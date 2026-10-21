#include "hex-common.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "dma-queue.h"
#include "htp-vtcm.h"
#include "hvx-utils.h"
#include "hvx-transpose.h"
#include "hex-fastdiv.h"
#include <string.h>

struct htp_concat_context {
    struct htp_ops_context * octx;
    uint32_t dim;
    uint32_t nrows_per_thread;
    uint32_t row_start;
    uint32_t nrows;
    uint32_t elem_start;
    uint32_t nelems;
    uint32_t nplanes;
    struct fastdiv_values div_ne0;
    struct fastdiv_values div_ne1;
    struct fastdiv_values div_ne2;
    const int32_t * win_offs; // recurrent conv ring: element offset per plane of src0 (NULL when absent)
};

static void concat_2d_f32_transposed(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t src0_ne0 = src0->ne[0];
    const uint32_t src1_ne0 = src1->ne[0];

    const uint32_t row_end = cctx->row_start + cctx->nrows;
    const uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    const uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    if (start_i >= end_i) return;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * spad1_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;

    const uint32_t block_i = 32;
    const uint32_t spad1_stride = block_i * sizeof(float);

    int32_t offsets[32] __attribute__((aligned(128)));
    for(int k=0; k<32; k++) {
        offsets[k] = k * spad1_stride;
    }
    HVX_Vector vv = *(HVX_Vector*)offsets;
    const uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 32);
    const uint32_t spad0_row_bytes = hex_round_up((src0_ne0 + src1_ne0_padded) * sizeof(float), VLEN);
    uint32_t mu = src1_ne0_padded * spad1_stride;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    for (uint32_t p = 0; p < cctx->nplanes; p++) {
        const uint32_t i3 = p / dst->ne[2];
        const uint32_t i2 = p - i3 * dst->ne[2];
        dma_addr_t src0_plane = src0->data + i2 * src0->nb[2] + i3 * src0->nb[3];
        if (cctx->win_offs) {
            src0_plane += (dma_addr_t) cctx->win_offs[i2] * src0->nb[0];
        }
        const dma_addr_t src1_plane = src1->data + i2 * src1->nb[2] + i3 * src1->nb[3];
        const dma_addr_t dst_plane  = dst->data  + i2 * dst->nb[2]  + i3 * dst->nb[3];

        for (uint32_t i = start_i; i < end_i; i += block_i) {
            uint32_t current_block_i = (end_i - i < block_i) ? (end_i - i) : block_i;

            uint32_t src1_width_bytes = current_block_i * sizeof(float);
            const dma_addr_t src1_addr = src1_plane + i * src1->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], src1_width_bytes, src1_ne0);

            uint32_t src0_row_bytes = src0_ne0 * sizeof(float);
            const dma_addr_t src0_addr = src0_plane + i * src0->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad0_base, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, current_block_i);

            dma_queue_pop(dma_q); // src1

            HVX_Vector * vtcm_tmp = (HVX_Vector *)(spad1_base + src1_ne0_padded * spad1_stride);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            for (uint32_t j = 0; j < src1_ne0_padded; j += 32) {
                #pragma unroll(4)
                for (uint32_t ii = 0; ii < current_block_i; ii++) {
                    size_t rt = (size_t)(spad1_base + j * spad1_stride + ii * sizeof(float));
                    Q6_vgather_ARMVw(&vtcm_tmp[ii], rt, mu, vv);
                    uint8_t * dst_ptr = spad0_base + ii * spad0_row_bytes + (src0_ne0 + j) * sizeof(float);
                    hvx_vmemu(dst_ptr) = vtcm_tmp[ii];
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

            dma_queue_pop(dma_q); // src0

            const dma_addr_t dst_addr = dst_plane + i * dst->nb[1];
            dma_queue_push(dma_q, dma_make_data(dst_addr, spad0_base), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(float), current_block_i);

            dma_queue_pop(dma_q);
        }
    }
}

// CONCAT_T_HVX (GGML_HEXAGON_CONCAT_T_HVX=1; the host sets op_params[1] = 1 on the descriptor): the f32 transposed window copy of concat_2d_f32_transposed with the per-element gathers replaced by an HVX 32x32 word transpose of the src1 token tile, and the DMA of block b + 1 in flight while block b is transposed (CONCAT_T_SLOTS slots per thread).
// It serves src1 columns M <= CONCAT_T_BLK only; the bytes written to dst are those of the gather path.
#define CONCAT_T_BLK   32   // channels per block: one 32-word HVX tile
#define CONCAT_T_SLOTS 3    // block b + 1 loads while block b is transposed; block b - 2 has retired its out DMA

struct concat_t_state {
    dma_queue *                       q;
    const struct htp_concat_context * cctx;
    const struct htp_tensor *         src0;
    const struct htp_tensor *         src1;
    const struct htp_tensor *         dst;
    uint8_t *                         spad0;                       // per slot: CONCAT_T_BLK rows of row0_bytes
    uint8_t *                         spad1;                       // per slot: src1 token rows of CONCAT_T_BLK channels
    uint32_t                          start_i;
    uint32_t                          end_i;
    uint32_t                          nbi;                         // channel blocks per plane
    uint32_t                          row0_bytes;
    uint32_t                          tk_pushed;                   // DMA tickets, same protocol as gated-delta-net-ops.c
    uint32_t                          tk_popped;
    uint32_t                          in_tk[CONCAT_T_SLOTS][2];    // [slot][0] src1 tile, [slot][1] src0 tile
    uint32_t                          out_tk[CONCAT_T_SLOTS];      // out DMA that reads the slot's row tile
    dma_addr_t                        blk_dst[CONCAT_T_SLOTS];     // first output row of the slot's block
    uint32_t                          blk_cur[CONCAT_T_SLOTS];     // channels of the slot's block
};

static inline uint32_t concat_t_push(struct concat_t_state * st, dma_data d, size_t dst_stride, size_t src_stride,
                                     size_t row_size, size_t nrows) {
    while (!dma_queue_push(st->q, d, dst_stride, src_stride, row_size, nrows)) {
        dma_queue_pop(st->q);   // ring full: retire the oldest entry, then retry
        st->tk_popped++;
    }
    return ++st->tk_pushed;
}

static inline void concat_t_wait(struct concat_t_state * st, uint32_t ticket) {
    while ((int32_t) (ticket - st->tk_popped) > 0) {
        dma_queue_pop(st->q);
        st->tk_popped++;
    }
}

// in-DMAs of block b into its slot: src1 token rows, then src0 state rows (FIFO: the src0 ticket retires both)
static inline void concat_t_issue(struct concat_t_state * st, uint32_t b) {
    const struct htp_concat_context * cctx = st->cctx;
    const struct htp_tensor * src0 = st->src0;
    const struct htp_tensor * src1 = st->src1;
    const struct htp_tensor * dst  = st->dst;

    const uint32_t p   = b / st->nbi;
    const uint32_t i   = st->start_i + (b - p * st->nbi) * CONCAT_T_BLK;
    const uint32_t cur = (st->end_i - i < CONCAT_T_BLK) ? (st->end_i - i) : CONCAT_T_BLK;
    const uint32_t s   = b % CONCAT_T_SLOTS;

    const uint32_t i3 = p / dst->ne[2];
    const uint32_t i2 = p - i3 * dst->ne[2];
    dma_addr_t src0_plane = src0->data + i2 * src0->nb[2] + i3 * src0->nb[3];
    if (cctx->win_offs) {
        src0_plane += (dma_addr_t) cctx->win_offs[i2] * src0->nb[0];
    }
    const dma_addr_t src1_plane = src1->data + i2 * src1->nb[2] + i3 * src1->nb[3];
    const dma_addr_t dst_plane  = dst->data  + i2 * dst->nb[2]  + i3 * dst->nb[3];

    // the out DMA of block b - CONCAT_T_SLOTS still reads this slot's row tile: retire it before the tile is refilled
    concat_t_wait(st, st->out_tk[s]);

    uint8_t * tile0 = st->spad0 + s * CONCAT_T_BLK * st->row0_bytes;
    uint8_t * tile1 = st->spad1 + s * CONCAT_T_BLK * CONCAT_T_BLK * sizeof(float);
    st->in_tk[s][0] = concat_t_push(st, dma_make_data(tile1, src1_plane + i * src1->nb[1]),
                                    CONCAT_T_BLK * sizeof(float), src1->nb[0], cur * sizeof(float), src1->ne[0]);
    st->in_tk[s][1] = concat_t_push(st, dma_make_data(tile0, src0_plane + i * src0->nb[1]),
                                    st->row0_bytes, src0->nb[1], src0->ne[0] * sizeof(float), cur);
    st->blk_dst[s] = dst_plane + i * dst->nb[1];
    st->blk_cur[s] = cur;
}

// block b: wait for its tiles, transpose the token tile into the row tiles (state columns stay in place), push out
static inline void concat_t_finish(struct concat_t_state * st, uint32_t b) {
    const uint32_t s = b % CONCAT_T_SLOTS;
    concat_t_wait(st, st->in_tk[s][1]);

    uint8_t * tile0 = st->spad0 + s * CONCAT_T_BLK * st->row0_bytes;
    uint8_t * tile1 = st->spad1 + s * CONCAT_T_BLK * CONCAT_T_BLK * sizeof(float);

    HVX_Vector m[CONCAT_T_BLK];
    HVX_Vector tmp[CONCAT_T_BLK];
    for (uint32_t k = 0; k < CONCAT_T_BLK; k++) {
        m[k] = *(const HVX_Vector *) (tile1 + k * CONCAT_T_BLK * sizeof(float));   // token row k (rows >= M are unused)
    }
    hvx_transpose_32x32_words(m, tmp);                  // m[c] lane k = token k of channel c
    const uint32_t tok_off = st->src0->ne[0] * sizeof(float);
    for (uint32_t c = 0; c < CONCAT_T_BLK; c++) {
        hvx_vmemu(tile0 + c * st->row0_bytes + tok_off) = m[c];
    }

    st->out_tk[s] = concat_t_push(st, dma_make_data(st->blk_dst[s], tile0), st->dst->nb[1], st->row0_bytes,
                                  (st->src0->ne[0] + st->src1->ne[0]) * sizeof(float), st->blk_cur[s]);
}

static void concat_2d_f32_transposed_hvx(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const uint32_t row_end = cctx->row_start + cctx->nrows;
    const uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    const uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    if (start_i >= end_i) return;

    struct concat_t_state st;
    memset(&st, 0, sizeof(st));
    st.q          = octx->ctx->dma[ith];
    st.cctx       = cctx;
    st.src0       = octx->src[0];
    st.src1       = octx->src[1];
    st.dst        = octx->dst;
    st.spad0      = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    st.spad1      = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    st.start_i    = start_i;
    st.end_i      = end_i;
    st.nbi        = (end_i - start_i + CONCAT_T_BLK - 1) / CONCAT_T_BLK;
    st.row0_bytes = hex_round_up((st.src0->ne[0] + CONCAT_T_BLK) * sizeof(float), VLEN);

    const uint32_t nblk = cctx->nplanes * st.nbi;

    concat_t_issue(&st, 0);
    for (uint32_t b = 0; b < nblk; b++) {
        if (b + 1 < nblk) {
            concat_t_issue(&st, b + 1);   // DMA of block b + 1 overlaps the transpose of block b
        }
        concat_t_finish(&st, b);
    }
    dma_queue_flush(st.q);
}

static void concat_2d_f16_transposed(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t src0_ne0 = src0->ne[0];
    const uint32_t src1_ne0 = src1->ne[0];

    const uint32_t row_end = cctx->row_start + cctx->nrows;
    const uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    const uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    if (start_i >= end_i) return;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * spad1_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;

    const uint32_t block_i = 64;
    const uint32_t spad1_stride = block_i * sizeof(__fp16);

    int16_t offsets[64] __attribute__((aligned(128)));
    for(int k=0; k<64; k++) {
        offsets[k] = k * spad1_stride;
    }
    HVX_Vector vv = *(HVX_Vector*)offsets;
    const uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 64);
    const uint32_t spad0_row_bytes = hex_round_up((src0_ne0 + src1_ne0_padded) * sizeof(__fp16), VLEN);
    uint32_t mu = src1_ne0_padded * spad1_stride;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    for (uint32_t p = 0; p < cctx->nplanes; p++) {
        const uint32_t i3 = p / dst->ne[2];
        const uint32_t i2 = p - i3 * dst->ne[2];
        const dma_addr_t src0_plane = src0->data + i2 * src0->nb[2] + i3 * src0->nb[3];
        const dma_addr_t src1_plane = src1->data + i2 * src1->nb[2] + i3 * src1->nb[3];
        const dma_addr_t dst_plane  = dst->data  + i2 * dst->nb[2]  + i3 * dst->nb[3];

        for (uint32_t i = start_i; i < end_i; i += block_i) {
            uint32_t current_block_i = (end_i - i < block_i) ? (end_i - i) : block_i;

            uint32_t src1_width_bytes = current_block_i * sizeof(__fp16);
            const dma_addr_t src1_addr = src1_plane + i * src1->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], src1_width_bytes, src1_ne0);

            uint32_t src0_row_bytes = src0_ne0 * sizeof(__fp16);
            const dma_addr_t src0_addr = src0_plane + i * src0->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad0_base, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, current_block_i);

            dma_queue_pop(dma_q); // src1

            HVX_Vector * vtcm_tmp = (HVX_Vector *)(spad1_base + src1_ne0_padded * spad1_stride);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            for (uint32_t j = 0; j < src1_ne0_padded; j += 64) {
                #pragma unroll(4)
                for (uint32_t ii = 0; ii < current_block_i; ii++) {
                    size_t rt = (size_t)(spad1_base + j * spad1_stride + ii * sizeof(__fp16));
                    Q6_vgather_ARMVh(&vtcm_tmp[ii], rt, mu, vv);
                    uint8_t * dst_ptr = spad0_base + ii * spad0_row_bytes + (src0_ne0 + j) * sizeof(__fp16);
                    hvx_vmemu(dst_ptr) = vtcm_tmp[ii];
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

            dma_queue_pop(dma_q); // src0

            const dma_addr_t dst_addr = dst_plane + i * dst->nb[1];
            dma_queue_push(dma_q, dma_make_data(dst_addr, spad0_base), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(__fp16), current_block_i);

            dma_queue_pop(dma_q);
        }
    }
}

static void concat_generic(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const int dim = cctx->dim;
    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    const uint32_t ne[4] = {dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]};

    // Per-device element range aligned to prevent false sharing
    const uint32_t elem_start = cctx->elem_start;
    const uint32_t nelems     = cctx->nelems;
    const uint32_t chunk_size = (nelems + nth - 1) / nth;

    const uint32_t start_idx = MIN(elem_start + ith * chunk_size, elem_start + nelems);
    const uint32_t end_idx   = MIN(start_idx + chunk_size, elem_start + nelems);

    // Naive scalar element-wise copy
    for (uint32_t idx = start_idx; idx < end_idx; idx++) {
        uint32_t idx_div_ne0 = fastdiv(idx, &cctx->div_ne0);
        uint32_t i0 = idx - idx_div_ne0 * ne[0];

        uint32_t idx_div_ne01 = fastdiv(idx_div_ne0, &cctx->div_ne1);
        uint32_t i1 = idx_div_ne0 - idx_div_ne01 * ne[1];

        uint32_t idx_div_ne012 = fastdiv(idx_div_ne01, &cctx->div_ne2);
        uint32_t i2 = idx_div_ne01 - idx_div_ne012 * ne[2];
        uint32_t i3 = idx_div_ne012;

        uint8_t * dst_ptr = (uint8_t *)dst->data + i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1] + i0 * dst->nb[0];

        uint32_t idx_dim = 0;
        if (dim == 0) idx_dim = i0;
        else if (dim == 1) idx_dim = i1;
        else if (dim == 2) idx_dim = i2;
        else if (dim == 3) idx_dim = i3;

        const struct htp_tensor * src = (idx_dim < src0->ne[dim]) ? src0 : src1;

        uint32_t s0 = i0;
        uint32_t s1 = i1;
        uint32_t s2 = i2;
        uint32_t s3 = i3;

        if (dim == 0 && src == src1) s0 -= src0->ne[0];
        if (dim == 1 && src == src1) s1 -= src0->ne[1];
        if (dim == 2 && src == src1) s2 -= src0->ne[2];
        if (dim == 3 && src == src1) s3 -= src0->ne[3];

        uint8_t * src_ptr = (uint8_t *)src->data + s3 * src->nb[3] + s2 * src->nb[2] + s1 * src->nb[1] + s0 * src->nb[0];

        if (type_size == 4) {
            *(float*)dst_ptr = *(float*)src_ptr;
        } else {
            *(__fp16*)dst_ptr = *(__fp16*)src_ptr;
        }
    }
}

static bool concat_dim1_contiguous_dma(struct htp_ops_context * octx, int dim, uint32_t type_size) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (dim != 1 || octx->ctx->mdev.count > 1 ||
        (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_I32) ||
        src0->type != dst->type || src1->type != dst->type ||
        src0->ne[0] != dst->ne[0] || src1->ne[0] != dst->ne[0] ||
        src0->ne[2] != dst->ne[2] || src1->ne[2] != dst->ne[2] ||
        src0->ne[3] != dst->ne[3] || src1->ne[3] != dst->ne[3] ||
        dst->ne[1] != src0->ne[1] + src1->ne[1] ||
        !htp_tensor_is_contiguous(src0, type_size) ||
        !htp_tensor_is_contiguous(src1, type_size) ||
        !htp_tensor_is_contiguous(dst, type_size)) {
        return false;
    }

    const uint32_t src0_row_size = src0->ne[0] * type_size;
    const uint32_t src1_row_size = src1->ne[0] * type_size;

    // v75+ dma_queue_push() writes a 2D descriptor directly and does not split overflow.
#if __HVX_ARCH__ >= 75
    if (src0_row_size > 0xffffffu || src1_row_size > 0xffffffu ||
        src0->nb[1] > 0xffffffu || src1->nb[1] > 0xffffffu || dst->nb[1] > 0xffffffu ||
        src0->ne[1] > UINT16_MAX || src1->ne[1] > UINT16_MAX) {
        return false;
    }
#endif

    dma_queue * q = octx->ctx->dma[0];

    for (uint32_t i3 = 0; i3 < dst->ne[3]; ++i3) {
        for (uint32_t i2 = 0; i2 < dst->ne[2]; ++i2) {
            dma_addr_t dst_addr  = dst->data  + i3 * dst->nb[3]  + i2 * dst->nb[2];
            dma_addr_t src0_addr = src0->data + i3 * src0->nb[3] + i2 * src0->nb[2];
            dma_addr_t src1_addr = src1->data + i3 * src1->nb[3] + i2 * src1->nb[2];

            if (!dma_queue_push(q, dma_make_data(dst_addr, src0_addr), dst->nb[1], src0->nb[1], src0_row_size, src0->ne[1])) {
                dma_queue_flush(q);
                dma_queue_push(q, dma_make_data(dst_addr, src0_addr), dst->nb[1], src0->nb[1], src0_row_size, src0->ne[1]);
            }

            dst_addr += src0->ne[1] * dst->nb[1];
            if (!dma_queue_push(q, dma_make_data(dst_addr, src1_addr), dst->nb[1], src1->nb[1], src1_row_size, src1->ne[1])) {
                dma_queue_flush(q);
                dma_queue_push(q, dma_make_data(dst_addr, src1_addr), dst->nb[1], src1->nb[1], src1_row_size, src1->ne[1]);
            }
        }
    }

    dma_queue_flush(q);
    return true;
}

int op_concat(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    int dim = octx->op_params[0];

    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;
    if (octx->src[2] && type_size != 4) {
        return HTP_STATUS_NO_SUPPORT;
    }
    bool is_src1_transposed  = (src1->nb[0] > src1->nb[1]);
    bool is_src0_transposed  = (src0->nb[0] > src0->nb[1]);

    if (!octx->src[2] && concat_dim1_contiguous_dma(octx, dim, type_size)) {
        return HTP_STATUS_OK;
    }

    uint32_t n_threads = octx->n_threads;
    struct htp_concat_context cctx;
    cctx.octx = octx;
    cctx.dim = dim;
    cctx.div_ne0 = init_fastdiv_values(dst->ne[0]);
    cctx.div_ne1 = init_fastdiv_values(dst->ne[1]);
    cctx.div_ne2 = init_fastdiv_values(dst->ne[2]);
    cctx.win_offs = octx->src[2] ? (const int32_t *) (uintptr_t) octx->src[2]->data : NULL;

    void (*worker_func)(unsigned int, unsigned int, void *) = concat_generic;

    const bool rows_ok = src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size;

    if (dim == 0 && is_src1_transposed && !is_src0_transposed && rows_ok) {
        const uint32_t total_rows = dst->ne[1];
        const size_t dst_data_row_size = dst->ne[0] * type_size;
        uint32_t row_start = 0;
        uint32_t nrows     = total_rows;
        if (octx->ctx->mdev.count > 1) {
            uint32_t rows_per_chunk = 0;
            htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_data_row_size, &rows_per_chunk);
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            row_start = range.start;
            nrows     = range.count;
        }

        if (nrows == 0) {
            return HTP_STATUS_OK;
        }

        cctx.row_start = row_start;
        cctx.nrows     = nrows;
        cctx.nplanes   = dst->ne[2] * dst->ne[3];

        uint32_t block_i = (type_size == 4) ? 32 : 64;

        cctx.nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

        // Allocate VTCM
        uint32_t spad1_stride = block_i * type_size;

        uint32_t src1_ne0_padded = hex_round_up(src1->ne[0], block_i);
        uint32_t spad0_row_bytes = hex_round_up((src0->ne[0] + src1_ne0_padded) * type_size, VLEN);

        // CONCAT_T_HVX: CONCAT_T_SLOTS slots of (row tile, token tile) per thread, no gather scratch (concat_2d_f32_transposed_hvx).
        // Its row stride equals spad0_row_bytes here: src1_ne0_padded is 32 for M <= 32.
        const bool t_hvx = type_size == 4 && (octx->op_params[1] & 1) && src1->ne[0] <= CONCAT_T_BLK;
        if (t_hvx) {
            octx->src0_spad.size_per_thread = CONCAT_T_SLOTS * CONCAT_T_BLK * spad0_row_bytes;
            octx->src1_spad.size_per_thread = CONCAT_T_SLOTS * CONCAT_T_BLK * spad1_stride;
        } else {
            octx->src0_spad.size_per_thread = block_i * spad0_row_bytes;
            octx->src1_spad.size_per_thread = src1_ne0_padded * spad1_stride + block_i * VLEN;
        }

        octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
        octx->src1_spad.size = n_threads * octx->src1_spad.size_per_thread;

        if (octx->src0_spad.size + octx->src1_spad.size > octx->ctx->vtcm_size) {
            return HTP_STATUS_VTCM_TOO_SMALL;
        }

        octx->src0_spad.data = octx->ctx->vtcm_base;
        octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
        octx->src0_spad.src  = NULL;
        octx->src1_spad.src  = NULL;

        if (type_size == 4) {
            worker_func = t_hvx ? concat_2d_f32_transposed_hvx : concat_2d_f32_transposed;
        } else {
            worker_func = concat_2d_f16_transposed;
        }
    } else {
        if (octx->src[2]) {
            return HTP_STATUS_NO_SUPPORT;
        }
        if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
            return HTP_STATUS_NO_SUPPORT;
        }

        const uint32_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
        uint32_t elem_start = 0;
        uint32_t nelems     = total_elements;
        if (octx->ctx->mdev.count > 1) {
            const uint32_t elems_per_chunk = HEX_L2_LINE_SIZE / type_size;
            const bool can_split = htp_tensor_mdev_data_aligned(dst) && htp_tensor_is_contiguous(dst, type_size) && !htp_tensor_is_permuted(dst);
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_elements, can_split ? elems_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            elem_start = range.start;
            nelems     = range.count;
        }

        if (nelems == 0) {
            return HTP_STATUS_OK;
        }

        cctx.elem_start = elem_start;
        cctx.nelems     = nelems;
    }

    work_queue_run(octx->ctx->work_queue, worker_func, &cctx, n_threads);
    return HTP_STATUS_OK;
}
