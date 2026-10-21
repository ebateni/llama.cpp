#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <stdatomic.h>
#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "htp-tensor.h"
#include "dma-queue.h"
#include "hex-profile.h"
#include "allreduce-ops.h"
#include "htp-fence.h"
#include "hvx-norm.h"
#include "hmx-act-tiles.h"

struct htp_allreduce_context {
    struct htp_ops_context * octx;
    uint32_t n_ranks;
    uint32_t n_dsts;
    uint32_t nelem;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t row_size_aligned;
    uint32_t rank_elem_start;
    uint32_t rank_nelem;
    uint32_t elems_per_thread;
    uint32_t block_elems;
    uint32_t vtcm_size_per_thread;
    bool     is_row_bcast;
    uint8_t * src_spad_base[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_base;
    uint8_t * res_spad_base;
};

#define DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_ADD_FN, HAS_ADD)                                       \
static void allreduce_thread_dma_1d_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                 \
    struct htp_allreduce_context * actx = (struct htp_allreduce_context *) data;                                \
    struct htp_ops_context * octx = actx->octx;                                                                 \
                                                                                                                \
    const uint32_t n_ranks     = actx->n_ranks;                                                                 \
    const uint32_t n_dsts      = actx->n_dsts;                                                                  \
    const uint32_t block_elems = actx->block_elems;                                                             \
                                                                                                                \
    const uint32_t dr  = actx->elems_per_thread;                                                                \
    const uint32_t ir0 = actx->rank_elem_start + dr * ith;                                                      \
    const uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_start + actx->rank_nelem);                               \
    if (ir0 >= ir1) return;                                                                                     \
                                                                                                                \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                      \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                    \
                                                                                                                \
    uint8_t * src_spad_base[HTP_ALLREDUCE_MAX_RANKS];                                                           \
    for (uint32_t s = 0; s < n_ranks; s++) {                                                                    \
        src_spad_base[s] = actx->src_spad_base[s] + (ith * actx->vtcm_size_per_thread);                         \
    }                                                                                                           \
    uint8_t * dst_spad_base = actx->dst_spad_base + (ith * actx->vtcm_size_per_thread);                         \
    uint8_t * res_spad_base = HAS_ADD ? (actx->res_spad_base + (ith * actx->vtcm_size_per_thread)) : NULL;      \
                                                                                                                \
    const size_t spad_half = actx->vtcm_size_per_thread / 2;                                                    \
    uint32_t ir_prefetch = ir0;                                                                                 \
    int spad_idx = 0;                                                                                           \
                                                                                                                \
    for (int k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                          \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir_prefetch);                                               \
        size_t   cur_bytes = cur_elems * sizeof(TYPE);                                                          \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                \
        for (uint32_t d = 0; d < n_dsts; d++) {                                                                 \
            dma_addr_t d_ddr = octx->dsts[d]->data + ir_prefetch * sizeof(TYPE);                                \
            dma_queue_push(dma_q, dma_make_data(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 0);            \
        }                                                                                                       \
        for (uint32_t s = 0; s < n_ranks; s++) {                                                                \
            uint8_t * s_spad = src_spad_base[s] + spad_idx * spad_half;                                         \
            const dma_addr_t s_ddr = octx->src[s]->data + ir_prefetch * sizeof(TYPE);                           \
            dma_queue_push(dma_q, dma_make_data(s_spad, s_ddr), cur_bytes, cur_bytes, cur_bytes, 1);            \
        }                                                                                                       \
        if (HAS_ADD) {                                                                                          \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                            \
            const dma_addr_t r_ddr = octx->src[2 * n_ranks]->data + ir_prefetch * sizeof(TYPE);                 \
            dma_queue_push(dma_q, dma_make_data(r_spad, r_ddr), cur_bytes, cur_bytes, cur_bytes, 1);            \
        }                                                                                                       \
        ir_prefetch += cur_elems;                                                                               \
        spad_idx ^= 1;                                                                                          \
    }                                                                                                           \
                                                                                                                \
    for (uint32_t ir = ir0; ir < ir1; ) {                                                                       \
        uint32_t cur_elems = MIN(block_elems, ir1 - ir);                                                        \
        size_t   cur_bytes = cur_elems * sizeof(TYPE);                                                          \
        uint8_t * d_spad = NULL;                                                                                \
        for (uint32_t d = 0; d < n_dsts; d++) {                                                                 \
            d_spad = (uint8_t *) dma_queue_pop(dma_q).src;                                                      \
        }                                                                                                       \
        uint8_t * s_spad[HTP_ALLREDUCE_MAX_RANKS];                                                              \
        for (uint32_t s = 0; s < n_ranks; s++) {                                                                \
            s_spad[s] = (uint8_t *) dma_queue_pop(dma_q).dst;                                                   \
        }                                                                                                       \
        uint8_t * r_spad = HAS_ADD ? (uint8_t *) dma_queue_pop(dma_q).dst : NULL;                               \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                       \
        HVX_ADD_FN(d_spad, s_spad[0], s_spad[1], cur_elems);                                                    \
        for (uint32_t s = 2; s < n_ranks; s++) {                                                                \
            HVX_ADD_FN(d_spad, d_spad, s_spad[s], cur_elems);                                                   \
        }                                                                                                       \
        if (HAS_ADD) {                                                                                          \
            HVX_ADD_FN(d_spad, d_spad, r_spad, cur_elems);                                                      \
        }                                                                                                       \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                        \
        for (uint32_t d = 0; d < n_dsts; d++) {                                                                 \
            dma_addr_t d_ddr = octx->dsts[d]->data + ir * sizeof(TYPE);                                         \
            dma_queue_push(dma_q, dma_make_data(d_ddr, d_spad), cur_bytes, cur_bytes, cur_bytes, 1);            \
        }                                                                                                       \
        if (ir_prefetch < ir1) {                                                                                \
            uint32_t next_elems = MIN(block_elems, ir1 - ir_prefetch);                                          \
            size_t   next_bytes = next_elems * sizeof(TYPE);                                                    \
            for (uint32_t s = 0; s < n_ranks; s++) {                                                            \
                const dma_addr_t s_next = octx->src[s]->data + ir_prefetch * sizeof(TYPE);                      \
                dma_queue_push(dma_q, dma_make_data(s_spad[s], s_next), next_bytes, next_bytes, next_bytes, 1); \
            }                                                                                                   \
            if (HAS_ADD) {                                                                                      \
                const dma_addr_t r_next = octx->src[2 * n_ranks]->data + ir_prefetch * sizeof(TYPE);            \
                dma_queue_push(dma_q, dma_make_data(r_spad, r_next), next_bytes, next_bytes, next_bytes, 1);    \
            }                                                                                                   \
            ir_prefetch += next_elems;                                                                          \
        }                                                                                                       \
        ir += cur_elems;                                                                                        \
    }                                                                                                           \
    dma_queue_flush(dma_q);                                                                                     \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_add_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     float,  hvx_add_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f16, __fp16, hvx_add_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(add_f32, float,  hvx_add_f32_aaa, 1)

#define DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_ADD_FN, HAS_ADD, IS_ROW_BCAST)                                                           \
static void allreduce_thread_dma_2d_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                                   \
    struct htp_allreduce_context * actx = (struct htp_allreduce_context *) data;                                                                  \
    struct htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    const uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    const uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    const uint32_t ne0              = actx->ne0;                                                                                                  \
    const uint32_t block_rows       = actx->block_elems;                                                                                          \
    const uint32_t row_size_aligned = actx->row_size_aligned;                                                                                     \
    const uint32_t row_bytes        = ne0 * sizeof(TYPE);                                                                                         \
                                                                                                                                                  \
    const uint32_t dr  = actx->elems_per_thread;                                                                                                  \
    const uint32_t r0  = actx->rank_elem_start + dr * ith;                                                                                        \
    const uint32_t r1  = MIN(r0 + dr, actx->rank_elem_start + actx->rank_nelem);                                                                  \
    if (r0 >= r1) return;                                                                                                                         \
                                                                                                                                                  \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    uint8_t * src_spad_base[HTP_ALLREDUCE_MAX_RANKS];                                                                                             \
    for (uint32_t s = 0; s < n_ranks; s++) {                                                                                                      \
        src_spad_base[s] = actx->src_spad_base[s] + (ith * actx->vtcm_size_per_thread);                                                           \
    }                                                                                                                                             \
    uint8_t * dst_spad_base = actx->dst_spad_base + (ith * actx->vtcm_size_per_thread);                                                           \
    uint8_t * res_spad_base = HAS_ADD ? (IS_ROW_BCAST ? actx->res_spad_base : (actx->res_spad_base + (ith * actx->vtcm_size_per_thread))) : NULL; \
                                                                                                                                                  \
    const size_t spad_half = actx->vtcm_size_per_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    int spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    for (int k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_rows = MIN(block_rows, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_base + spad_idx * spad_half;                                                                                  \
        for (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_queue_push(dma_q, dma_make_data(d_ddr, d_spad), octx->dsts[d]->nb[1], row_size_aligned, row_bytes, 0);                            \
        }                                                                                                                                         \
        for (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = src_spad_base[s] + spad_idx * spad_half;                                                                           \
            const dma_addr_t s_ddr = octx->src[s]->data + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_queue_push(dma_q, dma_make_data(s_spad, s_ddr), row_size_aligned, octx->src[s]->nb[1], row_bytes, cur_rows);                      \
        }                                                                                                                                         \
        if (HAS_ADD && !IS_ROW_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_base + spad_idx * spad_half;                                                                              \
            const dma_addr_t r_ddr = octx->src[2 * n_ranks]->data + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_queue_push(dma_q, dma_make_data(r_spad, r_ddr), row_size_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, cur_rows);            \
        }                                                                                                                                         \
        r_prefetch += cur_rows;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    for (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_rows = MIN(block_rows, r1 - r);                                                                                              \
        uint8_t * d_spad = NULL;                                                                                                                  \
        for (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            d_spad = (uint8_t *) dma_queue_pop(dma_q).src;                                                                                        \
        }                                                                                                                                         \
        uint8_t * s_spad[HTP_ALLREDUCE_MAX_RANKS];                                                                                                \
        for (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            s_spad[s] = (uint8_t *) dma_queue_pop(dma_q).dst;                                                                                     \
        }                                                                                                                                         \
        uint8_t * r_spad = (HAS_ADD && !IS_ROW_BCAST) ? (uint8_t *) dma_queue_pop(dma_q).dst : NULL;                                              \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        for (uint32_t row = 0; row < cur_rows; row++) {                                                                                           \
            uint8_t * d_row = d_spad + row * row_size_aligned;                                                                                    \
            const uint8_t * s0_row = s_spad[0] + row * row_size_aligned;                                                                          \
            const uint8_t * s1_row = s_spad[1] + row * row_size_aligned;                                                                          \
            HVX_ADD_FN(d_row, s0_row, s1_row, ne0);                                                                                               \
            for (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                const uint8_t * ss_row = s_spad[s] + row * row_size_aligned;                                                                      \
                HVX_ADD_FN(d_row, d_row, ss_row, ne0);                                                                                            \
            }                                                                                                                                     \
            if (HAS_ADD) {                                                                                                                        \
                const uint8_t * res_row = IS_ROW_BCAST ? res_spad_base : (r_spad + row * row_size_aligned);                                       \
                HVX_ADD_FN(d_row, d_row, res_row, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        for (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->data + r * octx->dsts[d]->nb[1];                                                                    \
            dma_queue_push(dma_q, dma_make_data(d_ddr, d_spad), octx->dsts[d]->nb[1], row_size_aligned, row_bytes, cur_rows);                     \
        }                                                                                                                                         \
        if (r_prefetch < r1) {                                                                                                                    \
            uint32_t next_rows = MIN(block_rows, r1 - r_prefetch);                                                                                \
            for (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                const dma_addr_t s_next = octx->src[s]->data + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_queue_push(dma_q, dma_make_data(s_spad[s], s_next), row_size_aligned, octx->src[s]->nb[1], row_bytes, next_rows);             \
            }                                                                                                                                     \
            if (HAS_ADD && !IS_ROW_BCAST) {                                                                                                       \
                const dma_addr_t r_next = octx->src[2 * n_ranks]->data + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_queue_push(dma_q, dma_make_data(r_spad, r_next), row_size_aligned, octx->src[2 * n_ranks]->nb[1], row_bytes, next_rows);      \
            }                                                                                                                                     \
            r_prefetch += next_rows;                                                                                                              \
        }                                                                                                                                         \
        r += cur_rows;                                                                                                                            \
    }                                                                                                                                             \
    dma_queue_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_add_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           float,  hvx_add_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f16,       __fp16, hvx_add_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_f32,       float,  hvx_add_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f16, __fp16, hvx_add_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(add_bcast_f32, float,  hvx_add_f32_aaa, 1, 1)

static int validate_allreduce(
    struct htp_ops_context * octx,
    const struct htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    if (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (kparams->vtcm_size_per_thread <= 0 || kparams->vtcm_size <= 0) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);
    const size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_count(
        n_ranks, octx->n_threads, has_add, kparams->is_row_bcast != 0);
    const size_t vtcm_size = n_vtcm_buffers * (size_t) kparams->vtcm_size_per_thread;
    if (vtcm_size != (size_t) kparams->vtcm_size) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    if (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    return HTP_STATUS_OK;
}

// MC05 norm gather, threaded (kparams->n_threads > 1; host: GGML_HEXAGON_AR_RMS_AG_MT=1). Rows are independent, so any split of the rank's rows gives the same bits. Thread ith owns dma[ith] and two src and two dst slots after the shared weight row.
// The queue is popped in push order (writes of block b-2, then the read of block b); each worker ends with one flush. Block b+2 is read into the src slot of block b once block b is normalized. The writes of block b are pushed asynchronously; the dst slot of block b is reused only after its writes are popped.
struct htp_rms_ag_context {
    struct htp_ops_context * octx;
    dma_addr_t               resid;
    uint8_t *                w_spad;
    uint8_t *                spad_base;
    uint32_t                 r0;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 blk_rows;
    uint32_t                 ne0;
    uint32_t                 row_bytes;
    uint32_t                 n_dsts;
    uint32_t                 vtcm_size_per_thread;
    float                    epsilon;
};

static void allreduce_rms_ag_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_rms_ag_context * rctx = (const struct htp_rms_ag_context *) data;
    struct htp_ops_context * octx = rctx->octx;

    const uint32_t row_begin = ith * rctx->rows_per_thread;
    if (row_begin >= rctx->nrows) return;
    const uint32_t n_rows = MIN(rctx->rows_per_thread, rctx->nrows - row_begin);
    const uint32_t blk    = rctx->blk_rows;
    const uint32_t rb     = rctx->row_bytes;
    const uint32_t n_blk  = (n_rows + blk - 1) / blk;
    const size_t   half   = (size_t) blk * rb;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * s_spad = rctx->spad_base + (size_t) ith * rctx->vtcm_size_per_thread;  // src slots 0 and 1
    uint8_t * d_spad = s_spad + 2 * half;                                              // dst slots 0 and 1

    // prologue: src blocks 0 and 1
    for (uint32_t b = 0; b < n_blk && b < 2; b++) {
        const size_t     nb  = (size_t) MIN(blk, n_rows - b * blk) * rb;
        const dma_addr_t src = rctx->resid + (dma_addr_t) (rctx->r0 + row_begin + b * blk) * rb;
        dma_queue_push(dma_q, dma_make_data(s_spad + (size_t) b * half, src), nb, nb, nb, 1);
    }

    for (uint32_t b = 0; b < n_blk; b++) {
        const uint32_t slot = b & 1;
        const uint32_t cur  = MIN(blk, n_rows - b * blk);
        const uint32_t row  = rctx->r0 + row_begin + b * blk;
        const size_t   nb   = (size_t) cur * rb;
        uint8_t * s_cur = s_spad + (size_t) slot * half;
        uint8_t * d_cur = d_spad + (size_t) slot * half;

        if (b >= 2) {
            // the dst slot was written by block b-2: its n_dsts writes are popped first
            for (uint32_t d = 0; d < rctx->n_dsts; d++) {
                (void) dma_queue_pop(dma_q);
            }
        }
        (void) dma_queue_pop(dma_q);  // src block b has landed

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);
        for (uint32_t i = 0; i < cur; i++) {
            hvx_fast_rms_norm_mul_f32(s_cur + (size_t) i * rb, rctx->w_spad, d_cur + (size_t) i * rb, (int) rctx->ne0, rctx->epsilon);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);

        for (uint32_t d = 0; d < rctx->n_dsts; d++) {
            dma_queue_push(dma_q, dma_make_data(octx->dsts[d]->data + (dma_addr_t) row * rb, d_cur), nb, nb, nb, 1);
        }
        if (b + 2 < n_blk) {
            // the src slot of block b is free again: read block b+2 into it
            const uint32_t next_row = rctx->r0 + row_begin + (b + 2) * blk;
            const size_t   next_nb  = (size_t) MIN(blk, n_rows - (b + 2) * blk) * rb;
            dma_queue_push(dma_q, dma_make_data(s_cur, rctx->resid + (dma_addr_t) next_row * rb), next_nb, next_nb, next_nb, 1);
        }
    }
    dma_queue_flush(dma_q);
}

// GGML_HEXAGON_GATHER_ACT_TILE: the norm gather writes fp16 HMX activation tiles (htp/hmx-act-tiles.h, the layout the ACT_TILE consumers read) instead of fp32 rows. The 16 rows fill one 1024-byte half of every tile, so each dst gets one 2D DMA of K/32 pieces (dst stride 2048).
// A staging slot is as large as 16 fp32 rows, so the per-thread layout is two src slots, two staging slots and two fp32 row buffers: (4 x 16 + 2) rows. Blocks are RMS_AG_TILE_ROWS rows starting at a multiple of 16 (host partition). A block is normalized in row pairs into two fp32 VTCM rows and converted with act_tile_row_pair_store into one tile row of staging (K/32 tiles of 2048 bytes, row index = global row % 32).
#define RMS_AG_TILE_ROWS 16

struct htp_rms_ag_tile_context {
    struct htp_ops_context * octx;
    dma_addr_t               resid;
    uint8_t *                w_spad;
    uint8_t *                spad_base;
    uint32_t                 r0;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 block_rows;    // rows per block: 16, or 8 (GGML_HEXAGON_AR_BLOCK_ROWS)
    uint32_t                 ne0;
    uint32_t                 row_bytes;
    uint32_t                 n_dsts;
    uint32_t                 vtcm_size_per_thread;
    float                    epsilon;
};

static void allreduce_rms_ag_tile_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_rms_ag_tile_context * rctx = (const struct htp_rms_ag_tile_context *) data;
    struct htp_ops_context * octx = rctx->octx;

    const uint32_t row_begin = ith * rctx->rows_per_thread;
    if (row_begin >= rctx->nrows) return;
    const uint32_t n_rows = MIN(rctx->rows_per_thread, rctx->nrows - row_begin);
    const uint32_t n_blk  = (n_rows + RMS_AG_TILE_ROWS - 1) / RMS_AG_TILE_ROWS;
    const uint32_t rb     = rctx->row_bytes;
    const uint32_t kt     = rctx->ne0 / HTP_MM_HMX_TILE_N_COLS;     // tiles per tile row
    const size_t   half   = (size_t) RMS_AG_TILE_ROWS * rb;          // == kt * 2048: 16 fp32 rows, one tile row of staging

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * s_spad = rctx->spad_base + (size_t) ith * rctx->vtcm_size_per_thread;  // src slots 0 and 1
    uint8_t * g_spad = s_spad + 2 * half;                                              // staging slots 0 and 1
    uint8_t * t_spad = g_spad + 2 * half;                                              // fp32 rows 0 and 1

    // prologue: src blocks 0 and 1
    for (uint32_t b = 0; b < n_blk && b < 2; b++) {
        const uint32_t cur   = MIN(RMS_AG_TILE_ROWS, n_rows - b * RMS_AG_TILE_ROWS);
        const size_t   nb    = (size_t) cur * rb;
        const dma_addr_t src = rctx->resid + (dma_addr_t) (rctx->r0 + row_begin + b * RMS_AG_TILE_ROWS) * rb;
        dma_queue_push(dma_q, dma_make_data(s_spad + (size_t) b * half, src), nb, nb, nb, 1);
    }

    for (uint32_t b = 0; b < n_blk; b++) {
        const uint32_t slot = b & 1;
        const uint32_t cur  = MIN(RMS_AG_TILE_ROWS, n_rows - b * RMS_AG_TILE_ROWS);
        const uint32_t row  = rctx->r0 + row_begin + b * RMS_AG_TILE_ROWS;  // first row of the block (multiple of 16)
        uint8_t * s_cur = s_spad + (size_t) slot * half;
        uint8_t * g_cur = g_spad + (size_t) slot * half;
        const uint32_t hoff = ((row % HTP_MM_HMX_TILE_N_ROWS) / RMS_AG_TILE_ROWS) * 1024;  // this block's half of every tile

        if (b >= 2) {
            // the staging slot was written by block b-2: its n_dsts writes are popped first
            for (uint32_t d = 0; d < rctx->n_dsts; d++) {
                (void) dma_queue_pop(dma_q);
            }
        }
        (void) dma_queue_pop(dma_q);  // src block b has landed

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);
        for (uint32_t i = 0; i + 1 < cur; i += 2) {
            hvx_fast_rms_norm_mul_f32(s_cur + (size_t) i * rb,       rctx->w_spad, t_spad,      (int) rctx->ne0, rctx->epsilon);
            hvx_fast_rms_norm_mul_f32(s_cur + (size_t) (i + 1) * rb, rctx->w_spad, t_spad + rb, (int) rctx->ne0, rctx->epsilon);
            act_tile_row_pair_store((__fp16 *) g_cur, (const float *) t_spad, (const float *) (t_spad + rb),
                                    (row + i) % HTP_MM_HMX_TILE_N_ROWS, rctx->ne0, rctx->ne0, true, true);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);

        for (uint32_t d = 0; d < rctx->n_dsts; d++) {
            const dma_addr_t dst = octx->dsts[d]->data +
                (dma_addr_t) (row / HTP_MM_HMX_TILE_N_ROWS) * kt * HTP_MM_HMX_TILE_N_ELMS * sizeof(__fp16) + hoff;
            dma_queue_push(dma_q, dma_make_data(dst, g_cur + hoff), 2048, 2048, 1024, kt);
        }
        if (b + 2 < n_blk) {
            // the src slot of block b is free again: read block b+2 into it
            const uint32_t next_cur = MIN(RMS_AG_TILE_ROWS, n_rows - (b + 2) * RMS_AG_TILE_ROWS);
            const uint32_t next_row = rctx->r0 + row_begin + (b + 2) * RMS_AG_TILE_ROWS;
            const size_t   next_nb  = (size_t) next_cur * rb;
            dma_queue_push(dma_q, dma_make_data(s_cur, rctx->resid + (dma_addr_t) next_row * rb), next_nb, next_nb, next_nb, 1);
        }
    }
    dma_queue_flush(dma_q);
}

// MC05: RMS_NORM_MUL behind the ALLREDUCE+ADD of this boundary. Inputs: [unused x n, fence x n, residual, norm weight]; outputs: norm dst x n.
// The residual is final on every rank once the ALLREDUCE+ADD exit barrier has passed, so rank r normalizes its own token rows with the RMS_NORM_MUL row kernel (identical per row) and writes them to every norm dst (all-gather).
static int op_allreduce_rms_ag(struct htp_ops_context * octx, const struct htp_allreduce_kernel_params * kparams,
                               uint32_t rank, uint32_t n_ranks) {
    const uint32_t fence_seq_entry = (uint32_t) octx->op_params[0];
    const uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];
    float epsilon = 0.0f;
    memcpy(&epsilon, &octx->op_params[2], sizeof(float));

    const uint32_t r0        = (uint32_t) kparams->rank_elem_start;
    const uint32_t nrows     = (uint32_t) kparams->rank_nelem;
    const uint32_t ne0       = (uint32_t) kparams->ne0;
    const uint32_t blk_rows  = (uint32_t) kparams->block_elems;
    const uint32_t row_bytes = ne0 * (uint32_t) sizeof(float);

    const struct htp_tensor * resid   = octx->src[2 * n_ranks];
    const struct htp_tensor * weight  = octx->src[2 * n_ranks + 1];
    const struct htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->data;

    // threaded gather: VTCM = shared weight row + n_threads x (2 src + 2 dst slots of blk_rows rows)
    const bool     mt        = kparams->n_threads > 1;
    const uint32_t n_threads = (uint32_t) kparams->n_threads;
    const bool     tile_out  = (octx->dsts[0]->flags & HTP_TENSOR_ACT_TILE) != 0;
    if (mt && (!htp_ops_context_set_n_threads(octx, n_threads) ||
               (size_t) kparams->vtcm_size > octx->ctx->vtcm_size ||
               (size_t) kparams->vtcm_size < row_bytes + (size_t) n_threads * 4 * blk_rows * row_bytes)) {
        htp_fence_write(my_fence, fence_seq_entry, HTP_STATUS_INVAL_PARAMS);
        htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INVAL_PARAMS);
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (tile_out && (nrows % RMS_AG_TILE_ROWS != 0 || ne0 % HTP_MM_HMX_TILE_N_COLS != 0 ||
                     (size_t) kparams->vtcm_size > octx->ctx->vtcm_size ||
                     (size_t) kparams->vtcm_size < row_bytes + (size_t) n_threads * (4 * RMS_AG_TILE_ROWS + 2) * row_bytes ||
                     (n_threads > 1 && !htp_ops_context_set_n_threads(octx, n_threads)))) {
        htp_fence_write(my_fence, fence_seq_entry, HTP_STATUS_INVAL_PARAMS);
        htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INVAL_PARAMS);
        return HTP_STATUS_INVAL_PARAMS;
    }

    // 1. entry barrier: every rank has finished its ALLREDUCE+ADD
    htp_fence_write(my_fence, fence_seq_entry, octx->status);
    for (uint32_t j = 0; j < n_ranks; j++) {
        if (j == rank) continue;
        const struct htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        while (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            if ((int32_t)(peer_seq - fence_seq_entry) >= 0) {
                if (peer_status > HTP_STATUS_OK) {
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    return peer_status;
                }
                break;
            }
            if (++spins > HTP_FENCE_TIMEOUT) {
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                return HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm volatile ("syncht" : : : "memory");

    // 2. normalize own rows and write each block to every norm dst
    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * w_spad = (uint8_t *) octx->ctx->vtcm_base;
    dma_sync_read(dma_q, w_spad, weight->data, row_bytes);

    if (tile_out) {
        // fp16 HMX tile output (GGML_HEXAGON_GATHER_ACT_TILE): 16-row blocks, whole blocks per thread (allreduce_rms_ag_tile_thread).
        struct htp_rms_ag_tile_context tctx;
        tctx.octx                 = octx;
        tctx.resid                = resid->data;
        tctx.w_spad               = w_spad;
        tctx.spad_base            = w_spad + row_bytes;
        tctx.r0                   = r0;
        tctx.nrows                = nrows;
        tctx.rows_per_thread      = RMS_AG_TILE_ROWS * ((nrows / RMS_AG_TILE_ROWS + n_threads - 1) / n_threads);
        tctx.ne0                  = ne0;
        tctx.row_bytes            = row_bytes;
        tctx.n_dsts               = n_ranks;
        tctx.vtcm_size_per_thread = (4 * RMS_AG_TILE_ROWS + 2) * row_bytes;
        tctx.epsilon              = epsilon;
        if (n_threads > 1) {
            work_queue_run(octx->ctx->work_queue, allreduce_rms_ag_tile_thread, &tctx, n_threads);
        } else {
            allreduce_rms_ag_tile_thread(1, 0, &tctx);
        }
    } else if (mt) {
        struct htp_rms_ag_context rctx;
        rctx.octx                 = octx;
        rctx.resid                = resid->data;
        rctx.w_spad               = w_spad;
        rctx.spad_base            = w_spad + row_bytes;
        rctx.r0                   = r0;
        rctx.nrows                = nrows;
        rctx.rows_per_thread      = (nrows + n_threads - 1) / n_threads;
        rctx.blk_rows             = blk_rows;
        rctx.ne0                  = ne0;
        rctx.row_bytes            = row_bytes;
        rctx.n_dsts               = n_ranks;
        rctx.vtcm_size_per_thread = 4 * blk_rows * row_bytes;
        rctx.epsilon              = epsilon;
        work_queue_run(octx->ctx->work_queue, allreduce_rms_ag_thread, &rctx, n_threads);
    } else {
        uint8_t * s_spad = w_spad + row_bytes;
        uint8_t * d_spad = s_spad + (size_t) blk_rows * row_bytes;
        for (uint32_t r = 0; r < nrows; r += blk_rows) {
            const uint32_t cur   = MIN(blk_rows, nrows - r);
            const uint32_t bytes = cur * row_bytes;
            const dma_addr_t off = (dma_addr_t) (r0 + r) * row_bytes;
            dma_sync_read(dma_q, s_spad, resid->data + off, bytes);
            for (uint32_t i = 0; i < cur; i++) {
                hvx_fast_rms_norm_mul_f32(s_spad + (size_t) i * row_bytes, w_spad, d_spad + (size_t) i * row_bytes, (int) ne0, epsilon);
            }
            for (uint32_t d = 0; d < n_ranks; d++) {
                dma_sync_write(dma_q, octx->dsts[d]->data + off, d_spad, bytes);
            }
        }
        dma_queue_flush(dma_q);
    }

    // 3. exit barrier: every rank has written its rows to every norm dst
    htp_fence_write(my_fence, fence_seq_exit, octx->status);
    for (uint32_t j = 0; j < n_ranks; j++) {
        if (j == rank) continue;
        const struct htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        while (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            if ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                if (peer_status > HTP_STATUS_OK) {
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    return peer_status;
                }
                break;
            }
            if (++spins > HTP_FENCE_TIMEOUT) {
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                return HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm volatile ("syncht" : : : "memory");
    return octx->status;
}

// AE-03 (GGML_HEXAGON_AR_RMS_FUSE=1): the owner-row ALLREDUCE+ADD of this rank and the RMS_NORM_MUL of the same rows, in one op. Each block of ADD_RMS_BLOCK_ROWS rows is reduced into VTCM and normalized there, so the residual is not read back and the boundary takes two fence barriers (entry, exit) instead of four. Inputs: [partial x n, fence x n, residual, weight]; outputs: [residual rows (dsts[0]), norm x n (dsts[1 + d])]. A slot set is n partial blocks, the residual block (the normalized rows overwrite it) and, for tiles, one staging block.
// The sum of an element is the AR reduce order ((p0 + p1) + p2) + res, and a normalized row is hvx_fast_rms_norm_mul_f32 of that sum: the same functions as the separate ops, so the bits are identical. Pipeline per thread, slot set s = b & 1: wait loads(b); reduce and normalize block b; wait writes(b - 1); start loads(b + 1); start writes(b). The norm outputs are fp32 rows, or fp16 HMX tiles for HTP_ALLREDUCE_KERNEL_ADD_RMS_TILE. Partial 0 receives the sum, which is the residual output of the block.
#define ADD_RMS_BLOCK_ROWS 16
#define ADD_RMS_MAX_RANKS  3   // 1 + n_ranks outputs must fit HTP_OP_MAX_OUTPUTS

struct htp_add_rms_context {
    struct htp_ops_context * octx;
    dma_addr_t               partial[ADD_RMS_MAX_RANKS];
    dma_addr_t               resid_in;
    dma_addr_t               resid_out;
    dma_addr_t               norm_out[ADD_RMS_MAX_RANKS];
    uint8_t *                w_spad;
    uint8_t *                spad_base;
    uint8_t *                res_base;       // VrAr: resident residual rows of this rank (VTCM); row 0 = first owner row
    bool                     res_in;         // resid_in rows are res_base (no load DMA)
    bool                     res_out;        // resid_out rows go to res_base (no store DMA)
    uint32_t                 r0;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 block_rows;     // rows per block: 16, or 8 (GGML_HEXAGON_AR_BLOCK_ROWS)
    uint32_t                 chk;
    volatile uint32_t *      chk_flags;
    uint32_t                 ne0;
    uint32_t                 row_bytes;
    uint32_t                 n_ranks;
    uint32_t                 vtcm_size_per_thread;
    bool                     tile;
    bool                     partial_f16;   // fp16 partial rows (HTP_ALLREDUCE_PARTIAL_F16): raw rows in the top half of each slot
    float                    epsilon;
    uint32_t                 rank;
    dma_addr_t               w_src;          // weight row (DDR), loaded into w_spad by thread 0
    volatile uint32_t        gate;           // 0 = entry fence pending, 1 = passed (or failed: gate_status)
    int                      gate_status;
};

#define ADD_RMS_LOADS_ALL   0
#define ADD_RMS_LOADS_LOCAL 1   // residual rows and this rank's own partial: written by this device before the op, no peer involved
#define ADD_RMS_LOADS_PEERS 2   // the other ranks' partials: valid after the entry fence

// Loads of one block into slot set `set`: the residual rows first, then the n partial rows. The queue is FIFO, so LOCAL then PEERS pops
// like ALL.
static void add_rms_push_loads_part(dma_queue * dma_q, const struct htp_add_rms_context * a, uint8_t * set, uint32_t row, uint32_t rows, int part) {
    const size_t half  = (size_t) a->block_rows * a->row_bytes;
    const size_t bytes = (size_t) rows * a->row_bytes;
    if (!a->res_in && part != ADD_RMS_LOADS_PEERS) {
        dma_queue_push(dma_q, dma_make_data(set + (size_t) a->n_ranks * half, a->resid_in + (dma_addr_t) row * a->row_bytes), bytes, bytes, bytes, 1);
    }
    for (uint32_t s = 0; s < a->n_ranks; s++) {
        if ((s == a->rank) != (part == ADD_RMS_LOADS_LOCAL) && part != ADD_RMS_LOADS_ALL) {
            continue;
        }
        if (a->partial_f16) {
            // fp16 partial row i lives at the START of fp32 row i of the partial buffer (producer: fp32 row stride). A compact layout (row stride ne0 halves) made norm row i overwrite partial rows 2i and 2i+1, owned by other ranks/blocks that may still be waiting to be loaded (NaN as soon as a thread loads a third block late).
            // A norm output (fp32 row i of a buffer that may alias a partial: ggml-alloc reuse) then overwrites only partial row i, which only its owner rank reads.
            dma_queue_push(dma_q, dma_make_data(set + (size_t) s * half + half / 2, a->partial[s] + (dma_addr_t) row * a->row_bytes),
                           a->row_bytes / 2, a->row_bytes, a->row_bytes / 2, rows);
        } else {
            dma_queue_push(dma_q, dma_make_data(set + (size_t) s * half, a->partial[s] + (dma_addr_t) row * a->row_bytes), bytes, bytes, bytes, 1);
        }
    }
}

static inline void add_rms_push_loads(dma_queue * dma_q, const struct htp_add_rms_context * a, uint8_t * set, uint32_t row, uint32_t rows) {
    add_rms_push_loads_part(dma_q, a, set, row, rows, ADD_RMS_LOADS_ALL);
}

// Writes of one block from slot set `set`: the residual rows, then one norm output per rank.
static void add_rms_push_writes(dma_queue * dma_q, const struct htp_add_rms_context * a, uint8_t * set, uint32_t row, uint32_t rows) {
    const size_t half  = (size_t) a->block_rows * a->row_bytes;
    const size_t bytes = (size_t) rows * a->row_bytes;
    if (!a->res_out) {
        dma_queue_push(dma_q, dma_make_data(a->resid_out + (dma_addr_t) row * a->row_bytes, set), bytes, bytes, bytes, 1);
    }
    if (!a->tile) {
        for (uint32_t d = 0; d < a->n_ranks; d++) {
            dma_queue_push(dma_q, dma_make_data(a->norm_out[d] + (dma_addr_t) row * a->row_bytes, set + (size_t) a->n_ranks * half), bytes, bytes, bytes, 1);
        }
    } else {
        // this block is a slice (hoff) of every fp16 tile row: kt pieces of block_rows * 64 bytes (1024 for 16 rows, 512 for 8) at a
        // 2048-byte stride (as allreduce_rms_ag_tile_thread)
        const uint32_t kt    = a->ne0 / HTP_MM_HMX_TILE_N_COLS;
        const uint32_t piece = a->block_rows * 64;
        const uint32_t hoff  = ((row % HTP_MM_HMX_TILE_N_ROWS) / a->block_rows) * piece;
        const uint8_t * g    = set + (size_t) (a->n_ranks + 1) * half;
        for (uint32_t d = 0; d < a->n_ranks; d++) {
            const dma_addr_t dst = a->norm_out[d] + (dma_addr_t) (row / HTP_MM_HMX_TILE_N_ROWS) * kt * HTP_MM_HMX_TILE_N_ELMS * sizeof(__fp16) + hoff;
            dma_queue_push(dma_q, dma_make_data(dst, g + hoff), 2048, 2048, piece, kt);
        }
    }
}

// Jobs the block pipeline pops: one per DMA pushed by add_rms_push_loads and add_rms_push_writes.
static inline uint32_t add_rms_loads_per_block(const struct htp_add_rms_context * a) { return a->n_ranks + (a->res_in ? 0 : 1); }
static inline uint32_t add_rms_writes_per_block(const struct htp_add_rms_context * a) { return a->n_ranks + (a->res_out ? 0 : 1); }

// fp16 partial rows (HTP_ALLREDUCE_PARTIAL_F16): the raw fp16 values of one partial (n_el contiguous halves) are widened exactly (x1.0) and either written to the fp32 accumulator (init) or added to it.
// Raw vector q is read before its two fp32 vectors are written; for the partial in the accumulator's own slot the raw vectors start at fp32 vector 640, so a write only reaches raw vectors already read.
static void add_rms_f16_accum(uint8_t * acc, const uint8_t * raw, uint32_t n_el, bool init) {
    const HVX_Vector one = hvx_vec_splat_f16(1.0);
    const uint32_t nvec = n_el / 64;   // 64 halves per vector
    _Pragma("clang loop pipeline(disable)")
    for (uint32_t q = 0; q < nvec; q++) {
        const HVX_Vector v = ((const HVX_Vector *) raw)[q];
        const HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v), one);
        const HVX_Vector lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
        const HVX_Vector hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
        HVX_Vector * d = (HVX_Vector *) acc;
        if (init) {
            d[2 * q]     = lo;
            d[2 * q + 1] = hi;
        } else {
            d[2 * q]     = hvx_vec_add_f32_f32(d[2 * q], lo);
            d[2 * q + 1] = hvx_vec_add_f32_f32(d[2 * q + 1], hi);
        }
    }
}

static int add_rms_wait_peers(struct htp_ops_context * octx, uint32_t rank, uint32_t n_ranks, uint32_t seq);

static void allreduce_add_rms_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_add_rms_context * a = (struct htp_add_rms_context *) data;
    struct htp_ops_context * octx = a->octx;

    const uint32_t BR     = a->block_rows;
    // Work spread evenly. Tile output needs whole blocks aligned to the tile row; fp32 output has no such constraint, so the rows are spread
    // (171 owner rows over 8 threads: 22,22,22,21,... rows) and only the last block of a thread is short.
    const uint32_t gran        = a->tile ? BR : 1;
    const uint32_t n_units_all = (a->nrows + gran - 1) / gran;
    const uint32_t u_base      = n_units_all / nth;
    const uint32_t u_extra     = n_units_all % nth;
    const uint32_t u_first     = ith * u_base + MIN(ith, u_extra);
    const uint32_t u_cnt       = u_base + (ith < u_extra ? 1 : 0);
    const uint32_t row_begin   = u_first * gran;
    if (u_cnt == 0 || row_begin >= a->nrows) return;
    const uint32_t n_rows = MIN(u_cnt * gran, a->nrows - row_begin);
    const uint32_t n_blk  = (n_rows + BR - 1) / BR;
    const uint32_t n      = a->n_ranks;
    const size_t   half   = (size_t) BR * a->row_bytes;
    // the tile staging block always holds a whole 32-row fp16 tile row (16 rows of fp32 bytes)
    const size_t   set_sz = (size_t) (n + 1) * half + (a->tile ? (size_t) 16 * a->row_bytes : 0);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    dma_queue * dma_q = octx->ctx->dma[ith];
    uint8_t * sets = a->spad_base + (size_t) ith * a->vtcm_size_per_thread;   // slot set 0, then slot set 1

    const uint32_t r_first = a->r0 + row_begin;
    // Entry fence overlapped with the loads of block 0 that do not depend on a peer: the weight row (thread 0), the residual rows and this
    // rank's own partial. Thread 0 waits for the peers, the others wait for thread 0; then the peers' partials of block 0 are pushed.
    if (ith == 0) {
        dma_queue_push(dma_q, dma_make_data(a->w_spad, a->w_src), a->row_bytes, a->row_bytes, a->row_bytes, 1);
    }
    add_rms_push_loads_part(dma_q, a, sets, r_first, MIN(BR, n_rows), ADD_RMS_LOADS_LOCAL);
    if (ith == 0) {
        const uint32_t seq = (uint32_t) octx->op_params[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        a->gate_status = add_rms_wait_peers(octx, a->rank, a->n_ranks, seq);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
        if (a->gate_status == HTP_STATUS_OK) {
            (void) dma_queue_pop(dma_q);   // the weight row: first in the queue, must have landed before any thread normalizes
        }
        __atomic_store_n(&a->gate, 1u, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&a->gate, __ATOMIC_ACQUIRE) == 0) {
            hex_pause();
        }
    }
    if (a->gate_status != HTP_STATUS_OK) {
        dma_queue_flush(dma_q);
        return;
    }
    add_rms_push_loads_part(dma_q, a, sets, r_first, MIN(BR, n_rows), ADD_RMS_LOADS_PEERS);
    if (n_blk > 1) {
        // block 1 takes the free second slot set now, so its loads run behind block 0's loads and overlap the reduce of block 0
        add_rms_push_loads(dma_q, a, sets + set_sz, r_first + BR, MIN(BR, n_rows - BR));
    }

    for (uint32_t b = 0; b < n_blk; b++) {
        uint8_t * set  = sets + (size_t) (b & 1) * set_sz;
        const uint32_t off  = b * BR;
        const uint32_t rows = MIN(BR, n_rows - off);
        const uint32_t row  = r_first + off;
        const uint32_t n_el = rows * a->ne0;
        uint8_t * res = set + (size_t) n * half;

        // 1. block b landed: the n partial blocks, then the residual block
        for (uint32_t s = 0; s < add_rms_loads_per_block(a); s++) {
            (void) dma_queue_pop(dma_q);
        }

        // 2. sum into partial 0, normalize the rows into the residual block, then fp16 tiles for ADD_RMS_TILE
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);
        // VrAr: the resident owner rows of this block (res_base is NULL without RES): the sum goes there with res_out, the residual is read there with res_in
        uint8_t * res_blk = a->res_base ? a->res_base + (size_t) (row_begin + off) * a->row_bytes : NULL;
        uint8_t * sum     = a->res_out ? res_blk : set;
        const uint8_t * rin = a->res_in ? res_blk : res;
        uint32_t raw_bad = 0;
        if (a->chk && a->partial_f16) {
            for (uint32_t s2 = 0; s2 < n && !raw_bad; s2++) {
                const uint16_t * rw = (const uint16_t *) (set + (size_t) s2 * half + half / 2);
                for (uint32_t i = 0; i < n_el; i++) { if ((rw[i] & 0x7c00) == 0x7c00) { raw_bad = 1; break; } }
            }
        }
        if (a->partial_f16) {
            // fp16 partials: partial 0 is widened into the accumulator, the others are added to it (same per-element order as below)
            add_rms_f16_accum(set, set + half / 2, n_el, true);
            for (uint32_t s = 1; s < n; s++) {
                add_rms_f16_accum(set, set + (size_t) s * half + half / 2, n_el, false);
            }
        } else {
            hvx_add_f32(set, set, set + half, n_el);
            for (uint32_t s = 2; s < n; s++) {
                hvx_add_f32(set, set, set + (size_t) s * half, n_el);
            }
        }
        if (a->chk) {
            uint32_t m = raw_bad ? 1u : 0u;
            const uint16_t * r0h = (const uint16_t *) (set + half / 2);
            (void) r0h;
            for (uint32_t s2 = 0; s2 < n; s2++) {
                const uint16_t * rw = (const uint16_t *) (set + (size_t) s2 * half + half / 2);
                (void) rw;
            }
            const float * f = (const float *) set; for (uint32_t i = 0; i < n_el; i++) { if (!(f[i] > -1e30f && f[i] < 1e30f)) { m |= 2u; break; } }
            const float * g = (const float *) rin; for (uint32_t i = 0; i < n_el; i++) { if (!(g[i] > -1e30f && g[i] < 1e30f)) { m |= 4u; break; } }
            if (m) {
                // first record only: (b, ith, flags, row)
                if (__sync_bool_compare_and_swap((uint32_t *) a->chk_flags, 0u, 1u)) {
                    ((volatile uint32_t *) a->chk_flags)[1] = (m << 24) | (b << 16) | (ith << 8) | (row & 0xff);
                    ((volatile uint32_t *) a->chk_flags)[2] = row;
                }
            }
        }
        hvx_add_f32(sum, set, rin, n_el);
        hvx_fast_rms_norm_mul_f32_rows_lm(sum, a->row_bytes, a->w_spad, res, a->row_bytes, rows, (int) a->ne0, a->epsilon, (int) ((octx->flags >> HTP_OPFLAGS_LM_SHIFT) & 3u));
        if (a->tile) {
            uint8_t * g = set + (size_t) (n + 1) * half;
            for (uint32_t i = 0; i + 1 < rows; i += 2) {
                act_tile_row_pair_store((__fp16 *) g, (const float *) (res + (size_t) i * a->row_bytes),
                                        (const float *) (res + (size_t) (i + 1) * a->row_bytes), (row + i) % HTP_MM_HMX_TILE_N_ROWS,
                                        a->ne0, a->ne0, true, true);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);

        // 3. the writes of block b-1 are done, so slot set (b-1) & 1 = (b+1) & 1 is free
        if (b >= 1) {
            for (uint32_t s = 0; s < add_rms_writes_per_block(a); s++) {
                (void) dma_queue_pop(dma_q);
            }
        }
        // 4. loads of block b+1 (block 1 was pushed in the prologue)
        if (b >= 1 && b + 1 < n_blk) {
            const uint32_t next_rows = MIN(BR, n_rows - (b + 1) * BR);
            add_rms_push_loads(dma_q, a, sets + (size_t) ((b + 1) & 1) * set_sz, row + BR, next_rows);
        }
        // 5. writes of block b
        add_rms_push_writes(dma_q, a, set, row, rows);
    }
    dma_queue_flush(dma_q);
}

// Waits until the fence of every peer has reached seq. Returns the first peer error, or HTP_STATUS_OK.
static int add_rms_wait_peers(struct htp_ops_context * octx, uint32_t rank, uint32_t n_ranks, uint32_t seq) {
    for (uint32_t j = 0; j < n_ranks; j++) {
        if (j == rank) continue;
        const struct htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        while (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_poll(peer_fence, &peer_seq, &peer_status);
            if ((int32_t)(peer_seq - seq) >= 0) {
                if (peer_status > HTP_STATUS_OK) {
                    return (int) peer_status;
                }
                break;
            }
            if (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: add-rms fence-wait TIMEOUT : rank %u waiting on %u seq 0x%x peer-seq 0x%x\n", rank, j, seq, peer_seq);
                return HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm volatile ("syncht" : : : "memory");
    return HTP_STATUS_OK;
}

static int op_allreduce_add_rms(struct htp_ops_context * octx, const struct htp_allreduce_kernel_params * kparams, uint32_t rank, uint32_t n_ranks) {
    const uint32_t fence_seq_entry = (uint32_t) octx->op_params[0];
    const uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];
    float epsilon = 0.0f;
    memcpy(&epsilon, &octx->op_params[2], sizeof(float));

    const bool     tile       = kparams->kernel_type == HTP_ALLREDUCE_KERNEL_ADD_RMS_TILE;
    const uint32_t ne0        = (uint32_t) kparams->ne0;
    const uint32_t row_bytes  = ne0 * (uint32_t) sizeof(float);
    const uint32_t n_threads  = (uint32_t) kparams->n_threads;
    const uint32_t per_thread = (uint32_t) kparams->vtcm_size_per_thread;
    const uint32_t r0         = (uint32_t) kparams->rank_elem_start;
    const uint32_t nrows      = (uint32_t) kparams->rank_nelem;
    const uint32_t mode      = (uint32_t) kparams->resid_mode;
    const bool     res_in    = (mode & HTP_ALLREDUCE_RES_IN) != 0;
    const bool     res_out   = (mode & HTP_ALLREDUCE_RES_OUT) != 0;
    const bool     pin       = (mode & HTP_ALLREDUCE_RES_PIN) != 0;
    const bool     flush     = (mode & HTP_ALLREDUCE_RES_FLUSH) != 0;
    const bool     unpin     = (mode & HTP_ALLREDUCE_RES_UNPIN) != 0;
    const size_t   res_bytes = (size_t) nrows * row_bytes;

    const struct htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->data;
    struct htp_thread_trace * tr0 = &octx->ctx->trace[0];

    const bool tile_ok = !tile || ((nrows % (uint32_t) kparams->block_elems) == 0 && (ne0 % HTP_MM_HMX_TILE_N_COLS) == 0 &&
                                   (octx->dsts[1]->flags & HTP_TENSOR_ACT_TILE) != 0);
    const bool res_ok = (mode & HTP_ALLREDUCE_RES_ANY) == 0 ||
                        ((size_t) kparams->resid_off >= (size_t) kparams->vtcm_size &&
                         (size_t) kparams->resid_off + res_bytes <= octx->ctx->vtcm_size_full);
    if (n_ranks > ADD_RMS_MAX_RANKS || !tile_ok || !res_ok || !htp_ops_context_set_n_threads(octx, n_threads) ||
        (size_t) kparams->vtcm_size > octx->ctx->vtcm_size || (size_t) kparams->vtcm_size < row_bytes + (size_t) n_threads * per_thread) {
        htp_fence_write(my_fence, fence_seq_entry, HTP_STATUS_INVAL_PARAMS);
        htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INVAL_PARAMS);
        return HTP_STATUS_INVAL_PARAMS;
    }

    // 1. entry barrier: every rank has finished its previous op, so the partial sums are complete. The wait itself runs in worker 0 of the
    // reduce (allreduce_add_rms_thread), overlapped with the loads of the data that is local to this device.
    htp_fence_write(my_fence, fence_seq_entry, octx->status);

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * w_spad = (uint8_t *) octx->ctx->vtcm_base;

    struct htp_add_rms_context actx;
    actx.octx = octx;
    for (uint32_t s = 0; s < n_ranks; s++) {
        actx.partial[s] = octx->src[s]->data;
        actx.norm_out[s] = octx->dsts[1 + s]->data;
    }
    actx.resid_in            = octx->src[2 * n_ranks]->data;
    actx.resid_out           = octx->dsts[0]->data;
    actx.w_spad              = w_spad;
    actx.w_src               = octx->src[2 * n_ranks + 1]->data;
    actx.rank                = rank;
    actx.gate                = 0;
    actx.gate_status         = HTP_STATUS_OK;
    actx.spad_base           = w_spad + row_bytes;
    actx.r0                  = r0;
    actx.nrows               = nrows;
    actx.rows_per_thread     = (uint32_t) kparams->elems_per_thread;
    actx.block_rows          = (uint32_t) kparams->block_elems;
    static volatile uint32_t chk_flags_store[4]; chk_flags_store[0]=chk_flags_store[1]=chk_flags_store[2]=0;
    actx.chk = (uint32_t) ((kparams->resid_mode >> 20) & 1);
    actx.chk_flags = chk_flags_store;
    actx.ne0                 = ne0;
    actx.row_bytes           = row_bytes;
    actx.n_ranks             = n_ranks;
    actx.vtcm_size_per_thread = per_thread;
    actx.tile                = tile;
    actx.epsilon             = epsilon;
    actx.res_base            = (mode & HTP_ALLREDUCE_RES_ANY) != 0 ? octx->ctx->vtcm_base + (size_t) kparams->resid_off : NULL;
    actx.res_in              = res_in;
    actx.res_out             = res_out;
    actx.partial_f16         = (mode & HTP_ALLREDUCE_PARTIAL_F16) != 0;
    int status;
    if (nrows > 0) {
        if (n_threads > 1) {
            work_queue_run(octx->ctx->work_queue, allreduce_add_rms_thread, &actx, n_threads);
        } else {
            allreduce_add_rms_thread(1, 0, &actx);
        }
        status = actx.gate_status;
    } else {
        htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
        dma_sync_read(dma_q, w_spad, actx.w_src, row_bytes);
        status = add_rms_wait_peers(octx, rank, n_ranks, fence_seq_entry);
        htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
    }
    if (status != HTP_STATUS_OK) {
        htp_fence_write(my_fence, fence_seq_exit, status);
        return status;
    }
    if (pin) {
        octx->ctx->vtcm_size = (size_t) kparams->resid_off;  // VrAr: the budget stays below the resident rows until the unpin op
    }

    // VrAr: flush the resident rows to resid_out (last boundary of the batch), then restore the VTCM budget
    if (flush && nrows > 0) {
        for (uint32_t off = 0; off < nrows; off += ADD_RMS_BLOCK_ROWS) {
            const uint32_t blk   = MIN(ADD_RMS_BLOCK_ROWS, nrows - off);
            const size_t   bytes = (size_t) blk * row_bytes;
            dma_queue_push(dma_q, dma_make_data(actx.resid_out + (dma_addr_t) (r0 + off) * row_bytes, actx.res_base + (size_t) off * row_bytes), bytes, bytes, bytes, 1);
            (void) dma_queue_pop(dma_q);
        }
    }
    if (unpin) {
        octx->ctx->vtcm_size = octx->ctx->vtcm_size_full;
    }

    // 3. exit barrier: every rank has written its residual rows and its norm rows to every norm dst
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
    htp_fence_write(my_fence, fence_seq_exit, octx->status);
    status = add_rms_wait_peers(octx, rank, n_ranks, fence_seq_exit);
    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
    return status != HTP_STATUS_OK ? status : octx->status;
}

int op_allreduce(struct htp_ops_context * octx) {
    if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx > 0) {
        return HTP_STATUS_OK;
    }

    const struct htp_allreduce_kernel_params * kparams = (const struct htp_allreduce_kernel_params *) octx->kernel_params;
    const struct htp_tensor * dst = octx->dst;

    const uint32_t rank    = (uint32_t) kparams->rank;
    const uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    if (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t fence_seq_entry = (uint32_t) octx->op_params[0];
    const uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    const struct htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->data;

    if (kparams->kernel_type == HTP_ALLREDUCE_KERNEL_RMS_AG) {
        return op_allreduce_rms_ag(octx, kparams, rank, n_ranks);
    }
    if (kparams->kernel_type == HTP_ALLREDUCE_KERNEL_ADD_RMS || kparams->kernel_type == HTP_ALLREDUCE_KERNEL_ADD_RMS_TILE) {
        return op_allreduce_add_rms(octx, kparams, rank, n_ranks);
    }
    const int status = validate_allreduce(octx, kparams, n_ranks);
    if (status != HTP_STATUS_OK) {
        if (status == HTP_STATUS_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, status);
        return status;
    }

    const bool has_add = (octx->op == HTP_OP_ALLREDUCE_ADD);
    const uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    // 1. Entry Barrier: Synchronize all ranks before reading
    struct htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    htp_fence_write(my_fence, fence_seq_entry, octx->status);

    for (uint32_t j = 0; j < n_ranks; j++) {
        if (j == rank) continue;
        const struct htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        while (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            if ((int32_t)(peer_seq - fence_seq_entry) >= 0) {
                if (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                    return peer_status;
                }
                break;
            }
            if (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_entry, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                return HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm volatile ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    // 2. Multi-threaded Reduction across assigned rank chunk
    if (nelem > 0) {
        const uint32_t n_threads            = (uint32_t) kparams->n_threads;
        const uint32_t block_elems          = (uint32_t) kparams->block_elems;
        const uint32_t elems_per_thread     = (uint32_t) kparams->elems_per_thread;
        const uint32_t vtcm_size_per_thread = (uint32_t) kparams->vtcm_size_per_thread;

        struct htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.row_size_aligned     = (uint32_t) kparams->row_size_aligned;
        actx.rank_elem_start      = (uint32_t) kparams->rank_elem_start;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_per_thread     = elems_per_thread;
        actx.block_elems          = block_elems;
        actx.vtcm_size_per_thread = vtcm_size_per_thread;
        actx.is_row_bcast         = (kparams->is_row_bcast != 0);

        work_queue_func_t reduce_fun = NULL;
        switch (kparams->kernel_type) {
            case HTP_ALLREDUCE_KERNEL_DMA_1D:
                if (has_add) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_add_f16 : allreduce_thread_dma_1d_add_f32;
                } else {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                break;
            case HTP_ALLREDUCE_KERNEL_DMA_2D:
                if (has_add) {
                    if (kparams->is_row_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_bcast_f16 : allreduce_thread_dma_2d_add_bcast_f32;
                    } else {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_add_f16 : allreduce_thread_dma_2d_add_f32;
                    }
                } else {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                break;
            default:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_NO_SUPPORT);
                return HTP_STATUS_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_base;
        for (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_base[s] = vtcm_ptr;
            vtcm_ptr += n_threads * vtcm_size_per_thread;
        }
        actx.dst_spad_base = vtcm_ptr;
        vtcm_ptr += n_threads * vtcm_size_per_thread;
        if (has_add) {
            actx.res_spad_base = vtcm_ptr;
            vtcm_ptr += (actx.is_row_bcast ? 1 : n_threads) * vtcm_size_per_thread;
        }

        if (has_add && actx.is_row_bcast) {
            const dma_addr_t r_ddr = octx->src[2 * n_ranks]->data;
            const uint32_t row_bytes = actx.ne0 * (dst->type == HTP_TYPE_F16 ? sizeof(__fp16) : sizeof(float));
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_push(dma_q, dma_make_data(actx.res_spad_base, r_ddr), actx.row_size_aligned, 0, row_bytes, 1);
            dma_queue_pop(dma_q);
        }

        work_queue_run(octx->ctx->work_queue, reduce_fun, &actx, n_threads);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    htp_fence_write(my_fence, fence_seq_exit, octx->status);

    for (uint32_t j = 0; j < n_ranks; j++) {
        if (j == rank) continue;
        const struct htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        while (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            if ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                if (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    return peer_status;
                }
                break;
            }
            if (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                return HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm volatile ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    return octx->status;
}
