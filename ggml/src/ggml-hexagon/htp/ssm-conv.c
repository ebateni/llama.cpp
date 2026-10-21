#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_ps.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <qurt_thread.h>
#include <stdatomic.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "dma-queue.h"
#include "hex-profile.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-utils.h"
#include "ssm-conv.h"

struct htp_ssm_conv_context {
    struct htp_ops_context *                  octx;
    const struct htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_per_thread;
    uint32_t                                  d_inner_tile;
    uint32_t                                  row_start;
    uint32_t                                  nrows;
    // CONCAT+SSM_CONV(+SILU) fusion (try_fuse_concat_ssm_conv, ggml-hexagon.cpp): when fused is set, octx->src[0] is only the small conv_states tail ([n_state_rows, d_inner, n_s]) and octx->src[2] is qkv_mixed_orig (token-major, [d_inner, n_qkv_tokens, n_s]) -- CONCAT's own DDR materialization of conv_input is skipped entirely; the two disjoint sources are read directly in the prefill worker.
    bool                                       fused;
    uint32_t                                   n_state_rows; // conv_kernel_size - 1, i.e. src0->ne[0] in fused mode
    bool                                       apply_silu;   // fuse silu(x) = x*sigmoid(x) into the output-store (prefill-only)
};

// In-register 32x32 fp32 transpose using std 5-stage HVX vshuff butterfly.
static inline void hvx_transpose_32x32_f32(HVX_Vector m[32]) {
    HVX_Vector tmp[32];

    // Stage 0 (R = -4): pair (2i, 2i+1) for i = 0..15. m -> tmp.
    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    // Stage 1 (R = -8): per block of 4, pair (b+0, b+2) and (b+1, b+3). tmp -> m.
    for (int b = 0; b < 32; b += 4) {
        HVX_VectorPair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_VectorPair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    // Stage 2 (R = -16): per block of 8, pair (b+i, b+i+4) for i = 0..3. m -> tmp.
    for (int b = 0; b < 32; b += 8) {
        for (int i = 0; i < 4; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 3 (R = -32): per block of 16, pair (b+i, b+i+8) for i = 0..7. tmp -> m.
    for (int b = 0; b < 32; b += 16) {
        for (int i = 0; i < 8; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 4 (R = -64): pair (i, i+16) for i = 0..15. m -> tmp -> m.
    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    for (int i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// HVX deinterleave for d_conv == 4: channel-major raw VTCM -> tap-major T VTCM
static inline void hvx_ssm_conv_unpack_to_T_4(const float * raw, float * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride) {
    for (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
        HVX_Vector v0 = *(const HVX_Vector *)(raw + (cb + 0) * 4);
        HVX_Vector v1 = *(const HVX_Vector *)(raw + (cb + 8) * 4);
        HVX_Vector v2 = *(const HVX_Vector *)(raw + (cb + 16) * 4);
        HVX_Vector v3 = *(const HVX_Vector *)(raw + (cb + 24) * 4);

        HVX_VectorPair p01 = Q6_W_vdeal_VVR(v1, v0, -4);
        HVX_VectorPair p23 = Q6_W_vdeal_VVR(v3, v2, -4);

        HVX_VectorPair p_w02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_VectorPair p_w13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        *(HVX_Vector *)(T + 0 * d_inner_stride + cb) = Q6_V_lo_W(p_w02);
        *(HVX_Vector *)(T + 1 * d_inner_stride + cb) = Q6_V_lo_W(p_w13);
        *(HVX_Vector *)(T + 2 * d_inner_stride + cb) = Q6_V_hi_W(p_w02);
        *(HVX_Vector *)(T + 3 * d_inner_stride + cb) = Q6_V_hi_W(p_w13);
    }
}

// HVX transpose for general d_conv <= 32: channel-major raw VTCM -> tap-major T VTCM
static inline void hvx_ssm_conv_unpack_to_T_gen(const float * raw, float * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    uint32_t __attribute__((aligned(VLEN))) mask_buf[VLEN_FP32] = { 0 };
    for (uint32_t j = 0; j < d_conv; ++j) {
        mask_buf[j] = 0xFFFFFFFF;
    }
    const HVX_Vector mask = *(const HVX_Vector *) mask_buf;

    for (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
        const uint32_t cb_n = MIN(VLEN_FP32, d_inner_per_thread - cb);
        HVX_Vector sub[32];
        for (uint32_t r = 0; r < cb_n; ++r) {
            const float * ch_ptr = raw + (cb + r) * d_conv;
            sub[r] = Q6_V_vand_VV(*(const HVX_UVector *) ch_ptr, mask);
        }
        for (uint32_t r = cb_n; r < 32; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        for (uint32_t j = 0; j < d_conv; ++j) {
            *(HVX_Vector *)(T + j * d_inner_stride + cb) = sub[j];
        }
    }
}

static inline void hvx_ssm_conv_unpack_to_T(const float * raw, float * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    if (d_conv == 4 && (d_inner_per_thread % VLEN_FP32 == 0)) {
        hvx_ssm_conv_unpack_to_T_4(raw, T, d_inner_per_thread, d_inner_stride);
    } else {
        hvx_ssm_conv_unpack_to_T_gen(raw, T, d_inner_per_thread, d_inner_stride, d_conv);
    }
}

// HVX 32x32 src0 transpose for prefill: src0 {tile_n, ncs} (VTCM) -> src0_T {ncs, d_inner_tile} (VTCM)
static inline void transpose_src0_block(const float * src0_block,
                                        uint32_t      ncs,
                                        uint32_t      cb_n,
                                        uint32_t      d_inner_tile,
                                        float *       src0_T_block_dst,
                                        uint32_t      cb) {
    const uint32_t T_TILE = VLEN_FP32;

    HVX_Vector __attribute__((aligned(VLEN))) sub[32];

    for (uint32_t t0 = 0; t0 < ncs; t0 += T_TILE) {
        const uint32_t t_n = MIN(T_TILE, ncs - t0);

        uint32_t __attribute__((aligned(VLEN))) mask_buf[VLEN_FP32] = { 0 };
        for (uint32_t k = 0; k < t_n; ++k) {
            mask_buf[k] = 0xFFFFFFFF;
        }
        const HVX_Vector mask = *(const HVX_Vector *) mask_buf;

        for (uint32_t r = 0; r < cb_n; ++r) {
            const float * src_row = src0_block + r * ncs + t0;
            sub[r] = (t_n == T_TILE) ? *(const HVX_UVector *) src_row : Q6_V_vand_VV(*(const HVX_UVector *) src_row, mask);
        }
        for (uint32_t r = cb_n; r < T_TILE; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        for (uint32_t r = 0; r < t_n; ++r) {
            float * dst = src0_T_block_dst + (t0 + r) * d_inner_tile + cb;
            if (cb_n == T_TILE) {
                *(HVX_UVector *) dst = sub[r];
            } else {
                hvx_vec_store_u(dst, cb_n * sizeof(float), sub[r]);
            }
        }
    }
}

// CONCAT+SSM_CONV(+SILU) fusion (try_fuse_concat_ssm_conv, ggml-hexagon.cpp): builds the same {d_inner_tile, ncs} VTCM destination layout as the unfused transpose_src0_block above, but for the qkv_mixed_orig portion (time >= n_state_rows) it is a straight strided DDR->VTCM DMA copy -- no transpose at all, since qkv_mixed_orig is already token-major/channel-contiguous, exactly the layout src0_T's channel axis wants. 3) needs an actual (scalar) gather- transpose, done by the sibling function below.
// Only the tiny conv_states tail (n_state_rows rows, e.g. DMA-pipelined: destinations are disjoint per column block, so this pushes the transfer to the DMA queue and returns immediately (caller must dma_queue_pop() once per push, matching this file's existing push/pop discipline elsewhere) instead of blocking on a synchronous unaligned-HVX-load-per-row copy -- letting the DMA engine's own transfer run concurrently with the state-rows scalar gather below.
static inline void transpose_src0_block_fused_dma_push(dma_queue *   dma_q,
                                                        const float * qkv_block,
                                                        uint32_t      qkv_stride_token,
                                                        uint32_t      n_state_rows,
                                                        uint32_t      ncs,
                                                        uint32_t      cb_n,
                                                        uint32_t      d_inner_tile,
                                                        float *       src0_T_block_dst,
                                                        uint32_t      cb) {
    const uint32_t n_qkv_rows = ncs - n_state_rows;
    if (n_qkv_rows == 0) {
        return;
    }
    float * dst_row0 = src0_T_block_dst + n_state_rows * d_inner_tile + cb;
    dma_queue_push(dma_q, dma_make_data(dst_row0, qkv_block),
                   d_inner_tile * sizeof(float), qkv_stride_token * sizeof(float),
                   cb_n * sizeof(float), n_qkv_rows);
}

// Scalar gather-transpose for the boundary (state-rows) part only -- paired with transpose_src0_block_fused_dma_push above, which handles the bulk qkv rows via DMA.
// n_state_rows is always < VLEN_FP32 (a handful of taps vs a 32-lane tile), so this only ever touches the first output tile and is at most a few dozen scalar stores total.
static inline void transpose_src0_state_rows_fused(const float * state_block,
                                                    uint32_t      state_stride_inner,
                                                    uint32_t      n_state_rows,
                                                    uint32_t      cb_n,
                                                    uint32_t      d_inner_tile,
                                                    float *       src0_T_block_dst,
                                                    uint32_t      cb) {
    for (uint32_t tt = 0; tt < n_state_rows; ++tt) {
        float * dst_row = src0_T_block_dst + tt * d_inner_tile + cb;
        for (uint32_t r = 0; r < cb_n; ++r) {
            dst_row[r] = state_block[r * state_stride_inner + tt];
        }
    }
}

// Single-row decode worker (n_t == 1)
static void ssm_conv_thread_f32_decode(unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_context *             scctx   = (struct htp_ssm_conv_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict src1 = octx->src[1];
    const struct htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    const uint32_t d_conv  = kparams->d_conv;
    const uint32_t d_inner = kparams->d_inner;
    const uint32_t n_s     = kparams->n_s;

    const uint32_t dr  = scctx->nrows_per_thread;
    const uint32_t ir0 = scctx->row_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    if (ir0 >= ir1) {
        return;
    }

    const uint32_t d_inner_per_thread = ir1 - ir0;
    const uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);

    const size_t src0_stride_seq_bytes = src0->nb[2];
    const size_t dst_stride_seq_bytes  = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    const size_t weight_bytes    = (size_t) d_inner_per_thread * d_conv * sizeof(float);
    const size_t weight_raw_size = hex_round_up(weight_bytes, 128);

    float * src1_raw = (float *) src1_spad_base;
    float * src1_T   = (float *) (src1_spad_base + weight_raw_size);

    float * src0_raw = (float *) src0_spad_base;
    float * src0_T   = (float *) (src0_spad_base + weight_raw_size);

    float * dst_spad = (float *) dst_spad_base;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    // 1. Fetch weights src1 from DDR into VTCM via DMA (DMA64-safe)
    const dma_addr_t src1_ddr = src1->data + ir0 * d_conv * sizeof(float);
    dma_queue_push(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);
    dma_queue_pop(dma_q);

    // 2. Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    const size_t input_bytes  = (size_t) d_inner_per_thread * d_conv * sizeof(float);
    const size_t output_bytes = (size_t) d_inner_per_thread * sizeof(float);

    // 3. Process each sequence
    for (uint32_t s = 0; s < n_s; ++s) {
        const dma_addr_t src0_ddr = src0->data + s * src0_stride_seq_bytes + ir0 * d_conv * sizeof(float);
        dma_queue_push(dma_q, dma_make_data((uint8_t *) src0_raw, src0_ddr), input_bytes, input_bytes, input_bytes, 1);
        dma_queue_pop(dma_q);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) s);
        hvx_ssm_conv_unpack_to_T(src0_raw, src0_T, d_inner_per_thread, d_inner_stride, d_conv);

        for (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
            const uint32_t cb_n = MIN(VLEN_FP32, d_inner_per_thread - cb);
            HVX_Vector acc = hvx_vec_splat_f32(0.0f);
            for (uint32_t j = 0; j < d_conv; ++j) {
                HVX_Vector x = *(const HVX_Vector *)(src0_T + j * d_inner_stride + cb);
                HVX_Vector w = *(const HVX_Vector *)(src1_T + j * d_inner_stride + cb);
                acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
            }
            HVX_Vector y = Q6_Vsf_equals_Vqf32(acc);
            if (cb_n == VLEN_FP32) {
                *(HVX_Vector *)(dst_spad + cb) = y;
            } else {
                hvx_vec_store_u(dst_spad + cb, cb_n * sizeof(float), y);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) s);

        const dma_addr_t dst_ddr = dst->data + s * dst_stride_seq_bytes + ir0 * sizeof(float);
        dma_queue_push(dma_q, dma_make_data(dst_ddr, (uint8_t *) dst_spad), output_bytes, output_bytes, output_bytes, 1);
        dma_queue_pop(dma_q);
    }

    FARF(HIGH, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

// Multi-token prefill worker (n_t > 1)
static uint32_t ssm_conv_prefill_enqueue_input(
        dma_queue * dma_q,
        bool fused,
        const struct htp_tensor * src0,
        const struct htp_tensor * src2,
        uint8_t * src0_spad_base,
        size_t src0_slot_size,
        size_t src0_tile_raw_bytes,
        uint32_t d_inner_tile,
        uint32_t tile_idx,
        uint32_t d_inner_per_thread,
        uint32_t ir0,
        uint32_t i3,
        uint32_t ncs,
        uint32_t n_state_rows,
        size_t src0_stride_inner_bytes,
        size_t src0_stride_seq_bytes,
        size_t qkv_stride_token_bytes,
        size_t qkv_stride_seq_bytes) {
    const uint32_t tile_off = tile_idx * d_inner_tile;
    const uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);
    uint8_t * slot_base = src0_spad_base + (size_t) (fused ? (tile_idx & 1) : 0) * src0_slot_size;
    float * src0_tile_raw = (float *) slot_base;
    float * src0_T = (float *) (slot_base + src0_tile_raw_bytes);
    uint32_t count = 0;

    if (fused) {
        const uint32_t n_qkv_rows = ncs - n_state_rows;
        if (n_state_rows > 0) {
            const dma_addr_t state_tile_ddr = src0->data +
                i3 * src0_stride_seq_bytes + (ir0 + tile_off) * src0_stride_inner_bytes;
            const size_t state_row_bytes = n_state_rows * sizeof(float);

            dma_queue_push(dma_q, dma_make_data((uint8_t *) src0_tile_raw, state_tile_ddr),
                           state_row_bytes, src0_stride_inner_bytes, state_row_bytes, tile_n);
            count++;
        }

        if (n_qkv_rows > 0) {
            const dma_addr_t qkv_tile_ddr = src2->data +
                i3 * qkv_stride_seq_bytes + (ir0 + tile_off) * sizeof(float);
            dma_queue_push(dma_q, dma_make_data((uint8_t *) (src0_T + n_state_rows * d_inner_tile), qkv_tile_ddr),
                           d_inner_tile * sizeof(float), qkv_stride_token_bytes,
                           tile_n * sizeof(float), n_qkv_rows);
            count++;
        }
    } else {
        const dma_addr_t src0_tile_ddr = src0->data +
            i3 * src0_stride_seq_bytes + (ir0 + tile_off) * src0_stride_inner_bytes;
        const size_t row_bytes = ncs * sizeof(float);

        dma_queue_push(dma_q, dma_make_data((uint8_t *) src0_tile_raw, src0_tile_ddr),
                       row_bytes, src0_stride_inner_bytes, row_bytes, tile_n);
        count++;
    }

    return count;
}

static inline void ssm_conv_prefill_wait_input(dma_queue * dma_q, uint32_t count) {
    while (count-- > 0) {
        dma_queue_pop(dma_q);
    }
}

static inline void ssm_conv_store_vec(float * p, uint32_t cb_n, HVX_Vector y) {
    if (cb_n == VLEN_FP32) {
        *(HVX_Vector *) p = y;
    } else {
        hvx_vec_store_u(p, cb_n * sizeof(float), y);
    }
}

static inline HVX_Vector ssm_conv_silu_vec(HVX_Vector y, HVX_Vector one, HVX_Vector max_exp, HVX_Vector min_exp) {
    HVX_Vector sig = hvx_vec_fast_sigmoid_f32_guard(y, one, max_exp, min_exp);
    return hvx_vec_mul_f32_f32(y, sig);
}

static inline HVX_Vector ssm_conv_mac_vec(HVX_Vector acc, HVX_Vector x, HVX_Vector w) {
    return Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
}

// SiLU in place over n_rows rows of cols floats (row stride out_stride), with the same per-vector operations as the SILU
// tail of the fused SSM_CONV. It runs after the MAC has stored y, as the original worker did, so the unrolled MAC keeps its schedule.
static void ssm_conv_silu_tile(float * out, uint32_t out_stride, uint32_t cols, uint32_t n_rows, HVX_Vector one,
                               HVX_Vector max_exp, HVX_Vector min_exp) {
    const uint32_t total_vecs = cols / VLEN_FP32;
    for (uint32_t u = 0; u < n_rows; ++u) {
        HVX_Vector * restrict v = (HVX_Vector *) (out + (size_t) u * out_stride);
        _Pragma("unroll(4)")
        for (uint32_t i = 0; i < total_vecs; i++) {
            HVX_Vector yv = v[i];
            v[i]          = ssm_conv_silu_vec(yv, one, max_exp, min_exp);
        }
    }
}

// Tile MAC (+ fused SiLU): out[u][c] = y or silu(y), y = sum_{j < d_conv} T[(u + j) * T_stride + c] * W[j * W_stride + c], for u < n_rows and c < cols. The d_conv == 4 path computes eight output rows per step: their dependent MAC and SiLU chains are independent, so issuing them together hides HVX latency (the one-row loop was latency bound; 4 rows measured 410 us, 8 rows 380 us for the same op), and the tap rows come from a sliding window (11 vector loads per 8 rows instead of 32).
// cols is a multiple of VLEN_FP32 in all callers. Per element this is the same operation sequence as the one-row loop it replaces (qf32 products added to a zero accumulator in ascending tap order, sf conversion, guarded SiLU), so the result is bit-identical.
static void ssm_conv_tile_mac(const float * T, uint32_t T_stride, const float * W, uint32_t W_stride, uint32_t d_conv,
                              uint32_t cols, uint32_t n_rows, float * out, uint32_t out_stride, bool silu,
                              HVX_Vector one, HVX_Vector max_exp, HVX_Vector min_exp) {
    const HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    uint32_t         u    = 0;

    if (d_conv == 4) {
        for (; u + 8 <= n_rows; u += 8) {
            const float * t = T + (size_t) u * T_stride;
            float *       o = out + (size_t) u * out_stride;
            for (uint32_t c = 0; c < cols; c += VLEN_FP32) {
                const uint32_t cb_n = MIN(VLEN_FP32, cols - c);

                const HVX_Vector w0 = *(const HVX_Vector *) (W + c);
                const HVX_Vector w1 = *(const HVX_Vector *) (W + (size_t) W_stride + c);
                const HVX_Vector w2 = *(const HVX_Vector *) (W + 2 * (size_t) W_stride + c);
                const HVX_Vector w3 = *(const HVX_Vector *) (W + 3 * (size_t) W_stride + c);

                const HVX_Vector x0  = *(const HVX_Vector *) (t + 0 * (size_t) T_stride + c);
                const HVX_Vector x1  = *(const HVX_Vector *) (t + 1 * (size_t) T_stride + c);
                const HVX_Vector x2  = *(const HVX_Vector *) (t + 2 * (size_t) T_stride + c);
                const HVX_Vector x3  = *(const HVX_Vector *) (t + 3 * (size_t) T_stride + c);
                const HVX_Vector x4  = *(const HVX_Vector *) (t + 4 * (size_t) T_stride + c);
                const HVX_Vector x5  = *(const HVX_Vector *) (t + 5 * (size_t) T_stride + c);
                const HVX_Vector x6  = *(const HVX_Vector *) (t + 6 * (size_t) T_stride + c);
                const HVX_Vector x7  = *(const HVX_Vector *) (t + 7 * (size_t) T_stride + c);
                const HVX_Vector x8  = *(const HVX_Vector *) (t + 8 * (size_t) T_stride + c);
                const HVX_Vector x9  = *(const HVX_Vector *) (t + 9 * (size_t) T_stride + c);
                const HVX_Vector x10 = *(const HVX_Vector *) (t + 10 * (size_t) T_stride + c);

                HVX_Vector a0 = ssm_conv_mac_vec(zero, x0, w0);
                a0            = ssm_conv_mac_vec(a0, x1, w1);
                a0            = ssm_conv_mac_vec(a0, x2, w2);
                a0            = ssm_conv_mac_vec(a0, x3, w3);
                HVX_Vector a1 = ssm_conv_mac_vec(zero, x1, w0);
                a1            = ssm_conv_mac_vec(a1, x2, w1);
                a1            = ssm_conv_mac_vec(a1, x3, w2);
                a1            = ssm_conv_mac_vec(a1, x4, w3);
                HVX_Vector a2 = ssm_conv_mac_vec(zero, x2, w0);
                a2            = ssm_conv_mac_vec(a2, x3, w1);
                a2            = ssm_conv_mac_vec(a2, x4, w2);
                a2            = ssm_conv_mac_vec(a2, x5, w3);
                HVX_Vector a3 = ssm_conv_mac_vec(zero, x3, w0);
                a3            = ssm_conv_mac_vec(a3, x4, w1);
                a3            = ssm_conv_mac_vec(a3, x5, w2);
                a3            = ssm_conv_mac_vec(a3, x6, w3);
                HVX_Vector a4 = ssm_conv_mac_vec(zero, x4, w0);
                a4            = ssm_conv_mac_vec(a4, x5, w1);
                a4            = ssm_conv_mac_vec(a4, x6, w2);
                a4            = ssm_conv_mac_vec(a4, x7, w3);
                HVX_Vector a5 = ssm_conv_mac_vec(zero, x5, w0);
                a5            = ssm_conv_mac_vec(a5, x6, w1);
                a5            = ssm_conv_mac_vec(a5, x7, w2);
                a5            = ssm_conv_mac_vec(a5, x8, w3);
                HVX_Vector a6 = ssm_conv_mac_vec(zero, x6, w0);
                a6            = ssm_conv_mac_vec(a6, x7, w1);
                a6            = ssm_conv_mac_vec(a6, x8, w2);
                a6            = ssm_conv_mac_vec(a6, x9, w3);
                HVX_Vector a7 = ssm_conv_mac_vec(zero, x7, w0);
                a7            = ssm_conv_mac_vec(a7, x8, w1);
                a7            = ssm_conv_mac_vec(a7, x9, w2);
                a7            = ssm_conv_mac_vec(a7, x10, w3);

                HVX_Vector y0 = Q6_Vsf_equals_Vqf32(a0);
                HVX_Vector y1 = Q6_Vsf_equals_Vqf32(a1);
                HVX_Vector y2 = Q6_Vsf_equals_Vqf32(a2);
                HVX_Vector y3 = Q6_Vsf_equals_Vqf32(a3);
                HVX_Vector y4 = Q6_Vsf_equals_Vqf32(a4);
                HVX_Vector y5 = Q6_Vsf_equals_Vqf32(a5);
                HVX_Vector y6 = Q6_Vsf_equals_Vqf32(a6);
                HVX_Vector y7 = Q6_Vsf_equals_Vqf32(a7);
                if (silu) {
                    y0 = ssm_conv_silu_vec(y0, one, max_exp, min_exp);
                    y1 = ssm_conv_silu_vec(y1, one, max_exp, min_exp);
                    y2 = ssm_conv_silu_vec(y2, one, max_exp, min_exp);
                    y3 = ssm_conv_silu_vec(y3, one, max_exp, min_exp);
                    y4 = ssm_conv_silu_vec(y4, one, max_exp, min_exp);
                    y5 = ssm_conv_silu_vec(y5, one, max_exp, min_exp);
                    y6 = ssm_conv_silu_vec(y6, one, max_exp, min_exp);
                    y7 = ssm_conv_silu_vec(y7, one, max_exp, min_exp);
                }
                ssm_conv_store_vec(o + c, cb_n, y0);
                ssm_conv_store_vec(o + 1 * (size_t) out_stride + c, cb_n, y1);
                ssm_conv_store_vec(o + 2 * (size_t) out_stride + c, cb_n, y2);
                ssm_conv_store_vec(o + 3 * (size_t) out_stride + c, cb_n, y3);
                ssm_conv_store_vec(o + 4 * (size_t) out_stride + c, cb_n, y4);
                ssm_conv_store_vec(o + 5 * (size_t) out_stride + c, cb_n, y5);
                ssm_conv_store_vec(o + 6 * (size_t) out_stride + c, cb_n, y6);
                ssm_conv_store_vec(o + 7 * (size_t) out_stride + c, cb_n, y7);
            }
        }
    }

    for (; u < n_rows; ++u) {
        const float * t = T + (size_t) u * T_stride;
        float *       o = out + (size_t) u * out_stride;
        for (uint32_t c = 0; c < cols; c += VLEN_FP32) {
            const uint32_t cb_n = MIN(VLEN_FP32, cols - c);
            HVX_Vector     acc  = zero;
            for (uint32_t j = 0; j < d_conv; ++j) {
                acc = ssm_conv_mac_vec(acc, *(const HVX_Vector *) (t + (size_t) j * T_stride + c),
                                       *(const HVX_Vector *) (W + (size_t) j * W_stride + c));
            }
            HVX_Vector y = Q6_Vsf_equals_Vqf32(acc);
            if (silu) {
                y = ssm_conv_silu_vec(y, one, max_exp, min_exp);
            }
            ssm_conv_store_vec(o + c, cb_n, y);
        }
    }
}

static void ssm_conv_thread_f32_prefill(unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_context *             scctx   = (struct htp_ssm_conv_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict src1 = octx->src[1];
    const struct htp_tensor * restrict dst  = octx->dst;

    const bool fused = kparams->fused_concat != 0;
    const uint32_t n_state_rows = fused ? kparams->n_state_rows : 0;
    const struct htp_tensor * restrict src2 = fused ? octx->src[2] : NULL;

    dma_queue * dma_q = octx->ctx->dma[ith];

    const uint32_t d_conv  = kparams->d_conv;
    const uint32_t d_inner = kparams->d_inner;
    const uint32_t n_t     = kparams->n_t;
    const uint32_t n_s     = kparams->n_s;
    const uint32_t ncs     = fused ? (n_state_rows + src2->ne[1]) : src0->ne[0];

    const uint32_t dr  = scctx->nrows_per_thread;
    const uint32_t ir0 = scctx->row_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    if (ir0 >= ir1) {
        return;
    }

    const uint32_t d_inner_per_thread = ir1 - ir0;
    const uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);
    const uint32_t d_inner_tile       = scctx->d_inner_tile;

    const size_t src0_stride_inner_bytes = src0->nb[1];
    const size_t src0_stride_seq_bytes   = src0->nb[2];
    const size_t dst_stride_token_bytes  = dst->nb[1];
    const size_t dst_stride_seq_bytes    = dst->nb[2];

    const size_t qkv_stride_token_bytes = fused ? src2->nb[1] : 0;
    const size_t qkv_stride_seq_bytes   = fused ? src2->nb[2] : 0;

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    const size_t weight_bytes    = (size_t) d_inner_per_thread * d_conv * sizeof(float);
    const size_t weight_raw_size = hex_round_up(weight_bytes, 128);

    float * src1_raw = (float *) src1_spad_base;
    float * src1_T   = (float *) (src1_spad_base + weight_raw_size);

    const uint32_t src0_raw_rows = fused ? n_state_rows : ncs;
    const size_t src0_tile_raw_bytes = hex_round_up(d_inner_tile * src0_raw_rows * sizeof(float), 128) + 128;
    const size_t src0_slot_size = fused ? octx->src0_spad.size_per_thread / 2 : octx->src0_spad.size_per_thread;
    float * dst_tile = (float *) dst_spad_base;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const HVX_Vector silu_one     = hvx_vec_splat_f32(1.f);
    const HVX_Vector silu_max_exp = hvx_vec_splat_f32(87.f);
    const HVX_Vector silu_min_exp = hvx_vec_splat_f32(-87.f);

    const dma_addr_t src1_ddr = src1->data + ir0 * d_conv * sizeof(float);
    dma_queue_push(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);
    dma_queue_pop(dma_q);

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    const uint32_t n_tiles = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
    const uint32_t C_TILE = VLEN_FP32;

    for (uint32_t i3 = 0; i3 < n_s; ++i3) {
        if (fused) {
            const uint32_t count = ssm_conv_prefill_enqueue_input(
                dma_q, true, src0, src2, src0_spad_base, src0_slot_size,
                src0_tile_raw_bytes, d_inner_tile, 0, d_inner_per_thread, ir0, i3, ncs,
                n_state_rows, src0_stride_inner_bytes, src0_stride_seq_bytes,
                qkv_stride_token_bytes, qkv_stride_seq_bytes);
            ssm_conv_prefill_wait_input(dma_q, count);
        }

        for (uint32_t tile_idx = 0; tile_idx < n_tiles; ++tile_idx) {
            const uint32_t tile_off = tile_idx * d_inner_tile;
            const uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);
            const uint32_t slot = fused ? (tile_idx & 1) : 0;
            uint8_t * slot_base = src0_spad_base + (size_t) slot * src0_slot_size;
            float * src0_tile_raw = (float *) slot_base;
            float * src0_T = (float *) (slot_base + src0_tile_raw_bytes);
            uint32_t next_input_count = 0;

            if (!fused) {
                const uint32_t count = ssm_conv_prefill_enqueue_input(
                    dma_q, false, src0, NULL, src0_spad_base, src0_slot_size,
                    src0_tile_raw_bytes, d_inner_tile, tile_idx, d_inner_per_thread, ir0, i3,
                    ncs, 0, src0_stride_inner_bytes, src0_stride_seq_bytes, 0, 0);
                ssm_conv_prefill_wait_input(dma_q, count);
            } else if (tile_idx + 1 < n_tiles) {
                next_input_count = ssm_conv_prefill_enqueue_input(
                    dma_q, true, src0, src2, src0_spad_base, src0_slot_size,
                    src0_tile_raw_bytes, d_inner_tile, tile_idx + 1, d_inner_per_thread, ir0, i3,
                    ncs, n_state_rows, src0_stride_inner_bytes, src0_stride_seq_bytes,
                    qkv_stride_token_bytes, qkv_stride_seq_bytes);
            }

            if (fused) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
                if (n_state_rows > 0) {
                    for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
                        const uint32_t cb_n = MIN(C_TILE, tile_n - cb);
                        transpose_src0_state_rows_fused(src0_tile_raw + cb * n_state_rows, n_state_rows,
                                n_state_rows, cb_n, d_inner_tile, src0_T, cb);
                    }
                }
            } else {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
                for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
                    const uint32_t cb_n = MIN(C_TILE, tile_n - cb);
                    transpose_src0_block(src0_tile_raw + cb * ncs, ncs, cb_n, d_inner_tile, src0_T, cb);
                }
            }

            ssm_conv_tile_mac(src0_T, d_inner_tile, src1_T + tile_off, d_inner_stride, d_conv, tile_n, n_t,
                              dst_tile, tile_n, false, silu_one, silu_max_exp, silu_min_exp);
            if (scctx->apply_silu) {
                ssm_conv_silu_tile(dst_tile, tile_n, tile_n, n_t, silu_one, silu_max_exp, silu_min_exp);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);

            if (next_input_count > 0) {
                ssm_conv_prefill_wait_input(dma_q, next_input_count);
            }

            const dma_addr_t dst_tile_ddr = dst->data +
                i3 * dst_stride_seq_bytes + (ir0 + tile_off) * sizeof(float);
            const size_t dst_row_bytes = tile_n * sizeof(float);
            dma_queue_push(dma_q, dma_make_data(dst_tile_ddr, (uint8_t *) dst_tile),
                           dst_stride_token_bytes, dst_row_bytes, dst_row_bytes, n_t);
            dma_queue_pop(dma_q);
        }
    }

    FARF(HIGH, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u fused %d silu %d\n",
         ith, nth, ncs, d_inner, n_s, 1u, ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], fused, scctx->apply_silu);
}
// Pipelined fused prefill (kparams->pipe). A step is one (i3, tile, token chunk) block. The DMA queue is FIFO: step s starts with [out(s-2), in(s), out(s-1), in(s+1)] queued, so it pops out(s-2) (frees output slot s&1), pops in(s), prefetches in(s+1) into slot (s+1)&1, computes and pushes out(s).
// Chunk c > 0 loads the qkv window [t0-3, t0+tn); chunk 0 also loads the raw state rows.
struct ssm_conv_pipe_ctx {
    const struct htp_tensor * src0;   // conv_states [n_state_rows, d_inner, n_s]
    const struct htp_tensor * src2;   // qkv_mixed_orig [d_inner, n_tokens, n_s]
    uint32_t                  n_t;
    uint32_t                  n_state_rows;
    uint32_t                  chunk;      // tokens per step
    uint32_t                  tile;       // d_inner_tile
    uint32_t                  ir0;
    uint32_t                  n_tiles;
    uint32_t                  n_chunks;
    size_t                    raw_bytes;  // raw state part of an input slot
};

static inline void ssm_conv_pipe_decode(const struct ssm_conv_pipe_ctx * pc, uint32_t s,
                                        uint32_t * i3, uint32_t * tile_idx, uint32_t * c) {
    const uint32_t per_seq = pc->n_tiles * pc->n_chunks;
    *i3       = s / per_seq;
    const uint32_t rem = s - *i3 * per_seq;
    *tile_idx = rem / pc->n_chunks;
    *c        = rem - *tile_idx * pc->n_chunks;
}

// Queues the input of one step into slot_base. Returns the number of DMA descriptors queued.
static uint32_t ssm_conv_pipe_push_in(dma_queue * dma_q, const struct ssm_conv_pipe_ctx * pc, uint8_t * slot_base,
                                      uint32_t i3, uint32_t tile_off, uint32_t tile_n, uint32_t c) {
    const uint32_t ns = pc->n_state_rows;
    const uint32_t t0 = c * pc->chunk;
    const uint32_t tn = MIN(pc->chunk, pc->n_t - t0);
    float * raw = (float *) slot_base;
    float * T   = (float *) (slot_base + pc->raw_bytes);

    if (t0 == 0) {
        const dma_addr_t state_ddr = pc->src0->data + i3 * pc->src0->nb[2] + (pc->ir0 + tile_off) * pc->src0->nb[1];
        const size_t state_row_bytes = ns * sizeof(float);
        dma_queue_push(dma_q, dma_make_data((uint8_t *) raw, state_ddr),
                       state_row_bytes, pc->src0->nb[1], state_row_bytes, tile_n);

        const dma_addr_t qkv_ddr = pc->src2->data + i3 * pc->src2->nb[2] + (pc->ir0 + tile_off) * sizeof(float);
        dma_queue_push(dma_q, dma_make_data((uint8_t *) (T + ns * pc->tile), qkv_ddr),
                       pc->tile * sizeof(float), pc->src2->nb[1], tile_n * sizeof(float), tn);
        return 2;
    }

    const dma_addr_t qkv_ddr = pc->src2->data + i3 * pc->src2->nb[2] + (pc->ir0 + tile_off) * sizeof(float) +
        (size_t) (t0 - ns) * pc->src2->nb[1];
    dma_queue_push(dma_q, dma_make_data((uint8_t *) T, qkv_ddr),
                   pc->tile * sizeof(float), pc->src2->nb[1], tile_n * sizeof(float), tn + ns);
    return 1;
}

static void ssm_conv_thread_f32_prefill_pipe(unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_context *             scctx   = (struct htp_ssm_conv_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict src1 = octx->src[1];
    const struct htp_tensor * restrict src2 = octx->src[2];
    const struct htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    const uint32_t d_conv = kparams->d_conv;
    const uint32_t n_t    = kparams->n_t;
    const uint32_t n_s    = kparams->n_s;
    const uint32_t ns     = kparams->n_state_rows;
    const uint32_t chunk  = kparams->pipe_chunk;

    const uint32_t dr  = scctx->nrows_per_thread;
    const uint32_t ir0 = scctx->row_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    if (ir0 >= ir1) {
        return;
    }

    const uint32_t d_inner_per_thread = ir1 - ir0;
    const uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);
    const uint32_t d_inner_tile       = scctx->d_inner_tile;

    const size_t dst_stride_token_bytes = dst->nb[1];
    const size_t dst_stride_seq_bytes   = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    const size_t weight_bytes    = (size_t) d_inner_per_thread * d_conv * sizeof(float);
    const size_t weight_raw_size = hex_round_up(weight_bytes, 128);

    float * src1_raw = (float *) src1_spad_base;
    float * src1_T   = (float *) (src1_spad_base + weight_raw_size);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const HVX_Vector silu_one     = hvx_vec_splat_f32(1.f);
    const HVX_Vector silu_max_exp = hvx_vec_splat_f32(87.f);
    const HVX_Vector silu_min_exp = hvx_vec_splat_f32(-87.f);

    const dma_addr_t src1_ddr = src1->data + ir0 * d_conv * sizeof(float);
    dma_queue_push(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);
    dma_queue_pop(dma_q);

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    const size_t raw_bytes = hex_round_up(d_inner_tile * ns * sizeof(float), 128) + 128;
    const size_t slot_in   = octx->src0_spad.size_per_thread / 2;
    const size_t slot_out  = octx->dst_spad.size_per_thread / 2;

    struct ssm_conv_pipe_ctx pc;
    pc.src0         = src0;
    pc.src2         = src2;
    pc.n_t          = n_t;
    pc.n_state_rows = ns;
    pc.chunk        = chunk;
    pc.tile         = d_inner_tile;
    pc.ir0          = ir0;
    pc.n_tiles      = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
    pc.n_chunks     = (n_t + chunk - 1) / chunk;
    pc.raw_bytes    = raw_bytes;

    const uint32_t n_steps = n_s * pc.n_tiles * pc.n_chunks;

    uint32_t i3       = 0;
    uint32_t tile_idx = 0;
    uint32_t c        = 0;
    ssm_conv_pipe_decode(&pc, 0, &i3, &tile_idx, &c);
    const uint32_t off0 = tile_idx * d_inner_tile;
    uint32_t cnt_cur = ssm_conv_pipe_push_in(dma_q, &pc, src0_spad_base, i3, off0,
                                             MIN(d_inner_tile, d_inner_per_thread - off0), c);

    for (uint32_t s = 0; s < n_steps; ++s) {
        ssm_conv_pipe_decode(&pc, s, &i3, &tile_idx, &c);
        const uint32_t tile_off = tile_idx * d_inner_tile;
        const uint32_t tile_n   = MIN(d_inner_tile, d_inner_per_thread - tile_off);
        const uint32_t t0       = c * chunk;
        const uint32_t tn       = MIN(chunk, n_t - t0);
        const uint32_t slot     = s & 1;

        if (s >= 2) {
            dma_queue_pop(dma_q);
        }
        ssm_conv_prefill_wait_input(dma_q, cnt_cur);

        uint32_t cnt_next = 0;
        if (s + 1 < n_steps) {
            uint32_t i3n = 0;
            uint32_t tin = 0;
            uint32_t cn  = 0;
            ssm_conv_pipe_decode(&pc, s + 1, &i3n, &tin, &cn);
            const uint32_t off_n = tin * d_inner_tile;
            cnt_next = ssm_conv_pipe_push_in(dma_q, &pc, src0_spad_base + (size_t) (slot ^ 1) * slot_in,
                                             i3n, off_n, MIN(d_inner_tile, d_inner_per_thread - off_n), cn);
        }

        uint8_t * in_base  = src0_spad_base + (size_t) slot * slot_in;
        float *   raw      = (float *) in_base;
        float *   T        = (float *) (in_base + raw_bytes);
        float *   out_tile = (float *) (dst_spad_base + (size_t) slot * slot_out);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        if (t0 == 0) {
            for (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
                const uint32_t cb_n = MIN(VLEN_FP32, tile_n - cb);
                transpose_src0_state_rows_fused(raw + cb * ns, ns, ns, cb_n, d_inner_tile, T, cb);
            }
        }
        ssm_conv_tile_mac(T, d_inner_tile, src1_T + tile_off, d_inner_stride, d_conv, tile_n, tn,
                          out_tile, tile_n, false, silu_one, silu_max_exp, silu_min_exp);
        if (scctx->apply_silu) {
            ssm_conv_silu_tile(out_tile, tile_n, tile_n, tn, silu_one, silu_max_exp, silu_min_exp);
        }

        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);

        const dma_addr_t dst_ddr = dst->data + i3 * dst_stride_seq_bytes + t0 * dst_stride_token_bytes +
            (ir0 + tile_off) * sizeof(float);
        dma_queue_push(dma_q, dma_make_data(dst_ddr, (uint8_t *) out_tile),
                       dst_stride_token_bytes, tile_n * sizeof(float), tile_n * sizeof(float), tn);

        cnt_cur = cnt_next;
    }

    ssm_conv_prefill_wait_input(dma_q, n_steps < 2 ? n_steps : 2);
    // drain the last output stores too: the op's DDR outputs must be complete when it returns (DMA-clean, htp-tensor.c)
    if ((octx->flags & HTP_OPFLAGS_DMA_CLEAN) != 0) {
        dma_queue_flush(dma_q);
    }

    FARF(HIGH, "ssm-conv-f32-prefill-pipe %d/%d: chunk %u tiles %u steps %u (%u:%u)\n",
         ith, nth, chunk, pc.n_tiles, n_steps, ir0, ir1);
}

int op_ssm_conv_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_ssm_conv_kernel_params * kparams = (const struct htp_ssm_conv_kernel_params *) octx->kernel_params;


    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->d_inner;

    if (octx->ctx->mdev.count > 1) {
        const uint32_t elems_per_chunk = VLEN_FP32;
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->d_inner,
            htp_tensor_mdev_data_aligned(dst) ? elems_per_chunk : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.count,
            &octx->ctx->mdev.count_div
        );
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const uint32_t n_threads = octx->n_threads;

    octx->src0_spad.size_per_thread = kparams->vtcm_src0_size_per_thread;
    octx->src1_spad.size_per_thread = kparams->vtcm_src1_size_per_thread;
    octx->dst_spad.size_per_thread  = kparams->vtcm_dst_size_per_thread;

    octx->src0_spad.size = kparams->vtcm_src0_size;
    octx->src1_spad.size = kparams->vtcm_src1_size;
    octx->dst_spad.size  = kparams->vtcm_dst_size;

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;
    octx->src0_spad.src  = NULL;
    octx->src1_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    const uint32_t raw_rpt            = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    const uint32_t d_inner_per_thread = hex_round_up(raw_rpt, VLEN_FP32);

    struct htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_per_thread = d_inner_per_thread,
        .d_inner_tile     = kparams->d_inner_tile,
        .row_start        = row_start,
        .nrows            = nrows,
        .fused            = (kparams->fused_concat != 0),
        .n_state_rows     = kparams->n_state_rows,
        .apply_silu       = (octx->op == HTP_OP_SSM_CONV_SILU),
    };

    FARF(HIGH, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : mode %s fused %d silu %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->n_t == 1 ? "decode" : "prefill", scctx.fused, scctx.apply_silu);

    if (kparams->n_t == 1) {
        // Neither the CONCAT-fusion nor the SILU-fusion decode worker path is implemented (try_fuse_concat_ssm_conv and try_fuse_ssm_conv_silu both gate on n_t > 1 host-side) -- if either ever reaches here, that is a host-side gating bug, not something to silently miscompute.
        if (scctx.fused || scctx.apply_silu) {
            FARF(ERROR, "ssm_conv: fused mode (concat=%d silu=%d) reached the decode "
                 "(n_t==1) worker, which implements neither; host-side fusion should "
                 "never trigger this\n", scctx.fused, scctx.apply_silu);
            return HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_decode, &scctx, n_threads);
    } else if (kparams->pipe) {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_prefill_pipe, &scctx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_prefill, &scctx, n_threads);
    }

    return HTP_STATUS_OK;
}

int op_ssm_conv(struct htp_ops_context * octx) {
    const struct htp_tensor * dst = octx->dst;

    switch (dst->type) {
        case HTP_TYPE_F32:
            return op_ssm_conv_f32(octx);
        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}

// Fused CONCAT + CPY + SSM_CONV + SILU (HTP_OP_SSM_CONV_CHAIN). The concat is never built: qkv tiles go by DMA
// into the conv input tile in VTCM, and the new conv state comes from the last n_state columns of that tile.

struct htp_ssm_conv_chain_context {
    struct htp_ops_context *                        octx;
    const struct htp_ssm_conv_chain_kernel_params * kparams;
    atomic_uint                                     state_barrier;
    atomic_uint                                     next_item;   // dynamic tile queue (balanced == 2)
    uint8_t *                                       shared_base; // dynamic: conv state, weights (raw, tap-major) of all channels
};

// Tap-major {n_rows, T_stride} -> channel-major {tile_n, n_rows}
static inline void hvx_ssm_conv_scatter_from_T(const float * T, uint32_t T_stride, uint32_t n_rows, float * out,
                                               uint32_t tile_n) {
    for (uint32_t j = 0; j < n_rows; ++j) {
        const float * src = T + (size_t) j * T_stride;
        float *       dst = out + j;
        for (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
            const uint32_t cb_n = MIN(VLEN_FP32, tile_n - cb);
            HVX_Vector     v    = *(const HVX_Vector *) (src + cb);
            for (uint32_t c = 0; c < cb_n; ++c) {
                const HVX_Vector r = Q6_V_vror_VR(v, c * (uint32_t) sizeof(float));
                hvx_vec_store_u(dst + (size_t) (cb + c) * n_rows, sizeof(float), r);
            }
        }
    }
}

static inline void ssm_conv_chain_push_state_in(dma_queue * dma_q, const struct htp_tensor * st_in,
                                                const struct htp_ssm_conv_chain_kernel_params * kp,
                                                float * stage, size_t seq_elems, uint32_t seq, uint32_t ch0, uint32_t n) {
    const size_t bytes = (size_t) n * kp->n_state * sizeof(float);
    float * dst = stage + (size_t) seq * seq_elems;
    const size_t     off = (size_t) seq * kp->st_in_seq_stride + (size_t) ch0 * kp->n_state * sizeof(float);
    const dma_addr_t src = st_in->data + off;
    dma_queue_push(dma_q, dma_make_data((uint8_t *) dst, src), bytes, bytes, bytes, 1);
}

// nt token rows (from t0) of tile_n channels into rows [n_state, n_state + nt) of the conv input tile
static inline void ssm_conv_chain_push_in(dma_queue * dma_q, const struct htp_tensor * qkv,
                                          const struct htp_ssm_conv_chain_kernel_params * kp,
                                          uint8_t * buf, uint32_t seq, uint32_t ch0, uint32_t tile_n,
                                          uint32_t t0, uint32_t nt) {
    float *          xt  = (float *) buf;
    const size_t     off = (size_t) seq * kp->qkv_seq_stride + (size_t) t0 * qkv->nb[0] + (size_t) ch0 * sizeof(float);
    const dma_addr_t src = qkv->data + off;
    dma_queue_push(dma_q, dma_make_data((uint8_t *) (xt + (size_t) kp->n_state * kp->d_inner_tile), src),
                   (size_t) kp->d_inner_tile * sizeof(float), qkv->nb[0], (size_t) tile_n * sizeof(float), nt);
}

// dynamic queue: token block b > 0 loads its own n_state history rows (tokens t0 - n_state ..) instead of taking them from block b - 1
static inline void ssm_conv_chain_push_in_hist(dma_queue * dma_q, const struct htp_tensor * qkv,
                                               const struct htp_ssm_conv_chain_kernel_params * kp,
                                               uint8_t * buf, uint32_t seq, uint32_t ch0, uint32_t tile_n,
                                               uint32_t t0, uint32_t nt) {
    float *          xt  = (float *) buf;
    const size_t     off = (size_t) seq * kp->qkv_seq_stride + (size_t) (t0 - kp->n_state) * qkv->nb[0] + (size_t) ch0 * sizeof(float);
    const dma_addr_t src = qkv->data + off;
    dma_queue_push(dma_q, dma_make_data((uint8_t *) xt, src),
                   (size_t) kp->d_inner_tile * sizeof(float), qkv->nb[0], (size_t) tile_n * sizeof(float), nt + kp->n_state);
}

// Output DMA of one tile: the conv output split at the q/k/v boundaries (one descriptor per part) and, with the
// last token block, the new conv state. Returns the number of descriptors pushed.
static inline uint32_t ssm_conv_chain_push_out(dma_queue * dma_q, const struct htp_ops_context * octx,
                                               const struct htp_ssm_conv_chain_kernel_params * kp,
                                               uint32_t n_q, uint32_t n_k, uint8_t * buf, uint8_t * dst_tile,
                                               uint32_t seq, uint32_t ch0, uint32_t tile_n,
                                               uint32_t t0, uint32_t nt, bool last_blk) {
    const struct htp_tensor * dst    = octx->dsts[0];
    const struct htp_tensor * st_out = octx->dsts[1];

    // q -> dst[2], k -> dst[3], the remaining channels -> dst[0]
    const uint32_t r_beg[3] = { 0, n_q, n_q + n_k };
    const uint32_t r_end[3] = { n_q, n_q + n_k, kp->d_inner };

    uint32_t n_desc = 0;
    for (uint32_t r = 0; r < 3; ++r) {
        const uint32_t c0 = MAX(ch0, r_beg[r]);
        const uint32_t c1 = MIN(ch0 + tile_n, r_end[r]);
        if (c1 <= c0) {
            continue;
        }
        dma_addr_t addr;
        size_t     stride;
        if (r == 2) {
            addr   = dst->data + (size_t) seq * kp->dst_seq_stride + (size_t) t0 * dst->nb[1] + c0 * sizeof(float);
            stride = dst->nb[1];
        } else {
            const struct htp_tensor * d = octx->dsts[2 + r];
            addr   = d->data + (size_t) seq * d->nb[3] + (size_t) t0 * d->nb[2] + (c0 - r_beg[r]) * sizeof(float);
            stride = d->nb[2];
        }
        dma_queue_push(dma_q, dma_make_data(addr, dst_tile + (size_t) (c0 - ch0) * sizeof(float)),
                       stride, (size_t) kp->d_inner_tile * sizeof(float), (size_t) (c1 - c0) * sizeof(float), nt);
        n_desc++;
    }

    if (last_blk) {
        const size_t state_bytes = (size_t) tile_n * kp->n_state * sizeof(float);
        float * state_out = (float *) (buf + kp->vtcm_xt_bytes);
        const size_t     off = (size_t) seq * kp->st_out_seq_stride + (size_t) ch0 * kp->n_state * sizeof(float);
        const dma_addr_t st  = st_out->data + off;
        dma_queue_push(dma_q, dma_make_data(st, (uint8_t *) state_out), state_bytes, state_bytes, state_bytes, 1);
        n_desc++;
    }

    return n_desc;
}

// In-place silu over the first n_vecs vectors of the conv output tile (rows of d_inner_tile floats, contiguous). Out of line on purpose: inlined, the compiler schedules it differently and the result is not bit-identical.
// Four vectors per step; tail uses the fp32 path. Same per-vector primitives as the fused SSM_CONV+SILU store (op_ssm_conv_f32 apply_silu): guarded fast sigmoid, then the product. fp16 sigmoid variant (GGML_HEXAGON_C2_SILU=3, numerics-changing, same scheme as norm_gated NG_SIG16): two fp32 vectors packed into one fp16 vector, hvx_vec_fast_sigmoid_f16 on 64 lanes, unpacked; y * sigmoid(y) stays fp32.
static void ssm_conv_chain_silu_h16(HVX_Vector * restrict v, uint32_t n, HVX_Vector one, HVX_Vector max_exp, HVX_Vector min_exp) {
    uint32_t i = 0;
    // eight vectors per step: four fp16 sigmoid chains in flight, as in norm_gated (hvx_silu_mul_rows_h16). Same per-element ops.
    for (; i + 8 <= n; i += 8) {
        const HVX_Vector sh0 = hvx_vec_fast_sigmoid_f16(hvx_vec_f32_to_f16_shuff(v[i],     v[i + 1]));
        const HVX_Vector sh1 = hvx_vec_fast_sigmoid_f16(hvx_vec_f32_to_f16_shuff(v[i + 2], v[i + 3]));
        const HVX_Vector sh2 = hvx_vec_fast_sigmoid_f16(hvx_vec_f32_to_f16_shuff(v[i + 4], v[i + 5]));
        const HVX_Vector sh3 = hvx_vec_fast_sigmoid_f16(hvx_vec_f32_to_f16_shuff(v[i + 6], v[i + 7]));
        const HVX_VectorPair p0 = hvx_vec_f16_to_f32_shuff(sh0);
        const HVX_VectorPair p1 = hvx_vec_f16_to_f32_shuff(sh1);
        const HVX_VectorPair p2 = hvx_vec_f16_to_f32_shuff(sh2);
        const HVX_VectorPair p3 = hvx_vec_f16_to_f32_shuff(sh3);
        v[i]     = hvx_vec_mul_f32_f32(v[i],     Q6_V_lo_W(p0));
        v[i + 1] = hvx_vec_mul_f32_f32(v[i + 1], Q6_V_hi_W(p0));
        v[i + 2] = hvx_vec_mul_f32_f32(v[i + 2], Q6_V_lo_W(p1));
        v[i + 3] = hvx_vec_mul_f32_f32(v[i + 3], Q6_V_hi_W(p1));
        v[i + 4] = hvx_vec_mul_f32_f32(v[i + 4], Q6_V_lo_W(p2));
        v[i + 5] = hvx_vec_mul_f32_f32(v[i + 5], Q6_V_hi_W(p2));
        v[i + 6] = hvx_vec_mul_f32_f32(v[i + 6], Q6_V_lo_W(p3));
        v[i + 7] = hvx_vec_mul_f32_f32(v[i + 7], Q6_V_hi_W(p3));
    }
    for (; i + 4 <= n; i += 4) {
        const HVX_Vector sh0 = hvx_vec_fast_sigmoid_f16(hvx_vec_f32_to_f16_shuff(v[i],     v[i + 1]));
        const HVX_Vector sh1 = hvx_vec_fast_sigmoid_f16(hvx_vec_f32_to_f16_shuff(v[i + 2], v[i + 3]));
        const HVX_VectorPair p0 = hvx_vec_f16_to_f32_shuff(sh0);
        const HVX_VectorPair p1 = hvx_vec_f16_to_f32_shuff(sh1);
        v[i]     = hvx_vec_mul_f32_f32(v[i],     Q6_V_lo_W(p0));
        v[i + 1] = hvx_vec_mul_f32_f32(v[i + 1], Q6_V_hi_W(p0));
        v[i + 2] = hvx_vec_mul_f32_f32(v[i + 2], Q6_V_lo_W(p1));
        v[i + 3] = hvx_vec_mul_f32_f32(v[i + 3], Q6_V_hi_W(p1));
    }
    for (; i < n; ++i) {
        const HVX_Vector yv  = v[i];
        const HVX_Vector sig = hvx_vec_fast_sigmoid_f32_guard(yv, one, max_exp, min_exp);
        v[i] = hvx_vec_mul_f32_f32(yv, sig);
    }
}

static void ssm_conv_chain_silu_h16_v2(HVX_Vector * restrict v, uint32_t n, HVX_Vector one, HVX_Vector max_exp, HVX_Vector min_exp) {
    uint32_t i = 0;
    // eight vectors per step: four fp16 sigmoid chains in flight, as in norm_gated (hvx_silu_mul_rows_h16). Same per-element ops.
    for (; i + 8 <= n; i += 8) {
        const HVX_Vector sh0 = hvx_vec_fast_sigmoid_f16_v2(hvx_vec_f32_to_f16_shuff(v[i],     v[i + 1]));
        const HVX_Vector sh1 = hvx_vec_fast_sigmoid_f16_v2(hvx_vec_f32_to_f16_shuff(v[i + 2], v[i + 3]));
        const HVX_Vector sh2 = hvx_vec_fast_sigmoid_f16_v2(hvx_vec_f32_to_f16_shuff(v[i + 4], v[i + 5]));
        const HVX_Vector sh3 = hvx_vec_fast_sigmoid_f16_v2(hvx_vec_f32_to_f16_shuff(v[i + 6], v[i + 7]));
        const HVX_VectorPair p0 = hvx_vec_f16_to_f32_shuff(sh0);
        const HVX_VectorPair p1 = hvx_vec_f16_to_f32_shuff(sh1);
        const HVX_VectorPair p2 = hvx_vec_f16_to_f32_shuff(sh2);
        const HVX_VectorPair p3 = hvx_vec_f16_to_f32_shuff(sh3);
        v[i]     = hvx_vec_mul_f32_f32(v[i],     Q6_V_lo_W(p0));
        v[i + 1] = hvx_vec_mul_f32_f32(v[i + 1], Q6_V_hi_W(p0));
        v[i + 2] = hvx_vec_mul_f32_f32(v[i + 2], Q6_V_lo_W(p1));
        v[i + 3] = hvx_vec_mul_f32_f32(v[i + 3], Q6_V_hi_W(p1));
        v[i + 4] = hvx_vec_mul_f32_f32(v[i + 4], Q6_V_lo_W(p2));
        v[i + 5] = hvx_vec_mul_f32_f32(v[i + 5], Q6_V_hi_W(p2));
        v[i + 6] = hvx_vec_mul_f32_f32(v[i + 6], Q6_V_lo_W(p3));
        v[i + 7] = hvx_vec_mul_f32_f32(v[i + 7], Q6_V_hi_W(p3));
    }
    for (; i + 4 <= n; i += 4) {
        const HVX_Vector sh0 = hvx_vec_fast_sigmoid_f16_v2(hvx_vec_f32_to_f16_shuff(v[i],     v[i + 1]));
        const HVX_Vector sh1 = hvx_vec_fast_sigmoid_f16_v2(hvx_vec_f32_to_f16_shuff(v[i + 2], v[i + 3]));
        const HVX_VectorPair p0 = hvx_vec_f16_to_f32_shuff(sh0);
        const HVX_VectorPair p1 = hvx_vec_f16_to_f32_shuff(sh1);
        v[i]     = hvx_vec_mul_f32_f32(v[i],     Q6_V_lo_W(p0));
        v[i + 1] = hvx_vec_mul_f32_f32(v[i + 1], Q6_V_hi_W(p0));
        v[i + 2] = hvx_vec_mul_f32_f32(v[i + 2], Q6_V_lo_W(p1));
        v[i + 3] = hvx_vec_mul_f32_f32(v[i + 3], Q6_V_hi_W(p1));
    }
    for (; i < n; ++i) {
        const HVX_Vector yv  = v[i];
        const HVX_Vector sig = hvx_vec_fast_sigmoid_f32_guard(yv, one, max_exp, min_exp);
        v[i] = hvx_vec_mul_f32_f32(yv, sig);
    }
}

// LOSSY_GDN conv levers. rsqrt with `iters` (1..3) Newton iterations (hvx_vec_rsqrt_f32 does 3).
static inline __attribute__((always_inline)) HVX_Vector ssm_conv_rsqrt_n(HVX_Vector in_vec, const uint32_t iters) {
    const HVX_Vector onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    const HVX_Vector threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);
    HVX_Vector x2 = Q6_Vqf32_vmpy_VsfVsf(in_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());
    HVX_Vector y = Q6_Vw_vasr_VwR(in_vec, 1);
    y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(RSQRT_CONST), y);
    HVX_Vector ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    HVX_Vector temp = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(temp));
    _Pragma("unroll(2)")
    for (uint32_t it = 1; it < iters; ++it) {
        y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
        ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
        ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
        temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
        temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
        temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);
    }
    return Q6_Vsf_equals_Vqf32(temp);
}

// Four heads of hd floats (hd % 32 == 0), in place. iters: rsqrt Newton iterations; fuse: out = x * (rsqrt * scale) + bias.
static inline __attribute__((always_inline)) void ssm_conv_qk_norm_x4_lossy_t(uint8_t * p0, uint8_t * p1, uint8_t * p2, uint8_t * p3, uint32_t hd,
                                                                HVX_Vector eps_v, HVX_Vector denom_v, HVX_Vector scl_v, HVX_Vector bias_v,
                                                                const uint32_t iters, const bool fuse) {
    HVX_Vector * v0 = (HVX_Vector *) p0, * v1 = (HVX_Vector *) p1, * v2 = (HVX_Vector *) p2, * v3 = (HVX_Vector *) p3;
    const uint32_t nvec = hd / VLEN_FP32;
    HVX_Vector a0 = Q6_V_vzero(), a1 = a0, a2 = a0, a3 = a0;
    _Pragma("unroll(4)")
    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector x0 = v0[i], x1 = v1[i], x2 = v2[i], x3 = v3[i];
        a0 = Q6_Vqf32_vadd_Vqf32Vqf32(a0, Q6_Vqf32_vmpy_VsfVsf(x0, x0));
        a1 = Q6_Vqf32_vadd_Vqf32Vqf32(a1, Q6_Vqf32_vmpy_VsfVsf(x1, x1));
        a2 = Q6_Vqf32_vadd_Vqf32Vqf32(a2, Q6_Vqf32_vmpy_VsfVsf(x2, x2));
        a3 = Q6_Vqf32_vadd_Vqf32Vqf32(a3, Q6_Vqf32_vmpy_VsfVsf(x3, x3));
    }
    a0 = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(a0));
    a1 = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(a1));
    a2 = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(a2));
    a3 = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(a3));
    HVX_Vector r0 = ssm_conv_rsqrt_n(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a0, denom_v), eps_v)), iters);
    HVX_Vector r1 = ssm_conv_rsqrt_n(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a1, denom_v), eps_v)), iters);
    HVX_Vector r2 = ssm_conv_rsqrt_n(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a2, denom_v), eps_v)), iters);
    HVX_Vector r3 = ssm_conv_rsqrt_n(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a3, denom_v), eps_v)), iters);
    if (fuse) {
        r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r0, scl_v));
        r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r1, scl_v));
        r2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r2, scl_v));
        r3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r3, scl_v));
        _Pragma("unroll(4)")
        for (uint32_t i = 0; i < nvec; i++) {
            v0[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(v0[i], r0), bias_v));
            v1[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(v1[i], r1), bias_v));
            v2[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(v2[i], r2), bias_v));
            v3[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(v3[i], r3), bias_v));
        }
    } else {
        _Pragma("unroll(4)")
        for (uint32_t i = 0; i < nvec; i++) {
            HVX_Vector m0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v0[i], r0));
            HVX_Vector m1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1[i], r1));
            HVX_Vector m2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v2[i], r2));
            HVX_Vector m3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3[i], r3));
            v0[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(m0, scl_v), bias_v));
            v1[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(m1, scl_v), bias_v));
            v2[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(m2, scl_v), bias_v));
            v3[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(m3, scl_v), bias_v));
        }
    }
}

static void ssm_conv_qk_norm_x4_lossy(uint8_t * p0, uint8_t * p1, uint8_t * p2, uint8_t * p3, uint32_t hd, HVX_Vector eps_v,
                                      HVX_Vector denom_v, HVX_Vector scl_v, HVX_Vector bias_v, uint32_t iters, bool fuse) {
#define SSM_QKN(IT, FU) ssm_conv_qk_norm_x4_lossy_t(p0, p1, p2, p3, hd, eps_v, denom_v, scl_v, bias_v, IT, FU)
    switch (iters * 2 + (fuse ? 1 : 0)) {
        case 2:  SSM_QKN(1, false); break;
        case 3:  SSM_QKN(1, true);  break;
        case 4:  SSM_QKN(2, false); break;
        case 5:  SSM_QKN(2, true);  break;
        case 7:  SSM_QKN(3, true);  break;
        default: SSM_QKN(3, false); break;
    }
#undef SSM_QKN
}

// LOSSY_GDN SILU_H16: the product y * sigmoid(y) is formed in fp16 on the packed vectors too (one hf multiply per two fp32 vectors
// instead of two fp32 multiplies); y is already rounded to fp16 for the sigmoid. Eight vectors per step; tail uses the fp32 path.
static void ssm_conv_chain_silu_h16_full(HVX_Vector * restrict v, uint32_t n, HVX_Vector one, HVX_Vector max_exp, HVX_Vector min_exp, bool v2) {
    uint32_t i = 0;
#define SILU_PACK(k) const HVX_Vector y##k = hvx_vec_f32_to_f16_shuff(v[i + 2 * (k)], v[i + 2 * (k) + 1]);
#define SILU_FIN(k) { const HVX_VectorPair p = hvx_vec_f16_to_f32_shuff(hvx_vec_mul_f16_f16(y##k, sh##k)); \
                      v[i + 2 * (k)] = Q6_V_lo_W(p); v[i + 2 * (k) + 1] = Q6_V_hi_W(p); }
    if (v2) {
        for (; i + 8 <= n; i += 8) {
            SILU_PACK(0) SILU_PACK(1) SILU_PACK(2) SILU_PACK(3)
            const HVX_Vector sh0 = hvx_vec_fast_sigmoid_f16_v2(y0), sh1 = hvx_vec_fast_sigmoid_f16_v2(y1);
            const HVX_Vector sh2 = hvx_vec_fast_sigmoid_f16_v2(y2), sh3 = hvx_vec_fast_sigmoid_f16_v2(y3);
            SILU_FIN(0) SILU_FIN(1) SILU_FIN(2) SILU_FIN(3)
        }
    } else {
        for (; i + 8 <= n; i += 8) {
            SILU_PACK(0) SILU_PACK(1) SILU_PACK(2) SILU_PACK(3)
            const HVX_Vector sh0 = hvx_vec_fast_sigmoid_f16(y0), sh1 = hvx_vec_fast_sigmoid_f16(y1);
            const HVX_Vector sh2 = hvx_vec_fast_sigmoid_f16(y2), sh3 = hvx_vec_fast_sigmoid_f16(y3);
            SILU_FIN(0) SILU_FIN(1) SILU_FIN(2) SILU_FIN(3)
        }
    }
#undef SILU_PACK
#undef SILU_FIN
    for (; i < n; ++i) {
        const HVX_Vector yv  = v[i];
        const HVX_Vector sig = hvx_vec_fast_sigmoid_f32_guard(yv, one, max_exp, min_exp);
        v[i] = hvx_vec_mul_f32_f32(yv, sig);
    }
}

static void __attribute__((noinline)) ssm_conv_chain_silu_tile_lossy(uint8_t * restrict tile, uint32_t n_vecs, uint32_t variant) {
    ssm_conv_chain_silu_h16_full((HVX_Vector *) tile, n_vecs, hvx_vec_splat_f32(1.f), hvx_vec_splat_f32(87.f), hvx_vec_splat_f32(-87.f), variant >= 4);
}

static void __attribute__((noinline)) ssm_conv_chain_silu_tile(uint8_t * restrict tile, uint32_t n_vecs, uint32_t variant) {
    HVX_Vector * restrict v = (HVX_Vector *) tile;
    const HVX_Vector      one     = hvx_vec_splat_f32(1.f);
    const HVX_Vector      max_exp = hvx_vec_splat_f32(87.f);
    const HVX_Vector      min_exp = hvx_vec_splat_f32(-87.f);
    if (variant >= 4) {   // SIG16_V2
        ssm_conv_chain_silu_h16_v2(v, n_vecs, one, max_exp, min_exp);
        return;
    }
    if (variant >= 3) {
        ssm_conv_chain_silu_h16(v, n_vecs, one, max_exp, min_exp);
        return;
    }
    if (variant >= 2) {   // C2Elt: two-iteration reciprocal (not bit-identical; PPL and HOLD gate)
        for (uint32_t i = 0; i < n_vecs; i++) {
            HVX_Vector yv  = v[i];
            HVX_Vector sig = hvx_vec_fast_sigmoid_f32_guard_r2(yv, one, max_exp, min_exp);
            v[i]           = hvx_vec_mul_f32_f32(yv, sig);
        }
        return;
    }
    for (uint32_t i = 0; i < n_vecs; i++) {
        HVX_Vector yv  = v[i];
        HVX_Vector sig = hvx_vec_fast_sigmoid_f32_guard(yv, one, max_exp, min_exp);
        v[i]           = hvx_vec_mul_f32_f32(yv, sig);
    }
}

// tile index -> (sequence, channel offset, channel count, first token, token count); token blocks are innermost
struct ssm_conv_chain_tile_pos {
    uint32_t seq, tile_off, tile_n, t0, nt;
};

static inline struct ssm_conv_chain_tile_pos ssm_conv_chain_tile(uint32_t idx, uint32_t n_tiles_seq, uint32_t n_blk,
                                                                 uint32_t n_tb, uint32_t n_t,
                                                                 uint32_t d_inner_tile, uint32_t d_inner_per_thread) {
    struct ssm_conv_chain_tile_pos p;
    const uint32_t b  = idx % n_blk;
    const uint32_t ct = idx / n_blk;
    p.seq      = ct / n_tiles_seq;
    p.tile_off = (ct - p.seq * n_tiles_seq) * d_inner_tile;
    p.tile_n   = MIN(d_inner_tile, d_inner_per_thread - p.tile_off);
    p.t0       = b * n_tb;
    p.nt       = MIN(n_tb, n_t - p.t0);
    return p;
}


// Dynamic tile queue (kp->balanced == 2): every thread stages the conv state and weights of an even share of the head units into shared VTCM, then all threads drain the items (sequence, head unit, token block) through an atomic counter.
// Items are independent: token block b > 0 loads its own n_state history rows, so any thread can run any item.
static void ssm_conv_chain_thread_dyn(unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_chain_context *                      ccctx = (struct htp_ssm_conv_chain_context *) data;
    struct htp_ops_context *                                 octx  = ccctx->octx;
    const struct htp_ssm_conv_chain_kernel_params * restrict kp    = ccctx->kparams;

    const struct htp_tensor * restrict st_in = octx->src[0];
    const struct htp_tensor * restrict qkv   = octx->src[1];
    const struct htp_tensor * restrict wgt   = octx->src[2];

    dma_queue * dma_q = octx->ctx->dma[ith];

    const uint32_t d_conv       = kp->d_conv;
    const uint32_t n_state      = kp->n_state;
    const uint32_t n_t          = kp->n_t;
    const uint32_t d_inner_tile = kp->d_inner_tile;
    const uint32_t hd           = kp->qk_head_dim;     // == d_inner_tile
    const uint32_t D            = kp->d_inner;
    const uint32_t n_units      = D / hd;

    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * tile_spad_base = src0_spad_base + kp->vtcm_state_stage_bytes;
    uint8_t * dst_spad_base  = octx->dst_spad.data + ith * octx->dst_spad.size_per_thread;

    // shared region: conv state [seq][ch][n_state], raw weights [ch][d_conv], tap-major weights [tap][D]
    const size_t state_bytes = hex_round_up((size_t) kp->n_s * n_state * D * sizeof(float), 128);
    const size_t wraw_bytes  = hex_round_up((size_t) D * d_conv * sizeof(float), 128);
    uint8_t *    shared      = ccctx->shared_base;
    float *      state_stage = (float *) shared;
    float *      wgt_raw     = (float *) (shared + state_bytes);
    float *      wgt_T       = (float *) (shared + state_bytes + wraw_bytes);

    const uint32_t n_q  = octx->dsts[2]->ne[0] * octx->dsts[2]->ne[1];
    const uint32_t n_k  = octx->dsts[3]->ne[0] * octx->dsts[3]->ne[1];
    const uint32_t n_qk = n_q + n_k;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const uint32_t lossy    = (octx->flags & HTP_OPFLAGS_LOSSY_CONV_MASK) >> HTP_OPFLAGS_LOSSY_CONV_SHIFT;
    const uint32_t qk_iters = (lossy & HTP_LOSSY_CONV_RSQRT1) ? 1u : (lossy & HTP_LOSSY_CONV_RSQRT2) ? 2u : 3u;
    const bool     qk_fuse  = (lossy & HTP_LOSSY_CONV_NORM_FUSE) != 0;
    const bool     qk_lossy = qk_iters != 3 || qk_fuse;
    const bool     silu_h16 = (lossy & HTP_LOSSY_CONV_SILU_H16) != 0 && (kp->silu_variant & 0xff) >= 3;

    const uint32_t abl = kp->silu_variant >> 8;
    const uint32_t n_tb  = kp->n_tb ? kp->n_tb : n_t;
    const uint32_t n_blk = (n_t + n_tb - 1) / n_tb;
    const uint32_t n_items = n_units * n_blk * kp->n_s;

    // staging: an even share of the head units per thread
    const uint32_t u0 = ith * n_units / nth, u1 = (ith + 1) * n_units / nth;
    if (u1 > u0) {
        const uint32_t c0 = u0 * hd, n = (u1 - u0) * hd;
        const size_t   wb = (size_t) n * d_conv * sizeof(float);
        dma_queue_push(dma_q, dma_make_data((uint8_t *) (wgt_raw + (size_t) c0 * d_conv), wgt->data + (size_t) c0 * d_conv * sizeof(float)),
                       wb, wb, wb, 1);
        for (uint32_t s = 0; s < kp->n_s; ++s) {
            ssm_conv_chain_push_state_in(dma_q, st_in, kp, state_stage + (size_t) c0 * n_state, (size_t) n_state * D, s, c0, n);
        }
    }

    // item -> (seq, unit, block); token blocks innermost, q/k units first
    #define DYN_POS(it, seq_, unit_, b_) \
        const uint32_t b_ = (it) % n_blk; const uint32_t su_ = (it) / n_blk; \
        const uint32_t unit_ = su_ % n_units; const uint32_t seq_ = su_ / n_units;

    // first two items before the barrier: their input DMA overlaps the other threads' staging
    // (item 0 of a block-0 tile needs only qkv; the state is read after the barrier)
    uint32_t q[4] = { n_items, n_items, n_items, n_items };
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t it = atomic_fetch_add(&ccctx->next_item, 1);
        q[i] = it;
        if (it < n_items) {
            DYN_POS(it, seq, unit, b)
            const uint32_t t0 = b * n_tb, nt = MIN(n_tb, n_t - t0);
            uint8_t * buf = tile_spad_base + (i & 1) * kp->vtcm_buf_stride;
            if (b == 0) {
                ssm_conv_chain_push_in(dma_q, qkv, kp, buf, seq, unit * hd, hd, t0, nt);
            } else {
                ssm_conv_chain_push_in_hist(dma_q, qkv, kp, buf, seq, unit * hd, hd, t0, nt);
            }
        }
    }

    if (u1 > u0) {
        const uint32_t c0 = u0 * hd, n = (u1 - u0) * hd;
        for (uint32_t s = 0; s < kp->n_s + 1; ++s) {
            dma_queue_pop(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) c0);
        hvx_ssm_conv_unpack_to_T(wgt_raw + (size_t) c0 * d_conv, wgt_T + c0, n, D, d_conv);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) c0);
    }


    asm volatile("syncht" : : : "memory");
    atomic_fetch_sub(&ccctx->state_barrier, 1);
    while (atomic_load(&ccctx->state_barrier) > 0) {
    }

    uint32_t out_ndesc[2] = { 0, 0 };
    uint32_t last_i = 0;
    bool     any    = false;

    for (uint32_t i = 0; q[i & 3] < n_items; ++i) {
        const uint32_t it = q[i & 3];
        DYN_POS(it, seq, unit, b)
        const uint32_t t0 = b * n_tb, nt = MIN(n_tb, n_t - t0);
        const uint32_t ch0 = unit * hd;
        const uint32_t tile_n = hd;
        const bool last_blk = t0 + nt == n_t;
        last_i = i;
        any    = true;

        uint8_t * buf       = tile_spad_base + (i & 1) * kp->vtcm_buf_stride;
        float *   xt        = (float *) buf;
        float *   state_out = (float *) (buf + kp->vtcm_xt_bytes);
        float *   dst_tile  = (float *) (dst_spad_base + (i & 1) * kp->vtcm_dst_tile_bytes);

        dma_queue_pop(dma_q);

        if (b == 0) {
            const float * stage = state_stage + (size_t) seq * n_state * D + (size_t) ch0 * n_state;
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) ch0);
            hvx_ssm_conv_unpack_to_T(stage, xt, tile_n, d_inner_tile, n_state);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) ch0);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ch0);
        if (!(abl & 4)) ssm_conv_tile_mac(xt, d_inner_tile, wgt_T + ch0, D, d_conv, tile_n, nt, dst_tile,
                          d_inner_tile, false, hvx_vec_splat_f32(1.f), hvx_vec_splat_f32(87.f), hvx_vec_splat_f32(-87.f));
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ch0);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_OUT, (uint16_t) ch0);
        if (abl & 1) {
        } else if (silu_h16) {
            ssm_conv_chain_silu_tile_lossy((uint8_t *) dst_tile, nt * d_inner_tile / VLEN_FP32, kp->silu_variant & 0xff);
        } else {
            ssm_conv_chain_silu_tile((uint8_t *) dst_tile, nt * d_inner_tile / VLEN_FP32, kp->silu_variant & 0xff);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_OUT, (uint16_t) ch0);

        if (ch0 < n_qk && !(abl & 2)) {
            const uint32_t n_heads = (MIN(ch0 + tile_n, n_qk) - ch0) / hd;
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_PREP, (uint16_t) ch0);
            const HVX_Vector eps_v   = hvx_vec_splat_f32(kp->qk_eps);
            const HVX_Vector denom_v = hvx_vec_inverse_f32(hvx_vec_splat_f32((float) hd));
            const HVX_Vector scl_v   = hvx_vec_splat_f32(kp->qk_scale);
            const HVX_Vector bias_v  = hvx_vec_splat_f32(kp->qk_bias);
            const uint32_t   n_units_h = nt * n_heads;
            uint32_t         u       = 0;
            for (; u + 4 <= n_units_h; u += 4) {
                uint8_t * p[4];
                for (uint32_t j = 0; j < 4; ++j) {
                    const uint32_t t = (u + j) / n_heads, h = (u + j) % n_heads;
                    p[j] = (uint8_t *) (dst_tile + (size_t) t * d_inner_tile) + (size_t) h * hd * sizeof(float);
                }
                if (qk_lossy) {
                    ssm_conv_qk_norm_x4_lossy(p[0], p[1], p[2], p[3], hd, eps_v, denom_v, scl_v, bias_v, qk_iters, qk_fuse);
                } else {
                    hvx_fast_rms_norm_scale_f32_x4(p[0], p[1], p[2], p[3], p[0], p[1], p[2], p[3], hd, eps_v, denom_v, scl_v, bias_v);
                }
            }
            for (; u < n_units_h; ++u) {
                const uint32_t t = u / n_heads, h = u % n_heads;
                uint8_t * head = (uint8_t *) (dst_tile + (size_t) t * d_inner_tile) + (size_t) h * hd * sizeof(float);
                hvx_fast_rms_norm_scale_f32_c(head, head, hd, eps_v, denom_v, scl_v, bias_v);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, (uint16_t) ch0);
        }

        if (last_blk) {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) ch0);
            hvx_ssm_conv_scatter_from_T(xt + (size_t) nt * d_inner_tile, d_inner_tile, n_state, state_out, tile_n);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) ch0);
        }

        out_ndesc[i & 1] = ssm_conv_chain_push_out(dma_q, octx, kp, n_q, n_k, buf, (uint8_t *) dst_tile, seq, ch0,
                                                   tile_n, t0, nt, last_blk);

        if (i >= 1) {
            for (uint32_t d = 0; d < out_ndesc[(i - 1) & 1]; ++d) {
                dma_queue_pop(dma_q);
            }
        }

        // claim and load the item two ahead into the buffer just consumed
        const uint32_t nx = atomic_fetch_add(&ccctx->next_item, 1);
        q[(i + 2) & 3] = nx;
        if (nx < n_items) {
            DYN_POS(nx, seq2, unit2, b2)
            const uint32_t t02 = b2 * n_tb, nt2 = MIN(n_tb, n_t - t02);
            uint8_t * nbuf = tile_spad_base + (i & 1) * kp->vtcm_buf_stride;
            if (b2 == 0) {
                ssm_conv_chain_push_in(dma_q, qkv, kp, nbuf, seq2, unit2 * hd, hd, t02, nt2);
            } else {
                ssm_conv_chain_push_in_hist(dma_q, qkv, kp, nbuf, seq2, unit2 * hd, hd, t02, nt2);
            }
        }
    }
    #undef DYN_POS

    if (any) {
        for (uint32_t d = 0; d < out_ndesc[last_i & 1]; ++d) {
            dma_queue_pop(dma_q);
        }
    }
}

static void ssm_conv_chain_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_chain_context *                      ccctx = (struct htp_ssm_conv_chain_context *) data;
    if (ccctx->kparams->balanced == 2) {
        ssm_conv_chain_thread_dyn(nth, ith, data);
        return;
    }
    struct htp_ops_context *                                 octx  = ccctx->octx;
    const struct htp_ssm_conv_chain_kernel_params * restrict kp    = ccctx->kparams;

    const struct htp_tensor * restrict st_in  = octx->src[0];
    const struct htp_tensor * restrict qkv    = octx->src[1];
    const struct htp_tensor * restrict wgt    = octx->src[2];
    const struct htp_tensor * restrict dst    = octx->dsts[0];
    const struct htp_tensor * restrict st_out = octx->dsts[1];

    dma_queue * dma_q = octx->ctx->dma[ith];

    const uint32_t d_conv       = kp->d_conv;
    const uint32_t n_state      = kp->n_state;
    const uint32_t n_t          = kp->n_t;
    const uint32_t d_inner_tile = kp->d_inner_tile;

    uint32_t ir0 = kp->d_inner_per_thread * ith;
    uint32_t ir1 = MIN(ir0 + kp->d_inner_per_thread, kp->d_inner);
    if (kp->balanced) {
        ir0 = ith ? (uint32_t) kp->ch_start[ith - 1] * VLEN_FP32 : 0;
        ir1 = ith + 1 < nth ? (uint32_t) kp->ch_start[ith] * VLEN_FP32 : kp->d_inner;
    }
    if (ir0 >= ir1) {
        atomic_fetch_sub(&ccctx->state_barrier, 1);
        return;
    }

    const uint32_t d_inner_per_thread = ir1 - ir0;
    const uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);

    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    float *   state_stage    = (float *) src0_spad_base;
    uint8_t * tile_spad_base = src0_spad_base + kp->vtcm_state_stage_bytes;
    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    const size_t weight_bytes = (size_t) d_inner_per_thread * d_conv * sizeof(float);

    float * wgt_raw = (float *) src1_spad_base;
    float * wgt_T   = (float *) (src1_spad_base + hex_round_up(weight_bytes, 128));

    // q/k l2-norm epilogue: channels [0, n_q) and [n_q, n_q + n_k) are whole heads of hd
    const uint32_t hd   = kp->qk_head_dim;
    const uint32_t n_q  = hd ? octx->dsts[2]->ne[0] * octx->dsts[2]->ne[1] : 0;
    const uint32_t n_k  = hd ? octx->dsts[3]->ne[0] * octx->dsts[3]->ne[1] : 0;
    const uint32_t n_qk = n_q + n_k;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    // LOSSY_GDN conv levers (GGML_HEXAGON_LOSSY_GDN bits 8..): rsqrt iterations / fused scale of the q/k norm
    const uint32_t lossy   = (octx->flags & HTP_OPFLAGS_LOSSY_CONV_MASK) >> HTP_OPFLAGS_LOSSY_CONV_SHIFT;
    const uint32_t qk_iters = (lossy & HTP_LOSSY_CONV_RSQRT1) ? 1u : (lossy & HTP_LOSSY_CONV_RSQRT2) ? 2u : 3u;
    const bool     qk_fuse  = (lossy & HTP_LOSSY_CONV_NORM_FUSE) != 0;
    const bool     qk_lossy = qk_iters != 3 || qk_fuse;
    const bool     silu_h16 = (lossy & HTP_LOSSY_CONV_SILU_H16) != 0 && kp->silu_variant >= 3;

    dma_queue_push(dma_q, dma_make_data((uint8_t *) wgt_raw, wgt->data + (size_t) ir0 * d_conv * sizeof(float)),
                   weight_bytes, weight_bytes, weight_bytes, 1);
    dma_queue_pop(dma_q);

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_to_T(wgt_raw, wgt_T, d_inner_per_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);

    // Ring (FIFO) push order: st(0..n_s-1) in0 in1 | out0 in2 | out1 in3 | ...; out(i) is 1 to 4 descriptors.
    // Token blocks are innermost per channel tile; block b > 0 takes its first n_state input rows from block b - 1.
    const uint32_t n_tb        = kp->n_tb ? kp->n_tb : n_t;
    const uint32_t n_blk       = (n_t + n_tb - 1) / n_tb;
    const uint32_t n_tiles_seq = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
    const uint32_t n_tiles     = n_tiles_seq * n_blk * kp->n_s;
    const uint32_t n_buf       = MIN(kp->n_buf, n_tiles);

    // dst[0] may be allocated on top of src[0], so the whole conv state is staged before any output DMA
    for (uint32_t s = 0; s < kp->n_s; ++s) {
        ssm_conv_chain_push_state_in(dma_q, st_in, kp, state_stage, (size_t) kp->n_state * kp->d_inner_per_thread, s, ir0, d_inner_per_thread);
    }

    for (uint32_t i = 0; i < 2 && i < n_tiles; ++i) {
        const struct ssm_conv_chain_tile_pos p = ssm_conv_chain_tile(i, n_tiles_seq, n_blk, n_tb, n_t, d_inner_tile,
                                                                     d_inner_per_thread);
        ssm_conv_chain_push_in(dma_q, qkv, kp, tile_spad_base + (i % n_buf) * kp->vtcm_buf_stride, p.seq,
                               ir0 + p.tile_off, p.tile_n, p.t0, p.nt);
    }

    for (uint32_t s = 0; s < kp->n_s; ++s) {
        dma_queue_pop(dma_q);
    }

    // rings are per thread, so wait until every thread has staged its conv state
    atomic_fetch_sub(&ccctx->state_barrier, 1);
    while (atomic_load(&ccctx->state_barrier) > 0) {
    }

    uint32_t out_ndesc[2] = { 0, 0 };

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const struct ssm_conv_chain_tile_pos p = ssm_conv_chain_tile(i, n_tiles_seq, n_blk, n_tb, n_t, d_inner_tile,
                                                                     d_inner_per_thread);
        const uint32_t tile_off = p.tile_off;
        const uint32_t tile_n   = p.tile_n;
        const uint32_t nt       = p.nt;
        const uint32_t ch0      = ir0 + tile_off;
        const bool     last_blk = p.t0 + nt == n_t;

        uint8_t * buf       = tile_spad_base + (i % n_buf) * kp->vtcm_buf_stride;
        float *   xt        = (float *) buf;
        float *   state_out = (float *) (buf + kp->vtcm_xt_bytes);
        float *   dst_tile  = (float *) (dst_spad_base + (i % n_buf) * kp->vtcm_dst_tile_bytes);

        dma_queue_pop(dma_q);

        if (p.t0 == 0) {
            const float * stage = state_stage + (size_t) p.seq * n_state * kp->d_inner_per_thread;
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
            hvx_ssm_conv_unpack_to_T(stage + (size_t) tile_off * n_state, xt, tile_n, d_inner_tile, n_state);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        ssm_conv_tile_mac(xt, d_inner_tile, wgt_T + tile_off, d_inner_stride, d_conv, tile_n, nt, dst_tile,
                          d_inner_tile, false, hvx_vec_splat_f32(1.f), hvx_vec_splat_f32(87.f), hvx_vec_splat_f32(-87.f));
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);

        // one silu pass over the whole tile, as the fused SSM_CONV+SILU store does
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_OUT, (uint16_t) tile_off);
        if (silu_h16) {
            ssm_conv_chain_silu_tile_lossy((uint8_t *) dst_tile, nt * d_inner_tile / VLEN_FP32, kp->silu_variant);
        } else {
            ssm_conv_chain_silu_tile((uint8_t *) dst_tile, nt * d_inner_tile / VLEN_FP32, kp->silu_variant);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_OUT, (uint16_t) tile_off);

        // q/k l2-norm of the tile's whole heads below n_qk: one fused RMS_NORM+SCALE pass per head, in place (bit-exact with the
        // two unfused ops, hvx_fast_rms_norm_scale_f32)
        if (ch0 < n_qk) {
            const uint32_t n_heads = (MIN(ch0 + tile_n, n_qk) - ch0) / hd;
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_PREP, (uint16_t) tile_off);
            if ((hd % VLEN_FP32) == 0) {
                // four heads at a time: the sum / reduce / rsqrt latency chains of the heads interleave (bit-exact with one at a time)
                const HVX_Vector eps_v   = hvx_vec_splat_f32(kp->qk_eps);
                const HVX_Vector denom_v = hvx_vec_inverse_f32(hvx_vec_splat_f32((float) hd));
                const HVX_Vector scl_v   = hvx_vec_splat_f32(kp->qk_scale);
                const HVX_Vector bias_v  = hvx_vec_splat_f32(kp->qk_bias);
                const uint32_t   n_units = nt * n_heads;
                uint32_t         u       = 0;
                for (; u + 4 <= n_units; u += 4) {
                    uint8_t * p[4];
                    for (uint32_t j = 0; j < 4; ++j) {
                        const uint32_t t = (u + j) / n_heads, h = (u + j) % n_heads;
                        p[j] = (uint8_t *) (dst_tile + (size_t) t * d_inner_tile) + (size_t) h * hd * sizeof(float);
                    }
                    if (qk_lossy) {
                        ssm_conv_qk_norm_x4_lossy(p[0], p[1], p[2], p[3], hd, eps_v, denom_v, scl_v, bias_v, qk_iters, qk_fuse);
                    } else {
                        hvx_fast_rms_norm_scale_f32_x4(p[0], p[1], p[2], p[3], p[0], p[1], p[2], p[3], hd, eps_v, denom_v, scl_v, bias_v);
                    }
                }
                for (; u < n_units; ++u) {
                    const uint32_t t = u / n_heads, h = u % n_heads;
                    uint8_t * head = (uint8_t *) (dst_tile + (size_t) t * d_inner_tile) + (size_t) h * hd * sizeof(float);
                    hvx_fast_rms_norm_scale_f32_c(head, head, hd, eps_v, denom_v, scl_v, bias_v);
                }
            } else {
                for (uint32_t t = 0; t < nt; ++t) {
                    uint8_t * row = (uint8_t *) (dst_tile + (size_t) t * d_inner_tile);
                    for (uint32_t h = 0; h < n_heads; ++h) {
                        uint8_t * head = row + (size_t) h * hd * sizeof(float);
                        hvx_fast_rms_norm_scale_f32(head, head, hd, kp->qk_eps, kp->qk_scale, kp->qk_bias);
                    }
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, (uint16_t) tile_off);
        }

        if (last_blk) {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) tile_off);
            hvx_ssm_conv_scatter_from_T(xt + (size_t) nt * d_inner_tile, d_inner_tile, n_state, state_out, tile_n);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) tile_off);
        }

        out_ndesc[i & 1] = ssm_conv_chain_push_out(dma_q, octx, kp, n_q, n_k, buf, (uint8_t *) dst_tile, p.seq, ch0,
                                                   tile_n, p.t0, nt, last_blk);

        if (i >= 1) {
            for (uint32_t d = 0; d < out_ndesc[(i - 1) & 1]; ++d) {
                dma_queue_pop(dma_q);
            }
        }

        // the next block of this channel tile starts with the last n_state conv input rows of this one
        if (!last_blk) {
            float * xt_next = (float *) (tile_spad_base + ((i + 1) % n_buf) * kp->vtcm_buf_stride);
            for (uint32_t j = 0; j < n_state; ++j) {
                const HVX_Vector * vs = (const HVX_Vector *) (xt + (size_t) (nt + j) * d_inner_tile);
                HVX_Vector *       vd = (HVX_Vector *) (xt_next + (size_t) j * d_inner_tile);
                for (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
                    vd[cb / VLEN_FP32] = vs[cb / VLEN_FP32];
                }
            }
        }

        if (i + 2 < n_tiles) {
            const struct ssm_conv_chain_tile_pos q =
                ssm_conv_chain_tile(i + 2, n_tiles_seq, n_blk, n_tb, n_t, d_inner_tile, d_inner_per_thread);
            ssm_conv_chain_push_in(dma_q, qkv, kp, tile_spad_base + ((i + 2) % n_buf) * kp->vtcm_buf_stride, q.seq,
                                   ir0 + q.tile_off, q.tile_n, q.t0, q.nt);
        }
    }

    for (uint32_t d = 0; d < out_ndesc[(n_tiles - 1) & 1]; ++d) {
        dma_queue_pop(dma_q);
    }

    FARF(HIGH, "ssm-conv-chain-f32 %d/%d: (%u:%u) state %ux%u x qkv %ux%u -> %ux%u + state %u\n",
         ith, nth, ir0, ir1, st_in->ne[0], st_in->ne[1], qkv->ne[0], qkv->ne[1], dst->ne[0], dst->ne[1], st_out->ne[0]);
}

int op_ssm_conv_chain(struct htp_ops_context * octx) {
    const struct htp_tensor * st_in  = octx->src[0];
    const struct htp_tensor * qkv    = octx->src[1];
    const struct htp_tensor * wgt    = octx->src[2];
    const struct htp_tensor * dst    = octx->dsts[0];
    const struct htp_tensor * st_out = octx->dsts[1];

    if (st_in->type != HTP_TYPE_F32 || qkv->type != HTP_TYPE_F32 || wgt->type != HTP_TYPE_F32 ||
        dst->type != HTP_TYPE_F32 || st_out->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (octx->ctx->mdev.count > 1) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_ssm_conv_chain_kernel_params * kparams = (const struct htp_ssm_conv_chain_kernel_params *) octx->kernel_params;

    if (kparams->d_conv == 0 || kparams->d_conv > 32 || kparams->n_state + 1 != kparams->d_conv ||
        kparams->d_inner == 0 || kparams->n_t == 0 || kparams->n_s == 0 || kparams->d_inner_tile == 0) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (kparams->n_s != dst->ne[2] || kparams->n_t != dst->ne[1] || kparams->d_inner != dst->ne[0]) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (kparams->n_s > 1 && (kparams->st_in_seq_stride == 0 || kparams->qkv_seq_stride == 0 ||
                             kparams->dst_seq_stride == 0 || kparams->st_out_seq_stride == 0)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    // token blocks and channel tiles: more than one tile needs two buffers, or the look-ahead DMA overwrites the tile in use
    const uint32_t n_tb        = kparams->n_tb ? kparams->n_tb : kparams->n_t;
    const uint32_t n_blk       = (kparams->n_t + n_tb - 1) / n_tb;
    const uint32_t n_tiles_seq = (kparams->d_inner_per_thread + kparams->d_inner_tile - 1) / kparams->d_inner_tile;
    if (kparams->n_buf == 0 || (kparams->n_buf < 2 && (uint64_t) n_tiles_seq * n_blk * kparams->n_s > 1)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const uint32_t hd = kparams->qk_head_dim;

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.size_per_thread = kparams->vtcm_src0_size_per_thread;
    octx->src1_spad.size_per_thread = kparams->vtcm_src1_size_per_thread;
    octx->dst_spad.size_per_thread  = kparams->vtcm_dst_size_per_thread;

    octx->src0_spad.size = kparams->vtcm_src0_size_per_thread * kparams->n_threads;
    octx->src1_spad.size = kparams->vtcm_src1_size_per_thread * kparams->n_threads;
    octx->dst_spad.size  = kparams->vtcm_dst_size_per_thread  * kparams->n_threads;

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;
    octx->src0_spad.src  = NULL;
    octx->src1_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    struct htp_ssm_conv_chain_context ccctx = {
        .octx    = octx,
        .kparams = kparams,
    };
    atomic_init(&ccctx.state_barrier, octx->n_threads);
    atomic_init(&ccctx.next_item, 0);
    ccctx.shared_base = octx->dst_spad.data + octx->dst_spad.size;

    FARF(HIGH, "ssm-conv-chain-f32: state (%ux%u) qkv (%ux%u) w (%ux%u) -> (%ux%u) : %s n_s %u tile %u\n",
         st_in->ne[0], st_in->ne[1], qkv->ne[0], qkv->ne[1], wgt->ne[0], wgt->ne[1],
         dst->ne[0], dst->ne[1], kparams->n_t == 1 ? "decode" : "prefill", kparams->n_s, kparams->d_inner_tile);
    FARF(HIGH, "ssm-conv-chain-f32: token block %u qk head_dim %u\n", n_tb, hd);

    work_queue_run(octx->ctx->work_queue, ssm_conv_chain_thread_f32, &ccctx, octx->n_threads);

    return HTP_STATUS_OK;
}
