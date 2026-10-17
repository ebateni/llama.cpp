#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "work-queue.h"
#include "hvx-utils.h"
#include "hex-utils.h"
#include "hvx-copy.h"
#include "hvx-quant.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "hex-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp/set-rows-ops.h"

#define set_rows_preamble                      \
    const uint32_t ne00 = octx->src[0]->ne[0]; \
    const uint32_t ne01 = octx->src[0]->ne[1]; \
    const uint32_t ne02 = octx->src[0]->ne[2]; \
    const uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    const uint32_t ne10 = octx->src[1]->ne[0]; \
    const uint32_t ne11 = octx->src[1]->ne[1]; \
    const uint32_t ne12 = octx->src[1]->ne[2]; \
    const uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    const uint32_t nb01 = octx->src[0]->nb[1]; \
    const uint32_t nb02 = octx->src[0]->nb[2]; \
    const uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    const uint32_t nb10 = octx->src[1]->nb[0]; \
    const uint32_t nb11 = octx->src[1]->nb[1]; \
    const uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    const uint32_t nb1 = octx->dst->nb[1];     \
    const uint32_t nb2 = octx->dst->nb[2];     \
    const uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    const uint32_t ne0 = octx->dst->ne[0];     \
    const uint32_t ne1 = octx->dst->ne[1];     \
    const uint32_t ne2 = octx->dst->ne[2];     \
    const uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    const uint32_t nr  = ne01;

struct set_rows_context {
    struct htp_ops_context * octx;
    const struct htp_set_rows_kernel_params * kparams;
    struct htp_set_rows_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_per_thread;
};

#define SET_ROWS_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_EXPR)                                                \
static void set_rows_thread_dma_##TYPE_NAME##_##IDX_TYPE(unsigned int nth, unsigned int ith, void *data) {       \
    struct set_rows_context * srctx = (struct set_rows_context *)data;                                           \
    struct htp_ops_context * octx = srctx->octx;                                                                 \
    const struct htp_set_rows_kernel_params * kparams = srctx->kparams;                                          \
    set_rows_preamble;                                                                                           \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    const uint32_t dr  = srctx->tasks_per_thread;                                                                \
    const uint32_t ir0 = srctx->task_start + dr * ith;                                                           \
    if (ir0 >= srctx->task_start + srctx->tasks) {                                                               \
        return;                                                                                                  \
    }                                                                                                            \
    const uint32_t ir1 = MIN(ir0 + dr, srctx->task_start + srctx->tasks);                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                     \
    const struct htp_set_rows_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_base + vtcm_layout->off_src0 + ith * vtcm_layout->src0_bytes_per_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_base + vtcm_layout->off_dst  + ith * vtcm_layout->dst_bytes_per_thread;    \
    const uint32_t src0_row_size = ne00 * sizeof(float);                                                         \
    const uint32_t dst_row_size  = htp_tensor_get_row_size(octx->dst->type, ne00);                               \
    const uint32_t nrows_per_thread = ir1 - ir0;                                                                 \
    const uint32_t total_steps = ne03 * ne02 * nrows_per_thread;                                                 \
    uint32_t pi_step = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    for (uint32_t step = 0, spad_idx = 0; step < total_steps && spad_idx < 2; ++step, spad_idx++) {              \
        uint32_t i = ir0 + pi_step;                                                                              \
        const dma_addr_t src0_data = octx->src[0]->data + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_queue_push(dma_q,                                                                                    \
                       dma_make_data(octx->dst->data,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_size),                     \
                       dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                          \
        dma_queue_push(dma_q,                                                                                    \
                       dma_make_data(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_size,                    \
                                     src0_data),                                                                 \
                       vtcm_layout->src0_spad_half_size, src0_row_size, src0_row_size, 1);                       \
        pi_step++;                                                                                               \
        if (pi_step == nrows_per_thread) {                                                                       \
            pi_step = 0;                                                                                         \
            pi02++;                                                                                              \
            if (pi02 == ne02) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_step = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_base = 0;                                                                                      \
    uint32_t ci12_base = 0;                                                                                      \
    for (uint32_t step = 0; step < total_steps; ++step) {                                                        \
        void * dst_spad = (void *) dma_queue_pop(dma_q).src;                                                     \
        void * src_spad = (void *) dma_queue_pop(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_step;                                                                              \
        const uintptr_t src1_addr = octx->src[1]->data + i*nb10 + ci11_base*nb11 + ci12_base*nb12;               \
        const IDX_TYPE i1 = *(const IDX_TYPE *)src1_addr;                                                        \
        const bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        const uint32_t target_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                 \
        if (valid_i1) {                                                                                          \
            COMPUTE_EXPR;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                  \
        if (valid_i1) {                                                                                          \
            const dma_addr_t dst_data = octx->dst->data + target_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_queue_push(dma_q,                                                                                \
                           dma_make_data(dst_data, dst_spad),                                                    \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 1);                      \
        } else {                                                                                                 \
            dma_queue_push(dma_q,                                                                                \
                           dma_make_data(octx->dst->data, dst_spad),                                             \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                      \
        }                                                                                                        \
        const uint32_t next_step = step + 2;                                                                     \
        if (next_step < total_steps) {                                                                           \
            uint32_t ni = ir0 + pi_step;                                                                         \
            const dma_addr_t psrc0_data = octx->src[0]->data + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_queue_push(dma_q,                                                                                \
                           dma_make_data(src_spad, psrc0_data),                                                  \
                           vtcm_layout->src0_spad_half_size, src0_row_size, src0_row_size, 1);                   \
            pi_step++;                                                                                           \
            if (pi_step == nrows_per_thread) {                                                                   \
                pi_step = 0;                                                                                     \
                pi02++;                                                                                          \
                if (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_step++;                                                                                               \
        if (ci_step == nrows_per_thread) {                                                                       \
            ci_step = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_base++;                                                                                         \
            if (ci11_base == ne11) {                                                                             \
                ci11_base = 0;                                                                                   \
            }                                                                                                    \
            if (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_base++;                                                                                     \
                if (ci12_base == ne12) {                                                                         \
                    ci12_base = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_queue_flush(dma_q);                                                                                      \
}

SET_ROWS_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_row_q8_0_f32(dst_spad, (const float *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_row_q8_0_f32(dst_spad, (const float *)src_spad, ne00); })

// Block path: when a thread's destination indices are consecutive and valid (the KV-cache write), B rows move with
// one 2D DMA in and one 2D DMA out instead of one descriptor pair per row. Only the DMA grouping differs from the
// per-row path (same conversion on the same rows), so results are identical.
#define SET_ROWS_BLOCK 16

#define SET_ROWS_THREAD_BLOCK_FN(TYPE_NAME, IDX_TYPE, COMPUTE_ROW)                                               \
static void set_rows_thread_block_##TYPE_NAME##_##IDX_TYPE(unsigned int nth, unsigned int ith, void *data) {     \
    struct set_rows_context * srctx = (struct set_rows_context *)data;                                           \
    struct htp_ops_context * octx = srctx->octx;                                                                 \
    set_rows_preamble;                                                                                           \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    const uint32_t dr  = srctx->tasks_per_thread;                                                                \
    const uint32_t ir0 = srctx->task_start + dr * ith;                                                           \
    if (ir0 >= srctx->task_start + srctx->tasks) {                                                               \
        return;                                                                                                  \
    }                                                                                                            \
    const uint32_t ir1 = MIN(ir0 + dr, srctx->task_start + srctx->tasks);                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                     \
    const struct htp_set_rows_vtcm_layout * vl = &srctx->vtcm_layout;                                            \
    const uint32_t src_stride_v = vl->src0_spad_half_size;                                                       \
    const uint32_t dst_stride_v = vl->dst_spad_half_size;                                                        \
    /* block layout: per thread 2 slots x BLOCK rows for the source, same for the destination */                \
    const size_t src_thr = (size_t) 2 * SET_ROWS_BLOCK * src_stride_v;                                           \
    const size_t dst_thr = (size_t) 2 * SET_ROWS_BLOCK * dst_stride_v;                                           \
    uint8_t * vtcm_src0 = srctx->vtcm_base + ith * src_thr;                                                      \
    uint8_t * vtcm_dst  = srctx->vtcm_base + (size_t) octx->n_threads * src_thr + ith * dst_thr;                 \
    const uint32_t src0_row_size = ne00 * sizeof(float);                                                         \
    const uint32_t dst_row_size  = htp_tensor_get_row_size(octx->dst->type, ne00);                               \
    const uint32_t nrows = ir1 - ir0;                                                                            \
    const uint32_t nblk  = (nrows + SET_ROWS_BLOCK - 1) / SET_ROWS_BLOCK;                                        \
    const uint8_t * idx_base = (const uint8_t *) octx->src[1]->data;                                             \
    for (uint32_t step = 0; step < MIN(nblk, 2u); ++step) {                                                      \
        const uint32_t i0  = ir0 + step * SET_ROWS_BLOCK;                                                        \
        const uint32_t cnt = MIN((uint32_t) SET_ROWS_BLOCK, ir1 - i0);                                           \
        dma_queue_push(dma_q, dma_make_data(octx->dst->data, vtcm_dst + step * SET_ROWS_BLOCK * dst_stride_v),   \
                       dst_row_size, dst_stride_v, dst_row_size, 0);                                             \
        dma_queue_push(dma_q, dma_make_data(vtcm_src0 + step * SET_ROWS_BLOCK * src_stride_v,                    \
                                            octx->src[0]->data + (size_t) i0 * nb01),                            \
                       src_stride_v, nb01, src0_row_size, cnt);                                                  \
    }                                                                                                            \
    for (uint32_t step = 0; step < nblk; ++step) {                                                               \
        uint8_t * dst_spad = (uint8_t *) dma_queue_pop(dma_q).src;                                               \
        uint8_t * src_spad = (uint8_t *) dma_queue_pop(dma_q).dst;                                               \
        const uint32_t i0  = ir0 + step * SET_ROWS_BLOCK;                                                        \
        const uint32_t cnt = MIN((uint32_t) SET_ROWS_BLOCK, ir1 - i0);                                           \
        const IDX_TYPE i1 = *(const IDX_TYPE *) (idx_base + (size_t) i0 * nb10);                                 \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                 \
        for (uint32_t r = 0; r < cnt; ++r) {                                                                     \
            void * dst_spad_r = dst_spad + r * dst_stride_v;                                                     \
            void * src_spad_r = src_spad + r * src_stride_v;                                                     \
            COMPUTE_ROW;                                                                                         \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                  \
        dma_queue_push(dma_q, dma_make_data(octx->dst->data + (size_t) (uint32_t) i1 * nb1, dst_spad),           \
                       nb1, dst_stride_v, dst_row_size, cnt);                                                    \
        const uint32_t next = step + 2;                                                                          \
        if (next < nblk) {                                                                                       \
            const uint32_t ni0  = ir0 + next * SET_ROWS_BLOCK;                                                   \
            const uint32_t ncnt = MIN((uint32_t) SET_ROWS_BLOCK, ir1 - ni0);                                     \
            dma_queue_push(dma_q, dma_make_data(src_spad, octx->src[0]->data + (size_t) ni0 * nb01),             \
                           src_stride_v, nb01, src0_row_size, ncnt);                                             \
        }                                                                                                        \
    }                                                                                                            \
    dma_queue_flush(dma_q);                                                                                      \
}

SET_ROWS_THREAD_BLOCK_FN(f32, int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad_r, (const uint8_t *)src_spad_r, ne00); })
SET_ROWS_THREAD_BLOCK_FN(f32, int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad_r, (const uint8_t *)src_spad_r, ne00); })
SET_ROWS_THREAD_BLOCK_FN(f16, int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad_r, (const uint8_t *)src_spad_r, ne00); })
SET_ROWS_THREAD_BLOCK_FN(f16, int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad_r, (const uint8_t *)src_spad_r, ne00); })

// all indices of every thread's range valid and consecutive (a single run over the whole task range)
static bool set_rows_indices_consecutive(const struct htp_ops_context * octx, uint32_t task_start, uint32_t tasks, bool is_i32, uint32_t ne1) {
    const uint8_t * base = (const uint8_t *) octx->src[1]->data;
    const uint32_t nb10 = octx->src[1]->nb[0];
    const int64_t first = is_i32 ? (int64_t) *(const int32_t *) (base + (size_t) task_start * nb10)
                                 : *(const int64_t *) (base + (size_t) task_start * nb10);
    if (first < 0 || (uint64_t) first + tasks > (uint64_t) ne1) {
        return false;
    }
    for (uint32_t k = 1; k < tasks; ++k) {
        const int64_t v = is_i32 ? (int64_t) *(const int32_t *) (base + (size_t) (task_start + k) * nb10)
                                 : *(const int64_t *) (base + (size_t) (task_start + k) * nb10);
        if (v != first + (int64_t) k) {
            return false;
        }
    }
    return true;
}

int op_set_rows(struct htp_ops_context * octx) {
    const struct htp_set_rows_kernel_params * kparams = (const struct htp_set_rows_kernel_params *)octx->kernel_params;
    set_rows_preamble;

    if (octx->src[0]->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (htp_tensor_is_extended(octx->src[1])) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_tensor * dst = octx->dst;
    const uint32_t total_tasks    = kparams->total_tasks;

    uint32_t task_start = 0;
    uint32_t tasks      = total_tasks;

    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0 && !htp_tensor_is_permuted(dst);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_tasks, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        task_start = range.start;
        tasks      = range.count;
    }

    if (tasks == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_threads = octx->n_threads;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_block((const void *)octx->src[1]->data, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    struct set_rows_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_start = task_start;
    srctx.tasks = tasks;
    srctx.tasks_per_thread = fastdiv(tasks + n_threads - 1, &octx->n_threads_div);

    htp_set_rows_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_threads);
    srctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;

    work_queue_func_t q_func = NULL;
    const bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    switch (octx->dst->type) {
        case HTP_TYPE_F32:  q_func = is_i32 ? set_rows_thread_dma_f32_int32_t  : set_rows_thread_dma_f32_int64_t;  break;
        case HTP_TYPE_F16:  q_func = is_i32 ? set_rows_thread_dma_f16_int32_t  : set_rows_thread_dma_f16_int64_t;  break;
        case HTP_TYPE_Q8_0: q_func = is_i32 ? set_rows_thread_dma_q8_0_int32_t : set_rows_thread_dma_q8_0_int64_t; break;
        default:            return HTP_STATUS_NO_SUPPORT;
    }

    // block path: 2D destination, dst rows of one thread consecutive, f32/f16 destination, enough VTCM
    if (ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 && ne11 == 1 &&
        (octx->dst->type == HTP_TYPE_F32 || octx->dst->type == HTP_TYPE_F16) &&
        (size_t) n_threads * 2 * SET_ROWS_BLOCK * (srctx.vtcm_layout.src0_spad_half_size + srctx.vtcm_layout.dst_spad_half_size) <= octx->ctx->vtcm_size &&
        set_rows_indices_consecutive(octx, task_start, tasks, is_i32, ne1) &&
        (nb01 | nb1 | srctx.vtcm_layout.src0_spad_half_size | srctx.vtcm_layout.dst_spad_half_size) <= DMA_MAX_2D_ROW_SIZE) {
        const bool f32 = octx->dst->type == HTP_TYPE_F32;
        work_queue_func_t b_func = f32 ? (is_i32 ? set_rows_thread_block_f32_int32_t : set_rows_thread_block_f32_int64_t)
                                       : (is_i32 ? set_rows_thread_block_f16_int32_t : set_rows_thread_block_f16_int64_t);
        work_queue_run(octx->ctx->work_queue, b_func, &srctx, n_threads);
        return HTP_STATUS_OK;
    }

    FARF(HIGH, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_bytes_per_thread * n_threads,
         srctx.vtcm_layout.dst_bytes_per_thread  * n_threads,
         n_threads);

    work_queue_run(octx->ctx->work_queue, q_func, &srctx, n_threads);

    return HTP_STATUS_OK;
}
