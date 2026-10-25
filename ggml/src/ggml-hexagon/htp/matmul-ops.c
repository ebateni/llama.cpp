#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_compute_res.h>

#include <math.h>
#include <string.h>
#include <stdatomic.h>

#include "dma-queue.h"
#include "hvx-utils.h"
#include "hvx-dump.h"
#include "hvx-arith.h"
#include "hvx-reduce.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "matmul-ops.h"
#include "hmx-act-tiles.h"
#include "hmx-swiglu-tiles.h"
#include "hmx-a8w8.h"
#include "htp-vtcm.h"

typedef struct {
    float        *dst;
    dma_addr_t    src2_addr;
    size_t        src2_bytes;
    dma_addr_t    act_dma_addr;
    dma_addr_t    weight;
    dma_queue *   weight_dma;
    int           m;
    int           k;
    int           n;
    int           act_stride;
    int           weight_stride;
    int           dst_stride;
    uint32_t      src2_stride;
    int           ne02;
    int           ne03;
    int           ne12;
    int           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        act_nb2;
    size_t        act_nb3;
    size_t        src2_nb2;
    size_t        src2_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
    int           r2;
    int           r3;
    struct fastdiv_values div_r2;
    struct fastdiv_values div_r3;
} hmx_mm_f16_f32_batched_params_t;

static bool htp_matmul_has_extended_weight(const struct htp_ops_context * octx, uint32_t n_weights) {
    for (uint32_t i = 0; i < n_weights; ++i) {
        if (htp_tensor_is_extended(octx->src[i])) {
            return true;
        }
    }
    return false;
}

struct htp_mm_context {
    const char * type;
    struct htp_ops_context * octx;
    const struct htp_tensor * act;

    void (*vec_dot_1x1)(const uint32_t n, float * restrict s0,
         const void * restrict vx0,
         const void * restrict vy0);

    void (*vec_dot_2x1)(const uint32_t n, float * restrict s0,
         const void * restrict vx0, const void * restrict vx1,
         const void * restrict vy0);

    void (*vec_dot_2x2)(const uint32_t n, float * restrict s0, float * restrict s1,
         const void * restrict vx0, const void * restrict vx1,
         const void * restrict vy0, const void * restrict vy1);

    void (*vec_dot_32x1)(const uint32_t n, float * restrict s,
         const void * restrict vx,
         const void * restrict vy, uint32_t valid_rows,
         const float * restrict sz);

    // Precomputed values
    uint32_t src0_nrows_per_thread;
    uint32_t src0_row_start;
    uint32_t src0_row_end;
    uint32_t src0_row_size_padded;
    uint32_t act_nrows;
    uint32_t cur_m_start;
    uint32_t cur_m_rows;

    struct fastdiv_values mm_div_ne12_ne1;
    struct fastdiv_values mm_div_ne1;
    struct fastdiv_values mm_div_r2;
    struct fastdiv_values mm_div_r3;
    struct fastdiv_values mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_first[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_ib_last[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_r[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_c[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_rows_per_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    const uint32_t * matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows;
    uint32_t mapping_stride;

    // Dynamic VTCM pointers allocated sequentially
    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_src2;
    uint8_t * vtcm_src3;
    uint8_t * vtcm_dst;
    uint8_t * vtcm_act_raw;

    // Cached strides
    uint32_t vtcm_src0_stride;
    uint32_t vtcm_src1_stride;
    uint32_t vtcm_src2_stride;
    uint32_t vtcm_src3_stride;
    uint32_t vtcm_act_raw_stride;

    // Cached thread offsets/sizes
    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_src2_size_per_thread;
    uint32_t vtcm_src3_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;
};

static int htp_mm_init_context(
    struct htp_ops_context * octx,
    const struct htp_mm_kernel_params * kparams
) {
    if (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (kparams->n_hmx) {
        if (kparams->n_act_threads <= 0 || kparams->n_act_threads > (int32_t) octx->n_threads) {
            return HTP_STATUS_INVAL_PARAMS;
        }
    }

    return HTP_STATUS_OK;
}

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
static const uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
    0x00, 0x00, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00, 0x00,
    0x00, 0x11, 0x10, 0x10, 0x10, 0x02, 0x00, 0x04, 0x00, 0x01, 0x02, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04,
    0x00, 0x00, 0x22, 0x20, 0x20, 0x20, 0x21, 0x22, 0x20, 0x24, 0x04, 0x00, 0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x02,
    0x00, 0x04, 0x00, 0x11, 0x12, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08,
    0x01, 0x02, 0x00, 0x04, 0x44, 0x40, 0x40, 0x40, 0x41, 0x40, 0x40, 0x40, 0x42, 0x40, 0x44, 0x40, 0x41, 0x42, 0x48,
    0x48, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x12, 0x10, 0x10, 0x10, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00,
    0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x22, 0x20, 0x24, 0x20, 0x21, 0x22, 0x20, 0x20,
};

// IQ4_NL dequantization LUT: maps 4-bit index (0-15) to int8 kvalue
// kvalues: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
static const uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#define htp_matmul_tensors_preamble                                 \
    const struct htp_tensor * restrict src0 = octx->src[0];         \
    const struct htp_tensor * restrict src1 = octx->src[1];         \
    const struct htp_tensor * restrict src2 = octx->src[2];         \
    const struct htp_tensor * restrict  dst = octx->dst;            \
                                                                    \
    const uint32_t ne00 = src0->ne[0];                              \
    const uint32_t ne01 = src0->ne[1];                              \
    const uint32_t ne02 = src0->ne[2];                              \
    const uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    const uint32_t ne10 = src1->ne[0];                              \
    const uint32_t ne11 = src1->ne[1];                              \
    const uint32_t ne12 = src1->ne[2];                              \
    const uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    const uint32_t ne20 = src2 ? src2->ne[0] : 0;                   \
    const uint32_t ne21 = src2 ? src2->ne[1] : 0;                   \
    const uint32_t ne22 = src2 ? src2->ne[2] : 0;                   \
    const uint32_t ne23 = src2 ? src2->ne[3] : 0;                   \
                                                                    \
    const uint32_t ne0 = dst->ne[0];                                \
    const uint32_t ne1 = dst->ne[1];                                \
    const uint32_t ne2 = dst->ne[2];                                \
    const uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    const uint32_t nb00 = src0->nb[0];                              \
    const uint32_t nb01 = src0->nb[1];                              \
    const uint32_t nb02 = src0->nb[2];                              \
    const uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    const uint32_t nb10 = src1->nb[0];                              \
    const uint32_t nb11 = src1->nb[1];                              \
    const uint32_t nb12 = src1->nb[2];                              \
    const uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    const uint32_t nb0 = dst->nb[0];                                \
    const uint32_t nb1 = dst->nb[1];                                \
    const uint32_t nb2 = dst->nb[2];                                \
    const uint32_t nb3 = dst->nb[3];

#define htp_matmul_preamble                                         \
    struct htp_mm_context * mmctx  = data;                          \
    struct htp_ops_context * octx  = mmctx->octx;                   \
    dma_queue * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#include "hvx-mm-kernels-tiled.h"
#include "hvx-mm-kernels-float.h"
#include "hmx-mm-kernels-tiled.h"

// Specialized repacked matmul macros
#define MATMUL_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                                       \
static void hvx_mm_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                                 \
    htp_matmul_preamble;                                                                                                                   \
                                                                                                                                           \
    const uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;                                                               \
    const uint32_t src1_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);                                              \
    const uint32_t cur_m_start = mmctx->cur_m_start;                                                                                       \
                                                                                                                                           \
    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;                                                  \
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);                                     \
                                                                                                                                           \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                 \
                                                                                                                                           \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                               \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                                       \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                                  \
                                                                                                                                           \
    const size_t dst_row_size  = nb1;                                                                                                      \
    const size_t src1_row_size = nb11;                                                                                                     \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                                    \
    const size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;                                                          \
                                                                                                                                           \
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;                                          \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                          \
    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                                       \
                                                                                                                                           \
    const dma_addr_t src0_row = src0->data;                                                                                                \
                                                                                                                                           \
    const uint32_t tile_size = TILE_SIZE;                                                                                                  \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                                       \
                                                                                                                                           \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                                      \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                                      \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                                    \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                             \
                                                                                                                                           \
    uint32_t ct_start = src0_start_row / 32;                                                                                               \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                          \
                                                                                                                                           \
    uint32_t push_ct = ct_start;                                                                                                           \
    if (src0_start_row < src0_end_row) {                                                                                                   \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                                         \
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                                        \
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                   \
        }                                                                                                                                  \
    }                                                                                                                                      \
                                                                                                                                           \
    if (src0_start_row >= src0_end_row) {                                                                                                  \
        return;                                                                                                                            \
    }                                                                                                                                      \
                                                                                                                                           \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                                      \
        const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;                                                                        \
                                                                                                                                           \
        int valid_rows = (int)ne0 - (int)(ct * 32);                                                                                        \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                                          \
                                                                                                                                           \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                             \
        uint32_t ir1 = 0;                                                                                                                  \
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                                           \
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                                    \
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                                    \
            float * restrict dst_row0 = (float *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));                                    \
            float * restrict dst_row1 = (float *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));                                    \
                                                                                                                                           \
            float * dst_ptr0 = &dst_row0[ct * 32];                                                                                         \
            float * dst_ptr1 = &dst_row1[ct * 32];                                                                                         \
                                                                                                                                           \
            const float * src2_ptr0 = NULL;                                                                                                \
            const float * src2_ptr1 = NULL;                                                                                                \
            if (src2) {                                                                                                                    \
                const float * restrict src2_row0 = (const float *) ((const uint8_t *) src2->data + ((cur_m_start + ir1+0) * src2_stride)); \
                const float * restrict src2_row1 = (const float *) ((const uint8_t *) src2->data + ((cur_m_start + ir1+1) * src2_stride)); \
                src2_ptr0 = &src2_row0[ct * 32];                                                                                           \
                src2_ptr1 = &src2_row1[ct * 32];                                                                                           \
            }                                                                                                                              \
            DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, src1_col0, src1_col1, valid_rows, src2_ptr0, src2_ptr1);                             \
        }                                                                                                                                  \
                                                                                                                                           \
        for (; ir1 < src1_nrows; ++ir1) {                                                                                                  \
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                                         \
            float * restrict dst_row          = (float *) (dst->data + ((cur_m_start + ir1) * dst_row_size));                              \
            float * dst_ptr = &dst_row[ct * 32];                                                                                           \
                                                                                                                                           \
            const float * src2_ptr = NULL;                                                                                                 \
            if (src2) {                                                                                                                    \
                const float * restrict src2_row = (const float *) ((const uint8_t *) src2->data + ((cur_m_start + ir1) * src2_stride));    \
                src2_ptr = &src2_row[ct * 32];                                                                                             \
            }                                                                                                                              \
            DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, src2_ptr);                                                                \
        }                                                                                                                                  \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                              \
                                                                                                                                           \
        if (push_ct < ct_end) {                                                                                                            \
            dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),                                             \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                          \
            push_ct++;                                                                                                                     \
        }                                                                                                                                  \
    }                                                                                                                                      \
}

#define MATVEC_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X1)                                                              \
static void hvx_mv_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                               \
    htp_matmul_preamble;                                                                                                 \
                                                                                                                         \
    const uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;                                             \
                                                                                                                         \
    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;                                \
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);                   \
                                                                                                                         \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                               \
                                                                                                                         \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;             \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                     \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                \
                                                                                                                         \
    const size_t dst_row_size  = nb1;                                                                                    \
    const size_t src1_row_size = nb11;                                                                                   \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                  \
                                                                                                                         \
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_thread * ith;                                   \
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                 \
    uint8_t * src1_data = mmctx->vtcm_src1;                                                                              \
                                                                                                                         \
    float * tmp = (float *) vtcm_dst_ptr;                                                                                \
                                                                                                                         \
    const dma_addr_t src0_row = src0->data;                                                                              \
                                                                                                                         \
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;                                                     \
    float * restrict dst_col          = (float *) dst->data;                                                             \
                                                                                                                         \
    const uint32_t tile_size = TILE_SIZE;                                                                                \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                     \
                                                                                                                         \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                    \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                    \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                  \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                           \
                                                                                                                         \
    uint32_t ct_start = src0_start_row / 32;                                                                             \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                        \
                                                                                                                         \
    uint32_t push_ct = ct_start;                                                                                         \
    if (src0_start_row < src0_end_row) {                                                                                 \
        if (src2) {                                                                                                      \
            float * vtcm_src2_ptr = (float *) mmctx->vtcm_src2 + src0_start_row;                                         \
            const dma_addr_t src2_addr = src2->data + src0_start_row * sizeof(float);                                    \
            int slice_size = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;                                          \
            if (slice_size > 0) {                                                                                        \
                dma_queue_push(dma_q, dma_make_data(vtcm_src2_ptr, src2_addr),                                           \
                               slice_size * sizeof(float), slice_size * sizeof(float), slice_size * sizeof(float), 1);   \
                dma_queue_pop_nowait(dma_q);                                                                             \
            }                                                                                                            \
        }                                                                                                                \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                       \
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                      \
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a); \
        }                                                                                                                \
    }                                                                                                                    \
                                                                                                                         \
    if (src0_start_row >= src0_end_row) {                                                                                \
        return;                                                                                                          \
    }                                                                                                                    \
                                                                                                                         \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                    \
        const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;                                                      \
                                                                                                                         \
        float * dst_ptr = &tmp[ct * 32 - src0_start_row];                                                                \
        int valid_rows = (int)ne0 - (int)(ct * 32);                                                                      \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                        \
                                                                                                                         \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                           \
        DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, NULL);                                                      \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                            \
                                                                                                                         \
        if (push_ct < ct_end) {                                                                                          \
            dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),                           \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                        \
            push_ct++;                                                                                                   \
        }                                                                                                                \
    }                                                                                                                    \
                                                                                                                         \
    int copy_cnt = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;                                                    \
    if (copy_cnt > 0) {                                                                                                  \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);                                                       \
        if (src2) {                                                                                                      \
            hvx_add_f32_uaa((uint8_t *) &dst_col[src0_start_row],                                                        \
                            (const uint8_t *) tmp,                                                                       \
                            (const uint8_t *) ((const float *) mmctx->vtcm_src2 + src0_start_row),                       \
                            copy_cnt);                                                                                   \
        } else {                                                                                                         \
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);                            \
        }                                                                                                                \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);                                                        \
    }                                                                                                                    \
}

#define MATMUL_NX_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                           \
static void hvx_mm_nx_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                     \
    struct htp_mm_context * mmctx = data;                                                                                         \
    struct htp_ops_context * octx = mmctx->octx;                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_weights = kparams->n_weights;                                                                                \
                                                                                                                                  \
    const struct htp_tensor * restrict act = octx->src[n_weights]; /* x */                                                        \
    const uint32_t ne10 = act->ne[0];                                                                                             \
    const uint32_t src1_nrows = act->ne[1] * act->ne[2] * act->ne[3];                                                             \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
                                                                                                                                  \
    uint8_t * restrict vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                               \
    uint8_t * restrict src1_data       = mmctx->vtcm_src1;                                                                        \
                                                                                                                                  \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                        \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
                                                                                                                                  \
    for (uint32_t widx = 0; widx < n_weights; widx++) {                                                                           \
        const struct htp_tensor * restrict src_w = octx->src[widx];                                                               \
        const struct htp_tensor * restrict dst   = octx->dsts[widx];                                                              \
        if (!src_w || !dst) continue;                                                                                             \
        dma_queue * dma_q = octx->ctx->dma[ith];                                                                                  \
                                                                                                                                  \
        const uint32_t ne00 = src_w->ne[0];                                                                                       \
        const uint32_t ne01 = src_w->ne[1];                                                                                       \
        const size_t dst_row_size = dst->nb[1];                                                                                   \
        const dma_addr_t src_w_row = src_w->data;                                                                                 \
                                                                                                                                  \
        uint32_t n_k_tiles_w = ne00 / 32;                                                                                         \
        uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                       \
                                                                                                                                  \
        uint32_t src0_start_row = 0;                                                                                              \
        uint32_t src0_end_row   = ne01;                                                                                           \
        if (octx->ctx->mdev.count > 1) {                                                                                          \
            const bool can_split = htp_tensor_can_row_partition(dst, sizeof(float));                                              \
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0,                        \
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div); \
            src0_start_row = range.start;                                                                                         \
            src0_end_row   = range.start + range.count;                                                                           \
        }                                                                                                                         \
                                                                                                                                  \
        const uint32_t nrows = src0_end_row - src0_start_row;                                                                     \
        uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);                                          \
        src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);                                                          \
                                                                                                                                  \
        const uint32_t start_row = src0_start_row + src0_nrows_per_thread * ith;                                                  \
        const uint32_t end_row   = MIN(start_row + src0_nrows_per_thread, src0_end_row);                                          \
        if (start_row >= end_row) continue;                                                                                       \
                                                                                                                                  \
        uint32_t ct_start = start_row / 32;                                                                                       \
        uint32_t ct_end   = (end_row + 31) / 32;                                                                                  \
                                                                                                                                  \
        uint32_t push_ct = ct_start;                                                                                              \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                                \
            dma_queue_push(dma_q, dma_make_data(vtcm_weight_ptr + d * tile_row_transfer_size_aligned,                             \
                           src_w_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);         \
        }                                                                                                                         \
                                                                                                                                  \
        for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                         \
            const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;                                                           \
            int valid_rows = (int)ne01 - (int)(ct * 32);                                                                          \
            valid_rows = MIN(32, MAX(0, valid_rows));                                                                             \
                                                                                                                                  \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                \
            uint32_t ir1 = 0;                                                                                                     \
            for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                              \
                const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                       \
                const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                       \
                                                                                                                                  \
                float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));                                     \
                float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));                                     \
                float * dst_ptr0 = &dst_row0[ct * 32];                                                                            \
                float * dst_ptr1 = &dst_row1[ct * 32];                                                                            \
                                                                                                                                  \
                DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, src1_col0, src1_col1, valid_rows, NULL, NULL);                          \
            }                                                                                                                     \
                                                                                                                                  \
            for (; ir1 < src1_nrows; ++ir1) {                                                                                     \
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                            \
                float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));                                          \
                float * dst_ptr = &dst_row[ct * 32];                                                                              \
                DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, NULL);                                                       \
            }                                                                                                                     \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                 \
                                                                                                                                  \
            if (push_ct < ct_end) {                                                                                               \
                dma_queue_push(dma_q, dma_make_data(w_tile, src_w_row + push_ct * tile_row_stride),                               \
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                             \
                push_ct++;                                                                                                        \
            }                                                                                                                     \
        }                                                                                                                         \
    }                                                                                                                             \
}

MATMUL_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_2D_REPACKED_IMPL(q6_k,       896,  tiled_vec_dot_q6_k_32x2,  tiled_vec_dot_q6_k_32x1)
MATMUL_2D_REPACKED_IMPL(q5_k,       768,  tiled_vec_dot_q5_k_32x2,  tiled_vec_dot_q5_k_32x1)
MATMUL_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

static void hvx_mm_transfer_src1_dma(
    struct htp_ops_context * octx,
    const struct htp_mm_kernel_params * kparams,
    const struct htp_tensor * src1,
    uint8_t * dst_base,
    size_t dst_row_size,
    uint32_t m_start,
    uint32_t m_rows
) {
    if (m_rows == 0) {
        return;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    const uint32_t ne0 = src1->ne[0];
    const size_t elem_size = (src1->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const size_t row_bytes = ne0 * elem_size;
    const size_t src1_nb1 = src1->nb[1];
    const dma_addr_t src_base = src1->data;

    const bool is_contiguous = (src1->nb[2] == src1->ne[1] * src1_nb1) &&
                               (src1->nb[3] == src1->ne[2] * src1->nb[2]);

    if (is_contiguous) {
        const dma_addr_t src_addr = src_base + m_start * src1_nb1;
        dma_queue_push(dma_q, dma_make_data(dst_base, src_addr),
                       dst_row_size, src1_nb1, row_bytes, m_rows);
        dma_queue_pop(dma_q);
    } else {
        const uint32_t ne12_ne1 = src1->ne[2] * src1->ne[1];
        const bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        for (uint32_t ir = 0; ir < m_rows; ++ir) {
            const uint32_t ir1 = m_start + ir;
            uint32_t i11, i12, i13;
            if (use_fastdiv) {
                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                const uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = fastdiv(rem, &kparams->div_ne1);
                i11 = rem - i12 * src1->ne[1];
            } else {
                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                const uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = src1->ne[1] ? rem / src1->ne[1] : 0;
                i11 = rem - i12 * src1->ne[1];
            }
            const dma_addr_t row_src = src_base + (i11 * src1->nb[1] +
                                                   i12 * src1->nb[2] +
                                                   i13 * src1->nb[3]);
            uint8_t * row_dst = dst_base + ir * dst_row_size;
            dma_queue_push(dma_q, dma_make_data(row_dst, row_src),
                           dst_row_size, src1_nb1, row_bytes, 1);
            dma_queue_pop(dma_q);
        }
    }

    if (dst_row_size > row_bytes) {
        struct htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
        const uint32_t pad_elems = (dst_row_size - row_bytes) / elem_size;
        if (elem_size == sizeof(float)) {
            for (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f32_u(dst_base + ir * dst_row_size + row_bytes, 0.0f, pad_elems);
            }
        } else {
            for (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f16_u(dst_base + ir * dst_row_size + row_bytes, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
    }
}

#define QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_size_expr)                                        \
static void name(unsigned int nth, unsigned int ith, void * data) {                                        \
    (void) nth;                                                                                            \
    struct htp_mm_context * mmctx = data;                                                                  \
    struct htp_ops_context * octx = mmctx->octx;                                                           \
    const struct htp_tensor * src = mmctx->act;                                                            \
    const uint32_t ne0 = src->ne[0];                                                                       \
    const uint32_t nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;                       \
    const uint32_t nrows_per_thread = mmctx->n_quant_rows_per_thread;                                      \
                                                                                                           \
    const uint32_t ir_first = nrows_per_thread * ith;                                                      \
    if (ir_first >= nrows) {                                                                               \
        return;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
                                                                                                           \
    uint8_t * restrict dst = mmctx->vtcm_src1;                                                             \
    const uint32_t ir_last = MIN(ir_first + nrows_per_thread, nrows);                                      \
    const size_t raw_row_size = mmctx->vtcm_act_raw_stride;                                                \
    const size_t dst_row_size = (dst_row_size_expr);                                                       \
                                                                                                           \
    const uint8_t * restrict src_data = (const uint8_t *) mmctx->vtcm_act_raw + (raw_row_size * ir_first); \
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);                             \
    kernel_fn(src_data, dst_data, NULL, ne0, ir_last - ir_first, raw_row_size, dst_row_size);              \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled, "quantize-f32-q8_0_tiled", quantize_f32_q8_0_tiled_kernel, htp_mm_q8_0_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled, "quantize-f32-q8_1_tiled", quantize_f32_q8_1_tiled_kernel, htp_mm_q8_1_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_f32,        "quantize-f32-f32",        quantize_f32_f32_kernel,        mmctx->vtcm_src1_stride)
QUANTIZE_IMPL(quantize_f32_f16,        "quantize-f32-f16",        quantize_f32_f16_kernel,        mmctx->vtcm_src1_stride)
QUANTIZE_IMPL(quantize_f16_f16,        "quantize-f16-f16",        quantize_f16_f16_kernel,        mmctx->vtcm_src1_stride)

static void quantize_f32_q8_0_tiled_block(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_mm_context * mmctx = data;
    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        return;
    }
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_block_kernel(
        (const float *) mmctx->vtcm_act_raw,
        mmctx->vtcm_src1,
        NULL,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

static void quantize_f32_q8_1_tiled_block(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_mm_context * mmctx = data;
    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        return;
    }
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_block_kernel(
        (const float *) mmctx->vtcm_act_raw,
        mmctx->vtcm_src1,
        NULL,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

MATVEC_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x1)
MATVEC_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x1)
MATVEC_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x1)
MATVEC_2D_REPACKED_IMPL(q5_k,       768,  tiled_vec_dot_q5_k_32x1)
MATVEC_2D_REPACKED_IMPL(q6_k,       896,  tiled_vec_dot_q6_k_32x1)
MATVEC_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x1)
MATVEC_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x1)

MATMUL_NX_2D_REPACKED_IMPL(q4_0,    576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_NX_2D_REPACKED_IMPL(q4_1,    640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_NX_2D_REPACKED_IMPL(q8_0,    1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_NX_2D_REPACKED_IMPL(iq4nl,   576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_NX_2D_REPACKED_IMPL(mxfp4,   544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)
MATMUL_NX_2D_REPACKED_IMPL(q5_k,    768,  tiled_vec_dot_q5_k_32x2,  tiled_vec_dot_q5_k_32x1)

#define MATMUL_4D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                                                        \
static void hvx_mm_4d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                                                  \
    htp_matmul_preamble;                                                                                                                                    \
                                                                                                                                                            \
    const uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;                                                                                \
    const uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);                                                               \
    const uint32_t cur_m_start = mmctx->cur_m_start;                                                                                                        \
                                                                                                                                                            \
    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;                                                                   \
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);                                                      \
                                                                                                                                                            \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                                  \
                                                                                                                                                            \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                                                \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                                                        \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                                                   \
                                                                                                                                                            \
    const size_t dst_row_size = nb1;                                                                                                                        \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                                                     \
                                                                                                                                                            \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                                           \
    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                                                        \
                                                                                                                                                            \
    const uint32_t tile_size = TILE_SIZE;                                                                                                                   \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                                                        \
                                                                                                                                                            \
    const uint32_t n_k_tiles_w = ne00 / 32;                                                                                                                 \
    const uint32_t n_k_tiles_a = ne10 / 32;                                                                                                                 \
    const uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                                               \
    const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                                        \
    const uint32_t src0_slice_stride = ((ne01 + 31) / 32) * tile_row_stride;                                                                                \
                                                                                                                                                            \
    const uint32_t ct_start = src0_start_row / 32;                                                                                                          \
    const uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                                     \
                                                                                                                                                            \
                                                                                                                                                            \
    if (src0_start_row >= src0_end_row || cur_m_rows == 0) {                                                                                                \
        return;                                                                                                                                             \
    }                                                                                                                                                       \
                                                                                                                                                            \
    const uint32_t total_batches = ne12 * ne13;                                                                                                             \
    const uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);                                                                                       \
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);                                                                       \
    b_end = MIN(b_end, total_batches);                                                                                                                      \
                                                                                                                                                            \
    uint32_t b_grp_start = b_start;                                                                                                                         \
    while (b_grp_start < b_end) {                                                                                                                           \
        const uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);                                                                                       \
        const uint32_t b2 = b_grp_start - b3 * ne12;                                                                                                        \
        const uint32_t i02 = fastdiv(b2, &kparams->div_r2);                                                                                                 \
        const uint32_t i03 = fastdiv(b3, &kparams->div_r3);                                                                                                 \
                                                                                                                                                            \
        uint32_t b_grp_end = b_grp_start + 1;                                                                                                               \
        while (b_grp_end < b_end) {                                                                                                                         \
            const uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);                                                                                 \
            const uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;                                                                                              \
            const uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);                                                                                     \
            const uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);                                                                                     \
            if (cur_i02 != i02 || cur_i03 != i03) {                                                                                                         \
                break;                                                                                                                                      \
            }                                                                                                                                               \
            b_grp_end++;                                                                                                                                    \
        }                                                                                                                                                   \
                                                                                                                                                            \
        const uint32_t slice_idx = i03 * ne02 + i02;                                                                                                        \
        const dma_addr_t src0_slice = src0->data + (size_t) slice_idx * src0_slice_stride;                                                                  \
                                                                                                                                                            \
        uint32_t push_ct = ct_start;                                                                                                                        \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                                                          \
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                                                         \
                           src0_slice + (size_t) push_ct * tile_row_stride),                                                                                \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                                           \
        }                                                                                                                                                   \
                                                                                                                                                            \
        for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                                                   \
            const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;                                                                                     \
                                                                                                                                                            \
            int valid_rows = (int)ne0 - (int)(ct * 32);                                                                                                     \
            valid_rows = MIN(32, MAX(0, valid_rows));                                                                                                       \
                                                                                                                                                            \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                                          \
            for (uint32_t b = b_grp_start; b < b_grp_end; b++) {                                                                                            \
                const uint32_t b_m_start = b * ne11;                                                                                                        \
                const uint32_t m_first = MAX(cur_m_start, b_m_start);                                                                                       \
                const uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);                                                                   \
                if (m_first >= m_last) continue;                                                                                                            \
                                                                                                                                                            \
                const uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);                                                                                     \
                const uint32_t cur_b2 = b - cur_b3 * ne12;                                                                                                  \
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;                                           \
                                                                                                                                                            \
                const uint32_t chunk_m_offset = m_first - cur_m_start;                                                                                      \
                const uint32_t dst_m_offset   = m_first - b_m_start;                                                                                        \
                const uint32_t batch_nrows    = m_last - m_first;                                                                                           \
                                                                                                                                                            \
                uint32_t ir1 = 0;                                                                                                                           \
                for (; ir1 + 1 < batch_nrows; ir1 += 2) {                                                                                                   \
                    const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (chunk_m_offset + ir1 + 0) * src1_stride);                          \
                    const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (chunk_m_offset + ir1 + 1) * src1_stride);                          \
                    float * restrict dst_row0 = (float *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);                                       \
                    float * restrict dst_row1 = (float *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);                                       \
                    float * dst_ptr0 = &dst_row0[ct * 32];                                                                                                  \
                    float * dst_ptr1 = &dst_row1[ct * 32];                                                                                                  \
                    DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, src1_col0, src1_col1, valid_rows, NULL, NULL);                                                \
                }                                                                                                                                           \
                for (; ir1 < batch_nrows; ++ir1) {                                                                                                          \
                    const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (chunk_m_offset + ir1) * src1_stride);                               \
                    float * restrict dst_row = (float *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);                                            \
                    float * dst_ptr = &dst_row[ct * 32];                                                                                                    \
                    DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, NULL);                                                                             \
                }                                                                                                                                           \
            }                                                                                                                                               \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                                           \
                                                                                                                                                            \
            if (push_ct < ct_end) {                                                                                                                         \
                dma_queue_push(dma_q, dma_make_data(w_tile, src0_slice + (size_t) push_ct * tile_row_stride),                                               \
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                                       \
                push_ct++;                                                                                                                                  \
            }                                                                                                                                               \
        }                                                                                                                                                   \
        b_grp_start = b_grp_end;                                                                                                                            \
    }                                                                                                                                                       \
    if (src2) {                                                                                                                                             \
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_rows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1); \
    }                                                                                                                                                       \
}

MATMUL_4D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_4D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_4D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_4D_REPACKED_IMPL(q6_k,       896,  tiled_vec_dot_q6_k_32x2,  tiled_vec_dot_q6_k_32x1)
MATMUL_4D_REPACKED_IMPL(q5_k,       768,  tiled_vec_dot_q5_k_32x2,  tiled_vec_dot_q5_k_32x1)
MATMUL_4D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_4D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

static void hvx_mm_2d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    const uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows
    const uint32_t src1_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;                          // src1 rows
    const uint32_t cur_m_start = mmctx->cur_m_start;

    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data     = mmctx->vtcm_src1;

    const dma_addr_t src0_row = src0->data;


    // Prefill vtcm with src0 rows
    if (src0_start_row < src0_end_row) {
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const int is0 = (ir0 - src0_start_row);
            if (is0 >= (int)n_prefetch) {
                break;
            }
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    if (src0_start_row >= src0_end_row) {
        return;
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process src1 columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);
            float * restrict dst_row0 = (float *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));
            float * restrict dst_row1 = (float *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);
        }

        // Handle remaining src1 rows (fallback to 2x1)
        for (; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
            float * restrict dst_row          = (float *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, src1_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        const int pr0 = (ir0 + n_prefetch);
        const int is0 = (pr0 - src0_start_row) & prefetch_mask;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #pragma unroll(2)
        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
            float * restrict dst_row          = (float *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, src1_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    if (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + src1_nrows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

static void hvx_mv_2d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;

    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * src1_data     = mmctx->vtcm_src1;

    float * tmp = (float *) vtcm_dst_ptr;

    const dma_addr_t src0_row = src0->data;
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
    float * restrict dst_col          = (float *) dst->data;

    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    if (src0_start_row < src0_end_row) {
        if (src2) {
            float * vtcm_src2_ptr = (float *) mmctx->vtcm_src2 + src0_start_row;
            const dma_addr_t src2_addr = src2->data + src0_start_row * sizeof(float);
            int slice_size = (int)src0_end_row - (int)src0_start_row;
            if (slice_size > 0) {
                dma_queue_push(dma_q, dma_make_data(vtcm_src2_ptr, src2_addr),
                               slice_size * sizeof(float), slice_size * sizeof(float), slice_size * sizeof(float), 1);
                dma_queue_pop_nowait(dma_q);
            }
        }
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const uint32_t is0 = (ir0 - src0_start_row);
            if (is0 >= n_prefetch) {
                break;
            }
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }


    if (src0_start_row >= src0_end_row) {
        return;
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, src1_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        const uint32_t pr0 = (ir0 + n_prefetch);
        const uint32_t is0 = (pr0 - src0_start_row) & prefetch_mask;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        const uint32_t ir0 = src0_end_row_x2;
        const uint32_t is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, src1_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    int copy_cnt = src0_end_row - src0_start_row;
    if (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
        if (src2) {
            hvx_add_f32_uaa((uint8_t *) &dst_col[src0_start_row],
                            (const uint8_t *) tmp,
                            (const uint8_t *) ((const float *) mmctx->vtcm_src2 + src0_start_row),
                            copy_cnt);
        } else {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
    }
}

static void hvx_mm_4d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    const uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    const uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    const uint32_t cur_m_start = mmctx->cur_m_start;

    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data     = mmctx->vtcm_src1;


    if (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        return;
    }

    const uint32_t total_batches = ne12 * ne13;
    const uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, total_batches);

    uint32_t b_grp_start = b_start;
    while (b_grp_start < b_end) {
        const uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        const uint32_t b2 = b_grp_start - b3 * ne12;
        const uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        const uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        while (b_grp_end < b_end) {
            const uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            const uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            const uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            const uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            if (cur_i02 != i02 || cur_i03 != i03) {
                break;
            }
            b_grp_end++;
        }

        const dma_addr_t src0_row = src0->data + ((size_t) i02 * nb02 + (size_t) i03 * nb03);

        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const int is0 = (ir0 - src0_start_row);
            if (is0 >= (int)n_prefetch) {
                break;
            }
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }

        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            for (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                const uint32_t b_m_start = b * ne11;
                const uint32_t m_first = MAX(cur_m_start, b_m_start);
                const uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                if (m_first >= m_last) continue;

                const uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                const uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                const uint32_t chunk_m_offset = m_first - cur_m_start;
                const uint32_t dst_m_offset   = m_first - b_m_start;
                const uint32_t batch_nrows    = m_last - m_first;

                uint32_t ir1 = 0;
                for (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (chunk_m_offset + ir1 + 0) * src1_stride);
                    const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (chunk_m_offset + ir1 + 1) * src1_stride);
                    float * restrict dst_row0 = (float *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);
                    float * restrict dst_row1 = (float *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);
                }
                for (; ir1 < batch_nrows; ++ir1) {
                    const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (chunk_m_offset + ir1) * src1_stride);
                    float * restrict dst_row          = (float *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, src1_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            const int pr0 = (ir0 + n_prefetch);
            const int is0 = (pr0 - src0_start_row) & prefetch_mask;
            if (pr0 < src0_end_row_x2) {
                dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) pr0 * src0_row_size),
                               src0_stride, src0_row_size, src0_row_size, 2);
            }
        }

        if (src0_end_row != src0_end_row_x2) {
            uint32_t ir0 = src0_end_row_x2;
            const int is0 = (ir0 - src0_start_row) & prefetch_mask;
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 1);
            const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            for (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                const uint32_t b_m_start = b * ne11;
                const uint32_t m_first = MAX(cur_m_start, b_m_start);
                const uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                if (m_first >= m_last) continue;

                const uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                const uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                const uint32_t chunk_m_offset = m_first - cur_m_start;
                const uint32_t dst_m_offset   = m_first - b_m_start;
                const uint32_t batch_nrows    = m_last - m_first;

                for (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (chunk_m_offset + ir1) * src1_stride);
                    float * restrict dst_row          = (float *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, src1_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_start = b_grp_end;
    }

    if (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_rows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * mmctx->mapping_stride + (i1)]

static void hvx_mm_id(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];

    const uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
    const uint32_t src1_nrows      = ne11;
    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);


    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    const uint32_t n_ids = ids->ne[0];  // n_expert_used
    const uint32_t n_as  = ne02;        // n_expert

    const uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;

    const size_t dst_row_size  = nb1;
    const size_t src1_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    const size_t src1_stride = mmctx->vtcm_src1_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) {
            continue;
        }

        const dma_addr_t src0_row = src0->data + cur_a * nb02;

        const uint32_t tile_size = htp_mm_get_weight_tile_size(src0->type);
        const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src0->type);
        const uint32_t n_k_tiles_w = ne00 / 32;
        const uint32_t n_k_tiles_a = ne10 / 32;
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
        }

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;

            int valid_rows = (int)ne01 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            for (uint32_t cid = 0; cid < cne1; ++cid) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                const int               rm1         = row_mapping.i1;  // expert idx
                const int               rm2         = row_mapping.i2;  // token idx

                const uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // src1 row idx
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (ir1 + rm2 * ne11 + 0) * src1_stride);
                float * restrict dst_row = (float *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            if (push_ct < ct_end) {
                dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                push_ct++;
            }
        }
    }
}

static void hvx_mv_id(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];

    const uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);


    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    assert(ne13 % ne03 == 0);

    const size_t dst_row_size  = nb1;
    const size_t src1_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    const uint32_t n_aids = src2->ne[0];  // num activated experts
    const uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        const int32_t eid = *(const int32_t *) ((const uint8_t *) src2->data + ie1 * src2->nb[0]);
        if (eid == -1) {
            continue;
        }
        assert(eid >= 0 && eid < (int32_t) n_ids);

        const dma_addr_t src0_row = src0->data + eid * nb02;
        const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
        float * restrict dst_row          = (float *) (dst->data + ie1 * nb1);

        const uint32_t tile_size = htp_mm_get_weight_tile_size(src0->type);
        const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src0->type);
        const uint32_t n_k_tiles_w = ne00 / 32;
        const uint32_t n_k_tiles_a = ne10 / 32;
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
        }

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;

            int valid_rows = (int)ne01 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            if (push_ct < ct_end) {
                dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                push_ct++;
            }
        }
    }
}

static void hvx_mv_id_nx(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = (struct htp_mm_context *) data;
    struct htp_ops_context * octx = mmctx->octx;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights      = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const struct htp_tensor * restrict ids  = octx->src[n_weights + 1];


    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    const uint32_t n_aids = ids->ne[0];
    const uint32_t n_ids  = src0->ne[2];

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        const int32_t eid = *(const int32_t *) ((const uint8_t *) ids->data + ie1 * ids->nb[0]);
        if (eid < 0) continue;
        assert(eid < (int32_t) n_ids);

        for (uint32_t p = 0; p < n_weights; ++p) {
            const struct htp_tensor * restrict src_w = octx->src[p];
            const struct htp_tensor * restrict dst   = octx->dsts[p];
            if (!src_w || !dst) continue;
            dma_queue * dma_q = octx->ctx->dma[ith];

            const uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            if (octx->ctx->mdev.count > 1) {
                const bool can_split = htp_tensor_can_row_partition(dst, sizeof(float));
                const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
                start_row = range.start;
                end_row   = range.start + range.count;
            }

            const uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            const uint32_t src0_start_row = start_row + src0_nrows_per_thread * ith;
            const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, end_row);
            if (src0_start_row >= src0_end_row) continue;

            const dma_addr_t src0_row = src_w->data + eid * src_w->nb[2];
            const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
            float * restrict dst_row = (float *) (dst->data + ie1 * dst->nb[1]);

            const uint32_t tile_size = htp_mm_get_weight_tile_size(src_w->type);
            const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src_w->type);
            const uint32_t n_k_tiles_w = src_w->ne[0] / 32;
            const uint32_t n_k_tiles_a = act->ne[0] / 32;
            const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;

            const uint32_t ct_start = src0_start_row / 32;
            const uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t push_ct = ct_start;
            for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
                dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
            }

            for (uint32_t ct = ct_start; ct < ct_end; ct++) {
                const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;

                int valid_rows = (int)src_w->ne[1] - (int)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                if (push_ct < ct_end) {
                    dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),
                                   aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                    push_ct++;
                }
            }
        }
    }
}

static void hvx_mm_id_nx(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = (struct htp_mm_context *) data;
    struct htp_ops_context * octx = mmctx->octx;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights      = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const struct htp_tensor * restrict ids  = octx->src[n_weights + 1];


    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    const uint32_t n_as = src0->ne[2];

    const uint32_t * matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows = mmctx->matrix_rows;

    const size_t src1_stride = mmctx->vtcm_src1_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;

        for (uint32_t p = 0; p < n_weights; ++p) {
            const struct htp_tensor * restrict src_w = octx->src[p];
            const struct htp_tensor * restrict dst   = octx->dsts[p];
            if (!src_w || !dst) continue;
            dma_queue * dma_q = octx->ctx->dma[ith];

            const uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            if (octx->ctx->mdev.count > 1) {
                const bool can_split = htp_tensor_can_row_partition(dst, sizeof(float));
                const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
                start_row = range.start;
                end_row   = range.start + range.count;
            }

            const uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            const uint32_t src0_start_row = start_row + src0_nrows_per_thread * ith;
            const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, end_row);
            if (src0_start_row >= src0_end_row) continue;

            const dma_addr_t src0_row = src_w->data + cur_a * src_w->nb[2];

            const uint32_t tile_size = htp_mm_get_weight_tile_size(src_w->type);
            const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src_w->type);
            const uint32_t n_k_tiles_w = src_w->ne[0] / 32;
            const uint32_t n_k_tiles_a = act->ne[0] / 32;
            const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;

            const uint32_t ct_start = src0_start_row / 32;
            const uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t push_ct = ct_start;
            for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
                dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
            }

            for (uint32_t ct = ct_start; ct < ct_end; ct++) {
                const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;

                int valid_rows = (int)src_w->ne[1] - (int)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                for (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                    const int rm1 = row_mapping.i1;
                    const int rm2 = row_mapping.i2;

                    const uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (ir1 + rm2 * act->ne[1]) * src1_stride);
                    float * restrict dst_row = (float *) (dst->data + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                if (push_ct < ct_end) {
                    dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),
                                   aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                    push_ct++;
                }
            }
        }
    }
}

static int hvx_mm_init_vec_dot(struct htp_mm_context * mmctx, enum htp_data_type type) {
    switch (type) {
        case HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            return 0;
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q4_K:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            return 0;
        case HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            return 0;
        case HTP_TYPE_Q5_K:
            mmctx->type         = "q5_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
            return 0;
        case HTP_TYPE_Q6_K:
            mmctx->type         = "q6_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
            return 0;
        case HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            return 0;
        case HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            return 0;
        default:
            return -1;
    }
}

static int hvx_mm_matmul(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = src1;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = ne01;
    const uint32_t src1_nrows = ne11 * ne12 * ne13;
    mmctx->act_nrows = src1_nrows;

    uint32_t src0_row_start = 0;
    uint32_t src0_row_end   = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_can_row_partition(dst, sizeof(float));
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        src0_row_start = range.start;
        src0_row_end   = range.start + range.count;
    }

    if (src0_row_start >= src0_row_end) {
        return HTP_STATUS_OK;
    }

    const uint32_t nrows = src0_row_end - src0_row_start;
    mmctx->src0_row_start = src0_row_start;
    mmctx->src0_row_end   = src0_row_end;

    bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                        src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                        src0->type == HTP_TYPE_MXFP4 || src0->type == HTP_TYPE_Q6_K ||
                        src0->type == HTP_TYPE_Q4_K || src0->type == HTP_TYPE_Q5_K);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
    if (is_repacked) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } else {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;
    size_t       src1_row_size = nb11;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);
    size_t       src1_row_size_padded;

    worker_callback_t quant_task_func;
    worker_callback_t matmul_job_func;
    uint32_t n_quant_tasks = 1;
    const bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    if (is_batched) {
        if (is_repacked) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_4d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:
                case HTP_TYPE_Q4_K:   matmul_job_func = hvx_mm_4d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_4d_repacked_q8_0;   break;
                case HTP_TYPE_Q6_K:   matmul_job_func = hvx_mm_4d_repacked_q6_k;   break;
                case HTP_TYPE_Q5_K:   matmul_job_func = hvx_mm_4d_repacked_q5_k;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_4d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_4d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            matmul_job_func = hvx_mm_4d;
        }
    } else if (src1_nrows > 1) {
        if (is_repacked) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:
                case HTP_TYPE_Q4_K:   matmul_job_func = hvx_mm_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_2d_repacked_q8_0;   break;
                case HTP_TYPE_Q6_K:   matmul_job_func = hvx_mm_2d_repacked_q6_k;   break;
                case HTP_TYPE_Q5_K:   matmul_job_func = hvx_mm_2d_repacked_q5_k;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            matmul_job_func = hvx_mm_2d;
        }
    } else {
        if (is_repacked) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mv_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:
                case HTP_TYPE_Q4_K:   matmul_job_func = hvx_mv_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mv_2d_repacked_q8_0;   break;
                case HTP_TYPE_Q5_K:   matmul_job_func = hvx_mv_2d_repacked_q5_k;   break;
                case HTP_TYPE_Q6_K:   matmul_job_func = hvx_mv_2d_repacked_q6_k;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mv_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mv_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            matmul_job_func = hvx_mv_2d;
        }
    }

    bool need_quant = true;

    switch (kparams->kernel_type) {
        case HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (src1->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (src1->type == HTP_TYPE_F32);
            mmctx->type            = (src1->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = vec_dot_f16_f16_aa_2x2;
            src1_row_size          = hex_round_up(ne10 * 2, 128);
            break;

        case HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = NULL;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = vec_dot_f32_f32_aa_2x2;
            src1_row_size          = hex_round_up(ne10 * 4, 128);
            break;

        case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
        case HTP_MM_KERNEL_HVX_QUANT_ROW:
        default:
            if (hvx_mm_init_vec_dot(mmctx, src0->type) != 0) {
                return HTP_STATUS_NO_SUPPORT;
            }

            const uint32_t qk = QK_Q8_0_TILED;
            const uint32_t nb = (ne10 + qk - 1) / qk;
            const uint32_t total_nb = src1_nrows * nb;

            if (src1_nrows < octx->n_threads && !is_batched) {
                n_quant_tasks = MIN(total_nb, octx->n_threads);
                quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
                for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
                    uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_first[ith] = ib_first;
                    mmctx->quant_ib_last[ith]  = ib_last;
                    mmctx->quant_r[ith]        = ib_first / nb;
                    mmctx->quant_c[ith]        = ib_first % nb;
                }
            } else {
                n_quant_tasks = MIN(src1_nrows, octx->n_threads);
                quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
            }
            src1_row_size = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
            break;
    }

    const uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < src1_nrows)
                           ? (uint32_t) kparams->m_chunk : src1_nrows;
    const uint32_t m_layout_rows = m_chunk;

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_rows, octx->n_threads,
                                 dst_row_size, src0_row_size, src1_row_size, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
    } else {
        mmctx->vtcm_src1_size_per_thread = fastdiv(L.src1_bytes, &octx->n_threads_div);
    }

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    const size_t vtcm_size = L.total_bytes;

    FARF(HIGH, "matmul-%s : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.src1_bytes, L.dst_bytes, vtcm_size);

    FARF(HIGH, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, src1->data, dst->data);

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src1     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_src2     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src2);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_src1_stride = src1_row_size;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
    } else if (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * sizeof(float), 128);
    } else {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    if (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < src1_nrows) {
        for (uint32_t m_start = 0; m_start < src1_nrows; m_start += m_chunk) {
            const uint32_t cur_m_rows = MIN(src1_nrows - m_start, m_chunk);
            mmctx->cur_m_start = m_start;
            mmctx->cur_m_rows  = cur_m_rows;

            if (need_quant) {
                hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_start, cur_m_rows);

                const uint32_t qk = QK_Q8_0_TILED;
                const uint32_t nb = (ne10 + qk - 1) / qk;
                const uint32_t total_nb = cur_m_rows * nb;
                uint32_t quant_tasks;
                work_queue_func_t q_func;
                if (cur_m_rows < octx->n_threads && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW)) {
                    quant_tasks = MIN(total_nb, octx->n_threads);
                    q_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
                    for (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_first = (total_nb * ith) / quant_tasks;
                        uint32_t ib_last  = (total_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_first[ith] = ib_first;
                        mmctx->quant_ib_last[ith]  = ib_last;
                        mmctx->quant_r[ith]        = ib_first / nb;
                        mmctx->quant_c[ith]        = ib_first % nb;
                    }
                } else {
                    quant_tasks = MIN(cur_m_rows, octx->n_threads);
                    q_func = quant_task_func;
                    mmctx->n_quant_rows_per_thread = (cur_m_rows + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_queue_run(octx->ctx->work_queue, q_func, mmctx, quant_tasks);
            } else {
                hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_src1, mmctx->vtcm_src1_stride, m_start, cur_m_rows);
            }

            work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
        }
    } else {
        mmctx->cur_m_start = 0;
        mmctx->cur_m_rows  = src1_nrows;

        if (need_quant) {
            hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, src1_nrows);
            mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
        } else {
            hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_src1, mmctx->vtcm_src1_stride, 0, src1_nrows);
        }

        work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
    }

    return HTP_STATUS_OK;
}

static void hvx_mm_nx_2d(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights = kparams->n_weights;

    const struct htp_tensor * restrict act = octx->src[n_weights];
    const uint32_t src1_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data     = mmctx->vtcm_src1;

    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];


    for (uint32_t widx = 0; widx < n_weights; widx++) {
        const struct htp_tensor * restrict src_w = octx->src[widx];
        const struct htp_tensor * restrict dst   = octx->dsts[widx];
        if (!src_w || !dst) continue;
        dma_queue * dma_q = octx->ctx->dma[ith];

        const uint32_t ne00 = src_w->ne[0];
        const uint32_t ne01 = src_w->ne[1];
        uint32_t start_row = 0;
        uint32_t end_row   = ne01;
        if (octx->ctx->mdev.count > 1) {
            const bool can_split = htp_tensor_can_row_partition(dst, sizeof(float));
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            start_row = range.start;
            end_row   = range.start + range.count;
        }

        const uint32_t nrows = end_row - start_row;
        uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
        src0_nrows_per_thread += (src0_nrows_per_thread & 1);

        const uint32_t src0_start_row  = start_row + src0_nrows_per_thread * ith;
        const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, end_row);
        const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);
        if (src0_start_row >= src0_end_row) continue;

        const size_t dst_row_size  = dst->nb[1];
        const size_t src0_row_size = src_w->nb[1];
        const size_t src0_stride   = hex_round_up(src0_row_size, 128);

        const dma_addr_t src0_row = src_w->data;

        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const int is0 = (ir0 - src0_start_row);
            if (is0 >= (int)n_prefetch) break;
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }

        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            uint32_t ir1 = 0;
            for (; ir1 + 1 < src1_nrows; ir1 += 2) {
                const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);
                const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);
                float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));
                float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));
                mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);
            }
            for (; ir1 < src1_nrows; ++ir1) {
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
                float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));
                mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, src1_col);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            const int pr0 = (ir0 + n_prefetch);
            const int is0 = (pr0 - src0_start_row) & prefetch_mask;
            if (pr0 < src0_end_row_x2) {
                dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                               src0_stride, src0_row_size, src0_row_size, 2);
            }
        }

        if (src0_end_row != src0_end_row_x2) {
            uint32_t ir0 = src0_end_row_x2;
            const int is0 = (ir0 - src0_start_row) & prefetch_mask;
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 1);
            const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
                float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));
                mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, src1_col);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }
    }
}

#define DEQUANTIZE_WORKER_LOOP_IMPL(SUFFIX)                                                     \
static void dequantize_tiled_worker_loop_##SUFFIX(unsigned int n, unsigned int i, void *data) { \
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;                         \
    struct htp_thread_trace * tr = &state->traces[i];                                           \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);                                  \
    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {      \
        int start = task_id * state->n_tiles_per_task;                                          \
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);              \
        dequantize_tiled_weight_to_fp16_task_##SUFFIX(state, start, end);                       \
    }                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);                                   \
}

DEQUANTIZE_WORKER_LOOP_IMPL(q4_0)
DEQUANTIZE_WORKER_LOOP_IMPL(q4_1)
DEQUANTIZE_WORKER_LOOP_IMPL(iq4_nl)
DEQUANTIZE_WORKER_LOOP_IMPL(mxfp4)
DEQUANTIZE_WORKER_LOOP_IMPL(q8_0)
DEQUANTIZE_WORKER_LOOP_IMPL(q6_k)
DEQUANTIZE_WORKER_LOOP_IMPL(q5_k)

static void convert_f16_worker_loop(unsigned int n, unsigned int i, void *data) {
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;
    struct htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {
        int start = task_id * state->n_tiles_per_task;
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        convert_f16_weight_to_fp16_tiles_task(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

static void quantize_f32_worker_loop(unsigned int n, unsigned int i, void *data) {
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;

    struct htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, i);

    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {
        int start = task_id * state->n_tiles_per_task;
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        quantize_f32_weight_to_fp16_tiles_task(state, start, end);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, i);
}

static void transfer_output_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) data;

    struct htp_thread_trace * tr = &st->traces[i];

    int start_chunk_idx = i * st->n_chunks_per_task;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);

    for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
        int    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        float        *dst      = st->dst      + chunk_idx * st->dst_stride;
        const float  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : NULL;
        transfer_output_chunk_fp16_to_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_size, st->n_cols, st->dst_stride, st->src2_stride, st->dst_cols);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);
}

typedef struct {
    struct htp_context            * ctx;
    struct htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    float                         * vtcm_f32_act;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    size_t                          vtcm_f32_act_bytes_per_thread;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
    bool                             is_hmx_int;
    float *                         vtcm_act_scales;
} activation_transfer_task_state_t;

typedef struct {
    struct htp_context            * ctx;
    struct htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    float                         * vtcm_f32_act;
    uint32_t                        n_rows;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_col_chunks;
    struct fastdiv_values           n_threads_div;
    size_t                          vtcm_f32_act_bytes;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
    bool                             is_hmx_int;
    float *                         vtcm_act_scales;
} activation_transfer_col_chunk_state_t;

static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_first,
        uint32_t c_len,
        float *thread_f32_act,
        struct htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    const uint32_t R = dma_step_rows;
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    const uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    // Push step 0
    if (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_push(dma_q, dma_make_data(thread_f32_act, act_dma_addr + (size_t) c_first * sizeof(float)),
                       c_len * sizeof(float), k_stride * sizeof(float), k_chunk_valid * sizeof(float), nrows_to_fetch);
    }
    // Push step 1
    if (n_steps > 1) {
        uint32_t next_r = R * 1;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            float *next_buf = thread_f32_act + 1 * R * c_len;
            dma_queue_push(dma_q, dma_make_data(next_buf, act_dma_addr + ((size_t) next_r * k_stride + c_first) * sizeof(float)),
                           c_len * sizeof(float), k_stride * sizeof(float), k_chunk_valid * sizeof(float), nrows_to_fetch);
        }
    }
    for (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        float *curr_buf = thread_f32_act;

        if (r < n_rows) {
            curr_buf = (float *) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        for (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t row_idx = r + (p << 1);
            float *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((row_idx + 0) < n_rows);
            bool r1_valid = ((row_idx + 1) < n_rows);

            transfer_activation_row_pair_fp32_to_fp16_col_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, row_idx, k_block, c_first, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_push(dma_q, dma_make_data(curr_buf, act_dma_addr + ((size_t) next_r * k_stride + c_first) * sizeof(float)),
                           c_len * sizeof(float), k_stride * sizeof(float), k_chunk_valid * sizeof(float), nrows_to_fetch);
        }
    }
}

static void transfer_activation_chunk_col_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_col_chunk_state_t *st = (activation_transfer_col_chunk_state_t *) data;
    struct htp_thread_trace * tr = &st->traces[i];

    uint32_t n_blocks = st->k_block / 32;
    uint32_t b_first = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last  = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first = b_first * 32;
    uint32_t c_last = b_last * 32;
    uint32_t c_len = c_last - c_first;

    if (c_len == 0) {
        return;
    }

    uint32_t k_chunk_valid = 0;
    if (st->k_valid > c_first) {
        k_chunk_valid = hex_smin(st->k_valid, c_last) - c_first;
    }

    __fp16 *dst = st->dst;

    size_t thread_scratch_bytes = hex_align_down(fastdiv(st->vtcm_f32_act_bytes, &st->n_threads_div), 128);
    float *thread_f32_act = (float *)((char *)st->vtcm_f32_act + i * thread_scratch_bytes);

    transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_rows, st->k_block, st->k_stride, k_chunk_valid,
        c_first, c_len, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
    );
}

static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_valid,
        float *thread_f32_act,
        struct htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift,
        bool is_hmx_int,
        float *vtcm_act_scales) {

    const uint32_t R = dma_step_rows;
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    const uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    // Push step 0
    if (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_push(dma_q, dma_make_data(thread_f32_act, act_dma_addr),
                       k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
    }
    // Push step 1 (if valid)
    if (n_steps > 1) {
        uint32_t next_r = R * 1;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            float *next_buf = thread_f32_act + 1 * R * k_block;
            dma_queue_push(dma_q, dma_make_data(next_buf, act_dma_addr + (size_t) next_r * k_stride * sizeof(float)),
                           k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
        }
    }
    for (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        float *curr_buf = thread_f32_act;

        if (r < n_rows) {
            curr_buf = (float *) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        for (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t row_idx = r + (p << 1);
            float *pair_buf = curr_buf + (p << 1) * k_block;
            bool r0_valid = ((row_idx + 0) < n_rows);
            bool r1_valid = ((row_idx + 1) < n_rows);

            if (is_hmx_int) {
                quantize_act_row_pair_fp32_to_ub((uint8_t *) vtcm_dst, vtcm_act_scales, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            } else {
                act_tile_row_pair_store(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_push(dma_q, dma_make_data(curr_buf, act_dma_addr + (size_t) next_r * k_stride * sizeof(float)),
                           k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
        }
    }
}

static void transfer_activation_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

    struct htp_thread_trace * tr = &st->traces[i];

    for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
        int    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_block;
        const dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * sizeof(float);

        float *thread_f32_act = (float *)((char *)st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread);
        const size_t n_k_tiles = st->k_block / HTP_MM_HMX_TILE_N_COLS;
        float * act_scales = st->vtcm_act_scales ? st->vtcm_act_scales + chunk_idx * n_k_tiles : NULL;
        transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift,
            st->is_hmx_int, act_scales
        );
    }
}

typedef struct {
    struct htp_thread_trace       * traces;
    const struct mmid_row_mapping * matrix_rows;
    __fp16                        * dst;
    const float                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    struct fastdiv_values           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        start_row;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

typedef struct {
    struct htp_thread_trace       * traces;
    const struct mmid_row_mapping * matrix_rows;
    const __fp16                  * vtcm_src;
    float                         * dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        n_cols;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        start_row;
    uint32_t                        cne1;
} output_transfer_scattered_task_state_t;

static void transfer_activation_chunk_gathered_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_gathered_task_state_t *st = data;
    struct htp_thread_trace * tr = &st->traces[i];
    int chunk_idx      = i;
    int chunk_size     = st->n_chunks_per_task;
    int vtcm_start_row = chunk_idx * chunk_size;
    int start_row      = st->start_row + vtcm_start_row;
    int n_rows         = hex_smin(st->cne1 - start_row, chunk_size);
    if (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

static void transfer_activation_chunk_gathered_worker_flat_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_gathered_task_state_t *st = data;
    struct htp_thread_trace * tr = &st->traces[i];
    int chunk_idx = i;
    int chunk_size = st->n_chunks_per_task;
    int vtcm_start_row = chunk_idx * chunk_size;
    int start_row = st->start_row + vtcm_start_row;
    int n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    if (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered_flat(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

static void transfer_output_chunk_scattered_worker_fn(unsigned int n, unsigned int i, void *data) {
    output_transfer_scattered_task_state_t *st = data;
    struct htp_thread_trace * tr = &st->traces[i];
    int chunk_idx = i;
    int chunk_size = st->n_chunks_per_task;
    int vtcm_start_row = chunk_idx * chunk_size;
    int start_row = st->start_row + vtcm_start_row;
    int n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    if (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_to_fp32_scattered(
            st->dst, st->vtcm_src, start_row, vtcm_start_row, n_rows, st->n_cols,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

static void dequantize_tiled_weight_chunk_to_fp16_tiles(
        struct htp_context *ctx, __fp16 *vtcm_dst,
        const void *weight_src_ddr,
        int n_cols, int k_block,
        size_t row_stride, int weight_type,
        int n_k_tiles, struct fastdiv_values n_k_tiles_div,
        worker_callback_t dequant_worker_fn, int n_threads) {

    assert(n_cols  % HTP_MM_HMX_TILE_N_COLS == 0);
    assert(k_block % HTP_MM_HMX_TILE_N_COLS == 0);

    size_t n_col_tiles = n_cols / HTP_MM_HMX_TILE_N_COLS;
    size_t n_tot_tiles = n_col_tiles * n_k_tiles;

    size_t n_tiles_per_task = (n_threads == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_threads);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_per_task - 1) / n_tiles_per_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_per_task = n_tiles_per_task;
    state.dst              = vtcm_dst;
    state.src              = (const uint8_t *)weight_src_ddr;
    state.n_cols           = n_cols;
    state.k_block          = k_block;
    state.row_stride       = row_stride;
    state.weight_type      = weight_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    state.tile_size = htp_mm_get_weight_tile_size(weight_type);
    state.aligned_tile_size = htp_mm_get_weight_aligned_tile_size(weight_type);

    if (state.n_tasks == 1 || n_threads == 1) {
        dequant_worker_fn(1, 0, &state);
    } else {
        int n_tasks = hex_smin((int) state.n_tasks, n_threads);
        worker_pool_run_func(ctx->worker_pool, dequant_worker_fn, &state, n_tasks);
    }
}

typedef struct {
    struct htp_context      * ctx;
    struct htp_thread_trace * traces;
    float                   * dst;
    const __fp16            * vtcm_src;
    const float             * src2;
    uint32_t                  n_rows;
    uint32_t                  n_cols;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_cols;
    struct fastdiv_values     n_threads_div;
} output_transfer_col_chunk_state_t;

static void transfer_output_chunk_col_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    (void) n;
    output_transfer_col_chunk_state_t *st = (output_transfer_col_chunk_state_t *) data;
    struct htp_thread_trace * tr = &st->traces[i];

    uint32_t n_blocks = st->n_cols / 32;
    uint32_t b_first  = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last   = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first  = b_first * 32;
    uint32_t c_last   = b_last * 32;
    uint32_t c_len    = c_last - c_first;

    if (c_len == 0) return;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);

    const __fp16 *vtcm_src = st->vtcm_src + b_first * HTP_MM_HMX_TILE_N_ELMS;
    const float *src2      = st->src2 ? (st->src2 + c_first) : NULL;
    float *dst             = st->dst + c_first;

    int chunk_dst_cols = (int)st->dst_cols - (int)c_first;
    if (chunk_dst_cols > 0) {
        transfer_output_chunk_fp16_to_fp32_col_chunk(
            dst, src2, vtcm_src, 0, st->n_rows, c_len, st->n_cols,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_cols
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

static void transfer_output_chunk_threaded(struct htp_context *ctx, float *dst, const float *src2, const __fp16 *vtcm_src,
                                              int n_rows, int n_cols, int dst_stride, uint32_t src2_stride, int dst_cols, int n_threads) {
    assert(n_cols % HTP_MM_HMX_TILE_N_COLS == 0);

    if (n_rows <= 0) return;

    uint32_t n_blocks = (uint32_t)n_cols / 32;
    if (n_threads > 1 && n_blocks >= (uint32_t)n_threads) {
        struct fastdiv_values n_threads_div = (n_threads == (int)ctx->n_threads) ? ctx->n_threads_div : init_fastdiv_values(n_threads);
        output_transfer_col_chunk_state_t col_state;
        col_state.dst = dst;
        col_state.src2 = src2;
        col_state.vtcm_src = vtcm_src;
        col_state.n_rows = (uint32_t)n_rows;
        col_state.n_cols = (uint32_t)n_cols;
        col_state.dst_stride = (uint32_t)dst_stride;
        col_state.src2_stride = src2_stride;
        col_state.dst_cols = (uint32_t)dst_cols;
        col_state.n_threads_div = n_threads_div;
        col_state.traces = ctx->trace;
        col_state.ctx = ctx;

        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_col_chunk_worker_fn, &col_state, n_threads);
        return;
    }

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : hmx_ceil_div(n_rows, n_threads);
    n_chunks_per_task        = hex_align_up(n_chunks_per_task, 2);

    int actual_threads = hmx_ceil_div(n_rows, n_chunks_per_task);

    output_transfer_task_state_t state;
    state.n_tasks           = actual_threads;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_per_task = n_chunks_per_task;
    state.dst               = dst;
    state.src2              = src2;
    state.vtcm_src          = vtcm_src;
    state.n_cols            = n_cols;
    state.dst_stride        = dst_stride;
    state.src2_stride       = src2_stride;
    state.dst_cols          = dst_cols;
    state.traces            = ctx->trace;

    if (actual_threads <= 1) {
        transfer_output_chunk_worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_worker_fn, &state, actual_threads);
    }
}

// HTP_OPFLAGS_MM_OUT_F16: the fp16 tiles are stored as fp16 rows, without widening. Tile vector r1 holds row 2*r1 in its even halfwords and row 2*r1+1 in its odd halfwords: the deal puts row r in the low 64 bytes and row r+1 in the high 64 bytes.
// The fp32 path widens these same values (x1.0, no src2), so the stored values are identical. dst_stride_bytes is the DDR row stride.
static void transfer_output_chunk_f16_rows(uint8_t * dst, size_t dst_stride_bytes, const __fp16 * vtcm_src, uint32_t start_row,
                                           uint32_t n_rows, uint32_t n_cols, uint32_t dst_cols) {
    const size_t tile_row_stride = (n_cols / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
    const size_t limit_c         = hex_smin(n_cols, dst_cols);
    for (size_t r = 0; r < n_rows; r += 2) {
        const size_t r_idx0     = start_row + r;
        const size_t r0         = r_idx0 / HTP_MM_HMX_TILE_N_ROWS;
        const size_t r1         = (r_idx0 % HTP_MM_HMX_TILE_N_ROWS) / 2;
        const __fp16 * row_base = vtcm_src + r0 * tile_row_stride;
        __fp16 * out0 = (__fp16 *) (dst + r * dst_stride_bytes);
        __fp16 * out1 = (__fp16 *) (dst + (r + 1) * dst_stride_bytes);
        const HVX_VectorPred q_lo64 = Q6_Q_vsetq_R(64);
        for (size_t c = 0; c < limit_c; c += HTP_MM_HMX_TILE_N_COLS) {
            const size_t c0    = c / HTP_MM_HMX_TILE_N_COLS;
            const HVX_Vector v = ((const HVX_Vector *) (row_base + c0 * HTP_MM_HMX_TILE_N_ELMS))[r1];
            const HVX_Vector d = Q6_Vh_vdeal_Vh(v);
            // Two adjacent column tiles starting on a 128-byte boundary fill whole cache lines of the row (64 columns = 128 B): a 64-byte masked store would make the cache fetch the rest of the line first (read for ownership) and made the fp16 store slower than the fp32 one (MUL_MAT +25 us/call measured on the board).
            if (((uintptr_t) (out0 + c) & 127) == 0 && c + 2 * HTP_MM_HMX_TILE_N_COLS <= limit_c) {
                const HVX_Vector v1 = ((const HVX_Vector *) (row_base + (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS))[r1];
                const HVX_Vector d1 = Q6_Vh_vdeal_Vh(v1);
                *(HVX_Vector *) (out0 + c) = Q6_V_vmux_QVV(q_lo64, d, Q6_V_vror_VR(d1, 64));
                if (r + 1 < n_rows) {
                    const HVX_Vector w1 = Q6_V_vmux_QVV(q_lo64, Q6_V_vror_VR(d, 64), d1);
                    if (((uintptr_t) (out1 + c) & 127) == 0) {
                        *(HVX_Vector *) (out1 + c) = w1;
                    } else {
                        hvx_vec_store_u(out1 + c, 128, w1);
                    }
                }
                c += HTP_MM_HMX_TILE_N_COLS;
                continue;
            }
            const size_t valid = hex_smin(limit_c - c, HTP_MM_HMX_TILE_N_COLS) * sizeof(__fp16);
            hvx_vec_store_u(out0 + c, valid, d);
            if (r + 1 < n_rows) {
                hvx_vec_store_u(out1 + c, valid, Q6_V_vror_VR(d, 64));
            }
        }
    }
}

typedef struct {
    uint8_t *      dst;
    size_t         dst_stride_bytes;
    const __fp16 * vtcm_src;
    uint32_t       n_rows;
    uint32_t       rows_per_task;
    uint32_t       n_cols;
    uint32_t       dst_cols;
} output_transfer_f16_task_state_t;

static void transfer_output_f16_worker_fn(unsigned int n, unsigned int i, void * data) {
    (void) n;
    output_transfer_f16_task_state_t * st = (output_transfer_f16_task_state_t *) data;
    const uint32_t r0 = i * st->rows_per_task;
    if (r0 >= st->n_rows) return;
    const uint32_t rows = hex_smin(st->n_rows - r0, st->rows_per_task);
    transfer_output_chunk_f16_rows(st->dst + (size_t) r0 * st->dst_stride_bytes, st->dst_stride_bytes, st->vtcm_src, r0, rows,
                                   st->n_cols, st->dst_cols);
}

// Row-partitioned fp16 store (rows are independent; a row pair always stays inside one tile row).
static void transfer_output_chunk_f16_threaded(struct htp_context * ctx, __fp16 * dst, size_t dst_stride_bytes, const __fp16 * vtcm_src,
                                               int n_rows, int n_cols, int dst_cols, int n_threads) {
    if (n_rows <= 0) return;
    output_transfer_f16_task_state_t st;
    st.dst              = (uint8_t *) dst;
    st.dst_stride_bytes = dst_stride_bytes;
    st.vtcm_src         = vtcm_src;
    st.n_rows           = (uint32_t) n_rows;
    st.n_cols           = (uint32_t) n_cols;
    st.dst_cols         = (uint32_t) dst_cols;
    st.rows_per_task    = hex_align_up(hmx_ceil_div((uint32_t) n_rows, (uint32_t) (n_threads > 0 ? n_threads : 1)), 2);
    const uint32_t n_tasks = hmx_ceil_div(st.n_rows, st.rows_per_task);
    if (n_tasks <= 1) {
        transfer_output_f16_worker_fn(1, 0, &st);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_output_f16_worker_fn, &st, n_tasks);
    }
}

struct activation_transfer_params {
    struct htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    int                           n_rows;
    int                           k_block;
    int                           k_stride;
    int                           n_threads;
    const struct fastdiv_values * act_threads_div;
    const struct fastdiv_values * k_div;
    int                           k_valid;
    float *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_bytes;
    bool                          is_hmx_int;
    float *                       vtcm_act_scales;
};

static void transfer_activation_chunk_threaded(const struct activation_transfer_params * params) {
    struct htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    const dma_addr_t              act_dma_addr       = params->act_dma_addr;
    int                           n_rows             = params->n_rows;
    int                           k_block            = params->k_block;
    int                           k_stride           = params->k_stride;
    int                           n_threads          = params->n_threads;
    const struct fastdiv_values * act_threads_div    = params->act_threads_div;
    const struct fastdiv_values * k_div              = params->k_div;
    int                           k_valid            = params->k_valid;
    float *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_bytes = params->vtcm_f32_act_bytes;
    const bool                    is_hmx_int          = params->is_hmx_int;

    if (n_rows <= 0) {
        return;
    }

    const size_t n_tasks = (n_rows + 31) >> 5;
    if (!is_hmx_int && n_threads > 1 && k_block > 32 && n_tasks < (size_t) n_threads) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_step_rows = 2;
        uint32_t dma_step_rows_shift = 1;
        if (vtcm_f32_act_bytes > 0) {
            size_t thread_scratch_bytes = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_threads_div), 128);
            size_t thread_scratch_elements = thread_scratch_bytes / sizeof(float);
            size_t dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
            if (dma_step_rows_max >= 4) {
                dma_step_rows = 4;
                dma_step_rows_shift = 2;
            }
        }

        activation_transfer_col_chunk_state_t col_state;
        col_state.dst = dst;
        col_state.act_dma_addr = act_dma_addr;
        col_state.n_rows = n_rows;
        col_state.k_block = k_block;
        col_state.k_stride = k_stride;
        col_state.k_valid = k_valid;
        col_state.n_col_chunks = n_threads;
        col_state.n_threads_div = *act_threads_div;
        col_state.vtcm_f32_act = vtcm_f32_act;
        col_state.vtcm_f32_act_bytes = vtcm_f32_act_bytes;
        col_state.traces = ctx->trace;
        col_state.ctx = ctx;
        col_state.dma_step_rows = dma_step_rows;
        col_state.dma_step_rows_shift = dma_step_rows_shift;

        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_col_chunk_worker_fn, &col_state, n_threads);
        return;
    }

    assert(k_block % HTP_MM_HMX_TILE_N_COLS == 0 && k_stride % HTP_MM_HMX_TILE_N_COLS == 0);

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks            = (n_threads == 1) ? 1 : hmx_ceil_div(n_tot_chunks, 32);
    state.n_tot_chunks       = n_tot_chunks;
    state.n_chunks_per_task  = n_chunks_per_task;
    state.dst                = dst;
    state.act_dma_addr       = act_dma_addr;
    state.k_block            = k_block;
    state.k_stride           = k_stride;
    state.k_valid            = k_valid;
    state.traces             = ctx->trace;
    state.ctx                = ctx;
    state.vtcm_f32_act       = vtcm_f32_act;
    state.is_hmx_int         = params->is_hmx_int;
    state.vtcm_act_scales    = params->vtcm_act_scales;

    state.vtcm_f32_act_bytes_per_thread = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_threads_div), 128);

    uint32_t dma_step_rows = 2;
    uint32_t dma_step_rows_shift = 1;
    if (state.vtcm_f32_act_bytes_per_thread > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_bytes_per_thread / sizeof(float);
        size_t dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
        if (dma_step_rows_max >= 4) {
            dma_step_rows = 4;
            dma_step_rows_shift = 2;
        }
    }
    state.dma_step_rows      = dma_step_rows;
    state.dma_step_rows_shift = dma_step_rows_shift;

    int active_threads = hex_smin(n_threads, (int)state.n_tasks);
    if (state.n_tasks == 1 || n_threads == 1) {
        transfer_activation_chunk_worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_worker_fn, &state, active_threads);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

typedef struct {
    int            kernel_type;
    __fp16 *       output;
    const void *   activation;
    const void *   weight;
    const __fp16 * scales;
    void *         bias;
    uint32_t       cvt_reg;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_job_t;

static int g_a8w8_pe_op = 1;   // plain MUL_MAT consumes pre-expanded blobs (bit 29 of A8W8_DBG clears it)
static dma_addr_t g_a8w8_pe_weight;   // plain MUL_MAT: address of the pre-expanded blob of src0 (0: none), set by hmx_mm_op_matmul
static int g_a8w8_dbg;
static inline float a8w8_clip_from_params(const struct htp_ops_context * octx) {
    float c; memcpy(&c, &octx->op_params[2], sizeof(c));
    return (c > 0.1f && c <= 1.0f) ? c : 1.0f;
}   // A8W8 debug / probe bits (op_params[3]): 1 = halve weight DMA bytes, 4 = deferred SWIGLU, bits 8..15/16..23 = first-last / second chunk tiles, 24 = one-block activation groups, 25 = skip act quant, 26 = skip weight expand, 27 = skip post, 28 = weight DMA reads 1024 B per tile (int8-sized) (timing probes 25..28: wrong results)

// ---- A8W8 (GGML_HEXAGON_MM_A8W8, HTP_OPFLAGS_MM_A8W8): SWIGLU gate/up with u8 x int8 HMX jobs (htp/hmx-a8w8.h) ----
typedef struct {
    // per row block
    const __fp16 * act16;      // fp16 act tiles of the row block (tile (r0,c0) at r0 * tile_k + c0)
    uint8_t *      act8;       // u8 64-row tiles: tile_k tiles of 2048 B per 64-row block
    float *        rscale;     // [n_rows64] row scales
    uint32_t       tile_k;
    uint32_t       n_blocks64; // 64-row blocks
    uint32_t       n_row_tiles;
    uint32_t       qb0, qnb;   // quantiser block range [qb0, qb0 + qnb); qnb = 0: all n_blocks64 blocks
    // per chunk
    const uint8_t * raw;       // raw Q4_0 tiles of the chunk: column tile ct, k tile kt at (ct * tile_k + kt) * aligned
    uint32_t        aligned;
    uint8_t *       w8;        // int8 tiles: (ct * tile_k + kt) * 1024
    float *         sw;        // [n_col_tiles][32]
    int32_t *       cs;        // [n_col_tiles][32]
    uint32_t        n_col_tiles;
    uint32_t        n_tasks;
    struct htp_thread_trace * traces;
} a8w8_state_t;

static void a8w8_quant_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    const a8w8_state_t * st = (const a8w8_state_t *) data;
    const uint32_t qnb   = st->qnb ? st->qnb : st->n_blocks64;
    const uint32_t total = qnb * 16;                                 // row quads: 16 per 64-row block
    const uint32_t t0 = (total * i) / st->n_tasks, t1 = (total * (i + 1)) / st->n_tasks;
    struct htp_thread_trace * tr = &st->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, i);
    for (uint32_t t = t0; t < t1 && !(g_a8w8_dbg & (1 << 25)); ++t) {
        const uint32_t blk = st->qb0 + t / 16, qd = t % 16;          // quad index inside the block: rows 4*qd .. 4*qd+3
        const uint32_t tile_row = blk * 2 + qd / 8;                  // 32-row tile row
        const uint32_t pi = (qd % 8) * 2;
        if (tile_row >= st->n_row_tiles) {
            // rows past the block: zero activations (+128 bias = 128) and unit scale
            for (uint32_t kt = 0; kt < st->tile_k; ++kt) {
                memset(st->act8 + (size_t) blk * st->tile_k * 2048 + (size_t) kt * 2048 + (qd * 4) * 32, 128, 128);
            }
            for (int r = 0; r < 4; ++r) st->rscale[blk * 64 + qd * 4 + r] = 1.f;
            continue;
        }
        a8w8_quant_quad(st->act16 + (size_t) tile_row * st->tile_k * 1024, st->tile_k, pi,
                        st->act8 + (size_t) blk * st->tile_k * 2048, qd * 4, st->rscale + blk * 64 + qd * 4);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, i);
}

static void a8w8_expand_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    g_a8w8_expand_probe = (g_a8w8_dbg >> 28) & 3;   // DBG bits 28 / 29: constant multipliers / skip tile unpack (timing probes, wrong results)
    const a8w8_state_t * st = (const a8w8_state_t *) data;
    const uint32_t c0 = (st->n_col_tiles * i) / st->n_tasks, c1 = (st->n_col_tiles * (i + 1)) / st->n_tasks;
    struct htp_thread_trace * tr = &st->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    for (uint32_t ct = c0; ct < c1 && !(g_a8w8_dbg & (1 << 26)); ++ct) {
        a8w8_expand_column_tile(st->w8 + (size_t) ct * st->tile_k * 1024, st->raw + (size_t) ct * st->tile_k * st->aligned, st->tile_k,
                                st->aligned, st->sw + ct * 32, st->cs + ct * 32);
        if (g_a8w8_dbg & (1 << 30)) {
            // probe: three phases of this column tile (scan = d scan for amax, setup = scalar scale setup, loop = per-k-tile expand) as W_PREP events (info 100 + phase)
            for (int ph = 0; ph < 3 && tr->count + 2 <= tr->max_events; ++ph) {
                tr->events[tr->count].id = HTP_TRACE_EVT_HVX_W_PREP; tr->events[tr->count].info = 100 + ph;          tr->events[tr->count].cycles = g_a8w8_stamp[ph];     tr->count++;
                tr->events[tr->count].id = HTP_TRACE_EVT_HVX_W_PREP; tr->events[tr->count].info = (100 + ph) | 0x8000; tr->events[tr->count].cycles = g_a8w8_stamp[ph + 1]; tr->count++;
            }
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

// Pre-expanded chunk: the blob sits in st->raw (n_col_tiles * (tile_k * 1024 + 256) B); w8 = the blob itself (job wstride), sw/cs = the 256 B tails.
static void a8w8_pe_unpack_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    const a8w8_state_t * st = (const a8w8_state_t *) data;
    const uint32_t ctb = st->tile_k * 1024 + HTP_A8W8_PE_TAIL;
    for (uint32_t ct = i; ct < st->n_col_tiles; ct += st->n_tasks) {
        const uint8_t * tail = st->raw + (size_t) ct * ctb + (size_t) st->tile_k * 1024;
        *(HVX_Vector *) (st->sw + ct * 32) = *(const HVX_Vector *) tail;
        *(HVX_Vector *) (st->cs + ct * 32) = *(const HVX_Vector *) (tail + 128);
    }
}

typedef struct {
    const uint8_t * act8;      // u8 64-row tiles of the block
    const uint8_t * w8;        // int8 tiles of the chunk
    uint8_t *       planes;    // ring of 4-plane slots (8 KB each)
    const uint32_t * tbl;      // 4 bias/scale tables of 256 B (exponents 24, 16, 8, 0)
    uint32_t        tile_k;
    uint32_t        n_blocks64;
    uint32_t        n_col_tiles;
    uint32_t        ring;
    volatile int *  produced;  // slots produced (VTCM, written by the HMX thread)
    volatile int *  qready;    // 64-row activation blocks quantised so far (VTCM, written by the dispatch thread); NULL: all ready
    uint32_t        wstride;   // bytes between column tiles of w8 (0: tile_k * 1024; pre-expanded blobs carry 256 B of scales after each column tile)
} a8w8_job_t;


typedef struct {
    const uint8_t * planes;    // ring of 8 KB slots
    const float *   sw;        // [n_col_tiles][32]
    const int32_t * cs;        // [n_col_tiles][32]
    const float *   rscale;    // [64 * n_blocks64]
    __fp16 *        out;       // chunk output tiles in the HMX output layout: tile (rt, ct) at (rt * n_col_tiles + ct) * 1024
    uint32_t        n_col_tiles;
    uint32_t        n_blocks64;
    uint32_t        n_row_tiles;
    uint32_t        ring;
    uint32_t        n_tasks;
    struct htp_thread_trace * traces;
    volatile int *  produced;  // NULL: all slots are ready
    // fused SWIGLU epilogue (up chunk only): after a slot's tiles are written, silu(gate) * up of the same tiles goes to the act-tile output
    const __fp16 * epi_g;      // gate chunk tiles (same layout as `out`); NULL: no epilogue
    __fp16 *       epi_out;    // act tiles of the whole output (DDR)
    uint32_t       epi_row0, epi_col0, epi_cols;
    int            epi_barrier;
} a8w8_post_state_t;

// task = (64-row block r, column tile c, row group chunk): 16 groups of 4 rows per slot; slots are post-processed in whole.
static void a8w8_post_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    const a8w8_post_state_t * st = (const a8w8_post_state_t *) data;
    const uint32_t n_slots = st->n_blocks64 * st->n_col_tiles;
    __fp16 scratch[2][1024] __attribute__((aligned(128)));
    float rs_loc[64] __attribute__((aligned(128)));      // row scales of the current 64-row block (stack copy: scalar loads from VTCM are slow)
    struct htp_thread_trace * tr = &st->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, i);
    uint32_t r_loaded = 0xffffffffu;
    // contiguous slot range per thread (adjacent threads would otherwise share the plane ring's cache lines and VTCM banks slot by slot);
    // with `produced` set, each slot is awaited on the HMX thread's counter
    const uint32_t sa = (uint32_t) (((uint64_t) n_slots * i) / st->n_tasks), sb = (uint32_t) (((uint64_t) n_slots * (i + 1)) / st->n_tasks);
    for (uint32_t slot = sa; slot < sb; ++slot) {
        const uint32_t r = slot / st->n_col_tiles, c = slot - r * st->n_col_tiles;
        if (st->produced) { while ((uint32_t) *st->produced <= slot) { } }
        if (r != r_loaded) { memcpy(rs_loc, st->rscale + r * 64, 64 * sizeof(float)); r_loaded = r; }
        const uint8_t * pl = st->planes + (size_t) slot * 8192;
        const HVX_Vector swv = *(const HVX_Vector *) (st->sw + c * 32);
        const HVX_Vector csv = *(const HVX_Vector *) (st->cs + c * 32);
        const uint32_t rt0 = r * 2, rt1 = r * 2 + 1;
        __fp16 * o0 = (rt0 < st->n_row_tiles) ? st->out + ((size_t) rt0 * st->n_col_tiles + c) * 1024 : scratch[0];
        __fp16 * o1 = (rt1 < st->n_row_tiles) ? st->out + ((size_t) rt1 * st->n_col_tiles + c) * 1024 : scratch[1];
        #pragma unroll(2)
        for (uint32_t grp = 0; grp < 16 && !(g_a8w8_dbg & (1 << 27)); ++grp) {
            a8w8_post_group(pl, swv, csv, rs_loc, o0, o1, grp);
        }
        if (st->epi_g) {
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t rt = rt0 + h;
                if (rt >= st->n_row_tiles) { break; }
                const size_t src = ((size_t) rt * st->n_col_tiles + c) * HTP_MM_HMX_TILE_N_ELMS;
                __fp16 * eo = st->epi_out + ((size_t) (st->epi_row0 + rt) * st->epi_cols + st->epi_col0 + c) * HTP_MM_HMX_TILE_N_ELMS;
                if (st->epi_barrier & 2)      { swiglu_tile_pair_sig16v2(eo, st->epi_g + src, st->out + src); }
                else if (st->epi_barrier & 1) { swiglu_tile_pair(eo, st->epi_g + src, st->out + src); }
                else                          { swiglu_tile_pair_nobar(eo, st->epi_g + src, st->out + src); }
            }
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, i);
}

static void a8w8_hmx_job(void * data) {
    a8w8_job_t * b = (a8w8_job_t *) data;
    int seq = 0;
    asm volatile("mxclracc\n");
    for (uint32_t r = 0; r < b->n_blocks64; ++r) {
        if (b->qready) { while ((uint32_t) *b->qready <= r) { } }
        for (uint32_t c = 0; c < b->n_col_tiles; ++c) {
            const uint8_t * a = b->act8 + (size_t) r * b->tile_k * 2048;
            const uint8_t * w = b->w8 + (size_t) c * (b->wstride ? b->wstride : b->tile_k * 1024);
            uint8_t * o = b->planes + (size_t) (seq % b->ring) * 8192;
            asm volatile("mxclracc\n");
            uint32_t left = b->tile_k;
            while (left) {
                const uint32_t n  = left > 32 ? 32 : left;
                const uint32_t ra = ((n - 1) << 11) | 31;
                const uint32_t rw = (uint32_t) (1024 * n - 128);
                asm volatile("{\n activation.ub = mxmem(%0, %1):deep:cm\n weight.b = mxmem(%2, %3)\n}\n" : : "r"(a), "r"(ra), "r"(w), "r"(rw));
                a += 2048 * n; w += 1024 * n; left -= n;
            }
            for (int k = 0; k < 4; ++k) {
                const uint32_t zero = 0;
                asm volatile("bias = mxmem2(%0)\n" : : "r"(b->tbl + k * 64));
                if (k < 3) asm volatile("{ mxmem(%0, %1):after:retain:cm.ub = acc }\n" : : "r"(o + k * 2048), "r"(zero));
                else       asm volatile("{ mxmem(%0, %1):after:cm.ub = acc }\n" : : "r"(o + k * 2048), "r"(zero));
            }
            ++seq;
            *b->produced = seq;
        }
    }
    // Leave both accumulator banks clean for the fp16 jobs that follow (they clear only the current bank and rely on the non-retain store to swap into a clean one):
    // clear this bank, swap with a dummy non-retain readout (dump slot after the plane ring), clear the other bank.
    {
        const uint32_t zero = 0;
        uint8_t * dump = b->planes + (size_t) b->ring * 8192;
        asm volatile("mxclracc\n");
        asm volatile("bias = mxmem2(%0)\n" : : "r"(b->tbl + 3 * 64));
        asm volatile("{ mxmem(%0, %1):after:cm.ub = acc }\n" : : "r"(dump), "r"(zero) : "memory");
        asm volatile("mxclracc\n");
    }
}

static void hmx_matmul_worker_fn(void * data) {
    hmx_matmul_job_t * job = (hmx_matmul_job_t *) data;
    FARF(HIGH, "hmx-mm-job: type %d n_row_tiles %u n_col_tiles %u n_dot_tiles %u",
         job->kernel_type, job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
    core_dot_chunk_fp16(job->output, (const __fp16 *) job->activation,
                        (const __fp16 *) job->weight, job->scales,
                        job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
}

static inline void hmx_matmul_job_init(hmx_matmul_job_t * job,
                                       __fp16 *           output,
                                       const void *       activation,
                                       const void *       weight,
                                       const __fp16 *     scales,
                                       uint32_t           n_row_tiles,
                                       uint32_t           n_col_tiles,
                                       uint32_t           n_dot_tiles) {
    job->kernel_type = HTP_MM_KERNEL_HMX_2D;
    job->output      = output;
    job->activation  = activation;
    job->weight      = weight;
    job->scales      = scales;
    job->bias        = NULL;
    job->cvt_reg     = 0;
    job->n_row_tiles = n_row_tiles;
    job->n_col_tiles = n_col_tiles;
    job->n_dot_tiles = n_dot_tiles;
}

// --- MM-03: chunk pipeline over the weights of one row block (hmx_mm_pipe_*) --------------------------------------------- Scheduling only. HMX accumulates each output tile over the dot tiles in the same fixed order whatever the column chunking, so chunk widths and chunk order never change an output bit. The exposed prologue (first weight DMA + dequant, HMX idle) and the exposed tail (last output transfer) halve.
// - HTP_OPFLAGS_MM_CROSS_WEIGHT: one chunk sequence over all weights of a NX group. Buffers alternate by global chunk number, so the parity carries across weights. Weight p+1's first chunk is fetched and dequantized while the last HMX job of weight p runs. Two effects, switched by op flags: - HTP_OPFLAGS_MM_SPLIT_EDGES: the first and last chunk of a weight are half width. With both flags clear the sequence is the per-weight order of the loops it replaces: DMA two chunks ahead, dequantize, submit the job, store the previous job's output.
#define HMX_MM_PIPE_MAX_W HTP_OP_MAX_OUTPUTS

typedef struct {
    uint32_t n_tiles;   // weight rows in 32-row tiles
    uint32_t nt;        // nominal chunk width in tiles
    uint32_t n_chunks;
    uint32_t edge;      // half-width first/last chunk in tiles (0 = plain chunking)
    uint32_t n_mid;     // full-width middle chunks (edge plan)
    uint32_t rem;       // remainder middle chunk in tiles (edge plan, 0 = none)
    uint32_t e1;        // width of chunk 1 when it is shrunk (HTP_OPFLAGS_MM_C1SMALL), 0 = off; the middle is then split into equal chunks
    uint32_t mid_k;     // number of middle chunks (e1 plan)
} hmx_mm_chunk_plan_t;

static inline void hmx_mm_chunk_plan_init(hmx_mm_chunk_plan_t * P, uint32_t n_tiles, uint32_t nt, bool split, bool c1small) {
    P->n_tiles = n_tiles;
    P->nt      = nt;
    P->edge    = 0;
    P->n_mid   = 0;
    P->rem     = 0;
    P->e1      = 0;
    P->mid_k   = 0;
    if (split && nt >= 2 && n_tiles > nt) {
        if (c1small && n_tiles > 2 * (nt / 2) + (nt / 2)) {
            // [edge, e1, mid..., edge]: chunk 1 is as narrow as chunk 0, so its weight DMA lands while the chunk-0 group jobs run;
            // the middle is divided into equal chunks of at most nt tiles.
            P->edge = nt / 2;
            P->e1   = nt / 2;
            const uint32_t rest = n_tiles - 2 * P->edge - P->e1;
            P->mid_k    = (rest + nt - 1) / nt;
            P->n_chunks = 3 + P->mid_k;
            return;
        }
        const uint32_t rest = n_tiles - 2 * (nt / 2);
        P->edge     = nt / 2;
        P->n_mid    = rest / nt;
        P->rem      = rest % nt;
        P->n_chunks = 2 + P->n_mid + (P->rem ? 1 : 0);
    } else {
        P->n_chunks = (n_tiles + nt - 1) / nt;
    }
}

// Chunk i of a plan: first tile and width, in tiles.
static inline void hmx_mm_chunk_get(const hmx_mm_chunk_plan_t * P, uint32_t i, uint32_t * t0, uint32_t * tw) {
    if (P->edge == 0) {
        *t0 = i * P->nt;
        *tw = (P->n_tiles - *t0 < P->nt) ? (P->n_tiles - *t0) : P->nt;
    } else if (i == 0) {
        *t0 = 0;
        *tw = P->edge;
    } else if (i + 1 == P->n_chunks) {
        *t0 = P->n_tiles - P->edge;
        *tw = P->edge;
    } else if (P->e1) {
        if (i == 1) {
            *t0 = P->edge;
            *tw = P->e1;
        } else {
            const uint32_t m    = i - 2;
            const uint32_t rest = P->n_tiles - 2 * P->edge - P->e1;
            const uint32_t base = rest / P->mid_k, extra = rest % P->mid_k;
            *tw = base + (m < extra ? 1 : 0);
            *t0 = P->edge + P->e1 + m * base + (m < extra ? m : extra);
        }
    } else {
        *t0 = P->edge + (i - 1) * P->nt;
        *tw = (i - 1 < P->n_mid) ? P->nt : P->rem;
    }
}

typedef struct {
    dma_addr_t    weight;         // weight rows in DDR
    size_t        weight_stride;  // src nb[1]
    size_t        n;              // weight rows (32-padded)
    float *       dst;            // output of this weight (row 0 of the op)
    size_t        dst_stride;     // floats per output row
    int           dst_cols;       // real output columns (crop)
    const float * src2;           // bias row in VTCM, added to every row; NULL = none
    size_t        src2_stride;    // floats per row of src2 (0 = broadcast row)
    bool          pe;             // HTP_TENSOR_A8W8: pre-expanded int8 weight blob (per column tile: n_k * 1024 B tiles, 128 B sw f32, 128 B cs i32)
} hmx_mm_pipe_w_t;

typedef struct {
    struct htp_context *  ctx;
    dma_queue *           dma;
    bool                  is_quant;
    int                   weight_type;
    int                   k;
    int                   n_k_tiles;
    struct fastdiv_values n_k_tiles_div;
    worker_callback_t     dequant_fn;
    size_t                row_stride;
    uint32_t              tile_size;
    uint32_t              dma_dst_stride;
    uint32_t              dma_width_bytes;
    int                   n_threads;
    int                   nc_tiles;     // nominal chunk width in tiles
    bool                  split;        // HTP_OPFLAGS_MM_SPLIT_EDGES
    bool                  c1small;      // HTP_OPFLAGS_MM_C1SMALL (applied only for K >= 3072: ffn_down)
    int                   n_rows;       // rows of the current block
    size_t                mr;           // first row of the current block
    const __fp16 *        act;          // fp16 activation tiles
    __fp16 *              scales;
    __fp16 *              raw[2];       // weight DMA buffers, by global chunk parity
    __fp16 *              wgt[2];       // dequantized weight tiles, by global chunk parity
    __fp16 *              out[2];       // HMX output tiles, by global chunk parity
    bool                    out_f16;      // HTP_OPFLAGS_MM_OUT_F16: the output rows are fp16 (transfer_output_chunk_f16_threaded)
    bool                    a8w8;         // HTP_OPFLAGS_MM_A8W8: u8 x int8 HMX jobs for this matmul (hmx_mm_pipe_run_a8w8)
} hmx_mm_pipe_t;

// Global chunk g -> weight index, first tile and width (tiles). first[j] = global index of weight j's chunk 0.
static inline void hmx_mm_pipe_locate(const hmx_mm_chunk_plan_t * plan, const uint32_t * first, int n_w, uint32_t g,
                                      int * w, uint32_t * t0, uint32_t * tw) {
    int j = 0;
    while (j + 1 < n_w && g >= first[j + 1]) {
        ++j;
    }
    hmx_mm_chunk_get(&plan[j], g - first[j], t0, tw);
    *w = j;
}

// DMA chunk g of the weight sequence into raw.
static inline void hmx_mm_pipe_issue(const hmx_mm_pipe_t * P, const hmx_mm_pipe_w_t * W, const hmx_mm_chunk_plan_t * plan,
                                     const uint32_t * first, int n_w, uint32_t g, void * raw) {
    int w; uint32_t t0, tw;
    hmx_mm_pipe_locate(plan, first, n_w, g, &w, &t0, &tw);
    const dma_addr_t src          = W[w].weight + (size_t) t0 * HTP_MM_HMX_TILE_N_COLS * W[w].weight_stride;
    const uint32_t   dma_src_stride = P->is_quant ? P->tile_size : (uint32_t) W[w].weight_stride;
    uint32_t         height         = P->is_quant ? tw * (uint32_t) P->n_k_tiles : tw * HTP_MM_HMX_TILE_N_COLS;
    if ((g_a8w8_dbg & 1) && P->a8w8) { height = (height + 1) / 2; }
    uint32_t         row_bytes      = P->dma_width_bytes;
    if ((g_a8w8_dbg & (1u << 31)) && P->a8w8 && P->is_quant) { row_bytes = 1024; }   // probe: int8-sized weight reads (overlapping garbage writes)
    dma_queue_push(P->dma, dma_make_data(raw, src), P->dma_dst_stride, dma_src_stride, row_bytes, height);
}

static inline void hmx_mm_pipe_dequant(const hmx_mm_pipe_t * P, __fp16 * dst, const void * raw, uint32_t tw) {
    dequantize_tiled_weight_chunk_to_fp16_tiles(P->ctx, dst, raw, (int) (tw * HTP_MM_HMX_TILE_N_COLS), P->k, P->row_stride,
                                                P->weight_type, P->n_k_tiles, P->n_k_tiles_div, P->dequant_fn, P->n_threads);
}

// Store the output of chunk g (fp16 tiles in out) to the weight's output rows, cropped to its real columns.
static inline void hmx_mm_pipe_store(const hmx_mm_pipe_t * P, const hmx_mm_pipe_w_t * W, const hmx_mm_chunk_plan_t * plan,
                                     const uint32_t * first, int n_w, uint32_t g, const __fp16 * out) {
    int w; uint32_t t0, tw;
    hmx_mm_pipe_locate(plan, first, n_w, g, &w, &t0, &tw);
    const size_t nc       = (size_t) t0 * HTP_MM_HMX_TILE_N_COLS;
    const int    dst_cols = W[w].dst_cols - (int) nc;
    if (dst_cols <= 0) {
        return;
    }
    const int    n_out_cols = (int) (tw * HTP_MM_HMX_TILE_N_COLS);
    if (P->out_f16) {
        // fp16 partial output (HTP_OPFLAGS_MM_OUT_F16): compact fp16 rows (row stride = dst_stride halves), columns in halves
        __fp16 * out16 = (__fp16 *) ((uint8_t *) W[w].dst + (size_t) P->mr * W[w].dst_stride * sizeof(float)) + nc;
        transfer_output_chunk_f16_threaded(P->ctx, out16, (size_t) W[w].dst_stride * sizeof(float), out, P->n_rows, n_out_cols, dst_cols, P->n_threads);
        return;
    }
    float *       dst  = W[w].dst + (size_t) P->mr * W[w].dst_stride + nc;
    const float * src2 = W[w].src2 ? W[w].src2 + (size_t) P->mr * W[w].src2_stride + nc : NULL;
    transfer_output_chunk_threaded(P->ctx, dst, src2, out, P->n_rows, n_out_cols,
                                   (int) W[w].dst_stride, (uint32_t) W[w].src2_stride, dst_cols, P->n_threads);
}

// First-chunk activation overlap (GGML_HEXAGON_MM_ACT_OVERLAP, HTP_OPFLAGS_MM_ACT_OVERLAP; fp16-tile producers only). Tiles keep their layout, so every output tile sees the same K order as in the whole-block job. (An fp32 -> fp16 conversion variant was measured slower and is not kept.)
// An M block of at least HMX_ACT_OVERLAP_MIN_ROWS rows is DMA'd in row groups (whole 32-row tile rows; the last group takes the rest), and chunk 0 runs as one HMX job per group: group r+1's tile DMA runs while job r executes.
#define HMX_ACT_OVERLAP_MIN_ROWS   256
#define HMX_ACT_OVERLAP_ROWS       128
#define HMX_ACT_OVERLAP_MAX_GROUPS 4

typedef struct {
    int        n_groups;                                // >= 2
    int        group_rows[HMX_ACT_OVERLAP_MAX_GROUPS];
    dma_addr_t tiles;                                   // the block's fp16 activation tiles in DDR
    size_t     k_tiles;                                 // K / 32 tile columns
} hmx_mm_act_stage_t;

// Groups of an M block of n_rows rows: ceil(n_rows / g) rounded up to whole tile rows, the last group takes the rest.
// Returns the number of groups; 0 keeps the original staging.
static int hmx_act_overlap_plan(size_t n_rows, int * group_rows) {
    if (n_rows < HMX_ACT_OVERLAP_MIN_ROWS) {
        return 0;
    }
    size_t g = n_rows / HMX_ACT_OVERLAP_ROWS;
    if (g > HMX_ACT_OVERLAP_MAX_GROUPS) {
        g = HMX_ACT_OVERLAP_MAX_GROUPS;
    }
    const size_t per = hex_align_up((n_rows + g - 1) / g, HTP_MM_HMX_TILE_N_ROWS);
    for (size_t r = 0; r + 1 < g; ++r) {
        group_rows[r] = (int) per;
    }
    group_rows[g - 1] = (int) (n_rows - per * (g - 1));
    return (int) g;
}

// Stage row group [row0, row0 + rows) by DMA of its fp16 tiles (queue q; the caller pops it in order).
static void hmx_act_group_dma(const hmx_mm_act_stage_t * S, dma_queue * q, __fp16 * vtcm_act, size_t row0, size_t rows) {
    const size_t tr0 = row0 / HTP_MM_HMX_TILE_N_ROWS;
    act_tiles_dma_push(q, vtcm_act + tr0 * S->k_tiles * HTP_MM_HMX_TILE_N_ELMS, S->tiles + tr0 * S->k_tiles * HTP_MM_HMX_TILE_SIZE,
                       (hex_align_up(rows, HTP_MM_HMX_TILE_N_ROWS) / HTP_MM_HMX_TILE_N_ROWS) * S->k_tiles * HTP_MM_HMX_TILE_SIZE);
}

// Chunk sequence of the weights W[0..n_w) for the row block P->mr .. Returns after every HMX job of the block is stored.
// P->mr + P->n_rows. The activation tiles must be in place unless stage is set: then they are staged here in row groups, overlapped with chunk 0 (hmx_mm_act_stage_t).
static void hmx_mm_pipe_run(const hmx_mm_pipe_t * P, const hmx_mm_pipe_w_t * W, int n_w, const hmx_mm_act_stage_t * stage) {
    hmx_mm_chunk_plan_t plan[HMX_MM_PIPE_MAX_W];
    uint32_t            first[HMX_MM_PIPE_MAX_W];
    uint32_t            total = 0;
    for (int w = 0; w < n_w; ++w) {
        hmx_mm_chunk_plan_init(&plan[w], (uint32_t) (W[w].n / HTP_MM_HMX_TILE_N_COLS), (uint32_t) P->nc_tiles, P->split, P->c1small && P->n_k_tiles >= 96);
        first[w] = total;
        total   += plan[w].n_chunks;
    }
    if (total == 0) {
        return;
    }

    hmx_matmul_job_t job[2];
    hmx_matmul_job_t job_pre;   // chunk 1 submitted ahead of the chunk-0 store (staged blocks)
    const uint32_t   n_row_tiles = (uint32_t) hmx_ceil_div((size_t) P->n_rows, HTP_MM_HMX_TILE_N_ROWS);
    const uint32_t   n_dot_tiles = (uint32_t) (P->k / HTP_MM_HMX_TILE_N_ROWS);
    const uint32_t   g_first     = stage ? 1 : 0;   // chunk 0 is complete before the loop when staged in groups

    if (stage && P->ctx->mm_early_pending) {
        P->ctx->mm_early_pending = false;   // chunk 0 of weight 0 is already in flight (htp_mm_early_weight_dma)
    } else {
        hmx_mm_pipe_issue(P, W, plan, first, n_w, 0, P->raw[0]);
    }
    if (stage) {
        // group tiles queue behind chunk 0's weight DMA and ahead of chunk 1's; the pops below follow this order
        size_t row0 = 0;
        for (int r = 0; r < stage->n_groups; ++r) {
            hmx_act_group_dma(stage, P->dma, (__fp16 *) P->act, row0, (size_t) stage->group_rows[r]);
            row0 += (size_t) stage->group_rows[r];
        }
    }
    if (total > 1) {
        hmx_mm_pipe_issue(P, W, plan, first, n_w, 1, P->raw[1]);
    }

    if (stage) {
        // chunk 0 (weight 0) as one HMX job per row group. At most two jobs are in flight (slot r & 1).
        uint32_t t00, tw0;
        hmx_mm_chunk_get(&plan[0], 0, &t00, &tw0);
        (void) t00;
        void * raw0 = (void *) dma_queue_pop(P->dma).dst;
        hmx_mm_pipe_dequant(P, P->wgt[0], raw0, tw0);
        if (total > 2) {
            hmx_mm_pipe_issue(P, W, plan, first, n_w, 2, raw0);
        }
        size_t row0 = 0;
        for (int r = 0; r < stage->n_groups; ++r) {
            const size_t rows = (size_t) stage->group_rows[r];
            dma_queue_pop(P->dma);   // group r's tiles have landed
            if (r >= 2) {
                hmx_queue_pop(P->ctx->hmx_queue);   // job r-2 has finished: its slot is free
            }
            const size_t tr0 = row0 / HTP_MM_HMX_TILE_N_ROWS;
            hmx_matmul_job_init(&job[r & 1], P->out[0] + tr0 * tw0 * HTP_MM_HMX_TILE_N_ELMS,
                                P->act + tr0 * n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS, P->wgt[0], P->scales,
                                (uint32_t) hmx_ceil_div(rows, HTP_MM_HMX_TILE_N_ROWS), tw0, n_dot_tiles);
            hmx_queue_push(P->ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job[r & 1]));
            row0 += rows;
        }
        // Chunk 1 is dequantized and submitted while the chunk-0 group jobs are still running, so its HMX job follows the last
        // group job without waiting for the chunk-0 output store and the dequant (measured gap ~33 us per ffn_down call).
        if (total > 1) {
            int      w1;
            uint32_t t01, tw1;
            hmx_mm_pipe_locate(plan, first, n_w, 1, &w1, &t01, &tw1);
            void * raw1 = (void *) dma_queue_pop(P->dma).dst;
            hmx_mm_pipe_dequant(P, P->wgt[1], raw1, tw1);
            if (total > 3) {
                hmx_mm_pipe_issue(P, W, plan, first, n_w, 3, raw1);
            }
            hmx_matmul_job_init(&job_pre, P->out[1], P->act, P->wgt[1], P->scales, n_row_tiles, tw1, n_dot_tiles);
            hmx_queue_push(P->ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_pre));
        }
        for (int r = stage->n_groups > 2 ? stage->n_groups - 2 : 0; r < stage->n_groups; ++r) {
            hmx_queue_pop(P->ctx->hmx_queue);
        }
        hmx_mm_pipe_store(P, W, plan, first, n_w, 0, P->out[0]);
    }

    for (uint32_t g = (stage && total > 1) ? 2 : g_first; g < total; ++g) {
        int      w;
        uint32_t t0, tw;
        hmx_mm_pipe_locate(plan, first, n_w, g, &w, &t0, &tw);

        // pop chunk g's raw DMA and dequantize it into buffer g&1 (job g-2 used that buffer and has been popped)
        void * curr_raw = (void *) dma_queue_pop(P->dma).dst;
        hmx_mm_pipe_dequant(P, P->wgt[g & 1], curr_raw, tw);
        // refill the raw buffer with chunk g+2
        if (g + 2 < total) {
            hmx_mm_pipe_issue(P, W, plan, first, n_w, g + 2, curr_raw);
        }
        // submit job g; then wait for job g-1 and store it (its output buffer is the one job g+1 reuses)
        hmx_matmul_job_init(&job[g & 1], P->out[g & 1], P->act, P->wgt[g & 1], P->scales, n_row_tiles, tw, n_dot_tiles);
        hmx_queue_push(P->ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job[g & 1]));
        if (g > g_first) {
            hmx_queue_pop(P->ctx->hmx_queue);
            hmx_mm_pipe_store(P, W, plan, first, n_w, g - 1, P->out[(g - 1) & 1]);
        }
    }
    if (total > g_first) {
        hmx_queue_pop(P->ctx->hmx_queue);
        hmx_mm_pipe_store(P, W, plan, first, n_w, total - 1, P->out[(total - 1) & 1]);
    }
}


// ---- A8W8 for the plain chunk pipeline (MUL_MAT / MUL_MAT_NX, HTP_OPFLAGS_MM_A8W8 + P->a8w8): the same dataflow as the gate/up kernel without the SWIGLU epilogue. VTCM (all inside regions the fp16 path owns): wgt[p]: [int8 tiles of chunk parity p (nct * tile_k KB)] [p = 0: act8 | rscale | sw, cs (2 parities) ; p = 1: plane ring of the parity-1 jobs + dump slot] act : plane ring of the parity-0 jobs + dump slot (the fp16 act tiles are dead after the quantiser) Pre-expanded weights: chunk g = tw consecutive column tiles of the blob (stride n_k * 1024 + 256 B per column tile) copied by one contiguous DMA into `dst` (tw * (n_k*1024+256) bytes); the expand step is replaced by a scale unpack (a8w8_pe_unpack_worker) from the tail of each column tile.
// The fp16 activation tiles of the row block are quantised to u8 once, every weight chunk is expanded Q4_0 -> int8 into the fp16 weight scratch of its parity, the HMX queue runs u8 x int8 jobs and the post workers write the fp16 output tiles to P->out[g & 1] (HMX output layout), which hmx_mm_pipe_store transfers exactly as in the fp16 path.
static inline void hmx_mm_a8w8_pe_issue(const hmx_mm_pipe_t * P, const hmx_mm_pipe_w_t * W, const hmx_mm_chunk_plan_t * plan,
                                        const uint32_t * first, int n_w, uint32_t g, void * dst) {
    int w; uint32_t t0, tw;
    hmx_mm_pipe_locate(plan, first, n_w, g, &w, &t0, &tw);
    const size_t ct_bytes = (size_t) P->n_k_tiles * 1024 + HTP_A8W8_PE_TAIL;
    dma_queue_push(P->dma, dma_make_data(dst, W[w].weight + (size_t) t0 * ct_bytes), ct_bytes * tw, ct_bytes * tw, ct_bytes * tw, 1);
}

// A8W8 first-chunk shaping (probe/tuning, op_params[3] bits 8..15 = first/last chunk tiles E, 16..23 = chunk-1 tiles, 24 = activation groups of one 64-row block):
// a narrow chunk 0 puts less weight DMA ahead of the first activation group, so the first HMX job starts earlier.
static inline void hmx_mm_a8w8_plan_init(hmx_mm_chunk_plan_t * P, uint32_t n_tiles, uint32_t nt, bool split, bool c1small) {
    const uint32_t E = (uint32_t) (g_a8w8_dbg >> 8) & 0xff, E1 = (uint32_t) (g_a8w8_dbg >> 16) & 0xff;
    if (E > 0 && E1 > 0 && n_tiles > 2 * E + E1 + nt) {
        P->n_tiles = n_tiles; P->nt = nt; P->edge = E; P->e1 = E1; P->n_mid = 0; P->rem = 0;
        const uint32_t rest = n_tiles - 2 * E - E1;
        P->mid_k    = (rest + nt - 1) / nt;
        P->n_chunks = 3 + P->mid_k;
        return;
    }
    hmx_mm_chunk_plan_init(P, n_tiles, nt, split, c1small);
}

static bool hmx_mm_a8w8_fits(uint32_t k, size_t mc, size_t nc_tiles, bool pe) {
    const size_t tile_k = k / 32;
    const size_t nb64   = (hex_align_up(mc, 64)) / 64;
    const size_t scr    = nc_tiles * tile_k * 2048;                                         // one weight scratch buffer (nc * k fp16 bytes)
    const size_t half   = hex_align_up(nc_tiles * (tile_k * 1024 + (pe ? HTP_A8W8_PE_TAIL : 0)), 2048);
    if (half > scr) return false;
    const size_t avail  = scr - half;                                                       // room behind the int8 tiles of parity p
    const size_t need0  = nb64 * tile_k * 2048 + nb64 * 256 + 4 * nc_tiles * 128 + 256;
    const size_t need1  = nb64 * nc_tiles * 8192 + 8192;
    const size_t actb   = hex_align_up(mc * k * 2, 2048);
    return need0 <= avail && need1 <= avail && need1 <= actb;
}

// reuse_act8: the u8 activations / row scales of the previous call (same row block) are still in place (NX: second and later weights)
static void hmx_mm_pipe_run_a8w8(const hmx_mm_pipe_t * P, const hmx_mm_pipe_w_t * W, int n_w, dma_addr_t act_src, size_t act_bytes, bool reuse_act8) {
    hmx_mm_chunk_plan_t plan[HMX_MM_PIPE_MAX_W];
    uint32_t            first[HMX_MM_PIPE_MAX_W];
    uint32_t            total = 0;
    for (int w = 0; w < n_w; ++w) {
        hmx_mm_a8w8_plan_init(&plan[w], (uint32_t) (W[w].n / HTP_MM_HMX_TILE_N_COLS), (uint32_t) P->nc_tiles, P->split, P->c1small && P->n_k_tiles >= 96);
        first[w] = total;
        total   += plan[w].n_chunks;
    }
    if (total == 0) {
        return;
    }
    struct htp_context * ctx = P->ctx;
    const uint32_t tile_k      = (uint32_t) P->n_k_tiles;
    const uint32_t n_row_tiles = (uint32_t) hmx_ceil_div((size_t) P->n_rows, HTP_MM_HMX_TILE_N_ROWS);
    const uint32_t n_blocks64  = (n_row_tiles + 1) / 2;
    const uint32_t nct_max     = (uint32_t) P->nc_tiles;
    const int      n_threads   = P->n_threads;
    const bool     pe          = W[0].pe;     // all weights of one call are pre-expanded or none (the caller checks)
    const uint32_t wstride     = pe ? tile_k * 1024 + HTP_A8W8_PE_TAIL : 0;

    const size_t half = hex_align_up((size_t) nct_max * (tile_k * 1024 + (pe ? HTP_A8W8_PE_TAIL : 0)), 2048);
    uint8_t * w8_par[2]     = { (uint8_t *) P->wgt[0], (uint8_t *) P->wgt[1] };
    uint8_t * act8          = (uint8_t *) P->wgt[0] + half;
    float *   rscale        = (float *) (act8 + (size_t) n_blocks64 * tile_k * 2048);
    float *   sw_par        = rscale + n_blocks64 * 64;                                    // [2][nct_max * 32]
    int32_t * cs_par        = (int32_t *) (sw_par + 2 * nct_max * 32);
    uint8_t * planes_par[2] = { (uint8_t *) P->act, (uint8_t *) P->wgt[1] + half };
    uint32_t * tbl          = (uint32_t *) P->scales;
    volatile int * produced_par[2] = { (volatile int *) ((uint8_t *) P->scales + 1024), (volatile int *) ((uint8_t *) P->scales + 1024 + 128) };
    {
        static const uint32_t ee1x[4] = { 24, 16, 8, 0 };
        for (int t = 0; t < 4; ++t) for (int q = 0; q < 64; ++q) tbl[t * 64 + q] = (q < 32) ? (ee1x[t] << 10) : 0;
    }

    a8w8_state_t st = { .act16 = P->act, .act8 = act8, .rscale = rscale, .tile_k = tile_k, .n_blocks64 = n_blocks64, .n_row_tiles = n_row_tiles,
                        .aligned = (uint32_t) P->dma_dst_stride, .traces = ctx->trace };
    a8w8_job_t jobs[2];
    a8w8_post_state_t ps;
    struct htp_thread_trace * tr0 = &ctx->trace[0];

    if (ctx->mm_early_pending) {
        ctx->mm_early_pending = false;   // chunk 0 is already in flight in raw[0] (htp_mm_early_weight_dma): same FIFO position as the push below
    } else {
        if (pe) hmx_mm_a8w8_pe_issue(P, W, plan, first, n_w, 0, w8_par[0]); else hmx_mm_pipe_issue(P, W, plan, first, n_w, 0, P->raw[0]);
    }
    // The activation tiles arrive in groups of two 64-row blocks (queued behind chunk 0's weights): each group is quantised as soon as it lands and HMX(0) starts after the first group (its job waits on the per-block counter qready).
    // Parity-0 plane slots overwrite the already quantised fp16 tiles from the front, which is safe while a block's slots (nct * 8 KB) fit into its fp16 tiles (tile_k * 4 KB).
    const bool     q_groups = act_src && n_blocks64 >= 2 && 2 * nct_max <= tile_k;   // parity-0 slots (nct * 8 KB per block) fit into the block's fp16 tiles
    const uint32_t gb       = (g_a8w8_dbg & (1 << 24)) ? 1u : 2u;    // 64-row blocks per activation group
    const uint32_t n_groups = q_groups ? (n_blocks64 + gb - 1) / gb : 0;
    volatile int * qready   = (volatile int *) ((uint8_t *) P->scales + 1024 + 256);
    if (q_groups) {
        const size_t tile_bytes = (size_t) tile_k * HTP_MM_HMX_TILE_SIZE;
        for (uint32_t gi = 0; gi < n_groups; ++gi) {
            const uint32_t rt0 = gi * 2 * gb, rt1 = (rt0 + 2 * gb < n_row_tiles) ? rt0 + 2 * gb : n_row_tiles;
            act_tiles_dma_push(P->dma, (void *) ((uint8_t *) P->act + (size_t) rt0 * tile_bytes), act_src + (size_t) rt0 * tile_bytes, (size_t) (rt1 - rt0) * tile_bytes);
        }
    } else if (act_src) {
        act_tiles_dma_push(P->dma, (void *) P->act, act_src, act_bytes);
    }
    if (total > 1) {
        if (pe) hmx_mm_a8w8_pe_issue(P, W, plan, first, n_w, 1, w8_par[1]); else hmx_mm_pipe_issue(P, W, plan, first, n_w, 1, P->raw[1]);
    }

    // chunk 0: expand while the activation DMA is in flight, then quantise and start HMX(0)
    {
        int w; uint32_t t0, tw;
        hmx_mm_pipe_locate(plan, first, n_w, 0, &w, &t0, &tw);
        void * raw = (void *) dma_queue_pop(P->dma).dst;
        st.raw = (const uint8_t *) raw; st.n_col_tiles = tw; st.w8 = w8_par[0]; st.sw = sw_par; st.cs = cs_par;
        st.n_tasks = (uint32_t) n_threads < tw ? (uint32_t) n_threads : tw;
        if (pe) {
            st.tile_k = tile_k; st.n_tasks = 1;
            a8w8_pe_unpack_worker(1, 0, &st);   // tails of the blob (already in w8_par[0] + ct stride)
        } else {
            worker_pool_run_func(ctx->worker_pool, a8w8_expand_worker, &st, (int) st.n_tasks);
            if (total > 2) {
                hmx_mm_pipe_issue(P, W, plan, first, n_w, 2, raw);
            }
        }
        *produced_par[0] = 0;
        jobs[0] = (a8w8_job_t) { .act8 = act8, .w8 = w8_par[0], .planes = planes_par[0], .tbl = tbl, .tile_k = tile_k, .n_blocks64 = n_blocks64,
                                 .n_col_tiles = tw, .ring = n_blocks64 * tw, .produced = produced_par[0], .qready = NULL, .wstride = wstride };
        if (q_groups) {
            *qready = 0;
            jobs[0].qready = qready;
            for (uint32_t gi = 0; gi < n_groups; ++gi) {
                dma_queue_pop(P->dma);
                st.qb0 = gi * gb; st.qnb = (gi * gb + gb <= n_blocks64) ? gb : (n_blocks64 - gi * gb);
                st.n_tasks = (uint32_t) n_threads < st.qnb * 16 ? (uint32_t) n_threads : st.qnb * 16;
                htp_trace_event_start(tr0, HTP_TRACE_EVT_HVX_A_QUANT, gi);
                worker_pool_run_func(ctx->worker_pool, a8w8_quant_worker, &st, (int) st.n_tasks);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_HVX_A_QUANT, gi);
                *qready = (int) (st.qb0 + st.qnb);
                if (gi == 0) {
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(a8w8_hmx_job, &jobs[0]));
                }
            }
            st.qb0 = 0; st.qnb = 0;
        } else {
            if (act_src) {
                dma_queue_pop(P->dma);
            }
            if (!reuse_act8) {
                st.n_tasks = (uint32_t) n_threads < n_blocks64 * 16 ? (uint32_t) n_threads : n_blocks64 * 16;
                htp_trace_event_start(tr0, HTP_TRACE_EVT_HVX_A_QUANT, 0);
                worker_pool_run_func(ctx->worker_pool, a8w8_quant_worker, &st, (int) st.n_tasks);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_HVX_A_QUANT, 0);
            }
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(a8w8_hmx_job, &jobs[0]));
        }
    }

    for (uint32_t g = 0; g < total; ++g) {
        const int b = (int) (g & 1);
        int w; uint32_t t0, tw;
        hmx_mm_pipe_locate(plan, first, n_w, g, &w, &t0, &tw);
        if (g + 1 < total) {
            int w1; uint32_t t01, tw1;
            hmx_mm_pipe_locate(plan, first, n_w, g + 1, &w1, &t01, &tw1);
            void * raw = (void *) dma_queue_pop(P->dma).dst;
            st.raw = (const uint8_t *) raw; st.n_col_tiles = tw1; st.w8 = w8_par[b ^ 1]; st.sw = sw_par + (b ^ 1) * nct_max * 32; st.cs = cs_par + (b ^ 1) * nct_max * 32;
            st.n_tasks = (uint32_t) n_threads < tw1 ? (uint32_t) n_threads : tw1;
            if (pe) {
                st.tile_k = tile_k; st.n_tasks = 1;
                a8w8_pe_unpack_worker(1, 0, &st);
            } else {
                worker_pool_run_func(ctx->worker_pool, a8w8_expand_worker, &st, (int) st.n_tasks);
                if (g + 3 < total) {
                    hmx_mm_pipe_issue(P, W, plan, first, n_w, g + 3, raw);
                }
            }
            *produced_par[b ^ 1] = 0;
            jobs[b ^ 1] = (a8w8_job_t) { .act8 = act8, .w8 = w8_par[b ^ 1], .planes = planes_par[b ^ 1], .tbl = tbl, .tile_k = tile_k, .n_blocks64 = n_blocks64,
                                         .n_col_tiles = tw1, .ring = n_blocks64 * tw1, .produced = produced_par[b ^ 1], .wstride = wstride };
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(a8w8_hmx_job, &jobs[b ^ 1]));
        }
        ps = (a8w8_post_state_t) { .planes = planes_par[b], .sw = sw_par + b * nct_max * 32, .cs = cs_par + b * nct_max * 32, .rscale = rscale,
                                   .out = P->out[b], .n_col_tiles = tw, .n_blocks64 = n_blocks64, .n_row_tiles = n_row_tiles,
                                   .ring = n_blocks64 * tw, .n_tasks = (uint32_t) n_threads, .traces = ctx->trace, .produced = produced_par[b],
                                   .epi_g = NULL };
        htp_trace_event_start(tr0, HTP_TRACE_EVT_INIT, 3);
        worker_pool_run_func(ctx->worker_pool, a8w8_post_worker, &ps, n_threads);
        htp_trace_event_stop(tr0, HTP_TRACE_EVT_INIT, 3);
        hmx_queue_pop(ctx->hmx_queue);                      // HMX(g) complete (all its slots were posted)
        if (pe && g + 2 < total) {
            hmx_mm_a8w8_pe_issue(P, W, plan, first, n_w, g + 2, w8_par[b]);   // the blob lands in the int8 buffer HMX(g) just released
        }
        hmx_mm_pipe_store(P, W, plan, first, n_w, g, P->out[b]);
    }
    hmx_init_column_scales(P->scales, Q6_V_vsplat_R(0x3c00));   // the bias tables overwrote the fp16 scale tile
}

static int hmx_mm_2d_f32(struct htp_context *ctx,
                                  dma_queue *weight_dma,
                                  float *restrict dst,
                                  dma_addr_t src2_addr,
                                  size_t src2_bytes,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t weight,
                                  int m, int k, int n,
                                  int act_stride,
                                  int weight_stride,
                                  int weight_type,
                                  int kernel_type,
                                  int k_valid,
                                  int dst_stride,
                                  uint32_t src2_stride,
                                  int dst_cols,
                                  int m_chunk,
                                  int n_chunk,
                                  int pipeline,
                                  int n_threads,
                                  int act_threads,
                                  const struct fastdiv_values * act_threads_div,
                                  const struct fastdiv_values * k_div,
                                  int tile_size,
                                  int aligned_tile_size,
                                  int vtcm_size,
                                  bool act_tiled, uint32_t op_flags) {
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    if (k % 32 != 0 || n % 32 != 0) { return -1; }
    if (!hex_is_aligned(dst, VLEN) || (act_dma_addr & (VLEN - 1)) != 0) { return -1; }


    size_t row_stride = htp_mm_get_tiled_row_stride(weight_type, k);
    if (row_stride == 0) {
        return -1;
    }

    worker_callback_t dequant_worker_fn = NULL;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_0; break;
        case HTP_TYPE_IQ4_NL: dequant_worker_fn = dequantize_tiled_worker_loop_iq4_nl; break;
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q4_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_1; break;
        case HTP_TYPE_MXFP4:  dequant_worker_fn = dequantize_tiled_worker_loop_mxfp4; break;
        case HTP_TYPE_Q8_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q8_0; break;
        case HTP_TYPE_Q5_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q5_k; break;
        case HTP_TYPE_Q6_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q6_k; break;
        case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
        case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
        default:
            return -1;
    }

    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const bool is_hmx_int = (kernel_type == HTP_MM_KERNEL_HMX_INT4 && weight_type == HTP_TYPE_Q4_0);
    if (is_hmx_int) {
        pipeline = 0;
    }

    const bool is_quant       = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);
    const size_t vec_dot_size = k * sizeof(__fp16);
    const size_t vtcm_budget  = ctx->vtcm_size;
    // elements per row -- reading `row_stride` bytes from DDR would run past the true row (into the next row, or off the buffer end on the last row). F32 weights are never K-padded (see ggml_hexagon_precompute_matmul_params_impl), so their row_stride is already exactly k_valid*sizeof(float) -- leave as-is.
    // Clamp the DMA's row_size (not its dst/src strides) to the true unpadded byte width; the never-DMA'd tail of each VTCM row is zeroed once below instead, so the tile converter reads real zeros, not whatever was previously in that VTCM buffer.
    const bool needs_k_pad = (weight_type == HTP_TYPE_F16) && ((size_t) k_valid * sizeof(__fp16) < row_stride);
    const uint32_t dma_dst_stride  = is_quant ? aligned_tile_size : row_stride;
    const uint32_t dma_src_stride  = is_quant ? tile_size : weight_stride;
    const uint32_t dma_width_bytes = is_quant ? tile_size :
        (needs_k_pad ? (uint32_t)((size_t) k_valid * sizeof(__fp16)) : row_stride);

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used      = vtcm_size;

    const size_t qweight_row_stride = is_quant ? (size_t)(n_k_tiles * aligned_tile_size) / 32 : 0;

    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, kernel_type, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, src2_bytes);

    vtcm_used = L.total_bytes;
    if (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_used, vtcm_budget, m, k, n, m_chunk_n_rows, n_chunk_n_cols);
        return -1;
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    if (needs_k_pad) {
        // The DMA above only ever writes [0, dma_width_bytes) of each row_stride-wide VTCM row; zero the whole buffer once up front so every row's never-DMA'd tail (k_valid..k_padded elements) reads as a real zero in convert_f16_weight_to_fp16_tiles_task, not stale VTCM contents from a previous op.
        memset(vtcm_weight_raw[0], 0, L.weight_area_bytes);
        if (pipeline) {
            memset(vtcm_weight_raw[1], 0, L.weight_area_bytes);
        }
    }

    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    float   *vtcm_f32_act    = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    void    *vtcm_scratch0   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]);
    void    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_scratch[1], pipeline);
    void    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    const bool has_src2      = (src2_bytes > 0 && src2_addr != 0);
    float   *vtcm_src2       = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_src2, has_src2);
    if (has_src2) {
        dma_queue_push(weight_dma, dma_make_data(vtcm_src2, src2_addr), hex_align_up(src2_bytes, 128), 0, src2_bytes, 1);
        dma_queue_pop(weight_dma);
    }

    FARF(HIGH, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, weight_type, m_chunk_n_rows, n_chunk_n_cols, vtcm_used, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    if (pipeline) {
        // Chunk pipeline (hmx_mm_pipe_run): one weight, one row block per iteration.
        hmx_mm_pipe_t P = {
            .ctx = ctx, .dma = weight_dma, .is_quant = is_quant, .weight_type = weight_type, .k = k,
            .n_k_tiles = n_k_tiles, .n_k_tiles_div = n_k_tiles_div, .dequant_fn = dequant_worker_fn,
            .row_stride = row_stride, .tile_size = (uint32_t) tile_size,
            .dma_dst_stride = dma_dst_stride, .dma_width_bytes = dma_width_bytes, .n_threads = n_threads,
            .nc_tiles = (int) (n_chunk_n_cols / HTP_MM_HMX_TILE_N_COLS), .split = (op_flags & HTP_OPFLAGS_MM_SPLIT_EDGES) != 0, .c1small = (op_flags & HTP_OPFLAGS_MM_C1SMALL) != 0,
            .act = vtcm_f16_act, .scales = vtcm_scales,
            .raw = { vtcm_weight_raw[0], vtcm_weight_raw[1] },
            .wgt = { (__fp16 *) vtcm_scratch0, (__fp16 *) vtcm_scratch1 },
            .out = { vtcm_output, (__fp16 *) vtcm_scratch2 },
        };
        const bool pe_ok = g_a8w8_pe_weight != 0 && g_a8w8_pe_op && (op_flags & HTP_OPFLAGS_MM_A8W8) != 0 && weight_type == HTP_TYPE_Q4_0 &&
                           hmx_mm_a8w8_fits((uint32_t) k, (size_t) m_chunk_n_rows, (size_t) (n_chunk_n_cols / HTP_MM_HMX_TILE_N_COLS), true);
        const hmx_mm_pipe_w_t W = {
            .pe = pe_ok,
            .weight = weight, .weight_stride = (size_t) weight_stride, .n = (size_t) n, .dst = dst,
            .dst_stride = (size_t) dst_stride, .dst_cols = dst_cols, .src2 = vtcm_src2, .src2_stride = src2_stride,
        };

        for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            struct activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + mr * act_stride * sizeof(float),
                .n_rows = (int) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .is_hmx_int = is_hmx_int,
                .vtcm_act_scales = (float *) vtcm_scales,
            };
            P.n_rows = (int) n_rows;
            P.out_f16 = (op_flags & HTP_OPFLAGS_MM_OUT_F16) != 0;
            P.mr     = mr;
            P.a8w8   = (op_flags & HTP_OPFLAGS_MM_A8W8) != 0 && weight_type == HTP_TYPE_Q4_0 && !is_hmx_int && n_rows >= 128 &&
                       hmx_mm_a8w8_fits((uint32_t) k, (size_t) m_chunk_n_rows, (size_t) (n_chunk_n_cols / HTP_MM_HMX_TILE_N_COLS), pe_ok);
            if (P.a8w8) {
                if (act_tiled) {
                    const size_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                    const size_t tile_off      = (mr / HTP_MM_HMX_TILE_N_ROWS) * n_k_tiles * HTP_MM_HMX_TILE_SIZE;
                    hmx_mm_pipe_w_t Wb = W; if (Wb.pe) { Wb.weight = g_a8w8_pe_weight; }
                    hmx_mm_pipe_run_a8w8(&P, &Wb, 1, act_dma_addr + tile_off, (n_rows_padded / HTP_MM_HMX_TILE_N_ROWS) * n_k_tiles * HTP_MM_HMX_TILE_SIZE, false);
                } else {
                    transfer_activation_chunk_threaded(&act_params);
                    hmx_mm_pipe_w_t Wb = W; if (Wb.pe) { Wb.weight = g_a8w8_pe_weight; }
                    hmx_mm_pipe_run_a8w8(&P, &Wb, 1, 0, 0, false);
                }
                continue;
            }
            int ov_rows[HMX_ACT_OVERLAP_MAX_GROUPS];
            const int ov_n = (act_tiled && (op_flags & HTP_OPFLAGS_MM_ACT_OVERLAP) != 0) ? hmx_act_overlap_plan(n_rows, ov_rows) : 0;
            if (ov_n > 1) {
                // activation DMA'd in row groups inside chunk 0 (hmx_mm_pipe_run)
                hmx_mm_act_stage_t stage = {
                    .n_groups = ov_n,
                    .tiles    = act_dma_addr + (mr / HTP_MM_HMX_TILE_N_ROWS) * n_k_tiles * HTP_MM_HMX_TILE_SIZE,
                    .k_tiles  = (size_t) n_k_tiles,
                };
                for (int r = 0; r < ov_n; ++r) {
                    stage.group_rows[r] = ov_rows[r];
                }
                hmx_mm_pipe_run(&P, &W, 1, &stage);
            } else {
                if (act_tiled) {
                    // Producer-written fp16 HMX tiles (HTP_TENSOR_ACT_TILE): DMA them, no conversion. The weight DMA
                    // queue is empty here, so the push/pop pair is FIFO-safe ahead of the weight prologue.
                    const size_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                    const size_t tile_off      = (mr / HTP_MM_HMX_TILE_N_ROWS) * n_k_tiles * HTP_MM_HMX_TILE_SIZE;
                    act_tiles_dma_push(weight_dma, vtcm_f16_act, act_dma_addr + tile_off,
                                       (n_rows_padded / HTP_MM_HMX_TILE_N_ROWS) * n_k_tiles * HTP_MM_HMX_TILE_SIZE);
                    dma_queue_pop(weight_dma);
                } else {
                    transfer_activation_chunk_threaded(&act_params);
                }
                hmx_mm_pipe_run(&P, &W, 1, NULL);
            }
        }
    } else {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_job_t job;
        for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            struct activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + mr * act_stride * sizeof(float),
                .n_rows = (int) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .is_hmx_int = is_hmx_int,
                .vtcm_act_scales = (float *) vtcm_scales,
            };
            if (act_tiled) {
                const size_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                const size_t tile_off      = (mr / HTP_MM_HMX_TILE_N_ROWS) * n_k_tiles * HTP_MM_HMX_TILE_SIZE;
                act_tiles_dma_push(weight_dma, vtcm_f16_act, act_dma_addr + tile_off,
                                   (n_rows_padded / HTP_MM_HMX_TILE_N_ROWS) * n_k_tiles * HTP_MM_HMX_TILE_SIZE);
                dma_queue_pop(weight_dma);
            } else {
                transfer_activation_chunk_threaded(&act_params);
            }

            // A0: Pre-fetch the first weight chunk (nc = 0)
            if (n > 0) {
                const size_t n_cols = hex_smin(n, n_chunk_n_cols);
                const uint32_t height = is_quant ? (n_cols / 32) * n_k_tiles : n_cols;
                dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
            }

            for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
                const size_t n_cols = hex_smin(n - nc, n_chunk_n_cols);
                const size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                // A: Wait for weight DMA
                void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;


                const void * wt_ptr = curr_raw;
                // B: Weight Dequantize (Threaded) - skip for HMX INT8/INT4
                if (!is_hmx_int) {
                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_scratch0, curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);
                    wt_ptr = vtcm_scratch0;
                }

                // Start weight DMA for the next chunk early
                const size_t nc_next = nc + n_chunk_n_cols;
                if (!is_hmx_int && nc_next < n) {
                    const size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                    const uint32_t height_next = is_quant ? (n_cols_next / 32) * n_k_tiles : n_cols_next;
                    dma_queue_push(weight_dma, dma_make_data(curr_raw, weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                }

                // C: HMX Compute
                if (is_hmx_int) {
                    float * out_f32 = (float *) vtcm_output;
                    const float * s2 = has_src2 ? (vtcm_src2 + mr * src2_stride + nc) : NULL;
                    hmx_int_chunk_job_t int_job = {
                        .output_f32  = out_f32,
                        .act_ub      = (const uint8_t *) vtcm_f16_act,
                        .wt_raw      = (const uint8_t *) curr_raw,
                        .scratch     = (uint8_t *) vtcm_scratch0,
                        .act_scales  = (const float *) vtcm_scales,
                        .n_rows      = (uint32_t) n_rows,
                        .n_cols      = (uint32_t) n_cols,
                        .k           = (uint32_t) k,
                        .src2        = s2,
                        .src2_stride = src2_stride,
                    };
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_chunk_int_worker, &int_job));
                    hmx_queue_pop(ctx->hmx_queue);

                    float * output_chunk = dst + (mr * dst_stride + nc);
                    int chunk_dst_cols = hex_smin(n_cols, dst_cols - (int) nc);
                    if (chunk_dst_cols > 0) {
                        for (size_t r = 0; r < n_rows; ++r) {
                            float * out_r = out_f32 + r * n_cols;
                            float * dst_r = output_chunk + r * dst_stride;
                            for (size_t c = 0; c < (size_t)chunk_dst_cols; c += 32) {
                                size_t valid_c = hex_smin((size_t)chunk_dst_cols - c, 32);
                                hvx_vec_store_u(dst_r + c, valid_c * sizeof(float), hvx_vmemu(out_r + c));
                            }
                        }
                    }

                    if (nc_next < n) {
                        const size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                        const uint32_t height_next = (n_cols_next / 32) * n_k_tiles;
                        dma_queue_push(weight_dma, dma_make_data(curr_raw, weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                    }
                } else {
                    hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, wt_ptr, vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                    hmx_queue_pop(ctx->hmx_queue);

                    float *output_chunk = dst + (mr * dst_stride + nc);
                    const float *src2_chunk = has_src2 ? (vtcm_src2 + mr * src2_stride + nc) : NULL;
                    int chunk_dst_cols = dst_cols - (int)nc;
                    if (chunk_dst_cols > 0) {
                        if ((op_flags & HTP_OPFLAGS_MM_OUT_F16) != 0) {
                            // fp16 partial output: the tiles are stored as fp16 rows at the fp32 row stride (columns in halves)
                            __fp16 * out16 = (__fp16 *) ((uint8_t *) dst + (size_t) mr * dst_stride * sizeof(float)) + nc;
                            transfer_output_chunk_f16_threaded(ctx, out16, (size_t) dst_stride * sizeof(float), vtcm_output, n_rows, n_cols, chunk_dst_cols, n_threads);
                        } else {
                            transfer_output_chunk_threaded(ctx, output_chunk, src2_chunk, vtcm_output, n_rows, n_cols, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                        }
                    }
                }
            }
        }
    }
    return 0;
}

// SWIGLU epilogue (GGML_HEXAGON_SWIGLU_OUT, htp/hmx-swiglu-tiles.h): gate/up fp16 chunk tiles -> fp16 act tiles.
typedef struct {
    const __fp16 * g;          // gate chunk tiles (VTCM, HMX output layout)
    const __fp16 * u;          // up chunk tiles (VTCM, same layout)
    __fp16 *       out;        // act tiles of the whole output (DDR)
    uint32_t       rt0;        // first chunk-relative row tile of this part
    uint32_t       n_rt;       // row tiles in this part
    uint32_t       n_col_tiles;
    uint32_t       out_row0;   // tile row of the chunk start (mr / 32)
    uint32_t       out_col0;   // tile column of the chunk start (chunk column / 32)
    uint32_t       out_cols;   // tile columns of the whole output (N / 32)
    uint32_t       n_tasks;
    int            barrier;    // bit 0: value barriers on (default; clear = GGML_HEXAGON_LX_FUSE bit 0, HTP_OPFLAGS_SWIGLU_NOBAR); bit 1: fp16 sigmoid (HTP_OPFLAGS_SWIGLU_SIG16)
} swiglu_epi_state_t;

static void swiglu_epi_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    const swiglu_epi_state_t * st = (const swiglu_epi_state_t *) data;
    const uint32_t n_tiles = st->n_rt * st->n_col_tiles;
    const uint32_t t_first = (n_tiles * i) / st->n_tasks;
    const uint32_t t_last  = (n_tiles * (i + 1)) / st->n_tasks;
    for (uint32_t t = t_first; t < t_last; t++) {
        const uint32_t rt  = st->rt0 + t / st->n_col_tiles;
        const uint32_t ct  = t % st->n_col_tiles;
        const size_t   src = ((size_t) rt * st->n_col_tiles + ct) * HTP_MM_HMX_TILE_N_ELMS;
        __fp16 * o = st->out + ((size_t) (st->out_row0 + rt) * st->out_cols + st->out_col0 + ct) * HTP_MM_HMX_TILE_N_ELMS;
        if (st->barrier & 4) {
            swiglu_tile_pair_sig16v2(o, st->g + src, st->u + src);
        } else if (st->barrier & 2) {
            swiglu_tile_pair_sig16(o, st->g + src, st->u + src);
        } else if (st->barrier & 1) {
            swiglu_tile_pair(o, st->g + src, st->u + src);
        } else {
            swiglu_tile_pair_nobar(o, st->g + src, st->u + src);
        }
    }
}

// One part of a chunk pair: part 0 = the first ceil(R / 2) of its R row tiles, part 1 = the remaining ones.
static void swiglu_pair_epi(struct htp_context * ctx, int n_threads, const __fp16 * g, const __fp16 * u, __fp16 * out,
                            uint32_t out_row0, uint32_t n_row_tiles, uint32_t out_col0, uint32_t n_col_tiles, uint32_t out_cols, int part, int barrier) {
    const uint32_t h0 = (n_row_tiles + 1) / 2;
    swiglu_epi_state_t st;
    st.g           = g;
    st.u           = u;
    st.out         = out;
    st.rt0         = (part == 1) ? h0 : 0;
    st.n_rt        = (part == 2) ? n_row_tiles : (part == 1) ? (n_row_tiles - h0) : h0;
    st.n_col_tiles = n_col_tiles;
    st.out_row0    = out_row0;
    st.out_col0    = out_col0;
    st.out_cols    = out_cols;
    st.barrier     = barrier;
    const uint32_t n_tiles = st.n_rt * st.n_col_tiles;
    if (n_tiles == 0) {
        return;
    }
    const uint32_t nt = (n_threads > 0) ? (uint32_t) n_threads : 1u;
    st.n_tasks = (nt < n_tiles) ? nt : n_tiles;
    if (st.n_tasks <= 1) {
        st.n_tasks = 1;
        swiglu_epi_worker(1, 0, &st);
    } else {
        worker_pool_run_func(ctx->worker_pool, swiglu_epi_worker, &st, (int) st.n_tasks);
    }
}

static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    struct htp_context * ctx = octx->ctx;
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const uint32_t n_weights = kparams->n_weights;
    if (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];

    const int weight_type = (int) src0->type;
    const int k           = (int) act->ne[0];
    const int k_valid     = (int) act->ne[0];
    const int m           = (int) (act->ne[1] * act->ne[2] * act->ne[3]);
    const int act_stride  = (int) (act->nb[1] / sizeof(float));
    const dma_addr_t act_dma_addr = act->data;
    const bool act_tiled = (act->flags & HTP_TENSOR_ACT_TILE) != 0;

    if (k % 32 != 0) { return HTP_STATUS_NO_SUPPORT; }
    if ((act_dma_addr & (VLEN - 1)) != 0) { return HTP_STATUS_NO_SUPPORT; }

    size_t row_stride = htp_mm_get_tiled_row_stride(weight_type, k);
    if (row_stride == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    worker_callback_t dequant_worker_fn = NULL;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_0; break;
        case HTP_TYPE_IQ4_NL: dequant_worker_fn = dequantize_tiled_worker_loop_iq4_nl; break;
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q4_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_1; break;
        case HTP_TYPE_MXFP4:  dequant_worker_fn = dequantize_tiled_worker_loop_mxfp4; break;
        case HTP_TYPE_Q8_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q8_0; break;
        case HTP_TYPE_Q5_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q5_k; break;
        case HTP_TYPE_Q6_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q6_k; break;
        case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
        case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
        default:
            return HTP_STATUS_NO_SUPPORT;
    }

    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const bool is_quant       = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);
    const size_t vtcm_budget  = ctx->vtcm_size;
    g_a8w8_dbg = (octx->flags & HTP_OPFLAGS_MM_A8W8) != 0 ? octx->op_params[3] : 0;
    g_a8w8_clip = a8w8_clip_from_params(octx);

    const int m_chunk_n_rows  = kparams->m_chunk;
    const int n_chunk_n_cols  = kparams->n_chunk;
    const int pipeline        = kparams->pipeline;
    const int n_threads       = octx->n_threads;
    const int act_threads     = kparams->n_act_threads;
    const struct fastdiv_values * act_threads_div = &kparams->div_n_act_threads;
    const struct fastdiv_values * k_div           = &kparams->div_ne00_padded;
    const int tile_size       = kparams->tile_size;
    const int aligned_tile_size = kparams->aligned_tile_size;

    const uint32_t dma_dst_stride  = is_quant ? aligned_tile_size : row_stride;
    const uint32_t dma_width_bytes = is_quant ? tile_size : row_stride;

    struct htp_mm_hmx_vtcm_layout L;
    // MM-02: optional bias row of the last weight (fused MUL_MAT+ADD, host try_fuse_nx_bias), read from src[n_weights + 1]
    const bool has_bias = kparams->vtcm_src2_size > 0;
    const struct htp_tensor * restrict bias = has_bias ? octx->src[n_weights + 1] : NULL;
    if (has_bias && (bias == NULL || pipeline == 0)) {
        return HTP_STATUS_NO_SUPPORT;  // the bias is applied by the HMX pipeline epilogue only
    }
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, (size_t) kparams->vtcm_src2_size);

    if (L.total_bytes > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.total_bytes, vtcm_budget, m, k, m_chunk_n_rows, n_chunk_n_cols);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };

    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    float   *vtcm_f32_act    = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    void    *vtcm_scratch0   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]);
    void    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_scratch[1], pipeline);
    void    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    float   *vtcm_src2      = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_src2, has_bias);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    int m_start = 0;
    int m_rows  = m;
    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_can_row_partition(octx->dsts[0], sizeof(float));
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        m_start = (int) range.start;
        m_rows  = (int) range.count;
    }

    if (m_rows == 0) {
        return HTP_STATUS_OK;
    }

    FARF(HIGH, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_start, m_start + m_rows, k, weight_type, m_chunk_n_rows, n_chunk_n_cols, L.total_bytes, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    if (has_bias) {
        // bias row (ssm_dt), once per op: pushed ahead of the weight DMAs on the same FIFO queue; the last weight's output adds it
        const size_t bias_bytes = (size_t) kparams->vtcm_src2_size;
        dma_queue_push(octx->ctx->dma[0], dma_make_data(vtcm_src2, bias->data), hex_align_up(bias_bytes, 128), 0, bias_bytes, 1);
        dma_queue_pop(octx->ctx->dma[0]);
    }

    const size_t mr_end = (size_t)(m_start + m_rows);

    if (pipeline) {
        // Chunk pipeline (hmx_mm_pipe_run). With HTP_OPFLAGS_MM_CROSS_WEIGHT the NX weights of a row block form one chunk
        // sequence (weight p+1's first chunk is fetched during the last HMX job of weight p); otherwise each weight runs alone.
        hmx_mm_pipe_t P = {
            .ctx = ctx, .dma = octx->ctx->dma[0], .is_quant = is_quant, .weight_type = weight_type, .k = k,
            .n_k_tiles = n_k_tiles, .n_k_tiles_div = n_k_tiles_div, .dequant_fn = dequant_worker_fn,
            .row_stride = row_stride, .tile_size = (uint32_t) tile_size,
            .dma_dst_stride = dma_dst_stride, .dma_width_bytes = dma_width_bytes, .n_threads = n_threads,
            .nc_tiles = n_chunk_n_cols / HTP_MM_HMX_TILE_N_COLS, .split = (octx->flags & HTP_OPFLAGS_MM_SPLIT_EDGES) != 0, .c1small = (octx->flags & HTP_OPFLAGS_MM_C1SMALL) != 0,
            .act = vtcm_f16_act, .scales = vtcm_scales,
            .raw = { vtcm_weight_raw[0], vtcm_weight_raw[1] },
            .wgt = { (__fp16 *) vtcm_scratch0, (__fp16 *) vtcm_scratch1 },
            .out = { vtcm_output, (__fp16 *) vtcm_scratch2 },
        };
        hmx_mm_pipe_w_t W[HMX_MM_PIPE_MAX_W];
        int n_w = 0;
        for (uint32_t p = 0; p < n_weights; p++) {
            const struct htp_tensor * restrict src_w = octx->src[p];
            const struct htp_tensor * restrict dst   = octx->dsts[p];
            if (!src_w || !dst) continue;
            const size_t n = src_w->ne[1];
            if (n == 0) continue;
            W[n_w++] = (hmx_mm_pipe_w_t) {
                .pe = (src_w->flags & HTP_TENSOR_A8W8) != 0 && (octx->flags & HTP_OPFLAGS_MM_A8W8) != 0,
                .weight = src_w->data, .weight_stride = src_w->nb[1], .n = n,
                .dst = (float *) dst->data, .dst_stride = dst->nb[1] / sizeof(float), .dst_cols = (int) dst->ne[0],
                .src2 = (has_bias && p + 1 == n_weights) ? vtcm_src2 : NULL, .src2_stride = 0,
            };
        }
        const bool cross = (octx->flags & HTP_OPFLAGS_MM_CROSS_WEIGHT) != 0;

        for (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            struct activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + mr * act_stride * sizeof(float),
                .n_rows = (int) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
            };
            P.n_rows = (int) n_rows;
            P.mr     = mr;
            bool any_pe = false, all_pe = true;
            for (int w = 0; w < n_w; ++w) { any_pe = any_pe || W[w].pe; all_pe = all_pe && W[w].pe; }
            P.a8w8   = (octx->flags & HTP_OPFLAGS_MM_A8W8) != 0 && weight_type == HTP_TYPE_Q4_0 && n_rows >= 128 && (all_pe || !any_pe) &&
                       hmx_mm_a8w8_fits((uint32_t) k, (size_t) m_chunk_n_rows, (size_t) (n_chunk_n_cols / HTP_MM_HMX_TILE_N_COLS), all_pe);
            if (P.a8w8) {
                // NX weights run as one chunk sequence (cross-weight prefetch) when CROSS_WEIGHT is set, else one weight at a time
                dma_addr_t asrc = 0; size_t abytes = 0;
                if (act_tiled) {
                    const size_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                    asrc   = act_dma_addr + (mr / HTP_MM_HMX_TILE_N_ROWS) * (k / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_SIZE;
                    abytes = (n_rows_padded / HTP_MM_HMX_TILE_N_ROWS) * (k / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_SIZE;
                } else {
                    transfer_activation_chunk_threaded(&act_params);
                }
                // pre-expanded weights live behind the tiled Q4_0 data of their tensor: point the descriptors at the blob for this path only
                hmx_mm_pipe_w_t Wb[HMX_MM_PIPE_MAX_W];
                for (int w = 0; w < n_w; w++) {
                    Wb[w] = W[w];
                    if (Wb[w].pe) { Wb[w].weight += hex_align_up((size_t) octx->src[w]->nb[3] * octx->src[w]->ne[3], 128); }
                }
                if (cross) {
                    hmx_mm_pipe_run_a8w8(&P, Wb, n_w, asrc, abytes, false);
                } else {
                    hmx_mm_pipe_run_a8w8(&P, &Wb[0], 1, asrc, abytes, false);
                    for (int w = 1; w < n_w; w++) {
                        hmx_mm_pipe_run_a8w8(&P, &Wb[w], 1, 0, 0, true);
                    }
                }
                continue;
            }
            int ov_rows[HMX_ACT_OVERLAP_MAX_GROUPS];
            const int ov_n = (act_tiled && (octx->flags & HTP_OPFLAGS_MM_ACT_OVERLAP) != 0) ? hmx_act_overlap_plan(n_rows, ov_rows) : 0;
            if (ov_n > 1) {
                // activation DMA'd in row groups inside the first chunk of the first weight (hmx_mm_pipe_run)
                hmx_mm_act_stage_t stage = {
                    .n_groups = ov_n,
                    .tiles    = act_dma_addr + (mr / HTP_MM_HMX_TILE_N_ROWS) * (k / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_SIZE,
                    .k_tiles  = (size_t) (k / HTP_MM_HMX_TILE_N_COLS),
                };
                for (int r = 0; r < ov_n; ++r) {
                    stage.group_rows[r] = ov_rows[r];
                }
                if (cross) {
                    hmx_mm_pipe_run(&P, W, n_w, &stage);
                } else {
                    hmx_mm_pipe_run(&P, &W[0], 1, &stage);
                    for (int w = 1; w < n_w; w++) {
                        hmx_mm_pipe_run(&P, &W[w], 1, NULL);
                    }
                }
            } else {
                if (act_tiled) {
                    const size_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                    const size_t tile_off      = (mr / HTP_MM_HMX_TILE_N_ROWS) * (k / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_SIZE;
                    act_tiles_dma_push(octx->ctx->dma[0], vtcm_f16_act, act_dma_addr + tile_off,
                                       (n_rows_padded / HTP_MM_HMX_TILE_N_ROWS) * (k / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_SIZE);
                    dma_queue_pop(octx->ctx->dma[0]);
                } else {
                    transfer_activation_chunk_threaded(&act_params);
                }
                if (cross) {
                    hmx_mm_pipe_run(&P, W, n_w, NULL);
                } else {
                    for (int w = 0; w < n_w; w++) {
                        hmx_mm_pipe_run(&P, &W[w], 1, NULL);
                    }
                }
            }
        }
    } else {
        hmx_matmul_job_t job;
        for (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            struct activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + mr * act_stride * sizeof(float),
                .n_rows = (int) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
            };
            if (act_tiled) {
                const size_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                const size_t tile_off      = (mr / HTP_MM_HMX_TILE_N_ROWS) * (k / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_SIZE;
                act_tiles_dma_push(octx->ctx->dma[0], vtcm_f16_act, act_dma_addr + tile_off,
                                   (n_rows_padded / HTP_MM_HMX_TILE_N_ROWS) * (k / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_SIZE);
                dma_queue_pop(octx->ctx->dma[0]);
            } else {
                transfer_activation_chunk_threaded(&act_params);
            }

            for (uint32_t p = 0; p < n_weights; p++) {
                const struct htp_tensor * restrict src_w = octx->src[p];
                const struct htp_tensor * restrict dst   = octx->dsts[p];
                if (!src_w || !dst) continue;

                const dma_addr_t weight       = src_w->data;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                float * dst_ptr            = (float *) dst->data;
                const size_t n             = src_w->ne[1];
                if (n == 0) continue;
                const size_t weight_stride = src_w->nb[1];
                const size_t dst_stride    = dst->nb[1] / sizeof(float);
                const int dst_cols         = (int) dst->ne[0];

                const uint32_t dma_src_stride = is_quant ? tile_size : weight_stride;

                if (n > 0) {
                    const size_t n_cols = hex_smin(n, n_chunk_n_cols);
                    const uint32_t height = is_quant ? (n_cols / 32) * n_k_tiles : n_cols;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
                }

                for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
                    const size_t n_cols = hex_smin(n - nc, n_chunk_n_cols);
                    const size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                    const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                    void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;

                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_scratch0, curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                    const size_t nc_next = nc + n_chunk_n_cols;
                    if (nc_next < n) {
                        const size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                        const uint32_t height_next = is_quant ? (n_cols_next / 32) * n_k_tiles : n_cols_next;
                        dma_queue_push(weight_dma, dma_make_data(curr_raw, weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                    }

                    hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, vtcm_scratch0, vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                    hmx_queue_pop(ctx->hmx_queue);

                    float *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    int chunk_dst_cols = dst_cols - (int)nc;
                    if (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, NULL, vtcm_output, n_rows, n_cols, dst_stride, 0, chunk_dst_cols, n_threads);
                    }
                }
            }
        }
    }

    return HTP_STATUS_OK;
}

// ImplSwigluOut: gate/up HMX NX with fused SWIGLU output (GGML_HEXAGON_SWIGLU_OUT; host: try_fuse_swiglu_out). gate_i / up_i stay valid until jobs 2i+4 / 2i+5 overwrite their slots, which is after both parts. Chunk pair i (columns [i*nc, (i+1)*nc)) is HMX jobs 2i (gate) and 2i+1 (up).
// Each loop step is one dequantization plus one part on the HVX threads while one HMX job runs. src[0] = gate weight, src[1] = up weight, src[2] = activation; dsts[0] = fp16 act tiles [N x M] in the htp/hmx-act-tiles.h layout, read by the FFN down (HTP_TENSOR_ACT_TILE). The epilogue of pair i runs in two parts: part 0 right after up_i is popped, part 1 right after gate_{i+1} is popped (the last pair runs both parts in the tail).
static inline void swiglu_nx_push_chunk(dma_queue * q, void * raw_dst, const struct htp_tensor * w, size_t nc0, size_t n_cols,
                                        int n_k_tiles, int tile_size, int aligned_tile_size) {
    uint32_t height = (uint32_t) ((n_cols / HTP_MM_HMX_TILE_N_ROWS) * (size_t) n_k_tiles);
    if (g_a8w8_dbg & 1) { height = (height + 1) / 2; }   // timing probe (wrong results): half the weight bytes
    const uint32_t row_bytes = (g_a8w8_dbg & (1u << 31)) ? 1024u : (uint32_t) tile_size;   // probe: int8-sized weight reads (overlapping garbage writes)
    dma_queue_push(q, dma_make_data(raw_dst, w->data + nc0 * w->nb[1]), (uint32_t) aligned_tile_size, (uint32_t) tile_size,
                   row_bytes, height);
}

// slots: gate[0], gate[1], up[0], up[1] chunk outputs; pair i uses slots i & 1 (gate) and 2 + (i & 1) (up).
static void swiglu_nx_pair_epi(struct htp_context * ctx, int n_threads, __fp16 * const * slots, __fp16 * out,
                               uint32_t out_row0, uint32_t n_row_tiles, uint32_t pair, uint32_t n_col_tiles, uint32_t out_cols, int part, int barrier) {
    swiglu_pair_epi(ctx, n_threads, slots[pair & 1], slots[2 + (pair & 1)], out, out_row0, n_row_tiles,
                    pair * n_col_tiles, n_col_tiles, out_cols, part, barrier);
}


// Pre-expanded chunk push for the SWIGLU kernel: columns [nc0, nc0 + n_cols) of weight w (blob behind the tiled Q4_0 data) into dst.
static inline void swiglu_nx_push_pe(dma_queue * q, void * dst, const struct htp_tensor * w, size_t nc0, size_t n_cols, int n_k_tiles) {
    const size_t ct_bytes = (size_t) n_k_tiles * 1024 + HTP_A8W8_PE_TAIL;
    const size_t blob     = hex_align_up((size_t) w->nb[3] * w->ne[3], 128);
    const size_t nct      = n_cols / HTP_MM_HMX_TILE_N_ROWS;
    dma_queue_push(q, dma_make_data(dst, w->data + blob + (nc0 / HTP_MM_HMX_TILE_N_ROWS) * ct_bytes), ct_bytes * nct, ct_bytes * nct, ct_bytes * nct, 1);
}

static int hmx_mm_nx_swiglu_2d(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    struct htp_context * ctx = octx->ctx;
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_tensor * restrict src_g = octx->src[0];
    const struct htp_tensor * restrict src_u = octx->src[1];
    const struct htp_tensor * restrict act   = octx->src[2];
    const struct htp_tensor * restrict dst   = octx->dsts[0];
    if (!src_g || !src_u || !act || !dst) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const int weight_type = (int) src_g->type;
    const int k          = (int) act->ne[0];
    const int m          = (int) (act->ne[1] * act->ne[2] * act->ne[3]);
    const int act_stride = (int) (act->nb[1] / sizeof(float));
    const dma_addr_t act_dma_addr = act->data;
    const bool act_tiled = (act->flags & HTP_TENSOR_ACT_TILE) != 0;

    if (weight_type != HTP_TYPE_Q4_0 || src_u->type != weight_type || src_u->ne[1] != src_g->ne[1]) {
        return HTP_STATUS_NO_SUPPORT;
    }
    // mdev.count > 1 would row-partition this op; the host only fuses single-session row layouts (see try_fuse_swiglu_out).
    if (k % 32 != 0 || (act_dma_addr & (VLEN - 1)) != 0 || ctx->mdev.count > 1) {
        return HTP_STATUS_NO_SUPPORT;
    }
    const size_t row_stride = htp_mm_get_tiled_row_stride(weight_type, k);
    if (row_stride == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);
    const worker_callback_t dequant_worker_fn = dequantize_tiled_worker_loop_q4_0;
    const size_t vtcm_budget = ctx->vtcm_size;

    const int m_chunk_n_rows    = kparams->m_chunk;
    const int n_chunk_n_cols    = kparams->n_chunk;
    const int n_threads         = octx->n_threads;
    const int barrier         = ((octx->flags & HTP_OPFLAGS_SWIGLU_NOBAR) == 0 ? 1 : 0) |
                                ((octx->flags & HTP_OPFLAGS_SWIGLU_SIG16) != 0 ? 2 : 0);  // LX_FUSE bit 0 clears the value barriers; bit 1 = fp16 sigmoid
    const int act_threads       = kparams->n_act_threads;
    const struct fastdiv_values * act_threads_div = &kparams->div_n_act_threads;
    const struct fastdiv_values * k_div           = &kparams->div_ne00_padded;
    const int tile_size         = kparams->tile_size;
    const int aligned_tile_size = kparams->aligned_tile_size;
    const size_t n = (size_t) src_g->ne[1];

    if (n_chunk_n_cols <= 0 || n_chunk_n_cols > HTP_MM_SWIGLU_NC || n_chunk_n_cols % HTP_MM_HMX_TILE_N_COLS != 0 ||
        n % (size_t) n_chunk_n_cols != 0 || (size_t) dst->ne[0] != n ||
        m_chunk_n_rows <= 0 || m_chunk_n_rows % HTP_MM_HMX_TILE_N_ROWS != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    struct htp_mm_swiglu_vtcm_layout L;
    htp_mm_swiglu_vtcm_layout_build(&L, (size_t) m_chunk_n_rows, (size_t) n_chunk_n_cols, (uint32_t) k,
                                    (uint32_t) aligned_tile_size, (uint32_t) act_threads);
    if (L.total_bytes > vtcm_budget) {
        FARF(ERROR, "hmx-mm-swiglu: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.total_bytes, vtcm_budget, m, k, m_chunk_n_rows, n_chunk_n_cols);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    void    * vtcm_weight_raw[2] = { VTCM_LAYOUT_PTR(void, base, L.off_weight[0]), VTCM_LAYOUT_PTR(void, base, L.off_weight[1]) };
    __fp16  * vtcm_f16_act       = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    float   * vtcm_f32_act       = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    __fp16  * vtcm_wbuf[2]       = { VTCM_LAYOUT_PTR(__fp16, base, L.off_scratch[0]), VTCM_LAYOUT_PTR(__fp16, base, L.off_scratch[1]) };
    __fp16  * vtcm_out[4]        = { VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]), VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[1]),
                                     VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[2]), VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[3]) };
    __fp16  * vtcm_scales        = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16
    const bool a8w8 = (octx->flags & HTP_OPFLAGS_MM_A8W8) != 0;
    bool a8w8_pe = a8w8 && (src_g->flags & HTP_TENSOR_A8W8) != 0 && (src_u->flags & HTP_TENSOR_A8W8) != 0 && n_chunk_n_cols % 32 == 0;
    if (a8w8_pe) {
        const size_t scratch_ = (size_t) n_chunk_n_cols * k * sizeof(uint16_t);
        const size_t nb64_    = ((size_t) m_chunk_n_rows + 63) / 64;
        const size_t pe_w_    = hex_align_up((size_t) (n_chunk_n_cols / 32) * ((size_t) (k / 32) * 1024 + HTP_A8W8_PE_TAIL), 2048);
        const size_t pe_a0_   = nb64_ * (size_t) (k / 32) * 2048 + nb64_ * 256 + 256;
        const size_t pe_a1_   = nb64_ * (size_t) (n_chunk_n_cols / 32) * 8192 + 2048 + 4 * (size_t) (n_chunk_n_cols / 32) * 32 * 4 + 256;
        a8w8_pe = pe_w_ + pe_a0_ <= scratch_ && pe_w_ + pe_a1_ <= scratch_;
        if (!a8w8_pe) { FARF(ERROR, "a8w8 gate/up: pre-expanded layout does not fit (w %zu a0 %zu a1 %zu scratch %zu): falling back", pe_w_, pe_a0_, pe_a1_, scratch_); }
    }
    g_a8w8_dbg = a8w8 ? octx->op_params[3] : 0;
    g_a8w8_clip = a8w8_clip_from_params(octx);
    (void) vtcm_f32_act;
    // A8W8 scratch (all in VTCM, carved from the fp32 activation staging area and the fp16 weight scratch that the dequant path does not use): act8 (u8 64-row tiles) + rscale : live for the whole row block -> in vtcm_f32_act (staging is free once the fp16 tiles are in place) w8 tiles + sw + cs of the chunk : vtcm_wbuf[p] (int8 tiles are 1/2 of the fp16 scratch) plane ring + bias tables : vtcm_scales tile and vtcm_out slots (planes are consumed before the slot is reused)
    uint32_t * a8_tbl = (uint32_t *) vtcm_scales;                 // 4 x 256 B tables (the scales tile is 2 KB)
    if (a8w8) {
        static const uint32_t ee1x[4] = { 24, 16, 8, 0 };
        for (int t = 0; t < 4; ++t) for (int q = 0; q < 64; ++q) a8_tbl[t * 64 + q] = (q < 32) ? (ee1x[t] << 10) : 0;
        if (act_tiled == false) {
            return HTP_STATUS_NO_SUPPORT;   // the int path quantises from fp16 act tiles; fp32 activations keep the fp16 path
        }
        {
            const size_t scratch = (size_t) n_chunk_n_cols * k * sizeof(uint16_t);
            const size_t nb64 = ((size_t) m_chunk_n_rows + 63) / 64;
            const size_t need_w = 2 * (size_t) (n_chunk_n_cols / 32) * (k / 32) * 1024;
            const size_t need_a = nb64 * (size_t) (k / 32) * 2048 + nb64 * 64 * 4 + 2048 + nb64 * (size_t) (n_chunk_n_cols / 32) * 8192 + 4 * (size_t) (n_chunk_n_cols / 32) * 32 * 4 + 2048;
            const size_t pe_w   = hex_align_up((size_t) (n_chunk_n_cols / 32) * ((size_t) (k / 32) * 1024 + HTP_A8W8_PE_TAIL), 2048);
            const size_t pe_a0  = nb64 * (size_t) (k / 32) * 2048 + nb64 * 256 + 256;                                   // act8 | rscale behind parity-0 blob
            const size_t pe_a1  = nb64 * (size_t) (n_chunk_n_cols / 32) * 8192 + 2048 + 4 * (size_t) (n_chunk_n_cols / 32) * 32 * 4 + 256;   // ring 0 | sw | cs behind parity-1 blob
            (void) pe_w; (void) pe_a0; (void) pe_a1;
            if (a8w8_pe) {
                if (nb64 * (size_t) (n_chunk_n_cols / 32) * 8192 + 2048 > (size_t) hex_align_up((size_t) m_chunk_n_rows, 32) * (size_t) k * 2) {
                    return HTP_STATUS_VTCM_TOO_SMALL;
                }
            } else
            if (need_w > scratch || need_a > scratch || nb64 * (size_t) (n_chunk_n_cols / 32) * 8192 + 2048 > (size_t) hex_align_up((size_t) m_chunk_n_rows, 32) * (size_t) k * 2) {
                return HTP_STATUS_VTCM_TOO_SMALL;
            }
        }
    }

    __fp16 *       out_tiles   = (__fp16 *) dst->data;
    const uint32_t n_col_tiles = (uint32_t) (n_chunk_n_cols / HTP_MM_HMX_TILE_N_COLS);
    const uint32_t out_cols    = (uint32_t) (n / HTP_MM_HMX_TILE_N_COLS);
    const uint32_t tile_k      = (uint32_t) (k / HTP_MM_HMX_TILE_N_ROWS);
    const int      n_chunk_cnt = (int) (n / (size_t) n_chunk_n_cols);
    const int      n_jobs      = 2 * n_chunk_cnt;
    dma_queue *    weight_dma  = octx->ctx->dma[0];
    hmx_matmul_job_t job_slots[2];
    hmx_matmul_job_t grp_job[2];   // chunk-0 row-group jobs (act overlap), kept apart from job_slots

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    for (size_t mr = 0; mr < (size_t) m; mr += (size_t) m_chunk_n_rows) {
        const size_t   n_rows      = hex_smin((size_t) m - mr, (size_t) m_chunk_n_rows);
        const uint32_t n_row_tiles = (uint32_t) hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
        const uint32_t out_row0    = (uint32_t) (mr / HTP_MM_HMX_TILE_N_ROWS);

        struct activation_transfer_params act_params = {
            .ctx                = ctx,
            .dst                = vtcm_f16_act,
            .act_dma_addr       = act_dma_addr + mr * act_stride * sizeof(float),
            .n_rows             = (int) n_rows,
            .k_block            = k,
            .k_stride           = act_stride,
            .n_threads          = act_threads,
            .act_threads_div    = act_threads_div,
            .k_div              = k_div,
            .k_valid            = k,
            .vtcm_f32_act       = vtcm_f32_act,
            .vtcm_f32_act_bytes = L.act_f32_bytes,
        };
        // First-chunk activation overlap (HTP_OPFLAGS_MM_ACT_OVERLAP, fp16-tile producers): the activation is DMA'd in row groups queued behind the gate chunk 0 weight, and chunk 0 runs as one HMX job per group (j == 0 below): the first job waits for one group, not the whole block.
        // Tiles keep their layout, so every output tile sees the same K order.
        bool early_w0 = false;   // A8W8: gate chunk 0 already sits in vtcm_weight_raw[0] (MM_EARLY_W)
        int ov_rows[HMX_ACT_OVERLAP_MAX_GROUPS];
        const int ov_n = (!a8w8 && act_tiled && (octx->flags & HTP_OPFLAGS_MM_ACT_OVERLAP) != 0) ? hmx_act_overlap_plan(n_rows, ov_rows) : 0;
        if (ov_n > 1) {
            hmx_mm_act_stage_t stage = {
                .n_groups = ov_n,
                .tiles    = act_dma_addr + (mr / HTP_MM_HMX_TILE_N_ROWS) * (size_t) n_k_tiles * HTP_MM_HMX_TILE_SIZE,
                .k_tiles  = (size_t) n_k_tiles,
            };
            if (ctx->mm_early_pending) {
                ctx->mm_early_pending = false;   // gate chunk 0 is already in flight (htp_mm_early_weight_dma)
            } else {
                swiglu_nx_push_chunk(weight_dma, vtcm_weight_raw[0], octx->src[0], 0, (size_t) n_chunk_n_cols,
                                     n_k_tiles, tile_size, aligned_tile_size);
            }
            size_t row0 = 0;
            for (int r = 0; r < ov_n; ++r) {
                hmx_act_group_dma(&stage, weight_dma, vtcm_f16_act, row0, (size_t) ov_rows[r]);
                row0 += (size_t) ov_rows[r];
            }
            swiglu_nx_push_chunk(weight_dma, vtcm_weight_raw[1], octx->src[1], 0, (size_t) n_chunk_n_cols,
                                 n_k_tiles, tile_size, aligned_tile_size);
        } else {
            if (a8w8 && ctx->mm_early_pending && !a8w8_pe) {
                // MM_EARLY_W pushed gate chunk 0 into vtcm_weight_raw[0] ahead of everything: it is the first queue entry (popped in the A8W8 block below)
                ctx->mm_early_pending = false;
                early_w0 = true;
            }
            if (a8w8 && !act_tiled) {
                return HTP_STATUS_NO_SUPPORT;
            }
            if (a8w8) {
                // the A8W8 block below issues the activation DMA itself so that the first weight expand overlaps it
            } else if (act_tiled) {
                const size_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                const size_t tile_off      = (mr / HTP_MM_HMX_TILE_N_ROWS) * (k / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_SIZE;
                act_tiles_dma_push(weight_dma, vtcm_f16_act, act_dma_addr + tile_off,
                                   (n_rows_padded / HTP_MM_HMX_TILE_N_ROWS) * (k / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_SIZE);
                dma_queue_pop(weight_dma);
            } else {
                transfer_activation_chunk_threaded(&act_params);
            }

            // Raw DMA ring: job j lands in vtcm_weight_raw[j & 1]; job j+2 reuses that buffer once job j has been popped.
            for (int j = 0; !a8w8 && j < 2 && j < n_jobs; j++) {
                const size_t nc_j = (size_t) (j >> 1) * (size_t) n_chunk_n_cols;
                swiglu_nx_push_chunk(weight_dma, vtcm_weight_raw[j & 1], octx->src[j & 1], nc_j,
                                     hex_smin(n - nc_j, (size_t) n_chunk_n_cols), n_k_tiles, tile_size, aligned_tile_size);
            }
        }

        if (a8w8) {
            // ---- A8W8 path for this row block (act fp16 tiles are in vtcm_f16_act).
            // VTCM carving: vtcm_wbuf[0]: two int8 weight tile sets (chunk parity) vtcm_wbuf[1]: act8 | rscale | plane ring (all slots of one chunk) | sw, cs per parity vtcm_out[2 * p + (i & 1)]: fp16 chunk outputs written by the post-process, consumed by the SWIGLU epilogue (unchanged) Pipeline per chunk j: HMX(j) runs while the workers expand chunk j+1 and post-process chunk j; HMX(j+1) is pushed as soon as both are done and the SWIGLU epilogue of a finished (gate, up) pair overlaps it.
            const uint32_t n_blocks64 = (n_row_tiles + 1) / 2;
            const uint32_t nct        = (uint32_t) (n_chunk_n_cols / HTP_MM_HMX_TILE_N_COLS);
            const uint32_t ring       = n_blocks64 * nct;
            // pre-expanded weights (HTP_TENSOR_A8W8 on both FFN weights): the chunk blob (nct * (tile_k * 1024 + PE_TAIL)) is DMA'd straight into the int8 buffer of its
            // parity: parity 0 at the start of vtcm_wbuf[0] (act8 | rscale behind it), parity 1 at the start of vtcm_wbuf[1] (plane ring 0 | sw | cs behind it).
            const bool pe = a8w8_pe;
            const size_t w8_bytes = pe ? hex_align_up((size_t) nct * (tile_k * 1024 + HTP_A8W8_PE_TAIL), 2048) : (size_t) nct * tile_k * 1024;
            uint8_t * act8   = pe ? (uint8_t *) vtcm_wbuf[0] + w8_bytes : (uint8_t *) vtcm_wbuf[1];
            float *   rscale = (float *) (act8 + (size_t) n_blocks64 * tile_k * 2048);
            uint8_t * planes_par[2];
            float *   sw_par; int32_t * cs_par;
            if (pe) {
                planes_par[0] = (uint8_t *) vtcm_wbuf[1] + w8_bytes;
                planes_par[1] = (uint8_t *) vtcm_f16_act;
                sw_par = (float *) (planes_par[0] + (size_t) ring * 8192 + 2048);
                cs_par = (int32_t *) (sw_par + 2 * nct * 32);
            } else {
                planes_par[0] = (uint8_t *) hex_align_up((size_t) (rscale + n_blocks64 * 64), 2048);
                planes_par[1] = (uint8_t *) vtcm_f16_act;   // [1]: the fp16 act tiles are dead after the quantiser
                sw_par = (float *) (planes_par[0] + (size_t) ring * 8192 + 2048);   // [2][nct * 32]; +2048: dump slot of the HMX job's bank cleanup
                cs_par = (int32_t *) (sw_par + 2 * nct * 32);                       // [2][nct * 32]
            }
            uint8_t * w8_par[2] = { (uint8_t *) vtcm_wbuf[0], pe ? (uint8_t *) vtcm_wbuf[1] : (uint8_t *) vtcm_wbuf[0] + w8_bytes };
            const uint32_t wstride = pe ? tile_k * 1024 + HTP_A8W8_PE_TAIL : 0;
            volatile int * produced_par[2] = { (volatile int *) ((uint8_t *) vtcm_scales + 1024), (volatile int *) ((uint8_t *) vtcm_scales + 1024 + 128) };

            a8w8_state_t st = { .act16 = vtcm_f16_act, .act8 = act8, .rscale = rscale, .tile_k = tile_k, .n_blocks64 = n_blocks64, .n_row_tiles = n_row_tiles,
                                .traces = ctx->trace };
            st.n_tasks = (uint32_t) n_threads < n_blocks64 * 16 ? (uint32_t) n_threads : n_blocks64 * 16;
            // Queue order: [gate chunk 0 (early or pushed here)], activation tiles, up chunk 0. Expanding gate chunk 0 does not need the activations, so it
            // runs while the activation DMA is in flight; the quantiser follows.
            if (a8w8_pe) {
                swiglu_nx_push_pe(weight_dma, w8_par[0], octx->src[0], 0, (size_t) n_chunk_n_cols, n_k_tiles);
            } else if (!early_w0) {
                swiglu_nx_push_chunk(weight_dma, vtcm_weight_raw[0], octx->src[0], 0, (size_t) n_chunk_n_cols, n_k_tiles, tile_size, aligned_tile_size);
            }
            // activation tiles in groups of two 64-row blocks (each quantised as soon as it lands; HMX(0) starts after the first group)
            const bool     q_groups = n_blocks64 >= 2;
            const uint32_t n_groups = q_groups ? (n_blocks64 + 1) / 2 : 1;
            {
                const size_t tile_off   = (mr / HTP_MM_HMX_TILE_N_ROWS) * (k / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_SIZE;
                const size_t tile_bytes = (size_t) tile_k * HTP_MM_HMX_TILE_SIZE;
                for (uint32_t gi = 0; gi < n_groups; ++gi) {
                    const uint32_t rt0 = q_groups ? gi * 4 : 0, rt1 = q_groups ? ((rt0 + 4 < n_row_tiles) ? rt0 + 4 : n_row_tiles) : n_row_tiles;
                    act_tiles_dma_push(weight_dma, (uint8_t *) vtcm_f16_act + (size_t) rt0 * tile_bytes, act_dma_addr + tile_off + (size_t) rt0 * tile_bytes,
                                       (size_t) (rt1 - rt0) * tile_bytes);
                }
            }
            if (n_jobs > 1) {
                if (a8w8_pe) {
                    swiglu_nx_push_pe(weight_dma, w8_par[1], octx->src[1], 0, (size_t) n_chunk_n_cols, n_k_tiles);
                } else {
                    swiglu_nx_push_chunk(weight_dma, vtcm_weight_raw[1], octx->src[1], 0, (size_t) n_chunk_n_cols, n_k_tiles, tile_size, aligned_tile_size);
                }
            }
            a8w8_job_t jobs[2];
            a8w8_post_state_t ps;
            struct htp_thread_trace * tr0 = &ctx->trace[0];
            // chunk 0: expand (the activation DMA is still running), then the activation quantiser
            void * raw_next = (void *) dma_queue_pop(weight_dma).dst;
            st.raw = (const uint8_t *) raw_next; st.aligned = (uint32_t) aligned_tile_size; st.n_col_tiles = nct;
            st.w8 = w8_par[0]; st.sw = sw_par; st.cs = cs_par;
            st.n_tasks = (uint32_t) n_threads < nct ? (uint32_t) n_threads : nct;
            if (a8w8_pe) {
                st.n_tasks = 1; a8w8_pe_unpack_worker(1, 0, &st);      // sw / cs tails of the landed blob
            } else {
                worker_pool_run_func(ctx->worker_pool, a8w8_expand_worker, &st, (int) st.n_tasks);
            }
            if (n_jobs > 2 && !a8w8_pe) {
                swiglu_nx_push_chunk(weight_dma, raw_next, octx->src[0], (size_t) 1 * n_chunk_n_cols,
                                     hex_smin(n - (size_t) n_chunk_n_cols, (size_t) n_chunk_n_cols), n_k_tiles, tile_size, aligned_tile_size);
            }
            volatile int * qready = (volatile int *) ((uint8_t *) vtcm_scales + 1024 + 256);
            *produced_par[0] = 0;
            *qready = 0;
            jobs[0] = (a8w8_job_t) { .act8 = act8, .w8 = w8_par[0], .planes = planes_par[0], .tbl = a8_tbl, .tile_k = tile_k, .n_blocks64 = n_blocks64,
                                     .n_col_tiles = nct, .ring = ring, .produced = produced_par[0], .qready = qready, .wstride = wstride };
            for (uint32_t gi = 0; gi < n_groups; ++gi) {
                dma_queue_pop(weight_dma);                                 // activation group landed
                st.qb0 = q_groups ? gi * 2 : 0;
                st.qnb = q_groups ? ((gi * 2 + 2 <= n_blocks64) ? 2 : 1) : n_blocks64;
                st.n_tasks = (uint32_t) n_threads < st.qnb * 16 ? (uint32_t) n_threads : st.qnb * 16;
                htp_trace_event_start(&ctx->trace[0], HTP_TRACE_EVT_HVX_A_QUANT, gi);
                worker_pool_run_func(ctx->worker_pool, a8w8_quant_worker, &st, (int) st.n_tasks);
                htp_trace_event_stop(&ctx->trace[0], HTP_TRACE_EVT_HVX_A_QUANT, gi);
                *qready = (int) (st.qb0 + st.qnb);
                if (gi == 0) {
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(a8w8_hmx_job, &jobs[0]));
                }
            }
            st.qb0 = 0; st.qnb = 0;

            // Per chunk j (b = j & 1): expand(j+1) -> push HMX(j+1) (its own plane ring) -> post(j) behind HMX(j) -> pop HMX(j). The HMX unit thus goes straight
            // from job j to job j+1 while the workers still post-process chunk j.
            for (int j = 0; j < n_jobs; j++) {
                const int p = j & 1, i = j >> 1, b = j & 1;
                if (j + 1 < n_jobs) {
                    raw_next = (void *) dma_queue_pop(weight_dma).dst;
                    st.raw = (const uint8_t *) raw_next; st.w8 = w8_par[b ^ 1]; st.sw = sw_par + (b ^ 1) * nct * 32; st.cs = cs_par + (b ^ 1) * nct * 32;
                    st.n_tasks = (uint32_t) n_threads < nct ? (uint32_t) n_threads : nct;
                    if (a8w8_pe) {
                        st.n_tasks = 1; a8w8_pe_unpack_worker(1, 0, &st);
                    } else {
                        worker_pool_run_func(ctx->worker_pool, a8w8_expand_worker, &st, (int) st.n_tasks);
                        if (j + 3 < n_jobs) {
                            const size_t nc_n = (size_t) ((j + 3) >> 1) * (size_t) n_chunk_n_cols;
                            swiglu_nx_push_chunk(weight_dma, raw_next, octx->src[(j + 3) & 1], nc_n,
                                                 hex_smin(n - nc_n, (size_t) n_chunk_n_cols), n_k_tiles, tile_size, aligned_tile_size);
                        }
                    }
                    *produced_par[b ^ 1] = 0;
                    jobs[b ^ 1] = (a8w8_job_t) { .act8 = act8, .w8 = w8_par[b ^ 1], .planes = planes_par[b ^ 1], .tbl = a8_tbl, .tile_k = tile_k, .n_blocks64 = n_blocks64,
                                                 .n_col_tiles = nct, .ring = ring, .produced = produced_par[b ^ 1], .wstride = wstride };
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(a8w8_hmx_job, &jobs[b ^ 1]));
                }
                // SWIGLU epilogue: the fused up post is kept only for the last pair (its tail is exposed anyway); for the others the epilogue of pair i-1 runs
                // behind the push of HMX(j+1) (gate_i job), so the expand + epilogue of the up post no longer sit between two HMX jobs.
                const bool defer_swiglu = (g_a8w8_dbg & 4) != 0 && j < n_jobs - 1;   // opt-in (A8W8_DBG=4): measured neutral
                if (defer_swiglu && p == 0 && i >= 1) {
                    swiglu_pair_epi(ctx, n_threads, vtcm_out[(i - 1) & 1], vtcm_out[2 + ((i - 1) & 1)], out_tiles, out_row0, n_row_tiles,
                                    (uint32_t) (i - 1) * n_col_tiles, n_col_tiles, out_cols, 2, (barrier & 2) ? 4 : barrier);
                }
                ps = (a8w8_post_state_t) { .planes = planes_par[b], .sw = sw_par + b * nct * 32, .cs = cs_par + b * nct * 32, .rscale = rscale,
                                           .out = vtcm_out[2 * p + (i & 1)], .n_col_tiles = nct, .n_blocks64 = n_blocks64, .n_row_tiles = n_row_tiles,
                                           .ring = ring, .n_tasks = (uint32_t) n_threads, .traces = ctx->trace, .produced = produced_par[b],
                                           .epi_g = (p == 1 && !defer_swiglu) ? vtcm_out[i & 1] : NULL, .epi_out = out_tiles, .epi_row0 = out_row0,
                                           .epi_col0 = (uint32_t) i * n_col_tiles, .epi_cols = out_cols, .epi_barrier = barrier };
                htp_trace_event_start(tr0, HTP_TRACE_EVT_INIT, 3);
                worker_pool_run_func(ctx->worker_pool, a8w8_post_worker, &ps, n_threads);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_INIT, 3);
                hmx_queue_pop(ctx->hmx_queue);                      // HMX(j) complete (all its slots were posted)
                if (a8w8_pe && j + 2 < n_jobs) {
                    // the int8 buffer of parity b is free again: fetch chunk j+2 (weight src[j & 1], column chunk (j+2)>>1) into it
                    const size_t nc_n = (size_t) ((j + 2) >> 1) * (size_t) n_chunk_n_cols;
                    swiglu_nx_push_pe(weight_dma, w8_par[b], octx->src[(j + 2) & 1], nc_n, hex_smin(n - nc_n, (size_t) n_chunk_n_cols), n_k_tiles);
                }
            }
            continue;   // next row block
        }

        // last pair's up job in two row groups (tail_g > 1 only when the block has at least two row tiles)
        const uint32_t tail_gr   = (n_row_tiles + 1) / 2;
        const uint32_t tail_g    = (n_row_tiles + tail_gr - 1) / tail_gr;
        int            tail_done = 0;
        for (int j = 0; j < n_jobs; j++) {
            const int    p      = j & 1;   // 0 gate, 1 up
            const int    i      = j >> 1;  // chunk pair
            const size_t nc_i   = (size_t) i * (size_t) n_chunk_n_cols;
            const size_t n_cols = hex_smin(n - nc_i, (size_t) n_chunk_n_cols);

            void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;
            dequantize_tiled_weight_chunk_to_fp16_tiles(ctx, vtcm_wbuf[p], curr_raw, (int) n_cols, k, row_stride, weight_type,
                                                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);
            if (j + 2 < n_jobs) {
                const size_t nc_n = (size_t) ((j + 2) >> 1) * (size_t) n_chunk_n_cols;
                swiglu_nx_push_chunk(weight_dma, curr_raw, octx->src[p], nc_n,
                                     hex_smin(n - nc_n, (size_t) n_chunk_n_cols), n_k_tiles, tile_size, aligned_tile_size);
            }

            if (j == 0 && ov_n > 1) {
                // chunk 0 (gate) as one HMX job per row group; the last group job stays outstanding until j = 1 pops it
                size_t row0 = 0;
                for (int r = 0; r < ov_n; ++r) {
                    dma_queue_pop(weight_dma);                  // group r's tiles have landed
                    if (r >= 2) {
                        hmx_queue_pop(ctx->hmx_queue);          // group r-2 complete: its job slot is free
                    }
                    const size_t tr0 = row0 / HTP_MM_HMX_TILE_N_ROWS;
                    hmx_matmul_job_init(&grp_job[r & 1], vtcm_out[0] + tr0 * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS,
                                        vtcm_f16_act + tr0 * tile_k * HTP_MM_HMX_TILE_N_ELMS, vtcm_wbuf[0], vtcm_scales,
                                        (uint32_t) hmx_ceil_div((size_t) ov_rows[r], HTP_MM_HMX_TILE_N_ROWS), n_col_tiles, tile_k);
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &grp_job[r & 1]));
                    row0 += (size_t) ov_rows[r];
                }
                hmx_queue_pop(ctx->hmx_queue);                  // all group jobs but the last complete
                continue;
            }
            if (j == n_jobs - 1 && tail_g > 1) {
                // last up job as tail_g row groups: the epilogue of group r-1 runs while group r's job executes, so only the
                // last group's epilogue stays after the HMX work. Each group keeps its two epilogue parts (row halves).
                for (uint32_t r = 0; r <= tail_g; ++r) {
                    if (r < tail_g) {
                        const uint32_t rt0 = r * tail_gr;
                        const uint32_t nrt = (tail_gr < n_row_tiles - rt0) ? tail_gr : (n_row_tiles - rt0);
                        hmx_matmul_job_init(&grp_job[r & 1], vtcm_out[2 + (i & 1)] + (size_t) rt0 * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS,
                                            vtcm_f16_act + (size_t) rt0 * tile_k * HTP_MM_HMX_TILE_N_ELMS, vtcm_wbuf[1], vtcm_scales,
                                            nrt, n_col_tiles, tile_k);
                        hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &grp_job[r & 1]));
                    }
                    hmx_queue_pop(ctx->hmx_queue);      // r = 0: gate of the last pair; else group r-1 complete
                    if (r == 0) {
                        if (i >= 1) {                    // the general step j-1 -> j: part 1 of the previous pair
                            swiglu_nx_pair_epi(ctx, n_threads, vtcm_out, out_tiles, out_row0, n_row_tiles, (uint32_t) (i - 1),
                                               n_col_tiles, out_cols, 1, barrier);
                        }
                    } else {
                        const uint32_t g_rt0 = (r - 1) * tail_gr;
                        const uint32_t g_nrt = (tail_gr < n_row_tiles - g_rt0) ? tail_gr : (n_row_tiles - g_rt0);
                        const __fp16 * gg = vtcm_out[i & 1] + (size_t) g_rt0 * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
                        const __fp16 * uu = vtcm_out[2 + (i & 1)] + (size_t) g_rt0 * n_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
                        swiglu_pair_epi(ctx, n_threads, gg, uu, out_tiles, out_row0 + g_rt0, g_nrt, (uint32_t) i * n_col_tiles,
                                        n_col_tiles, out_cols, 0, barrier);
                        swiglu_pair_epi(ctx, n_threads, gg, uu, out_tiles, out_row0 + g_rt0, g_nrt, (uint32_t) i * n_col_tiles,
                                        n_col_tiles, out_cols, 1, barrier);
                    }
                }
                tail_done = 1;
                continue;
            }

            hmx_matmul_job_init(&job_slots[j & 1], vtcm_out[2 * p + (i & 1)], vtcm_f16_act, vtcm_wbuf[p], vtcm_scales,
                                n_row_tiles, (uint32_t) (n_cols / HTP_MM_HMX_TILE_N_COLS), tile_k);
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[j & 1]));

            if (j > 0) {
                hmx_queue_pop(ctx->hmx_queue);  // job j-1 complete
                const int q = j - 1;
                if ((q & 1) != 0) {
                    swiglu_nx_pair_epi(ctx, n_threads, vtcm_out, out_tiles, out_row0, n_row_tiles, (uint32_t) (q >> 1),
                                       n_col_tiles, out_cols, 0, barrier);
                } else if (q >= 2) {
                    swiglu_nx_pair_epi(ctx, n_threads, vtcm_out, out_tiles, out_row0, n_row_tiles, (uint32_t) ((q >> 1) - 1),
                                       n_col_tiles, out_cols, 1, barrier);
                }
            }
        }

        if (!tail_done) {
            hmx_queue_pop(ctx->hmx_queue);  // last job (up of the last pair) complete
            const uint32_t last = (uint32_t) (n_chunk_cnt - 1);
            swiglu_nx_pair_epi(ctx, n_threads, vtcm_out, out_tiles, out_row0, n_row_tiles, last, n_col_tiles, out_cols, 0, barrier);
            swiglu_nx_pair_epi(ctx, n_threads, vtcm_out, out_tiles, out_row0, n_row_tiles, last, n_col_tiles, out_cols, 1, barrier);
        }
    }

    return HTP_STATUS_OK;
}

static inline dma_addr_t hmx_mm_weight_batch_data(const hmx_mm_f16_f32_batched_params_t *params,
                                                         int dst_b2, int dst_b3) {
    const size_t b2_idx = (params->r2 <= 1) ? (size_t) dst_b2 : (size_t) fastdiv((uint32_t) dst_b2, &params->div_r2);
    const size_t b3_idx = (params->r3 <= 1) ? (size_t) dst_b3 : (size_t) fastdiv((uint32_t) dst_b3, &params->div_r3);
    return params->weight + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
}

static inline dma_addr_t hmx_mm_act_batch_addr(const hmx_mm_f16_f32_batched_params_t *params,
                                                int dst_b2, int dst_b3) {
    return params->act_dma_addr + dst_b2 * params->act_nb2 +
           dst_b3 * params->act_nb3;
}

static inline float *hmx_mm_dst_batch_ptr(const hmx_mm_f16_f32_batched_params_t *params,
                                              int dst_b2, int dst_b3) {
    return (float *) ((uint8_t *) params->dst +
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

static int hmx_mm_f16_f32_batched_simple(struct htp_context *ctx,
                                                        const hmx_mm_f16_f32_batched_params_t *params,
                                                        int m_chunk, int n_chunk, int pipeline, int n_threads, int act_threads, int vtcm_size,
                                                        const struct fastdiv_values * act_threads_div, const struct fastdiv_values * k_div) {
    int ret = 0;
    for (int b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        for (int b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            dma_addr_t cur_src2_addr = params->src2_addr ? (params->src2_addr +
                                       b2 * params->src2_nb2 +
                                       b3 * params->src2_nb3) : 0;
            ret = hmx_mm_2d_f32(ctx, params->weight_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                cur_src2_addr, params->src2_bytes,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_weight_batch_data(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->weight_stride * (int)sizeof(__fp16),
                                HTP_TYPE_F16, HTP_MM_KERNEL_HMX_F16_BATCHED, params->k, params->dst_stride, params->src2_stride, params->n,
                                m_chunk, n_chunk, pipeline, n_threads, act_threads,
                                act_threads_div, k_div, 0, 0, vtcm_size, false, 0);
        }
    }
    return ret;
}

static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_batched_params_t *params,
                               int m_chunk, int n_chunk, int pipeline, int n_threads, int act_threads,
                               const struct fastdiv_values * act_threads_div,
                               const struct fastdiv_values * k_div,
                               int vtcm_size) {
    if (params->act_stride < params->k || params->weight_stride < params->k || params->dst_stride < params->n) { return -1; }
    if (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { return -1; }
    if (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { return -1; }
    if (params->k % 32 != 0 || params->n % 32 != 0) { return -1; }
    if (!hex_is_aligned(params->dst, VLEN) || (params->act_dma_addr & (VLEN - 1)) != 0) { return -1; }

    const int group_size = params->r2;
    const size_t vtcm_budget  = ctx->vtcm_size;

    // Check if the precomputed parameters are grouped or simple. Grouped path is only valid if group_size > 1 and it fits within VTCM budget.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    bool run_grouped = (group_size > 1 && (size_t)vtcm_size <= vtcm_budget);
    if (!run_grouped) {
        return hmx_mm_f16_f32_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size, act_threads_div, k_div);
    }

    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const size_t vec_dot_size = params->k * sizeof(__fp16);

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used = vtcm_size;

    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_size, false, act_threads, 0, params->src2_bytes);

    if (L.total_bytes > vtcm_budget) {
        FARF(HIGH, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        return hmx_mm_f16_f32_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size, act_threads_div, k_div);
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight     = VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    void    *vtcm_scratch0   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]);
    void    *vtcm_scratch1   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    float   *vtcm_f32_act    = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);

    const bool has_src2      = (params->src2_bytes > 0 && params->src2_addr != 0);
    float   *vtcm_src2       = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_src2, has_src2);
    if (has_src2) {
        dma_queue_push(params->weight_dma, dma_make_data(vtcm_src2, params->src2_addr), hex_align_up(params->src2_bytes, 128), 0, params->src2_bytes, 1);
        dma_queue_pop(params->weight_dma);
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(HIGH, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, group_size, params->ne13,
            m_chunk_n_rows, n_chunk_n_cols,
            L.total_bytes, vtcm_budget);

    const size_t fp16_row_bytes   = (size_t) params->k * sizeof(__fp16);
    const size_t weight_row_bytes = (size_t) params->weight_stride * sizeof(__fp16);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;

    for (int b3 = 0; b3 < params->ne13; ++b3) {
        for (int b2_base = 0; b2_base < params->ne12; b2_base += group_size) {
            const dma_addr_t weight_group = hmx_mm_weight_batch_data(params, b2_base, b3);
            dma_queue * weight_dma = params->weight_dma;

            for (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_rows) {
                const size_t n_rows = hex_smin((size_t) params->m - mr, m_chunk_n_rows);
                const size_t n_row_tiles = hmx_ceil_div((int) n_rows, HTP_MM_HMX_TILE_N_ROWS);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                for (int g = 0; g < group_size; ++g) {
                    const dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_base + g, b3) +
                                                      mr * params->act_stride * sizeof(float);
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                    struct activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_rows = (int) n_rows,
                        .k_block = params->k,
                        .k_stride = params->act_stride,
                        .n_threads = act_threads,
                        .act_threads_div = act_threads_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_bytes = L.act_f32_bytes,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                // Prologue: Push A0 and A1 (if exists)
                {
                    const size_t n_cols_first = hex_smin((size_t) params->n, n_chunk_n_cols);
                    dma_queue_push(weight_dma, dma_make_data(vtcm_scratch0, weight_group),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_first);
                }
                if (n_chunk_n_cols < (size_t) params->n) {
                    const size_t n_cols_second = hex_smin((size_t) params->n - n_chunk_n_cols, n_chunk_n_cols);
                    dma_queue_push(weight_dma, dma_make_data(vtcm_scratch1, weight_group + params->weight_stride * sizeof(__fp16)),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_second);
                }

                for (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_cols) {
                    const size_t n_cols      = hex_smin((size_t) params->n - nc, n_chunk_n_cols);
                    const size_t n_col_tiles = hmx_ceil_div((int) n_cols, HTP_MM_HMX_TILE_N_COLS);

                    {
                        void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;

                        hmx_interleave_rows_to_tiles(vtcm_weight, (const __fp16 *) curr_raw, n_cols, params->k, params->k, 0, n_cols);

                        const size_t nc_next = nc + n_chunk_n_cols * 2;
                        if (nc_next < (size_t) params->n) {
                            const size_t n_cols_next = hex_smin((size_t) params->n - nc_next, n_chunk_n_cols);
                            const dma_addr_t next_weight_chunk = weight_group + nc_next * params->weight_stride * sizeof(__fp16);

                            dma_queue_push(weight_dma, dma_make_data(curr_raw, next_weight_chunk),
                                              fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_next);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    for (int g = 0; g < group_size; ++g) {
                        {
                            const __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                            hmx_matmul_job_init(&job, vtcm_output, vtcm_act_g, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, params->k / 32);
                            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                            hmx_queue_pop(ctx->hmx_queue);
                        }

                        {
                            float *output = hmx_mm_dst_batch_ptr(params, b2_base + g, b3) + mr * params->dst_stride + nc;
                            const float *src2_chunk = has_src2 ? (vtcm_src2 + mr * params->src2_stride + nc) : NULL;
                            int chunk_dst_cols = params->n - (int)nc;
                            if (chunk_dst_cols > 0) {
                                transfer_output_chunk_threaded(ctx, output, src2_chunk, vtcm_output, (int) n_rows, (int) n_cols,
                                                               params->dst_stride, params->src2_stride, chunk_dst_cols, n_threads);
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}

static void transfer_activation_chunk_gathered_threaded(
            struct htp_context *ctx,
            __fp16 *dst,
            const float *src,
            int start_row,
            int n_rows,
            int k_block,
            const struct mmid_row_mapping *matrix_rows,
            int cur_a,
            int mapping_stride,
            int ne11,
            size_t nb11,
            size_t nb12,
            int cne1,
            int n_threads,
            int k_valid) {
    if (n_rows <= 0) return;
    int chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    int actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = actual_threads,
        .n_tot_chunks      = n_rows,
        .n_chunks_per_task = chunks_per_thread,
        .k_block           = k_block,
        .matrix_rows       = matrix_rows,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = ne11 > 1 ? init_fastdiv_values(ne11) : (struct fastdiv_values){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .start_row         = start_row,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    worker_callback_t worker_fn = ne11 == 1 ? transfer_activation_chunk_gathered_worker_flat_fn :
                                              transfer_activation_chunk_gathered_worker_fn;

    if (actual_threads <= 1) {
        worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, worker_fn, &state, actual_threads);
    }
}

static void transfer_output_chunk_scattered_threaded(
            struct htp_context *ctx,
            float *dst,
            const __fp16 *vtcm_src,
            int start_row,
            int n_rows,
            int n_cols,
            const struct mmid_row_mapping *matrix_rows,
            int cur_a,
            int mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            int cne1,
            int n_threads) {
    if (n_rows <= 0) return;
    int chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    int actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = actual_threads,
        .n_tot_chunks      = n_rows,
        .n_chunks_per_task = chunks_per_thread,
        .n_cols            = n_cols,
        .matrix_rows       = matrix_rows,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .start_row         = start_row,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    if (actual_threads <= 1) {
        transfer_output_chunk_scattered_worker_fn(1, 0, &state);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_scattered_worker_fn, &state, actual_threads);
    }
}

static int hmx_mm_id_2d_f32(struct htp_context *ctx,
                                         dma_queue *weight_dma,
                                         float *restrict dst,
                                         const float *activation,
                                         dma_addr_t weight,
                                         int m, int k, int n,
                                         int k_valid,
                                         int ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         int weight_stride,
                                         int weight_type,
                                         const struct mmid_row_mapping *matrix_rows,
                                         int cur_a,
                                         int mapping_stride,
                                         int m_start,
                                         int m_end,
                                         int n_threads) {
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const int cne1 = m;
    const int m_padded = hex_align_up(m, 32);

    if (k % 32 != 0 || n % 32 != 0) { return -1; }
    if (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { return -1; }

    size_t row_stride = htp_mm_get_tiled_row_stride(weight_type, k);
    if (row_stride == 0) {
        return -1;
    }

    worker_callback_t dequant_worker_fn = NULL;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_0; break;
        case HTP_TYPE_IQ4_NL: dequant_worker_fn = dequantize_tiled_worker_loop_iq4_nl; break;
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q4_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_1; break;
        case HTP_TYPE_MXFP4:  dequant_worker_fn = dequantize_tiled_worker_loop_mxfp4; break;
        case HTP_TYPE_Q8_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q8_0; break;
        case HTP_TYPE_Q5_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q5_k; break;
        case HTP_TYPE_Q6_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q6_k; break;
        case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
        case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
        default:
            return -1;
    }

    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const bool is_quant   = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);

    const size_t vec_dot_size = k * sizeof(__fp16);
    const size_t vtcm_budget  = ctx->vtcm_size;
    size_t vtcm_used = 0;

    int tile_size = htp_mm_get_weight_tile_size(weight_type);
    int aligned_tile_size = htp_mm_get_weight_aligned_tile_size(weight_type);

    const uint32_t dma_dst_stride  = is_quant ? aligned_tile_size : row_stride;
    const uint32_t dma_src_stride  = is_quant ? tile_size : weight_stride;
    const uint32_t dma_width_bytes = is_quant ? tile_size : row_stride;

    const size_t qweight_row_stride = is_quant ? (size_t)(n_k_tiles * aligned_tile_size) / 32 : 0;
    const size_t weight_row_stride = is_quant ? qweight_row_stride : row_stride;

    size_t size_per_n = 0, size_per_m = 0, size_per_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(weight_type, k, /*pipeline=*/false, aligned_tile_size,
                                  &size_per_n, &size_per_m, &size_per_mn);

    const size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
    if (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, size_per_n, size_per_m, size_per_mn,
                           m_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_COST_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_padded * HTP_MM_HMX_COST_A_CONVERT, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_padded, k, n, vtcm_budget);
        return -1;
    }

    const size_t weight_area_size = hex_align_up(n_chunk_n_cols * weight_row_stride, HTP_MM_HMX_TILE_SIZE);
    const size_t act_area_size    = hex_align_up(m_chunk_n_rows * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
    const size_t output_area_size = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * sizeof(__fp16), HTP_MM_HMX_TILE_SIZE);

    size_t scratch0_size = hex_align_up(n_chunk_n_cols * vec_dot_size, HTP_MM_HMX_TILE_SIZE);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight   = weight_area_size ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size) : NULL;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_size);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
    void    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_size);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_used = vtcm_ptr - (uint8_t *) ctx->vtcm_base;
    if (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_used, vtcm_budget);
        return -1;
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;

    for (size_t mr = (size_t) m_start; mr < (size_t) m_end; mr += m_chunk_n_rows) {
        const size_t n_rows = hex_smin((size_t) m_end - mr, m_chunk_n_rows);
        const size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (int) mr, (int) n_rows, k,
            matrix_rows, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_threads, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        if (n > 0) {
            const size_t n_cols = hex_smin((size_t) n, n_chunk_n_cols);
            const uint32_t height = is_quant ? (n_cols / 32) * n_k_tiles : n_cols;
            dma_queue_push(weight_dma, dma_make_data(vtcm_weight, weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height);
        }

        for (size_t nc = 0; nc < (size_t) n; nc += n_chunk_n_cols) {
            const size_t n_cols = hex_smin((size_t) n - nc, n_chunk_n_cols);
            const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

            // A: Wait for weight DMA
            void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_weight_chunk_to_fp16_tiles(
                ctx, vtcm_scratch0, curr_raw,
                n_cols, k, row_stride, weight_type,
                n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads
            );

            // Start weight DMA for the next chunk early
            const size_t nc_next = nc + n_chunk_n_cols;
            if (nc_next < (size_t) n) {
                const size_t n_cols_next = hex_smin((size_t) n - nc_next, n_chunk_n_cols);
                const uint32_t height_next = is_quant ? (n_cols_next / 32) * n_k_tiles : n_cols_next;
                dma_queue_push(weight_dma, dma_make_data(curr_raw, weight + nc_next * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, vtcm_scratch0, vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
            hmx_queue_pop(ctx->hmx_queue);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (int) mr, (int) n_rows, (int) n_cols,
                matrix_rows, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_threads);
        }
    }

    return 0;
}

// --- Dispatchers and Public Entry Points ---

static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    const int k_valid = (int) src0->ne[0];
    // F16 weights may be HMX-tile-padded past their true K (see ggml_hexagon_precompute_matmul_params_impl / hmx_mm_2d_f32's needs_k_pad zero-fill) -- hmx_mm_2d_f32 itself hard-requires k % 32 == 0, so pass the padded value as `k` while k_valid (below) stays the true, unpadded tensor dimension for DMA-width/tail-masking purposes.
    int k = (src0->type == HTP_TYPE_F16) ? (int) hex_round_up((uint32_t) k_valid, 32) : k_valid;
    int n = (int) src0->ne[1];
    const int m_total    = (int) src1->ne[1];
    const int act_stride = (int)(src1->nb[1] / sizeof(float));
    const int wgt_stride = (int)(src0->nb[1] / sizeof(__fp16));

    int m_start = 0;
    int m_rows  = m_total;
    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_can_row_partition(dst, sizeof(float));
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_total, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        m_start = (int) range.start;
        m_rows  = (int) range.count;
    }

    if (m_rows == 0) {
        return HTP_STATUS_OK;
    }

    dma_addr_t src2_addr = 0;
    size_t src2_bytes = 0;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    if (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / sizeof(float));
        src2_addr = src2->data + m_start * src2_stride * sizeof(float);
        src2_bytes = (size_t) kparams->vtcm_src2_size;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    const int dst_stride = (int)(dst->nb[1] / sizeof(float));
    float       * dst_ptr = (float *) dst->data + m_start * dst_stride;
    const dma_addr_t act_addr = src1->data + m_start * act_stride * sizeof(float);

    int ret = -1;
    g_a8w8_dbg = (octx->flags & HTP_OPFLAGS_MM_A8W8) != 0 ? octx->op_params[3] : 0;
    g_a8w8_clip = a8w8_clip_from_params(octx);
    g_a8w8_pe_weight = ((octx->flags & HTP_OPFLAGS_MM_A8W8) != 0 && (src0->flags & HTP_TENSOR_A8W8) != 0) ?
                       src0->data + hex_align_up((size_t) src0->nb[3] * src0->ne[3], 128) : 0;
    const int n_threads = kparams->n_threads;
    if (kparams->kernel_type == HTP_MM_KERNEL_HMX_F16_BATCHED) {
        hmx_mm_f16_f32_batched_params_t batch_params = {
            .dst             = dst_ptr,
            .src2_addr       = src2_addr,
            .src2_bytes      = src2_bytes,
            .act_dma_addr    = act_addr,
            .weight          = src0->data,
            .weight_dma      = octx->ctx->dma[0],
            .m               = m_rows,
            .k               = k,
            .n               = n,
            .act_stride      = act_stride,
            .weight_stride   = wgt_stride,
            .dst_stride      = dst_stride,
            .src2_stride     = src2_stride,
            .ne02            = ne02,
            .ne03            = ne03,
            .ne12            = ne12,
            .ne13            = ne13,
            .src0_nb2        = src0->nb[2],
            .src0_nb3        = src0->nb[3],
            .act_nb2         = src1->nb[2],
            .act_nb3         = src1->nb[3],
            .dst_nb2         = dst->nb[2],
            .dst_nb3         = dst->nb[3],
            .src2_nb2        = src2_nb2,
            .src2_nb3        = src2_nb3,
            .r2              = (ne02 > 0) ? (ne12 / ne02) : 1,
            .r3              = (ne03 > 0) ? (ne13 / ne03) : 1,
            .div_r2          = kparams->div_r2,
            .div_r3          = kparams->div_r3,
        };
        ret = hmx_mm_f16_f32_batched(octx->ctx, &batch_params,
                                     kparams->m_chunk, kparams->n_chunk,
                                     kparams->pipeline, n_threads,
                                     kparams->n_act_threads,
                                     &kparams->div_n_act_threads,
                                     &kparams->div_ne00_padded,
                                     kparams->vtcm_size);
    } else {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_bytes,
            act_addr, src0->data,
            m_rows, k, n, act_stride, (int) src0->nb[1], (int) src0->type, kparams->kernel_type, (int) src1->ne[0],
            dst_stride, src2_stride, (int)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_threads,
            kparams->n_act_threads,
            &kparams->div_n_act_threads,
            &kparams->div_ne00_padded,
            kparams->tile_size, kparams->aligned_tile_size, kparams->vtcm_size,
            (src1->flags & HTP_TENSOR_ACT_TILE) != 0, octx->flags
        );
    }

    if (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        return HTP_STATUS_INTERNAL_ERR;
    }
    return HTP_STATUS_OK;
}

// HTP_OPFLAGS_MM_EARLY_W: push the first weight chunk of an HMX prefill matmul whose activation is staged in row groups (hmx_mm_pipe_run / hmx_mm_nx_swiglu_2d stage path) at op entry, ahead of the op bookkeeping and the input cache flush. Anything the kernel does not recognise leaves mm_early_pending set and proc_op_req drains the queue after the op.
// Weights are static in DDR (never dirty), the VTCM destination (raw buffer 0) is free between ops, and the kernel's first pipeline run takes over the queue entry exactly where it would have pushed it (the FIFO order raw0, activation groups, raw1 is unchanged).
void htp_mm_early_weight_dma(struct htp_ops_context * octx) {
    struct htp_context * ctx = octx->ctx;
    const struct htp_mm_kernel_params * kp = (const struct htp_mm_kernel_params *) octx->kernel_params;
    if (!ctx->hmx_queue || !kp->n_hmx || !kp->pipeline || kp->kernel_type != HTP_MM_KERNEL_HMX_2D || kp->vtcm_src2_size > 0 ||
        kp->m_chunk <= 0 || kp->n_chunk <= 0 || (octx->flags & HTP_OPFLAGS_MM_ACT_OVERLAP) == 0) {
        return;
    }
    const struct htp_tensor * w   = octx->src[0];
    const struct htp_tensor * act = NULL;
    uint32_t k = 0, m = 0;
    bool swiglu = false;
    if (octx->op == HTP_OP_MUL_MAT) {
        act = octx->src[1];
        if (!w || !act) return;
        k = w->ne[0];
        m = act->ne[1];
    } else if (octx->op == HTP_OP_MUL_MAT_NX) {
        swiglu = (octx->flags & HTP_OPFLAGS_SWIGLU_OUT) != 0;
        act    = swiglu ? octx->src[2] : octx->src[kp->n_weights];
        if (!w || !act) return;
        k = act->ne[0];
        m = act->ne[1] * act->ne[2] * act->ne[3];
    } else {
        return;
    }
    if ((w->flags & HTP_TENSOR_A8W8) != 0) return;   // pre-expanded blobs have their own DMA shape
    if ((act->flags & HTP_TENSOR_ACT_TILE) == 0 || w->type != HTP_TYPE_Q4_0 || k == 0 || k % 32 != 0 || w->ne[1] == 0 || w->ne[1] % 32 != 0) {
        return;
    }
    int ov_rows[HMX_ACT_OVERLAP_MAX_GROUPS];
    if (hmx_act_overlap_plan(hex_smin((size_t) m, (size_t) kp->m_chunk), ov_rows) <= 1) {
        return;
    }
    const size_t n_tiles = w->ne[1] / 32;
    const size_t nt      = (size_t) kp->n_chunk / 32;
    size_t       off_w0, total;
    size_t       tw;
    if (swiglu) {
        if (kp->n_chunk > HTP_MM_SWIGLU_NC || w->ne[1] % (uint32_t) kp->n_chunk != 0) return;
        struct htp_mm_swiglu_vtcm_layout L;
        htp_mm_swiglu_vtcm_layout_build(&L, (size_t) kp->m_chunk, (size_t) kp->n_chunk, k, (uint32_t) kp->aligned_tile_size, (uint32_t) kp->n_act_threads);
        off_w0 = L.off_weight[0];
        total  = L.total_bytes;
        tw     = nt;
    } else {
        struct htp_mm_hmx_vtcm_layout L;
        htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, HTP_TYPE_Q4_0, k, (size_t) kp->m_chunk, (size_t) kp->n_chunk, 1, true,
                                     (uint32_t) kp->n_act_threads, (uint32_t) kp->aligned_tile_size, 0);
        off_w0 = L.off_weight[0];
        total  = L.total_bytes;
        const bool split = (octx->flags & HTP_OPFLAGS_MM_SPLIT_EDGES) != 0;
        tw = (split && nt >= 2 && n_tiles > nt) ? nt / 2 : hex_smin(n_tiles, nt);
        if ((octx->flags & HTP_OPFLAGS_MM_A8W8) != 0) {
            const uint32_t E = (uint32_t) (octx->op_params[3] >> 8) & 0xff, E1 = (uint32_t) (octx->op_params[3] >> 16) & 0xff;
            if (E > 0 && E1 > 0 && n_tiles > 2 * E + E1 + nt) tw = E;   // must match hmx_mm_a8w8_plan_init
        }
    }
    if (total > ctx->vtcm_size || tw == 0 || !dma_queue_empty(ctx->dma[0])) {
        return;
    }
    const uint32_t n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    dma_queue_push(ctx->dma[0], dma_make_data(ctx->vtcm_base + off_w0, w->data), (uint32_t) kp->aligned_tile_size, (uint32_t) kp->tile_size,
                   (uint32_t) kp->tile_size, (uint32_t) (tw * n_k_tiles));
    ctx->mm_early_pending = true;
}

int op_matmul(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const int status = htp_mm_init_context(octx, kparams);
    if (status != HTP_STATUS_OK) {
        return status;
    }

    if (kparams->n_hmx) {
        return hmx_mm_op_matmul(octx, kparams);
    }

    return hvx_mm_matmul(octx);
}

static int hmx_mm_op_matmul_id(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx
) {
    const uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    htp_matmul_tensors_preamble;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const int n_ids = octx->src[2]->ne[0];
    const int n_as  = ne02;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;

        const int m_padded = hex_align_up(cne1, 32);
        int m_start = 0, m_end = m_padded;
        if (octx->ctx->mdev.count > 1) {
            const bool can_split = htp_tensor_mdev_data_aligned(dst) && (uint32_t) cne1 >= octx->ctx->mdev.count;
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            m_start = (int) range.start;
            m_end   = (int) (range.start + range.count);
        }
        if (m_start >= m_end) continue;

        int ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (float*) dst->data, (float*) src1->data,
                                   src0->data + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (int) src0->nb[1], (int) src0->type,
                                   matrix_rows, cur_a, mmctx->mapping_stride,
                                   m_start, m_end, (int) octx->n_threads);
        if (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    return HTP_STATUS_OK;
}

static int hvx_mm_matmul_id(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    const uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    const uint32_t act_nrows            = mmctx->act_nrows;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const struct htp_tensor * restrict ids = octx->src[2];
    const size_t src0_row_size = nb01;

    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne10 + qk - 1) / qk;
    const uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    if (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
        for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
    }
    size_t src1_row_size  = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_threads,
                                 0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, true, false);

    const size_t vtcm_size = L.total_bytes;

    FARF(HIGH, "matmul-id-%s : src0-spad-size %zu src1-spad-size %zu src2-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.src1_bytes, L.dst_bytes, vtcm_size);

    FARF(HIGH, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         src1->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src1     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_src2     = NULL;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->src2_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    mmctx->vtcm_src0_stride    = src0_row_size_padded;
    mmctx->vtcm_src1_stride    = src1_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
    mmctx->vtcm_src2_size_per_thread = 0;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_threads);

    return HTP_STATUS_OK;
}

static int hmx_mm_op_matmul_id_nx(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx
) {
    const uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const int n_as = src0->ne[2];

    for (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;

        const int m_padded = hex_align_up(cne1, 32);
        int m_start = 0, m_end = m_padded;
        if (octx->ctx->mdev.count > 1) {
            bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.count;
            for (uint32_t p = 0; p < n_weights && can_split; ++p) {
                const struct htp_tensor * restrict dst = octx->dsts[p];
                can_split = !dst || htp_tensor_mdev_data_aligned(dst);
            }
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            m_start = (int) range.start;
            m_end   = (int) (range.start + range.count);
        }
        if (m_start >= m_end) continue;

        for (uint32_t p = 0; p < n_weights; ++p) {
            const struct htp_tensor * restrict src_w = octx->src[p];
            const struct htp_tensor * restrict dst   = octx->dsts[p];
            if (!src_w || !dst) continue;

            int ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (float*) dst->data, (float*) act->data,
                                       src_w->data + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (int) src_w->nb[1], (int) src_w->type,
                                       matrix_rows, cur_a, mmctx->mapping_stride,
                                       m_start, m_end, (int) octx->n_threads);
            if (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                return HTP_STATUS_NO_SUPPORT;
            }
        }
    }

    return HTP_STATUS_OK;
}

static int hvx_mm_matmul_id_nx(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    const uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    const uint32_t act_nrows            = mmctx->act_nrows;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const struct htp_tensor * restrict ids  = octx->src[n_weights + 1];
    const size_t src0_row_size = src0->nb[1];

    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (act->ne[0] + qk - 1) / qk;
    const uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    if (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
        for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
    }
    size_t src1_row_size = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(act->ne[0]) : htp_mm_q8_0_tiled_row_size(act->ne[0]);

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, true, false);

    const size_t vtcm_size = L.total_bytes;

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_src1     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    octx->src0_spad.src = NULL;
    octx->src1_spad.src = NULL;
    octx->src2_spad.src = NULL;
    octx->src3_spad.src = NULL;
    octx->dst_spad.src  = NULL;

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_src1_stride    = src1_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * sizeof(float), QK_Q8_0_TILED * sizeof(float));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    FARF(HIGH, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, src1 %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.total_bytes, octx->ctx->vtcm_size, octx->n_threads);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_src1_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_threads);

    return HTP_STATUS_OK;
}

static inline void scan_expert_ids_n(
    const struct htp_tensor * ids,
    const uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    struct mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    const size_t ids_nb1 = ids->nb[1];
    const uint8_t * ids_data = (const uint8_t *) ids->data;

    for (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        const int32_t * row_ptr = (const int32_t *) (ids_data + iid1 * ids_nb1);
        for (uint32_t id = 0; id < n_ids; ++id) {
            const int32_t i02 = row_ptr[id];
            if (i02 == -1) {
                continue;
            }
            assert(i02 >= 0 && i02 < n_as);

            if (matrix_rows) {
                matrix_rows[i02 * mapping_stride + counts[i02]] = (struct mmid_row_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

// the matmuls only write the rows that an expert owns, so the rows of the skipped slots are zeroed here
static inline void zero_skipped_dst_rows(
    const struct htp_tensor * ids,
    uint32_t n_ids,
    const struct htp_tensor * dst,
    uint32_t ne0,
    size_t nb1,
    size_t nb2
) {
    const uint8_t * ids_data = (const uint8_t *) ids->data;

    for (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        for (uint32_t id = 0; id < n_ids; ++id) {
            const int32_t i02 = *(const int32_t *) (ids_data + iid1 * ids->nb[1] + id * ids->nb[0]);
            if (i02 != -1) {
                continue;
            }
            memset((uint8_t *) (dst->data + id * nb1 + iid1 * nb2), 0, ne0 * sizeof(float));
        }
    }
}

static inline void scan_expert_ids(
    const struct htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    struct mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    const size_t ids_nb0 = ids->nb[0];

    if (ids_nb0 == 4) {
        switch (n_ids) {
            case 8:  scan_expert_ids_n(ids, 8,     n_as, counts, matrix_rows, mapping_stride); break;
            case 4:  scan_expert_ids_n(ids, 4,     n_as, counts, matrix_rows, mapping_stride); break;
            case 2:  scan_expert_ids_n(ids, 2,     n_as, counts, matrix_rows, mapping_stride); break;
            default: scan_expert_ids_n(ids, n_ids, n_as, counts, matrix_rows, mapping_stride); break;
        }
    } else {
        // Strided fallback
        const size_t ids_nb1 = ids->nb[1];
        const uint8_t * ids_data = (const uint8_t *) ids->data;
        for (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            const int32_t * row_ptr = (const int32_t *) (ids_data + iid1 * ids_nb1);
            for (uint32_t id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const uint8_t *) row_ptr + id * ids_nb0);
                if (i02 == -1) {
                    continue;
                }
                assert(i02 >= 0 && i02 < n_as);

                if (matrix_rows) {
                    matrix_rows[i02 * mapping_stride + counts[i02]] = (struct mmid_row_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

int op_matmul_id(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;

    const int status = htp_mm_init_context(octx, kparams);
    if (status != HTP_STATUS_OK) {
        return status;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = src1;

    const struct htp_tensor * restrict ids = octx->src[2];
    if (htp_tensor_is_extended(ids) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    const uint32_t src0_nrows = ne01;  // per expert
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    // row groups
    const int n_ids = ids->ne[0];  // n_expert_used
    const int n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    struct mmid_row_mapping * matrix_rows = NULL;

    if (src1_nrows > 1) {
        const size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
        assert(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((const void *) ids->data, ids->ne[1] * ids->nb[1]);

        memset(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, NULL, 0);

        uint32_t max_count = hvx_reduce_max_i32((const uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_count > 0 ? max_count : 1;

        size_t matrix_row_map_size  = n_as * mapping_stride * sizeof(struct mmid_row_mapping);
        const size_t total_map_size = matrix_row_counts_size + matrix_row_map_size;

        if (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            if (!mapping_buf) {
                return HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (struct mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        memset(matrix_row_counts, 0, n_as * sizeof(uint32_t));
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, matrix_rows, mapping_stride);
    }

    zero_skipped_dst_rows(ids, n_ids, dst, ne0, nb1, nb2);

    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_rows          = matrix_rows;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_row_size_padded = src0_row_size_padded;
    mmctx->act_nrows            = src1_nrows;
    mmctx->cur_m_start          = 0;
    mmctx->cur_m_rows           = src1_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    int s;
    if (kparams->n_hmx && kparams->kernel_type != HTP_MM_KERNEL_HMX_INT4 && kparams->kernel_type != HTP_MM_KERNEL_HMX_INT8) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } else {
        uint32_t src0_row_start = 0;
        uint32_t src0_row_end   = src0_nrows;
        if (octx->ctx->mdev.count > 1) {
            const bool can_split = htp_tensor_can_row_partition(dst, sizeof(float));
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            src0_row_start = range.start;
            src0_row_end   = range.start + range.count;
        }

        if (src0_row_start >= src0_row_end) {
            if (mapping_buf != octx->ctx->ddr_spad_base) {
                free(mapping_buf);
            }
            return HTP_STATUS_OK;
        }

        const uint32_t nrows = src0_row_end - src0_row_start;
        mmctx->src0_row_start = src0_row_start;
        mmctx->src0_row_end   = src0_row_end;

        mmctx->src0_nrows_per_thread = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);

        if (hvx_mm_init_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, src1_nrows > 1 ? hvx_mm_id : hvx_mv_id);
        } else {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    if (mapping_buf != octx->ctx->ddr_spad_base) {
        free(mapping_buf);
    }

    return s;
}

int op_matmul_id_nx(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;

    const int status = htp_mm_init_context(octx, kparams);
    if (status != HTP_STATUS_OK) {
        return status;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    const uint32_t n_weights = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const struct htp_tensor * restrict ids  = octx->src[n_weights + 1];
    if (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    for (uint32_t p = 0; p < n_weights; p++) {
        if (octx->dsts[p] && htp_tensor_is_extended(octx->dsts[p])) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    const int n_ids = ids->ne[0];
    const int n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    struct mmid_row_mapping * matrix_rows = NULL;

    if (act_nrows > 1) {
        const size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
        assert(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((const void *) ids->data, ids->ne[1] * ids->nb[1]);

        memset(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, NULL, 0);

        uint32_t max_count = hvx_reduce_max_i32((const uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_count > 0 ? max_count : 1;

        size_t matrix_row_map_size  = n_as * mapping_stride * sizeof(struct mmid_row_mapping);
        const size_t total_map_size = matrix_row_counts_size + matrix_row_map_size;

        if (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            if (!mapping_buf) {
                return HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (struct mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        memset(matrix_row_counts, 0, n_as * sizeof(uint32_t));
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, matrix_rows, mapping_stride);
    }

    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_rows          = matrix_rows;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_row_size_padded = src0_row_size_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_start          = 0;
    mmctx->cur_m_rows           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    int s;
    if (kparams->n_hmx && kparams->kernel_type != HTP_MM_KERNEL_HMX_INT4 && kparams->kernel_type != HTP_MM_KERNEL_HMX_INT8) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } else {
        if (hvx_mm_init_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx : hvx_mv_id_nx);
        } else {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    if (mapping_buf != octx->ctx->ddr_spad_base) {
        free(mapping_buf);
    }

    return s;
}
int op_matmul_nx(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const int status = htp_mm_init_context(octx, kparams);
    if (status != HTP_STATUS_OK) {
        return status;
    }

    if (kparams->n_hmx && kparams->kernel_type != HTP_MM_KERNEL_HMX_INT4 && kparams->kernel_type != HTP_MM_KERNEL_HMX_INT8) {
        if ((octx->flags & HTP_OPFLAGS_SWIGLU_OUT) != 0) {
            return hmx_mm_nx_swiglu_2d(octx, kparams);
        }
        return hmx_mm_nx_2d_f32(octx, kparams);
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const uint32_t n_weights = kparams->n_weights;

    const struct htp_tensor * restrict src0 = octx->src[0]; // first weight
    const struct htp_tensor * restrict act  = octx->src[n_weights]; // activation x

    bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                        src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                        src0->type == HTP_TYPE_MXFP4 || src0->type == HTP_TYPE_Q4_K ||
                        src0->type == HTP_TYPE_Q5_K);

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    if (hvx_mm_init_vec_dot(mmctx, src0->type) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (act->ne[0] + qk - 1) / qk;
    const uint32_t total_nb = act_nrows * nb;

    worker_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    if (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
        for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
    }

    const size_t src1_row_size = htp_mm_weight_has_offset(src0->type)
                               ? htp_mm_q8_1_tiled_row_size(act->ne[0])
                               : htp_mm_q8_0_tiled_row_size(act->ne[0]);

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, false, true);

    const size_t vtcm_size = L.total_bytes;

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_src1     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    octx->src0_spad.src  = NULL;
    octx->src1_spad.src  = NULL;
    octx->src2_spad.src  = NULL;
    octx->src3_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    mmctx->vtcm_src0_stride    = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_src1_stride    = src1_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * sizeof(float), QK_Q8_0_TILED * sizeof(float));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    // Run fused matmul
    const uint32_t n_matmul_jobs = octx->n_threads;
    worker_callback_t matmul_job_func;
    if (is_repacked) {
        switch (src0->type) {
            case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_nx_2d_repacked_q4_0;   break;
            case HTP_TYPE_Q4_1:
            case HTP_TYPE_Q4_K:   matmul_job_func = hvx_mm_nx_2d_repacked_q4_1;   break;
            case HTP_TYPE_Q5_K:   matmul_job_func = hvx_mm_nx_2d_repacked_q5_k;   break;
            case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_nx_2d_repacked_q8_0;   break;
            case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_nx_2d_repacked_iq4nl;  break;
            case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_nx_2d_repacked_mxfp4;  break;
            default:
                htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
                return HTP_STATUS_NO_SUPPORT;
        }
    } else {
        matmul_job_func = hvx_mm_nx_2d;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_src1_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, n_matmul_jobs);

    return HTP_STATUS_OK;
}

// --- HMX microbenchmark (experiment; host GGML_HEXAGON_HMX_BENCH, ggml-hexagon.cpp) ----------------------------------- Isolated HMX job loop on VTCM-resident data: no DMA, no dequant, no output transfer.
// Mirrors core_dot_chunk_fp16: per output tile CLRACC, ceil(n_dot/32) DEEP MACs, STORE_AFTER. Variants: 0 CLRACC + MAC + STORE_AFTER per output tile (production sequence) 1 MAC + STORE_AFTER per output tile, one CLRACC per iteration (no CLRACC between tiles) 2 CLRACC + MAC per output tile, no store 3 MAC only (accumulator never cleared or stored) n_writers > 0: that many HVX threads store 128-byte vectors into a private VTCM window (top 1 MiB) for the whole job.
typedef struct {
    uint32_t       variant;
    uint32_t       n_row, n_col, n_dot, n_iters;
    const __fp16 * act;
    const __fp16 * wgt;
    __fp16 *       out;
    __fp16 *       scales;
    atomic_int *   stop;
    uint64_t       cycles;
    uint64_t       usec;
} hmx_bench_job_t;

static inline void hmx_bench_mac(const __fp16 * row_tiles, const __fp16 * col_tiles, uint32_t n_dot) {
    if (n_dot <= 32) {
        const uint32_t range = 2048u * n_dot - 1;
        asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_tiles), "r"(col_tiles));
        return;
    }
    for (uint32_t l = 0; l < n_dot / 32; ++l) {
        asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(row_tiles), "r"(col_tiles));
        row_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
        col_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
    }
    const uint32_t rem = n_dot % 32;
    if (rem > 0) {
        const uint32_t range = 2048u * rem - 1;
        asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_tiles), "r"(col_tiles));
    }
}

static void hmx_bench_job_fn(void * data) {
    hmx_bench_job_t * b = (hmx_bench_job_t *) data;
    const size_t      dot_stride = (size_t) b->n_dot * HTP_MM_HMX_TILE_N_ELMS;
    const uint32_t    variant    = b->variant;

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) b->scales));
    asm volatile(HMX_CLRACC_F16());

    const uint64_t q0 = HAP_perf_get_qtimer_count();
    const uint64_t c0 = hex_get_cycles();
    if (variant >= 18 && variant <= 20) {
        // Full int sequences incl. readout (clracc, K MACs, 4 byte-plane cvt+store per accumulator set). Results are verified by the caller. Scale tables: 4 x 256 B at `scales` (word i<32 = e<<10, rest 0), e = 20,12,4,0.
        // 18: A8W8 act deep:cm + weight.b -> 1 set (64 rows x 32 cols), wt 1 KB / k-tile 19: A8W4 act deep:cm + weight.n:2x -> 2 sets (64 x 64), wt 1 KB / k-tile 20: A8W4 act deep:cm + weight.n:2x:deep-> 4 sets (64 x 128), wt 2 KB / k-tile Output planes: set s plane k at out + (s*4+k)*2048.
        const uint32_t nsets = (variant == 18) ? 1 : (variant == 19) ? 2 : 4;
        const size_t   wt    = (variant == 20) ? 2048 : 1024;
        uint32_t * tbl = (uint32_t *) b->scales;
        static const uint32_t ee2x[4] = { 20, 12, 4, 0 };   // 2x weights: cvtacc_s01 shifts the accumulator left by 4
        static const uint32_t ee1x[4] = { 24, 16, 8, 0 };
        const uint32_t * ee = (variant == 18) ? ee1x : ee2x;
        for (int k = 0; k < 4; ++k) {
            for (int i = 0; i < 64; ++i) tbl[k * 64 + i] = (i < 32) ? (ee[k] << 10) : 0;
        }
        const uint32_t chunk = 32;
        for (uint32_t it = 0; it < b->n_iters; ++it) {
            for (uint32_t r = 0; r < b->n_row; ++r) {
                const uint8_t * a0 = (const uint8_t *) b->act + (size_t) r * b->n_dot * 2048;
                for (uint32_t c = 0; c < b->n_col; ++c) {
                    const uint8_t * w = (const uint8_t *) b->wgt + (size_t) c * b->n_dot * wt;
                    const uint8_t * a = a0;
                    uint8_t * o = (uint8_t *) b->out;
                    asm volatile("mxclracc\n");
                    uint32_t left = b->n_dot;
                    while (left) {
                        const uint32_t n = left > chunk ? chunk : left;
                        const uint32_t ra = ((n - 1) << 11) | 31;
                        const uint32_t rw = (uint32_t) (wt * n - 128);
                        if (variant == 18) {
                            asm volatile("{\n activation.ub = mxmem(%0, %1):deep:cm\n weight.b = mxmem(%2, %3)\n}\n" : : "r"(a), "r"(ra), "r"(w), "r"(rw));
                        } else if (variant == 19) {
                            asm volatile("{\n activation.ub = mxmem(%0, %1):deep:cm\n weight.n = mxmem(%2, %3):2x\n}\n" : : "r"(a), "r"(ra), "r"(w), "r"(rw));
                        } else {
                            asm volatile("{\n activation.ub = mxmem(%0, %1):deep:cm\n weight.n = mxmem(%2, %3):2x:deep\n}\n" : : "r"(a), "r"(ra), "r"(w), "r"(rw));
                        }
                        a += 2048 * n; w += wt * n; left -= n;
                    }
                    // readout
                    if (variant == 18) {
                        for (int k = 0; k < 4; ++k) {
                            const uint32_t zero = 0;
                            asm volatile("bias = mxmem2(%0)\n" : : "r"(tbl + k * 64));
                            if (k < 3) asm volatile("{ mxmem(%0, %1):after:retain:cm.ub = acc }\n" : : "r"(o + k * 2048), "r"(zero));
                            else       asm volatile("{ mxmem(%0, %1):after:cm.ub = acc }\n" : : "r"(o + k * 2048), "r"(zero));
                        }
                    } else {
                        const uint32_t rs = 3;
                        for (uint32_t s = 0; s < nsets; ++s) {
                            if (s == 2) {
                                const uint32_t rs2 = 2;   // clear + swap to the second accumulator bank
                                asm volatile("bias = mxmem2(%0)\n" : : "r"(tbl));
                                asm volatile("{ cvt.ub = acc(%0):sc0 }\n" : : "r"(rs2));
                            }
                            for (int k = 0; k < 4; ++k) {
                                const uint32_t zero = 0;
                                asm volatile("bias = mxmem2(%0)\n" : : "r"(tbl + k * 64));
                                if (s & 1) asm volatile("{ cvt.ub = acc(%0):sc1 }\n" : : "r"(rs));
                                else       asm volatile("{ cvt.ub = acc(%0):sc0 }\n" : : "r"(rs));
                                asm volatile("{ mxmem(%0, %1):cm = cvt }\n" : : "r"(o + (s * 4 + k) * 2048), "r"(zero));
                            }
                        }
                    }
                }
            }
        }
        goto bench_done;
    }
    if (variant >= 8 && variant <= 17) {
        // Correct-form int dataflow MAC probes (HexKL / hmxconv register forms).
        // Act tile = 64 rows x 32 ch u8 (2048 B). MAC-only (garbage results). 8: A8W8 act deep:cm + weight.b wt 1024 B / k-tile, chunk 32 9: A8W8 same, chunk 64 (act dC 63) 10: A8W4 N1X: weight.n wt 512 B / k-tile 11: A8W4 2x: weight.n:2x (two column sets) wt 1024 B / k-tile 12: A8W4 2x:deep (weight.n:2x:deep) wt 2048 B / k-tile 13: A8W8 weight.b:deep wt 2048 B / k-tile 14: A8W4 N1X weight deep (weight.n:deep) wt 1024 B / k-tile 15: fp16 reference of the production sequence (same loop, 32-row tiles): act.hf deep + weight.hf, wt 2048 B
        const size_t wt = (variant == 10) ? 512 : (variant == 8 || variant == 9 || variant == 11 || variant == 14) ? 1024 : (variant == 16 || variant == 17) ? 4096 : 2048;
        const uint32_t chunk = (variant == 9) ? 64 : (variant == 17) ? 1 : 32;
        for (uint32_t it = 0; it < b->n_iters; ++it) {
            for (uint32_t r = 0; r < b->n_row; ++r) {
                const uint8_t * a0 = (const uint8_t *) b->act + (size_t) r * b->n_dot * 2048;
                for (uint32_t c = 0; c < b->n_col; ++c) {
                    const uint8_t * w = (const uint8_t *) b->wgt + (size_t) c * b->n_dot * wt;
                    const uint8_t * a = a0;
                    asm volatile("mxclracc\n");
                    uint32_t left = b->n_dot;
                    while (left) {
                        const uint32_t n = left > chunk ? chunk : left;
                        const uint32_t ra = ((n - 1) << 11) | 31;
                        const uint32_t rw = (uint32_t) (wt * n - 128);
                        switch (variant) {
                        case 8: case 9:
                            asm volatile("{\n activation.ub = mxmem(%0, %1):deep:cm\n weight.b = mxmem(%2, %3)\n}\n" : : "r"(a), "r"(ra), "r"(w), "r"(rw)); break;
                        case 10:
                            asm volatile("{\n activation.ub = mxmem(%0, %1):deep:cm\n weight.n = mxmem(%2, %3)\n}\n" : : "r"(a), "r"(ra), "r"(w), "r"(rw)); break;
                        case 11:
                            asm volatile("{\n activation.ub = mxmem(%0, %1):deep:cm\n weight.n = mxmem(%2, %3):2x\n}\n" : : "r"(a), "r"(ra), "r"(w), "r"(rw)); break;
                        case 12:
                            asm volatile("{\n activation.ub = mxmem(%0, %1):deep:cm\n weight.n = mxmem(%2, %3):2x:deep\n}\n" : : "r"(a), "r"(ra), "r"(w), "r"(rw)); break;
                        case 13:
                            asm volatile("{\n activation.ub = mxmem(%0, %1):deep:cm\n weight.b = mxmem(%2, %3):deep\n}\n" : : "r"(a), "r"(ra), "r"(w), "r"(rw)); break;
                        case 14:
                            asm volatile("{\n activation.ub = mxmem(%0, %1):deep:cm\n weight.n = mxmem(%2, %3):deep\n}\n" : : "r"(a), "r"(ra), "r"(w), "r"(rw)); break;
                        case 16: {   // fp16 weight deep: 2 weight sets (64 output channels) per activation crouton
                            const uint32_t rr = 2048u * n - 1;
                            const uint32_t rwd = 4096u * n - 1;
                            asm volatile("{\n activation.hf = mxmem(%0, %1):deep\n weight.hf = mxmem(%2, %3):deep\n}\n" : : "r"(a), "r"(rr), "r"(w), "r"(rwd));
                        } break;
                        case 17: {   // fp16 weight deep, 1 crouton per issue (PRM 4.3.2 form)
                            const uint32_t rwd = 4096u - 1;
                            asm volatile("{\n activation.hf = mxmem(%0, %1)\n weight.hf = mxmem(%2, %3):deep\n}\n" : : "r"(a), "r"(2047u), "r"(w), "r"(rwd));
                        } break;
                        default: {
                            const uint32_t rr = 2048u * n - 1;
                            asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(rr), "r"(a), "r"(w));
                        } break;
                        }
                        a += 2048 * n; w += wt * n; left -= n;
                    }
                }
            }
        }
        goto bench_done;
    }
    if (variant == 5 || variant == 6 || variant == 7) {
        // MAC-only int8 probes (results are garbage, only cycles matter). 5: act.ub deep + weight.b, 1024 B tiles, range 1024*n-1
        // 6: same with 2048 B act tiles (paired). 7: int4 weights (weight.n :2x:deep) 512 B tiles.
        const size_t at = (variant == 6) ? 2048 : 1024;
        const size_t wt = (variant == 7) ? 512 : 1024;
        for (uint32_t it = 0; it < b->n_iters; ++it) {
            for (uint32_t r = 0; r < b->n_row; ++r) {
                const uint8_t * a = (const uint8_t *) b->act + (size_t) r * b->n_dot * at;
                for (uint32_t c = 0; c < b->n_col; ++c) {
                    const uint8_t * w = (const uint8_t *) b->wgt + (size_t) c * b->n_dot * wt;
                    asm volatile("mxclracc\n");
                    uint32_t left = b->n_dot;
                    while (left) {
                        const uint32_t n = left > 32 ? 32 : left;
                        const uint32_t ra = (uint32_t) (at * n - 1), rw = (uint32_t) (wt * n - 1);
                        (void) rw;
                        if (variant == 7) {
                            asm volatile(HMX_LOAD_MPY_INT4("%1", "%2", "%0") : : "r"(ra), "r"(a), "r"(w));
                        } else {
                            asm volatile(HMX_LOAD_MPY_INT8("%1", "%2", "%0") : : "r"(ra), "r"(a), "r"(w));
                        }
                        a += at * n; w += wt * n; left -= n;
                    }
                }
            }
        }
        goto bench_done;
    }
    if (variant == 4) {
        // int8 HMX (hmx-int-kernels.S): per output tile: clracc, bias, n_dot x (act.ub deep, wgt.b), cvt.uh + store. act tile 2048 B / k-tile, wgt 1024 B / k-tile.
        uint32_t * bias_tbl = (uint32_t *) b->scales;   // must live in VTCM (bias = mxmem2)
        for (int i = 0; i < 64; ++i) bias_tbl[i] = 0x4000;
        for (uint32_t it = 0; it < b->n_iters; ++it) {
            for (uint32_t r = 0; r < b->n_row; ++r) {
                const uint8_t * a = (const uint8_t *) b->act + (size_t) r * b->n_dot * 2048;
                for (uint32_t c = 0; c < b->n_col; ++c) {
                    uint16_t * o = (uint16_t *) b->out + ((size_t) r * b->n_col + c) * 1024;
                    hmx_tile_dot_read_u16_int8_v81(o, a, (const uint8_t *) b->wgt + (size_t) c * b->n_dot * 1024, bias_tbl, b->n_dot);
                }
            }
        }
        goto bench_done;
    }
    for (uint32_t it = 0; it < b->n_iters; ++it) {
        if (variant == 1) {
            asm volatile(HMX_CLRACC_F16());
        }
        for (uint32_t r = 0; r < b->n_row; ++r) {
            const __fp16 * row_base = b->act + (size_t) r * dot_stride;
            for (uint32_t c = 0; c < b->n_col; ++c) {
                __fp16 * out_tile = b->out + ((size_t) r * b->n_col + c) * HTP_MM_HMX_TILE_N_ELMS;
                if (variant == 0 || variant == 2) {
                    asm volatile(HMX_CLRACC_F16());
                }
                hmx_bench_mac(row_base, b->wgt + (size_t) c * dot_stride, b->n_dot);
                if (variant == 0 || variant == 1) {
                    asm volatile(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(out_tile), "r"(0) : "memory");
                }
            }
        }
    }
bench_done:;
    const uint64_t c1 = hex_get_cycles();
    const uint64_t q1 = HAP_perf_get_qtimer_count();
    b->cycles = c1 - c0;
    b->usec   = HAP_perf_qtimer_count_to_us(q1 - q0);
    if (b->stop) {
        atomic_store(b->stop, 1);
    }
}

typedef struct {
    uint8_t *    base;
    size_t       span;
    atomic_int * stop;
    uint64_t     bytes[HTP_MAX_NTHREADS];
} hmx_bench_wr_t;

static void hmx_bench_writer_fn(unsigned int n, unsigned int i, void * data) {
    (void) n;
    hmx_bench_wr_t * w = (hmx_bench_wr_t *) data;
    uint8_t *        p = w->base + (size_t) i * w->span;
    const HVX_Vector v = Q6_V_vsplat_R(0x3c003c00);
    uint64_t         bytes = 0;
    while (!atomic_load(w->stop)) {
        for (size_t off = 0; off < w->span; off += 128) {
            *(HVX_Vector *) (p + off) = v;
        }
        bytes += w->span;
    }
    w->bytes[i] = bytes;
}


// --- A8W8 pipeline bench (variant 21): the HMX queue thread runs [clracc, KG k-tile MACs (act.ub deep + wgt.b), 4 byte-plane readouts] per 64x32 output tile into a ring of plane slots; n_writers HVX workers concurrently merge the 4 byte planes into int32 (HexKL vshuff sequence), convert to f32, multiply by rowscale*colscale and accumulate into an f32 [64][32] partial.
// n_row = KG (k-tiles per readout), n_col = column units, n_dot = k-tiles per column unit.
typedef struct {
    const uint8_t * act; const uint8_t * wgt; uint8_t * planes; uint32_t n_dot, n_col, kg, n_iters, ring, consume;
    volatile int * produced; volatile int * consumed; volatile int * stop;
    uint32_t * tbl; uint64_t cycles;
} hmx_pipe_job_t;

static void hmx_pipe_job_fn(void * data) {
    hmx_pipe_job_t * b = (hmx_pipe_job_t *) data;
    const uint64_t c0 = hex_get_cycles();
    int seq = 0;
    for (uint32_t it = 0; it < b->n_iters; ++it) {
        for (uint32_t c = 0; c < b->n_col; ++c) {
            for (uint32_t g0 = 0; g0 < b->n_dot; g0 += b->kg) {
                const uint32_t ng = (b->n_dot - g0 < b->kg) ? (b->n_dot - g0) : b->kg;
                const uint8_t * a = b->act + (size_t) g0 * 2048;
                const uint8_t * w = b->wgt + ((size_t) c * b->n_dot + g0) * 1024;
                uint8_t * o = b->planes + (size_t) (seq % b->ring) * 8192;
                if (b->consume) {
                    for (;;) {   // slot reuse: every consumer thread must be past slot (seq - ring)
                        int mn = 0x7fffffff;
                        for (uint32_t q = 0; q < b->consume; ++q) { const int v = b->consumed[q * 32]; if (v < mn) mn = v; }
                        if (seq - mn < (int) b->ring) break;
                    }
                }
                asm volatile("mxclracc\n");
                uint32_t left = ng;
                while (left) {
                    const uint32_t n = left > 32 ? 32 : left;
                    const uint32_t ra = ((n - 1) << 11) | 31;
                    const uint32_t rw = (uint32_t) (1024 * n - 128);
                    asm volatile("{\n activation.ub = mxmem(%0, %1):deep:cm\n weight.b = mxmem(%2, %3)\n}\n" : : "r"(a), "r"(ra), "r"(w), "r"(rw));
                    a += 2048 * n; w += 1024 * n; left -= n;
                }
                for (int k = 0; k < 4; ++k) {
                    const uint32_t zero = 0;
                    asm volatile("bias = mxmem2(%0)\n" : : "r"(b->tbl + k * 64));
                    if (k < 3) asm volatile("{ mxmem(%0, %1):after:retain:cm.ub = acc }\n" : : "r"(o + k * 2048), "r"(zero));
                    else       asm volatile("{ mxmem(%0, %1):after:cm.ub = acc }\n" : : "r"(o + k * 2048), "r"(zero));
                }
                ++seq;
                *b->produced = seq;
            }
        }
    }
    b->cycles = hex_get_cycles() - c0;
    *b->stop = 1;
}

typedef struct {
    uint8_t * planes; float * accs; const float * rscale; const float * cscale; uint32_t ring;
    volatile int * produced; volatile int * consumed; volatile int * stop; volatile int * done;   // done[ring]
    uint64_t busy[HTP_MAX_NTHREADS]; uint32_t mism;
    const uint8_t * ref_act; const uint8_t * ref_wgt; uint32_t ref_kt;
} hmx_pipe_wr_t;

static void hmx_pipe_consumer_fn(unsigned int n, unsigned int i, void * data) {
    hmx_pipe_wr_t * w = (hmx_pipe_wr_t *) data;
    uint64_t busy = 0;
    int seq = 0;
    const uint32_t groups = 16 / n;                      // groups of 4 rows (n = 1, 2, 4, 8)
    const HVX_Vector cs = *(const HVX_Vector *) w->cscale;   // 32 column scales
    for (;;) {
        while (seq >= *w->produced) {
            if (*w->stop && seq >= *w->produced) { w->busy[i] = busy; return; }
        }
        const uint64_t t0 = hex_get_cycles();
        const uint8_t * pl = w->planes + (size_t) (seq % w->ring) * 8192;
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t grp = i * groups + g;         // rows 4*grp .. 4*grp+3
            const HVX_Vector v0 = *(const HVX_Vector *) (pl + 0 * 2048 + grp * 128);
            const HVX_Vector v1 = *(const HVX_Vector *) (pl + 1 * 2048 + grp * 128);
            const HVX_Vector v2 = *(const HVX_Vector *) (pl + 2 * 2048 + grp * 128);
            const HVX_Vector v3 = *(const HVX_Vector *) (pl + 3 * 2048 + grp * 128);
            HVX_VectorPair p0 = Q6_W_vshuff_VVR(v1, v0, -1);
            HVX_VectorPair p1 = Q6_W_vshuff_VVR(v3, v2, -1);
            HVX_VectorPair p2 = Q6_W_vshuff_VVR(Q6_V_lo_W(p1), Q6_V_lo_W(p0), -2);
            HVX_VectorPair p3 = Q6_W_vshuff_VVR(Q6_V_hi_W(p1), Q6_V_hi_W(p0), -2);
            const HVX_Vector r4[4] = { Q6_V_lo_W(p2), Q6_V_hi_W(p2), Q6_V_lo_W(p3), Q6_V_hi_W(p3) };   // int32 rows 4*grp .. 4*grp+3
            HVX_Vector * ap = (HVX_Vector *) (w->accs + (size_t) grp * 4 * 32);
            for (int r = 0; r < 4; ++r) {
                const HVX_Vector rs = hvx_vec_splat_f32(w->rscale[grp * 4 + r]);
                const HVX_Vector sc = hvx_vec_mul_f32_f32(rs, cs);
                ap[r] = hvx_vec_add_f32_f32(ap[r], hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(r4[r]), sc));
            }
        }
        busy += hex_get_cycles() - t0;
        ++seq;
        w->consumed[i * 32] = seq;   // per-thread progress (own cache-line-sized slot in VTCM), no atomics
    }
}


// --- variant 22: HVX post-process in isolation (no HMX). n_row = stage (1 shuffle+store int32, 2 +f32 convert/scale, 3 +accumulate), n_col = slots, n_dot unused,
//     n_writers threads; each thread handles 16/n_writers groups of 4 rows of each slot. Returns wall cycles in *cycles.
typedef struct { uint8_t * planes; float * accs; const float * rscale; const float * cscale; uint32_t stage, slots, iters; } hmx_pp_t;
static void hmx_pp_fn(unsigned int n, unsigned int i, void * data) {
    hmx_pp_t * w = (hmx_pp_t *) data;
    const uint32_t groups = 16 / n;
    const HVX_Vector cs = *(const HVX_Vector *) w->cscale;
    for (uint32_t it = 0; it < w->iters; ++it) for (uint32_t sl = 0; sl < w->slots; ++sl) {
        const uint8_t * pl = w->planes + (size_t) sl * 8192;
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t grp = i * groups + g;
            const HVX_Vector v0 = *(const HVX_Vector *) (pl + 0 * 2048 + grp * 128);
            const HVX_Vector v1 = *(const HVX_Vector *) (pl + 1 * 2048 + grp * 128);
            const HVX_Vector v2 = *(const HVX_Vector *) (pl + 2 * 2048 + grp * 128);
            const HVX_Vector v3 = *(const HVX_Vector *) (pl + 3 * 2048 + grp * 128);
            HVX_VectorPair p0 = Q6_W_vshuff_VVR(v1, v0, -1);
            HVX_VectorPair p1 = Q6_W_vshuff_VVR(v3, v2, -1);
            HVX_VectorPair p2 = Q6_W_vshuff_VVR(Q6_V_lo_W(p1), Q6_V_lo_W(p0), -2);
            HVX_VectorPair p3 = Q6_W_vshuff_VVR(Q6_V_hi_W(p1), Q6_V_hi_W(p0), -2);
            const HVX_Vector r4[4] = { Q6_V_lo_W(p2), Q6_V_hi_W(p2), Q6_V_lo_W(p3), Q6_V_hi_W(p3) };
            HVX_Vector * ap = (HVX_Vector *) (w->accs + (size_t) grp * 4 * 32);
            for (int r = 0; r < 4; ++r) {
                if (w->stage == 1) {
                    ap[r] = r4[r];
                } else {
                    const HVX_Vector rs = hvx_vec_splat_f32(w->rscale[grp * 4 + r]);
                    const HVX_Vector sc = hvx_vec_mul_f32_f32(rs, cs);
                    const HVX_Vector pr = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(r4[r]), sc);
                    if (w->stage == 2) ap[r] = pr; else ap[r] = hvx_vec_add_f32_f32(ap[r], pr);
                }
            }
        }
    }
}


// variant 23: optimized whole-K post-process, no accumulate: per 64x32 slot: merge 4 planes -> int32 -> f32 -> *(rowscale*colscale) -> fp16 tile pairs (2 rows per vector)
//   scale vectors (rowscale[r] * colscale[0..31]) are precomputed once per slot column (w->accs scratch holds 64 vectors) -> the per-slot loop is loads + 4 vshuff + 4 cvt + 4 vmpy + 2 pack
static void hmx_pp2_fn(unsigned int n, unsigned int i, void * data) {
    hmx_pp_t * w = (hmx_pp_t *) data;
    const uint32_t groups = 16 / n;
    HVX_Vector * restrict sv = (HVX_Vector *) (w->accs + 64 * 32 * 2);           // 64 scale vectors (8 KB)
    if (i == 0) {
        const HVX_Vector cs = *(const HVX_Vector *) w->cscale;
        for (uint32_t r = 0; r < 64; ++r) sv[r] = hvx_vec_mul_f32_f32(hvx_vec_splat_f32(w->rscale[r]), cs);
    }
    for (uint32_t it = 0; it < w->iters; ++it) for (uint32_t sl = 0; sl < w->slots; ++sl) {
        const HVX_Vector * restrict pl = (const HVX_Vector *) (w->planes + (size_t) sl * 8192);
        HVX_Vector * restrict out = (HVX_Vector *) (w->accs + (size_t) sl * 0);
        #pragma unroll(2)
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t grp = i * groups + g;
            const HVX_Vector v0 = pl[0 * 16 + grp], v1 = pl[1 * 16 + grp], v2 = pl[2 * 16 + grp], v3 = pl[3 * 16 + grp];
            HVX_VectorPair p0 = Q6_W_vshuff_VVR(v1, v0, -1);
            HVX_VectorPair p1 = Q6_W_vshuff_VVR(v3, v2, -1);
            HVX_VectorPair p2 = Q6_W_vshuff_VVR(Q6_V_lo_W(p1), Q6_V_lo_W(p0), -2);
            HVX_VectorPair p3 = Q6_W_vshuff_VVR(Q6_V_hi_W(p1), Q6_V_hi_W(p0), -2);
            const HVX_Vector q0 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(Q6_V_lo_W(p2)), sv[grp * 4 + 0]);
            const HVX_Vector q1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(Q6_V_hi_W(p2)), sv[grp * 4 + 1]);
            const HVX_Vector q2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(Q6_V_lo_W(p3)), sv[grp * 4 + 2]);
            const HVX_Vector q3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(Q6_V_hi_W(p3)), sv[grp * 4 + 3]);
            out[grp * 2 + 0] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
            out[grp * 2 + 1] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q3, q2));
        }
    }
}


// variant 24: Q4M tile expand cost. Raw tile (640 B: 512 B nibble pairs + 32 B multipliers) -> int8 HMX weight tile (1 KB). n_col tiles per thread, n_writers threads.
typedef struct { const uint8_t * raw; uint8_t * out; uint32_t tiles, iters; } hmx_ex_t;
static inline void q4m_expand_tile(uint8_t * restrict dst, const uint8_t * restrict src) {
    const HVX_Vector mask = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector vm = *(const HVX_Vector *) (src + 512);
    HVX_VectorPair p1 = Q6_W_vshuff_VVR(vm, vm, -1);
    HVX_VectorPair p2 = Q6_W_vshuff_VVR(Q6_V_lo_W(p1), Q6_V_lo_W(p1), -2);
    HVX_Vector m4 = Q6_V_lo_W(p2);
    HVX_Vector m8 = Q6_Vb_vadd_VbVb(m4, m4); m8 = Q6_Vb_vadd_VbVb(m8, m8); m8 = Q6_Vb_vadd_VbVb(m8, m8);
    #pragma unroll(4)
    for (int t = 0; t < 4; t++) {
        HVX_Vector v = *(const HVX_Vector *) (src + t * 128);
        HVX_Vector lo = Q6_V_vand_VV(v, mask);
        HVX_Vector hi = Q6_Vub_vlsr_VubR(v, 4);
        HVX_VectorPair pl = Q6_Wuh_vmpy_VubVub(lo, m4);
        HVX_VectorPair ph = Q6_Wuh_vmpy_VubVub(hi, m4);
        HVX_Vector rl = Q6_Vb_vshuffe_VbVb(Q6_V_hi_W(pl), Q6_V_lo_W(pl));
        HVX_Vector rh = Q6_Vb_vshuffe_VbVb(Q6_V_hi_W(ph), Q6_V_lo_W(ph));
        *(HVX_Vector *) (dst + t * 128)       = Q6_Vb_vsub_VbVb(rl, m8);
        *(HVX_Vector *) (dst + (t + 4) * 128) = Q6_Vb_vsub_VbVb(rh, m8);
    }
}
static void hmx_ex_fn(unsigned int n, unsigned int i, void * data) {
    hmx_ex_t * w = (hmx_ex_t *) data;
    const uint32_t per = w->tiles / n;
    for (uint32_t it = 0; it < w->iters; ++it)
        for (uint32_t t = i * per; t < (i + 1) * per; ++t) q4m_expand_tile(w->out + (size_t) t * 1024, w->raw + (size_t) t * 640);
}


// variant 25: board self-check of the real A8W8 chain: random fp16 act tiles + random Q4_0-layout tiles -> quant / expand / HMX / post, compared with a scalar float
// reference built from the SAME intermediate int8 data. Returns mismatch count in *writer_bytes (0 = ok), rel err x 1e6 in *usec.
static int hmx_a8w8_selfcheck(struct htp_context * ctx, uint32_t nb64, uint32_t nct, uint32_t tile_k, uint64_t * cycles, uint64_t * usec, uint64_t * writer_bytes) {
    uint8_t * base = (uint8_t *) ctx->vtcm_base;
    const size_t act16_b = (size_t) nb64 * 2 * tile_k * 2048;
    uint8_t * act16 = base + 4096;
    uint8_t * act8  = act16 + act16_b;
    float *   rscale = (float *) (act8 + (size_t) nb64 * tile_k * 2048);
    uint8_t * raw   = (uint8_t *) hex_align_up((size_t) (rscale + nb64 * 64), 2048);
    uint8_t * w8    = raw + (size_t) nct * tile_k * 640;
    float *   sw    = (float *) (w8 + (size_t) nct * tile_k * 1024);
    int32_t * cs    = (int32_t *) (sw + nct * 32);
    uint8_t * planes = (uint8_t *) hex_align_up((size_t) (cs + nct * 32), 2048);
    __fp16 *  out   = (__fp16 *) (planes + (size_t) nb64 * nct * 8192);
    uint32_t * tbl  = (uint32_t *) base;
    static const uint32_t ee1x[4] = { 24, 16, 8, 0 };
    for (int t = 0; t < 4; ++t) for (int q = 0; q < 64; ++q) tbl[t * 64 + q] = (q < 32) ? (ee1x[t] << 10) : 0;
    uint32_t sd = 0x2468ace1u;
    for (size_t i = 0; i < act16_b / 2; ++i) { sd = sd * 1664525u + 1013904223u; ((__fp16 *) act16)[i] = (__fp16) ((float) ((sd >> 8) & 0xffff) / 65536.f - 0.5f); }
    for (uint32_t c = 0; c < nct; ++c) for (uint32_t kt = 0; kt < tile_k; ++kt) {
        uint8_t * t = raw + ((size_t) c * tile_k + kt) * 640;
        for (int q = 0; q < 512; ++q) { sd = sd * 1664525u + 1013904223u; t[q] = (uint8_t) (sd >> 24); }
        __fp16 * d = (__fp16 *) (t + 512);
        for (int n = 0; n < 32; ++n) { sd = sd * 1664525u + 1013904223u; const float mg = 0.01f * (1.f + (float) ((sd >> 8) % 8)); d[n] = (__fp16) (((sd >> 4) & 1) ? mg : -mg); }
        for (int q = 576; q < 640; ++q) t[q] = 0;
    }
    a8w8_state_t st = { .traces = ctx->trace, .act16 = (const __fp16 *) act16, .act8 = act8, .rscale = rscale, .tile_k = tile_k, .n_blocks64 = nb64, .n_row_tiles = nb64 * 2 };
    st.n_tasks = 1; a8w8_quant_worker(1, 0, &st);
    st.raw = raw; st.aligned = 640; st.w8 = w8; st.sw = sw; st.cs = cs; st.n_col_tiles = nct; st.n_tasks = 1; a8w8_expand_worker(1, 0, &st);
    volatile int produced = 0;
    a8w8_job_t job = { .act8 = act8, .w8 = w8, .planes = planes, .tbl = tbl, .tile_k = tile_k, .n_blocks64 = nb64, .n_col_tiles = nct, .ring = nb64 * nct, .produced = &produced };
    const uint64_t c0 = hex_get_cycles();
    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(a8w8_hmx_job, &job));
    hmx_queue_pop(ctx->hmx_queue);
    a8w8_post_state_t ps = { .planes = planes, .sw = sw, .cs = cs, .rscale = rscale, .out = out, .n_col_tiles = nct, .n_blocks64 = nb64, .n_row_tiles = nb64 * 2, .ring = nb64 * nct, .n_tasks = 1, .traces = ctx->trace };
    a8w8_post_worker(1, 0, &ps);
    *cycles = hex_get_cycles() - c0;
    // checks: (1) int32 accumulators of every slot vs scalar (act8 x w8)  (2) final fp16 vs float
    uint64_t bad_int = 0; double se = 0, sr = 0;
    for (uint32_t r = 0; r < nb64; ++r) for (uint32_t c = 0; c < nct; ++c) {
        const uint8_t * pl = planes + ((size_t) r * nct + c) * 8192;
        for (uint32_t row = 0; row < 64; row += 5) for (uint32_t n = 0; n < 32; ++n) {
            int64_t acc = 0;
            for (uint32_t kt = 0; kt < tile_k; ++kt) for (uint32_t k = 0; k < 32; ++k)
                acc += (int) act8[((size_t) r * tile_k + kt) * 2048 + row * 32 + k] * (int8_t) w8[((size_t) c * tile_k + kt) * 1024 + (k / 4) * 128 + n * 4 + (k % 4)];
            uint32_t g = 0; for (int p = 0; p < 4; ++p) g |= (uint32_t) pl[p * 2048 + row * 32 + n] << (8 * p);
            if ((int32_t) g != (int32_t) acc) bad_int++;
            // float reference of the final value
            int64_t wsum = 0; for (uint32_t kt = 0; kt < tile_k; ++kt) for (uint32_t k = 0; k < 32; ++k) wsum += (int8_t) w8[((size_t) c * tile_k + kt) * 1024 + (k / 4) * 128 + n * 4 + (k % 4)];
            const double ref = (double) (acc - 128 * wsum) * (double) rscale[r * 64 + row] * (double) sw[c * 32 + n];
            const uint32_t rt = r * 2 + row / 32, rw = row % 32;
            const double got = (double) (float) out[((size_t) rt * nct + c) * 1024 + (rw / 2) * 64 + 2 * n + (rw & 1)];
            se += (got - ref) * (got - ref); sr += ref * ref;
        }
    }
    *usec = (uint64_t) (1e6 * sqrt(se / (sr > 0 ? sr : 1)));
    *writer_bytes = bad_int;
    return 0;
}

int htp_hmx_bench_run(struct htp_context * ctx, uint32_t variant, uint32_t n_row, uint32_t n_col, uint32_t n_dot,
                      uint32_t n_iters, uint32_t n_writers, uint64_t * cycles, uint64_t * usec, uint64_t * writer_bytes) {
    if (variant == 25) { return hmx_a8w8_selfcheck(ctx, n_row, n_col, n_dot, cycles, usec, writer_bytes); }
    if (!ctx || !ctx->hmx_queue || !ctx->vtcm_base || n_row == 0 || n_col == 0 || n_dot == 0 || n_iters == 0 ||
        n_dot > 320 || n_writers > HTP_MAX_NTHREADS || variant > 25) {
        return -1;
    }
    const size_t wr_window = n_writers ? (size_t) (1u << 20) : 0;
    const size_t act_bytes = (size_t) n_row * n_dot * HTP_MM_HMX_TILE_SIZE;
    const size_t wgt_bytes = (size_t) n_col * n_dot * HTP_MM_HMX_TILE_SIZE;
    size_t       out_bytes = (size_t) n_row * n_col * HTP_MM_HMX_TILE_SIZE;
    if (variant >= 18 && out_bytes < 16 * 2048) out_bytes = 16 * 2048;
    if (variant == 24) out_bytes = (size_t) n_col * 2048 + 4096;
    if (variant == 21 || variant == 22) out_bytes = 8 * 8192 + 64 * 32 * 4 + 1024 + 4096 * 2;   // 4 sets x 4 byte planes x 2 KB
    uint8_t *    base      = ctx->vtcm_base;
    const size_t act_off   = 2048;
    const size_t wgt_off   = act_off + act_bytes;
    const size_t out_off   = wgt_off + wgt_bytes;
    const size_t wr_off    = ctx->vtcm_size_full - wr_window;
    if (out_off + out_bytes > wr_off) {
        return -2;
    }

    __fp16 * scales = (__fp16 *) base;
    hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
    __fp16 * act = (__fp16 *) (base + act_off);
    __fp16 * wgt = (__fp16 *) (base + wgt_off);
    __fp16 * out = (__fp16 *) (base + out_off);
    uint32_t seed = 0x12345678u;
    for (size_t i = 0; i < (act_bytes + wgt_bytes) / 2; ++i) {
        seed = seed * 1664525u + 1013904223u;
        ((uint16_t *) (base + act_off))[i] = (uint16_t) (0x3800u | ((seed >> 16) & 0x3ffu));
    }

    if (variant >= 18) {
        // random u8 activations, random weight bytes (int8 for v18, two int4 nibbles for v19/v20)
        uint32_t sd = 0x9e3779b9u;
        for (size_t i = 0; i < act_bytes + wgt_bytes; ++i) {
            sd = sd * 1664525u + 1013904223u;
            ((uint8_t *) (base + act_off))[i] = (uint8_t) (sd >> 24);
        }
        // int sequences address act/wgt per row tile / col unit with their own strides: re-derive extents below
    }
    if (variant == 25) {
        return hmx_a8w8_selfcheck(ctx, n_row, n_col, n_dot, cycles, usec, writer_bytes);
    }
    if (variant == 24) {
        static hmx_ex_t ex;
        uint8_t * raw = (uint8_t *) out;                 // out area: raw tiles then expanded tiles (n_col tiles)
        ex = (hmx_ex_t) { .raw = raw, .out = raw + (size_t) n_col * 640 + 128, .tiles = n_col, .iters = n_iters };
        for (size_t q = 0; q < (size_t) n_col * 640; ++q) raw[q] = (uint8_t) (q * 2654435761u >> 24);
        if (n_col % (n_writers ? n_writers : 1)) return -4;
        const uint64_t c0 = hex_get_cycles();
        work_queue_run(ctx->work_queue, hmx_ex_fn, &ex, n_writers ? n_writers : 1);
        *cycles = hex_get_cycles() - c0; *usec = 0; *writer_bytes = 0;
        return 0;
    }
    if (variant == 22) {
        static hmx_pp_t pp;
        uint8_t * pl = (uint8_t *) out;
        float * accs = (float *) (base + out_off + 8 * 8192);
        float * cs   = accs + 64 * 32;
        float * rs   = cs + 32;
        for (int q = 0; q < 32; ++q) cs[q] = 1e-3f * (q + 1);
        for (int q = 0; q < 64; ++q) rs[q] = 1e-2f * (q + 1);
        memset(pl, 0x5a, 8 * 8192);
        pp = (hmx_pp_t) { .planes = pl, .accs = accs, .rscale = rs, .cscale = cs, .stage = n_row, .slots = 8, .iters = n_iters };
        if (16 % (n_writers ? n_writers : 1)) return -4;
        const uint64_t c0 = hex_get_cycles();
        work_queue_run(ctx->work_queue, n_dot == 23 ? hmx_pp2_fn : hmx_pp_fn, &pp, n_writers ? n_writers : 1);
        *cycles = hex_get_cycles() - c0; *usec = 0; *writer_bytes = 0;
        return 0;
    }
    if (variant == 21) {
        // n_row = KG (k-tiles per readout), n_col = column units, n_dot = k-tiles per unit, n_writers = HVX consumers (0 = HMX only)
        const uint32_t kg = n_row;
        if (kg == 0 || kg > n_dot || (n_writers && (16 % n_writers))) return -4;
        volatile int * flags = (volatile int *) (base + out_off + 8 * 8192 + 64 * 32 * 4 + 1024);   // VTCM tail of the out area
        volatile int *pr = flags, *st = flags + 32, *co = flags + 64, *dn = flags;
        *pr = 0; *st = 0; for (int q = 0; q < 16; ++q) co[q * 32] = 0;
        uint8_t * pl = (uint8_t *) out;                        // 8 slots x 8 KB
        uint32_t * tbl = (uint32_t *) base + 0;   // 1 KB of scale/bias tables at the VTCM base (below act_off)
        static const uint32_t ee1x[4] = { 24, 16, 8, 0 };
        for (int k = 0; k < 4; ++k) for (int q = 0; q < 64; ++q) tbl[k * 64 + q] = (q < 32) ? (ee1x[k] << 10) : 0;
        float * accs = (float *) (base + out_off + 8 * 8192);
        float * cs   = accs + 64 * 32;
        float * rs   = cs + 32;
        for (int q = 0; q < 32; ++q) cs[q] = 1e-3f * (q + 1);
        for (int q = 0; q < 64; ++q) rs[q] = 1e-2f * (q + 1);
        memset(accs, 0, 64 * 32 * sizeof(float));
        hmx_pipe_job_t pj = { .act = (const uint8_t *) act, .wgt = (const uint8_t *) wgt, .planes = pl, .n_dot = n_dot, .n_col = n_col, .kg = kg,
                              .n_iters = n_iters, .ring = 8, .consume = n_writers, .produced = pr, .consumed = co, .stop = st, .tbl = tbl, .cycles = 0 };
        static hmx_pipe_wr_t pw;
        memset(&pw, 0, sizeof(pw));
        pw.planes = pl; pw.accs = accs; pw.rscale = rs; pw.cscale = cs; pw.ring = 8; pw.produced = pr; pw.consumed = co; pw.stop = st; pw.done = dn;
        if (!hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_pipe_job_fn, &pj))) return -3;
        if (n_writers) work_queue_run(ctx->work_queue, hmx_pipe_consumer_fn, &pw, n_writers);
        hmx_queue_pop(ctx->hmx_queue);
        if (n_iters == 1 && n_col == 1) {
            // slot (n_dot/kg - 1) % ring holds the last readout of the single column unit: verify its 4 planes (1x exponents 24/16/8/0) against a scalar reference
            const uint32_t ns = (n_dot + kg - 1) / kg; const uint32_t last = ns - 1; const uint32_t ng = n_dot - last * kg;
            const uint8_t * pls = pl + (size_t) (last % 8) * 8192;
            const uint8_t * ab = (const uint8_t *) act + (size_t) last * kg * 2048;
            const uint8_t * wb8 = (const uint8_t *) wgt + (size_t) last * kg * 1024;
            uint32_t bad = 0;
            for (uint32_t row = 0; row < 64; ++row) for (uint32_t col = 0; col < 32; ++col) {
                int64_t acc = 0;
                for (uint32_t kt = 0; kt < ng; ++kt) for (uint32_t k = 0; k < 32; ++k)
                    acc += (int) ab[(size_t) kt * 2048 + row * 32 + k] * (int8_t) wb8[(size_t) kt * 1024 + (k / 4) * 128 + col * 4 + (k % 4)];
                uint32_t g = 0; for (int k = 0; k < 4; ++k) g |= (uint32_t) pls[k * 2048 + row * 32 + col] << (8 * k);
                if ((int32_t) g != (int32_t) acc) bad++;
            }
            pw.mism = bad;
        }
        uint64_t bsum = 0; for (uint32_t q = 0; q < n_writers; ++q) bsum += pw.busy[q];
        *cycles = pj.cycles; *usec = bsum; *writer_bytes = pw.mism;   // usec field carries total HVX busy cycles; writer_bytes = int32 mismatches
        return 0;
    }
    atomic_int stop;
    atomic_store(&stop, 0);
    hmx_bench_job_t job = {
        .variant = variant, .n_row = n_row, .n_col = n_col, .n_dot = n_dot, .n_iters = n_iters,
        .act = act, .wgt = wgt, .out = out, .scales = scales, .stop = n_writers ? &stop : NULL,
        .cycles = 0, .usec = 0,
    };
    hmx_bench_wr_t wr = { .base = base + wr_off, .span = n_writers ? wr_window / n_writers : 0, .stop = &stop };
    memset(wr.bytes, 0, sizeof(wr.bytes));

    if (!hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_bench_job_fn, &job))) {
        return -3;
    }
    if (n_writers) {
        work_queue_run(ctx->work_queue, hmx_bench_writer_fn, &wr, n_writers);
    }
    hmx_queue_pop(ctx->hmx_queue);

    if (variant >= 18) {
        // verify tile (row tile 0, col unit n_col-1 is the last one written; recompute that one) against a scalar reference
        const uint32_t nsets = (variant == 18) ? 1 : (variant == 19) ? 2 : 4;
        const size_t   wt    = (variant == 20) ? 2048 : 1024;
        const uint32_t c     = n_col - 1, r = n_row - 1;
        const uint8_t * ab = (const uint8_t *) act + (size_t) r * n_dot * 2048;
        const uint8_t * wb8 = (const uint8_t *) wgt + (size_t) c * n_dot * wt;
        const uint8_t * ob = (const uint8_t *) out;
        uint64_t bad = 0;
        for (uint32_t s = 0; s < nsets; ++s) {
            for (uint32_t row = 0; row < 64; ++row) {
                for (uint32_t n = 0; n < 32; ++n) {
                    int64_t acc = 0;
                    for (uint32_t kt = 0; kt < n_dot; ++kt) {
                        const uint8_t * at = ab + (size_t) kt * 2048;
                        const uint8_t * wtile = wb8 + (size_t) kt * wt + ((s >> 1) ? 1024 : 0);
                        for (uint32_t k = 0; k < 32; ++k) {
                            const uint8_t wbyte = wtile[(k / 4) * 128 + n * 4 + (k % 4)];
                            int wv;
                            if (variant == 18) wv = (int8_t) wbyte;
                            else wv = (s & 1) ? ((int8_t) wbyte) >> 4 : ((int8_t) (wbyte << 4)) >> 4;
                            acc += (int) at[row * 32 + k] * wv;
                        }
                    }
                    uint32_t g = 0;
                    for (int k = 0; k < 4; ++k) g |= (uint32_t) ob[(size_t) (s * 4 + k) * 2048 + row * 32 + n] << (8 * k);
                    if ((int32_t) g != (int32_t) acc) bad++;
                }
            }
        }
        // cycles = job.cycles; usec = job.usec; writer_bytes = bad; // mismatching int32 results (0 = bit-exact) for verified variants
        return 0;
    }

    uint64_t wb = 0;
    for (uint32_t i = 0; i < n_writers; ++i) {
        wb += wr.bytes[i];
    }
    // cycles = job.cycles; usec = job.usec; writer_bytes = wb;
    return 0;
}
