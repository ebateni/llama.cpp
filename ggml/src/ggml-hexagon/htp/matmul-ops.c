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
#include "hvx-sigmoid.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "matmul-ops.h"
#include "htp-vtcm.h"

typedef struct {
    float        *dst;
    dma_addr_t    bias_addr;
    size_t        bias_bytes;
    dma_addr_t    act_dma_addr;
    dma_addr_t    weight;
    dma_queue *   weight_dma;
    int           m;
    int           k;
    int           n;
    int           act_stride;
    uint32_t      act_elem_size;  // 4=F32 src1, 2=F16 src1
    int           weight_stride;
    int           dst_stride;
    uint32_t      bias_stride;
    int           ne02;
    int           ne03;
    int           ne12;
    int           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        act_nb2;
    size_t        act_nb3;
    size_t        bias_nb2;
    size_t        bias_nb3;
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
    uint8_t * vtcm_act;
    uint8_t * vtcm_bias;
    uint8_t * vtcm_dst;
    uint8_t * vtcm_act_raw;

    // Cached strides
    uint32_t vtcm_src0_stride;
    uint32_t vtcm_act_stride;
    uint32_t vtcm_act_raw_stride;

    // Cached thread offsets/sizes
    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_act_size_per_thread;
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
    const struct htp_tensor * restrict  act = src1;                 \
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
#include "hmx-mm-kernels-int8.h"

// Specialized repacked matmul macros
#define MATMUL_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                                       \
static void hvx_mm_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                                 \
    htp_matmul_preamble;                                                                                                                   \
                                                                                                                                           \
    const uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;                                                               \
    const uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);                                               \
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
    const size_t act_row_size = nb11;                                                                                                      \
    const size_t act_stride = mmctx->vtcm_act_stride;                                                                                      \
    const size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;                                                          \
                                                                                                                                           \
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;                                          \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                          \
    uint8_t * restrict act_data = mmctx->vtcm_act;                                                                                         \
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
        for (; ir1 + 1 < act_nrows; ir1 += 2) {                                                                                            \
            const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (ir1+0) * act_stride);                                       \
            const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (ir1+1) * act_stride);                                       \
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
            DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, src2_ptr0, src2_ptr1);                               \
        }                                                                                                                                  \
                                                                                                                                           \
        for (; ir1 < act_nrows; ++ir1) {                                                                                                   \
            const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);                                            \
            float * restrict dst_row          = (float *) (dst->data + ((cur_m_start + ir1) * dst_row_size));                              \
            float * dst_ptr = &dst_row[ct * 32];                                                                                           \
                                                                                                                                           \
            const float * src2_ptr = NULL;                                                                                                 \
            if (src2) {                                                                                                                    \
                const float * restrict src2_row = (const float *) ((const uint8_t *) src2->data + ((cur_m_start + ir1) * src2_stride));    \
                src2_ptr = &src2_row[ct * 32];                                                                                             \
            }                                                                                                                              \
            DOT_2X1(ne10, dst_ptr, w_tile, act_col, valid_rows, src2_ptr);                                                                 \
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
    const size_t act_row_size = nb11;                                                                                    \
    const size_t act_stride = mmctx->vtcm_act_stride;                                                                    \
                                                                                                                         \
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_thread * ith;                                   \
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                 \
    uint8_t * act_data = mmctx->vtcm_act;                                                                                \
                                                                                                                         \
    float * tmp = (float *) vtcm_dst_ptr;                                                                                \
                                                                                                                         \
    const dma_addr_t src0_row = src0->data;                                                                              \
                                                                                                                         \
    const uint8_t * restrict act_col = (const uint8_t *) act_data;                                                       \
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
            float * vtcm_bias_ptr = (float *) mmctx->vtcm_bias + src0_start_row;                                         \
            const dma_addr_t src2_addr = src2->data + src0_start_row * sizeof(float);                                    \
            int slice_size = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;                                          \
            if (slice_size > 0) {                                                                                        \
                dma_queue_push(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),                                           \
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
        DOT_2X1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL);                                                       \
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
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],                                                        \
                            (const uint8_t *) tmp,                                                                       \
                            (const uint8_t *) ((const float *) mmctx->vtcm_bias + src0_start_row),                       \
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
    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];                                                              \
    const size_t act_stride = mmctx->vtcm_act_stride;                                                                             \
                                                                                                                                  \
    uint8_t * restrict vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                               \
    uint8_t * restrict act_data        = mmctx->vtcm_act;                                                                         \
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
            for (; ir1 + 1 < act_nrows; ir1 += 2) {                                                                               \
                const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (ir1+0) * act_stride);                          \
                const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (ir1+1) * act_stride);                          \
                                                                                                                                  \
                float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));                                     \
                float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));                                     \
                float * dst_ptr0 = &dst_row0[ct * 32];                                                                            \
                float * dst_ptr1 = &dst_row1[ct * 32];                                                                            \
                                                                                                                                  \
                DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, NULL, NULL);                            \
            }                                                                                                                     \
                                                                                                                                  \
            for (; ir1 < act_nrows; ++ir1) {                                                                                      \
                const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);                               \
                float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));                                          \
                float * dst_ptr = &dst_row[ct * 32];                                                                              \
                DOT_2X1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL);                                                        \
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
MATMUL_2D_REPACKED_IMPL(q3_k,       512,  tiled_vec_dot_q3_k_32x2,  tiled_vec_dot_q3_k_32x1)
MATMUL_2D_REPACKED_IMPL(q2_k,       512,  tiled_vec_dot_q2_k_32x2,  tiled_vec_dot_q2_k_32x1)
MATMUL_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

static void hvx_mm_transfer_act_dma(
    struct htp_ops_context * octx,
    const struct htp_mm_kernel_params * kparams,
    const struct htp_tensor * act,
    uint8_t * dst_base,
    size_t dst_row_size,
    uint32_t m_start,
    uint32_t m_rows
) {
    if (m_rows == 0) {
        return;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    const uint32_t ne0 = act->ne[0];
    const size_t elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const size_t row_bytes = ne0 * elem_size;
    const size_t act_nb1 = act->nb[1];
    const dma_addr_t act_base = act->data;

    const bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    if (is_contiguous) {
        const dma_addr_t src_addr = act_base + m_start * act_nb1;
        dma_queue_push(dma_q, dma_make_data(dst_base, src_addr),
                       dst_row_size, act_nb1, row_bytes, m_rows);
        dma_queue_pop(dma_q);
    } else {
        const uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        const bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        for (uint32_t ir = 0; ir < m_rows; ++ir) {
            const uint32_t ir1 = m_start + ir;
            uint32_t i11, i12, i13;
            if (use_fastdiv) {
                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                const uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = fastdiv(rem, &kparams->div_ne1);
                i11 = rem - i12 * act->ne[1];
            } else {
                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                const uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = act->ne[1] ? rem / act->ne[1] : 0;
                i11 = rem - i12 * act->ne[1];
            }
            const dma_addr_t row_src = act_base + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * row_dst = dst_base + ir * dst_row_size;
            dma_queue_push(dma_q, dma_make_data(row_dst, row_src),
                           dst_row_size, act_nb1, row_bytes, 1);
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
    uint8_t * restrict dst = mmctx->vtcm_act;                                                              \
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
QUANTIZE_IMPL(quantize_f32_q8_1_s16_tiled, "quantize-f32-q8_1_s16_tiled", quantize_f32_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_f32,        "quantize-f32-f32",        quantize_f32_f32_kernel,        mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f32_f16,        "quantize-f32-f16",        quantize_f32_f16_kernel,        mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f16_f16,        "quantize-f16-f16",        quantize_f16_f16_kernel,        mmctx->vtcm_act_stride)

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
        mmctx->vtcm_act,
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
        mmctx->vtcm_act,
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

static void quantize_f32_q8_1_s16_tiled_block(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_mm_context * mmctx = data;
    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        return;
    }
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_block_kernel(
        (const float *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
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

// q8_1 for weight types with offsets (q8_1_s16 for Q2_K), otherwise q8_0
static inline worker_callback_t htp_mm_act_quant_row_func(int weight_type) {
    if (weight_type == HTP_TYPE_Q2_K) {
        return quantize_f32_q8_1_s16_tiled;
    }
    return htp_mm_weight_has_offset(weight_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

static inline worker_callback_t htp_mm_act_quant_block_func(int weight_type) {
    if (weight_type == HTP_TYPE_Q2_K) {
        return quantize_f32_q8_1_s16_tiled_block;
    }
    return htp_mm_weight_has_offset(weight_type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
}

#define MATVEC_2D_REPACKED_PK_IMPL(SUFFIX, TILE_SIZE, DOT_2X1)                                                              \
static void hvx_mv_2d_repacked_pk_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                               \
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
    const size_t act_row_size = nb11;                                                                                    \
    const size_t act_stride = mmctx->vtcm_act_stride;                                                                    \
                                                                                                                         \
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_thread * ith;                                   \
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                 \
    uint8_t * act_data = mmctx->vtcm_act;                                                                                \
                                                                                                                         \
    float * tmp = (float *) vtcm_dst_ptr;                                                                                \
                                                                                                                         \
    const dma_addr_t src0_row = src0->data;                                                                              \
                                                                                                                         \
    const uint8_t * restrict act_col = (const uint8_t *) act_data;                                                       \
    float * restrict dst_col          = (float *) dst->data;                                                             \
                                                                                                                         \
    const uint32_t tile_size = TILE_SIZE;                                                                                \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                     \
                                                                                                                         \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                    \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                    \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                  \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                          \
    const uint32_t pk_bytes = n_k_tiles_a * tile_size;                                                                                                             \
                                                                                                                         \
    uint32_t ct_start = src0_start_row / 32;                                                                             \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                        \
                                                                                                                         \
    uint32_t push_ct = ct_start;                                                                                         \
    if (src0_start_row < src0_end_row) {                                                                                 \
        if (src2) {                                                                                                      \
            float * vtcm_bias_ptr = (float *) mmctx->vtcm_bias + src0_start_row;                                         \
            const dma_addr_t src2_addr = src2->data + src0_start_row * sizeof(float);                                    \
            int slice_size = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;                                          \
            if (slice_size > 0) {                                                                                        \
                dma_queue_push(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),                                           \
                               slice_size * sizeof(float), slice_size * sizeof(float), slice_size * sizeof(float), 1);   \
                dma_queue_pop_nowait(dma_q);                                                                             \
            }                                                                                                            \
        }                                                                                                                \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                       \
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                      \
                           src0_row + push_ct * tile_row_stride), pk_bytes, pk_bytes, pk_bytes, 1); \
        }                                                                                                                \
    }                                                                                                                    \
                                                                                                                         \
    if (src0_start_row >= src0_end_row) {                                                                                \
        return;                                                                                                          \
    }                                                                                                                    \
                                                                                                                         \
    uint32_t ct = ct_start;                                                                                              \
    for (; ct < ct_end; ct++) {                                                                                          \
        const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;                                                      \
                                                                                                                         \
        float * dst_ptr = &tmp[ct * 32 - src0_start_row];                                                                \
        int valid_rows = (int)ne0 - (int)(ct * 32);                                                                      \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                        \
                                                                                                                         \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                           \
        DOT_2X1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL);                                                       \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                            \
                                                                                                                         \
        if (push_ct < ct_end) {                                                                                          \
            dma_queue_push(dma_q, dma_make_data((void *) w_tile, src0_row + push_ct * tile_row_stride),                  \
                           pk_bytes, pk_bytes, pk_bytes, 1);                                                             \
            push_ct++;                                                                                                   \
        }                                                                                                                \
    }                                                                                                                    \
                                                                                                                         \
    int copy_cnt = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;                                                    \
    if (copy_cnt > 0) {                                                                                                  \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);                                                       \
        if (octx->op == HTP_OP_MUL_MAT_CAP) {                                                                            \
            float cap_s1, cap_s2;                                                                                        \
            memcpy(&cap_s1, &octx->op_params[0], sizeof(float));                                                         \
            memcpy(&cap_s2, &octx->op_params[1], sizeof(float));                                                         \
            hvx_scale_offset_f32_aa((uint8_t *) tmp, (const uint8_t *) tmp, copy_cnt, cap_s1, 0.0f);                     \
            hvx_tanh_f32_aa((uint8_t *) tmp, (const uint8_t *) tmp, copy_cnt);                                           \
            hvx_scale_offset_f32_aa((uint8_t *) tmp, (const uint8_t *) tmp, copy_cnt, cap_s2, 0.0f);                     \
        }                                                                                                                \
        if (octx->op == HTP_OP_MUL_MAT_GELU_MUL) {                                                                       \
            /* dst = gelu(W x) * y for this thread's slice; y = src[3] (m=1, one row) */                                 \
            const float * restrict y_row = (const float *) octx->src[3]->data;                                           \
            float * restrict d_row = (float *) dst->data;                                                                \
            const int nv = copy_cnt / 32;                                                                                \
            const int nl = copy_cnt % 32;                                                                                \
            const HVX_Vector * restrict vt = (const HVX_Vector *) tmp;                                                   \
            for (int i = 0; i < nv; i++) {                                                                               \
                HVX_Vector r = hvx_vec_mul_f32_f32(hvx_vec_gelu_f32(vt[i]), hvx_vmemu(&y_row[src0_start_row + i * 32]));  \
                hvx_vec_store_u(&d_row[src0_start_row + i * 32], 32 * sizeof(float), r);                                 \
            }                                                                                                            \
            if (nl) {                                                                                                    \
                HVX_Vector r = hvx_vec_mul_f32_f32(hvx_vec_gelu_f32(vt[nv]), hvx_vmemu(&y_row[src0_start_row + nv * 32]));\
                hvx_vec_store_u(&d_row[src0_start_row + nv * 32], nl * sizeof(float), r);                                \
            }                                                                                                            \
        } else if (src2) {                                                                                               \
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],                                                        \
                            (const uint8_t *) tmp,                                                                       \
                            (const uint8_t *) ((const float *) mmctx->vtcm_bias + src0_start_row),                       \
                            copy_cnt);                                                                                   \
        } else {                                                                                                         \
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);                            \
        }                                                                                                                \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);                                                        \
    }                                                                                                                    \
}

#define MATMUL_NX_2D_REPACKED_PK_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                           \
static void hvx_mm_nx_2d_repacked_pk_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                     \
    struct htp_mm_context * mmctx = data;                                                                                         \
    struct htp_ops_context * octx = mmctx->octx;                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_weights = kparams->n_weights;                                                                                \
                                                                                                                                  \
    const struct htp_tensor * restrict act = octx->src[n_weights]; /* x */                                                        \
    const uint32_t ne10 = act->ne[0];                                                                                             \
    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];                                                              \
    const size_t act_stride = mmctx->vtcm_act_stride;                                                                             \
                                                                                                                                  \
    uint8_t * restrict vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                               \
    uint8_t * restrict act_data        = mmctx->vtcm_act;                                                                         \
                                                                                                                                  \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                        \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                          \
    const uint32_t pk_bytes = n_k_tiles_a * tile_size;                                                                                                                      \
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
                           src_w_row + push_ct * tile_row_stride), pk_bytes, pk_bytes, pk_bytes, 1);         \
        }                                                                                                                         \
                                                                                                                                  \
        uint32_t ct = ct_start;                                                                                                   \
        for (; ct < ct_end; ct++) {                                                                                               \
            const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;                                                           \
            int valid_rows = (int)ne01 - (int)(ct * 32);                                                                          \
            valid_rows = MIN(32, MAX(0, valid_rows));                                                                             \
                                                                                                                                  \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                \
            uint32_t ir1 = 0;                                                                                                     \
            for (; ir1 + 1 < act_nrows; ir1 += 2) {                                                                               \
                const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (ir1+0) * act_stride);                          \
                const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (ir1+1) * act_stride);                          \
                                                                                                                                  \
                float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));                                     \
                float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));                                     \
                float * dst_ptr0 = &dst_row0[ct * 32];                                                                            \
                float * dst_ptr1 = &dst_row1[ct * 32];                                                                            \
                                                                                                                                  \
                DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, NULL, NULL);                            \
            }                                                                                                                     \
                                                                                                                                  \
            for (; ir1 < act_nrows; ++ir1) {                                                                                      \
                const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);                               \
                float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));                                          \
                float * dst_ptr = &dst_row[ct * 32];                                                                              \
                DOT_2X1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL);                                                        \
            }                                                                                                                     \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                 \
                                                                                                                                  \
            if (push_ct < ct_end) {                                                                                               \
                dma_queue_push(dma_q, dma_make_data(w_tile, src_w_row + push_ct * tile_row_stride),                               \
                               pk_bytes, pk_bytes, pk_bytes, 1);                                             \
                push_ct++;                                                                                                        \
            }                                                                                                                     \
        }                                                                                                                         \
    }                                                                                                                             \
}

// Decode gate+up with the gelu-gated epilogue fused (q4_0, m=1). Gate and up tiles of one output
// tile go through the same DMA queue back to back; dst = gelu(gate) * up.
static void hvx_mm_nx_geglu_pk_q4_0(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const struct htp_tensor * restrict src_g = octx->src[0];
    const struct htp_tensor * restrict src_u = octx->src[1];
    const struct htp_tensor * restrict act   = octx->src[2];
    const struct htp_tensor * restrict dst   = octx->dsts[0];

    const uint32_t ne10 = act->ne[0];
    const uint8_t * restrict act_data = mmctx->vtcm_act;
    uint8_t * restrict vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    const uint32_t n_prefetch = kparams->n_prefetch;  // slots; one tile pair uses two
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    const uint32_t tile_size = 576;
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);
    const uint32_t n_k_tiles_a = ne10 / 32;
    const uint32_t slot_stride = n_k_tiles_a * aligned_tile_size;
    const uint32_t pk_bytes = n_k_tiles_a * tile_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    const uint32_t ne01 = src_g->ne[1];
    const uint32_t tile_row_stride = (src_g->ne[0] / 32) * tile_size;
    const dma_addr_t g_base = src_g->data;
    const dma_addr_t u_base = src_u->data;
    float * restrict dst_f = (float *) dst->data;

    uint32_t src0_start_row = 0;
    uint32_t src0_end_row   = ne01;
    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_can_row_partition(dst, sizeof(float));
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0,
                                                     octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        src0_start_row = range.start;
        src0_end_row   = range.start + range.count;
    }

    const uint32_t nrows = src0_end_row - src0_start_row;
    uint32_t nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
    nrows_per_thread = hex_round_up(nrows_per_thread, 32);

    const uint32_t start_row = src0_start_row + nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + nrows_per_thread, src0_end_row);
    if (start_row >= end_row) return;

    const uint32_t ct_start = start_row / 32;
    const uint32_t ct_end   = (end_row + 31) / 32;
    const uint32_t n_pairs  = n_prefetch / 2;

    const HVX_Vector v_coef_a_times_sqrt = hvx_vec_splat_f32(HVX_GELU_COEF_A * HVX_SQRT_2_OVER_PI);
    const HVX_Vector v_sqrt_2_pi         = hvx_vec_splat_f32(HVX_SQRT_2_OVER_PI);
    const HVX_Vector v_one               = hvx_vec_splat_f32(1.0f);
    const HVX_Vector v_max_exp           = hvx_vec_splat_f32(87.0f);
    const HVX_Vector v_min_exp           = hvx_vec_splat_f32(-87.0f);

    uint32_t push_ct = ct_start;
    for (uint32_t d = 0; d < n_pairs && push_ct < ct_end; d++, push_ct++) {
        dma_queue_push(dma_q, dma_make_data(vtcm_weight_ptr + (2 * d) * slot_stride, g_base + push_ct * tile_row_stride),
                       pk_bytes, pk_bytes, pk_bytes, 1);
        dma_queue_push(dma_q, dma_make_data(vtcm_weight_ptr + (2 * d + 1) * slot_stride, u_base + push_ct * tile_row_stride),
                       pk_bytes, pk_bytes, pk_bytes, 1);
    }

    for (uint32_t ct = ct_start; ct < ct_end; ct++) {
        const uint8_t * g_tile = (void *) dma_queue_pop(dma_q).dst;
        const uint8_t * u_tile = (void *) dma_queue_pop(dma_q).dst;

        int valid_rows = (int) ne01 - (int) (ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        HVX_Vector vg = tiled_vec_dot_q4_0_32x1_pk_v(ne10, g_tile, act_data);
        HVX_Vector vu = tiled_vec_dot_q4_0_32x1_pk_v(ne10, u_tile, act_data);
        HVX_Vector res = hvx_vec_geglu_f32(vg, vu, v_coef_a_times_sqrt, v_sqrt_2_pi, v_one, v_max_exp, v_min_exp);
        hvx_vec_store_u(&dst_f[ct * 32], valid_rows * sizeof(float), res);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        if (push_ct < ct_end) {
            dma_queue_push(dma_q, dma_make_data((void *) g_tile, g_base + push_ct * tile_row_stride),
                           pk_bytes, pk_bytes, pk_bytes, 1);
            dma_queue_push(dma_q, dma_make_data((void *) u_tile, u_base + push_ct * tile_row_stride),
                           pk_bytes, pk_bytes, pk_bytes, 1);
            push_ct++;
        }
    }
}

MATVEC_2D_REPACKED_PK_IMPL(q4_0, 576, tiled_vec_dot_q4_0_32x1_pk)
MATMUL_NX_2D_REPACKED_PK_IMPL(q4_0, 576, tiled_vec_dot_q4_0_32x2, tiled_vec_dot_q4_0_32x1_pk)

MATVEC_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x1)
MATVEC_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x1)
MATVEC_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x1)
MATVEC_2D_REPACKED_IMPL(q5_k,       768,  tiled_vec_dot_q5_k_32x1)
MATVEC_2D_REPACKED_IMPL(q6_k,       896,  tiled_vec_dot_q6_k_32x1)
MATVEC_2D_REPACKED_IMPL(q3_k,       512,  tiled_vec_dot_q3_k_32x1)
MATVEC_2D_REPACKED_IMPL(q2_k,       512,  tiled_vec_dot_q2_k_32x1)
MATVEC_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x1)
MATVEC_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x1)

MATMUL_NX_2D_REPACKED_IMPL(q4_0,    576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_NX_2D_REPACKED_IMPL(q4_1,    640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_NX_2D_REPACKED_IMPL(q8_0,    1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_NX_2D_REPACKED_IMPL(iq4nl,   576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_NX_2D_REPACKED_IMPL(mxfp4,   544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)
MATMUL_NX_2D_REPACKED_IMPL(q5_k,    768,  tiled_vec_dot_q5_k_32x2,  tiled_vec_dot_q5_k_32x1)
MATMUL_NX_2D_REPACKED_IMPL(q3_k,    512,  tiled_vec_dot_q3_k_32x2,  tiled_vec_dot_q3_k_32x1)
MATMUL_NX_2D_REPACKED_IMPL(q2_k,    512,  tiled_vec_dot_q2_k_32x2,  tiled_vec_dot_q2_k_32x1)

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
    const size_t act_stride = mmctx->vtcm_act_stride;                                                                                                       \
                                                                                                                                                            \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                                           \
    uint8_t * restrict act_data = mmctx->vtcm_act;                                                                                                          \
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
                    const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (chunk_m_offset + ir1 + 0) * act_stride);                             \
                    const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (chunk_m_offset + ir1 + 1) * act_stride);                             \
                    float * restrict dst_row0 = (float *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);                                       \
                    float * restrict dst_row1 = (float *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);                                       \
                    float * dst_ptr0 = &dst_row0[ct * 32];                                                                                                  \
                    float * dst_ptr1 = &dst_row1[ct * 32];                                                                                                  \
                    DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, NULL, NULL);                                                  \
                }                                                                                                                                           \
                for (; ir1 < batch_nrows; ++ir1) {                                                                                                          \
                    const uint8_t * restrict act_col = (const uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);                                  \
                    float * restrict dst_row = (float *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);                                            \
                    float * dst_ptr = &dst_row[ct * 32];                                                                                                    \
                    DOT_2X1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL);                                                                              \
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
MATMUL_4D_REPACKED_IMPL(q3_k,       512,  tiled_vec_dot_q3_k_32x2,  tiled_vec_dot_q3_k_32x1)
MATMUL_4D_REPACKED_IMPL(q2_k,       512,  tiled_vec_dot_q2_k_32x2,  tiled_vec_dot_q2_k_32x1)
MATMUL_4D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_4D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

static void hvx_mm_2d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    const uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows
    const uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;
    const uint32_t cur_m_start = mmctx->cur_m_start;

    const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t act_row_size = nb11;

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;

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
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        for (; ir1 + 1 < act_nrows; ir1 += 2) {
            const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (ir1+0) * act_stride);
            const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (ir1+1) * act_stride);
            float * restrict dst_row0 = (float *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));
            float * restrict dst_row1 = (float *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        for (; ir1 < act_nrows; ++ir1) {
            const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);
            float * restrict dst_row          = (float *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
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
        for (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);
            float * restrict dst_row          = (float *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    if (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + act_nrows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
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
    const size_t act_row_size  = nb11;

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * act_data      = mmctx->vtcm_act;

    float * tmp = (float *) vtcm_dst_ptr;

    const dma_addr_t src0_row = src0->data;
    const uint8_t * restrict act_col = (const uint8_t *) act_data;
    float * restrict dst_col         = (float *) dst->data;

    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    if (src0_start_row < src0_end_row) {
        if (src2) {
            float * vtcm_bias_ptr = (float *) mmctx->vtcm_bias + src0_start_row;
            const dma_addr_t src2_addr = src2->data + src0_start_row * sizeof(float);
            int slice_size = (int)src0_end_row - (int)src0_start_row;
            if (slice_size > 0) {
                dma_queue_push(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),
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
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, act_col);
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
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    int copy_cnt = src0_end_row - src0_start_row;
    if (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
        if (src2) {
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (const uint8_t *) tmp,
                            (const uint8_t *) ((const float *) mmctx->vtcm_bias + src0_start_row),
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
    const size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;


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
                    const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (chunk_m_offset + ir1 + 0) * act_stride);
                    const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (chunk_m_offset + ir1 + 1) * act_stride);
                    float * restrict dst_row0 = (float *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);
                    float * restrict dst_row1 = (float *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                for (; ir1 < batch_nrows; ++ir1) {
                    const uint8_t * restrict act_col = (const uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    float * restrict dst_row          = (float *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
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
                    const uint8_t * restrict act_col = (const uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    float * restrict dst_row          = (float *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
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
    const uint32_t act_nrows       = ne11;
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
    const size_t act_row_size  = htp_mm_q8_0_tiled_row_size(ne10);

    const size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

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

                const uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                const uint8_t * restrict act_col = (const uint8_t *) (act_data + (ir1 + rm2 * ne11 + 0) * act_stride);
                float * restrict dst_row = (float *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, NULL);
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
    const size_t act_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    const uint32_t n_aids = src2->ne[0];  // num activated experts
    const uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        const int32_t eid = *(const int32_t *) ((const uint8_t *) src2->data + ie1 * src2->nb[0]);
        if (eid < 0) {
            continue;
        }
        assert(eid < (int32_t) n_ids);

        const dma_addr_t src0_row = src0->data + eid * nb02;
        const uint8_t * restrict act_col = (const uint8_t *) act_data;
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
            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, NULL);
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
    uint8_t * restrict act_data = mmctx->vtcm_act;

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
            const uint8_t * restrict act_col = (const uint8_t *) act_data;
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
                mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, NULL);
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

    const size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

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
                    const uint8_t * restrict act_col = (const uint8_t *) (act_data + (ir1 + rm2 * act->ne[1]) * act_stride);
                    float * restrict dst_row = (float *) (dst->data + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, NULL);
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
        case HTP_TYPE_Q3_K:
            mmctx->type         = "q3_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q3_k_32x1;
            return 0;
        case HTP_TYPE_Q2_K:
            mmctx->type         = "q2_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_k_32x1;
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
    mmctx->act = act;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = ne01;
    const uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

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
                        src0->type == HTP_TYPE_Q4_K || src0->type == HTP_TYPE_Q5_K ||
                        src0->type == HTP_TYPE_Q3_K || src0->type == HTP_TYPE_Q2_K);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
    if (is_repacked) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } else {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;
    size_t       act_row_size  = nb11;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

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
                case HTP_TYPE_Q3_K:   matmul_job_func = hvx_mm_4d_repacked_q3_k;   break;
                case HTP_TYPE_Q2_K:   matmul_job_func = hvx_mm_4d_repacked_q2_k;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_4d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_4d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            matmul_job_func = hvx_mm_4d;
        }
    } else if (act_nrows > 1) {
        if (is_repacked) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:
                case HTP_TYPE_Q4_K:   matmul_job_func = hvx_mm_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_2d_repacked_q8_0;   break;
                case HTP_TYPE_Q6_K:   matmul_job_func = hvx_mm_2d_repacked_q6_k;   break;
                case HTP_TYPE_Q5_K:   matmul_job_func = hvx_mm_2d_repacked_q5_k;   break;
                case HTP_TYPE_Q3_K:   matmul_job_func = hvx_mm_2d_repacked_q3_k;   break;
                case HTP_TYPE_Q2_K:   matmul_job_func = hvx_mm_2d_repacked_q2_k;   break;
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
                case HTP_TYPE_Q4_0:   matmul_job_func = (((act->ne[0]) / 32) % 2 == 0) ? hvx_mv_2d_repacked_pk_q4_0 : hvx_mv_2d_repacked_q4_0; break;
                case HTP_TYPE_Q4_1:
                case HTP_TYPE_Q4_K:   matmul_job_func = hvx_mv_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mv_2d_repacked_q8_0;   break;
                case HTP_TYPE_Q5_K:   matmul_job_func = hvx_mv_2d_repacked_q5_k;   break;
                case HTP_TYPE_Q6_K:   matmul_job_func = hvx_mv_2d_repacked_q6_k;   break;
                case HTP_TYPE_Q3_K:   matmul_job_func = hvx_mv_2d_repacked_q3_k;   break;
                case HTP_TYPE_Q2_K:   matmul_job_func = hvx_mv_2d_repacked_q2_k;   break;
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
            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (act->type == HTP_TYPE_F32);
            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = vec_dot_f16_f16_aa_2x2;
            act_row_size           = kparams->act_row_size;
            break;

        case HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = NULL;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = vec_dot_f32_f32_aa_2x2;
            act_row_size           = kparams->act_row_size;
            break;

        case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
        case HTP_MM_KERNEL_HVX_QUANT_ROW:
        default:
            if (hvx_mm_init_vec_dot(mmctx, src0->type) != 0) {
                return HTP_STATUS_NO_SUPPORT;
            }

            const uint32_t qk = QK_Q8_0_TILED;
            const uint32_t nb = (ne10 + qk - 1) / qk;
            const uint32_t total_nb = act_nrows * nb;

            if (act_nrows < octx->n_threads && !is_batched) {
                n_quant_tasks = MIN(total_nb, octx->n_threads);
                quant_task_func = htp_mm_act_quant_block_func(src0->type);
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
                quant_task_func = htp_mm_act_quant_row_func(src0->type);
            }
            act_row_size = kparams->act_row_size;
            break;
    }

    const uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    const uint32_t m_layout_rows = m_chunk;

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_rows, octx->n_threads,
                                 dst_row_size, src0_row_size, act_row_size, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        mmctx->vtcm_act_size_per_thread = L.act_bytes;
    } else {
        mmctx->vtcm_act_size_per_thread = fastdiv(L.act_bytes, &octx->n_threads_div);
    }

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    const size_t vtcm_size = L.total_bytes;

    FARF(HIGH, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);

    FARF(HIGH, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, act->data, dst->data);

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_act_stride  = act_row_size;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
    } else if (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * sizeof(float), 128);
    } else {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    if (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        for (uint32_t m_start = 0; m_start < act_nrows; m_start += m_chunk) {
            const uint32_t cur_m_rows = MIN(act_nrows - m_start, m_chunk);
            mmctx->cur_m_start = m_start;
            mmctx->cur_m_rows  = cur_m_rows;

            if (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_start, cur_m_rows);

                const uint32_t qk = QK_Q8_0_TILED;
                const uint32_t nb = (ne10 + qk - 1) / qk;
                const uint32_t total_nb = cur_m_rows * nb;
                uint32_t quant_tasks;
                work_queue_func_t q_func;
                if (cur_m_rows < octx->n_threads && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW)) {
                    quant_tasks = MIN(total_nb, octx->n_threads);
                    q_func = htp_mm_act_quant_block_func(src0->type);
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
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_start, cur_m_rows);
            }

            work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
        }
    } else {
        mmctx->cur_m_start = 0;
        mmctx->cur_m_rows  = act_nrows;

        if (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
        } else {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
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
    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    const size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;

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
            for (; ir1 + 1 < act_nrows; ir1 += 2) {
                const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (ir1+0) * act_stride);
                const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (ir1+1) * act_stride);
                float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));
                float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));
                mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
            }
            for (; ir1 < act_nrows; ++ir1) {
                const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);
                float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));
                mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
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
            for (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
                const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);
                float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));
                mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
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
DEQUANTIZE_WORKER_LOOP_IMPL(q3_k)
DEQUANTIZE_WORKER_LOOP_IMPL(q2_k)

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
    uint32_t                        act_elem_size;  // 4=F32 src1, 2=F16 src1
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

static void transfer_activation_chunk_to_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_size,
        struct htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    const uint32_t R = dma_step_rows;
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    const uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    const bool act_is_f16 = (act_elem_size == sizeof(__fp16));
    const size_t row_bytes = (size_t) k_block * act_elem_size;  // staging row (DMA dst stride)
    const size_t src_row_bytes = (size_t) k_stride * act_elem_size;  // DDR row (DMA src stride)
    const size_t width_bytes = (size_t) k_valid * act_elem_size;

    // Push step 0
    if (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_push(dma_q, dma_make_data(thread_act, act_dma_addr),
                       row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
    }
    // Push step 1 (if valid)
    if (n_steps > 1) {
        uint32_t next_r = R * 1;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            uint8_t *next_buf = thread_act + 1 * R * row_bytes;
            dma_queue_push(dma_q, dma_make_data(next_buf, act_dma_addr + (size_t) next_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
    for (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        uint8_t *curr_buf = thread_act;

        if (r < n_rows) {
            curr_buf = (uint8_t *) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        if (act_is_f16) {
            for (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                const __fp16 *pair_buf = (const __fp16 *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_f16_to_f16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        } else {
            for (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                const float *pair_buf = (const float *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_fp32_to_fp16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_push(dma_q, dma_make_data(curr_buf, act_dma_addr + (size_t) next_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
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
        const dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_size;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread;
        transfer_activation_chunk_to_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_act, st->act_elem_size, tr, st->dma_step_rows, st->dma_step_rows_shift
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

    assert(k_block % HTP_MM_HMX_TILE_N_COLS == 0);

    size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);
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

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_blocks = hmx_ceil_div(st->n_cols, 32);
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

// fused geglu epilogue: dst = gelu(gate) * up, gate/up are HMX fp16 output tiles in VTCM
#define HTP_GELU_COEF_A    0.044715f
#define HTP_SQRT_2_OVER_PI 0.79788456080286535587989211986876f

// sigmoid taking t = log2(e) * x, so the caller folds the scale into the gelu constants.
// Two things make this much cheaper than hvx_vec_fast_sigmoid_f16: 1 + 2^-|t| always lies
// in [1, 2], so a degree-5 polynomial in w = 2^-|t| replaces the full-range
// hvx_vec_inverse_f16; and the negative branch is w/(1+w) rather than 1 - 1/(1+w), which
// also removes the f16 cancellation that collapses sigmoid(x) to 0 for x < -9.
static inline HVX_Vector hvx_vec_sigmoid_log2_f16(HVX_Vector t) {
    // u = -|t|, by forcing the sign bit. Clamped so the exponent add below stays in range.
    HVX_Vector u = Q6_Vhf_vmax_VhfVhf(Q6_V_vor_VV(t, Q6_Vh_vsplat_R(0x8000)), hvx_vec_splat_f16(-24.0f));

    // w = 2^u: split u into int k and frac in [-0.5, 0.5], poly the frac, add k to the exponent.
    HVX_Vector k  = Q6_Vh_equals_Vhf(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(u, hvx_vec_splat_f16(0.5f))));
    HVX_Vector fr = Q6_Vqf16_vsub_VhfVhf(u, Q6_Vhf_equals_Vh(k));
    HVX_Vector y  = Q6_Vqf16_vmpy_Vqf16Vhf(fr, hvx_vec_splat_f16(0.00967604f));
    y = Q6_Vqf16_vmpy_Vqf16Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(y, hvx_vec_splat_f16(0.05592204f)), fr);
    y = Q6_Vqf16_vmpy_Vqf16Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(y, hvx_vec_splat_f16(0.24022107f)), fr);
    y = Q6_Vqf16_vmpy_Vqf16Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(y, hvx_vec_splat_f16(0.69312103f)), fr);
    HVX_Vector w = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(y, hvx_vec_splat_f16(1.0f)));
    // 2^k underflows f16 for k <= -14; the mantissa stays in [0.7, 1.5] so testing k alone is enough.
    w = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(-13), k), Q6_V_vzero(),
                      Q6_Vh_vaslacc_VhVhR(w, k, 10));

    // s = 1/(1+w) as a minimax degree-5 polynomial, valid because w is in [0, 1].
    HVX_Vector s = Q6_Vqf16_vmpy_VhfVhf(w, hvx_vec_splat_f16(-0.10765349f));
    s = Q6_Vqf16_vmpy_Vqf16Vhf(Q6_Vqf16_vadd_Vqf16Vhf(s, hvx_vec_splat_f16(0.42599635f)), w);
    s = Q6_Vqf16_vmpy_Vqf16Vhf(Q6_Vqf16_vadd_Vqf16Vhf(s, hvx_vec_splat_f16(-0.77778286f)), w);
    s = Q6_Vqf16_vmpy_Vqf16Vhf(Q6_Vqf16_vadd_Vqf16Vhf(s, hvx_vec_splat_f16(0.95607545f)), w);
    s = Q6_Vqf16_vmpy_Vqf16Vhf(Q6_Vqf16_vadd_Vqf16Vhf(s, hvx_vec_splat_f16(-0.99662270f)), w);
    s = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(s, hvx_vec_splat_f16(0.99995645f)));

    // t < 0: sigmoid = w/(1+w). The f16 sign bit makes an int16 compare a valid sign test.
    return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), t), hvx_vec_mul_f16_f16(w, s), s);
}

// gelu(x) = x * sigmoid(2 * sqrt(2/pi) * x * (1 + a * x^2)), with the factor 2 and log2(e)
// folded into the constants. f16 math fits 64 lanes per vector instead of 32, so the pair of
// output rows is done in one pass. x^2 is clamped because sigmoid saturates long before the
// square can overflow f16.
static inline HVX_Vector hvx_vec_geglu_f16(HVX_Vector x, HVX_Vector g, HVX_Vector c_a, HVX_Vector c_s, HVX_Vector x2_max) {
    HVX_Vector x2   = Q6_Vhf_vmin_VhfVhf(hvx_vec_mul_f16_f16(x, x), x2_max);
    HVX_Vector coef = hvx_vec_add_f16_f16(hvx_vec_mul_f16_f16(x2, c_a), c_s);
    HVX_Vector sig  = hvx_vec_sigmoid_log2_f16(hvx_vec_mul_f16_f16(x, coef));
    return hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, sig), g);
}

static void transfer_output_geglu_col_chunk(float *restrict dst, const __fp16 *restrict vtcm_gate, const __fp16 *restrict vtcm_up,
                                            uint32_t n_rows, uint32_t c_len, uint32_t total_n_cols, uint32_t dst_stride, uint32_t dst_cols) {
    const size_t tile_row_stride = (total_n_cols / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;

    const HVX_Vector one_h  = hvx_vec_splat_f16(1.0);
    const HVX_Vector c_a    = hvx_vec_splat_f16(2.0f * EXP_LOG2E_F * HTP_GELU_COEF_A * HTP_SQRT_2_OVER_PI);
    const HVX_Vector c_s    = hvx_vec_splat_f16(2.0f * EXP_LOG2E_F * HTP_SQRT_2_OVER_PI);
    const HVX_Vector x2_max = hvx_vec_splat_f16(256.0f);

    const size_t limit_c         = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned = limit_c & ~(size_t)(HTP_MM_HMX_TILE_N_COLS - 1);

    for (size_t r = 0; r < n_rows; r += 2) {
        const size_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;
        const size_t r1 = (r % HTP_MM_HMX_TILE_N_ROWS) / 2;
        const __fp16 *g_base = vtcm_gate + r0 * tile_row_stride;
        const __fp16 *u_base = vtcm_up   + r0 * tile_row_stride;
        float *out0 = dst + r * dst_stride;
        float *out1 = out0 + dst_stride;

        if (r + 1 < n_rows) {
            #pragma unroll(2)
            for (size_t c = 0; c < limit_c_aligned; c += HTP_MM_HMX_TILE_N_COLS) {
                const size_t t = (c / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
                HVX_Vector res = hvx_vec_geglu_f16(((const HVX_Vector *) (g_base + t))[r1],
                                                   ((const HVX_Vector *) (u_base + t))[r1], c_a, c_s, x2_max);
                HVX_VectorPair rp = Q6_Wqf32_vmpy_VhfVhf(res, one_h);
                hvx_vmemu(out0 + c) = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(rp));
                hvx_vmemu(out1 + c) = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(rp));
            }
        } else {
            #pragma unroll(2)
            for (size_t c = 0; c < limit_c_aligned; c += HTP_MM_HMX_TILE_N_COLS) {
                const size_t t = (c / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
                HVX_Vector res = hvx_vec_geglu_f16(((const HVX_Vector *) (g_base + t))[r1],
                                                   ((const HVX_Vector *) (u_base + t))[r1], c_a, c_s, x2_max);
                hvx_vmemu(out0 + c) = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(res, one_h)));
            }
        }

        // ragged tail: fewer than 32 valid destination columns left
        if (limit_c_aligned < limit_c) {
            const size_t c     = limit_c_aligned;
            const size_t valid = limit_c - c;
            const size_t t     = (c / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
            HVX_Vector res = hvx_vec_geglu_f16(((const HVX_Vector *) (g_base + t))[r1],
                                               ((const HVX_Vector *) (u_base + t))[r1], c_a, c_s, x2_max);
            HVX_VectorPair rp = Q6_Wqf32_vmpy_VhfVhf(res, one_h);

            hvx_vec_store_u(out0 + c, valid * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(rp)));
            if (r + 1 < n_rows) {
                hvx_vec_store_u(out1 + c, valid * sizeof(float), Q6_Vsf_equals_Vqf32(Q6_V_hi_W(rp)));
            }
        }
    }
}

typedef struct {
    struct htp_thread_trace * traces;
    float                   * dst;
    const __fp16            * vtcm_gate;
    const __fp16            * vtcm_up;
    uint32_t                  n_rows;
    uint32_t                  n_cols;
    uint32_t                  dst_stride;
    uint32_t                  dst_cols;
    struct fastdiv_values     n_threads_div;
} output_geglu_state_t;

static void transfer_output_geglu_worker_fn(unsigned int n, unsigned int i, void *data) {
    (void) n;
    output_geglu_state_t * st = (output_geglu_state_t *) data;
    struct htp_thread_trace * tr = &st->traces[i];

    const uint32_t n_blocks = st->n_cols / 32;
    const uint32_t c_first  = fastdiv(n_blocks * i,       &st->n_threads_div) * 32;
    const uint32_t c_last   = fastdiv(n_blocks * (i + 1), &st->n_threads_div) * 32;
    if (c_last == c_first || c_first >= st->dst_cols) return;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
    const size_t toff = (c_first / 32) * HTP_MM_HMX_TILE_N_ELMS;
    transfer_output_geglu_col_chunk(st->dst + c_first, st->vtcm_gate + toff, st->vtcm_up + toff,
                                    st->n_rows, c_last - c_first, st->n_cols, st->dst_stride, st->dst_cols - c_first);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

static void transfer_output_geglu_threaded(struct htp_context *ctx, float *dst, const __fp16 *vtcm_gate, const __fp16 *vtcm_up,
                                           int n_rows, int n_cols, int dst_stride, int dst_cols, int n_threads) {
    const uint32_t n_blocks = (uint32_t) n_cols / 32;
    const int nth = (int) hex_smin((uint32_t) n_threads, n_blocks);

    output_geglu_state_t st;
    st.traces        = ctx->trace;
    st.dst           = dst;
    st.vtcm_gate     = vtcm_gate;
    st.vtcm_up       = vtcm_up;
    st.n_rows        = (uint32_t) n_rows;
    st.n_cols        = (uint32_t) n_cols;
    st.dst_stride    = (uint32_t) dst_stride;
    st.dst_cols      = (uint32_t) dst_cols;
    st.n_threads_div = (nth == (int) ctx->n_threads) ? ctx->n_threads_div : init_fastdiv_values(nth);

    if (nth <= 1) {
        transfer_output_geglu_worker_fn(1, 0, &st);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_output_geglu_worker_fn, &st, nth);
    }
}

// Fused geglu epilogue writing f16 straight into HMX activation-tile layout. The HMX output tile
// layout and the HMX activation tile layout are identical (both pack a row pair per vector, see
// transfer_activation_row_pair_fp32_to_fp16_col_chunk), so the consumer matmul can DMA this in with
// no conversion at all. Tile-at-a-time so each 2KB destination tile is written sequentially.
static void transfer_output_geglu_tiled_col_chunk(__fp16 *restrict dst, const __fp16 *restrict vtcm_gate,
                                                 const __fp16 *restrict vtcm_up, uint32_t n_rows, uint32_t c_len,
                                                 uint32_t total_n_cols, uint32_t dst_tile_col, uint32_t dst_k_tiles) {
    const size_t src_row_stride = (total_n_cols / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
    const size_t dst_row_stride = (size_t) dst_k_tiles * HTP_MM_HMX_TILE_N_ELMS;

    const HVX_Vector c_a    = hvx_vec_splat_f16(2.0f * EXP_LOG2E_F * HTP_GELU_COEF_A * HTP_SQRT_2_OVER_PI);
    const HVX_Vector c_s    = hvx_vec_splat_f16(2.0f * EXP_LOG2E_F * HTP_SQRT_2_OVER_PI);
    const HVX_Vector x2_max = hvx_vec_splat_f16(256.0f);

    const uint32_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    for (uint32_t ct = 0; ct * HTP_MM_HMX_TILE_N_COLS < c_len; ++ct) {
        const size_t so = (size_t) ct * HTP_MM_HMX_TILE_N_ELMS;
        const size_t doff = (size_t) (dst_tile_col + ct) * HTP_MM_HMX_TILE_N_ELMS;
        for (uint32_t rt = 0; rt < n_row_tiles; ++rt) {
            const HVX_Vector * g = (const HVX_Vector *) (vtcm_gate + rt * src_row_stride + so);
            const HVX_Vector * u = (const HVX_Vector *) (vtcm_up + rt * src_row_stride + so);
            HVX_Vector *       o = (HVX_Vector *) (dst + rt * dst_row_stride + doff);
            #pragma unroll(4)
            for (uint32_t v = 0; v < HTP_MM_HMX_TILE_N_ROWS / 2; ++v) {
                o[v] = hvx_vec_geglu_f16(g[v], u[v], c_a, c_s, x2_max);
            }
        }
    }
}

typedef struct {
    struct htp_thread_trace * traces;
    __fp16                  * dst;
    const __fp16            * vtcm_gate;
    const __fp16            * vtcm_up;
    uint32_t                  n_rows;
    uint32_t                  n_cols;
    uint32_t                  dst_tile_col;
    uint32_t                  dst_k_tiles;
    struct fastdiv_values     n_threads_div;
} output_geglu_tiled_state_t;

static void transfer_output_geglu_tiled_worker_fn(unsigned int n, unsigned int i, void *data) {
    (void) n;
    output_geglu_tiled_state_t * st = (output_geglu_tiled_state_t *) data;
    struct htp_thread_trace * tr = &st->traces[i];

    const uint32_t n_blocks = st->n_cols / HTP_MM_HMX_TILE_N_COLS;
    const uint32_t b_first  = fastdiv(n_blocks * i, &st->n_threads_div);
    const uint32_t b_last   = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    if (b_last == b_first) return;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, b_first * HTP_MM_HMX_TILE_N_COLS);
    transfer_output_geglu_tiled_col_chunk(st->dst, st->vtcm_gate + (size_t) b_first * HTP_MM_HMX_TILE_N_ELMS,
                                          st->vtcm_up + (size_t) b_first * HTP_MM_HMX_TILE_N_ELMS, st->n_rows,
                                          (b_last - b_first) * HTP_MM_HMX_TILE_N_COLS, st->n_cols,
                                          st->dst_tile_col + b_first, st->dst_k_tiles);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, b_first * HTP_MM_HMX_TILE_N_COLS);
}

static void transfer_output_geglu_tiled_threaded(struct htp_context *ctx, __fp16 *dst, const __fp16 *vtcm_gate,
                                                const __fp16 *vtcm_up, int n_rows, int n_cols, int dst_tile_col,
                                                int dst_k_tiles, int n_threads) {
    const uint32_t n_blocks = (uint32_t) n_cols / HTP_MM_HMX_TILE_N_COLS;
    if (!n_blocks) return;
    const int nth = (int) hex_smin((uint32_t) n_threads, n_blocks);

    output_geglu_tiled_state_t st;
    st.traces        = ctx->trace;
    st.dst           = dst;
    st.vtcm_gate     = vtcm_gate;
    st.vtcm_up       = vtcm_up;
    st.n_rows        = (uint32_t) n_rows;
    st.n_cols        = (uint32_t) n_cols;
    st.dst_tile_col  = (uint32_t) dst_tile_col;
    st.dst_k_tiles   = (uint32_t) dst_k_tiles;
    st.n_threads_div = (nth == (int) ctx->n_threads) ? ctx->n_threads_div : init_fastdiv_values(nth);

    if (nth <= 1) {
        transfer_output_geglu_tiled_worker_fn(1, 0, &st);
    } else {
        worker_pool_run_func(ctx->worker_pool, transfer_output_geglu_tiled_worker_fn, &st, nth);
    }
}

static void transfer_output_chunk_threaded(struct htp_context *ctx, float *dst, const float *src2, const __fp16 *vtcm_src,
                                              int n_rows, int n_cols, int dst_stride, uint32_t src2_stride, int dst_cols, int n_threads) {
    if (n_rows <= 0) return;

    uint32_t n_blocks = hmx_ceil_div((uint32_t) n_cols, 32);
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
    uint32_t                      act_elem_size;  // 4=F32 src1, 2=F16 src1
    // src is already f16 in HMX tile layout (produced by a fused producer): copy it in verbatim
    bool                          src_tiled_f16;
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
    // element size of the activation rows (4 = F32, 2 = F16).
    const uint32_t                act_elem_size      = params->act_elem_size ? params->act_elem_size : (uint32_t) sizeof(float);

    if (n_rows <= 0) {
        return;
    }

    // Already tiled: the layout matches VTCM exactly, so one contiguous copy replaces all HVX work.
    if (params->src_tiled_f16) {
        const size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
        const size_t row_bytes   = (size_t) k_block * HTP_MM_HMX_TILE_N_ROWS * sizeof(__fp16);
        dma_queue_push(ctx->dma[0], dma_make_data(dst, act_dma_addr), row_bytes, row_bytes, row_bytes, n_row_tiles);
        dma_queue_pop(ctx->dma[0]);
        return;
    }

    const size_t n_tasks = (n_rows + 31) >> 5;
    if (act_elem_size == sizeof(float) && n_threads > 1 && k_block > 32 && n_tasks < (size_t) n_threads) {
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
    state.act_elem_size = act_elem_size;

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
    __fp16 *       output;
    const __fp16 * activation;
    const __fp16 * weight;
    const __fp16 * scales;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_job_t;

static void hmx_matmul_worker_fn(void * data) {
    hmx_matmul_job_t * job = (hmx_matmul_job_t *) data;
    FARF(HIGH, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
    core_dot_chunk_fp16(job->output, job->activation, job->weight, job->scales, job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
}

static inline void hmx_matmul_job_init(hmx_matmul_job_t * job,
                                       __fp16 *           output,
                                       const __fp16 *     activation,
                                       const __fp16 *     weight,
                                       const __fp16 *     scales,
                                       uint32_t           n_row_tiles,
                                       uint32_t           n_col_tiles,
                                       uint32_t           n_dot_tiles) {
    job->output      = output;
    job->activation  = activation;
    job->weight      = weight;
    job->scales      = scales;
    job->n_row_tiles = n_row_tiles;
    job->n_col_tiles = n_col_tiles;
    job->n_dot_tiles = n_dot_tiles;
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
                                  uint32_t act_elem_size,  // 4=F32 src1, 2=F16 src1
                                  int weight_stride,
                                  int weight_type,
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
                                  bool act_tiled_f16) {
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    // Quantized weights are repacked and padded to 32, so we it has to be 32-aligned.
    // Only F16/F32 weights can be non-32-aligned, they will be padded in the following kernel.
    const bool wtype_is_quant = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);
    if (k % 32 != 0 || (wtype_is_quant && n % 32 != 0)) {
        return -1;
    }
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
        case HTP_TYPE_Q3_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q3_k; break;
        case HTP_TYPE_Q2_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q2_k; break;
        case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
        case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
        default:
            return -1;
    }

    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const bool is_quant       = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);
    const size_t vec_dot_size = k * sizeof(__fp16);
    const size_t vtcm_budget  = ctx->vtcm_size;

    const uint32_t dma_dst_stride  = is_quant ? aligned_tile_size : row_stride;
    const uint32_t dma_src_stride  = is_quant ? tile_size : weight_stride;
    const uint32_t dma_width_bytes = is_quant ? tile_size : row_stride;

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used      = vtcm_size;

    // A tiled activation is addressed in whole 32-row tiles, so the chunk must be a tile multiple.
    // Shrinking is always safe: VTCM was sized for the larger chunk.
    if (act_tiled_f16) {
        m_chunk_n_rows = hex_align_down(m_chunk_n_rows, HTP_MM_HMX_TILE_N_ROWS);
        if (m_chunk_n_rows == 0) { m_chunk_n_rows = HTP_MM_HMX_TILE_N_ROWS; }
    }

    // Every m-block re-streams and re-dequantizes all weights, so a small tail block is bound by the
    // weight stream, not by HMX. Spread rows evenly over the same number of blocks (only shrinks the chunk).
    if (m_chunk_n_rows > 0 && (size_t) m > m_chunk_n_rows) {
        const size_t n_mblocks = hmx_ceil_div(m, m_chunk_n_rows);
        const size_t balanced  = hex_align_up(hmx_ceil_div(m, n_mblocks), HTP_MM_HMX_TILE_N_ROWS);
        if (balanced < m_chunk_n_rows) {
            m_chunk_n_rows = balanced;
        }
    }

    const size_t qweight_row_stride = is_quant ? (size_t)(n_k_tiles * aligned_tile_size) / 32 : 0;

    struct htp_mm_hmx_vtcm_layout L;
    // A tiled f16 activation is DMAed straight into VTCM, so it needs no f32 conversion scratch.
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline,
                                 act_tiled_f16 ? 0 : act_threads, aligned_tile_size, src2_bytes);

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

    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    float   *vtcm_f32_act    = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    void    *vtcm_scratch0   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]);
    void    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_scratch[1], pipeline);
    void    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    const bool has_bias      = (src2_bytes > 0 && src2_addr != 0);
    float   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_bias, has_bias);
    if (has_bias) {
        dma_queue_push(weight_dma, dma_make_data(vtcm_bias, src2_addr), hex_align_up(src2_bytes, 128), 0, src2_bytes, 1);
        dma_queue_pop(weight_dma);
    }

    FARF(HIGH, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, weight_type, m_chunk_n_rows, n_chunk_n_cols, vtcm_used, vtcm_budget);

    int n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    if (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_job_t job_slots[2];  // persistent double-buffered job descriptors

        for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            void *vtcm_weight_bufs[2] = { vtcm_scratch0, vtcm_scratch1 };
            void *vtcm_output_bufs[2] = { vtcm_output,   vtcm_scratch2 };

            struct activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (act_tiled_f16 ? (size_t) mr * k * sizeof(__fp16)
                                                              : (size_t) mr * act_stride * act_elem_size),
                .n_rows = (int) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_size = act_elem_size,
                .src_tiled_f16 = act_tiled_f16,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            const size_t   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
            const uint32_t height_A0 = is_quant ? hmx_ceil_div(n_cols_A0, 32) * n_k_tiles : n_cols_A0;
            dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

            if (1 < n_chunk_cnt) {
                const size_t   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                const uint32_t height_A1 = is_quant ? hmx_ceil_div(n_cols_A1, 32) * n_k_tiles : n_cols_A1;
                dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[1], weight + n_chunk_n_cols * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            for (int i = 0; i < n_chunk_cnt; ++i) {
                const size_t nc    = i * n_chunk_n_cols;
                const size_t nc_p2 = nc + 2 * n_chunk_n_cols;

                const size_t n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                const size_t n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                // 1. pop A_i
                void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 2], curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                if (i + 2 < n_chunk_cnt) {
                    const uint32_t height_p2 = is_quant ? hmx_ceil_div(n_cols_p2, 32) * n_k_tiles : n_cols_p2;
                    dma_queue_push(weight_dma, dma_make_data(curr_raw, weight + nc_p2 * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                }

                // 4. submit C_i
                hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2],
                                    vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                    hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                if (i > 0) {
                    hmx_queue_pop(ctx->hmx_queue);
                    const size_t nc_prev = (i - 1) * n_chunk_n_cols;
                    const size_t n_cols_prev = hex_smin(n - nc_prev, n_chunk_n_cols);
                    float *output_chunk = dst + (mr * dst_stride + nc_prev);
                    const float *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_prev) : NULL;
                    int chunk_dst_cols = dst_cols - (int)nc_prev;
                    if (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_queue_pop(ctx->hmx_queue);
            const size_t nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
            const size_t n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
            float *output_chunk = dst + (mr * dst_stride + nc_last);
            const float *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_last) : NULL;
            int chunk_dst_cols = dst_cols - (int)nc_last;
            if (chunk_dst_cols > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, src2_stride, chunk_dst_cols, n_threads);
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
                .act_dma_addr = act_dma_addr + (act_tiled_f16 ? (size_t) mr * k * sizeof(__fp16)
                                                              : (size_t) mr * act_stride * act_elem_size),
                .n_rows = (int) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_size = act_elem_size,
                .src_tiled_f16 = act_tiled_f16,
            };
            transfer_activation_chunk_threaded(&act_params);

            // A0: Pre-fetch the first weight chunk (nc = 0)
            if (n > 0) {
                const size_t n_cols = hex_smin(n, n_chunk_n_cols);
                const uint32_t height = is_quant ? hmx_ceil_div(n_cols, 32) * n_k_tiles : n_cols;
                dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
            }

            for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
                const size_t n_cols = hex_smin(n - nc, n_chunk_n_cols);
                const size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                const size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                // A: Wait for weight DMA
                void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_scratch0, curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // Start weight DMA for the next chunk early
                const size_t nc_next = nc + n_chunk_n_cols;
                if (nc_next < n) {
                    const size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                    const uint32_t height_next = is_quant ? hmx_ceil_div(n_cols_next, 32) * n_k_tiles : n_cols_next;
                    dma_queue_push(weight_dma, dma_make_data(curr_raw, weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, vtcm_scratch0, vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                hmx_queue_pop(ctx->hmx_queue);

                // D: Output Store
                float *output_chunk = dst + (mr * dst_stride + nc);
                const float *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : NULL;
                int chunk_dst_cols = dst_cols - (int)nc;
                if (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_rows, n_cols, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                }
            }
        }
    }

    return 0;
}

// geglu(W0 x, W1 x): per N-chunk, gate and up are computed into the two (pipeline) output buffers and combined
// in the epilogue, so the fp32 gate/up tensors never go to DDR
static int hmx_mm_nx_geglu_2d_f32(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    struct htp_context * ctx = octx->ctx;
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    if (kparams->n_weights != 2 || !kparams->pipeline) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const struct htp_tensor * restrict w_gate = octx->src[0];
    const struct htp_tensor * restrict w_up   = octx->src[1];
    const struct htp_tensor * restrict act    = octx->src[2];
    const struct htp_tensor * restrict dst    = octx->dsts[0];

    const int weight_type = (int) w_gate->type;
    const int k           = (int) act->ne[0];
    const int m           = (int) (act->ne[1] * act->ne[2] * act->ne[3]);
    const int act_stride  = (int) (act->nb[1] / sizeof(float));
    const size_t n        = w_gate->ne[1];

    // Tiled mode hands the geglu result to a fused consumer matmul as f16 in HMX activation-tile
    // layout, so dst is addressed by tile index rather than by row stride.
    const bool tiled_out = (octx->op == HTP_OP_MUL_MAT_NX_GEGLU_T);

    if (k % 32 != 0 || (act->data & (VLEN - 1)) != 0 || w_up->ne[1] != n || w_up->type != w_gate->type) {
        return HTP_STATUS_NO_SUPPORT;
    }
    // Tiled mode has no room for column padding: the consumer derives k from the same tile count.
    if (tiled_out && (n % HTP_MM_HMX_TILE_N_COLS != 0 || (size_t) dst->ne[0] != n ||
                      (dst->data & (VLEN - 1)) != 0)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const size_t row_stride = htp_mm_get_tiled_row_stride(weight_type, k);
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
    if (row_stride == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const bool is_quant          = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);
    // Tiled output is addressed in whole 32-row tiles; shrinking the chunk only frees VTCM.
    const int  m_chunk_n_rows    = tiled_out ? (int) hex_smax(hex_align_down((uint32_t) kparams->m_chunk, HTP_MM_HMX_TILE_N_ROWS), HTP_MM_HMX_TILE_N_ROWS)
                                             : kparams->m_chunk;
    const int  n_chunk_n_cols    = kparams->n_chunk;
    const int  n_threads         = octx->n_threads;
    const int  tile_size         = kparams->tile_size;
    const int  aligned_tile_size = kparams->aligned_tile_size;

    const uint32_t dma_dst_stride  = is_quant ? aligned_tile_size : row_stride;
    const uint32_t dma_width_bytes = is_quant ? tile_size : row_stride;

    struct htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, true,
                                 kparams->n_act_threads, aligned_tile_size, 0);
    const size_t extra_off = hex_align_up(L.total_bytes, HTP_MM_HMX_TILE_SIZE);
    if (extra_off + 2 * L.output_area_bytes > ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    __fp16 * vtcm_weight_raw[2] = { VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]), VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[1]) };
    __fp16 * vtcm_wbuf[2]       = { VTCM_LAYOUT_PTR(__fp16, base, L.off_scratch[0]), VTCM_LAYOUT_PTR(__fp16, base, L.off_scratch[1]) };
    __fp16 * vtcm_obuf[4]       = { VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]), VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[1]),
                                    VTCM_LAYOUT_PTR(__fp16, base, extra_off),
                                    VTCM_LAYOUT_PTR(__fp16, base, extra_off + L.output_area_bytes) };
    __fp16 * vtcm_f16_act       = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    float  * vtcm_f32_act       = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    __fp16 * vtcm_scales        = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    int m_start = 0;
    int m_rows  = m;
    if (ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_can_row_partition(dst, sizeof(float));
        const uint32_t unit = tiled_out ? HTP_MM_HMX_TILE_N_ROWS : 1;
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m, can_split ? unit : 0, ctx->mdev.idx, ctx->mdev.count, &ctx->mdev.count_div);
        m_start = (int) range.start;
        m_rows  = (int) range.count;
    }
    if (m_rows == 0) {
        return HTP_STATUS_OK;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    dma_queue * weight_dma = ctx->dma[0];
    const struct htp_tensor * ws[2] = { w_gate, w_up };
    const uint32_t dma_src_stride[2] = { is_quant ? (uint32_t) tile_size : (uint32_t) w_gate->nb[1],
                                         is_quant ? (uint32_t) tile_size : (uint32_t) w_up->nb[1] };

    float * dst_ptr          = (float *) dst->data;
    const size_t dst_stride  = dst->nb[1] / sizeof(float);
    const int dst_cols       = (int) dst->ne[0];
    __fp16 * const dst_tiled = (__fp16 *) dst->data;
    const int dst_k_tiles    = (int) n / HTP_MM_HMX_TILE_N_COLS;
    const int n_chunk_cnt    = hmx_ceil_div(n, n_chunk_n_cols);
    const int n_steps        = 2 * n_chunk_cnt; // step s: chunk s/2, weight s%2 (0 = gate, 1 = up)

    #define GEGLU_PUSH_W(step, buf) do {                                                                        \
        const int    _c    = (step) / 2;                                                                        \
        const int    _w    = (step) % 2;                                                                        \
        const size_t _nc   = (size_t) _c * n_chunk_n_cols;                                                      \
        const size_t _cols = hex_smin(n - _nc, (size_t) n_chunk_n_cols);                                        \
        const uint32_t _h  = is_quant ? (_cols / 32) * n_k_tiles : _cols;                                       \
        dma_queue_push(weight_dma, dma_make_data((buf), ws[_w]->data + _nc * ws[_w]->nb[1]),                    \
                       dma_dst_stride, dma_src_stride[_w], dma_width_bytes, _h);                                \
    } while (0)

    hmx_matmul_job_t job_slots[4];
    const size_t mr_end = (size_t) (m_start + m_rows);

    for (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
        const size_t n_rows = hex_smin(mr_end - mr, (size_t) m_chunk_n_rows);

        struct activation_transfer_params act_params = {
            .ctx = ctx,
            .dst = vtcm_f16_act,
            .act_dma_addr = act->data + mr * act_stride * sizeof(float),
            .n_rows = (int) n_rows,
            .k_block = k,
            .k_stride = act_stride,
            .n_threads = kparams->n_act_threads,
            .act_threads_div = &kparams->div_n_act_threads,
            .k_div = &kparams->div_ne00_padded,
            .k_valid = k,
            .vtcm_f32_act = vtcm_f32_act,
            .vtcm_f32_act_bytes = L.act_f32_bytes,
        };
        transfer_activation_chunk_threaded(&act_params);

        GEGLU_PUSH_W(0, vtcm_weight_raw[0]);
        if (n_steps > 1) {
            GEGLU_PUSH_W(1, vtcm_weight_raw[1]);
        }

        // step s: chunk c = s/2, weight w = s%2; the output for (c, w) goes to obuf[(c%2)*2 + w]
        // the epilogue of chunk c-1 runs on HVX after both HMX jobs of chunk c have been queued
        for (int s = 0; s < n_steps; ++s) {
            const int    c      = s / 2;
            const int    w      = s % 2;
            const size_t nc     = (size_t) c * n_chunk_n_cols;
            const size_t n_cols = hex_smin(n - nc, (size_t) n_chunk_n_cols);

            void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;

            // scratch w was last read by HMX job s-2, which has been popped (see below) before we get here
            dequantize_tiled_weight_chunk_to_fp16_tiles(ctx, vtcm_wbuf[w], curr_raw, n_cols, k, row_stride, weight_type,
                                                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

            if (s + 2 < n_steps) {
                GEGLU_PUSH_W(s + 2, curr_raw);
            }

            const int ob = (c % 2) * 2 + w;
            hmx_matmul_job_init(&job_slots[ob], vtcm_obuf[ob], vtcm_f16_act, vtcm_wbuf[w], vtcm_scales,
                                hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS), hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS),
                                k / HTP_MM_HMX_TILE_N_ROWS);
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[ob]));

            // retire job s-1 so that scratch (1-w) is free for step s+1
            if (s > 0) {
                hmx_queue_pop(ctx->hmx_queue);
            }

            // after queueing chunk c's up-job, finish chunk c-1 (both of its jobs are retired by now)
            if (w == 1 && c > 0) {
                const size_t nc_prev = nc - n_chunk_n_cols;
                const int    pb      = ((c - 1) % 2) * 2;
                const int    dcols   = dst_cols - (int) nc_prev;
                if (tiled_out) {
                    transfer_output_geglu_tiled_threaded(ctx, dst_tiled + (mr / HTP_MM_HMX_TILE_N_ROWS) * (size_t) dst_k_tiles * HTP_MM_HMX_TILE_N_ELMS,
                                                         vtcm_obuf[pb], vtcm_obuf[pb + 1], (int) n_rows, n_chunk_n_cols,
                                                         (int) (nc_prev / HTP_MM_HMX_TILE_N_COLS), dst_k_tiles, n_threads);
                } else if (dcols > 0) {
                    transfer_output_geglu_threaded(ctx, dst_ptr + mr * dst_stride + nc_prev, vtcm_obuf[pb], vtcm_obuf[pb + 1],
                                                   (int) n_rows, n_chunk_n_cols, (int) dst_stride, dcols, n_threads);
                }
            }
        }

        // drain the last job and finish the last chunk
        hmx_queue_pop(ctx->hmx_queue);
        {
            const int    c_last  = n_chunk_cnt - 1;
            const size_t nc_last = (size_t) c_last * n_chunk_n_cols;
            const int    pb      = (c_last % 2) * 2;
            const int    dcols   = dst_cols - (int) nc_last;
            if (tiled_out) {
                transfer_output_geglu_tiled_threaded(ctx, dst_tiled + (mr / HTP_MM_HMX_TILE_N_ROWS) * (size_t) dst_k_tiles * HTP_MM_HMX_TILE_N_ELMS,
                                                     vtcm_obuf[pb], vtcm_obuf[pb + 1], (int) n_rows,
                                                     (int) hex_smin(n - nc_last, (size_t) n_chunk_n_cols),
                                                     (int) (nc_last / HTP_MM_HMX_TILE_N_COLS), dst_k_tiles, n_threads);
            } else if (dcols > 0) {
                transfer_output_geglu_threaded(ctx, dst_ptr + mr * dst_stride + nc_last, vtcm_obuf[pb], vtcm_obuf[pb + 1],
                                               (int) n_rows, (int) hex_smin(n - nc_last, (size_t) n_chunk_n_cols),
                                               (int) dst_stride, dcols, n_threads);
            }
        }
    }
    #undef GEGLU_PUSH_W

    return HTP_STATUS_OK;
}


// ---- int8 HMX gate+up GEGLU (activation.ub x weight.b:deep, 64 filters per issue) ----

typedef struct {
    uint8_t *       out;
    const uint8_t * act;
    const uint8_t * wgt;
    const uint8_t * bias;
    uint32_t        n_row_tiles;
    uint32_t        n_groups;
    uint32_t        n_k;
    size_t          wgt_stride;
    size_t          bias_stride;
} hmx_int8_job_t;

static void hmx_int8_job_fn(void * data) {
    hmx_int8_job_t * j = (hmx_int8_job_t *) data;
    hmx_int8_core_dot_chunk(j->out, j->act, j->wgt, j->bias, j->n_row_tiles, j->n_groups, j->n_k, j->wgt_stride, j->bias_stride);
}

// per-chunk side tables of one expanded weight set (256 B aligned; bias tables first)
struct hmx_int8_meta {
    uint8_t * bias;     // [n_cols/64][512 B]
    float *   s_n;      // [n_cols]
    float *   col_mul;  // [n_cols]
    int32_t * colsum;   // [n_cols]
    int32_t * abssum;   // [n_cols]
};

typedef struct {
    struct htp_thread_trace * traces;
    uint8_t *                 dst;
    const uint8_t *           src;
    struct hmx_int8_meta      meta;
    uint32_t                  n_ct;
    uint32_t                  n_k_tiles;
    uint32_t                  aligned_tile_size;
    uint32_t                  n_cols;
    uint32_t                  reduce;
} int8_expand_state_t;

static void int8_expand_worker_fn(unsigned int n, unsigned int i, void * data) {
    int8_expand_state_t * st = (int8_expand_state_t *) data;
    const uint32_t p0 = (st->n_ct * i) / n, p1 = (st->n_ct * (i + 1)) / n;
    if (p0 == p1) { return; }
    htp_trace_event_start(&st->traces[i], HTP_TRACE_EVT_HVX_W_DEQUANT, p0);
    hmx_int8_expand_q4_0(st->dst, st->meta.s_n, st->meta.colsum, st->meta.abssum, st->src, p0, p1,
                         st->n_k_tiles, st->aligned_tile_size);
    hmx_int8_make_bias((uint32_t *) st->meta.bias, st->meta.col_mul, st->meta.s_n, st->meta.colsum,
                           st->meta.abssum, p0, p1, st->reduce);
    htp_trace_event_stop(&st->traces[i], HTP_TRACE_EVT_HVX_W_DEQUANT, p0);
}

typedef struct {
    struct htp_context *      ctx;
    struct htp_thread_trace * traces;
    uint8_t *                 dst;
    dma_addr_t                x;          // DDR address of row 0 of this m-block
    float *                   a_scale;
    float *                   stage;      // n_threads * 2 buffers * 2 rows * k floats
    uint32_t                  m;
    uint32_t                  k;
    uint32_t                  x_stride;   // floats
    uint32_t                  n_rt;
    const float *             signs;      // non-NULL: rotate each row (Hadamard) before quantizing
    uint8_t *                 dst2;       // non-NULL: A16 second pass (low residual pass) destination
    float *                   a_scale2;   // A16 second pass row scales
    bool                      tiled_f16;  // x is the tiled-f16 activation (HMX tile layout, from a fused GEGLU_T producer)
    uint32_t                  pair0;      // first row pair (block-local) handled by this call
    uint32_t                  pair_n;     // number of row pairs handled by this call
    uint32_t                  first_w;    // 1: worker 0 (the caller, owner of dma[0] = weight DMA queue) takes no work
    const float *             inv_s;      // non-NULL: x *= inv_s[k] (static activation smoothing, VTCM copy, 128 B aligned)
    uint32_t                  abl;        // bit 0: timing ablation, skip the quantize compute
    uint32_t                  old_quant;  // 1: use the original quantizer (A/B timing and correctness reference)
    uint32_t                  rot32;      // 1: fused in-lane WHT over 32 elements (inv_s carries sign / (sqrt(32) * s_k))
} int8_quant_state_t;

static void int8_quant_worker_fn(unsigned int n, unsigned int i, void * data) {
    int8_quant_state_t * st = (int8_quant_state_t *) data;
    // threads split the call's row PAIRS; with first_w the caller (thread 0, weight DMA owner) does nothing
    if (i < st->first_w) { return; }
    const uint32_t wi = i - st->first_w, wn = n - st->first_w;
    const uint32_t p0 = st->pair0 + (st->pair_n * wi) / wn, p1 = st->pair0 + (st->pair_n * (wi + 1)) / wn;
    if (p0 == p1) { return; }
    htp_trace_event_start(&st->traces[i], HTP_TRACE_EVT_HVX_A_QUANT, p0);
    dma_queue * q = st->ctx->dma[i];
    const uint32_t k = st->k, nkc = k / 32;
    float * stage[2] = { st->stage + (size_t) i * 4 * k, st->stage + (size_t) i * 4 * k + 2 * (size_t) k };
    const uint32_t n_pairs = p1 - p0;
    const size_t row_bytes = (size_t) k * sizeof(float), src_stride = (size_t) st->x_stride * sizeof(float);

    // tiled-f16 input: row pair p of row tile rt is the 128 B vector p of each of the nkc 2 KB tiles of that row tile
    // (lo halves = row 2p, hi halves = row 2p + 1). Two f16 stripe buffers sit behind the f32 pair area of the thread's stage.
    uint8_t * h16[2] = { (uint8_t *) stage[0] + 2 * row_bytes, (uint8_t *) stage[0] + 2 * row_bytes + (size_t) nkc * 128 };
    const HVX_Vector one16 = hvx_vec_splat_f16(1.0f);
    #define Q8_PUSH(li) do {                                                                                   \
        const uint32_t _gp  = p0 + (li);                                                                        \
        const uint32_t _row = 2 * _gp;                                                                          \
        if (_row < st->m) {                                                                                     \
            if (st->tiled_f16) {                                                                                \
                dma_queue_push(q, dma_make_data(h16[(li) & 1], st->x + (size_t) ((_gp >> 4) * nkc) * 2048 + (_gp & 15) * 128), \
                               128, 2048, 128, nkc);                                                            \
            } else {                                                                                            \
                dma_queue_push(q, dma_make_data(stage[(li) & 1], st->x + (size_t) _row * src_stride), row_bytes, src_stride, row_bytes, \
                               hex_smin(2u, st->m - _row));                                                     \
            }                                                                                                   \
        }                                                                                                       \
    } while (0)

    Q8_PUSH(0);
    if (n_pairs > 1) { Q8_PUSH(1); }
    float sc[2];
    for (uint32_t li = 0; li < n_pairs; ++li) {
        const uint32_t gp  = p0 + li;
        const uint32_t row = 2 * gp;
        const float * x0 = NULL, * x1 = NULL;
        if (row < st->m) {
            const float * buf = (const float *) dma_queue_pop(q).dst;
            if (st->tiled_f16) {
                const HVX_Vector * hv = (const HVX_Vector *) buf;
                HVX_Vector * w0 = (HVX_Vector *) stage[0];
                HVX_Vector * w1 = (HVX_Vector *) (stage[0] + k);
                for (uint32_t kc = 0; kc < nkc; ++kc) {
                    const HVX_VectorPair vp = Q6_Wqf32_vmpy_VhfVhf(hv[kc], one16);
                    w0[kc] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
                    w1[kc] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
                }
                buf = stage[0];
            }
            x0 = buf;
            x1 = row + 1 < st->m ? buf + k : NULL;
        }
        const bool fast = !st->signs && !st->abl && x0 && !st->old_quant;   // no rotation: smoothing is fused into the quantizer (A8 and A16)
        if (st->inv_s && !fast) {
            const HVX_Vector * sv = (const HVX_Vector *) st->inv_s;
            if (x0) { HVX_Vector * xv = (HVX_Vector *) x0; for (uint32_t kc = 0; kc < nkc; ++kc) { xv[kc] = hvx_vec_mul_f32_f32(xv[kc], sv[kc]); } }
            if (x1) { HVX_Vector * xv = (HVX_Vector *) x1; for (uint32_t kc = 0; kc < nkc; ++kc) { xv[kc] = hvx_vec_mul_f32_f32(xv[kc], sv[kc]); } }
        }
        if (st->signs) {
            if (x0) { hmx_int8_rotate_row((HVX_Vector *) x0, (const HVX_Vector *) st->signs, k); }
            if (x1) { hmx_int8_rotate_row((HVX_Vector *) x1, (const HVX_Vector *) st->signs, k); }
        }
        const size_t doff = (size_t) (gp >> 4) * nkc * HMX_INT8_CROUTON_A + (gp & 15) * 128;
        uint8_t * dp = st->dst + doff;
        if (st->abl) {   // ablation: DMA + pop only, no quantize compute (wrong results, timing only)
            if (x0) { st->a_scale[row] = 1.0f; }
            if (x1) { st->a_scale[row + 1] = 1.0f; }
            if (li + 2 < n_pairs) { Q8_PUSH(li + 2); }
            continue;
        }
        if (st->dst2 && fast) {
            float sc16[4];
            hmx_int8_quant_pair_a16_fast(dp, st->dst2 + doff, sc16, (float *) x0, (float *) x1, k, st->inv_s);
            st->a_scale[row] = sc16[0]; st->a_scale2[row] = sc16[2];
            if (x1) { st->a_scale[row + 1] = sc16[1]; st->a_scale2[row + 1] = sc16[3]; }
        } else if (st->dst2) {
            float sc16[4];
            hmx_int8_quant_pair_a16(dp, st->dst2 + doff, sc16, x0, x1, k);
            if (x0) { st->a_scale[row] = sc16[0]; st->a_scale2[row] = sc16[2]; }
            if (x1) { st->a_scale[row + 1] = sc16[1]; st->a_scale2[row + 1] = sc16[3]; }
        } else if (fast) {
            hmx_int8_quant_pair_fast(dp, sc, (float *) x0, (float *) x1, k, st->inv_s, st->rot32);
            st->a_scale[row] = sc[0];
            if (x1) { st->a_scale[row + 1] = sc[1]; }
        } else {
            hmx_int8_quant_pair(dp, sc, x0, x1, k);
            if (x0) { st->a_scale[row] = sc[0]; }
            if (x1) { st->a_scale[row + 1] = sc[1]; }
        }
        if (li + 2 < n_pairs) { Q8_PUSH(li + 2); }
    }
    #undef Q8_PUSH
    htp_trace_event_stop(&st->traces[i], HTP_TRACE_EVT_HVX_A_QUANT, p0);
}

// int16 accumulator tiles -> scaled f16 tiles, in place: tile[v] (word c = rows 2v, 2v+1 of channel c)
typedef struct {
    struct htp_thread_trace * traces;
    uint8_t *                 tiles;
    const float *             col_mul;
    const uint32_t *          a_scale16;  // row pair (2v, 2v+1) as one word of two f16
    uint32_t                  n_rt;
    uint32_t                  n_ct;
} int8_convert_state_t;

static void int8_convert_worker_fn(unsigned int n, unsigned int i, void * data) {
    int8_convert_state_t * st = (int8_convert_state_t *) data;
    const uint32_t c0 = (st->n_ct * i) / n, c1 = (st->n_ct * (i + 1)) / n;
    if (c0 == c1) { return; }
    htp_trace_event_start(&st->traces[i], HTP_TRACE_EVT_HVX_W_PREP, c0);
    // per-column f16 scale vectors (word = channel, both halves) built once per column tile
    HVX_Vector cm16[32];
    for (uint32_t ct = c0; ct < c1; ct += 32) {
        const uint32_t cn = hex_smin(32u, c1 - ct);
        for (uint32_t t = 0; t < cn; ++t) {
            const HVX_Vector cm = ((const HVX_Vector *) st->col_mul)[ct + t];
            cm16[t] = hvx_vec_f32_to_f16_shuff(cm, cm);
        }
        for (uint32_t rt = 0; rt < st->n_rt; ++rt) {
            const uint32_t * as = st->a_scale16 + rt * 16;
            for (uint32_t v = 0; v < 16; ++v) {
                const HVX_Vector asv = Q6_V_vsplat_R(as[v]);
                for (uint32_t t = 0; t < cn; ++t) {
                    HVX_Vector * tv = (HVX_Vector *) (st->tiles + ((size_t) rt * st->n_ct + ct + t) * 2048) + v;
                    const HVX_Vector a = Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(*tv), asv);
                    *tv = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(a, cm16[t]));
                }
            }
        }
    }
    htp_trace_event_stop(&st->traces[i], HTP_TRACE_EVT_HVX_W_PREP, c0);
}


// Fused epilogue for the int8 path (tiled f16 output): int16 accumulator tiles of gate and up are scaled to f16
// (row scale * column scale) in registers and fed straight into the GEGLU.
typedef struct {
    struct htp_thread_trace * traces;
    __fp16 *                  dst;
    const uint8_t *           gate;
    const uint8_t *           up;
    const float *             cm_gate;
    const float *             cm_up;
    const float *             ca_gate;   // per-column two-level-bias remainder (NULL when the record has none)
    const float *             ca_up;
    const uint32_t *          a_scale16;
    uint32_t                  n_rt;
    uint32_t                  n_ct;
    uint32_t                  dst_tile_col;
    uint32_t                  dst_k_tiles;
} int8_geglu_state_t;

// Mirrors transfer_output_geglu_tiled_col_chunk, with int16 accumulator inputs scaled by row/column f16 factors.
static void int8_geglu_tiled_col_chunk(__fp16 *restrict dst, const int16_t *restrict gate, const int16_t *restrict up,
                                       const uint32_t *restrict as16, const float *restrict cm_gate,
                                       const float *restrict cm_up, const float *restrict ca_gate,
                                       const float *restrict ca_up, uint32_t n_rows, uint32_t c_len,
                                       uint32_t total_n_cols, uint32_t dst_tile_col, uint32_t dst_k_tiles) {
    const size_t src_row_stride = (total_n_cols / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
    const size_t dst_row_stride = (size_t) dst_k_tiles * HTP_MM_HMX_TILE_N_ELMS;

    const HVX_Vector c_a    = hvx_vec_splat_f16(2.0f * EXP_LOG2E_F * HTP_GELU_COEF_A * HTP_SQRT_2_OVER_PI);
    const HVX_Vector c_s    = hvx_vec_splat_f16(2.0f * EXP_LOG2E_F * HTP_SQRT_2_OVER_PI);
    const HVX_Vector x2_max = hvx_vec_splat_f16(256.0f);

    const uint32_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    for (uint32_t ct = 0; ct * HTP_MM_HMX_TILE_N_COLS < c_len; ++ct) {
        const size_t so = (size_t) ct * HTP_MM_HMX_TILE_N_ELMS;
        const size_t doff = (size_t) (dst_tile_col + ct) * HTP_MM_HMX_TILE_N_ELMS;
        const HVX_Vector cmg0 = ((const HVX_Vector *) cm_gate)[ct];
        const HVX_Vector cmu0 = ((const HVX_Vector *) cm_up)[ct];
        const HVX_Vector cmg  = hvx_vec_f32_to_f16_shuff(cmg0, cmg0);
        const HVX_Vector cmu  = hvx_vec_f32_to_f16_shuff(cmu0, cmu0);
        HVX_Vector cag = Q6_V_vzero(), cau = Q6_V_vzero();
        if (ca_gate) { const HVX_Vector x = ((const HVX_Vector *) ca_gate)[ct]; cag = hvx_vec_f32_to_f16_shuff(x, x); }
        if (ca_up)   { const HVX_Vector x = ((const HVX_Vector *) ca_up)[ct];   cau = hvx_vec_f32_to_f16_shuff(x, x); }
        for (uint32_t rt = 0; rt < n_row_tiles; ++rt) {
            const HVX_Vector * g = (const HVX_Vector *) (gate + rt * src_row_stride + so);
            const HVX_Vector * u = (const HVX_Vector *) (up + rt * src_row_stride + so);
            HVX_Vector *       o = (HVX_Vector *) (dst + rt * dst_row_stride + doff);
            const uint32_t *   a = as16 + rt * 16;
            #pragma unroll(4)
            for (uint32_t v = 0; v < HTP_MM_HMX_TILE_N_ROWS / 2; ++v) {
                const HVX_Vector asv = Q6_V_vsplat_R(a[v]);
                HVX_Vector gf  = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vh(g[v]), asv), cmg);
                HVX_Vector uf  = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vh(u[v]), asv), cmu);
                // remainder of the truncated bias word (accumulator domain) scales with the row factor; zero when the word is exact
                gf = hvx_vec_add_f16_f16(gf, hvx_vec_mul_f16_f16(cag, asv));
                uf = hvx_vec_add_f16_f16(uf, hvx_vec_mul_f16_f16(cau, asv));
                o[v] = hvx_vec_geglu_f16(gf, uf, c_a, c_s, x2_max);
            }
        }
    }
}

static void int8_geglu_tiled_worker_fn(unsigned int n, unsigned int i, void * data) {
    int8_geglu_state_t * st = (int8_geglu_state_t *) data;
    const uint32_t c0 = (st->n_ct * i) / n, c1 = (st->n_ct * (i + 1)) / n;
    if (c0 == c1) { return; }
    htp_trace_event_start(&st->traces[i], HTP_TRACE_EVT_HVX_O_PROC, c0);
    int8_geglu_tiled_col_chunk(st->dst, (const int16_t *) st->gate + c0 * HTP_MM_HMX_TILE_N_ELMS,
                               (const int16_t *) st->up + c0 * HTP_MM_HMX_TILE_N_ELMS, st->a_scale16,
                               st->cm_gate + c0 * 32, st->cm_up + c0 * 32,
                               st->ca_gate ? st->ca_gate + c0 * 32 : NULL, st->ca_up ? st->ca_up + c0 * 32 : NULL,
                               st->n_rt * 32, (c1 - c0) * 32,
                               st->n_ct * 32, st->dst_tile_col + c0, st->dst_k_tiles);
    htp_trace_event_stop(&st->traces[i], HTP_TRACE_EVT_HVX_O_PROC, c0);
}

#define HMX_INT8_MAX_ROWS 1024

struct hmx_int8_layout {
    size_t off_act, off_ascale, off_ascale16, off_stage, off_raw[2], off_w8[2], off_meta[4], off_obuf[4], total;
    size_t raw_bytes, w8_bytes, meta_bytes, obuf_bytes, bias_bytes;
};

static bool hmx_int8_layout_build(struct hmx_int8_layout * L, uint32_t k, size_t mc, size_t nc, uint32_t aligned_tile_size) {
    const size_t nkt = k / 32;
    const size_t act_bytes = (mc / 32) * nkt * HMX_INT8_CROUTON_A;
    L->raw_bytes  = hex_align_up((nc / 32) * nkt * aligned_tile_size, 2048);
    L->w8_bytes   = (nc / 64) * nkt * HMX_INT8_CROUTON_W;
    L->bias_bytes = (nc / 64) * 512;
    L->meta_bytes = hex_align_up(L->bias_bytes, 256) + 4 * hex_align_up(nc * 4, 256);
    L->obuf_bytes = (mc / 32) * (nc / 32) * 2048;
    size_t off = 0;
    L->off_act = off;      off += hex_align_up(act_bytes, 2048);
    L->off_ascale = off;   off += hex_align_up(mc * 4, 256);
    L->off_ascale16 = off; off += hex_align_up(mc * 2, 256);
    L->off_stage = off;    off += 8 * 4 * (size_t) k * sizeof(float);
    for (int i = 0; i < 2; ++i) { L->off_raw[i] = off; off += L->raw_bytes; }
    for (int i = 0; i < 2; ++i) { L->off_w8[i] = off;  off += hex_align_up(L->w8_bytes, 2048); }
    for (int i = 0; i < 4; ++i) { L->off_meta[i] = off; off += L->meta_bytes; }
    off = hex_align_up(off, 2048);
    for (int i = 0; i < 4; ++i) { L->off_obuf[i] = off; off += L->obuf_bytes; }
    L->total = off;
    return true;
}

static bool hmx_int8_solve_geglu(uint32_t k, uint32_t n, uint32_t m, uint32_t aligned_tile_size, size_t budget,
                                 size_t * mc_out, size_t * nc_out) {
    const size_t m_pad = hex_align_up(m, 32);
    size_t best_cost = SIZE_MAX, best_mc = 0, best_nc = 0;
    for (size_t mc = m_pad; mc >= 32; mc -= 32) {
        for (size_t nc = hex_align_down(hex_smin(n, 2048), 64); nc >= 64; nc -= 64) {
            struct hmx_int8_layout L;
            hmx_int8_layout_build(&L, k, mc, nc, aligned_tile_size);
            if (L.total > budget) { continue; }
            const size_t mb = (m + mc - 1) / mc, nb = (n + nc - 1) / nc;
            const size_t cost = mb * n * HTP_MM_HMX_COST_W_DEQUANT + nb * mb * 64;
            if (cost < best_cost) { best_cost = cost; best_mc = mc; best_nc = nc; }
            break;
        }
        if (mc == 32) { break; }
    }
    if (!best_mc) { return false; }
    *mc_out = best_mc; *nc_out = best_nc;
    return true;
}

static void hmx_int8_meta_ptrs(struct hmx_int8_meta * M, uint8_t * base, size_t off, const struct hmx_int8_layout * L, size_t nc) {
    uint8_t * p = base + off;
    M->bias = p;                                   p += hex_align_up(L->bias_bytes, 256);
    M->s_n = (float *) p;                          p += hex_align_up(nc * 4, 256);
    M->col_mul = (float *) p;                      p += hex_align_up(nc * 4, 256);
    M->colsum = (int32_t *) p;                     p += hex_align_up(nc * 4, 256);
    M->abssum = (int32_t *) p;
}

static int hmx_mm_nx_geglu_2d_int8(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    struct htp_context * ctx = octx->ctx;
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_tensor * restrict w_gate = octx->src[0];
    const struct htp_tensor * restrict w_up   = octx->src[1];
    const struct htp_tensor * restrict act    = octx->src[2];
    const struct htp_tensor * restrict dst    = octx->dsts[0];

    const int    k          = (int) act->ne[0];
    const int    m          = (int) (act->ne[1] * act->ne[2] * act->ne[3]);
    const int    act_stride = (int) (act->nb[1] / sizeof(float));
    const size_t n          = w_gate->ne[1];
    const bool   tiled_out  = (octx->op == HTP_OP_MUL_MAT_NX_GEGLU_T);

    if (w_gate->type != HTP_TYPE_Q4_0 || w_up->type != HTP_TYPE_Q4_0 || act->type != HTP_TYPE_F32 || k % 32 != 0 ||
        n % 64 != 0 || m < 64 || w_up->ne[1] != n || ctx->mdev.count > 1 || (act->data & (VLEN - 1)) != 0 ||
        (tiled_out && ((size_t) dst->ne[0] != n || (dst->data & (VLEN - 1)) != 0))) {
        return HTP_STATUS_NO_SUPPORT;
    }
#ifdef HTP_Q4_0_TILE_LAYOUT_V2
    // Merge guard: the expand below (hmx_int8_expand_q4_0) reads the legacy Q4_0 tile bytes; the tiles are v2 here.
    // Refuse unconditionally; the caller falls through to the fp16 HMX geglu (the host also never selects this path).
    return HTP_STATUS_NO_SUPPORT;
#endif

    const uint32_t tile_size         = kparams->tile_size;
    const uint32_t aligned_tile_size = kparams->aligned_tile_size;
    size_t mc = 0, nc = 0;
    if (!hmx_int8_solve_geglu(k, n, m, aligned_tile_size, ctx->vtcm_size, &mc, &nc)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((tiled_out && (mc % 32)) || mc > HMX_INT8_MAX_ROWS) { return HTP_STATUS_NO_SUPPORT; }

    const int n_k_tiles = k / 32;
    const int n_threads = octx->n_threads;
    struct hmx_int8_layout L;
    hmx_int8_layout_build(&L, k, mc, nc, aligned_tile_size);

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    uint8_t * a_u8     = base + L.off_act;
    float *   a_scale  = (float *) (base + L.off_ascale);
    uint32_t * a_scale16 = (uint32_t *) (base + L.off_ascale16);
    uint32_t   as16_ddr[HMX_INT8_MAX_ROWS / 2] __attribute__((aligned(128)));
    (void) a_scale16;
    uint8_t * raw[2]   = { base + L.off_raw[0], base + L.off_raw[1] };
    uint8_t * w8[2]    = { base + L.off_w8[0], base + L.off_w8[1] };
    uint8_t * obuf[4]  = { base + L.off_obuf[0], base + L.off_obuf[1], base + L.off_obuf[2], base + L.off_obuf[3] };
    struct hmx_int8_meta meta[4];
    for (int i = 0; i < 4; ++i) { hmx_int8_meta_ptrs(&meta[i], base, L.off_meta[i], &L, nc); }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    dma_queue * weight_dma = ctx->dma[0];
    const struct htp_tensor * ws[2] = { w_gate, w_up };

    __fp16 * const dst_tiled = (__fp16 *) dst->data;
    float *  const dst_ptr   = (float *) dst->data;
    const size_t dst_stride  = dst->nb[1] / sizeof(float);
    const int dst_cols       = (int) dst->ne[0];
    const int dst_k_tiles    = (int) n / 32;
    const int n_chunk_cnt    = hmx_ceil_div(n, nc);
    const int n_steps        = 2 * n_chunk_cnt;

    #define GEGLU8_PUSH_W(step, buf) do {                                                                       \
        const int    _c    = (step) / 2;                                                                        \
        const int    _w    = (step) % 2;                                                                        \
        const size_t _nc   = (size_t) _c * nc;                                                                  \
        const size_t _cols = hex_smin(n - _nc, nc);                                                             \
        dma_queue_push(weight_dma, dma_make_data((buf), ws[_w]->data + _nc * ws[_w]->nb[1]),                    \
                       aligned_tile_size, tile_size, tile_size, (_cols / 32) * n_k_tiles);                      \
    } while (0)

    hmx_int8_job_t job_slots[4];
    #define GEGLU8_EPILOGUE(c_, mr_, n_rows_) do {                                                              \
        const size_t _ncc  = (size_t) (c_) * nc;                                                                \
        const size_t _cols = hex_smin(n - _ncc, nc);                                                            \
        const int    _pb   = ((c_) % 2) * 2;                                                                    \
        const int    _dcols = dst_cols - (int) _ncc;                                                            \
        if (tiled_out) {                                                                                        \
            int8_geglu_state_t gs = { ctx->trace, dst_tiled + ((mr_) / 32) * (size_t) dst_k_tiles * 1024,       \
                                      obuf[_pb], obuf[_pb + 1], meta[_pb].col_mul, meta[_pb + 1].col_mul,       \
                                      NULL, NULL, as16_ddr, (uint32_t) hmx_ceil_div((n_rows_), 32), (uint32_t) (_cols / 32),\
                                      (uint32_t) (_ncc / 32), (uint32_t) dst_k_tiles };                         \
            const int nth = (int) hex_smin((uint32_t) n_threads, gs.n_ct);                                      \
            if (nth <= 1) { int8_geglu_tiled_worker_fn(1, 0, &gs); } else { worker_pool_run_func(ctx->worker_pool, int8_geglu_tiled_worker_fn, &gs, nth); } \
        } else if (_dcols > 0) {                                                                                \
            for (int _w = 0; _w < 2; ++_w) {                                                                    \
                int8_convert_state_t cs = { ctx->trace, obuf[_pb + _w], meta[_pb + _w].col_mul, as16_ddr,       \
                                            (uint32_t) hmx_ceil_div((n_rows_), 32), (uint32_t) (_cols / 32) };  \
                const int nth = (int) hex_smin((uint32_t) n_threads, cs.n_ct);                                  \
                if (nth <= 1) { int8_convert_worker_fn(1, 0, &cs); } else { worker_pool_run_func(ctx->worker_pool, int8_convert_worker_fn, &cs, nth); } \
            }                                                                                                   \
            transfer_output_geglu_threaded(ctx, dst_ptr + (mr_) * dst_stride + _ncc, (const __fp16 *) obuf[_pb],\
                                           (const __fp16 *) obuf[_pb + 1], (int) (n_rows_), (int) _cols, (int) dst_stride, _dcols, n_threads); \
        }                                                                                                       \
    } while (0)

    for (size_t mr = 0; mr < (size_t) m; mr += mc) {
        const size_t n_rows = hex_smin((size_t) m - mr, mc);
        const uint32_t n_rt = hmx_ceil_div(n_rows, 32);

        memset(a_scale, 0, hex_align_up((size_t) n_rt * 32, 64) * sizeof(float));
        {
            int8_quant_state_t qs = { ctx, ctx->trace, a_u8, act->data + mr * act_stride * sizeof(float), a_scale,
                                      (float *) (base + L.off_stage), (uint32_t) n_rows, (uint32_t) k, (uint32_t) act_stride, n_rt, NULL, NULL, NULL, false, 0, n_rt * 16, 0, NULL, 0, 0, 0 };
            const int nth = (int) hex_smin((uint32_t) n_threads, n_rt);
            if (nth <= 1) { int8_quant_worker_fn(1, 0, &qs); } else { worker_pool_run_func(ctx->worker_pool, int8_quant_worker_fn, &qs, nth); }
        }
        for (uint32_t r = 0; r < n_rt * 32; r += 64) {
            hvx_vmem((__fp16 *) as16_ddr + r) = hvx_vec_f32_to_f16(hvx_vmem(a_scale + r), hvx_vmem(a_scale + r + 32));
        }

        GEGLU8_PUSH_W(0, raw[0]);
        if (n_steps > 1) { GEGLU8_PUSH_W(1, raw[1]); }

        for (int s = 0; s < n_steps; ++s) {
            const int    c      = s / 2;
            const int    w      = s % 2;
            const size_t ncc    = (size_t) c * nc;
            const size_t n_cols = hex_smin(n - ncc, nc);
            const int    ob     = (c % 2) * 2 + w;

            void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;

            {
                int8_expand_state_t es = { ctx->trace, w8[w], (const uint8_t *) curr_raw, meta[ob], (uint32_t) (n_cols / 32),
                                           (uint32_t) n_k_tiles, aligned_tile_size, (uint32_t) n_cols, (uint32_t) 4 };
                const int nth = (int) hex_smin((uint32_t) n_threads, es.n_ct);
                if (nth <= 1) { int8_expand_worker_fn(1, 0, &es); } else { worker_pool_run_func(ctx->worker_pool, int8_expand_worker_fn, &es, nth); }
            }

            if (s + 2 < n_steps) { GEGLU8_PUSH_W(s + 2, curr_raw); }

            job_slots[ob] = (hmx_int8_job_t) { obuf[ob], a_u8, w8[w], meta[ob].bias, n_rt, (uint32_t) (n_cols / 64), (uint32_t) n_k_tiles, (size_t) n_k_tiles * HMX_INT8_CROUTON_W, 512 };
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_int8_job_fn, &job_slots[ob]));

            if (s > 0) { hmx_queue_pop(ctx->hmx_queue); }

            if (w == 1 && c > 0) { GEGLU8_EPILOGUE(c - 1, mr, n_rows); }
        }

        hmx_queue_pop(ctx->hmx_queue);
        GEGLU8_EPILOGUE(n_chunk_cnt - 1, mr, n_rows);
    }
    #undef GEGLU8_PUSH_W
    #undef GEGLU8_EPILOGUE

    return HTP_STATUS_OK;
}


// Rotated int8 gate+up: weights come as pre-rotated int8 records (hex-int8-pack.h), the activation rows are rotated on HVX.
static int hmx_mm_nx_geglu_2d_int8_rot(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    struct htp_context * ctx = octx->ctx;
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_tensor * restrict w_gate = octx->src[0];
    const struct htp_tensor * restrict w_up   = octx->src[1];
    const struct htp_tensor * restrict act    = octx->src[2];
    const struct htp_tensor * restrict dst    = octx->dsts[0];

    const int    k          = (int) act->ne[0];
    const int    m          = (int) (act->ne[1] * act->ne[2] * act->ne[3]);
    const int    act_stride = (int) (act->nb[1] / sizeof(float));
    const size_t n          = w_gate->ne[1];
    const bool   tiled_out  = (octx->op == HTP_OP_MUL_MAT_NX_GEGLU_T);

    if (w_gate->type != HTP_TYPE_Q4_0 || w_up->type != HTP_TYPE_Q4_0 || act->type != HTP_TYPE_F32 || k != 1536 ||
        n % 64 != 0 || m < 64 || w_up->ne[1] != n || ctx->mdev.count > 1 || (act->data & (VLEN - 1)) != 0 ||
        (tiled_out && ((size_t) dst->ne[0] != n || (dst->data & (VLEN - 1)) != 0))) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const int      n_k_tiles = k / 32;
    const size_t   rec_bytes = (size_t) n_k_tiles * 2048 + 2048;                // one 64-column record (2 KB multiple)
    // int8 data starts after the tiled Q4_0 data of the tensor (128 B aligned), see ggml_hexagon_int8_rot_offset()
    const size_t   int8_off  = hex_align_up((size_t) (k / 32) * 18 * hex_align_up(n, 32), 128);   // = (n/32)*(k/32)*576 tiled bytes

    // VTCM: act u8 | row scales | f32 stage | [2 raw record chunks] | meta (col_mul) | 4 output buffers
    // chunk = nc columns = nc/64 records; raw chunk holds records contiguously
    size_t mc = 0, nc = 0;
    {
        const size_t m_pad = hex_align_up(m, 32);
        size_t best = SIZE_MAX;
        for (size_t mc_c = m_pad; mc_c >= 32; mc_c -= 32) {
            for (size_t nc_c = hex_align_down(hex_smin(n, 1024), 64); nc_c >= 64; nc_c -= 64) {
                const size_t act_b = (mc_c / 32) * n_k_tiles * HMX_INT8_CROUTON_A;
                const size_t tot = hex_align_up(act_b, 2048) + hex_align_up(mc_c * 4, 256) + hex_align_up(mc_c * 2, 256) + hex_smin(8u, hmx_ceil_div(mc_c, 32)) * 4 * (size_t) k * 4 + (kparams->smooth ? hex_align_up((size_t) k * 4, 2048) : 0) +
                                   2 * hex_align_up((nc_c / 64) * rec_bytes, 2048) + 4 * hex_align_up(2 * nc_c * 4, 256) +
                                   4 * hex_align_up((mc_c / 32) * (nc_c / 32) * 2048, 2048) + k * 4 + 4096;
                if (tot > ctx->vtcm_size) { continue; }
                const size_t mb = (m + mc_c - 1) / mc_c, nb = (n + nc_c - 1) / nc_c;
                const size_t cost = mb * n * 2 + nb * mb * 64;
                if (cost < best) { best = cost; mc = mc_c; nc = nc_c; }
                break;
            }
            if (mc_c == 32) { break; }
        }
    }
    if (!mc || mc > HMX_INT8_MAX_ROWS || (tiled_out && (mc % 32))) { return HTP_STATUS_NO_SUPPORT; }

    const int n_threads = octx->n_threads;
    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    size_t off = 0;
    #define ROT_ALLOC(var, bytes) do { var = off; off += hex_align_up((bytes), 2048); } while (0)
    size_t o_act, o_asc, o_asc16, o_stage, o_raw[2], o_meta[4], o_obuf[4], o_sign;
    ROT_ALLOC(o_act, (mc / 32) * n_k_tiles * HMX_INT8_CROUTON_A);
    ROT_ALLOC(o_asc, mc * 4);
    ROT_ALLOC(o_asc16, mc * 2);
    ROT_ALLOC(o_stage, hex_smin(8u, hmx_ceil_div(mc, 32)) * 4 * (size_t) k * 4);
    ROT_ALLOC(o_sign, (size_t) k * 4);
    size_t o_smooth = 0;
    if (kparams->smooth) { ROT_ALLOC(o_smooth, (size_t) k * 4); }
    for (int i = 0; i < 2; ++i) { ROT_ALLOC(o_raw[i], (nc / 64) * rec_bytes); }
    for (int i = 0; i < 4; ++i) { ROT_ALLOC(o_meta[i], 2 * nc * 4); }
    for (int i = 0; i < 4; ++i) { ROT_ALLOC(o_obuf[i], (mc / 32) * (nc / 32) * 2048); }
    #undef ROT_ALLOC
    if (off > ctx->vtcm_size) { return HTP_STATUS_VTCM_TOO_SMALL; }

    uint8_t *  a_u8      = base + o_act;
    float *    a_scale   = (float *) (base + o_asc);
    static float g_rot_signs[1536] __attribute__((aligned(128)));
    static int   g_rot_signs_init = 0;
    if (!g_rot_signs_init) { hmx_int8_make_signs(g_rot_signs, 1536); g_rot_signs_init = 1; }
    float *    signs     = (float *) (base + o_sign);
    hmx_int8_make_signs(signs, 1536);
    (void) g_rot_signs; (void) g_rot_signs_init;
    uint32_t   as16_ddr[HMX_INT8_MAX_ROWS / 2] __attribute__((aligned(128)));
    uint8_t *  raw[2]    = { base + o_raw[0], base + o_raw[1] };
    uint8_t *  obuf[4]   = { base + o_obuf[0], base + o_obuf[1], base + o_obuf[2], base + o_obuf[3] };
    float *    colmul[4] = { (float *) (base + o_meta[0]), (float *) (base + o_meta[1]), (float *) (base + o_meta[2]), (float *) (base + o_meta[3]) };
    float *    coladd[4] = { (float *) (base + o_meta[0]) + nc, (float *) (base + o_meta[1]) + nc, (float *) (base + o_meta[2]) + nc, (float *) (base + o_meta[3]) + nc };
    (void) o_asc16;

    // static activation smoothing: 1/s_k behind the packed records of the gate weight; copy to VTCM once
    const float * inv_s_vtcm = NULL;
    if (kparams->smooth) {
        float * dstv = (float *) (base + o_smooth);
        dma_queue_push(ctx->dma[0], dma_make_data(dstv, w_gate->data + int8_off + (n / 64) * rec_bytes), (size_t) k * 4, (size_t) k * 4, (size_t) k * 4, 1);
        (void) dma_queue_pop(ctx->dma[0]);
        inv_s_vtcm = dstv;
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    dma_queue * weight_dma = ctx->dma[0];
    const struct htp_tensor * ws[2] = { w_gate, w_up };

    __fp16 * const dst_tiled = (__fp16 *) dst->data;
    float *  const dst_ptr   = (float *) dst->data;
    const size_t dst_stride  = dst->nb[1] / sizeof(float);
    const int dst_cols       = (int) dst->ne[0];
    const int dst_k_tiles    = (int) n / 32;
    const int n_chunk_cnt    = hmx_ceil_div(n, nc);
    const int n_steps        = 2 * n_chunk_cnt;

    #define ROT_PUSH_W(step, buf) do {                                                                          \
        const int    _c   = (step) / 2;                                                                         \
        const int    _w   = (step) % 2;                                                                         \
        const size_t _nc  = (size_t) _c * nc;                                                                   \
        const size_t _cols = hex_smin(n - _nc, nc);                                                             \
        const size_t _bytes = (_cols / 64) * rec_bytes;                                                         \
        dma_queue_push(weight_dma, dma_make_data((buf), ws[_w]->data + int8_off + (_nc / 64) * rec_bytes),      \
                       _bytes, _bytes, _bytes, 1);                                                              \
    } while (0)

    hmx_int8_job_t job_slots[4];
    #define ROT_EPILOGUE(c_, mr_, n_rows_) do {                                                                 \
        const size_t _ncc  = (size_t) (c_) * nc;                                                                \
        const size_t _cols = hex_smin(n - _ncc, nc);                                                            \
        const int    _pb   = ((c_) % 2) * 2;                                                                    \
        const int    _dcols = dst_cols - (int) _ncc;                                                            \
        if (tiled_out) {                                                                                        \
            int8_geglu_state_t gs = { ctx->trace, dst_tiled + ((mr_) / 32) * (size_t) dst_k_tiles * 1024,       \
                                      obuf[_pb], obuf[_pb + 1], colmul[_pb], colmul[_pb + 1],                   \
                                      kparams->bias2 ? coladd[_pb] : NULL, kparams->bias2 ? coladd[_pb + 1] : NULL, as16_ddr, \
                                      (uint32_t) hmx_ceil_div((n_rows_), 32), (uint32_t) (_cols / 32),\
                                      (uint32_t) (_ncc / 32), (uint32_t) dst_k_tiles };                         \
            const int nth = (int) hex_smin((uint32_t) n_threads, gs.n_ct);                                      \
            if (nth <= 1) { int8_geglu_tiled_worker_fn(1, 0, &gs); } else { worker_pool_run_func(ctx->worker_pool, int8_geglu_tiled_worker_fn, &gs, nth); } \
        } else if (_dcols > 0) {                                                                                \
            for (int _w = 0; _w < 2; ++_w) {                                                                    \
                int8_convert_state_t cs = { ctx->trace, obuf[_pb + _w], colmul[_pb + _w], as16_ddr,             \
                                            (uint32_t) hmx_ceil_div((n_rows_), 32), (uint32_t) (_cols / 32) };  \
                const int nth = (int) hex_smin((uint32_t) n_threads, cs.n_ct);                                  \
                if (nth <= 1) { int8_convert_worker_fn(1, 0, &cs); } else { worker_pool_run_func(ctx->worker_pool, int8_convert_worker_fn, &cs, nth); } \
            }                                                                                                   \
            transfer_output_geglu_threaded(ctx, dst_ptr + (mr_) * dst_stride + _ncc, (const __fp16 *) obuf[_pb],\
                                           (const __fp16 *) obuf[_pb + 1], (int) (n_rows_), (int) _cols, (int) dst_stride, _dcols, n_threads); \
        }                                                                                                       \
    } while (0)

    for (size_t mr = 0; mr < (size_t) m; mr += mc) {
        const size_t n_rows = hex_smin((size_t) m - mr, mc);
        const uint32_t n_rt = hmx_ceil_div(n_rows, 32);

        memset(a_scale, 0, hex_align_up((size_t) n_rt * 32, 64) * sizeof(float));
        {
            int8_quant_state_t qs = { ctx, ctx->trace, a_u8, act->data + mr * act_stride * sizeof(float), a_scale,
                                      (float *) (base + o_stage), (uint32_t) n_rows, (uint32_t) k, (uint32_t) act_stride, n_rt, (kparams->int_mm == 2 || kparams->int_mm == 4) ? signs : NULL, NULL, NULL, false, 0, n_rt * 16, 0, inv_s_vtcm, 0, 0, (uint32_t) ((kparams->int8_split >> 4) & 1) | ((uint32_t) ((kparams->int8_split >> 5) & 7) << 1) };   // 2/4: rotate activations; 3/5: weights-only rotation, act unrotated
            const int nth = (int) hex_smin((uint32_t) n_threads, n_rt);
            if (nth <= 1) { int8_quant_worker_fn(1, 0, &qs); } else { worker_pool_run_func(ctx->worker_pool, int8_quant_worker_fn, &qs, nth); }
        }
        for (uint32_t r = 0; r < n_rt * 32; r += 64) {
            hvx_vmem((__fp16 *) as16_ddr + r) = hvx_vec_f32_to_f16(hvx_vmem(a_scale + r), hvx_vmem(a_scale + r + 32));
        }

        ROT_PUSH_W(0, raw[0]);
        if (n_steps > 1) { ROT_PUSH_W(1, raw[1]); }

        // step s uses raw[s % 2] and obuf[ob]; raw[s % 2] is refilled with step s + 2 only after job s has retired
        for (int s = 0; s < n_steps; ++s) {
            const int    c      = s / 2;
            const int    w      = s % 2;
            const size_t ncc    = (size_t) c * nc;
            const size_t n_cols = hex_smin(n - ncc, nc);
            const int    ob     = (c % 2) * 2 + w;

            uint8_t * curr_raw = (uint8_t *) dma_queue_pop(weight_dma).dst;

            {
                const size_t n_rec = n_cols / 64;
                for (size_t r = 0; r < n_rec; ++r) {
                    const float * cm = (const float *) (curr_raw + r * rec_bytes + (size_t) n_k_tiles * 2048 + 512);
                    hvx_vmem(colmul[ob] + r * 64)      = hvx_vmem(cm);
                    hvx_vmem(colmul[ob] + r * 64 + 32) = hvx_vmem(cm + 32);
                    const float * ca = (const float *) (curr_raw + r * rec_bytes + (size_t) n_k_tiles * 2048 + 768);
                    hvx_vmem(coladd[ob] + r * 64)      = hvx_vmem(ca);
                    hvx_vmem(coladd[ob] + r * 64 + 32) = hvx_vmem(ca + 32);
                }
            }

            job_slots[ob] = (hmx_int8_job_t) { obuf[ob], a_u8, curr_raw, curr_raw + (size_t) n_k_tiles * 2048, n_rt,
                                               (uint32_t) (n_cols / 64), (uint32_t) n_k_tiles, rec_bytes, rec_bytes };
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_int8_job_fn, &job_slots[ob]));

            if (s > 0) {
                hmx_queue_pop(ctx->hmx_queue);                       // retire job s - 1
                if (s + 1 < n_steps) {
                    ROT_PUSH_W(s + 1, raw[(s - 1) % 2]);             // its raw buffer takes step s + 1 (same parity as s - 1)
                }
            }
            if (w == 1 && c > 0) { ROT_EPILOGUE(c - 1, mr, n_rows); }
        }

        hmx_queue_pop(ctx->hmx_queue);
        ROT_EPILOGUE(n_chunk_cnt - 1, mr, n_rows);
    }
    #undef ROT_PUSH_W
    #undef ROT_EPILOGUE

    return HTP_STATUS_OK;
}

// one sign table for all int8 jobs (i/sqrt(512), the Hadamard rotation pattern; see hex-int8-pack.h)
static const float * get_int8_signs(void) {
    static float tbl[4096] __attribute__((aligned(128)));
    static int   init = 0;
    if (!init) { hmx_int8_make_signs(tbl, 4096); init = 1; }
    return tbl;
}

// ---- int8 packet-loaded matmul with f32 output (qkv, o-proj, down, PLE model_proj) ----
// Weights are the packed int8 records produced by hex-int8-pack.h behind the Q4_0 tensor (GGML_HEXAGON_INT_MM >= 2).

typedef struct {
    struct htp_thread_trace * traces;
    float *                   dst;          // [rows][dst_stride]
    const uint8_t *           tiles16;      // [n_rt][n_ct] 2 KB int16 tiles (pass 1)
    const uint8_t *           tiles16_b;    // A16 pass 2 tiles (NULL for A8)
    const float *             cm;           // [n_ct * 32]
    const float *             a_scale;      // [rows]
    const float *             a_scale_b;    // A16 pass 2 row scales (NULL for A8)
    const float *             cm_add;       // per-column additive zero-point remainder (two-level bias, see hex-int8-pack.h)
    uint32_t                  n_rt, n_ct, dst_stride, dst_cols, col0, c_first_base;
    uint32_t                  n_rows_stop;   // rows >= this are padding (m not a multiple of 32)
    float *                   sink;          // timing ablation only (INT_ABL bit 1): stores go to this per-thread VTCM area instead of DDR
} int8_f32_state_t;

// Threads split the row pairs (not the column tiles) and walk the column tiles inside, so each thread writes whole
// contiguous destination rows (like the fp16 transfer) and every thread has work even when nc is as small as 64.
static void int8_f32_out_worker_fn(unsigned int n, unsigned int i, void * data) {
    int8_f32_state_t * st = (int8_f32_state_t *) data;
    const uint32_t n_pairs = st->n_rt * 16;
    const uint32_t p0 = (n_pairs * i) / n, p1 = (n_pairs * (i + 1)) / n;
    if (p0 == p1 || st->col0 >= st->dst_cols) { return; }
    htp_trace_event_start(&st->traces[i], HTP_TRACE_EVT_HVX_O_PROC, p0);
    const uint32_t ct_valid = hex_smin(st->n_ct, (st->dst_cols - st->col0 + 31) / 32);
    for (uint32_t p = p0; p < p1; ++p) {
        const uint32_t rt = p >> 4, v = p & 15;
        const uint32_t row0 = rt * 32 + 2 * v;
        if (row0 >= st->n_rows_stop) { break; }
        const bool second = (row0 + 1) < st->n_rows_stop;
        const HVX_Vector as0 = hvx_vec_splat_f32(st->a_scale[row0]);
        const HVX_Vector as1 = hvx_vec_splat_f32(st->a_scale[row0 + 1]);
        HVX_Vector bs0 = Q6_V_vzero(), bs1 = Q6_V_vzero();
        if (st->tiles16_b) { bs0 = hvx_vec_splat_f32(st->a_scale_b[row0]); bs1 = hvx_vec_splat_f32(st->a_scale_b[row0 + 1]); }
        const HVX_Vector ras0 = hvx_vec_add_f32_f32(as0, bs0), ras1 = hvx_vec_add_f32_f32(as1, bs1);
        float * o0 = st->dst + (size_t) row0 * st->dst_stride + st->col0;
        float * o1 = o0 + st->dst_stride;
        const uint8_t * t  = st->tiles16 + ((size_t) rt * st->n_ct * 2048) + (size_t) v * 128;
        const uint8_t * tb = st->tiles16_b ? st->tiles16_b + ((size_t) rt * st->n_ct * 2048) + (size_t) v * 128 : NULL;
        #pragma unroll(2)
        for (uint32_t ct = 0; ct < ct_valid; ++ct) {
            const HVX_Vector cmv = hvx_vmem(st->cm + ct * 32);
            const HVX_Vector x = *(const HVX_Vector *) (t + (size_t) ct * 2048);
            HVX_Vector a0 = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(x, 16), 16)), as0);
            HVX_Vector a1 = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(x, 16)), as1);
            if (tb) {
                const HVX_Vector xb = *(const HVX_Vector *) (tb + (size_t) ct * 2048);
                a0 = hvx_vec_add_f32_f32(a0, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(xb, 16), 16)), bs0));
                a1 = hvx_vec_add_f32_f32(a1, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(xb, 16)), bs1));
            }
            HVX_Vector s0 = hvx_vec_mul_f32_f32(a0, cmv);
            HVX_Vector s1 = hvx_vec_mul_f32_f32(a1, cmv);
            if (st->cm_add) {
                // remainder of the truncated bias word lives in the accumulator domain: it scales with the row factor of
                // every pass (A16: both passes carry the same bias word)
                const HVX_Vector cav = hvx_vmem(st->cm_add + ct * 32);
                s0 = hvx_vec_add_f32_f32(s0, hvx_vec_mul_f32_f32(cav, ras0));
                s1 = hvx_vec_add_f32_f32(s1, hvx_vec_mul_f32_f32(cav, ras1));
            }
            const uint32_t col = ct * 32;
            const uint32_t ncol = hex_smin(32u, st->dst_cols - (st->col0 + col));
            if (st->sink) {   // ablation: same ALU work, stores to VTCM
                float * sk = st->sink + (size_t) i * 1024 + (col & 511);
                *(HVX_Vector *) sk = s0; *(HVX_Vector *) (sk + 512) = s1;
                continue;
            }
            if (ncol == 32) {
                hvx_vmemu(o0 + col) = s0;
                if (second) { hvx_vmemu(o1 + col) = s1; }
            } else {
                hvx_vec_store_u(o0 + col, ncol * 4, s0);
                if (second) { hvx_vec_store_u(o1 + col, ncol * 4, s1); }
            }
        }
    }
    htp_trace_event_stop(&st->traces[i], HTP_TRACE_EVT_HVX_O_PROC, p0);
}

typedef struct {
    uint8_t *       out;
    const uint8_t * act;
    const uint8_t * wgt;
    const uint8_t * bias;
    uint32_t        n_row_tiles, n_groups, n_k;
    size_t          wgt_stride, bias_stride;
} hmx_int8_job1_t;

static void hmx_int8_job1_fn(void * data) {
    hmx_int8_job1_t * j = (hmx_int8_job1_t *) data;
    hmx_int8_core_dot_chunk(j->out, j->act, j->wgt, j->bias, j->n_row_tiles, j->n_groups, j->n_k, j->wgt_stride, j->bias_stride);
}

static int hmx_mm_nx_2d_int8_f32(struct htp_ops_context * octx, const struct htp_mm_kernel_params * kparams) {
    struct htp_context * ctx = octx->ctx;
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    // plain MUL_MAT (o-proj, down, PLE) has kparams->n_weights == 0: one weight, act in src[1], dst in dsts[0]
    const uint32_t n_weights = kparams->n_weights ? kparams->n_weights : 1;
    if (n_weights > 4) { return HTP_STATUS_NO_SUPPORT; }
    if (octx->op == HTP_OP_MUL_MAT_ADD) { return HTP_STATUS_NO_SUPPORT; }
    const bool act_tiled = (octx->op == HTP_OP_MUL_MAT_AT);   // x = tiled f16 activation from the fused GEGLU_T producer
    // q8_in: the activation is the int8 payload + row scales written by RMS_NORM_MUL_Q8 (host sets kparams->q8_in only when it emitted that
    // producer for this exact tensor). The payload replaces the f32 rows; every requirement below must hold or the op reports NO_SUPPORT
    // loudly (the host guarantees the int8 path runs, so a refusal here is a bug and must never fall back to the f32 path).
    const bool q8_in = kparams->q8_in != 0;

    const struct htp_tensor * restrict act = octx->src[n_weights];
    const int    k          = (int) act->ne[0];
    const int    m          = (int) (act->ne[1] * act->ne[2] * act->ne[3]);
    const int    act_stride = (int) (act->nb[1] / sizeof(float));

    if (act->type != HTP_TYPE_F32 || k % 32 != 0 || m < 1 || m > HMX_INT8_MAX_ROWS || ctx->mdev.count > 1 ||
        (act->data & (VLEN - 1)) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (q8_in && (kparams->int_mm != 3 || kparams->int_mm == 6 || act_tiled || m < 64 || (m % 32) != 0 || kparams->int8_split != 0 || !kparams->smooth || n_weights != 3)) {
        return HTP_STATUS_INVAL_PARAMS;   // payload mode needs plain A8, one quantize block, no m-split, m >= 64 (the producer's own limits)
    }
    // per-weight checks (fused NX with several weights shares one activation); n_max sizes the VTCM layout
    size_t n_max = 0;
    for (uint32_t wi = 0; wi < n_weights; ++wi) {
        const struct htp_tensor * w   = octx->src[wi];
        const struct htp_tensor * dst = octx->dsts[wi];
        if (!w || !dst || (w->type != HTP_TYPE_Q4_0 && w->type != HTP_TYPE_F16) || w->ne[1] % 64 != 0 || dst->ne[0] > w->ne[1] ||
            (dst->data & (VLEN - 1)) != 0) {
            return HTP_STATUS_NO_SUPPORT;
        }
        n_max = hex_smax(n_max, (size_t) w->ne[1]);
    }

    const bool     a16       = (kparams->int_mm == 6);
    const int      n_k_tiles = k / 32;
    const size_t   rec_bytes = (size_t) n_k_tiles * 2048 + 2048;
    const int      n_threads = octx->n_threads;

    // VTCM: act u8 | row scales f32 | f32 quant stage | 2 raw record chunks | 4 col_mul areas | 2 int16 out buffers
    size_t mc = 0, nc = 0;
    {
        const size_t m_pad = hex_align_up(m, 32);
        size_t best = SIZE_MAX;
        for (size_t mc_c = m_pad; mc_c >= 32; mc_c -= 32) {
            for (size_t nc_c = hex_align_down(hex_smin(n_max, 1024), 64); nc_c >= 64; nc_c -= 64) {
                const size_t act_b = (mc_c / 32) * n_k_tiles * HMX_INT8_CROUTON_A;
                const size_t passes = a16 ? 2 : 1;
                const size_t tot = passes * hex_align_up(act_b, 2048) + passes * hex_align_up(mc_c * 4, 256) + hex_smin(8u, hmx_ceil_div(mc_c, 32)) * 4 * (size_t) k * 4 + (kparams->smooth ? hex_align_up((size_t) k * 4, 2048) : 0) +
                                   2 * hex_align_up((nc_c / 64) * rec_bytes, 2048) + 2 * hex_align_up(2 * nc_c * 4, 256) +
                                   passes * 2 * hex_align_up((mc_c / 32) * (nc_c / 32) * 2048, 2048);
                if (tot > ctx->vtcm_size) { continue; }
                // cycles: the packed records are re-streamed once per m block at ~8 B/cycle (measured weight DMA: 9.4 MB in
                // 1.2M cycles for down), plus ~16k cycles of queue/dispatch bubble per chunk step
                const size_t mb = (m + mc_c - 1) / mc_c, nb = (n_max + nc_c - 1) / nc_c;
                const size_t cost = mb * n_max * ((size_t) k + 32) / 8 + nb * mb * 16384;
                if (cost < best) { best = cost; mc = mc_c; nc = nc_c; }
                break;
            }
            if (mc_c == 32) { break; }
        }
    }
    if (!mc) { return HTP_STATUS_NO_SUPPORT; }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    size_t off = 0;
    size_t o_act, o_act2, o_asc, o_asc2, o_stage, o_raw[2], o_meta[2], o_obuf[2], o_obuf2[2];
    #define I8F_ALLOC(var, bytes) do { var = off; off += hex_align_up((bytes), 2048); } while (0)
    I8F_ALLOC(o_act, (mc / 32) * n_k_tiles * HMX_INT8_CROUTON_A);
    o_act2 = o_act; if (a16) { I8F_ALLOC(o_act2, (mc / 32) * n_k_tiles * HMX_INT8_CROUTON_A); }
    I8F_ALLOC(o_asc, mc * 4);
    o_asc2 = o_asc; if (a16) { I8F_ALLOC(o_asc2, mc * 4); }
    I8F_ALLOC(o_stage, hex_smin(8u, hmx_ceil_div(mc, 32)) * 4 * (size_t) k * 4);
    size_t o_smooth = 0;
    if (kparams->smooth) { I8F_ALLOC(o_smooth, (size_t) k * 4); }
    for (int i = 0; i < 2; ++i) { I8F_ALLOC(o_raw[i], (nc / 64) * rec_bytes); }
    for (int i = 0; i < 2; ++i) { I8F_ALLOC(o_meta[i], 2 * nc * 4); }
    for (int i = 0; i < 2; ++i) { I8F_ALLOC(o_obuf[i], (mc / 32) * (nc / 32) * 2048); }
    for (int i = 0; i < 2; ++i) { o_obuf2[i] = o_obuf[i]; if (a16) { I8F_ALLOC(o_obuf2[i], (mc / 32) * (nc / 32) * 2048); } }
    #undef I8F_ALLOC
    if (off > ctx->vtcm_size) { return HTP_STATUS_VTCM_TOO_SMALL; }

    uint8_t *  a_u8     = base + o_act;
    uint8_t *  a_u8_2   = base + o_act2;
    float *    a_scale  = (float *) (base + o_asc);
    float *    a_scale_2 = (float *) (base + o_asc2);
    uint8_t *  raw[2]   = { base + o_raw[0], base + o_raw[1] };
    uint8_t *  obuf[2]  = { base + o_obuf[0], base + o_obuf[1] };
    uint8_t *  obuf2[2] = { base + o_obuf2[0], base + o_obuf2[1] };
    float *    colmul[2] = { (float *) (base + o_meta[0]), (float *) (base + o_meta[1]) };
    float *    coladd[2] = { (float *) (base + o_meta[0]) + nc, (float *) (base + o_meta[1]) + nc };   // col_mul then col_add, nc floats each

    // static activation smoothing: 1/s_k sits behind the packed records of weight 0 (see hex-int8-pack.h); copy it to VTCM once
    const float * inv_s_vtcm = NULL;
    if (kparams->smooth) {
        const struct htp_tensor * w0 = octx->src[0];
        const size_t sm_off = (w0->type == HTP_TYPE_F16 ? hex_align_up((size_t) k * w0->ne[1] * 2, 128) : hex_align_up((size_t) (k / 32) * 18 * hex_align_up((size_t) w0->ne[1], 32), 128)) +
                              (size_t) (w0->ne[1] / 64) * rec_bytes;
        float * dstv = (float *) (base + o_smooth);
        dma_queue_push(ctx->dma[0], dma_make_data(dstv, w0->data + sm_off), (size_t) k * 4, (size_t) k * 4, (size_t) k * 4, 1);
        (void) dma_queue_pop(ctx->dma[0]);
        inv_s_vtcm = dstv;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    dma_queue * weight_dma = ctx->dma[0];

    #define I8F_PUSH(step, buf) do {                                                                         \
        const int    _c    = (step);                                                                         \
        const size_t _nc   = (size_t) _c * nc;                                                               \
        const size_t _cols = hex_smin(n - _nc, nc);                                                          \
        const size_t _bytes = (_cols / 64) * rec_bytes;                                                      \
        dma_queue_push(weight_dma, dma_make_data((buf), w->data + int8_off + (_nc / 64) * rec_bytes),        \
                       _bytes, _bytes, _bytes, 1);                                                           \
    } while (0)

    hmx_int8_job1_t job_slots[4];   // [ob * 2 + pass]; a slot is reused only after its job has been popped

    // Quantize a row-pair range of the current m block. first_w = 1 keeps the caller (owner of the weight DMA queue dma[0],
    // whose FIFO would otherwise hand back weight entries) out of the activation DMA.
    #define I8F_QUANT(mr_, n_rows_, n_rt_, pair0_, pair_n_, first_w_) do {                                         \
        int8_quant_state_t _qs = { ctx, ctx->trace, a_u8,                                                           \
                                   act_tiled ? act->data + (size_t) ((mr_) / 32) * n_k_tiles * 2048 : act->data + (mr_) * act_stride * sizeof(float), a_scale, \
                                   (float *) (base + o_stage), (uint32_t) (n_rows_), (uint32_t) k, (uint32_t) act_stride, (n_rt_), \
                                   kparams->int_mm == 2 ? (const float *) get_int8_signs() : NULL,                  \
                                   a16 ? a_u8_2 : NULL, a16 ? a_scale_2 : NULL, act_tiled, (pair0_), (pair_n_), (first_w_), inv_s_vtcm, (uint32_t) ((kparams->int8_split >> 2) & 1), (uint32_t) ((kparams->int8_split >> 3) & 1), (uint32_t) ((kparams->int8_split >> 4) & 1) | ((uint32_t) ((kparams->int8_split >> 5) & 7) << 1) }; \
        const int _nth = (int) hex_smin(hex_smin((uint32_t) n_threads, (uint32_t) (pair_n_) + (first_w_)),             \
                                        hex_smin(8u, (uint32_t) hmx_ceil_div(mc, 32)));                              \
        if (_nth <= 1) { int8_quant_worker_fn(1, 0, &_qs); } else { worker_pool_run_func(ctx->worker_pool, int8_quant_worker_fn, &_qs, _nth); } \
    } while (0)

    for (size_t mr = 0; mr < (size_t) m; mr += mc) {
        const size_t n_rows = hex_smin((size_t) m - mr, mc);
        const uint32_t n_rt = hmx_ceil_div(n_rows, 32);

        memset(a_scale, 0, hex_align_up((size_t) n_rt * 32, 64) * sizeof(float));
        if (a16) { memset(a_scale_2, 0, hex_align_up((size_t) n_rt * 32, 64) * sizeof(float)); }

        // Two row sub-blocks: sub 0 is quantized up front, sub 1 is quantized by the HVX threads while the HMX works on sub 0
        // (the row tiles are independent in a_u8 / a_scale / the output buffers). The weights are streamed once per sub-block.
        const bool     split    = (kparams->int8_split & 1) && n_rt >= 2 && n_threads >= 3;
        const uint32_t rt_split = split ? (n_rt + 1) / 2 : n_rt;
        if (q8_in) {
            // payload from RMS_NORM_MUL_Q8: n_rt croutons-rows of (k/32) * 2048 B at the m-block start, row scales behind the whole payload.
            // Same bytes the f32 -> int8 quantize would have produced into a_u8 / a_scale for these rows.
            const size_t pay_total = (size_t) (m / 32) * n_k_tiles * HMX_INT8_CROUTON_A;
            const size_t scale_off = hex_align_up(pay_total, 128);
            const uint8_t * pay = (const uint8_t *) (uintptr_t) act->data + (size_t) (mr / 32) * n_k_tiles * HMX_INT8_CROUTON_A;
            const size_t   pbytes = (size_t) n_rt * n_k_tiles * HMX_INT8_CROUTON_A;
            const uint8_t * scl = (const uint8_t *) (uintptr_t) act->data + scale_off + mr * sizeof(float);
            const size_t   sbytes = hex_align_up((size_t) n_rt * 32 * sizeof(float), 128);
            dma_queue_push(ctx->dma[0], dma_make_data(a_u8, pay), pbytes, pbytes, pbytes, 1);
            dma_queue_pop(ctx->dma[0]);
            dma_queue_push(ctx->dma[0], dma_make_data((uint8_t *) a_scale, scl), sbytes, sbytes, sbytes, 1);
            dma_queue_pop(ctx->dma[0]);
            // m % 32 == 0 (required above): no padded rows, the DMA'd scales cover exactly the n_rt * 32 rows
        } else {
            I8F_QUANT(mr, n_rows, n_rt, 0, rt_split * 16, 0);
        }

        for (int sub = 0; sub < (split ? 2 : 1); ++sub) {
        const uint32_t rt_lo   = sub ? rt_split : 0;
        const uint32_t rt_hi   = sub ? n_rt : rt_split;
        const uint32_t n_rt_s  = rt_hi - rt_lo;
        const size_t   row_lo  = (size_t) rt_lo * 32;
        const size_t   rows_s  = hex_smin(n_rows, (size_t) rt_hi * 32) - row_lo;
        const size_t   act_off = (size_t) rt_lo * n_k_tiles * HMX_INT8_CROUTON_A;

        // fused NX: weights share the activation quantization above, then run one after another reusing the same buffers
        for (uint32_t wi = 0; wi < n_weights; ++wi) {
        const struct htp_tensor * restrict w   = octx->src[wi];
        const struct htp_tensor * restrict dst = octx->dsts[wi];
        const size_t   n        = w->ne[1];
        const int      dst_stride = (int) (dst->nb[1] / sizeof(float));
        const int      dst_cols   = (int) dst->ne[0];
        const size_t   int8_off = w->type == HTP_TYPE_F16 ? hex_align_up((size_t) k * n * 2, 128) : hex_align_up((size_t) (k / 32) * 18 * hex_align_up(n, 32), 128);
        const int      n_chunk_cnt = hmx_ceil_div(n, nc);
        float * const  dst_ptr  = (float *) dst->data + (mr + row_lo) * dst_stride;

        I8F_PUSH(0, raw[0]);
        if (n_chunk_cnt > 1) { I8F_PUSH(1, raw[1]); }

        for (int c = 0; c < n_chunk_cnt; ++c) {
            const size_t ncc = (size_t) c * nc;
            const size_t n_cols = hex_smin(n - ncc, nc);
            const int    ob = c % 2;
            uint8_t * curr_raw = (uint8_t *) dma_queue_pop(weight_dma).dst;

            {
                const size_t n_rec = n_cols / 64;
                for (size_t r = 0; r < n_rec; ++r) {
                    const float * cm = (const float *) (curr_raw + r * rec_bytes + (size_t) n_k_tiles * 2048 + 512);
                    hvx_vmem(colmul[ob] + r * 64)      = hvx_vmem(cm);
                    hvx_vmem(colmul[ob] + r * 64 + 32) = hvx_vmem(cm + 32);
                    const float * ca = (const float *) (curr_raw + r * rec_bytes + (size_t) n_k_tiles * 2048 + 768);
                    hvx_vmem(coladd[ob] + r * 64)      = hvx_vmem(ca);
                    hvx_vmem(coladd[ob] + r * 64 + 32) = hvx_vmem(ca + 32);
                }
            }

            job_slots[ob * 2 + 0] = (hmx_int8_job1_t) { obuf[ob], a_u8 + act_off, curr_raw, curr_raw + (size_t) n_k_tiles * 2048, n_rt_s,
                                                        (uint32_t) (n_cols / 64), (uint32_t) n_k_tiles, rec_bytes, rec_bytes };
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_int8_job1_fn, &job_slots[ob * 2 + 0]));
            if (a16) {
                job_slots[ob * 2 + 1] = (hmx_int8_job1_t) { obuf2[ob], a_u8_2 + act_off, curr_raw, curr_raw + (size_t) n_k_tiles * 2048, n_rt_s,
                                                            (uint32_t) (n_cols / 64), (uint32_t) n_k_tiles, rec_bytes, rec_bytes };
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_int8_job1_fn, &job_slots[ob * 2 + 1]));
            }

            // overlap: HMX job c is running; quantize slice c of the second sub-block on workers 1..n-1 (weight DMA owner
            // stays out). One slice per chunk of the first weight, so every pair is done before sub-block 1 starts.
            if (split && sub == 0 && wi == 0) {
                const uint32_t P = (n_rt - rt_split) * 16;
                const uint32_t pa = (uint32_t) (((uint64_t) P * c) / n_chunk_cnt), pb = (uint32_t) (((uint64_t) P * (c + 1)) / n_chunk_cnt);
                if (pb > pa) { I8F_QUANT(mr, n_rows, n_rt, rt_split * 16 + pa, pb - pa, 1); }
            }

            if (c > 0) {
                hmx_queue_pop(ctx->hmx_queue);                       // retire job c-1 (pass 1)
                if (a16) { hmx_queue_pop(ctx->hmx_queue); }          // retire job c-1 (pass 2)
                if (c + 1 < n_chunk_cnt) { I8F_PUSH(c + 1, raw[(c - 1) % 2]); }
            }

            if (c > 0) {
                const size_t ncc_p = (size_t) (c - 1) * nc;
                const uint32_t ncols_p = (uint32_t) hex_smin(n - ncc_p, nc);
                int8_f32_state_t st = { ctx->trace, dst_ptr, obuf[(c - 1) % 2],
                                        a16 ? obuf2[(c - 1) % 2] : NULL, colmul[(c - 1) % 2], a_scale + row_lo, a16 ? a_scale_2 + row_lo : NULL,
                                        kparams->bias2 ? coladd[(c - 1) % 2] : NULL,
                                        n_rt_s, ncols_p / 32, (uint32_t) dst_stride, (uint32_t) dst_cols, (uint32_t) ncc_p, 0,
                                        (uint32_t) rows_s, (kparams->int8_split & 2) ? (float *) (base + o_stage) : NULL };
                const int nth = (int) hex_smin((uint32_t) n_threads, st.n_rt * 16);
                if (nth <= 1) { int8_f32_out_worker_fn(1, 0, &st); } else { worker_pool_run_func(ctx->worker_pool, int8_f32_out_worker_fn, &st, nth); }
            }
        }

        hmx_queue_pop(ctx->hmx_queue);
        if (a16) { hmx_queue_pop(ctx->hmx_queue); }
        {
            const int    c_last = n_chunk_cnt - 1;
            const size_t ncc_p  = (size_t) c_last * nc;
            const uint32_t ncols_p = (uint32_t) hex_smin(n - ncc_p, nc);
            int8_f32_state_t st = { ctx->trace, dst_ptr, obuf[c_last % 2],
                                    a16 ? obuf2[c_last % 2] : NULL, colmul[c_last % 2], a_scale + row_lo, a16 ? a_scale_2 + row_lo : NULL,
                                    kparams->bias2 ? coladd[c_last % 2] : NULL,
                                    n_rt_s, ncols_p / 32, (uint32_t) dst_stride, (uint32_t) dst_cols, (uint32_t) ncc_p, 0,
                                    (uint32_t) rows_s, (kparams->int8_split & 2) ? (float *) (base + o_stage) : NULL };
            const int nth = (int) hex_smin((uint32_t) n_threads, st.n_rt * 16);
            if (nth <= 1) { int8_f32_out_worker_fn(1, 0, &st); } else { worker_pool_run_func(ctx->worker_pool, int8_f32_out_worker_fn, &st, nth); }
        }
        }   // wi
        }   // sub
    }
    #undef I8F_QUANT
    #undef I8F_PUSH

    return HTP_STATUS_OK;
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
    const uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const int act_stride  = (int) (act->nb[1] / act_elem_size);
    const dma_addr_t act_dma_addr = act->data;

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
        case HTP_TYPE_Q3_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q3_k; break;
        case HTP_TYPE_Q2_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q2_k; break;
        case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
        case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
        default:
            return HTP_STATUS_NO_SUPPORT;
    }

    const int n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    const struct fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    const bool is_quant       = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);
    const size_t vtcm_budget  = ctx->vtcm_size;

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
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, 0);

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

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    int m_start = 0;
    int m_rows  = m;
    if (octx->ctx->mdev.count > 1) {
        bool can_split = htp_tensor_can_row_partition(act, act_elem_size);
        if (can_split) {
            for (uint32_t p = 0; p < n_weights; ++p) {
                const struct htp_tensor * restrict dst = octx->dsts[p];
                if (!htp_tensor_can_row_partition(dst, sizeof(float))) {
                    can_split = false;
                    break;
                }
            }
        }
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

    const size_t mr_end = (size_t)(m_start + m_rows);

    if (pipeline) {
        hmx_matmul_job_t job_slots[2];

        for (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            const size_t n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            void *vtcm_weight_bufs[2] = { vtcm_scratch0, vtcm_scratch1 };
            void *vtcm_output_bufs[2] = { vtcm_output,   vtcm_scratch2 };

            struct activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                .n_rows = (int) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_size = act_elem_size,
            };
            transfer_activation_chunk_threaded(&act_params);

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
                const int n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_cols);

                const uint32_t dma_src_stride = is_quant ? tile_size : weight_stride;

                const size_t   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
                const uint32_t height_A0 = is_quant ? hmx_ceil_div(n_cols_A0, 32) * n_k_tiles : n_cols_A0;
                dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], weight),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

                if (1 < n_chunk_cnt) {
                    const size_t   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                    const uint32_t height_A1 = is_quant ? hmx_ceil_div(n_cols_A1, 32) * n_k_tiles : n_cols_A1;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[1], weight + n_chunk_n_cols * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
                }

                for (int i = 0; i < n_chunk_cnt; ++i) {
                    const size_t nc    = i * n_chunk_n_cols;
                    const size_t nc_p2 = nc + 2 * n_chunk_n_cols;

                    const size_t n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                    const size_t n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                    void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;

                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 2], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                    if (i + 2 < n_chunk_cnt) {
                        const uint32_t height_p2 = is_quant ? hmx_ceil_div(n_cols_p2, 32) * n_k_tiles : n_cols_p2;
                        dma_queue_push(weight_dma, dma_make_data(curr_raw, weight + nc_p2 * weight_stride),
                                       dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                    }

                    hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2],
                                        vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                        hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                    if (i > 0) {
                        hmx_queue_pop(ctx->hmx_queue);
                        const size_t nc_prev = (i - 1) * n_chunk_n_cols;
                        const size_t n_cols_prev = hex_smin(n - nc_prev, n_chunk_n_cols);
                        float *output_chunk = dst_ptr + (mr * dst_stride + nc_prev);
                        int chunk_dst_cols = dst_cols - (int)nc_prev;
                        if (chunk_dst_cols > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, NULL, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, 0, chunk_dst_cols, n_threads);
                        }
                    }
                }

                hmx_queue_pop(ctx->hmx_queue);
                const size_t nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
                const size_t n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
                float *output_chunk = dst_ptr + (mr * dst_stride + nc_last);
                int chunk_dst_cols = dst_cols - (int)nc_last;
                if (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, NULL, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, 0, chunk_dst_cols, n_threads);
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
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                .n_rows = (int) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_size = act_elem_size,
            };
            transfer_activation_chunk_threaded(&act_params);

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
                    const uint32_t height = is_quant ? hmx_ceil_div(n_cols, 32) * n_k_tiles : n_cols;
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
                        const uint32_t height_next = is_quant ? hmx_ceil_div(n_cols_next, 32) * n_k_tiles : n_cols_next;
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
            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
                                       b2 * params->bias_nb2 +
                                       b3 * params->bias_nb3) : 0;
            ret = hmx_mm_2d_f32(ctx, params->weight_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                cur_bias_addr, params->bias_bytes,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_weight_batch_data(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->act_elem_size, params->weight_stride * (int)sizeof(__fp16),
                                HTP_TYPE_F16, params->k, params->dst_stride, params->bias_stride, params->n,
                                m_chunk, n_chunk, pipeline, n_threads, act_threads,
                                act_threads_div, k_div, 0, 0, vtcm_size, false);
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
    // N (the weight row count) does not have to be 32-aligned:
    if (params->k % 32 != 0) { return -1; }
    if (!hex_is_aligned(params->dst, VLEN) || (params->act_dma_addr & (VLEN - 1)) != 0) { return -1; }

    const int group_size = params->r2;
    const size_t vtcm_budget  = ctx->vtcm_size;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid if group_size > 1 and it fits within VTCM budget.
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
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_size, false, act_threads, 0, params->bias_bytes);

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

    const bool has_bias      = (params->bias_bytes > 0 && params->bias_addr != 0);
    float   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_bias, has_bias);
    if (has_bias) {
        dma_queue_push(params->weight_dma, dma_make_data(vtcm_bias, params->bias_addr), hex_align_up(params->bias_bytes, 128), 0, params->bias_bytes, 1);
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
                                                      (size_t) mr * params->act_stride * params->act_elem_size;
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
                        .act_elem_size = params->act_elem_size,
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
                    const dma_addr_t second_weight_chunk = weight_group + n_chunk_n_cols * params->weight_stride * sizeof(__fp16);
                    dma_queue_push(weight_dma, dma_make_data(vtcm_scratch1, second_weight_chunk),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_second);
                }

                for (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_cols) {
                    const size_t n_cols      = hex_smin((size_t) params->n - nc, n_chunk_n_cols);
                    const size_t n_col_tiles = hmx_ceil_div((int) n_cols, HTP_MM_HMX_TILE_N_COLS);

                    {
                        void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;

                        const size_t n_cols_tiled = hex_align_up(n_cols, HTP_MM_HMX_TILE_N_COLS);

                        hmx_interleave_rows_to_tiles(vtcm_weight, (const __fp16 *) curr_raw, (uint32_t) n_cols, params->k, params->k, 0, (uint32_t) n_cols_tiled);

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
                            const float *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : NULL;
                            int chunk_dst_cols = params->n - (int)nc;
                            if (chunk_dst_cols > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (int) n_rows, (int) n_cols,
                                                               params->dst_stride, params->bias_stride, chunk_dst_cols, n_threads);
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
    const int m_core = m_end - m_start;
    const int m_core_padded = hex_align_up(m_core > 0 ? m_core : 1, 32);

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
        case HTP_TYPE_Q3_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q3_k; break;
        case HTP_TYPE_Q2_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q2_k; break;
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
                           m_core_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_COST_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_COST_A_CONVERT, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
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
            const uint32_t height = is_quant ? hmx_ceil_div(n_cols, 32) * n_k_tiles : n_cols;
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
                const uint32_t height_next = is_quant ? hmx_ceil_div(n_cols_next, 32) * n_k_tiles : n_cols_next;
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

    int k = (int) src0->ne[0];
    int n = (int) src0->ne[1];
    const int m_total    = (int) act->ne[1];
    const uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    const int act_stride = (int) (act->nb[1] / act_elem_size);
    const int wgt_stride = (int)(src0->nb[1] / sizeof(__fp16));

    // Activation already f16 in HMX tile layout (emitted by a fused geglu producer).
    const bool act_tiled_f16 = (octx->op == HTP_OP_MUL_MAT_AT);

    // The tiled layout carries no column padding and only the 2d kernel understands it.
    if (act_tiled_f16 && (k != (int) src1->ne[0] || kparams->kernel_type == HTP_MM_KERNEL_HMX_F16_BATCHED)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    int m_start = 0;
    int m_rows  = m_total;
    if (octx->ctx->mdev.count > 1) {
        bool can_split = htp_tensor_can_row_partition(dst, sizeof(float)) &&
                         htp_tensor_can_row_partition(act, act_elem_size);
        if (src2 && src2->ne[1] > 1 && !htp_tensor_can_row_partition(src2, sizeof(float))) {
            can_split = false;
        }
        // Must match the producer's granularity so both derive the same tile range.
        const uint32_t unit = act_tiled_f16 ? HTP_MM_HMX_TILE_N_ROWS : 1;
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_total, can_split ? unit : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
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
        src2_bytes = (size_t) kparams->vtcm_bias_size;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    const int dst_stride = (int)(dst->nb[1] / sizeof(float));
    float       * dst_ptr = (float *) dst->data + m_start * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    const dma_addr_t act_addr = act_tiled_f16
        ? act->data + (size_t) (m_start / HTP_MM_HMX_TILE_N_ROWS) * k * HTP_MM_HMX_TILE_N_ROWS * sizeof(__fp16)
        : act->data + (size_t) m_start * act_stride * act_elem_size;

    int ret = -1;
    const int n_threads = kparams->n_threads;
    if (kparams->kernel_type == HTP_MM_KERNEL_HMX_F16_BATCHED) {
        hmx_mm_f16_f32_batched_params_t batch_params = {
            .dst             = dst_ptr,
            .bias_addr       = src2_addr,
            .bias_bytes      = src2_bytes,
            .act_dma_addr    = act_addr,
            .weight          = src0->data,
            .weight_dma      = octx->ctx->dma[0],
            .m               = m_rows,
            .k               = k,
            .n               = n,
            .act_stride      = act_stride,
            .act_elem_size = act_elem_size,
            .weight_stride   = wgt_stride,
            .dst_stride      = dst_stride,
            .bias_stride     = src2_stride,
            .ne02            = ne02,
            .ne03            = ne03,
            .ne12            = ne12,
            .ne13            = ne13,
            .src0_nb2        = src0->nb[2],
            .src0_nb3        = src0->nb[3],
            .act_nb2         = act->nb[2],
            .act_nb3         = act->nb[3],
            .dst_nb2         = dst->nb[2],
            .dst_nb3         = dst->nb[3],
            .bias_nb2        = src2_nb2,
            .bias_nb3        = src2_nb3,
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
            m_rows, k, n, act_stride, act_elem_size, (int) src0->nb[1], (int) src0->type, (int) act->ne[0],
            dst_stride, src2_stride, (int)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_threads,
            kparams->n_act_threads,
            &kparams->div_n_act_threads,
            &kparams->div_ne00_padded,
            kparams->tile_size, kparams->aligned_tile_size, kparams->vtcm_size,
            act_tiled_f16
        );
    }

    if (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        return HTP_STATUS_INTERNAL_ERR;
    }
    return HTP_STATUS_OK;
}

static inline void htp_mm_tensor_collapse_rows(struct htp_tensor * c, const struct htp_tensor * t, uint32_t stride) {
    *c = *t;
    c->ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c->ne[2] = 1;
    c->ne[3] = 1;
    c->nb[1] = stride;
    c->nb[2] = c->nb[1] * c->ne[1];
    c->nb[3] = c->nb[2];
}

static int op_matmul_impl(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const int status = htp_mm_init_context(octx, kparams);
    if (status != HTP_STATUS_OK) {
        return status;
    }

    if (kparams->q8_in) { return HTP_STATUS_INVAL_PARAMS; }   // payload activations exist only for the fused NX op

    if (kparams->n_hmx) {
        if (kparams->int_mm >= 2) {
            const int rc8 = hmx_mm_nx_2d_int8_f32(octx, kparams);
            if (rc8 != HTP_STATUS_NO_SUPPORT) { return rc8; }
        }
        return hmx_mm_op_matmul(octx, kparams);
    }

    // The HVX path reads f32 activations and cannot consume the tiled f16 layout.
    if (octx->op == HTP_OP_MUL_MAT_AT) {
        return HTP_STATUS_NO_SUPPORT;
    }

    return hvx_mm_matmul(octx);
}

int op_matmul(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    if (kparams->collapse) {
        const struct htp_tensor * act  = octx->src[1];
        const struct htp_tensor * dst  = octx->dst;
        const uint32_t s_act = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        const uint32_t sd    = (dst->ne[1] > 1) ? dst->nb[1] : ((dst->ne[2] > 1) ? dst->nb[2] : dst->nb[3]);
        struct htp_tensor act_collapsed, dst_collapsed;
        htp_mm_tensor_collapse_rows(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_rows(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        struct htp_tensor src2_collapsed;
        const struct htp_tensor * src2 = octx->src[2];
        if (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            const uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_rows(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        const int status = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        if (src2) {
            octx->src[2] = src2;
        }
        return status;
    }

    return op_matmul_impl(octx);
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

    const bool mdev_split = (octx->ctx->mdev.count > 1) && htp_tensor_can_row_partition(dst, sizeof(float));
    if (octx->ctx->mdev.count > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
        return HTP_STATUS_OK;
    }
    uint32_t n_active = 0;
    if (mdev_split) {
        for (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            if (matrix_row_counts[a] > 0) n_active++;
        }
    }
    const bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.count);

    uint32_t target_dev = 0;
    for (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;

        const int m_padded = hex_align_up(cne1, 32);
        int m_start = 0, m_end = m_padded;
        if (expert_split) {
            const bool my_expert = (target_dev == octx->ctx->mdev.idx);
            if (++target_dev == octx->ctx->mdev.count) {
                target_dev = 0;
            }
            if (!my_expert) continue;
        } else if (mdev_split) {
            const bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.count;
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            m_start = (int) range.start;
            m_end   = (int) (range.start + range.count);
        }
        if (m_start >= m_end) continue;

        int ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (float*) dst->data, (float*) act->data,
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
        quant_task_func = htp_mm_act_quant_block_func(src0->type);
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
        quant_task_func = htp_mm_act_quant_row_func(src0->type);
    }
    size_t act_row_size   = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, true, false);

    const size_t vtcm_size = L.total_bytes;

    FARF(HIGH, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);

    FARF(HIGH, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         act->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = NULL;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_row_size_padded;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

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

    bool mdev_split = (octx->ctx->mdev.count > 1);
    for (uint32_t p = 0; p < n_weights && mdev_split; ++p) {
        const struct htp_tensor * restrict dst = octx->dsts[p];
        mdev_split = htp_tensor_can_row_partition(dst, sizeof(float));
    }
    if (octx->ctx->mdev.count > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
        return HTP_STATUS_OK;
    }
    uint32_t n_active = 0;
    if (mdev_split) {
        for (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            if (matrix_row_counts[a] > 0) n_active++;
        }
    }
    const bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.count);

    uint32_t target_dev = 0;
    for (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;

        const int m_padded = hex_align_up(cne1, 32);
        int m_start = 0, m_end = m_padded;
        if (expert_split) {
            const bool my_expert = (target_dev == octx->ctx->mdev.idx);
            if (++target_dev == octx->ctx->mdev.count) {
                target_dev = 0;
            }
            if (!my_expert) continue;
        } else if (mdev_split) {
            const bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.count;
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
        quant_task_func = htp_mm_act_quant_block_func(src0->type);
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
        quant_task_func = htp_mm_act_quant_row_func(src0->type);
    }
    size_t act_row_size  = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(act->ne[0]) : htp_mm_q8_0_tiled_row_size(act->ne[0]);

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, true, false);

    const size_t vtcm_size = L.total_bytes;

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * sizeof(float), QK_Q8_0_TILED * sizeof(float));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    FARF(HIGH, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.total_bytes, octx->ctx->vtcm_size, octx->n_threads);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

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
            if (i02 < 0) {
                continue;
            }
            assert(i02 < n_as);

            if (matrix_rows) {
                matrix_rows[i02 * mapping_stride + counts[i02]] = (struct mmid_row_mapping) { id, iid1 };
            }
            counts[i02] += 1;
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
                if (i02 < 0) {
                    continue;
                }
                assert(i02 < n_as);

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
    mmctx->act = act;

    const struct htp_tensor * restrict ids = octx->src[2];
    if (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    const uint32_t src0_nrows = ne01;  // per expert
    const uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    const int n_ids = ids->ne[0];  // n_expert_used
    const int n_as  = ne02;        // n_expert

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
    if (kparams->n_hmx) {
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
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id : hvx_mv_id);
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
        if (htp_tensor_is_extended(octx->dsts[p])) {
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
    if (kparams->n_hmx) {
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
static int op_matmul_nx_impl(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const int status = htp_mm_init_context(octx, kparams);
    if (status != HTP_STATUS_OK) {
        return status;
    }

    if (kparams->q8_in) {
        // the activation bytes are int8 payload + scales: ONLY hmx_mm_nx_2d_int8_f32 can read them. Every other route would read them as f32.
        if (octx->op != HTP_OP_MUL_MAT_NX || !kparams->n_hmx || kparams->int_mm != 3 || kparams->collapse) { return HTP_STATUS_INVAL_PARAMS; }
        const int rc8 = hmx_mm_nx_2d_int8_f32(octx, kparams);
        return rc8 == HTP_STATUS_NO_SUPPORT ? HTP_STATUS_INVAL_PARAMS : rc8;
    }

    // Merge decision: int8 HMX geglu (int branch) first, then HMX f32 geglu;
    // n_hmx == 0 falls through to the HVX geglu kernel (fuse branch).
    if ((octx->op == HTP_OP_MUL_MAT_NX_GEGLU || octx->op == HTP_OP_MUL_MAT_NX_GEGLU_T) && kparams->n_hmx) {
        if (kparams->int_mm) {
            const int rc = kparams->int_mm >= 2 ? hmx_mm_nx_geglu_2d_int8_rot(octx, kparams) : hmx_mm_nx_geglu_2d_int8(octx, kparams);
            if (rc != HTP_STATUS_NO_SUPPORT) { return rc; }
        }
        return hmx_mm_nx_geglu_2d_f32(octx, kparams);
    }

    if (kparams->n_hmx) {
        if (kparams->int_mm >= 2) {
            const int rc8 = hmx_mm_nx_2d_int8_f32(octx, kparams);
            if (rc8 != HTP_STATUS_NO_SUPPORT) { return rc8; }
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
                        src0->type == HTP_TYPE_Q5_K || src0->type == HTP_TYPE_Q3_K ||
                        src0->type == HTP_TYPE_Q2_K);

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
        quant_task_func = htp_mm_act_quant_block_func(src0->type);
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
        quant_task_func = htp_mm_act_quant_row_func(src0->type);
    }

    const size_t act_row_size = htp_mm_weight_has_offset(src0->type)
                               ? htp_mm_q8_1_tiled_row_size(act->ne[0])
                               : htp_mm_q8_0_tiled_row_size(act->ne[0]);

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, false, true);

    const size_t vtcm_size = L.total_bytes;

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * sizeof(float), QK_Q8_0_TILED * sizeof(float));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    // Run fused matmul
    const uint32_t n_matmul_jobs = octx->n_threads;
    worker_callback_t matmul_job_func;
    if (is_repacked) {
        switch (src0->type) {
            case HTP_TYPE_Q4_0:
                if (octx->op == HTP_OP_MUL_MAT_NX_GEGLU) {
                    matmul_job_func = hvx_mm_nx_geglu_pk_q4_0;
                } else {
                    matmul_job_func = (act_nrows == 1 && (act->ne[0] / 32) % 2 == 0) ? hvx_mm_nx_2d_repacked_pk_q4_0 : hvx_mm_nx_2d_repacked_q4_0;
                }
                break;
            case HTP_TYPE_Q4_1:
            case HTP_TYPE_Q4_K:   matmul_job_func = hvx_mm_nx_2d_repacked_q4_1;   break;
            case HTP_TYPE_Q5_K:   matmul_job_func = hvx_mm_nx_2d_repacked_q5_k;   break;
            case HTP_TYPE_Q3_K:   matmul_job_func = hvx_mm_nx_2d_repacked_q3_k;   break;
            case HTP_TYPE_Q2_K:   matmul_job_func = hvx_mm_nx_2d_repacked_q2_k;   break;
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

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, n_matmul_jobs);

    return HTP_STATUS_OK;
}

int op_matmul_nx(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    if (kparams->collapse) {
        const uint32_t n_weights = kparams->n_weights;
        const struct htp_tensor * act = octx->src[n_weights];
        const uint32_t s1 = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        struct htp_tensor act_collapsed;
        struct htp_tensor dsts_collapsed[HTP_OP_MAX_OUTPUTS];
        const struct htp_tensor * orig_dsts[HTP_OP_MAX_OUTPUTS];

        htp_mm_tensor_collapse_rows(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        for (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            const struct htp_tensor * d = octx->dsts[p];
            const uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_rows(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        const int status = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        for (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        return status;
    }

    return op_matmul_nx_impl(octx);
}
