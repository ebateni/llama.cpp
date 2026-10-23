#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "hex-fastdiv.h"
#include "hvx-exp.h"
#include "hvx-sigmoid.h"
#include "hvx-utils.h"
#include "unary-ops.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hex-profile.h"
#include "hmx-mm-kernels-int8.h"
#define HMX_INT8_MAX_ROWS_Q8 1024   // must equal HMX_INT8_MAX_ROWS in matmul-ops.c (the plain int8 op refuses m > 1024)

struct htp_unary_context;

typedef void (*unary_compute_fn_t)(const void * restrict src,
                                   void * restrict dst,
                                   uint32_t num_rows,
                                   const struct htp_unary_context * uctx);

typedef void (*unary_rms_norm_mul_compute_fn_t)(const void * restrict src,
                                                const void * restrict weight,
                                                void * restrict dst,
                                                uint32_t num_rows,
                                                const struct htp_unary_context * uctx);

typedef void (*unary_add_rms_norm_mul_compute_fn_t)(const void * restrict src0,
                                                    const void * restrict src1,
                                                    const void * restrict weight,
                                                    void * restrict sum,
                                                    void * restrict dst,
                                                    uint32_t num_rows,
                                                    const struct htp_unary_context * uctx);

typedef void (*unary_tri_compute_fn_t)(const void * restrict src,
                                       void * restrict dst,
                                       uint32_t num_rows,
                                       uint32_t ir,
                                       const struct htp_unary_context * uctx);

typedef void (*unary_tile_compute_fn_t)(void * restrict dst,
                                        const void * restrict src,
                                        uint32_t tw,
                                        const struct htp_unary_context * uctx);

typedef void (*unary_tiled_tri_compute_fn_t)(const void * restrict src,
                                             void * restrict dst,
                                             uint32_t tile_elems,
                                             uint32_t col_start,
                                             uint32_t i01,
                                             uint32_t ne0,
                                             int32_t ttype);

struct htp_unary_context {
    struct htp_ops_context * octx;
    const struct htp_unary_kernel_params * kparams;

    void *                    compute;

    dma_addr_t                data_src0;
    dma_addr_t                data_src1;            // weight/scale tensor for RMS_NORM_MUL
    dma_addr_t                data_src2;            // second addend for ADD_RMS_NORM_MUL
    dma_addr_t                data_dst;

    size_t                    src0_data_row_size;   // actual data bytes per row
    size_t                    src1_data_row_size;
    size_t                    dst_data_row_size;    // actual data bytes per row

    size_t                    src0_row_size_aligned;
    size_t                    src1_row_size_aligned;
    size_t                    dst_row_size_aligned;

    size_t                    src0_vtcm_half_size;
    size_t                    src1_vtcm_half_size;
    size_t                    dst_vtcm_half_size;

    uint32_t                  block;
    uint32_t                  src0_nrows;
    uint32_t                  src0_nrows_per_thread;
    uint32_t                  row_start;
    uint32_t                  nc;
    uint32_t                  col_tile;             // tiled mode
    bool                      broadcast_weight;

    uint8_t *                 vtcm_src0;
    uint8_t *                 vtcm_src1;
    uint8_t *                 vtcm_src2;
    uint8_t *                 vtcm_sum;
    uint8_t *                 vtcm_dst;

    size_t                    vtcm_src0_size_per_thread;
    size_t                    vtcm_src1_size_per_thread;
    size_t                    vtcm_dst_size_per_thread;
    size_t                    vtcm_src2_size_per_thread;
    size_t                    vtcm_sum_size_per_thread;

    // RMS_NORM_MUL_ADD only: the scale sits after the weight rows in VTCM
    bool                      has_scale;

    // RMS_NORM_MUL_Q8 only: int8 activation payload (HMX crouton layout) + per-row scales written to dst
    uint8_t *                 q8_inv_s;        // VTCM copy of the 1/s_k smoothing vector (or NULL)
    size_t                    q8_stage_per_thread;   // stride of the per-thread VTCM blocks
    uint32_t                  q8_nrows;        // total rows (m) of the tensor
    uint32_t                  q8_scale_off;    // byte offset of the f32 row scales inside dst
};

// Convert flat row index to DDR byte offset using the tensor's actual strides.
// ir = i1 + ne1*(i2 + ne2*i3)  =>  offset = i1*nb1 + i2*nb2 + i3*nb3
static inline size_t unary_row_offset(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      const struct fastdiv_values * div_ne1,
                                      const struct fastdiv_values * div_ne2,
                                      const struct fastdiv_values * div_ne12,
                                      size_t nb1, size_t nb2, size_t nb3) {
    const uint32_t i1 = fastmodulo(ir, ne1, div_ne1);
    const uint32_t ir_div_ne1 = fastdiv(ir, div_ne1);
    const uint32_t i2 = fastmodulo(ir_div_ne1, ne2, div_ne2);
    const uint32_t i3 = fastdiv(ir, div_ne12);
    return i1 * nb1 + i2 * nb2 + i3 * nb3;
}

// Safe DMA block size from row `ir`: clamp to the tighter dim-1 slice
// boundary of src and dst so the nb1 stride stays valid for all rows.
static inline uint32_t unary_block_size(uint32_t ir,
                                        uint32_t end_row,
                                        uint32_t block,
                                        bool src_contig,
                                        bool dst_contig,
                                        uint32_t ne1,
                                        const struct fastdiv_values * div_ne1) {
    uint32_t limit = MIN(block, end_row - ir);

    if (!src_contig || !dst_contig) {
        const uint32_t slice_end = (fastdiv(ir, div_ne1) + 1) * ne1;
        limit = MIN(limit, slice_end - ir);
    }

    return limit;
}

#define htp_unary_preamble            \
    const uint32_t ne00 = src->ne[0]; \
    const uint32_t ne01 = src->ne[1]; \
    const uint32_t ne02 = src->ne[2]; \
    const uint32_t ne03 = src->ne[3]; \
                                      \
    const uint32_t ne0 = dst->ne[0];  \
    const uint32_t ne1 = dst->ne[1];  \
    const uint32_t ne2 = dst->ne[2];  \
    const uint32_t ne3 = dst->ne[3];  \
                                      \
    const uint32_t nb00 = src->nb[0]; \
    const uint32_t nb01 = src->nb[1]; \
    const uint32_t nb02 = src->nb[2]; \
    const uint32_t nb03 = src->nb[3]; \
                                      \
    const uint32_t nb0 = dst->nb[0];  \
    const uint32_t nb1 = dst->nb[1];  \
    const uint32_t nb2 = dst->nb[2];  \
    const uint32_t nb3 = dst->nb[3];

#define htp_unary_op_preamble                                         \
    int32_t * op_params = uctx->octx->op_params;                      \
    const uint32_t ne0 = uctx->nc;                                    \
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned; \
    const size_t dst_row_size_aligned = uctx->dst_row_size_aligned;

static void scale_f32(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float scale = 0.f;
    float bias  = 0.f;
    memcpy(&scale, &op_params[0], sizeof(float));
    memcpy(&bias,  &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0, scale, bias);
    }
}

static void clamp_f32(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float min = 0.f;
    float max = 0.f;
    memcpy(&min, &op_params[0], sizeof(float));
    memcpy(&max, &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_clamp_scalar_f32(dst_local, src_local, min, max, ne0);
    }
}

static void leaky_relu_f32(const void * restrict src,
                           void * restrict dst,
                           const uint32_t num_rows,
                           const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float negative_slope = 0.f;
    memcpy(&negative_slope, &op_params[0], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_leaky_relu_scalar_f32(dst_local, src_local, negative_slope, ne0);
    }
}

static void rms_norm_f32(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_f32((const uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

static void rms_norm_mul_f32(const void * restrict src,
                             const void * restrict weight,
                             void * restrict dst,
                             const uint32_t num_rows,
                             const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        const uint8_t * restrict w_local   = (const uint8_t *)weight + (uctx->broadcast_weight ? 0 : ir * uctx->src1_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_mul_f32(src_local, w_local, dst_local, ne0, epsilon);
    }
}

static void add_rms_norm_mul_f32(const void * restrict src0,
                                 const void * restrict src1,
                                 const void * restrict weight,
                                 void * restrict sum,
                                 void * restrict dst,
                                 const uint32_t num_rows,
                                 const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict s0_local = (const uint8_t *) src0 + (ir * src0_row_size_aligned);
        const uint8_t * restrict s1_local = (const uint8_t *) src1 + (ir * src0_row_size_aligned);
        const uint8_t * restrict w_local  = (const uint8_t *) weight + (uctx->broadcast_weight ? 0 : ir * uctx->src1_row_size_aligned);
        uint8_t * restrict sum_local      = (uint8_t *) sum + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local      = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_fast_add_rms_norm_mul_f32(s0_local, s1_local, w_local, sum_local, dst_local, ne0, epsilon);
    }
}

// y = rms_norm(x) * w1 + r [* s]; out = rms_norm(y) * w2. w2 sits one row after w1 in VTCM.
// Without w2 the op only produces y, which the thread loop then stages in the out buffer.
static void rms_norm_mul_add_f32(const void * restrict src0,
                                 const void * restrict src1,
                                 const void * restrict weight,
                                 void * restrict sum,
                                 void * restrict dst,
                                 const uint32_t num_rows,
                                 const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float eps1 = 0.f, eps2 = 0.f;
    memcpy(&eps1, &op_params[0], sizeof(float));
    memcpy(&eps2, &op_params[1], sizeof(float));

    const bool has_w2 = (uctx->octx->dsts[1] != NULL);
    const uint8_t * w1 = (const uint8_t *) weight;
    const uint8_t * w2 = has_w2 ? w1 + uctx->src1_row_size_aligned : NULL;
    const float * scale = uctx->has_scale ? (const float *) (w1 + (has_w2 ? 2 : 1) * uctx->src1_row_size_aligned) : NULL;
    uint8_t * y_base = has_w2 ? (uint8_t *) sum : (uint8_t *) dst;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict x_local = (const uint8_t *) src0 + (ir * src0_row_size_aligned);
        const uint8_t * restrict r_local = (const uint8_t *) src1 + (ir * src0_row_size_aligned);
        uint8_t * restrict y_local       = y_base + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local     = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_mul_add_f32(x_local, r_local, w1, w2, y_local, dst_local, ne0, eps1, eps2, scale);
    }
}

static void norm_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f32((const uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

static void sqr_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void sqrt_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void scale_f16(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float scale = 0.f;
    float bias  = 0.f;
    memcpy(&scale, &op_params[0], sizeof(float));
    memcpy(&bias,  &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0, scale, bias);
    }
}

static void clamp_f16(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float min = 0.f;
    float max = 0.f;
    memcpy(&min, &op_params[0], sizeof(float));
    memcpy(&max, &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_clamp_scalar_f16(dst_local, src_local, (_Float16) min, (_Float16) max, ne0);
    }
}

static void rms_norm_f16(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_f16((const uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

static void norm_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f16((const uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

static void sqr_f16(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void sqrt_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void abs_f16(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_abs_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void log_f16(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_log_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void step_f16(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_step_f16_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void l2_norm_f16(const void * restrict src,
                        void * restrict dst,
                        const uint32_t num_rows,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_f = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_l2_norm_f16((const uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

static void neg_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

static void exp_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_exp_f32(dst_local, src_local, ne0, false);
    }
}

static void sigmoid_f32(const void * restrict src,
                        void * restrict dst,
                        const uint32_t num_rows,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

// silu(x) = x * sigmoid(x)
static void silu_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
        hvx_mul_f32_aaa(dst_local, src_local, dst_local, ne0);
    }
}

static inline void hvx_gelu_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    for (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_f32(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), hvx_vec_gelu_f32(vsrc[i]));
    }
}

// GELU_QUICK uses x * sigmoid(1.702 * x).
static __attribute__((noinline)) HVX_Vector hvx_vec_gelu_quick_f32(HVX_Vector x) {
    const HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    const HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);
    const HVX_Vector scaled  = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(1.702f));
    const HVX_Vector sigmoid = hvx_vec_fast_sigmoid_f32_guard(scaled, one, max_exp, min_exp);

    return hvx_vec_mul_f32_f32(x, sigmoid);
}

static inline void hvx_gelu_quick_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    for (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_quick_f32(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), hvx_vec_gelu_quick_f32(vsrc[i]));
    }
}

static void gelu_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_f32_aa(dst_local, src_local, ne0);
    }
}

static void gelu_quick_f32(const void * restrict src,
                           void * restrict dst,
                           const uint32_t num_rows,
                           const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_quick_f32_aa(dst_local, src_local, ne0);
    }
}

static void gelu_erf_f32(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_gelu_erf_f32_aa(dst_local, src_local, ne0);
    }
}

static void tri_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ir,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    const int32_t ttype = op_params[0];
    const HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    const uint32_t nvec  = ne0 / VLEN_FP32;
    const uint32_t nloe  = ne0 % VLEN_FP32;

    const uint32_t ne01 = uctx->octx->src[0]->ne[1];

    for (uint32_t b = 0; b < num_rows; b++) {
        const uint32_t abs_row = ir + b;
        const uint32_t i01     = abs_row % ne01;

        const HVX_Vector * restrict v_src = (const HVX_Vector *) ((const uint8_t *) src + b * src0_row_size_aligned);
        HVX_Vector * restrict v_dst       = (HVX_Vector *) ((uint8_t *) dst + b * dst_row_size_aligned);

        uint32_t boundary;
        int      keep_left;
        switch (ttype) {
            case 0: boundary = i01;     keep_left = 0; break;  // keep col >= row
            case 1: boundary = i01 + 1; keep_left = 0; break;  // keep col > row
            case 2: boundary = i01 + 1; keep_left = 1; break;  // keep col <= row
            case 3: boundary = i01;     keep_left = 1; break;  // keep col < row
            default: boundary = 0; keep_left = 0; break;
        }
        if (boundary > ne0) boundary = ne0;

        // Full HVX vectors - each starts at a 128-byte aligned offset
        for (uint32_t i = 0; i < nvec; i++) {
            const uint32_t vec_start = i * VLEN_FP32;
            const uint32_t vec_end   = vec_start + VLEN_FP32;
            if (keep_left) {
                if (vec_end <= boundary) {
                    v_dst[i] = v_src[i];
                } else if (vec_start >= boundary) {
                    v_dst[i] = zero;
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * sizeof(float));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
                }
            } else {
                if (vec_end <= boundary) {
                    v_dst[i] = zero;
                } else if (vec_start >= boundary) {
                    v_dst[i] = v_src[i];
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * sizeof(float));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
                }
            }
        }

        // Tail elements (row_elems not a multiple of VLEN_FP32)
        if (nloe > 0) {
            const uint32_t abs_start = nvec * VLEN_FP32;
            const uint32_t abs_end   = abs_start + nloe;
            HVX_Vector     tail_val;
            if (keep_left) {
                if (abs_end <= boundary) {
                    tail_val = v_src[nvec];
                } else if (abs_start >= boundary) {
                    tail_val = zero;
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                    tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
                }
            } else {
                if (abs_end <= boundary) {
                    tail_val = zero;
                } else if (abs_start >= boundary) {
                    tail_val = v_src[nvec];
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                    tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
                }
            }
            hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(float), tail_val);
        }
    }
}

static void softplus_f32(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    // softplus(x) = log(1 + exp(x))
    // Match CPU reference: ggml_compute_softplus_f32() in ggml-impl.h
    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const float * restrict src_f = (const float *)((const uint8_t *)src + (ir * src0_row_size_aligned));
        float * restrict dst_f       = (float *)((uint8_t *)dst + (ir * dst_row_size_aligned));

        for (uint32_t i = 0; i < ne0; i++) {
            float x = src_f[i];
            // For x > 20: softplus(x) ~ x (avoids exp overflow)
            dst_f[i] = (x > 20.0f) ? x : logf(1.0f + expf(x));
        }
    }
}

static void l2_norm_f32(const void * restrict src,
                        void * restrict dst,
                        const uint32_t num_rows,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_f = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_l2_norm_f32((const uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

static void tanh_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

static void abs_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_abs_f32_aa(dst_local, src_local, ne0);
    }
}

static void relu_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_max_scalar_f32(dst_local, src_local, 0.0f, ne0);
    }
}

static void step_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_step_f32_aa(dst_local, src_local, ne0);
    }
}

static void log_f32(const void * restrict src,
                    void * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_log_f32_aa(dst_local, src_local, ne0);
    }
}

#// Pointwise unary ops on one column tile in VTCM.
static void tile_scale_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    float scale = 0.f;
    float bias  = 0.f;
    memcpy(&scale, &uctx->octx->op_params[0], sizeof(float));
    memcpy(&bias,  &uctx->octx->op_params[1], sizeof(float));
    hvx_scale_offset_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw, scale, bias);
}

static void tile_clamp_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    float min = 0.f;
    float max = 0.f;
    memcpy(&min, &uctx->octx->op_params[0], sizeof(float));
    memcpy(&max, &uctx->octx->op_params[1], sizeof(float));
    hvx_clamp_scalar_f32((uint8_t *) dst, (const uint8_t *) src, min, max, tw);
}

static void tile_leaky_relu_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    float negative_slope = 0.f;
    memcpy(&negative_slope, &uctx->octx->op_params[0], sizeof(float));
    hvx_leaky_relu_scalar_f32((uint8_t *) dst, (const uint8_t *) src, negative_slope, tw);
}

static void tile_sqr_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_sqr_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_sqrt_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_sqrt_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_neg_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_scale_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw, -1.0f);
}

static void tile_exp_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_exp_f32((uint8_t *) dst, (const uint8_t *) src, tw, false);
}

static void tile_sigmoid_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_silu_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
    hvx_mul_f32_aaa((uint8_t *) dst, (const uint8_t *) src, (uint8_t *) dst, tw);
}

static void tile_gelu_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_gelu_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_gelu_quick_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_gelu_quick_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_gelu_erf_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_gelu_erf_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_softplus_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    const float * restrict sf = (const float *) src;
    float * restrict df       = (float *) dst;
    for (uint32_t i = 0; i < tw; i++) {
        float x = sf[i];
        df[i] = (x > 20.0f) ? x : logf(1.0f + expf(x));
    }
}

static void tile_tanh_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_tanh_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_abs_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_abs_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_log_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_log_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tile_relu_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_max_scalar_f32((uint8_t *) dst, (const uint8_t *) src, 0.0f, tw);
}

static void tile_step_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_step_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
}

static void tri_apply_tile_f32(const void * restrict src, void * restrict dst,
                               uint32_t tile_elems, uint32_t col_start, uint32_t i01,
                               uint32_t ne0, int32_t ttype) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    const HVX_Vector zero = hvx_vec_splat_f32(0.0f);

    uint32_t boundary;
    int      keep_left;
    switch (ttype) {
        case 0: boundary = i01;     keep_left = 0; break;
        case 1: boundary = i01 + 1; keep_left = 0; break;
        case 2: boundary = i01 + 1; keep_left = 1; break;
        case 3: boundary = i01;     keep_left = 1; break;
        default: boundary = 0; keep_left = 0; break;
    }
    if (boundary > ne0) boundary = ne0;

    const uint32_t nvec = tile_elems / VLEN_FP32;
    const uint32_t nloe = tile_elems % VLEN_FP32;

    for (uint32_t i = 0; i < nvec; i++) {
        const uint32_t abs_start = col_start + i * VLEN_FP32;
        const uint32_t abs_end   = abs_start + VLEN_FP32;
        if (keep_left) {
            if (abs_end <= boundary) {
                v_dst[i] = v_src[i];
            } else if (abs_start >= boundary) {
                v_dst[i] = zero;
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
            }
        } else {
            if (abs_end <= boundary) {
                v_dst[i] = zero;
            } else if (abs_start >= boundary) {
                v_dst[i] = v_src[i];
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
            }
        }
    }

    if (nloe > 0) {
        const uint32_t abs_start = col_start + nvec * VLEN_FP32;
        const uint32_t abs_end   = abs_start + nloe;
        HVX_Vector     tail_val;
        if (keep_left) {
            if (abs_end <= boundary) {
                tail_val = v_src[nvec];
            } else if (abs_start >= boundary) {
                tail_val = zero;
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
            }
        } else {
            if (abs_end <= boundary) {
                tail_val = zero;
            } else if (abs_start >= boundary) {
                tail_val = v_src[nvec];
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
            }
        }
        hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(float), tail_val);
    }
}

// 1. Standard row-block unary task (F32 and F16).
static void unary_thread_row_block(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    const uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    const size_t src0_data_row_size = uctx->src0_data_row_size;
    const size_t dst_data_row_size  = uctx->dst_data_row_size;
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    const uint32_t src0_nrows = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src = uctx->data_src0;
    const dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    const size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    const size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    const bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    const bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;

    const uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    const uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    const uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    if (BLOCK == 0) {
        FARF(ERROR, "unary-row-block : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);
        return;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_queue_push(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        const size_t src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_push(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    unary_compute_fn_t compute = (unary_compute_fn_t) uctx->compute;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        void * dst_vtcm  = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src0_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_size, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        const size_t dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_push(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        const uint32_t next_ir = ir + block_size;
        if (next_ir < src0_end_row) {
            const uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            const uint32_t pref_ir = next_ir + next_block_size;
            if (pref_ir < src0_end_row) {
                const uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                const size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_push(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 2. RMS_NORM_MUL row-block task with weight buffer.
static void unary_thread_rms_norm_mul_f32(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    const uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    const size_t src0_data_row_size = uctx->src0_data_row_size;
    const size_t dst_data_row_size  = uctx->dst_data_row_size;
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    const uint32_t src0_nrows = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src  = uctx->data_src0;
    const dma_addr_t data_src1 = uctx->data_src1;
    const dma_addr_t data_dst  = uctx->data_dst;

    const struct htp_tensor * src1 = octx->src[1];
    const uint32_t nb11 = src1->nb[1];
    const uint32_t nb12 = src1->nb[2];
    const uint32_t nb13 = src1->nb[3];
    const uint32_t nb11_bc = (src1->ne[1] > 1) ? nb11 : 0;
    const uint32_t nb12_bc = (src1->ne[2] > 1) ? nb12 : 0;
    const uint32_t nb13_bc = (src1->ne[3] > 1) ? nb13 : 0;
    const bool src1_contig = ((nb12 == (size_t)ne01 * nb11) && (nb13 == (size_t)ne02 * nb12));

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * src1_vtcm_data = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_size_per_thread)) : NULL;
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    const size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    const size_t src1_vtcm_half_size = uctx->src1_vtcm_half_size;
    const size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    const bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    const bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;

    const bool src1_needs_row_clip = !uctx->broadcast_weight && !src1_contig;
    const bool block_src0_contig = src0_contig && !src1_needs_row_clip;
    const bool block_dst_contig  = dst_contig  && !src1_needs_row_clip;

    const uint32_t src0_max_block = block_src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    const uint32_t dst_max_block  = block_dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    const uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    if (BLOCK == 0) {
        FARF(ERROR, "unary-rms-norm-mul : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);
        return;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    if (uctx->broadcast_weight) {
        dma_queue_push(dma_q, dma_make_data(src1_vtcm_data, data_src1),
                       uctx->src1_row_size_aligned, 0, uctx->src1_data_row_size, 1);
        dma_queue_flush(dma_q);
    }

    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_src0_contig, block_dst_contig,
                                                     ne01, div_ne01);

        dma_queue_push(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        const size_t src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_push(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        if (!uctx->broadcast_weight) {
            const size_t src1_off = src1_contig ? (ir * nb11) :
                unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc, nb13_bc);
            dma_queue_push(dma_q,
                dma_make_data(src1_vtcm_data + (vtcm_idx * src1_vtcm_half_size), data_src1 + src1_off),
                uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, block_size);
        }

        ir += block_size;
    }

    unary_rms_norm_mul_compute_fn_t compute = (unary_rms_norm_mul_compute_fn_t) uctx->compute;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_src0_contig, block_dst_contig,
                                                     ne01, div_ne01);

        void * dst_vtcm  = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src0_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;
        void * src1_vtcm = NULL;
        if (!uctx->broadcast_weight) {
            src1_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        const void * w = uctx->broadcast_weight ? (const void *) src1_vtcm_data : src1_vtcm;
        compute(src0_vtcm, w, dst_vtcm, block_size, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        const size_t dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_push(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        const uint32_t next_ir = ir + block_size;
        if (next_ir < src0_end_row) {
            const uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, block_src0_contig,
                                                              block_dst_contig, ne01, div_ne01);
            const uint32_t pref_ir = next_ir + next_block_size;
            if (pref_ir < src0_end_row) {
                const uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, block_src0_contig,
                                                                  block_dst_contig, ne01, div_ne01);
                const size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_push(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);

                if (!uctx->broadcast_weight) {
                    const size_t src1_pref_off = src1_contig ? (pref_ir * nb11) :
                        unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc,
                                          nb13_bc);
                    dma_queue_push(dma_q,
                        dma_make_data(src1_vtcm, data_src1 + src1_pref_off),
                        uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, pref_block_size);
                }
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 2a. RMS_NORM_MUL_Q8: rms_norm(x) * w per ROW PAIR, then the int8 quantizer of the plain int8 matmul (hmx_int8_quant_pair_fast), writing crouton payload and f32 row scales to dst.
// Each thread owns whole pairs, so nothing is shared. dst: payload n_rt * (k/32) * 2048 B, then f32 scales at q8_scale_off.
static void unary_thread_rms_norm_mul_q8(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const uint32_t k = uctx->nc;
    const uint32_t m = uctx->q8_nrows;
    const uint32_t nkc = k / 32;
    const uint32_t n_pairs = (m + 1) / 2;
    const uint32_t n_thr = octx->n_threads;
    const uint32_t p0 = (uint32_t) (((uint64_t) n_pairs * ith) / n_thr);
    const uint32_t p1 = (uint32_t) (((uint64_t) n_pairs * (ith + 1)) / n_thr);
    if (p0 >= p1) { return; }

    const size_t src_stride = octx->src[0]->nb[1];
    const size_t row_al     = uctx->src0_row_size_aligned;
    dma_queue * dma_q = octx->ctx->dma[ith];

    float epsilon = 0.f;
    memcpy(&epsilon, octx->op_params, sizeof(float));

    // one block per thread: [weight row | 2 x (2 input rows) | x0 | x1 | 2 x crouton words (nkc * 128) | row scales], stride q8_stage_per_thread
    uint8_t * blk    = uctx->vtcm_src1 + (size_t) ith * uctx->q8_stage_per_thread;
    uint8_t * w_vtcm = blk;
    uint8_t * pair[2] = { blk + row_al, blk + 3 * row_al };
    uint8_t * stg    = blk + 5 * row_al;
    float *   x0     = (float *) stg;
    float *   x1     = (float *) (stg + row_al);
    uint8_t * words[2] = { stg + 2 * row_al, stg + 2 * row_al + (size_t) nkc * 128 };
    float *   scl    = (float *) (stg + 2 * row_al + (size_t) nkc * 256);

    dma_queue_push(dma_q, dma_make_data(w_vtcm, uctx->data_src1), uctx->src1_row_size_aligned, 0, uctx->src1_data_row_size, 1);
    dma_queue_pop(dma_q);

    // Software pipeline on the thread's DMA queue (strict FIFO): inputs run two pairs ahead, the payload scatter of pair g completes
    // while pair g+1 computes. Queue order per iteration g: [out(g-2)] in(g) are popped in push order, so pops never mismatch.
    #define Q8_IN(g_) do {                                                                                                         \
        const uint32_t _r = 2 * (g_);                                                                                              \
        dma_queue_push(dma_q, dma_make_data(pair[((g_) - p0) & 1], (uint8_t *) (uintptr_t) uctx->data_src0 + (size_t) _r * src_stride), \
                       row_al, src_stride, uctx->src0_data_row_size, (_r + 1 < m) ? 2 : 1);                                         \
    } while (0)

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, p0);
    const uint32_t r_first = 2 * p0;
    Q8_IN(p0);
    if (p0 + 1 < p1) { Q8_IN(p0 + 1); }
    for (uint32_t gp = p0; gp < p1; ++gp) {
        const uint32_t li  = gp - p0;
        const uint32_t row = 2 * gp;
        const bool     two = (row + 1 < m);
        if (li >= 2) { dma_queue_pop(dma_q); }   // out(gp - 2): frees words[li & 1]
        dma_queue_pop(dma_q);                    // in(gp)
        uint8_t * in = pair[li & 1];
        hvx_fast_rms_norm_mul_f32(in, w_vtcm, (uint8_t *) x0, (int) k, epsilon);
        if (two) { hvx_fast_rms_norm_mul_f32(in + row_al, w_vtcm, (uint8_t *) x1, (int) k, epsilon); }

        float sc[2] = { 0.0f, 0.0f };
        hmx_int8_quant_pair_fast_ex(words[li & 1], sc, x0, two ? x1 : NULL, k, (const float *) uctx->q8_inv_s, 0, 128);
        scl[row - r_first] = sc[0];
        if (two) { scl[row + 1 - r_first] = sc[1]; }

        // the nkc words of this pair go to (row_pair>>4)*nkc*2048 + (row_pair&15)*128 + kc*2048 in DDR
        const size_t doff = (size_t) (gp >> 4) * nkc * HMX_INT8_CROUTON_A + (size_t) (gp & 15) * 128;
        dma_queue_push(dma_q, dma_make_data((uint8_t *) (uintptr_t) uctx->data_dst + doff, words[li & 1]), HMX_INT8_CROUTON_A, 128, 128, nkc);
        if (gp + 2 < p1) { Q8_IN(gp + 2); }
    }
    #undef Q8_IN
    // drain the outstanding scatters (the last two pairs; with a single pair only one)
    const uint32_t n_p = p1 - p0;
    dma_queue_pop(dma_q);
    if (n_p >= 2) { dma_queue_pop(dma_q); }
    // this thread's row scales (one contiguous run)
    const uint32_t r_last = (2 * p1 < m) ? 2 * p1 : m;
    const size_t   sbytes = (size_t) (r_last - r_first) * sizeof(float);
    dma_queue_push(dma_q, dma_make_data((uint8_t *) (uintptr_t) uctx->data_dst + uctx->q8_scale_off + (size_t) r_first * sizeof(float), scl),
                   sbytes, sbytes, sbytes, 1);
    dma_queue_pop(dma_q);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, p0);
}

// 2b. ADD_RMS_NORM_MUL: add the two residual inputs, keep the sum in VTCM for the norm,
// then write both the sum (next layer's residual) and the normalized result.
static void unary_thread_add_rms_norm_mul_f32(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dsts[0];

    htp_unary_preamble;

    const uint32_t nrows_per_thread = uctx->src0_nrows_per_thread;
    const uint32_t start_row = uctx->row_start + nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + nrows_per_thread, uctx->row_start + uctx->src0_nrows);

    if (start_row >= end_row) {
        return;
    }

    const size_t row_aligned = uctx->src0_row_size_aligned;
    const size_t row_bytes   = uctx->src0_data_row_size;
    const uint32_t nb21      = octx->src[2]->nb[1];

    // dsts[0] is the normalized result, dsts[1] is the residual sum.
    // RMS_NORM_MUL_ADD without a second norm has only dsts[0], which then holds the sum.
    const bool       has_sum  = (octx->dsts[1] != NULL);
    const dma_addr_t data_sum = has_sum ? octx->dsts[1]->data : 0;
    const uint32_t   nb_sum   = has_sum ? octx->dsts[1]->nb[1] : 0;

    uint8_t * s0_vtcm  = uctx->vtcm_src0 + ith * uctx->vtcm_src0_size_per_thread;
    uint8_t * s2_vtcm  = uctx->vtcm_src2 + ith * uctx->vtcm_src2_size_per_thread;
    uint8_t * w_vtcm   = uctx->vtcm_src1 + ith * uctx->vtcm_src1_size_per_thread;
    uint8_t * sum_vtcm = uctx->vtcm_sum  + ith * uctx->vtcm_sum_size_per_thread;
    uint8_t * dst_vtcm = uctx->vtcm_dst  + ith * uctx->vtcm_dst_size_per_thread;

    const size_t s0_half  = uctx->vtcm_src0_size_per_thread / 2;
    const size_t s2_half  = uctx->vtcm_src2_size_per_thread / 2;
    const size_t sum_half = uctx->vtcm_sum_size_per_thread / 2;
    const size_t dst_half = uctx->vtcm_dst_size_per_thread / 2;

    const uint32_t BLOCK = uctx->block;
    dma_queue * dma_q = octx->ctx->dma[ith];

    dma_queue_push(dma_q, dma_make_data(w_vtcm, uctx->data_src1),
                   uctx->src1_row_size_aligned, 0, uctx->src1_data_row_size, 1);
    if (octx->op == HTP_OP_RMS_NORM_MUL_ADD) {
        uint32_t w_rows = 1;
        if (has_sum) {
            dma_queue_push(dma_q, dma_make_data(w_vtcm + uctx->src1_row_size_aligned, octx->src[3]->data),
                           uctx->src1_row_size_aligned, 0, uctx->src1_data_row_size, 1);
            w_rows = 2;
        }
        if (uctx->has_scale) {
            dma_queue_push(dma_q, dma_make_data(w_vtcm + w_rows * uctx->src1_row_size_aligned, octx->src[4]->data),
                           128, 0, sizeof(float), 1);
        }
    }
    dma_queue_flush(dma_q);

    for (uint32_t ir = start_row, slot = 0; ir < end_row && slot < 2; slot++) {
        const uint32_t bs = MIN(BLOCK, end_row - ir);
        // zero-height descriptors only hand the VTCM slot pointers back to the loop below
        dma_queue_push(dma_q, dma_make_data(uctx->data_dst, dst_vtcm + slot * dst_half),
                       nb1, row_aligned, row_bytes, 0);
        if (has_sum) {
            dma_queue_push(dma_q, dma_make_data(data_sum, sum_vtcm + slot * sum_half),
                           nb_sum, row_aligned, row_bytes, 0);
        }
        dma_queue_push(dma_q, dma_make_data(s0_vtcm + slot * s0_half, uctx->data_src0 + ir * nb01),
                       row_aligned, nb01, row_bytes, bs);
        dma_queue_push(dma_q, dma_make_data(s2_vtcm + slot * s2_half, uctx->data_src2 + ir * nb21),
                       row_aligned, nb21, row_bytes, bs);
        ir += bs;
    }

    unary_add_rms_norm_mul_compute_fn_t compute = (unary_add_rms_norm_mul_compute_fn_t) uctx->compute;

    for (uint32_t ir = start_row; ir < end_row; ) {
        const uint32_t bs = MIN(BLOCK, end_row - ir);

        void * out = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * sum = has_sum ? (void *) (uintptr_t) dma_queue_pop(dma_q).src : NULL;
        void * s0  = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;
        void * s2  = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(&octx->ctx->trace[ith], HTP_TRACE_EVT_HVX_COMP, ir);
        compute(s0, s2, w_vtcm, sum, out, bs, uctx);
        htp_trace_event_stop(&octx->ctx->trace[ith], HTP_TRACE_EVT_HVX_COMP, ir);

        // The output-write descriptors double as the next iteration's buffer handles, so push order must match pop order (dst, then sum).
        dma_queue_push(dma_q, dma_make_data(uctx->data_dst + ir * nb1, out), nb1, row_aligned, row_bytes, bs);
        if (has_sum) {
            dma_queue_push(dma_q, dma_make_data(data_sum + ir * nb_sum, sum), nb_sum, row_aligned, row_bytes, bs);
        }

        const uint32_t next_ir = ir + bs;
        if (next_ir < end_row) {
            const uint32_t pref_ir = next_ir + MIN(BLOCK, end_row - next_ir);
            if (pref_ir < end_row) {
                const uint32_t pbs = MIN(BLOCK, end_row - pref_ir);
                dma_queue_push(dma_q, dma_make_data(s0, uctx->data_src0 + pref_ir * nb01), row_aligned, nb01, row_bytes, pbs);
                dma_queue_push(dma_q, dma_make_data(s2, uctx->data_src2 + pref_ir * nb21), row_aligned, nb21, row_bytes, pbs);
            }
        }
        ir = next_ir;
    }

    dma_queue_flush(dma_q);
}

// 3. TRI row-block task with row index ir.
static void unary_thread_tri_f32(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    const uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    const size_t src0_data_row_size = uctx->src0_data_row_size;
    const size_t dst_data_row_size  = uctx->dst_data_row_size;
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    const uint32_t src0_nrows = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src = uctx->data_src0;
    const dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    const size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    const size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    const bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    const bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;

    const uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    const uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    const uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    if (BLOCK == 0) {
        FARF(ERROR, "unary-tri : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);
        return;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_queue_push(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        const size_t src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_push(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    unary_tri_compute_fn_t compute = (unary_tri_compute_fn_t) uctx->compute;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        void * dst_vtcm  = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src0_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_size, ir, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        const size_t dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_push(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        const uint32_t next_ir = ir + block_size;
        if (next_ir < src0_end_row) {
            const uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            const uint32_t pref_ir = next_ir + next_block_size;
            if (pref_ir < src0_end_row) {
                const uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                const size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_push(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 4. Pointwise tiled unary task.
static void unary_thread_tiled(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    const uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    const uint32_t col_tile              = uctx->col_tile;

    const uint32_t src0_nrows     = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src = uctx->data_src0;
    const dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    const size_t src0_half = uctx->src0_vtcm_half_size;
    const size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;
    const struct fastdiv_values * div_tpr   = &uctx->kparams->div_tpr;

    const uint32_t tiles_per_row = (ne0 + col_tile - 1) / col_tile;

    const bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    const bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    const uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    for (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        const uint32_t row  = src0_start_row + t / tiles_per_row;
        const uint32_t col  = (t % tiles_per_row) * col_tile;
        const uint32_t tw   = MIN(col_tile, ne0 - col);
        const size_t   tb   = (size_t) tw * sizeof(float);
        const size_t   soff = (src0_contig ? (row * nb01) :
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) col * sizeof(float);

        dma_queue_push(dma_q, dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_push(dma_q, dma_make_data(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    unary_tile_compute_fn_t compute = (unary_tile_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    for (uint32_t t = 0; t < total_tiles; t++) {
        void * dst_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        const uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(dst_vtcm, src_vtcm, tw, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        const size_t doff = (dst_contig ? (row * nb1) :
                             unary_row_offset(row, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) col * sizeof(float);
        const size_t tb   = (size_t) tw * sizeof(float);
        dma_queue_push(dma_q, dma_make_data(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        const uint32_t pt = t + 2;
        if (pt < total_tiles) {
            const uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            const size_t   ptb  = (size_t) ptw * sizeof(float);
            const size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * sizeof(float);
            dma_queue_push(dma_q, dma_make_data(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        if (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
        }

        ptile_in_row++;
        pcol += col_tile;
        if (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

// 5. TRI tiled task.
static void unary_thread_tiled_tri_f32(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    const uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    const int32_t * op_params            = octx->op_params;
    const uint32_t col_tile              = uctx->col_tile;

    const uint32_t src0_nrows     = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src = uctx->data_src0;
    const dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    const size_t src0_half = uctx->src0_vtcm_half_size;
    const size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    const struct fastdiv_values * div_ne01  = &uctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &uctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &uctx->kparams->div_ne012;
    const struct fastdiv_values * div_tpr   = &uctx->kparams->div_tpr;

    const uint32_t tiles_per_row = (ne0 + col_tile - 1) / col_tile;
    const int32_t  tri_ttype     = op_params[0];

    const bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    const bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    const uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    for (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        const uint32_t row  = src0_start_row + t / tiles_per_row;
        const uint32_t col  = (t % tiles_per_row) * col_tile;
        const uint32_t tw   = MIN(col_tile, ne0 - col);
        const size_t   tb   = (size_t) tw * sizeof(float);
        const size_t   soff = (src0_contig ? (row * nb01) :
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) col * sizeof(float);

        dma_queue_push(dma_q, dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_push(dma_q, dma_make_data(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    unary_tiled_tri_compute_fn_t compute = (unary_tiled_tri_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;
    uint32_t i01 = fastmodulo(row, ne01, div_ne01);

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    for (uint32_t t = 0; t < total_tiles; t++) {
        void * dst_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * src_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        const uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(src_vtcm, dst_vtcm, tw, col, i01, ne0, tri_ttype);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        const size_t doff = (dst_contig ? (row * nb1) :
                             unary_row_offset(row, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) col * sizeof(float);
        const size_t tb   = (size_t) tw * sizeof(float);
        dma_queue_push(dma_q, dma_make_data(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        const uint32_t pt = t + 2;
        if (pt < total_tiles) {
            const uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            const size_t   ptb  = (size_t) ptw * sizeof(float);
            const size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * sizeof(float);
            dma_queue_push(dma_q, dma_make_data(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        if (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
            i01++;
            if (i01 == ne01) {
                i01 = 0;
            }
        }

        ptile_in_row++;
        pcol += col_tile;
        if (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

static int execute_op_unary(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dsts[0];

    const bool is_f16 = (src0->type == HTP_TYPE_F16);

    const char * op_type = NULL;

    switch (octx->op) {
        case HTP_OP_NORM:            op_type = is_f16 ? "norm-f16"     : "norm-f32";         break;
        case HTP_OP_RMS_NORM:        op_type = is_f16 ? "rmsnorm-f16"  : "rmsnorm-f32";      break;
        case HTP_OP_RMS_NORM_MUL:    op_type = "rmsnorm-mul-f32";                            break;
        case HTP_OP_ADD_RMS_NORM_MUL: op_type = "add-rmsnorm-mul-f32";                        break;
        case HTP_OP_RMS_NORM_MUL_ADD: op_type = "rmsnorm-mul-add-f32";                        break;
        case HTP_OP_SCALE:           op_type = is_f16 ? "scale-f16"    : "scale-f32";        break;
        case HTP_OP_CLAMP:           op_type = is_f16 ? "clamp-f16"    : "clamp-f32";        break;
        case HTP_OP_LEAKY_RELU:      op_type = "leaky-relu-f32";                             break;
        case HTP_OP_SQR:             op_type = is_f16 ? "sqr-f16"      : "sqr-f32";          break;
        case HTP_OP_SQRT:            op_type = is_f16 ? "sqrt-f16"     : "sqrt-f32";         break;
        case HTP_OP_UNARY_NEG:       op_type = "neg-f32";                                    break;
        case HTP_OP_UNARY_EXP:       op_type = "exp-f32";                                    break;
        case HTP_OP_UNARY_SIGMOID:   op_type = "sigmoid-f32";                                break;
        case HTP_OP_UNARY_SILU:      op_type = "silu-f32";                                   break;
        case HTP_OP_UNARY_GELU:      op_type = "gelu-f32";                                   break;
        case HTP_OP_UNARY_GELU_QUICK: op_type = "gelu-quick-f32";                             break;
        case HTP_OP_UNARY_GELU_ERF:  op_type = "gelu-erf-f32";                               break;
        case HTP_OP_UNARY_SOFTPLUS:  op_type = "softplus-f32";                               break;
        case HTP_OP_UNARY_TANH:      op_type = "tanh-f32";                                   break;
        case HTP_OP_UNARY_ABS:       op_type = is_f16 ? "abs-f16"      : "abs-f32";          break;
        case HTP_OP_UNARY_LOG:       op_type = is_f16 ? "log-f16"      : "log-f32";          break;
        case HTP_OP_UNARY_RELU:      op_type = "relu-f32";                                   break;
        case HTP_OP_UNARY_STEP:      op_type = is_f16 ? "step-f16"     : "step-f32";         break;
        case HTP_OP_L2_NORM:         op_type = is_f16 ? "l2norm-f16"   : "l2norm-f32";       break;
        case HTP_OP_TRI:             op_type = "tri-f32";                                    break;
        default:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    // F16 only has row-block kernels for this subset of ops (see the dispatch switch
    // below) - reject everything else up front, before touching kparams/VTCM.
    if (is_f16) {
        switch (octx->op) {
            case HTP_OP_NORM:
            case HTP_OP_RMS_NORM:
            case HTP_OP_SCALE:
            case HTP_OP_CLAMP:
            case HTP_OP_SQR:
            case HTP_OP_SQRT:
            case HTP_OP_L2_NORM:
            case HTP_OP_UNARY_ABS:
            case HTP_OP_UNARY_LOG:
            case HTP_OP_UNARY_STEP:
                break;
            default:
                FARF(ERROR, "unary-%s: not supported for F16\n", op_type);
                return HTP_STATUS_NO_SUPPORT;
        }
    }

    const struct htp_unary_kernel_params * kparams = (const struct htp_unary_kernel_params *) octx->kernel_params;

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const size_t elem_size = is_f16 ? sizeof(_Float16) : sizeof(float);
    const size_t src0_data_row_size = src0->ne[0] * elem_size;
    const size_t dst_data_row_size  = dst->ne[0]  * elem_size;

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_size, (uint32_t) dst_data_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = octx->n_threads;

    const size_t src0_row_size_aligned = kparams->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = kparams->dst_row_size_aligned;

    // Always 0 for F16 - htp_unary_vtcm_layout_build() keeps F16 on the row-block path,
    // since only F32 has unary_task_f32_tiled_* kernels.
    const uint32_t col_tile = kparams->col_tile;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = kparams->src1_row_size_aligned;
    bool broadcast_weight = kparams->broadcast_weight;
    const struct htp_tensor * src1 = NULL;

    // RMS_NORM_MUL fusion is F32-only (its weight tensor is always F32; see
    // try_fuse_node()'s type guard), so this never triggers when is_f16 is true.
    if (octx->op == HTP_OP_RMS_NORM_MUL || octx->op == HTP_OP_ADD_RMS_NORM_MUL || octx->op == HTP_OP_RMS_NORM_MUL_ADD) {
        src1 = octx->src[1];
        src1_data_row_size = src1->ne[0] * sizeof(float);
    }

    if (octx->ctx->vtcm_size < (size_t)kparams->vtcm_size) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size, (size_t)kparams->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = NULL;
    octx->src1_spad.src = NULL;
    octx->dst_spad.src  = NULL;

    FARF(HIGH, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %u src1-vtcm-size %u dst-vtcm-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->vtcm_src0_size, kparams->vtcm_src1_size, kparams->vtcm_dst_size);

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    struct htp_unary_context uctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .src0_nrows            = nrows,
        .row_start             = row_start,

        .data_src0             = src0->data,
        .data_src1             = src1 ? src1->data : 0,
        .data_src2             = (octx->op == HTP_OP_ADD_RMS_NORM_MUL || octx->op == HTP_OP_RMS_NORM_MUL_ADD) ? octx->src[2]->data : 0,
        .data_dst              = dst->data,

        .src0_data_row_size    = src0_data_row_size,
        .src1_data_row_size    = src1_data_row_size,
        .dst_data_row_size     = dst_data_row_size,

        .src0_row_size_aligned = src0_row_size_aligned,
        .src1_row_size_aligned = src1_row_size_aligned,
        .dst_row_size_aligned  = dst_row_size_aligned,

        .src0_vtcm_half_size   = kparams->vtcm_src0_size_per_thread / 2,
        .src1_vtcm_half_size   = (octx->op == HTP_OP_RMS_NORM_MUL) ? (kparams->vtcm_src1_size_per_thread / (broadcast_weight ? 1 : 2)) : 0,
        .dst_vtcm_half_size    = kparams->vtcm_dst_size_per_thread / 2,

        .block                 = kparams->block,
        .nc                    = src0->ne[0],
        .col_tile              = col_tile,
        .broadcast_weight      = broadcast_weight,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, base, 0),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, kparams->vtcm_src0_size, kparams->vtcm_src1_size > 0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, base, kparams->vtcm_src0_size + kparams->vtcm_src1_size),
        .vtcm_src2             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base,
                                     kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size,
                                     kparams->vtcm_src2_size > 0),
        .vtcm_sum              = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base,
                                     kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size +
                                     kparams->vtcm_src2_size,
                                     kparams->vtcm_sum_size > 0),

        .vtcm_src0_size_per_thread = kparams->vtcm_src0_size_per_thread,
        .vtcm_src1_size_per_thread = kparams->vtcm_src1_size_per_thread,
        .vtcm_dst_size_per_thread  = kparams->vtcm_dst_size_per_thread,
        .vtcm_src2_size_per_thread = kparams->vtcm_src2_size_per_thread,
        .vtcm_sum_size_per_thread  = kparams->vtcm_sum_size_per_thread,

        .has_scale             = (octx->op == HTP_OP_RMS_NORM_MUL_ADD && octx->src[4] != NULL),
    };

    FARF(HIGH, "%s: %s mode (col_tile %u)\n", op_type, col_tile ? "tiled" : "row-block", col_tile);

    worker_callback_t task_func = NULL;
    void * compute_func = NULL;

    if (col_tile) {
        task_func = unary_thread_tiled;
        switch (octx->op) {
            case HTP_OP_SCALE:           compute_func = (void *) tile_scale_f32;          break;
            case HTP_OP_CLAMP:           compute_func = (void *) tile_clamp_f32;          break;
            case HTP_OP_LEAKY_RELU:      compute_func = (void *) tile_leaky_relu_f32;     break;
            case HTP_OP_SQR:             compute_func = (void *) tile_sqr_f32;            break;
            case HTP_OP_SQRT:            compute_func = (void *) tile_sqrt_f32;           break;
            case HTP_OP_UNARY_NEG:       compute_func = (void *) tile_neg_f32;            break;
            case HTP_OP_UNARY_EXP:       compute_func = (void *) tile_exp_f32;            break;
            case HTP_OP_UNARY_SIGMOID:   compute_func = (void *) tile_sigmoid_f32;        break;
            case HTP_OP_UNARY_SILU:      compute_func = (void *) tile_silu_f32;           break;
            case HTP_OP_UNARY_GELU:      compute_func = (void *) tile_gelu_f32;           break;
            case HTP_OP_UNARY_GELU_QUICK: compute_func = (void *) tile_gelu_quick_f32;     break;
            case HTP_OP_UNARY_GELU_ERF:  compute_func = (void *) tile_gelu_erf_f32;       break;
            case HTP_OP_UNARY_SOFTPLUS:  compute_func = (void *) tile_softplus_f32;       break;
            case HTP_OP_UNARY_TANH:      compute_func = (void *) tile_tanh_f32;           break;
            case HTP_OP_UNARY_ABS:       compute_func = (void *) tile_abs_f32;            break;
            case HTP_OP_UNARY_LOG:       compute_func = (void *) tile_log_f32;            break;
            case HTP_OP_UNARY_RELU:      compute_func = (void *) tile_relu_f32;           break;
            case HTP_OP_UNARY_STEP:      compute_func = (void *) tile_step_f32;           break;
            case HTP_OP_TRI:
                task_func    = unary_thread_tiled_tri_f32;
                compute_func = (void *) tri_apply_tile_f32;
                break;
            default:                     break;
        }
    } else if (is_f16) {
        task_func = unary_thread_row_block;
        switch (octx->op) {
            case HTP_OP_NORM:            compute_func = (void *) norm_f16;                break;
            case HTP_OP_RMS_NORM:        compute_func = (void *) rms_norm_f16;            break;
            case HTP_OP_SCALE:           compute_func = (void *) scale_f16;               break;
            case HTP_OP_CLAMP:           compute_func = (void *) clamp_f16;               break;
            case HTP_OP_SQR:             compute_func = (void *) sqr_f16;                 break;
            case HTP_OP_SQRT:            compute_func = (void *) sqrt_f16;                break;
            case HTP_OP_L2_NORM:         compute_func = (void *) l2_norm_f16;             break;
            case HTP_OP_UNARY_ABS:       compute_func = (void *) abs_f16;                 break;
            case HTP_OP_UNARY_LOG:       compute_func = (void *) log_f16;                 break;
            case HTP_OP_UNARY_STEP:      compute_func = (void *) step_f16;                break;
            default:                     break;
        }
    } else {
        task_func = unary_thread_row_block;
        switch (octx->op) {
            case HTP_OP_NORM:            compute_func = (void *) norm_f32;                break;
            case HTP_OP_RMS_NORM:        compute_func = (void *) rms_norm_f32;            break;
            case HTP_OP_RMS_NORM_MUL:
                task_func    = unary_thread_rms_norm_mul_f32;
                compute_func = (void *) rms_norm_mul_f32;
                break;
            case HTP_OP_ADD_RMS_NORM_MUL:
                task_func    = unary_thread_add_rms_norm_mul_f32;
                compute_func = (void *) add_rms_norm_mul_f32;
                break;
            case HTP_OP_RMS_NORM_MUL_ADD:
                task_func    = unary_thread_add_rms_norm_mul_f32;
                compute_func = (void *) rms_norm_mul_add_f32;
                break;
            case HTP_OP_SCALE:           compute_func = (void *) scale_f32;               break;
            case HTP_OP_CLAMP:           compute_func = (void *) clamp_f32;               break;
            case HTP_OP_LEAKY_RELU:      compute_func = (void *) leaky_relu_f32;          break;
            case HTP_OP_SQR:             compute_func = (void *) sqr_f32;                 break;
            case HTP_OP_SQRT:            compute_func = (void *) sqrt_f32;                break;
            case HTP_OP_UNARY_NEG:       compute_func = (void *) neg_f32;                 break;
            case HTP_OP_UNARY_EXP:       compute_func = (void *) exp_f32;                 break;
            case HTP_OP_UNARY_SIGMOID:   compute_func = (void *) sigmoid_f32;             break;
            case HTP_OP_UNARY_SILU:      compute_func = (void *) silu_f32;                break;
            case HTP_OP_UNARY_GELU:      compute_func = (void *) gelu_f32;                break;
            case HTP_OP_UNARY_GELU_QUICK: compute_func = (void *) gelu_quick_f32;          break;
            case HTP_OP_UNARY_GELU_ERF:  compute_func = (void *) gelu_erf_f32;            break;
            case HTP_OP_UNARY_SOFTPLUS:  compute_func = (void *) softplus_f32;            break;
            case HTP_OP_UNARY_TANH:      compute_func = (void *) tanh_f32;                break;
            case HTP_OP_UNARY_ABS:       compute_func = (void *) abs_f32;                 break;
            case HTP_OP_UNARY_LOG:       compute_func = (void *) log_f32;                 break;
            case HTP_OP_UNARY_RELU:      compute_func = (void *) relu_f32;                break;
            case HTP_OP_UNARY_STEP:      compute_func = (void *) step_f32;                break;
            case HTP_OP_L2_NORM:         compute_func = (void *) l2_norm_f32;             break;
            case HTP_OP_TRI:
                task_func    = unary_thread_tri_f32;
                compute_func = (void *) tri_f32;
                break;
            default:                     break;
        }
    }

    if (!task_func || !compute_func) {
        FARF(ERROR, "execute_op_unary: task function is NULL for op %d\n", octx->op);
        return HTP_STATUS_NO_SUPPORT;
    }

    uctx.compute = compute_func;
    work_queue_run(octx->ctx->work_queue, task_func, &uctx, n_threads);

    return err;
}

// RMS_NORM_MUL_Q8 entry: src[0]=f32 rows [k x m], src[1]=f32 norm weight [k], src[2]=qkv weight 0 (locates 1/s_k), dst=payload ceil(m/32)*(k/32)*2048 B then m f32 row scales at round_up(payload, 128).
// Any unmet precondition returns NO_SUPPORT; the host guarantees the consumer is the plain int8 op.
static int execute_op_rms_norm_mul_q8(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * wq   = octx->src[2];
    const struct htp_tensor * dst  = octx->dsts[0];
    const struct htp_unary_kernel_params * kparams = (const struct htp_unary_kernel_params *) octx->kernel_params;

    if (!src0 || !src1 || !wq || !dst || src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || octx->ctx->mdev.count > 1) {
        return HTP_STATUS_NO_SUPPORT;
    }
    const uint32_t k = src0->ne[0];
    const uint32_t m = src0->ne[1] * src0->ne[2] * src0->ne[3];
    if (k % 64 != 0 || m < 64 || (m % 32) != 0 || m > HMX_INT8_MAX_ROWS_Q8 || src1->ne[0] != k || (src0->data & 127) || (dst->data & 127) ||
        (src0->nb[1] & 127) || src1->ne[1] * src1->ne[2] * src1->ne[3] != 1) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const uint32_t n_thr = octx->n_threads;
    const uint32_t nkc   = k / 32;
    const size_t row_al  = hex_round_up((size_t) k * sizeof(float), 128);
    const size_t payload = (size_t) ((m + 31) / 32) * nkc * HMX_INT8_CROUTON_A;
    const size_t scale_off = hex_round_up(payload, 128);

    // VTCM: smoothing vector | per thread: [weight row | 2 input rows | stage x0 x1 words scales]
    const size_t inv_s_bytes = hex_round_up((size_t) k * sizeof(float), 128);
    const size_t scl_bytes   = hex_round_up(((size_t) ((m + n_thr - 1) / n_thr) + 2) * sizeof(float), 128);
    const size_t per_thr     = row_al /*w*/ + 4 * row_al /*2 x in pair*/ + 2 * row_al /*x0 x1*/ + (size_t) nkc * 256 /*2 x words*/ + scl_bytes;
    if (inv_s_bytes + (size_t) n_thr * per_thr > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }
    uint8_t * base = (uint8_t *) octx->ctx->vtcm_base;
    uint8_t * inv_s = base;
    uint8_t * thr0  = base + inv_s_bytes;

    // 1/s_k sits 128-aligned behind the packed records of weight 0 (same address math as the plain int8 op, F16 branch included)
    const size_t n_w   = wq->ne[1];
    const size_t rec_b = (size_t) nkc * 2048 + 2048;
    const size_t i8off = wq->type == HTP_TYPE_F16 ? hex_round_up((size_t) k * n_w * 2, 128) : hex_round_up((size_t) nkc * 18 * hex_round_up(n_w, 32), 128);
    dma_queue * dq0 = octx->ctx->dma[0];
    dma_queue_push(dq0, dma_make_data(inv_s, wq->data + i8off + (n_w / 64) * rec_b), (size_t) k * 4, (size_t) k * 4, (size_t) k * 4, 1);
    dma_queue_pop(dq0);

    struct htp_unary_context uctx;
    memset(&uctx, 0, sizeof(uctx));
    uctx.octx                  = octx;
    uctx.kparams               = kparams;
    uctx.data_src0             = src0->data;
    uctx.data_src1             = src1->data;
    uctx.data_dst              = dst->data;
    uctx.src0_data_row_size    = (size_t) k * sizeof(float);
    uctx.src1_data_row_size    = (size_t) k * sizeof(float);
    uctx.src0_row_size_aligned = row_al;
    uctx.src1_row_size_aligned = row_al;
    uctx.nc                    = k;
    uctx.vtcm_src1             = thr0;          // base of the per-thread blocks
    uctx.q8_stage_per_thread   = per_thr;
    uctx.q8_inv_s              = inv_s;
    uctx.q8_nrows              = m;
    uctx.q8_scale_off          = (uint32_t) scale_off;

    work_queue_run(octx->ctx->work_queue, unary_thread_rms_norm_mul_q8, &uctx, n_thr);
    return HTP_STATUS_OK;
}

int op_unary(struct htp_ops_context * octx) {
    if (octx->op == HTP_OP_RMS_NORM_MUL_Q8) {
        return execute_op_rms_norm_mul_q8(octx);
    }
    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
        case HTP_TYPE_F16:
            return execute_op_unary(octx);

        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
