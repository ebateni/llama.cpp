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
#include "hmx-act-tiles.h"

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
    bool                      tile_out;                 // RMS_NORM_MUL: also writes fp16 HMX activation tiles to dsts[1]
    bool                      skip_f32;                 // tile output only (HTP_OPFLAGS_ACT_TILE_ONLY): fp32 rows stay in VTCM
    dma_addr_t                data_tile;
    uint8_t *                 vtcm_tile;
    size_t                    vtcm_tile_size_per_thread;

    uint8_t *                 vtcm_src0;
    uint8_t *                 vtcm_src1;
    uint8_t *                 vtcm_dst;

    size_t                    vtcm_src0_size_per_thread;
    size_t                    vtcm_src1_size_per_thread;
    size_t                    vtcm_dst_size_per_thread;
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

// RMS_NORM+SCALE: GDN's Q/K L2-norm (build_gdn_l2_norm).
// SCALE's scale/bias are host scalars from the fused-away SCALE node's own op_params (packed into this fused op's op_params[1]/op_params[2], epsilon stays at op_params[0] matching plain RMS_NORM's layout) -- no second DMA'd buffer, unlike RMS_NORM_MUL's weight tensor.
static void rms_norm_scale_f32(const void * restrict src,
                               void * restrict dst,
                               const uint32_t num_rows,
                               const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    float scale_mul = 0.f;
    float bias = 0.f;
    memcpy(&epsilon,   &op_params[0], sizeof(float));
    memcpy(&scale_mul, &op_params[1], sizeof(float));
    memcpy(&bias,      &op_params[2], sizeof(float));

    if (op_params[3] >= 1 && (ne0 % 32) == 0) {
        // C2Elt (GGML_HEXAGON_C2_RS=1|2): the RMS_NORM+SCALE primitive with its per-op constants computed once per call
        const HVX_Vector eps_v       = hvx_vec_splat_f32(epsilon);
        const HVX_Vector denom_v     = hvx_vec_inverse_f32(hvx_vec_splat_f32((float) ne0));
        const HVX_Vector scale_mul_v = hvx_vec_splat_f32(scale_mul);
        const HVX_Vector bias_v      = hvx_vec_splat_f32(bias);
        for (uint32_t ir = 0; ir < num_rows; ir++) {
            const uint8_t * restrict src_local = (const uint8_t *) src + (ir * src0_row_size_aligned);
            uint8_t *       restrict dst_local = (uint8_t *) dst + (ir * dst_row_size_aligned);
            hvx_fast_rms_norm_scale_f32_c(src_local, dst_local, ne0, eps_v, denom_v, scale_mul_v, bias_v);
        }
        return;
    }


    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_scale_f32((const uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon, scale_mul, bias);
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

// gelu(x) = x * sigmoid(1.702 * x)  (quick/sigmoid approximation, matches CPU GELU_QUICK reference)
static void gelu_f32(const void * restrict src,
                     void * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_mul_scalar_f32(dst_local, src_local, 1.702f, ne0);
        hvx_sigmoid_f32_aa(dst_local, dst_local, ne0);
        hvx_mul_f32_aaa(dst_local, src_local, dst_local, ne0);
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

// softplus(x) = log(1 + exp(x)). Match CPU reference: ggml_compute_softplus_f32() in ggml-impl.h (same x>20 overflow guard). The Q6_V_vmux_QVV select below replicates the scalar reference's own `x>20 -> return x` guard exactly (same vmux idiom as hvx_vec_exp_f32_guard in hvx-exp.h) -- this is not optional.
// Correctness-critical: hvx_vec_exp_f32 internally clamps its input to (-88, 88) BEFORE exponentiating (hvx-exp.h) -- for x>20 (definitely x>88) a naive log(1+exp(x)) composition would silently return a wrong, saturated ~88.72 instead of the correct x. Vectorized via this backend's existing hvx_vec_exp_f32/hvx_vec_log_f32 primitives (hvx-exp.h/hvx-log.h, already used elsewhere in this file/header set -- no new numerical approximation introduced).
static inline void hvx_softplus_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const HVX_Vector twenty = hvx_vec_splat_f32(20.0f);
    const HVX_Vector one    = hvx_vec_splat_f32(1.0f);

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    #pragma unroll(4)
    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector x              = v_src[i];
        HVX_VectorPred pred_gt20  = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_Vector expx           = hvx_vec_exp_f32(x);
        HVX_Vector logv           = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
        v_dst[i]                  = Q6_V_vmux_QVV(pred_gt20, x, logv);
    }

    if (nloe > 0) {
        HVX_Vector x             = v_src[nvec];
        HVX_VectorPred pred_gt20 = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_Vector expx          = hvx_vec_exp_f32(x);
        HVX_Vector logv          = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
        HVX_Vector sp            = Q6_V_vmux_QVV(pred_gt20, x, logv);
        hvx_vec_store_a((void *) &v_dst[nvec], nloe * sizeof(float), sp);
    }
}

static void softplus_f32(const void * restrict src,
                         void * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_softplus_f32_aa(dst_local, src_local, ne0);
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
    hvx_mul_scalar_f32((uint8_t *) dst, (const uint8_t *) src, 1.702f, tw);
    hvx_sigmoid_f32_aa((uint8_t *) dst, (uint8_t *) dst, tw);
    hvx_mul_f32_aaa((uint8_t *) dst, (const uint8_t *) src, (uint8_t *) dst, tw);
}

static void tile_softplus_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
    (void) uctx;
    hvx_softplus_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
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

    // fp16 HMX activation tiles (tile output, see htp/hmx-act-tiles.h): per thread, two staging halves of one
    // 32-row tile block. A tile block is K/32 contiguous tiles in DDR, so each block is a single DMA.
    uint8_t * tile_vtcm_data = uctx->vtcm_tile ? (uctx->vtcm_tile + (ith * uctx->vtcm_tile_size_per_thread)) : NULL;
    const size_t tile_half        = uctx->vtcm_tile_size_per_thread / 2;
    const size_t tile_block_bytes = (size_t) (uctx->nc / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_SIZE;

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

        if (!uctx->skip_f32) {
            dma_queue_push(dma_q,
                dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
                nb1, dst_row_size_aligned, dst_data_row_size, 0);
        }
        if (uctx->tile_out) {
            // Tile credit (zero rows): popped with the dst credit above, before src0/src1.
            dma_queue_push(dma_q, dma_make_data(uctx->data_tile, tile_vtcm_data + (vtcm_idx * tile_half)),
                           128, 128, 128, 0);
        }

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

    for (uint32_t ir = src0_start_row, blk = 0; ir < src0_end_row; blk++) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_src0_contig, block_dst_contig,
                                                     ne01, div_ne01);

        // Tile-only: the fp32 rows of a block live in one VTCM scratch half (no DDR round trip), so no dst credit.
        void * dst_vtcm  = uctx->skip_f32 ? (void *) (dst_vtcm_data + ((blk & 1) * dst_vtcm_half_size))
                                          : (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * tile_vtcm = NULL;
        if (uctx->tile_out) {
            tile_vtcm = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        }
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
        if (!uctx->skip_f32) {
            dma_queue_push(dma_q,
                dma_make_data(data_dst + dst_off, dst_vtcm),
                nb1, dst_row_size_aligned, dst_data_row_size, block_size);
        }
        if (uctx->tile_out) {
            // fp32 rows of this block -> fp16 HMX tiles with the matmul activation converter. Rows past block_size
            // (M tail) convert to zero, so the DMA'd tile block is exact.
            __fp16 * tile_dst = (__fp16 *) tile_vtcm;
            for (uint32_t r = 0; r < HTP_MM_HMX_TILE_N_ROWS; r += 2) {
                const float * row0 = (const float *) ((const uint8_t *) dst_vtcm + (size_t) r * dst_row_size_aligned);
                const float * row1 = (const float *) ((const uint8_t *) dst_vtcm + (size_t) (r + 1) * dst_row_size_aligned);
                act_tile_row_pair_store(tile_dst, row0, row1, r, uctx->nc, uctx->nc, r < block_size, (r + 1) < block_size);
            }
            const size_t tile_off = (size_t) (ir / HTP_MM_HMX_TILE_N_ROWS) * tile_block_bytes;
            dma_queue_push(dma_q, dma_make_data(uctx->data_tile + tile_off, tile_vtcm),
                           tile_block_bytes, tile_block_bytes, tile_block_bytes, 1);
        }

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
    const struct htp_tensor * dst  = octx->dst;

    const bool is_f16 = (src0->type == HTP_TYPE_F16);

    const char * op_type = NULL;

    switch (octx->op) {
        case HTP_OP_NORM:            op_type = is_f16 ? "norm-f16"     : "norm-f32";         break;
        case HTP_OP_RMS_NORM:        op_type = is_f16 ? "rmsnorm-f16"  : "rmsnorm-f32";      break;
        case HTP_OP_RMS_NORM_MUL:    op_type = "rmsnorm-mul-f32";                            break;
        case HTP_OP_RMS_NORM_SCALE:  op_type = "rmsnorm-scale-f32";                          break;
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
    if (octx->op == HTP_OP_RMS_NORM_MUL) {
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
    const bool tile_out = (octx->op == HTP_OP_RMS_NORM_MUL) && octx->dsts[1] != NULL &&
                          (octx->dsts[1]->flags & HTP_TENSOR_ACT_TILE) != 0;
    // Tile output needs 32-row aligned block starts, so each thread's row range is rounded up to whole tile blocks.
    uint32_t nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    if (tile_out) {
        nrows_per_thread = (nrows_per_thread + HTP_MM_HMX_TILE_N_ROWS - 1) & ~((uint32_t) HTP_MM_HMX_TILE_N_ROWS - 1);
    }
    struct htp_unary_context uctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .src0_nrows_per_thread = nrows_per_thread,
        .src0_nrows            = nrows,
        .row_start             = row_start,

        .data_src0             = src0->data,
        .data_src1             = (octx->op == HTP_OP_RMS_NORM_MUL) ? src1->data : 0,
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

        .vtcm_src0_size_per_thread = kparams->vtcm_src0_size_per_thread,
        .vtcm_src1_size_per_thread = kparams->vtcm_src1_size_per_thread,
        .vtcm_dst_size_per_thread  = kparams->vtcm_dst_size_per_thread,
        .skip_f32                  = tile_out && (octx->flags & HTP_OPFLAGS_ACT_TILE_ONLY) != 0,
        .tile_out                  = tile_out,
        .data_tile                 = tile_out ? octx->dsts[1]->data : 0,
        .vtcm_tile                 = tile_out ? VTCM_LAYOUT_PTR(uint8_t, base, kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size) : NULL,
        .vtcm_tile_size_per_thread = kparams->vtcm_tile_size_per_thread,
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
            case HTP_OP_RMS_NORM_SCALE:  compute_func = (void *) rms_norm_scale_f32;      break;
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

int op_unary(struct htp_ops_context * octx) {
    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
        case HTP_TYPE_F16:
            return execute_op_unary(octx);

        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}

// --------------------------------------------------------------------------------------------- GDN gate prep, fused (host: try_fuse_gdn_gate_mul/_sigmoid; GGML_HEXAGON_FUSE_S150 covers T <= 32 and GGML_HEXAGON_GDN_GATE_PF covers prefill up to GDN_GATE_MAX_T). Elementwise only (no reductions). Layout is host-checked: x [H,T] contiguous; A [H]; beta and beta_sig are [1,H,T,1] (rows of H floats at stride nb[2]); gate [H,T] contiguous. Rows are split into whole blocks of GDN_GATE_BLOCK over the session threads.
// The output rows are the same rows as the input rows, so an output may share memory with an input. Replaces SOFTPLUS(x) -> MUL(A) -> gate and SIGMOID(beta) -> beta_sig: gate[t][h] = softplus(x[t][h]) * A[h] (x = alpha + dt, biased by the MUL_MAT+ADD fusion) beta_sig[t][h] = sigmoid(beta[t][h]) Uses the same per-element HVX primitives as the unfused ops (hvx_softplus_f32_aa, hvx_mul_f32_aaa, hvx_sigmoid_f32_aa), rows staged in 128-byte VTCM slots, so the bits match the unfused chain. Each thread stages its block of x (and beta) by DMA, computes in VTCM, and writes gate (and beta_sig) back by DMA before the next block.
#define GDN_GATE_MAX_H     32
#define GDN_GATE_MAX_T     1024  // same bound as the host (ggml-hexagon.cpp try_fuse_gdn_gate_mul)
#define GDN_GATE_BLOCK     32
#define GDN_GATE_SLOT      128
// per-thread VTCM: x, beta, gate, beta_sig blocks and the staged A row
#define GDN_GATE_THREAD_BYTES ((4 * GDN_GATE_BLOCK + 1) * GDN_GATE_SLOT)

// a full DMA queue is drained (flush waits for in-flight transfers), then the push is retried
#define GDN_GATE_DMA_PUSH(q, ...)                  \
    do {                                           \
        if (!dma_queue_push((q), __VA_ARGS__)) {   \
            dma_queue_flush((q));                  \
            dma_queue_push((q), __VA_ARGS__);      \
        }                                          \
    } while (0)

struct htp_gdn_gate_context {
    struct htp_ops_context * octx;
    uint32_t                 H;
    uint32_t                 T;
    uint32_t                 rows_per_thread;   // multiple of GDN_GATE_BLOCK
    uint8_t *                vtcm;              // GDN_GATE_THREAD_BYTES per thread
    int                      has_sigmoid;
};

static void gdn_gate_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_gdn_gate_context * gctx = (const struct htp_gdn_gate_context *) data;
    struct htp_ops_context *            octx = gctx->octx;

    const struct htp_tensor * x    = octx->src[0];
    const struct htp_tensor * a    = octx->src[1];
    const struct htp_tensor * beta = octx->src[2];
    const struct htp_tensor * gate = octx->dsts[0];
    const struct htp_tensor * bsig = octx->dsts[1];

    const uint32_t H   = gctx->H;
    const uint32_t ir0 = ith * gctx->rows_per_thread;
    if (ir0 >= gctx->T) {
        return;
    }
    const uint32_t ir1 = MIN(ir0 + gctx->rows_per_thread, gctx->T);

    const size_t row_bytes = (size_t) H * sizeof(float);

    uint8_t * const base = gctx->vtcm + (size_t) ith * GDN_GATE_THREAD_BYTES;
    uint8_t * const v_x  = base;
    uint8_t * const v_b  = v_x + GDN_GATE_BLOCK * GDN_GATE_SLOT;
    uint8_t * const v_g  = v_b + GDN_GATE_BLOCK * GDN_GATE_SLOT;
    uint8_t * const v_s  = v_g + GDN_GATE_BLOCK * GDN_GATE_SLOT;
    uint8_t * const v_a  = v_s + GDN_GATE_BLOCK * GDN_GATE_SLOT;

    dma_queue *                  dma_q = octx->ctx->dma[ith];
    struct htp_thread_trace *    tr    = &octx->ctx->trace[ith];

    // the one-row A weight is staged once per thread
    GDN_GATE_DMA_PUSH(dma_q, dma_make_data(v_a, a->data), row_bytes, row_bytes, row_bytes, 1);
    dma_queue_pop(dma_q);

    for (uint32_t ir = ir0; ir < ir1; ir += GDN_GATE_BLOCK) {
        const uint32_t n = MIN((uint32_t) GDN_GATE_BLOCK, ir1 - ir);

        // stage the x rows (and the beta rows) of this block; both transfers are in flight before the first wait
        GDN_GATE_DMA_PUSH(dma_q, dma_make_data(v_x, x->data + (size_t) ir * x->nb[1]),
                          GDN_GATE_SLOT, x->nb[1], row_bytes, n);
        if (gctx->has_sigmoid) {
            GDN_GATE_DMA_PUSH(dma_q, dma_make_data(v_b, beta->data + (size_t) ir * beta->nb[2]),
                              GDN_GATE_SLOT, beta->nb[2], row_bytes, n);
        }
        dma_queue_pop(dma_q);
        if (gctx->has_sigmoid) {
            dma_queue_pop(dma_q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        for (uint32_t t = 0; t < n; t++) {
            uint8_t * sx = v_x + t * GDN_GATE_SLOT;
            uint8_t * sg = v_g + t * GDN_GATE_SLOT;
            hvx_softplus_f32_aa(sg, sx, H);    // softplus(x)     (same as the SOFTPLUS op)
            hvx_mul_f32_aaa(sg, sg, v_a, H);   // * A            (same as the MUL op, A broadcast over rows)
            if (gctx->has_sigmoid) {
                hvx_sigmoid_f32_aa(v_s + t * GDN_GATE_SLOT, v_b + t * GDN_GATE_SLOT, H);  // sigmoid(beta) (same as SIGMOID)
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // write back: gate rows (dst stride nb[1]) and beta_sig rows (dst stride nb[2])
        GDN_GATE_DMA_PUSH(dma_q, dma_make_data(gate->data + (size_t) ir * gate->nb[1], v_g),
                          gate->nb[1], GDN_GATE_SLOT, row_bytes, n);
        if (gctx->has_sigmoid) {
            GDN_GATE_DMA_PUSH(dma_q, dma_make_data(bsig->data + (size_t) ir * bsig->nb[2], v_s),
                              bsig->nb[2], GDN_GATE_SLOT, row_bytes, n);
        }
        // both writes are in flight before the first wait (one dma_queue_pop waits for one transfer)
        dma_queue_pop(dma_q);
        if (gctx->has_sigmoid) {
            dma_queue_pop(dma_q);
        }
    }
}

int op_gdn_gate(struct htp_ops_context * octx) {
    const struct htp_tensor * x    = octx->src[0];
    const struct htp_tensor * a    = octx->src[1];
    const struct htp_tensor * beta = octx->src[2];
    const struct htp_tensor * gate = octx->dsts[0];
    const struct htp_tensor * bsig = octx->dsts[1];

    if (x->type != HTP_TYPE_F32 || a->type != HTP_TYPE_F32 || gate->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    const int has_sigmoid = beta != NULL && bsig != NULL;
    if (has_sigmoid && (beta->type != HTP_TYPE_F32 || bsig->type != HTP_TYPE_F32)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t H = x->ne[0];
    const uint32_t T = x->ne[1];
    if (H == 0 || H > GDN_GATE_MAX_H || T == 0 || T > GDN_GATE_MAX_T || a->ne[0] != H) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t n_blocks = (T + GDN_GATE_BLOCK - 1) / GDN_GATE_BLOCK;
    const uint32_t nth      = MIN(octx->n_threads, n_blocks);
    const uint32_t rows_per_thread = ((n_blocks + nth - 1) / nth) * GDN_GATE_BLOCK;

    if (!htp_ops_context_set_n_threads(octx, nth)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (octx->ctx->vtcm_size < (size_t) nth * GDN_GATE_THREAD_BYTES) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    struct htp_gdn_gate_context gctx = {
        .octx            = octx,
        .H               = H,
        .T               = T,
        .rows_per_thread = rows_per_thread,
        .vtcm            = (uint8_t *) octx->ctx->vtcm_base,
        .has_sigmoid     = has_sigmoid,
    };

    work_queue_run(octx->ctx->work_queue, gdn_gate_thread, &gctx, nth);

    return HTP_STATUS_OK;
}

// --------------------------------------------------------------------------------------------- Attention gate multiply, fused (host: try_fuse_gate_mul, GGML_HEXAGON_GATE_MUL=1): MUL(x, SIGMOID(g)) becomes one op: y[t][w] = x[t][w] * sigmoid(g[t][k][s]), w = k*S + s, k < NSEG, s < S x and y are contiguous [W, T] tensors (W = NSEG*S). Tokens are split into blocks over the session threads; each thread stages its block by DMA on its own queue (x rows, then g), computes in VTCM and writes y back by DMA.
// g is the SIGMOID input: the contiguous [W, T] tensor (NSEG = 1) or the [S, NSEG, T] gate view (NSEG = heads). Same bits as the unfused chain: the same per-element HVX primitives in the same operand order (hvx_sigmoid_f32_aa, then hvx_mul_f32_aaa with the sigmoid result as src1). Only DMA touches DDR, so the op is DMA-clean. op_params: [0] = S, [1] = NSEG, [2] = head stride (bytes), [3] = token stride (bytes); the host guarantees token stride == NSEG * head stride, so the g segments of a token block form one strided 2D transfer.
#define GATE_MUL_BLOCK_T 8      // tokens per staged block
#define GATE_MUL_MAX_W   4096   // W = NSEG*S floats per token (16 KB per row)

#define GATE_MUL_DMA_PUSH(q, ...)                  \
    do {                                           \
        if (!dma_queue_push((q), __VA_ARGS__)) {   \
            dma_queue_flush((q));                  \
            dma_queue_push((q), __VA_ARGS__);      \
        }                                          \
    } while (0)

struct htp_gate_mul_context {
    struct htp_ops_context * octx;
    uint32_t                 W;                 // floats per token row (NSEG * S)
    uint32_t                 S;                 // floats per g segment
    uint32_t                 NSEG;              // g segments per token
    uint32_t                 hs;                // g head stride (bytes)
    uint32_t                 ts;                // g token stride (bytes)
    uint32_t                 T;
    uint32_t                 tokens_per_thread; // multiple of GATE_MUL_BLOCK_T
    uint8_t *                vtcm;
    size_t                   thread_bytes;      // 4 * GATE_MUL_BLOCK_T * W * 4: two block sets of x and g rows
    bool                     tile_out;          // dsts[1] = fp16 HMX activation tiles (GGML_HEXAGON_GATE_TILE): y rows go out as tiles
};

static void gate_mul_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_gate_mul_context * gctx = (const struct htp_gate_mul_context *) data;
    struct htp_ops_context *            octx = gctx->octx;

    const struct htp_tensor * x = octx->src[0];
    const struct htp_tensor * g = octx->src[1];
    const struct htp_tensor * y = octx->dst;

    const uint32_t t0 = ith * gctx->tokens_per_thread;
    if (t0 >= gctx->T) {
        return;
    }
    const uint32_t t1 = MIN(t0 + gctx->tokens_per_thread, gctx->T);

    const uint32_t W     = gctx->W;
    const uint32_t NSEG  = gctx->NSEG;
    const size_t   row_b = (size_t) W * sizeof(float);
    const size_t   seg_b = (size_t) gctx->S * sizeof(float);
    const size_t   blk_b = (size_t) GATE_MUL_BLOCK_T * row_b;
    const uint32_t n_blk = (t1 - t0 + GATE_MUL_BLOCK_T - 1) / GATE_MUL_BLOCK_T;

    // two block sets, block b uses set b & 1: x rows and g rows. The product is written back over the g rows (DMA out).
    uint8_t * const base   = gctx->vtcm + (size_t) ith * gctx->thread_bytes;
    uint8_t * const v_x[2] = { base,         base + 2 * blk_b };
    uint8_t * const v_g[2] = { base + blk_b, base + 3 * blk_b };

    dma_queue * dma_q = octx->ctx->dma[ith];

    // stage block b_idx (tokens t0 + b_idx * GATE_MUL_BLOCK_T ...): x rows (DDR stride nb[1]), then the g segments
    #define GATE_MUL_PUSH_IN(b_idx)                                                                      \
        do {                                                                                             \
            const uint32_t tt_ = t0 + (b_idx) * GATE_MUL_BLOCK_T;                                        \
            const uint32_t nn_ = MIN((uint32_t) GATE_MUL_BLOCK_T, t1 - tt_);                             \
            GATE_MUL_DMA_PUSH(dma_q, dma_make_data(v_x[(b_idx) & 1], x->data + (size_t) tt_ * x->nb[1]), \
                              row_b, x->nb[1], row_b, nn_);                                              \
            GATE_MUL_DMA_PUSH(dma_q, dma_make_data(v_g[(b_idx) & 1], g->data + (size_t) tt_ * gctx->ts), \
                              seg_b, gctx->hs, seg_b, nn_ * NSEG);                                      \
        } while (0)

    if (n_blk == 0) {
        return;
    }
    GATE_MUL_PUSH_IN(0);
    for (uint32_t b = 0; b < n_blk; b++) {
        const uint32_t cur = b & 1;
        const uint32_t tt  = t0 + b * GATE_MUL_BLOCK_T;
        const uint32_t n   = MIN((uint32_t) GATE_MUL_BLOCK_T, t1 - tt);

        dma_queue_pop(dma_q);            // x of block b
        dma_queue_pop(dma_q);            // g of block b
        if (b >= 1) {
            dma_queue_pop(dma_q);        // y of block b - 1 (its set is refilled by block b + 1 below)
        }
        if (b + 1 < n_blk) {
            GATE_MUL_PUSH_IN(b + 1);     // prefetch overlaps the compute of block b
        }

        for (uint32_t j = 0; j < n; j++) {
            uint8_t * xr = v_x[cur] + (size_t) j * row_b;
            uint8_t * gr = v_g[cur] + (size_t) j * row_b;
            hvx_sigmoid_f32_aa(gr, gr, W);      // sigmoid(g): the same primitive as SIGMOID
            hvx_mul_f32_aaa(gr, xr, gr, W);     // x * sigmoid(g): the same primitive and operand order as MUL
        }

        if (gctx->tile_out) {
            // fp32 rows -> fp16 tiles (htp/hmx-act-tiles.h): token pair (t, t + 1) fills vector t / 2 of every 32-column tile of the token's 32-token tile row (the conversion of act_tile_row_pair_store: hvx_vec_f32_to_f16_shuff(row t, row t + 1)).
            // A block of 8 tokens is vectors 4 * ((tt % 32) / 8) .. The tiles are staged in the (consumed) x set of this block. +3 of each tile: one 512-byte row per column tile, tile stride 2048 bytes.
            const uint32_t cpr = W / 32;
            uint8_t * tile_stage = v_x[cur];
            for (uint32_t p = 0; p < GATE_MUL_BLOCK_T / 2; p++) {
                const HVX_Vector * ra = (const HVX_Vector *) (v_g[cur] + (size_t) (2 * p) * row_b);
                const HVX_Vector * rb = (const HVX_Vector *) (v_g[cur] + (size_t) (2 * p + 1) * row_b);
                for (uint32_t c = 0; c < cpr; c++) {
                    ((HVX_Vector *) (tile_stage + (size_t) c * 512))[p] = hvx_vec_f32_to_f16_shuff(ra[c], rb[c]);
                }
            }
            const size_t tile_off = ((size_t) (tt / 32) * cpr) * HTP_MM_HMX_TILE_SIZE + (size_t) ((tt % 32) / 2) * 128;
            GATE_MUL_DMA_PUSH(dma_q, dma_make_data(octx->dsts[1]->data + tile_off, tile_stage), HTP_MM_HMX_TILE_SIZE, 512, 512, cpr);
        } else {
            // y rows tt .. tt+n-1 from the g set
            GATE_MUL_DMA_PUSH(dma_q, dma_make_data(y->data + (size_t) tt * y->nb[1], v_g[cur]), y->nb[1], row_b, row_b, n);
        }
    }
    dma_queue_pop(dma_q);                // y of the last block

    #undef GATE_MUL_PUSH_IN
}

int op_gate_mul(struct htp_ops_context * octx) {
    const struct htp_tensor * x = octx->src[0];
    const struct htp_tensor * g = octx->src[1];
    const struct htp_tensor * y = octx->dst;

    if (x->type != HTP_TYPE_F32 || g->type != HTP_TYPE_F32 || y->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const int32_t S    = octx->op_params[0];
    const int32_t NSEG = octx->op_params[1];
    const int32_t hs   = octx->op_params[2];
    const int32_t ts   = octx->op_params[3];
    const uint32_t W   = x->ne[0];
    const uint32_t T   = x->ne[1];
    if (S <= 0 || NSEG <= 0 || hs <= 0 || ts <= 0 || T == 0 || (uint32_t) S * (uint32_t) NSEG != W || W > GATE_MUL_MAX_W ||
        (W % 32) != 0 || (S % 32) != 0 || hs < S * (int32_t) sizeof(float) || ts != NSEG * hs ||
        x->ne[2] != 1 || x->ne[3] != 1 || y->ne[0] != W || y->ne[1] != T || y->ne[2] != 1 || y->ne[3] != 1 ||
        x->nb[0] != sizeof(float) || y->nb[0] != sizeof(float) || g->nb[0] != sizeof(float) ||
        x->nb[1] != (size_t) W * sizeof(float) || y->nb[1] != (size_t) W * sizeof(float)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t n_blocks          = (T + GATE_MUL_BLOCK_T - 1) / GATE_MUL_BLOCK_T;
    const uint32_t nth               = MIN(octx->n_threads, n_blocks);
    const uint32_t tokens_per_thread = ((n_blocks + nth - 1) / nth) * GATE_MUL_BLOCK_T;

    if (!htp_ops_context_set_n_threads(octx, nth)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const size_t thread_bytes = (size_t) 4 * GATE_MUL_BLOCK_T * W * sizeof(float);
    if (octx->ctx->vtcm_size < (size_t) nth * thread_bytes) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const bool tile_out = octx->dsts[1] != NULL && (octx->dsts[1]->flags & HTP_TENSOR_ACT_TILE) != 0;
    if (tile_out && ((T % 32) != 0 || (octx->flags & HTP_OPFLAGS_ACT_TILE_ONLY) == 0 || (W % 32) != 0)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    struct htp_gate_mul_context gctx = {
        .tile_out          = tile_out,
        .octx              = octx,
        .W                 = W,
        .S                 = (uint32_t) S,
        .NSEG              = (uint32_t) NSEG,
        .hs                = (uint32_t) hs,
        .ts                = (uint32_t) ts,
        .T                 = T,
        .tokens_per_thread = tokens_per_thread,
        .vtcm              = (uint8_t *) octx->ctx->vtcm_base,
        .thread_bytes      = thread_bytes,
    };

    work_queue_run(octx->ctx->work_queue, gate_mul_thread, &gctx, nth);

    return HTP_STATUS_OK;
}

// --------------------------------------------------------------------------------------------- Gated output norm, fused (host: try_fuse_norm_gated, GGML_HEXAGON_NORM_GATED=1): RMS_NORM_MUL(x, w) followed by GLU_SWIGLU(z, normalized) = silu(z) * normalized becomes one op: dst[r][c] = RMS_NORM(x[r], eps)[c] * w[c] * SILU(z[r][c]), SILU(z) = z * sigmoid(z) x, z and dst are contiguous F32 tensors of ne0 floats per row (ne0 % 32 == 0, rows = ne1 * ne2 * ne3); w is one row (broadcast over rows).
// The arithmetic is the RMS_NORM_MUL primitive (hvx_fast_rms_norm_mul_f32) followed by the SWIGLU kernel's primitives in the same order (hvx_sigmoid_f32_aa, hvx_mul_f32_aaa), so the result is bit-identical to the unfused chain. Each thread works on whole blocks of GDN_NG_BLOCK rows and double buffers them: the next block's x and z are DMA'd while the current block is computed, and each output block is written back by DMA one block later. Output rows are the input rows, so dst may alias x or z exactly.
#define GDN_NG_BLOCK 32
// C2Elt: the product of hvx_mul_f32_aaa (its HVX_OP_MUL_F32 comes from hvx-arith.h and is not visible here): the IEEE
// Q6_Vsf_vmpy_VsfVsf on v79 and up, the qf32 product with its conversion before that.
#if __HVX_ARCH__ < 79
#define C2_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define C2_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif
// C2Elt SIG16 (GGML_HEXAGON_NG_SIG16=1, numerics-changing, default off): o *= z * sigmoid(z) for two rows of nvec fp32 vectors (nvec % 4 == 0). The four sigmoid chains of a group are independent and issued together so their latency overlaps; the z * sigmoid and normalized-row products stay in fp32.
// Each pair of fp32 vectors is packed into one fp16 vector, the sigmoid runs on 64 lanes in fp16 (hvx_vec_fast_sigmoid_f16), and the result is unpacked to the same two fp32 lanes.
static inline void hvx_silu_mul_rows_h16(HVX_Vector * o0, const HVX_Vector * z0, HVX_Vector * o1, const HVX_Vector * z1, uint32_t nvec) {
    for (uint32_t i = 0; i < nvec; i += 4) {
        HVX_Vector sh0 = hvx_vec_fast_sigmoid_f16(hvx_vec_f32_to_f16_shuff(z0[i],     z0[i + 1]));
        HVX_Vector sh1 = hvx_vec_fast_sigmoid_f16(hvx_vec_f32_to_f16_shuff(z0[i + 2], z0[i + 3]));
        HVX_Vector sh2 = hvx_vec_fast_sigmoid_f16(hvx_vec_f32_to_f16_shuff(z1[i],     z1[i + 1]));
        HVX_Vector sh3 = hvx_vec_fast_sigmoid_f16(hvx_vec_f32_to_f16_shuff(z1[i + 2], z1[i + 3]));
        HVX_VectorPair p0 = hvx_vec_f16_to_f32_shuff(sh0);
        HVX_VectorPair p1 = hvx_vec_f16_to_f32_shuff(sh1);
        HVX_VectorPair p2 = hvx_vec_f16_to_f32_shuff(sh2);
        HVX_VectorPair p3 = hvx_vec_f16_to_f32_shuff(sh3);
        o0[i]     = C2_MUL_F32(o0[i],     C2_MUL_F32(z0[i],     Q6_V_lo_W(p0)));
        o0[i + 1] = C2_MUL_F32(o0[i + 1], C2_MUL_F32(z0[i + 1], Q6_V_hi_W(p0)));
        o0[i + 2] = C2_MUL_F32(o0[i + 2], C2_MUL_F32(z0[i + 2], Q6_V_lo_W(p1)));
        o0[i + 3] = C2_MUL_F32(o0[i + 3], C2_MUL_F32(z0[i + 3], Q6_V_hi_W(p1)));
        o1[i]     = C2_MUL_F32(o1[i],     C2_MUL_F32(z1[i],     Q6_V_lo_W(p2)));
        o1[i + 1] = C2_MUL_F32(o1[i + 1], C2_MUL_F32(z1[i + 1], Q6_V_hi_W(p2)));
        o1[i + 2] = C2_MUL_F32(o1[i + 2], C2_MUL_F32(z1[i + 2], Q6_V_lo_W(p3)));
        o1[i + 3] = C2_MUL_F32(o1[i + 3], C2_MUL_F32(z1[i + 3], Q6_V_hi_W(p3)));
    }
}


// SIG16_V2 (GGML_HEXAGON_SIG16_V2=1): the same with the accurate, shorter fp16 sigmoid (hvx_vec_fast_sigmoid_f16_v2).
static inline void hvx_silu_mul_rows_h16_v2(HVX_Vector * o0, const HVX_Vector * z0, HVX_Vector * o1, const HVX_Vector * z1, uint32_t nvec) {
    for (uint32_t i = 0; i < nvec; i += 4) {
        HVX_Vector sh0 = hvx_vec_fast_sigmoid_f16_v2(hvx_vec_f32_to_f16_shuff(z0[i],     z0[i + 1]));
        HVX_Vector sh1 = hvx_vec_fast_sigmoid_f16_v2(hvx_vec_f32_to_f16_shuff(z0[i + 2], z0[i + 3]));
        HVX_Vector sh2 = hvx_vec_fast_sigmoid_f16_v2(hvx_vec_f32_to_f16_shuff(z1[i],     z1[i + 1]));
        HVX_Vector sh3 = hvx_vec_fast_sigmoid_f16_v2(hvx_vec_f32_to_f16_shuff(z1[i + 2], z1[i + 3]));
        HVX_VectorPair p0 = hvx_vec_f16_to_f32_shuff(sh0);
        HVX_VectorPair p1 = hvx_vec_f16_to_f32_shuff(sh1);
        HVX_VectorPair p2 = hvx_vec_f16_to_f32_shuff(sh2);
        HVX_VectorPair p3 = hvx_vec_f16_to_f32_shuff(sh3);
        o0[i]     = C2_MUL_F32(o0[i],     C2_MUL_F32(z0[i],     Q6_V_lo_W(p0)));
        o0[i + 1] = C2_MUL_F32(o0[i + 1], C2_MUL_F32(z0[i + 1], Q6_V_hi_W(p0)));
        o0[i + 2] = C2_MUL_F32(o0[i + 2], C2_MUL_F32(z0[i + 2], Q6_V_lo_W(p1)));
        o0[i + 3] = C2_MUL_F32(o0[i + 3], C2_MUL_F32(z0[i + 3], Q6_V_hi_W(p1)));
        o1[i]     = C2_MUL_F32(o1[i],     C2_MUL_F32(z1[i],     Q6_V_lo_W(p2)));
        o1[i + 1] = C2_MUL_F32(o1[i + 1], C2_MUL_F32(z1[i + 1], Q6_V_hi_W(p2)));
        o1[i + 2] = C2_MUL_F32(o1[i + 2], C2_MUL_F32(z1[i + 2], Q6_V_lo_W(p3)));
        o1[i + 3] = C2_MUL_F32(o1[i + 3], C2_MUL_F32(z1[i + 3], Q6_V_hi_W(p3)));
    }
}


struct htp_norm_gated_context {
    struct htp_ops_context * octx;
    uint32_t                 ne0;
    uint32_t                 row;             // ne0 * sizeof(float) rounded up to 128 bytes (VTCM row stride)
    uint32_t                 n_rows;
    uint32_t                 rows_per_thread; // multiple of GDN_NG_BLOCK
    uint8_t *                vtcm;            // per-thread regions of gdn_ng_thread_bytes()
    float                    eps;
    uint32_t                 variant;         // C2Elt: 0 = original row loop, 1 = per-op constants hoisted, 2 = also row pairs
    uint32_t                 stagger;         // C2Elt: 1 = 128 B stagger between the blocks of a thread (VTCM bank conflicts)
    uint32_t                 sig16;           // C2Elt SIG16: fp16 sigmoid for the row-pair path (op_params[1] bit 3)
};

// per thread: x and z for two blocks, output for two blocks, the weight row, and two SiLU scratch rows (variant 2)
static inline size_t gdn_ng_thread_bytes(uint32_t row) {
    return (size_t) (6 * GDN_NG_BLOCK + 3) * row + 6 * 128 + 128;   // + stagger slack (C2Elt bit 2) for the six blocks and the rows
}

static void norm_gated_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_norm_gated_context * gctx = (const struct htp_norm_gated_context *) data;
    struct htp_ops_context *              octx = gctx->octx;

    const struct htp_tensor * x   = octx->src[0];
    const struct htp_tensor * w   = octx->src[1];
    const struct htp_tensor * z   = octx->src[2];
    const struct htp_tensor * dst = octx->dsts[0];

    const uint32_t ne0   = gctx->ne0;
    const uint32_t row   = gctx->row;
    const size_t   row_bytes = (size_t) ne0 * sizeof(float);

    const uint32_t r0 = ith * gctx->rows_per_thread;
    if (r0 >= gctx->n_rows) {
        return;
    }
    const uint32_t r1      = MIN(r0 + gctx->rows_per_thread, gctx->n_rows);
    const uint32_t n_blk   = (r1 - r0 + GDN_NG_BLOCK - 1) / GDN_NG_BLOCK;
    const uint32_t slot    = GDN_NG_BLOCK * row + (gctx->stagger ? 128u : 0u);  // C2Elt: 128 B stagger avoids bank conflicts at power-of-two strides

    uint8_t * const base  = gctx->vtcm + (size_t) ith * gdn_ng_thread_bytes(row);
    uint8_t * const in_x[2] = { base + 0 * slot, base + 1 * slot };
    uint8_t * const in_z[2] = { base + 2 * slot, base + 3 * slot };
    uint8_t * const out[2]  = { base + 4 * slot, base + 5 * slot };
    uint8_t * const w_row   = base + 6 * slot;
    uint8_t * const s_row   = w_row + row;
    uint8_t * const s_row1  = s_row + row;                   // variant 2: second SiLU scratch row
    const uint32_t  nvec    = ne0 / VLEN_FP32;
    const HVX_Vector eps_v  = hvx_vec_splat_f32(gctx->eps);  // per-op constants, computed once per thread
    const HVX_Vector denom_v = hvx_vec_inverse_f32(hvx_vec_splat_f32((float) ne0));  // 1/n: the RMS_NORM mean factor
    const HVX_Vector one_v  = hvx_vec_splat_f32(1.f);        // the sigmoid constants of the SWIGLU / SIGMOID primitive
    const HVX_Vector max_v  = hvx_vec_splat_f32(87.f);
    const HVX_Vector min_v  = hvx_vec_splat_f32(-87.f);

    dma_queue *               dma_q = octx->ctx->dma[ith];
    struct htp_thread_trace * tr    = &octx->ctx->trace[ith];

    // the weight row is staged once per thread
    GDN_GATE_DMA_PUSH(dma_q, dma_make_data(w_row, w->data), row_bytes, row_bytes, row_bytes, 1);
    dma_queue_pop(dma_q);

    // block b covers rows [r0 + b * GDN_NG_BLOCK, ...) and uses buffer set b & 1
    #define GDN_NG_PUSH_IN(b_idx)                                                                   \
        do {                                                                                        \
            const uint32_t rr_ = r0 + (b_idx) * GDN_NG_BLOCK;                                       \
            const uint32_t nn_ = MIN((uint32_t) GDN_NG_BLOCK, r1 - rr_);                            \
            GDN_GATE_DMA_PUSH(dma_q, dma_make_data(in_x[(b_idx) & 1], x->data + (size_t) rr_ * x->nb[1]), \
                              row, x->nb[1], row_bytes, nn_);                                       \
            GDN_GATE_DMA_PUSH(dma_q, dma_make_data(in_z[(b_idx) & 1], z->data + (size_t) rr_ * z->nb[1]), \
                              row, z->nb[1], row_bytes, nn_);                                       \
        } while (0)

    GDN_NG_PUSH_IN(0);
    for (uint32_t b = 0; b < n_blk; b++) {
        const uint32_t cur = b & 1;
        const uint32_t rb  = r0 + b * GDN_NG_BLOCK;
        const uint32_t n   = MIN((uint32_t) GDN_NG_BLOCK, r1 - rb);

        if (b + 1 < n_blk) {
            GDN_NG_PUSH_IN(b + 1);
        }
        dma_queue_pop(dma_q);            // x of block b
        dma_queue_pop(dma_q);            // z of block b
        if (b >= 1) {
            dma_queue_pop(dma_q);        // output of block b - 1 (its buffer is reused by block b + 1)
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) rb);
        uint32_t t = 0;
        if (gctx->variant >= 2) {
            // C2Elt variant 2: two rows per step. Each element takes the same operations in the same order as the single-row
            // path below; the two rows' chains are issued back to back so their packets can interleave.
            for (; t + 1 < n; t += 2) {
                const HVX_Vector * x0 = (const HVX_Vector *) (in_x[cur] + (size_t) t * row);
                const HVX_Vector * x1 = (const HVX_Vector *) (in_x[cur] + (size_t) (t + 1) * row);
                const HVX_Vector * z0 = (const HVX_Vector *) (in_z[cur] + (size_t) t * row);
                const HVX_Vector * z1 = (const HVX_Vector *) (in_z[cur] + (size_t) (t + 1) * row);
                HVX_Vector *       o0 = (HVX_Vector *) (out[cur] + (size_t) t * row);
                HVX_Vector *       o1 = (HVX_Vector *) (out[cur] + (size_t) (t + 1) * row);
                HVX_Vector *       s0 = (HVX_Vector *) s_row;
                HVX_Vector *       s1 = (HVX_Vector *) s_row1;
                hvx_fast_rms_norm_mul_f32_c((const uint8_t *) x0, (const uint8_t *) w_row, (uint8_t *) o0, ne0, eps_v, denom_v);
                hvx_fast_rms_norm_mul_f32_c((const uint8_t *) x1, (const uint8_t *) w_row, (uint8_t *) o1, ne0, eps_v, denom_v);
                if (gctx->sig16 == 2) {
                    hvx_silu_mul_rows_h16_v2(o0, z0, o1, z1, nvec);
                } else if (gctx->sig16) {
                    hvx_silu_mul_rows_h16(o0, z0, o1, z1, nvec);
                } else {
                if (gctx->variant >= 3) {   // variant 3: two-iteration reciprocal (not bit-identical; PPL and HOLD gate)
                    for (uint32_t i = 0; i < nvec; i++) {
                        s0[i] = hvx_vec_fast_sigmoid_f32_guard_r2(z0[i], one_v, max_v, min_v);
                        s1[i] = hvx_vec_fast_sigmoid_f32_guard_r2(z1[i], one_v, max_v, min_v);
                    }
                } else {
                    for (uint32_t i = 0; i < nvec; i++) {   // sigmoid(z) of both rows: the SWIGLU / SIGMOID primitive
                        s0[i] = hvx_vec_fast_sigmoid_f32_guard(z0[i], one_v, max_v, min_v);
                        s1[i] = hvx_vec_fast_sigmoid_f32_guard(z1[i], one_v, max_v, min_v);
                    }
                }
                for (uint32_t i = 0; i < nvec; i++) {   // z * sigmoid(z): hvx_mul_f32_aaa(s, z, s)
                    s0[i] = C2_MUL_F32(z0[i], s0[i]);
                    s1[i] = C2_MUL_F32(z1[i], s1[i]);
                }
                for (uint32_t i = 0; i < nvec; i++) {   // times the normalized row: hvx_mul_f32_aaa(o, o, s)
                    o0[i] = C2_MUL_F32(o0[i], s0[i]);
                    o1[i] = C2_MUL_F32(o1[i], s1[i]);
                }
                }
            }
        }
        for (; t < n; t++) {
            const uint8_t * xr   = in_x[cur] + (size_t) t * row;
            const uint8_t * zr   = in_z[cur] + (size_t) t * row;
            uint8_t *       orow = out[cur] + (size_t) t * row;
            if (gctx->variant >= 1) {
                hvx_fast_rms_norm_mul_f32_c(xr, w_row, orow, ne0, eps_v, denom_v);   // RMS_NORM_MUL primitive, constants hoisted
            } else {
                hvx_fast_rms_norm_mul_f32(xr, w_row, orow, ne0, gctx->eps);  // RMS_NORM_MUL primitive
            }
            if (gctx->variant >= 3) {   // variant 3: two-iteration reciprocal (not bit-identical; PPL and HOLD gate)
                HVX_Vector *       sv = (HVX_Vector *) s_row;
                const HVX_Vector * zv = (const HVX_Vector *) zr;
                for (uint32_t i = 0; i < nvec; i++) {
                    sv[i] = hvx_vec_fast_sigmoid_f32_guard_r2(zv[i], one_v, max_v, min_v);
                }
            } else {
                hvx_sigmoid_f32_aa(s_row, zr, ne0);                          // SWIGLU: sigmoid(z) ...
            }
            hvx_mul_f32_aaa(s_row, zr, s_row, ne0);                      // ... times z
            hvx_mul_f32_aaa(orow, orow, s_row, ne0);                     // times the normalized row
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) rb);

        GDN_GATE_DMA_PUSH(dma_q, dma_make_data(dst->data + (size_t) rb * dst->nb[1], out[cur]),
                          dst->nb[1], row, row_bytes, n);
    }
    dma_queue_pop(dma_q);                // output of the last block

    #undef GDN_NG_PUSH_IN
}

// --------------------------------------------------------------------------------------------- Gated output norm with fp16 HMX activation tile output (host: GGML_HEXAGON_NG_TILE=1 + HTP_OPFLAGS_ACT_TILE_ONLY, dsts[1] = tile view). x, z: [ne0, nh, n_tok] fp32, row (t, h) at t * nh + h; the matmul activation is [K = nh * ne0, M = n_tok] with column h * ne0 + c.
// Same arithmetic as norm_gated_thread (row pairs), but the result goes to the FFN-style fp16 tile layout (htp/hmx-act-tiles.h) that ssm_out's HMX matmul consumes directly: no fp32 output, no fp32 -> fp16 activation convert in the matmul prologue. A unit is (32-token block tb, head group g of NG_TILE_HEADS heads): its x/z rows are a 2-D DMA (32 rows of nhu * ne0 floats), its output tiles (nhu * ne0/32 tiles of tile row tb, first column tile h0 * ne0/32) are one contiguous DMA.
#define NG_TILE_HEADS 4

struct htp_norm_gated_tile_context {
    struct htp_ops_context * octx;
    uint32_t                 ne0;
    uint32_t                 nh;
    uint32_t                 n_tok;
    uint32_t                 n_units;
    uint32_t                 units_per_thread;
    uint8_t *                vtcm;
    size_t                   thread_bytes;
    float                    eps;
    uint32_t                 sig16;
    uint32_t                 lm_norm;   // LM_NORM mode (op_params[1] bits 12-13)
};

static inline size_t ng_tile_buf_bytes(uint32_t ne0) {            // x or z or o buffer of one unit
    return (size_t) 32 * NG_TILE_HEADS * ne0 * sizeof(float);
}
static inline size_t ng_tile_tile_bytes(uint32_t ne0) {           // tile buffer of one unit
    return (size_t) NG_TILE_HEADS * (ne0 / 32) * HTP_MM_HMX_TILE_SIZE;
}
static inline size_t ng_tile_thread_bytes(uint32_t ne0) {
    return 5 * ng_tile_buf_bytes(ne0) + 2 * ng_tile_tile_bytes(ne0) + 4 * (size_t) ne0 * sizeof(float);
}

static void norm_gated_tile_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct htp_norm_gated_tile_context * g = (const struct htp_norm_gated_tile_context *) data;
    struct htp_ops_context * octx = g->octx;

    const struct htp_tensor * x  = octx->src[0];
    const struct htp_tensor * w  = octx->src[1];
    const struct htp_tensor * z  = octx->src[2];
    const struct htp_tensor * dt = octx->dsts[1];

    const uint32_t ne0 = g->ne0, nh = g->nh;
    const uint32_t cpr = ne0 / 32;                                 // column tiles per head
    const size_t   row = (size_t) ne0 * sizeof(float);             // ne0 % 32 == 0: a multiple of 128 bytes
    const uint32_t G   = (nh + NG_TILE_HEADS - 1) / NG_TILE_HEADS;
    const uint32_t nvec = ne0 / VLEN_FP32;

    const uint32_t u0 = ith * g->units_per_thread;
    const uint32_t u1 = MIN(u0 + g->units_per_thread, g->n_units);
    if (u0 >= u1) {
        return;
    }

    const size_t buf = ng_tile_buf_bytes(ne0);
    uint8_t * const base = g->vtcm + (size_t) ith * g->thread_bytes;
    uint8_t * const x_b[2] = { base, base + buf };
    uint8_t * const z_b[2] = { base + 2 * buf, base + 3 * buf };
    uint8_t * const o_b    = base + 4 * buf;
    uint8_t * const w_row  = o_b + buf;
    uint8_t * const s_row0 = w_row + row;
    uint8_t * const s_row1 = s_row0 + row;
    uint8_t * const t_b[2] = { s_row1 + row, s_row1 + row + ng_tile_tile_bytes(ne0) };

    const HVX_Vector eps_v   = hvx_vec_splat_f32(g->eps);
    const HVX_Vector denom_v = hvx_vec_inverse_f32(hvx_vec_splat_f32((float) ne0));
    const HVX_Vector one_v   = hvx_vec_splat_f32(1.f);
    const HVX_Vector max_v   = hvx_vec_splat_f32(87.f);
    const HVX_Vector min_v   = hvx_vec_splat_f32(-87.f);

    dma_queue *               dma_q = octx->ctx->dma[ith];
    struct htp_thread_trace * tr    = &octx->ctx->trace[ith];

    GDN_GATE_DMA_PUSH(dma_q, dma_make_data(w_row, w->data), row, row, row, 1);
    dma_queue_pop(dma_q);

    #define NG_TILE_PUSH_IN(u_idx)                                                                                   \
        do {                                                                                                         \
            const uint32_t tb_  = (u_idx) / G;                                                                       \
            const uint32_t h0_  = ((u_idx) % G) * NG_TILE_HEADS;                                                     \
            const uint32_t nhu_ = MIN((uint32_t) NG_TILE_HEADS, nh - h0_);                                           \
            const size_t   off_ = ((size_t) tb_ * 32 * nh + h0_) * row;                                              \
            GDN_GATE_DMA_PUSH(dma_q, dma_make_data(x_b[((u_idx) - u0) & 1], x->data + off_), nhu_ * row, nh * row, nhu_ * row, 32); \
            GDN_GATE_DMA_PUSH(dma_q, dma_make_data(z_b[((u_idx) - u0) & 1], z->data + off_), nhu_ * row, nh * row, nhu_ * row, 32); \
        } while (0)

    NG_TILE_PUSH_IN(u0);
    for (uint32_t u = u0; u < u1; u++) {
        const uint32_t k   = u - u0;
        const uint32_t cur = k & 1;
        const uint32_t tb  = u / G;
        const uint32_t h0  = (u % G) * NG_TILE_HEADS;
        const uint32_t nhu = MIN((uint32_t) NG_TILE_HEADS, nh - h0);
        const uint32_t nrows = 32 * nhu;

        if (u + 1 < u1) {
            NG_TILE_PUSH_IN(u + 1);
        }
        dma_queue_pop(dma_q);            // x of the unit
        dma_queue_pop(dma_q);            // z of the unit
        if (k >= 1) {
            dma_queue_pop(dma_q);        // tile output of unit k - 1 (its tile buffer is reused by unit k + 1)
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) u);
        for (uint32_t i = 0; i + 3 < nrows; i += 4) {
            const uint8_t * xr = x_b[cur] + (size_t) i * row;
            const HVX_Vector * z0 = (const HVX_Vector *) (z_b[cur] + (size_t) i * row);
            const HVX_Vector * z1 = (const HVX_Vector *) (z_b[cur] + (size_t) (i + 1) * row);
            const HVX_Vector * z2 = (const HVX_Vector *) (z_b[cur] + (size_t) (i + 2) * row);
            const HVX_Vector * z3 = (const HVX_Vector *) (z_b[cur] + (size_t) (i + 3) * row);
            uint8_t * orow = o_b + (size_t) i * row;
            HVX_Vector * o0 = (HVX_Vector *) orow;
            HVX_Vector * o1 = (HVX_Vector *) (orow + row);
            HVX_Vector * o2 = (HVX_Vector *) (orow + 2 * row);
            HVX_Vector * o3 = (HVX_Vector *) (orow + 3 * row);
            if (g->lm_norm) {
                hvx_fast_rms_norm_mul_f32_c4p(xr, xr + row, xr + 2 * row, xr + 3 * row, w_row, orow, orow + row, orow + 2 * row, orow + 3 * row,
                                              ne0, eps_v, denom_v, (int) g->lm_norm);
            } else {
                hvx_fast_rms_norm_mul_f32_c4(xr, xr + row, xr + 2 * row, xr + 3 * row, w_row, orow, orow + row, orow + 2 * row, orow + 3 * row,
                                             ne0, eps_v, denom_v);
            }
            if (g->sig16) {
                hvx_silu_mul_rows_h16(o0, z0, o1, z1, nvec);
                hvx_silu_mul_rows_h16(o2, z2, o3, z3, nvec);
            } else {
                HVX_Vector * s0 = (HVX_Vector *) s_row0;
                HVX_Vector * s1 = (HVX_Vector *) s_row1;
                for (int pr = 0; pr < 2; pr++) {
                    const HVX_Vector * za = pr ? z2 : z0;
                    const HVX_Vector * zb = pr ? z3 : z1;
                    HVX_Vector *       oa = pr ? o2 : o0;
                    HVX_Vector *       ob = pr ? o3 : o1;
                    for (uint32_t v = 0; v < nvec; v++) {
                        s0[v] = hvx_vec_fast_sigmoid_f32_guard(za[v], one_v, max_v, min_v);
                        s1[v] = hvx_vec_fast_sigmoid_f32_guard(zb[v], one_v, max_v, min_v);
                    }
                    for (uint32_t v = 0; v < nvec; v++) {
                        s0[v] = C2_MUL_F32(za[v], s0[v]);
                        s1[v] = C2_MUL_F32(zb[v], s1[v]);
                    }
                    for (uint32_t v = 0; v < nvec; v++) {
                        oa[v] = C2_MUL_F32(oa[v], s0[v]);
                        ob[v] = C2_MUL_F32(ob[v], s1[v]);
                    }
                }
            }
        }
        // fp32 rows -> fp16 tiles: token pair (t, t + 1) of head j fills vector t / 2 of every column tile of the head
        // (the conversion of act_tile_row_pair_store: hvx_vec_f32_to_f16_shuff(row t, row t + 1))
        for (uint32_t t = 0; t < 32; t += 2) {
            for (uint32_t j = 0; j < nhu; j++) {
                const HVX_Vector * ra = (const HVX_Vector *) (o_b + ((size_t) t * nhu + j) * row);
                const HVX_Vector * rb = (const HVX_Vector *) (o_b + ((size_t) (t + 1) * nhu + j) * row);
                for (uint32_t c = 0; c < cpr; c++) {
                    HVX_Vector * tile = (HVX_Vector *) (t_b[cur] + ((size_t) j * cpr + c) * HTP_MM_HMX_TILE_SIZE);
                    tile[t / 2] = hvx_vec_f32_to_f16_shuff(ra[c], rb[c]);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) u);

        const size_t tile_off = ((size_t) tb * (nh * cpr) + (size_t) h0 * cpr) * HTP_MM_HMX_TILE_SIZE;
        const size_t tile_bytes = (size_t) nhu * cpr * HTP_MM_HMX_TILE_SIZE;
        GDN_GATE_DMA_PUSH(dma_q, dma_make_data(dt->data + tile_off, t_b[cur]), tile_bytes, tile_bytes, tile_bytes, 1);
    }
    dma_queue_pop(dma_q);                // tile output of the last unit

    #undef NG_TILE_PUSH_IN
}

int op_rms_norm_mul_silu(struct htp_ops_context * octx) {
    const struct htp_tensor * x   = octx->src[0];
    const struct htp_tensor * w   = octx->src[1];
    const struct htp_tensor * z   = octx->src[2];
    const struct htp_tensor * dst = octx->dsts[0];

    if (!x || !w || !z || !dst || x->type != HTP_TYPE_F32 || w->type != HTP_TYPE_F32 ||
        z->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t ne0 = x->ne[0];
    if (ne0 == 0 || (ne0 % 32) != 0 || w->ne[0] != ne0 || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1) {
        return HTP_STATUS_NO_SUPPORT;
    }
    for (int d = 0; d < 4; d++) {
        if (z->ne[d] != x->ne[d] || dst->ne[d] != x->ne[d]) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }
    // rows are contiguous in x, z and dst (the host checks this too)
    if (x->nb[0] != sizeof(float) || z->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
        x->nb[1] != ne0 * sizeof(float) || z->nb[1] != ne0 * sizeof(float) || dst->nb[1] != ne0 * sizeof(float)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->dsts[1] != NULL && (octx->dsts[1]->flags & HTP_TENSOR_ACT_TILE) != 0) {
        // fp16 HMX tile output (host GGML_HEXAGON_NG_TILE): x, z rows are (token, head) with the heads inner
        const uint32_t nh    = x->ne[1];
        const uint32_t n_tok = x->ne[2] * x->ne[3];
        if ((n_tok % 32) != 0 || x->ne[3] != 1 || ne0 % 128 != 0 || (octx->flags & HTP_OPFLAGS_ACT_TILE_ONLY) == 0) {
            return HTP_STATUS_NO_SUPPORT;
        }
        const uint32_t G       = (nh + NG_TILE_HEADS - 1) / NG_TILE_HEADS;
        const uint32_t n_units = (n_tok / 32) * G;
        const uint32_t nth     = MIN(octx->n_threads, n_units);
        const size_t   tb      = ng_tile_thread_bytes(ne0);
        if (!htp_ops_context_set_n_threads(octx, nth)) {
            return HTP_STATUS_INVAL_PARAMS;
        }
        if ((size_t) octx->ctx->vtcm_size < (size_t) nth * tb) {
            return HTP_STATUS_VTCM_TOO_SMALL;
        }
        struct htp_norm_gated_tile_context tctx = {
            .octx             = octx,
            .ne0              = ne0,
            .nh               = nh,
            .n_tok            = n_tok,
            .n_units          = n_units,
            .units_per_thread = (n_units + nth - 1) / nth,
            .vtcm             = (uint8_t *) octx->ctx->vtcm_base,
            .thread_bytes     = tb,
        };
        memcpy(&tctx.eps, octx->op_params, sizeof(float));
        tctx.sig16 = (((uint32_t) octx->op_params[1] >> 3) & 1u) ? 1u : 0u;
        tctx.lm_norm = ((uint32_t) octx->op_params[1] >> 12) & 3u;
        work_queue_run(octx->ctx->work_queue, norm_gated_tile_thread, &tctx, nth);
        return HTP_STATUS_OK;
    }

    const uint32_t n_rows   = x->ne[1] * x->ne[2] * x->ne[3];
    const uint32_t row      = hex_round_up(ne0 * sizeof(float), 128);
    const uint32_t n_blocks = (n_rows + GDN_NG_BLOCK - 1) / GDN_NG_BLOCK;
    if (n_blocks == 0) {
        return HTP_STATUS_OK;
    }
    const uint32_t nth             = MIN(octx->n_threads, n_blocks);
    const uint32_t rows_per_thread = ((n_blocks + nth - 1) / nth) * GDN_NG_BLOCK;

    if (!htp_ops_context_set_n_threads(octx, nth)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if ((size_t) octx->ctx->vtcm_size < (size_t) nth * gdn_ng_thread_bytes(row)) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    struct htp_norm_gated_context gctx = {
        .octx            = octx,
        .ne0             = ne0,
        .row             = row,
        .n_rows          = n_rows,
        .rows_per_thread = rows_per_thread,
        .vtcm            = (uint8_t *) octx->ctx->vtcm_base,
    };
    memcpy(&gctx.eps, octx->op_params, sizeof(float));
    gctx.variant = (uint32_t) octx->op_params[1] & 3u;     // C2Elt: GGML_HEXAGON_C2_NG bits 0-1: 0 original, 1 hoisted, 2 + row pairs, 3 + two-iteration reciprocal
    gctx.stagger = ((uint32_t) octx->op_params[1] >> 2) & 1u;  // C2Elt: GGML_HEXAGON_C2_NG bit 2: 128 B stagger of the VTCM blocks
    gctx.sig16   = (((uint32_t) octx->op_params[1] >> 3) & 1u) && (ne0 % 128 == 0) && (n_rows % 2 == 0) ? (((uint32_t) octx->op_params[1] >> 4) & 1u ? 2u : 1u) : 0u;  // bit 4: SIG16_V2

    work_queue_run(octx->ctx->work_queue, norm_gated_thread, &gctx, nth);

    return HTP_STATUS_OK;
}
