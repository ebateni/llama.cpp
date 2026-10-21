#ifndef HVX_NORM_H
#define HVX_NORM_H

#include <stdint.h>
#include "hvx-base.h"
#include "hvx-reduce.h"
#include "hvx-inverse.h"
#include "hvx-sqrt.h"
#include "hvx-repl.h"

static inline void hvx_fast_rms_norm_f32(const uint8_t * restrict src,
                                         uint8_t * restrict dst,
                                         const int num_elems,
                                         float     epsilon) {

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;    // number of full vectors
    const int nloe = num_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Reduce HVX sum
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    // Scale full vectors
    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector result = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, result);
    }
}

static inline void hvx_fast_rms_norm_mul_f32(const uint8_t * restrict src,
                                             const uint8_t * restrict weight,
                                             uint8_t * restrict dst,
                                             const int num_elems,
                                             float     epsilon) {
    const HVX_Vector * restrict v_src    = (const HVX_Vector *) src;
    const HVX_Vector * restrict v_weight = (const HVX_Vector *) weight;
    HVX_Vector * restrict v_dst          = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;    // number of full vectors
    const int nloe = num_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Reduce HVX sum
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    // Scale and multiply
    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector result = Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(result);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector result = Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[nvec]);
        HVX_Vector res_v = Q6_Vsf_equals_Vqf32(result);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

// RMS_NORM+SCALE fusion for GDN's Q/K L2-norm (build_gdn_l2_norm in models.h: ggml_scale(ggml_rms_norm(x, eps/n), 1/sqrt(n))). SCALE's multiplier/bias are host scalars baked into op_params (ggml_scale_impl), not a device tensor like MUL's weight -- unlike hvx_fast_rms_norm_mul_f32 above, there is no second buffer to DMA in.
// This kernel reproduces that exact two-step rounding (round x*rsqrt(mean+eps) to f32 first, THEN multiply-add by scale_mul/bias and round again) rather than collapsing the whole row into one algebraically-equivalent-but-differently-rounded pass. Bit-exact with the unfused two-op sequence by construction: the unfused SCALE kernel (scale_f32/hvx_scale_offset_f32_aa) reads RMS_NORM's already-rounded-to-f32 output back from DRAM and applies its own separate qf32 multiply+add+round on top.
static inline void hvx_fast_rms_norm_scale_f32(const uint8_t * restrict src,
                                               uint8_t * restrict dst,
                                               const int num_elems,
                                               float     epsilon,
                                               float     scale_mul,
                                               float     bias) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;    // number of full vectors
    const int nloe = num_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Reduce HVX sum
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    HVX_Vector rms_scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));
    HVX_Vector scale_mul_v = hvx_vec_splat_f32(scale_mul);
    HVX_Vector bias_v      = hvx_vec_splat_f32(bias);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1  = v_src[i];
        HVX_Vector rms = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, rms_scale_v)); // RMS_NORM's own rounding
        HVX_Vector out = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(rms, scale_mul_v), bias_v); // SCALE's own rounding
        v_dst[i] = Q6_Vsf_equals_Vqf32(out);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1  = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector rms = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, rms_scale_v));
        HVX_Vector out = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(rms, scale_mul_v), bias_v);
        HVX_Vector result = Q6_Vsf_equals_Vqf32(out);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, result);
    }
}

static inline void hvx_fast_norm_f32(const uint8_t * restrict src,
                                     uint8_t * restrict dst,
                                     const int num_elems,
                                     float     epsilon) {

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;    // number of full vectors
    const int nloe = num_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares and sum of values for full vectors
    HVX_Vector sum_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector sum_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq_v, v2);
        sum_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq_v, v2);
        sum_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Reduce HVX sums
    sum_sq_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_sq_v));
    sum_x_v  = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_x_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_sq_v      = Q6_Vqf32_vmpy_VsfVsf(sum_sq_v, denom_v);
    HVX_Vector mean_x_v       = Q6_Vqf32_vmpy_VsfVsf(sum_x_v,  denom_v);
    HVX_Vector mean_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(mean_x_v), Q6_Vsf_equals_Vqf32(mean_x_v));
    HVX_Vector var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(mean_sq_v, mean_x_sq_v);
    HVX_Vector var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    // scale = rsqrt(variance + epsilon),  mean_x broadcast for subtraction
    HVX_Vector scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_Vector mean_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(mean_x_v));

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, mean_x_b);
        HVX_Vector v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v3);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, mean_x_b);
        HVX_Vector v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        HVX_Vector result = Q6_Vsf_equals_Vqf32(v3);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, result);
    }
}

static inline void hvx_fast_l2_norm_f32(const uint8_t * restrict src,
                                        uint8_t * restrict dst,
                                        const int num_elems,
                                        float     epsilon) {

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    HVX_Vector sum_v = hvx_vec_splat_f32(0.0f);

    const int nvec = num_elems / VLEN_FP32;
    const int nloe = num_elems % VLEN_FP32;

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v         = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, sq);
    }

    // Include tail elements in the sum-of-squares using a predicate mask
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v         = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, sq);
    }

    // Compute scale = 1/fmax(sqrt(sum), epsilon) entirely in HVX registers.
    // hvx_vec_rsqrt_f32 + hvx_vec_inverse_f32 avoids scalar extraction.
    HVX_Vector sum_sf    = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));
    HVX_Vector rsqrt_v   = hvx_vec_rsqrt_f32(sum_sf);              // 1/sqrt(sum)
    HVX_Vector sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);            // sqrt(sum)
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_Vector denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);  // fmax(sqrt(sum), epsilon)
    HVX_Vector scale_v   = hvx_vec_inverse_f32(denom_v);            // 1/fmax(sqrt(sum), epsilon)

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        v_dst[i]      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector result = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, result);
    }
}

// F16 norm kernels: reduce and scale in f32 (via promote/narrow), matching the
// precision-preserving pattern used by the flash-attn f16 kernels.

static inline void hvx_fast_rms_norm_f16(const uint8_t * restrict src,
                                         uint8_t * restrict dst,
                                         const int num_elems,
                                         float     epsilon) {

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP16;    // number of full f16 vectors
    const int nloe = num_elems % VLEN_FP16;    // leftover elements

    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_to_f16(r0, r1);
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_Vector result = hvx_vec_f32_to_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, result);
    }
}

static inline void hvx_fast_norm_f16(const uint8_t * restrict src,
                                     uint8_t * restrict dst,
                                     const int num_elems,
                                     float     epsilon) {

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP16;
    const int nloe = num_elems % VLEN_FP16;

    HVX_Vector sum_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector sum_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        sum_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        sum_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        sum_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        sum_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        sum_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        sum_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        sum_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        sum_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    sum_sq_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_sq_v));
    sum_x_v  = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_x_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_sq_v      = Q6_Vqf32_vmpy_VsfVsf(sum_sq_v, denom_v);
    HVX_Vector mean_x_v       = Q6_Vqf32_vmpy_VsfVsf(sum_x_v,  denom_v);
    HVX_Vector mean_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(mean_x_v), Q6_Vsf_equals_Vqf32(mean_x_v));
    HVX_Vector var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(mean_sq_v, mean_x_sq_v);
    HVX_Vector var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    HVX_Vector scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_Vector mean_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(mean_x_v));

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), mean_x_b);
        HVX_Vector d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), mean_x_b);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        v_dst[i] = hvx_vec_f32_to_f16(r0, r1);
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), mean_x_b);
        HVX_Vector d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), mean_x_b);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        HVX_Vector result = hvx_vec_f32_to_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, result);
    }
}

static inline void hvx_fast_l2_norm_f16(const uint8_t * restrict src,
                                        uint8_t * restrict dst,
                                        const int num_elems,
                                        float     epsilon) {

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP16;
    const int nloe = num_elems % VLEN_FP16;

    HVX_Vector sum_v = hvx_vec_splat_f32(0.0f);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector p0 = Q6_V_lo_W(p);
        HVX_Vector p1 = Q6_V_hi_W(p);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    HVX_Vector sum_sf    = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));
    HVX_Vector rsqrt_v   = hvx_vec_rsqrt_f32(sum_sf);
    HVX_Vector sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_Vector denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);
    HVX_Vector scale_v   = hvx_vec_inverse_f32(denom_v);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_to_f16(r0, r1);
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32(v1);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_Vector result = hvx_vec_f32_to_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, result);
    }
}

// C2Elt (GGML_HEXAGON_C2_NG / GGML_HEXAGON_C2_RS = 1..2): hvx_fast_rms_norm_mul_f32 and hvx_fast_rms_norm_scale_f32 with the per-op constants passed in (epsilon, 1/n as an HVX vector, scale, bias).
// The code shape is kept line for line, so the compiler makes the same fusion decisions. A first version with restructured loops changed the PPL per chunk (cabin 1.7254 for the hoisted variant against 1.7274 for OFF), so the loops stay as in the originals and only the constant lines move out.
static inline void hvx_fast_rms_norm_mul_f32_c(const uint8_t * restrict src, const uint8_t * restrict weight, uint8_t * restrict dst,
                                               const int num_elems, HVX_Vector epsilon_v, HVX_Vector denom_v) {
    const HVX_Vector * restrict v_src    = (const HVX_Vector *) src;
    const HVX_Vector * restrict v_weight = (const HVX_Vector *) weight;
    HVX_Vector * restrict v_dst          = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;    // number of full vectors
    const int nloe = num_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Reduce HVX sum
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));

    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    // Scale and multiply
    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector result = Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(result);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector result = Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[nvec]);
        HVX_Vector res_v = Q6_Vsf_equals_Vqf32(result);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

// Four rows at once (num_elems % 32 == 0): the same arithmetic per row as hvx_fast_rms_norm_mul_f32_c, but four independent
// sum -> reduce -> rsqrt -> scale chains are issued together so their latencies overlap (bit-identical results).
static inline void hvx_fast_rms_norm_mul_f32_c4(const uint8_t * restrict src0, const uint8_t * restrict src1, const uint8_t * restrict src2,
                                                const uint8_t * restrict src3, const uint8_t * restrict weight,
                                                uint8_t * restrict dst0, uint8_t * restrict dst1, uint8_t * restrict dst2,
                                                uint8_t * restrict dst3, const int num_elems, HVX_Vector epsilon_v, HVX_Vector denom_v) {
    const HVX_Vector * restrict vs[4] = { (const HVX_Vector *) src0, (const HVX_Vector *) src1, (const HVX_Vector *) src2, (const HVX_Vector *) src3 };
    HVX_Vector * restrict       vd[4] = { (HVX_Vector *) dst0, (HVX_Vector *) dst1, (HVX_Vector *) dst2, (HVX_Vector *) dst3 };
    const HVX_Vector * restrict v_weight = (const HVX_Vector *) weight;
    const int nvec = num_elems / VLEN_FP32;

    HVX_Vector sum[4];
    #pragma unroll(4)
    for (int k = 0; k < 4; k++) {
        sum[k] = Q6_V_vsplat_R(0x00000000);
    }
    for (int i = 0; i < nvec; i++) {
        #pragma unroll(4)
        for (int k = 0; k < 4; k++) {
            const HVX_Vector v1 = vs[k][i];
            sum[k] = Q6_Vqf32_vadd_Vqf32Vqf32(sum[k], Q6_Vqf32_vmpy_VsfVsf(v1, v1));
        }
    }
    HVX_Vector scale[4];
    #pragma unroll(4)
    for (int k = 0; k < 4; k++) {
        const HVX_Vector sum_v         = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum[k]));
        const HVX_Vector mean_v        = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
        const HVX_Vector mean_epsilon  = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);
        scale[k] = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon));
    }
    for (int i = 0; i < nvec; i++) {
        const HVX_Vector w = v_weight[i];
        #pragma unroll(4)
        for (int k = 0; k < 4; k++) {
            const HVX_Vector v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vs[k][i], scale[k]));
            vd[k][i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3, w));
        }
    }
}

// LM_NORM (GGML_HEXAGON_LM_NORM, numerics-changing, default off). mode 2: two iterations (rel. error ~5e-6); mode 3: one iteration with a retuned seed constant and polynomial coefficients (max rel.
// hvx_vec_rsqrt_f32 runs three Newton iterations, enough for the last fp32 bit; the consumers only keep ~fp16 accuracy. error 8.9e-4 over 2^-12..2^12).
static inline HVX_Vector hvx_vec_rsqrt_f32_lm(HVX_Vector in_vec, int mode) {
    if (mode <= 1) {
        return hvx_vec_rsqrt_f32(in_vec);
    }
    HVX_Vector onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_Vector threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);
    HVX_Vector x2, y, ypower2, temp;
    if (mode == 2) {
        x2 = Q6_Vqf32_vmpy_VsfVsf(in_vec, onehalf);
        x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());
        y  = Q6_Vw_vasr_VwR(in_vec, 1);
        y  = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(RSQRT_CONST), y);
        ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
        ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
        temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
        temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
        temp    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(temp));
        y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
        ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
        ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
        temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
        temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
        temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);
        return Q6_Vsf_equals_Vqf32(temp);
    }
    // mode 3: y0 = 0x5f37b1ca - (bits >> 1); y = y0 * (1.49869704 - 0.497808891 * x * y0^2)
    y  = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x5f37b1ca), Q6_Vw_vasr_VwR(in_vec, 1));
    ypower2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y, y));
    temp    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(in_vec, ypower2));
    temp    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(temp, Q6_V_vsplat_R(0x3efee0cf)));   // 0.497808891
    temp    = Q6_Vqf32_vsub_VsfVsf(Q6_V_vsplat_R(0x3fbfd54e), temp);                          // 1.49869704
    temp    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(temp));
    return Q6_Vsf_equals_Vqf32(temp);
}

// hvx_fast_rms_norm_mul_f32_c4 with the LM_NORM changes (mode 1..3): the four row sums come out of one hvx_vec_reduce_sum_f32x4 (lane j of the result holds the sum of row j mod 4), the four (mean + eps) values share one rsqrt evaluation (lanes 0..3), and each row's scale is that lane broadcast.
// Mode 1 keeps the three-iteration rsqrt (only the summation order of the row sums differs).
static inline void hvx_fast_rms_norm_mul_f32_c4p(const uint8_t * restrict src0, const uint8_t * restrict src1, const uint8_t * restrict src2,
                                                 const uint8_t * restrict src3, const uint8_t * restrict weight,
                                                 uint8_t * restrict dst0, uint8_t * restrict dst1, uint8_t * restrict dst2,
                                                 uint8_t * restrict dst3, const int num_elems, HVX_Vector epsilon_v, HVX_Vector denom_v, int mode) {
    const HVX_Vector * restrict vs[4] = { (const HVX_Vector *) src0, (const HVX_Vector *) src1, (const HVX_Vector *) src2, (const HVX_Vector *) src3 };
    HVX_Vector * restrict       vd[4] = { (HVX_Vector *) dst0, (HVX_Vector *) dst1, (HVX_Vector *) dst2, (HVX_Vector *) dst3 };
    const HVX_Vector * restrict v_weight = (const HVX_Vector *) weight;
    const int nvec = num_elems / VLEN_FP32;

    HVX_Vector_x4 sum;
    #pragma unroll(4)
    for (int k = 0; k < 4; k++) {
        sum.v[k] = Q6_V_vsplat_R(0x00000000);
    }
    for (int i = 0; i < nvec; i++) {
        #pragma unroll(4)
        for (int k = 0; k < 4; k++) {
            const HVX_Vector v1 = vs[k][i];
            sum.v[k] = Q6_Vqf32_vadd_Vqf32Vqf32(sum.v[k], Q6_Vqf32_vmpy_VsfVsf(v1, v1));
        }
    }
    #pragma unroll(4)
    for (int k = 0; k < 4; k++) {
        sum.v[k] = Q6_Vsf_equals_Vqf32(sum.v[k]);
    }
    const HVX_Vector sum_v        = hvx_vec_reduce_sum_f32x4(sum);
    const HVX_Vector mean_v       = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    const HVX_Vector mean_epsilon = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);
    const HVX_Vector rs           = hvx_vec_rsqrt_f32_lm(Q6_Vsf_equals_Vqf32(mean_epsilon), mode);
    HVX_Vector scale[4];
    #pragma unroll(4)
    for (int k = 0; k < 4; k++) {
        scale[k] = hvx_vec_repl_f32(Q6_V_vror_VR(rs, 4 * k));
    }
    for (int i = 0; i < nvec; i++) {
        const HVX_Vector w = v_weight[i];
        #pragma unroll(4)
        for (int k = 0; k < 4; k++) {
            const HVX_Vector v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vs[k][i], scale[k]));
            vd[k][i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3, w));
        }
    }
}

// n rows (row r at src + r * src_stride, dst + r * dst_stride; num_elems % 32 == 0), the same arithmetic per row as
// hvx_fast_rms_norm_mul_f32: groups of four rows through the interleaved kernel, the remainder one row at a time. src may equal dst.
static inline void hvx_fast_rms_norm_mul_f32_rows(const uint8_t * src, size_t src_stride, const uint8_t * weight, uint8_t * dst, size_t dst_stride,
                                                  uint32_t n_rows, const int num_elems, float epsilon) {
    uint32_t r = 0;
    if ((num_elems % VLEN_FP32) == 0 && n_rows >= 4) {
        const HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);
        const HVX_Vector denom_v   = hvx_vec_inverse_f32(hvx_vec_splat_f32((float) num_elems));
        for (; r + 3 < n_rows; r += 4) {
            hvx_fast_rms_norm_mul_f32_c4(src + r * src_stride, src + (r + 1) * src_stride, src + (r + 2) * src_stride, src + (r + 3) * src_stride,
                                         weight, dst + r * dst_stride, dst + (r + 1) * dst_stride, dst + (r + 2) * dst_stride,
                                         dst + (r + 3) * dst_stride, num_elems, epsilon_v, denom_v);
        }
    }
    for (; r < n_rows; r++) {
        hvx_fast_rms_norm_mul_f32(src + r * src_stride, weight, dst + r * dst_stride, num_elems, epsilon);
    }
}

// LM_NORM variant of the above (mode 0 = original path).
static inline void hvx_fast_rms_norm_mul_f32_rows_lm(const uint8_t * src, size_t src_stride, const uint8_t * weight, uint8_t * dst, size_t dst_stride,
                                                     uint32_t n_rows, const int num_elems, float epsilon, int mode) {
    if (mode == 0 || (num_elems % VLEN_FP32) != 0 || n_rows < 4) {
        hvx_fast_rms_norm_mul_f32_rows(src, src_stride, weight, dst, dst_stride, n_rows, num_elems, epsilon);
        return;
    }
    const HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);
    const HVX_Vector denom_v   = hvx_vec_inverse_f32(hvx_vec_splat_f32((float) num_elems));
    uint32_t r = 0;
    for (; r + 3 < n_rows; r += 4) {
        hvx_fast_rms_norm_mul_f32_c4p(src + r * src_stride, src + (r + 1) * src_stride, src + (r + 2) * src_stride, src + (r + 3) * src_stride,
                                      weight, dst + r * dst_stride, dst + (r + 1) * dst_stride, dst + (r + 2) * dst_stride,
                                      dst + (r + 3) * dst_stride, num_elems, epsilon_v, denom_v, mode);
    }
    for (; r < n_rows; r++) {
        hvx_fast_rms_norm_mul_f32(src + r * src_stride, weight, dst + r * dst_stride, num_elems, epsilon);
    }
}

static inline void hvx_fast_rms_norm_scale_f32_c(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems,
                                                 HVX_Vector epsilon_v, HVX_Vector denom_v, HVX_Vector scale_mul_v, HVX_Vector bias_v) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;    // number of full vectors
    const int nloe = num_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Reduce HVX sum
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));

    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    HVX_Vector rms_scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1  = v_src[i];
        HVX_Vector rms = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, rms_scale_v)); // RMS_NORM's own rounding
        HVX_Vector out = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(rms, scale_mul_v), bias_v); // SCALE's own rounding
        v_dst[i] = Q6_Vsf_equals_Vqf32(out);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1  = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector rms = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, rms_scale_v));
        HVX_Vector out = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(rms, scale_mul_v), bias_v);
        HVX_Vector result = Q6_Vsf_equals_Vqf32(out);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, result);
    }
}

// Four rows at once (in place or not) of hvx_fast_rms_norm_scale_f32_c for num_elems % 32 == 0: every row runs exactly the op
// sequence of the single-row primitive, the four independent latency chains (sum, reduce, rsqrt) are interleaved. Bit-exact.
static inline void hvx_fast_rms_norm_scale_f32_x4(uint8_t * restrict d0, uint8_t * restrict d1, uint8_t * restrict d2, uint8_t * restrict d3,
                                                  const uint8_t * restrict s0, const uint8_t * restrict s1,
                                                  const uint8_t * restrict s2, const uint8_t * restrict s3, const int num_elems,
                                                  HVX_Vector epsilon_v, HVX_Vector denom_v, HVX_Vector scale_mul_v, HVX_Vector bias_v) {
    const HVX_Vector * restrict v0 = (const HVX_Vector *) s0;
    const HVX_Vector * restrict v1 = (const HVX_Vector *) s1;
    const HVX_Vector * restrict v2 = (const HVX_Vector *) s2;
    const HVX_Vector * restrict v3 = (const HVX_Vector *) s3;
    const int nvec = num_elems / VLEN_FP32;

    HVX_Vector a0 = Q6_V_vsplat_R(0x00000000), a1 = a0, a2 = a0, a3 = a0;
    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
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

    HVX_Vector r0 = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a0, denom_v), epsilon_v)));
    HVX_Vector r1 = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a1, denom_v), epsilon_v)));
    HVX_Vector r2 = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a2, denom_v), epsilon_v)));
    HVX_Vector r3 = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a3, denom_v), epsilon_v)));

    HVX_Vector * restrict o0 = (HVX_Vector *) d0;
    HVX_Vector * restrict o1 = (HVX_Vector *) d1;
    HVX_Vector * restrict o2 = (HVX_Vector *) d2;
    HVX_Vector * restrict o3 = (HVX_Vector *) d3;
    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector x0 = v0[i], x1 = v1[i], x2 = v2[i], x3 = v3[i];
        HVX_Vector m0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, r0));
        HVX_Vector m1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x1, r1));
        HVX_Vector m2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, r2));
        HVX_Vector m3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x3, r3));
        o0[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(m0, scale_mul_v), bias_v));
        o1[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(m1, scale_mul_v), bias_v));
        o2[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(m2, scale_mul_v), bias_v));
        o3[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(m3, scale_mul_v), bias_v));
    }
}

#endif // HVX_NORM_H
