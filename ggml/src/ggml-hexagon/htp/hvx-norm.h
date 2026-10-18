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

// y = src0 + src1; dst = rms_norm(y) * weight. y is also written out (it feeds the next layer).
static inline void hvx_fast_add_rms_norm_mul_f32(const uint8_t * restrict src0,
                                                 const uint8_t * restrict src1,
                                                 const uint8_t * restrict weight,
                                                 uint8_t * restrict sum,
                                                 uint8_t * restrict dst,
                                                 const int num_elems,
                                                 float     epsilon) {
    const HVX_Vector * restrict v_src0   = (const HVX_Vector *) src0;
    const HVX_Vector * restrict v_src1   = (const HVX_Vector *) src1;
    const HVX_Vector * restrict v_weight = (const HVX_Vector *) weight;
    HVX_Vector * restrict v_sum          = (HVX_Vector *) sum;
    HVX_Vector * restrict v_dst          = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;
    const int nloe = num_elems % VLEN_FP32;

    HVX_Vector sum_v     = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    // pass 1: add and keep the sum in VTCM, accumulating sum-of-squares on the fly
    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_src0[i], v_src1[i]));
        v_sum[i] = v1;
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(v1, v1));
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_src0[nvec], v_src1[nvec]));
        hvx_vec_store_a(&v_sum[nvec], nloe * 4, v1);
        HVX_Vector v1m = Q6_V_vand_QV(bmask, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(v1m, v1m));
    }

    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);
    HVX_Vector scale_v        = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    // pass 2: normalize the VTCM-resident sum and apply the weight
    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v_sum[i], scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        v_dst[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[i]));
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_sum[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector res_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[nvec]));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

#if __HVX_ARCH__ < 79
#define HVX_NORM_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_NORM_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define HVX_NORM_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_NORM_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

static inline HVX_Vector hvx_rms_scale_f32(HVX_Vector sum_v, int num_elems, float epsilon) {
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));
    HVX_Vector denom_v = hvx_vec_inverse_f32(hvx_vec_splat_f32((float) num_elems));
    HVX_Vector mean_v  = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    return hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(mean_v, hvx_vec_splat_f32(epsilon))));
}

// y = rms_norm(x) * w1 + r, then y *= s (if scale), then dst = rms_norm(y) * w2 (if w2).
// Each step rounds like the standalone RMS_NORM+MUL, ADD and MUL ops.
static inline void hvx_fast_rms_norm_mul_add_f32(const uint8_t * restrict x,
                                                 const uint8_t * restrict r,
                                                 const uint8_t * restrict w1,
                                                 const uint8_t * restrict w2,
                                                 uint8_t * restrict y,
                                                 uint8_t * restrict dst,
                                                 const int num_elems,
                                                 float     eps1,
                                                 float     eps2,
                                                 const float * scale) {
    const HVX_Vector * restrict v_x  = (const HVX_Vector *) x;
    const HVX_Vector * restrict v_r  = (const HVX_Vector *) r;
    const HVX_Vector * restrict v_w1 = (const HVX_Vector *) w1;
    const HVX_Vector * restrict v_w2 = (const HVX_Vector *) w2;
    HVX_Vector * restrict v_y        = (HVX_Vector *) y;
    HVX_Vector * restrict v_dst      = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;
    const int nloe = num_elems % VLEN_FP32;
    const HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);

    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);
    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(v_x[i], v_x[i]));
    }
    if (nloe > 0) {
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_x[nvec]);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(v1, v1));
    }
    const HVX_Vector s1_v = hvx_rms_scale_f32(sum_v, num_elems, eps1);

    if (!w2) {
        // only y is needed: no second sum of squares
        const HVX_Vector sc_v = scale ? hvx_vec_splat_f32(*scale) : Q6_V_vzero();
        if (scale) {
            #pragma unroll(4)
            for (int i = 0; i < nvec; i++) {
                HVX_Vector n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_x[i], s1_v));
                n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(n, v_w1[i]));
                v_y[i] = HVX_NORM_MUL_F32(HVX_NORM_ADD_F32(n, v_r[i]), sc_v);
            }
        } else {
            #pragma unroll(4)
            for (int i = 0; i < nvec; i++) {
                HVX_Vector n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_x[i], s1_v));
                n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(n, v_w1[i]));
                v_y[i] = HVX_NORM_ADD_F32(n, v_r[i]);
            }
        }
        if (nloe > 0) {
            HVX_Vector n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_x[nvec], s1_v));
            n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(n, v_w1[nvec]));
            HVX_Vector v = HVX_NORM_ADD_F32(n, v_r[nvec]);
            if (scale) {
                v = HVX_NORM_MUL_F32(v, sc_v);
            }
            hvx_vec_store_a(&v_y[nvec], nloe * 4, v);
        }
        return;
    }

    sum_v = Q6_V_vsplat_R(0x00000000);
    if (scale) {
        const HVX_Vector sc_v = hvx_vec_splat_f32(*scale);
        #pragma unroll(4)
        for (int i = 0; i < nvec; i++) {
            HVX_Vector n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_x[i], s1_v));
            n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(n, v_w1[i]));
            HVX_Vector v = HVX_NORM_MUL_F32(HVX_NORM_ADD_F32(n, v_r[i]), sc_v);
            v_y[i] = v;
            sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(v, v));
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < nvec; i++) {
            HVX_Vector n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_x[i], s1_v));
            n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(n, v_w1[i]));
            HVX_Vector v = HVX_NORM_ADD_F32(n, v_r[i]);
            v_y[i] = v;
            sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(v, v));
        }
    }
    if (nloe > 0) {
        HVX_Vector n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_x[nvec], s1_v));
        n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(n, v_w1[nvec]));
        HVX_Vector v = HVX_NORM_ADD_F32(n, v_r[nvec]);
        if (scale) {
            v = HVX_NORM_MUL_F32(v, hvx_vec_splat_f32(*scale));
        }
        hvx_vec_store_a(&v_y[nvec], nloe * 4, v);
        v = Q6_V_vand_QV(bmask, v);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(v, v));
    }

    const HVX_Vector s2_v = hvx_rms_scale_f32(sum_v, num_elems, eps2);
    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector n = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_y[i], s2_v));
        v_dst[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(n, v_w2[i]));
    }
    if (nloe > 0) {
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_y[nvec]);
        HVX_Vector n  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, s2_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(n, v_w2[nvec])));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Row-interleaved variants: HVX_NORM_X independent rows go through exactly the intrinsic sequence of the single-row
// kernels above (same per-row accumulation order, same rounding points, so the result is bit-identical). One row is a
// short dependent chain (sum-of-squares accumulate -> horizontal reduce -> inverse -> 3x Newton rsqrt), so a single row
// is latency-bound; X rows in flight overlap those chains. Only rows with num_elems % 32 == 0 take this path; callers
// run the leftover rows (and ragged rows) through the single-row kernels.
#ifndef HVX_NORM_X
#define HVX_NORM_X 4
#endif

#define HVX_NORM_ROW(base, stride, r) ((const HVX_Vector *) ((const uint8_t *) (base) + (size_t) (r) * (stride)))
#define HVX_NORM_ROWW(base, stride, r) ((HVX_Vector *) ((uint8_t *) (base) + (size_t) (r) * (stride)))

// dst[r] = rms_norm(src[r]) * weight, X rows; weight is a single row (wstride == 0) or one row per input row.
static inline void hvx_fast_rms_norm_mul_f32_xn(const uint8_t * src, size_t sstride,
                                                const uint8_t * weight, size_t wstride,
                                                uint8_t * dst, size_t dstride,
                                                const int num_elems, float epsilon) {
    const int nvec = num_elems / VLEN_FP32;
    HVX_Vector sum_v[HVX_NORM_X];
    #pragma unroll
    for (int r = 0; r < HVX_NORM_X; ++r) { sum_v[r] = Q6_V_vsplat_R(0x00000000); }

    for (int i = 0; i < nvec; i++) {
        #pragma unroll
        for (int r = 0; r < HVX_NORM_X; ++r) {
            HVX_Vector v1 = HVX_NORM_ROW(src, sstride, r)[i];
            sum_v[r] = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v[r], Q6_Vqf32_vmpy_VsfVsf(v1, v1));
        }
    }

    const HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);
    const HVX_Vector denom_v   = hvx_vec_inverse_f32(hvx_vec_splat_f32((float) num_elems));
    HVX_Vector scale_v[HVX_NORM_X];
    #pragma unroll
    for (int r = 0; r < HVX_NORM_X; ++r) {
        HVX_Vector s = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v[r]));
        HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(s, denom_v);
        HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);
        scale_v[r] = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));
    }

    for (int i = 0; i < nvec; i++) {
        #pragma unroll
        for (int r = 0; r < HVX_NORM_X; ++r) {
            HVX_Vector v1 = HVX_NORM_ROW(src, sstride, r)[i];
            HVX_Vector v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v[r]));
            HVX_Vector result = Q6_Vqf32_vmpy_VsfVsf(v3, HVX_NORM_ROW(weight, wstride, r)[i]);
            HVX_NORM_ROWW(dst, dstride, r)[i] = Q6_Vsf_equals_Vqf32(result);
        }
    }
}

#undef HVX_NORM_ADD_F32
#undef HVX_NORM_MUL_F32

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

#endif // HVX_NORM_H
