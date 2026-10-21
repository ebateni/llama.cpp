#ifndef HVX_SIGMOID_H
#define HVX_SIGMOID_H

#include "hvx-base.h"
#include "hvx-inverse.h"
#include "hvx-exp.h"

#define FAST_SIGMOID_LOG2F (0x3fb8aa3b)  // 1.442695022
#define FAST_SIGMOID_C1    (0x3d009076)  // 0.03138777
#define FAST_SIGMOID_C2    (0x3e8d74bd)  // 0.276281267
#define FAST_SIGMOID_C3    (0x3f000000)  // 0.5

static inline HVX_Vector hvx_vec_fast_sigmoid_f32(HVX_Vector v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_Vector in_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3          = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    HVX_Vector v3_exponent = Q6_Vw_vasl_VwR(v3, 1);
    v3_exponent            = Q6_Vuw_vlsr_VuwR(v3_exponent, 24);
    v3_exponent            = Q6_Vw_vadd_VwVw(in_int, v3_exponent);
    v3                     = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_Vector res = hvx_vec_inverse_f32(v5);
    res            = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    return Q6_Vsf_equals_Vqf32(res);
}

static inline HVX_Vector hvx_vec_fast_sigmoid_f32_guard(HVX_Vector v,
                                                         HVX_Vector one,
                                                         HVX_Vector max_exp,
                                                         HVX_Vector min_exp) {
    const HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    const HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_Vector out = hvx_vec_fast_sigmoid_f32(v);
    out            = Q6_V_vmux_QVV(pred_max, out, one);
    return Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

// C2Elt variant (GGML_HEXAGON_C2_NG=3, GGML_HEXAGON_C2_SILU=2): the guarded fast sigmoid above with two Newton refinements of the reciprocal instead of three. The default guard is unchanged.
// Max relative error of the reciprocal over 1e-6..1e8: 2.0e-5 (two) against 1.5e-7 (three). Not bit-identical; the gate for it is PPL and HOLD-100 (research/C2Elt.md).
static inline HVX_Vector hvx_vec_inverse_f32_2it(HVX_Vector v_sf) {
    HVX_Vector inv_aprox_sf = Q6_V_vsplat_R(0x7EEEEBB3);
    HVX_Vector two_sf       = hvx_vec_splat_f32(2.0);
    HVX_Vector i_sf         = Q6_Vw_vsub_VwVw(inv_aprox_sf, v_sf);
    HVX_Vector r_qf;
    r_qf = Q6_Vqf32_vmpy_VsfVsf(i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v_sf)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));
    return Q6_Vsf_equals_Vqf32(r_qf);
}

static inline HVX_Vector hvx_vec_fast_sigmoid_f32_r2(HVX_Vector v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_Vector in_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_Vector res = hvx_vec_inverse_f32_2it(v5);
    res            = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    return Q6_Vsf_equals_Vqf32(res);
}

static inline HVX_Vector hvx_vec_fast_sigmoid_f32_guard_r2(HVX_Vector v, HVX_Vector one, HVX_Vector max_exp, HVX_Vector min_exp) {
    const HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    const HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_Vector out = hvx_vec_fast_sigmoid_f32_r2(v);
    out            = Q6_V_vmux_QVV(pred_max, out, one);
    return Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

static inline HVX_Vector hvx_vec_tanh_f32(HVX_Vector x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    HVX_Vector two = hvx_vec_splat_f32(2.0f);
    HVX_Vector one = hvx_vec_splat_f32(1.0f);
    HVX_Vector x2  = Q6_Vqf32_vmpy_VsfVsf(x, two);

    HVX_Vector max_exp = hvx_vec_splat_f32(87.f);
    HVX_Vector min_exp = hvx_vec_splat_f32(-87.f);

    HVX_Vector sig2x = hvx_vec_fast_sigmoid_f32_guard(Q6_Vsf_equals_Vqf32(x2), one, max_exp, min_exp);

    HVX_Vector res = Q6_Vqf32_vmpy_VsfVsf(sig2x, two);
    res = Q6_Vqf32_vsub_Vqf32Vsf(res, one);
    return Q6_Vsf_equals_Vqf32(res);
}

#define hvx_sigmoid_loop_body(dst_type, src_type, vec_store)                                  \
    do {                                                                                      \
        dst_type * restrict vdst = (dst_type *) dst;                                          \
        src_type * restrict vsrc = (src_type *) src;                                          \
                                                                                              \
        const HVX_Vector one     = hvx_vec_splat_f32(1.f);                                    \
        const HVX_Vector max_exp = hvx_vec_splat_f32(87.f);                                   \
        const HVX_Vector min_exp = hvx_vec_splat_f32(-87.f);                                  \
                                                                                              \
        const uint32_t epv  = 128 / sizeof(float);                                            \
        const uint32_t nvec = n / epv;                                                        \
        const uint32_t nloe = n % epv;                                                        \
                                                                                              \
        uint32_t i = 0;                                                                       \
                                                                                              \
        _Pragma("unroll(4)")                                                                  \
        for (; i < nvec; i++) {                                                               \
             vdst[i] = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp);        \
        }                                                                                     \
        if (nloe) {                                                                           \
             HVX_Vector tmp = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp); \
             vec_store((void *) &vdst[i], nloe * sizeof(float), tmp);                         \
        }                                                                                     \
    } while(0)

#define hvx_tanh_loop_body(dst_type, src_type, vec_store)             \
    do {                                                              \
        dst_type * restrict vdst = (dst_type *) dst;                  \
        src_type * restrict vsrc = (src_type *) src;                  \
                                                                      \
        const uint32_t epv  = 128 / sizeof(float);                    \
        const uint32_t nvec = n / epv;                                \
        const uint32_t nloe = n % epv;                                \
                                                                      \
        uint32_t i = 0;                                               \
                                                                      \
        _Pragma("unroll(4)")                                          \
        for (; i < nvec; i++) {                                       \
             vdst[i] = hvx_vec_tanh_f32(vsrc[i]);                     \
        }                                                             \
        if (nloe) {                                                   \
             HVX_Vector tmp = hvx_vec_tanh_f32(vsrc[i]);              \
             vec_store((void *) &vdst[i], nloe * sizeof(float), tmp); \
        }                                                             \
    } while(0)

static inline void hvx_sigmoid_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_sigmoid_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_sigmoid_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_sigmoid_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_sigmoid_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_sigmoid_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_tanh_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_tanh_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline HVX_Vector hvx_vec_fast_sigmoid_f16(HVX_Vector x_v) {
    const HVX_Vector v_one       = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_neg_log2e = hvx_vec_splat_f16(-EXP_LOG2E_F);
    const HVX_Vector em_mask     = Q6_Vh_vsplat_R(0x7FFF);

    // Compute absolute value of x_v
    HVX_Vector abs_x = Q6_V_vand_VV(x_v, em_mask);

    // Compute u = -abs_x * log2(e) <= 0.
    HVX_Vector u = hvx_vec_mul_f16_f16(abs_x, v_neg_log2e);

    // Clamp input to prevent underflow in exp2
    const HVX_Vector v_clamp_min = hvx_vec_splat_f16(-24.0f);
    u = Q6_Vhf_vmax_VhfVhf(v_clamp_min, u);

    HVX_Vector exp_val = hvx_vec_exp2_f16(u);
    HVX_Vector denom   = hvx_vec_add_f16_f16(v_one, exp_val);
    HVX_Vector sig_abs = hvx_vec_inverse_f16(denom);

    // check if x_v < 0 (using integer comparison on absolute value)
    HVX_VectorPred is_neg = Q6_Q_vcmp_gt_VhVh(abs_x, x_v);

    // If x_v < 0, return 1.0f - sig_abs
    HVX_Vector sig_neg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_one, sig_abs));
    return Q6_V_vmux_QVV(is_neg, sig_neg, sig_abs);
}

#if __HVX_ARCH__ >= 79
// Accurate fp16 sigmoid (opt5000 conv5, GGML_HEXAGON_SIG16_V2): hvx_vec_fast_sigmoid_f16 goes through hvx_vec_exp2_f16, whose qf16 Horner has a wrong leading coefficient (up to 0.14 absolute error near |x| = 0.35..0.75).
// This version evaluates e = 2^(-|x| log2 e) with a degree-3 fp16 polynomial on the fractional part (no qf16 round trips, fused hf multiply-add) and sigmoid(|x|) = 1 / (1 + e) as a degree-4 polynomial of e in [0, 1]. Max abs error 7.3e-4 over all fp16 inputs |x| <= 17.
static inline HVX_Vector hvx_vec_fast_sigmoid_f16_v2(HVX_Vector x_v) {
    const HVX_Vector v_one   = Q6_Vh_vsplat_R(0x3c00);
    const HVX_Vector half    = Q6_Vh_vsplat_R(0x3800);
    HVX_Vector abs_x = Q6_V_vand_VV(x_v, Q6_Vh_vsplat_R(0x7FFF));
    HVX_Vector u     = Q6_Vhf_vmpy_VhfVhf(abs_x, Q6_Vh_vsplat_R(0xbdc5));    // -|x| log2(e)
    u                = Q6_Vhf_vmax_VhfVhf(Q6_Vh_vsplat_R(0xca80), u);        // >= -13
    HVX_Vector k     = Q6_Vh_equals_Vhf(Q6_Vhf_vsub_VhfVhf(u, half));        // round-to-nearest integer part, as in hvx_vec_exp2_f16 (fr in [-0.5, 0.5))
    HVX_Vector fr    = Q6_Vhf_vsub_VhfVhf(u, Q6_Vhf_equals_Vh(k));
    HVX_Vector y     = Q6_Vhf_vmpyacc_VhfVhfVhf(Q6_Vh_vsplat_R(0x33c3), Q6_Vh_vsplat_R(0x2b0d), fr);
    y                = Q6_Vhf_vmpyacc_VhfVhfVhf(Q6_Vh_vsplat_R(0x398c), y, fr);
    y                = Q6_Vhf_vmpyacc_VhfVhfVhf(v_one, y, fr);
    HVX_Vector e     = Q6_Vh_vaslacc_VhVhR(y, k, 10);                         // y * 2^k
    HVX_Vector s     = Q6_Vhf_vmpyacc_VhfVhfVhf(Q6_Vh_vsplat_R(0xb830), Q6_Vh_vsplat_R(0x30b8), e);
    s                = Q6_Vhf_vmpyacc_VhfVhfVhf(Q6_Vh_vsplat_R(0x3ae1), s, e);
    s                = Q6_Vhf_vmpyacc_VhfVhfVhf(Q6_Vh_vsplat_R(0xbbde), s, e);
    s                = Q6_Vhf_vmpyacc_VhfVhfVhf(Q6_Vh_vsplat_R(0x3bff), s, e);
    HVX_VectorPred is_neg = Q6_Q_vcmp_gt_VhVh(abs_x, x_v);
    return Q6_V_vmux_QVV(is_neg, Q6_Vhf_vsub_VhfVhf(v_one, s), s);
}
#else
static inline HVX_Vector hvx_vec_fast_sigmoid_f16_v2(HVX_Vector x_v) { return hvx_vec_fast_sigmoid_f16(x_v); }
#endif

static inline HVX_Vector hvx_vec_tanh_f16(HVX_Vector x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    const HVX_Vector v_two = hvx_vec_splat_f16(2.0f);

    HVX_Vector x2 = hvx_vec_mul_f16_f16(x, v_two);
    HVX_Vector sig2x = hvx_vec_fast_sigmoid_f16(x2);

    const HVX_Vector v_neg_one = hvx_vec_splat_f16(-1.0f);
    return hvx_vec_add_f16_f16(hvx_vec_mul_f16_f16(sig2x, v_two), v_neg_one);
}

#endif /* HVX_SIGMOID_H */
