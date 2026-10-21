#ifndef HMX_SWIGLU_TILES_H
#define HMX_SWIGLU_TILES_H

// SWIGLU output epilogue of the fused gate/up HMX matmul (GGML_HEXAGON_SWIGLU_OUT, ImplSwigluOut). 2. silu(g) * u runs the same primitives in the same order as swiglu_f32 + hvx_mul_mul_f32_aa: s = hvx_vec_fast_sigmoid_f32_guard(g); t = HVX_OP_MUL(g, s); o = HVX_OP_MUL(t, u). 3. fp32 -> fp16 is hvx_vec_f32_to_f16_shuff on the same row pair (even row lo, odd row hi), as act_tile_row_pair_store. Output: fp16 activation tile in the htp/hmx-act-tiles.h layout (tile (r0, c0) at r0 * (N/32) + c0), which the FFN down matmul consumes as src1 with HTP_TENSOR_ACT_TILE. Every step is lane-wise (no cross-lane reduction), so grouping elements into vectors does not change any result.
// Bit identity with the unfused path (HMX fp16 out -> fp32 out buffer -> swiglu_f32 -> fp32 -> fp16 act transfer): 1. Input: one (mr, nc) tile pair of the gate and up HMX outputs, fp16, in the HMX output layout: tile (r0, c0) at index r0 * (nc/32) + c0, HVX vector r1 holds row 2*r1 (lo half of Q6_Wqf32_vmpy_VhfVhf) and row 2*r1+1 (hi). HVX_OP_MUL (hvx-sqrt.h) expands to the same expression as hvx-arith.h's HVX_OP_MUL_F32 for every __HVX_ARCH__; the latter is #undef'd at the end of hvx-arith.h, so both products here use HVX_OP_MUL. fp16 -> fp32 is exact: Q6_Wqf32_vmpy_VhfVhf by 1.0 is the conversion transfer_output_chunk_fp16_to_fp32 uses.

#include "hvx-utils.h"
#include "hmx-act-tiles.h"
#include "matmul-ops.h"

// One 32x32 tile pair: g, u (VTCM, HMX output layout) -> o (act tile, DDR). barrier != 0: value barriers on (default);
// barrier == 0 (LxFuseFree, GGML_HEXAGON_LX_FUSE bit 0, HTP_OPFLAGS_SWIGLU_NOBAR): the same math without them.
__attribute__((always_inline))
static inline void swiglu_tile_pair_impl(__fp16 * restrict o, const __fp16 * restrict g, const __fp16 * restrict u, const int barrier) {
    const HVX_Vector one16   = hvx_vec_splat_f16(1.0);
    const HVX_Vector one     = hvx_vec_splat_f32(1.f);
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.f);
    const HVX_Vector min_exp = hvx_vec_splat_f32(-87.f);

    const HVX_Vector * restrict vg = (const HVX_Vector *) g;
    const HVX_Vector * restrict vu = (const HVX_Vector *) u;
    HVX_Vector * restrict vo       = (HVX_Vector *) o;

    _Pragma("unroll(2)")
    for (uint32_t r1 = 0; r1 < HTP_MM_HMX_TILE_N_ROWS / 2; r1++) {
        const HVX_VectorPair gp = Q6_Wqf32_vmpy_VhfVhf(vg[r1], one16);
        const HVX_VectorPair up = Q6_Wqf32_vmpy_VhfVhf(vu[r1], one16);

        const HVX_Vector g0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(gp));
        const HVX_Vector g1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(gp));
        const HVX_Vector u0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(up));
        const HVX_Vector u1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(up));

        const HVX_Vector s0 = hvx_vec_fast_sigmoid_f32_guard(g0, one, max_exp, min_exp);
        const HVX_Vector s1 = hvx_vec_fast_sigmoid_f32_guard(g1, one, max_exp, min_exp);

        HVX_Vector o0 = HVX_OP_MUL(HVX_OP_MUL(g0, s0), u0);
        HVX_Vector o1 = HVX_OP_MUL(HVX_OP_MUL(g1, s1), u1);
        // Value barriers on the fp32 products before the fp16 pack (ImplSwigluOut, impl/SwigluFix.md): under -flto the
        // sf round trip of the packing folds away and changes about 0.04% of fp16 outputs by 1 ulp.
        if (barrier) {
            __asm__ volatile("" : "+v"(o0), "+v"(o1));
        }

        vo[r1] = hvx_vec_f32_to_f16_shuff(o0, o1);
    }
}

// SIG16 (GGML_HEXAGON_SWIGLU_SIG16, numerics-changing, default off): the sigmoid runs once per fp16 vector (64 lanes: rows 2*r1 and 2*r1+1
// of the gate tile) with hvx_vec_fast_sigmoid_f16 and is widened back to the two fp32 lane sets; g * s * u and the fp16 pack stay in fp32.
__attribute__((always_inline))
static inline void swiglu_tile_pair_sig16(__fp16 * restrict o, const __fp16 * restrict g, const __fp16 * restrict u) {
    const HVX_Vector one16 = hvx_vec_splat_f16(1.0);

    const HVX_Vector * restrict vg = (const HVX_Vector *) g;
    const HVX_Vector * restrict vu = (const HVX_Vector *) u;
    HVX_Vector * restrict vo       = (HVX_Vector *) o;

    _Pragma("unroll(2)")
    for (uint32_t r1 = 0; r1 < HTP_MM_HMX_TILE_N_ROWS / 2; r1++) {
        const HVX_VectorPair gp = Q6_Wqf32_vmpy_VhfVhf(vg[r1], one16);
        const HVX_VectorPair up = Q6_Wqf32_vmpy_VhfVhf(vu[r1], one16);

        const HVX_Vector g0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(gp));
        const HVX_Vector g1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(gp));
        const HVX_Vector u0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(up));
        const HVX_Vector u1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(up));

        const HVX_VectorPair sp = hvx_vec_f16_to_f32_shuff(hvx_vec_fast_sigmoid_f16(vg[r1]));
        const HVX_Vector s0 = Q6_V_lo_W(sp);
        const HVX_Vector s1 = Q6_V_hi_W(sp);

        const HVX_Vector o0 = HVX_OP_MUL(HVX_OP_MUL(g0, s0), u0);
        const HVX_Vector o1 = HVX_OP_MUL(HVX_OP_MUL(g1, s1), u1);

        vo[r1] = hvx_vec_f32_to_f16_shuff(o0, o1);
    }
}

// c + b * v in fp16 with the same ops the toolchain emits for Q6_Vhf_vmpyacc_VhfVhfVhf (vmpy.dv, add of qf32(c), convert), but with qf32(c) hoisted.
#define SWG_MAC(cq, b, v) ({ HVX_VectorPair swg_p = Q6_Wqf32_vmpy_VhfVhf((b), (v));     Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(swg_p), (cq)), Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(swg_p), (cq)))); })

// All-fp16 SWIGLU (A8W8 fused up post, SWIGLU_SIG16): silu(g) * u = hf(hf(g * sigmoid_v2(g)) * u) on 64 lanes per vector, no f32 round trip.
// Numerics: two extra fp16 roundings (relative 2^-11 each) compared with the fp32 chain, which rounds once to fp16 at the end. Same op sequence as hvx_vec_fast_sigmoid_f16_v2, with the 13 splat constants hoisted out of the loop and four independent vectors per iteration so the dependent polynomial chain pipelines (the compiler otherwise rematerialises the splats and serialises two chains).
__attribute__((always_inline))
static inline void swiglu_tile_pair_sig16v2(__fp16 * restrict o, const __fp16 * restrict g, const __fp16 * restrict u) {
    const HVX_Vector * restrict vg = (const HVX_Vector *) g;
    const HVX_Vector * restrict vu = (const HVX_Vector *) u;
    HVX_Vector * restrict vo       = (HVX_Vector *) o;
    const HVX_Vector v_one = Q6_Vh_vsplat_R(0x3c00), half = Q6_Vh_vsplat_R(0x3800), absm = Q6_Vh_vsplat_R(0x7FFF);
    const HVX_Vector nlog2e = Q6_Vh_vsplat_R(0xbdc5), clampm = Q6_Vh_vsplat_R(0xca80);
    const HVX_Vector e1 = Q6_Vh_vsplat_R(0x2b0d), s1 = Q6_Vh_vsplat_R(0x30b8);
    // qf32 form of the polynomial constants (what the toolchain's vmpyacc expansion recomputes per use as vmpy(c, 1.0)): lanes of the pair are identical
    const HVX_Vector one16 = Q6_Vh_vsplat_R(0x3c00);
    const HVX_Vector e0 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vsplat_R(0x33c3), one16)), e2 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vsplat_R(0x398c), one16));
    const HVX_Vector e3 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(one16, one16));
    const HVX_Vector s0 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vsplat_R(0xb830), one16)), s2 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vsplat_R(0x3ae1), one16));
    const HVX_Vector s3 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vsplat_R(0xbbde), one16)), s4 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vsplat_R(0x3bff), one16));
    #pragma clang loop unroll(disable)
    for (uint32_t r1 = 0; r1 < HTP_MM_HMX_TILE_N_ROWS / 2; r1 += 4) {
        HVX_Vector x[4], ax[4], k[4], fr[4], y[4], e[4], s[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) { x[i] = vg[r1 + i]; }
        #pragma unroll
        for (int i = 0; i < 4; i++) { ax[i] = Q6_V_vand_VV(x[i], absm); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { y[i] = Q6_Vhf_vmpy_VhfVhf(ax[i], nlog2e); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { y[i] = Q6_Vhf_vmax_VhfVhf(clampm, y[i]); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { k[i] = Q6_Vh_equals_Vhf(Q6_Vhf_vsub_VhfVhf(y[i], half)); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { fr[i] = Q6_Vhf_vsub_VhfVhf(y[i], Q6_Vhf_equals_Vh(k[i])); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { y[i] = SWG_MAC(e0, e1, fr[i]); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { y[i] = SWG_MAC(e2, y[i], fr[i]); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { y[i] = SWG_MAC(e3, y[i], fr[i]); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { e[i] = Q6_Vh_vaslacc_VhVhR(y[i], k[i], 10); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { s[i] = SWG_MAC(s0, s1, e[i]); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { s[i] = SWG_MAC(s2, s[i], e[i]); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { s[i] = SWG_MAC(s3, s[i], e[i]); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { s[i] = SWG_MAC(s4, s[i], e[i]); }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            HVX_VectorPred is_neg = Q6_Q_vcmp_gt_VhVh(ax[i], x[i]);
            s[i] = Q6_V_vmux_QVV(is_neg, Q6_Vhf_vsub_VhfVhf(v_one, s[i]), s[i]);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) { y[i] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(x[i], s[i])); }
        #pragma unroll
        for (int i = 0; i < 4; i++) { vo[r1 + i] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(y[i], vu[r1 + i])); }
    }
}

static inline void swiglu_tile_pair(__fp16 * restrict o, const __fp16 * restrict g, const __fp16 * restrict u) {
    swiglu_tile_pair_impl(o, g, u, 1);
}

// LxFuseFree (GGML_HEXAGON_LX_FUSE bit 0): the same epilogue without the value barriers.
static inline void swiglu_tile_pair_nobar(__fp16 * restrict o, const __fp16 * restrict g, const __fp16 * restrict u) {
    swiglu_tile_pair_impl(o, g, u, 0);
}

#endif /* HMX_SWIGLU_TILES_H */
