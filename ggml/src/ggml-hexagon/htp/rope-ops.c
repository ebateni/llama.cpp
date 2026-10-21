#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "dma-queue.h"
#include "hvx-utils.h"
#include "hvx-norm.h"
#include "hex-fastdiv.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "rope-ops.h"

// Redefined the rope type constants as we can't include ggml.h
#define HTP_ROPE_TYPE_NORMAL 0
#define HTP_ROPE_TYPE_NEOX   2
#define HTP_ROPE_TYPE_MROPE  8
#define HTP_ROPE_TYPE_VISION 24
#define HTP_ROPE_TYPE_IMROPE 40

#define htp_rope_preamble              \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];

struct htp_rope_context {
    int32_t n_dims;
    int32_t n_offs;
    int32_t mode;
    int32_t n_ctx_orig;
    int32_t sections[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    float theta_scale;
    float theta_scale_32;
    float theta_powers[32];
    float corr_dims[2];

    uint32_t src0_nrows_per_thread;

    struct htp_ops_context * octx;

    uint8_t * vtcm_base;
    size_t    spad_per_thread;
    size_t    theta_cache_offset;

    size_t src0_row_size;
    size_t src0_row_stride;
    size_t dst_row_size;
    size_t dst_row_stride;
    size_t src0_row_size_aligned;
    uint32_t src0_nrows;
    uint32_t row_start;
    uint32_t nrows;

    struct fastdiv_values div_ne2_ne1;
    struct fastdiv_values div_ne1;

    const float * freq_factors;

    // ROPE_NX only: fuses a second (src,dst) tensor pair -- e.g. The second pair has its own ne1 (head count) and row size/stride, but the same ne2 (tokens)/ne3 (seqs) as the first pair.
    // K alongside Q -- sharing the same inp_pos/sections/freq params, so mrope_cache_init/rope_cache_init runs once per token instead of once per tensor.
    bool     rope_fast;      // HTP_OPFLAGS_ROPE_FAST
    bool     fuse_norm;      // HTP_OPFLAGS_ROPE_NORM: RMS_NORM+MUL of the A rows (weight src[4]) and B rows (weight src[5]) before the rotation
    float    norm_eps;
    bool     kv;              // HTP_OPFLAGS_ROPE_KV: the K rows (B) are also written to the K cache (dsts[2]) as fp16 at row idx_k[tok], and the V rows (src[7]) to dsts[3] at idx_v[tok]
    size_t   kv_off;          // per-thread spad offset of the KV staging area
    size_t   v_row_bytes;     // V row bytes (fp32)
    size_t   kc_row_bytes;    // K cache row bytes (fp16) == ne1B * ne0 * 2
    size_t   vc_row_bytes;
    size_t   w_bytes_aligned;
    bool     has_pair;
    uint32_t ne1_b;
    size_t   src0b_row_size;
    size_t   src0b_row_stride;
    size_t   dstb_row_size;
    size_t   dstb_row_stride;
    size_t   src0b_row_size_aligned;
};

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);

    return (1 - MIN(1, MAX(0, y)));
}

// Compute one (cos, sin) pair into cache[i0], cache[i0+1] applying YaRN scaling.
static inline void rope_yarn_one(float theta, float freq_scale, float * corr_dims,
                                 uint32_t i0, float ext_factor, float mscale,
                                 float * cache) {
    float theta_extrap = theta;

    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta_final  = theta_interp;
    float mscale_final = mscale;

    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta_final    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale_final  *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    const uint32_t b = i0 / 64;
    const uint32_t k = (i0 % 64) / 2;
    cache[b * 64 + k]      = cosf(theta_final) * mscale_final;
    cache[b * 64 + 32 + k] = sinf(theta_final) * mscale_final;
}

// 32 thetas -> 32 deinterleaved pairs [cos[32] | sin[32]] at cache[i0].
static inline void rope_cache_hvx_32(float * cache, uint32_t i0,
                                     HVX_Vector v_theta,
                                     const float * freq_factors,
                                     HVX_Vector v_freq_scale,
                                     HVX_Vector v_mscale) {
    if (freq_factors) {
        HVX_Vector v_ff = hvx_vmemu(freq_factors + i0 / 2);
        v_theta = hvx_vec_mul_f32_f32(v_theta, hvx_vec_inverse_f32(v_ff));
    }

    HVX_Vector v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);
    HVX_Vector vcos;
    HVX_Vector vsin;
    hvx_vec_sincos_f32(v_theta_final, &vcos, &vsin);
    vcos = hvx_vec_mul_f32_f32(vcos, v_mscale);
    vsin = hvx_vec_mul_f32_f32(vsin, v_mscale);

    if (((uintptr_t) (cache + i0)) % 128 == 0) {
        hvx_vmem(cache + i0 + 0)  = vcos;
        hvx_vmem(cache + i0 + 32) = vsin;
    } else {
        hvx_vec_store_u(cache + i0 + 0,  32 * sizeof(float), vcos);
        hvx_vec_store_u(cache + i0 + 32, 32 * sizeof(float), vsin);
    }
}

static __attribute__((noinline)) void rope_cache_init(const float    theta_base,
                            const float    freq_scale,
                            const float *  freq_factors,
                            float *        corr_dims,
                            const uint32_t n_cache,
                            const float    ext_factor,
                            const float    mscale,
                            float *        cache,
                            const float    theta_scale,
                            const float *  theta_powers,
                            const float    theta_scale_32) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    if (ext_factor == 0.0f) {
        // Fast path: fully vectorized
        // We process 32 pairs (64 elements) per iteration.
        const uint32_t n_blocks = n_cache / 64;

        HVX_Vector v_theta_powers = hvx_vmemu(theta_powers);
        HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_Vector v_mscale = hvx_vec_splat_f32(mscale);

        float theta_block = theta_base;

        for (uint32_t b = 0; b < n_blocks; b++) {
            uint32_t i0 = b * 64;
            HVX_Vector v_theta_base = hvx_vec_splat_f32(theta_block);
            HVX_Vector v_theta = hvx_vec_mul_f32_f32(v_theta_base, v_theta_powers);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            theta_block *= theta_scale_32;
        }

        // Leftovers
        float theta = theta_block;
        for (uint32_t i0 = n_blocks * 64; i0 < n_cache; i0 += 2) {
            const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta *= theta_scale;
        }
    } else {
        float theta = theta_base;
        for (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
            const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta *= theta_scale;
        }
    }
}

static inline float mrope_pick_theta(float theta_t, float theta_h, float theta_w, float theta_e,
                                     int sector, const int32_t sections[4], int sec_w, int sec_e,
                                     bool is_imrope) {
    if (is_imrope) {
        if      (sector % 3 == 0 && sector < 3 * sections[0]) { return theta_t; }
        else if (sector % 3 == 1 && sector < 3 * sections[1]) { return theta_h; }
        else if (sector % 3 == 2 && sector < 3 * sections[2]) { return theta_w; }
        else                                                   { return theta_e; }
    }
    if      (sector < sections[0]) { return theta_t; }
    else if (sector < sec_w)       { return theta_h; }
    else if (sector < sec_e)       { return theta_w; }
    else                           { return theta_e; }
}

// lane j is 1 when (j % 3) == rem
static const float __attribute__((aligned(128))) mrope_mod3_eq0[32] = {
    1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0
};
static const float __attribute__((aligned(128))) mrope_mod3_eq1[32] = {
    0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
};
static const float __attribute__((aligned(128))) mrope_mod3_eq2[32] = {
    0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0
};

static const float __attribute__((aligned(128))) mrope_k_ramp[32] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
};

static inline HVX_VectorPred mrope_mask_eq1(const float * m) {
    return Q6_Q_vcmp_gt_VsfVsf(hvx_vmemu(m), Q6_V_vzero());
}

// IMROPE without wrap: theta[k] = pos[k % 3] * scale^k
static inline HVX_Vector mrope_thetas_imrope_mod3(float pos_t, float pos_h, float pos_w,
                                                  uint32_t k0, HVX_Vector v_powers, float scale_block) {
    const int r = (int) (k0 % 3);
    const float * mt = (r == 0) ? mrope_mod3_eq0 : (r == 1) ? mrope_mod3_eq2 : mrope_mod3_eq1;
    const float * mh = (r == 0) ? mrope_mod3_eq1 : (r == 1) ? mrope_mod3_eq0 : mrope_mod3_eq2;

    HVX_Vector v = hvx_vec_splat_f32(pos_w);
    v = Q6_V_vmux_QVV(mrope_mask_eq1(mh), hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(mrope_mask_eq1(mt), hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    return hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_block));
}

// Contiguous MROPE without wrap: theta[k] = pos[section(k)] * scale^k
static inline HVX_Vector mrope_thetas_contig(float pos_t, float pos_h, float pos_w, float pos_e,
                                             uint32_t k0, int s0, int sec_w, int sec_e,
                                             HVX_Vector v_powers, float scale_block) {
    HVX_Vector v_k = hvx_vec_add_f32_f32(hvx_vec_splat_f32((float) k0), hvx_vmemu(mrope_k_ramp));
    HVX_VectorPred lt_s0 = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((float) s0), v_k);
    HVX_VectorPred lt_sw = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((float) sec_w), v_k);
    HVX_VectorPred lt_se = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((float) sec_e), v_k);

    HVX_Vector v = hvx_vec_splat_f32(pos_e);
    v = Q6_V_vmux_QVV(lt_se, hvx_vec_splat_f32(pos_w), v);
    v = Q6_V_vmux_QVV(lt_sw, hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(lt_s0, hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    return hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_block));
}

// largest pair index k < n_pairs with k % 3 == r lies inside the 3 * sec range of its section (IMROPE interleave)
static inline bool mrope_last_in_section(int n_pairs, int r, int sec) {
    const int kmax = (n_pairs - 1) - ((((n_pairs - 1) - r) % 3) + 3) % 3;
    return kmax < 0 || kmax < 3 * sec;
}

// pos_t/h/w/e: the four position ids for this sequence step (t=time, h=height, w=width, e=extra).
// sections[4]: number of head dims assigned to each position component.
static __attribute__((noinline)) void mrope_cache_init(const float    pos_t,
                             const float    pos_h,
                             const float    pos_w,
                             const float    pos_e,
                             const int32_t  sections[4],
                             const bool     is_imrope,
                             const bool     indep_sects,
                             const float    freq_scale,
                             const float *  freq_factors,
                             float *        corr_dims,
                             const uint32_t n_cache,
                             const float    ext_factor,
                             const float    mscale,
                             float *        cache,
                             const float    theta_scale,
                             const float *  theta_powers,
                             const float    theta_scale_32,
                             const bool     rope_fast) {
    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const int sec_w     = sections[0] + sections[1];
    const int sec_e     = sec_w + sections[2];
    const uint32_t n_pairs = n_cache / 2;

    const bool no_wrap = (sect_dims > 0) && (n_pairs <= (uint32_t) sect_dims);
    // GGML_HEXAGON_ROPE_FAST (numerics-changing rounding of theta only): the mod-3 interleave also covers sections whose 3*n reaches only past the last pair of its residue class (qwen35: 11/11/10, 32 pairs: k = 29 is the last w pair, 3*10 = 30).
    // The generic loop below then runs per pair in scalar code (theta *= scale chains), which dominated ROPE.
    const bool imrope_mod3 = is_imrope && !indep_sects && no_wrap
        && sections[0] > 0 && sections[1] > 0 && sections[2] > 0
        && (rope_fast ? (mrope_last_in_section((int) n_pairs, 0, sections[0]) && mrope_last_in_section((int) n_pairs, 1, sections[1]) && mrope_last_in_section((int) n_pairs, 2, sections[2])) :
            (n_pairs <= (uint32_t) (3 * sections[0]) && n_pairs <= (uint32_t) (3 * sections[1]) && n_pairs <= (uint32_t) (3 * sections[2])));
    const bool contig = !is_imrope && !indep_sects && no_wrap;

    if (ext_factor == 0.0f && (imrope_mod3 || contig)) {
        HVX_Vector v_powers     = hvx_vmemu(theta_powers);
        HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_Vector v_mscale     = hvx_vec_splat_f32(mscale);
        float scale_block = 1.0f;
        const uint32_t n_blocks = n_cache / 64;

        for (uint32_t b = 0; b < n_blocks; b++) {
            const uint32_t i0 = b * 64;
            const uint32_t k0 = b * 32;
            HVX_Vector v_theta = imrope_mod3
                ? mrope_thetas_imrope_mod3(pos_t, pos_h, pos_w, k0, v_powers, scale_block)
                : mrope_thetas_contig(pos_t, pos_h, pos_w, pos_e, k0, sections[0], sec_w, sec_e,
                                      v_powers, scale_block);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            scale_block *= theta_scale_32;
        }

        float theta_k = scale_block;
        for (uint32_t k = n_blocks * 32; k < n_pairs; k++) {
            const uint32_t i0 = 2 * k;
            const float pos = mrope_pick_theta(pos_t, pos_h, pos_w, pos_e,
                                               (int) k, sections, sec_w, sec_e, is_imrope);
            const float ff = freq_factors ? freq_factors[k] : 1.0f;
            rope_yarn_one(pos * theta_k / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta_k *= theta_scale;
        }
        return;
    }

    float theta_t = pos_t;
    float theta_h = pos_h;
    float theta_w = pos_w;
    float theta_e = pos_e;

    const bool use_hvx = (ext_factor == 0.0f);
    float __attribute__((aligned(128))) thetas[32];
    uint32_t n_thetas = 0;
    uint32_t block_i0 = 0;

    HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
    HVX_Vector v_mscale     = hvx_vec_splat_f32(mscale);

    // Same values as before, without the per-pair integer modulo (a library call on Hexagon): sector counts up and wraps at
    // sect_dims, so the loop below costs a few cycles per pair instead of ~80.
    int sector = 0;
    for (uint32_t i0 = 0; i0 < n_cache; i0 += 2, sector = (sector + 1 == sect_dims) ? 0 : sector + 1) {

        if (indep_sects) {
            // Reset theta when crossing into a new section.
            if      (sector == 0)           { theta_t = pos_t; }
            else if (sector == sections[0]) { theta_h = pos_h; }
            else if (sector == sec_w)       { theta_w = pos_w; }
            else if (sector == sec_e)       { theta_e = pos_e; }
        }

        const float theta = mrope_pick_theta(theta_t, theta_h, theta_w, theta_e,
                                             sector, sections, sec_w, sec_e, is_imrope);

        if (use_hvx) {
            if (n_thetas == 0) {
                block_i0 = i0;
            }
            thetas[n_thetas++] = theta;
            if (n_thetas == 32) {
                rope_cache_hvx_32(cache, block_i0, hvx_vmemu(thetas), freq_factors, v_freq_scale, v_mscale);
                n_thetas = 0;
            }
        } else {
            const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
        }

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }

    for (uint32_t k = 0; k < n_thetas; k++) {
        const uint32_t i0 = block_i0 + 2 * k;
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn_one(thetas[k] / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
    }
}

#define M_PI 3.1415926535897932384626433

static void rope_corr_dims(int     n_dims,
                           int     n_ctx_orig,
                           float   freq_base,
                           float   beta_fast,
                           float   beta_slow,
                           float * dims) {
    float start = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (float) M_PI)) / (2 * logf(freq_base)));
    float end   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (float) M_PI)) / (2 * logf(freq_base)));
    dims[0]     = MAX(0, start);
    dims[1]     = MIN(n_dims - 1, end);
}

static inline void hvx_rope_neox_mul(HVX_Vector v0, HVX_Vector v1, HVX_Vector vcos, HVX_Vector vsin,
                                     HVX_Vector * o0, HVX_Vector * o1) {
    HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, vcos);
    HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, vsin);
    HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, vcos);
    HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, vsin);
    *o0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s));
    *o1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c));
}

// theta_cache full 32-pair blocks are deinterleaved [cos | sin].
static inline void hvx_rope_neox_f32_aa(float * restrict dst, const float * restrict src0, uint32_t ne, const float * restrict theta_cache) {
    const uint32_t he = ne / 2;
    const uint32_t nvec = he / 32;
    const uint32_t nloe = he % 32;

    if (nloe == 0) {
        const HVX_Vector * vs = (const HVX_Vector *) src0;
        const HVX_Vector * vt = (const HVX_Vector *) theta_cache;
        HVX_Vector * vd = (HVX_Vector *) dst;
        for (uint32_t i = 0; i < nvec; i++) {
            HVX_Vector o0, o1;
            hvx_rope_neox_mul(vs[i], vs[nvec + i], vt[i * 2 + 0], vt[i * 2 + 1], &o0, &o1);
            vd[i] = o0;
            vd[nvec + i] = o1;
        }
        return;
    }

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector o0, o1;
        hvx_rope_neox_mul(((const HVX_Vector *) src0)[i],
                          hvx_vmemu(src0 + he + i * 32),
                          ((const HVX_Vector *) theta_cache)[i * 2 + 0],
                          ((const HVX_Vector *) theta_cache)[i * 2 + 1],
                          &o0, &o1);
        ((HVX_Vector *) dst)[i] = o0;
        hvx_vmemu(dst + he + i * 32) = o1;
    }

    HVX_Vector v0 = hvx_vmemu(src0 + nvec * 32);
    HVX_Vector v1 = hvx_vmemu(src0 + he + nvec * 32);
    HVX_Vector vcos = hvx_vmemu(theta_cache + nvec * 64);
    HVX_Vector vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);
    HVX_Vector o0, o1;
    hvx_rope_neox_mul(v0, v1, vcos, vsin, &o0, &o1);
    hvx_vec_store_u(dst + nvec * 32, nloe * sizeof(float), o0);
    hvx_vec_store_u(dst + he + nvec * 32, nloe * sizeof(float), o1);
}

static inline void hvx_rope_f32_aa(float * restrict dst, const float * restrict src0, uint32_t ne, const float * restrict theta_cache) {
    const uint32_t nvec = ne / 64;
    const uint32_t nloe = ne % 64;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v0 = ((const HVX_Vector *) src0)[i * 2 + 0];
        HVX_Vector v1 = ((const HVX_Vector *) src0)[i * 2 + 1];

        HVX_Vector vcos = ((const HVX_Vector *) theta_cache)[i * 2 + 0];
        HVX_Vector vsin = ((const HVX_Vector *) theta_cache)[i * 2 + 1];

        HVX_VectorPair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

        ((HVX_Vector *) dst)[i * 2 + 0] = Q6_V_lo_W(vstore);
        ((HVX_Vector *) dst)[i * 2 + 1] = Q6_V_hi_W(vstore);
    }

    if (nloe > 0) {
        if (nloe <= 32) {
            HVX_Vector v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_Vector vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_Vector vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_VectorPair vx0_x1 = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, -4);

            HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            hvx_vec_store_u(dst + nvec * 64, nloe * sizeof(float), Q6_V_lo_W(vstore));
        } else {
            HVX_Vector v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_Vector v1 = hvx_vmemu(src0 + nvec * 64 + 32);

            HVX_Vector vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_Vector vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_VectorPair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

            HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            ((HVX_Vector *) dst)[nvec * 2 + 0] = Q6_V_lo_W(vstore);
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * sizeof(float), Q6_V_hi_W(vstore));
        }
    }
}

static void inline rope_basic_f32_inplace(struct htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, const float * restrict theta_cache) {
    const uint32_t n_offs = rctx->n_offs;
    #pragma unroll(4)
    for (uint32_t i = 0; i < nr; i++) {
        float * s = (float *) (src + i * rctx->src0_row_size_aligned);
        hvx_rope_f32_aa(s + n_offs, s + n_offs, rctx->n_dims, theta_cache);
    }
}

static void inline rope_neox_f32_inplace(struct htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, uint32_t ne, const float * restrict theta_cache) {
    const uint32_t n_offs = rctx->n_offs;
    #pragma unroll(4)
    for (uint32_t i = 0; i < nr; i++) {
        float * s = (float *) (src + i * rctx->src0_row_size_aligned);
        hvx_rope_neox_f32_aa(s + n_offs, s + n_offs, ne, theta_cache);
    }
}

static void rope_job_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_rope_context * rctx = (struct htp_rope_context *) data;
    struct htp_ops_context * octx = rctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * src2 = octx->src[2];
    const struct htp_tensor * dst  = octx->dst;

    htp_rope_preamble;

    const uint32_t src0_nrows = rctx->nrows;
    const uint32_t src0_nrows_per_thread = rctx->src0_nrows_per_thread;

    const uint32_t src0_start_row = rctx->row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, rctx->row_start + src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    const int32_t mode    = rctx->mode;
    // MROPE, IMROPE and VISION use NEOX-style pairing for the rotation
    const bool    is_neox = (mode & HTP_ROPE_TYPE_NEOX) || (mode & HTP_ROPE_TYPE_MROPE);
    const bool    is_vision = (mode == HTP_ROPE_TYPE_VISION);

    // VTCM setup
    uint8_t * src0_spad_base = rctx->vtcm_base + (ith * rctx->spad_per_thread);
    float *   theta_cache    = (float *) (src0_spad_base);
              src0_spad_base = src0_spad_base + rctx->theta_cache_offset;

    dma_queue * dma_q = octx->ctx->dma[ith];
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    const int32_t * pos = (const int32_t *) (uintptr_t) src1->data;
    const float * freq_factors = rctx->freq_factors;

    const uint32_t i3_start = fastdiv(src0_start_row, &rctx->div_ne2_ne1);
    const uint32_t rem      = fastmodulo(src0_start_row, ne2 * ne1, &rctx->div_ne2_ne1);
    const uint32_t i2_start = fastdiv(rem, &rctx->div_ne1);
    const uint32_t i1_start = fastmodulo(rem, ne1, &rctx->div_ne1);

    uint32_t ir = src0_start_row;
    uint32_t prev_i2 = (uint32_t) -1;
    uint32_t cur_slot = 0;

    for (uint32_t i3 = i3_start; i3 < ne3; i3++) { // batch
        const uint32_t i2_init = (i3 == i3_start) ? i2_start : 0;
        for (uint32_t i2 = i2_init; i2 < ne2; i2++) { // seq-len
            const uint32_t i1_init = (i3 == i3_start && i2 == i2_start) ? i1_start : 0;
            for (uint32_t i1 = i1_init; i1 < ne1; ) { // attn-heads
                if (ir >= src0_end_row) goto done;

                // Rows in this block
                const uint32_t nrows = MIN(src0_end_row - ir, ne1 - i1);

                // Depth before prefetch
                const uint32_t dma_depth = dma_queue_depth(dma_q);

                // Prefetch up to 2 blocks
                const uint32_t p_nrows = MIN(nrows, 2 * HTP_ROPE_SPAD_BLOCK);
                for (uint32_t pr = 0; pr < p_nrows; pr += HTP_ROPE_SPAD_BLOCK) {
                    const uint32_t pnr = MIN(nrows - pr, HTP_ROPE_SPAD_BLOCK);
                    const uint32_t slot = (cur_slot + pr / HTP_ROPE_SPAD_BLOCK) % HTP_ROPE_SPAD_NSLOTS;
                    uint8_t * spad_slot = rope_spad_slot(src0_spad_base, slot, rctx->src0_row_size_aligned);
                    const dma_addr_t src0_data = src0->data + i3 * nb03 + i2 * nb02 + (i1 + pr) * nb01;

                    // Dummy DMA transaction for sequencing (interleaving wr, rd, wr, rd, ...)
                    dma_queue_push(dma_q, dma_make_data(dst->data, spad_slot), 0, 0, 0, 0);

                    dma_queue_push(dma_q, dma_make_data(spad_slot, src0_data),
                        rctx->src0_row_size_aligned, rctx->src0_row_stride, rctx->src0_row_size, pnr);
                }

                // Update theta cache
                if (i2 != prev_i2) {
                    prev_i2 = i2;

                    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                    // VISION rotates the full row; other modes only rotate n_dims.
                    const uint32_t n_cache = is_vision ? ne0 : (uint32_t) rctx->n_dims;
                    const bool is_mrope = (rctx->mode & HTP_ROPE_TYPE_MROPE) != 0;
                    if (is_mrope) {
                        // src1 holds four position arrays stacked along ne0:
                        // pos[i2], pos[i2+ne2], pos[i2+ne2*2], pos[i2+ne2*3]
                        const bool is_imrope = (rctx->mode == HTP_ROPE_TYPE_IMROPE);
                        mrope_cache_init(
                            (float) pos[i2],
                            (float) pos[i2 + ne2],
                            (float) pos[i2 + ne2 * 2],
                            (float) pos[i2 + ne2 * 3],
                            rctx->sections, is_imrope, is_vision,
                            rctx->freq_scale, freq_factors, rctx->corr_dims,
                            n_cache, rctx->ext_factor, rctx->attn_factor,
                            theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32, rctx->rope_fast);
                    } else {
                       rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                                        n_cache, rctx->ext_factor, rctx->attn_factor,
                                        theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                }

                // Skip output DMA transactions from prev block (if any)
                for (uint32_t d = 0; d < dma_depth; d++) { dma_queue_pop_nowait(dma_q); }

                // Compute loop
                const uint32_t ne = is_vision ? ne0 : rctx->n_dims;
                const uint32_t base_i1 = i1;
                const uint32_t base_ir = ir;

                for (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr) {
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_BLOCK);
                    const uint32_t slot = (cur_slot + cr / HTP_ROPE_SPAD_BLOCK) % HTP_ROPE_SPAD_NSLOTS;
                    const uint32_t cur_ir   = base_ir + cr;
                    const uint32_t cur_i1   = base_i1 + cr;

                    dma_queue_pop(dma_q);
                    uint8_t * cur_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

                    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);
                    if (is_neox || is_vision) {
                        rope_neox_f32_inplace(rctx, cur_spad, cnr, ne, theta_cache);
                    } else {
                        rope_basic_f32_inplace(rctx, cur_spad, cnr, theta_cache);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);

                    const dma_addr_t dst_data = dst->data + i3 * nb3 + i2 * nb2 + cur_i1 * nb1;
                    dma_queue_push(dma_q, dma_make_data(dst_data, cur_spad),
                        rctx->dst_row_stride, rctx->src0_row_size_aligned, rctx->dst_row_size, cnr);

                    // Prefetch 2 blocks ahead into the slot just freed
                    if ((cr + 2 * HTP_ROPE_SPAD_BLOCK) < nrows) {
                        const uint32_t p_cr   = cr + 2 * HTP_ROPE_SPAD_BLOCK;
                        const uint32_t pnr    = MIN(nrows - p_cr, HTP_ROPE_SPAD_BLOCK);
                        const uint32_t p_slot = (cur_slot + p_cr / HTP_ROPE_SPAD_BLOCK) % HTP_ROPE_SPAD_NSLOTS;
                        uint8_t * p_spad      = rope_spad_slot(src0_spad_base, p_slot, rctx->src0_row_size_aligned);
                        const dma_addr_t p_src0_data = src0->data + i3 * nb03 + i2 * nb02 + (base_i1 + p_cr) * nb01;

                        dma_queue_push(dma_q, dma_make_data(p_spad, p_src0_data),
                            rctx->src0_row_size_aligned, rctx->src0_row_stride, rctx->src0_row_size, pnr);
                    }
                }

                const uint32_t n_chunks = (nrows + HTP_ROPE_SPAD_BLOCK - 1) / HTP_ROPE_SPAD_BLOCK;
                cur_slot = (cur_slot + n_chunks) % HTP_ROPE_SPAD_NSLOTS;

                ir += nrows;
                i1 += nrows;
            }
        }
    }

done:
    dma_queue_flush(dma_q);

    FARF(HIGH, "rope-f32: %d/%d: (%u:%u)\n", ith, nth, src0_start_row, src0_end_row);
}

static int execute_op_rope_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * src2 = octx->src[2];
    const struct htp_tensor * dst  = octx->dst;

    switch (octx->op) {
        case HTP_OP_ROPE:
            break;

        default:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_rope_kernel_params * kparams = (const struct htp_rope_kernel_params *) octx->kernel_params;
    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    assert(octx->ctx->vtcm_size >= kparams->vtcm_size);

    if (htp_tensor_is_extended(src1)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const size_t dst_data_row_size = dst->ne[0] * sizeof(float);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, sizeof(float), (uint32_t) dst_data_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = octx->n_threads;

    const uint32_t ne0 = dst->ne[0];
    const size_t src0_row_size   = src0->ne[0] * sizeof(float);
    const size_t src0_row_stride = src0->nb[1];
    const size_t dst_row_size    = dst->ne[0] * sizeof(float);
    const size_t dst_row_stride  = dst->nb[1];

    struct htp_rope_context rctx;
    memset(&rctx, 0, sizeof(struct htp_rope_context));

    rctx.octx                  = octx;
    rctx.vtcm_base             = (uint8_t *) octx->ctx->vtcm_base;
    rctx.spad_per_thread       = kparams->spad_per_thread;
    rctx.theta_cache_offset    = kparams->theta_cache_offset;

    if (src2) {
        dma_queue * dma_q = octx->ctx->dma[0];
        const size_t ff_size = src2->ne[0] * sizeof(float);
        float * vtcm_freq_factors = (float *) (rctx.vtcm_base + kparams->freq_factors_offset);
        dma_queue_push(dma_q, dma_make_data(vtcm_freq_factors, src2->data),
                       kparams->freq_factors_size, 0, ff_size, 1);
        dma_queue_pop(dma_q);
        rctx.freq_factors = vtcm_freq_factors;
    }

    const int32_t * op_params = &octx->op_params[0];
    rctx.rope_fast  = (octx->flags & HTP_OPFLAGS_ROPE_FAST) != 0;
    rctx.n_dims     = ((const int32_t *) op_params)[1];
    rctx.mode       = ((const int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((const int32_t *) op_params)[4];
    rctx.n_offs     = ((const int32_t *) op_params)[15];

    memcpy(&rctx.freq_base,   (int32_t *) op_params + 5,  sizeof(float));
    memcpy(&rctx.freq_scale,  (int32_t *) op_params + 6,  sizeof(float));
    memcpy(&rctx.ext_factor,  (int32_t *) op_params + 7,  sizeof(float));
    memcpy(&rctx.attn_factor, (int32_t *) op_params + 8,  sizeof(float));
    memcpy(&rctx.beta_fast,   (int32_t *) op_params + 9,  sizeof(float));
    memcpy(&rctx.beta_slow,   (int32_t *) op_params + 10, sizeof(float));
    memcpy(&rctx.sections,    (int32_t *) op_params + 11, sizeof(int) * 4);

    rctx.theta_scale = powf(rctx.freq_base, -2.0f / rctx.n_dims);
    rctx.theta_powers[0] = 1.0f;
    for (int j = 1; j < 32; j++) {
        rctx.theta_powers[j] = rctx.theta_powers[j - 1] * rctx.theta_scale;
    }
    rctx.theta_scale_32 = rctx.theta_powers[31] * rctx.theta_scale;

    rope_corr_dims(rctx.n_dims, rctx.n_ctx_orig, rctx.freq_base, rctx.beta_fast, rctx.beta_slow, rctx.corr_dims);

    rctx.src0_row_size         = src0_row_size;
    rctx.src0_row_stride       = src0_row_stride;
    rctx.dst_row_size          = dst_row_size;
    rctx.dst_row_stride        = dst_row_stride;
    rctx.src0_row_size_aligned = kparams->src0_row_size_aligned;

    rctx.src0_nrows            = nrows;
    rctx.nrows                 = nrows;
    rctx.row_start             = row_start;
    rctx.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    rctx.div_ne2_ne1           = kparams->div_ne2_ne1;
    rctx.div_ne1               = kparams->div_ne1;

    FARF(HIGH, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_factor, rctx.theta_scale, rctx.attn_factor);

    work_queue_run(octx->ctx->work_queue, rope_job_f32, &rctx, n_threads);

    return err;
}

// ROPE_NX: fuses two independent ROPE calls (e.g. This sacrifices some of rope_job_f32's DMA/compute overlap in exchange for a data-flow simple enough to reason about without the class of cross-thread VTCM-visibility risk that a literal port of a deeper pipeline would introduce for a first implementation of a 2-tensor fused kernel. Deliberately simpler than rope_job_f32's cross-token double-buffered pipeline: this kernel partitions work by TOKEN (not by row) across threads, and each thread processes its tokens with straightforward blocking DMA (fetch A's rows, fetch B's rows, compute theta_cache once, rotate A's rows, rotate B's rows, write back) -- no cross-tensor VTCM slot rotation, no multi-token-ahead prefetch depth.
// K); both share ne2 (tokens)/ne3 (seqs), only ne1 (head count) may differ. Q, K) that share the same inp_pos/ freq_factors/op_params (enforced host-side by is_mergeable_rope_pair) into one dispatch, computing theta_cache once per token instead of once per tensor. Tensor A is src[0]/ dsts[0] (e.g. Q), tensor B is src[3]/dsts[1] (e.g.
static void rope_nx_job_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_rope_context * rctx = (struct htp_rope_context *) data;
    struct htp_ops_context * octx = rctx->octx;

    const struct htp_tensor * srcA = octx->src[0];
    const struct htp_tensor * srcB = octx->src[3];
    const struct htp_tensor * dstA = octx->dsts[0];
    const struct htp_tensor * dstB = octx->dsts[1];

    const uint32_t ne1A = dstA->ne[1];
    const uint32_t ne1B = dstB->ne[1];
    const uint32_t ne2  = dstA->ne[2];
    const uint32_t ne3  = dstA->ne[3];
    const uint32_t n_tokens = ne2 * ne3;

    const uint32_t tokens_per_thread = rctx->src0_nrows_per_thread; // repurposed as token count/thread
    const uint32_t tok_start = tokens_per_thread * ith;
    const uint32_t tok_end   = MIN(tok_start + tokens_per_thread, n_tokens);
    if (tok_start >= tok_end) {
        return;
    }

    const int32_t mode      = rctx->mode;
    const bool    is_neox   = (mode & HTP_ROPE_TYPE_NEOX) || (mode & HTP_ROPE_TYPE_MROPE);
    const bool    is_vision = (mode == HTP_ROPE_TYPE_VISION);
    const bool    is_mrope  = (mode & HTP_ROPE_TYPE_MROPE) != 0;
    const bool    is_imrope = (mode == HTP_ROPE_TYPE_IMROPE);

    uint8_t * spad_base = rctx->vtcm_base + (ith * rctx->spad_per_thread);
    float *   theta_cache = (float *) spad_base;
    // two token buffers (A rows then B rows each): token t uses buffer t & 1, so the loads of token t + 1 overlap the compute of t
    const size_t tok_bytes = (size_t) ne1A * rctx->src0_row_size_aligned + (size_t) ne1B * rctx->src0b_row_size_aligned;
    uint8_t * a_spad_[2] = { spad_base + rctx->theta_cache_offset, spad_base + rctx->theta_cache_offset + tok_bytes };
    uint8_t * b_spad_[2] = { a_spad_[0] + ne1A * rctx->src0_row_size_aligned, a_spad_[1] + ne1A * rctx->src0_row_size_aligned };

    dma_queue * dma_q = octx->ctx->dma[ith];
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    const struct htp_tensor * srcPos = octx->src[1];
    const int32_t * pos = (const int32_t *) (uintptr_t) srcPos->data;
    const float * freq_factors = rctx->freq_factors;

    const uint32_t n_cache = is_vision ? dstA->ne[0] : (uint32_t) rctx->n_dims;

    #define ROPE_NX_PUSH_LOAD(tok_)                                                                                  \
        do {                                                                                                         \
            const uint32_t i3_ = (tok_) / ne2, i2_ = (tok_) % ne2;                                                   \
            dma_queue_push(dma_q, dma_make_data(a_spad_[(tok_) & 1], srcA->data + i3_ * srcA->nb[3] + i2_ * srcA->nb[2]), \
                rctx->src0_row_size_aligned, rctx->src0_row_stride, rctx->src0_row_size, ne1A);                      \
            dma_queue_push(dma_q, dma_make_data(b_spad_[(tok_) & 1], srcB->data + i3_ * srcB->nb[3] + i2_ * srcB->nb[2]), \
                rctx->src0b_row_size_aligned, rctx->src0b_row_stride, rctx->src0b_row_size, ne1B);                   \
            if (rctx->kv) {                                                                                          \
                dma_queue_push(dma_q, dma_make_data(v_in_[(tok_) & 1], kv_srcV->data + i3_ * kv_srcV->nb[3] + i2_ * kv_srcV->nb[1]), \
                    rctx->v_row_bytes, rctx->v_row_bytes, rctx->v_row_bytes, 1);                                     \
            }                                                                                                        \
        } while (0)

    // KV cache write (ROPE_KV): V rows are loaded per token next to A/B, K/V are converted to fp16 and DMA'd to their cache rows
    const struct htp_tensor * kv_srcV = rctx->kv ? octx->src[7] : NULL;
    const struct htp_tensor * kv_dstK = rctx->kv ? octx->dsts[2] : NULL;
    const struct htp_tensor * kv_dstV = rctx->kv ? octx->dsts[3] : NULL;
    const int64_t * kv_idxK = rctx->kv ? (const int64_t *) (uintptr_t) octx->src[6]->data : NULL;
    const int64_t * kv_idxV = rctx->kv ? (const int64_t *) (uintptr_t) octx->src[8]->data : NULL;
    uint8_t * v_in_[2]  = { NULL, NULL };
    uint8_t * kc_out_[2] = { NULL, NULL };
    uint8_t * vc_out_[2] = { NULL, NULL };
    if (rctx->kv) {
        uint8_t * kvb = spad_base + rctx->kv_off;
        const size_t vb = hex_round_up((uint32_t) rctx->v_row_bytes, 128);
        const size_t kb = hex_round_up((uint32_t) rctx->kc_row_bytes, 128);
        const size_t cb = hex_round_up((uint32_t) rctx->vc_row_bytes, 128);
        v_in_[0] = kvb;            v_in_[1] = kvb + vb;
        kc_out_[0] = kvb + 2 * vb; kc_out_[1] = kc_out_[0] + kb;
        vc_out_[0] = kc_out_[1] + kb; vc_out_[1] = vc_out_[0] + cb;
    }
    const uint32_t kv_ne1_k = rctx->kv ? (uint32_t) kv_dstK->ne[1] : 0;
    const uint32_t kv_ne1_v = rctx->kv ? (uint32_t) kv_dstV->ne[1] : 0;

    uint8_t * wq_spad = NULL;
    uint8_t * wk_spad = NULL;
    if (rctx->fuse_norm) {
        const struct htp_tensor * wq = octx->src[4];
        const struct htp_tensor * wk = octx->src[5];
        wq_spad = spad_base + rctx->theta_cache_offset + 2 * tok_bytes;
        wk_spad = wq_spad + rctx->w_bytes_aligned;
        dma_queue_push(dma_q, dma_make_data(wq_spad, wq->data), rctx->w_bytes_aligned, 0, wq->ne[0] * sizeof(float), 1);
        dma_queue_push(dma_q, dma_make_data(wk_spad, wk->data), rctx->w_bytes_aligned, 0, wk->ne[0] * sizeof(float), 1);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
    }

    ROPE_NX_PUSH_LOAD(tok_start);
    for (uint32_t tok = tok_start; tok < tok_end; tok++) {
        const uint32_t i3 = tok / ne2;
        const uint32_t i2 = tok % ne2;
        const uint32_t cur = tok & 1;

        // Compute theta_cache once for this token, shared by both A and B (overlaps the token's load DMA).
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
        if (is_mrope) {
            mrope_cache_init(
                (float) pos[i2], (float) pos[i2 + ne2], (float) pos[i2 + ne2 * 2], (float) pos[i2 + ne2 * 3],
                rctx->sections, is_imrope, is_vision,
                rctx->freq_scale, freq_factors, rctx->corr_dims,
                n_cache, rctx->ext_factor, rctx->attn_factor,
                theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32, rctx->rope_fast);
        } else {
            rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                             n_cache, rctx->ext_factor, rctx->attn_factor,
                             theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);

        dma_queue_pop(dma_q);   // A rows of tok
        dma_queue_pop(dma_q);   // B rows of tok
        if (rctx->kv) {
            dma_queue_pop(dma_q);   // V row of tok
        }
        if (tok > tok_start) {
            dma_queue_pop(dma_q);   // A store of tok - 1 (its buffer is refilled below)
            dma_queue_pop(dma_q);   // B store of tok - 1
            if (rctx->kv) {
                dma_queue_pop(dma_q);   // K cache store of tok - 1
                dma_queue_pop(dma_q);   // V cache store of tok - 1
            }
        }
        if (tok + 1 < tok_end) {
            ROPE_NX_PUSH_LOAD(tok + 1);
        }

        // Rotate A's and B's rows in place, using the shared theta_cache.
        uint8_t * a_spad = a_spad_[cur];
        uint8_t * b_spad = b_spad_[cur];
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, tok);
        if (rctx->fuse_norm) {
            // the fused-away RMS_NORM+MUL ops: the same primitive on the same rows, so the rotation sees identical inputs
            hvx_fast_rms_norm_mul_f32_rows(a_spad, rctx->src0_row_size_aligned, wq_spad, a_spad, rctx->src0_row_size_aligned, ne1A, (int) srcA->ne[0], rctx->norm_eps);
            hvx_fast_rms_norm_mul_f32_rows(b_spad, rctx->src0b_row_size_aligned, wk_spad, b_spad, rctx->src0b_row_size_aligned, ne1B, (int) srcB->ne[0], rctx->norm_eps);
        }
        if (is_neox || is_vision) {
            rope_neox_f32_inplace(rctx, a_spad, ne1A, is_vision ? dstA->ne[0] : rctx->n_dims, theta_cache);
            rope_neox_f32_inplace(rctx, b_spad, ne1B, is_vision ? dstB->ne[0] : rctx->n_dims, theta_cache);
        } else {
            rope_basic_f32_inplace(rctx, a_spad, ne1A, theta_cache);
            rope_basic_f32_inplace(rctx, b_spad, ne1B, theta_cache);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, tok);

        dma_queue_push(dma_q, dma_make_data(dstA->data + i3 * dstA->nb[3] + i2 * dstA->nb[2], a_spad),
            rctx->dst_row_stride, rctx->src0_row_size_aligned, rctx->dst_row_size, ne1A);
        dma_queue_push(dma_q, dma_make_data(dstB->data + i3 * dstB->nb[3] + i2 * dstB->nb[2], b_spad),
            rctx->dstb_row_stride, rctx->src0b_row_size_aligned, rctx->dstb_row_size, ne1B);
        if (rctx->kv) {
            // the same conversion as SET_ROWS (hvx_copy_f16_f32_uu) on the rotated K rows and the V row, then the scatter by index
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, tok);
            hvx_copy_f16_f32_uu(kc_out_[cur], b_spad, (uint32_t) (ne1B * (rctx->dstb_row_size / sizeof(float))));
            hvx_copy_f16_f32_uu(vc_out_[cur], v_in_[cur], (uint32_t) (rctx->v_row_bytes / sizeof(float)));
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, tok);
            const uint64_t ik = (uint64_t) kv_idxK[i2];
            const uint64_t iv = (uint64_t) kv_idxV[i2];
            dma_queue_push(dma_q, dma_make_data(kv_dstK->data + (ik < kv_ne1_k ? ik : 0) * kv_dstK->nb[1], kc_out_[cur]),
                rctx->kc_row_bytes, rctx->kc_row_bytes, rctx->kc_row_bytes, ik < kv_ne1_k ? 1 : 0);
            dma_queue_push(dma_q, dma_make_data(kv_dstV->data + (iv < kv_ne1_v ? iv : 0) * kv_dstV->nb[1], vc_out_[cur]),
                rctx->vc_row_bytes, rctx->vc_row_bytes, rctx->vc_row_bytes, iv < kv_ne1_v ? 1 : 0);
        }
    }
    #undef ROPE_NX_PUSH_LOAD

    dma_queue_flush(dma_q);

    FARF(HIGH, "rope-nx-f32: %d/%d: tok (%u:%u)\n", ith, nth, tok_start, tok_end);
}

static int execute_op_rope_nx_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * srcA = octx->src[0];
    const struct htp_tensor * srcB = octx->src[3];
    const struct htp_tensor * dstA = octx->dsts[0];
    const struct htp_tensor * dstB = octx->dsts[1];

    if (!srcA || !srcB || !dstA || !dstB) {
        FARF(ERROR, "rope-nx-f32: missing required src/dst tensor\n");
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t ne1A = dstA->ne[1];
    const uint32_t ne1B = dstB->ne[1];
    const uint32_t n_tokens = dstA->ne[2] * dstA->ne[3];

    if (ne1A > HTP_ROPE_SPAD_NROWS || ne1B > HTP_ROPE_SPAD_NROWS) {
        FARF(ERROR, "rope-nx-f32: head count (%u,%u) exceeds per-token spad capacity %u\n",
             ne1A, ne1B, (unsigned) HTP_ROPE_SPAD_NROWS);
        return HTP_STATUS_NO_SUPPORT;
    }
    if (n_tokens == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = MIN(octx->n_threads, n_tokens);

    const size_t srcA_row_size   = srcA->ne[0] * sizeof(float);
    const size_t srcA_row_stride = srcA->nb[1];
    const size_t dstA_row_size   = dstA->ne[0] * sizeof(float);
    const size_t dstA_row_stride = dstA->nb[1];

    const size_t srcB_row_size   = srcB->ne[0] * sizeof(float);
    const size_t srcB_row_stride = srcB->nb[1];
    const size_t dstB_row_size   = dstB->ne[0] * sizeof(float);
    const size_t dstB_row_stride = dstB->nb[1];

    const size_t srcA_row_size_aligned = hex_round_up((uint32_t) srcA_row_size, 128);
    const size_t srcB_row_size_aligned = hex_round_up((uint32_t) srcB_row_size, 128);
    const size_t theta_cache_size_aligned = hex_round_up((uint32_t) (dstA->ne[0] * sizeof(float)), 256);

    // Per-thread spad: theta_cache (shared) + A's ne1A rows + B's ne1B rows. Single-
    // buffered by design (see rope_nx_job_f32's own comment) -- no ×NSLOTS multiplier.
    const bool fuse_norm = (octx->flags & HTP_OPFLAGS_ROPE_NORM) != 0;
    if (fuse_norm && (!octx->src[4] || !octx->src[5] || srcA->ne[0] != srcB->ne[0])) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const size_t w_bytes_aligned = fuse_norm ? hex_round_up((uint32_t) (srcA->ne[0] * sizeof(float)), 128) : 0;
    const bool kv = (octx->flags & HTP_OPFLAGS_ROPE_KV) != 0;
    size_t kv_bytes = 0, v_row_bytes = 0, kc_row_bytes = 0, vc_row_bytes = 0;
    if (kv) {
        const struct htp_tensor * sv = octx->src[7];
        const struct htp_tensor * dk = octx->dsts[2];
        const struct htp_tensor * dv = octx->dsts[3];
        if (!sv || !dk || !dv || !octx->src[6] || !octx->src[8] || sv->type != HTP_TYPE_F32 || dk->type != HTP_TYPE_F16 || dv->type != HTP_TYPE_F16 ||
            dstB->nb[1] != dstB->ne[0] * sizeof(float) || srcB_row_size != srcB_row_size_aligned || (dstB->ne[0] * ne1B) % 32 != 0 ||
            dk->ne[0] != dstB->ne[0] * ne1B || dv->ne[0] != sv->ne[0] || sv->ne[2] != 1 || sv->ne[3] != 1 || sv->ne[1] != dstA->ne[2] ||
            dstA->ne[3] != 1 || octx->src[6]->type != HTP_TYPE_I64 || octx->src[8]->type != HTP_TYPE_I64 || sv->ne[0] % 32 != 0) {
            return HTP_STATUS_NO_SUPPORT;
        }
        v_row_bytes  = sv->ne[0] * sizeof(float);
        kc_row_bytes = dk->ne[0] * sizeof(__fp16);
        vc_row_bytes = dv->ne[0] * sizeof(__fp16);
        kv_bytes     = 2 * hex_round_up((uint32_t) v_row_bytes, 128) + 2 * hex_round_up((uint32_t) kc_row_bytes, 128) + 2 * hex_round_up((uint32_t) vc_row_bytes, 128);
    }
    const size_t spad_per_thread = theta_cache_size_aligned
        + 2 * (ne1A * srcA_row_size_aligned
        + ne1B * srcB_row_size_aligned) + 2 * w_bytes_aligned + kv_bytes;

    const size_t total_vtcm_needed = spad_per_thread * n_threads;
    if (octx->ctx->vtcm_size < total_vtcm_needed) {
        FARF(ERROR, "rope-nx-f32: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, total_vtcm_needed);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    struct htp_rope_context rctx;
    memset(&rctx, 0, sizeof(struct htp_rope_context));

    rctx.octx      = octx;
    rctx.vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    const struct htp_tensor * src2 = octx->src[2];
    if (src2) {
        dma_queue * dma_q0 = octx->ctx->dma[0];
        const size_t ff_size = src2->ne[0] * sizeof(float);
        const size_t ff_size_aligned = hex_round_up((uint32_t) ff_size, 256);
        // Freq-factors staging lives right after every thread's own spad region.
        float * vtcm_ff = (float *) (rctx.vtcm_base + spad_per_thread * n_threads);
        dma_queue_push(dma_q0, dma_make_data(vtcm_ff, src2->data), ff_size_aligned, 0, ff_size, 1);
        dma_queue_pop(dma_q0);
        rctx.freq_factors = vtcm_ff;
    }

    const int32_t * op_params = &octx->op_params[0];
    rctx.rope_fast  = (octx->flags & HTP_OPFLAGS_ROPE_FAST) != 0;
    rctx.n_dims     = ((const int32_t *) op_params)[1];
    rctx.mode       = ((const int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((const int32_t *) op_params)[4];
    rctx.n_offs     = ((const int32_t *) op_params)[15];

    memcpy(&rctx.freq_base,   (int32_t *) op_params + 5,  sizeof(float));
    memcpy(&rctx.freq_scale,  (int32_t *) op_params + 6,  sizeof(float));
    memcpy(&rctx.ext_factor,  (int32_t *) op_params + 7,  sizeof(float));
    memcpy(&rctx.attn_factor, (int32_t *) op_params + 8,  sizeof(float));
    memcpy(&rctx.beta_fast,   (int32_t *) op_params + 9,  sizeof(float));
    memcpy(&rctx.beta_slow,   (int32_t *) op_params + 10, sizeof(float));
    memcpy(&rctx.sections,    (int32_t *) op_params + 11, sizeof(int) * 4);

    rctx.theta_scale = powf(rctx.freq_base, -2.0f / rctx.n_dims);
    rctx.theta_powers[0] = 1.0f;
    for (int j = 1; j < 32; j++) {
        rctx.theta_powers[j] = rctx.theta_powers[j - 1] * rctx.theta_scale;
    }
    rctx.theta_scale_32 = rctx.theta_powers[31] * rctx.theta_scale;

    rope_corr_dims(rctx.n_dims, rctx.n_ctx_orig, rctx.freq_base, rctx.beta_fast, rctx.beta_slow, rctx.corr_dims);

    rctx.fuse_norm       = fuse_norm;
    rctx.kv              = kv;
    rctx.kv_off          = theta_cache_size_aligned + 2 * (ne1A * srcA_row_size_aligned + ne1B * srcB_row_size_aligned) + 2 * w_bytes_aligned;
    rctx.v_row_bytes     = v_row_bytes;
    rctx.kc_row_bytes    = kc_row_bytes;
    rctx.vc_row_bytes    = vc_row_bytes;
    rctx.w_bytes_aligned = w_bytes_aligned;
    memcpy(&rctx.norm_eps, &octx->kernel_params[0], sizeof(float));
    rctx.src0_row_size   = srcA_row_size;
    rctx.src0_row_stride = srcA_row_stride;
    rctx.dst_row_size    = dstA_row_size;
    rctx.dst_row_stride  = dstA_row_stride;
    rctx.src0_row_size_aligned = srcA_row_size_aligned;
    rctx.theta_cache_offset    = theta_cache_size_aligned;
    rctx.spad_per_thread       = spad_per_thread;

    rctx.has_pair = true;
    rctx.ne1_b    = ne1B;
    rctx.src0b_row_size          = srcB_row_size;
    rctx.src0b_row_stride        = srcB_row_stride;
    rctx.dstb_row_size           = dstB_row_size;
    rctx.dstb_row_stride         = dstB_row_stride;
    rctx.src0b_row_size_aligned  = srcB_row_size_aligned;

    // token-unit partitioning (src0_nrows/src0_nrows_per_thread repurposed as token counts)
    rctx.src0_nrows            = n_tokens;
    rctx.src0_nrows_per_thread = (n_tokens + n_threads - 1) / n_threads;

    FARF(HIGH, "rope-nx-f32 n-tokens %u n-dims %d ne1A %u ne1B %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n",
         n_tokens, rctx.n_dims, ne1A, ne1B, rctx.ext_factor, rctx.theta_scale, rctx.attn_factor);

    work_queue_run(octx->ctx->work_queue, rope_nx_job_f32, &rctx, n_threads);

    return err;
}

int op_rope(struct htp_ops_context * octx) {
    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
            if (octx->op == HTP_OP_ROPE_NX) {
                return execute_op_rope_nx_f32(octx);
            }
            return execute_op_rope_f32(octx);

        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
