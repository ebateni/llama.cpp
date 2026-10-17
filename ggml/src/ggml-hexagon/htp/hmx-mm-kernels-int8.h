// INT8 HMX matmul helpers: Q4_0 -> int8 weight crouton expand, per-row u8 activation quantizer, output convert.
// Scalar reference versions (the HVX versions replace them one by one).
//
// Layouts (verified on v81, see analysis/hmx-int-log.md):
//   activation crouton (32 spatial x 32 k) = 2 KB, data only in odd bytes: (s/2)*128 + k*4 + 1 + 2*(s&1), unsigned
//   weight crouton (32 k x 32 out) = 1 KB: (k/4)*128 + c*4 + (k%4), signed; a 64-filter group holds
//     per k crouton: set0 (cols 0..31) 1 KB then set1 (cols 32..63) 1 KB
//   output tile (32 spatial x 32 out) = 2 KB int16: vector = 2 spatials, word = channel, 16-bit halves

#ifndef HMX_MM_KERNELS_INT8_H
#define HMX_MM_KERNELS_INT8_H

#include <math.h>
#include <stdint.h>

#include "hvx-utils.h"
#include "hvx-inverse.h"
#include "hvx-reduce.h"

#define HMX_INT8_CROUTON_A 2048
#define HMX_INT8_CROUTON_W 2048  // two 1 KB sets (64 filters) per k crouton

// Q4_0 tiled weights (tile = 512 B nibbles + 32 fp16 scales, tile index ct * n_k_tiles + kt)
// -> int8 crouton groups. Group g = ct / 2, set = ct & 1. n_col_tiles must be even.
// Per output row: s_n = dmax / 15.875; per block ratio r = round(|d| / dmax * 15.875 * 16) / 16;
// w8 = clip(round(sign(d) * q * r), -127, 127).
// Outputs per column: s_n, sum(w8), sum(|w8|).
static inline void hmx_int8_expand_q4_0_ref(uint8_t * dst, float * s_n, int32_t * colsum, int32_t * abssum,
                                            const uint8_t * src, uint32_t ct0, uint32_t ct1, uint32_t n_k_tiles,
                                            uint32_t aligned_tile_size) {
    for (uint32_t ct = ct0; ct < ct1; ++ct) {
        uint8_t * grp = dst + (size_t) (ct >> 1) * n_k_tiles * HMX_INT8_CROUTON_W + (ct & 1) * 1024;
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t col = ct * 32 + row;
            float dmax = 0.0f;
            for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
                const uint8_t * tile = src + ((size_t) ct * n_k_tiles + kt) * aligned_tile_size;
                const float d = fabsf((float) ((const __fp16 *) (tile + 512))[row]);
                dmax = d > dmax ? d : dmax;
            }
            s_n[col] = dmax / 15.875f;
            int32_t sum = 0, asum = 0;
            for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
                const uint8_t * tile = src + ((size_t) ct * n_k_tiles + kt) * aligned_tile_size;
                const float d    = (float) ((const __fp16 *) (tile + 512))[row];
                const float sgn  = d < 0 ? -1.0f : 1.0f;
                const int   rq   = dmax > 0 ? (int) floorf(fabsf(d) / dmax * 254.0f + 0.5f) : 0;
                uint8_t * cr = grp + (size_t) kt * HMX_INT8_CROUTON_W;
                asum += rq;
                for (uint32_t k = 0; k < 32; ++k) {
                    const uint8_t b = tile[(k >> 1) * 32 + row];
                    const int q = (int) ((k & 1) ? (b >> 4) : (b & 0xF)) - 8;
                    int w8 = (q * (int) sgn * rq + 8) >> 4;
                    w8 = w8 > 127 ? 127 : (w8 < -127 ? -127 : w8);
                    cr[(k >> 2) * 128 + row * 4 + (k & 3)] = (uint8_t) (int8_t) w8;
                    sum += w8;
                }
            }
            colsum[col] = sum;
            abssum[col] = asum;
        }
    }
}

// f32 rows -> u8 (q + 128) odd-byte crouton layout, per row scale = absmax / 127.
// dst: [ceil(m/32)][k/32] croutons; rows >= m are zero (u8 = 128).
static inline void hmx_int8_quant_act_rows_ref(uint8_t * dst, float * a_scale, const float * x, uint32_t m,
                                               uint32_t k, uint32_t x_stride) {
    const uint32_t n_rt = (m + 31) / 32, nkc = k / 32;
    for (uint32_t rt = 0; rt < n_rt; ++rt) {
        for (uint32_t s = 0; s < 32; ++s) {
            const uint32_t row = rt * 32 + s;
            const float * xr = x + (size_t) row * x_stride;
            float amax = 0.0f;
            if (row < m) {
                for (uint32_t i = 0; i < k; ++i) {
                    const float a = fabsf(xr[i]);
                    amax = a > amax ? a : amax;
                }
            }
            const float inv = amax > 0 ? 127.0f / amax : 0.0f;
            if (row < m) { a_scale[row] = amax / 127.0f; }
            for (uint32_t kc = 0; kc < nkc; ++kc) {
                uint8_t * cr = dst + ((size_t) rt * nkc + kc) * HMX_INT8_CROUTON_A;
                for (uint32_t kk = 0; kk < 32; ++kk) {
                    const int q = row < m ? (int) roundf(xr[kc * 32 + kk] * inv) : 0;
                    cr[(s >> 1) * 128 + kk * 4 + 1 + 2 * (s & 1)] = (uint8_t) (q + 128);
                }
            }
        }
    }
}

// Bias tables: per 64-filter group 512 B (set0 256 B, set1 256 B); per set 32 low words (fp16 scale bits)
// then 32 high words (signed integer added to acc). ibias = -128 * colsum cancels the +128 activation offset.
// Per-column shift keeps |acc| / 2^sh inside int16 (bound 127 * sum|w8|), minus `reduce` bits of headroom.
// col_mul[col] = s_n * 2^sh converts the int16 readout to the real column scale.
static inline void hmx_int8_make_bias_ref(uint32_t * bias, float * col_mul, const float * s_n, const int32_t * colsum,
                                          const int32_t * abssum, uint32_t c0, uint32_t c1, uint32_t reduce) {
    for (uint32_t col = c0; col < c1; ++col) {
        const uint32_t bnd = (uint32_t) abssum[col] << 11;
        uint32_t sh = 0;
        while ((bnd >> sh) >= 32768u) { sh++; }
        sh = sh > reduce ? sh - reduce : 0;
        const uint32_t g = col / 64, st = (col / 32) & 1, c = col & 31;
        uint32_t * t = bias + g * 128 + st * 64;
        t[c]      = 0x4000u - (sh << 10);
        t[32 + c] = (uint32_t) (-128 * colsum[col]) + (sh ? (1u << (sh - 1)) : 0u);
        col_mul[col] = s_n[col] * (float) (1u << sh);
    }
}

// int16 output tiles ([rt][g][set] 2 KB) -> f32 rows: dst[row][col] = acc16 * a_scale[row] * col_mul[col]
static inline void hmx_int8_out_convert_ref(float * dst, uint32_t dst_stride, const int16_t * o, uint32_t m,
                                            uint32_t n_cols, const float * a_scale, const float * col_mul) {
    const uint32_t n_rt = (m + 31) / 32, ng = n_cols / 64;
    for (uint32_t rt = 0; rt < n_rt; ++rt) {
        for (uint32_t g = 0; g < ng; ++g) {
            for (uint32_t st = 0; st < 2; ++st) {
                const int16_t * t = o + (size_t) ((rt * ng + g) * 2 + st) * 1024;
                for (uint32_t v = 0; v < 16; ++v) {
                    for (uint32_t c = 0; c < 32; ++c) {
                        for (uint32_t h = 0; h < 2; ++h) {
                            const uint32_t row = rt * 32 + 2 * v + h;
                            if (row >= m) { continue; }
                            const uint32_t col = g * 64 + st * 32 + c;
                            dst[(size_t) row * dst_stride + col] = (float) t[v * 64 + c * 2 + h] * a_scale[row] * col_mul[col];
                        }
                    }
                }
            }
        }
    }
}


// ---- HVX versions ----

// Tiled Q4_0 columns [ct0, ct1) -> int8 crouton groups (same result as the ref, ratio table in 1/16 steps).
// Per 32-column tile: pass 1 finds dmax per channel, pass 2 expands each k tile with
// w8 = clip((q * R + 8) >> 4), R = signed round(|d| / dmax * 254).
static inline void hmx_int8_expand_q4_0(uint8_t * dst, float * s_n, int32_t * colsum, int32_t * abssum,
                                        const uint8_t * src, uint32_t ct0, uint32_t ct1, uint32_t n_k_tiles,
                                        uint32_t aligned_tile_size) {
    const HVX_Vector mask_lo = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector mask_e  = Q6_Vh_vsplat_R(0x00ff);
    const HVX_Vector ones_b  = Q6_V_vsplat_R(0x01010101);
    const HVX_Vector half_f  = hvx_vec_splat_f32(0.5f);
    const HVX_Vector tiny_f  = hvx_vec_splat_f32(1e-30f);
    const HVX_Vector k254_f  = hvx_vec_splat_f32(254.0f);
    const HVX_Vector m16     = Q6_V_vsplat_R(0xffff);
    const HVX_Vector zero    = Q6_V_vzero();

    for (uint32_t ct = ct0; ct < ct1; ++ct) {
        uint8_t *       grp  = dst + (size_t) (ct >> 1) * n_k_tiles * HMX_INT8_CROUTON_W + (ct & 1) * 1024;
        const uint8_t * tcol = src + (size_t) ct * n_k_tiles * aligned_tile_size;

        HVX_Vector dmax = tiny_f;
        for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
            const HVX_Vector sc = hvx_vmemu(tcol + kt * aligned_tile_size + 512);
            dmax = Q6_Vsf_vmax_VsfVsf(dmax, hvx_vec_abs_f32(Q6_V_lo_W(hvx_vec_f16_to_f32(sc))));
        }
        const HVX_Vector invk = hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(dmax), k254_f);
        hvx_vmem(s_n + ct * 32) = hvx_vec_mul_f32_f32(dmax, hvx_vec_splat_f32(1.0f / 15.875f));

        HVX_Vector vsum = zero, vabs = zero;
        for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
            const uint8_t * tile = tcol + (size_t) kt * aligned_tile_size;
            uint8_t *       cr   = grp + (size_t) kt * HMX_INT8_CROUTON_W;

            const HVX_Vector scf = Q6_V_lo_W(hvx_vec_f16_to_f32(hvx_vmemu(tile + 512)));
            HVX_Vector R = Q6_Vw_equals_Vsf(hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vec_abs_f32(scf), invk), half_f));
            R = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(zero, scf), Q6_Vw_vsub_VwVw(zero, R), R);
            vabs = Q6_Vw_vadd_VwVw(vabs, Q6_Vw_vabs_Vw(R));
            const HVX_Vector Rm = Q6_V_vand_VV(R, m16);
            const HVX_Vector Rd = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Rm, 16), Rm);
            // ph halfword i = bytes (2i, 2i+1) = channel i/2, and Rd halfword i = R[i/2]; the :rnd:sat narrowing
            // interleaves its two sources (even byte from Vv, odd byte from Vu), which restores the byte order.
            const HVX_Vector cR = Q6_Vh_vsub_VhVh(zero, Q6_Vh_vasl_VhR(Rd, 3));   // -8R
            #pragma unroll(4)
            for (uint32_t g = 0; g < 4; ++g) {
                const HVX_Vector vq = hvx_vmem(tile + g * 128);
                const HVX_Vector lo = Q6_V_vand_VV(vq, mask_lo);
                const HVX_Vector hi = Q6_Vub_vlsr_VubR(vq, 4);
                const HVX_VectorPair P = Q6_W_vshuff_VVR(hi, lo, -1);
                const HVX_Vector ph0 = Q6_Vh_vshuff_Vh(Q6_V_lo_W(P));
                const HVX_Vector ph1 = Q6_Vh_vshuff_Vh(Q6_V_hi_W(P));
                const HVX_Vector w0 = Q6_Vb_vasr_VhVhR_rnd_sat(
                    Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(Q6_Vuh_vlsr_VuhR(ph0, 8), Rd), cR),
                    Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(Q6_V_vand_VV(ph0, mask_e), Rd), cR), 4);
                const HVX_Vector w1 = Q6_Vb_vasr_VhVhR_rnd_sat(
                    Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(Q6_Vuh_vlsr_VuhR(ph1, 8), Rd), cR),
                    Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(Q6_V_vand_VV(ph1, mask_e), Rd), cR), 4);
                hvx_vmem(cr + (g * 2 + 0) * 128) = w0;
                hvx_vmem(cr + (g * 2 + 1) * 128) = w1;
                vsum = Q6_Vw_vrmpyacc_VwVbVb(vsum, w0, ones_b);
                vsum = Q6_Vw_vrmpyacc_VwVbVb(vsum, w1, ones_b);
            }
        }
        hvx_vmem(colsum + ct * 32) = vsum;
        hvx_vmem(abssum + ct * 32) = vabs;
    }
}

// One row pair (rows 2p, 2p+1 of a 32-row tile) -> u8 (q + 128) odd-byte layout at dst_pair (crouton 0, vector p).
// x0/x1 must be 128 B aligned (VTCM staging), NULL for rows past the end. Writes the row scales to sc[0], sc[1].
static inline void hmx_int8_quant_pair(uint8_t * dst_pair, float * sc, const float * x0, const float * x1, uint32_t k) {
    const uint32_t nkc = k / 32;
    const HVX_Vector abs_mask = Q6_V_vsplat_R(0x7fffffff);
    const HVX_Vector sgn_mask = Q6_V_vsplat_R(0x80000000);
    const HVX_Vector half_bits = Q6_V_vsplat_R(0x3f000000);
    const HVX_Vector c127 = hvx_vec_splat_f32(127.0f);
    const HVX_Vector tiny = hvx_vec_splat_f32(1e-30f);
    const HVX_Vector off128 = Q6_V_vsplat_R(128);
    float tmp[32] __attribute__((aligned(128)));
    HVX_Vector inv0 = Q6_V_vzero(), inv1 = Q6_V_vzero();
    if (x0) {
        HVX_Vector mx = tiny;
        for (uint32_t i = 0; i < nkc; ++i) { mx = Q6_Vsf_vmax_VsfVsf(mx, Q6_V_vand_VV(((const HVX_Vector *) x0)[i], abs_mask)); }
        mx = hvx_vec_reduce_max_f32(mx);
        hvx_vmem(tmp) = mx;
        sc[0] = tmp[0] * (1.0f / 127.0f);
        inv0 = hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(mx), c127);
    }
    if (x1) {
        HVX_Vector mx = tiny;
        for (uint32_t i = 0; i < nkc; ++i) { mx = Q6_Vsf_vmax_VsfVsf(mx, Q6_V_vand_VV(((const HVX_Vector *) x1)[i], abs_mask)); }
        mx = hvx_vec_reduce_max_f32(mx);
        hvx_vmem(tmp) = mx;
        sc[1] = tmp[0] * (1.0f / 127.0f);
        inv1 = hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(mx), c127);
    }
    for (uint32_t kc = 0; kc < nkc; ++kc) {
        HVX_Vector u0 = off128, u1 = off128;
        if (x0) {
            const HVX_Vector y = hvx_vec_mul_f32_f32(((const HVX_Vector *) x0)[kc], inv0);
            const HVX_Vector h = Q6_V_vor_VV(Q6_V_vand_VV(y, sgn_mask), half_bits);
            u0 = Q6_Vw_vadd_VwVw(Q6_Vw_equals_Vsf(hvx_vec_add_f32_f32(y, h)), off128);
        }
        if (x1) {
            const HVX_Vector y = hvx_vec_mul_f32_f32(((const HVX_Vector *) x1)[kc], inv1);
            const HVX_Vector h = Q6_V_vor_VV(Q6_V_vand_VV(y, sgn_mask), half_bits);
            u1 = Q6_Vw_vadd_VwVw(Q6_Vw_equals_Vsf(hvx_vec_add_f32_f32(y, h)), off128);
        }
        *(HVX_Vector *) (dst_pair + (size_t) kc * HMX_INT8_CROUTON_A) = Q6_V_vor_VV(Q6_Vw_vasl_VwR(u1, 24), Q6_Vw_vasl_VwR(u0, 8));
    }
}

// forward declaration used below

// A16: two int8 passes for a row pair. dst_hi/dst_lo point at the pair's 128-byte stride inside its 32-row tile
// (caller passes dst + rt*nkc*2048 + p*128); sc = { s_hi(row0), s_hi(row1), s_lo(row0), s_lo(row1) }.
// Both passes are full-scale int8 in their own scale, so the window shift is the same for both (bound 127 * sum|w8|)
// and the epilogue combines o1 * s_hi + o2 * s_lo. Store layout is the odd-byte one: word k = channel k, byte 1 = row 2p, byte 3 = row 2p+1.
static inline void hmx_int8_quant_pair_a16(uint8_t * dst_hi, uint8_t * dst_lo, float * sc,
                                           const float * x0, const float * x1, uint32_t k) {
    const uint32_t nkc = k / 32;
    const HVX_Vector abs_mask = Q6_V_vsplat_R(0x7fffffff);
    const HVX_Vector sgn_mask = Q6_V_vsplat_R(0x80000000);
    const HVX_Vector half_bits = Q6_V_vsplat_R(0x3f000000);
    const HVX_Vector c127 = hvx_vec_splat_f32(127.0f);
    const HVX_Vector tiny = hvx_vec_splat_f32(1e-30f);
    const HVX_Vector off128 = Q6_V_vsplat_R(128);
    float tmp[32] __attribute__((aligned(128)));
    HVX_Vector inv[2] = { Q6_V_vzero(), Q6_V_vzero() };
    HVX_Vector ilo[2] = { Q6_V_vzero(), Q6_V_vzero() };
    const float * xs[2] = { x0, x1 };

    // pass A: row max.  pass B: hi quantization, residual t = y - q written back over x (the staging buffer is
    // private and writable), residual max.  pass C: lo quantization from t only (no recomputation of y and q).
    for (int r = 0; r < 2; ++r) {
        const float * x = xs[r];
        if (!x) { sc[r] = 0.0f; sc[2 + r] = 0.0f; continue; }
        HVX_Vector mx = tiny;
        for (uint32_t i = 0; i < nkc; ++i) { mx = Q6_Vsf_vmax_VsfVsf(mx, Q6_V_vand_VV(((const HVX_Vector *) x)[i], abs_mask)); }
        mx = hvx_vec_reduce_max_f32(mx);
        hvx_vmem(tmp) = mx;
        sc[r] = tmp[0] * (1.0f / 127.0f);
        inv[r] = hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(mx), c127);
    }

    HVX_Vector tmax[2] = { tiny, tiny };
    for (uint32_t kc = 0; kc < nkc; ++kc) {
        HVX_Vector uhi[2];
        for (int r = 0; r < 2; ++r) {
            if (!xs[r]) { uhi[r] = off128; continue; }
            HVX_Vector * xp = (HVX_Vector *) xs[r] + kc;
            const HVX_Vector y = hvx_vec_mul_f32_f32(*xp, inv[r]);
            const HVX_Vector h = Q6_V_vor_VV(Q6_V_vand_VV(y, sgn_mask), half_bits);
            const HVX_Vector q = Q6_Vw_equals_Vsf(hvx_vec_add_f32_f32(y, h));
            uhi[r] = Q6_Vw_vadd_VwVw(q, off128);
            const HVX_Vector t = hvx_vec_sub_f32_f32(y, Q6_Vsf_equals_Vw(q));
            *xp = t;
            tmax[r] = Q6_Vsf_vmax_VsfVsf(tmax[r], Q6_V_vand_VV(t, abs_mask));
        }
        *(HVX_Vector *) (dst_hi + (size_t) kc * HMX_INT8_CROUTON_A) = Q6_V_vor_VV(Q6_Vw_vasl_VwR(uhi[1], 24), Q6_Vw_vasl_VwR(uhi[0], 8));
    }
    for (int r = 0; r < 2; ++r) {
        if (!xs[r]) { continue; }
        const HVX_Vector tm = hvx_vec_reduce_max_f32(tmax[r]);
        hvx_vmem(tmp) = tm;
        const float amax_t = tmp[0];
        sc[2 + r] = sc[r] * amax_t * (1.0f / 127.0f);
        ilo[r] = amax_t > 0.0f ? hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(tm), c127) : Q6_V_vzero();
    }

    for (uint32_t kc = 0; kc < nkc; ++kc) {
        HVX_Vector ulo[2];
        for (int r = 0; r < 2; ++r) {
            if (!xs[r]) { ulo[r] = off128; continue; }
            const HVX_Vector t = hvx_vec_mul_f32_f32(((const HVX_Vector *) xs[r])[kc], ilo[r]);
            const HVX_Vector h2 = Q6_V_vor_VV(Q6_V_vand_VV(t, sgn_mask), half_bits);
            ulo[r] = Q6_Vw_vadd_VwVw(Q6_Vw_equals_Vsf(hvx_vec_add_f32_f32(t, h2)), off128);
        }
        *(HVX_Vector *) (dst_lo + (size_t) kc * HMX_INT8_CROUTON_A) = Q6_V_vor_VV(Q6_Vw_vasl_VwR(ulo[1], 24), Q6_Vw_vasl_VwR(ulo[0], 8));
    }
}

// f32 rows (DDR or any memory, k % 32 == 0, 128 B aligned rows) -> u8 odd-byte croutons, per row scale = absmax / 127.
static inline void hmx_int8_quant_act_rows(uint8_t * dst, float * a_scale, const float * x, uint32_t m,
                                           uint32_t k, uint32_t x_stride) {
    const uint32_t n_rt = (m + 31) / 32, nkc = k / 32;
    float sc[2];
    for (uint32_t rt = 0; rt < n_rt; ++rt) {
        for (uint32_t p = 0; p < 16; ++p) {
            const uint32_t r0 = rt * 32 + 2 * p;
            const float * x0 = r0 < m ? x + (size_t) r0 * x_stride : NULL;
            const float * x1 = r0 + 1 < m ? x + (size_t) (r0 + 1) * x_stride : NULL;
            hmx_int8_quant_pair(dst + (size_t) rt * nkc * HMX_INT8_CROUTON_A + p * 128, sc, x0, x1, k);
            if (x0) { a_scale[r0] = sc[0]; }
            if (x1) { a_scale[r0 + 1] = sc[1]; }
        }
    }
}

// Per 32-column tile ct in [ct0, ct1): bias-table words (scale exponent shift, ibias) and col_mul = s_n * 2^sh.
// bound = 2048 * sum|R| >= 127 * sum|w8| >= max |sum a8*w8|; sh = max(0, bitlen(bound) - 15 - reduce) keeps the int16 window unclipped.
static inline void hmx_int8_make_bias(uint32_t * bias, float * col_mul, const float * s_n, const int32_t * colsum,
                                      const int32_t * abssum, uint32_t ct0, uint32_t ct1, uint32_t reduce) {
    const HVX_Vector lim   = Q6_V_vsplat_R(17 - reduce);
    const HVX_Vector zero  = Q6_V_vzero();
    const HVX_Vector c4000 = Q6_V_vsplat_R(0x4000);
    for (uint32_t ct = ct0; ct < ct1; ++ct) {
        const HVX_Vector bound = Q6_Vw_vasl_VwR(hvx_vmem(abssum + ct * 32), 11);
        const HVX_Vector sh = Q6_Vw_vmax_VwVw(zero, Q6_Vw_vsub_VwVw(lim, Q6_Vuw_vcl0_Vuw(bound)));
        const HVX_Vector sn = hvx_vmem(s_n + ct * 32);
        uint32_t * t = bias + (ct >> 1) * 128 + (ct & 1) * 64;
        hvx_vmem(t)      = Q6_Vw_vsub_VwVw(c4000, Q6_Vw_vasl_VwR(sh, 10));
        // the window read truncates: add half an LSB (2^(sh-1), 0 for sh == 0) so that it rounds to nearest
        const HVX_Vector half_lsb = Q6_Vw_vlsr_VwVw(Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(1), sh), Q6_V_vsplat_R(1));
        hvx_vmem(t + 32) = Q6_Vw_vadd_VwVw(Q6_Vw_vsub_VwVw(zero, Q6_Vw_vasl_VwR(hvx_vmem(colsum + ct * 32), 7)), half_lsb);
        hvx_vmem(col_mul + ct * 32) = hvx_vec_mul_f32_f32(sn, Q6_Vw_vadd_VwVw(Q6_V_vsplat_R(0x3f800000), Q6_Vw_vasl_VwR(sh, 23)));
    }
}

// Scale one 2 KB int16 accumulator tile to f16 in place: row pair scale (word of two f16) and column scale (f16 per channel).
static inline void hmx_int8_scale_tile(HVX_Vector * tv, const uint32_t * as16, HVX_Vector cm16) {
    #pragma unroll(4)
    for (uint32_t v = 0; v < 16; ++v) {
        const HVX_Vector asv = Q6_V_vsplat_R(as16[v]);
        const HVX_Vector a = Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(tv[v]), asv);
        tv[v] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(a, cm16));
    }
}

// Core loop. act: [n_row_tiles][n_k] croutons, wgt: groups of 64 filters, each n_k 2 KB croutons, group stride wgt_stride bytes,
// bias: 512 B per group (256 B aligned), group stride bias_stride bytes, out: [n_row_tiles][n_groups][2] 2 KB int16 tiles.
// Must run on the HMX queue thread.
static void hmx_int8_core_dot_chunk(uint8_t * restrict out, const uint8_t * restrict act, const uint8_t * restrict wgt,
                                    const uint8_t * restrict bias, uint32_t n_row_tiles, uint32_t n_groups,
                                    uint32_t n_k, size_t wgt_stride, size_t bias_stride) {
    asm volatile("mxclracc.hf\n" ::: "memory");
    for (uint32_t rt = 0; rt < n_row_tiles; ++rt) {
        for (uint32_t g = 0; g < n_groups; ++g) {
            const uint8_t * a0 = act + (size_t) rt * n_k * HMX_INT8_CROUTON_A;
            const uint8_t * w0 = wgt + (size_t) g * wgt_stride;
            const uint8_t * b0 = bias + (size_t) g * bias_stride;
            uint8_t *       o  = out + (size_t) (rt * n_groups + g) * 2 * 2048;
            asm volatile("mxclracc\n" ::: "memory");
            uint32_t left = n_k, done = 0;
            while (left) {
                const uint32_t n = left > 32 ? 32 : left;
                const uint32_t r = 2048 * n - 1;
                const uint8_t * a = a0 + (size_t) done * HMX_INT8_CROUTON_A;
                const uint8_t * w = w0 + (size_t) done * HMX_INT8_CROUTON_W;
                asm volatile("{\n activation.ub = mxmem(%0, %1):deep\n weight.b = mxmem(%2, %3):deep\n}\n"
                             :: "r"(a), "r"(r), "r"(w), "r"(r) : "memory");
                left -= n;
                done += n;
            }
            asm volatile("bias = mxmem2(%0)\n" :: "r"(b0) : "memory");
            asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(o), "r"(0) : "memory");
            asm volatile("bias = mxmem2(%0)\n" :: "r"(b0 + 256) : "memory");
            asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(o + 2048), "r"(0) : "memory");
        }
    }
}

// ---- Hadamard rotation of an activation row (matches hex-int8-pack.h on the weight side) ----
// Row = 3 blocks of 512 floats. x'[i] = sign[i] * x[i] / sqrt(512) is mixed by the unnormalised Walsh-Hadamard transform inside each block:
// WHT over index bits = 5 in-vector stages (lane bit) + 4 cross-vector stages (vector index within the block).

static const uint32_t hmx_int8_iota[32] __attribute__((aligned(128))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
};

// signs[i] = sign_hash(i) / sqrt(512) as f32
static inline void hmx_int8_make_signs(float * signs, uint32_t k) {
    for (uint32_t i = 0; i < k; ++i) {
        signs[i] = (((i * 2654435761u) >> 20) & 1u ? -1.0f : 1.0f) * 0.044194173824159216f;
    }
}

// Row WHT (3 blocks of 512 floats = 16 vectors each). Lane stages (bit j of the word index, j = 0..4) use an even/odd split:
// for a pair of vectors (a, b), vdeal on words separates even and odd lanes of the concatenation, so one add and one sub
// perform the lane-bit-0 butterfly of both vectors at once. Higher lane bits are reached by changing the deal block size.
// One row = 3 blocks x (5 lane stages x 8 pairs + 4 vector stages x 8 pairs) = 3 x 72 pair-butterflies.
static inline __attribute__((always_inline)) void hmx_int8_wht_pair_bit(HVX_Vector * a, HVX_Vector * b, int j) {
    const int bytes = 4 << j;
    HVX_VectorPair P = Q6_W_vdeal_VVR(*b, *a, bytes);      // lo = lanes with bit j == 0 (a then b), hi = lanes with bit j == 1
    HVX_Vector     e = Q6_V_lo_W(P), o = Q6_V_hi_W(P);
    HVX_VectorPair R = Q6_W_vshuff_VVR(hvx_vec_sub_f32_f32(e, o), hvx_vec_add_f32_f32(e, o), bytes);
    *a = Q6_V_lo_W(R);
    *b = Q6_V_hi_W(R);
}

// f16 variant: the row is converted to f16 (8 vectors per 512-block instead of 16), the WHT runs on halfwords, and the result is
// converted back to f32 for the quantizer. Rotation precision ~1e-3 (f16 mantissa) vs the A8 quantization error ~1e-2, so the extra
// error is negligible, and the butterfly work halves. Sign multiply stays in f32 (identical sign handling to the f32 variant).
static inline void hmx_int8_rotate_row_f16(HVX_Vector * v, const HVX_Vector * sg, uint32_t k) {
    HVX_Vector h[8];
    for (uint32_t blk = 0; blk < k / 512; ++blk) {
        HVX_Vector * b = v + blk * 16;
        #pragma unroll(4)
        for (uint32_t i = 0; i < 8; ++i) {
            const HVX_Vector x0 = hvx_vec_mul_f32_f32(b[2 * i],     sg[blk * 16 + 2 * i]);
            const HVX_Vector x1 = hvx_vec_mul_f32_f32(b[2 * i + 1], sg[blk * 16 + 2 * i + 1]);
            h[i] = hvx_vec_f32_to_f16(x0, x1);              // 64 halves in element order
        }
        // lane stages: bit j of the halfword index (j = 0..4), pairs of vectors -> 4 pairs per stage
        #pragma unroll
        for (int j = 0; j < 5; ++j) {
            const int bytes = 2 << j;
            #pragma unroll
            for (uint32_t i = 0; i < 8; i += 2) {
                HVX_VectorPair P = Q6_W_vdeal_VVR(h[i + 1], h[i], bytes);
                const HVX_Vector e = Q6_V_lo_W(P), o = Q6_V_hi_W(P);
                HVX_VectorPair R = Q6_W_vshuff_VVR(hvx_vec_sub_f16_f16(e, o), hvx_vec_add_f16_f16(e, o), bytes);
                h[i] = Q6_V_lo_W(R);
                h[i + 1] = Q6_V_hi_W(R);
            }
        }
        // vector stages: bit 5..7 of the element index = bit 0..2 of the vector index
        #pragma unroll
        for (uint32_t hh = 1; hh < 8; hh <<= 1) {
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                if (i & hh) { continue; }
                const HVX_Vector a = h[i], c = h[i + hh];
                h[i] = hvx_vec_add_f16_f16(a, c);
                h[i + hh] = hvx_vec_sub_f16_f16(a, c);
            }
        }
        #pragma unroll(4)
        for (uint32_t i = 0; i < 8; ++i) {
            const HVX_VectorPair p = hvx_vec_f16_to_f32(h[i]);   // 64 halves -> two f32 vectors in element order
            b[2 * i] = Q6_V_lo_W(p);
            b[2 * i + 1] = Q6_V_hi_W(p);
        }
    }
}

// one row of k floats (VTCM, 128 B aligned), in place; fully unrolled so the 16 vectors stay in registers
static inline void hmx_int8_rotate_row(HVX_Vector * v, const HVX_Vector * sg, uint32_t k) {
    for (uint32_t blk = 0; blk < k / 512; ++blk) {
        HVX_Vector * b = v + blk * 16;
        const HVX_Vector * g = sg + blk * 16;
        HVX_Vector t0  = hvx_vec_mul_f32_f32(b[0],  g[0]),  t1  = hvx_vec_mul_f32_f32(b[1],  g[1]);
        HVX_Vector t2  = hvx_vec_mul_f32_f32(b[2],  g[2]),  t3  = hvx_vec_mul_f32_f32(b[3],  g[3]);
        HVX_Vector t4  = hvx_vec_mul_f32_f32(b[4],  g[4]),  t5  = hvx_vec_mul_f32_f32(b[5],  g[5]);
        HVX_Vector t6  = hvx_vec_mul_f32_f32(b[6],  g[6]),  t7  = hvx_vec_mul_f32_f32(b[7],  g[7]);
        HVX_Vector t8  = hvx_vec_mul_f32_f32(b[8],  g[8]),  t9  = hvx_vec_mul_f32_f32(b[9],  g[9]);
        HVX_Vector t10 = hvx_vec_mul_f32_f32(b[10], g[10]), t11 = hvx_vec_mul_f32_f32(b[11], g[11]);
        HVX_Vector t12 = hvx_vec_mul_f32_f32(b[12], g[12]), t13 = hvx_vec_mul_f32_f32(b[13], g[13]);
        HVX_Vector t14 = hvx_vec_mul_f32_f32(b[14], g[14]), t15 = hvx_vec_mul_f32_f32(b[15], g[15]);
        #define WP(j) do { hmx_int8_wht_pair_bit(&t0,&t1,j); hmx_int8_wht_pair_bit(&t2,&t3,j); hmx_int8_wht_pair_bit(&t4,&t5,j); \
                           hmx_int8_wht_pair_bit(&t6,&t7,j); hmx_int8_wht_pair_bit(&t8,&t9,j); hmx_int8_wht_pair_bit(&t10,&t11,j); \
                           hmx_int8_wht_pair_bit(&t12,&t13,j); hmx_int8_wht_pair_bit(&t14,&t15,j); } while (0)
        WP(0); WP(1); WP(2); WP(3); WP(4);
        #undef WP
        #define BF(x, y) do { HVX_Vector _a = x, _c = y; x = hvx_vec_add_f32_f32(_a, _c); y = hvx_vec_sub_f32_f32(_a, _c); } while (0)
        BF(t0,t1); BF(t2,t3); BF(t4,t5); BF(t6,t7); BF(t8,t9); BF(t10,t11); BF(t12,t13); BF(t14,t15);
        BF(t0,t2); BF(t1,t3); BF(t4,t6); BF(t5,t7); BF(t8,t10); BF(t9,t11); BF(t12,t14); BF(t13,t15);
        BF(t0,t4); BF(t1,t5); BF(t2,t6); BF(t3,t7); BF(t8,t12); BF(t9,t13); BF(t10,t14); BF(t11,t15);
        BF(t0,t8); BF(t1,t9); BF(t2,t10); BF(t3,t11); BF(t4,t12); BF(t5,t13); BF(t6,t14); BF(t7,t15);
        #undef BF
        b[0]=t0; b[1]=t1; b[2]=t2; b[3]=t3; b[4]=t4; b[5]=t5; b[6]=t6; b[7]=t7;
        b[8]=t8; b[9]=t9; b[10]=t10; b[11]=t11; b[12]=t12; b[13]=t13; b[14]=t14; b[15]=t15;
    }
}

#endif // HMX_MM_KERNELS_INT8_H
