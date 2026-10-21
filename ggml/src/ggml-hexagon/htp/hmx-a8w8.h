#ifndef HMX_A8W8_H
#define HMX_A8W8_H

// A8W8 HMX dataflow primitives (GGML_HEXAGON_MM_A8W8): u8 activations (+128 bias) x int8 weights, int32 accumulation in the HMX, four byte-plane readout, HVX post-process. Per-weight error ~ int8. Weights: the Q4_0 tile (nibbles, 512 B) requantised to int8 with a per-column scale s[n] shared by the readout group: w_int8[k][n] = round((nibble[k][n] - 8) * C[n][kt] / 8) C = round(d[n][kt] * 127 / amax_n), signed byte (|w| <= 127) and s[n] = amax_n * 8 / 127 applied after the integer accumulation (amax_n = max_kt |d[n][kt]| of the group).
// Output of one HMX int job (64 activation rows x 32 columns, K = n_k_tiles * 32): acc_u8[r][n] = sum_k (q_act[r][k] + 128) * w[k][n] q_act in -127..127 true[r][n] = (acc_u8[r][n] - 128 * sum_k w[k][n]) * rowscale[r] * s[n] Each primitive was verified bit-exact (integer parts) or within fp16 rounding (scaled parts) against scalar references in hexagon-sim (v79) and on the board (v81).

#include "hvx-utils.h"
#include "hvx-inverse.h"
#include "matmul-ops.h"

// ---- 1. Source: the production Q4_0 tiled layout (htp: repack_q4_0_tiled): raw[cp*32 + n] low nibble = q[k=2cp][n], high nibble = q[k=2cp+1][n] (cp = 0..15, n = 0..31; vector t = bytes t*128..
// Q4M tile expand. holds cp = 4t..4t+3). dst: int8 HMX weight tile (1 KB) with byte g*128 + n*4 + j = (q[4g+j][n] - 8) * m[n]. vm: lanes 0..31 = m[n] (signed int8). *csum accumulates sum_k w[k][n] per column (int32 lane n) for the activation-bias correction.
static inline void a8w8_expand_q4m_tile(uint8_t * restrict dst, const uint8_t * restrict src, HVX_Vector vm, HVX_Vector * csum) {
    const HVX_Vector mask = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector i8   = Q6_Vb_vsplat_R(8);
    const HVX_Vector ones = Q6_Vb_vsplat_R(1);
    // m4: lane n*4+j = m[n]
    HVX_VectorPair p1 = Q6_W_vshuff_VVR(vm, vm, -1);
    HVX_VectorPair p2 = Q6_W_vshuff_VVR(Q6_V_lo_W(p1), Q6_V_lo_W(p1), -2);
    HVX_Vector m4 = Q6_V_lo_W(p2);
    HVX_Vector acc = *csum;
    #pragma unroll(4)
    for (int t = 0; t < 4; t++) {
        HVX_Vector v  = *(const HVX_Vector *) (src + t * 128);
        HVX_Vector lo = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(v, mask), i8);        // k even (nibble 0)
        HVX_Vector hi = Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(v, 4), i8);       // k odd
        HVX_VectorPair pr = Q6_W_vshuff_VVR(hi, lo, -1);                  // halfword lanes: (k=2cp, 2cp+1) for (c4, n) = lane/32, lane%32 ; lo_W: c4 0,1 ; hi_W: c4 2,3
        HVX_Vector A = Q6_V_lo_W(pr), B = Q6_V_hi_W(pr);
        HVX_VectorPair xa = Q6_W_vshuff_VVR(Q6_V_vror_VR(A, 64), A, -2);   // interleave (c4=0,n),(c4=1,n) halfwords -> byte n*4 + j
        HVX_VectorPair xb = Q6_W_vshuff_VVR(Q6_V_vror_VR(B, 64), B, -2);
        HVX_Vector qa = Q6_V_lo_W(xa), qb = Q6_V_lo_W(xb);
        HVX_VectorPair pa = Q6_Wh_vmpy_VbVb(qa, m4);                       // int16 products nib * C   (|nib| <= 8, |C| <= 127)
        HVX_VectorPair pb = Q6_Wh_vmpy_VbVb(qb, m4);
        // w = round(nib * C / 8) = (p + 4) >> 3 (|w| <= 127), packed back to bytes (even / odd product lanes interleaved) by one round-shift-saturate-pack
        HVX_Vector wa = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(pa), Q6_V_lo_W(pa), 3);
        HVX_Vector wb = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(pb), Q6_V_lo_W(pb), 3);
        *(HVX_Vector *) (dst + (2 * t) * 128)     = wa;
        *(HVX_Vector *) (dst + (2 * t + 1) * 128) = wb;
        acc = Q6_Vw_vadd_VwVw(acc, Q6_Vw_vrmpy_VbVb(wa, ones));            // per 4 bytes (j = 0..3 of one n): sums into lane n
        acc = Q6_Vw_vadd_VwVw(acc, Q6_Vw_vrmpy_VbVb(wb, ones));
    }
    *csum = acc;
}

// Q4M parameters of one column tile ct over n_k tiles (raw tiles of `aligned` bytes, fp16 block scales d[n] at +512): s[n] = max_kt |d| / 15 (1 if all 0),
// multipliers m = round(d * 15 / amax). Writes the expanded int8 tiles to w8 (1 KB each), s to sw[32] (f32) and 128 * sum_k w to csum[32] (int32).
static inline uint32_t a8w8_cyc(void) { uint32_t c; __asm__ volatile("%0 = c14" : "=r"(c)); return c; }
// phase cycle stamps of the last a8w8_expand_column_tile call on this thread (debug probe, read by a8w8_expand_worker)
static __thread uint32_t g_a8w8_stamp[4];
static int g_a8w8_expand_probe;   // 1: constant multipliers (skip the per-k-tile m derivation); 2: skip the tile unpack/multiply (timing probes, wrong results)
static inline void a8w8_expand_column_tile(uint8_t * restrict w8, const uint8_t * restrict raw, uint32_t n_k, uint32_t aligned,
                                           float * restrict sw, int32_t * restrict csum) {
    g_a8w8_stamp[0] = a8w8_cyc();
    const HVX_VectorPred dmask = Q6_Q_vsetq_R(64);                         // bytes 0..63 of the +512 vector = d[0..31] (fp16); the rest is tile padding
    const HVX_Vector absmask = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector mx = Q6_V_vzero();
    for (uint32_t kt = 0; kt < n_k; kt++) {
        HVX_Vector d = Q6_V_vmux_QVV(dmask, *(const HVX_Vector *) (raw + (size_t) kt * aligned + 512), Q6_V_vzero());
        mx = Q6_Vh_vmax_VhVh(mx, Q6_V_vand_VV(d, absmask));
    }
    g_a8w8_stamp[1] = a8w8_cyc();
    // vectorised per-column setup (was 32 scalar iterations, ~4.8k cycles per column tile on silicon): fp16 amax -> sw = amax * 8/127 (1 if amax == 0; within 1 ulp of the
    // scalar division) and vinv = 127 / amax / 256 as fp16 (0 if amax == 0; identical to the scalar path for every finite fp16 amax, checked exhaustively in hexagon-sim)
    HVX_Vector vinv;
    {
        const HVX_Vector zero = Q6_V_vzero();
        const HVX_Vector a    = Q6_V_lo_W(hvx_vec_f16_to_f32(mx));                    // columns 0..31 as fp32 (lanes 32..63 of mx are zero)
        const HVX_VectorPred pos = Q6_Q_vcmp_gt_VwVw(a, zero);
        const HVX_Vector r    = hvx_vec_inverse_f32(a);
        HVX_Vector swv = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, hvx_vec_splat_f32(8.0f / 127.0f)));
        swv = Q6_V_vmux_QVV(pos, swv, hvx_vec_splat_f32(1.0f));
        *(HVX_Vector *) sw = swv;
        HVX_Vector iv = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r, hvx_vec_splat_f32(127.0f / 256.0f)));
        iv = Q6_V_vmux_QVV(pos, iv, zero);
        vinv = hvx_vec_f32_to_f16(iv, zero);
    }
    const HVX_Vector sgn = Q6_Vh_vsplat_R(0x8000), half = Q6_Vh_vsplat_R(0x3800);
    const HVX_Vector c256 = Q6_Vh_vsplat_R(0x5c00);                          // fp16 256.0
    g_a8w8_stamp[2] = a8w8_cyc();
    HVX_Vector cs = Q6_V_vzero();
    // pair the m derivation of two k-tiles per vector: lanes 0..31 = d of kt, lanes 32..63 = d of kt + 1 (the +512 vector of a tile holds d[0..31] in its first 64 bytes);
    // the 64 fp16 lanes then go through one inv-multiply chain, and the two byte vectors of 32 multipliers are split back out with a pack / rotate
    const HVX_VectorPred hi32 = Q6_Q_vsetq_R(64);                           // bytes 0..63 select the first tile's d (lanes 0..31)
    HVX_Vector vinvh = Q6_V_vor_VV(Q6_V_vand_VV(vinv, Q6_V_vmux_QVV(hi32, Q6_V_vsplat_R(-1), Q6_V_vzero())), Q6_V_vror_VR(Q6_V_vand_VV(vinv, Q6_V_vmux_QVV(hi32, Q6_V_vsplat_R(-1), Q6_V_vzero())), 64));
    uint32_t kt = 0;
    for (; kt + 2 <= n_k; kt += 2) {
        const uint8_t * t0 = raw + (size_t) kt * aligned, * t1 = t0 + aligned;
        HVX_Vector d0 = *(const HVX_Vector *) (t0 + 512), d1 = *(const HVX_Vector *) (t1 + 512);
        HVX_Vector d  = Q6_V_vmux_QVV(hi32, d0, Q6_V_vror_VR(d1, 64));      // lanes 0..31: d(kt), lanes 32..63: d(kt+1)
        HVX_Vector y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(d, vinvh));
        y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(y, c256));
        y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(y, Q6_V_vor_VV(Q6_V_vand_VV(y, sgn), half)));
        HVX_Vector mq = Q6_Vh_equals_Vhf(y);
        HVX_Vector vmb = Q6_Vb_vpack_VhVh_sat(mq, mq);                       // bytes 0..31 = m(kt)[0..31], bytes 32..63 = m(kt+1)[0..31] (then repeated)
        HVX_Vector vm0 = vmb;                                                // bytes 0..31 used by a8w8_expand_q4m_tile
        HVX_Vector vm1 = Q6_V_vror_VR(vmb, 32);                              // bytes 0..31 = m(kt+1)
        a8w8_expand_q4m_tile(w8 + (size_t) kt * 1024, t0, vm0, &cs);
        a8w8_expand_q4m_tile(w8 + (size_t) (kt + 1) * 1024, t1, vm1, &cs);
    }
    for (; kt < n_k; kt++) {
        const uint8_t * t = raw + (size_t) kt * aligned;
        HVX_Vector d = Q6_V_vmux_QVV(dmask, *(const HVX_Vector *) (t + 512), Q6_V_vzero());
        HVX_Vector y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(d, vinv));
        y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(y, c256));
        y = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(y, Q6_V_vor_VV(Q6_V_vand_VV(y, sgn), half)));
        HVX_Vector mq = Q6_Vh_equals_Vhf(y);
        HVX_Vector vm = Q6_Vb_vpack_VhVh_sat(mq, mq);
        a8w8_expand_q4m_tile(w8 + (size_t) kt * 1024, t, vm, &cs);
    }
    // 128 * sum (activation bias), lanes = columns
    HVX_Vector c128 = Q6_Vw_vasl_VwR(cs, 7);
    *(HVX_Vector *) csum = c128;
    g_a8w8_stamp[3] = a8w8_cyc();
}

// ---- 2. Activation clip factor c in (0, 1]: scale = amax * c / 127 (values beyond c * amax saturate). activation quantizer: fp16 act tiles (hmx-act-tiles.h layout, n_k_tiles tiles per tile row, tile (r0,c0) at r0*n_k_tiles+c0) -> u8 64-row tiles ---- One call handles the four rows (2*pi ..
// Returns the four row scales through rs[0..3]. Set per op from op_params[2] (float bits) by matmul-ops.c. 2*pi+3) of tile row r0 (pi even: row pairs pi and pi+1 are the vectors pi and pi+1 of every tile): scans all K for the per-row abs max, computes the scales (rs[row] = amax/127) and writes the +128-biased int8 values into q[kt] (64-row tile, row*32 bytes; the four rows are one aligned 128 B vector per k tile, row_in_block = multiple of 4).
static float g_a8w8_clip = 1.0f;

static inline void a8w8_quant_quad(const __fp16 * restrict tiles /* tile (r0, 0) */, uint32_t n_k_tiles, uint32_t pi,
                                   uint8_t * restrict q /* 64-row u8 tiles, 2048 B per k tile */, uint32_t row_in_block, float * rs) {
    const HVX_Vector absmask = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector mA = Q6_V_vzero(), mB = Q6_V_vzero();
    const __fp16 * tp = tiles + pi * 64;
    #pragma unroll(4)
    for (uint32_t c0 = 0; c0 < n_k_tiles; c0++) {
        mA = Q6_Vh_vmax_VhVh(mA, Q6_V_vand_VV(*(const HVX_Vector *) (tp + (size_t) c0 * 1024), absmask));
        mB = Q6_Vh_vmax_VhVh(mB, Q6_V_vand_VV(*(const HVX_Vector *) (tp + (size_t) c0 * 1024 + 64), absmask));
    }
    for (int sh = 64; sh >= 4; sh >>= 1) { mA = Q6_Vh_vmax_VhVh(mA, Q6_V_vror_VR(mA, sh)); mB = Q6_Vh_vmax_VhVh(mB, Q6_V_vror_VR(mB, sh)); }   // lanes 0 / 1: max of the even / odd row
    // lanes 0 / 1 of mA / mB hold the row maxima as one 32-bit word each: extract to scalars and rebuild the inverse-scale vectors with word splats
    // (a vector store followed by scalar loads, and 64 scalar stores followed by a vector load, stall on store forwarding)
    const uint32_t wA = (uint32_t) Q6_R_vextract_VR(mA, 0), wB = (uint32_t) Q6_R_vextract_VR(mB, 0);
    const uint16_t ah[4] = { (uint16_t) wA, (uint16_t) (wA >> 16), (uint16_t) wB, (uint16_t) (wB >> 16) };
    uint32_t fb[4];
    for (int r = 0; r < 4; r++) {
        __fp16 amax_h; memcpy(&amax_h, &ah[r], sizeof(amax_h));
        const float a  = (float) amax_h;
        const float sc = a > 0.f ? a * g_a8w8_clip / 127.f : 1.f;
        rs[r] = sc;
        // 1/s can leave the fp16 range (amax < 0.002 -> 1/s > 65504): keep the factor as (1/s) * 2^-8 and undo the 2^-8 after the multiply.
        const __fp16 f = (__fp16) (1.f / sc * (1.f / 256.f));
        uint16_t fh; memcpy(&fh, &f, sizeof(fh));
        fb[r] = fh;
    }
    // even halfword lanes belong to the even row of the pair, odd lanes to the odd row
    const HVX_Vector vinvA = Q6_V_vsplat_R((int) (fb[0] | (fb[1] << 16))), vinvB = Q6_V_vsplat_R((int) (fb[2] | (fb[3] << 16)));
    const HVX_Vector sgn = Q6_Vh_vsplat_R(0x8000), half = Q6_Vh_vsplat_R(0x3800), c128 = Q6_Vh_vsplat_R(128), c256 = Q6_Vh_vsplat_R(0x5c00);
    #pragma unroll(2)
    for (uint32_t c0 = 0; c0 < n_k_tiles; c0++) {
        const HVX_Vector xA = *(const HVX_Vector *) (tp + (size_t) c0 * 1024), xB = *(const HVX_Vector *) (tp + (size_t) c0 * 1024 + 64);
        HVX_Vector yA = Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_vmpy_VhfVhf(xA, vinvA), c256);
        HVX_Vector yB = Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_vmpy_VhfVhf(xB, vinvB), c256);
        yA = Q6_Vhf_vadd_VhfVhf(yA, Q6_V_vor_VV(Q6_V_vand_VV(yA, sgn), half));
        yB = Q6_Vhf_vadd_VhfVhf(yB, Q6_V_vor_VV(Q6_V_vand_VV(yB, sgn), half));
        const HVX_Vector qA = Q6_Vh_vadd_VhVh(Q6_Vh_equals_Vhf(yA), c128);        // truncation toward zero of y +- 0.5 = round half away
        const HVX_Vector qB = Q6_Vh_vadd_VhVh(Q6_Vh_equals_Vhf(yB), c128);
        // deal: lanes 0..31 even row, 32..63 odd row; pack(B, A) = bytes 0..63 rows (A even, A odd), 64..127 rows (B even, B odd)
        *(HVX_Vector *) (q + (size_t) c0 * 2048 + row_in_block * 32) = Q6_Vub_vpack_VhVh_sat(Q6_Vh_vdeal_Vh(qB), Q6_Vh_vdeal_Vh(qA));
    }
}

// ---- 3. swv = s[n] (f32 lanes = columns), csum128 = 128 * sum_k w[k][n] (int32), rs = 64 row scales. out0 / out1 = 32-row fp16 tiles (rows 0..31 / 32..63), HMX output layout (vector r1/2: rows r1, r1+1).
// readout post-process: 4 byte planes of a 64x32 int32 accumulator -> fp16 output (two 32-row HMX output tiles) ---- planes: 4 x 2 KB at pl (plane k = byte k of the int32, readout exponents 24/16/8/0).
static inline void a8w8_post_group(const uint8_t * restrict pl, HVX_Vector swv, HVX_Vector csum128, const float * restrict rs,
                                   __fp16 * restrict out0, __fp16 * restrict out1, uint32_t grp) {
    const HVX_Vector v0 = *(const HVX_Vector *) (pl + 0 * 2048 + grp * 128);
    const HVX_Vector v1 = *(const HVX_Vector *) (pl + 1 * 2048 + grp * 128);
    const HVX_Vector v2 = *(const HVX_Vector *) (pl + 2 * 2048 + grp * 128);
    const HVX_Vector v3 = *(const HVX_Vector *) (pl + 3 * 2048 + grp * 128);
    HVX_VectorPair p0 = Q6_W_vshuff_VVR(v1, v0, -1);
    HVX_VectorPair p1 = Q6_W_vshuff_VVR(v3, v2, -1);
    HVX_VectorPair p2 = Q6_W_vshuff_VVR(Q6_V_lo_W(p1), Q6_V_lo_W(p0), -2);
    HVX_VectorPair p3 = Q6_W_vshuff_VVR(Q6_V_hi_W(p1), Q6_V_hi_W(p0), -2);
    const HVX_Vector a[4] = { Q6_Vw_vsub_VwVw(Q6_V_lo_W(p2), csum128), Q6_Vw_vsub_VwVw(Q6_V_hi_W(p2), csum128),
                              Q6_Vw_vsub_VwVw(Q6_V_lo_W(p3), csum128), Q6_Vw_vsub_VwVw(Q6_V_hi_W(p3), csum128) };
    HVX_Vector q[4];
    for (int r = 0; r < 4; r++) {
        HVX_Vector t = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(a[r]), swv);                       // acc * s[n]
        q[r] = Q6_Vqf32_vmpy_Vqf32Vqf32(t, Q6_Vqf32_vadd_VsfVsf(hvx_vec_splat_f32(rs[grp * 4 + r]), Q6_V_vzero()));   // * row scale
    }
    __fp16 * o = (grp < 8) ? out0 : out1;
    const uint32_t pi = (grp & 7) * 2;                                      // row pair index within the 32-row tile
    ((HVX_Vector *) o)[pi]     = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q[1], q[0]));
    ((HVX_Vector *) o)[pi + 1] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q[3], q[2]));
}

#endif // HMX_A8W8_H
