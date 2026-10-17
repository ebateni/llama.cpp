// Host-side packing of Hadamard-rotated int8 weights for the HMX int8 matmul path (see htp/hmx-mm-kernels-int8.h).
//
// For a Q4_0 weight [n rows (output channels)][k] the rows are dequantized, multiplied by a fixed +-1 sign pattern, mixed by a
// normalized Walsh-Hadamard transform over blocks of HEX_INT8_WHT_BLOCK elements, quantized to int8 with one scale per row, and stored as
// records of 64 output channels:
//   [k/32 croutons of 2 KB: set0 (cols 0..31, 1 KB) then set1 (cols 32..63); byte (kk/4)*128 + c*4 + (kk%4)]
//   [HMX bias table, 2 x 256 B: 32 scale words (fp16 bits of 2^-shift * 2) then 32 words of -128*colsum + rounding]
//   [col_mul: 64 floats = row scale * 2^shift]
// The activation side applies the same sign pattern and transform on the DSP before its own int8 quantization.

#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <thread>
#include <vector>

#define HEX_INT8_WHT_BLOCK 512

// Zero-point (input bias) handling. The HMX bias high word carries the per-column integer added to the accumulator before the
// readout; the k = 12288 class needs |ibias| up to 1.36e6, beyond the ~4e5 the lab has verified, so the correction is split in two
// levels: the bias word carries the value truncated to HEX_INT8_BIAS_BITS and the exact remainder is applied on HVX as a per-column
// f32 term (col_add). With HEX_INT8_BIAS_BITS = 32 the remainder is exactly zero and this is bit-identical to a single-level bias;
// e56.py on the device decides the largest width the hardware honours (M33).
#ifndef HEX_INT8_BIAS_BITS
#define HEX_INT8_BIAS_BITS 32
#endif

static inline float hex_int8_sign(uint32_t k) {
    return ((k * 2654435761u) >> 20) & 1u ? -1.0f : 1.0f;
}

static inline size_t hex_int8_record_bytes(uint32_t k) {
    return (size_t) (k / 32) * 2048 + 2048;   // croutons, then bias tables (512 B) + col_mul (256 B), padded to a 2 KB multiple
}

static inline size_t hex_int8_extra_bytes(uint32_t k, uint32_t n, bool smooth = false) {
    // smoothing vector (1/s_k, k floats, 128 B padded) follows the records
    return (size_t) (n / 64) * hex_int8_record_bytes(k) + (smooth ? (((size_t) k * 4 + 127) & ~(size_t) 127) : 0);
}


// Scheme 1: the shipped W4A8 numerics, produced at LOAD time instead of on the fly. Each Q4_0 block has a fp16 scale d_g; with
// dmax = max_g|d_g| the row scale is s_n = dmax/15.875 and each block gets the Q4.4 ratio R_g = round(|d_g|/dmax*254), so the
// stored int8 is clip((q * sign(d_g) * R_g + 8) >> 4) with q in -8..7. This is exactly what the DSP's on-the-fly expand produces
// (hmx_int8_expand_q4_0_ref in htp/hmx-mm-kernels-int8.h), so a class measured with this scheme reports the W4A8 numbers without
// changing any kernel; the DDR cost is the same 1 B/weight as scheme 0 (only the on-the-fly expand saves those bytes).
static inline void hex_int8_pack_q44(uint8_t * dst, const uint8_t * src_rows, uint32_t n, uint32_t k, uint32_t reduce) {
    const size_t   rec_bytes = hex_int8_record_bytes(k);
    const uint32_t nkc = k / 32;
    for (uint32_t c = 0; c < n; ++c) {
        const uint8_t * rp = src_rows + (size_t) c * nkc * 18;
        float dmax = 0.0f;
        for (uint32_t b = 0; b < nkc; ++b) {
            uint16_t dh; memcpy(&dh, rp + b * 18, 2);
            const uint32_t sg = (dh >> 15) & 1, ex = (dh >> 10) & 31, ma = dh & 1023;
            float d = (ex == 0) ? ldexpf((float) ma, -24) : (ex == 31 ? 0.0f : ldexpf((float) (ma | 1024), (int) ex - 25));
            if (sg) { d = -d; }
            const float ad = fabsf(d);
            dmax = ad > dmax ? ad : dmax;
        }
        const float sn = dmax / 15.875f;
        uint8_t * rec = dst + (size_t) (c / 64) * rec_bytes;
        const uint32_t set = (c / 32) & 1, cc = c & 31;
        int64_t colsum = 0, abssum = 0;
        for (uint32_t b = 0; b < nkc; ++b) {
            uint16_t dh; memcpy(&dh, rp + b * 18, 2);
            const uint32_t sg = (dh >> 15) & 1, ex = (dh >> 10) & 31, ma = dh & 1023;
            float d = (ex == 0) ? ldexpf((float) ma, -24) : (ex == 31 ? 0.0f : ldexpf((float) (ma | 1024), (int) ex - 25));
            if (sg) { d = -d; }
            const int sgn = (d < 0.0f) ? -1 : 1;
            const int R = dmax > 0.0f ? (int) floorf(fabsf(d) / dmax * 254.0f + 0.5f) : 0;
            const uint8_t * qs = rp + b * 18 + 2;
            for (uint32_t i = 0; i < 16; ++i) {
                for (int half = 0; half < 2; ++half) {
                    const int q = (int) (half ? (qs[i] >> 4) : (qs[i] & 0xF)) - 8;
                    int w8 = (q * sgn * R + 8) >> 4;
                    w8 = w8 > 127 ? 127 : (w8 < -127 ? -127 : w8);
                    const uint32_t kk = b * 32 + i + (half ? 16 : 0);
                    rec[(size_t) (kk / 32) * 2048 + set * 1024 + ((kk % 32) / 4) * 128 + cc * 4 + (kk % 4)] = (uint8_t) (int8_t) w8;
                    colsum += w8; abssum += w8 < 0 ? -w8 : w8;
                }
            }
        }
        uint64_t bound = (uint64_t) (127 * abssum);
        uint32_t sh = 0;
        while ((bound >> sh) >= 32768u) { sh++; }
        sh = sh > reduce ? sh - reduce : 0;
        // half-LSB term makes the window truncation round-to-nearest; two-level bias as in the rotated packer
        const int64_t ibias_true = -128 * colsum + (sh ? (1LL << (sh - 1)) : 0);
        int64_t ib_hw = ibias_true;
        if (HEX_INT8_BIAS_BITS < 32) {
            const int64_t m = (int64_t) 1 << HEX_INT8_BIAS_BITS;
            ib_hw = ((ibias_true % m) + m + m / 2) % m - m / 2;
        }
        uint32_t * bias = (uint32_t *) (rec + (size_t) nkc * 2048 + set * 256);
        bias[cc]      = 0x4000u - (sh << 10);
        bias[32 + cc] = (uint32_t) (int32_t) ib_hw;
        float * cm = (float *) (rec + (size_t) nkc * 2048 + 512);
        cm[set * 32 + cc] = sn * (float) (1u << sh);
        float * ca = (float *) (rec + (size_t) nkc * 2048 + 768);
        ca[set * 32 + cc] = (float) (ibias_true - ib_hw) * sn;   // x row scale(s) is applied by the epilogue
    }
}

// dst: n/64 records. src_rows: n rows of k/32 blocks (18 bytes each: fp16 d, 16 bytes of nibbles; low nibble = element i, high = i + 16)
static inline void hex_int8_pack_rotated(uint8_t * dst, const uint8_t * src_rows, uint32_t n, uint32_t k, uint32_t reduce, bool rotate = true,
                                          const float * smooth = nullptr) {
    const size_t   rec_bytes = hex_int8_record_bytes(k);
    const uint32_t nkc = k / 32;
    const float    c_norm = 1.0f / sqrtf((float) HEX_INT8_WHT_BLOCK);

    std::vector<float> sgn(k);
    for (uint32_t i = 0; i < k; ++i) { sgn[i] = rotate ? hex_int8_sign(i) * c_norm : 1.0f; }

    auto work = [&](uint32_t r0, uint32_t r1) {
        std::vector<float> row(k);
        for (uint32_t c = r0; c < r1; ++c) {
            const uint8_t * rp = src_rows + (size_t) c * nkc * 18;
            for (uint32_t b = 0; b < nkc; ++b) {
                uint16_t dh; memcpy(&dh, rp + b * 18, 2);
                // fp16 -> f32
                const uint32_t sg = (dh >> 15) & 1, ex = (dh >> 10) & 31, ma = dh & 1023;
                float d;
                if (ex == 0)       { d = ldexpf((float) ma, -24); }
                else if (ex == 31) { d = 0.0f; }
                else               { d = ldexpf((float) (ma | 1024), (int) ex - 25); }
                if (sg) { d = -d; }
                const uint8_t * qs = rp + b * 18 + 2;
                for (uint32_t i = 0; i < 16; ++i) {
                    row[b * 32 + i]      = (float) ((int) (qs[i] & 0xF) - 8) * d;
                    row[b * 32 + i + 16] = (float) ((int) (qs[i] >> 4) - 8) * d;
                }
            }
            // smoothing: W' = W * s_k per input channel (the DSP divides the activation by s_k)
            if (smooth) { for (uint32_t i = 0; i < k; ++i) { row[i] *= smooth[i]; } }
            for (uint32_t i = 0; i < k; ++i) { row[i] *= sgn[i]; }
            for (uint32_t blk = 0; rotate && blk + HEX_INT8_WHT_BLOCK <= k; blk += HEX_INT8_WHT_BLOCK) {
                float * x = row.data() + blk;
                for (uint32_t h = 1; h < HEX_INT8_WHT_BLOCK; h <<= 1) {
                    for (uint32_t i = 0; i < HEX_INT8_WHT_BLOCK; i += 2 * h) {
                        for (uint32_t j = i; j < i + h; ++j) {
                            const float a = x[j], bb = x[j + h];
                            x[j] = a + bb; x[j + h] = a - bb;
                        }
                    }
                }
            }
            float amax = 0.0f;
            for (uint32_t i = 0; i < k; ++i) { amax = fmaxf(amax, fabsf(row[i])); }
            const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
            const float inv = 1.0f / s;

            uint8_t * rec = dst + (size_t) (c / 64) * rec_bytes;
            const uint32_t set = (c / 32) & 1, cc = c & 31;
            int64_t colsum = 0, abssum = 0;
            for (uint32_t i = 0; i < k; ++i) {
                int q = (int) lrintf(row[i] * inv);
                q = q > 127 ? 127 : (q < -127 ? -127 : q);
                const uint32_t kc = i / 32, kk = i % 32;
                rec[(size_t) kc * 2048 + set * 1024 + (kk / 4) * 128 + cc * 4 + (kk % 4)] = (uint8_t) (int8_t) q;
                colsum += q; abssum += q < 0 ? -q : q;
            }
            // window shift: |acc| <= 127 * sum|w8|; keep the int16 readout inside range, minus `reduce` bits of headroom
            const uint64_t bound = (uint64_t) (127 * abssum);
            uint32_t sh = 0;
            while ((bound >> sh) >= 32768u) { sh++; }
            sh = sh > reduce ? sh - reduce : 0;
            const int64_t ibias_true = (int64_t) (-128 * colsum) + (sh ? (1LL << (sh - 1)) : 0);
            int64_t ibias_hw = ibias_true;
            if (HEX_INT8_BIAS_BITS < 32) {
                const int64_t m = (int64_t) 1 << HEX_INT8_BIAS_BITS;
                ibias_hw = ((ibias_true % m) + m + m / 2) % m - m / 2;   // sign-extended low bits
            }
            uint32_t * bias = (uint32_t *) (rec + (size_t) nkc * 2048 + set * 256);
            bias[cc]      = 0x4000u - (sh << 10);
            bias[32 + cc] = (uint32_t) (int32_t) ibias_hw;
            float * cm = (float *) (rec + (size_t) nkc * 2048 + 512);
            cm[set * 32 + cc] = s * (float) (1u << sh);
            // col_add = (ibias_true - ibias_hw) / 2^sh * col_mul = delta * s; the epilogue multiplies it by the row scale(s)
            float * ca = (float *) (rec + (size_t) nkc * 2048 + 768);
            ca[set * 32 + cc] = (float) (ibias_true - ibias_hw) * s;
        }
    };

    const uint32_t nth = 8;
    std::vector<std::thread> th;
    for (uint32_t t = 0; t < nth; ++t) {
        const uint32_t r0 = (uint32_t) ((uint64_t) n * t / nth) & ~63u, r1 = t + 1 == nth ? n : (uint32_t) ((uint64_t) n * (t + 1) / nth) & ~63u;
        th.emplace_back(work, r0, r1);
    }
    for (auto & t : th) { t.join(); }
}
