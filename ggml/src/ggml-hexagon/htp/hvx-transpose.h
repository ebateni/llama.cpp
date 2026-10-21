#ifndef HVX_TRANSPOSE_H
#define HVX_TRANSPOSE_H

#include "hvx-base.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"

// In-register transpose of a 32x32 tile of 32-bit words: on entry m[i] is row i, on exit m[i] is column i, i.e.
// lane j of m[i] holds lane i of the input's m[j]. tmp is scratch for 32 vectors. Shared by the GDN D-matrix packing (gated-delta-net-ops.c) and the conv-window CONCAT (concat-ops.c, GGML_HEXAGON_CONCAT_T_HVX).
static inline void hvx_transpose_32x32_words(HVX_Vector * restrict m, HVX_Vector * restrict tmp) {
    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    for (int b = 0; b < 32; b += 4) {
        HVX_VectorPair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_VectorPair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    for (int b = 0; b < 32; b += 8) {
        for (int i = 0; i < 4; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    for (int b = 0; b < 32; b += 16) {
        for (int i = 0; i < 8; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    for (int i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

#endif // HVX_TRANSPOSE_H
