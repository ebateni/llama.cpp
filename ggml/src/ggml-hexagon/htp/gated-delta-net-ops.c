#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <HAP_farf.h>

#include "hvx-base.h"
#include "hvx-copy.h"
#include "hvx-reduce.h"
#include "hvx-transpose.h"
#include "hvx-exp.h"
#include "hvx-log.h"
#include "hvx-sigmoid.h"
#include "dma-queue.h"
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hmx-utils.h"
#include "hmx-fa-kernels.h"
#include "hmx-queue.h"
#include "gated-delta-net-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

struct htp_gdn_context {
    struct htp_ops_context * octx;
    const struct htp_gdn_kernel_params * kparams;
    struct htp_gdn_vtcm_layout layout;
    struct htp_gdn_io_layout io;
    uint8_t * vtcm_base;
    uint32_t row_start;
    uint32_t nrows;
    uint32_t n_parts;   // state-row parts per head (htp_gdn_col_parts); work units = nrows * n_parts
};

static inline HVX_Vector gdn_mul_dot_f32(float * restrict dst, const HVX_Vector * restrict mul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd   = hvx_vmemu(dst + i * epv);
        HVX_Vector vm   = mul[i];
        HVX_Vector vdot = dot[i];
        HVX_Vector out  = hvx_vec_mul_f32_f32(vd, vm);
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm = mul[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vm);
        hvx_vec_store_u(dst + off, nloe * sizeof(float), out);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out, vdot), zero));
    }

    return hvx_vec_reduce_sum_f32(acc);
}

static inline HVX_Vector gdn_mul_scalar_dot_f32(float * restrict dst, HVX_Vector vmul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd   = hvx_vmemu(dst + i * epv);
        HVX_Vector vdot = dot[i];
        HVX_Vector out  = hvx_vec_mul_f32_f32(vd, vmul);
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vmul);
        hvx_vec_store_u(dst + off, nloe * sizeof(float), out);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out, vdot), zero));
    }

    return hvx_vec_reduce_sum_f32(acc);
}

static inline HVX_Vector gdn_add_scaled_dot_f32(float * restrict dst, const HVX_Vector * restrict src,
        HVX_Vector vscale, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd   = hvx_vmemu(dst + i * epv);
        HVX_Vector vs   = src[i];
        HVX_Vector vdot = dot[i];
        HVX_Vector out  = hvx_vec_add_f32_f32(vd, hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs   = src[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out = hvx_vec_add_f32_f32(hvx_vmemu(dst + off), hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vec_store_u(dst + off, nloe * sizeof(float), out);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out, vdot), zero));
    }

    return hvx_vec_reduce_sum_f32(acc);
}

static inline HVX_Vector gdn_mul_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3,
        const HVX_Vector * restrict mul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vm   = mul[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm = mul[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    return hvx_vec_reduce_sum_f32x4(acc);
}

static inline HVX_Vector gdn_mul_scalar_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3,
        HVX_Vector vmul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    return hvx_vec_reduce_sum_f32x4(acc);
}

static inline HVX_Vector gdn_add_scaled_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3,
        const HVX_Vector * restrict src, const float * restrict scale,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_splat_f32(scale[0]);
    const HVX_Vector scale1 = hvx_vec_splat_f32(scale[1]);
    const HVX_Vector scale2 = hvx_vec_splat_f32(scale[2]);
    const HVX_Vector scale3 = hvx_vec_splat_f32(scale[3]);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs   = src[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs   = src[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    return hvx_vec_reduce_sum_f32x4(acc);
}

static inline HVX_Vector gdn_mul_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        const HVX_Vector * restrict mul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vm   = mul[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vm);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vm);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vm);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm = mul[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vm);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vm);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vm);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, nloe * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, nloe * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, nloe * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, nloe * sizeof(float), out7);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_add_f32_f32(acc4, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_add_f32_f32(acc5, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_add_f32_f32(acc6, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_add_f32_f32(acc7, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_Vector rA = hvx_vec_reduce_sum_f32x4(accA);
    HVX_Vector rB = hvx_vec_reduce_sum_f32x4(accB);
    HVX_VectorPred q16 = Q6_Q_vsetq2_R(16);
    return Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

static inline HVX_Vector gdn_mul_scalar_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        HVX_Vector vmul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vmul);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vmul);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vmul);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vmul);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vmul);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vmul);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, nloe * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, nloe * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, nloe * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, nloe * sizeof(float), out7);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_add_f32_f32(acc4, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_add_f32_f32(acc5, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_add_f32_f32(acc6, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_add_f32_f32(acc7, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_Vector rA = hvx_vec_reduce_sum_f32x4(accA);
    HVX_Vector rB = hvx_vec_reduce_sum_f32x4(accB);
    HVX_VectorPred q16 = Q6_Q_vsetq2_R(16);
    return Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

static inline HVX_Vector gdn_add_scaled_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        const HVX_Vector * restrict src, const float * restrict scale,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_splat_f32(scale[0]);
    const HVX_Vector scale1 = hvx_vec_splat_f32(scale[1]);
    const HVX_Vector scale2 = hvx_vec_splat_f32(scale[2]);
    const HVX_Vector scale3 = hvx_vec_splat_f32(scale[3]);
    const HVX_Vector scale4 = hvx_vec_splat_f32(scale[4]);
    const HVX_Vector scale5 = hvx_vec_splat_f32(scale[5]);
    const HVX_Vector scale6 = hvx_vec_splat_f32(scale[6]);
    const HVX_Vector scale7 = hvx_vec_splat_f32(scale[7]);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs   = src[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_Vector out4 = hvx_vec_add_f32_f32(hvx_vmemu(dst4 + i * epv), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_Vector out5 = hvx_vec_add_f32_f32(hvx_vmemu(dst5 + i * epv), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_Vector out6 = hvx_vec_add_f32_f32(hvx_vmemu(dst6 + i * epv), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_Vector out7 = hvx_vec_add_f32_f32(hvx_vmemu(dst7 + i * epv), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs   = src[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_Vector out4 = hvx_vec_add_f32_f32(hvx_vmemu(dst4 + off), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_Vector out5 = hvx_vec_add_f32_f32(hvx_vmemu(dst5 + off), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_Vector out6 = hvx_vec_add_f32_f32(hvx_vmemu(dst6 + off), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_Vector out7 = hvx_vec_add_f32_f32(hvx_vmemu(dst7 + off), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, nloe * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, nloe * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, nloe * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, nloe * sizeof(float), out7);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_add_f32_f32(acc4, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_add_f32_f32(acc5, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_add_f32_f32(acc6, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_add_f32_f32(acc7, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_Vector rA = hvx_vec_reduce_sum_f32x4(accA);
    HVX_Vector rB = hvx_vec_reduce_sum_f32x4(accB);
    HVX_VectorPred q16 = Q6_Q_vsetq2_R(16);
    return Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

// One token of the recurrence on state rows [j0, j0 + nj) of one head. With j0 = 0 and nj = S_v this is the whole head.
// s_part holds those rows (row 0 = row j0); attn_out and v_t are the head's vectors, indexed by absolute row j. Rows are processed in the same 8-, 4- and 1-row blocks as before, so each row's arithmetic is unchanged; a part that is a multiple of 8 rows runs only 8-row blocks, like every row of the unsplit head.
static inline void gdn_step_kda_f32(
    float * restrict s_part,
    float * restrict attn_out,
    const float * restrict q_t,
    const float * restrict k_t,
    const float * restrict v_t,
    const float * restrict g_t,
    float beta_val,
    float scale,
    uint32_t S_v,
    uint32_t j0,
    uint32_t nj
) {
    const uint32_t epv  = 128 / sizeof(float);
    const uint32_t nvec = S_v / epv;
    const uint32_t nloe = S_v % epv;

    HVX_Vector vq[4];
    HVX_Vector vk[4];
    HVX_Vector vg[4];

    for (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
        vg[i] = hvx_vec_exp_f32(hvx_vmemu(g_t + i * epv));
    }
    if (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
        vg[nvec] = hvx_vec_exp_f32(hvx_vmemu(g_t + nvec * epv));
    }

    const HVX_Vector vbeta  = hvx_vec_splat_f32(beta_val);
    const HVX_Vector vscale = hvx_vec_splat_f32(scale);

    float delta[8] __attribute__((aligned(128)));

    uint32_t r = 0;                                  // local row inside the part
    for (; r + 8 <= nj; r += 8) {
        float * row0 = s_part + (uint64_t) (r + 0) * S_v;
        float * row1 = s_part + (uint64_t) (r + 1) * S_v;
        float * row2 = s_part + (uint64_t) (r + 2) * S_v;
        float * row3 = s_part + (uint64_t) (r + 3) * S_v;
        float * row4 = s_part + (uint64_t) (r + 4) * S_v;
        float * row5 = s_part + (uint64_t) (r + 5) * S_v;
        float * row6 = s_part + (uint64_t) (r + 6) * S_v;
        float * row7 = s_part + (uint64_t) (r + 7) * S_v;
        const uint32_t j = j0 + r;

        HVX_Vector vsums = gdn_mul_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                            vg, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_add_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j, 8 * sizeof(float), res_attn);
    }
    for (; r + 4 <= nj; r += 4) {
        float * row0 = s_part + (uint64_t) (r + 0) * S_v;
        float * row1 = s_part + (uint64_t) (r + 1) * S_v;
        float * row2 = s_part + (uint64_t) (r + 2) * S_v;
        float * row3 = s_part + (uint64_t) (r + 3) * S_v;
        const uint32_t j = j0 + r;

        HVX_Vector vsums = gdn_mul_dot4_f32(row0, row1, row2, row3, vg, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_add_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j, 4 * sizeof(float), res_attn);
    }
    for (; r < nj; ++r) {
        float * row = s_part + (uint64_t) r * S_v;
        const uint32_t j = j0 + r;
        HVX_Vector vsum = gdn_mul_dot_f32(row, vg, vk, S_v);
        HVX_Vector vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_Vector vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_Vector vres = gdn_add_scaled_dot_f32(row, vk, vdj, vq, S_v);
        attn_out[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

// Scalar-gate counterpart of gdn_step_kda_f32 (same row range contract).
static inline void gdn_step_scalar_f32(
    float * restrict s_part,
    float * restrict attn_out,
    const float * restrict q_t,
    const float * restrict k_t,
    const float * restrict v_t,
    const float * restrict g_t,
    float beta_val,
    float scale,
    uint32_t S_v,
    uint32_t j0,
    uint32_t nj
) {
    const uint32_t epv  = 128 / sizeof(float);
    const uint32_t nvec = S_v / epv;
    const uint32_t nloe = S_v % epv;

    HVX_Vector vq[4];
    HVX_Vector vk[4];

    for (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
    }
    if (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
    }

    const HVX_Vector vgate  = hvx_vec_exp_f32(hvx_vec_splat_f32(g_t[0]));
    const HVX_Vector vbeta  = hvx_vec_splat_f32(beta_val);
    const HVX_Vector vscale = hvx_vec_splat_f32(scale);

    float delta[8] __attribute__((aligned(128)));

    uint32_t r = 0;                                  // local row inside the part
    for (; r + 8 <= nj; r += 8) {
        float * row0 = s_part + (uint64_t) (r + 0) * S_v;
        float * row1 = s_part + (uint64_t) (r + 1) * S_v;
        float * row2 = s_part + (uint64_t) (r + 2) * S_v;
        float * row3 = s_part + (uint64_t) (r + 3) * S_v;
        float * row4 = s_part + (uint64_t) (r + 4) * S_v;
        float * row5 = s_part + (uint64_t) (r + 5) * S_v;
        float * row6 = s_part + (uint64_t) (r + 6) * S_v;
        float * row7 = s_part + (uint64_t) (r + 7) * S_v;
        const uint32_t j = j0 + r;

        HVX_Vector vsums = gdn_mul_scalar_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vgate, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_add_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j, 8 * sizeof(float), res_attn);
    }
    for (; r + 4 <= nj; r += 4) {
        float * row0 = s_part + (uint64_t) (r + 0) * S_v;
        float * row1 = s_part + (uint64_t) (r + 1) * S_v;
        float * row2 = s_part + (uint64_t) (r + 2) * S_v;
        float * row3 = s_part + (uint64_t) (r + 3) * S_v;
        const uint32_t j = j0 + r;

        HVX_Vector vsums = gdn_mul_scalar_dot4_f32(row0, row1, row2, row3, vgate, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_add_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j, 4 * sizeof(float), res_attn);
    }
    for (; r < nj; ++r) {
        float * row = s_part + (uint64_t) r * S_v;
        const uint32_t j = j0 + r;
        HVX_Vector vsum = gdn_mul_scalar_dot_f32(row, vgate, vk, S_v);
        HVX_Vector vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_Vector vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_Vector vres = gdn_add_scaled_dot_f32(row, vk, vdj, vq, S_v);
        attn_out[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

// --------------------------------------------------------------------------------------------------------------- HTP_GDN_SNAP_VERSION kernels (D2Snap). Requires S_v % 32 == 0 (no tail columns) and S_v % 8 == 0 (8-row blocks). ---------------------------------------------------------------------------------------------------------------
// The arithmetic is gdn_mul_scalar_dot8_f32 / gdn_add_scaled_dot8_f32 of gdn_step_scalar_f32, element for element and in the same order: pass 1 reads the previous version and stores nothing; pass 2 recomputes the decayed row from the previous version and writes the new version to a separate buffer.

// pass 1: kv sums of 8 rows, the decayed rows are not stored
static inline HVX_Vector gdn_dot8_nostore_f32(const float * restrict src0, const float * restrict src1,
        const float * restrict src2, const float * restrict src3, const float * restrict src4,
        const float * restrict src5, const float * restrict src6, const float * restrict src7,
        HVX_Vector vmul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();

    const uint32_t epv  = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(src0 + i * epv), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(src1 + i * epv), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(src2 + i * epv), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(src3 + i * epv), vmul);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(src4 + i * epv), vmul);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(src5 + i * epv), vmul);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(src6 + i * epv), vmul);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(src7 + i * epv), vmul);

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_Vector rA = hvx_vec_reduce_sum_f32x4(accA);
    HVX_Vector rB = hvx_vec_reduce_sum_f32x4(accB);
    HVX_VectorPred q16 = Q6_Q_vsetq2_R(16);
    return Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

// pass 2: new version of 8 rows: dst = decay(src) + k * delta (stored), and the q dot products of the new rows
static inline HVX_Vector gdn_upd_dot8_f32(const float * restrict src0, const float * restrict src1,
        const float * restrict src2, const float * restrict src3, const float * restrict src4,
        const float * restrict src5, const float * restrict src6, const float * restrict src7,
        float * restrict dst0, float * restrict dst1, float * restrict dst2, float * restrict dst3,
        float * restrict dst4, float * restrict dst5, float * restrict dst6, float * restrict dst7,
        HVX_Vector vmul, const HVX_Vector * restrict vk, const float * restrict scale,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_splat_f32(scale[0]);
    const HVX_Vector scale1 = hvx_vec_splat_f32(scale[1]);
    const HVX_Vector scale2 = hvx_vec_splat_f32(scale[2]);
    const HVX_Vector scale3 = hvx_vec_splat_f32(scale[3]);
    const HVX_Vector scale4 = hvx_vec_splat_f32(scale[4]);
    const HVX_Vector scale5 = hvx_vec_splat_f32(scale[5]);
    const HVX_Vector scale6 = hvx_vec_splat_f32(scale[6]);
    const HVX_Vector scale7 = hvx_vec_splat_f32(scale[7]);

    const uint32_t epv  = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs   = vk[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(src0 + i * epv), vmul), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(src1 + i * epv), vmul), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(src2 + i * epv), vmul), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(src3 + i * epv), vmul), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_Vector out4 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(src4 + i * epv), vmul), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_Vector out5 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(src5 + i * epv), vmul), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_Vector out6 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(src6 + i * epv), vmul), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_Vector out7 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(src7 + i * epv), vmul), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_Vector rA = hvx_vec_reduce_sum_f32x4(accA);
    HVX_Vector rB = hvx_vec_reduce_sum_f32x4(accB);
    HVX_VectorPred q16 = Q6_Q_vsetq2_R(16);
    return Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

// one token of one unit (rows j0 .. j0 + nj - 1 of a head), out of place: cur = decay(prev) + k * delta,
// attn_out[j0 + j] = (cur . q) * scale. prev and cur hold the unit's nj rows (nj % 8 == 0); the q, k vectors span all S_v columns.
static inline void gdn_step_ver_f32(
    const float * restrict prev,
    float * restrict cur,
    float * restrict attn_out,
    const float * restrict q_t,
    const float * restrict k_t,
    const float * restrict v_t,
    const float * restrict g_t,
    float beta_val,
    float scale,
    uint32_t S_v,
    uint32_t j0,
    uint32_t nj
) {
    const uint32_t epv  = 128 / sizeof(float);
    const uint32_t nvec = S_v / epv;

    HVX_Vector vq[4];
    HVX_Vector vk[4];
    for (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
    }

    const HVX_Vector vgate  = hvx_vec_exp_f32(hvx_vec_splat_f32(g_t[0]));
    const HVX_Vector vbeta  = hvx_vec_splat_f32(beta_val);
    const HVX_Vector vscale = hvx_vec_splat_f32(scale);

    float delta[8] __attribute__((aligned(128)));

    for (uint32_t j = 0; j < nj; j += 8) {
        const float * p0 = prev + (uint64_t) (j + 0) * S_v;
        const float * p1 = prev + (uint64_t) (j + 1) * S_v;
        const float * p2 = prev + (uint64_t) (j + 2) * S_v;
        const float * p3 = prev + (uint64_t) (j + 3) * S_v;
        const float * p4 = prev + (uint64_t) (j + 4) * S_v;
        const float * p5 = prev + (uint64_t) (j + 5) * S_v;
        const float * p6 = prev + (uint64_t) (j + 6) * S_v;
        const float * p7 = prev + (uint64_t) (j + 7) * S_v;
        float * c0 = cur + (uint64_t) (j + 0) * S_v;
        float * c1 = cur + (uint64_t) (j + 1) * S_v;
        float * c2 = cur + (uint64_t) (j + 2) * S_v;
        float * c3 = cur + (uint64_t) (j + 3) * S_v;
        float * c4 = cur + (uint64_t) (j + 4) * S_v;
        float * c5 = cur + (uint64_t) (j + 5) * S_v;
        float * c6 = cur + (uint64_t) (j + 6) * S_v;
        float * c7 = cur + (uint64_t) (j + 7) * S_v;

        HVX_Vector vsums = gdn_dot8_nostore_f32(p0, p1, p2, p3, p4, p5, p6, p7, vgate, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j0 + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_upd_dot8_f32(p0, p1, p2, p3, p4, p5, p6, p7,
                                            c0, c1, c2, c3, c4, c5, c6, c7,
                                            vgate, vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j0 + j, 8 * sizeof(float), res_attn);
    }
}

static void gated_delta_net_f32_pp_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_gdn_context * gctx = (struct htp_gdn_context *) data;
    struct htp_ops_context * octx = gctx->octx;
    const struct htp_gdn_kernel_params * kparams = gctx->kparams;

    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t S_v      = kparams->S_v;
    const uint32_t H        = kparams->H;
    const uint32_t n_tokens = kparams->n_tokens;
    const uint32_t n_seqs   = kparams->n_seqs;
    const uint32_t K        = kparams->K;
    // unit = (head, part): part p holds state rows [p * nj, (p + 1) * nj) of its head (nj = S_v / n_parts)
    const uint32_t n_parts  = gctx->n_parts;
    const uint32_t nj       = S_v / n_parts;
    const uint32_t n_el_u   = nj * S_v;
    const uint32_t n_units  = gctx->nrows * n_parts;
    // interval snapshot writes: op_params[3] = d0, op_params[4] = n (n == 0 keeps the default full write).
    const int64_t snap_d0 = (int32_t) octx->op_params[3];
    const int64_t snap_n  = (int32_t) octx->op_params[4];

    if (ith >= n_units) {
        return;
    }

    const struct htp_tensor * dst_cache = octx->dsts[1];
    const float scale = kparams->scale;
    float * dst_base       = (float *) (uintptr_t) dst->data;
    float * state_out_base = dst_cache ? (float *) (uintptr_t) dst_cache->data : (dst_base + S_v * H * n_tokens * n_seqs);

    dma_queue * dma_q = octx->ctx->dma[ith];
    const struct htp_gdn_vtcm_layout * layout = &gctx->layout;
    float * s_work[2];
    s_work[0] = (float *) (gctx->vtcm_base + layout->bytes_per_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / sizeof(float);

    const struct fastdiv_values * fd_H   = &kparams->div_H;
    const struct fastdiv_values * fd_q1  = &kparams->div_q1;
    const struct fastdiv_values * fd_k1  = &kparams->div_k1;
    const struct fastdiv_values * fd_rq3 = &kparams->div_rq3;
    const struct fastdiv_values * fd_rk3 = &kparams->div_rk3;

    const uint32_t state_seq_stride = kparams->state_seq_stride;
    const uint64_t state_size_per_snap = (uint64_t) kparams->state_size_per_snap;
    const dma_addr_t state_out_dma_base = dst_cache ? dst_cache->data : (dst->data + S_v * H * n_tokens * n_seqs * sizeof(float));

    // this thread takes units ith, ith + nth, ...; prefetch the first two of them
    uint32_t u_prefetch = ith;
    int spad_idx = 0;

    // Prefetch preamble (up to 2 units)
    for (int step = 0; step < 2 && u_prefetch < n_units; step++) {
        const uint32_t ir   = gctx->row_start + u_prefetch / n_parts;
        const uint32_t part = u_prefetch % n_parts;
        const uint32_t piv1 = fastmodulo(ir, H, fd_H);
        const uint32_t piv3 = fastdiv(ir, fd_H);
        const uint64_t part_el = (uint64_t) part * n_el_u;
        dma_addr_t ps_in  = state->data + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v + part_el) * sizeof(float);
        dma_addr_t ps_out = state_out_dma_base + (((uint64_t) piv3 * H + piv1) * S_v * S_v + part_el) * sizeof(float);

        // Push dummy write-back
        dma_queue_push(dma_q, dma_make_data(ps_out, s_work[spad_idx]),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), 0);

        // Push fetch
        dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), nj);

        u_prefetch += nth;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    int curr_spad_idx = 0;
    for (uint32_t u = ith; u < n_units; u += nth) {
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        float * s_work_curr = s_work[curr_spad_idx];

        const uint32_t ir   = gctx->row_start + u / n_parts;
        const uint32_t part = u % n_parts;
        const uint32_t j0   = part * nj;                 // first state row of this unit inside its head
        const uint64_t part_el = (uint64_t) j0 * S_v;    // elements of this part inside a head

        const uint32_t iv1 = fastmodulo(ir, H, fd_H);
        const uint32_t iv3 = fastdiv(ir, fd_H);

        const uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        const uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        const uint32_t iq3 = fastdiv(iv3, fd_rq3);
        const uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_out  = state_out_dma_base + (((uint64_t) iv3 * H + iv1) * S_v * S_v + part_el) * sizeof(float);
        float * attn_data = dst_base + ((uint64_t) iv3 * n_tokens * H + iv1) * S_v;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        for (uint32_t t = 0; t < n_tokens; ++t) {
            const float * q_t = (const float *) ((const uint8_t *) (uintptr_t) q->data +
                    (uint64_t) iq3 * q->nb[3] + (uint64_t) t * q->nb[2] + (uint64_t) iq1 * q->nb[1]);
            const float * k_t = (const float *) ((const uint8_t *) (uintptr_t) k->data +
                    (uint64_t) ik3 * k->nb[3] + (uint64_t) t * k->nb[2] + (uint64_t) ik1 * k->nb[1]);
            const float * v_t = (const float *) ((const uint8_t *) (uintptr_t) v->data +
                    (uint64_t) iv3 * v->nb[3] + (uint64_t) t * v->nb[2] + (uint64_t) iv1 * v->nb[1]);
            const float * g_t = (const float *) ((const uint8_t *) (uintptr_t) g->data +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) t * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
            const float beta_val = *(const float *) ((const uint8_t *) (uintptr_t) beta->data +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) t * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);

            if (kparams->kda) {
                gdn_step_kda_f32(s_work_curr, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v, j0, nj);
            } else {
                gdn_step_scalar_f32(s_work_curr, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v, j0, nj);
            }

            if (K > 1) {
                // slot = distance d = n_tokens-1-t; slot 0 (final state) is written by the write-back after the loop.
                // Interval mode (op_params[4] = n != 0): only d in [snap_d0, snap_d0+snap_n) is written.
                const int64_t d = (int64_t) n_tokens - 1 - (int64_t) t;
                const bool in_range = (snap_n == 0) || (d >= snap_d0 && d < snap_d0 + snap_n);
                if (d > 0 && d < (int64_t) K && in_range) {
                    float * curr_state_o = state_out_base + (uint64_t) d * state_size_per_snap + ((uint64_t) iv3 * H + iv1) * S_v * S_v + part_el;
                    hvx_copy_f32_uu((uint8_t *) curr_state_o, (const uint8_t *) s_work_curr, n_el_u);
                }
            }

            attn_data += (uint64_t) S_v * H;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back (this part)
        dma_queue_push(dma_q, dma_make_data(s_out, s_work_curr),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), nj);

        // Prefetch next unit (if any)
        if (u_prefetch < n_units) {
            const uint32_t ir_p   = gctx->row_start + u_prefetch / n_parts;
            const uint32_t part_p = u_prefetch % n_parts;
            const uint32_t piv1 = fastmodulo(ir_p, H, fd_H);
            const uint32_t piv3 = fastdiv(ir_p, fd_H);
            dma_addr_t ps_in = state->data + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v
                                              + (uint64_t) part_p * n_el_u) * sizeof(float);

            dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                           S_v * sizeof(float), S_v * sizeof(float),
                           S_v * sizeof(float), nj);

            u_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_queue_flush(dma_q);
}

static void gated_delta_net_f32_tg_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_gdn_context * gctx = (struct htp_gdn_context *) data;
    struct htp_ops_context * octx = gctx->octx;
    const struct htp_gdn_kernel_params * kparams = gctx->kparams;

    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t S_v      = kparams->S_v;
    const uint32_t H        = kparams->H;
    const uint32_t n_seqs   = kparams->n_seqs;
    const uint32_t row_end  = gctx->row_start + gctx->nrows;

    if (ith >= gctx->nrows) {
        return;
    }

    const struct htp_tensor * dst_cache = octx->dsts[1];
    const float scale = kparams->scale;
    float * dst_base  = (float *) (uintptr_t) dst->data;

    dma_queue * dma_q = octx->ctx->dma[ith];
    const struct htp_gdn_vtcm_layout * layout = &gctx->layout;
    float * s_work[2];
    s_work[0] = (float *) (gctx->vtcm_base + layout->bytes_per_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / sizeof(float);

    const struct fastdiv_values * fd_H   = &kparams->div_H;
    const struct fastdiv_values * fd_q1  = &kparams->div_q1;
    const struct fastdiv_values * fd_k1  = &kparams->div_k1;
    const struct fastdiv_values * fd_rq3 = &kparams->div_rq3;
    const struct fastdiv_values * fd_rk3 = &kparams->div_rk3;

    const uint32_t state_seq_stride = kparams->state_seq_stride;
    const dma_addr_t state_out_dma_base = dst_cache ? dst_cache->data : (dst->data + S_v * H * n_seqs * sizeof(float));

    uint32_t ir_prefetch = gctx->row_start + ith;
    int spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    for (int step = 0; step < 2 && ir_prefetch < row_end; step++) {
        const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_in  = state->data + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * sizeof(float);
        dma_addr_t ps_out = state_out_dma_base + ((uint64_t) piv3 * H + piv1) * S_v * S_v * sizeof(float);

        // Push dummy write-back
        dma_queue_push(dma_q, dma_make_data(ps_out, s_work[spad_idx]),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), 0);

        // Push fetch
        dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    int curr_spad_idx = 0;
    for (uint32_t ir = gctx->row_start + ith; ir < row_end; ir += nth) {
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        float * s_work_curr = s_work[curr_spad_idx];

        const uint32_t iv1 = fastmodulo(ir, H, fd_H);
        const uint32_t iv3 = fastdiv(ir, fd_H);

        const uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        const uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        const uint32_t iq3 = fastdiv(iv3, fd_rq3);
        const uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_out  = state_out_dma_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v * sizeof(float);
        float * attn_data = dst_base + ((uint64_t) iv3 * H + iv1) * S_v;

        const float * q_t = (const float *) ((const uint8_t *) (uintptr_t) q->data +
                (uint64_t) iq3 * q->nb[3] + (uint64_t) iq1 * q->nb[1]);
        const float * k_t = (const float *) ((const uint8_t *) (uintptr_t) k->data +
                (uint64_t) ik3 * k->nb[3] + (uint64_t) ik1 * k->nb[1]);
        const float * v_t = (const float *) ((const uint8_t *) (uintptr_t) v->data +
                (uint64_t) iv3 * v->nb[3] + (uint64_t) iv1 * v->nb[1]);
        const float * g_t = (const float *) ((const uint8_t *) (uintptr_t) g->data +
                (uint64_t) iv3 * g->nb[3] + (uint64_t) iv1 * g->nb[1]);
        const float beta_val = *(const float *) ((const uint8_t *) (uintptr_t) beta->data +
                (uint64_t) iv3 * beta->nb[3] + (uint64_t) iv1 * beta->nb[1]);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        if (kparams->kda) {
            gdn_step_kda_f32(s_work_curr, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v, 0, S_v);
        } else {
            gdn_step_scalar_f32(s_work_curr, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v, 0, S_v);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_queue_push(dma_q, dma_make_data(s_out, s_work_curr),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);

        // Prefetch next block (if any)
        if (ir_prefetch < row_end) {
            const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_in = state->data + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * sizeof(float);

            dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                           S_v * sizeof(float), S_v * sizeof(float),
                           S_v * sizeof(float), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_queue_flush(dma_q);
}

// HTP_GDN_OPT_UP_DMA token-gen path (upstream 9b47b8ad4, c6c532daa): state, q, k, v, g, beta and out move through
// HTP_GDN_HVX_N_BUFS DMA buffers per thread. Used for n_tokens == 1 on the SYNC snapshot path only.
static void gated_delta_net_f32_tg_dma_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_gdn_context * gctx = (struct htp_gdn_context *) data;
    struct htp_ops_context * octx = gctx->octx;
    const struct htp_gdn_kernel_params * kparams = gctx->kparams;

    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t S_v      = kparams->S_v;
    const uint32_t H        = kparams->H;
    const uint32_t n_seqs   = kparams->n_seqs;
    const uint32_t row_end  = gctx->row_start + gctx->nrows;

    if (ith >= gctx->nrows) {
        return;
    }

    const struct htp_tensor * dst_cache = octx->dsts[1];
    const float scale = kparams->scale;

    dma_queue * dma_q = octx->ctx->dma[ith];
    const struct htp_gdn_vtcm_layout * layout = &gctx->layout;

    uint8_t * thread_base = (uint8_t *) gctx->vtcm_base + layout->bytes_per_thread * ith;

    float * s_work[HTP_GDN_HVX_N_BUFS];
    float * q_spad[HTP_GDN_HVX_N_BUFS];
    float * k_spad[HTP_GDN_HVX_N_BUFS];
    float * v_spad[HTP_GDN_HVX_N_BUFS];
    float * g_spad[HTP_GDN_HVX_N_BUFS];
    float * b_spad[HTP_GDN_HVX_N_BUFS];
    float * out_spad[HTP_GDN_HVX_N_BUFS];

    for (int b = 0; b < HTP_GDN_HVX_N_BUFS; ++b) {
        s_work[b]   = (float *) (thread_base + layout->off_state + b * layout->state_aligned);
        q_spad[b]   = (float *) (thread_base + layout->off_q     + b * layout->vec_aligned);
        k_spad[b]   = (float *) (thread_base + layout->off_k     + b * layout->vec_aligned);
        v_spad[b]   = (float *) (thread_base + layout->off_v     + b * layout->vec_aligned);
        g_spad[b]   = (float *) (thread_base + layout->off_g     + b * layout->vec_aligned);
        b_spad[b]   = (float *) (thread_base + layout->off_b     + b * 128);
        out_spad[b] = (float *) (thread_base + layout->off_out   + b * layout->vec_aligned);
    }

    const struct fastdiv_values * fd_H   = &kparams->div_H;
    const struct fastdiv_values * fd_q1  = &kparams->div_q1;
    const struct fastdiv_values * fd_k1  = &kparams->div_k1;
    const struct fastdiv_values * fd_rq3 = &kparams->div_rq3;
    const struct fastdiv_values * fd_rk3 = &kparams->div_rk3;

    const uint32_t state_seq_stride = kparams->state_seq_stride;
    const dma_addr_t state_out_dma_base = dst_cache ? dst_cache->data : (dst->data + S_v * H * n_seqs * sizeof(float));

    const size_t vec_bytes = (size_t) S_v * sizeof(float);
    const size_t g_bytes   = kparams->kda ? vec_bytes : sizeof(float);
    const size_t b_bytes   = sizeof(float);

    uint32_t ir_prefetch = gctx->row_start + ith;
    int spad_idx = 0;

    // Prefetch preamble (1 step)
    for (int step = 0; step < HTP_GDN_HVX_N_BUFS - 1 && ir_prefetch < row_end; step++) {
        const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);

        const uint32_t piq1 = fastmodulo(piv1, q->ne[1], fd_q1);
        const uint32_t pik1 = fastmodulo(piv1, k->ne[1], fd_k1);
        const uint32_t piq3 = fastdiv(piv3, fd_rq3);
        const uint32_t pik3 = fastdiv(piv3, fd_rk3);

        dma_addr_t ps_in  = state->data + (piv3 * state_seq_stride + piv1 * S_v * S_v) * sizeof(float);
        dma_addr_t ps_out = state_out_dma_base + (piv3 * H + piv1) * S_v * S_v * sizeof(float);
        dma_addr_t pa_out = dst->data + (piv3 * H + piv1) * S_v * sizeof(float);

        dma_addr_t pq_in = q->data + piq3 * q->nb[3] + piq1 * q->nb[1];
        dma_addr_t pk_in = k->data + pik3 * k->nb[3] + pik1 * k->nb[1];
        dma_addr_t pv_in = v->data + piv3 * v->nb[3] + piv1 * v->nb[1];
        dma_addr_t pg_in = g->data + piv3 * g->nb[3] + piv1 * g->nb[1];
        dma_addr_t pb_in = beta->data + piv3 * beta->nb[3] + piv1 * beta->nb[1];

        // Push dummy write-backs
        dma_queue_push(dma_q, dma_make_data(ps_out, s_work[spad_idx]),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), 0);
        dma_queue_push(dma_q, dma_make_data(pa_out, out_spad[spad_idx]),
                       vec_bytes, layout->vec_aligned,
                       vec_bytes, 0);

        // Push fetches
        dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);
        dma_queue_push(dma_q, dma_make_data(q_spad[spad_idx], pq_in),
                       layout->vec_aligned, 0,
                       vec_bytes, 1);
        dma_queue_push(dma_q, dma_make_data(k_spad[spad_idx], pk_in),
                       layout->vec_aligned, 0,
                       vec_bytes, 1);
        dma_queue_push(dma_q, dma_make_data(v_spad[spad_idx], pv_in),
                       layout->vec_aligned, 0,
                       vec_bytes, 1);
        dma_queue_push(dma_q, dma_make_data(g_spad[spad_idx], pg_in),
                       layout->vec_aligned, 0,
                       g_bytes, 1);
        dma_queue_push(dma_q, dma_make_data(b_spad[spad_idx], pb_in),
                       128, 0,
                       b_bytes, 1);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    int curr_spad_idx = 0;
    for (uint32_t ir = gctx->row_start + ith; ir < row_end; ir += nth) {
        // Pop write-backs and fetches for current slot
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        float * s_work_curr  = s_work[curr_spad_idx];
        float * out_curr     = out_spad[curr_spad_idx];
        const float * q_curr = q_spad[curr_spad_idx];
        const float * k_curr = k_spad[curr_spad_idx];
        const float * v_curr = v_spad[curr_spad_idx];
        const float * g_curr = g_spad[curr_spad_idx];
        const float beta_val = *b_spad[curr_spad_idx];

        const uint32_t iv1 = fastmodulo(ir, H, fd_H);
        const uint32_t iv3 = fastdiv(ir, fd_H);

        dma_addr_t s_out = state_out_dma_base + (iv3 * H + iv1) * S_v * S_v * sizeof(float);
        dma_addr_t a_out = dst->data + (iv3 * H + iv1) * S_v * sizeof(float);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        if (kparams->kda) {
            gdn_step_kda_f32(s_work_curr, out_curr, q_curr, k_curr, v_curr, g_curr, beta_val, scale, S_v, 0, S_v);
        } else {
            gdn_step_scalar_f32(s_work_curr, out_curr, q_curr, k_curr, v_curr, g_curr, beta_val, scale, S_v, 0, S_v);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-backs
        dma_queue_push(dma_q, dma_make_data(s_out, s_work_curr),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);
        dma_queue_push(dma_q, dma_make_data(a_out, out_curr),
                       vec_bytes, layout->vec_aligned,
                       vec_bytes, 1);

        // Prefetch next block (if any)
        if (ir_prefetch < row_end) {
            const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);

            const uint32_t piq1 = fastmodulo(piv1, q->ne[1], fd_q1);
            const uint32_t pik1 = fastmodulo(piv1, k->ne[1], fd_k1);
            const uint32_t piq3 = fastdiv(piv3, fd_rq3);
            const uint32_t pik3 = fastdiv(piv3, fd_rk3);

            dma_addr_t ps_in = state->data + (piv3 * state_seq_stride + piv1 * S_v * S_v) * sizeof(float);
            dma_addr_t pq_in = q->data + piq3 * q->nb[3] + piq1 * q->nb[1];
            dma_addr_t pk_in = k->data + pik3 * k->nb[3] + pik1 * k->nb[1];
            dma_addr_t pv_in = v->data + piv3 * v->nb[3] + piv1 * v->nb[1];
            dma_addr_t pg_in = g->data + piv3 * g->nb[3] + piv1 * g->nb[1];
            dma_addr_t pb_in = beta->data + piv3 * beta->nb[3] + piv1 * beta->nb[1];

            dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                           S_v * sizeof(float), S_v * sizeof(float),
                           S_v * sizeof(float), S_v);
            dma_queue_push(dma_q, dma_make_data(q_spad[spad_idx], pq_in),
                           layout->vec_aligned, 0,
                           vec_bytes, 1);
            dma_queue_push(dma_q, dma_make_data(k_spad[spad_idx], pk_in),
                           layout->vec_aligned, 0,
                           vec_bytes, 1);
            dma_queue_push(dma_q, dma_make_data(v_spad[spad_idx], pv_in),
                           layout->vec_aligned, 0,
                           vec_bytes, 1);
            dma_queue_push(dma_q, dma_make_data(g_spad[spad_idx], pg_in),
                           layout->vec_aligned, 0,
                           g_bytes, 1);
            dma_queue_push(dma_q, dma_make_data(b_spad[spad_idx], pb_in),
                           128, 0,
                           b_bytes, 1);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_queue_flush(dma_q);
}

struct htp_gdn_hmx_gemm_task {
    const __fp16 * row_tiles;
    const __fp16 * col_tiles;
    __fp16 *       out_tiles;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
    uint32_t       dot_stride;
    uint8_t *      hmx_scales;
};

static inline void htp_gdn_hmx_gemm_exec(const struct htp_gdn_hmx_gemm_task * task) {
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)task->hmx_scales));

    const size_t dot_stride = task->dot_stride;
    for (uint32_t r = 0; r < task->n_row_tiles; ++r) {
        const __fp16 * r_tiles = task->row_tiles + r * dot_stride;
        const __fp16 * c_tiles = task->col_tiles;
        __fp16 *       o_tile  = task->out_tiles + r * task->n_col_tiles * HMX_FP16_TILE_N_ELMS;

        for (uint32_t c = 0; c < task->n_col_tiles; ++c) {
            hmx_fa_qk_dot_tile(r_tiles, c_tiles, o_tile, task->n_dot_tiles);
            c_tiles += dot_stride;
            o_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

static void htp_gdn_hmx_gemm_worker(void * data) {
    htp_gdn_hmx_gemm_exec((const struct htp_gdn_hmx_gemm_task *) data);
}

// HTP_GDN_OPT_HMX_BATCH: one queue descriptor runs the task of every head of the batch for one GEMM kind
struct htp_gdn_hmx_gemm_batch {
    const struct htp_gdn_hmx_gemm_task * tasks;
    uint32_t                             n;
};

static void htp_gdn_hmx_gemm_batch_worker(void * data) {
    const struct htp_gdn_hmx_gemm_batch * b = (const struct htp_gdn_hmx_gemm_batch *) data;
    for (uint32_t i = 0; i < b->n; ++i) {
        htp_gdn_hmx_gemm_exec(&b->tasks[i]);
    }
}

// Tile rows [h_lo, h_hi) of a 64 x 64 tile set -> chunk rows 32 h .. 32 h + 31 (one vector per row pair half).
static inline void gdn_unpack_64x64_tiles_rows(
    HVX_Vector * restrict rows,
    const __fp16 * restrict tiles,
    uint32_t h_lo,
    uint32_t h_hi
) {
    for (uint32_t h = h_lo; h < h_hi; ++h) {
        const HVX_Vector * t0 = (const HVX_Vector *) (tiles + (2 * h + 0) * HMX_FP16_TILE_N_ELMS);
        const HVX_Vector * t1 = (const HVX_Vector *) (tiles + (2 * h + 1) * HMX_FP16_TILE_N_ELMS);

        for (uint32_t r = 0; r < 16; ++r) {
            HVX_VectorPair vp = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
            rows[32 * h + 2 * r + 0] = Q6_V_lo_W(vp);
            rows[32 * h + 2 * r + 1] = Q6_V_hi_W(vp);
        }
    }
}

// Tile rows [h_lo, h_hi) of the 64 x 64 rows (32 rows per tile row) -> the 64 x 64 tiles (inverse of the unpack below).
static inline void gdn_pack_64x64_rows_to_tiles(
    __fp16 * restrict tiles,
    const HVX_Vector * restrict rows,
    uint32_t h_lo,
    uint32_t h_hi
) {
    for (uint32_t h = h_lo; h < h_hi; ++h) {
        HVX_Vector * t0 = (HVX_Vector *) (tiles + (2 * h + 0) * HMX_FP16_TILE_N_ELMS);
        HVX_Vector * t1 = (HVX_Vector *) (tiles + (2 * h + 1) * HMX_FP16_TILE_N_ELMS);

        for (uint32_t r = 0; r < 16; ++r) {
            HVX_VectorPair vp = Q6_W_vshuff_VVR(rows[32 * h + 2 * r + 1], rows[32 * h + 2 * r + 0], -2);
            t0[r] = Q6_V_lo_W(vp);
            t1[r] = Q6_V_hi_W(vp);
        }
    }
}

// fp16 tiles -> fp32 rows of tile rows [r0_lo, r0_hi) (32 rows each); S_v / 32 tiles per tile row. Used for the chunk
// tiles (64 rows) and the state tiles (S_v rows) alike.
static inline void gdn_unpack_tiles_to_f32_rows(
    float * restrict dst_f32,
    const __fp16 * restrict tiles,
    uint32_t S_v,
    uint32_t r0_lo,
    uint32_t r0_hi
) {
    const uint32_t n_col_tiles = S_v / 32;
    for (uint32_t r0 = r0_lo; r0 < r0_hi; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_VectorPair p0 = hvx_vec_f16_to_f32(Q6_V_lo_W(vp01));
                HVX_VectorPair p1 = hvx_vec_f16_to_f32(Q6_V_hi_W(vp01));

                float * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                float * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

// fp16 tiles -> fp16 rows: columns [64 d_lo, 64 d_hi) (64-column halves d of S_v) of all 64 rows.
static inline void gdn_unpack_64xS_tiles_to_f16_cols(
    __fp16 * restrict dst_f16,
    const __fp16 * restrict tiles,
    uint32_t S_v,
    uint32_t d_lo,
    uint32_t d_hi
) {
    const uint32_t n_col_tiles = S_v / 32;
    for (uint32_t r0 = 0; r0 < 2; ++r0) {
        for (uint32_t d = d_lo; d < d_hi; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                __fp16 * out0 = dst_f16 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                __fp16 * out1 = dst_f16 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0) = Q6_V_lo_W(vp01);
                hvx_vmem(out1) = Q6_V_hi_W(vp01);
            }
        }
    }
}

// Rows [r_lo, r_hi) (even bounds) of an f32 chunk: f16 row tiles, row tiles scaled by the per-row scales (when
// dst_prime_tiles is set) and row-major f16 (when dst_f16 is set).
static inline void gdn_f32_to_hmx_row_tiles_and_f16_rows(
    __fp16 * restrict dst_tiles,
    __fp16 * restrict dst_prime_tiles,
    __fp16 * restrict dst_f16,
    const float * restrict src,
    const __fp16 * restrict scale_per_row,
    uint32_t r_lo,
    uint32_t r_hi,
    uint32_t n_cols
) {
    const uint32_t n_col_tiles = n_cols / 32;
    const uint32_t * scale_pairs = (const uint32_t *) scale_per_row;

    for (uint32_t r = r_lo; r < r_hi; r += 2) {
        uint32_t r0 = r / 32;
        uint32_t r1 = (r % 32) / 2;
        const float * p0 = src + (r + 0) * n_cols;
        const float * p1 = src + (r + 1) * n_cols;

        HVX_Vector v_scale;
        if (dst_prime_tiles) {
            uint32_t scale_pair = scale_pairs ? scale_pairs[r / 2] : 0x3c003c00;
            v_scale = Q6_V_vsplat_R(scale_pair);
        }

        for (uint32_t c = 0; c < n_col_tiles; c += 2) {
            HVX_Vector v0_0 = hvx_vmem(p0 + (c + 0) * 32);
            HVX_Vector v1_0 = hvx_vmem(p1 + (c + 0) * 32);
            HVX_Vector v0_1 = hvx_vmem(p0 + (c + 1) * 32);
            HVX_Vector v1_1 = hvx_vmem(p1 + (c + 1) * 32);

            HVX_Vector vh0 = hvx_vec_f32_to_f16_shuff(v0_0, v1_0);
            HVX_Vector vh1 = hvx_vec_f32_to_f16_shuff(v0_1, v1_1);
            __fp16 * tile0 = dst_tiles + (r0 * n_col_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
            __fp16 * tile1 = dst_tiles + (r0 * n_col_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
            ((HVX_Vector *) tile0)[r1] = vh0;
            ((HVX_Vector *) tile1)[r1] = vh1;

            if (dst_prime_tiles) {
                HVX_Vector vh0_s = hvx_vec_mul_f16_f16(vh0, v_scale);
                HVX_Vector vh1_s = hvx_vec_mul_f16_f16(vh1, v_scale);
                __fp16 * tile0_s = dst_prime_tiles + (r0 * n_col_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
                __fp16 * tile1_s = dst_prime_tiles + (r0 * n_col_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
                ((HVX_Vector *) tile0_s)[r1] = vh0_s;
                ((HVX_Vector *) tile1_s)[r1] = vh1_s;
            }

            if (dst_f16) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(vh1, vh0, -2);
                hvx_vmem(dst_f16 + (r + 0) * n_cols + c * 32) = Q6_V_lo_W(vp01);
                hvx_vmem(dst_f16 + (r + 1) * n_cols + c * 32) = Q6_V_hi_W(vp01);
            }
        }
    }
}

// Interleave the row-major f16 rows [r_lo, r_hi) of columns [c_lo, c_hi) (64-column steps) into column tiles: tile
// (dim tile c / 32, row tile r / 32), n_row_tiles row tiles per dim tile. Full range = hmx_interleave_cols_to_tiles.
static inline void gdn_interleave_cols_to_tiles_range(
    __fp16 * restrict tiles_out,
    const __fp16 * restrict src,
    uint32_t n_rows,
    uint32_t c_lo,
    uint32_t c_hi,
    size_t src_stride,
    uint32_t n_row_tiles,
    uint32_t r_lo,
    uint32_t r_hi
) {
    const size_t tile_stride_elms = (size_t) n_row_tiles * HMX_FP16_TILE_N_ELMS;

    for (uint32_t r = r_lo; r < r_hi; r += 2) {
        const bool next_row_valid = (r + 1) < r_hi && (r + 1) < n_rows;

        const HVX_Vector * pv_in0 = (const HVX_Vector *) (src + r * src_stride + c_lo);
        const HVX_Vector * pv_in1 = next_row_valid ? (const HVX_Vector *) (src + (r + 1) * src_stride + c_lo) : NULL;

        const uint32_t r0      = r / HMX_FP16_TILE_N_ROWS;
        const uint32_t r1_half = (r % HMX_FP16_TILE_N_ROWS) / 2;

        __fp16 *     tb0     = tiles_out + (size_t) (c_lo / HMX_FP16_TILE_N_COLS) * tile_stride_elms +
                               (size_t) r0 * HMX_FP16_TILE_N_ELMS;
        __fp16 *     tb1     = tb0 + tile_stride_elms;
        const size_t tb_step = 2 * tile_stride_elms;

        if (pv_in1) {
            for (uint32_t c = c_lo; c < c_hi; c += 64) {
                HVX_Vector     v0             = *pv_in0++;
                HVX_Vector     v1             = *pv_in1++;
                HVX_VectorPair vp             = Q6_W_vshuff_VVR(v1, v0, -2);
                ((HVX_Vector *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_Vector *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_step;
                tb1 += tb_step;
            }
        } else {
            const HVX_Vector vzero = Q6_V_vzero();
            for (uint32_t c = c_lo; c < c_hi; c += 64) {
                HVX_Vector     v0             = *pv_in0++;
                HVX_VectorPair vp             = Q6_W_vshuff_VVR(vzero, v0, -2);
                ((HVX_Vector *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_Vector *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_step;
                tb1 += tb_step;
            }
        }
    }
}

// d^T tiles of the column halves [ch_lo, ch_hi) of d (S_v / 64 halves of 64 columns); m and tmp: 32 scratch vectors each.
static inline void gdn_pack_d_t_row_tiles_range(
    __fp16 * restrict dst_tiles,
    const __fp16 * restrict src_d,
    uint32_t S_v,
    HVX_Vector * restrict m,
    HVX_Vector * restrict tmp,
    uint32_t ch_lo,
    uint32_t ch_hi
) {
    for (uint32_t col_half = ch_lo; col_half < ch_hi; ++col_half) {
        uint32_t r0_base = col_half * 2;
        for (uint32_t c0 = 0; c0 < 2; ++c0) {
            for (uint32_t s_local = 0; s_local < 32; ++s_local) {
                uint32_t s = c0 * 32 + s_local;
                m[s_local] = hvx_vmem(src_d + s * S_v + col_half * 64);
            }

            hvx_transpose_32x32_words(m, tmp);

            uint32_t tile0_idx = (r0_base + 0) * 2 + c0;
            uint32_t tile1_idx = (r0_base + 1) * 2 + c0;
            HVX_Vector * t0 = (HVX_Vector *)(dst_tiles + tile0_idx * HMX_FP16_TILE_N_ELMS);
            HVX_Vector * t1 = (HVX_Vector *)(dst_tiles + tile1_idx * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                t0[r] = m[r];
                t1[r] = m[16 + r];
            }
        }
    }
}

// Forward substitution of one 32 x 32 unit-lower block: a_rows[t] = row t of the block inverse, one 64-lane f16 vector per row. Every row still receives its terms in increasing k, each product and difference rounded to f16 as in the row-by-row recurrence, so the result is bit-identical.
// The chains are named vectors: an array of vector accumulators is not kept in registers. Rows are solved in groups of 8: the terms of the rows before a group go to its 8 rows as 8 independent chains, then the group's own rows follow in order.
#define GDN_SOLVE_SUB(acc, lv, av) (acc) = hvx_vec_sub_f16_f16((acc), hvx_vec_mul_f16_f16(Q6_Vh_vsplat_R(lv), (av)))

static inline HVX_Vector gdn_solve_fin(uint32_t t, HVX_Vector acc, HVX_VectorPred q_mask64, HVX_Vector v_one_f16) {
    HVX_VectorPred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
    return Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, acc));
}

static inline void gdn_solve_unit_lower_32(HVX_Vector * restrict a_rows, const uint16_t (*restrict l)[32],
                                           HVX_VectorPred q_mask64, HVX_Vector v_one_f16, uint32_t band) {
    for (uint32_t g = 0; g < 32; g += 8) {
        HVX_Vector c0 = Q6_V_vzero(), c1 = Q6_V_vzero(), c2 = Q6_V_vzero(), c3 = Q6_V_vzero(), c4 = Q6_V_vzero(), c5 = Q6_V_vzero(), c6 = Q6_V_vzero(), c7 = Q6_V_vzero();
        // LOSSY_GDN SOLVE_B16 / B8: only the `band` rows before the group feed the group (band = 32: all rows, exact)
        for (uint32_t k = g > band ? g - band : 0; k < g; ++k) {
            const HVX_Vector a_k = a_rows[k];
            GDN_SOLVE_SUB(c0, l[g + 0][k], a_k);
            GDN_SOLVE_SUB(c1, l[g + 1][k], a_k);
            GDN_SOLVE_SUB(c2, l[g + 2][k], a_k);
            GDN_SOLVE_SUB(c3, l[g + 3][k], a_k);
            GDN_SOLVE_SUB(c4, l[g + 4][k], a_k);
            GDN_SOLVE_SUB(c5, l[g + 5][k], a_k);
            GDN_SOLVE_SUB(c6, l[g + 6][k], a_k);
            GDN_SOLVE_SUB(c7, l[g + 7][k], a_k);
        }
        const HVX_Vector a0 = gdn_solve_fin(g + 0, c0, q_mask64, v_one_f16);
        a_rows[g + 0] = a0;
        GDN_SOLVE_SUB(c1, l[g + 1][g + 0], a0);
        GDN_SOLVE_SUB(c2, l[g + 2][g + 0], a0);
        GDN_SOLVE_SUB(c3, l[g + 3][g + 0], a0);
        GDN_SOLVE_SUB(c4, l[g + 4][g + 0], a0);
        GDN_SOLVE_SUB(c5, l[g + 5][g + 0], a0);
        GDN_SOLVE_SUB(c6, l[g + 6][g + 0], a0);
        GDN_SOLVE_SUB(c7, l[g + 7][g + 0], a0);
        const HVX_Vector a1 = gdn_solve_fin(g + 1, c1, q_mask64, v_one_f16);
        a_rows[g + 1] = a1;
        GDN_SOLVE_SUB(c2, l[g + 2][g + 1], a1);
        GDN_SOLVE_SUB(c3, l[g + 3][g + 1], a1);
        GDN_SOLVE_SUB(c4, l[g + 4][g + 1], a1);
        GDN_SOLVE_SUB(c5, l[g + 5][g + 1], a1);
        GDN_SOLVE_SUB(c6, l[g + 6][g + 1], a1);
        GDN_SOLVE_SUB(c7, l[g + 7][g + 1], a1);
        const HVX_Vector a2 = gdn_solve_fin(g + 2, c2, q_mask64, v_one_f16);
        a_rows[g + 2] = a2;
        GDN_SOLVE_SUB(c3, l[g + 3][g + 2], a2);
        GDN_SOLVE_SUB(c4, l[g + 4][g + 2], a2);
        GDN_SOLVE_SUB(c5, l[g + 5][g + 2], a2);
        GDN_SOLVE_SUB(c6, l[g + 6][g + 2], a2);
        GDN_SOLVE_SUB(c7, l[g + 7][g + 2], a2);
        const HVX_Vector a3 = gdn_solve_fin(g + 3, c3, q_mask64, v_one_f16);
        a_rows[g + 3] = a3;
        GDN_SOLVE_SUB(c4, l[g + 4][g + 3], a3);
        GDN_SOLVE_SUB(c5, l[g + 5][g + 3], a3);
        GDN_SOLVE_SUB(c6, l[g + 6][g + 3], a3);
        GDN_SOLVE_SUB(c7, l[g + 7][g + 3], a3);
        const HVX_Vector a4 = gdn_solve_fin(g + 4, c4, q_mask64, v_one_f16);
        a_rows[g + 4] = a4;
        GDN_SOLVE_SUB(c5, l[g + 5][g + 4], a4);
        GDN_SOLVE_SUB(c6, l[g + 6][g + 4], a4);
        GDN_SOLVE_SUB(c7, l[g + 7][g + 4], a4);
        const HVX_Vector a5 = gdn_solve_fin(g + 5, c5, q_mask64, v_one_f16);
        a_rows[g + 5] = a5;
        GDN_SOLVE_SUB(c6, l[g + 6][g + 5], a5);
        GDN_SOLVE_SUB(c7, l[g + 7][g + 5], a5);
        const HVX_Vector a6 = gdn_solve_fin(g + 6, c6, q_mask64, v_one_f16);
        a_rows[g + 6] = a6;
        GDN_SOLVE_SUB(c7, l[g + 7][g + 6], a6);
        const HVX_Vector a7 = gdn_solve_fin(g + 7, c7, q_mask64, v_one_f16);
        a_rows[g + 7] = a7;
    }
}

// Diagonal blocks of the 64 x 64 inverse. Each block reads only its own rows of rows_kk and decay_m, so the two blocks can run as two jobs.
// side 0: block 0 (tokens 0..31; writes tile00 and zeroes tile01); side 1: block 1 (tokens 32..63; writes tile11, the off-diagonal tile l10 and neg_a11); side 2: both.
static __attribute__((noinline)) void gdn_build_inv_l_blocks(
    __fp16 * restrict inv_row_tiles,
    const HVX_Vector * restrict rows_kk,
    const __fp16 * restrict decay_m,
    const float * restrict beta,
    __fp16 * restrict l10_tile,
    __fp16 * restrict neg_a11_tile,
    uint32_t side,
    uint32_t band
) {
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq2_R(64);

    uint16_t beta_u16[64] __attribute__((aligned(128)));
    uint16_t l00[32][32]  __attribute__((aligned(128)));
    uint16_t l11[32][32]  __attribute__((aligned(128)));

    HVX_Vector * restrict p_l00 = (HVX_Vector *) l00;
    HVX_Vector * restrict p_l11 = (HVX_Vector *) l11;
    HVX_Vector * restrict p_l10_tile = (HVX_Vector *) l10_tile;

    HVX_Vector * restrict tile00 = (HVX_Vector *) (inv_row_tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * restrict tile01 = (HVX_Vector *) (inv_row_tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * restrict tile11 = (HVX_Vector *) (inv_row_tiles + 3 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * restrict p_neg_a11 = (HVX_Vector *) neg_a11_tile;

    hvx_vmem(beta_u16) = hvx_vec_f32_to_f16(hvx_vmem(beta + 0), hvx_vmem(beta + 32));

    if (side != 1) {
        for (uint32_t r = 0; r < 16; ++r) {
            tile01[r] = Q6_V_vzero();
        }
        for (uint32_t r = 0; r < 16; ++r) {
            uint32_t t0 = 2 * r;
            uint32_t t1 = t0 + 1;

            HVX_Vector v_d0 = hvx_vmem(decay_m + t0 * 64);
            HVX_Vector v_d1 = hvx_vmem(decay_m + t1 * 64);
            HVX_Vector v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
            HVX_Vector v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

            HVX_Vector r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t0], v_d0), v_b0);
            HVX_Vector r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t1], v_d1), v_b1);

            p_l00[r] = Q6_V_vmux_QVV(q_mask64, r0, Q6_V_vror_VR(r1, 64));
        }
    }

    if (side != 0) {
        for (uint32_t r = 0; r < 16; ++r) {
            uint32_t t0 = 32 + 2 * r;
            uint32_t t1 = t0 + 1;

            HVX_Vector v_d0 = hvx_vmem(decay_m + t0 * 64);
            HVX_Vector v_d1 = hvx_vmem(decay_m + t1 * 64);
            HVX_Vector v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
            HVX_Vector v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

            HVX_Vector r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t0], v_d0), v_b0);
            HVX_Vector r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t1], v_d1), v_b1);

            HVX_VectorPair vp_l10 = Q6_W_vshuff_VVR(r1, r0, -2);
            p_l10_tile[r] = Q6_V_lo_W(vp_l10);
            p_l11[r] = Q6_V_vmux_QVV(q_mask64, Q6_V_vror_VR(r0, 64), r1);
        }
    }

    HVX_Vector a_rows[32];
    if (side != 1) {
        gdn_solve_unit_lower_32(a_rows, l00, q_mask64, v_one_f16, band);
        for (uint32_t r = 0; r < 16; ++r) {
            HVX_VectorPair vp = Q6_W_vshuff_VVR(a_rows[2 * r + 1], a_rows[2 * r + 0], -2);
            tile00[r] = Q6_V_lo_W(vp);
        }
    }

    if (side != 0) {
        gdn_solve_unit_lower_32(a_rows, l11, q_mask64, v_one_f16, band);
        for (uint32_t r = 0; r < 16; ++r) {
            HVX_VectorPair vp = Q6_W_vshuff_VVR(a_rows[2 * r + 1], a_rows[2 * r + 0], -2);
            tile11[r] = Q6_V_lo_W(vp);

            HVX_Vector n0 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_rows[2 * r + 0]);
            HVX_Vector n1 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_rows[2 * r + 1]);
            HVX_VectorPair vp_neg = Q6_W_vshuff_VVR(n1, n0, -2);
            p_neg_a11[r] = Q6_V_lo_W(vp_neg);
        }
    }
}


static inline void gdn_dma_push_chunk_gb(
    dma_queue * dma_q,
    float * vtcm_g_raw,
    float * vtcm_b_raw,
    const struct htp_tensor * g,
    const struct htp_tensor * beta,
    uint32_t iv3,
    uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_size,
    uint32_t n_batch
) {
    const dma_addr_t g_dma    = g->data + (uint64_t) iv3 * g->nb[3] + (uint64_t) t_chunk * g->nb[2] + (uint64_t) iv1 * g->nb[1];
    const dma_addr_t beta_dma = beta->data + (uint64_t) iv3 * beta->nb[3] + (uint64_t) t_chunk * beta->nb[2] + (uint64_t) iv1 * beta->nb[1];
    const uint32_t row_bytes  = n_batch * sizeof(float);

    dma_queue_push(dma_q, dma_make_data(vtcm_g_raw, g_dma), row_bytes, g->nb[2], row_bytes, chunk_size);
    dma_queue_push(dma_q, dma_make_data(vtcm_b_raw, beta_dma), row_bytes, beta->nb[2], row_bytes, chunk_size);
}

// QKSHARE: g and beta rows span all H heads of the sequence (the column of a head is its head index)
static inline void gdn_dma_push_chunk_gb_full(
    dma_queue * dma_q,
    float * vtcm_g_raw,
    float * vtcm_b_raw,
    const struct htp_tensor * g,
    const struct htp_tensor * beta,
    uint32_t iv3,
    uint32_t t_chunk,
    uint32_t chunk_size,
    uint32_t H
) {
    const dma_addr_t g_dma    = g->data + (uint64_t) iv3 * g->nb[3] + (uint64_t) t_chunk * g->nb[2];
    const dma_addr_t beta_dma = beta->data + (uint64_t) iv3 * beta->nb[3] + (uint64_t) t_chunk * beta->nb[2];
    const uint32_t row_bytes  = H * sizeof(float);

    dma_queue_push(dma_q, dma_make_data(vtcm_g_raw, g_dma), row_bytes, g->nb[2], row_bytes, chunk_size);
    dma_queue_push(dma_q, dma_make_data(vtcm_b_raw, beta_dma), row_bytes, beta->nb[2], row_bytes, chunk_size);
}

// Same result as hmx_interleave_rows_to_tiles(dst, src, n_rows, k, k, r_lo, r_hi) (src: row-major f16, row stride k, k % 64 == 0; r_lo, r_hi multiples of 32) without vscatter: every block of 32 rows x 64 columns is a 32 x 32 word transpose (a word = the column pair of one row), which gives 32 tile rows = the two k tiles of that column half.
// Pure data movement, bit-identical. scr: 64 vectors of VTCM scratch.
static inline void gdn_rows_to_col_tiles_range(__fp16 * restrict dst, const __fp16 * restrict src, uint32_t k,
                                               uint32_t r_lo, uint32_t r_hi, HVX_Vector * restrict scr) {
    HVX_Vector * restrict m   = scr;
    HVX_Vector * restrict tmp = scr + 32;
    const uint32_t n_k_tiles = k / 32;
    for (uint32_t ct = r_lo / 32; ct < r_hi / 32; ++ct) {
        for (uint32_t d = 0; d < k / 64; ++d) {
            for (uint32_t j = 0; j < 32; ++j) {
                m[j] = hvx_vmem(src + (ct * 32 + j) * k + d * 64);
            }
            hvx_transpose_32x32_words(m, tmp);
            HVX_Vector * t0 = (HVX_Vector *) (dst + (size_t) (ct * n_k_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            HVX_Vector * t1 = (HVX_Vector *) (dst + (size_t) (ct * n_k_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);
            for (uint32_t r = 0; r < 16; ++r) {
                t0[r] = m[r];
                t1[r] = m[16 + r];
            }
        }
    }
}

// State rows [j_lo, j_hi) of a head: f32 -> f16 row-major copy, then the column tiles of those rows.
static inline void gdn_pack_s_col_tiles_rows(
    __fp16 * restrict vtcm_s_col_tiles,
    __fp16 * restrict vtcm_s_f16,
    const float * restrict vtcm_s_state,
    uint32_t S_v,
    uint32_t j_lo,
    uint32_t j_hi,
    HVX_Vector * restrict scr
) {
    for (uint32_t j = j_lo; j < j_hi; ++j) {
        for (uint32_t i = 0; i < S_v; i += 64) {
            HVX_Vector v0 = hvx_vmem(vtcm_s_state + j * S_v + i + 0);
            HVX_Vector v1 = (i + 32 < S_v) ? hvx_vmem(vtcm_s_state + j * S_v + i + 32) : Q6_V_vzero();
            hvx_vmem(vtcm_s_f16 + j * S_v + i) = hvx_vec_f32_to_f16(v0, v1);
        }
    }
    gdn_rows_to_col_tiles_range(vtcm_s_col_tiles, vtcm_s_f16, S_v, j_lo, j_hi, scr);
}

struct htp_gdn_head_ptrs {
    float *  s_state;
    __fp16 * s_f16;
    __fp16 * s_col_tiles;
    __fp16 * s_update_tiles;

    float * q_f32[2];
    float * k_f32[2];
    float * v_f32[2];
    float * g_f32[2];
    float * b_f32[2];
    float * o_f32[2];

    __fp16 * k_f16;
    __fp16 * v_prime_f16;
    __fp16 * delta_f16;
    __fp16 * d_f16;

    __fp16 * q_row_tiles;
    __fp16 * q_prime_row_tiles;
    __fp16 * k_row_tiles;
    __fp16 * k_col_tiles;
    __fp16 * k_prime_row_tiles;
    __fp16 * k_col_tiles_64x128;
    __fp16 * kk_tiles;
    __fp16 * qk_tiles;
    __fp16 * v_inter_tiles;
    __fp16 * o_inter_tiles;
    __fp16 * inv_row_tiles;
    __fp16 * a_row_tiles;
    __fp16 * v_prime_col_tiles;
    __fp16 * delta_tiles;
    __fp16 * delta_col_tiles;
    __fp16 * o_intra_tiles;
    __fp16 * d_row_tiles;

    __fp16 * gamma;
    float *  lambda_init;
    __fp16 * lambda_init_f16;
    __fp16 * decay_m;
    __fp16 * decay_a;

    HVX_Vector * rows_kk;
    HVX_Vector * rows_qk;
    HVX_Vector * rows_inv;
    HVX_Vector * rows_a;

    HVX_Vector * vtcm_m;
    HVX_Vector * vtcm_tmp;

    uint32_t iv1;
    uint32_t iv3;
    uint32_t iq1;
    uint32_t ik1;
    uint32_t iq3;
    uint32_t ik3;
    dma_addr_t state_in_dma;
    dma_addr_t state_out_dma;
    uint32_t gb_col;   // column of this head in the raw g/beta rows (batch slot, or head index with QKSHARE)
    uint8_t  own_q;    // 1: this head loads the q of its Q/K head; 0: it reuses an earlier slot's q buffers (QKSHARE)
    uint8_t  own_k;    // same for k
};

// q and k are loaded once per Q/K head (owners, QKSHARE); v is loaded per head. Returns the descriptor count.
static inline uint32_t gdn_dma_push_chunk_inputs_slot(
    dma_queue * dma_q,
    const struct htp_gdn_head_ptrs * head,
    uint32_t buf,
    const struct htp_tensor * q,
    const struct htp_tensor * k,
    const struct htp_tensor * v,
    uint32_t t_chunk,
    uint32_t chunk_size,
    uint32_t S_v
) {
    const dma_addr_t q_dma = q->data + (uint64_t) head->iq3 * q->nb[3] + (uint64_t) t_chunk * q->nb[2] + (uint64_t) head->iq1 * q->nb[1];
    const dma_addr_t k_dma = k->data + (uint64_t) head->ik3 * k->nb[3] + (uint64_t) t_chunk * k->nb[2] + (uint64_t) head->ik1 * k->nb[1];
    const dma_addr_t v_dma = v->data + (uint64_t) head->iv3 * v->nb[3] + (uint64_t) t_chunk * v->nb[2] + (uint64_t) head->iv1 * v->nb[1];

    uint32_t n = 0;
    if (head->own_q) {
        dma_queue_push(dma_q, dma_make_data(head->q_f32[buf], q_dma), S_v * sizeof(float), q->nb[2], S_v * sizeof(float), chunk_size);
        n++;
    }
    if (head->own_k) {
        dma_queue_push(dma_q, dma_make_data(head->k_f32[buf], k_dma), S_v * sizeof(float), k->nb[2], S_v * sizeof(float), chunk_size);
        n++;
    }
    dma_queue_push(dma_q, dma_make_data(head->v_f32[buf], v_dma), S_v * sizeof(float), v->nb[2], S_v * sizeof(float), chunk_size);
    return n + 1;
}

// Layout part of a head's pointers: depends only on the VTCM layout and the batch slot h, not on the op.
static inline void gdn_init_head_layout(
    struct htp_gdn_head_ptrs * head,
    const struct htp_gdn_hmx_vtcm_layout * L,
    uint8_t * vtcm_base,
    uint32_t h
) {
    head->s_state        = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_s_state + h * L->st_state_f32);
    head->s_f16          = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_s_f16 + h * L->st_state_f16);
    head->s_col_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_s_col_tiles + h * L->st_state_tiles);
    head->s_update_tiles = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_s_update_tiles + h * L->st_state_tiles);

    head->q_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_q_f32[0] + h * L->st_dma_chunk);
    head->q_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L->off_q_f32[1] + h * L->st_dma_chunk) : head->q_f32[0];
    head->k_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_k_f32[0] + h * L->st_dma_chunk);
    head->k_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L->off_k_f32[1] + h * L->st_dma_chunk) : head->k_f32[0];
    head->v_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_v_f32[0] + h * L->st_dma_chunk);
    head->v_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L->off_v_f32[1] + h * L->st_dma_chunk) : head->v_f32[0];
    head->g_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_g_f32[0] + h * L->st_dma_scalar);
    head->g_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L->off_g_f32[1] + h * L->st_dma_scalar) : head->g_f32[0];
    head->b_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_b_f32[0] + h * L->st_dma_scalar);
    head->b_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L->off_b_f32[1] + h * L->st_dma_scalar) : head->b_f32[0];
    head->o_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_o_f32[0] + h * L->st_dma_chunk);
    head->o_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L->off_o_f32[1] + h * L->st_dma_chunk) : head->o_f32[0];

    head->k_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_f16 + h * L->st_act_f16);
    head->v_prime_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_v_prime_f16 + h * L->st_act_f16);
    head->delta_f16   = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_delta_f16 + h * L->st_act_f16);
    head->d_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_d_f16 + h * L->st_act_f16);

    head->q_row_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_q_row_tiles + h * L->st_tile_64xSv);
    head->q_prime_row_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_q_prime_row_tiles + h * L->st_tile_64xSv);
    head->k_row_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_row_tiles + h * L->st_tile_64xSv);
    head->k_col_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_col_tiles + h * L->st_tile_64xSv);
    head->k_prime_row_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_prime_row_tiles + h * L->st_tile_64xSv);
    head->k_col_tiles_64x128 = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_col_tiles_64x128 + h * L->st_tile_64xSv);
    head->kk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_kk_tiles + h * L->st_tile_64x64);
    head->qk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_qk_tiles + h * L->st_tile_64x64);
    head->v_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_v_inter_tiles + h * L->st_tile_64xSv);
    head->o_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_o_inter_tiles + h * L->st_tile_64xSv);
    head->inv_row_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_inv_row_tiles + h * L->st_tile_64x64);
    head->a_row_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_a_row_tiles + h * L->st_tile_64x64);
    head->v_prime_col_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_v_prime_col_tiles + h * L->st_tile_64xSv);
    head->delta_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_delta_tiles + h * L->st_tile_64xSv);
    head->delta_col_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_delta_col_tiles + h * L->st_tile_64xSv);
    head->o_intra_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_o_intra_tiles + h * L->st_tile_64xSv);
    head->d_row_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_d_row_tiles + h * L->st_tile_64xSv);

    head->gamma           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_gamma + h * L->st_gamma);
    head->lambda_init_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_gamma + h * L->st_gamma + 128);
    head->lambda_init     = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_lambda_init + h * L->st_gamma);
    head->decay_m     = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_decay_m + h * L->st_decay);
    head->decay_a     = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_decay_a + h * L->st_decay);

    head->rows_kk  = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_rows_kk + h * L->st_row_vecs);
    head->rows_qk  = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_rows_qk + h * L->st_row_vecs);
    head->rows_inv = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_rows_inv + h * L->st_row_vecs);
    head->rows_a   = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_rows_a + h * L->st_row_vecs);

    head->vtcm_m   = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_thread_scratch + h * L->st_scratch);
    head->vtcm_tmp = head->vtcm_m + 32;
}

// Per-op part: tensor coordinates, DMA addresses and the q/k ownership flags of head slot h.
static inline void gdn_init_head_op(
    struct htp_gdn_head_ptrs * head,
    const struct htp_gdn_hmx_vtcm_layout * L,
    uint8_t * vtcm_base,
    uint32_t h,
    uint32_t iv1,
    uint32_t iv3,
    const struct htp_tensor * q,
    const struct htp_tensor * k,
    const struct htp_tensor * state,
    const struct htp_tensor * dst,
    const struct htp_tensor * dst_cache,
    const struct htp_gdn_kernel_params * kparams,
    uint32_t S_v,
    uint32_t H,
    uint32_t n_tokens
) {
    head->s_state = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_s_state + h * L->st_state_f32); // destination of the state-in DMA
    head->iv1    = iv1;
    head->gb_col = h;
    head->own_q  = 1;
    head->own_k  = 1;
    head->iv3 = iv3;
    head->iq1 = fastmodulo(head->iv1, q->ne[1], &kparams->div_q1);
    head->ik1 = fastmodulo(head->iv1, k->ne[1], &kparams->div_k1);
    head->iq3 = fastdiv(head->iv3, &kparams->div_rq3);
    head->ik3 = fastdiv(head->iv3, &kparams->div_rk3);

    head->state_in_dma = state->data +
        ((uint64_t) head->iv3 * kparams->state_seq_stride + (uint64_t) head->iv1 * S_v * S_v) * sizeof(float);

    head->state_out_dma = dst_cache ?
        (dst_cache->data + ((uint64_t) head->iv3 * H + head->iv1) * S_v * S_v * sizeof(float)) :
        (dst->data + ((uint64_t) S_v * H * n_tokens * kparams->n_seqs + (uint64_t) (head->iv3 * H + head->iv1) * S_v * S_v) * sizeof(float));
}

struct htp_gdn_batch_context {
    struct htp_gdn_head_ptrs * heads;
    const float *              vtcm_g_raw;
    const float *              vtcm_b_raw;
    uint32_t                   curr_buf;
    uint32_t                   c;
    uint32_t                   n_batch;
    uint32_t                   S_v;
    float                      scale;
    struct htp_ops_context *   octx;
    const struct htp_gdn_kernel_params * kparams;
    uint32_t                   n_jobs;      // HVX jobs per phase; a batch wider than n_jobs runs in passes
    uint32_t                   gb_row;      // floats per token row of the raw g/beta buffers
    uint32_t                   qk_pre;      // QKSHARE: q/k padding of the last chunk is written by gdn_zero_pad_qk_worker
    uint32_t                   state_merge; // HTP_GDN_OPT_STATE_MERGE
    uint32_t                   n_slices;       // jobs per head of the current pass: 1, or 2 with HTP_GDN_FIX_PASS_SLICE
    uint32_t                   slice_shift;    // log2(n_slices): job mapping and per-slice extents use shifts, no division
    uint32_t                   split_ok;       // pass slicing allowed (HTP_GDN_FIX_PASS_SLICE; chunked HMX, n_tokens >= 64)
    uint32_t                   prologue_early; // HTP_GDN_FIX_PROLOGUE: chunk-0 inputs are pushed before the state-in wait
    uint32_t                   hmx_split_ok;   // HTP_GDN_FIX_HMX_SPLIT: DL and OA are issued per pass (chunked HMX, batched GEMMs, n_tokens >= 64)
    HVX_Vector *               scratch;        // per-job scratch blocks (64 HVX vectors each) for the phase-4 transposes
    uint32_t                   scratch_stride; // bytes between the per-job scratch blocks (layout st_scratch)
    uint32_t                   lossy;          // HTP_LOSSY_GDN_* bits (GGML_HEXAGON_LOSSY_GDN)
    const float *              gate_a;         // GDN_GATE_FOLD: ssm_a[H]; the raw g/beta rows are alpha+dt / beta and phase 1a applies softplus*A / sigmoid
};

// Job i of a pass: head i >> slice_shift, slice i & (n_slices - 1). With two slices, slice s owns half s of the chunk
// rows (tile row s) or of the state rows of its head and does that half of each phase; with one slice a job is the head.
static inline struct htp_gdn_head_ptrs * gdn_job_head(const struct htp_gdn_batch_context * bctx, unsigned int i) {
    return &bctx->heads[i >> bctx->slice_shift];
}

static inline uint32_t gdn_job_slice(const struct htp_gdn_batch_context * bctx, unsigned int i) {
    return i & (bctx->n_slices - 1);
}

static void gdn_hvx_init_state_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    struct htp_gdn_head_ptrs * head = gdn_job_head(bctx, i);
    const uint32_t s_rows = bctx->S_v >> bctx->slice_shift;
    const uint32_t s_lo   = gdn_job_slice(bctx, i) * s_rows;
    gdn_pack_s_col_tiles_rows(head->s_col_tiles, head->s_f16, head->s_state, bctx->S_v, s_lo, s_lo + s_rows,
                              (HVX_Vector *) ((uint8_t *) bctx->scratch + (size_t) i * bctx->scratch_stride));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
}

static inline __attribute__((unused)) HVX_Vector hvx_clamp_neg20_0(HVX_Vector v, HVX_Vector v_zero, HVX_Vector v_neg20) {
    HVX_VectorPred p_gt = Q6_Q_vcmp_gt_VsfVsf(v, v_zero);
    v = Q6_V_vmux_QVV(p_gt, v_zero, v);
    HVX_VectorPred p_lt = Q6_Q_vcmp_gt_VsfVsf(v_neg20, v);
    return Q6_V_vmux_QVV(p_lt, v_neg20, v);
}

static inline HVX_Vector hvx_prefix_scan_f32(HVX_Vector v, HVX_Vector carry_in) {
    const HVX_Vector zero = Q6_V_vzero();

    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_vec_add_f32_f32(v, carry_in);

    return v;
}

static inline HVX_Vector hvx_splat_last_f32(HVX_Vector v) {
    return hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

static void gdn_hvx_phase1a_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    struct htp_gdn_head_ptrs * head = gdn_job_head(bctx, i);
    const uint32_t curr_buf = bctx->curr_buf;
    const uint32_t S_v = bctx->S_v;
    const uint32_t gb_row = bctx->gb_row;
    const uint32_t sl = gdn_job_slice(bctx, i);
    const uint32_t r_per = 64u >> bctx->slice_shift;
    const uint32_t r_lo = sl * r_per;
    const uint32_t r_hi = r_lo + r_per;

    // every slice scans the gates of all 64 tokens. Slice 0 keeps them in the head's rows (phases 1b to 3 read them);
    // the other slices gather into the head's scratch block, which is free in this phase (vgather writes VTCM only).
    float * g_buf = (sl == 0) ? head->g_f32[curr_buf] : (float *) head->vtcm_m;
    float * b_buf = (sl == 0) ? head->b_f32[curr_buf] : ((float *) head->vtcm_m) + 64;

    if (gb_row == 1) {
        hvx_vmem(g_buf + 0)  = hvx_vmem(bctx->vtcm_g_raw + 0);
        hvx_vmem(g_buf + 32) = hvx_vmem(bctx->vtcm_g_raw + 32);
        hvx_vmem(b_buf + 0)  = hvx_vmem(bctx->vtcm_b_raw + 0);
        hvx_vmem(b_buf + 32) = hvx_vmem(bctx->vtcm_b_raw + 32);
    } else {
        int32_t offsets[32] __attribute__((aligned(128)));
        for (int k = 0; k < 32; ++k) {
            offsets[k] = k * gb_row * sizeof(float);
        }
        HVX_Vector vv = *(const HVX_Vector *) offsets;
        const size_t rt_g = (size_t) ((const uint8_t *) bctx->vtcm_g_raw + head->gb_col * sizeof(float));
        const size_t rt_b = (size_t) ((const uint8_t *) bctx->vtcm_b_raw + head->gb_col * sizeof(float));
        const size_t mu   = 64 * gb_row * sizeof(float);

        Q6_vgather_ARMVw((HVX_Vector *) (g_buf + 0),  rt_g, mu, vv);
        Q6_vgather_ARMVw((HVX_Vector *) (g_buf + 32), rt_g + 32 * gb_row * sizeof(float), mu, vv);
        Q6_vgather_ARMVw((HVX_Vector *) (b_buf + 0),  rt_b, mu, vv);
        Q6_vgather_ARMVw((HVX_Vector *) (b_buf + 32), rt_b + 32 * gb_row * sizeof(float), mu, vv);
    }

    if (bctx->gate_a) {
        // GDN_GATE folded in: g = softplus(x) * A[head], beta = sigmoid(raw), the same per-lane primitives as op_gdn_gate (unary-ops.c)
        const HVX_Vector v_20    = hvx_vec_splat_f32(20.0f);
        const HVX_Vector v_one   = hvx_vec_splat_f32(1.0f);
        const HVX_Vector v_max   = hvx_vec_splat_f32(87.f);
        const HVX_Vector v_min   = hvx_vec_splat_f32(-87.f);
        const HVX_Vector v_a     = hvx_vec_splat_f32(bctx->gate_a[head->gb_col]);
        const HVX_Vector x0 = hvx_vmem(g_buf + 0);
        const HVX_Vector x1 = hvx_vmem(g_buf + 32);
        const HVX_VectorPred p0 = Q6_Q_vcmp_gt_VsfVsf(x0, v_20);
        const HVX_VectorPred p1 = Q6_Q_vcmp_gt_VsfVsf(x1, v_20);
        const HVX_Vector e0 = hvx_vec_exp_f32(x0);
        const HVX_Vector e1 = hvx_vec_exp_f32(x1);
        const HVX_Vector l0 = hvx_vec_log_f32(hvx_vec_add_f32_f32(v_one, e0));
        const HVX_Vector l1 = hvx_vec_log_f32(hvx_vec_add_f32_f32(v_one, e1));
        hvx_vmem(g_buf + 0)  = hvx_vec_mul_f32_f32(Q6_V_vmux_QVV(p0, x0, l0), v_a);
        hvx_vmem(g_buf + 32) = hvx_vec_mul_f32_f32(Q6_V_vmux_QVV(p1, x1, l1), v_a);
        if (sl == 0) {   // beta is read from the head's rows only (phase 3); the other slice gathers it into scratch and never uses it
            hvx_vmem(b_buf + 0)  = hvx_vec_fast_sigmoid_f32_guard(hvx_vmem(b_buf + 0),  v_one, v_max, v_min);
            hvx_vmem(b_buf + 32) = hvx_vec_fast_sigmoid_f32_guard(hvx_vmem(b_buf + 32), v_one, v_max, v_min);
        }
    }

    const uint32_t t_chunk = bctx->c * 64;
    const uint32_t valid_tokens = hex_smin(64, bctx->kparams->n_tokens - t_chunk);
    if (valid_tokens < 64) {
        for (uint32_t t = valid_tokens; t < 64; ++t) {
            g_buf[t] = 0.0f;
            b_buf[t] = 0.0f;
        }
        // q, k and v padding rows: the rows of this slice
        const HVX_Vector vzero = Q6_V_vzero();
        const uint32_t t_pad = (valid_tokens > r_lo) ? valid_tokens : r_lo;
        for (uint32_t t = t_pad; t < r_hi; ++t) {
            for (uint32_t j = 0; j < S_v; j += 32) {
                if (!bctx->qk_pre) {
                    hvx_vmem(head->q_f32[curr_buf] + t * S_v + j) = vzero;
                    hvx_vmem(head->k_f32[curr_buf] + t * S_v + j) = vzero;
                }
                hvx_vmem(head->v_f32[curr_buf] + t * S_v + j) = vzero;
            }
        }
    }

    const HVX_Vector v_g0 = hvx_vmem(g_buf + 0);
    const HVX_Vector v_g1 = hvx_vmem(g_buf + 32);

    HVX_Vector v_gamma0 = hvx_prefix_scan_f32(v_g0, Q6_V_vzero());
    HVX_Vector v_carry  = hvx_splat_last_f32(v_gamma0);
    HVX_Vector v_gamma1 = hvx_prefix_scan_f32(v_g1, v_carry);

    const HVX_Vector v_zero  = Q6_V_vzero();
    const HVX_Vector v_neg20 = hvx_vec_splat_f32(-20.0f);

    if (sl == 0) {
        hvx_vmem(head->gamma) = hvx_vec_f32_to_f16(v_gamma0, v_gamma1);
    }

    HVX_Vector v_l0 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma0, v_zero, v_neg20));
    HVX_Vector v_l1 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma1, v_zero, v_neg20));

    // f16 lambda of all 64 rows in a local block: the K rows of this slice are scaled with it
    __fp16 lam16[64] __attribute__((aligned(128)));
    hvx_vmem(lam16) = hvx_vec_f32_to_f16(v_l0, v_l1);
    if (sl == 0) {
        hvx_vmem(head->lambda_init + 0)  = v_l0;
        hvx_vmem(head->lambda_init + 32) = v_l1;
        hvx_vmem(head->lambda_init_f16)  = hvx_vec_f32_to_f16(v_l0, v_l1);
    }

    gdn_f32_to_hmx_row_tiles_and_f16_rows(head->k_row_tiles, head->k_prime_row_tiles, head->k_f16,
                                          head->k_f32[curr_buf], lam16, r_lo, r_hi, S_v);
    gdn_rows_to_col_tiles_range(head->k_col_tiles, head->k_f16, S_v, r_lo, r_hi, sl == 0 ? head->rows_inv : head->rows_a);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

static void gdn_hvx_phase1b_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    struct htp_gdn_head_ptrs * head = gdn_job_head(bctx, i);
    const uint32_t curr_buf = bctx->curr_buf;
    const uint32_t S_v = bctx->S_v;
    const uint32_t S   = bctx->n_slices;
    const uint32_t sl  = gdn_job_slice(bctx, i);
    const uint32_t r_per = 64u >> bctx->slice_shift;
    const uint32_t r_lo  = sl * r_per;
    const uint32_t r_hi  = r_lo + r_per;

    gdn_f32_to_hmx_row_tiles_and_f16_rows(head->q_row_tiles, head->q_prime_row_tiles, NULL,
                                          head->q_f32[curr_buf], head->lambda_init_f16, r_lo, r_hi, S_v);

    const uint16_t * gamma_u16 = (const uint16_t *) head->gamma;
    const HVX_Vector v_gamma   = hvx_vmem(head->gamma);
    const bool decay_v2 = (bctx->lossy & HTP_LOSSY_GDN_DECAY_V2) != 0;

    if (bctx->lossy & HTP_LOSSY_GDN_DLAST_K) {
        // decay_last[s] = decay_a[63][s] folds into the K rows of the S-update operand (phase 4 no longer scales delta)
        uint16_t dl[64] __attribute__((aligned(128)));
        {
            HVX_Vector diff = hvx_vec_sub_f16_f16(Q6_Vh_vsplat_R(gamma_u16[63]), v_gamma);
            HVX_VectorPred p_gt = Q6_Q_vcmp_gt_VhfVhf(diff, Q6_V_vzero());
            diff = Q6_V_vmux_QVV(p_gt, Q6_V_vzero(), diff);
            diff = Q6_Vhf_vmax_VhfVhf(hvx_vec_splat_f16(-20.0f), diff);
            HVX_Vector e = decay_v2 ? hvx_vec_exp2_f16_v2(hvx_vec_mul_f16_f16(diff, hvx_vec_splat_f16(1.4426950408889634f)))
                                    : hvx_vec_exp2_f16(hvx_vec_mul_f16_f16(diff, hvx_vec_splat_f16(1.4426950408889634f)));
            HVX_VectorPred lt = Q6_Q_vsetq2_R(2 * 63);
            hvx_vmem(dl) = Q6_V_vmux_QVV(lt, e, hvx_vec_splat_f16(1.0f));
        }
        for (uint32_t r = r_lo; r < r_hi; r += 2) {
            const HVX_Vector * pv_in0 = (const HVX_Vector *) (head->k_f16 + (size_t) r * S_v);
            const HVX_Vector * pv_in1 = (const HVX_Vector *) (head->k_f16 + (size_t) (r + 1) * S_v);
            const HVX_Vector s0 = Q6_Vh_vsplat_R(dl[r]);
            const HVX_Vector s1 = Q6_Vh_vsplat_R(dl[r + 1]);
            const uint32_t r0 = r / HMX_FP16_TILE_N_ROWS;
            const uint32_t r1_half = (r % HMX_FP16_TILE_N_ROWS) / 2;
            __fp16 * tb0 = head->k_col_tiles_64x128 + (size_t) r0 * HMX_FP16_TILE_N_ELMS;
            __fp16 * tb1 = tb0 + 2 * HMX_FP16_TILE_N_ELMS;
            for (uint32_t c = 0; c < S_v; c += 64) {
                HVX_Vector     v0 = hvx_vec_mul_f16_f16(*pv_in0++, s0);
                HVX_Vector     v1 = hvx_vec_mul_f16_f16(*pv_in1++, s1);
                HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
                ((HVX_Vector *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_Vector *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += 4 * HMX_FP16_TILE_N_ELMS;
                tb1 += 4 * HMX_FP16_TILE_N_ELMS;
            }
        }
    } else {
        gdn_interleave_cols_to_tiles_range(head->k_col_tiles_64x128, head->k_f16, 64, 0, S_v, S_v, 2, r_lo, r_hi);
    }

    const HVX_Vector v_zero_f16  = Q6_V_vzero();
    const HVX_Vector v_neg20_f16 = hvx_vec_splat_f16(-20.0f);
    const HVX_Vector v_log2e_f16 = hvx_vec_splat_f16(1.4426950408889634f);
    const HVX_Vector v_one_f16   = hvx_vec_splat_f16(1.0f);

    if (sl == 0) {
        hvx_vmem(head->decay_m + 0) = Q6_V_vzero();
        hvx_vmem(head->decay_a + 0) = Q6_V_vand_QV(Q6_Q_vsetq2_R(2), v_one_f16);
    }

    // row pairs (t0, t1) = (2p + 1, 2p + 2) for p = 0..30; slice 0 takes p in [0, 16), slice 1 takes p in [16, 31)
    const uint32_t p_lo = (S == 1) ? 0 : (sl == 0 ? 0 : 16);
    const uint32_t p_hi = (S == 1) ? 31 : (sl == 0 ? 16 : 31);
    for (uint32_t p = p_lo; p < p_hi; ++p) {
        uint32_t t0 = 2 * p + 1;
        uint32_t t1 = t0 + 1;

        HVX_Vector v_gamma_t0  = Q6_Vh_vsplat_R(gamma_u16[t0]);
        HVX_Vector v_gamma_t1  = Q6_Vh_vsplat_R(gamma_u16[t1]);

        HVX_Vector diff0       = hvx_vec_sub_f16_f16(v_gamma_t0, v_gamma);
        HVX_Vector diff1       = hvx_vec_sub_f16_f16(v_gamma_t1, v_gamma);

        HVX_VectorPred p_gt0   = Q6_Q_vcmp_gt_VhfVhf(diff0, v_zero_f16);
        HVX_VectorPred p_gt1   = Q6_Q_vcmp_gt_VhfVhf(diff1, v_zero_f16);

        diff0                  = Q6_V_vmux_QVV(p_gt0, v_zero_f16, diff0);
        diff1                  = Q6_V_vmux_QVV(p_gt1, v_zero_f16, diff1);

        diff0                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff0);
        diff1                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff1);

        HVX_Vector diff_log2e0 = hvx_vec_mul_f16_f16(diff0, v_log2e_f16);
        HVX_Vector diff_log2e1 = hvx_vec_mul_f16_f16(diff1, v_log2e_f16);

        HVX_Vector v_exp0      = decay_v2 ? hvx_vec_exp2_f16_v2(diff_log2e0) : hvx_vec_exp2_f16(diff_log2e0);
        HVX_Vector v_exp1      = decay_v2 ? hvx_vec_exp2_f16_v2(diff_log2e1) : hvx_vec_exp2_f16(diff_log2e1);

        HVX_VectorPred mask_lt0 = Q6_Q_vsetq2_R(2 * t0);
        HVX_VectorPred mask_lt1 = Q6_Q_vsetq2_R(2 * t1);

        HVX_Vector v_m0         = Q6_V_vand_QV(mask_lt0, v_exp0);
        HVX_Vector v_m1         = Q6_V_vand_QV(mask_lt1, v_exp1);

        HVX_VectorPred mask_le0 = Q6_Q_vsetq2_R(2 * (t0 + 1));
        HVX_VectorPred mask_le1 = Q6_Q_vsetq2_R(2 * (t1 + 1));

        HVX_VectorPred mask_diag0 = Q6_Q_and_QQn(mask_le0, mask_lt0);
        HVX_VectorPred mask_diag1 = Q6_Q_and_QQn(mask_le1, mask_lt1);

        HVX_Vector v_a0         = Q6_V_vmux_QVV(mask_diag0, v_one_f16, v_m0);
        HVX_Vector v_a1         = Q6_V_vmux_QVV(mask_diag1, v_one_f16, v_m1);

        hvx_vmem(head->decay_m + t0 * 64) = v_m0;
        hvx_vmem(head->decay_a + t0 * 64) = v_a0;
        hvx_vmem(head->decay_m + t1 * 64) = v_m1;
        hvx_vmem(head->decay_a + t1 * 64) = v_a1;
    }

    if (sl == S - 1) {
        HVX_Vector v_gamma_t     = Q6_Vh_vsplat_R(gamma_u16[63]);
        HVX_Vector diff          = hvx_vec_sub_f16_f16(v_gamma_t, v_gamma);
        HVX_VectorPred p_gt      = Q6_Q_vcmp_gt_VhfVhf(diff, v_zero_f16);
        diff                     = Q6_V_vmux_QVV(p_gt, v_zero_f16, diff);
        diff                     = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff);

        HVX_Vector diff_log2e    = hvx_vec_mul_f16_f16(diff, v_log2e_f16);
        HVX_Vector v_exp         = decay_v2 ? hvx_vec_exp2_f16_v2(diff_log2e) : hvx_vec_exp2_f16(diff_log2e);

        HVX_VectorPred mask_lt_t = Q6_Q_vsetq2_R(2 * 63);
        HVX_Vector v_m           = Q6_V_vand_QV(mask_lt_t, v_exp);

        HVX_VectorPred mask_le_t = Q6_Q_vcmp_eq_VhVh(v_zero_f16, v_zero_f16);
        HVX_VectorPred mask_diag = Q6_Q_and_QQn(mask_le_t, mask_lt_t);

        HVX_Vector v_a           = Q6_V_vmux_QVV(mask_diag, v_one_f16, v_m);

        hvx_vmem(head->decay_m + 63 * 64) = v_m;
        hvx_vmem(head->decay_a + 63 * 64) = v_a;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

// HTP_GDN_OPT_UP_LEGACY_EXP: decay matrix with the pre-268f3874d arithmetic: f32 gamma (the same prefix scan as phase 1a),
// f32 differences, f32 exp, packed to f16. Everything except the decay matrix is phase 1b.
static void gdn_hvx_phase1b_f32exp_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t curr_buf = bctx->curr_buf;
    const uint32_t S_v = bctx->S_v;

    gdn_f32_to_hmx_row_tiles_and_f16_rows(head->q_row_tiles, head->q_prime_row_tiles, NULL,
                                          head->q_f32[curr_buf], head->lambda_init_f16, 0, 64, S_v);

    hmx_interleave_cols_to_tiles(head->k_col_tiles_64x128, head->k_f16, 64, S_v, S_v, 2, 0, 64);

    const HVX_Vector v_zero  = Q6_V_vzero();
    const HVX_Vector v_neg20 = hvx_vec_splat_f32(-20.0f);
    const HVX_Vector v_gamma0 = hvx_prefix_scan_f32(hvx_vmem(head->g_f32[curr_buf] + 0), Q6_V_vzero());
    const HVX_Vector v_gamma1 = hvx_prefix_scan_f32(hvx_vmem(head->g_f32[curr_buf] + 32), hvx_splat_last_f32(v_gamma0));

    float gamma_f32[64] __attribute__((aligned(128)));
    hvx_vmem(gamma_f32 + 0)  = v_gamma0;
    hvx_vmem(gamma_f32 + 32) = v_gamma1;

    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);

    for (uint32_t t = 0; t < 64; ++t) {
        HVX_Vector v_gamma_t = hvx_vec_splat_f32(gamma_f32[t]);
        HVX_Vector diff0 = hvx_vec_sub_f32_f32(v_gamma_t, v_gamma0);
        HVX_Vector diff1 = hvx_vec_sub_f32_f32(v_gamma_t, v_gamma1);

        diff0 = hvx_clamp_neg20_0(diff0, v_zero, v_neg20);
        diff1 = hvx_clamp_neg20_0(diff1, v_zero, v_neg20);

        HVX_Vector exp0 = hvx_vec_exp_f32(diff0);
        HVX_Vector exp1 = hvx_vec_exp_f32(diff1);

        HVX_Vector v_exp = hvx_vec_f32_to_f16(exp0, exp1);

        HVX_Vector v_m, v_a;
        if (t == 0) {
            v_m = Q6_V_vzero();
            v_a = Q6_V_vand_QV(Q6_Q_vsetq2_R(2), v_one_f16);
        } else {
            HVX_VectorPred mask_lt_t = Q6_Q_vsetq2_R(2 * t);
            v_m = Q6_V_vand_QV(mask_lt_t, v_exp);

            HVX_VectorPred mask_le_t = (t == 63) ? Q6_Q_vcmp_eq_VhVh(v_zero, v_zero) : Q6_Q_vsetq2_R(2 * (t + 1));
            HVX_VectorPred mask_diag = Q6_Q_and_QQn(mask_le_t, mask_lt_t);
            v_a = Q6_V_vmux_QVV(mask_diag, v_one_f16, v_m);
        }

        hvx_vmem(head->decay_m + t * 64) = v_m;
        hvx_vmem(head->decay_a + t * 64) = v_a;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

static void gdn_hvx_phase2_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);

    struct htp_gdn_head_ptrs * head = gdn_job_head(bctx, i);
    const uint32_t curr_buf = bctx->curr_buf;
    const uint32_t band = (bctx->lossy & HTP_LOSSY_GDN_SOLVE_B8) ? 8u : (bctx->lossy & HTP_LOSSY_GDN_SOLVE_B16) ? 16u : 32u;

    // one job per head: both diagonal blocks; two slices: slice s builds diagonal block s from its KK tile row
    if (bctx->n_slices == 1) {
        gdn_unpack_64x64_tiles_rows(head->rows_kk, head->kk_tiles, 0, 2);
        gdn_build_inv_l_blocks(head->inv_row_tiles, head->rows_kk, head->decay_m, head->b_f32[curr_buf],
                               (__fp16 *) head->vtcm_m, (__fp16 *) head->vtcm_m + HMX_FP16_TILE_N_ELMS, 2, band);
    } else {
        const uint32_t sl = gdn_job_slice(bctx, i);
        gdn_unpack_64x64_tiles_rows(head->rows_kk, head->kk_tiles, sl, sl + 1);
        gdn_build_inv_l_blocks(head->inv_row_tiles, head->rows_kk, head->decay_m, head->b_f32[curr_buf],
                               (__fp16 *) head->vtcm_m, (__fp16 *) head->vtcm_m + HMX_FP16_TILE_N_ELMS, sl, band);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);
}

// v_prime_f16 = (v - v_inter) * beta for the 32-row tile rows [r0_lo, r0_hi); v_inter is unpacked from its tiles in the same pass.
// Per element the same operations as the unpack to f32 rows followed by the (v - v_inter) * beta loop. LOSSY_GDN VPRIME_F16: v' = (v - v_inter) * beta in fp16 (v is narrowed to fp16, as the HMX operand v' is anyway).
static inline void gdn_vprime_from_tiles_rows_f16(struct htp_gdn_head_ptrs * head, uint32_t S_v, const HVX_VectorAlias * local_b,
                                                  uint32_t curr_buf, uint32_t r0_lo, uint32_t r0_hi) {
    const uint32_t n_col_tiles = S_v / 32;
    for (uint32_t r0 = r0_lo; r0 < r0_hi; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (head->v_inter_tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (head->v_inter_tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);
            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t t  = r0 * 32 + 2 * r + h;
                    const HVX_Vector vb = hvx_vec_splat_f16((_Float16) local_b[t / 32].fp32[t % 32]);
                    const float * vv  = head->v_f32[curr_buf] + t * S_v + d * 64;
                    const HVX_Vector vi = (h == 0) ? Q6_V_lo_W(vp01) : Q6_V_hi_W(vp01);
                    const HVX_Vector v16 = hvx_vec_f32_to_f16(hvx_vmem(vv + 0), hvx_vmem(vv + 32));
                    hvx_vmem(head->v_prime_f16 + t * S_v + d * 64) = hvx_vec_mul_f16_f16(hvx_vec_sub_f16_f16(v16, vi), vb);
                }
            }
        }
    }
}

static inline void gdn_vprime_from_tiles_rows(struct htp_gdn_head_ptrs * head, uint32_t S_v, const HVX_VectorAlias * local_b,
                                              uint32_t curr_buf, uint32_t r0_lo, uint32_t r0_hi) {
    const uint32_t n_col_tiles = S_v / 32;
    for (uint32_t r0 = r0_lo; r0 < r0_hi; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (head->v_inter_tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (head->v_inter_tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);
            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_VectorPair pa   = hvx_vec_f16_to_f32(Q6_V_lo_W(vp01)); // row 2r
                HVX_VectorPair pb   = hvx_vec_f16_to_f32(Q6_V_hi_W(vp01)); // row 2r + 1
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t t  = r0 * 32 + 2 * r + h;
                    const HVX_Vector vb = hvx_vec_splat_f32(local_b[t / 32].fp32[t % 32]);
                    const float * vv  = head->v_f32[curr_buf] + t * S_v + d * 64;
                    const HVX_VectorPair p = (h == 0) ? pa : pb;
                    HVX_Vector vp0 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(hvx_vmem(vv + 0),  Q6_V_lo_W(p)), vb);
                    HVX_Vector vp1 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(hvx_vmem(vv + 32), Q6_V_hi_W(p)), vb);
                    hvx_vmem(head->v_prime_f16 + t * S_v + d * 64) = hvx_vec_f32_to_f16(vp0, vp1);
                }
            }
        }
    }
}

static void gdn_hvx_phase3_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);

    struct htp_gdn_head_ptrs * head = gdn_job_head(bctx, i);
    const uint32_t curr_buf = bctx->curr_buf;
    const uint32_t S_v = bctx->S_v;
    // chunk rows [r_lo, r_hi) of the slice (one tile row with two slices)
    const uint32_t r_per = 64u >> bctx->slice_shift;
    const uint32_t r_lo  = gdn_job_slice(bctx, i) * r_per;
    const uint32_t r_hi  = r_lo + r_per;

    // v_prime = (v - v_inter) * beta for the rows [r_lo, r_hi): the v_inter tiles are unpacked in the same pass
    HVX_VectorAlias local_b[2];
    local_b[0].v = hvx_vmem(head->b_f32[curr_buf] + 0);
    local_b[1].v = hvx_vmem(head->b_f32[curr_buf] + 32);
    if (bctx->lossy & HTP_LOSSY_GDN_VPRIME_F16) {
        gdn_vprime_from_tiles_rows_f16(head, S_v, local_b, curr_buf, r_lo / 32, r_hi / 32);
    } else {
        gdn_vprime_from_tiles_rows(head, S_v, local_b, curr_buf, r_lo / 32, r_hi / 32);
    }

    gdn_interleave_cols_to_tiles_range(head->v_prime_col_tiles, head->v_prime_f16, 64, 0, S_v, S_v, 2, r_lo, r_hi);

    gdn_unpack_64x64_tiles_rows(head->rows_qk, head->qk_tiles, r_lo / 32, r_hi / 32);
    for (uint32_t t = r_lo; t < r_hi; ++t) {
        HVX_Vector v_decay_a = hvx_vmem(head->decay_a + t * 64);
        head->rows_a[t]      = hvx_vec_mul_f16_f16(head->rows_qk[t], v_decay_a);
    }
    gdn_pack_64x64_rows_to_tiles(head->a_row_tiles, head->rows_a, r_lo / 32, r_hi / 32);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);
}

static void gdn_hvx_phase4_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);

    struct htp_gdn_head_ptrs * head = gdn_job_head(bctx, i);
    const uint32_t S_v = bctx->S_v;

    // D_PACK: with one slice, slice 0 does the whole head (the other slices of a pass return at once).
    // With two slices (the second pass of a batch wider than n_threads), slice s takes column half s of d: every step below is per column half, so the halves are independent.
    const uint32_t sl = gdn_job_slice(bctx, i);
    uint32_t d_lo = 0;
    uint32_t d_hi = S_v / 64;
    if (bctx->n_slices == 2) {
        d_lo = sl;
        d_hi = sl + 1;
    } else if (sl != 0) {
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);
        return;
    }

    gdn_unpack_64xS_tiles_to_f16_cols(head->delta_f16, head->delta_tiles, S_v, d_lo, d_hi);
    gdn_interleave_cols_to_tiles_range(head->delta_col_tiles, head->delta_f16, 64, d_lo * 64, d_hi * 64, S_v, 2, 0, 64);

    const uint16_t * decay_last = (const uint16_t *) (head->decay_a + 63 * 64);
    const HVX_Vector vzero      = Q6_V_vzero();
    // HTP_GDN_OPT_UP_LEGACY_ACC: pre-7d4b7c4d9 product without the zero-decay mask (the two differ only for non-finite deltas)
    const bool legacy_acc = (bctx->kparams->opt & HTP_GDN_OPT_UP_LEGACY_ACC) != 0;

    // LOSSY_GDN DLAST_K: decay_last was folded into the K rows of the S-update operand (phase 1b), d = delta
    const bool dlast_k = (bctx->lossy & HTP_LOSSY_GDN_DLAST_K) != 0;
    if (!dlast_k) {
        for (uint32_t s = 0; s < 64; ++s) {
            HVX_Vector vs         = Q6_Vh_vsplat_R(decay_last[s]);
            HVX_VectorPred p_zero = Q6_Q_vcmp_eq_VhVh(vs, vzero);
            for (uint32_t j = d_lo * 64; j < d_hi * 64; j += 64) {
                HVX_Vector vd   = hvx_vmem(head->delta_f16 + s * S_v + j);
                HVX_Vector prod = hvx_vec_mul_f16_f16(vd, vs);
                hvx_vmem(head->d_f16 + s * S_v + j) = legacy_acc ? prod : Q6_V_vmux_QVV(p_zero, vzero, prod);
            }
        }
    }

    // d^T tiles of the slice's column halves; the transposes use scratch block i of the batch (the per-head blocks hold
    // the l10/neg_a11 and inverse operands, which were consumed by IA/IB before this phase)
    HVX_Vector * scr = (HVX_Vector *) ((uint8_t *) bctx->scratch + (size_t) i * bctx->scratch_stride);
    gdn_pack_d_t_row_tiles_range(head->d_row_tiles, dlast_k ? head->delta_f16 : head->d_f16, S_v, scr, scr + 32, d_lo, d_hi);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);
}

// OUT part 2 of 2 is in gdn_hvx_phase5_worker. o_inter is unpacked into the output rows by part 1 (gdn_hvx_phase5a_worker),
// which runs while the HMX computes o_intra; the add with o_intra and the scale follow.
static void gdn_hvx_phase5a_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_REM, info);

    struct htp_gdn_head_ptrs * head = gdn_job_head(bctx, i);
    // slice s unpacks tile row s (chunk rows 32 s .. 32 s + 31)
    const uint32_t h_per = 2u >> bctx->slice_shift;
    const uint32_t h_lo  = gdn_job_slice(bctx, i) * h_per;
    gdn_unpack_tiles_to_f32_rows(head->o_f32[bctx->curr_buf], head->o_inter_tiles, bctx->S_v, h_lo, h_lo + h_per);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_REM, info);
}

// o = (o_inter + o_intra) * scale for the 32-row tile rows [r0_lo, r0_hi). Per element the same operations as the unpack of both tile sets followed by the add and the scale.
// o_inter is already in the output rows (gdn_hvx_phase5a_worker); o_intra is unpacked from its tiles in this pass.
static inline void gdn_out_add_tiles_rows(float * restrict o, const __fp16 * restrict intra_tiles, uint32_t S_v,
                                          uint32_t r0_lo, uint32_t r0_hi, HVX_Vector vscale) {
    const uint32_t n_col_tiles = S_v / 32;
    for (uint32_t r0 = r0_lo; r0 < r0_hi; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (intra_tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (intra_tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);
            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_VectorPair pa   = hvx_vec_f16_to_f32(Q6_V_lo_W(vp01)); // row 2r
                HVX_VectorPair pb   = hvx_vec_f16_to_f32(Q6_V_hi_W(vp01)); // row 2r + 1

                float * o0 = o + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                float * o1 = o0 + S_v;
                HVX_Vector a00 = hvx_vmem(o0 + 0);
                HVX_Vector a01 = hvx_vmem(o0 + 32);
                HVX_Vector a10 = hvx_vmem(o1 + 0);
                HVX_Vector a11 = hvx_vmem(o1 + 32);

                hvx_vmem(o0 + 0)  = hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(a00, Q6_V_lo_W(pa)), vscale);
                hvx_vmem(o0 + 32) = hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(a01, Q6_V_hi_W(pa)), vscale);
                hvx_vmem(o1 + 0)  = hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(a10, Q6_V_lo_W(pb)), vscale);
                hvx_vmem(o1 + 32) = hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(a11, Q6_V_hi_W(pb)), vscale);
            }
        }
    }
}

static void gdn_hvx_phase5_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_OUT, info);

    struct htp_gdn_head_ptrs * head = gdn_job_head(bctx, i);
    // slice s owns tile row s of the output rows (see gdn_hvx_phase5a_worker)
    const uint32_t h_per = 2u >> bctx->slice_shift;
    const uint32_t h_lo  = gdn_job_slice(bctx, i) * h_per;
    gdn_out_add_tiles_rows(head->o_f32[bctx->curr_buf], head->o_intra_tiles, bctx->S_v, h_lo, h_lo + h_per,
                           hvx_vec_splat_f32(bctx->scale));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_OUT, info);
}

// HTP_GDN_OPT_STATE_MERGE: S <- S * lambda_last + s_update, with the update read straight from its tiles and the f16
// row-major copy of the new S (for the column-tile repack) written in the same pass. Same operations, same order.
static inline void gdn_state_update_pack_fused(struct htp_gdn_head_ptrs * head, uint32_t S_v, HVX_Vector v_l, bool pack) {
    const uint32_t n_tiles = S_v / 32;
    for (uint32_t r0 = 0; r0 < n_tiles; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (head->s_update_tiles + (r0 * n_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (head->s_update_tiles + (r0 * n_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_VectorPair pu[2];
                pu[0] = hvx_vec_f16_to_f32(Q6_V_lo_W(vp01)); // row 2r
                pu[1] = hvx_vec_f16_to_f32(Q6_V_hi_W(vp01)); // row 2r + 1

                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t row = r0 * 32 + 2 * r + half;
                    float * s = head->s_state + row * S_v + d * 64;

                    HVX_Vector n0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(s + 0),  v_l), Q6_V_lo_W(pu[half]));
                    HVX_Vector n1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(s + 32), v_l), Q6_V_hi_W(pu[half]));
                    hvx_vmem(s + 0)  = n0;
                    hvx_vmem(s + 32) = n1;
                    if (pack) {
                        hvx_vmem(head->s_f16 + row * S_v + d * 64) = hvx_vec_f32_to_f16(n0, n1);
                    }
                }
            }
        }
    }
}

// Non-merged state update of the 32-row tile rows [r0_lo, r0_hi): S <- S * lambda + upd, with upd unpacked from the f16 update tiles in the same pass (no f32 copy of the update).
// With pack, the f16 row-major copy of the new S is written too (source of the next chunk's column tiles). Per element the same operations as gdn_unpack_tiles_to_f32_rows, the update loop and the f16 repack of gdn_pack_s_col_tiles_rows.
static inline void gdn_state_axpy_tiles_rows(struct htp_gdn_head_ptrs * head, uint32_t S_v, HVX_Vector v_l,
                                             uint32_t r0_lo, uint32_t r0_hi, bool pack) {
    const uint32_t n_col_tiles = S_v / 32;
    for (uint32_t r0 = r0_lo; r0 < r0_hi; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (head->s_update_tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (head->s_update_tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_VectorPair pa   = hvx_vec_f16_to_f32(Q6_V_lo_W(vp01)); // row 2r of the tile row, cols d*64 .. +63
                HVX_VectorPair pb   = hvx_vec_f16_to_f32(Q6_V_hi_W(vp01)); // row 2r + 1

                const uint32_t row0 = r0 * 32 + 2 * r;
                float * s0 = head->s_state + row0 * S_v + d * 64;
                float * s1 = s0 + S_v;
                HVX_Vector a0 = hvx_vmem(s0 + 0);
                HVX_Vector a1 = hvx_vmem(s0 + 32);
                HVX_Vector b0 = hvx_vmem(s1 + 0);
                HVX_Vector b1 = hvx_vmem(s1 + 32);

                HVX_Vector n00 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(a0, v_l), Q6_V_lo_W(pa));
                HVX_Vector n01 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(a1, v_l), Q6_V_hi_W(pa));
                HVX_Vector n10 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(b0, v_l), Q6_V_lo_W(pb));
                HVX_Vector n11 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(b1, v_l), Q6_V_hi_W(pb));

                hvx_vmem(s0 + 0)  = n00;
                hvx_vmem(s0 + 32) = n01;
                hvx_vmem(s1 + 0)  = n10;
                hvx_vmem(s1 + 32) = n11;

                if (pack) {
                    __fp16 * f0 = head->s_f16 + row0 * S_v + d * 64;
                    hvx_vmem(f0)       = hvx_vec_f32_to_f16(n00, n01);
                    hvx_vmem(f0 + S_v) = hvx_vec_f32_to_f16(n10, n11);
                }
            }
        }
    }
}

// LOSSY_GDN STATE_F16: S <- S * lambda + update on the row-major f16 copy (fused hf multiply-add); the f32 state is not touched.
static inline HVX_Vector gdn_hf_fma(HVX_Vector acc, HVX_Vector a, HVX_Vector b) {
#if __HVX_ARCH__ >= 79
    return Q6_Vhf_vmpyacc_VhfVhfVhf(acc, a, b);
#else
    return hvx_vec_add_f16_f16(acc, hvx_vec_mul_f16_f16(a, b));
#endif
}

static inline void gdn_state_axpy_f16_rows(struct htp_gdn_head_ptrs * head, uint32_t S_v, HVX_Vector v_l16, uint32_t r0_lo, uint32_t r0_hi) {
    const uint32_t n_col_tiles = S_v / 32;
    for (uint32_t r0 = r0_lo; r0 < r0_hi; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (head->s_update_tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (head->s_update_tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);
            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                __fp16 * f0 = head->s_f16 + (r0 * 32 + 2 * r) * S_v + d * 64;
                hvx_vmem(f0)       = gdn_hf_fma(Q6_V_lo_W(vp01), hvx_vmem(f0), v_l16);
                hvx_vmem(f0 + S_v) = gdn_hf_fma(Q6_V_hi_W(vp01), hvx_vmem(f0 + S_v), v_l16);
            }
        }
    }
}

// last chunk of LOSSY_GDN STATE_F16: the f32 state = widened f16 state, rows [j_lo, j_hi)
static inline void gdn_state_f16_to_f32_rows(struct htp_gdn_head_ptrs * head, uint32_t S_v, uint32_t j_lo, uint32_t j_hi) {
    for (uint32_t j = j_lo; j < j_hi; ++j) {
        for (uint32_t i = 0; i < S_v; i += 64) {
            HVX_VectorPair p = hvx_vec_f16_to_f32(hvx_vmem(head->s_f16 + j * S_v + i));
            hvx_vmem(head->s_state + j * S_v + i + 0)  = Q6_V_lo_W(p);
            hvx_vmem(head->s_state + j * S_v + i + 32) = Q6_V_hi_W(p);
        }
    }
}

static void gdn_hvx_phase6_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);

    struct htp_gdn_head_ptrs * head = gdn_job_head(bctx, i);
    const uint32_t S_v = bctx->S_v;
    const uint32_t c = bctx->c;
    const uint32_t n_chunks = bctx->kparams->n_chunks;

    HVX_VectorAlias last_lambda;
    last_lambda.v = hvx_vmem(head->lambda_init + 32);
    HVX_Vector v_l_final = hvx_vec_splat_f32(last_lambda.fp32[31]);

    if (bctx->state_merge) {
        gdn_state_update_pack_fused(head, S_v, v_l_final, c + 1 < n_chunks);
        if (c + 1 < n_chunks) {
            hmx_interleave_rows_to_tiles(head->s_col_tiles, head->s_f16, S_v, S_v, S_v, 0, S_v);
        }
    } else {
        // slice s owns the state rows [s_lo, s_hi): their tile rows, their update, their f16 repack and col tiles
        const uint32_t s_rows = S_v >> bctx->slice_shift;
        const uint32_t s_lo   = gdn_job_slice(bctx, i) * s_rows;
        const uint32_t s_hi   = s_lo + s_rows;
        const bool     pack   = c + 1 < n_chunks;
        if ((bctx->lossy & HTP_LOSSY_GDN_STATE_F16) && n_chunks > 1) {
            if (pack) {
                gdn_state_axpy_f16_rows(head, S_v, hvx_vec_splat_f16(last_lambda.fp32[31]), s_lo / 32, s_hi / 32);
            } else {
                gdn_state_f16_to_f32_rows(head, S_v, s_lo, s_hi);
                gdn_state_axpy_tiles_rows(head, S_v, v_l_final, s_lo / 32, s_hi / 32, false);
            }
        } else {
            gdn_state_axpy_tiles_rows(head, S_v, v_l_final, s_lo / 32, s_hi / 32, pack);
        }
        if (pack) {
            gdn_rows_to_col_tiles_range(head->s_col_tiles, head->s_f16, S_v, s_lo, s_hi,
                                        (HVX_Vector *) ((uint8_t *) bctx->scratch + (size_t) i * bctx->scratch_stride));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);
}


// --------------------------------------------------------------------------------------------------------------------- Chunked-HMX driver. Each GEMM kind is one HMX descriptor for the whole batch (HTP_GDN_OPT_HMX_BATCH) or one per head (legacy). Pops follow the issue order of the descriptors.
// --------------------------------------------------------------------------------------------------------------------- A batch of n heads of one sequence runs the chunk loop together. Each phase is one work-queue run over the heads (passes of n_jobs heads when n > n_jobs).

enum htp_gdn_gemm_kind {
    GDN_GEMM_KK    = 0, // kk       = K K^T                (k_row_tiles, k_col_tiles)
    GDN_GEMM_QK    = 1, // qk       = Q K^T                (q_row_tiles, k_col_tiles)
    GDN_GEMM_VI    = 2, // v_inter  = K' S                 (k_prime_row_tiles, s_col_tiles)
    GDN_GEMM_OI    = 3, // o_inter  = Q' S                 (q_prime_row_tiles, s_col_tiles)
    GDN_GEMM_DL    = 4, // delta    = inv v'               (inv_row_tiles, v_prime_col_tiles)
    GDN_GEMM_OA    = 5, // o_intra  = A delta              (a_row_tiles, delta_col_tiles)
    GDN_GEMM_SU    = 6, // s_update = D^T K                (d_row_tiles, k_col_tiles_64x128)
    GDN_GEMM_IA    = 7, // inverse, block 0               (vtcm_m, inv_row_tiles)
    GDN_GEMM_IB    = 8, // inverse, block 1               (vtcm_m + 1 tile, vtcm_tmp)
    GDN_GEMM_COUNT = 9,
};

static inline void gdn_gemm_fill(struct htp_gdn_hmx_gemm_task * t, uint32_t kind, const struct htp_gdn_head_ptrs * hp,
                                 uint32_t n_sv_tiles, uint8_t * scales) {
    const __fp16 * row = NULL;
    const __fp16 * col = NULL;
    __fp16 *       out = NULL;
    uint32_t       nr  = 2;
    uint32_t       nc  = 2;
    uint32_t       nd  = n_sv_tiles;

    switch (kind) {
        case GDN_GEMM_KK: row = hp->k_row_tiles;       col = hp->k_col_tiles;        out = hp->kk_tiles;      break;
        case GDN_GEMM_QK: row = hp->q_row_tiles;       col = hp->k_col_tiles;        out = hp->qk_tiles;      break;
        case GDN_GEMM_VI: row = hp->k_prime_row_tiles; col = hp->s_col_tiles;        out = hp->v_inter_tiles; nc = n_sv_tiles; break;
        case GDN_GEMM_OI: row = hp->q_prime_row_tiles; col = hp->s_col_tiles;        out = hp->o_inter_tiles; nc = n_sv_tiles; break;
        case GDN_GEMM_DL: row = hp->inv_row_tiles;     col = hp->v_prime_col_tiles;  out = hp->delta_tiles;   nc = n_sv_tiles; nd = 2; break;
        case GDN_GEMM_OA: row = hp->a_row_tiles;       col = hp->delta_col_tiles;    out = hp->o_intra_tiles; nc = n_sv_tiles; nd = 2; break;
        case GDN_GEMM_SU: row = hp->d_row_tiles;       col = hp->k_col_tiles_64x128; out = hp->s_update_tiles; nr = n_sv_tiles; nc = n_sv_tiles; nd = 2; break;
        case GDN_GEMM_IA: row = (const __fp16 *) hp->vtcm_m; col = hp->inv_row_tiles; out = (__fp16 *) hp->vtcm_tmp;
                          nr = 1; nc = 1; nd = 1; break;
        default:          row = (const __fp16 *) hp->vtcm_m + HMX_FP16_TILE_N_ELMS; col = (const __fp16 *) hp->vtcm_tmp;
                          out = hp->inv_row_tiles + 2 * HMX_FP16_TILE_N_ELMS; nr = 1; nc = 1; nd = 1; break;
    }

    t->row_tiles   = row;
    t->col_tiles   = col;
    t->out_tiles   = out;
    t->n_row_tiles = nr;
    t->n_col_tiles = nc;
    t->n_dot_tiles = nd;
    t->dot_stride  = nd * HMX_FP16_TILE_N_ELMS;
    t->hmx_scales  = scales;
}

// Fills the GEMM task of kind `kind` for every head of the batch. The tasks depend on the batch only (head pointers,
// tile counts, scales), not on the chunk, so they are filled once per batch before the chunk loop.
static inline void gdn_hmx_fill(struct htp_gdn_hmx_gemm_task * tasks, uint32_t kind, const struct htp_gdn_head_ptrs * heads,
                                uint32_t n_batch, uint32_t n_sv_tiles, uint8_t * scales) {
    for (uint32_t h = 0; h < n_batch; ++h) {
        gdn_gemm_fill(&tasks[h], kind, &heads[h], n_sv_tiles, scales);
    }
}

// Issues one GEMM kind for every head of the batch (tasks already filled): one descriptor (batched) or one per head (legacy).
static inline void gdn_hmx_issue(hmx_queue_t q, struct htp_gdn_hmx_gemm_task * tasks, struct htp_gdn_hmx_gemm_batch * gb,
                                 uint32_t n_batch, bool batched) {
    if (batched) {
        gb->tasks = tasks;
        gb->n     = n_batch;
        hmx_queue_push(q, hmx_queue_make_desc(htp_gdn_hmx_gemm_batch_worker, gb));
    } else {
        for (uint32_t h = 0; h < n_batch; ++h) {
            hmx_queue_push(q, hmx_queue_make_desc(htp_gdn_hmx_gemm_worker, &tasks[h]));
        }
    }
}

// Pops the descriptors of one kind, in issue order (waits for their completion).
static inline void gdn_hmx_wait(hmx_queue_t q, uint32_t n_batch, bool batched) {
    const uint32_t n = batched ? 1 : n_batch;
    for (uint32_t i = 0; i < n; ++i) {
        hmx_queue_pop(q);
    }
}

// Several GEMM kinds as ONE queue descriptor (HTP_GDN_OPT_HMX_BATCH): every push costs the producer a futex wake (~1.1 us),
// so kinds whose results are consumed at the same point of the chunk are issued together. The tasks run in list order.
#define GDN_HMX_GROUP_MAX 3
struct htp_gdn_hmx_gemm_group {
    const struct htp_gdn_hmx_gemm_task * tasks[GDN_HMX_GROUP_MAX];
    uint32_t                             n_lists;
    uint32_t                             n;
};

static void htp_gdn_hmx_gemm_group_worker(void * data) {
    const struct htp_gdn_hmx_gemm_group * g = (const struct htp_gdn_hmx_gemm_group *) data;
    for (uint32_t l = 0; l < g->n_lists; ++l) {
        for (uint32_t i = 0; i < g->n; ++i) {
            htp_gdn_hmx_gemm_exec(&g->tasks[l][i]);
        }
    }
}

// Issues the kinds kinds[0..n_k) (tasks already filled) for every head of the batch; returns the number of pops to wait.
static inline uint32_t gdn_hmx_issue_group(hmx_queue_t q, struct htp_gdn_hmx_gemm_group * grp,
                                           struct htp_gdn_hmx_gemm_task (*tasks)[HTP_GDN_MAX_BATCH],
                                           const uint32_t * kinds, uint32_t n_k, uint32_t n_batch, bool batched) {
    if (batched) {
        grp->n_lists = n_k;
        grp->n       = n_batch;
        for (uint32_t l = 0; l < n_k; ++l) {
            grp->tasks[l] = tasks[kinds[l]];
        }
        hmx_queue_push(q, hmx_queue_make_desc(htp_gdn_hmx_gemm_group_worker, grp));
        return 1;
    }
    for (uint32_t l = 0; l < n_k; ++l) {
        for (uint32_t h = 0; h < n_batch; ++h) {
            hmx_queue_push(q, hmx_queue_make_desc(htp_gdn_hmx_gemm_worker, &tasks[kinds[l]][h]));
        }
    }
    return n_k * n_batch;
}

static inline void gdn_hmx_wait_n(hmx_queue_t q, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        hmx_queue_pop(q);
    }
}

// One pass of a phase: the jobs of heads [h0, h0 + n_jobs) of the batch, exactly as gdn_run_phase_ex runs them for h0.
static inline void gdn_run_pass(work_queue_t wp, work_queue_func_t fn, const struct htp_gdn_batch_context * bctx, uint32_t h0) {
    struct htp_gdn_batch_context pass = *bctx;
    pass.heads       = bctx->heads + h0;
    pass.n_batch     = hex_smin(bctx->n_jobs, bctx->n_batch - h0);
    pass.n_slices    = (bctx->split_ok != 0 && 2 * pass.n_batch <= bctx->n_jobs) ? 2 : 1;
    pass.slice_shift = pass.n_slices >> 1;
    work_queue_run(wp, fn, &pass, pass.n_batch * pass.n_slices);
}

// HTP_GDN_FIX_HMX_SPLIT: one batched descriptor for GEMM kind `kind` over heads [h0, h0 + n) (tasks already filled). Its
// batch object is separate from the other descriptors of the kind, because the queue reads the object when the descriptor runs.
static inline void gdn_hmx_issue_range(hmx_queue_t q, struct htp_gdn_hmx_gemm_task * tasks, struct htp_gdn_hmx_gemm_batch * gb,
                                       uint32_t h0, uint32_t n) {
    gb->tasks = tasks + h0;
    gb->n     = n;
    hmx_queue_push(q, hmx_queue_make_desc(htp_gdn_hmx_gemm_batch_worker, gb));
}

// A work-queue run has one job per head; a batch wider than n_jobs runs in passes of n_jobs heads. The head passes of a phase touch disjoint heads (and the scratch block of job i is only used by thread i), so a batch wider than n_jobs runs all its passes in ONE work-queue run: thread i executes job i of pass 0, then job i of pass 1 (when it has one).
// This drops the barrier and wake-up between the passes. With HTP_GDN_FIX_PASS_SLICE a pass of at most n_jobs/2 heads runs two jobs per head (n_slices = 2, see gdn_job_head).
#define GDN_MAX_MERGED_PASSES 2
struct htp_gdn_merged_phase {
    work_queue_func_t                  fn;
    uint32_t                           n_pass;
    uint32_t                           n_jobs[GDN_MAX_MERGED_PASSES];
    struct htp_gdn_batch_context       pass[GDN_MAX_MERGED_PASSES];
};

static void gdn_merged_phase_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    const struct htp_gdn_merged_phase * m = (const struct htp_gdn_merged_phase *) data;
    for (uint32_t p = 0; p < m->n_pass; ++p) {
        if (i < m->n_jobs[p]) {
            m->fn(m->n_jobs[p], i, (void *) &m->pass[p]);
        }
    }
}

static inline void gdn_run_phase_ex(work_queue_t wp, work_queue_func_t fn, const struct htp_gdn_batch_context * bctx,
                                    bool allow_slices) {
    const uint32_t n_pass = (bctx->n_batch + bctx->n_jobs - 1) / bctx->n_jobs;
    if (n_pass >= 2 && n_pass <= GDN_MAX_MERGED_PASSES) {
        struct htp_gdn_merged_phase m;
        m.fn     = fn;
        m.n_pass = n_pass;
        for (uint32_t p = 0; p < n_pass; ++p) {
            const uint32_t h0 = p * bctx->n_jobs;
            m.pass[p]          = *bctx;
            m.pass[p].heads    = bctx->heads + h0;
            m.pass[p].n_batch  = hex_smin(bctx->n_jobs, bctx->n_batch - h0);
            m.pass[p].n_slices = (allow_slices && bctx->split_ok != 0 && 2 * m.pass[p].n_batch <= bctx->n_jobs) ? 2 : 1;
            m.pass[p].slice_shift = m.pass[p].n_slices >> 1;
            m.n_jobs[p]        = m.pass[p].n_batch * m.pass[p].n_slices;
        }
        work_queue_run(wp, gdn_merged_phase_worker, &m, m.n_jobs[0]);
        return;
    }
    for (uint32_t h0 = 0; h0 < bctx->n_batch; h0 += bctx->n_jobs) {
        struct htp_gdn_batch_context pass = *bctx;
        pass.heads    = bctx->heads + h0;
        pass.n_batch  = hex_smin(bctx->n_jobs, bctx->n_batch - h0);
        pass.n_slices = (allow_slices && bctx->split_ok != 0 && 2 * pass.n_batch <= bctx->n_jobs) ? 2 : 1;
        pass.slice_shift = pass.n_slices >> 1;
        work_queue_run(wp, fn, &pass, pass.n_batch * pass.n_slices);
    }
}

static inline void gdn_run_phase(work_queue_t wp, work_queue_func_t fn, const struct htp_gdn_batch_context * bctx) {
    gdn_run_phase_ex(wp, fn, bctx, true);
}

// QKSHARE: padding of a short last chunk in q and k, written once per Q/K head (owners); phase 1a skips it.
static void gdn_zero_pad_qk_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    const struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t S_v   = bctx->S_v;
    const uint32_t valid = hex_smin(64, bctx->kparams->n_tokens - bctx->c * 64);
    const HVX_Vector vzero = Q6_V_vzero();

    for (uint32_t t = valid; t < 64; ++t) {
        for (uint32_t j = 0; j < S_v; j += 32) {
            if (head->own_q) {
                hvx_vmem(head->q_f32[bctx->curr_buf] + t * S_v + j) = vzero;
            }
            if (head->own_k) {
                hvx_vmem(head->k_f32[bctx->curr_buf] + t * S_v + j) = vzero;
            }
        }
    }
}

// V head (within its sequence) at batch position pos. QKSHARE: the V heads of one Q/K head are adjacent,
// pos = g * r + m -> head g + m * Hq with r = H / Hq (a permutation of 0..H-1); otherwise the identity.
static inline uint32_t gdn_batch_head_iv1(uint32_t pos, uint32_t H, uint32_t Hq, bool qkshare) {
    if (!qkshare || Hq == 0 || (H % Hq) != 0) {
        return pos;
    }
    const uint32_t r = H / Hq;
    return (pos / r) + (pos % r) * Hq;
}

// Chunk-0 inputs of a batch: q, k, v of every head (q and k once per Q/K head with QKSHARE) and the g and beta rows.
static inline void gdn_push_chunk0_inputs(dma_queue * dma_q, const struct htp_gdn_head_ptrs * heads, uint32_t n_batch,
                                          const struct htp_tensor * q, const struct htp_tensor * k, const struct htp_tensor * v,
                                          const struct htp_tensor * g, const struct htp_tensor * beta, uint32_t iv3,
                                          uint32_t head_in_seq, bool qkshare, uint32_t H, float * vtcm_g_raw0,
                                          float * vtcm_b_raw0, uint32_t chunk0_tokens, uint32_t S_v) {
    for (uint32_t h = 0; h < n_batch; ++h) {
        gdn_dma_push_chunk_inputs_slot(dma_q, &heads[h], 0, q, k, v, 0, chunk0_tokens, S_v);
    }
    if (qkshare) {
        gdn_dma_push_chunk_gb_full(dma_q, vtcm_g_raw0, vtcm_b_raw0, g, beta, iv3, 0, chunk0_tokens, H);
    } else {
        gdn_dma_push_chunk_gb(dma_q, vtcm_g_raw0, vtcm_b_raw0, g, beta, iv3, head_in_seq, 0, chunk0_tokens, n_batch);
    }
}

static int gated_delta_net_f32_hmx_chunked(
    struct htp_ops_context * octx,
    const struct htp_gdn_kernel_params * kparams,
    uint32_t row_start,
    uint32_t nrows
) {
    // The first chunk runs ~35 us slower than later ones (cold instruction path: other ops evicted the phase workers from
    // the L2). Prefetch the contiguous range of the phase workers (init_state .. solve, ~20 KB) while the state DMA is in flight.
    {
        const uintptr_t c0 = (uintptr_t) (void *) gdn_hvx_init_state_worker;
        const uintptr_t c1 = (uintptr_t) (void *) gdn_build_inv_l_blocks;
        const uintptr_t lo = (c0 < c1 ? c0 : c1) & ~(uintptr_t) 127;
        const uintptr_t hi = ((c0 < c1 ? c1 : c0) + 0x700 + 127) & ~(uintptr_t) 127;
        if (hi - lo <= 65536) {
            hex_l2fetch((const void *) lo, 128, 128, (uint32_t) ((hi - lo) >> 7));
        }
    }
    const struct htp_tensor * q         = octx->src[0];
    const struct htp_tensor * k         = octx->src[1];
    const struct htp_tensor * v         = octx->src[2];
    const struct htp_tensor * g         = octx->src[3];
    const struct htp_tensor * beta      = octx->src[4];
    const struct htp_tensor * state     = octx->src[5];
    const struct htp_tensor * dst       = octx->dst;
    const struct htp_tensor * dst_cache = octx->dsts[1];

    const uint32_t S_v        = kparams->S_v;
    const uint32_t H          = kparams->H;
    const uint32_t n_tokens   = kparams->n_tokens;
    const float    scale      = kparams->scale;
    const uint32_t chunk_size = kparams->chunk_size;
    const uint32_t n_chunks   = kparams->n_chunks;
    const uint32_t n_sv_tiles = S_v / 32;
    const uint32_t opt        = kparams->opt;
    const bool     batched    = (opt & HTP_GDN_OPT_HMX_BATCH) != 0;
    const bool     qkshare    = (opt & HTP_GDN_OPT_QKSHARE) != 0;
    const uint32_t gb_row     = htp_gdn_gb_row(opt, H, kparams->n_heads_batch);
    const uint32_t Hq         = q->ne[1];

    struct htp_gdn_hmx_vtcm_layout L;
    htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_size, kparams->n_heads_batch, kparams->n_threads, kparams->pipeline != 0, gb_row,
                                  htp_gdn_skew_bytes(kparams->fix, kparams->skew_units, n_tokens));

    if (L.total_bytes > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    float * vtcm_g_raw[2] = {
        VTCM_LAYOUT_PTR(float, vtcm_base, L.off_g_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L.off_g_raw[1]) : VTCM_LAYOUT_PTR(float, vtcm_base, L.off_g_raw[0])
    };
    float * vtcm_b_raw[2] = {
        VTCM_LAYOUT_PTR(float, vtcm_base, L.off_b_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L.off_b_raw[1]) : VTCM_LAYOUT_PTR(float, vtcm_base, L.off_b_raw[0])
    };

    uint8_t * vtcm_scales_1 = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, L.off_scales_1);
    hmx_init_column_scales(vtcm_scales_1, Q6_V_vsplat_R(0x3c00));

    hmx_queue_t  hmx_q = octx->ctx->hmx_queue;
    dma_queue *  dma_q = octx->ctx->dma[0];
    work_queue_t wp    = octx->ctx->work_queue;

    // static: batch state of up to HTP_GDN_MAX_BATCH heads (ops run one at a time on the session)
    static struct htp_gdn_head_ptrs      heads[HTP_GDN_MAX_BATCH];
    static struct htp_gdn_hmx_gemm_task  gemm_tasks[GDN_GEMM_COUNT][HTP_GDN_MAX_BATCH];
    static struct htp_gdn_hmx_gemm_batch gemm_batch[GDN_GEMM_COUNT];
    static struct htp_gdn_hmx_gemm_batch gemm_batch_p2[GDN_GEMM_COUNT]; // second-pass descriptors of the split kinds (DL, OA)
    static struct htp_gdn_hmx_gemm_group gemm_grp[3];   // KK+VI, QK+OI, IA+IB (one queue descriptor each)

    struct htp_gdn_batch_context bctx;
    bctx.heads       = heads;
    bctx.vtcm_g_raw  = NULL;
    bctx.vtcm_b_raw  = NULL;
    bctx.curr_buf    = 0;
    bctx.c           = 0;
    bctx.n_batch     = 1;
    bctx.S_v         = S_v;
    bctx.scale       = scale;
    bctx.octx        = octx;
    bctx.kparams     = kparams;
    bctx.n_jobs      = octx->n_threads;
    bctx.gb_row      = gb_row;
    bctx.qk_pre      = qkshare ? 1 : 0;
    bctx.gate_a      = NULL;
    bctx.lossy       = (octx->flags & HTP_OPFLAGS_LOSSY_GDN_MASK) >> HTP_OPFLAGS_LOSSY_GDN_SHIFT;
    if (octx->flags & HTP_OPFLAGS_GDN_GATE_FOLD) {
        float * const a_vtcm = VTCM_LAYOUT_PTR(float, vtcm_base, L.off_gate_a);
        dma_queue_push(dma_q, dma_make_data((uint8_t *) a_vtcm, octx->src[6]->data), 128, 128, H * sizeof(float), 1);
        dma_queue_pop(dma_q);
        bctx.gate_a = a_vtcm;
    }
    bctx.state_merge = (opt & HTP_GDN_OPT_STATE_MERGE) ? 1 : 0;
    bctx.n_slices      = 1;
    bctx.slice_shift   = 0;
    bctx.scratch        = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L.off_thread_scratch);
    bctx.scratch_stride = (uint32_t) L.st_scratch;
    // HTP_GDN_FIX bits, chunked HMX with n_tokens >= 64 only (the decode and verify batches keep the original path)
    const bool fix_on   = n_tokens >= 64;
    bctx.split_ok       = (fix_on && (kparams->fix & HTP_GDN_FIX_PASS_SLICE) != 0 && !bctx.state_merge &&
                           (opt & HTP_GDN_OPT_UP_LEGACY_EXP) == 0) ? 1 : 0;
    bctx.prologue_early = (fix_on && (kparams->fix & HTP_GDN_FIX_PROLOGUE) != 0) ? 1 : 0;
    bctx.hmx_split_ok   = (fix_on && (kparams->fix & HTP_GDN_FIX_HMX_SPLIT) != 0 && batched) ? 1 : 0;

    uint32_t n_batch = 1;
    for (uint32_t r = row_start; r < row_start + nrows; r += n_batch) {
        const uint32_t head_in_seq         = fastmodulo(r, H, &kparams->div_H);
        const uint32_t iv3                 = fastdiv(r, &kparams->div_H);
        const uint32_t heads_left_in_seq   = H - head_in_seq;
        const uint32_t heads_left_in_range = (row_start + nrows) - r;
        n_batch = hex_smin((uint32_t) kparams->n_heads_batch, hex_smin(heads_left_in_seq, heads_left_in_range));

        // wake the HMX queue thread now: its first sleep-to-run latency (about 10 us) is otherwise paid by the first GEMM issue
        hmx_queue_signal(hmx_q, HMX_QUEUE_WAKEUP);
        // Per-op head coordinates first (cheap): the state-in DMA only needs the state slot and its source address. The
        // layout pointers (about 8 us for 11 heads) are filled while that DMA is in flight.
        for (uint32_t h = 0; h < n_batch; ++h) {
            const uint32_t iv1 = gdn_batch_head_iv1(head_in_seq + h, H, Hq, qkshare);
            gdn_init_head_op(&heads[h], &L, vtcm_base, h, iv1, iv3, q, k, state, dst, dst_cache, kparams, S_v, H, n_tokens);
            heads[h].gb_col = qkshare ? iv1 : h;
        }
        for (uint32_t h = 0; h < n_batch; ++h) {
            dma_queue_push(dma_q, dma_make_data(heads[h].s_state, heads[h].state_in_dma),
                           S_v * sizeof(float), S_v * sizeof(float), S_v * sizeof(float), S_v);
        }
        for (uint32_t h = 0; h < n_batch; ++h) {
            gdn_init_head_layout(&heads[h], &L, vtcm_base, h);
        }

        // q/k sharing: the first slot of each (sequence, Q/K head) loads q and k; later slots alias its buffers
        uint32_t n_qk_desc = 0;
        for (uint32_t h = 0; h < n_batch; ++h) {
            if (qkshare) {
                for (uint32_t o = 0; o < h; ++o) {
                    if (heads[o].iq3 == heads[h].iq3 && heads[o].iq1 == heads[h].iq1) {
                        heads[h].own_q    = 0;
                        heads[h].q_f32[0] = heads[o].q_f32[0];
                        heads[h].q_f32[1] = heads[o].q_f32[1];
                        break;
                    }
                }
                for (uint32_t o = 0; o < h; ++o) {
                    if (heads[o].ik3 == heads[h].ik3 && heads[o].ik1 == heads[h].ik1) {
                        heads[h].own_k    = 0;
                        heads[h].k_f32[0] = heads[o].k_f32[0];
                        heads[h].k_f32[1] = heads[o].k_f32[1];
                        break;
                    }
                }
            }
            n_qk_desc += (uint32_t) heads[h].own_q + heads[h].own_k + 1; // q (owner), k (owner), v
        }
        const uint32_t n_in_desc = n_qk_desc + 2; // + g and beta of the chunk

        bctx.n_batch = n_batch;
        const bool split = bctx.hmx_split_ok != 0 && n_batch > bctx.n_jobs && n_batch <= 2 * bctx.n_jobs;
        bctx.gb_row  = qkshare ? H : n_batch; // raw g/beta row stride = the batch width (QKSHARE: all H heads)

        // (HMX queue wake-up was issued at the top of the batch)
        // chunk-0 inputs: HTP_GDN_FIX_PROLOGUE pushes them behind the state-in DMA (they do not depend on the state)
        const uint32_t chunk0_tokens = hex_smin(chunk_size, n_tokens);
        const bool early_c0 = bctx.prologue_early != 0 && n_chunks > 0;
        if (early_c0) {
            gdn_push_chunk0_inputs(dma_q, heads, n_batch, q, k, v, g, beta, iv3, head_in_seq, qkshare, H,
                                   vtcm_g_raw[0], vtcm_b_raw[0], chunk0_tokens, S_v);
        }
        for (uint32_t h = 0; h < n_batch; ++h) {
            dma_queue_pop(dma_q); // state-in: the FIFO retires the state descriptors first
        }

        if (n_chunks > 0) {
            gdn_run_phase(wp, gdn_hvx_init_state_worker, &bctx);
            if (!early_c0) {
                gdn_push_chunk0_inputs(dma_q, heads, n_batch, q, k, v, g, beta, iv3, head_in_seq, qkshare, H,
                                       vtcm_g_raw[0], vtcm_b_raw[0], chunk0_tokens, S_v);
            }
        }

        for (uint32_t kind = 0; kind < GDN_GEMM_COUNT; ++kind) {
            gdn_hmx_fill(gemm_tasks[kind], kind, heads, n_batch, n_sv_tiles, vtcm_scales_1);
        }

        for (uint32_t c = 0; c < n_chunks; ++c) {
            const uint32_t curr_buf     = c & 1;
            const uint32_t next_buf     = (c + 1) & 1;
            const uint32_t t_chunk      = c * chunk_size;
            const uint32_t valid_tokens = hex_smin(chunk_size, n_tokens - t_chunk);

            bctx.curr_buf   = curr_buf;
            bctx.c          = c;
            bctx.vtcm_g_raw = vtcm_g_raw[curr_buf];
            bctx.vtcm_b_raw = vtcm_b_raw[curr_buf];

            // inputs of this chunk (pushed one chunk ahead), then the prefetch of the next chunk
            for (uint32_t i = 0; i < n_in_desc; ++i) {
                dma_queue_pop(dma_q);
            }

            if (c + 1 < n_chunks) {
                const uint32_t next_t_chunk = (c + 1) * chunk_size;
                const uint32_t next_tokens  = hex_smin(chunk_size, n_tokens - next_t_chunk);
                for (uint32_t h = 0; h < n_batch; ++h) {
                    gdn_dma_push_chunk_inputs_slot(dma_q, &heads[h], next_buf, q, k, v, next_t_chunk, next_tokens, S_v);
                }
                if (qkshare) {
                    gdn_dma_push_chunk_gb_full(dma_q, vtcm_g_raw[next_buf], vtcm_b_raw[next_buf], g, beta, iv3, next_t_chunk, next_tokens, H);
                } else {
                    gdn_dma_push_chunk_gb(dma_q, vtcm_g_raw[next_buf], vtcm_b_raw[next_buf], g, beta, iv3, head_in_seq, next_t_chunk, next_tokens, n_batch);
                }
            }

            if (c > 0) {
                for (uint32_t h = 0; h < n_batch; ++h) {
                    dma_queue_pop(dma_q); // output of the previous chunk
                }
            }

            if (qkshare && valid_tokens < 64) {
                gdn_run_phase_ex(wp, gdn_zero_pad_qk_worker, &bctx, false);
            }
            gdn_run_phase(wp, gdn_hvx_phase1a_worker, &bctx);

            {
                static const uint32_t k_g1[2] = { GDN_GEMM_KK, GDN_GEMM_VI };
                static const uint32_t k_g2[2] = { GDN_GEMM_QK, GDN_GEMM_OI };
                static const uint32_t k_g3[2] = { GDN_GEMM_IA, GDN_GEMM_IB };
                // kk and v_inter (one descriptor): both are done long before the end of phase 1b
                const uint32_t n1 = gdn_hmx_issue_group(hmx_q, &gemm_grp[0], gemm_tasks, k_g1, 2, n_batch, batched);
                gdn_run_phase(wp, (kparams->opt & HTP_GDN_OPT_UP_LEGACY_EXP) ? gdn_hvx_phase1b_f32exp_worker : gdn_hvx_phase1b_worker, &bctx);
                gdn_hmx_wait_n(hmx_q, n1);

                // qk and o_inter: o_inter (1.9 us) runs under the solve phase
                const uint32_t n2 = gdn_hmx_issue_group(hmx_q, &gemm_grp[1], gemm_tasks, k_g2, 2, n_batch, batched);
                gdn_run_phase(wp, gdn_hvx_phase2_worker, &bctx);
                gdn_hmx_wait_n(hmx_q, n2);

                const uint32_t n3 = gdn_hmx_issue_group(hmx_q, &gemm_grp[2], gemm_tasks, k_g3, 2, n_batch, batched);
                if (split) {
                    // DL of pass 1 is queued behind IA and IB (FIFO) and runs under pass 2 of the phase
                    gdn_run_pass(wp, gdn_hvx_phase3_worker, &bctx, 0);
                    gdn_hmx_issue_range(hmx_q, gemm_tasks[GDN_GEMM_DL], &gemm_batch[GDN_GEMM_DL], 0, bctx.n_jobs);
                    gdn_run_pass(wp, gdn_hvx_phase3_worker, &bctx, bctx.n_jobs);
                    gdn_hmx_issue_range(hmx_q, gemm_tasks[GDN_GEMM_DL], &gemm_batch_p2[GDN_GEMM_DL], bctx.n_jobs, n_batch - bctx.n_jobs);
                } else {
                    gdn_run_phase(wp, gdn_hvx_phase3_worker, &bctx);
                }
                gdn_hmx_wait_n(hmx_q, n3); // inverse blocks 0 and 1
            }

            if (split) {
                gdn_hmx_wait(hmx_q, 1, true);   // delta, pass 1
                gdn_run_pass(wp, gdn_hvx_phase4_worker, &bctx, 0);
                gdn_hmx_issue_range(hmx_q, gemm_tasks[GDN_GEMM_OA], &gemm_batch[GDN_GEMM_OA], 0, bctx.n_jobs);
                gdn_hmx_wait(hmx_q, 1, true);   // delta, pass 2
                gdn_run_pass(wp, gdn_hvx_phase4_worker, &bctx, bctx.n_jobs);
                gdn_hmx_issue_range(hmx_q, gemm_tasks[GDN_GEMM_OA], &gemm_batch_p2[GDN_GEMM_OA], bctx.n_jobs, n_batch - bctx.n_jobs);
                gdn_hmx_issue(hmx_q, gemm_tasks[GDN_GEMM_SU], &gemm_batch[GDN_GEMM_SU], n_batch, batched);
                gdn_run_phase(wp, gdn_hvx_phase5a_worker, &bctx); // o_inter, while the HMX computes o_intra
                gdn_hmx_wait(hmx_q, 1, true);   // o_intra, pass 1
                gdn_run_pass(wp, gdn_hvx_phase5_worker, &bctx, 0);
                gdn_hmx_wait(hmx_q, 1, true);   // o_intra, pass 2
                gdn_run_pass(wp, gdn_hvx_phase5_worker, &bctx, bctx.n_jobs);
            } else {
                gdn_hmx_issue(hmx_q, gemm_tasks[GDN_GEMM_DL], &gemm_batch[GDN_GEMM_DL], n_batch, batched);
                gdn_hmx_wait(hmx_q, n_batch, batched); // delta
                gdn_run_phase(wp, gdn_hvx_phase4_worker, &bctx);

                gdn_hmx_issue(hmx_q, gemm_tasks[GDN_GEMM_OA], &gemm_batch[GDN_GEMM_OA], n_batch, batched);
                gdn_hmx_issue(hmx_q, gemm_tasks[GDN_GEMM_SU], &gemm_batch[GDN_GEMM_SU], n_batch, batched);
                gdn_run_phase(wp, gdn_hvx_phase5a_worker, &bctx); // o_inter, while the HMX computes o_intra
                gdn_hmx_wait(hmx_q, n_batch, batched); // o_intra
                gdn_run_phase(wp, gdn_hvx_phase5_worker, &bctx);
            }

            for (uint32_t h = 0; h < n_batch; ++h) {
                const dma_addr_t attn_chunk_dma = dst->data +
                    ((uint64_t) heads[h].iv3 * n_tokens * H + (uint64_t) t_chunk * H + heads[h].iv1) * S_v * sizeof(float);
                dma_queue_push(dma_q, dma_make_data(attn_chunk_dma, heads[h].o_f32[curr_buf]),
                               dst->nb[1], S_v * sizeof(float), S_v * sizeof(float), valid_tokens);
            }

            gdn_hmx_wait(hmx_q, n_batch, batched); // s_update
            gdn_run_phase(wp, gdn_hvx_phase6_worker, &bctx);
        }

        if (n_chunks > 0) {
            for (uint32_t h = 0; h < n_batch; ++h) {
                dma_queue_pop(dma_q);
            }
        }

        for (uint32_t h = 0; h < n_batch; ++h) {
            dma_queue_push(dma_q, dma_make_data(heads[h].state_out_dma, heads[h].s_state),
                           S_v * sizeof(float), S_v * sizeof(float), S_v * sizeof(float), S_v);
        }
        for (uint32_t h = 0; h < n_batch; ++h) {
            dma_queue_pop(dma_q);
        }
    }

    dma_queue_flush(dma_q);
    return HTP_STATUS_OK;
}

// --------------------------------------------------------------------------------------------------------------- Ticket path (scalar recurrence): f16 state storage (LLAMA_GDN_STATE_F16) and/or async snapshot planes (GGML_HEXAGON_GDN_SNAP=async). The default path (gated_delta_net_f32_pp_thread / _tg_thread) is not used here.
// Every DMA pushed on this thread's queue takes a ticket; the queue retires in FIFO order, so waiting for a ticket retires everything issued before it. --------------------------------------------------------------------------------------------------------------- A buffer is refilled or reused only after the ticket of its last reader.

struct htp_gdn_tickets {
    dma_queue * q;
    uint32_t    pushed;
    uint32_t    popped;
};

static inline uint32_t gdn_tk_push(struct htp_gdn_tickets * tk, dma_data d, size_t dst_stride, size_t src_stride,
                                   size_t row_size, size_t nrows) {
    while (!dma_queue_push(tk->q, d, dst_stride, src_stride, row_size, nrows)) {
        dma_queue_pop(tk->q);   // ring full: retire the oldest entry, then retry
        tk->popped++;
    }
    return ++tk->pushed;
}

static inline void gdn_tk_wait(struct htp_gdn_tickets * tk, uint32_t ticket) {
    while ((int32_t) (ticket - tk->popped) > 0) {
        dma_queue_pop(tk->q);
        tk->popped++;
    }
}

// nrows rows of S_v contiguous elements: one head, or one part of a head (cache side and VTCM side alike)
static inline uint32_t gdn_tk_fetch(struct htp_gdn_tickets * tk, uint8_t * dst, dma_addr_t src, uint32_t nrows, uint32_t S_v, size_t el) {
    const size_t row = (size_t) S_v * el;
    return gdn_tk_push(tk, dma_make_data(dst, src), row, row, row, nrows);
}

static inline uint32_t gdn_tk_store(struct htp_gdn_tickets * tk, dma_addr_t dst, const uint8_t * src, uint32_t nrows, uint32_t S_v, size_t el) {
    const size_t row = (size_t) S_v * el;
    return gdn_tk_push(tk, dma_make_data(dst, src), row, row, row, nrows);
}

// f16 storage <-> f32 compute. n is a multiple of 64 (S_v * S_v for S_v in {16, 32, 64, 128}).
static inline void gdn_f16_to_f32(float * dst, const uint8_t * src, uint32_t n) {
    const __fp16 * s = (const __fp16 *) src;
    for (uint32_t i = 0; i < n; i += 64) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(hvx_vmemu(s + i));
        *(HVX_UVector *) (dst + i)      = Q6_V_lo_W(p);
        *(HVX_UVector *) (dst + i + 32) = Q6_V_hi_W(p);
    }
}

static inline void gdn_f32_to_f16(uint8_t * dst, const float * src, uint32_t n) {
    __fp16 * d = (__fp16 *) dst;
    for (uint32_t i = 0; i < n; i += 64) {
        HVX_Vector v0 = hvx_vmemu(src + i);
        HVX_Vector v1 = hvx_vmemu(src + i + 32);
        *(HVX_UVector *) (d + i) = hvx_vec_f32_to_f16(v0, v1);
    }
}

// carried state rounded to f16 precision in place (f32 container): what a decode step that stores f16 leaves behind
static inline void gdn_round_f16_inplace(float * s, uint32_t n) {
    for (uint32_t i = 0; i < n; i += 64) {
        HVX_Vector v0 = hvx_vmemu(s + i);
        HVX_Vector v1 = hvx_vmemu(s + i + 32);
        HVX_VectorPair p = hvx_vec_f16_to_f32(hvx_vec_f32_to_f16(v0, v1));
        *(HVX_UVector *) (s + i)      = Q6_V_lo_W(p);
        *(HVX_UVector *) (s + i + 32) = Q6_V_hi_W(p);
    }
}

// one plane (snapshot or final state) from the VTCM f32 state to f16 or f32 element storage
static inline void gdn_stage_plane(uint8_t * dst, const float * src, uint32_t n, uint8_t to_f16) {
    if (to_f16) {
        gdn_f32_to_f16(dst, src, n);
    } else {
        hvx_copy_f32_uu(dst, (const uint8_t *) src, n);
    }
}

static void gated_delta_net_f32_ticket_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_gdn_context * gctx = (struct htp_gdn_context *) data;
    struct htp_ops_context * octx = gctx->octx;
    const struct htp_gdn_kernel_params * kparams = gctx->kparams;
    const struct htp_gdn_io_layout * io = &gctx->io;

    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;
    const struct htp_tensor * dst_cache = octx->dsts[1];

    const uint32_t S_v      = kparams->S_v;
    const uint32_t H        = kparams->H;
    const uint32_t n_tokens = kparams->n_tokens;
    const uint32_t n_seqs   = kparams->n_seqs;
    const uint32_t K        = kparams->K;
    const uint32_t n_el     = S_v * S_v;
    // unit = (head, part): part p holds state rows [p * nj, (p + 1) * nj) of its head (nj = S_v / n_parts)
    const uint32_t n_parts  = gctx->n_parts;
    const uint32_t nj       = S_v / n_parts;
    const uint32_t n_el_u   = nj * S_v;
    const uint32_t n_units  = gctx->nrows * n_parts;
    const int64_t snap_d0   = (int32_t) octx->op_params[3];
    const int64_t snap_n    = (int32_t) octx->op_params[4];
    const float   scale     = kparams->scale;

    if (ith >= n_units) {
        return;
    }

    const bool   in_f16   = state->type == HTP_TYPE_F16;
    const bool   out_f16  = dst_cache != NULL && dst_cache->type == HTP_TYPE_F16;
    const bool   snap_on  = K > 1 && kparams->snap_mode != HTP_GDN_SNAP_SKIP;
    const bool   snap_async = kparams->snap_mode == HTP_GDN_SNAP_ASYNC || kparams->snap_mode == HTP_GDN_SNAP_VERSION;
    const size_t el_in    = in_f16 ? sizeof(uint16_t) : sizeof(float);
    const size_t el_out   = out_f16 ? sizeof(uint16_t) : sizeof(float);

    float *      dst_base   = (float *) (uintptr_t) dst->data;
    const dma_addr_t plane_base = dst_cache ? dst_cache->data : (dst->data + S_v * H * n_tokens * n_seqs * sizeof(float));
    const uint64_t   plane_stride = (uint64_t) kparams->state_size_per_snap * el_out;   // bytes between planes

    uint8_t * tbase = (uint8_t *) gctx->vtcm_base + io->bytes_per_thread * ith;
    float *   s_work[2] = { (float *) tbase, (float *) (tbase + io->state_aligned) };
    uint8_t * io_in[2]  = { tbase + io->off_io_in,  tbase + io->off_io_in  + io->io_aligned };
    uint8_t * io_out[2] = { tbase + io->off_io_out, tbase + io->off_io_out + io->io_aligned };
    uint8_t * ring[HTP_GDN_SNAP_RING];
    for (uint32_t r = 0; r < HTP_GDN_SNAP_RING; r++) {
        ring[r] = tbase + io->off_ring + r * io->ring_slot;
    }

    const struct fastdiv_values * fd_H   = &kparams->div_H;
    const struct fastdiv_values * fd_q1  = &kparams->div_q1;
    const struct fastdiv_values * fd_k1  = &kparams->div_k1;
    const struct fastdiv_values * fd_rq3 = &kparams->div_rq3;
    const struct fastdiv_values * fd_rk3 = &kparams->div_rk3;

    struct htp_gdn_tickets tk = { octx->ctx->dma[ith], 0, 0 };
    uint32_t fetch_tk[2] = { 0, 0 };                     // ticket of the state load that fills buffer b
    uint32_t store_tk[2] = { 0, 0 };                     // ticket of the last final-state store reading io_out[b]
    uint32_t ring_tk[HTP_GDN_SNAP_RING];
    uint32_t snap_count = 0;
    for (uint32_t r = 0; r < HTP_GDN_SNAP_RING; r++) {
        ring_tk[r] = 0;
    }

    // this thread takes units ith, ith + nth, ...; prefetch the first two of them (buffers 0 and 1)
    uint32_t u_prefetch = ith;
    for (int b = 0; b < 2 && u_prefetch < n_units; b++) {
        const uint32_t ir   = gctx->row_start + u_prefetch / n_parts;
        const uint32_t part = u_prefetch % n_parts;
        const uint32_t piv1 = fastmodulo(ir, H, fd_H);
        const uint32_t piv3 = fastdiv(ir, fd_H);
        const dma_addr_t ps_in = state->data + (uint64_t) piv3 * state->nb[3] + (uint64_t) piv1 * n_el * el_in
                                 + (uint64_t) part * n_el_u * el_in;
        fetch_tk[b] = gdn_tk_fetch(&tk, in_f16 ? io_in[b] : (uint8_t *) s_work[b], ps_in, nj, S_v, el_in);
        u_prefetch += nth;
    }

    int curr = 0;
    for (uint32_t u = ith; u < n_units; u += nth) {
        const uint32_t ir   = gctx->row_start + u / n_parts;
        const uint32_t part = u % n_parts;
        const uint32_t j0   = part * nj;                 // first state row of this unit inside its head

        gdn_tk_wait(&tk, fetch_tk[curr]);
        float * s_cur = s_work[curr];
        if (in_f16) {
            gdn_f16_to_f32(s_cur, io_in[curr], n_el_u);
        }

        const uint32_t iv1 = fastmodulo(ir, H, fd_H);
        const uint32_t iv3 = fastdiv(ir, fd_H);
        const uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        const uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        const uint32_t iq3 = fastdiv(iv3, fd_rq3);
        const uint32_t ik3 = fastdiv(iv3, fd_rk3);

        const uint64_t   head_off = ((uint64_t) iv3 * H + iv1) * n_el * el_out;   // byte offset of this head in a plane
        const uint64_t   part_off = (uint64_t) j0 * S_v * el_out;                  // and of this part inside it
        const dma_addr_t s_out_addr = plane_base + head_off + part_off;
        float * attn_data = dst_base + ((uint64_t) iv3 * n_tokens * H + iv1) * S_v;

        for (uint32_t t = 0; t < n_tokens; ++t) {
            const float * q_t = (const float *) ((const uint8_t *) (uintptr_t) q->data +
                    (uint64_t) iq3 * q->nb[3] + (uint64_t) t * q->nb[2] + (uint64_t) iq1 * q->nb[1]);
            const float * k_t = (const float *) ((const uint8_t *) (uintptr_t) k->data +
                    (uint64_t) ik3 * k->nb[3] + (uint64_t) t * k->nb[2] + (uint64_t) ik1 * k->nb[1]);
            const float * v_t = (const float *) ((const uint8_t *) (uintptr_t) v->data +
                    (uint64_t) iv3 * v->nb[3] + (uint64_t) t * v->nb[2] + (uint64_t) iv1 * v->nb[1]);
            const float * g_t = (const float *) ((const uint8_t *) (uintptr_t) g->data +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) t * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
            const float beta_val = *(const float *) ((const uint8_t *) (uintptr_t) beta->data +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) t * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);

            if (kparams->kda) {
                gdn_step_kda_f32(s_cur, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v, j0, nj);
            } else {
                gdn_step_scalar_f32(s_cur, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v, j0, nj);
            }

            if (in_f16) {
                gdn_round_f16_inplace(s_cur, n_el_u);
            }

            if (snap_on) {
                // plane d = distance from the last token; plane 0 (the final state) is stored after the loop
                const int64_t d = (int64_t) n_tokens - 1 - (int64_t) t;
                const bool in_range = (snap_n == 0) || (d >= snap_d0 && d < snap_d0 + snap_n);
                if (d > 0 && d < (int64_t) K && in_range) {
                    const dma_addr_t snap_addr = plane_base + (uint64_t) d * plane_stride + head_off + part_off;
                    if (snap_async) {
                        const uint32_t slot = snap_count % HTP_GDN_SNAP_RING;
                        gdn_tk_wait(&tk, ring_tk[slot]);                 // the previous plane in this slot has landed
                        gdn_stage_plane(ring[slot], s_cur, n_el_u, out_f16);
                        ring_tk[slot] = gdn_tk_store(&tk, snap_addr, ring[slot], nj, S_v, el_out);
                        snap_count++;
                    } else {
                        gdn_stage_plane((uint8_t *) (uintptr_t) snap_addr, s_cur, n_el_u, out_f16);
                    }
                }
            }

            attn_data += (uint64_t) S_v * H;
        }

        // final state = plane 0, this part only
        if (out_f16) {
            gdn_tk_wait(&tk, store_tk[curr]);                // io_out[curr] is reused: retire its previous store
            gdn_f32_to_f16(io_out[curr], s_cur, n_el_u);
            store_tk[curr] = gdn_tk_store(&tk, s_out_addr, io_out[curr], nj, S_v, el_out);
        } else {
            store_tk[curr] = gdn_tk_store(&tk, s_out_addr, (const uint8_t *) s_cur, nj, S_v, el_out);
        }

        // this buffer receives unit u + 2 * nth (same parity), issued after the store above
        if (u_prefetch < n_units) {
            const uint32_t ir_p   = gctx->row_start + u_prefetch / n_parts;
            const uint32_t part_p = u_prefetch % n_parts;
            const uint32_t piv1 = fastmodulo(ir_p, H, fd_H);
            const uint32_t piv3 = fastdiv(ir_p, fd_H);
            const dma_addr_t ps_in = state->data + (uint64_t) piv3 * state->nb[3] + (uint64_t) piv1 * n_el * el_in
                                     + (uint64_t) part_p * n_el_u * el_in;
            fetch_tk[curr] = gdn_tk_fetch(&tk, in_f16 ? io_in[curr] : (uint8_t *) s_work[curr], ps_in, nj, S_v, el_in);
            u_prefetch += nth;
        }

        curr ^= 1;
    }
    dma_queue_flush(tk.q);
}

// --------------------------------------------------------------------------------------------------------------- HTP_GDN_SNAP_VERSION thread (D2Snap). A buffer is rewritten only after the ticket of the DMA that reads it has retired (tickets are FIFO per thread queue). ---------------------------------------------------------------------------------------------------------------
// Rows are independent, so the head/row assignment is the ticket thread's. Per head, the states of versions 0..n_tokens live in a ring of HTP_GDN_VER_RING VTCM buffers: version v+1 = step(version v), written out of place. Plane d (d = distance from the last token) is DMA'd straight from its version buffer, the final state (plane 0) from the last version, so the staging copy of the ASYNC path disappears.
static void gated_delta_net_f32_ver_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_gdn_context * gctx = (struct htp_gdn_context *) data;
    struct htp_ops_context * octx = gctx->octx;
    const struct htp_gdn_kernel_params * kparams = gctx->kparams;

    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;
    const struct htp_tensor * dst_cache = octx->dsts[1];

    const uint32_t S_v      = kparams->S_v;
    const uint32_t H        = kparams->H;
    const uint32_t n_tokens = kparams->n_tokens;
    const uint32_t n_seqs   = kparams->n_seqs;
    const uint32_t K        = kparams->K;
    const size_t   head_bytes = (size_t) S_v * S_v * sizeof(float);
    // unit = (head, part): part p holds state rows [p * nj, (p + 1) * nj) of its head (htp_gdn_col_parts, COLSPLIT)
    const uint32_t n_parts    = gctx->n_parts;
    const uint32_t nj         = S_v / n_parts;
    const size_t   unit_bytes = (size_t) nj * S_v * sizeof(float);
    const uint32_t n_units    = gctx->nrows * n_parts;
    const int64_t snap_d0   = (int32_t) octx->op_params[3];
    const int64_t snap_n    = (int32_t) octx->op_params[4];
    const float   scale     = kparams->scale;

    if (ith >= n_units) {
        return;
    }

    float *          dst_base   = (float *) (uintptr_t) dst->data;
    const dma_addr_t plane_base = dst_cache ? dst_cache->data : (dst->data + S_v * H * n_tokens * n_seqs * sizeof(float));
    const uint64_t   plane_stride = (uint64_t) kparams->state_size_per_snap * sizeof(float);   // bytes between planes

    const struct fastdiv_values * fd_H   = &kparams->div_H;
    const struct fastdiv_values * fd_q1  = &kparams->div_q1;
    const struct fastdiv_values * fd_k1  = &kparams->div_k1;
    const struct fastdiv_values * fd_rq3 = &kparams->div_rq3;
    const struct fastdiv_values * fd_rk3 = &kparams->div_rk3;

    uint8_t * tbase = (uint8_t *) gctx->vtcm_base + gctx->layout.bytes_per_thread * ith;
    float *   vbuf[HTP_GDN_VER_RING];
    uint32_t  free_tk[HTP_GDN_VER_RING];   // ticket of the last DMA that reads buffer r (retired before r is written)
    for (uint32_t r = 0; r < HTP_GDN_VER_RING; r++) {
        vbuf[r]    = (float *) (tbase + (size_t) r * head_bytes);
        free_tk[r] = 0;
    }

    struct htp_gdn_tickets tk = { octx->ctx->dma[ith], 0, 0 };
    uint32_t vc = 0;                       // version counter: version vc lives in buffer vc % HTP_GDN_VER_RING

    for (uint32_t u = ith; u < n_units; u += nth) {
        const uint32_t ir   = gctx->row_start + u / n_parts;
        const uint32_t part = u % n_parts;
        const uint32_t j0   = part * nj;                 // first state row of this unit inside its head

        const uint32_t iv1 = fastmodulo(ir, H, fd_H);
        const uint32_t iv3 = fastdiv(ir, fd_H);
        const uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        const uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        const uint32_t iq3 = fastdiv(iv3, fd_rq3);
        const uint32_t ik3 = fastdiv(iv3, fd_rk3);

        const uint64_t   unit_off  = ((uint64_t) iv3 * H + iv1) * head_bytes + (uint64_t) j0 * S_v * sizeof(float);   // byte offset of this unit in a plane
        const dma_addr_t plane0    = plane_base + unit_off;
        const dma_addr_t ps_in     = state->data + (uint64_t) iv3 * state->nb[3] + (uint64_t) iv1 * head_bytes +
                                     (uint64_t) j0 * S_v * sizeof(float);
        float *          attn_data = dst_base + ((uint64_t) iv3 * n_tokens * H + iv1) * S_v;

        // version 0: the input state of this unit (rows j0 .. j0 + nj - 1 of the head)
        const uint32_t b0 = vc % HTP_GDN_VER_RING;
        gdn_tk_wait(&tk, free_tk[b0]);
        gdn_tk_wait(&tk, gdn_tk_push(&tk, dma_make_data(vbuf[b0], ps_in), unit_bytes, unit_bytes, unit_bytes, 1));
        vc++;

        for (uint32_t t = 0; t < n_tokens; ++t) {
            const float * q_t = (const float *) ((const uint8_t *) (uintptr_t) q->data +
                    (uint64_t) iq3 * q->nb[3] + (uint64_t) t * q->nb[2] + (uint64_t) iq1 * q->nb[1]);
            const float * k_t = (const float *) ((const uint8_t *) (uintptr_t) k->data +
                    (uint64_t) ik3 * k->nb[3] + (uint64_t) t * k->nb[2] + (uint64_t) ik1 * k->nb[1]);
            const float * v_t = (const float *) ((const uint8_t *) (uintptr_t) v->data +
                    (uint64_t) iv3 * v->nb[3] + (uint64_t) t * v->nb[2] + (uint64_t) iv1 * v->nb[1]);
            const float * g_t = (const float *) ((const uint8_t *) (uintptr_t) g->data +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) t * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
            const float beta_val = *(const float *) ((const uint8_t *) (uintptr_t) beta->data +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) t * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);

            const uint32_t bp = (vc - 1) % HTP_GDN_VER_RING;
            const uint32_t bc = vc % HTP_GDN_VER_RING;
            gdn_tk_wait(&tk, free_tk[bc]);
            gdn_step_ver_f32(vbuf[bp], vbuf[bc], attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v, j0, nj);
            vc++;

            // plane d = distance from the last token; plane 0 (the final state) is always written
            const int64_t d        = (int64_t) n_tokens - 1 - (int64_t) t;
            const bool    in_range = (snap_n == 0) || (d >= snap_d0 && d < snap_d0 + snap_n);
            if (d == 0) {
                free_tk[bc] = gdn_tk_push(&tk, dma_make_data(plane0, vbuf[bc]), unit_bytes, unit_bytes, unit_bytes, 1);
            } else if (d < (int64_t) K && in_range) {
                const dma_addr_t snap_addr = plane_base + (uint64_t) d * plane_stride + unit_off;
                free_tk[bc] = gdn_tk_push(&tk, dma_make_data(snap_addr, vbuf[bc]), unit_bytes, unit_bytes, unit_bytes, 1);
            }

            attn_data += (uint64_t) S_v * H;
        }
    }
    dma_queue_flush(tk.q);
}

int op_gated_delta_net(struct htp_ops_context * octx) {
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    if (q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F32 || v->type != HTP_TYPE_F32 ||
        g->type != HTP_TYPE_F32 || beta->type != HTP_TYPE_F32 ||
        (state->type != HTP_TYPE_F32 && state->type != HTP_TYPE_F16) ||
        dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t S_v      = v->ne[0];
    const uint32_t H        = v->ne[1];
    const uint32_t n_tokens = v->ne[2];
    const uint32_t n_seqs   = v->ne[3];
    const uint32_t K        = octx->op_params[0];

    if (S_v == 0 || S_v > HTP_GDN_MAX_SV || H == 0 || n_tokens == 0 || n_seqs == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }
    const bool gate_fold = (octx->flags & HTP_OPFLAGS_GDN_GATE_FOLD) != 0;
    if ((g->ne[0] != 1 && g->ne[0] != S_v) || (beta->ne[0] != 1 && !gate_fold)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] == 0 || k->ne[1] == 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] == 0 || k->ne[3] == 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }
    // state holds s0 only: [S_v, S_v, H, n_seqs]
    if (state->ne[0] != S_v || state->ne[1] != S_v || state->ne[2] != H || state->ne[3] != n_seqs) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        return HTP_STATUS_NO_SUPPORT;
    }

    for (int i = 0; i < 5; i++) {
        if (htp_tensor_is_extended(octx->src[i])) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }
    if (gate_fold && (octx->src[6] == NULL || octx->src[6]->type != HTP_TYPE_F32 || octx->src[6]->ne[0] != H || H > 32)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (htp_tensor_is_extended(octx->dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (octx->dsts[1]) {
        const struct htp_tensor * dst_cache = octx->dsts[1];
        if ((dst_cache->type != HTP_TYPE_F32 && dst_cache->type != HTP_TYPE_F16) || htp_tensor_is_extended(dst_cache)) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    const struct htp_gdn_kernel_params * kparams = (const struct htp_gdn_kernel_params *) octx->kernel_params;
    struct htp_gdn_kernel_params kparams_local;
    if (!kparams || kparams->S_v == 0) {
        const uint32_t rq3 = n_seqs / q->ne[3];
        const uint32_t rk3 = n_seqs / k->ne[3];
        const uint32_t total_rows = H * n_seqs;
        uint32_t n_threads = (total_rows < octx->n_threads) ? total_rows : octx->n_threads;
        if (n_threads == 0) {
            n_threads = 1;
        }

        memset(&kparams_local, 0, sizeof(kparams_local));
        kparams_local.n_threads           = n_threads;
        kparams_local.S_v                 = S_v;
        kparams_local.H                   = H;
        kparams_local.n_tokens            = n_tokens;
        kparams_local.n_seqs              = n_seqs;
        kparams_local.K                   = K;
        kparams_local.total_rows          = total_rows;
        kparams_local.rows_per_thread     = (total_rows + n_threads - 1) / n_threads;
        // K > 1 chunked HMX needs the host's HTP_GDN_OPT_PROMPT_HMX decision (carried in kparams); this fallback has no kparams, so K == 1 only
        const bool can_use_hmx = (octx->ctx->hmx_enabled) &&
                                 (S_v % 64 == 0) &&
                                 (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                                 (g->ne[0] == 1) &&
                                 (K == 1) &&
                                 (state->type != HTP_TYPE_F16);

        struct htp_gdn_hmx_vtcm_layout hmx_layout_local;
        struct htp_gdn_vtcm_layout hvx_layout_local;
        uint32_t n_heads_batch = 1;

        if (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout_local, S_v, HTP_GDN_CHUNK_SIZE, H, total_rows, octx->ctx->vtcm_size, n_threads, true, 0, 0, &n_heads_batch)) {
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
            kparams_local.pipeline        = hmx_layout_local.pipeline ? 1 : 0;
            kparams_local.chunk_size      = HTP_GDN_CHUNK_SIZE;
            kparams_local.n_chunks        = (n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
            kparams_local.n_heads_batch   = (uint16_t) n_heads_batch;
            kparams_local.vtcm_size       = (uint32_t) hmx_layout_local.total_bytes;
            kparams_local.state_aligned   = (uint32_t) hmx_layout_local.state_f32_bytes;
            kparams_local.vtcm_per_thread = (uint32_t) (hmx_layout_local.total_bytes / (n_threads > 0 ? n_threads : 1));
        } else {
            htp_gdn_vtcm_layout_build(&hvx_layout_local, S_v, n_threads);
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
            kparams_local.pipeline        = 0;
            kparams_local.n_heads_batch   = 1;
            kparams_local.state_aligned   = (uint32_t) hvx_layout_local.state_aligned;
            kparams_local.vtcm_per_thread = (uint32_t) hvx_layout_local.bytes_per_thread;
            kparams_local.vtcm_size       = (uint32_t) hvx_layout_local.total_bytes;
        }
        kparams_local.kda                 = (g->ne[0] == S_v) ? 1 : 0;
        kparams_local.scale               = 1.0f / sqrtf((float) S_v);
        kparams_local.state_seq_stride    = (uint32_t) (state->nb[3] / sizeof(float));
        kparams_local.state_size_per_snap = S_v * S_v * H * n_seqs;

        kparams_local.div_H         = init_fastdiv_values(H);
        kparams_local.div_q1        = init_fastdiv_values(q->ne[1]);
        kparams_local.div_k1        = init_fastdiv_values(k->ne[1]);
        kparams_local.div_rq3       = init_fastdiv_values(rq3);
        kparams_local.div_rk3       = init_fastdiv_values(rk3);
        kparams_local.div_n_threads = init_fastdiv_values(n_threads);

        kparams = &kparams_local;
    }

    const uint32_t total_rows = kparams->total_rows;
    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                               ((dst->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows,
            can_split ? 1 : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.count,
            &octx->ctx->mdev.count_div
        );
        row_start = range.start;
        nrows     = range.count;
    } else if (octx->op_params[1] != 0) {
        row_start = octx->op_params[1];
        nrows     = octx->op_params[2];
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    // f16 state (LLAMA_GDN_STATE_F16) and non-default snapshot modes run on the ticket path; f16 state never uses HMX
    const bool in_f16     = state->type == HTP_TYPE_F16;
    const bool out_f16    = octx->dsts[1] != NULL && octx->dsts[1]->type == HTP_TYPE_F16;
    const bool use_ticket = kparams->snap_mode != HTP_GDN_SNAP_SYNC || in_f16 || out_f16;
    // D2Snap: VERSION (GGML_HEXAGON_GDN_SNAP=version) on decode-sized batches with f32 state and cache
    const bool use_ver = kparams->snap_mode == HTP_GDN_SNAP_VERSION && !in_f16 && !out_f16 && !kparams->kda &&
                         kparams->n_tokens < HTP_GDN_VER_MAX_TOKENS && (S_v % 32) == 0;
    if (in_f16 && kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) {
        return HTP_STATUS_NO_SUPPORT;
    }
    // the chunked HMX kernel writes the final state (plane 0) only: K > 1 may run it only on the host's decision for the
    // typed final-state range (opt bit, op_params[3..4] = 0, 1); otherwise planes 1..K-1 would be left stale
    if (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED && K > 1 &&
            !((kparams->opt & HTP_GDN_OPT_PROMPT_HMX) && (int32_t) octx->op_params[3] == 0 && (int32_t) octx->op_params[4] == 1)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (gate_fold && (kparams->kernel_type != HTP_GDN_KERNEL_HMX_CHUNKED || !(kparams->opt & HTP_GDN_OPT_QKSHARE))) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }
    if (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) {
        return gated_delta_net_f32_hmx_chunked(octx, kparams, row_start, nrows);
    }

    // scalar pp and ticket threads split each head into n_parts units (htp_gdn_col_parts); the tg thread stays unsplit
    const bool     scalar_split = use_ticket || n_tokens > 1;
    const uint32_t n_parts      = scalar_split ? htp_gdn_col_parts(S_v, nrows, kparams->n_threads, htp_gdn_col_split_log2(kparams->fix)) : 1;
    const uint32_t n_units      = nrows * n_parts;
    const uint32_t n_threads    = (n_units < kparams->n_threads) ? n_units : kparams->n_threads;

    struct htp_gdn_context gctx;
    gctx.octx      = octx;
    gctx.kparams   = kparams;
    gctx.row_start = row_start;
    gctx.nrows     = nrows;
    gctx.n_parts   = n_parts;
    gctx.vtcm_base = octx->ctx->vtcm_base;

    if (use_ver) {
        gctx.layout.state_aligned    = (size_t) S_v * S_v * sizeof(float);
        gctx.layout.bytes_per_thread = (size_t) HTP_GDN_VER_RING * gctx.layout.state_aligned;
        gctx.layout.total_bytes      = gctx.layout.bytes_per_thread * n_threads;
    } else if (use_ticket) {
        htp_gdn_io_layout_build(&gctx.io, S_v, in_f16, out_f16,
                                kparams->snap_mode == HTP_GDN_SNAP_ASYNC || kparams->snap_mode == HTP_GDN_SNAP_VERSION);
        gctx.layout.state_aligned    = gctx.io.state_aligned;
        gctx.layout.bytes_per_thread = gctx.io.bytes_per_thread;
        gctx.layout.total_bytes      = gctx.io.bytes_per_thread * n_threads;
    } else {
        htp_gdn_vtcm_layout_select(&gctx.layout, S_v, n_threads, kparams->opt);
    }

    if (gctx.layout.total_bytes > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    FARF(HIGH, "gated-delta-net-f32: q(%ux%ux%ux%u) k(%ux%ux%ux%u) v(%ux%ux%ux%u) state(%ux%ux%ux%u) -> (%ux%ux%ux%u) : "
         "vtcm-size %zu n_threads %u\n",
         q->ne[0], q->ne[1], q->ne[2], q->ne[3],
         k->ne[0], k->ne[1], k->ne[2], k->ne[3],
         v->ne[0], v->ne[1], v->ne[2], v->ne[3],
         state->ne[0], state->ne[1], state->ne[2], state->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         gctx.layout.total_bytes, n_threads);

    if (use_ver) {
        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_ver_thread, &gctx, n_threads);
    } else if (use_ticket) {
        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_ticket_thread, &gctx, n_threads);
    } else if (n_tokens == 1) {
        work_queue_func_t tg_fn = (kparams->opt & HTP_GDN_OPT_UP_DMA) ? gated_delta_net_f32_tg_dma_thread : gated_delta_net_f32_tg_thread;
        work_queue_run(octx->ctx->work_queue, tg_fn, &gctx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_pp_thread, &gctx, n_threads);
    }

    return HTP_STATUS_OK;
}
