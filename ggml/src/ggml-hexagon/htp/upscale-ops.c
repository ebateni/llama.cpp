#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "hex-common.h"
#include "htp-ops.h"
#include "htp-tensor.h"

#define HTP_SCALE_MODE_BILINEAR      1
#define HTP_SCALE_FLAG_ALIGN_CORNERS (1u << 8)
#define HTP_SCALE_FLAG_ANTIALIAS     (1u << 9)

struct htp_upscale_context {
    struct htp_ops_context * octx;

    const struct htp_tensor * src;
    const struct htp_tensor * dst;

    uint32_t mode;
    uint32_t out_w;
    uint32_t out_h;
    uint32_t n_channels;

    uint32_t in_w;
    uint32_t in_h;

    float sf_x;
    float sf_y;
    float pixel_offset;

    uint32_t rows_per_thread;
    uint32_t row_start;
    uint32_t total_rows;
};

static void upscale_job_per_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_upscale_context * uctx = (const struct htp_upscale_context *) data;

    const struct htp_tensor * src = uctx->src;
    const struct htp_tensor * dst = uctx->dst;

    const uint32_t row_start = uctx->row_start + uctx->rows_per_thread * ith;
    const uint32_t row_end   = MIN(row_start + uctx->rows_per_thread, uctx->row_start + uctx->total_rows);

    const uint8_t * src_data = (const uint8_t *) htp_tensor_data(src);
    uint8_t *       dst_data = (uint8_t *)       htp_tensor_data(dst);

    const uint32_t nb00 = src->nb[0];
    const uint32_t nb01 = src->nb[1];
    const uint32_t nb02 = src->nb[2];
    const uint32_t nb0  = dst->nb[0];
    const uint32_t nb1  = dst->nb[1];
    const uint32_t nb2  = dst->nb[2];

    struct htp_thread_trace * tr = &uctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    for (uint32_t row = row_start; row < row_end; row++) {
        const uint32_t y  = row % uctx->out_h;
        const uint32_t ch = row / uctx->out_h;

        float fy = ((float) y + uctx->pixel_offset) / uctx->sf_y;
        fy = fmaxf(0.0f, fminf(fy, (float) (uctx->in_h - 1)));

        const uint32_t y0 = (uint32_t) floorf(fy);
        const uint32_t y1 = MIN(y0 + 1, uctx->in_h - 1);
        const float    wy = fy - (float) y0;

        const uint8_t * src_row0 = src_data + (size_t) y0 * nb01 + (size_t) ch * nb02;
        const uint8_t * src_row1 = src_data + (size_t) y1 * nb01 + (size_t) ch * nb02;
        uint8_t *       dst_row  = dst_data + (size_t) y  * nb1  + (size_t) ch * nb2;

        for (uint32_t x = 0; x < uctx->out_w; x++) {
            float fx = ((float) x + uctx->pixel_offset) / uctx->sf_x;
            fx = fmaxf(0.0f, fminf(fx, (float) (uctx->in_w - 1)));

            const uint32_t x0 = (uint32_t) floorf(fx);
            const uint32_t x1 = MIN(x0 + 1, uctx->in_w - 1);
            const float    wx = fx - (float) x0;

            const float * p00 = (const float *) (src_row0 + (size_t) x0 * nb00);
            const float * p10 = (const float *) (src_row0 + (size_t) x1 * nb00);
            const float * p01 = (const float *) (src_row1 + (size_t) x0 * nb00);
            const float * p11 = (const float *) (src_row1 + (size_t) x1 * nb00);

            const float v0 = *p00 + (*p10 - *p00) * wx;
            const float v1 = *p01 + (*p11 - *p01) * wx;

            float * out = (float *) (dst_row + (size_t) x * nb0);
            *out = v0 + (v1 - v0) * wy;
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(HIGH, "upscale %d/%d: (%ux%ux%u) -> (%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2],
         dst->ne[0], dst->ne[1], dst->ne[2], row_start, row_end);
}

int op_upscale(struct htp_ops_context * octx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    if (!src || !dst || src->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t mode = (uint32_t) octx->op_params[0];
    if ((mode & 0xFF) != HTP_SCALE_MODE_BILINEAR || (mode & HTP_SCALE_FLAG_ANTIALIAS)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (src->ne[3] != 1 || dst->ne[3] != 1 || src->ne[2] != dst->ne[2]) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t out_w = dst->ne[0];
    const uint32_t out_h = dst->ne[1];
    const uint32_t in_w  = src->ne[0];
    const uint32_t in_h  = src->ne[1];

    if (out_w == 0 || out_h == 0 || in_w == 0 || in_h == 0) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const bool align_corners = (mode & HTP_SCALE_FLAG_ALIGN_CORNERS) != 0;

    float sf_x;
    float sf_y;

    if (align_corners) {
        sf_x = (out_w > 1 && in_w > 1) ? (float) (out_w - 1) / (float) (in_w - 1) : (float) out_w / (float) in_w;
        sf_y = (out_h > 1 && in_h > 1) ? (float) (out_h - 1) / (float) (in_h - 1) : (float) out_h / (float) in_h;
    } else {
        sf_x = (float) out_w / (float) in_w;
        sf_y = (float) out_h / (float) in_h;
    }

    const uint32_t total_rows = out_h * dst->ne[2];

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, 4, out_w * 4, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = octx->n_threads;

    struct htp_upscale_context uctx = {
        .octx             = octx,
        .src              = src,
        .dst              = dst,
        .mode             = mode,
        .out_w            = out_w,
        .out_h            = out_h,
        .n_channels       = dst->ne[2],
        .in_w             = in_w,
        .in_h             = in_h,
        .sf_x             = sf_x,
        .sf_y             = sf_y,
        .pixel_offset     = align_corners ? 0.0f : 0.5f,
        .rows_per_thread  = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .row_start        = row_start,
        .total_rows       = nrows,
    };

    work_queue_run(octx->ctx->work_queue, upscale_job_per_thread, &uctx, n_threads);

    return HTP_STATUS_OK;
}
