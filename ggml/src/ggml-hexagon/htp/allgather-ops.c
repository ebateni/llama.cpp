#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <stdatomic.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "htp-tensor.h"
#include "dma-queue.h"
#include "hex-profile.h"
#include "allgather-ops.h"
#include "htp-fence.h"

struct htp_allgather_context {
    struct htp_ops_context * octx;
    uint32_t n_ranks;
    uint32_t row_bytes;
    uint32_t total_rows;
    uint32_t row_offset[HTP_ALLGATHER_MAX_RANKS];
    uint32_t row_count[HTP_ALLGATHER_MAX_RANKS];
};

// Each thread owns a contiguous slice of the full output rows and, for every
// output row it owns, DMAs that row from the rank that produced it (that rank's
// compacted source buffer) into this rank's full destination buffer.
// Pure data movement — no reduction.
static void allgather_thread_dma(unsigned int nth, unsigned int ith, void * data) {
    struct htp_allgather_context * actx = (struct htp_allgather_context *) data;
    struct htp_ops_context * octx = actx->octx;

    const uint32_t n_ranks    = actx->n_ranks;
    const uint32_t row_bytes  = actx->row_bytes;
    const uint32_t total_rows = actx->total_rows;

    // Partition the full output rows across threads.
    const uint32_t rows_per_thread = (total_rows + nth - 1) / nth;
    const uint32_t gr0 = rows_per_thread * ith;
    const uint32_t gr1 = MIN(gr0 + rows_per_thread, total_rows);
    if (gr0 >= gr1) {
        return;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    const struct htp_tensor * dst = octx->dst;

    // Walk each rank's block; copy the intersection with this thread's row range.
    for (uint32_t s = 0; s < n_ranks; s++) {
        const uint32_t off = actx->row_offset[s];
        const uint32_t cnt = actx->row_count[s];
        if (cnt == 0) {
            continue;
        }
        const uint32_t blk0 = off;
        const uint32_t blk1 = off + cnt;

        const uint32_t lo = MAX(gr0, blk0);
        const uint32_t hi = MIN(gr1, blk1);
        if (lo >= hi) {
            continue;
        }

        const struct htp_tensor * src = octx->src[s];           // rank s compacted slice [row_bytes x cnt]
        const uint32_t nrows = hi - lo;

        // destination rows [lo, hi) in the full output; source rows [lo-off, ...) in the compacted slice
        const dma_addr_t d_ddr = (dma_addr_t) (dst->data + (uint64_t) lo         * dst->nb[1]);
        const dma_addr_t s_ddr = (dma_addr_t) (src->data + (uint64_t) (lo - off) * src->nb[1]);

        dma_queue_push(dma_q, dma_make_data(d_ddr, s_ddr), dst->nb[1], src->nb[1], row_bytes, nrows);
    }

    // Drain this thread's queued copies.
    for (uint32_t s = 0; s < n_ranks; s++) {
        const uint32_t off = actx->row_offset[s];
        const uint32_t cnt = actx->row_count[s];
        if (cnt == 0) {
            continue;
        }
        const uint32_t lo = MAX(gr0, off);
        const uint32_t hi = MIN(gr1, off + cnt);
        if (lo >= hi) {
            continue;
        }
        dma_queue_pop(dma_q);
    }
    dma_queue_flush(dma_q);
}

static int validate_allgather(
    struct htp_ops_context * octx,
    const struct htp_allgather_kernel_params * kparams,
    uint32_t n_ranks
) {
    if (kparams->n_threads == 0 || (uint32_t) kparams->n_threads > octx->ctx->n_threads) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    octx->n_threads = (uint32_t) kparams->n_threads;

    if (kparams->row_bytes <= 0) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    return HTP_STATUS_OK;
}

int op_allgather(struct htp_ops_context * octx) {
    // Only the lead sub-device of an mdev group participates (mirrors op_allreduce).
    if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx > 0) {
        return HTP_STATUS_OK;
    }

    const struct htp_allgather_kernel_params * kparams = (const struct htp_allgather_kernel_params *) octx->kernel_params;
    const struct htp_tensor * dst = octx->dst;

    const uint32_t rank    = (uint32_t) kparams->rank;
    const uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    if (n_ranks < 2 || n_ranks > HTP_ALLGATHER_MAX_RANKS || rank >= n_ranks) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t fence_seq_entry = (uint32_t) octx->op_params[0];
    const uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    const struct htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->data;

    const int status = validate_allgather(octx, kparams, n_ranks);
    if (status != HTP_STATUS_OK) {
        if (status == HTP_STATUS_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allgather unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, status);
        return status;
    }

    // 1. Entry barrier: every rank must have finished producing its own slice
    //    before any rank reads peers' slices.
    struct htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    htp_fence_write(my_fence, fence_seq_entry, octx->status);

    for (uint32_t j = 0; j < n_ranks; j++) {
        if (j == rank) continue;
        const struct htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        while (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            if ((int32_t)(peer_seq - fence_seq_entry) >= 0) {
                if (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allgather entry peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                    return peer_status;
                }
                break;
            }
            if (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allgather entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_entry, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);
                return HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm volatile ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_entry);

    // 2. Fan-in copy of every rank's slice into this rank's full output buffer.
    uint32_t total_rows = 0;
    struct htp_allgather_context actx;
    actx.octx      = octx;
    actx.n_ranks   = n_ranks;
    actx.row_bytes = (uint32_t) kparams->row_bytes;
    for (uint32_t s = 0; s < n_ranks; s++) {
        actx.row_offset[s] = (uint32_t) kparams->row_offset[s];
        actx.row_count[s]  = (uint32_t) kparams->row_count[s];
        total_rows += actx.row_count[s];
    }
    actx.total_rows = total_rows;

    if (total_rows > 0) {
        work_queue_run(octx->ctx->work_queue, allgather_thread_dma, &actx, octx->n_threads);
    }

    // 3. Exit barrier: ensure all fan-in writes have landed before any rank
    //    proceeds to consume the gathered (now mirrored) tensor.
    htp_trace_event_start(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
    asm volatile ("syncht" : : : "memory");

    htp_fence_write(my_fence, fence_seq_exit, octx->status);

    for (uint32_t j = 0; j < n_ranks; j++) {
        if (j == rank) continue;
        const struct htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->data;
        uint64_t spins = 0;
        while (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            if ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                if (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: allgather exit peer %u failed with status %u\n", j, peer_status);
                    htp_fence_write(my_fence, fence_seq_exit, peer_status);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    return peer_status;
                }
                break;
            }
            if (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allgather exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_STATUS_INTERNAL_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                return HTP_STATUS_INTERNAL_ERR;
            }
            hex_pause();
        }
    }
    asm volatile ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    return octx->status;
}
