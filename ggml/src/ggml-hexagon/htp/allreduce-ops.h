#ifndef ALLREDUCE_OPS_H
#define ALLREDUCE_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define HTP_ALLREDUCE_MAX_RANKS 4

#ifdef __cplusplus
extern "C" {
#endif

enum htp_allreduce_kernel_type {
    HTP_ALLREDUCE_KERNEL_UNSUPPORTED = 0,
    HTP_ALLREDUCE_KERNEL_DMA_1D,
    HTP_ALLREDUCE_KERNEL_DMA_2D,
    HTP_ALLREDUCE_KERNEL_RMS_AG,      // MC05: row-chunked RMS_NORM_MUL with all-gather to every norm dst (gap-fused, switch 3)
    HTP_ALLREDUCE_KERNEL_ADD_RMS,       // AE-03: owner-row ALLREDUCE+ADD and RMS_NORM_MUL of the same rows in one op, fp32 norm rows (GGML_HEXAGON_AR_RMS_FUSE)
    HTP_ALLREDUCE_KERNEL_ADD_RMS_TILE,  // AE-03: same, fp16 HMX tile norm outputs (GATHER_ACT_TILE)
};

// VrAr step A (GGML_HEXAGON_VR_AR bit 4), ADD_RMS only: the owner rows of the residual stream stay in VTCM at resid_off for a prefill batch.
#define HTP_ALLREDUCE_RES_IN    (1 << 0)  // resid_in rows are the resident block (no load DMA)
#define HTP_ALLREDUCE_RES_OUT   (1 << 1)  // resid_out rows go to the resident block (no store DMA)
#define HTP_ALLREDUCE_RES_PIN   (1 << 2)  // the VTCM budget becomes resid_off until RES_UNPIN
#define HTP_ALLREDUCE_RES_FLUSH (1 << 3)  // the resident rows are also written to resid_out (DDR)
#define HTP_ALLREDUCE_RES_UNPIN (1 << 4)  // the VTCM budget is restored after this op
#define HTP_ALLREDUCE_RES_ANY   (HTP_ALLREDUCE_RES_IN | HTP_ALLREDUCE_RES_OUT | HTP_ALLREDUCE_RES_PIN | HTP_ALLREDUCE_RES_FLUSH | HTP_ALLREDUCE_RES_UNPIN)
#define HTP_ALLREDUCE_PARTIAL_F16 (1 << 5)  // fp16 partial rows (GGML_HEXAGON_AR_PARTIAL_F16): each partial is compact fp16 rows (row stride = ne0 halves); loads widen them in VTCM

static inline size_t htp_allreduce_vtcm_buffer_count(
    uint32_t n_ranks,
    uint32_t n_threads,
    bool has_add,
    bool is_row_bcast
) {
    return (size_t) (n_ranks + 1) * n_threads + (has_add ? (is_row_bcast ? 1 : n_threads) : 0);
}

struct htp_allreduce_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_threads;
    int32_t block_elems;          // 1D: block_elems, 2D: block_rows
    int32_t elems_per_thread;     // 1D: nelem_per_thread, 2D: nrows_per_thread
    int32_t vtcm_size_per_thread;
    int32_t vtcm_size;
    int32_t kernel_type;
    int32_t ne0;
    int32_t ne1;
    int32_t row_size_aligned;
    int32_t rank_elem_start;
    int32_t rank_nelem;
    int32_t n_dsts;
    int32_t is_row_bcast;
    int32_t resid_mode;           // VrAr: HTP_ALLREDUCE_RES_* bits; 0 = plain path
    int32_t resid_off;            // VrAr: VTCM offset of the resident residual rows (the VTCM budget while pinned)
};

#ifdef __cplusplus
}
#endif

#endif /* ALLREDUCE_OPS_H */
