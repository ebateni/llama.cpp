#ifndef ALLGATHER_OPS_H
#define ALLGATHER_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define HTP_ALLGATHER_MAX_RANKS 4

#ifdef __cplusplus
extern "C" {
#endif

// All-gather concatenates each rank's contiguous row-block (a slice along the
// row dimension = ne1*ne2*ne3) into the full output tensor on every rank.
// Modeled on the allreduce collective but with NO reduction: it is a pure
// DMA fan-in of every peer's slice into this rank's full-size output buffer.
//
// Buffer layout (see ggml_backend_meta_buffer_init_tensor_impl): under an
// AXIS_1 (token) split each rank's SOURCE tensor is the compacted
// [ne0, rows_r] slice; the MIRRORED destination is the full [ne0, total_rows]
// tensor. row_offset[r] is the starting row of rank r's slice in the output.

enum htp_allgather_kernel_type {
    HTP_ALLGATHER_KERNEL_UNSUPPORTED = 0,
    HTP_ALLGATHER_KERNEL_DMA = 1,
};

struct htp_allgather_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_threads;
    int32_t row_bytes;                              // bytes per row (ne0 * elem_size), == contiguous slice stride
    int32_t kernel_type;
    int32_t row_offset[HTP_ALLGATHER_MAX_RANKS];    // starting row of each rank's slice in the full output
    int32_t row_count[HTP_ALLGATHER_MAX_RANKS];     // number of rows contributed by each rank
};

#ifdef __cplusplus
}
#endif

#endif /* ALLGATHER_OPS_H */
