#ifndef HTP_SSM_CONV_H
#define HTP_SSM_CONV_H

#include <stdint.h>

#include "hex-fastdiv.h"
#include "htp-ops.h"

struct htp_ssm_conv_kernel_params {
    uint32_t n_threads;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_per_thread;
    uint32_t d_inner_tile;

    uint32_t src0_row_size_aligned;
    uint32_t src1_row_size_aligned;
    uint32_t dst_row_size_aligned;

    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;

    uint32_t vtcm_src0_size;
    uint32_t vtcm_src1_size;
    uint32_t vtcm_dst_size;
    uint32_t vtcm_size;

    // CONCAT+SSM_CONV(+SILU) fusion (try_fuse_concat_ssm_conv, ggml-hexagon.cpp): n_state_rows is conv_kernel_size - 1, i.e.
    // src0->ne[0] when fused_concat is set. fused_concat==0 means the normal unfused dispatch (src0 = full conv_input, computed as before).
    uint32_t fused_concat;
    uint32_t n_state_rows;
    // pipe: pipelined fused prefill (GGML_HEXAGON_SSM_CONV_PIPE, host-set, default 0); pipe_chunk = tokens per step.
    uint32_t pipe;
    uint32_t pipe_chunk;

    struct fastdiv_values div_n_threads;
};

// HTP_OP_SSM_CONV_CHAIN: src conv state, qkv (transposed view), conv1d; dst silu(conv) {d_inner, n_t}, new conv state.
// With qk_head_dim != 0, channels [0, n_q + n_k) get the per-head RMS_NORM + SCALE and go to dst[2] q and dst[3] k.
#define HTP_SSM_CONV_CHAIN_MAX_BALANCED_THREADS 9

struct htp_ssm_conv_chain_kernel_params {
    uint32_t n_threads;
    uint32_t d_conv;
    uint32_t n_state;             // d_conv - 1
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_per_thread;  // multiple of 32
    uint32_t d_inner_tile;        // multiple of 32

    // byte strides between sequences
    uint32_t st_in_seq_stride;
    uint32_t qkv_seq_stride;
    uint32_t dst_seq_stride;
    uint32_t st_out_seq_stride;

    uint32_t n_buf;                   // tile buffers per thread
    uint32_t vtcm_xt_bytes;           // {n_state + block rows, d_inner_tile} conv input tile
    uint32_t vtcm_state_stage_bytes;  // conv state of the thread's channels, all sequences
    uint32_t vtcm_buf_stride;         // xt + state_out
    uint32_t vtcm_dst_tile_bytes;     // {block rows, d_inner_tile}

    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;

    uint32_t vtcm_size;           // n_threads x (src0 + src1 + dst per thread)

    // balanced != 0: thread i > 0 starts at channel 32 * ch_start[i - 1] instead of i * d_inner_per_thread;
    // no range is wider than d_inner_per_thread and, with the q/k epilogue, every range is whole heads
    uint32_t balanced;
    uint8_t  ch_start[HTP_SSM_CONV_CHAIN_MAX_BALANCED_THREADS - 1];

    uint32_t n_tb;         // token rows per block, 0 = all n_t tokens in one block
    uint32_t qk_head_dim;  // 0 = no q/k l2-norm epilogue; else a multiple of 32 that divides the tile
    float    qk_eps;       // RMS_NORM eps
    float    qk_scale;     // SCALE scale and bias
    float    qk_bias;
    uint32_t silu_variant; // C2Elt: 0 = original SiLU tile; 1 = four vectors per step; 2 = also the two-iteration reciprocal
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
static_assert(sizeof(struct htp_ssm_conv_chain_kernel_params) <= 128, "htp_ssm_conv_chain_kernel_params is too large");
#else
_Static_assert(sizeof(struct htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
_Static_assert(sizeof(struct htp_ssm_conv_chain_kernel_params) <= 128, "htp_ssm_conv_chain_kernel_params is too large");
#endif

#endif // HTP_SSM_CONV_H
