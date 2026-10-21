#ifndef HTP_GATED_DELTA_NET_OPS_H
#define HTP_GATED_DELTA_NET_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "hex-fastdiv.h"
#include "hex-common.h"
#include "htp-vtcm.h"

#define HTP_GDN_MAX_SV     128
#define HTP_GDN_CHUNK_SIZE 64
#define HTP_GDN_MIN_TOKENS 8

#define HTP_GDN_MAX_BATCH  16 // heads per batch with HTP_GDN_OPT_WIDE (HVX jobs run head passes of up to n_threads)

// kparams->opt bits (host: GGML_HEXAGON_GDN_HMX_BATCH / _WIDE / _QKSHARE / _STATE_MERGE / _PROMPT_HMX; all default off)
#define HTP_GDN_OPT_HMX_BATCH   (1u << 0) // one HMX descriptor per GEMM kind per chunk, covering all heads of the batch
#define HTP_GDN_OPT_WIDE        (1u << 1) // batches of up to HTP_GDN_MAX_BATCH heads
#define HTP_GDN_OPT_QKSHARE     (1u << 2) // V heads that share a Q/K head are batched together; q/k DMA once per Q/K head
#define HTP_GDN_OPT_STATE_MERGE (1u << 3) // chunk state update and f16 repack in one HVX pass (state stays f32)
#define HTP_GDN_OPT_PROMPT_HMX  (1u << 4) // K > 1 chunked HMX on the typed final-state range (op_params[3..4] = 0, 1); the host sets it on that path only
#define HTP_GDN_OPT_UP_DMA        (1u << 5) // GGML_HEXAGON_GDN_UP bit 0: DMA for all tensors on the SYNC token-gen path (upstream 9b47b8ad4, c6c532daa)
#define HTP_GDN_OPT_UP_LEGACY_EXP (1u << 6) // GGML_HEXAGON_GDN_UP bit 1: HMX chunk decays with f32 exp (pre-268f3874d arithmetic)
#define HTP_GDN_OPT_UP_LEGACY_ACC (1u << 7) // GGML_HEXAGON_GDN_UP bit 2: HMX phase 4 without the zero-decay mask (pre-7d4b7c4d9 arithmetic)
#define HTP_GDN_OPT_UP_ALL        (HTP_GDN_OPT_UP_DMA | HTP_GDN_OPT_UP_LEGACY_EXP | HTP_GDN_OPT_UP_LEGACY_ACC) // GGML_HEXAGON_GDN_UP bits: no HMX layout effect

// kparams->fix bits (host: GGML_HEXAGON_GDN_FIX=bitmask; default 0). Chunked HMX path only, and only for n_tokens >= 64.
#define HTP_GDN_FIX_PASS_SLICE (1u << 0) // a pass of at most n_threads/2 heads runs each head as two jobs (row halves)
#define HTP_GDN_FIX_PROLOGUE   (1u << 1) // chunk-0 input DMA is pushed before the state-in wait and init_state
#define HTP_GDN_FIX_HEAD_SKEW  (1u << 2) // per-head VTCM pad (kparams->skew_units x 128 B): the heads of a batch start on different VTCM banks
#define HTP_GDN_FIX_HMX_SPLIT  (1u << 3) // chunked HMX, batched GEMMs, two passes: DL and OA are issued per pass so pass 1 runs under the next pass's HVX phase
#define HTP_GDN_FIX_COLSPLIT_SHIFT 4 // bits 4-5 of kparams->fix: GGML_HEXAGON_GDN_COLSPLIT=N (0..3); not a FIX flag, packed in the byte
static inline uint32_t htp_gdn_col_split_log2(uint8_t fix) { return ((uint32_t) fix >> HTP_GDN_FIX_COLSPLIT_SHIFT) & 3u; }

#ifndef HMX_FP16_TILE_SIZE
#define HMX_FP16_TILE_SIZE 2048
#endif

enum htp_gdn_kernel_type {
    HTP_GDN_KERNEL_HVX_RECURRENT = 0,
    HTP_GDN_KERNEL_HMX_CHUNKED   = 1,
};

// K > 1 snapshot-plane write policy (kparams->snap_mode, set from GGML_HEXAGON_GDN_SNAP on the host)
enum htp_gdn_snap_mode {
    HTP_GDN_SNAP_SYNC        = 0, // default: synchronous HVX stores of each plane (legacy path, unchanged)
    HTP_GDN_SNAP_SKIP        = 1, // debug: no snapshot planes are written (slot 0, the final state, still is)
    HTP_GDN_SNAP_ASYNC       = 2, // VC-04: planes are staged in a per-thread VTCM ring and pushed with DMA
    HTP_GDN_SNAP_SYNC_TICKET = 3, // control: ticket path with synchronous plane writes (isolates the code path)
    HTP_GDN_SNAP_VERSION     = 4, // D2Snap: out-of-place state versions in a per-thread VTCM ring; planes are DMA'd from the version
                                  // (no staging copy); two-pass step without the intermediate store. Used for n_tokens < HTP_GDN_VER_MAX_TOKENS
                                  // with f32 state and cache and S_v % 32 == 0; otherwise the ASYNC ticket path
};

// VTCM staging slots per thread for HTP_GDN_SNAP_ASYNC
#define HTP_GDN_SNAP_RING 4

// HTP_GDN_SNAP_VERSION: state versions per thread (one version = one head's state, 64 KiB at S_v = 128)
#define HTP_GDN_VER_RING       4
#define HTP_GDN_VER_MAX_TOKENS 64

struct htp_gdn_kernel_params {
    uint8_t  kernel_type;
    uint8_t  pipeline;
    uint16_t chunk_size;
    uint16_t n_chunks;
    uint16_t n_heads_batch;
    uint8_t  snap_mode;      // enum htp_gdn_snap_mode
    uint8_t  opt;           // HTP_GDN_OPT_* bits (host GGML_HEXAGON_GDN_* switches); takes the former reserved byte
    uint8_t  fix;           // HTP_GDN_FIX_* bits (host GGML_HEXAGON_GDN_FIX); fits the padding before n_threads
    uint8_t  skew_units;    // HTP_GDN_FIX_HEAD_SKEW pad per head in 128-byte units (host GGML_HEXAGON_GDN_SKEW); the former padding byte

    uint32_t n_threads;
    uint32_t S_v;
    uint32_t H;
    uint32_t n_tokens;
    uint32_t n_seqs;
    uint32_t K;

    uint32_t total_rows;
    uint32_t row_start;
    uint32_t nrows;
    uint32_t rows_per_thread;

    uint32_t kda;
    uint32_t state_aligned;
    uint32_t vtcm_per_thread;
    uint32_t vtcm_size;
    uint32_t state_seq_stride;
    uint32_t state_size_per_snap;

    float    scale;

    struct fastdiv_values div_H;
    struct fastdiv_values div_q1;
    struct fastdiv_values div_k1;
    struct fastdiv_values div_rq3;
    struct fastdiv_values div_rk3;
    struct fastdiv_values div_n_threads;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#endif

#define HTP_GDN_HVX_N_BUFS 2 // ring depth of the token-gen buffers (HTP_GDN_OPT_UP_DMA)

struct htp_gdn_vtcm_layout {
    size_t state_aligned;
    size_t vec_aligned;     // HTP_GDN_OPT_UP_DMA layout only
    size_t off_state;       // HTP_GDN_OPT_UP_DMA layout only: offsets inside the per-thread block
    size_t off_q;
    size_t off_k;
    size_t off_v;
    size_t off_g;
    size_t off_b;
    size_t off_out;
    size_t bytes_per_thread;
    size_t total_bytes;
};

static inline void htp_gdn_vtcm_layout_build(
    struct htp_gdn_vtcm_layout * layout,
    uint32_t S_v,
    uint32_t n_threads
) {
    size_t state_bytes = (size_t) S_v * S_v * sizeof(float);
    layout->state_aligned = hex_round_up(state_bytes, 128);
    layout->bytes_per_thread = 2 * layout->state_aligned;
    layout->total_bytes = layout->bytes_per_thread * n_threads;
}

// HTP_GDN_OPT_UP_DMA layout (upstream 9b47b8ad4 / c6c532daa): HTP_GDN_HVX_N_BUFS buffers each of state, q, k, v, g, beta
// and out per thread, all filled by DMA; the thread block is rounded to 128 bytes.
static inline void htp_gdn_vtcm_layout_build_dma(
    struct htp_gdn_vtcm_layout * layout,
    uint32_t S_v,
    uint32_t n_threads
) {
    const size_t state_bytes = (size_t) S_v * S_v * sizeof(float);
    const size_t vec_bytes   = (size_t) S_v * sizeof(float);

    layout->state_aligned = hex_round_up(state_bytes, 128);
    layout->vec_aligned   = hex_round_up(vec_bytes, 128);

    size_t off = 0;
    layout->off_state = off; off += HTP_GDN_HVX_N_BUFS * layout->state_aligned;
    layout->off_q     = off; off += HTP_GDN_HVX_N_BUFS * layout->vec_aligned;
    layout->off_k     = off; off += HTP_GDN_HVX_N_BUFS * layout->vec_aligned;
    layout->off_v     = off; off += HTP_GDN_HVX_N_BUFS * layout->vec_aligned;
    layout->off_g     = off; off += HTP_GDN_HVX_N_BUFS * layout->vec_aligned;
    layout->off_b     = off; off += HTP_GDN_HVX_N_BUFS * 128;
    layout->off_out   = off; off += HTP_GDN_HVX_N_BUFS * layout->vec_aligned;

    layout->bytes_per_thread = hex_round_up(off, 128);
    layout->total_bytes      = layout->bytes_per_thread * n_threads;
}

// VTCM layout of the non-ticket, non-HMX path: the base layout unless HTP_GDN_OPT_UP_DMA is set.
// Host and device both call this with kparams->opt, so they always agree.
static inline void htp_gdn_vtcm_layout_select(
    struct htp_gdn_vtcm_layout * layout,
    uint32_t S_v,
    uint32_t n_threads,
    uint8_t  opt
) {
    if (opt & HTP_GDN_OPT_UP_DMA) {
        htp_gdn_vtcm_layout_build_dma(layout, S_v, n_threads);
    } else {
        htp_gdn_vtcm_layout_build(layout, S_v, n_threads);
    }
}

// VTCM block of one thread on the ticket path (f16 state storage and/or async snapshot ring), in this order: [s_work0][s_work1][io_in0][io_in1][io_out0][io_out1][ring x HTP_GDN_SNAP_RING] io_in exists iff the state is f16, io_out iff the cache state is f16, the ring iff the snap mode is ASYNC.
struct htp_gdn_io_layout {
    size_t state_aligned;   // one f32 state buffer
    size_t io_aligned;      // one f16 state buffer (io_in / io_out)
    size_t off_io_in;       // offset of io_in[0] in the thread block
    size_t off_io_out;      // offset of io_out[0]
    size_t off_ring;        // offset of ring[0]
    size_t ring_slot;       // bytes per ring slot (0 unless ASYNC)
    size_t bytes_per_thread;
};

static inline void htp_gdn_io_layout_build(
    struct htp_gdn_io_layout * L,
    uint32_t S_v,
    uint8_t  in_f16,
    uint8_t  out_f16,
    uint8_t  snap_async
) {
    const size_t n_el = (size_t) S_v * S_v;
    L->state_aligned    = hex_round_up(n_el * sizeof(float), 128);
    L->io_aligned       = hex_round_up(n_el * sizeof(uint16_t), 128);
    L->off_io_in        = 2 * L->state_aligned;
    L->off_io_out       = L->off_io_in + (in_f16 ? 2 * L->io_aligned : 0);
    L->off_ring         = L->off_io_out + (out_f16 ? 2 * L->io_aligned : 0);
    L->ring_slot        = snap_async ? hex_round_up(n_el * (out_f16 ? sizeof(uint16_t) : sizeof(float)), 128) : 0;
    L->bytes_per_thread = L->off_ring + HTP_GDN_SNAP_RING * L->ring_slot;
}

// Column split (GGML_HEXAGON_GDN_COLSPLIT=N; scalar pp and ticket threads only). p is chosen to minimise the modelled makespan ceil(units / threads) * (S_v / p), ties keep the smaller p.
// p = 2^l with l <= max_log2 (at most 8). Returns 1 when unsplit. A head's S_v state rows are cut into p parts of S_v / p rows, and each (head, part) is one work unit. A part holds whole 8-row blocks of the scalar kernel, so every state row gets the same arithmetic as in the unsplit kernel (bit-identical), and no two units share a row (no cross-thread reduction).
static inline uint32_t htp_gdn_col_parts(uint32_t S_v, uint32_t nrows, uint32_t nth_max, uint32_t max_log2) {
    if (nrows == 0 || nth_max == 0) {
        return 1;
    }
    uint32_t best = 1;
    uint32_t nth  = nrows < nth_max ? nrows : nth_max;
    uint64_t best_cost = (uint64_t) ((nrows + nth - 1) / nth) * S_v;
    for (uint32_t l = 1; l <= max_log2 && l <= 3; l++) {
        const uint32_t p = 1u << l;
        if (S_v % p != 0 || (S_v / p) % 8 != 0) {
            break;
        }
        const uint32_t units = nrows * p;
        nth = units < nth_max ? units : nth_max;
        const uint64_t cost = (uint64_t) ((units + nth - 1) / nth) * (S_v / p);
        if (cost < best_cost) {
            best      = p;
            best_cost = cost;
        }
    }
    return best;
}

struct htp_gdn_hmx_vtcm_layout {
    size_t off_s_state;
    size_t off_s_f16;
    size_t off_s_col_tiles;
    size_t off_s_update_tiles;

    size_t off_q_f32[2];
    size_t off_k_f32[2];
    size_t off_v_f32[2];
    size_t off_g_f32[2];
    size_t off_b_f32[2];
    size_t off_g_raw[2];
    size_t off_b_raw[2];
    size_t off_o_f32[2];

    size_t off_k_f16;
    size_t off_v_prime_f16;
    size_t off_delta_f16;
    size_t off_d_f16;

    size_t off_q_row_tiles;
    size_t off_q_prime_row_tiles;
    size_t off_k_row_tiles;
    size_t off_k_col_tiles;
    size_t off_k_prime_row_tiles;
    size_t off_k_col_tiles_64x128;
    size_t off_kk_tiles;
    size_t off_qk_tiles;
    size_t off_v_inter_tiles;
    size_t off_o_inter_tiles;
    size_t off_inv_row_tiles;
    size_t off_a_row_tiles;
    size_t off_v_prime_col_tiles;
    size_t off_delta_tiles;
    size_t off_delta_col_tiles;
    size_t off_o_intra_tiles;
    size_t off_d_row_tiles;

    size_t off_gamma;
    size_t off_lambda_init;
    size_t off_decay_m;
    size_t off_decay_a;
    size_t off_rows_kk;
    size_t off_rows_qk;
    size_t off_rows_inv;
    size_t off_rows_a;

    size_t off_thread_scratch;
    size_t off_scales_1;
    size_t off_gate_a;       // 128 B: ssm_a[H] staged for HTP_OPFLAGS_GDN_GATE_FOLD

    size_t state_f32_bytes;  // one f32 state, unpadded (kparams->state_aligned)
    // per-head strides = unpadded block + skew_bytes; tile and scratch strides keep the 2 KB HMX alignment
    size_t st_state_f32;
    size_t st_state_f16;
    size_t st_state_tiles;
    size_t st_dma_chunk;
    size_t st_dma_scalar;
    size_t st_act_f16;
    size_t st_tile_64xSv;
    size_t st_tile_64x64;
    size_t st_decay;
    size_t st_row_vecs;
    size_t st_gamma;
    size_t st_scratch;
    size_t skew_bytes;

    uint32_t n_heads_batch;
    uint32_t n_threads;
    bool     pipeline;
    size_t   total_bytes;
};

static inline void htp_gdn_hmx_vtcm_layout_build(
    struct htp_gdn_hmx_vtcm_layout * L,
    uint32_t S_v,
    uint32_t chunk_size,
    uint32_t n_heads_batch,
    uint32_t n_threads,
    bool     pipeline,
    uint32_t gb_row,
    uint32_t skew_bytes
) {
    memset(L, 0, sizeof(*L));
    L->n_heads_batch = n_heads_batch;
    L->n_threads     = n_threads;
    L->pipeline      = pipeline;
    L->skew_bytes    = skew_bytes;

    const size_t bh = (size_t) n_heads_batch;
    const size_t nth = (size_t) (n_threads > 0 ? n_threads : 1);

    const size_t state_f32_sz   = hex_round_up(S_v * S_v * sizeof(float), 2048);
    const size_t state_f16_sz   = hex_round_up(S_v * S_v * sizeof(__fp16), 2048);
    const size_t n_sv_tiles     = S_v / 32;
    const size_t state_tiles_sz = n_sv_tiles * n_sv_tiles * 2048;

    const size_t dma_chunk_sz   = hex_round_up(chunk_size * S_v * sizeof(float), 2048);
    const size_t dma_scalar_sz  = hex_round_up(chunk_size * sizeof(float), 128);

    const size_t act_f16_sz     = hex_round_up(chunk_size * S_v * sizeof(__fp16), 2048);
    const size_t tile_64xSv_sz  = 2 * n_sv_tiles * 2048;
    const size_t tile_64x64_sz  = 4 * 2048;

    const size_t decay_sz       = 64 * 64 * sizeof(__fp16);
    const size_t row_vecs_sz    = 64 * 128;
    const size_t thread_scratch_sz = 64 * 128;

    // skew_bytes pads every per-head block (a 128-byte multiple, HVX rows). Blocks with HMX operands (tiles, inverse
    // scratch) pad by skew_bytes rounded up to 2 KB, so every tile stays 2 KB aligned.
    const size_t pad  = skew_bytes;
    const size_t tpad = hex_round_up(skew_bytes, 2048);
    L->state_f32_bytes = state_f32_sz;
    L->st_state_f32    = state_f32_sz + pad;
    L->st_state_f16    = state_f16_sz + pad;
    L->st_state_tiles  = state_tiles_sz + tpad;
    L->st_dma_chunk    = dma_chunk_sz + pad;
    L->st_dma_scalar   = dma_scalar_sz + pad;
    L->st_act_f16      = act_f16_sz + pad;
    L->st_tile_64xSv   = tile_64xSv_sz + tpad;
    L->st_tile_64x64   = tile_64x64_sz + tpad;
    L->st_decay        = decay_sz + pad;
    L->st_row_vecs     = row_vecs_sz + pad;
    L->st_gamma        = dma_scalar_sz + pad;
    L->st_scratch      = thread_scratch_sz + tpad;

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_s_state,        bh * L->st_state_f32);
    VTCM_LAYOUT_ALLOC(off, off_s_f16,          bh * L->st_state_f16);
    off = hex_align_up(off, HMX_FP16_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off, off_s_col_tiles,    bh * L->st_state_tiles);
    off = hex_align_up(off, HMX_FP16_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off, off_s_update_tiles, bh * L->st_state_tiles);

    VTCM_LAYOUT_ALLOC(off, off_q_f32[0], bh * L->st_dma_chunk);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_q_f32[1], bh * L->st_dma_chunk, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_k_f32[0], bh * L->st_dma_chunk);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_k_f32[1], bh * L->st_dma_chunk, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_v_f32[0], bh * L->st_dma_chunk);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_v_f32[1], bh * L->st_dma_chunk, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_g_f32[0], bh * L->st_dma_scalar);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_g_f32[1], bh * L->st_dma_scalar, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_b_f32[0], bh * L->st_dma_scalar);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_b_f32[1], bh * L->st_dma_scalar, pipeline);
    const size_t raw_gb_sz = hex_round_up(gb_row * chunk_size * sizeof(float), 128);
    VTCM_LAYOUT_ALLOC(off, off_g_raw[0], raw_gb_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_g_raw[1], raw_gb_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_b_raw[0], raw_gb_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_b_raw[1], raw_gb_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_o_f32[0], bh * L->st_dma_chunk);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_o_f32[1], bh * L->st_dma_chunk, pipeline);

    VTCM_LAYOUT_ALLOC(off, off_k_f16,       bh * L->st_act_f16);
    VTCM_LAYOUT_ALLOC(off, off_v_prime_f16, bh * L->st_act_f16);
    VTCM_LAYOUT_ALLOC(off, off_delta_f16,   bh * L->st_act_f16);
    VTCM_LAYOUT_ALLOC(off, off_d_f16,       bh * L->st_act_f16);

    off = hex_align_up(off, HMX_FP16_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off, off_q_row_tiles,        bh * L->st_tile_64xSv);
    VTCM_LAYOUT_ALLOC(off, off_q_prime_row_tiles,  bh * L->st_tile_64xSv);
    VTCM_LAYOUT_ALLOC(off, off_k_row_tiles,        bh * L->st_tile_64xSv);
    VTCM_LAYOUT_ALLOC(off, off_k_col_tiles,        bh * L->st_tile_64xSv);
    VTCM_LAYOUT_ALLOC(off, off_k_prime_row_tiles,  bh * L->st_tile_64xSv);
    VTCM_LAYOUT_ALLOC(off, off_k_col_tiles_64x128, bh * L->st_tile_64xSv);
    VTCM_LAYOUT_ALLOC(off, off_kk_tiles,           bh * L->st_tile_64x64);
    VTCM_LAYOUT_ALLOC(off, off_qk_tiles,           bh * L->st_tile_64x64);
    VTCM_LAYOUT_ALLOC(off, off_v_inter_tiles,      bh * L->st_tile_64xSv);
    VTCM_LAYOUT_ALLOC(off, off_o_inter_tiles,      bh * L->st_tile_64xSv);
    VTCM_LAYOUT_ALLOC(off, off_inv_row_tiles,      bh * L->st_tile_64x64);
    VTCM_LAYOUT_ALLOC(off, off_a_row_tiles,        bh * L->st_tile_64x64);
    VTCM_LAYOUT_ALLOC(off, off_v_prime_col_tiles,  bh * L->st_tile_64xSv);
    VTCM_LAYOUT_ALLOC(off, off_delta_tiles,        bh * L->st_tile_64xSv);
    VTCM_LAYOUT_ALLOC(off, off_delta_col_tiles,    bh * L->st_tile_64xSv);
    VTCM_LAYOUT_ALLOC(off, off_o_intra_tiles,      bh * L->st_tile_64xSv);
    VTCM_LAYOUT_ALLOC(off, off_d_row_tiles,        bh * L->st_tile_64xSv);

    VTCM_LAYOUT_ALLOC(off, off_gamma,       bh * L->st_gamma);
    VTCM_LAYOUT_ALLOC(off, off_lambda_init, bh * L->st_gamma);
    VTCM_LAYOUT_ALLOC(off, off_decay_m,     bh * L->st_decay);
    VTCM_LAYOUT_ALLOC(off, off_decay_a,     bh * L->st_decay);
    VTCM_LAYOUT_ALLOC(off, off_rows_kk,     bh * L->st_row_vecs);
    VTCM_LAYOUT_ALLOC(off, off_rows_qk,     bh * L->st_row_vecs);
    VTCM_LAYOUT_ALLOC(off, off_rows_inv,    bh * L->st_row_vecs);
    VTCM_LAYOUT_ALLOC(off, off_rows_a,      bh * L->st_row_vecs);

    const size_t n_scratch = (nth > bh) ? nth : bh; // one scratch block per head of the batch
    off = hex_align_up(off, HMX_FP16_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off, off_thread_scratch, n_scratch * L->st_scratch);
    off = hex_align_up(off, HMX_FP16_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off, off_scales_1,       HMX_FP16_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off, off_gate_a,         128);

    L->total_bytes = off;
}

// Batch geometry of the chunked-HMX path. gb_row = floats per token row of the raw g/beta buffers: with QKSHARE the
// row spans all H heads of the sequence (addressed by head index), otherwise it spans the batch (addressed by slot).
static inline uint32_t htp_gdn_gb_row(uint32_t opt, uint32_t H, uint32_t n_heads_batch) {
    return (opt & HTP_GDN_OPT_QKSHARE) ? H : n_heads_batch;
}

// HTP_GDN_FIX_HEAD_SKEW pad per head in bytes (kparams->skew_units x 128). Chunked HMX with n_tokens >= 64 only.
static inline uint32_t htp_gdn_skew_bytes(uint8_t fix, uint8_t skew_units, uint32_t n_tokens) {
    return ((fix & HTP_GDN_FIX_HEAD_SKEW) != 0 && n_tokens >= 64) ? (uint32_t) skew_units * 128u : 0u;
}

static inline bool htp_gdn_hmx_try_batch(
    struct htp_gdn_hmx_vtcm_layout * layout_out,
    uint32_t S_v,
    uint32_t chunk_size,
    uint32_t H,
    uint32_t bh,
    size_t   vtcm_budget,
    uint32_t n_threads,
    bool     pipeline,
    uint32_t opt,
    uint32_t skew_bytes
) {
    htp_gdn_hmx_vtcm_layout_build(layout_out, S_v, chunk_size, bh, n_threads, pipeline, htp_gdn_gb_row(opt, H, bh), skew_bytes);
    return layout_out->total_bytes <= vtcm_budget;
}

static inline bool htp_gdn_hmx_solve_layout(
    struct htp_gdn_hmx_vtcm_layout * layout_out,
    uint32_t S_v,
    uint32_t chunk_size,
    uint32_t H,
    uint32_t total_rows,
    size_t   vtcm_budget,
    uint32_t n_threads,
    bool     pipeline,
    uint32_t opt,
    uint32_t skew_bytes,
    uint32_t * n_heads_batch_out
) {
    const bool wide = (opt & HTP_GDN_OPT_WIDE) != 0;

    // legacy: at most 8 heads and at most one head per HVX thread; wide: up to HTP_GDN_MAX_BATCH heads
    uint32_t max_batch = wide ? HTP_GDN_MAX_BATCH : 8;
    if (max_batch > total_rows) {
        max_batch = total_rows;
    }
    if (!wide && max_batch > n_threads) {
        max_batch = n_threads;
    }

    // wide: the whole row range as one batch when it fits (3 cores: 10 or 12 heads per device)
    if (wide && max_batch > 0 &&
            htp_gdn_hmx_try_batch(layout_out, S_v, chunk_size, H, max_batch, vtcm_budget, n_threads, pipeline, opt, skew_bytes)) {
        *n_heads_batch_out = max_batch;
        return true;
    }

    static const uint32_t candidates[] = { 8, 6, 4, 2, 1 };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        const uint32_t bh = candidates[i];
        if (bh > max_batch) {
            continue;
        }
        if (htp_gdn_hmx_try_batch(layout_out, S_v, chunk_size, H, bh, vtcm_budget, n_threads, pipeline, opt, skew_bytes)) {
            *n_heads_batch_out = bh;
            return true;
        }
    }
    if (pipeline && !wide && (opt & ~HTP_GDN_OPT_UP_ALL) == 0) {
        // legacy fallback (unchanged): non-pipelined layout; the new arms never take it, they require the DMA pipeline
        return htp_gdn_hmx_solve_layout(layout_out, S_v, chunk_size, H, total_rows, vtcm_budget, n_threads, false, opt, skew_bytes, n_heads_batch_out);
    }
    return false;
}

#endif // HTP_GATED_DELTA_NET_OPS_H
