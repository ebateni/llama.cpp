#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <memory>
#include <chrono>
#include <mutex>
#include <thread>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <regex>
#include <queue>
#include <deque>
#include <algorithm>
#include <cmath>
#include <initializer_list>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#       define NOMINMAX
#    endif
#    include <windows.h>
#    include <sal.h>
#else
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"

#include <AEEStdErr.h>
#include <dspqueue.h>
#include <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-hexagon.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "htp-opnode.h"
#include "htp-ops.h"
#include "htp/matmul-ops.h"
#include "htp/binary-ops.h"
#include "htp/flash-attn-ops.h"
#include "htp/unary-ops.h"
#include "htp/get-rows-ops.h"
#include "htp/set-rows-ops.h"
#include "htp/softmax-ops.h"
#include "htp/rope-ops.h"
#include "htp/ssm-conv.h"
#include "htp/gated-delta-net-ops.h"
#include "htp/argsort-ops.h"
#include "htp_iface.h"
#include "htp-drv.h"

using intvec  = std::vector<int>;
using uintvec = std::vector<unsigned int>;
using u32vec  = std::vector<uint32_t>;

#define GGML_HEXAGON_MAX_SESSIONS          16

#define GGML_HEXAGON_FENCE_SLOT_SIZE       128

struct ggml_hexagon_device_config {
    int         physical_idx = 0;
    int         virtual_idx  = 0;
    int         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

static ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

static int    opt_arch    = 0; // autodetect
static size_t opt_ndev    = 1;
static size_t opt_nhvx    = 0; // use all
static int    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
static int      opt_lane_threads = 0;  // GGML_HEXAGON_LANE_THREADS=K2: HVX lane B threads (Phase-1 overlap experiment; 0 = off)
static int      opt_lane_a_mb    = 0;  // GGML_HEXAGON_LANE_A_MB=M: VTCM budget of the main lane in MB (lane B gets the rest)
static uint32_t opt_lane_ops     = 0;  // GGML_HEXAGON_LANE_OPS: bit 0 ssm, 1 normg, 2 qk-norm, 3 rope, 4 gatemul (HVX-only ops sent to lane B)

// HTP_OPFLAGS_LANE_B for the HVX-only ops selected by GGML_HEXAGON_LANE_OPS (Phase-1 lane experiment, see htp/main.c)
static uint32_t ggml_hexagon_lane_flag(uint32_t opcode) {
    if (opt_lane_threads <= 0 || opt_lane_ops == 0) {
        return 0;
    }
    if ((opt_lane_ops & 1)  && (opcode == HTP_OP_SSM_CONV || opcode == HTP_OP_SSM_CONV_SILU)) return HTP_OPFLAGS_LANE_B;
    if ((opt_lane_ops & 2)  && opcode == HTP_OP_RMS_NORM_MUL_SILU) return HTP_OPFLAGS_LANE_B;
    if ((opt_lane_ops & 4)  && opcode == HTP_OP_RMS_NORM_SCALE)    return HTP_OPFLAGS_LANE_B;
    if ((opt_lane_ops & 8)  && opcode == HTP_OP_ROPE)              return HTP_OPFLAGS_LANE_B;
    if ((opt_lane_ops & 16) && opcode == HTP_OP_GATE_MUL)          return HTP_OPFLAGS_LANE_B;
    return 0;
}

static size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
static size_t opt_mbuf    = 1ul * 1024 * 1024 * 1024; // max buffer size
static int    opt_etm     = 0;
static int    opt_verbose = 0;
static int    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
static bool   opt_hostbuf = false;
static bool   opt_dma64   = false;

static int    opt_mm_select = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
static int    opt_fa_select = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
static int    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
static int    opt_gdn_hmx  = 1; // GGML_HEXAGON_GDN_HMX: 0 = no HMX chunked path for GDN (scalar recurrence for K=1)
static int    opt_gdn_prompt_hmx = 0; // GGML_HEXAGON_GDN_PROMPT_HMX=1: K > 1 batches whose typed snapshot range is the final state (op_params[3..4] = 0, 1) may use the chunked HMX path, which writes plane 0 only
static int    opt_gdn_snap = 0; // GGML_HEXAGON_GDN_SNAP: 0 = sync (default), 1 = skip, 2 = async (VC-04), 3 = sync via ticket path, 4 = version (D2Snap)
static uint32_t opt_gdn_opt = 0; // HTP_GDN_OPT_* bits: GGML_HEXAGON_GDN_HMX_BATCH / _WIDE / _QKSHARE / _STATE_MERGE (0 = legacy)
static uint32_t opt_gdn_fix = 0; // GGML_HEXAGON_GDN_FIX bitmask: HTP_GDN_FIX_* (kparams->fix; default 0 = off)
static uint32_t opt_gdn_skew_units = 1; // GGML_HEXAGON_GDN_SKEW=<bytes> (default 128): per-head VTCM pad of HTP_GDN_FIX_HEAD_SKEW, in 128-byte units; 0 = no pad
static int    opt_gdn_colsplit = 0; // GGML_HEXAGON_GDN_COLSPLIT=N: scalar pp/ticket GDN threads split each head into 2^N state-row parts (0 = off)
static int    opt_concat_t_hvx = 0; // GGML_HEXAGON_CONCAT_T_HVX=1: transposed conv-window CONCAT uses the HVX-transpose pipeline (concat-ops.c)
static int    opt_ar_select = 2; // 2 = fused ALLREDUCE+ADD (DMA, default), 1 = unfused ALLREDUCE (DMA), 0 = fallback to CPY+FENCE
static int    opt_ar_gap_fuse = 0; // GGML_HEXAGON_AR_GAP_FUSE: 0 = off (default), 1 = gap-fused ALLREDUCE+ADD, chunked reduce-scatter to all dsts (MC01), 2 = gap-fused, each rank reduces its full range into its own dst
static int    opt_act_tile = 0; // GGML_HEXAGON_ACT_TILE: 1 = RMS_NORM_MUL also writes fp16 HMX activation tiles (fp32 kept), matmul consumers DMA them; 2 = tile-only where every reader is an HMX MUL_MAT (default 0 = off)
static int    opt_gather_act_tile = 0; // GGML_HEXAGON_GATHER_ACT_TILE=1: MC05 norm gather writes fp16 HMX activation tiles when every reader is an HMX matmul (needs AR_GAP_FUSE=3; RS_ROW owner rows are 16-aligned to match; default off)
static int    opt_ar_rms_fuse = 0; // GGML_HEXAGON_AR_RMS_FUSE=1: the owner-row ALLREDUCE+ADD and its RMS_NORM_MUL run as one op (AE-03); needs AR_GAP_FUSE=3 and RS_ROW; prefill batches only
static int    opt_ar_partial_f16 = 0; // GGML_HEXAGON_AR_PARTIAL_F16=1: fused ALLREDUCE+ADD+RMS partial producers write fp16 rows (needs AR_RMS_FUSE)
static constexpr uint32_t AR_RMS_AG_TILE_ROWS = 16; // row block of the tile-output norm gather (htp allreduce-ops.c RMS_AG_TILE_ROWS)
// Batches below PREFILL_MIN_ROWS (decode, speculative verify) take the original path for every prefill-only optimization of this branch:
// the act-tile plan, the norm-gather tile admission and deferral, and the owner-row alignment of the gather.
static constexpr uint32_t PREFILL_MIN_ROWS = 64;
// Owner-row partition of the RS-ROW reduce-scatter and of the MC05 norm gather: ceil(ne1 / n_ranks) rows per rank.
// With GGML_HEXAGON_GATHER_ACT_TILE the chunk is rounded up to whole 16-row tile blocks, for the fp32 and the tile gather alike, so the reduce and the gather always agree on the owner rows.
static uint32_t ar_rs_row_chunk_rows(uint32_t ne1, uint32_t n_ranks) {
    uint32_t rows = (ne1 + n_ranks - 1) / n_ranks;
    if (opt_gather_act_tile && ne1 >= PREFILL_MIN_ROWS) {  // prefill-sized outputs only; below the gate: ceil(ne1 / n_ranks)
        rows = (rows + AR_RMS_AG_TILE_ROWS - 1) / AR_RMS_AG_TILE_ROWS * AR_RMS_AG_TILE_ROWS;
    }
    return rows;
}
static inline bool ar_gap_chunked(void) { return opt_ar_gap_fuse != 2; } // modes 1 and 3: rank-chunked reduce-scatter; mode 2: full range
// GGML_HEXAGON_RS_ROW=1 (default 0, only with GGML_HEXAGON_AR_GAP_FUSE=3): the MC05 reduce-scatter cuts the rows (not the flat elements) of batches with at least opt_rs_row_min_rows rows (GGML_HEXAGON_RS_ROW_MIN_ROWS, default 64, floor 16), and the residual ADD writes its dst on the owner rank only where ggml-backend-meta.cpp allows it.
// Smaller batches keep the unchanged path: no candidates, no owner attempt, flat chunks, full writes.
static int      opt_rs_row          = 0;
static uint32_t opt_rs_row_min_rows = 64;
static inline bool ar_rs_row_rows(uint32_t ne1) { return opt_rs_row != 0 && ne1 >= opt_rs_row_min_rows; }

// GGML_HEXAGON_AR_RMS_AG_MT=1: the MC05 norm gather runs on all session threads (row ranges per thread, dma[ith]). Default 0: one thread.
static bool ar_rms_ag_mt(void) {
    static const bool on = [] {
        const char * v = getenv("GGML_HEXAGON_AR_RMS_AG_MT");
        return v != nullptr && atoi(v) != 0;
    }();
    return on;
}

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
static u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

static int opt_opbatch  = 1280; // max number of ops in a batch
static int opt_hmx_min_rows = 2; // min activation rows routed to HMX matmul
static int opt_opqueue  = 32;   // max number of pending batches
static int opt_optrace  = 0;    // trace buffer size per thread (0 means default)
static int opt_oppoll   = 0;    // polling for batch completions
static int opt_opfusion = 1;    // enable/disable op fusion
static int opt_fuse_s150 = 0;   // speed150 switches, default off: GDN gate prep fusion (T <= 32) + NX N%32 guard
static int opt_gdn_gate_pf = 0; // GGML_HEXAGON_GDN_GATE_PF=1 (default off): GDN gate prep fusion also for prefill batches (T <= GDN_GATE_HOST_MAX_T)
static int opt_gate_mul = 0;    // GGML_HEXAGON_GATE_MUL=1 (default off): attention gate MUL(x, SIGMOID(g)) becomes one GATE_MUL op (unary-ops.c op_gate_mul)
static const int64_t GDN_GATE_HOST_MAX_T = 1024; // rows per GDN_GATE op with GGML_HEXAGON_GDN_GATE_PF (== GDN_GATE_MAX_T in htp/unary-ops.c)
static int opt_norm_gated = 0;  // GGML_HEXAGON_NORM_GATED=1 (default off): RMS_NORM_MUL + SWIGLU(z, normalized) -> RMS_NORM_MUL_SILU (qwen35 gated output norm)
static uint32_t opt_lossy_gdn = 0; // GGML_HEXAGON_LOSSY_GDN=<bitmask> (default 0, numerics-changing): bits 0..7 GATED_DELTA_NET (HTP_LOSSY_GDN_*), bits 8..13 SSM_CONV_CHAIN (HTP_LOSSY_CONV_*)
static int opt_gdn_gate_fold = 0; // GGML_HEXAGON_GDN_GATE_FOLD=1 (default off, needs GDN_QKSHARE + GDN_HMX, prefill): the GDN gate SOFTPLUS*A and beta SIGMOID run in the GATED_DELTA_NET phase 1a (no GDN_GATE op; bit-exact)
static int opt_rope_kv = 0;   // GGML_HEXAGON_ROPE_KV=1: the K/V cache SET_ROWS ops that follow ROPE_NX are folded into it (HTP_OPFLAGS_ROPE_KV)
static int opt_gate_tile = 0; // GGML_HEXAGON_GATE_TILE=1 (default off, needs GATE_MUL): the attention gate multiply writes the fp16 HMX activation tiles of the attn_output matmul directly
static int opt_ng_tile = 0;   // GGML_HEXAGON_NG_TILE=1 (default off): the gated output norm writes the fp16 HMX activation tiles of the ssm_out matmul directly (needs NORM_GATED, ACT_TILE)
static int opt_conv_dyn = 0; // GGML_HEXAGON_CONV_DYN=1: SSM_CONV_CHAIN+QK_NORM tiles (head unit x token block) drained by all threads through an atomic counter (bit-exact)
static int opt_conv_split32 = 0; // GGML_HEXAGON_CONV_SPLIT32=1: SSM_CONV_CHAIN thread ranges at 32-channel granularity in the v region (bit-exact)
static uint32_t opt_conv_split32_cq = 107, opt_conv_split32_cv = 62; // GGML_HEXAGON_CONV_SPLIT32_COST=cq,cv (relative cost of a q/k / v head)
static int opt_fa_exp_v2 = 0; // GGML_HEXAGON_FA_EXP_V2=1: flash-attn softmax exp2 with the accurate native-hf polynomial (numerics-changing)
static int opt_lm_fa_skip = 0; // GGML_HEXAGON_LM_FA_SKIP=1 (default off, exact for causal masks): FLASH_ATTN_EXT softmax skips row vectors fully masked in a KV block
static int opt_lm_norm = 0;  // GGML_HEXAGON_LM_NORM=1..3 (default off, numerics-changing): AR/norm_gated RMS norm with 4-row packed rsqrt; 2 = 2 Newton iterations, 3 = 1 iteration (retuned)
static int opt_sig16_v2 = 0;  // GGML_HEXAGON_SIG16_V2=1 (default off, numerics-changing): accurate shorter fp16 sigmoid in the NG_SIG16 / C2_SILU=3 paths
static int opt_ng_sig16 = 0;  // GGML_HEXAGON_NG_SIG16=1 (default off, numerics-changing): gated output norm sigmoid in fp16 (needs C2_NG >= 2)
static int opt_rope_fast = 0;   // GGML_HEXAGON_ROPE_FAST=1 (default off, theta rounding differs): vectorised MROPE theta cache for qwen35's 11/11/10 sections
static int opt_rope_norm = 0;   // GGML_HEXAGON_ROPE_NORM=1 (default off, bit-exact; needs ROPE_NX): attention q/k RMS_NORM+MUL run inside the ROPE_NX op
static int opt_dma_clean = 0;   // GGML_HEXAGON_DMA_CLEAN=1 (default off): DMA-only ops skip the flush of inputs DMA wrote in the batch (htp-tensor.c)
static int opt_ar_diag   = 0;   // diagnostic only: log why ALLREDUCE+ADD is not fused
static int opt_ssm_conv_pipe = 0; // pipelined fused SSM_CONV prefill (token chunks, DMA double buffer), default off
static int opt_ssm_conv_chain = 0; // GGML_HEXAGON_SSM_CONV_CHAIN (default 0): 1 = CONCAT+CPY+SSM_CONV+SILU of the GDN conv block as one HTP_OP_SSM_CONV_CHAIN; 2 = also the per-head q/k RMS_NORM+SCALE epilogue (2 implies 1)
static uint32_t opt_ssm_conv_chain_tb = 0; // GGML_HEXAGON_SSM_CONV_CHAIN_TB=<tokens> (default 0): token block size of the non-QK chain; 0 = whole-token tiles when they fit in VTCM
static int opt_c2_ng = 0; // GGML_HEXAGON_C2_NG (C2Elt, default 0): gated output norm variant: 0 original, 1 per-op constants hoisted, 2 also two rows per step (bit-identical)
static int opt_c2_rs = 0; // GGML_HEXAGON_C2_RS (C2Elt, default 0): fused RMS_NORM+SCALE (q/k L2 norm) variant, same meaning
static int opt_c2_silu = 0; // GGML_HEXAGON_C2_SILU (C2Elt, default 0): SSM_CONV_CHAIN SiLU tile variant: 0 original, 1 four vectors per step, 2 also the two-iteration reciprocal
static const int64_t SSM_CONV_CHAIN_MIN_ROWS = 64; // the chain is only matched when the CONCAT carries >= 64 tokens per sequence; smaller batches keep the unfused path
static int opt_dispatch_1pass = 0; // GGML_HEXAGON_DISPATCH_1PASS=1 (speed150): prepare every batch, then issue the writes back to back
static int opt_cycle_timing   = 0; // LLAMA_CYCLE_TIMING>=2: host timestamps of each batch write, printed at synchronize

// host timestamps of one batch write (LLAMA_CYCLE_TIMING>=2); printed and cleared by ggml_hexagon_dispatch_log_flush()
struct ggml_hexagon_disp_rec {
    const char * name;
    uint64_t     seq;
    uint64_t     t_begin;
    uint64_t     t_w0;
    uint64_t     t_w1;
};
static std::vector<ggml_hexagon_disp_rec> opt_disp_log;
static int opt_hmx_int = 0;    // enable native HMX INT8/INT4 matrix multiplication
static int opt_swiglu_out = 0;      // ImplSwigluOut: FFN gate/up NX writes silu(g)*u as fp16 act tiles read by the FFN down, GGML_HEXAGON_SWIGLU_OUT=1
static int opt_swiglu_nc = HTP_MM_SWIGLU_NC; // GGML_HEXAGON_SWIGLU_NC (512 or 768; default 768): fused gate/up SWIGLU chunk width (VrAr step 0)
static uint32_t opt_ar_block_rows = 16;  // GGML_HEXAGON_AR_BLOCK_ROWS=8|16: rows per block of the fused AR+ADD+RMS op (8 spreads 171 owner rows over all 8 threads)
static uint32_t opt_vr_ar = 0;  // GGML_HEXAGON_VR_AR bitmask (VrAr): 1 = C, 2 = B, 4 = A (residual owner rows resident); prefill batches with AR_RMS_FUSE
// GGML_HEXAGON_ELT_SMALL bitmask (default 0; design and A/B in impl/ImplEltSmall.md). Batch gate: GGML_HEXAGON_ELT_SMALL_MIN_ROWS (default 64, floor 16); a graph below it takes no bit.
// Bits: 1 AE-06 beta SIGMOID on [H,T] (qwen35.cpp), 2 AE-07 unary/RMS row blocks capped at GGML_HEXAGON_ELT_SMALL_ROW_CAP rows (not tile output), 4 AE-09 FA gate SIGMOID without CONT (qwen35.cpp), 8 AE-10 FA Q-norm RMS_NORM+MUL fusion on a strided src0, 16 AE-16 GDN pos0 state clear without SCALE-by-0.
static int      opt_elt_small          = 0;
static uint32_t opt_elt_small_min_rows = 64;
static constexpr uint32_t GGML_HEXAGON_ELT_SMALL_ROW_CAP = 32;
static int opt_swiglu_out_min_rows = 64; // GGML_HEXAGON_SWIGLU_OUT_MIN_ROWS: batches below this keep the original path (no admission scan, no fusion)
static int opt_swiglu_sig16 = 0;  // GGML_HEXAGON_SWIGLU_SIG16=1 (default off, numerics-changing): fp16 sigmoid in the FFN SWIGLU_OUT epilogue
static int opt_lx_fuse = 0;   // GGML_HEXAGON_LX_FUSE bitmask (LxFuseFree, default 0): bit 0 = SWIGLU_OUT epilogue without the B1 value barriers
// Row counts the fused SWIGLU output takes: at least the batch gate, and whole 32-row tiles (partial last tiles diverge on device, impl/SwigluFix.md).
static inline bool swiglu_out_rows_ok(int64_t rows) {
    return rows >= opt_swiglu_out_min_rows && rows % 32 == 0;
}
// GGML_HEXAGON_MM_PROLOGUE (MM-03, default 0): HMX MUL_MAT/NX op flags. 1 = half-width first/last weight chunk + cross-weight
// prefetch, 2 = cross-weight prefetch only, 3 = half-width edges only. Scheduling only (bit-identical); htp/htp-ops.h.
static uint32_t opt_mm_prologue_flags = 0;
static uint32_t opt_ar_dma_clean_flags = 0;   // GGML_HEXAGON_AR_DMA_CLEAN=1: HTP_OPFLAGS_AR_DMA_CLEAN on ALLREDUCE ops of DMA_CLEAN graphs (DMA-only reader/writer)
static uint32_t opt_mm_a8w8_ops = 0;   // GGML_HEXAGON_MM_A8W8 bitmask: 1 = SWIGLU gate/up, 2 = ffn_down (plain MUL_MAT, K >= 3072), 4 = ssm_out/attn_output (plain MUL_MAT, K < 3072), 8 = plain MUL_MAT_NX (qkv)
static uint32_t opt_a8w8_pre = 0;      // GGML_HEXAGON_A8W8_PRE bitmask: pre-expanded int8 weight blob appended to the tiled Q4_0 weight (1 = ffn_gate/ffn_up, 2 = ffn_down, 4 = all other Q4_0 matmul weights except lm_head)
static uint32_t opt_mm_a8w8_flags = 0; // GGML_HEXAGON_MM_A8W8=1: SWIGLU gate/up through u8 x int8 HMX jobs (htp/hmx-a8w8.h)
static uint32_t opt_mm_c1small_flags = 0;     // GGML_HEXAGON_MM_C1SMALL=1: HTP_OPFLAGS_MM_C1SMALL on prefill MUL_MAT (not NX)
static uint32_t opt_mm_early_w_flags = 0;     // GGML_HEXAGON_MM_EARLY_W=1: HTP_OPFLAGS_MM_EARLY_W on prefill MUL_MAT/NX (needs MM_ACT_OVERLAP): first weight chunk DMA at op entry
static uint32_t opt_mm_act_overlap_flags = 0;  // GGML_HEXAGON_MM_ACT_OVERLAP=1: HTP_OPFLAGS_MM_ACT_OVERLAP on prefill MUL_MAT/NX (matmul-ops.c hmx_mm_pipe_run)
static uint32_t mm_prologue_op_flags(int v) {
    switch (v) {
        case 1:  return HTP_OPFLAGS_MM_SPLIT_EDGES | HTP_OPFLAGS_MM_CROSS_WEIGHT;
        case 2:  return HTP_OPFLAGS_MM_CROSS_WEIGHT;
        case 3:  return HTP_OPFLAGS_MM_SPLIT_EDGES;
        default: return 0;
    }
}
// GGML_HEXAGON_NX_ALPHABETA (MM-02, default 0): ssm_alpha/beta (N=10/12) join the HMX MUL_MAT_NX group of qkv+z, and the ssm_dt bias of alpha is fused into that group's last weight (try_fuse_nx_bias).
// Needs GGML_HEXAGON_FUSE_S150 and the LLAMA_S150_GDN_ORDER graph order (qwen35.cpp expands alpha and its ADD right behind beta).
static int opt_nx_alphabeta = 0;
// MM-02 / MM-03 prefill gate: both switches engage only for batches of at least this many rows. qwen35.cpp applies the same threshold to ubatch.n_tokens (build_layer_attn_linear).
// Smaller batches (decode M=1, draft verify M=17, the 4-token chunk) keep the original graph order, fusion and kernel schedule.
static constexpr int64_t GGML_HEXAGON_MM_PREFILL_MIN_ROWS = 64;
static inline int64_t ggml_hexagon_src1_rows(const struct ggml_tensor * t) {
    return t->ne[1] * t->ne[2] * t->ne[3];
}
// Batch gate of the ELT_SMALL bits: a graph's token count is ne[1] of its first MUL_MAT activation (one node read). Below
// opt_elt_small_min_rows the graph gets no bit, so its plan, row blocks and fusions keep the original path.
static int ggml_hexagon_elt_small_graph_bits(const ggml_cgraph * graph) {
    if (opt_elt_small == 0) {
        return 0;
    }
    for (int i = 0; i < graph->n_nodes; i++) {
        if (graph->nodes[i]->op == GGML_OP_MUL_MAT) {
            return graph->nodes[i]->src[1]->ne[1] >= (int64_t) opt_elt_small_min_rows ? opt_elt_small : 0;
        }
    }
    return 0;
}
static int opt_enqueue_fast = 1; // fast op-enqueue (decode-only): replay a pre-built op-descriptor
                                 // plan for a cached subgraph uid instead of re-running the per-op try_fuse+fit_op+add_op host path on every repeat of the same graph (steady-state decode re-enqueues the same subgraph from scratch every single token, unlike prefill's one-shot dispatch of the whole batch). GGML_HEXAGON_EARLY_FLUSH_OPS=N (prefill, default 32, 0 = off): repeated early batch flush, see ggml_hexagon_early_flush.
                                 // Ported (ALLREDUCE-plan-replay code excluded: provably dead on this single-device build, see ggml_backend_hexagon_comm_allreduce_tensor's n_backends<2 early-return with opt_ndev's own default of 1) from the old private fork's "fast op-enqueue" mechanism. Set GGML_HEXAGON_ENQUEUE_FAST=0 to disable.
static int opt_early_flush_ops = 32;

enum ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ADD  = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL   = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ADD    = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX     = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX  = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY        = (1 << 6), // 64
    GGML_HEXAGON_FUSE_SSM_CONV_SILU  = (1 << 7), // 128
    GGML_HEXAGON_FUSE_CONCAT_SSM_CONV = (1 << 8), // 256
    GGML_HEXAGON_FUSE_ROPE_NX         = (1 << 9), // 512
    GGML_HEXAGON_FUSE_RMS_NORM_SCALE   = (1 << 10), // 1024
};

static inline bool ggml_hexagon_is_fusion_enabled(int flag) {
    if (opt_opfusion <= 0) return false;
    if (opt_opfusion == 1) return true; // 1 enables all
    return (opt_opfusion & flag) != 0;
}

static std::regex* opt_opfilter = NULL; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    if (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

static const char * status_to_str(uint32_t status) {
    switch (status) {
        case HTP_STATUS_OK:
            return "OK";
        case HTP_STATUS_NO_SUPPORT:
            return "NO-SUPPORT";
        case HTP_STATUS_INVAL_PARAMS:
            return "INVAL-PARAMS";
        case HTP_STATUS_VTCM_TOO_SMALL:
            return "VTCM-TOO-SMALL";
        case HTP_STATUS_INTERNAL_ERR:
            return "INTERNAL-ERROR";
        default: {
            static thread_local char buf[128];
            if (status >= HTP_STATUS_VTCM_TOO_SMALL + 0x10) {
                const uint32_t c = status - HTP_STATUS_VTCM_TOO_SMALL - 0x10;
                snprintf(buf, sizeof(buf), "DIAG-AR flags=%u block=%u thread=%u row=%u", c & 0xf, (c >> 4) & 0xff, (c >> 12) & 0xff, c >> 20);
                return buf;
            }
            if (status >= 0x20 && status < 0x400) {
                snprintf(buf, sizeof(buf), "A8W8-CHECK bad_tiles_pct=%u rt0_all_bad=%u ct0_all_bad=%u", (status - 0x20) & 0x7f, (status >> 7) & 1, (status >> 8) & 1);
                return buf;
            }
            return "UNKNOWN";
        }
    }
}

// ** debug helpers

static void ggml_hexagon_dump_op_exec(const std::string &sess_name, const htp_opnode & node, const uint32_t req_flags) {
    if (!opt_verbose) return;

    htp_opformat fmt(node);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                node.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

static void ggml_hexagon_dump_op_supp(const std::string &sess_name, const struct ggml_tensor * op, bool supp) {
    if (!opt_verbose) return;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

static const char * htp_event_name(uint16_t id) {
    switch (id) {
        case HTP_TRACE_EVT_DMA:            return "DMA";
        case HTP_TRACE_EVT_HVX_COMP:       return "HVX_COMP";
        case HTP_TRACE_EVT_HVX_A_QUANT:    return "HVX_A_QUANT";
        case HTP_TRACE_EVT_HVX_A_PREP:     return "HVX_A_PREP";
        case HTP_TRACE_EVT_HVX_W_DEQUANT:  return "HVX_W_DEQUANT";
        case HTP_TRACE_EVT_HVX_W_PREP:     return "HVX_W_PREP";
        case HTP_TRACE_EVT_HVX_O_PROC:     return "HVX_O_PROC";
        case HTP_TRACE_EVT_HVX_FA_QK:      return "HVX_QK_FA";
        case HTP_TRACE_EVT_HVX_FA_SFM:     return "HVX_SFM_FA";
        case HTP_TRACE_EVT_HVX_FA_Q_PREP:  return "HVX_Q_PREP";
        case HTP_TRACE_EVT_HVX_FA_K_PREP:  return "HVX_K_PREP";
        case HTP_TRACE_EVT_HVX_FA_V_PREP:  return "HVX_V_PREP";
        case HTP_TRACE_EVT_HVX_GDN_PREP:   return "HVX_GDN_PREP";
        case HTP_TRACE_EVT_HVX_GDN_SOLVE:  return "HVX_GDN_SOLVE";
        case HTP_TRACE_EVT_HVX_GDN_V_PREP: return "HVX_GDN_V_PREP";
        case HTP_TRACE_EVT_HVX_GDN_D_PREP: return "HVX_GDN_D_PREP";
        case HTP_TRACE_EVT_HVX_GDN_OUT:    return "HVX_GDN_OUT";
        case HTP_TRACE_EVT_HVX_GDN_STATE:  return "HVX_GDN_STATE";
        case HTP_TRACE_EVT_HVX_GDN_REM:    return "HVX_GDN_REM";
        case HTP_TRACE_EVT_HMX_COMP:       return "HMX_COMP";
        case HTP_TRACE_EVT_L2FLUSH:        return "L2FLUSH";
        case HTP_TRACE_EVT_INIT:           return "INIT";
        case HTP_TRACE_EVT_BUFF:           return "BUFF";
        case HTP_TRACE_EVT_FENCE:          return "FENCE";
        default:                           return "UNKNOWN";
    }
}

static void ggml_hexagon_dump_op_prof(const std::string &sess_name, const htp_opnode & node, const htp_prof_desc & pd) {
    if (!opt_profile) return;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    const uint32_t * pmu = pd.pmu;

    char pmu_str[256] = "";
    if (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, sizeof(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(node);
    float mhz = op_usec > 0 ? (float) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            node.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

static void ggml_hexagon_dump_batch_prof(const std::string & sess_name, const htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    float batch_mhz = rsp.usecs > 0 ? (float) batch_cycles / rsp.usecs : 0.0f;

    char evt_str[256] = "----";
    if (opt_profile == 3) {
        snprintf(evt_str, sizeof(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (unsigned long long) batch_cycles, (unsigned long long) rsp.cycles_start, batch_mhz);
}

static void ggml_hexagon_dump_trace_events(const std::string & sess_name, const htp_opbatch_rsp & rsp,
                                           const htp_trace_desc * trace_events, uint32_t n_traces) {
    if (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        for (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        for (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            for (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                const auto & e = trace_events[t * n_traces + idx];
                bool is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

enum ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_REPACK    = (1 << 0),
    GGML_HEXAGON_TENSOR_WEIGHT    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
    GGML_HEXAGON_TENSOR_ACT_TILE  = (1 << 4),
    GGML_HEXAGON_TENSOR_A8W8PRE   = (1 << 5),
};

static inline bool ggml_hexagon_is_repack_type(enum ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
           type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q6_K ||
           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K;
}

// Size of one repacked row in the DSP tiled layout. The Q6_K, Q5_K and Q4_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other repack types the tile has the same size as the ggml blocks.
static inline size_t ggml_hexagon_tiled_row_size(enum ggml_type type, int64_t ne0) {
    if (type == GGML_TYPE_Q6_K) {
        return (size_t) (ne0 / 32) * (HTP_MM_WEIGHT_TILE_SIZE_Q6_K / 32);
    }
    if (type == GGML_TYPE_Q4_K) {
        return (size_t) (ne0 / 32) * (HTP_MM_WEIGHT_TILE_SIZE_Q4_1 / 32);
    }
    if (type == GGML_TYPE_Q5_K) {
        return (size_t) (ne0 / 32) * (HTP_MM_WEIGHT_TILE_SIZE_Q5_K / 32);
    }
    return ggml_row_size(type, ne0);
}


// ** A8W8 pre-expanded int8 weights (GGML_HEXAGON_A8W8_PRE): per Q4_0 weight tensor an int8 copy of the A8W8 expand output is stored behind the tiled Q4_0 data (same allocation; blob offset = round_up(tiled bytes, 128)).
// Per column tile ct (32 weight rows): n_k * 1 KB int8 HMX weight tiles, sw f32[32] at +0, cs int32[32] at +128, padded to HTP_A8W8_PE_TAIL (1 KB) so every column tile stays 1 KB aligned -- exactly what htp/hmx-a8w8.h a8w8_expand_column_tile() produces from the Q4_0 tiles.
static bool ggml_hexagon_a8w8_pre_match(const struct ggml_tensor * t) {
    if (opt_a8w8_pre == 0 || t->type != GGML_TYPE_Q4_0 || t->ne[2] != 1 || t->ne[3] != 1 || (t->ne[0] % 32) != 0) {
        return false;
    }
    const char * n = t->name;
    uint32_t want = 4u;
    if (strstr(n, "ffn_gate") || strstr(n, "ffn_up")) want = 1u;
    else if (strstr(n, "ffn_down")) want = 2u;
    else if (strstr(n, "output.weight") || strstr(n, "token_embd") || !strstr(n, "weight")) want = 0u;
    return (opt_a8w8_pre & want) != 0;
}
static inline size_t ggml_hexagon_a8w8_pre_q4_bytes(const struct ggml_tensor * t) {
    return ggml_hexagon_tiled_row_size(t->type, hex_round_up(t->ne[0], 32)) * hex_round_up(t->ne[1], 32);
}
static inline size_t ggml_hexagon_a8w8_pre_blob_off(const struct ggml_tensor * t) {
    return hex_round_up(ggml_hexagon_a8w8_pre_q4_bytes(t), 128);
}
static inline size_t ggml_hexagon_a8w8_pre_blob_bytes(const struct ggml_tensor * t) {
    const size_t n_k = hex_round_up(t->ne[0], 32) / 32, n_ct = hex_round_up(t->ne[1], 32) / 32;
    return n_ct * (n_k * 1024 + HTP_A8W8_PE_TAIL);
}

static inline bool ggml_hexagon_is_hmx_weight_type(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || ggml_hexagon_is_repack_type(type);
}

struct ggml_hexagon_session;

static void ggml_hexagon_precompute_matmul_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
);

static void ggml_hexagon_precompute_fused_matmul_add_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
);

static void ggml_hexagon_precompute_unary_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_unary_kernel_params * kparams,
    bool                             tile_out = false
);

static bool ggml_hexagon_precompute_binary_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_binary_kernel_params * kparams
);

static void ggml_hexagon_precompute_get_rows_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_get_rows_kernel_params * kparams
);

static void ggml_hexagon_precompute_set_rows_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_set_rows_kernel_params * kparams
);

static void ggml_hexagon_precompute_softmax_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_softmax_kernel_params * kparams
);

static void ggml_hexagon_precompute_rope_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_rope_kernel_params * kparams
);

static void ggml_hexagon_precompute_ssm_conv_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_ssm_conv_kernel_params * kparams,
    bool fused_concat = false,
    uint32_t n_state_rows = 0
);

static void ggml_hexagon_precompute_gated_delta_net_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_gdn_kernel_params * kparams
);

static void ggml_hexagon_precompute_sort_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    bool is_top_k,
    struct htp_sort_kernel_params * kparams
);

static bool ggml_hexagon_precompute_swiglu_nx_params(const struct ggml_hexagon_session * sess, const struct ggml_tensor * w0,
                                                     const struct ggml_tensor * x, struct htp_mm_kernel_params * kparams);

static void ggml_hexagon_precompute_fused_mmnx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    int32_t n_weights,
    size_t src2_size,
    struct htp_mm_kernel_params * kparams
);

static void ggml_hexagon_precompute_fused_mmidnx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
);

static bool ggml_hexagon_precompute_allreduce_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    bool has_add,
    bool is_row_bcast,
    struct htp_allreduce_kernel_params * kparams,
    bool chunk_add = false
);

static bool mm_is_hmx_eligible(const ggml_tensor * t);
static htp_op_code op_remap_to_htp(const ggml_tensor * t);
static bool is_supported_mul_mat_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams);
static bool is_supported_mul_mat_id_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams);
static bool is_mergeable_mul_mat(const ggml_tensor * t);
static bool is_mergeable_mul_mat_pair(const ggml_tensor * n1, const ggml_tensor * n2);
static bool is_mergeable_mul_mat_id(const ggml_tensor * t);
static bool is_mergeable_mul_mat_id_pair(const ggml_tensor * n1, const ggml_tensor * n2);
static bool is_mergeable_rope_pair(const ggml_tensor * n1, const ggml_tensor * n2);

// ** backend sessions

struct ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_size { 0 };
    uint32_t             flags { 0 };
};

static inline bool ggml_hexagon_tensor_is_fuseable(const struct ggml_tensor * t) {
    if (!t->extra) return false;
    auto extra = (const struct ggml_hexagon_tensor_extra *) t->extra;
    return (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

static inline bool ggml_hexagon_tensors_overlap(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    const uintptr_t a0 = (uintptr_t) a->data;
    const uintptr_t b0 = (uintptr_t) b->data;
    const uintptr_t a1 = a0 + ggml_nbytes(a);
    const uintptr_t b1 = b0 + ggml_nbytes(b);

    return a0 < b1 && b0 < a1;
}

struct htp_opnode;

struct ggml_hexagon_opbatch;
struct ggml_hexagon_opqueue;
struct ggml_hexagon_shared_buffer;
struct ggml_hexagon_fence_buffer;
struct ggml_hexagon_session;
struct ggml_backend_hexagon_device_context;

struct ggml_hexagon_mdev_group {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

struct ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    volatile uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

struct ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    volatile uint32_t *    fence_slot   = nullptr;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

// ============================================================================ fast op-enqueue (decode-only port, AR-replay code excluded -- opt_ndev's own default of 1 and ggml_backend_hexagon_comm_allreduce_tensor's n_backends<2 early-return mean the allreduce comm hook never even constructs on this single-device build, so the AR-specific plan-capture/replay code the old private fork's version also carried is provably unreachable and was not ported): pre-built subgraph replay plans. A plan captures that trajectory once (the opbatch state at entry plus the exact bufs/tens/ops the slow path appended, in order); replay re-appends the same descriptors without re-running the per-op host work (map churn, htp_opnode vector copies, fusion predicate chains, capacity checks).
// Steady-state decode re-submits the same cached subgraph uid every token (llama_decode's meta graph is identical shape-for-shape token to token, per upstream ggml's own graph->uid/can_reuse mechanism -- unmodified by this port), so the slow enqueue path (per-op try_fuse + fit_op + add_op in ggml_hexagon_session::enqueue_op) re-derives an identical op/buffer/tensor descriptor trajectory on every token even though it was already computed. Any mismatch falls back to the slow path, which produces the identical batch -- so this is a strict performance optimization over unchanged numerics and scheduling semantics: same ops, same order, same fusion, same flush points. A replay is only valid when the live batch is in exactly the recorded entry state (the four counters) AND every entry-tensor reference resolves to the recorded index in the live maps (same lookup order as add_tensor).
struct htp_graph_plan {
    // opbatch state at the point this plan's ops start (recorded slow run).
    uint32_t sig_n_ops  = 0;
    uint32_t sig_n_tens = 0;
    uint32_t sig_n_bufs = 0;
    size_t   sig_b_vmem = 0;

    // Tensors appended by this plan, in add order. htp_tensor.bi/.ti are valid
    // as-is (see above). new_tens_fd[k] identifies the buffer for validation.
    std::vector<htp_tensor>          new_tens;
    std::vector<const ggml_tensor *> new_tens_ptr;
    std::vector<uint32_t>            new_tens_fd;
    // t->data of each new tensor, captured at record time: replay must NOT dereference new_tens_ptr (the ggml objects are freed after their token); the d_map key is this data address (stable: per-session sbuf memory).
    std::vector<uint64_t>            new_tens_data;
    std::vector<htp_buf_desc>        new_bufs;

    // Entry tensors referenced by this plan's ops: indexed by batch tensor index (valid for indices < sig_n_tens); resolved through the live maps at replay time.
    std::vector<const ggml_tensor *> entry_tens;
    // t->data of each entry tensor, captured at record time.
    // Replay validates entry references against this (live h_tens[idx].data + h_bufs base must match) instead of dereferencing entry_tens -- those ggml objects are freed after their token, and their memory is reused by later graphs.
    std::vector<uint64_t> entry_tens_data;
    struct op_entry {
        htp_op_desc desc; // recorded verbatim, post-fusion
        // Per slot (src[0..9] then dst[0..3]): -1 = no tensor 0x4000|k = tensor appended by this plan's replay (k indexes new_tens) 0xC000|k = entry tensor k (verified against the live maps)
        int16_t slot[HTP_OP_MAX_INPUTS + HTP_OP_MAX_OUTPUTS];
    };
    std::vector<op_entry> ops;

    // htp_opnode mirror of the appended ops (op_cache/profiling parity with the
    // slow path; only consumed when GGML_HEXAGON_PROFILE != 0).
    std::vector<htp_opnode> nodes;

    // A flush inside the recorded span moved a batch boundary away from the
    // recorded one; such plans are discarded at record time.
    bool dirty = false;
    // Recorded with the gap-fused residual ADDs of its subgraph skipped (ggml_backend_hexagon_comm_allreduce_add_tensor): replay only
    // under the same skip set (plan->gap_skip == has_gap_skip).
    bool gap_skip = false;
};

// MC05 norm gather (ggml_backend_hexagon_comm_allreduce_add_rms_tensor) deferred to the subgraph that reads its outputs
// (GGML_HEXAGON_GATHER_ACT_TILE). One record per session and boundary.
struct ggml_hexagon_pending_rms_ag {
    uint32_t                         rank            = 0;
    uint32_t                         n_ranks         = 0;
    uint32_t                         fence_seq_entry = 0;
    uint32_t                         fence_seq_exit  = 0;
    float                            epsilon         = 0.0f;
    std::vector<const ggml_tensor *> sync_tensors;
    const ggml_tensor *              residual = nullptr;
    const ggml_tensor *              weight   = nullptr;
    std::vector<const ggml_tensor *> norm_tensors;
};

// AE-03 (GGML_HEXAGON_AR_RMS_FUSE): the owner-row ALLREDUCE+ADD and the RMS_NORM_MUL of one boundary, enqueued as one op by the consuming
// subgraph (ggml_hexagon_enqueue_pending_add_rms). The partials are read at that point, before any node of the consuming subgraph runs.
struct ggml_hexagon_pending_add_rms {
    bool partial_f16 = false;   // fp16 partial rows: the producers wrote fp16 rows (HTP_ALLREDUCE_PARTIAL_F16 on the consumer)
    uint32_t                         rank            = 0;
    uint32_t                         n_ranks         = 0;
    uint32_t                         fence_seq_entry = 0;
    uint32_t                         fence_seq_exit  = 0;
    float                            epsilon         = 0.0f;
    std::vector<const ggml_tensor *> partials;                // the n partial sums of the boundary (all ranks)
    std::vector<const ggml_tensor *> sync_tensors;            // the n fences
    const ggml_tensor *              residual  = nullptr;    // this rank's residual input (the ADD source)
    const ggml_tensor *              resid_out = nullptr;    // this rank's residual output (the ADD node, owner rows)
    const ggml_tensor *              weight    = nullptr;    // RMS_NORM_MUL weight
    std::vector<const ggml_tensor *> norm_tensors;            // the n norm outputs (the MUL nodes)
    std::vector<ggml_hexagon_session *> owners;               // the n sessions owning the norm dsts (tile mode: their norm tile buffers)
};

struct ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    int              phys_idx;
    int              virt_idx;
    bool             valid_session;
    bool             valid_handle;
    bool             valid_queue;
    bool             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<int, std::unique_ptr<ggml_hexagon_shared_buffer>> cloned_buffers;
    std::unordered_set<ggml_hexagon_session *>                           virt_peers;
    std::unordered_set<ggml_hexagon_session *>                           phys_peers;
    // GGML_HEXAGON_DISPATCH_1PASS: the batch prepared by dispatch_prepare(), written by dispatch_issue()
    htp_opbatch_req disp_req     {};
    dspqueue_buffer disp_dbuf    {};
    uint64_t        disp_begin   = 0;
    bool            disp_pending = false;

    uint32_t n_threads   = 0;
    uint32_t n_hvx       = 0;
    uint32_t n_hmx       = 0;
    uint64_t vtcm_size   = 0;
    uint64_t vtcm_size_hw = 0;   // VTCM as allocated; vtcm_size is the budget (VrAr shrinks it for a batch)
    bool                                    vr_a_pinned   = false;   // VrAr A: the residual owner rows of the forward are resident (ggml_hexagon_vr_ar_begin)
    const void *                            vr_a_out_data = nullptr;  // data of the last resident residual output (the next boundary reads it)
    uint32_t                                vr_a_res_off  = 0;        // VTCM offset of the resident rows (the budget while pinned)
    size_t   max_vmem    = 0;
    uint32_t fence_seq   = 0;

    std::atomic<uint64_t> batch_req_seq{0};
    std::atomic<uint64_t> batch_rsp_seq{0};
    std::atomic<uint32_t> last_error{HTP_STATUS_OK};

    struct htp_cached_graph {
        uint64_t                        uid = 0;
        std::vector<htp_opnode>         nodes;
        std::unique_ptr<htp_graph_plan> plan;
    };
    static constexpr size_t HTP_GRAPH_CACHE_MAX = 512;
    std::vector<htp_cached_graph> cached_graphs;

    // Per-node kernel_params memoization, keyed by ggml_hexagon_node_hash (pure function of op/type/shape/stride/op_params/weight-usage, never of tensor data pointers). alternating target-verify / MTP-draft decode steps. This cache lets the ggml_hexagon_precompute_*_params calls below be skipped on a hit regardless of uid churn, without touching any raw tensor pointer: the cached blob is plain POD (int32_t[HTP_OP_MAX_KERN_PARAMS]), and the htp_opnode carrying the live ggml_tensor* is still built fresh from the CURRENT graph on every single call.
    // uid-based cached_graphs above almost never hits in steady-state decode because ggml_graph_next_uid() assigns a fresh global id on every split_graph() call even when the exact same graph topology (and therefore the exact same kernel_params) recurs -- e.g.
    struct htp_kparams_entry {
        uint64_t hash = 0;
        int32_t  kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};
    };
    static constexpr size_t HTP_KPARAMS_CACHE_MAX = 256;
    std::vector<htp_kparams_entry> kparams_cache;
    std::atomic<uint64_t> kparams_hits   { 0 };
    std::atomic<uint64_t> kparams_misses { 0 };

    // A replay ever failing after pre-validation disables the fast path for
    // the rest of this session's lifetime (the slow path remains correct).
    std::atomic<uint64_t> n_replay          { 0 };
    bool                  fast_enq_disabled = false;
    // LLAMA_CYCLE_TIMING>=2 host split of one pass (hostovh diagnostics): graph_compute time, first compute, last sync end
    uint64_t ov_enq_us       = 0;
    uint32_t ov_n_compute    = 0;
    uint64_t ov_first_us     = 0;
    uint64_t ov_last_sync_us = 0;
    // Residual ADDs executed by the gap-fused comm hook for the next subgraph; graph_compute drops them and clears this list.
    std::vector<const ggml_tensor *> ar_gap_skip;

    mutable std::unordered_set<const ggml_tensor *> needs_repack;

    ggml_hexagon_mdev_group                mdev;
    ggml_backend_dev_t                     dev       = nullptr;
    ggml_backend_hexagon_device_context *  dev_ctx   = nullptr;
    ggml_hexagon_fence_buffer *            fence_buf = nullptr;

    ggml_hexagon_session(const ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_count = 0) noexcept(false);
    ~ggml_hexagon_session() noexcept(true);

    const char* c_name() const { return name.c_str(); }

    void allocate(const ggml_hexagon_device_config & config) noexcept(false);
    void release() noexcept(true);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    void      free_fence(void * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, volatile uint32_t *> cpy_fence_slots;

    // fp16 HMX activation tiles (GGML_HEXAGON_ACT_TILE, layout in htp/hmx-act-tiles.h). act_tile_ok lists the producers the plan admits for the graph being enqueued; act_tile_live maps a fused producer's fp32 output to its view.
    // act_tile_buf is the scratch of RMS_NORM+MUL producers; consumers read per-(K, M) views of it (act_tile_views).
    ggml_backend_buffer_t                                        act_tile_buf = nullptr;
    ggml_hexagon_tensor_extra                                    act_tile_extra;
    std::unordered_map<uint64_t, std::unique_ptr<ggml_tensor>>   act_tile_views;
    std::unordered_set<const ggml_tensor *>                      act_tile_ok;
    std::unordered_set<const ggml_tensor *>                      act_tile_skip_ok;    // GGML_HEXAGON_ACT_TILE=2: producers whose fp32 output may be dropped
    std::unordered_set<const ggml_tensor *>                      act_tile_skip_live;  // dropped-fp32 producers fused into the graph being enqueued
    std::unordered_map<const ggml_tensor *, const ggml_tensor *> act_tile_live;
    // FFN SWIGLU output tiles (GGML_HEXAGON_SWIGLU_OUT, layout htp/hmx-act-tiles.h): written by the gate/up NX, read by the FFN
    // down through act_tile_live. swiglu_ok lists the GLU outputs the plan admits for the graph being enqueued.
    ggml_backend_buffer_t                                        swiglu_buf = nullptr;
    ggml_hexagon_tensor_extra                                    swiglu_extra;
    std::unordered_map<uint64_t, std::unique_ptr<ggml_tensor>>   swiglu_views;
    std::unordered_set<const ggml_tensor *>                      swiglu_ok;
    std::unordered_set<const ggml_tensor *>                      norm_rope_only;   // RMS_NORM+MUL outputs whose only reader is a ROPE (ROPE_NORM)
    std::unordered_set<const ggml_tensor *>                      gatetile_ok;   // attention gate MUL outputs whose only readers are HMX 2D matmuls (GGML_HEXAGON_GATE_TILE)
    std::unordered_map<const ggml_tensor *, std::vector<const ggml_tensor *>> ngtile_ok;   // gated-norm GLU outputs the plan admits for tile output -> the reshaped views the ssm_out matmul reads
    // MC05 norm gathers deferred from the comm hook to the subgraph that reads their outputs (GGML_HEXAGON_GATHER_ACT_TILE).
    std::vector<ggml_hexagon_pending_rms_ag>                       pending_rms_ag;
    // AE-03: fused owner-row ALLREDUCE+ADD+RMS_NORM+MUL records (GGML_HEXAGON_AR_RMS_FUSE), enqueued by the consuming subgraph.
    std::vector<ggml_hexagon_pending_add_rms>                      pending_add_rms;
    // fp16 tile views of MC05 norm outputs (GGML_HEXAGON_GATHER_ACT_TILE), keyed by the output's data pointer.
    ggml_hexagon_tensor_extra                                      gather_tile_extra;
    std::unordered_map<const void *, std::unique_ptr<ggml_tensor>> gather_views;
    const ggml_tensor * gather_tile_view(const ggml_tensor * P);
    // AE-03 fp16 norm output tiles (GGML_HEXAGON_AR_RMS_FUSE): one buffer per session, holding the tiles of the norm outputs it owns. The
    // fused op writes every rank's tiles here, never into a partial sum (ggml_hexagon_enqueue_pending_add_rms). Views are keyed by (K, M).
    ggml_backend_buffer_t                                        norm_tile_buf = nullptr;
    ggml_hexagon_tensor_extra                                    norm_tile_extra;
    std::unordered_map<uint64_t, std::unique_ptr<ggml_tensor>>   norm_tile_views;
    const ggml_tensor * norm_tile_view(int64_t K, int64_t M);
    // ELT_SMALL bits of the graph being computed (ggml_hexagon_elt_small_graph_bits): 0 below the batch gate.
    int elt_small = 0;
    // AE-16 zero state (GGML_HEXAGON_ELT_SMALL bit 4, ggml_hexagon_zero_state_plan): per graph, the clears to skip and the GDN
    // nodes that read zeros; zero_state_src is the GDN state source (zero_state_buf with the state's type and strides).
    std::unordered_set<const ggml_tensor *>                      zero_state_clear_skip;
    std::unordered_set<const ggml_tensor *>                      zero_state_gdn;
    // the state GET_ROWS that only feeds a zero_state_gdn (its reshape has no other reader): not enqueued
    std::unordered_set<const ggml_tensor *>                      zero_state_dead;
    // the CONCATs of conv chains whose conv state is a fresh zero state: the fused chain reads zeros (ggml_hexagon_zero_state_plan_conv)
    std::unordered_set<const ggml_tensor *>                      zero_state_conv;
    ggml_backend_buffer_t                                        zero_state_buf = nullptr;
    ggml_hexagon_tensor_extra                                    zero_state_extra;
    ggml_tensor                                                  zero_state_src {};
    ggml_tensor                                                  zero_state_conv_src {};   // separate descriptor: the batch keys tensors by pointer and the conv state has another shape

    void enqueue_mdev_group();
    void enqueue_op(const htp_opnode & node);
    void enqueue_cpy(const ggml_tensor * src, ggml_tensor * dst, const ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    void enqueue_fence(const ggml_tensor * sync_tensor, uint32_t fence_seq = 0, bool wait = true);
    void enqueue_allreduce(const ggml_tensor * dst, const std::vector<const ggml_tensor *> & src_tensors,
                           const std::vector<const ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);
    void enqueue_allreduce_add(const std::vector<const ggml_tensor *> & src_tensors,
                               const std::vector<const ggml_tensor *> & sync_tensors, const ggml_tensor * residual,
                               const std::vector<const ggml_tensor *> & dst_tensors, uint32_t rank, uint32_t n_ranks,
                               uint32_t fence_seq_entry, uint32_t fence_seq_exit, bool chunked, bool owner_only = false);
    void enqueue_allreduce_rms_ag(const std::vector<const ggml_tensor *> & sync_tensors, const ggml_tensor * residual,
                                  const ggml_tensor * weight, const std::vector<const ggml_tensor *> & norm_tensors, uint32_t rank,
                                  uint32_t n_ranks, uint32_t fence_seq_entry, uint32_t fence_seq_exit, float epsilon, bool tile);
    void enqueue_allreduce_add_rms(const std::vector<const ggml_tensor *> & partials, const std::vector<const ggml_tensor *> & sync_tensors,
                                   const ggml_tensor * residual, const ggml_tensor * weight, const ggml_tensor * resid_out,
                                   const std::vector<const ggml_tensor *> & norm_tensors, uint32_t rank, uint32_t n_ranks,
                                   uint32_t fence_seq_entry, uint32_t fence_seq_exit, float epsilon, bool tile,
                                   uint32_t resid_mode = 0, uint32_t resid_off = 0);

    // fast op-enqueue (see htp_graph_plan): plan_begin -- capture the entry opbatch state before a slow subgraph run plan_finish -- build the subgraph plan from the final opbatch state after a slow run; false = plan discarded (nothing to replay, or dirty) plan_replay -- validate + append a subgraph plan to the live batch (fast path)
    void plan_begin(htp_graph_plan & plan);
    bool plan_finish(htp_graph_plan & plan);
    bool plan_replay(const htp_graph_plan & plan);

    // kparams_cache lookup/store (see htp_kparams_entry above).
    // Returns true and fills `out` on a hit; a linear scan is fine since this only runs on the already-slow op-descriptor-build path it's meant to shortcut, and HTP_KPARAMS_CACHE_MAX bounds it to a small, fixed cost.
    bool kparams_lookup(uint64_t hash, int32_t out[HTP_OP_MAX_KERN_PARAMS]) {
        for (const auto & e : kparams_cache) {
            if (e.hash == hash) {
                memcpy(out, e.kernel_params, sizeof(e.kernel_params));
                kparams_hits++;
                kparams_log_periodic();
                return true;
            }
        }
        kparams_misses++;
        kparams_log_periodic();
        return false;
    }
    void kparams_log_periodic() {
        const uint64_t total = kparams_hits.load() + kparams_misses.load();
        if (opt_verbose && total > 0 && (total % 500) == 0) {
            GGML_LOG_INFO("ggml-hex: %s kparams-cache: hits %llu misses %llu hit-rate %.1f%%\n",
                           this->name.c_str(), (unsigned long long) kparams_hits.load(),
                           (unsigned long long) kparams_misses.load(),
                           100.0 * (double) kparams_hits.load() / (double) total);
        }
    }
    void kparams_store(uint64_t hash, const int32_t in[HTP_OP_MAX_KERN_PARAMS]) {
        if (kparams_cache.size() >= HTP_KPARAMS_CACHE_MAX) {
            kparams_cache.erase(kparams_cache.begin());
        }
        htp_kparams_entry e;
        e.hash = hash;
        memcpy(e.kernel_params, in, sizeof(e.kernel_params));
        kparams_cache.push_back(e);
    }

    void start_batch();
    void flush_sync(bool all = true);
    void flush_async();
    void flush_batch(size_t min_ops = 1);
    void flush_peers();
    void flush_pending(bool all = true);
    void flush_async_1pass();
    void dispatch_collect(std::vector<ggml_hexagon_session *> & out);
    void collect_peers_1pass(std::vector<ggml_hexagon_session *> & out);
    void dispatch_prepare(size_t min_ops);
    void dispatch_issue();
    void dispatch_write(htp_opbatch_req & req, dspqueue_buffer & dbuf, uint64_t t_begin);

    ggml_hexagon_shared_buffer * mmap_tensor(const ggml_tensor * t);
    bool clone_buffer(const ggml_hexagon_shared_buffer*);
    void release_buffer(const ggml_hexagon_shared_buffer*);
    void unclone_buffer(const ggml_hexagon_shared_buffer*);

    void add_peer(ggml_hexagon_session * peer) {
        if (this->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } else {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

struct ggml_backend_hexagon_device_context {
    int                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_buffer_type buffer_type       = {};
    ggml_backend_buffer_type host_buffer_type  = {};
    ggml_backend_buffer_type fence_buffer_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(int dev_id, const ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    const char * c_name() const { return config.name.c_str(); }

    ggml_hexagon_session * session() {
        if (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        return sess.get();
    }
};

struct ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(const std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        this->dev_ctx = dev_ctx;
        this->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

struct ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    int       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        if (!base) {
            throw std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        if (fd < 0) {
            rpcmem_free(base);
            throw std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        this->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        if (base) {
            rpcmem_free(base);
        }
    }
};

struct ggml_hexagon_shared_buffer {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    bool     mapped;
    bool     pinned;
    bool     extended;

    const char * c_name() const { return sess->c_name(); }
    uint8_t *    base()   const { return mem ? mem->base : nullptr; }
    size_t       size()   const { return mem ? mem->size : 0;  }
    int          fd()     const { return mem ? mem->fd   : -1; }

    void mmap(bool extended = false) {
        if (!this->mem)   return;
        if (this->mapped) return;

        GGML_ASSERT(!this->pinned || !extended);

        this->extended = extended;
        fastrpc_map_flags flags = this->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        int err = fastrpc_mmap(sess->domain_id, fd(), (void *) base(), 0, size(), flags);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (unsigned) err);
            throw std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (void *) base(), size(), fd(), pinned, extended);

        this->mapped = true;
    }

    void unmap() {
        if (!this->mapped) return;

        if (!this->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        if (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (void *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (void *) base(), size(), fd());

        this->mapped = false;
    }

    void alloc(size_t size) {
        if (this->mem) return;

        this->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (void *) base(), this->size(), fd(), (int) pinned);
        if (this->pinned) {
            mmap();
        }
    }

    void free() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (void *) base(), size(), fd());
        this->mem  = nullptr;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, bool pinned = false) {
        this->sess     = sess;
        this->mapped   = false;
        this->pinned   = pinned;
        this->extended = false;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_offset = (size + 4095) & ~4095;
        size_t total_size   = guard_offset + 4096;
        if (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            total_size = (total_size + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(total_size);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, const ggml_hexagon_shared_buffer & other) {
        this->sess   = sess;
        this->mem    = other.mem;
        this->mapped = false;
        this->pinned = other.pinned;
        this->extended = other.extended;
    }

    ~ggml_hexagon_shared_buffer() {
        free();
        for (auto * extra : tensor_extra) {
            delete extra;
        }
    }
};

struct ggml_hexagon_fence_buffer : public ggml_hexagon_shared_buffer {
    uint32_t              slot_count = 0;
    uint32_t              slot_head  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_buffer   backend_buffer{};

    ggml_hexagon_fence_buffer(ggml_hexagon_session * sess, ggml_backend_buffer_type_t buft, size_t size)
        : ggml_hexagon_shared_buffer(sess, size, false /* pinned */),
          slot_count(size / GGML_HEXAGON_FENCE_SLOT_SIZE),
          slot_head(0) {
        backend_buffer.buft    = buft;
        backend_buffer.context = static_cast<ggml_hexagon_shared_buffer *>(this);
        backend_buffer.size    = size;
        mmap(false);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        if (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        } else if (slot_head + n_slots <= slot_count) {
            uint32_t slot = slot_head;
            slot_head += n_slots;
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        }
        if (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_SLOT_SIZE);
        }
        return ptr;
    }

    void free_slot(void * ptr, uint32_t n_slots = 1) {
        if (!ptr) return;
        uint32_t slot = ((uint8_t *) ptr - base()) / GGML_HEXAGON_FENCE_SLOT_SIZE;
        for (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

inline uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    return ptr;
}

inline void ggml_hexagon_session::free_fence(void * ptr, uint32_t n_slots) {
    if (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

static void ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    sbuf->sess->unclone_buffer(sbuf);
    delete sbuf;
}

static void * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    return sbuf->base();
}

static enum ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (void *) sbuf->base(), tensor->data, ggml_nbytes(tensor));

    auto extra = new ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;
    if (ggml_hexagon_is_repack_type(tensor->type)) {
        if (sess->needs_repack.count(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_REPACK;
            sess->needs_repack.erase(tensor);
        }
    }

    return GGML_STATUS_SUCCESS;
}

// ** Repack helpers for tiled quantized weights

static void unpack_q4_0_quants(uint8_t * qs, const block_q4_0 * x, unsigned int bi) {
    static const int qk = QK4_0;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const int x0             = (x->qs[i] & 0x0F);
        const int x1             = (x->qs[i] >> 4);
        qs[bi * qk + i + 0]      = x0;
        qs[bi * qk + i + qk / 2] = x1;
    }
}

static void pack_q4_0_quants(block_q4_0 * x, const uint8_t * qs, unsigned int bi) {
    static const int qk = QK4_0;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const uint8_t x0 = qs[bi * qk + i + 0];
        const uint8_t x1 = qs[bi * qk + i + qk / 2];
        x->qs[i]         = x0 | (x1 << 4);
    }
}

static void unpack_q4_1_quants(uint8_t * qs, const block_q4_1 * x, unsigned int bi) {
    static const int qk = QK4_1;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const int x0             = (x->qs[i] & 0x0F);
        const int x1             = (x->qs[i] >> 4);
        qs[bi * qk + i + 0]      = x0;
        qs[bi * qk + i + qk / 2] = x1;
    }
}

static void pack_q4_1_quants(block_q4_1 * x, const uint8_t * qs, unsigned int bi) {
    static const int qk = QK4_1;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const uint8_t x0 = qs[bi * qk + i + 0];
        const uint8_t x1 = qs[bi * qk + i + qk / 2];
        x->qs[i]         = x0 | (x1 << 4);
    }
}

static void unpack_mxfp4_quants(uint8_t * qs, const block_mxfp4 * x, unsigned int bi) {
    static const int qk = QK_MXFP4;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const int x0             = (x->qs[i] & 0x0F);
        const int x1             = (x->qs[i] >> 4);
        qs[bi * qk + i + 0]      = x0;
        qs[bi * qk + i + qk / 2] = x1;
    }
}

static void pack_mxfp4_quants(block_mxfp4 * x, const uint8_t * qs, unsigned int bi) {
    static const int qk = QK_MXFP4;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const uint8_t x0 = qs[bi * qk + i + 0];
        const uint8_t x1 = qs[bi * qk + i + qk / 2];
        x->qs[i]         = x0 | (x1 << 4);
    }
}

// repack q4_0 data into q4_0_tiled tensor
static void repack_q4_0_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    const block_q4_0 * src_matrix = (const block_q4_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    size_t slice_size = ne1 * ggml_row_size(t->type, ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const block_q4_0 * src_slice = src_matrix + (slice_idx - start_slice) * (ne1 * (ne0 / 32));
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = 0; ct < n_col_tiles; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                uint8_t tile_quants[32][32];
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r < ne1 && kt < ne0 / 32) {
                        unpack_q4_0_quants(tile_quants[row], &src_slice[r * (ne0 / 32) + kt], 0);
                    } else {
                        memset(tile_quants[row], 8, 32);
                    }
                }

                for (int cp = 0; cp < 16; cp++) {
                    for (int row = 0; row < 32; row++) {
                        tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                    }
                }

                ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_slice[r * (ne0 / 32) + kt].d : 0;
                }
            }
        }
    }
}


static inline float a8w8_h(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }   // round to fp16 and back

// scalar mirror of htp/hmx-a8w8.h a8w8_expand_column_tile for one column tile; tiles = the tiled Q4_0 tiles of this column tile (n_k tiles, 576 B)
static void a8w8_pre_expand_ct(uint8_t * dst /* n_k*1024 + 128 + 128 */, const uint8_t * tiles, int n_k) {
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    float amax[32];
    for (int n = 0; n < 32; n++) {
        float m = 0.f;
        for (int kt = 0; kt < n_k; kt++) {
            const float d = fabsf(ggml_fp16_to_fp32(((const ggml_half *) (tiles + kt * tile_size + 512))[n]));
            m = d > m ? d : m;
        }
        amax[n] = m;
    }
    float *   sw = (float *)   (dst + (size_t) n_k * 1024);
    int32_t * cs = (int32_t *) (dst + (size_t) n_k * 1024 + 128);
    float inv[32];
    for (int n = 0; n < 32; n++) {
        const float a = amax[n];
        sw[n]  = a > 0.f ? a * 8.f / 127.f : 1.f;
        inv[n] = a > 0.f ? a8w8_h(127.f / a / 256.f) : 0.f;
    }
    int64_t csum[32] = { 0 };
    for (int kt = 0; kt < n_k; kt++) {
        const uint8_t * t = tiles + kt * tile_size;
        uint8_t * w8 = dst + (size_t) kt * 1024;
        for (int n = 0; n < 32; n++) {
            const float d = ggml_fp16_to_fp32(((const ggml_half *) (t + 512))[n]);
            float y = a8w8_h(d * inv[n]);
            y = a8w8_h(y * 256.f);
            y = a8w8_h(y + (y < 0.f || (y == 0.f && std::signbit(y)) ? -0.5f : 0.5f));
            int C = (int) y;   // truncation toward zero
            C = C > 127 ? 127 : (C < -128 ? -128 : C);
            for (int g = 0; g < 8; g++) {
                for (int j = 0; j < 4; j++) {
                    const int k   = 4 * g + j;
                    const int cp  = k >> 1;
                    const int nib = (k & 1) ? (t[cp * 32 + n] >> 4) : (t[cp * 32 + n] & 0x0F);
                    int p = (nib - 8) * C;
                    p = (p + 4) >> 3;
                    p = p > 127 ? 127 : (p < -128 ? -128 : p);
                    w8[g * 128 + n * 4 + j] = (uint8_t) (int8_t) p;
                    csum[n] += p;
                }
            }
        }
    }
    for (int n = 0; n < 32; n++) { cs[n] = (int32_t) (csum[n] * 128); }
}

static void a8w8_pre_build(ggml_tensor * t) {
    const int n_k  = (int) (hex_round_up(t->ne[0], 32) / 32);
    const int n_ct = (int) (hex_round_up(t->ne[1], 32) / 32);
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    uint8_t * q4   = (uint8_t *) t->data;
    uint8_t * blob = q4 + ggml_hexagon_a8w8_pre_blob_off(t);
    const size_t ct_bytes = (size_t) n_k * 1024 + HTP_A8W8_PE_TAIL;
    const int nth = (int) std::min<unsigned>(std::max(1u, std::thread::hardware_concurrency()), 16u);
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < nth; i++) {
        th.emplace_back([&]() {
            for (int ct; (ct = next.fetch_add(1)) < n_ct; ) {
                a8w8_pre_expand_ct(blob + (size_t) ct * ct_bytes, q4 + (size_t) ct * n_k * tile_size, n_k);
            }
        });
    }
    for (auto & x : th) { x.join(); }
}

// repack q4_0_tiled tensor into q4_0 data
static void repack_tiled_q4_0(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    block_q4_0 * dst_matrix = (block_q4_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    size_t slice_size = ne1 * ggml_row_size(t->type, ne0);
    size_t row_size_bytes = ggml_row_size(t->type, ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        size_t cur_start_byte = (std::max)(offset, (size_t) slice_idx * slice_size);
        size_t cur_end_byte   = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_size);
        size_t slice_offset_start = cur_start_byte - (size_t) slice_idx * slice_size;
        size_t slice_offset_end   = cur_end_byte - (size_t) slice_idx * slice_size;

        int64_t start_row = slice_offset_start / row_size_bytes;
        int64_t end_row   = (slice_offset_end + row_size_bytes - 1) / row_size_bytes;
        end_row = (std::min)(end_row, ne1);

        int start_ct = start_row / 32;
        int end_ct   = (end_row + 31) / 32;
        end_ct = (std::min)(end_ct, n_col_tiles);

        block_q4_0 * dst_slice = dst_matrix + (cur_start_byte - offset) / sizeof(block_q4_0);
        const uint8_t * matrix_src = (const uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = start_ct; ct < end_ct; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                uint8_t tile_quants[32][32];
                for (int cp = 0; cp < 16; cp++) {
                    for (int row = 0; row < 32; row++) {
                        uint8_t val = tile_src[cp * 32 + row];
                        tile_quants[row][2 * cp + 0] = val & 0x0F;
                        tile_quants[row][2 * cp + 1] = val >> 4;
                    }
                }

                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / 32) {
                        pack_q4_0_quants(&dst_slice[(r - start_row) * (ne0 / 32) + kt], tile_quants[row], 0);
                    }
                }

                const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / 32) {
                        dst_slice[(r - start_row) * (ne0 / 32) + kt].d = scale_src[row];
                    }
                }
            }
        }
    }
}

// repack q4_1 data into q4_1_tiled tensor
static void repack_q4_1_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    const block_q4_1 * src_matrix = (const block_q4_1 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    size_t slice_size = ne1 * ggml_row_size(t->type, ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const block_q4_1 * src_slice = src_matrix + (slice_idx - start_slice) * (ne1 * (ne0 / 32));
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = 0; ct < n_col_tiles; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                uint8_t tile_quants[32][32];
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r < ne1 && kt < ne0 / 32) {
                        unpack_q4_1_quants(tile_quants[row], &src_slice[r * (ne0 / 32) + kt], 0);
                    } else {
                        memset(tile_quants[row], 0, 32);
                    }
                }

                for (int cp = 0; cp < 16; cp++) {
                    for (int row = 0; row < 32; row++) {
                        tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                    }
                }

                ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r < ne1 && kt < ne0 / 32) {
                        scale_dst[2 * row + 0] = src_slice[r * (ne0 / 32) + kt].d;
                        scale_dst[2 * row + 1] = src_slice[r * (ne0 / 32) + kt].m;
                    } else {
                        scale_dst[2 * row + 0] = 0;
                        scale_dst[2 * row + 1] = 0;
                    }
                }
            }
        }
    }
}

// repack q4_1_tiled tensor into q4_1 data
static void repack_tiled_q4_1(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    block_q4_1 * dst_matrix = (block_q4_1 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    size_t slice_size = ne1 * ggml_row_size(t->type, ne0);
    size_t row_size_bytes = ggml_row_size(t->type, ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        size_t cur_start_byte = (std::max)(offset, (size_t) slice_idx * slice_size);
        size_t cur_end_byte   = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_size);
        size_t slice_offset_start = cur_start_byte - (size_t) slice_idx * slice_size;
        size_t slice_offset_end   = cur_end_byte - (size_t) slice_idx * slice_size;

        int64_t start_row = slice_offset_start / row_size_bytes;
        int64_t end_row   = (slice_offset_end + row_size_bytes - 1) / row_size_bytes;
        end_row = (std::min)(end_row, ne1);

        int start_ct = start_row / 32;
        int end_ct   = (end_row + 31) / 32;
        end_ct = (std::min)(end_ct, n_col_tiles);

        block_q4_1 * dst_slice = dst_matrix + (cur_start_byte - offset) / sizeof(block_q4_1);
        const uint8_t * matrix_src = (const uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = start_ct; ct < end_ct; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                uint8_t tile_quants[32][32];
                for (int cp = 0; cp < 16; cp++) {
                    for (int row = 0; row < 32; row++) {
                        uint8_t val = tile_src[cp * 32 + row];
                        tile_quants[row][2 * cp + 0] = val & 0x0F;
                        tile_quants[row][2 * cp + 1] = val >> 4;
                    }
                }

                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / 32) {
                        pack_q4_1_quants(&dst_slice[(r - start_row) * (ne0 / 32) + kt], tile_quants[row], 0);
                    }
                }

                const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / 32) {
                        dst_slice[(r - start_row) * (ne0 / 32) + kt].d = scale_src[2 * row];
                        dst_slice[(r - start_row) * (ne0 / 32) + kt].m = scale_src[2 * row + 1];
                    }
                }
            }
        }
    }
}

// repack q8_0 data into q8_0_tiled tensor
static void repack_q8_0_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    const block_q8_0 * src_matrix = (const block_q8_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    size_t slice_size = ne1 * ggml_row_size(t->type, ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const block_q8_0 * src_slice = src_matrix + (slice_idx - start_slice) * (ne1 * (ne0 / 32));
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = 0; ct < n_col_tiles; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                for (int cp = 0; cp < 16; cp++) {
                    int col0 = cp * 2;
                    int col1 = col0 + 1;
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        const block_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_slice[r * (ne0 / 32) + kt] : NULL;
                        tile_dst[cp * 64 + 2 * row + 0] = b ? b->qs[col0] : 0;
                        tile_dst[cp * 64 + 2 * row + 1] = b ? b->qs[col1] : 0;
                    }
                }

                ggml_half * scale_dst = (ggml_half *)(tile_dst + 1024);
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_slice[r * (ne0 / 32) + kt].d : 0;
                }
            }
        }
    }
}

// repack q8_0_tiled tensor into q8_0 data
static void repack_tiled_q8_0(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    block_q8_0 * dst_matrix = (block_q8_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    size_t slice_size = ne1 * ggml_row_size(t->type, ne0);
    size_t row_size_bytes = ggml_row_size(t->type, ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        size_t cur_start_byte = (std::max)(offset, (size_t) slice_idx * slice_size);
        size_t cur_end_byte   = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_size);
        size_t slice_offset_start = cur_start_byte - (size_t) slice_idx * slice_size;
        size_t slice_offset_end   = cur_end_byte - (size_t) slice_idx * slice_size;

        int64_t start_row = slice_offset_start / row_size_bytes;
        int64_t end_row   = (slice_offset_end + row_size_bytes - 1) / row_size_bytes;
        end_row = (std::min)(end_row, ne1);

        int start_ct = start_row / 32;
        int end_ct   = (end_row + 31) / 32;
        end_ct = (std::min)(end_ct, n_col_tiles);

        block_q8_0 * dst_slice = dst_matrix + (cur_start_byte - offset) / sizeof(block_q8_0);
        const uint8_t * matrix_src = (const uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = start_ct; ct < end_ct; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                for (int cp = 0; cp < 16; cp++) {
                    int col0 = cp * 2;
                    int col1 = col0 + 1;
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r >= start_row && r < end_row && kt < ne0 / 32) {
                            block_q8_0 & b = dst_slice[(r - start_row) * (ne0 / 32) + kt];
                            b.qs[col0] = tile_src[cp * 64 + 2 * row + 0];
                            b.qs[col1] = tile_src[cp * 64 + 2 * row + 1];
                        }
                    }
                }

                const ggml_half * scale_src = (const ggml_half *)(tile_src + 1024);
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / 32) {
                        dst_slice[(r - start_row) * (ne0 / 32) + kt].d = scale_src[row];
                    }
                }
            }
        }
    }
}

// repack mxfp4 data into mxfp4_tiled tensor
static void repack_mxfp4_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    const block_mxfp4 * src_matrix = (const block_mxfp4 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_MXFP4;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    size_t slice_size = ne1 * ggml_row_size(t->type, ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const block_mxfp4 * src_slice = src_matrix + (slice_idx - start_slice) * (ne1 * (ne0 / 32));
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = 0; ct < n_col_tiles; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                uint8_t tile_quants[32][32];
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r < ne1 && kt < ne0 / 32) {
                        unpack_mxfp4_quants(tile_quants[row], &src_slice[r * (ne0 / 32) + kt], 0);
                    } else {
                        memset(tile_quants[row], 0, 32);
                    }
                }

                for (int cp = 0; cp < 16; cp++) {
                    for (int row = 0; row < 32; row++) {
                        tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                    }
                }

                uint8_t * scale_dst = tile_dst + 512;
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_slice[r * (ne0 / 32) + kt].e : 0;
                }
            }
        }
    }
}

// repack mxfp4_tiled tensor into mxfp4 data
static void repack_tiled_mxfp4(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    block_mxfp4 * dst_matrix = (block_mxfp4 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_MXFP4;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    size_t slice_size = ne1 * ggml_row_size(t->type, ne0);
    size_t row_size_bytes = ggml_row_size(t->type, ne0);
    int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        size_t cur_start_byte = (std::max)(offset, (size_t) slice_idx * slice_size);
        size_t cur_end_byte   = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_size);
        size_t slice_offset_start = cur_start_byte - (size_t) slice_idx * slice_size;
        size_t slice_offset_end   = cur_end_byte - (size_t) slice_idx * slice_size;

        int64_t start_row = slice_offset_start / row_size_bytes;
        int64_t end_row   = (slice_offset_end + row_size_bytes - 1) / row_size_bytes;
        end_row = (std::min)(end_row, ne1);

        int start_ct = start_row / 32;
        int end_ct   = (end_row + 31) / 32;
        end_ct = (std::min)(end_ct, n_col_tiles);

        block_mxfp4 * dst_slice = dst_matrix + (cur_start_byte - offset) / sizeof(block_mxfp4);
        const uint8_t * matrix_src = (const uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = start_ct; ct < end_ct; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                uint8_t tile_quants[32][32];
                for (int cp = 0; cp < 16; cp++) {
                    for (int row = 0; row < 32; row++) {
                        uint8_t val = tile_src[cp * 32 + row];
                        tile_quants[row][2 * cp + 0] = val & 0x0F;
                        tile_quants[row][2 * cp + 1] = val >> 4;
                    }
                }

                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / 32) {
                        pack_mxfp4_quants(&dst_slice[(r - start_row) * (ne0 / 32) + kt], tile_quants[row], 0);
                    }
                }

                const uint8_t * scale_src = tile_src + 512;
                for (int row = 0; row < 32; row++) {
                    int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / 32) {
                        dst_slice[(r - start_row) * (ne0 / 32) + kt].e = scale_src[row];
                    }
                }
            }
        }
    }
}

// unsigned 6-bit value (0..63) of element e of a Q6_K block, same bit layout as dequantize_row_q6_K
static inline uint8_t q6_K_get_quant(const block_q6_K * b, int e) {
    const int c = e / 128;
    const int w = e % 128;
    const int g = w / 32;
    const int l = w % 32;
    const uint8_t * ql = b->ql + c * 64;
    const uint8_t * qh = b->qh + c * 32;
    uint8_t lo, hi;
    switch (g) {
        case 0:  lo = ql[l]      & 0xF; hi = (qh[l] >> 0) & 3; break;
        case 1:  lo = ql[l + 32] & 0xF; hi = (qh[l] >> 2) & 3; break;
        case 2:  lo = ql[l]      >> 4;  hi = (qh[l] >> 4) & 3; break;
        default: lo = ql[l + 32] >> 4;  hi = (qh[l] >> 6) & 3; break;
    }
    return (uint8_t) (lo | (hi << 4));
}

// tile layout: see HTP_MM_WEIGHT_TILE_SIZE_Q6_K in htp/matmul-ops.h
static void repack_q6_K_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);

    const block_q6_K * src_matrix = (const block_q6_K *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    GGML_ASSERT(ne0 % QK_K == 0);

    const int n_col_tiles = ne1_padded / 32;
    const int n_k_tiles   = ne0_padded / 32;
    const size_t tile_size   = HTP_MM_WEIGHT_TILE_SIZE_Q6_K;
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;

    const int64_t sb_per_row = ne0 / QK_K;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q6_K * src_slice = src_matrix + (i3 * ne2 + i2) * (ne1 * sb_per_row);
            uint8_t * matrix_dst = (uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            memset(matrix_dst, 0, matrix_size);  // padding rows and the OR-ed nibbles below need zeroed tiles

            for (int64_t r = 0; r < ne1; r++) {
                const int ct  = (int) (r / 32);
                const int row = (int) (r % 32);
                const block_q6_K * src_row = src_slice + r * sb_per_row;

                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const int kt_local = kt % 8;  // k-tile within the super-block
                    const block_q6_K * b = &src_row[kt / 8];
                    const float d = GGML_FP16_TO_FP32(b->d);

                    uint8_t * tile = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_size;
                    uint8_t * lo_pl = tile;
                    uint8_t * hi_pl = tile + 512;
                    ggml_half * sc_pl = (ggml_half *) (tile + 768);

                    for (int lk = 0; lk < 32; lk++) {
                        const uint8_t q6 = q6_K_get_quant(b, kt_local * 32 + lk);
                        const int g   = lk >> 2;
                        const int pos = row * 4 + (lk & 3);
                        lo_pl[(g >> 1) * 128 + pos] |= (uint8_t) ((q6 & 0xF) << ((g & 1) * 4));
                        hi_pl[(g >> 2) * 128 + pos] |= (uint8_t) ((q6 >> 4) << ((g & 3) * 2));
                    }
                    for (int sub = 0; sub < 2; sub++) {
                        sc_pl[sub * 32 + row] = GGML_FP32_TO_FP16(d * (float) b->scales[kt_local * 2 + sub]);
                    }
                }
            }
        }
    }

    GGML_UNUSED(size);
}

// Reverse of repack_q6_K_tiled. Unpacks quants losslessly and normalizes sub-block scales. Read-back only.
static void repack_tiled_q6_K(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);

    block_q6_K * dst_matrix = (block_q6_K *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    GGML_ASSERT(ne0 % QK_K == 0);

    const int n_col_tiles = ne1_padded / 32;
    const int n_k_tiles   = ne0_padded / 32;
    const size_t tile_size   = HTP_MM_WEIGHT_TILE_SIZE_Q6_K;
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;

    const int64_t sb_per_row = ne0 / QK_K;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q6_K *    dst_slice  = dst_matrix + (i3 * ne2 + i2) * (ne1 * sb_per_row);
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int64_t r = 0; r < ne1; r++) {
                const int ct  = (int) (r / 32);
                const int row = (int) (r % 32);
                block_q6_K * dst_row = dst_slice + r * sb_per_row;

                for (int64_t sb = 0; sb < sb_per_row; sb++) {
                    block_q6_K * b = &dst_row[sb];
                    memset(b, 0, sizeof(block_q6_K));

                    float sub_scales[16];
                    for (int kt_local = 0; kt_local < 8; kt_local++) {
                        const int kt = sb * 8 + kt_local;
                        const uint8_t *   tile  = matrix_src + ((size_t) ct * n_k_tiles + kt) * tile_size;
                        const uint8_t *   lo_pl = tile;
                        const uint8_t *   hi_pl = tile + 512;
                        const ggml_half * sc_pl = (const ggml_half *) (tile + 768);

                        const int c = kt_local / 4;
                        const int g = kt_local % 4;
                        uint8_t * ql = b->ql + c * 64;
                        uint8_t * qh = b->qh + c * 32;

                        for (int lk = 0; lk < 32; lk++) {
                            const int g_tile = lk >> 2;
                            const int pos    = row * 4 + (lk & 3);
                            const uint8_t lo = (lo_pl[(g_tile >> 1) * 128 + pos] >> ((g_tile & 1) * 4)) & 0xF;
                            const uint8_t hi = (hi_pl[(g_tile >> 2) * 128 + pos] >> ((g_tile & 3) * 2)) & 3;

                            switch (g) {
                                case 0:
                                    ql[lk]      |= lo;
                                    qh[lk]      |= (hi << 0);
                                    break;
                                case 1:
                                    ql[lk + 32] |= lo;
                                    qh[lk]      |= (hi << 2);
                                    break;
                                case 2:
                                    ql[lk]      |= (lo << 4);
                                    qh[lk]      |= (hi << 4);
                                    break;
                                default:
                                    ql[lk + 32] |= (lo << 4);
                                    qh[lk]      |= (hi << 6);
                                    break;
                            }
                        }

                        for (int sub = 0; sub < 2; sub++) {
                            sub_scales[kt_local * 2 + sub] = GGML_FP16_TO_FP32(sc_pl[sub * 32 + row]);
                        }
                    }

                    float max_abs_scale = 0.0f;
                    for (int s = 0; s < 16; s++) {
                        float abs_scale = fabsf(sub_scales[s]);
                        if (abs_scale > max_abs_scale) {
                            max_abs_scale = abs_scale;
                        }
                    }

                    if (max_abs_scale == 0.0f) {
                        b->d = GGML_FP32_TO_FP16(0.0f);
                        memset(b->scales, 0, sizeof(b->scales));
                    } else {
                        float d_flt = max_abs_scale / 127.0f;
                        b->d = GGML_FP32_TO_FP16(d_flt);
                        float d_actual = GGML_FP16_TO_FP32(b->d);
                        float inv_d = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                        for (int s = 0; s < 16; s++) {
                            int sc = (int) roundf(sub_scales[s] * inv_d);
                            b->scales[s] = (int8_t) (std::max)(-128, (std::min)(127, sc));
                        }
                    }
                }
            }
        }
    }

    GGML_UNUSED(size);
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// tile layout: see HTP_MM_WEIGHT_TILE_SIZE_Q4_1 in htp/matmul-ops.h
static void repack_q4_K_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);

    const block_q4_K * src_matrix = (const block_q4_K *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    GGML_ASSERT(ne0 % QK_K == 0);

    const int n_col_tiles = ne1_padded / 32;
    const int n_k_tiles   = ne0_padded / 32;
    const size_t tile_size   = HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;

    const int64_t sb_per_row = ne0 / QK_K;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q4_K * src_slice = src_matrix + (i3 * ne2 + i2) * (ne1 * sb_per_row);
            uint8_t * matrix_dst = (uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            memset(matrix_dst, 0, matrix_size);

            for (int64_t r = 0; r < ne1; r++) {
                const int ct  = (int) (r / 32);
                const int row = (int) (r % 32);
                const block_q4_K * src_row = src_slice + r * sb_per_row;

                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const int kt_local = kt % 8;
                    const block_q4_K * b = &src_row[kt / 8];
                    const float d = GGML_FP16_TO_FP32(b->d);
                    const float dmin = GGML_FP16_TO_FP32(b->dmin);

                    uint8_t * tile_dst = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_size;

                    uint8_t sc, m;
                    get_scale_min_k4(kt_local, b->scales, &sc, &m);

                    const float D = d * (float) sc;
                    const float M = -dmin * (float) m;

                    const uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                    const int shift = (kt_local & 1) ? 4 : 0;

                    for (int cp = 0; cp < 16; cp++) {
                        const uint8_t q0 = (qs_sub[2 * cp + 0] >> shift) & 0x0F;
                        const uint8_t q1 = (qs_sub[2 * cp + 1] >> shift) & 0x0F;
                        tile_dst[cp * 32 + row] = (uint8_t) ((q1 << 4) | q0);
                    }

                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                    scale_dst[2 * row + 0] = GGML_FP32_TO_FP16(D);
                    scale_dst[2 * row + 1] = GGML_FP32_TO_FP16(M);
                }
            }
        }
    }

    GGML_UNUSED(size);
}

// Reverse of repack_q4_K_tiled. Unpacks quants and normalizes scales/mins. Read-back only.
static void repack_tiled_q4_K(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);

    block_q4_K * dst_matrix = (block_q4_K *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    GGML_ASSERT(ne0 % QK_K == 0);

    const int n_col_tiles = ne1_padded / 32;
    const int n_k_tiles   = ne0_padded / 32;
    const size_t tile_size   = HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;

    const int64_t sb_per_row = ne0 / QK_K;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q4_K * dst_slice = dst_matrix + (i3 * ne2 + i2) * (ne1 * sb_per_row);
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int64_t r = 0; r < ne1; r++) {
                const int ct  = (int) (r / 32);
                const int row = (int) (r % 32);
                block_q4_K * dst_row = dst_slice + r * sb_per_row;

                for (int64_t sb = 0; sb < sb_per_row; sb++) {
                    block_q4_K * b = &dst_row[sb];
                    memset(b, 0, sizeof(block_q4_K));

                    float sub_scales[8];
                    float sub_mins[8];

                    for (int kt_local = 0; kt_local < 8; kt_local++) {
                        const int kt = sb * 8 + kt_local;
                        const uint8_t * tile_src = matrix_src + ((size_t) ct * n_k_tiles + kt) * tile_size;
                        const ggml_half * scale_src = (const ggml_half *) (tile_src + 512);

                        uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                        const int shift = (kt_local & 1) ? 4 : 0;

                        for (int cp = 0; cp < 16; cp++) {
                            const uint8_t val = tile_src[cp * 32 + row];
                            const uint8_t q0 = val & 0x0F;
                            const uint8_t q1 = val >> 4;
                            qs_sub[2 * cp + 0] |= (uint8_t) (q0 << shift);
                            qs_sub[2 * cp + 1] |= (uint8_t) (q1 << shift);
                        }

                        const float D = GGML_FP16_TO_FP32(scale_src[2 * row + 0]);
                        const float M = GGML_FP16_TO_FP32(scale_src[2 * row + 1]);
                        sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                        sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
                    }

                    float max_scale = 0.0f;
                    float max_min   = 0.0f;
                    for (int j = 0; j < 8; j++) {
                        if (sub_scales[j] > max_scale) max_scale = sub_scales[j];
                        if (sub_mins[j]   > max_min)   max_min   = sub_mins[j];
                    }

                    float inv_scale = 0.0f;
                    if (max_scale > 0.0f) {
                        b->d = GGML_FP32_TO_FP16(max_scale / 63.0f);
                        const float d_actual = GGML_FP16_TO_FP32(b->d);
                        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
                    } else {
                        b->d = GGML_FP32_TO_FP16(0.0f);
                    }

                    float inv_min = 0.0f;
                    if (max_min > 0.0f) {
                        b->dmin = GGML_FP32_TO_FP16(max_min / 63.0f);
                        const float dmin_actual = GGML_FP16_TO_FP32(b->dmin);
                        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
                    } else {
                        b->dmin = GGML_FP32_TO_FP16(0.0f);
                    }

                    for (int j = 0; j < 8; j++) {
                        uint8_t ls = (uint8_t) roundf(inv_scale * sub_scales[j]);
                        uint8_t lm = (uint8_t) roundf(inv_min * sub_mins[j]);
                        ls = (std::min)((uint8_t) 63, ls);
                        lm = (std::min)((uint8_t) 63, lm);
                        if (j < 4) {
                            b->scales[j]     = ls;
                            b->scales[j + 4] = lm;
                        } else {
                            b->scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                            b->scales[j - 4] |= ((ls >> 4) << 6);
                            b->scales[j - 0] |= ((lm >> 4) << 6);
                        }
                    }
                }
            }
        }
    }

    GGML_UNUSED(size);
}

// tile layout: see HTP_MM_WEIGHT_TILE_SIZE_Q5_K in htp/matmul-ops.h
static void repack_q5_K_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);

    const block_q5_K * src_matrix = (const block_q5_K *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    GGML_ASSERT(ne0 % QK_K == 0);

    const int n_col_tiles = ne1_padded / 32;
    const int n_k_tiles   = ne0_padded / 32;
    const size_t tile_size   = HTP_MM_WEIGHT_TILE_SIZE_Q5_K;
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;

    const int64_t sb_per_row = ne0 / QK_K;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q5_K * src_slice = src_matrix + (i3 * ne2 + i2) * (ne1 * sb_per_row);
            uint8_t * matrix_dst = (uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            memset(matrix_dst, 0, matrix_size);

            for (int64_t r = 0; r < ne1; r++) {
                const int ct  = (int) (r / 32);
                const int row = (int) (r % 32);
                const block_q5_K * src_row = src_slice + r * sb_per_row;

                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const int kt_local = kt % 8;
                    const block_q5_K * b = &src_row[kt / 8];
                    const float d = GGML_FP16_TO_FP32(b->d);
                    const float dmin = GGML_FP16_TO_FP32(b->dmin);

                    uint8_t * tile_dst = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_size;
                    uint8_t * plane    = tile_dst + 640;

                    uint8_t sc, m;
                    get_scale_min_k4(kt_local, b->scales, &sc, &m);

                    const float D = d * (float) sc;
                    const float M = -dmin * (float) m;

                    const uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                    const int shift = (kt_local & 1) ? 4 : 0;
                    const uint8_t hbit = (uint8_t) (1 << kt_local);

                    for (int cp = 0; cp < 16; cp++) {
                        const uint8_t q0 = (qs_sub[2 * cp + 0] >> shift) & 0x0F;
                        const uint8_t q1 = (qs_sub[2 * cp + 1] >> shift) & 0x0F;
                        tile_dst[cp * 32 + row] = (uint8_t) ((q1 << 4) | q0);

                        const int i    = cp / 4;
                        const int lane = (cp % 4) * 32 + row;
                        if (b->qh[2 * cp + 0] & hbit) {
                            plane[lane] |= (uint8_t) (1 << (2 * i));
                        }
                        if (b->qh[2 * cp + 1] & hbit) {
                            plane[lane] |= (uint8_t) (1 << (2 * i + 1));
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                    scale_dst[2 * row + 0] = GGML_FP32_TO_FP16(D);
                    scale_dst[2 * row + 1] = GGML_FP32_TO_FP16(M);
                }
            }
        }
    }

    GGML_UNUSED(size);
}

// Reverse of repack_q5_K_tiled. Unpacks quants losslessly and normalizes scales/mins. Read-back only.
static void repack_tiled_q5_K(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);

    block_q5_K * dst_matrix = (block_q5_K *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    GGML_ASSERT(ne0 % QK_K == 0);

    const int n_col_tiles = ne1_padded / 32;
    const int n_k_tiles   = ne0_padded / 32;
    const size_t tile_size   = HTP_MM_WEIGHT_TILE_SIZE_Q5_K;
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;

    const int64_t sb_per_row = ne0 / QK_K;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q5_K * dst_slice = dst_matrix + (i3 * ne2 + i2) * (ne1 * sb_per_row);
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int64_t r = 0; r < ne1; r++) {
                const int ct  = (int) (r / 32);
                const int row = (int) (r % 32);
                block_q5_K * dst_row = dst_slice + r * sb_per_row;

                for (int64_t sb = 0; sb < sb_per_row; sb++) {
                    block_q5_K * b = &dst_row[sb];
                    memset(b, 0, sizeof(block_q5_K));

                    float sub_scales[8];
                    float sub_mins[8];

                    for (int kt_local = 0; kt_local < 8; kt_local++) {
                        const int kt = sb * 8 + kt_local;
                        const uint8_t * tile_src  = matrix_src + ((size_t) ct * n_k_tiles + kt) * tile_size;
                        const uint8_t * plane     = tile_src + 640;
                        const ggml_half * scale_src = (const ggml_half *) (tile_src + 512);

                        uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                        const int shift = (kt_local & 1) ? 4 : 0;
                        const uint8_t hbit = (uint8_t) (1 << kt_local);

                        for (int cp = 0; cp < 16; cp++) {
                            const uint8_t val = tile_src[cp * 32 + row];
                            const uint8_t q0 = val & 0x0F;
                            const uint8_t q1 = val >> 4;
                            qs_sub[2 * cp + 0] |= (uint8_t) (q0 << shift);
                            qs_sub[2 * cp + 1] |= (uint8_t) (q1 << shift);

                            const int i    = cp / 4;
                            const int lane = (cp % 4) * 32 + row;
                            if (plane[lane] & (1 << (2 * i))) {
                                b->qh[2 * cp + 0] |= hbit;
                            }
                            if (plane[lane] & (1 << (2 * i + 1))) {
                                b->qh[2 * cp + 1] |= hbit;
                            }
                        }

                        const float D = GGML_FP16_TO_FP32(scale_src[2 * row + 0]);
                        const float M = GGML_FP16_TO_FP32(scale_src[2 * row + 1]);
                        sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                        sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
                    }

                    float max_scale = 0.0f;
                    float max_min   = 0.0f;
                    for (int j = 0; j < 8; j++) {
                        if (sub_scales[j] > max_scale) max_scale = sub_scales[j];
                        if (sub_mins[j]   > max_min)   max_min   = sub_mins[j];
                    }

                    float inv_scale = 0.0f;
                    if (max_scale > 0.0f) {
                        b->d = GGML_FP32_TO_FP16(max_scale / 63.0f);
                        const float d_actual = GGML_FP16_TO_FP32(b->d);
                        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
                    } else {
                        b->d = GGML_FP32_TO_FP16(0.0f);
                    }

                    float inv_min = 0.0f;
                    if (max_min > 0.0f) {
                        b->dmin = GGML_FP32_TO_FP16(max_min / 63.0f);
                        const float dmin_actual = GGML_FP16_TO_FP32(b->dmin);
                        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
                    } else {
                        b->dmin = GGML_FP32_TO_FP16(0.0f);
                    }

                    for (int j = 0; j < 8; j++) {
                        uint8_t ls = (uint8_t) roundf(inv_scale * sub_scales[j]);
                        uint8_t lm = (uint8_t) roundf(inv_min * sub_mins[j]);
                        ls = (std::min)((uint8_t) 63, ls);
                        lm = (std::min)((uint8_t) 63, lm);
                        if (j < 4) {
                            b->scales[j]     = ls;
                            b->scales[j + 4] = lm;
                        } else {
                            b->scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                            b->scales[j - 4] |= ((ls >> 4) << 6);
                            b->scales[j - 0] |= ((lm >> 4) << 6);
                        }
                    }
                }
            }
        }
    }

    GGML_UNUSED(size);
}

static void repack_tensor_tiled(ggml_tensor * tensor, const void * data, size_t size) {
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            repack_q4_0_tiled(tensor, data, 0, size);
            if (ggml_hexagon_a8w8_pre_match(tensor)) {
                a8w8_pre_build(tensor);
                ((ggml_hexagon_tensor_extra *) tensor->extra)->flags |= GGML_HEXAGON_TENSOR_A8W8PRE;
            }
            break;

        case GGML_TYPE_Q4_1:
            repack_q4_1_tiled(tensor, data, 0, size);
            break;

        case GGML_TYPE_Q4_K:
            repack_q4_K_tiled(tensor, data, 0, size);
            break;

        case GGML_TYPE_Q8_0:
            repack_q8_0_tiled(tensor, data, 0, size);
            break;

        case GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, data, 0, size);
            break;

        case GGML_TYPE_MXFP4:
            repack_mxfp4_tiled(tensor, data, 0, size);
            break;

        case GGML_TYPE_Q5_K:
            repack_q5_K_tiled(tensor, data, 0, size);
            break;

        case GGML_TYPE_Q6_K:
            repack_q6_K_tiled(tensor, data, 0, size);
            break;

        default:
            break;
    }
}

static void ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   const void *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    if (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        if (ggml_hexagon_is_repack_type(tensor->type)) {
            extra->flags |= GGML_HEXAGON_TENSOR_REPACK;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, data, offset, size, (int) buffer->usage, extra->flags);

    if ((extra->flags & GGML_HEXAGON_TENSOR_REPACK) == 0) {
        memcpy((char *) tensor->data + offset, data, size);
        return;
    }

    if (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        repack_tensor_tiled(tensor, data, size);
        return;
    }

    if (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    memcpy(extra->shadow_buf.data() + offset, data, size);
    extra->shadow_size += size;

    if (extra->shadow_size >= ggml_nbytes(tensor)) {
        repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

static void ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   const ggml_tensor *   tensor,
                                                   void *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, data, offset, size, (int) buffer->usage, extra->flags);

    if ((extra->flags & GGML_HEXAGON_TENSOR_REPACK) == 0) {
        memcpy(data, (const char *) tensor->data + offset, size);
        return;
    }

    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q4_0(data, tensor, offset, size);
            break;

        case GGML_TYPE_Q4_1:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q4_1(data, tensor, offset, size);
            break;

        case GGML_TYPE_Q4_K:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q4_K(data, tensor, offset, size);
            break;

        case GGML_TYPE_Q5_K:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q5_K(data, tensor, offset, size);
            break;

        case GGML_TYPE_Q8_0:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q8_0(data, tensor, offset, size);
            break;

        case GGML_TYPE_IQ4_NL:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q4_0(data, tensor, offset, size);
            break;

        case GGML_TYPE_MXFP4:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_mxfp4(data, tensor, offset, size);
            break;

        case GGML_TYPE_Q6_K:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q6_K(data, tensor, offset, size);
            break;

        default:
            memcpy(data, (const char *) tensor->data + offset, size);
            break;
    }
}

static bool ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   const struct ggml_tensor * src,
                                                   struct ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    return false;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

static void ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                          ggml_tensor *         tensor,
                                                          const void *          data,
                                                          size_t                offset,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    if (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        if (ggml_hexagon_is_repack_type(tensor->type)) {
            extra->flags |= GGML_HEXAGON_TENSOR_REPACK;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (int) buffer->usage, extra->flags);

    if ((extra->flags & GGML_HEXAGON_TENSOR_REPACK) == 0) {
        for (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (const uint8_t *) data + i * stride_data, size);
        }
        return;
    }

    if (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    for (size_t i = 0; i < n_copies; i++) {
        memcpy(extra->shadow_buf.data() + offset + i * stride_tensor, (const uint8_t *) data + i * stride_data, size);
    }
    extra->shadow_size += n_copies * size;

    if (extra->shadow_size >= ggml_nbytes(tensor)) {
        repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

static void ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                      const ggml_tensor *   tensor,
                                                      void *                data,
                                                      size_t                offset,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (int) buffer->usage);

    if ((extra->flags & GGML_HEXAGON_TENSOR_REPACK) == 0) {
        for (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *)data + i * stride_data, (const uint8_t *)tensor->data + offset + i * stride_tensor, size);
        }
        return;
    }

    size_t temp_size      = n_copies > 0 ? (n_copies - 1) * stride_tensor + size : 0;
    size_t slice_size     = tensor->ne[1] * ggml_row_size(tensor->type, tensor->ne[0]);
    size_t slice_offset   = offset % slice_size;
    size_t row_size_bytes = ggml_row_size(tensor->type, tensor->ne[0]);

    GGML_ASSERT((slice_offset % row_size_bytes) == 0 && "offset must be aligned to row boundary");
    GGML_ASSERT((temp_size % row_size_bytes)    == 0 && "temp_size must be a multiple of row size");
    GGML_ASSERT((slice_offset / row_size_bytes) % 32 == 0 && "offset must be aligned to tile size (32 rows)");
    GGML_ASSERT((offset + temp_size) <= ggml_nbytes(tensor));

    std::vector<uint8_t> temp_buf(temp_size);

    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            repack_tiled_q4_0(temp_buf.data(), tensor, offset, temp_size);
            break;

        case GGML_TYPE_Q4_1:
            repack_tiled_q4_1(temp_buf.data(), tensor, offset, temp_size);
            break;

        case GGML_TYPE_Q4_K:
            repack_tiled_q4_K(temp_buf.data(), tensor, offset, temp_size);
            break;

        case GGML_TYPE_Q5_K:
            repack_tiled_q5_K(temp_buf.data(), tensor, offset, temp_size);
            break;

        case GGML_TYPE_Q8_0:
            repack_tiled_q8_0(temp_buf.data(), tensor, offset, temp_size);
            break;

        case GGML_TYPE_IQ4_NL:
            repack_tiled_q4_0(temp_buf.data(), tensor, offset, temp_size);
            break;

        case GGML_TYPE_MXFP4:
            repack_tiled_mxfp4(temp_buf.data(), tensor, offset, temp_size);
            break;

        case GGML_TYPE_Q6_K:
            repack_tiled_q6_K(temp_buf.data(), tensor, offset, temp_size);
            break;

        default:
            memcpy(temp_buf.data(), (const uint8_t *) tensor->data + offset, temp_size);
            break;
    }

    for (size_t i = 0; i < n_copies; i++) {
        memcpy((uint8_t *) data + i * stride_data, temp_buf.data() + i * stride_tensor, size);
    }
}

static void ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (void *) sbuf->base(), sbuf->size());
    memset(sbuf->base(), value, sbuf->size());
}

static ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ NULL,
};

// ** backend buffer type

static void ggml_backend_hexagon_host_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                        ggml_tensor *         tensor,
                                                        const void *          data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                        const ggml_tensor *   tensor,
                                                        void *                data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_hexagon_host_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ NULL,
};

// ** backend buffer type

static const char * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    return static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    if (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        return nullptr;
    }
    try {
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size, false);
        return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        return nullptr;
    }
}

static ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    if (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        return nullptr;
    }
    try {
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size, false);
        return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_host_buffer_interface, sbuf, size);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        return nullptr;
    }
}

static size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;  // HVX alignment
    GGML_UNUSED(buft);
}

static size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * t) {
    if (ggml_hexagon_is_repack_type(t->type)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t sz = ggml_hexagon_tiled_row_size(t->type, ne0) * ne1 * ne2 * ne3;
        if (ggml_hexagon_a8w8_pre_match(t)) {
            sz = ggml_hexagon_a8w8_pre_blob_off(t) + ggml_hexagon_a8w8_pre_blob_bytes(t);
        }
        return sz;
    }
    return ggml_nbytes(t);

    GGML_UNUSED(buft);
}

static size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return opt_mbuf;
    GGML_UNUSED(buft);
}

static bool ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;
    GGML_UNUSED(buft);
}

static bool ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    .get_name         = ggml_backend_hexagon_buffer_type_name,
    .alloc_buffer     = ggml_backend_hexagon_buffer_type_alloc_buffer,
    .get_alignment    = ggml_backend_hexagon_buffer_type_get_alignment,
    .get_max_size     = ggml_backend_hexagon_buffer_type_get_max_size,
    .get_alloc_size   = ggml_backend_hexagon_buffer_type_get_alloc_size,
    .is_host          = ggml_backend_hexagon_buffer_type_is_host,
};

static ggml_backend_buffer_type_i ggml_backend_hexagon_host_buffer_type_interface = {
    .get_name = ggml_backend_hexagon_buffer_type_name,
    .alloc_buffer = ggml_backend_hexagon_host_buffer_type_alloc_buffer,
    .get_alignment = ggml_backend_hexagon_buffer_type_get_alignment,
    .get_max_size = ggml_backend_hexagon_buffer_type_get_max_size,
    .get_alloc_size = ggml_backend_hexagon_buffer_type_get_alloc_size,
    .is_host = ggml_backend_hexagon_host_buffer_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(int dev_id, const ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    buffer_type.device  = dev;
    buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    buffer_type.context = new ggml_backend_hexagon_buffer_type_context(config.name, this);

    host_buffer_type.device  = dev;
    host_buffer_type.iface   = ggml_backend_hexagon_host_buffer_type_interface;
    host_buffer_type.context = new ggml_backend_hexagon_buffer_type_context(config.name + "-HOST", this);

    fence_buffer_type.device  = dev;
    fence_buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    fence_buffer_type.context = new ggml_backend_hexagon_buffer_type_context(config.name + "-FENCE", this);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(host_buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(fence_buffer_type.context);
}

static bool ggml_backend_buffer_is_hexagon(const struct ggml_backend_buffer * b) {
    return b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

// fp16 HMX activation tiles (GGML_HEXAGON_ACT_TILE). Only one producer's tiles are live at a time, so the plan admits a producer only if all its matmul consumers run before the next candidate producer in node order overwrites the scratch.
// Layout: htp/hmx-act-tiles.h. Each session owns one scratch buffer.
static constexpr size_t GGML_HEXAGON_ACT_TILE_BYTES = (size_t) 16 << 20;

static bool ggml_hexagon_act_tile_reserve(ggml_hexagon_session * sess) {
    if (sess->act_tile_buf) {
        return true;
    }
    try {
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, GGML_HEXAGON_ACT_TILE_BYTES, false);
        sess->act_tile_buf = ggml_backend_buffer_init(nullptr, ggml_backend_hexagon_buffer_interface, sbuf, GGML_HEXAGON_ACT_TILE_BYTES);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s activation tile buffer allocation failed: %s\n", sess->c_name(), exc.what());
        return false;
    }
    sess->act_tile_extra.flags = GGML_HEXAGON_TENSOR_ACT_TILE;
    return true;
}

// Batch view of the scratch as an F16 [k x m] activation. Cached per shape: the batch keys tensors by pointer.
static const ggml_tensor * ggml_hexagon_act_tile_view(ggml_hexagon_session * sess, int64_t k, int64_t m) {
    const uint64_t key = ((uint64_t) k << 32) | (uint64_t) m;
    auto it = sess->act_tile_views.find(key);
    if (it != sess->act_tile_views.end()) {
        return it->second.get();
    }
    auto t = std::make_unique<ggml_tensor>();
    t->type   = GGML_TYPE_F16;
    t->ne[0]  = k;
    t->ne[1]  = m;
    t->ne[2]  = 1;
    t->ne[3]  = 1;
    t->nb[0]  = sizeof(uint16_t);
    t->nb[1]  = (size_t) k * sizeof(uint16_t);
    t->nb[2]  = t->nb[1] * (size_t) m;
    t->nb[3]  = t->nb[2];
    t->buffer = sess->act_tile_buf;
    t->data   = static_cast<ggml_hexagon_shared_buffer *>(sess->act_tile_buf->context)->base();
    t->extra  = &sess->act_tile_extra;
    const ggml_tensor * view = t.get();
    sess->act_tile_views.emplace(key, std::move(t));
    sess->mmap_tensor(view);
    return view;
}

// FFN SWIGLU output tiles (GGML_HEXAGON_SWIGLU_OUT): one DDR buffer per session, layout htp/hmx-act-tiles.h.
static constexpr size_t GGML_HEXAGON_SWIGLU_OUT_BYTES = (size_t) 16 << 20;

static bool ggml_hexagon_swiglu_reserve(ggml_hexagon_session * sess) {
    if (sess->swiglu_buf) {
        return true;
    }
    try {
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, GGML_HEXAGON_SWIGLU_OUT_BYTES, false);
        sess->swiglu_buf = ggml_backend_buffer_init(nullptr, ggml_backend_hexagon_buffer_interface, sbuf, GGML_HEXAGON_SWIGLU_OUT_BYTES);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s SWIGLU output tile buffer allocation failed: %s\n", sess->c_name(), exc.what());
        return false;
    }
    sess->swiglu_extra.flags = GGML_HEXAGON_TENSOR_ACT_TILE;
    return true;
}

// Batch view of the SWIGLU output buffer as an F16 [n x m] activation. Cached per shape: the batch keys tensors by pointer.
static const ggml_tensor * ggml_hexagon_swiglu_view(ggml_hexagon_session * sess, int64_t n, int64_t m) {
    const uint64_t key = ((uint64_t) n << 32) | (uint64_t) m;
    auto it = sess->swiglu_views.find(key);
    if (it != sess->swiglu_views.end()) {
        return it->second.get();
    }
    auto t = std::make_unique<ggml_tensor>();
    t->type   = GGML_TYPE_F16;
    t->ne[0]  = n;
    t->ne[1]  = m;
    t->ne[2]  = 1;
    t->ne[3]  = 1;
    t->nb[0]  = sizeof(uint16_t);
    t->nb[1]  = (size_t) n * sizeof(uint16_t);
    t->nb[2]  = t->nb[1] * (size_t) m;
    t->nb[3]  = t->nb[2];
    t->buffer = sess->swiglu_buf;
    t->data   = static_cast<ggml_hexagon_shared_buffer *>(sess->swiglu_buf->context)->base();
    t->extra  = &sess->swiglu_extra;
    const ggml_tensor * view = t.get();
    sess->swiglu_views.emplace(key, std::move(t));
    sess->mmap_tensor(view);
    return view;
}

// fp16 tile view of an MC05 norm output (GGML_HEXAGON_GATHER_ACT_TILE, layout htp/hmx-act-tiles.h): the same bytes as the fp32 output,
// read as tiles. Keyed by the output's data pointer; one view per output buffer.
const ggml_tensor * ggml_hexagon_session::gather_tile_view(const ggml_tensor * P) {
    auto it = gather_views.find(P->data);
    if (it != gather_views.end() && it->second->buffer == P->buffer) {
        return it->second.get();
    }
    gather_tile_extra.flags = GGML_HEXAGON_TENSOR_ACT_TILE;
    auto t = std::make_unique<ggml_tensor>();
    t->type   = GGML_TYPE_F16;
    t->ne[0]  = P->ne[0];
    t->ne[1]  = P->ne[1];
    t->ne[2]  = 1;
    t->ne[3]  = 1;
    t->nb[0]  = sizeof(uint16_t);
    t->nb[1]  = (size_t) P->ne[0] * sizeof(uint16_t);
    t->nb[2]  = t->nb[1] * (size_t) P->ne[1];
    t->nb[3]  = t->nb[2];
    t->buffer = P->buffer;
    t->data   = P->data;
    t->extra  = &gather_tile_extra;
    const ggml_tensor * view = t.get();
    gather_views[P->data] = std::move(t);
    mmap_tensor(view);
    return view;
}

// AE-03 norm output tiles (GGML_HEXAGON_AR_RMS_FUSE): one buffer per session of GGML_HEXAGON_NORM_TILE_BYTES, reserved lazily like the
// activation tile scratch. It holds the fp16 tiles (K * M * 2 bytes) of the norm outputs this session owns, for one prefill batch.
static constexpr size_t GGML_HEXAGON_NORM_TILE_BYTES = (size_t) 16 << 20;

static bool ggml_hexagon_norm_tile_reserve(ggml_hexagon_session * sess) {
    if (sess->norm_tile_buf) {
        return true;
    }
    try {
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, GGML_HEXAGON_NORM_TILE_BYTES, false);
        sess->norm_tile_buf = ggml_backend_buffer_init(nullptr, ggml_backend_hexagon_buffer_interface, sbuf, GGML_HEXAGON_NORM_TILE_BYTES);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s norm tile buffer allocation failed: %s\n", sess->c_name(), exc.what());
        return false;
    }
    sess->norm_tile_extra.flags = GGML_HEXAGON_TENSOR_ACT_TILE;
    return true;
}

// Batch view of this session's norm tile buffer as an F16 [K x M] activation (the layout of gather_tile_view). Cached per shape; the
// buffer is mapped here, and the peers that write into it clone it when they enqueue the fused op (enqueue_op).
const ggml_tensor * ggml_hexagon_session::norm_tile_view(int64_t K, int64_t M) {
    GGML_ASSERT(this->norm_tile_buf);
    const uint64_t key = ((uint64_t) K << 32) | (uint64_t) M;
    auto it = norm_tile_views.find(key);
    if (it != norm_tile_views.end()) {
        return it->second.get();
    }
    auto t = std::make_unique<ggml_tensor>();
    t->type   = GGML_TYPE_F16;
    t->ne[0]  = K;
    t->ne[1]  = M;
    t->ne[2]  = 1;
    t->ne[3]  = 1;
    t->nb[0]  = sizeof(uint16_t);
    t->nb[1]  = (size_t) K * sizeof(uint16_t);
    t->nb[2]  = t->nb[1] * (size_t) M;
    t->nb[3]  = t->nb[2];
    t->buffer = this->norm_tile_buf;
    t->data   = static_cast<ggml_hexagon_shared_buffer *>(this->norm_tile_buf->context)->base();
    t->extra  = &this->norm_tile_extra;
    const ggml_tensor * view = t.get();
    norm_tile_views.emplace(key, std::move(t));
    mmap_tensor(view);
    return view;
}

// A matmul runs the HMX 2D kernel (the only path that reads fp16 activation tiles). Other kernels keep the fp32 activation.
static inline bool act_tile_mm_hmx(const htp_mm_kernel_params * kp) {
    return kp->n_hmx && kp->kernel_type == HTP_MM_KERNEL_HMX_2D;
}

// Batch rows: the row dimension of the first RMS_NORM output near the head of the graph (the batch's activation rows, 0 if none).
// O(1) per graph, so decode graphs skip the prefill-only host scans below PREFILL_MIN_ROWS.
static int64_t ggml_hexagon_graph_rows(const ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes && i < 64; i++) {
        const ggml_tensor * n = graph->nodes[i];
        if (n->op == GGML_OP_RMS_NORM) {
            return n->ne[1];
        }
    }
    return 0;
}

// Graph pre-pass (slow path only, batches >= PREFILL_MIN_ROWS). An RMS_NORM+MUL output P [M x K] is a tile producer for this graph when M >= 64, K % 32 == 0, at least one MUL_MAT reads it as src1, and every such MUL_MAT runs before the next RMS_NORM+MUL candidate.
// A producer takes effect only when its fused op is emitted (try_fuse_rms_norm_mul). GGML_HEXAGON_ACT_TILE=2 also drops its fp32 output (act_tile_skip_ok) when no other op reads P and every MUL_MAT reader runs the HMX 2D kernel.
static void ggml_hexagon_act_tile_plan(ggml_hexagon_session * sess, const ggml_cgraph * graph) {
    std::vector<int> cand;
    for (int i = 0; i + 1 < graph->n_nodes; i++) {
        if (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            cand.push_back(i + 1);
        }
    }
    for (size_t c = 0; c < cand.size(); c++) {
        const ggml_tensor * P = graph->nodes[cand[c]];
        const int64_t K = P->ne[0];
        const int64_t M = P->ne[1];
        if (P->type != GGML_TYPE_F32 || P->ne[2] != 1 || P->ne[3] != 1 || M < PREFILL_MIN_ROWS || K % 32 != 0) {
            continue;
        }
        if ((size_t) ((M + 31) / 32 * 32) * (size_t) K * sizeof(uint16_t) > GGML_HEXAGON_ACT_TILE_BYTES) {
            continue;
        }
        const int next = c + 1 < cand.size() ? cand[c + 1] : graph->n_nodes;
        int  last        = -1;
        bool only_hmx_mm = true;  // no reader of P other than a MUL_MAT (src1 == P) on the HMX 2D kernel
        for (int j = cand[c] + 1; j < graph->n_nodes; j++) {
            const ggml_tensor * n = graph->nodes[j];
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                const ggml_tensor * src = n->src[s];
                if (!src || (src != P && src->view_src != P)) {
                    continue;
                }
                if (n->op != GGML_OP_MUL_MAT || s != 1 || src != P || n->src[0] == P) {
                    only_hmx_mm = false;
                    continue;
                }
                last = j;
                struct htp_mm_kernel_params kp;
                memset(&kp, 0, sizeof(kp));
                ggml_hexagon_precompute_matmul_params(sess, n->src[0], n->src[1], n, &kp);
                only_hmx_mm = only_hmx_mm && act_tile_mm_hmx(&kp);
            }
        }
        if (last < 0 || last >= next) {
            continue;
        }
        if (!ggml_hexagon_act_tile_reserve(sess)) {
            return;
        }
        sess->act_tile_ok.insert(P);
        if (opt_act_tile >= 2 && only_hmx_mm) {
            sess->act_tile_skip_ok.insert(P);
        }
    }
}

// AE-16 zero state (GGML_HEXAGON_ELT_SMALL bit 4). One DDR buffer per session, zero for the session's life: no DSP op writes a
// GDN state source, so the buffer stays zero.
static constexpr size_t GGML_HEXAGON_ZERO_STATE_BYTES = (size_t) 8 << 20;

static bool ggml_hexagon_zero_state_reserve(ggml_hexagon_session * sess) {
    if (sess->zero_state_buf) {
        return true;
    }
    try {
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, GGML_HEXAGON_ZERO_STATE_BYTES, false);
        sess->zero_state_buf = ggml_backend_buffer_init(&sess->dev_ctx->buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, GGML_HEXAGON_ZERO_STATE_BYTES);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s zero-state buffer allocation failed: %s\n", sess->c_name(), exc.what());
        return false;
    }
    memset(static_cast<ggml_hexagon_shared_buffer *>(sess->zero_state_buf->context)->base(), 0, GGML_HEXAGON_ZERO_STATE_BYTES);
    sess->zero_state_extra = {};
    return true;
}

// The GDN state source that stands in for a zeroed state: the state's type and strides, data in zero_state_buf.
static const ggml_tensor * ggml_hexagon_zero_state_src(ggml_hexagon_session * sess, const ggml_tensor * state, bool conv = false) {
    ggml_tensor & z = conv ? sess->zero_state_conv_src : sess->zero_state_src;
    z.type   = state->type;
    memcpy(z.ne, state->ne, sizeof(z.ne));
    memcpy(z.nb, state->nb, sizeof(z.nb));
    z.buffer = sess->zero_state_buf;
    z.data   = static_cast<ggml_hexagon_shared_buffer *>(sess->zero_state_buf->context)->base();
    z.extra  = &sess->zero_state_extra;
    return &z;
}

// AE-16 pre-pass (slow path, GGML_HEXAGON_ELT_SMALL bit 4). Each GDN that is not paired logs its reason under GGML_HEXAGON_VERBOSE. Other backends ignore the marker and keep the clear. This pass pairs that clear with its state copy and GDN only when the chain is SCALE -> GET_ROWS -> RESHAPE -> GDN, nothing else uses GET_ROWS or the RESHAPE, and n_seqs == 1.
// The clear is then dead on this backend: the GDN reads zeros from zero_state_buf. qwen35's build_rs marks the clear of a fresh single-sequence SSM state with op_params[2] = 1 on the SCALE by 0.
static void ggml_hexagon_zero_state_plan(ggml_hexagon_session * sess, const ggml_cgraph * graph) {
    auto root = [](const ggml_tensor * t) {
        while (t->view_src) {
            t = t->view_src;
        }
        return t;
    };
    auto n_users = [graph](const ggml_tensor * t) {
        int n = 0;
        for (int j = 0; j < graph->n_nodes; j++) {
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                n += graph->nodes[j]->src[k] == t;
            }
        }
        return n;
    };
    for (int i = 0; i < graph->n_nodes; i++) {
        const ggml_tensor * gdn = graph->nodes[i];
        if (gdn->op != GGML_OP_GATED_DELTA_NET) {
            continue;
        }
        const char *        why   = nullptr;
        const ggml_tensor * st    = gdn->src[5];
        const ggml_tensor * g     = st ? st->src[0] : nullptr;
        const ggml_tensor * clear = nullptr;
        if (!st || st->op != GGML_OP_RESHAPE) {
            why = "state source is not a reshape";
        } else if (!g || g->op != GGML_OP_GET_ROWS || !g->src[0] || st->ne[3] != 1) {
            why = "state copy is not GET_ROWS or n_seqs != 1";
        } else {
            for (int j = 0; j < graph->n_nodes && !clear; j++) {
                const ggml_tensor * sc = graph->nodes[j];
                if (sc->op == GGML_OP_SCALE && sc->src[0] && ggml_get_op_params_i32(sc, 2) == 1 && root(sc->src[0]) == root(g->src[0])) {
                    clear = sc;
                }
            }
            if (!clear) {
                why = "no marked clear of this state in the graph";
            } else if (n_users(g) != 1) {
                why = "state copy has other users";
            } else if (n_users(st) != 1) {
                why = "state reshape has other users";
            } else if (ggml_nbytes(st) > GGML_HEXAGON_ZERO_STATE_BYTES) {
                why = "state larger than the zero buffer";
            } else if (!ggml_hexagon_zero_state_reserve(sess)) {
                why = "zero buffer unavailable";
            }
        }
        if (why) {
            HEX_VERBOSE("ggml-hex: %s zero-state GDN (#%d) kept: %s\n", sess->c_name(), i, why);
            continue;
        }
        sess->zero_state_clear_skip.insert(clear);
        sess->zero_state_gdn.insert(gdn);
        sess->zero_state_dead.insert(g);
        HEX_VERBOSE("ggml-hex: %s zero-state GDN (#%d): fresh-state clear skipped\n", sess->c_name(), i);
    }
}

// FFN SWIGLU output plan (GGML_HEXAGON_SWIGLU_OUT). try_fuse_swiglu_out emits the fused op only when the gate/up NX fusion happened; the readers then see the tile view.
// A GLU SWIGLU is admitted when its gate and up are single-use MUL_MATs and every reader of its output is an HMX MUL_MAT that takes it as src1 with K equal to the GLU width (the 3-core K split).
static void ggml_hexagon_swiglu_out_plan(ggml_hexagon_session * sess, const ggml_cgraph * graph) {
    // Batch gate (GGML_HEXAGON_SWIGLU_OUT_MIN_ROWS): a graph is planned only if some SWIGLU has fusable rows. Decode and
    // verify graphs have none and skip the admission scan. The first SWIGLU is not decisive: GDN gate SWIGLUs come earlier.
    bool any_fusable = false;
    for (int i = 0; i < graph->n_nodes && !any_fusable; i++) {
        const ggml_tensor * g = graph->nodes[i];
        any_fusable = g->op == GGML_OP_GLU && ggml_get_glu_op(g) == GGML_GLU_OP_SWIGLU && swiglu_out_rows_ok(g->ne[1]);
    }
    if (!any_fusable) {
        return;
    }
    auto count_uses = [graph](const ggml_tensor * t) {
        int n = 0;
        for (int j = 0; j < graph->n_nodes; j++) {
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (graph->nodes[j]->src[s] == t) {
                    n++;
                }
            }
        }
        return n;
    };
    for (int i = 0; i < graph->n_nodes; i++) {
        const ggml_tensor * glu = graph->nodes[i];
        if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || glu->src[1] == nullptr) {
            continue;
        }
        const ggml_tensor * gate = glu->src[0];
        const ggml_tensor * up   = glu->src[1];
        if (gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT || count_uses(gate) != 1 || count_uses(up) != 1) {
            continue;
        }
        if (glu->type != GGML_TYPE_F32 || glu->ne[2] != 1 || glu->ne[3] != 1 || !swiglu_out_rows_ok(glu->ne[1])) {
            continue;
        }
        bool admit   = true;
        int  readers = 0;
        for (int j = i + 1; j < graph->n_nodes && admit; j++) {
            const ggml_tensor * n = graph->nodes[j];
            for (int s = 0; s < GGML_MAX_SRC && admit; s++) {
                if (n->src[s] != glu) {
                    continue;
                }
                readers++;
                htp_mm_kernel_params kp;
                admit = n->op == GGML_OP_MUL_MAT && s == 1 && n->src[0] != nullptr && n->src[0]->ne[0] == glu->ne[0];
                if (admit) {
                    ggml_hexagon_precompute_matmul_params(sess, n->src[0], n->src[1], n, &kp);
                    admit = act_tile_mm_hmx(&kp);
                }
            }
        }
        if (admit && readers > 0) {
            sess->swiglu_ok.insert(glu);
        }
    }
}

// Gated output norm tile plan (GGML_HEXAGON_NG_TILE). A GLU SWIGLU whose only readers are reshape views read as src1 by HMX 2D MUL_MATs
// (the ssm_out projection, K = heads * head_dim) may write the matmul's fp16 activation tiles directly (try_fuse_norm_gated).
static void ggml_hexagon_ng_tile_plan(ggml_hexagon_session * sess, const ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        const ggml_tensor * glu = graph->nodes[i];
        if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || glu->src[1] == nullptr) {
            continue;
        }
        const int64_t nh = glu->ne[1], ntok = glu->ne[2];
        if (glu->type != GGML_TYPE_F32 || glu->ne[3] != 1 || ntok < 64 || ntok % 32 != 0 || glu->ne[0] % 128 != 0) {
            continue;
        }
        const int64_t K = glu->ne[0] * nh;
        if ((size_t) ntok * (size_t) K * sizeof(uint16_t) > GGML_HEXAGON_ACT_TILE_BYTES) {
            continue;
        }
        std::vector<const ggml_tensor *> views;
        bool admit = true;
        for (int j = i + 1; j < graph->n_nodes && admit; j++) {
            const ggml_tensor * n = graph->nodes[j];
            for (int s = 0; s < GGML_MAX_SRC && admit; s++) {
                const ggml_tensor * src = n->src[s];
                if (!src || (src != glu && src->view_src != glu)) {
                    continue;
                }
                if (src == glu && n->op == GGML_OP_RESHAPE) {
                    views.push_back(n);
                } else if (src->view_src == glu && n->op == GGML_OP_MUL_MAT && s == 1 && src->ne[0] == K && src->ne[1] == ntok &&
                           src->ne[2] == 1 && src->ne[3] == 1 && ggml_is_contiguous(src)) {
                    htp_mm_kernel_params kp;
                    memset(&kp, 0, sizeof(kp));
                    ggml_hexagon_precompute_matmul_params(sess, n->src[0], n->src[1], n, &kp);
                    admit = act_tile_mm_hmx(&kp);
                    bool known = false;
                    for (const ggml_tensor * v : views) known = known || v == src;
                    if (!known) views.push_back(src);
                } else {
                    admit = false;
                }
            }
        }
        if (admit && !views.empty()) {
            sess->ngtile_ok[glu] = views;
        }
    }
}

// Attention gate tile plan (GGML_HEXAGON_GATE_TILE). A MUL(x, SIGMOID(g)) whose output is read only as src1 by HMX 2D MUL_MATs (the
// attn_output projection, K = heads * head_dim) may write the matmul's fp16 activation tiles directly (try_fuse_gate_mul).
static void ggml_hexagon_gate_tile_plan(ggml_hexagon_session * sess, const ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        const ggml_tensor * mul = graph->nodes[i];
        if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32 || mul->ne[2] != 1 || mul->ne[3] != 1 ||
            mul->ne[1] < 64 || mul->ne[1] % 32 != 0 || mul->ne[0] % 32 != 0) {
            continue;
        }
        auto is_sigmoid = [](const ggml_tensor * t) {
            auto sig = [](const ggml_tensor * u) { return u && u->op == GGML_OP_UNARY && ggml_get_unary_op(u) == GGML_UNARY_OP_SIGMOID; };
            return sig(t) || (t && t->op == GGML_OP_RESHAPE && sig(t->src[0]));
        };
        if (!is_sigmoid(mul->src[0]) && !is_sigmoid(mul->src[1])) {
            continue;
        }
        const int64_t K = mul->ne[0], M = mul->ne[1];
        if ((size_t) M * (size_t) K * sizeof(uint16_t) > GGML_HEXAGON_ACT_TILE_BYTES) {
            continue;
        }
        bool admit   = true;
        int  readers = 0;
        for (int j = i + 1; j < graph->n_nodes && admit; j++) {
            const ggml_tensor * n = graph->nodes[j];
            for (int s = 0; s < GGML_MAX_SRC && admit; s++) {
                const ggml_tensor * src = n->src[s];
                if (!src || (src != mul && src->view_src != mul)) {
                    continue;
                }
                readers++;
                if (src == mul && n->op == GGML_OP_MUL_MAT && s == 1 && src->ne[0] == K && src->ne[1] == M && ggml_is_contiguous(src)) {
                    htp_mm_kernel_params kp;
                    memset(&kp, 0, sizeof(kp));
                    ggml_hexagon_precompute_matmul_params(sess, n->src[0], n->src[1], n, &kp);
                    admit = act_tile_mm_hmx(&kp);
                } else {
                    admit = false;
                }
            }
        }
        if (admit && readers > 0) {
            sess->gatetile_ok.insert(mul);
        }
    }
}

struct ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<int, int>                b_map; // buffer fd   to index
    std::unordered_map<const ggml_tensor*, int> t_map; // tensor ptr  to index
    std::unordered_multimap<void*, int>         d_map; // tensor data to index

    unsigned int n_bufs;     // num buffers in the batch
    unsigned int n_tens;     // num tensors ...
    unsigned int n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    unsigned int n_bufs_max;
    unsigned int n_tens_max;
    unsigned int n_ops_max;
    size_t       b_vmem_max;

    // fast op-enqueue recording state: set while a plan is being captured from a
    // slow run; any flush_batch() inside the capture window marks the plan dirty.
    bool         rec_active     = false;
    bool         rec_flush_seen = false;
    // The active plan's sig_n_ops: ops at lower indexes already belong to a finished plan of this batch and are frozen (their recorded descriptors are what plan replay re-enqueues); a fusion that would rewrite one of those ops in place must be refused (see try_fuse's guard below).
    // Set by plan_begin, cleared on reset.
    unsigned int rec_plan_sig   = 0;
    void reset() {
        n_bufs = 0;
        n_tens = 0;
        n_ops  = 0;
        b_vmem = 0;
        rec_plan_sig = 0;

        b_map.clear();
        t_map.clear();
        d_map.clear();
        ops.resize(n_ops_max);
    }

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_size, size_t max_vmem) {
        this->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_size;
        n_tens_max = std::min<size_t>(n_ops_max + n_ops_max * HTP_OP_MAX_INPUTS, HTP_OP_MAX_TENSORS);

        b_vmem_max = max_vmem;

        ops.resize(n_ops_max);

        h_bufs.resize(n_bufs_max);
        h_tens.resize(n_tens_max);
        h_ops.resize(n_ops_max);

        b_map.reserve(n_bufs_max);
        t_map.reserve(n_tens_max);
        d_map.reserve(n_tens_max);

        GGML_LOG_INFO("ggml-hex: %s op batching: n-bufs %u n-tensors %u n-ops %u vmem %zu\n",
                sess->c_name(), n_bufs_max, n_tens_max, n_ops_max, b_vmem_max);

        reset();
    }

    bool empty() const { return n_ops == 0; }

    // add buffer and return its index
    int add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        auto it = b_map.find(sbuf->fd());
        if (it != b_map.end()) { return it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        int bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        if (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (void*) sbuf->base(), (size_t) b.size, b_vmem);

        return bi;
    }

    bool same_shape(const htp_tensor * h, const ggml_tensor * t) const {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;

        int64_t ne0 = t->ne[0];
        int64_t ne1 = t->ne[1];
        const bool is_repack = (extra->flags & GGML_HEXAGON_TENSOR_REPACK) != 0;
        if (is_repack) {
            ne0 = hex_round_up(ne0, 32);
            ne1 = hex_round_up(ne1, 32);
        }
        int64_t nb1 = is_repack ? (int64_t) ggml_hexagon_tiled_row_size(t->type, ne0) : t->nb[1];
        int64_t nb2 = is_repack ? nb1 * ne1      : t->nb[2];
        int64_t nb3 = is_repack ? nb2 * t->ne[2] : t->nb[3];

        return (h->type == t->type) &&
               (h->ne[0] == ne0) && (h->ne[1] == ne1) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == t->nb[0]) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    int add_tensor(const ggml_tensor * t) {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;
        auto sbuf  = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        for (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            if (same_shape(h, t)) { return it->second; }
        }

        // Lookup by tensor ptr
        auto it = t_map.find(t);
        if (it != t_map.end()) { return it->second; }

        // Add new tensor to the batch
        int ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->data, ti});

        uint64_t t_offset = (uint8_t *) t->data - sbuf->base();
        size_t   t_size   = ggml_nbytes(t);

        htp_tensor &h = h_tens[ti];
        h.bi    = add_buffer(sbuf);
        h.ti    = ti;
        h.data  = t_offset;
        h.type  = t->type;

        const bool is_repack = (extra->flags & GGML_HEXAGON_TENSOR_REPACK) != 0;
        if (is_repack) {
            h.ne[0] = hex_round_up(t->ne[0], 32);
            h.ne[1] = hex_round_up(t->ne[1], 32);
            h.ne[2] = t->ne[2];
            h.ne[3] = t->ne[3];

            h.nb[0] = t->nb[0];
            h.nb[1] = ggml_hexagon_tiled_row_size(t->type, h.ne[0]);
            h.nb[2] = h.nb[1] * h.ne[1];
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_size  = h.size;
        } else {
            h.size  = t_size;
            h.ne[0] = t->ne[0]; h.ne[1] = t->ne[1]; h.ne[2] = t->ne[2]; h.ne[3] = t->ne[3];
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        if ((extra->flags & GGML_HEXAGON_TENSOR_WEIGHT) != 0) {
            h.flags |= HTP_TENSOR_WEIGHT;
        }
        if ((extra->flags & GGML_HEXAGON_TENSOR_REPACK) != 0) {
            h.flags |= HTP_TENSOR_REPACK;
        }
        if ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }
        if ((extra->flags & GGML_HEXAGON_TENSOR_ACT_TILE) != 0) {
            h.flags |= HTP_TENSOR_ACT_TILE;
        }
        if ((extra->flags & GGML_HEXAGON_TENSOR_A8W8PRE) != 0) {
            h.flags |= HTP_TENSOR_A8W8;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (void*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        return ti;
    }

    // Lookup-only mirror of add_tensor's resolution order (data+shape via d_map, then pointer via t_map); returns the batch tensor index or -1.
    // Used by fast op-enqueue's frozen-plan cross-boundary fusion patch to resolve a tensor to its plan-relative index without adding anything to the batch.
    int lookup_tensor(const ggml_tensor * t) const {
        auto range = d_map.equal_range(t->data);
        for (auto it = range.first; it != range.second; ++it) {
            const htp_tensor * h = &h_tens[it->second];
            if (same_shape(h, t)) { return it->second; }
        }
        auto it = t_map.find(t);
        if (it != t_map.end()) { return it->second; }
        return -1;
    }

    // Batch input for an fp32 activation: the fp16 tile view of a live producer output (GGML_HEXAGON_ACT_TILE), else t.
    const ggml_tensor * act_tile_src(const ggml_tensor * t) const {
        auto it = sess->act_tile_live.find(t);
        return it == sess->act_tile_live.end() ? t : it->second;
    }
    // Activation input of a MUL_MAT op: the tile view when the op runs the HMX 2D kernel.
    const ggml_tensor * act_tile_mm_src(const htp_opnode & node, const ggml_tensor * t) const {
        return act_tile_mm_hmx((const htp_mm_kernel_params *) node.kernel_params) ? act_tile_src(t) : t;
    }

    bool fit_op(const htp_opnode & node) const {
        if (n_ops >= n_ops_max) return false;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        int seen_bufs[HTP_OP_MAX_BUFS];
        int n_seen_bufs = 0;

        auto fit_tensor = [&](const ggml_tensor *t) {
            if (!t) return;
            if (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                int fd = sbuf->fd();
                if (!b_map.count(fd)) {
                    for (int i = 0; i < n_seen_bufs; i++) {
                        if (seen_bufs[i] == fd) return;
                    }
                    if (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    if (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        for (const auto * src : node.get_inputs()) {
            fit_tensor(node.opcode == HTP_OP_MUL_MAT ? act_tile_mm_src(node, src) : src);
        }
        for (const auto * output : node.get_outputs()) {
            fit_tensor(output);
        }

        if ((extra_bufs + n_bufs) > n_bufs_max) return false;
        if ((extra_tens + n_tens) > n_tens_max) return false;
        if ((extra_vmem + b_vmem) > b_vmem_max) return false;

        return true;
    }

    // assumes that fit_op() was called first and returned true
    void add_op(const htp_opnode & node) {
        // Add new op

        unsigned int n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = node;

        htp_op_desc &o = h_ops[n];
        memcpy(o.params,        node.node->op_params, sizeof(node.node->op_params));
        memcpy(o.kernel_params, node.kernel_params,   sizeof(o.kernel_params));
        o.opcode = node.opcode;
        // MM-03 flags only for prefill-sized MUL_MAT (decode and draft batches keep the original chunk schedule); the MM_ACT_OVERLAP
        // flag likewise for prefill-sized MUL_MAT and MUL_MAT_NX (matmul-ops.c hmx_act_overlap_plan)
        const bool mm_op      = node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_NX;
        const bool mm_prefill = mm_op && ggml_hexagon_src1_rows(node.src1()) >= GGML_HEXAGON_MM_PREFILL_MIN_ROWS;
        // AR nodes are built outside the graph loop (node.dma_clean unset): AR_DMA_CLEAN gives them DMA_CLEAN + AR_DMA_CLEAN for prefill-sized batches
        const bool ar_op      = node.opcode == HTP_OP_ALLREDUCE || node.opcode == HTP_OP_ALLREDUCE_ADD;
        const bool ar_clean   = ar_op && opt_ar_dma_clean_flags != 0 && opt_dma_clean != 0 && node.node->ne[1] >= 64;
        o.flags  = ((node.dma_clean || ar_clean) ? HTP_OPFLAGS_DMA_CLEAN : 0) |
                   ((node.opcode == HTP_OP_MUL_MAT && mm_prefill) ? opt_mm_prologue_flags : 0) |
                   (mm_prefill ? opt_mm_act_overlap_flags : 0) |
                   (mm_prefill ? opt_mm_early_w_flags : 0) |
                   ((node.opcode == HTP_OP_MUL_MAT && mm_prefill) ? opt_mm_c1small_flags : 0) |
                   (ar_clean ? opt_ar_dma_clean_flags : 0) |
                   ggml_hexagon_lane_flag(node.opcode);
        o.flags |= node.extra_flags;
        if (node.opcode == HTP_OP_MUL_MAT && mm_prefill && node.src0() != nullptr && opt_mm_a8w8_ops != 0) {
            // plain MUL_MAT (also the seed of a later MUL_MAT_NX fusion, which inherits these flags), selected by weight name:
            //   2 = ffn_down, 4 = ssm_out / attn_output, 8 = everything else except the lm_head (qkv, attn_q/k/v, ...)
            const char * wn = node.src0()->name;
            uint32_t want = 8u;
            if (strstr(wn, "ffn_down")) want = 2u;
            else if (strstr(wn, "ssm_out") || strstr(wn, "attn_output")) want = 4u;
            else if (strstr(wn, "output.weight")) want = 16u;      // lm_head (opt-in)
            else if (strstr(wn, "token_embd")) want = 0u;
            if ((opt_mm_a8w8_ops & want) != 0) { o.flags |= opt_mm_a8w8_flags; static const int dbg = getenv("GGML_HEXAGON_A8W8_DBG") ? atoi(getenv("GGML_HEXAGON_A8W8_DBG")) : 0; o.params[3] = dbg; static const float clip = getenv("GGML_HEXAGON_A8W8_CLIP") ? (float) atof(getenv("GGML_HEXAGON_A8W8_CLIP")) : 1.0f; memcpy(&o.params[2], &clip, sizeof(clip)); }
        }
        if (opt_rope_fast && (node.opcode == HTP_OP_ROPE || node.opcode == HTP_OP_ROPE_NX)) {
            o.flags |= HTP_OPFLAGS_ROPE_FAST;
        }
        if (node.opcode == HTP_OP_FLASH_ATTN_EXT && opt_fa_exp_v2) {
            o.flags |= HTP_OPFLAGS_FA_EXP_V2;
        }
        if (opt_lossy_gdn != 0) {
            if (node.opcode == HTP_OP_GATED_DELTA_NET) {
                o.flags |= (opt_lossy_gdn & 0xffu) << HTP_OPFLAGS_LOSSY_GDN_SHIFT;
            } else if (node.opcode == HTP_OP_SSM_CONV_CHAIN) {
                o.flags |= ((opt_lossy_gdn >> 8) & 0x3fu) << HTP_OPFLAGS_LOSSY_CONV_SHIFT;
            }
        }
        if (node.opcode == HTP_OP_FLASH_ATTN_EXT && opt_lm_fa_skip) {
            o.flags |= 8u << HTP_OPFLAGS_LM_SHIFT;
        }
        if (node.opcode == HTP_OP_ALLREDUCE_ADD && opt_lm_norm) {
            o.flags |= ((uint32_t) opt_lm_norm & 3u) << HTP_OPFLAGS_LM_SHIFT;
        }
        if (node.opcode == HTP_OP_CONCAT && opt_concat_t_hvx) {
            o.params[1] = 1;   // CONCAT_T_HVX: the DSP takes the pipelined transposed copy (concat-ops.c)
        }

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        auto inputs = node.get_inputs();
        if (node.opcode == HTP_OP_MUL_MAT) {
            // fp16 tile activation (GGML_HEXAGON_ACT_TILE): consume the producer's tile view instead of its fp32 output.
            for (auto & t : inputs) {
                t = act_tile_mm_src(node, t);
            }
        }
        for (unsigned int i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        auto outputs = node.get_outputs();
        for (unsigned int i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    void sort_buffers() {
        if (n_bufs <= 1) return;

        std::vector<int> order(n_bufs);
        for (unsigned int i = 0; i < n_bufs; i++) { order[i] = (int) i; }

        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return h_bufs[a].size > h_bufs[b].size;
        });

        bool already_sorted = true;
        for (unsigned int i = 0; i < n_bufs; i++) {
            if (order[i] != (int) i) {
                already_sorted = false;
                break;
            }
        }
        if (already_sorted) return;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        for (unsigned int new_bi = 0; new_bi < n_bufs; new_bi++) {
            int old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        for (unsigned int i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        for (unsigned int i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    void update_mdev_group(uint32_t mdev_idx) {
        if (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    bool try_fuse_common(std::initializer_list<const ggml_tensor *> tensors) const {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        int seen_bufs[HTP_OP_MAX_BUFS];
        int n_seen_bufs = 0;

        for (const auto * t : tensors) {
            if (!t || t_map.count(t)) {
                continue;
            }
            extra_tens++;
            auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            int fd = sbuf->fd();
            if (!b_map.count(fd)) {
                bool found = false;
                for (int i = 0; i < n_seen_bufs; i++) {
                    if (seen_bufs[i] == fd) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    if (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    if (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        if ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            return false;
        }

        return true;
    }

    bool try_fuse_common(const ggml_tensor * t1, const ggml_tensor * t2) const {
        return try_fuse_common({t1, t2});
    }

    bool try_fuse_allreduce_add(const htp_opnode & node) {
        if (n_ops == 0 || opt_ar_select != 2) return false;
        if (node.opcode != HTP_OP_ADD) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_ALLREDUCE) return false;

        auto * ar_kparams = (struct htp_allreduce_kernel_params *) last_node.kernel_params;
        const uint32_t rank    = (uint32_t) ar_kparams->rank;
        const uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        const ggml_tensor * ar_local = last_node.inputs[rank];
        const ggml_tensor * add_src0 = node.src0();
        const ggml_tensor * add_src1 = node.src1();
        const ggml_tensor * add_dst  = node.dst();

        if (!ggml_hexagon_tensor_is_fuseable(ar_local)) return false;

        const ggml_tensor * res_tensor;
        if (add_src0 == ar_local || add_src0->data == ar_local->data) {
            res_tensor = add_src1;
        } else if (add_src1 == ar_local || add_src1->data == ar_local->data) {
            res_tensor = add_src0;
        } else {
            return false;
        }

        if (ar_local->type != res_tensor->type) return false;

        const bool is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        const bool is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        if (!is_same_shape && !is_row_bcast) return false;

        if (is_same_shape) {
            if (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                return false;
            }
            if (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                return false;
            }
        }
        if (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) {
            return false;
        }

        for (uint32_t r = 0; r < n_ranks; r++) {
            const ggml_tensor * ar_src = last_node.inputs[r];
            if (ggml_hexagon_tensors_overlap(add_dst, ar_src)) {
                HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
                return false;
            }
        }

        struct htp_allreduce_kernel_params new_kparams;
        if (!ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, true, is_row_bcast, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            return false;
        }

        if (!try_fuse_common(res_tensor, add_dst)) {
            return false;
        }

        last_node.opcode = HTP_OP_ALLREDUCE_ADD;
        last_node.name   = "ALLREDUCE+ADD";
        last_node.inputs.push_back(res_tensor);
        last_node.outputs.clear();
        last_node.outputs.push_back(add_dst);
        last_node.fused.push_back(node.node);
        memcpy(last_node.kernel_params, &new_kparams, sizeof(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ADD;
        memcpy(o.kernel_params, &new_kparams, sizeof(new_kparams));

        o.src[2 * n_ranks] = add_tensor(res_tensor);
        o.dst[0]           = add_tensor(add_dst);
        for (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }

    bool try_fuse_rms_norm_mul(const htp_opnode & node) {
        if (n_ops == 0) return false;
        if (node.opcode != HTP_OP_MUL) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_RMS_NORM) return false;

        const ggml_tensor * mul_src0 = node.src0();
        const ggml_tensor * mul_src1 = node.src1();
        const ggml_tensor * rms_out  = last_node.dst();

        if (!ggml_hexagon_tensor_is_fuseable(rms_out)) return false;

        const ggml_tensor * weight;
        if (mul_src0 == rms_out || mul_src0->data == rms_out->data) {
            weight = mul_src1;
        } else if (mul_src1 == rms_out || mul_src1->data == rms_out->data) {
            weight = mul_src0;
        } else {
            return false;
        }

        const ggml_tensor * src0 = last_node.src0();

        if (src0->ne[0] != weight->ne[0] || src0->ne[0] != node.dst()->ne[0]) {
            return false;
        }

        const bool is_row_bcast = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
        const bool is_same_shape = (src0->ne[0] == weight->ne[0] && src0->ne[1] == weight->ne[1] &&
                                    src0->ne[2] == weight->ne[2] && src0->ne[3] == weight->ne[3]);
        if (!is_row_bcast && !is_same_shape) return false;

        if (!ggml_are_same_shape(src0, node.dst())) {
            return false;
        }
        // AE-10 (GGML_HEXAGON_ELT_SMALL bit 3): the FA Q-norm input is a strided view. Its rows are contiguous, the DSP row
        // kernel reads src0 rows with nb[1] (2-D DMA, src0_contig path), and dst is contiguous, so a strided src0 may fuse.
        if ((sess->elt_small & 8) != 0) {
            if (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous(node.dst())) {
                return false;
            }
        } else if (ggml_is_contiguous(src0) != ggml_is_contiguous(node.dst())) {
            return false;
        }

        struct htp_unary_kernel_params new_kparams;
        // Tile output (GGML_HEXAGON_ACT_TILE): only for producers the graph plan admitted.
        const bool tile_wanted = sess->act_tile_ok.count(node.dst()) != 0;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, weight, node.dst(), &new_kparams, tile_wanted
        );
        const bool tile_on = tile_wanted && new_kparams.vtcm_tile_size_per_thread != 0;

        if ((size_t) new_kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_size, sess->vtcm_size);
            return false;
        }

        if (!try_fuse_common(weight, node.dst())) {
            return false;
        }

        last_node.opcode = HTP_OP_RMS_NORM_MUL;
        last_node.name   = "RMS_NORM+MUL";
        last_node.inputs.clear();
        last_node.inputs.push_back(src0);
        last_node.inputs.push_back(weight);
        last_node.outputs.clear();
        last_node.outputs.push_back(node.dst());
        last_node.fused.push_back(node.node);
        memcpy(last_node.kernel_params, &new_kparams, sizeof(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        memcpy(o.kernel_params, &new_kparams, sizeof(new_kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(weight);
        for (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(node.dst());
        for (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }
        if (tile_on) {
            // dsts[1]: fp16 tile view of this producer's output; consumers read it through act_tile_src().
            const ggml_tensor * view = ggml_hexagon_act_tile_view(sess, node.dst()->ne[0], node.dst()->ne[1]);
            sess->act_tile_live[node.dst()] = view;
            o.dst[1] = add_tensor(view);
            if (sess->act_tile_skip_ok.count(node.dst()) != 0) {
                // No fp32 output: every reader is an HMX MUL_MAT (ggml_hexagon_act_tile_plan).
                o.flags |= HTP_OPFLAGS_ACT_TILE_ONLY;
                sess->act_tile_skip_live.insert(node.dst());
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }

    // Gated output norm (qwen35 build_norm_gated with LLAMA_S150_GDN_SWIGLU=1): GLU_SWIGLU(z, normalized) with normalized = RMS_NORM_MUL(x, w) becomes RMS_NORM_MUL_SILU(x, w, z) at the GLU's slot (GGML_HEXAGON_NORM_GATED=1).
    // The normalized tensor is tagged by the graph pass (its MUL feeds only the GLU). Kernel: htp/unary-ops.c op_rms_norm_mul_silu; its arithmetic is RMS_NORM_MUL followed by the SWIGLU primitives, so it is bit-identical.
    bool try_fuse_norm_gated(const htp_opnode & node) {
        if (n_ops == 0 || node.opcode != HTP_OP_GLU_SWIGLU || node.node == nullptr) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        htp_op_desc & o        = h_ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_RMS_NORM_MUL || last_node.inputs.size() != 2 || last_node.outputs.size() != 1 ||
            (o.flags & HTP_OPFLAGS_ACT_TILE_ONLY) != 0 || o.dst[1] != 0xffff) {
            return false;
        }

        const ggml_tensor * z       = node.src0();  // gate
        const ggml_tensor * nrm     = node.src1();  // RMS_NORM_MUL output (normalized)
        const ggml_tensor * out     = node.dst();
        const ggml_tensor * rms_out = last_node.dst();
        if (!z || !nrm || !out || !rms_out) return false;
        if (out->ne[2] * out->ne[3] < 64) return false; // prefill only: batches under 64 tokens keep the unfused chain
        if (!(nrm == rms_out || nrm->data == rms_out->data)) return false;
        if (!ggml_hexagon_tensor_is_fuseable(rms_out)) return false;

        const ggml_tensor * x = last_node.inputs[0];
        const ggml_tensor * w = last_node.inputs[1];
        if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || z->type != GGML_TYPE_F32 ||
            nrm->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32) {
            return false;
        }
        if (x->ne[0] == 0 || (x->ne[0] % 32) != 0 || w->ne[0] != x->ne[0] ||
            w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1) {
            return false;
        }
        if (!ggml_is_contiguous(x) || !ggml_is_contiguous(w) || !ggml_is_contiguous(z) ||
            !ggml_is_contiguous(nrm) || !ggml_is_contiguous(out)) {
            return false;
        }
        if (!ggml_are_same_shape(x, z) || !ggml_are_same_shape(x, nrm) || !ggml_are_same_shape(x, out)) {
            return false;
        }
        // the fused op writes out block by block while it reads x, w and z: only an exact alias is allowed
        for (const ggml_tensor * t : { x, w, z }) {
            if (ggml_hexagon_tensors_overlap(out, t) && !(out->data == t->data && ggml_nbytes(out) == ggml_nbytes(t))) {
                return false;
            }
        }
        if (!try_fuse_common({ z, out })) return false;

        last_node.opcode  = HTP_OP_RMS_NORM_MUL_SILU;
        last_node.name    = "RMS_NORM+MUL+SWIGLU";
        last_node.inputs  = { x, w, z };
        last_node.outputs = { out };
        last_node.fused.push_back(node.node);
        memset(last_node.kernel_params, 0, sizeof(last_node.kernel_params));

        o.opcode = HTP_OP_RMS_NORM_MUL_SILU;
        o.params[1] = ((int64_t) x->ne[2] * x->ne[3] >= PREFILL_MIN_ROWS) ? (opt_c2_ng | (opt_ng_sig16 ? 8 : 0) | (opt_ng_sig16 && opt_sig16_v2 ? 16 : 0) | ((opt_lm_norm & 3) << 12)) : 0;  // C2Elt: prefill batches only (n_seq_tokens * n_seqs >= 64); bit 3 = SIG16
        memset(o.kernel_params, 0, sizeof(o.kernel_params));
        o.src[0] = add_tensor(x);
        o.src[1] = add_tensor(w);
        o.src[2] = add_tensor(z);
        for (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(out);
        for (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        // fp16 tile output for the ssm_out matmul (GGML_HEXAGON_NG_TILE): fp32 output dropped, readers get the tile view
        auto ngt = sess->ngtile_ok.find(out);
        if (ngt != sess->ngtile_ok.end() && ggml_hexagon_act_tile_reserve(sess) &&
            (size_t) sess->n_threads * (32 * 4 * (size_t) out->ne[0] * 4 * 5 + 2 * 4 * (size_t) (out->ne[0] / 32) * 2048 + 4 * (size_t) out->ne[0] * 4) <= sess->vtcm_size) {
            const int64_t K = out->ne[0] * out->ne[1];
            const ggml_tensor * view = ggml_hexagon_act_tile_view(sess, K, out->ne[2]);
            o.dst[1] = add_tensor(view);
            o.flags |= HTP_OPFLAGS_ACT_TILE_ONLY;
            for (const ggml_tensor * v : ngt->second) {
                sess->act_tile_live[v] = view;
                sess->act_tile_skip_live.insert(v);
            }
            sess->act_tile_skip_live.insert(out);
            last_node.name = "RMS_NORM+MUL+SWIGLU(tile)";
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL+SWIGLU (#%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }

    // GDN's Q/K L2-norm (build_gdn_l2_norm, models.h: ggml_scale(ggml_rms_norm(x, eps/n), 1/sqrt(n))) is emitted as RMS_NORM followed by a scalar SCALE, not the codebase's native L2_NORM op (L2_NORM's own formula -- rsqrt(fmax(sum, eps)), no mean-divide -- is a different, incompatible normalization; see hvx_fast_l2_norm_f32 vs.
    // hvx_fast_rms_norm_f32 in hvx-norm.h). Structurally this mirrors try_fuse_rms_norm_mul above, but SCALE's multiplier/bias are host-side scalars baked into the SCALE node's own op_params (ggml_scale_impl), not a device tensor like MUL's weight operand -- so there is no second src to DMA in, and no new VTCM buffer beyond RMS_NORM's own.
    bool try_fuse_rms_norm_scale(const htp_opnode & node) {
        if (n_ops == 0) return false;
        if (node.opcode != HTP_OP_SCALE) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_RMS_NORM) return false;

        const ggml_tensor * scale_src0 = node.src0();
        const ggml_tensor * rms_out    = last_node.dst();

        if (scale_src0 != rms_out && scale_src0->data != rms_out->data) return false;
        if (!ggml_hexagon_tensor_is_fuseable(rms_out)) return false;

        const ggml_tensor * src0 = last_node.src0();

        // F32-only, same scope as RMS_NORM_MUL above: the fused kernel below reinterprets
        // rows as HVX_Vector f32 lanes directly, no F16 promote/narrow path exists for it.
        if (src0->type != GGML_TYPE_F32 || node.dst()->type != GGML_TYPE_F32) return false;

        if (!ggml_are_same_shape(rms_out, node.dst())) return false;
        if (ggml_is_contiguous(rms_out) != ggml_is_contiguous(node.dst())) return false;
        if (!ggml_are_same_shape(src0, node.dst())) return false;

        struct htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_SCALE, src0, nullptr, node.dst(), &new_kparams
        );

        if ((size_t) new_kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_SCALE fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_size, sess->vtcm_size);
            return false;
        }

        if (!try_fuse_common({node.dst()})) {
            return false;
        }

        float scale_bias[2];
        memcpy(scale_bias, node.node->op_params, sizeof(scale_bias));

        last_node.opcode = HTP_OP_RMS_NORM_SCALE;
        last_node.name   = "RMS_NORM+SCALE";
        last_node.inputs.clear();
        last_node.inputs.push_back(src0);
        last_node.outputs.clear();
        last_node.outputs.push_back(node.dst());
        last_node.fused.push_back(node.node);
        memcpy(last_node.kernel_params, &new_kparams, sizeof(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_SCALE;
        // params[0] (eps) is already correct from add_op()'s original RMS_NORM memcpy; append SCALE's own scale/bias in params[1]/params[2], matching rms_norm_scale_f32's op_params[0]/[1]/[2] read order.
        memcpy(&o.params[1], scale_bias, sizeof(scale_bias));
        o.params[3] = ((int64_t) src0->ne[1] * src0->ne[2] * src0->ne[3] >= 1024) ? opt_c2_rs : 0;  // C2Elt: prefill rows only
        memcpy(o.kernel_params, &new_kparams, sizeof(new_kparams));

        o.src[0] = add_tensor(src0);
        for (uint32_t s = 1; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(node.dst());
        for (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+SCALE (#%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }

    bool try_fuse_mul_mat_add(const htp_opnode & node) {
        if (n_ops == 0) return false;
        if (node.opcode != HTP_OP_ADD) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode == HTP_OP_MUL_MAT_NX) return try_fuse_nx_bias(node);  // MM-02: bias of the NX's last weight
        if (last_node.opcode != HTP_OP_MUL_MAT) return false;

        const ggml_tensor * add_src0 = node.src0();
        const ggml_tensor * add_src1 = node.src1();
        const ggml_tensor * mm_out   = last_node.dst();

        if (!ggml_hexagon_tensor_is_fuseable(mm_out)) return false;

        const ggml_tensor * src2;
        if (add_src0 == mm_out || add_src0->data == mm_out->data) {
            src2 = add_src1;
        } else if (add_src1 == mm_out || add_src1->data == mm_out->data) {
            src2 = add_src0;
        } else {
            return false;
        }

        const ggml_tensor * src0 = last_node.src0();
        const ggml_tensor * src1 = last_node.src1();

        if (src2->type != GGML_TYPE_F32) return false;

        const struct htp_mm_kernel_params * orig_kparams = (const struct htp_mm_kernel_params *) last_node.kernel_params;
        struct htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_add_params(sess, src0, src1, src2, node.dst(), &kparams);
        if (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            return false;
        }
        if (sess->act_tile_skip_live.count(src1) != 0 && !act_tile_mm_hmx(&kparams)) {
            return false;  // a dropped-fp32 activation needs its fused consumer on the HMX kernel
        }

        const int src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        const bool can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        if (!can_fuse) return false;

        if ((size_t) kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
            return false;
        }

        if (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            if (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_threads < orig_kparams->n_act_threads) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_threads, kparams.n_act_threads);
                return false;
            }
        }

        if (!try_fuse_common(src2, node.dst())) {
            return false;
        }

        last_node.opcode = HTP_OP_MUL_MAT_ADD;
        last_node.name   = "MUL_MAT+ADD";
        last_node.inputs.clear();
        last_node.inputs.push_back(src0);
        last_node.inputs.push_back(src1);
        last_node.inputs.push_back(src2);
        last_node.outputs.clear();
        last_node.outputs.push_back(node.dst());
        last_node.fused.push_back(node.node);
        memcpy(last_node.kernel_params, &kparams, sizeof(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ADD;
        memcpy(o.kernel_params, &kparams, sizeof(kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(act_tile_mm_hmx(&kparams) ? act_tile_src(src1) : src1);
        o.src[2] = add_tensor(src2);
        for (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(node.dst());
        for (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }

    // MM-02 (GGML_HEXAGON_NX_ALPHABETA): a MUL_MAT+ADD whose MUL_MAT is the last weight of a MUL_MAT_NX (ssm_alpha joined the qkv+z+beta group) takes the ADD's broadcast row (ssm_dt) as the bias of that weight.
    // The DSP adds it in the HMX output epilogue of the last weight (src[n_weights + 1]); that weight's NX output becomes the ADD output.
    bool try_fuse_nx_bias(const htp_opnode & node) {
        if (!opt_nx_alphabeta || n_ops == 0 || node.opcode != HTP_OP_ADD) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_MUL_MAT_NX || (h_ops[n_ops - 1].flags & HTP_OPFLAGS_SWIGLU_OUT) != 0) return false;

        const uint32_t n_w = (uint32_t) last_node.outputs.size();
        if (n_w == 0 || n_w > HTP_OP_MAX_OUTPUTS || last_node.inputs.size() != n_w + 1) return false;  // n_w = 4 (qkv, z, beta, alpha) is valid

        const ggml_tensor * mm_out   = last_node.outputs.back();
        const ggml_tensor * add_src0 = node.src0();
        const ggml_tensor * add_src1 = node.src1();
        const ggml_tensor * src2;
        if (add_src0 == mm_out || add_src0->data == mm_out->data) {
            src2 = add_src1;
        } else if (add_src1 == mm_out || add_src1->data == mm_out->data) {
            src2 = add_src0;
        } else {
            return false;
        }

        if (!ggml_hexagon_tensor_is_fuseable(mm_out)) return false;
        if (src2->type != GGML_TYPE_F32 || src2->ne[1] != 1 || src2->ne[2] != 1 || src2->ne[3] != 1) return false;
        if (src2->ne[0] != mm_out->ne[0] || !ggml_is_contiguous(src2)) return false;

        const struct htp_mm_kernel_params * orig = (const struct htp_mm_kernel_params *) last_node.kernel_params;
        if (orig->vtcm_src2_size != 0 || orig->n_hmx == 0 || orig->pipeline == 0) return false;

        const ggml_tensor * w0 = last_node.inputs[0];
        const ggml_tensor * x  = last_node.inputs[n_w];
        if (ggml_hexagon_src1_rows(x) < GGML_HEXAGON_MM_PREFILL_MIN_ROWS) return false;  // decode and draft batches keep the original path
        const size_t src2_size = hex_round_up(ggml_nbytes(src2), 128);

        struct htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, (int32_t) n_w, src2_size, &kparams);
        if (!is_supported_mul_mat_nx_kernel(w0, &kparams) || kparams.n_hmx == 0 || kparams.pipeline == 0) return false;
        if (sess->act_tile_skip_live.count(x) != 0 && !act_tile_mm_hmx(&kparams)) return false;
        if (kparams.m_chunk < orig->m_chunk || kparams.n_chunk < orig->n_chunk || kparams.n_act_threads < orig->n_act_threads) {
            return false;  // HMX efficiency would drop (same rule as the standalone MUL_MAT+ADD fusion)
        }
        if ((size_t) kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip NX bias fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
            return false;
        }
        if (!try_fuse_common(src2, node.dst())) return false;

        last_node.inputs.push_back(src2);
        last_node.outputs.back() = node.dst();
        last_node.fused.push_back(node.node);
        memcpy(last_node.kernel_params, &kparams, sizeof(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        memcpy(o.kernel_params, &kparams, sizeof(kparams));
        o.src[n_w + 1] = add_tensor(src2);
        o.dst[n_w - 1] = add_tensor(node.dst());

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX+ADD (bias on weight %u, #%u)\n", sess->c_name(), n_w - 1, n_ops - 1);
        return true;
    }

    bool try_fuse_mul_mat_nx(const htp_opnode & node) {
        if (n_ops == 0 || node.opcode != HTP_OP_MUL_MAT) return false;
        if (!is_mergeable_mul_mat(node.node)) return false;

        const ggml_tensor * w_in = node.src0();
        const ggml_tensor * x_in = node.src1();
        const ggml_tensor * d_in = node.dst();

        htp_opnode & last_node = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        if (last_node.opcode == HTP_OP_MUL_MAT_NX) {
            if ((h_ops[n_ops - 1].flags & HTP_OPFLAGS_SWIGLU_OUT) != 0) {
                return false;  // a SWIGLU-output NX takes no further weights
            }
            if (opt_nx_alphabeta && ((const struct htp_mm_kernel_params *) last_node.kernel_params)->vtcm_src2_size != 0) {
                return false;  // the last weight carries a fused bias (try_fuse_nx_bias): nothing may follow it
            }
            const uint32_t curr_n = (uint32_t) last_node.outputs.size();
            if (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                return false;
            }

            const ggml_tensor * w0 = last_node.inputs[0];
            const ggml_tensor * x  = last_node.inputs[curr_n];

            if (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0]) {
                return false;
            }
            if (!last_node.fused.empty() && (mm_is_hmx_eligible(last_node.fused[0]) != mm_is_hmx_eligible(node.node))) {
                return false;
            }

            struct htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, 0, &kparams);
            if (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                return false;
            }
            if (sess->act_tile_skip_live.count(x) != 0 && !act_tile_mm_hmx(&kparams)) {
                return false;
            }
            if ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                return false;
            }

            if (!try_fuse_common(w_in, d_in)) {
                return false;
            }

            last_node.inputs[curr_n] = w_in;
            last_node.inputs.push_back(x);
            last_node.outputs.push_back(d_in);
            last_node.fused.push_back(node.node);
            memcpy(last_node.kernel_params, &kparams, sizeof(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, sizeof(kparams));

            for (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = add_tensor(act_tile_mm_hmx(&kparams) ? act_tile_src(last_node.inputs[s]) : last_node.inputs[s]);
            }
            for (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            for (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_node.outputs[d]);
            }
            for (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            return true;
        }

        // Case 2: last_node is single MUL_MAT
        if (last_node.opcode == HTP_OP_MUL_MAT) {
            if (!is_mergeable_mul_mat_pair(last_node.node, node.node)) {
                return false;
            }

            const ggml_tensor * w0 = last_node.src0();
            const ggml_tensor * x  = last_node.src1();
            const ggml_tensor * w1 = node.src0();

            struct htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, 0, &kparams);
            if (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                return false;
            }
            if (sess->act_tile_skip_live.count(x) != 0 && !act_tile_mm_hmx(&kparams)) {
                return false;
            }
            if ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                return false;
            }

            if (!try_fuse_common(w1, node.dst())) {
                return false;
            }

            const ggml_tensor * dst_0 = last_node.dst();
            const ggml_tensor * dst_1 = node.dst();

            last_node.opcode = HTP_OP_MUL_MAT_NX;
            last_node.name   = "MUL_MAT_NX";
            last_node.inputs.clear();
            last_node.inputs.push_back(w0);
            last_node.inputs.push_back(w1);
            last_node.inputs.push_back(x);
            last_node.outputs.clear();
            last_node.outputs.push_back(dst_0);
            last_node.outputs.push_back(dst_1);
            last_node.fused.push_back(node.node);
            memcpy(last_node.kernel_params, &kparams, sizeof(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            memcpy(o.kernel_params, &kparams, sizeof(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(act_tile_mm_hmx(&kparams) ? act_tile_src(x) : x);
            for (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            for (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            return true;
        }

        return false;
    }

    // FFN SWIGLU output (GGML_HEXAGON_SWIGLU_OUT, admitted by ggml_hexagon_swiglu_out_plan): the gate/up NX writes silu(g)*u as
    // fp16 act tiles. The GLU op and the gate/up fp32 outputs disappear, and the FFN down reads the tiles through act_tile_live.
    bool try_fuse_swiglu_out(const htp_opnode & node) {
        if (n_ops == 0 || node.opcode != HTP_OP_GLU_SWIGLU || node.node == nullptr) return false;
        if (!swiglu_out_rows_ok(node.node->ne[1])) return false;
        if (sess->swiglu_ok.count(node.node) == 0) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        htp_op_desc & o        = h_ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_MUL_MAT_NX || last_node.outputs.size() != 2 || last_node.inputs.size() != 3 ||
            (o.flags & HTP_OPFLAGS_SWIGLU_OUT) != 0) {
            return false;
        }

        // NX outputs are (dst of inputs[0], dst of inputs[1]); the GLU takes gate = src0 and up = src1.
        const ggml_tensor * out      = node.node;
        const ggml_tensor * gate_out = out->src[0];
        const ggml_tensor * up_out   = out->src[1];
        int g = -1;
        if (last_node.outputs[0] == gate_out) {
            g = 0;
        } else if (last_node.outputs[1] == gate_out) {
            g = 1;
        }
        if (g < 0 || last_node.outputs[1 - g] != up_out) return false;

        const ggml_tensor * w_gate = last_node.inputs[g];
        const ggml_tensor * w_up   = last_node.inputs[1 - g];
        const ggml_tensor * x      = last_node.inputs[2];
        if (w_up->type != w_gate->type || w_up->ne[0] != w_gate->ne[0] || w_up->ne[1] != w_gate->ne[1] ||
            out->ne[0] != w_gate->ne[1]) {
            return false;
        }
        if ((size_t) hex_round_up((int) out->ne[1], 32) * (size_t) out->ne[0] * sizeof(uint16_t) > GGML_HEXAGON_SWIGLU_OUT_BYTES) {
            return false;
        }

        struct htp_mm_kernel_params kparams;
        if (!ggml_hexagon_precompute_swiglu_nx_params(sess, w_gate, x, &kparams) ||
            !is_supported_mul_mat_nx_kernel(w_gate, &kparams)) {
            return false;
        }
        if ((size_t) kparams.vtcm_size > sess->vtcm_size || !ggml_hexagon_swiglu_reserve(sess)) {
            return false;
        }
        const ggml_tensor * view = ggml_hexagon_swiglu_view(sess, out->ne[0], out->ne[1]);
        if (!try_fuse_common({view})) return false;

        last_node.name = "MUL_MAT_NX+SWIGLU";
        last_node.inputs.clear();
        last_node.inputs.push_back(w_gate);
        last_node.inputs.push_back(w_up);
        last_node.inputs.push_back(x);
        last_node.outputs.clear();
        last_node.outputs.push_back(out);
        last_node.fused.push_back(node.node);
        memcpy(last_node.kernel_params, &kparams, sizeof(kparams));

        o.flags |= HTP_OPFLAGS_SWIGLU_OUT;
        if ((opt_lx_fuse & 1) != 0) o.flags |= HTP_OPFLAGS_SWIGLU_NOBAR;
        if (opt_swiglu_sig16) o.flags |= HTP_OPFLAGS_SWIGLU_SIG16;
        o.flags &= ~opt_mm_a8w8_flags;
        if (opt_mm_a8w8_ops & 1) o.flags |= opt_mm_a8w8_flags;
        if (opt_mm_a8w8_ops & 1) { static const int dbg = getenv("GGML_HEXAGON_A8W8_DBG") ? atoi(getenv("GGML_HEXAGON_A8W8_DBG")) : 0; o.params[3] = dbg; static const float clip = getenv("GGML_HEXAGON_A8W8_CLIP") ? (float) atof(getenv("GGML_HEXAGON_A8W8_CLIP")) : 1.0f; memcpy(&o.params[2], &clip, sizeof(clip)); }
        memcpy(o.kernel_params, &kparams, sizeof(kparams));
        o.src[0] = add_tensor(w_gate);
        o.src[1] = add_tensor(w_up);
        o.src[2] = add_tensor(act_tile_mm_hmx(&kparams) ? act_tile_src(x) : x);
        for (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(view);
        for (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        // The FFN down reads the tiles: its src1 (the GLU output) maps to the view through act_tile_src.
        sess->act_tile_live[node.node] = view;
        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX+SWIGLU (#%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }

    // Fuses two independent ROPE calls (e.g. Requires the two ROPE nodes to already be adjacent in the op list (see qwen35.cpp's build_layer_attn forced-expand hint) -- this matcher never reorders nodes itself, it only recognizes an adjacency that already exists.
    // Q and K) sharing the same inp_pos/ freq_factors/op_params into one ROPE_NX dispatch that computes the theta_cache once instead of once per tensor.
    bool try_fuse_rope_nx(const htp_opnode & node) {
        if (sess->mdev.count > 1) return false;
        if (n_ops == 0 || node.opcode != HTP_OP_ROPE) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_ROPE) return false;

        if (!is_mergeable_rope_pair(last_node.node, node.node)) return false;

        const ggml_tensor * a_in  = last_node.node->src[0];
        const ggml_tensor * pos   = last_node.node->src[1];
        const ggml_tensor * ff    = last_node.node->src[2]; // may be null
        const ggml_tensor * a_out = last_node.dst();
        const ggml_tensor * b_in  = node.node->src[0];
        const ggml_tensor * b_out = node.dst();
        if (!a_in || !pos || !a_out || !b_in || !b_out) return false;

        // Authoritative cap: HTP_ROPE_SPAD_NSLOTS*HTP_ROPE_SPAD_BLOCK (rope-ops.h) -- each token's row block for either tensor must fit the DSP-side per-token spad in one shot (no chunking).
        // Duplicated here as a literal since that constant is private to rope-ops.c/.h; matches the check execute_op_rope_nx_f32 makes again on-device.
        constexpr int64_t k_rope_nx_max_heads = 32; // HTP_ROPE_SPAD_NSLOTS(4) * HTP_ROPE_SPAD_BLOCK(8)
        if (a_in->ne[1] > k_rope_nx_max_heads || b_in->ne[1] > k_rope_nx_max_heads) {
            return false;
        }

        if (!try_fuse_common({b_in, b_out})) {
            return false;
        }

        last_node.opcode = HTP_OP_ROPE_NX;
        last_node.name   = "ROPE_NX";
        last_node.inputs.clear();
        last_node.inputs.push_back(a_in);
        last_node.inputs.push_back(pos);
        last_node.inputs.push_back(ff);
        last_node.inputs.push_back(b_in);
        last_node.outputs.clear();
        last_node.outputs.push_back(a_out);
        last_node.outputs.push_back(b_out);
        last_node.fused.push_back(node.node);
        // kernel_params unused by ROPE/ROPE_NX (op_params carries everything); leave as-is.

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ROPE_NX;

        o.src[0] = add_tensor(a_in);
        o.src[1] = add_tensor(pos);
        o.src[2] = ff ? add_tensor(ff) : 0xffff;
        o.src[3] = add_tensor(b_in);
        for (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(a_out);
        o.dst[1] = add_tensor(b_out);
        for (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        if (opt_rope_norm) try_fold_norms_into_rope_nx();

        HEX_VERBOSE("ggml-hex: %s fused ROPE_NX (#%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }

    // GGML_HEXAGON_ROPE_KV: [ROPE_NX, SET_ROWS(K cache), SET_ROWS(V cache)] -> one ROPE_NX that also stores the rotated K rows and the V rows into the fp16 caches (the same fp32 -> fp16 conversion and row scatter as SET_ROWS). it looks at [ROPE_NX, SET_ROWS(K)] + node(SET_ROWS V).
    // Called for each SET_ROWS after the ROPE_NX: the first one (K) is held back as a pending record by returning false only when the pair cannot complete, so the matcher folds only the V op, when ops[n-1] is the SET_ROWS(K) that was enqueued behind the ROPE_NX; i.e.
    bool try_fuse_rope_kv(const htp_opnode & node) {
        if (sess->mdev.count > 1 || n_ops < 2 || node.opcode != HTP_OP_SET_ROWS) return false;
        htp_opnode & kset = ops[n_ops - 1];
        htp_opnode & rope = ops[n_ops - 2];
        if (kset.opcode != HTP_OP_SET_ROWS || rope.opcode != HTP_OP_ROPE_NX) return false;
        if (rec_active && (n_ops - 2) < rec_plan_sig) return false;
        htp_op_desc & hr = h_ops[n_ops - 2];
        if ((hr.flags & HTP_OPFLAGS_ROPE_KV) != 0 || rope.outputs.size() != 2 || rope.inputs.size() > 6) return false;

        const ggml_tensor * k_rot = rope.outputs[1];                 // rotated K, [256, n_kv_heads, T]
        const ggml_tensor * k_src = kset.node->src[0];               // 2-D view of k_rot  [n_kv_heads * 256, T]
        const ggml_tensor * k_idx = kset.node->src[1];
        const ggml_tensor * k_cache = kset.dst();
        const ggml_tensor * v_src = node.node->src[0];
        const ggml_tensor * v_idx = node.node->src[1];
        const ggml_tensor * v_cache = node.dst();
        if (!k_src || !k_idx || !k_cache || !v_src || !v_idx || !v_cache) return false;
        if (k_src->data != k_rot->data || k_src->ne[1] != k_rot->ne[2] || k_src->ne[0] != k_rot->ne[0] * k_rot->ne[1]) return false;
        if (k_rot->nb[1] != k_rot->ne[0] * sizeof(float) || k_rot->ne[3] != 1) return false;
        if (k_cache->type != GGML_TYPE_F16 || v_cache->type != GGML_TYPE_F16 || k_cache->ne[0] != k_src->ne[0] || v_cache->ne[0] != v_src->ne[0]) return false;
        if (k_idx->type != GGML_TYPE_I64 || v_idx->type != GGML_TYPE_I64 || k_idx->ne[0] != k_src->ne[1] || v_idx->ne[0] != v_src->ne[1]) return false;
        if (k_src->ne[2] != 1 || k_src->ne[3] != 1 || v_src->ne[2] != 1 || v_src->ne[3] != 1 || k_idx->ne[1] != 1 || v_idx->ne[1] != 1) return false;
        if (v_src->type != GGML_TYPE_F32 || v_src->nb[0] != sizeof(float) || v_src->ne[0] % 32 != 0 || v_src->ne[1] != k_src->ne[1]) return false;
        if (k_cache->ne[2] != 1 || k_cache->ne[3] != 1 || v_cache->ne[2] != 1 || v_cache->ne[3] != 1) return false;
        // the V rows must not be written by an earlier op of this batch after the ROPE_NX started (V comes from the NX before the rope): data is ready
        if (!try_fuse_common({v_src, v_idx, k_idx, k_cache, v_cache})) return false;
        if (n_ops < 2 || rope.inputs.size() != 4 + (hr.src[4] != 0xffff ? 2 : 0)) return false;

        // inputs: ... [4]=wq [5]=wk (ROPE_NORM) or none; src[6]=K idx, src[7]=V rows, src[8]=V idx
        hr.src[6] = add_tensor(k_idx);
        hr.src[7] = add_tensor(v_src);
        hr.src[8] = add_tensor(v_idx);
        hr.src[9] = 0xffff;
        hr.dst[2] = add_tensor(k_cache);
        hr.dst[3] = add_tensor(v_cache);
        hr.flags |= HTP_OPFLAGS_ROPE_KV;
        rope.inputs.resize(6, nullptr);
        rope.inputs.push_back(k_idx);
        rope.inputs.push_back(v_src);
        rope.inputs.push_back(v_idx);
        rope.outputs.push_back(k_cache);
        rope.outputs.push_back(v_cache);
        rope.fused.push_back(kset.node);
        rope.fused.push_back(node.node);
        rope.name += "+KV";
        n_ops -= 1;   // drop the SET_ROWS(K) record (the V op was never added)
        HEX_VERBOSE("ggml-hex: %s folded K/V SET_ROWS into ROPE_NX (#%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }

    // GGML_HEXAGON_ROPE_NORM: [RMS_NORM+MUL(q), RMS_NORM+MUL(k), ROPE_NX] -> one ROPE_NX that normalises the loaded rows first
    // (src[4] / src[5] = the two norm weights, op eps in kernel_params[0]). The two norm ops are removed from the batch.
    bool try_fold_norms_into_rope_nx() {
        if (n_ops < 3) return false;
        const unsigned L = n_ops - 1;
        if (rec_active && (L - 2) < rec_plan_sig) return false;
        htp_opnode & rope = ops[L];
        htp_opnode & nq   = ops[L - 2];
        htp_opnode & nk   = ops[L - 1];
        htp_op_desc & hr  = h_ops[L];
        htp_op_desc & hq  = h_ops[L - 2];
        htp_op_desc & hk  = h_ops[L - 1];
        if (rope.opcode != HTP_OP_ROPE_NX || nq.opcode != HTP_OP_RMS_NORM_MUL || nk.opcode != HTP_OP_RMS_NORM_MUL) return false;
        if (nq.inputs.size() != 2 || nk.inputs.size() != 2 || nq.outputs.size() != 1 || nk.outputs.size() != 1) return false;
        if (hq.dst[1] != 0xffff || hk.dst[1] != 0xffff || (hq.flags & HTP_OPFLAGS_ACT_TILE_ONLY) || (hk.flags & HTP_OPFLAGS_ACT_TILE_ONLY)) return false;
        const ggml_tensor * a_in = rope.inputs[0];
        const ggml_tensor * b_in = rope.inputs[3];
        if (nq.outputs[0] != a_in || nk.outputs[0] != b_in) return false;
        if (!sess->norm_rope_only.count(a_in) || !sess->norm_rope_only.count(b_in)) return false;
        if (hq.params[0] != hk.params[0]) return false;                       // same epsilon
        const ggml_tensor * qs = nq.inputs[0];
        const ggml_tensor * ks = nk.inputs[0];
        const ggml_tensor * wq = nq.inputs[1];
        const ggml_tensor * wk = nk.inputs[1];
        for (const ggml_tensor * t : { qs, ks }) {
            if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || t->ne[0] != 256 || t->ne[3] != 1) return false;
            if (t->nb[1] % 128 != 0 && t->ne[1] > 1) { /* DMA rows only need the byte stride to fit */ }
        }
        for (const ggml_tensor * t : { wq, wk }) {
            if (t->type != GGML_TYPE_F32 || t->ne[0] != 256 || t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) return false;
        }
        if (qs->ne[1] != a_in->ne[1] || qs->ne[2] != a_in->ne[2] || ks->ne[1] != b_in->ne[1] || ks->ne[2] != b_in->ne[2]) return false;
        // the fused op reads the pre-norm rows and writes the rotated rows: they may share memory only in place, token for token
        const ggml_tensor * qd = rope.outputs[0];
        const ggml_tensor * kd = rope.outputs[1];
        auto overlap_ok = [](const ggml_tensor * src, const ggml_tensor * dst) {
            return !ggml_hexagon_tensors_overlap(src, dst) ||
                   (src->data == dst->data && src->nb[1] == dst->nb[1] && src->nb[2] == dst->nb[2]);
        };
        if (!overlap_ok(qs, qd) || !overlap_ok(ks, kd) || ggml_hexagon_tensors_overlap(qs, kd) || ggml_hexagon_tensors_overlap(ks, qd)) return false;
        // the weights are small, resident for the op
        if (!try_fuse_common({qs, ks, wq, wk})) return false;

        const ggml_tensor * pos = rope.inputs[1];
        const ggml_tensor * ff  = rope.inputs[2];
        std::vector<ggml_tensor *> fused;
        fused.push_back(nq.node);  for (auto * f : nq.fused) fused.push_back(f);
        fused.push_back(nk.node);  for (auto * f : nk.fused) fused.push_back(f);
        for (auto * f : rope.fused) fused.push_back(f);

        rope.inputs.clear();
        rope.inputs.push_back(qs);
        rope.inputs.push_back(pos);
        rope.inputs.push_back(ff);
        rope.inputs.push_back(ks);
        rope.inputs.push_back(wq);
        rope.inputs.push_back(wk);
        rope.fused = fused;
        rope.name  = "ROPE_NX+NORM";

        hr.flags |= HTP_OPFLAGS_ROPE_NORM;
        memcpy(&hr.kernel_params[0], &hq.params[0], sizeof(float));
        hr.src[0] = add_tensor(qs);
        hr.src[3] = add_tensor(ks);
        hr.src[4] = add_tensor(wq);
        hr.src[5] = add_tensor(wk);
        for (uint32_t s = 6; s < HTP_OP_MAX_INPUTS; s++) hr.src[s] = 0xffff;

        // drop the two norm ops (batch order: the rope op moves down two slots)
        ops[L - 2]   = std::move(ops[L]);
        h_ops[L - 2] = hr;
        n_ops -= 2;
        HEX_VERBOSE("ggml-hex: %s folded q/k RMS_NORM+MUL into ROPE_NX (#%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }

    bool try_fuse_mul_mat_id_nx(const htp_opnode & node) {
        if (n_ops == 0 || node.opcode != HTP_OP_MUL_MAT_ID) return false;
        if (!is_mergeable_mul_mat_id(node.node)) return false;

        const ggml_tensor * w_in   = node.src0();
        const ggml_tensor * x_in   = node.src1();
        const ggml_tensor * ids_in = node.node->src[2];
        const ggml_tensor * d_in   = node.dst();

        htp_opnode & last_node = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        if (last_node.opcode == HTP_OP_MUL_MAT_ID_NX) {
            const uint32_t curr_n = (uint32_t) last_node.outputs.size();
            if (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                return false;
            }

            const ggml_tensor * w0  = last_node.inputs[0];
            const ggml_tensor * x   = last_node.inputs[curr_n];
            const ggml_tensor * ids = last_node.inputs[curr_n + 1];

            if (x_in != x || ids_in != ids || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2]) {
                return false;
            }
            if (!last_node.fused.empty() && (mm_is_hmx_eligible(last_node.fused[0]) != mm_is_hmx_eligible(node.node))) {
                return false;
            }

            struct htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
            if (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                return false;
            }
            if ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                return false;
            }

            if (!try_fuse_common(w_in, d_in)) {
                return false;
            }

            last_node.inputs[curr_n] = w_in;
            last_node.inputs[curr_n + 1] = x;
            last_node.inputs.push_back(ids);
            last_node.outputs.push_back(d_in);
            last_node.fused.push_back(node.node);
            memcpy(last_node.kernel_params, &kparams, sizeof(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, sizeof(kparams));

            for (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = add_tensor(last_node.inputs[s]);
            }
            for (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            for (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_node.outputs[d]);
            }
            for (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            return true;
        }

        // Case 2: last_node is single MUL_MAT_ID
        if (last_node.opcode == HTP_OP_MUL_MAT_ID) {
            if (!is_mergeable_mul_mat_id_pair(last_node.node, node.node)) {
                return false;
            }

            const ggml_tensor * w0  = last_node.src0();
            const ggml_tensor * x   = last_node.src1();
            const ggml_tensor * ids = last_node.node->src[2];
            const ggml_tensor * w1  = node.src0();

            struct htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, node.dst(), 2, &kparams);
            if (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                return false;
            }
            if ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                return false;
            }

            if (!try_fuse_common(w1, node.dst())) {
                return false;
            }

            const ggml_tensor * dst_0 = last_node.dst();
            const ggml_tensor * dst_1 = node.dst();

            last_node.opcode = HTP_OP_MUL_MAT_ID_NX;
            last_node.name   = "MUL_MAT_ID_NX";
            last_node.inputs.clear();
            last_node.inputs.push_back(w0);
            last_node.inputs.push_back(w1);
            last_node.inputs.push_back(x);
            last_node.inputs.push_back(ids);
            last_node.outputs.clear();
            last_node.outputs.push_back(dst_0);
            last_node.outputs.push_back(dst_1);
            last_node.fused.push_back(node.node);
            memcpy(last_node.kernel_params, &kparams, sizeof(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            memcpy(o.kernel_params, &kparams, sizeof(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            o.src[3] = add_tensor(ids);
            for (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            for (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            return true;
        }

        return false;
    }


    // SSM_CONV(_SILU): the GDN conv-prep block always runs conv1d straight into silu with no bias in between (qwen35.cpp: conv_output_silu = ggml_silu(ctx0, conv_output_proper)). Restricted to the HVX prefill path (n_t > 1): the decode worker (ssm_conv_thread_f32_decode) has no SILU epilogue implemented (mirrors the private fork's own precedent of excluding decode -- its scalar per-token path would need a per-element libm expf() call instead of a vectorized guard, measured there as a ~5% tg128 regression when attempted).
    // Decode's SSM_CONV cost is negligible either way (~8-9 usec/call in this model's own profile) so excluding it costs nothing. The DSP-side kernel already supports fusing them (HTP_OP_SSM_CONV_SILU dispatches to op_ssm_conv_f32 with scctx.apply_silu=true, writing silu(y)=y*sigmoid(y) directly at the conv's own output-store site instead of round-tripping the [d_inner, n_t] conv output tensor through a separate HTP dispatch) -- this matcher wires that already- built kernel path to the graph's own SSM_CONV->SILU adjacency.
    bool try_fuse_ssm_conv_silu(const htp_opnode & node) {
        if (sess->mdev.count > 1) return false;
        if (n_ops == 0) return false;
        if (node.opcode != HTP_OP_UNARY_SILU) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_SSM_CONV) return false;
        if (last_node.outputs.size() != 1) return false;

        const ggml_tensor * conv_out  = last_node.dst();
        const ggml_tensor * silu_node = node.node;
        const ggml_tensor * silu_src  = node.src0();
        const ggml_tensor * silu_dst  = node.dst();

        if (!silu_src || !silu_dst || !silu_dst->data) return false;
        if (silu_src != conv_out) return false;
        if (conv_out->type != GGML_TYPE_F32 || silu_dst->type != GGML_TYPE_F32) return false;
        if ((conv_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (silu_node->flags & GGML_TENSOR_FLAG_OUTPUT)) return false;
        if (!ggml_hexagon_tensor_is_fuseable(conv_out)) return false; // must be SSM_CONV's only real consumer

        // Decode gate: n_t (conv_out->ne[1]) > 1 required, matching op_ssm_conv_f32's own apply_silu decode-path defensive NO-SUPPORT guard (ssm-conv.c) -- an n_t==1 shape safely skips this fusion instead of ever reaching that guard.
        if (conv_out->ne[1] <= 1) return false;

        // Same-shape sanity (SSM_CONV's dst and SILU's dst are always the same shape by construction -- ggml_unary_impl's ggml_dup_tensor -- but check explicitly rather than assume, since this is exactly the kind of precondition a future refactor could silently break).
        if (!ggml_are_same_shape(conv_out, silu_dst) || !ggml_is_contiguous(silu_dst)) return false;

        if (!try_fuse_common({silu_dst})) {
            return false;
        }

        last_node.opcode = HTP_OP_SSM_CONV_SILU;
        last_node.name    = "SSM_CONV+SILU";
        last_node.outputs.clear();
        last_node.outputs.push_back(silu_dst);
        last_node.fused.push_back(node.node);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_SSM_CONV_SILU;
        // kernel_params (VTCM tile sizes, d_inner/n_t/d_conv, etc.) were already precomputed by ggml_hexagon_precompute_ssm_conv_params for the unfused SSM_CONV node and stay valid unchanged -- they depend only on tensor shapes, which are identical between conv_out and silu_dst (checked above), not on which buffer the output lands in.
        o.dst[0] = add_tensor(silu_dst);
        for (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused SSM_CONV+SILU (#%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }

    // CONCAT(conv_states, TRANSPOSE(qkv_mixed)) -> SSM_CONV(_SILU): the GDN conv-input build materializes conv_input to DDR via CONCAT purely so SSM_CONV can read it right back and re-transpose it into VTCM for its own staging -- CONCAT's own HVX gather- transpose and SSM_CONV's transpose_src0_block are literal mathematical inverses for the qkv_mixed portion of the tensor (qkv_mixed's pre-transpose layout is already token-major/channel-contiguous, exactly what SSM_CONV's VTCM staging wants).
    // Prerequisite (src/models/delta-net-base.cpp + qwen35.cpp's build_conv_state / build_layer_attn_linear graph-reorder, landed separately and verified neutral in isolation): CONCAT+SSM_CONV must be dispatch-adjacent and CONCAT must have exactly one real consumer for this fusion to trigger -- true for every prefill call (n_tokens >= conv_kernel_size - 1) once that reorder redirects the state-update CPY to read from qkv_mixed directly instead of from conv_input.
    bool try_fuse_concat_ssm_conv(const htp_opnode & node) {
        if (sess->mdev.count > 1) return false;
        if (n_ops == 0) return false;
        if (node.opcode != HTP_OP_SSM_CONV && node.opcode != HTP_OP_SSM_CONV_SILU) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_CONCAT) return false;
        if (ggml_get_op_params_i32(last_node.node, 0) != 0) return false; // must concat along dim 0

        const ggml_tensor * conv_input     = last_node.dst();
        const ggml_tensor * conv_states     = last_node.src0();
        const ggml_tensor * qkv_transposed = last_node.src1();
        if (!conv_input || !conv_states || !qkv_transposed) return false;
        if (last_node.node->src[2]) return false; // CONCAT with conv ring offsets is never fused

        const ggml_tensor * conv_ssm_src = node.src0();
        if (!conv_ssm_src || conv_ssm_src != conv_input) return false;
        if (!ggml_hexagon_tensor_is_fuseable(conv_input)) return false; // must be CONCAT's only real consumer

        // qkv_transposed must be a plain GGML_OP_TRANSPOSE view (CONCAT's src1 built by build_conv_state's `qkv_mixed = ggml_transpose(ctx0, qkv_mixed)`), and its source (qkv_mixed_orig) must be token-major/channel-contiguous (nb[0] == type_size) -- exactly the layout the fused kernel's "no transpose needed" bulk-copy path assumes.
        // A non-plain-TRANSPOSE src1 (shouldn't occur for this call site, but guard against future refactors) safely skips the fusion instead of miscomputing.
        if (qkv_transposed->op != GGML_OP_TRANSPOSE) return false;
        const ggml_tensor * qkv_mixed_orig = qkv_transposed->src[0];
        if (!qkv_mixed_orig || qkv_mixed_orig->nb[0] != ggml_type_size(qkv_mixed_orig->type)) return false;

        // conv_states must be genuinely small (the tiny conv-kernel-history tail, not a pathological huge state) and itself contiguous (transpose_src0_state_rows_fused's per-tap scalar gather assumes contiguous rows of n_state_rows floats each -- true for the build_rs + reshape_3d chain that produces it).
        const int64_t n_state_rows = conv_states->ne[0];
        constexpr int64_t k_hvx_vlen_fp32 = 32; // matches htp/hvx-types.h's VLEN_FP32 (128B/4B), not visible from the host build
        if (n_state_rows <= 0 || n_state_rows >= k_hvx_vlen_fp32) return false;
        if (!ggml_is_contiguous(conv_states)) return false;

        // qkv_mixed_orig's token count must exactly cover conv_input's remaining span --
        // true by construction from build_conv_state, checked here defensively.
        if (qkv_mixed_orig->ne[1] != conv_input->ne[0] - n_state_rows) return false;
        if (qkv_mixed_orig->ne[2] != conv_states->ne[2]) return false; // n_seqs must match

        // Fused kernel's transpose_src0_state_rows_fused/DMA-bulk-copy path is prefill-only (same n_t>1 gate op_ssm_conv_f32's decode-mode NO-SUPPORT guard enforces) -- gate here too so an ineligible shape safely falls back to the unfused CONCAT+SSM_CONV path instead of ever hitting that guard at dispatch time.
        // d_inner must also be a multiple of the HVX tile width for the per-tile transpose/DMA math to divide evenly (matches op_ssm_conv_f32's own kparams->d_inner_tile rounding).
        const int64_t d_inner = conv_states->ne[1];
        const int64_t n_t     = node.dst()->ne[1];
        if (d_inner < k_hvx_vlen_fp32 || n_t <= 1) return false;

        if (!try_fuse_common({conv_states, qkv_mixed_orig, node.dst()})) {
            return false;
        }

        last_node.opcode = node.opcode; // HTP_OP_SSM_CONV or HTP_OP_SSM_CONV_SILU
        last_node.name   = (node.opcode == HTP_OP_SSM_CONV_SILU) ? "SSM_CONV+SILU(fused)" : "SSM_CONV(fused)";
        last_node.outputs.clear();
        last_node.outputs.push_back(node.dst());
        last_node.fused.push_back(node.node);

        // Recompute the staging layout for the fused path. Its qkv portion is already
        // token-major in DDR, and only the small recurrent-state tail needs raw staging.
        struct htp_ssm_conv_kernel_params new_kparams;
        ggml_hexagon_precompute_ssm_conv_params(
            sess, node.node->src[0], node.src1(), node.dst(), &new_kparams,
            true, (uint32_t) n_state_rows
        );
        if ((size_t) new_kparams.vtcm_size > sess->vtcm_size) {
            return false;
        }
        memcpy(last_node.kernel_params, &new_kparams, sizeof(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = node.opcode;
        memcpy(o.kernel_params, &new_kparams, sizeof(new_kparams));

        // Fused inputs: [0]=conv_states, [1]=conv_kernel weight (node.src1()), [2]=
        // qkv_mixed_orig. CONCAT's own inputs (conv_states, qkv_transposed) are dropped.
        last_node.inputs.clear();
        last_node.inputs.push_back(conv_states);
        last_node.inputs.push_back(node.src1());
        last_node.inputs.push_back(qkv_mixed_orig);

        o.src[0] = add_tensor(conv_states);
        o.src[1] = add_tensor(node.src1());
        o.src[2] = add_tensor(qkv_mixed_orig);
        for (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(node.dst());
        for (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused CONCAT+%s (#%u)\n", sess->c_name(),
                    node.opcode == HTP_OP_SSM_CONV_SILU ? "SSM_CONV+SILU" : "SSM_CONV", n_ops - 1);
        return true;
    }

    bool try_fuse_gdn_cpy(const htp_opnode & node) {
        if (n_ops == 0 || node.opcode != HTP_OP_CPY) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_GATED_DELTA_NET) return false;
        // interval snapshot mode (op_params[4] = n != 0): the fused CPY would cover slots the op did not write; never fuse.
        if (ggml_get_op_params_i32(last_node.node, 4) != 0) return false;
        if (last_node.outputs.size() != 1) return false;

        const ggml_tensor * gdn_out  = last_node.dst();
        const ggml_tensor * cpy_node = node.node;
        const ggml_tensor * cpy_src  = node.src0();
        const ggml_tensor * cpy_dst  = node.dst();

        if (!cpy_src || !cpy_dst || !cpy_dst->data) return false;
        const ggml_tensor * gdn_state = last_node.node->src[5];
        if (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32) return false;
        // f16 cache only together with an f16 GDN state (LLAMA_GDN_STATE_F16); otherwise the cache is f32
        if (cpy_dst->type != GGML_TYPE_F32 && !(cpy_dst->type == GGML_TYPE_F16 && gdn_state->type == GGML_TYPE_F16)) return false;
        if ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_node->flags & GGML_TENSOR_FLAG_OUTPUT)) return false;

        const ggml_tensor * v = last_node.node->src[2];
        if (!v) return false;

        // in-place state (LLAMA_RS_INPLACE_STATE): src[5] is the cache cell that dst[1] rewrites. The kernel is verified
        // for a single session (mdev.count == 1); a multi-session group takes the unfused GDN + CPY pair (no aliasing inside one op).
        if (sess->mdev.count > 1 && ggml_hexagon_tensors_overlap(gdn_state, cpy_dst)) return false;

        const int64_t S_v      = v->ne[0];
        const int64_t H        = v->ne[1];
        const int64_t n_tokens = v->ne[2];
        const int64_t n_seqs   = v->ne[3];
        const int64_t K        = ggml_get_op_params_i32(last_node.node, 0);
        const size_t  tail_off = (size_t) S_v * H * n_tokens * n_seqs * sizeof(float);

        const int64_t D         = S_v * S_v * H;
        const int64_t n_written = std::min<int64_t>(n_tokens, K);

        if (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_out && cpy_src->view_src->data != gdn_out->data) ||
            cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
            return false;
        }

        const size_t elem = ggml_type_size(cpy_dst->type);
        if (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != elem) {
            return false;
        }
        if (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * elem) {
            return false;
        }
        if (n_written > 1) {
            if (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * elem) {
                return false;
            }
        }

        if (!try_fuse_common({cpy_dst})) {
            return false;
        }

        last_node.name += "+CPY";
        last_node.outputs.push_back(cpy_dst);
        last_node.fused.push_back(node.node);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = add_tensor(cpy_dst);
        for (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }

    // speed150 GDN gate prep fusion (kernel: unary-ops.c op_gdn_gate). SOFTPLUS is rewritten in place into GDN_GATE(x, A) -> gate; SIGMOID then extends it with beta as input 2 and beta_sig as output 1.
    // Needs the graph order from LLAMA_S150_GDN_ORDER (qwen35.cpp): SOFTPLUS -> MUL(A) adjacent, and SIGMOID(beta) right after the MUL.
    bool try_fuse_gdn_gate_mul(const htp_opnode & node) {
        if (n_ops == 0 || node.opcode != HTP_OP_MUL) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_UNARY_SOFTPLUS || last_node.outputs.size() != 1) return false;

        const ggml_tensor * sp_dst = last_node.dst();
        const ggml_tensor * m0     = node.src0();
        const ggml_tensor * m1     = node.src1();
        const ggml_tensor * a      = nullptr;
        if (m0 == sp_dst || m0->data == sp_dst->data) {
            a = m1;
        } else if (m1 == sp_dst || m1->data == sp_dst->data) {
            a = m0;
        } else {
            return false;
        }

        const ggml_tensor * x = last_node.inputs[0];   // alpha + dt (MUL_MAT+ADD output)
        const ggml_tensor * g = node.dst();
        if (!x || !a || !g) return false;
        if (x->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32) return false;
        // GGML_HEXAGON_GDN_GATE_PF adds prefill batches only (64 <= rows <= GDN_GATE_HOST_MAX_T). Batches below 64 rows keep
        // exactly the GGML_HEXAGON_FUSE_S150 behaviour (rows <= 32), so without that switch they are unchanged.
        const bool pf_rows   = opt_gdn_gate_pf && x->ne[1] >= 64 && x->ne[1] <= GDN_GATE_HOST_MAX_T;
        const bool s150_rows = opt_fuse_s150 && x->ne[1] <= 32;
        if (x->ne[2] != 1 || x->ne[3] != 1 || x->ne[0] > 32 || x->ne[1] < 1 || !(pf_rows || s150_rows)) return false;
        if (!ggml_is_contiguous(x) || !ggml_is_contiguous(g)) return false;
        if (a->ne[0] != x->ne[0] || a->ne[1] != 1 || a->ne[2] != 1 || a->ne[3] != 1 || !ggml_is_contiguous(a)) return false;
        if (g->ne[0] != x->ne[0] || g->ne[1] != x->ne[1] || g->ne[2] != 1 || g->ne[3] != 1) return false;

        if (!try_fuse_common({a, g})) return false;

        last_node.opcode = HTP_OP_GDN_GATE;
        last_node.name   = "GDN_GATE";
        last_node.inputs.clear();
        last_node.inputs.push_back(x);
        last_node.inputs.push_back(a);
        last_node.outputs.clear();
        last_node.outputs.push_back(g);
        last_node.fused.push_back(node.node);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_GDN_GATE;
        o.src[0] = add_tensor(x);
        o.src[1] = add_tensor(a);
        for (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(g);
        for (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GDN_GATE (softplus+mul, #%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }

    bool try_fuse_gdn_gate_sigmoid(const htp_opnode & node) {
        if (n_ops == 0 || node.opcode != HTP_OP_UNARY_SIGMOID) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_GDN_GATE || last_node.inputs.size() != 2 || last_node.outputs.size() != 1) {
            return false;
        }

        const ggml_tensor * x = last_node.inputs[0];
        const ggml_tensor * b = node.src0();
        const ggml_tensor * s = node.dst();
        const int64_t H = x->ne[0];
        const int64_t T = x->ne[1];
        if (b->type != GGML_TYPE_F32 || s->type != GGML_TYPE_F32) return false;
        // beta/beta_sig: [1, H, T, 1] (decode and verify, up to 32 tokens) or [H, 1, T, 1] (AE-06 prefill layout, qwen35.cpp).
        // Both are H floats per token at nb[2] (op_gdn_gate reads one token row at beta->nb[2]).
        const bool beta4 = b->ne[0] == 1 && b->ne[1] == H && b->ne[2] == T && b->ne[3] == 1 &&
                           s->ne[0] == 1 && s->ne[1] == H && s->ne[2] == T && s->ne[3] == 1;
        const bool beta3 = b->ne[0] == H && b->ne[1] == 1 && b->ne[2] == T && b->ne[3] == 1 &&
                           s->ne[0] == H && s->ne[1] == 1 && s->ne[2] == T && s->ne[3] == 1;
        if (!beta4 && !beta3) return false;
        if (!ggml_is_contiguous(b) || !ggml_is_contiguous(s)) return false;
        if (b->nb[2] != (size_t) H * sizeof(float) || s->nb[2] != (size_t) H * sizeof(float)) return false;

        if (!try_fuse_common({b, s})) return false;

        last_node.inputs.push_back(b);
        last_node.outputs.push_back(s);
        last_node.fused.push_back(node.node);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.src[2] = add_tensor(b);
        o.dst[1] = add_tensor(s);

        HEX_VERBOSE("ggml-hex: %s fused GDN_GATE (+sigmoid, #%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }
    // Attention gate fusion (GGML_HEXAGON_GATE_MUL=1, kernel unary-ops.c op_gate_mul): MUL(x, SIGMOID(g)) with x [W, T] contiguous and g the [D, H, T] gate view (uniform token stride, D contiguous floats per head) becomes one GATE_MUL op y = x * sigmoid(g).
    // The SIGMOID is enqueued right before its MUL (attention graph order), so it is rewritten in place; its output is consumed only by that MUL (the gate sigmoid of build_layer_attn), so nothing else reads it.
    bool try_fuse_gate_mul(const htp_opnode & node) {
        if (n_ops == 0 || node.opcode != HTP_OP_MUL) return false;

        htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_UNARY_SIGMOID || last_node.inputs.size() != 1 || last_node.outputs.size() != 1) {
            return false;
        }

        const ggml_tensor * sg_dst = last_node.dst();
        const ggml_tensor * m0     = node.src0();
        const ggml_tensor * m1     = node.src1();
        const ggml_tensor * x      = nullptr;   // the MUL operand that is not the sigmoid
        const ggml_tensor * sg     = nullptr;   // the MUL operand that is the sigmoid output (reshaped view)
        if (m1 == sg_dst || m1->data == sg_dst->data) {
            x  = m0;
            sg = m1;
        } else if (m0 == sg_dst || m0->data == sg_dst->data) {
            x  = m1;
            sg = m0;
        } else {
            return false;
        }

        const ggml_tensor * g = last_node.inputs[0];   // the sigmoid input: the gate view
        const ggml_tensor * y = node.dst();
        if (!x || !sg || !g || !y) return false;
        if (x->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32 || y->type != GGML_TYPE_F32 || sg->type != GGML_TYPE_F32) {
            return false;
        }

        // gate layout: the contiguous [W, T] SIGMOID input (NSEG = 1, the CONT'd gate), or the [S, NSEG, T] gate view
        // (NSEG = heads, S = head dim, strided by the gate view); the token stride must be NSEG * head stride.
        const int64_t W = x->ne[0];
        const int64_t T = x->ne[1];
        int64_t S = 0, NSEG = 0, hs = 0, ts = 0;
        if (g->ne[2] == 1 && g->ne[3] == 1 && g->ne[0] == W && g->ne[1] == T) {
            S = W; NSEG = 1; hs = g->nb[1]; ts = g->nb[1];
        } else if (g->ne[3] == 1 && g->ne[2] == T && g->ne[0] * g->ne[1] == W) {
            S = g->ne[0]; NSEG = g->ne[1]; hs = g->nb[1]; ts = g->nb[2];
            if (ts != NSEG * hs) return false;
        } else {
            return false;
        }
        // prefill batches only (decode and small batches keep the unfused chain), GATE_MUL_MAX_W / 32-float alignment
        if (T < 64 || W > 4096 || W % 32 != 0 || S % 32 != 0) return false;
        if (x->ne[2] != 1 || x->ne[3] != 1) return false;
        if (y->ne[0] != W || y->ne[1] != T || y->ne[2] != 1 || y->ne[3] != 1) return false;
        if (sg->ne[0] != W || sg->ne[1] != T || sg->ne[2] != 1 || sg->ne[3] != 1) return false;
        if (g->nb[0] != sizeof(float) || hs < S * (int64_t) sizeof(float) || hs > INT32_MAX || ts > INT32_MAX) return false;
        if (x->nb[0] != sizeof(float) || x->nb[1] != (size_t) W * sizeof(float)) return false;
        if (y->nb[0] != sizeof(float) || y->nb[1] != (size_t) W * sizeof(float)) return false;
        if (sg->nb[0] != sizeof(float) || sg->nb[1] != (size_t) W * sizeof(float)) return false;

        // The fused op reads g (the gate input) while it writes y, and it drops the SIGMOID output. Overlap with g can come from the graph allocator reusing g's buffer for the MUL output after the SIGMOID that last used it.
        // Refuse unless the memory ranges are disjoint or y == x (in place: each token block is read before it is written).
        auto ranges_overlap = [](const void * a, size_t na, const void * b, size_t nb) {
            const uintptr_t a0 = (uintptr_t) a;
            const uintptr_t b0 = (uintptr_t) b;
            return a0 < b0 + nb && b0 < a0 + na;
        };
        const size_t g_span = (size_t) (T - 1) * (size_t) ts + (size_t) NSEG * (size_t) hs;   // extent of the gate input
        if (ranges_overlap(g->data, g_span, y->data, ggml_nbytes(y)) || ranges_overlap(g->data, g_span, x->data, ggml_nbytes(x))) {
            return false;
        }
        if (x->data != y->data && ranges_overlap(x->data, ggml_nbytes(x), y->data, ggml_nbytes(y))) {
            return false;
        }
        if (!try_fuse_common({x, g, y})) return false;

        last_node.opcode = HTP_OP_GATE_MUL;
        last_node.name   = "GATE_MUL";
        last_node.inputs.clear();
        last_node.inputs.push_back(x);
        last_node.inputs.push_back(g);
        last_node.outputs.clear();
        last_node.outputs.push_back(y);
        last_node.fused.push_back(node.node);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_GATE_MUL;
        o.params[0] = (int32_t) S;     // GATE_MUL op_params: segment floats, segments per token, head stride, token stride
        o.params[1] = (int32_t) NSEG;
        o.params[2] = (int32_t) hs;
        o.params[3] = (int32_t) ts;
        o.src[0] = add_tensor(x);
        o.src[1] = add_tensor(g);
        for (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(y);
        for (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        // fp16 tile output for the attn_output matmul (GGML_HEXAGON_GATE_TILE): fp32 output dropped, the reader gets the tile view
        if (sess->gatetile_ok.count(node.node) != 0 && ggml_hexagon_act_tile_reserve(sess)) {
            const ggml_tensor * view = ggml_hexagon_act_tile_view(sess, W, T);
            o.dst[1] = add_tensor(view);
            o.flags |= HTP_OPFLAGS_ACT_TILE_ONLY;
            sess->act_tile_live[node.node] = view;
            sess->act_tile_skip_live.insert(node.node);
            last_node.name = "GATE_MUL(tile)";
        }

        HEX_VERBOSE("ggml-hex: %s fused GATE_MUL (sigmoid+mul, #%u)\n", sess->c_name(), n_ops - 1);
        return true;
    }


    // speed150 diagnostic (GGML_HEXAGON_FUSE_AR_DIAG=1): for an ADD right after ALLREDUCE, logs each
    // precondition of try_fuse_allreduce_add (first 24 occurrences). Changes no behaviour.
    void ar_add_diag(const htp_opnode & node) {
        static int n_logged = 0;
        if (n_logged >= 24 || n_ops == 0 || node.opcode != HTP_OP_ADD) return;
        const htp_opnode & last_node = ops[n_ops - 1];
        if (last_node.opcode != HTP_OP_ALLREDUCE) return;
        n_logged++;

        const auto * ar_kparams = (const struct htp_allreduce_kernel_params *) last_node.kernel_params;
        const uint32_t rank    = (uint32_t) ar_kparams->rank;
        const uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        const ggml_tensor * ar_local = last_node.inputs[rank];
        const ggml_tensor * add_src0 = node.src0();
        const ggml_tensor * add_src1 = node.src1();
        const ggml_tensor * add_dst  = node.dst();
        const bool fuseable  = ar_local && ggml_hexagon_tensor_is_fuseable(ar_local);
        const bool src0_is   = add_src0 == ar_local || (ar_local && add_src0->data == ar_local->data);
        const bool src1_is   = add_src1 == ar_local || (ar_local && add_src1->data == ar_local->data);
        const ggml_tensor * res = src0_is ? add_src1 : add_src0;
        const bool type_ok   = res && ar_local && ar_local->type == res->type;
        const bool shape_ok  = res && ar_local->ne[0] == res->ne[0] && ar_local->ne[1] == res->ne[1] &&
                               ar_local->ne[2] == res->ne[2] && ar_local->ne[3] == res->ne[3];
        const bool bufs_ok   = res && try_fuse_common(res, add_dst);
        struct htp_allreduce_kernel_params tmp;
        const bool solver_ok = ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, rank, n_ranks, true, false, &tmp);
        GGML_LOG_INFO("ggml-hex: AR+ADD diag #%d: fuseable=%d src_match=%d(src0=%d,src1=%d) type_ok=%d shape_ok=%d "
                      "try_fuse_common=%d solver=%d | n_bufs=%u/%u n_tens=%u/%u b_vmem=%zu/%zu\n",
                      n_logged, (int) fuseable, (int) (src0_is || src1_is), (int) src0_is, (int) src1_is,
                      (int) type_ok, (int) shape_ok, (int) bufs_ok, (int) solver_ok,
                      n_bufs, n_bufs_max, n_tens, n_tens_max, b_vmem, b_vmem_max);
    }

    bool try_fuse(const htp_opnode & node) {
        if (opt_ar_diag) ar_add_diag(node);
        if (!opt_opfusion) return false;
        // fast op-enqueue plan-boundary guard: every fusion below rewrites the last-enqueued op in place. While a plan is being captured (rec_active), ops at indexes below rec_plan_sig already belong to an earlier subgraph's finished plan (plan_finish already recorded their descriptors): rewriting one of those op's in-place descriptor here would make that earlier plan's replay re-enqueue the stale (pre-fusion) descriptor and silently drop the fused-away node (e.g.
        // Refuse the fusion outright so the node enqueues plain into the CURRENT subgraph's plan instead, where it is recorded correctly and survives replay. a residual ADD whose matching MUL_MAT sits in a previous subgraph, at a batch boundary two different graph->uid dispatches happened to share without an intervening flush).
        if (rec_active && n_ops > 0 && (n_ops - 1) < rec_plan_sig) return false;
        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ADD) && try_fuse_allreduce_add(node)) return true;
        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(node))  return true;
        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_SCALE) && try_fuse_rms_norm_scale(node)) return true;
        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ADD)   && try_fuse_mul_mat_add(node))   return true;
        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(node))    return true;
        if (opt_swiglu_out && try_fuse_swiglu_out(node)) return true;
        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(node)) return true;
        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(node))       return true;
        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_SSM_CONV_SILU) && try_fuse_ssm_conv_silu(node)) return true;
        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_CONCAT_SSM_CONV) && try_fuse_concat_ssm_conv(node)) return true;
        if (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ROPE_NX)        && try_fuse_rope_nx(node))        return true;
        if (opt_rope_kv && try_fuse_rope_kv(node)) return true;
        if ((opt_fuse_s150 || opt_gdn_gate_pf) && try_fuse_gdn_gate_mul(node))     return true;
        if ((opt_fuse_s150 || opt_gdn_gate_pf) && try_fuse_gdn_gate_sigmoid(node)) return true;
        if (opt_gate_mul && try_fuse_gate_mul(node)) return true;
        if (opt_norm_gated && try_fuse_norm_gated(node)) return true;
        return false;
    }
};

struct ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

struct ggml_hexagon_opqueue {
    // Shared buffer for storing batches
    ggml_hexagon_shared_buffer *shm_buf;
    size_t                      shm_blk_size;
    size_t                      depth;
    const bool keep_ops;   // op_cache is read only for profiling (and its size assert); 1-pass dispatch skips the copy otherwise

    using opvec = std::vector<htp_opnode>;

    std::vector<opvec>          op_cache;       // per batch op cache
    std::vector<uint64_t>       start_usec;     // per batch start time

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, size_t batch_size, size_t depth) : depth(depth), keep_ops(!opt_dispatch_1pass || opt_profile != 0) {
        size_t n_bufs    = HTP_OP_MAX_BUFS;
        size_t n_ops     = batch_size;
        size_t n_tensors = n_ops * HTP_OP_MAX_OUTPUTS + n_ops * HTP_OP_MAX_INPUTS;

        size_t tr_size = 0;
        if (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * sizeof(htp_trace_desc);
        }

        shm_blk_size = sizeof(htp_buf_desc)  * n_bufs    +
                       sizeof(htp_tensor)    * n_tensors +
                       sizeof(htp_op_desc)   * n_ops     +
                       sizeof(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = new ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, true /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        if (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size(), shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    size_t shm_size() const { return shm_buf ? shm_buf->size() : 0; }

    // push new batch
    bool push(htp_opbatch_req& req, dspqueue_buffer& dbuf, const ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(sizeof(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(sizeof(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(sizeof(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(sizeof(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(sizeof(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(sizeof(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        if (seq - shm_buf->sess->batch_rsp_seq > depth) { return false; }

        const uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        if (keep_ops) { op_cache[slot] = op_batch->ops; }
        start_usec[slot] = ggml_time_us();

        const size_t b_size = sizeof(htp_buf_desc)  * req.n_bufs;
        const size_t t_size = sizeof(htp_tensor)    * req.n_tensors;
        const size_t o_size = sizeof(htp_op_desc)   * req.n_ops;
        const size_t p_size = sizeof(htp_prof_desc) * req.n_ops;

        size_t tr_size = 0;
        if (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * sizeof(htp_trace_desc);
        } else {
            req.n_traces = 0;
        }

        dbuf.ptr      = shm_buf->base() + ((size_t) slot * shm_blk_size);
        dbuf.fd       = shm_buf->fd();
        dbuf.flags    = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
        dbuf.offset   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->base();
        dbuf.size     = b_size + t_size + o_size + p_size + tr_size;

        GGML_ASSERT(dbuf.size <= shm_blk_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_size;
        uint8_t * t_ptr = m_ptr; m_ptr += t_size;
        uint8_t * o_ptr = m_ptr;

        memcpy(b_ptr, (void *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (void *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (void *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        if (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            for (unsigned int i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (void *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            for (unsigned int i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (unsigned long long) t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        return true;
    }

    void pop(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        const uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        const size_t b_size = sizeof(htp_buf_desc)  * rsp.n_bufs;
        const size_t t_size = sizeof(htp_tensor)    * rsp.n_tensors;
        const size_t o_size = sizeof(htp_op_desc)   * rsp.n_ops;
        const size_t p_size = sizeof(htp_prof_desc) * rsp.n_ops;

        size_t tr_size = 0;
        uint32_t n_traces = 0;
        if (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * sizeof(htp_trace_desc);
        }

        const size_t m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        if (rsp.n_ops > 0) {
            if (keep_ops) {
                GGML_ASSERT(rsp.n_ops <= op_cache[slot].size());
            }

            const htp_prof_desc * pd = (const htp_prof_desc *) p_ptr;
            const htp_trace_desc * trace_events = nullptr;
            if (opt_profile == 3) {
                trace_events = (const htp_trace_desc *) (p_ptr + p_size);
            }

            if (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            for (uint32_t i = 0; i < rsp.n_ops; i++) {
                if (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, op_cache[slot][i], pd[i]);
                }
            }

            if (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

void ggml_hexagon_session::flush_peers() {
    auto vpeers = std::move(virt_peers);
    virt_peers.clear();
    for (auto * peer : vpeers) {
        peer->flush_sync();
    }

    auto ppeers = std::move(phys_peers);
    phys_peers.clear();
    for (auto * peer : ppeers) {
        peer->flush_async();
    }

    for (auto & sub : this->mdev.sessions) {
        sub->flush_peers();
    }
}

void ggml_hexagon_session::flush_async() {
    if (opt_dispatch_1pass) {
        flush_async_1pass();
        return;
    }
    flush_peers();
    flush_batch();
}

void ggml_hexagon_session::flush_pending(bool all) {
    for (auto & sub : this->mdev.sessions) {
        sub->flush_pending(all);
        if (sub->last_error > HTP_STATUS_OK) {
            this->last_error = sub->last_error.load();
        }
    }

    while (this->batch_rsp_seq < this->batch_req_seq) {
        struct htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        struct dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        const uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        int err = dspqueue_read(this->queue, &flags, 1, &n_dbufs, &dbuf, sizeof(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        if (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            continue;
        }

        if (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        if (rsp_size != sizeof(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", this->c_name(), rsp_size, n_dbufs);
        }

        if (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", this->c_name(), status_to_str(rsp.status));
            this->last_error = rsp.status;
            for (auto & sub : this->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->pop(rsp, dbuf);

        GGML_ASSERT(rsp.seq == this->batch_rsp_seq + 1);
        this->batch_rsp_seq = rsp.seq;

        if (!all) break;
    }
}

void ggml_hexagon_session::flush_sync(bool all) {
    const uint64_t t_s0 = opt_cycle_timing >= 2 ? ggml_time_us() : 0;
    flush_async();
    const uint64_t t_s1 = opt_cycle_timing >= 2 ? ggml_time_us() : 0;
    flush_pending(all);
    if (opt_cycle_timing >= 2) {
        const uint64_t t_s2 = ggml_time_us();
        if (ov_n_compute) {
            long long first_w = 0, last_w = 0;
            if (!opt_disp_log.empty()) {
                uint64_t mn = UINT64_MAX, mx = 0;
                for (const auto & r : opt_disp_log) { mn = std::min<uint64_t>(mn, r.t_w0); mx = std::max<uint64_t>(mx, r.t_w1); }
                first_w = (long long) ((int64_t) mn - (int64_t) ov_first_us);
                last_w  = (long long) ((int64_t) t_s2 - (int64_t) mx);
            }
            GGML_LOG_INFO("hostovh %s first-write +%lld us after first compute, last write %lld us before sync end\n", c_name(), first_w, last_w);
            GGML_LOG_INFO("hostovh %s pass: computes %u enq %llu us | last-sync->first-compute %lld us | flush %llu us | wait %llu us | first-compute->sync %llu us\n",
                          c_name(), ov_n_compute, (unsigned long long) ov_enq_us,
                          (long long) ((int64_t) ov_first_us - (int64_t) ov_last_sync_us),
                          (unsigned long long) (t_s1 - t_s0), (unsigned long long) (t_s2 - t_s1),
                          (unsigned long long) (t_s2 - ov_first_us));
            ov_enq_us = 0;
            ov_n_compute = 0;
        }
        ov_last_sync_us = t_s2;
    }
}

void ggml_hexagon_session::start_batch() {
    if (this->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

void ggml_hexagon_session::flush_batch(size_t min_ops) {
    if (op_batch->n_ops < min_ops) { return; }
    const uint64_t t_begin = opt_cycle_timing >= 2 ? ggml_time_us() : 0;

    // fast op-enqueue: a flush inside a plan-capture window moved a batch boundary away from the recorded one; mark the in-progress plan dirty (read-only here -- plan_finish reads rec_flush_seen and discards it).
    if (op_batch->rec_active) {
        op_batch->rec_active     = false;
        op_batch->rec_flush_seen = true;
    }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    const uint64_t seq = ++this->batch_req_seq;

    op_batch->update_mdev_group(this->mdev.idx);

    if (!op_queue->push(req, dbuf, op_batch, seq)) {
        flush_pending(false);
        op_queue->push(req, dbuf, op_batch, seq);
    }

    for (auto & sub : this->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_buffer sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        if (!sub->op_queue->push(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(false);
            sub->op_queue->push(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        sub->dispatch_write(sub_req, sub_dbuf, t_begin);
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", this->c_name(), dbuf.ptr, dbuf.size);

    dispatch_write(req, dbuf, t_begin);

    op_batch->reset();
}

// GGML_HEXAGON_DISPATCH_1PASS (speed150): the same batches as flush_peers() + flush_batch(), in one pass. Every batch is
// prepared (sorted, copied into its shared slot) before any dspqueue_write is issued, so the writes go out back to back.
void ggml_hexagon_session::flush_async_1pass() {
    std::vector<ggml_hexagon_session *> group;
    dispatch_collect(group);
    for (auto * sess : group) {
        sess->dispatch_prepare(1);
    }
    for (auto * sess : group) {
        sess->dispatch_issue();
    }
}

// post-order closure over the peers, with the recursion's visiting rules; nothing is written here
void ggml_hexagon_session::dispatch_collect(std::vector<ggml_hexagon_session *> & out) {
    collect_peers_1pass(out);
    if (std::find(out.begin(), out.end(), this) == out.end()) {
        out.push_back(this);
    }
}

void ggml_hexagon_session::collect_peers_1pass(std::vector<ggml_hexagon_session *> & out) {
    auto vpeers = std::move(virt_peers);
    virt_peers.clear();
    for (auto * peer : vpeers) {
        peer->flush_sync();   // virtual peers stay sequential: they wait for their own responses
    }

    auto ppeers = std::move(phys_peers);
    phys_peers.clear();
    for (auto * peer : ppeers) {
        peer->dispatch_collect(out);
    }

    for (auto & sub : this->mdev.sessions) {
        sub->collect_peers_1pass(out);
    }
}

// first half of flush_batch(): sort, stamp the sequence, copy the descriptors into the shared slot (no write yet)
void ggml_hexagon_session::dispatch_prepare(size_t min_ops) {
    disp_pending = false;
    if (op_batch->n_ops < min_ops) { return; }

    disp_begin = opt_cycle_timing >= 2 ? ggml_time_us() : 0;

    if (op_batch->rec_active) {
        op_batch->rec_active     = false;
        op_batch->rec_flush_seen = true;
    }

    op_batch->sort_buffers();

    const uint64_t seq = ++this->batch_req_seq;

    op_batch->update_mdev_group(this->mdev.idx);

    disp_req  = htp_opbatch_req {};
    disp_dbuf = dspqueue_buffer {};
    if (!op_queue->push(disp_req, disp_dbuf, op_batch, seq)) {
        flush_pending(false);
        op_queue->push(disp_req, disp_dbuf, op_batch, seq);
    }
    disp_pending = true;

    for (auto & sub : this->mdev.sessions) {
        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        sub->disp_req  = htp_opbatch_req {};
        sub->disp_dbuf = dspqueue_buffer {};
        if (!sub->op_queue->push(sub->disp_req, sub->disp_dbuf, op_batch, seq)) {
            sub->flush_pending(false);
            sub->op_queue->push(sub->disp_req, sub->disp_dbuf, op_batch, seq);
        }
        sub->disp_begin   = disp_begin;
        sub->disp_pending = true;
    }
}

// second half of flush_batch(): the writes (mdev subs first, then this session, as before), then the op batch is free
void ggml_hexagon_session::dispatch_issue() {
    if (!disp_pending) { return; }

    for (auto & sub : this->mdev.sessions) {
        if (sub->disp_pending) {
            sub->dispatch_write(sub->disp_req, sub->disp_dbuf, sub->disp_begin);
            sub->disp_pending = false;
        }
    }
    dispatch_write(disp_req, disp_dbuf, disp_begin);
    disp_pending = false;

    op_batch->reset();
}

// one dspqueue_write of a batch; with LLAMA_CYCLE_TIMING>=2 the host times are kept for ggml_hexagon_dispatch_log_flush()
void ggml_hexagon_session::dispatch_write(htp_opbatch_req & req, dspqueue_buffer & dbuf, uint64_t t_begin) {
    const uint64_t t_w0 = opt_cycle_timing >= 2 ? ggml_time_us() : 0;
    int err = dspqueue_write(this->queue, 0, 1, &dbuf, sizeof(req), (const uint8_t*) &req, DSPQUEUE_TIMEOUT);
    if (opt_cycle_timing >= 2) {
        if (opt_disp_log.size() >= 4096) {
            opt_disp_log.clear();
        }
        opt_disp_log.push_back({ this->c_name(), req.seq, t_begin, t_w0, (uint64_t) ggml_time_us() });
    }
    if (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", this->c_name(), (unsigned) err);
    }
}

void ggml_hexagon_session::enqueue_op(const htp_opnode & node) {
    auto clone_tensor_buffer = [this](const ggml_tensor * t) {
        auto sbuf = this->mmap_tensor(t);
        if (!sbuf) return;
        if (sbuf->sess != this) {
            this->clone_buffer(sbuf);
        }
        for (auto & sub : this->mdev.sessions) {
            sub->clone_buffer(sbuf);
        }
    };

    for (auto t : node.get_inputs()) {
        clone_tensor_buffer(t);
    }
    for (auto t : node.get_outputs()) {
        clone_tensor_buffer(t);
    }

    if (opt_opfusion && op_batch->try_fuse(node)) {
        return;
    }

    if (!op_batch->fit_op(node)) {
        flush_async();
    }

    if (op_batch->empty()) {
        start_batch();
    }

    if (!op_batch->fit_op(node)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->add_op(node);
}

void ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_node(HTP_OP_MDEV_GROUP);

    uint8_t * fence_slot = this->mdev_fence_slot;

    static ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.buffer       = &this->fence_buf->backend_buffer;
    dummy_t.extra        = &fence_extra;
    dummy_t.data         = (void *) fence_slot;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.ne[1]        = (int64_t) this->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) this->mdev.idx;

    ggml_tensor * node = group_node.add_dummy(dummy_t);
    node->src[0] = node;
    group_node.init(node);
    group_node.outputs.clear();
    group_node.name = "MDEV_GROUP";

    for (auto & sub : this->mdev.sessions) {
        sub->clone_buffer(this->fence_buf);
    }

    op_batch->add_op(group_node);
}

void ggml_hexagon_session::enqueue_cpy(const ggml_tensor * src, ggml_tensor * dst, const ggml_tensor * sync_tensor, uint32_t fence_seq) {
    const bool with_fence = sync_tensor != nullptr;
    htp_opnode cpy_node(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* node = cpy_node.add_dummy(*dst);
    node->op     = GGML_OP_CPY;
    node->src[0] = const_cast<ggml_tensor *>(src);
    node->src[1] = with_fence ? cpy_node.add_dummy(*sync_tensor) : nullptr;
    if (with_fence) {
        node->op_params[0] = (int32_t) fence_seq;
    }

    cpy_node.init(node);
    if (with_fence) {
        cpy_node.name = "CPY+FENCE";
    }
    this->enqueue_op(cpy_node);
}

void ggml_hexagon_session::enqueue_fence(const ggml_tensor * sync_tensor, uint32_t fence_seq, bool wait) {
    htp_opnode sync_node(HTP_OP_FENCE);

    ggml_tensor* node = sync_node.add_dummy(*sync_tensor);
    node->op           = GGML_OP_NONE;
    node->src[0]       = node;
    node->op_params[0] = (int32_t) fence_seq;
    node->op_params[1] = wait ? 0 : 1;

    sync_node.init(node);
    sync_node.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    this->enqueue_op(sync_node);
}

static bool ggml_hexagon_precompute_allreduce_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    bool has_add,
    bool is_row_bcast,
    struct htp_allreduce_kernel_params * kparams,
    bool chunk_add
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    const uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    const uint32_t elem_size = (dst->type == GGML_TYPE_F16) ? sizeof(ggml_fp16_t) : sizeof(float);
    const bool is_contiguous = ggml_is_contiguous(dst);

    const uint32_t ne0 = (uint32_t) dst->ne[0];
    const uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    const bool use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    // Ranks reduce disjoint chunks and write them to all n_dsts (reduce-scatter). A fused ADD reduces the full range into its own dst
    // unless chunk_add asks for the chunked form (gap-fused MC01).
    const bool chunked = !has_add || chunk_add;
    if (!chunked) {
        kparams->n_dsts = 1;
        if (use_1d) {
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } else {
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } else {
        kparams->n_dsts = (int32_t) n_ranks;
        if (use_1d && has_add && ar_rs_row_rows(ne1) && ne0 % 128 == 0) {
            // RS-ROW: row-aligned chunks, the row ranges of the norm gather (ar_rs_row_chunk_rows). Element order per output is
            // unchanged (each element still sums ((s0+s1)+s2)+res), so the values are the same; only the owner of a row changes.
            const uint32_t rank_chunk_rows = ar_rs_row_chunk_rows(ne1, n_ranks);
            const uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            const uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            kparams->rank_elem_start       = (int32_t) (rank_r0 * ne0);
            kparams->rank_nelem            = (int32_t) ((rank_r1 - rank_r0) * ne0);
        } else if (use_1d) {
            const uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            const uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            const uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            const uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } else {
            const uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            const uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            const uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            const uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    if (use_1d) {
        const uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        const size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_count(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        if (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        while ((size_t) kparams->vtcm_size > sess->vtcm_size && block_elems > 128) {
            const size_t max_bytes_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((size_t) (max_bytes_per_buf / elem_size), 128);
            if (block_elems < 128) break;
            kparams->block_elems          = block_elems;
            kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
        }

        if (sess->vtcm_size < (size_t) kparams->vtcm_size || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            return false;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        return true;
    } else {
        const uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        const size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_count(n_ranks, n_threads, has_add, is_row_bcast);

        const uint32_t row_bytes = ne0 * elem_size;
        const uint32_t row_size_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_size_aligned = row_size_aligned;

        const uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        while ((size_t) kparams->vtcm_size > sess->vtcm_size && block_rows > 1) {
            const size_t max_rows_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2 * row_size_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
            if (max_rows_per_buf == 0) break;
        }

        if (sess->vtcm_size < (size_t) kparams->vtcm_size || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            return false;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        return true;
    }
}

void ggml_hexagon_session::enqueue_allreduce(
    const ggml_tensor * dst,
    const std::vector<const ggml_tensor *> & src_tensors,
    const std::vector<const ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_node(HTP_OP_ALLREDUCE);

    ggml_tensor* node = ar_node.add_dummy(*dst);
    node->op           = GGML_OP_NONE;
    node->op_params[0] = (int32_t) fence_seq_entry;
    node->op_params[1] = (int32_t) fence_seq_exit;

    ar_node.init(node);

    ar_node.inputs.clear();
    for (size_t i = 0; i < src_tensors.size(); i++) {
        ar_node.inputs.push_back(src_tensors[i]);
    }
    for (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_node.inputs.push_back(ar_node.add_dummy(*sync_tensors[i]));
    }

    ar_node.outputs.clear();
    for (size_t i = 0; i < src_tensors.size(); i++) {
        ar_node.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        this, dst, rank, n_ranks, false, false,
        (struct htp_allreduce_kernel_params *) ar_node.kernel_params
    );

    ar_node.name = "ALLREDUCE";
    this->enqueue_op(ar_node);
}

// MC05 norm gather: rank r normalizes token rows [r0, r0 + rows) of the ne1 rows, split like the chunked reduce.
// One thread: blocks of 32 rows in one slot set (weight row, src block, dst block). GGML_HEXAGON_AR_RMS_AG_MT=1: the rows are split over the session threads; each thread has two src and two dst slots of blk_rows rows after the shared weight row (allreduce_rms_ag_thread in the HTP).
static bool ggml_hexagon_precompute_rms_ag_params(const ggml_hexagon_session * sess, const ggml_tensor * dst, uint32_t rank,
                                                  uint32_t n_ranks, bool tile, struct htp_allreduce_kernel_params * kparams) {
    memset(kparams, 0, sizeof(*kparams));
    if (tile) {
        // fp16 tile output (GGML_HEXAGON_GATHER_ACT_TILE): owner rows in whole 16-row blocks, threads take whole blocks. VTCM is the
        // shared weight row plus (4 x 16 + 2) fp32 rows per thread (allreduce_rms_ag_tile_thread in the HTP).
        const uint32_t ne0_t      = (uint32_t) dst->ne[0];
        const uint32_t ne1_t      = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
        const uint32_t row_bytes  = ne0_t * (uint32_t) sizeof(float);
        const uint32_t rows_per   = ((ne1_t + n_ranks - 1) / n_ranks + AR_RMS_AG_TILE_ROWS - 1) / AR_RMS_AG_TILE_ROWS * AR_RMS_AG_TILE_ROWS;
        const uint32_t r0_t       = (std::min)(rank * rows_per, ne1_t);
        const uint32_t r1_t       = (std::min)(r0_t + rows_per, ne1_t);
        const uint32_t rows_t     = r1_t - r0_t;
        const size_t   per_thread = (size_t) (4 * AR_RMS_AG_TILE_ROWS + 2) * row_bytes;
        uint32_t n_threads_t      = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rows_t / AR_RMS_AG_TILE_ROWS));
        while (n_threads_t > 1 && row_bytes + n_threads_t * per_thread > sess->vtcm_size) {
            n_threads_t--;
        }
        if (row_bytes + n_threads_t * per_thread > sess->vtcm_size) {
            return false;
        }
        kparams->rank                 = (int32_t) rank;
        kparams->n_ranks              = (int32_t) n_ranks;
        kparams->n_threads            = (int32_t) n_threads_t;
        kparams->block_elems          = (int32_t) AR_RMS_AG_TILE_ROWS;
        kparams->kernel_type          = HTP_ALLREDUCE_KERNEL_RMS_AG;
        kparams->n_dsts               = (int32_t) n_ranks;
        kparams->ne0                  = (int32_t) ne0_t;
        kparams->ne1                  = (int32_t) ne1_t;
        kparams->row_size_aligned     = (int32_t) row_bytes;
        kparams->rank_elem_start      = (int32_t) r0_t;
        kparams->rank_nelem           = (int32_t) rows_t;
        kparams->vtcm_size_per_thread = (int32_t) per_thread;
        kparams->vtcm_size            = (int32_t) (row_bytes + n_threads_t * per_thread);
        return true;
    }
    const uint32_t ne0       = (uint32_t) dst->ne[0];
    const uint32_t ne1       = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    const uint32_t rows_per  = ar_rs_row_chunk_rows(ne1, n_ranks);
    const uint32_t r0        = (std::min)(rank * rows_per, ne1);
    const uint32_t r1        = (std::min)(r0 + rows_per, ne1);
    const uint32_t row_bytes = ne0 * (uint32_t) sizeof(float);
    const uint32_t rows      = r1 - r0;

    uint32_t n_threads = 1;
    uint32_t blk_rows  = 32;
    if (ar_rms_ag_mt() && rows > 1) {
        n_threads = (std::min)(sess->n_threads, rows);
        blk_rows  = (std::min)(16u, (std::max)(1u, (rows + 2 * n_threads - 1) / (2 * n_threads)));
        // weight row + n_threads x (2 src + 2 dst slots) must fit VTCM: shrink the block first, then the thread count
        while (n_threads > 1 && row_bytes + (size_t) n_threads * 4 * blk_rows * row_bytes > sess->vtcm_size) {
            if (blk_rows > 1) {
                blk_rows = (blk_rows + 1) / 2;
            } else {
                --n_threads;
            }
        }
        if (n_threads < 2) {
            n_threads = 1;
            blk_rows  = 32;
        }
    }

    kparams->rank                 = (int32_t) rank;
    kparams->n_ranks              = (int32_t) n_ranks;
    kparams->n_threads            = (int32_t) n_threads;
    kparams->block_elems          = (int32_t) blk_rows;
    kparams->kernel_type          = HTP_ALLREDUCE_KERNEL_RMS_AG;
    kparams->n_dsts               = (int32_t) n_ranks;
    kparams->ne0                  = (int32_t) ne0;
    kparams->ne1                  = (int32_t) ne1;
    kparams->row_size_aligned     = (int32_t) row_bytes;
    kparams->rank_elem_start      = (int32_t) r0;
    kparams->rank_nelem           = (int32_t) rows;
    if (n_threads > 1) {
        kparams->vtcm_size_per_thread = (int32_t) (4 * blk_rows * row_bytes);                             // 2 src + 2 dst slots
        kparams->vtcm_size            = (int32_t) (row_bytes + n_threads * 4 * blk_rows * row_bytes);     // shared weight row first
    } else {
        kparams->vtcm_size_per_thread = (int32_t) ((1 + 2 * blk_rows) * row_bytes); // weight row, src block, dst block
        kparams->vtcm_size            = kparams->vtcm_size_per_thread;
    }
    return true;
}

void ggml_hexagon_session::enqueue_allreduce_add(
    const std::vector<const ggml_tensor *> & src_tensors,
    const std::vector<const ggml_tensor *> & sync_tensors,
    const ggml_tensor * residual,
    const std::vector<const ggml_tensor *> & dst_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit,
    bool     chunked,
    bool     owner_only
) {
    htp_opnode ar_node(HTP_OP_ALLREDUCE_ADD);

    ggml_tensor * node = ar_node.add_dummy(*dst_tensors[rank]);
    node->op           = GGML_OP_NONE;
    node->op_params[0] = (int32_t) fence_seq_entry;
    node->op_params[1] = (int32_t) fence_seq_exit;

    ar_node.init(node);

    // Inputs: n partials, n fences, then the residual at src[2 * n_ranks] (the layout try_fuse_allreduce_add also produces).
    ar_node.inputs.clear();
    for (size_t i = 0; i < src_tensors.size(); i++) {
        ar_node.inputs.push_back(src_tensors[i]);
    }
    for (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_node.inputs.push_back(ar_node.add_dummy(*sync_tensors[i]));
    }
    ar_node.inputs.push_back(residual);

    // Outputs: chunked (reduce-scatter) writes every rank's dst, otherwise this rank's dst only.
    ar_node.outputs.clear();
    if (chunked) {
        for (size_t i = 0; i < dst_tensors.size(); i++) {
            ar_node.outputs.push_back(dst_tensors[i]);
        }
    } else {
        ar_node.outputs.push_back(dst_tensors[rank]);
    }

    ggml_hexagon_precompute_allreduce_params(
        this, dst_tensors[rank], rank, n_ranks, true, false,
        (struct htp_allreduce_kernel_params *) ar_node.kernel_params, chunked
    );

    if (owner_only) {
        // RS-ROW: this rank writes only its own dst (its rows of the residual). The peers' dsts keep their rows; ggml-backend-meta.cpp
        // only enables this when no node reads them. The rows are the same ranges as the norm gather that follows.
        ar_node.outputs.clear();
        ar_node.outputs.push_back(dst_tensors[rank]);
        ((struct htp_allreduce_kernel_params *) ar_node.kernel_params)->n_dsts = 1;
    }

    ar_node.name = "ALLREDUCE+ADD";
    this->enqueue_op(ar_node);
}

void ggml_hexagon_session::enqueue_allreduce_rms_ag(const std::vector<const ggml_tensor *> & sync_tensors, const ggml_tensor * residual,
                                                    const ggml_tensor * weight, const std::vector<const ggml_tensor *> & norm_tensors,
                                                    uint32_t rank, uint32_t n_ranks, uint32_t fence_seq_entry, uint32_t fence_seq_exit,
                                                    float epsilon, bool tile) {
    htp_opnode ar_node(HTP_OP_ALLREDUCE);

    ggml_tensor * node = ar_node.add_dummy(*norm_tensors[rank]);
    node->op           = GGML_OP_NONE;
    node->op_params[0] = (int32_t) fence_seq_entry;
    node->op_params[1] = (int32_t) fence_seq_exit;
    memcpy(&node->op_params[2], &epsilon, sizeof(float));

    ar_node.init(node);

    // ALLREDUCE layout: n unused slots, n fences, then the residual at src[2 * n_ranks] and the norm weight after it.
    ar_node.inputs.clear();
    for (uint32_t i = 0; i < n_ranks; i++) {
        ar_node.inputs.push_back(nullptr);
    }
    for (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_node.inputs.push_back(ar_node.add_dummy(*sync_tensors[i]));
    }
    ar_node.inputs.push_back(residual);
    ar_node.inputs.push_back(weight);

    ar_node.outputs.clear();
    for (size_t i = 0; i < norm_tensors.size(); i++) {
        ar_node.outputs.push_back(norm_tensors[i]);
    }

    struct htp_allreduce_kernel_params kparams;
    if (!ggml_hexagon_precompute_rms_ag_params(this, norm_tensors[rank], rank, n_ranks, tile, &kparams)) {
        GGML_ABORT("ggml-hex: %s RMS_NORM+MUL(AG) tile output does not fit VTCM", this->c_name());
    }
    memcpy(ar_node.kernel_params, &kparams, sizeof(kparams));

    ar_node.name = "RMS_NORM+MUL(AG)";
    this->enqueue_op(ar_node);
}

// AE-03 fused owner-row op (op_allreduce_add_rms in the HTP). Blocks of 16 rows are split over the session threads.
// Rows: this rank's RS-ROW owner rows (ar_rs_row_chunk_rows), the rows its reduce writes. VTCM: the shared weight row, then per thread two slot sets of (n_ranks + 1) fp32 blocks, plus one fp16 staging block per set for tile output (HTP_ALLREDUCE_KERNEL_ADD_RMS_TILE).
static bool ggml_hexagon_precompute_add_rms_params(const ggml_hexagon_session * sess, const ggml_tensor * resid_out, uint32_t rank,
                                                   uint32_t n_ranks, bool tile, struct htp_allreduce_kernel_params * kparams,
                                                   uint32_t resid_mode = 0, uint32_t resid_off = 0) {
    memset(kparams, 0, sizeof(*kparams));
    const uint32_t block_rows = opt_ar_block_rows;   // rows per block: 16 (default) or 8; the HTP reads kparams->block_elems
    const uint32_t ne0       = (uint32_t) resid_out->ne[0];
    const uint32_t ne1       = (uint32_t) (resid_out->ne[1] * resid_out->ne[2] * resid_out->ne[3]);
    const uint32_t row_bytes = ne0 * (uint32_t) sizeof(float);
    const uint32_t rows_per  = ar_rs_row_chunk_rows(ne1, n_ranks);
    const uint32_t r0        = (std::min)(rank * rows_per, ne1);
    const uint32_t r1        = (std::min)(r0 + rows_per, ne1);
    const uint32_t rows      = r1 - r0;
    if (tile && (rows % block_rows != 0 || ne0 % 32 != 0)) {
        return false;
    }
    const size_t block_bytes = (size_t) block_rows * row_bytes;
    const size_t set_bytes   = (size_t) (n_ranks + 1) * block_bytes + (tile ? (size_t) 16 * row_bytes : 0);   // the tile staging block is always one 32-row fp16 tile row
    const size_t per_thread  = 2 * set_bytes;
    const uint32_t n_blocks  = (rows + block_rows - 1) / block_rows;
    uint32_t n_threads       = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, n_blocks));
    if (const char * v = getenv("GGML_HEXAGON_AR_MAX_THREADS")) n_threads = (std::min)(n_threads, (uint32_t) atoi(v));
    while (n_threads > 1 && row_bytes + n_threads * per_thread > sess->vtcm_size) {
        n_threads--;
    }
    if (row_bytes + n_threads * per_thread > sess->vtcm_size) {
        return false;
    }
    kparams->rank                 = (int32_t) rank;
    kparams->n_ranks              = (int32_t) n_ranks;
    kparams->n_threads            = (int32_t) n_threads;
    kparams->block_elems          = (int32_t) block_rows;
    kparams->elems_per_thread     = (int32_t) (((n_blocks + n_threads - 1) / n_threads) * block_rows);   // rows per thread
    kparams->kernel_type          = tile ? HTP_ALLREDUCE_KERNEL_ADD_RMS_TILE : HTP_ALLREDUCE_KERNEL_ADD_RMS;
    kparams->n_dsts               = (int32_t) n_ranks;
    kparams->ne0                  = (int32_t) ne0;
    kparams->ne1                  = (int32_t) ne1;
    kparams->row_size_aligned     = (int32_t) row_bytes;
    kparams->rank_elem_start      = (int32_t) r0;
    kparams->rank_nelem           = (int32_t) rows;
    kparams->vtcm_size_per_thread = (int32_t) per_thread;
    kparams->vtcm_size            = (int32_t) (row_bytes + n_threads * per_thread);
    kparams->resid_mode           = (int32_t) resid_mode;
    kparams->resid_off            = (int32_t) resid_off;
    if (getenv("GGML_HEXAGON_AR_SERIAL")) kparams->resid_mode |= (1 << 20);   // DIAG
    return true;
}

void ggml_hexagon_session::enqueue_allreduce_add_rms(const std::vector<const ggml_tensor *> & partials,
                                                     const std::vector<const ggml_tensor *> & sync_tensors,
                                                     const ggml_tensor * residual, const ggml_tensor * weight,
                                                     const ggml_tensor * resid_out,
                                                     const std::vector<const ggml_tensor *> & norm_tensors, uint32_t rank,
                                                     uint32_t n_ranks, uint32_t fence_seq_entry, uint32_t fence_seq_exit,
                                                     float epsilon, bool tile, uint32_t resid_mode, uint32_t resid_off) {
    htp_opnode ar_node(HTP_OP_ALLREDUCE_ADD);

    ggml_tensor * node = ar_node.add_dummy(*resid_out);
    node->op           = GGML_OP_NONE;
    node->op_params[0] = (int32_t) fence_seq_entry;
    node->op_params[1] = (int32_t) fence_seq_exit;
    memcpy(&node->op_params[2], &epsilon, sizeof(float));

    ar_node.init(node);

    // Inputs: n partials, n fences, the residual, the norm weight. Outputs: the residual rows (dsts[0]), then the n norm outputs.
    ar_node.inputs.clear();
    for (size_t i = 0; i < partials.size(); i++) {
        ar_node.inputs.push_back(partials[i]);
    }
    for (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_node.inputs.push_back(ar_node.add_dummy(*sync_tensors[i]));
    }
    ar_node.inputs.push_back(residual);
    ar_node.inputs.push_back(weight);

    ar_node.outputs.clear();
    ar_node.outputs.push_back(resid_out);
    for (size_t i = 0; i < norm_tensors.size(); i++) {
        ar_node.outputs.push_back(norm_tensors[i]);
    }

    struct htp_allreduce_kernel_params kparams;
    if (!ggml_hexagon_precompute_add_rms_params(this, resid_out, rank, n_ranks, tile, &kparams, resid_mode, resid_off)) {
        GGML_ABORT("ggml-hex: %s ALLREDUCE+ADD+RMS_NORM+MUL does not fit VTCM", this->c_name());
    }
    memcpy(ar_node.kernel_params, &kparams, sizeof(kparams));

    ar_node.name = "ALLREDUCE+ADD+RMS_NORM+MUL(AG)";
    this->enqueue_op(ar_node);
}

// ============================================================================ fast op-enqueue: plan capture and replay (decode-only port; see htp_graph_plan) Slot encoding (htp_graph_plan::op_entry::slot): -1 = no tensor 0xC000|k = entry tensor k (k < sig_n_tens; live-verified by pointer at replay) 0x4000|k = tensor appended by this plan's replay (k indexes new_tens) ============================================================================

void ggml_hexagon_session::plan_begin(htp_graph_plan & plan) {
    plan.sig_n_ops  = op_batch->n_ops;
    plan.sig_n_tens = op_batch->n_tens;
    plan.sig_n_bufs = op_batch->n_bufs;
    plan.sig_b_vmem = op_batch->b_vmem;
    op_batch->rec_active     = true;
    op_batch->rec_flush_seen = false;
    op_batch->rec_plan_sig   = plan.sig_n_ops;
}

// Build `plan` from the final batch state after a slow subgraph run.
bool ggml_hexagon_session::plan_finish(htp_graph_plan & plan) {
    auto * b = op_batch;
    b->rec_active = false;
    const bool flush_seen = b->rec_flush_seen;
    b->rec_flush_seen = false;

    if (flush_seen) { plan.dirty = true; return false; }
    if (b->n_ops == plan.sig_n_ops) { return false; } // nothing appended; nothing to replay

    plan.new_bufs.assign(b->h_bufs.begin() + plan.sig_n_bufs, b->h_bufs.begin() + b->n_bufs);
    plan.new_tens.assign(b->h_tens.begin() + plan.sig_n_tens, b->h_tens.begin() + b->n_tens);
    plan.new_tens_fd.clear();
    plan.new_tens_data.clear();
    plan.new_tens_ptr.clear();
    plan.entry_tens.assign(plan.sig_n_tens, nullptr);
    std::vector<std::pair<int, const ggml_tensor *>> new_ptrs;
    new_ptrs.reserve(b->n_tens - plan.sig_n_tens);
    for (const auto & kv : b->t_map) {
        if (kv.second < (int) plan.sig_n_tens) {
            plan.entry_tens[kv.second] = kv.first;
        } else {
            new_ptrs.emplace_back(kv.second - plan.sig_n_tens, kv.first);
        }
    }
    plan.entry_tens_data.clear();
    plan.entry_tens_data.resize(plan.sig_n_tens, 0);
    for (uint32_t i = 0; i < plan.sig_n_tens; i++) {
        if (plan.entry_tens[i]) plan.entry_tens_data[i] = (uint64_t) plan.entry_tens[i]->data;
    }
    std::sort(new_ptrs.begin(), new_ptrs.end(),
              [](const auto & a, const auto & c) { return a.first < c.first; });
    if (new_ptrs.size() != plan.new_tens.size()) { plan.dirty = true; return false; }
    for (auto & p : new_ptrs) {
        plan.new_tens_ptr.push_back(p.second);
        plan.new_tens_fd.push_back(b->h_bufs[plan.new_tens[p.first].bi].fd);
        plan.new_tens_data.push_back((uint64_t) p.second->data);
    }

    plan.ops.clear();
    plan.ops.reserve(b->n_ops - plan.sig_n_ops);
    for (uint32_t i = plan.sig_n_ops; i < b->n_ops; i++) {
        htp_graph_plan::op_entry e {};
        const htp_op_desc * d = &b->h_ops[i];
        memcpy(&e.desc, d, sizeof(htp_op_desc));
        bool bad = false;
        const uint32_t n_slots = HTP_OP_MAX_INPUTS + HTP_OP_MAX_OUTPUTS;
        for (uint32_t s = 0; s < n_slots; s++) {
            uint16_t idx = (s < HTP_OP_MAX_INPUTS) ? d->src[s] : d->dst[s - HTP_OP_MAX_INPUTS];
            if (idx == 0xffff) {
                e.slot[s] = -1;
            } else if (idx < plan.sig_n_tens) {
                if (!plan.entry_tens[idx]) { bad = true; break; }
                e.slot[s] = (int16_t)(0xC000 | idx);
            } else {
                e.slot[s] = (int16_t)(0x4000 | (idx - plan.sig_n_tens));
            }
        }
        if (bad) { plan.dirty = true; return false; }
        plan.ops.push_back(e);
    }
    plan.nodes.assign(b->ops.begin() + plan.sig_n_ops, b->ops.begin() + b->n_ops);
    return true;
}

// Validate + append `plan` to the live batch. Returns false with the batch
// left untouched when the live state is not the recorded entry state.
bool ggml_hexagon_session::plan_replay(const htp_graph_plan & plan) {
    auto * b = op_batch;

    if (b->n_ops  != plan.sig_n_ops  || b->n_tens != plan.sig_n_tens ||
        b->n_bufs != plan.sig_n_bufs || b->b_vmem != plan.sig_b_vmem) return false;
    if (b->n_ops + plan.ops.size()       >= b->n_ops_max) return false;
    if (b->n_bufs + plan.new_bufs.size() > b->n_bufs_max) return false;
    if (b->n_tens + plan.new_tens.size() > b->n_tens_max) return false;

    // Pass 1 (read-only): every entry-tensor reference still identifies the recorded memory (live h_tens[idx] offset + buffer base must equal the recorded t->data; no dereference of the recorded ggml objects, which are freed after their token); every new buffer is absent from the live batch; every new tensor's entry-buffer reference is the recorded buffer.
    for (const auto & e : plan.ops) {
        for (uint32_t s = 0; s < HTP_OP_MAX_INPUTS + HTP_OP_MAX_OUTPUTS; s++) {
            const int16_t m = e.slot[s];
            if (m == -1) continue;               // no tensor
            if ((m & 0xC000) != 0xC000) continue; // new_tens (0x4000|k): not an entry ref
            const uint32_t idx = (uint16_t) m & 0x3FFF; // entry k (0xC000|k is negative int16!)
            if (idx >= (uint32_t) plan.entry_tens_data.size() || !plan.entry_tens_data[idx]) return false;
            const htp_tensor & ht = b->h_tens[idx];
            const uint64_t live = (uint64_t) b->h_bufs[ht.bi].base + ht.data;
            if (live != plan.entry_tens_data[idx]) return false;
        }
    }
    for (const auto & nb : plan.new_bufs) {
        if (b->b_map.find(nb.fd) != b->b_map.end()) return false;
    }
    for (size_t k = 0; k < plan.new_tens.size(); k++) {
        const htp_tensor & nt = plan.new_tens[k];
        if (nt.bi < plan.sig_n_bufs && b->h_bufs[nt.bi].fd != plan.new_tens_fd[k]) return false;
    }

    // Pass 2: pure append; nothing left to fail.
    for (const auto & nb : plan.new_bufs) {
        uint32_t bi = b->n_bufs++;
        GGML_ASSERT(bi < b->n_bufs_max);
        b->h_bufs[bi] = nb;
        b->b_map[nb.fd] = bi;
        if ((nb.flags & HTP_BUF_EXTENDED) == 0) {
            b->b_vmem += nb.size;
        }
    }
    for (size_t k = 0; k < plan.new_tens.size(); k++) {
        const htp_tensor & nt = plan.new_tens[k];
        uint32_t ti = b->n_tens++;
        GGML_ASSERT(ti < b->n_tens_max);
        b->h_tens[ti] = nt;
        b->t_map[plan.new_tens_ptr[k]] = ti;
        b->d_map.insert({(void *) plan.new_tens_data[k], ti});
    }
    for (const auto & e : plan.ops) {
        uint32_t n = b->n_ops++;
        GGML_ASSERT(n < b->n_ops_max);
        b->h_ops[n] = e.desc;
    }
    if (plan.nodes.size() == plan.ops.size()) {
        for (size_t i = 0; i < plan.nodes.size(); i++) {
            b->ops[plan.sig_n_ops + i] = plan.nodes[i];
        }
    }
    return true;
}

ggml_hexagon_shared_buffer * ggml_hexagon_session::mmap_tensor(const ggml_tensor * t) {
    if (!t) return nullptr;

    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
    if (!sbuf->mapped) {
        const bool is_weight = ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        const bool extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    return sbuf;
}

bool ggml_hexagon_session::clone_buffer(const ggml_hexagon_shared_buffer * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    if (sbuf->sess == this) return true;

    auto mem = sbuf->mem;
    int   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    if (this->cloned_buffers.find(fd) != this->cloned_buffers.end()) return true;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", this->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    auto clone = std::make_unique<ggml_hexagon_shared_buffer>(this, *sbuf);
    try {
        clone->mmap(sbuf->extended);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", this->c_name(), exc.what());
        return false;
    }

    this->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(this);
    return true;
}

void ggml_hexagon_session::release_buffer(const ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    int   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    auto it = this->cloned_buffers.find(fd);
    if (it != this->cloned_buffers.end()) {
        auto clone = std::move(it->second);
        this->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(this);
}

void ggml_hexagon_session::unclone_buffer(const ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    for (auto * sess : sessions) {
        sess->release_buffer(sbuf);
    }
}

static size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    const size_t MiB = 1024 * 1024;
    const size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t step = 256u * MiB;

    try {
        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;
        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;
        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;

        while (1) {
            sbufs.push_back(new ggml_hexagon_shared_buffer(sess, step, true));
            vmem += step;
        }
    } catch (...) { }

    for (auto b : sbufs) { delete b; }

    return vmem - step; // backoff to account for overhead from internal mappings
}

// GGML_HEXAGON_HMX_BENCH=1 (experiment): isolated HMX job loop on the DSP (matmul-ops.c htp_hmx_bench_run). One log line per
// config; the process exits after the sweep. cyc/crouton = cycles / (iters * n_row * n_col * n_dot), cyc/tile = cycles / (iters * n_row * n_col).
static void ggml_hexagon_hmx_bench(remote_handle64 handle) {
    struct bench_cfg { const char * name; uint32_t variant, n_row, n_col, n_dot, n_writers; };
    static const bench_cfg cfgs[] = {
        { "v0 prod r16 c11 d80",       0, 16, 11,  80, 0 },
        { "v1 swap r16 c11 d80",       1, 16, 11,  80, 0 },
        { "v2 nostore r16 c11 d80",    2, 16, 11,  80, 0 },
        { "v3 maconly r16 c11 d80",    3, 16, 11,  80, 0 },
        { "v0 r16 c11 d32",            0, 16, 11,  32, 0 },
        { "v0 r16 c11 d64",            0, 16, 11,  64, 0 },
        { "v0 r16 c11 d160",           0, 16, 11, 160, 0 },
        { "v0 r4 c4 d320",             0,  4,  4, 320, 0 },
        { "v1 r4 c4 d320",             1,  4,  4, 320, 0 },
        { "v2 r4 c4 d320",             2,  4,  4, 320, 0 },
        { "v3 r4 c4 d320",             3,  4,  4, 320, 0 },
        { "v0 r1 c11 d80",             0,  1, 11,  80, 0 },
        { "v0 r16 c1 d80",             0, 16,  1,  80, 0 },
        { "v0 r16 c11 d80 w1",         0, 16, 11,  80, 1 },
        { "v0 r16 c11 d80 w2",         0, 16, 11,  80, 2 },
        { "v0 r16 c11 d80 w4",         0, 16, 11,  80, 4 },
        { "v3 r16 c11 d80 w2",         3, 16, 11,  80, 2 },
        { "v3 r16 c11 d80 w4",         3, 16, 11,  80, 4 },
        { "v4 int8 r16 c11 d80",       4, 16, 11,  80, 0 },
        { "v4 int8 r4 c4 d320",        4,  4,  4, 320, 0 },
        { "v4 int8 r16 c11 d32",       4, 16, 11,  32, 0 },
        { "v5 int8 1KB r16 c11 d80",   5, 16, 11,  80, 0 },
        { "v5 int8 1KB r4 c4 d320",    5,  4,  4, 320, 0 },
        { "v6 int8 2KB r16 c11 d80",   6, 16, 11,  80, 0 },
        { "v7 int4 r16 c11 d80",       7, 16, 11,  80, 0 },
        // correct-form int dataflow (n_row = 64-row act tiles for v8..v14, 32-row tiles for v15; n_col = weight units)
        { "v15 fp16 r16 c11 d80",      15, 16, 11,  80, 0 },
        { "v8 a8w8 r8 c11 d80",        8,  8, 11,  80, 0 },
        { "v9 a8w8 ch64 r8 c11 d80",   9,  8, 11,  80, 0 },
        { "v10 a8w4 n1x r8 c11 d80",   10, 8, 11,  80, 0 },
        { "v11 a8w4 2x r8 c11 d80",    11, 8, 11,  80, 0 },
        { "v12 a8w4 2x deep r8 c11 d80", 12, 8, 11, 80, 0 },
        { "v13 a8w8 wdeep r8 c11 d80", 13, 8, 11,  80, 0 },
        { "v14 a8w4 n1x wdeep r8 c11 d80", 14, 8, 11, 80, 0 },
        { "v15 fp16 r4 c4 d320",       15,  4,  4, 320, 0 },
        { "v16 fp16 wdeep r16 c5 d80", 16, 16,  5,  80, 0 },
        { "v17 fp16 wdeep 1cr r16 c5 d80", 17, 16, 5, 80, 0 },
        { "v16 fp16 wdeep r4 c2 d320", 16,  4,  2, 320, 0 },
        { "v15 fp16 r16 c5 d80",       15, 16,  5,  80, 0 },
        // v21: n_row = KG (k-tiles per readout), n_col = column units, n_dot = k-tiles per unit, n_writers = HVX consumers
        { "v25 a8w8 selfcheck b2 c3 k16", 25, 2, 3, 16, 0 }, { "v25 a8w8 selfcheck b8 c24 k80", 25, 8, 24, 80, 0 },
        { "v24 q4m expand w1", 24, 1, 480, 1, 1 },
        { "v24 q4m expand w2", 24, 1, 480, 1, 2 },
        { "v24 q4m expand w4", 24, 1, 480, 1, 4 },
        { "v24 q4m expand w6", 24, 1, 480, 1, 6 },
        { "v24 q4m expand w8", 24, 1, 480, 1, 8 },
        { "v22 pp2 whole-K w1", 22, 3, 8, 23, 1 },
        { "v22 pp2 whole-K w2", 22, 3, 8, 23, 2 },
        { "v22 pp2 whole-K w4", 22, 3, 8, 23, 4 },
        { "v22 pp2 whole-K w8", 22, 3, 8, 23, 8 },
        { "v22 pp stage1 w1", 22, 1, 8, 1, 1 },
        { "v22 pp stage1 w2", 22, 1, 8, 1, 2 },
        { "v22 pp stage1 w4", 22, 1, 8, 1, 4 },
        { "v22 pp stage1 w8", 22, 1, 8, 1, 8 },
        { "v22 pp stage2 w1", 22, 2, 8, 1, 1 },
        { "v22 pp stage2 w2", 22, 2, 8, 1, 2 },
        { "v22 pp stage2 w4", 22, 2, 8, 1, 4 },
        { "v22 pp stage2 w8", 22, 2, 8, 1, 8 },
        { "v22 pp stage3 w1", 22, 3, 8, 1, 1 },
        { "v22 pp stage3 w2", 22, 3, 8, 1, 2 },
        { "v22 pp stage3 w4", 22, 3, 8, 1, 4 },
        { "v22 pp stage3 w8", 22, 3, 8, 1, 8 },
        { "v21 verify kg8 w0", 21, 8, 1, 8, 0 }, { "v21 verify kg2 w0", 21, 2, 1, 8, 0 },
        { "v21 pipe kg8 w0",  21, 8, 8, 80, 0 }, { "v21 pipe kg8 w1", 21, 8, 8, 80, 1 }, { "v21 pipe kg8 w2", 21, 8, 8, 80, 2 },
        { "v21 pipe kg8 w4",  21, 8, 8, 80, 4 }, { "v21 pipe kg8 w8", 21, 8, 8, 80, 8 },
        { "v21 pipe kg16 w0", 21, 16, 8, 80, 0 }, { "v21 pipe kg16 w2", 21, 16, 8, 80, 2 }, { "v21 pipe kg16 w4", 21, 16, 8, 80, 4 },
        { "v21 pipe kg4 w0",  21, 4, 8, 80, 0 }, { "v21 pipe kg4 w4", 21, 4, 8, 80, 4 }, { "v21 pipe kg4 w8", 21, 4, 8, 80, 8 },
        { "v21 pipe kg2 w0",  21, 2, 8, 80, 0 }, { "v21 pipe kg2 w8", 21, 2, 8, 80, 8 },
        { "v18 a8w8 full r4 c4 d80",   18,  4,  4,  80, 0 },
        { "v19 a8w4 2x full r4 c4 d80", 19,  4,  4,  80, 0 },
        { "v20 a8w4 2x deep full r4 c2 d80", 20, 4, 2, 80, 0 },
        { "v18 a8w8 full r4 c4 d8",    18,  4,  4,   8, 0 },
        { "v19 a8w4 2x full r4 c4 d8", 19,  4,  4,   8, 0 },
        { "v20 a8w4 2x deep full r4 c2 d8", 20, 4, 2, 8, 0 },
        { "v18 a8w8 full r4 c4 d320",  18,  4,  4, 320, 0 },
        { "v19 a8w4 2x full r4 c4 d320", 19, 4,  4, 320, 0 },
        { "v20 a8w4 2x deep full r4 c2 d320", 20, 4, 2, 320, 0 },
        { "v8 a8w8 r4 c4 d320",         8,  4,  4, 320, 0 },
        { "v10 a8w4 n1x r4 c4 d320",    10,  4,  4, 320, 0 },
        { "v11 a8w4 2x r4 c4 d320",     11,  4,  4, 320, 0 },
        { "v12 a8w4 2x deep r4 c4 d320", 12, 4, 4, 320, 0 },
    };
    for (const auto & c : cfgs) {
        const uint32_t iters = (c.variant == 21 && strstr(c.name, "verify")) ? 1 : c.variant == 22 ? 2000 : c.variant == 24 ? 200 : c.variant == 25 ? 1 : c.variant == 21 ? std::max<uint32_t>(1, 400000u / (c.n_col * c.n_dot * 32)) : std::max<uint32_t>(1, 400000u / (c.n_row * c.n_col * c.n_dot));
        htp_iface_hmx_bench_cfg cfg{ c.variant, c.n_row, c.n_col, c.n_dot, iters, c.n_writers };
        unsigned long long cycles = 0, usec = 0, wbytes = 0;
        const int err = htp_iface_hmx_bench(handle, &cfg, &cycles, &usec, &wbytes);
        const double tiles    = (double) iters * c.n_row * c.n_col;
        const double croutons = tiles * c.n_dot;
        GGML_LOG_ERROR("hmx-bench %-24s iters %6u err 0x%x cycles %12llu usec %9llu MHz %7.1f cyc/crouton %7.3f cyc/tile %8.1f writer MB/s %8.1f wb_raw %llu\n",
                       c.name, iters, (unsigned) err, cycles, usec,
                       usec ? (double) cycles / (double) usec : 0.0,
                       cycles / croutons, cycles / tiles,
                       usec ? (double) wbytes / (double) usec : 0.0, wbytes);
    }
}

void ggml_hexagon_session::allocate(const ggml_hexagon_device_config & config) noexcept(false) {
    int phys_idx = config.physical_idx;
    int virt_idx = config.virtual_idx;

    this->valid_session = false;
    this->valid_handle  = false;
    this->valid_queue   = false;
    this->valid_iface   = false;

    this->name          = config.name;
    this->phys_idx      = phys_idx;
    this->virt_idx      = virt_idx;
    this->domain_id     = config.domain_id;
    this->session_id    = 0;
    this->batch_req_seq = 0;
    this->batch_rsp_seq = 0;
    this->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", this->name.c_str(), this->domain_id, phys_idx, virt_idx);

    if (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        throw std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    const std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    if (virt_idx > 0) {
        struct remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<char *>(dom_name.c_str());
        n.session_name     = const_cast<char *>(this->name.c_str());
        n.session_name_len = this->name.size();
        n.session_id       = virt_idx;

        int err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (void *) &n, sizeof(n));
        if (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           this->c_name(), phys_idx, virt_idx, err);
            throw std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        this->session_id    = n.session_id;
        this->domain_id     = n.effective_domain_id;
        this->valid_session = true;
    } else {
        struct remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<char *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        int err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (void *) &eff, sizeof(eff));
        if (err == AEE_SUCCESS) {
            this->domain_id = eff.effective_domain_id;
        } else {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           this->name.c_str(), err, this->domain_id);
        }
    }

    // Enable unsigned modules
    {
        struct remote_rpc_control_unsigned_module u;
        u.domain = this->domain_id;
        u.enable = 1;
        int err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *) &u, sizeof(u));
        if (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", this->c_name(), err);
            throw std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    char session_uri[256];
    {
        char htp_uri[256];
        snprintf(htp_uri, sizeof(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        struct remote_rpc_get_uri u = {};
        u.session_id      = this->session_id;
        u.domain_name     = const_cast<char *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<char *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = sizeof(session_uri);

        int err = remote_session_control(FASTRPC_GET_URI, (void *) &u, sizeof(u));
        if (err != AEE_SUCCESS) {
            snprintf(session_uri, sizeof(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), this->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          this->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    int err = htp_iface_open(session_uri, &this->handle);
    if (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", this->c_name(), session_uri, err);
        throw std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    this->valid_handle = true;

    // Query HW info and resolve session options
    {
        unsigned int hw_n_threads = 0;
        unsigned int hw_n_hvx     = 0;
        unsigned int hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        int hw_err = htp_iface_hwinfo(this->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        if (hw_err == 0) {
            const uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            this->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_threads) : max_n_threads;
            this->n_hvx     = this->n_threads;
            this->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            this->vtcm_size = (uint64_t)hw_vtcm_size;
            this->vtcm_size_hw = this->vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          this->c_name(), this->n_threads, this->n_hvx, this->n_hmx,
                          (unsigned long long)(this->vtcm_size / (1024 * 1024)));
        } else {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", this->c_name(), hw_err);
            const uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            this->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_threads;
            this->n_hvx     = this->n_threads;
            this->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            this->vtcm_size = 8 * 1024 * 1024;
            this->vtcm_size_hw = this->vtcm_size;
        }
    }
    // Lane experiment: the host plans kparams against sess->vtcm_size, so it must match the DSP main-lane budget
    if (opt_lane_threads > 0) {
        if (opt_lane_a_mb <= 0) {
            opt_lane_a_mb = 8;
        }
        if (this->n_threads + (uint32_t) opt_lane_threads > HTP_MAX_NTHREADS) {
            GGML_LOG_WARN("ggml-hex: GGML_HEXAGON_LANE_THREADS=%d too large for %u threads; clamped\n", opt_lane_threads, this->n_threads);
            opt_lane_threads = (int) HTP_MAX_NTHREADS - (int) this->n_threads;
        }
    }
    if (opt_lane_a_mb > 0 && (uint64_t) opt_lane_a_mb * 1024 * 1024 < this->vtcm_size) {
        this->vtcm_size = (uint64_t) opt_lane_a_mb * 1024 * 1024;
    }

    // Enable FastRPC QoS mode
    {
        struct remote_rpc_control_latency l;
        l.enable = 1;

        int err = remote_handle64_control(this->handle, DSPRPC_CONTROL_LATENCY, (void *) &l, sizeof(l));
        if (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", this->c_name(),
                  this->session_id, this->domain_id, session_uri, (unsigned long) this->handle);

    const size_t req_q_size = (sizeof(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    const size_t rsp_q_size = (sizeof(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(this->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (void *) this,  // Callback context
                          &queue);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", this->name.c_str(), (unsigned) err);
        throw std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    this->valid_queue = true;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &this->queue_id);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        throw std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    if (opt_etm) {
        err = htp_iface_etm(this->handle, 1);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    this->op_queue = new ggml_hexagon_opqueue(this, opt_opbatch, opt_opqueue);

    this->fence_buf = new ggml_hexagon_fence_buffer(this, &dev_ctx->fence_buffer_type, 64 * 1024);
    if (this->mdev.count > 1) {
        this->mdev_fence_slot = this->alloc_fence(this->mdev.count);
    }

    if (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(this);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", this->c_name(), opt_vmem);
    }
    const size_t shm_size = this->op_queue->shm_size();
    this->max_vmem = (opt_vmem > shm_size) ? (opt_vmem - shm_size) : opt_vmem;

    this->op_batch = new ggml_hexagon_opbatch(this, opt_opbatch, this->max_vmem);

    // Start dspqueue/opbatch processing
    // lane packing (htp/main.c): bits 0-7 HMX on/off, 8-15 HVX threads of lane B, 16-23 main-lane VTCM budget (MB)
    const uint32_t hmx_lane_pack = (opt_nhmx ? 1u : 0u) | ((uint32_t) opt_lane_threads << 8) | ((uint32_t) opt_lane_a_mb << 16);
    err = htp_iface_start(this->handle, this->session_id, this->queue_id, this->n_threads, hmx_lane_pack, this->max_vmem);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", this->c_name(), (unsigned) err);
        throw std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    this->valid_iface = true;

    if (getenv("GGML_HEXAGON_HMX_BENCH")) {
        ggml_hexagon_hmx_bench(this->handle);
        _exit(0);
    }

    if (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(this->handle, opt_profile, &pmu_conf);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }
}

void ggml_hexagon_session::release() noexcept(true) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", this->name.c_str());
    if (this->act_tile_buf) {
        // fp16 activation tiles: views first, then the scratch buffer and its rpcmem block.
        this->act_tile_views.clear();
        this->gather_views.clear();
        this->pending_rms_ag.clear();
        this->pending_add_rms.clear();
        this->act_tile_ok.clear();
        this->act_tile_live.clear();
        delete static_cast<ggml_hexagon_shared_buffer *>(this->act_tile_buf->context);
        delete this->act_tile_buf;
        this->act_tile_buf = nullptr;
    }

    if (this->swiglu_buf) {
        // SWIGLU output tiles: views first, then the buffer and its rpcmem block.
        this->swiglu_views.clear();
        this->swiglu_ok.clear();
        delete static_cast<ggml_hexagon_shared_buffer *>(this->swiglu_buf->context);
        delete this->swiglu_buf;
        this->swiglu_buf = nullptr;
    }

    if (this->norm_tile_buf) {
        // AE-03 norm output tiles: views first, then the peers' clones (they wrote into this buffer), then the buffer and its rpcmem block.
        this->norm_tile_views.clear();
        this->act_tile_live.clear();
        ggml_hexagon_shared_buffer * sbuf = static_cast<ggml_hexagon_shared_buffer *>(this->norm_tile_buf->context);
        this->unclone_buffer(sbuf);
        delete sbuf;
        delete this->norm_tile_buf;
        this->norm_tile_buf = nullptr;
    }

    if (this->zero_state_buf) {
        this->zero_state_clear_skip.clear();
        this->zero_state_dead.clear();
        this->zero_state_gdn.clear();
        this->zero_state_conv.clear();
        delete static_cast<ggml_hexagon_shared_buffer *>(this->zero_state_buf->context);
        delete this->zero_state_buf;
        this->zero_state_buf = nullptr;
    }

    const uint64_t kp_hits   = this->kparams_hits.load();
    const uint64_t kp_misses = this->kparams_misses.load();
    const uint64_t kp_total  = kp_hits + kp_misses;
    if (kp_total > 0) {
        GGML_LOG_INFO("ggml-hex: %s kparams-cache: hits %llu misses %llu hit-rate %.1f%%\n",
                       this->name.c_str(), (unsigned long long) kp_hits, (unsigned long long) kp_misses,
                       100.0 * (double) kp_hits / (double) kp_total);
    }

    this->mdev.sessions.clear();

    int err;

    if (this->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(this->handle);
        if (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    delete this->op_batch;
    delete this->op_queue;
    for (auto & it : this->cpy_fence_slots) {
        free_fence((void *) it.second, 1);
    }
    this->cpy_fence_slots.clear();

    if (this->fence_buf) {
        unclone_buffer(this->fence_buf);
        delete this->fence_buf;
        this->fence_buf = nullptr;
    }
    while (!this->cloned_buffers.empty()) {
        release_buffer(this->cloned_buffers.begin()->second.get());
    }

    if (opt_etm) {
        err = htp_iface_etm(this->handle, 0);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    if (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(this->handle, 0, &pmu_conf);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    if (this->valid_queue) {
        err = dspqueue_close(queue);
        if (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (unsigned) err);
        }
    }

    if (this->valid_handle) {
        htp_iface_close(this->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(const ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_count) noexcept(false) {
    this->dev        = dev;
    this->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    this->mdev.idx   = mdev_idx;
    this->mdev.count = mdev_count > 0 ? mdev_count : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)this) & 0xFFFF;

    try {
        allocate(config);
        if (this->mdev.idx == 0 && !config.mdev_group.empty()) {
            for (size_t i = 0; i < config.mdev_group.size(); i++) {
                this->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], this->dev, (uint32_t) (i + 1), this->mdev.count));
            }
        }
    } catch (const std::exception & exc) {
        release();
        throw;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(true) {
    release();
}

// ** backend interface

static bool ggml_hexagon_flash_attn_is_hmx_eligible(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * q,
    const struct ggml_tensor * k,
    const struct ggml_tensor * v,
    const struct ggml_tensor * sinks
) {
    if (sess->n_hmx == 0) {
        return false;
    }

    if (opt_fa_select < 2) {
        return false;
    }

    if ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        return false;
    }

    const uint32_t DK = q->ne[0];
    const uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    if (DK % 8 != 0 || DV % 8 != 0) {
        return false;
    }

    // Fall back to HVX for small token counts if head dimension is small (DK <= 128)
    const uint32_t neq1 = q->ne[1];
    if (DK <= 128 && neq1 < 5) {
        return false;
    }

    return true;

    GGML_UNUSED(sinks);
}

static bool ggml_hexagon_precompute_flash_attn_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_fa_kernel_params * kparams
) {
    if (opt_fa_select < 1) {
        return false;
    }

    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * q    = op->src[0];
    const struct ggml_tensor * k    = op->src[1];
    const struct ggml_tensor * v    = op->src[2];
    const struct ggml_tensor * mask = op->src[3];
    const struct ggml_tensor * dst  = op;

    const uint32_t neq0 = q->ne[0];  // head_dim (DK)
    const uint32_t neq1 = q->ne[1];  // n_tokens
    const uint32_t neq2 = q->ne[2];  // n_heads

    const uint32_t nek1 = k->ne[1];  // kv_len

    const uint32_t nev0 = v->ne[0];  // head_dim (DV)

    const uint32_t DK = neq0;
    const uint32_t DV = nev0;

    const uint32_t n_kv_heads = k->ne[2];
    const uint32_t G          = neq2 / n_kv_heads;

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], sizeof(float));
    memcpy(&max_bias,      &op->op_params[1], sizeof(float));
    memcpy(&logit_softcap, &op->op_params[2], sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->is_q_fp32 = (q->type == GGML_TYPE_F32) ? 1 : 0;
    kparams->is_dst_fp32 = (dst->type == GGML_TYPE_F32) ? 1 : 0;
    kparams->G = G;

    const uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    const struct ggml_tensor * sinks = op->src[4];
    if (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        const uint32_t DK_pad = hex_round_up(DK, 64);
        const uint32_t DV_pad = hex_round_up(DV, 64);
        size_t Br = 0, Bc = 0;
        int ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, kparams->is_q_fp32 != 0, sinks != nullptr, n_head);
        if (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, kparams->is_q_fp32 != 0, sinks != nullptr, n_head);

            const size_t row_vec_bytes = hex_align_up(Bc * sizeof(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            const size_t m_line_bytes = hex_align_up(Bc * sizeof(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_line_bytes / sizeof(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_values(G);
            if (mask) {
                kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            return true;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    const size_t size_q_row_padded = hex_round_up(q->ne[0] * (kparams->is_q_fp32 ? 4 : 2), 128);
    const size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    const size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, kparams->is_q_fp32 != 0, mask != nullptr, sinks != nullptr, n_head, sess->n_threads);

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_values(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_values(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_values(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_values(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_values(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_values(q->ne[3]/v->ne[3]);
    if (mask) {
        kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    return true;
}

static bool ggml_hexagon_supported_flash_attn_ext(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * src3 = op->src[3];
    const struct ggml_tensor * src4 = op->src[4];
    const struct ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    if ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        return false;
    }

    if (src3 && src3->type != GGML_TYPE_F16) {  // mask
        return false;
    }

    if (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        return false;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return false;
    }

    if (dst->ne[3] != 1) {
        return false;
    }

    struct htp_fa_kernel_params kparams;
    if (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        return false;
    }

    if ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        return false;
    }

    return true;
}

// The chunked HMX kernel writes the final state (snapshot plane 0) only. A K > 1 op may use it when its typed snapshot
// range is exactly that (op_params[3] = d0 == 0, op_params[4] = n == 1, see llama_set_rs_snap_range(ctx, 0, 1)) and the switch is on.
static bool ggml_hexagon_gdn_hmx_snap_ok(const struct ggml_tensor * op, int64_t K) {
    if (K == 1) {
        return true;
    }
    return opt_gdn_prompt_hmx != 0 && ggml_get_op_params_i32(op, 3) == 0 && ggml_get_op_params_i32(op, 4) == 1;
}

static bool ggml_hexagon_supported_gated_delta_net(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    if (opt_gdn_select < 1) {
        return false;
    }

    const struct ggml_tensor * q     = op->src[0];
    const struct ggml_tensor * k     = op->src[1];
    const struct ggml_tensor * v     = op->src[2];
    const struct ggml_tensor * g     = op->src[3];
    const struct ggml_tensor * beta  = op->src[4];
    const struct ggml_tensor * state = op->src[5];
    const struct ggml_tensor * dst   = op;

    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 ||
        (state->type != GGML_TYPE_F32 && state->type != GGML_TYPE_F16) ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t S_v      = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(op, 0);

    if (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        return false;
    }
    if (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        return false;
    }
    if ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        return false;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    if (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        return false;
    }
    if (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        return false;
    }

    const uint32_t total_rows = (uint32_t) (H * n_seqs);
    const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    const bool can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                                 ggml_hexagon_gdn_hmx_snap_ok(op, K) &&
                                 opt_gdn_hmx != 0 && (state->type != GGML_TYPE_F16);

    if (can_use_hmx) {
        struct htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        if (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_SIZE, (uint32_t) H, total_rows, sess->vtcm_size, n_threads, true, opt_gdn_opt,
                                      htp_gdn_skew_bytes((uint8_t) opt_gdn_fix, (uint8_t) opt_gdn_skew_units, n_tokens), &n_heads_batch)) {
            return false;
        }
    } else {
        struct htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_select(&layout, (uint32_t) S_v, n_threads, (uint8_t) opt_gdn_opt);
        if (layout.total_bytes > sess->vtcm_size) {
            return false;
        }
    }

    return true;
}

static bool ggml_hexagon_matmul_is_hmx_eligible(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int ne01_padded,
    bool is_matmul_id,
    bool is_batched
) {
    if (src1->type != GGML_TYPE_F32) {
        return false;
    }

    const int ne00  = src0->ne[0];
    const int ne11  = src1->ne[1];
    const int ne12  = src1->ne[2];
    const int wtype = src0->type;

    // HMX weight tile requires N to be 32-aligned.
    if (ne01_padded % 32 != 0) {
        return false;
    }

    // HMX supports F16, F32, and repack quantized types.
    if (!ggml_hexagon_is_hmx_weight_type((ggml_type) wtype)) {
        return false;
    }

    // HMX paths require K aligned to 32 (F16 weights get K-padded on the device side instead -- see hmx_mm_2d_f32's needs_k_pad -- so they are exempt from this hard requirement).
    if (wtype != GGML_TYPE_F16 && ne00 % 32 != 0) {
        return false;
    }

    // Quantized HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    if (!is_matmul_id && is_batched && wtype != GGML_TYPE_F16) {
        return false;
    }

    // HMX assumes contiguous row-major layout.
    if (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        return false;
    }

    // Native Q4_0 integer matmul targets speculative verify batches. Single-row draft and bulk prefill keep their tuned paths.
    const int m = is_matmul_id ? ne12 : ne11;
    const bool native_int4 = opt_hmx_int && wtype == GGML_TYPE_Q4_0 && !is_matmul_id && !is_batched && m > 1 && m <= 8;
    if (!native_int4 && m < opt_hmx_min_rows) {
        return false;
    }

    return true;

    GGML_UNUSED(dst);
}

static bool ggml_hexagon_precompute_hmx_mm_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int wtype,
    int ne00_padded,
    int ne01_padded,
    int ne02,
    int ne11,
    int ne12,
    int ne11_padded,
    bool is_matmul_id,
    bool is_batched,
    size_t src2_size,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    const int aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
    const int m = is_matmul_id ? ne12 : ne11;
    const bool native_int4 = opt_hmx_int && wtype == GGML_TYPE_Q4_0 && !is_matmul_id && !is_batched && m > 1 && m <= 8;
    const int requested_kernel = native_int4 ? HTP_MM_KERNEL_HMX_INT4 : HTP_MM_KERNEL_HMX_2D;
    const bool pipeline = !native_int4 && (is_matmul_id ? false : htp_mm_hmx_pipeline(ne11));
    const int n_threads = (int)sess->n_threads;
    const int ne10 = src1->ne[0];

    const bool is_batched_val = is_matmul_id ? false : is_batched;
    const int group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    bool use_grouped = false;
    int act_threads_selected = 0;

    if (is_batched_val && wtype == GGML_TYPE_F16 && group_size > 1) {
        // Try grouped path first
        if (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, ne11, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = true;
        }
    }

    if (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        const int m_id_rows = (dst && is_matmul_id) ? (int) ((size_t) dst->ne[1] * dst->ne[2]) : 0;
        if (!htp_mm_hmx_solve_2d_params(requested_kernel, wtype, ne00_padded, m_id_rows, ne01_padded, ne11_padded, ne11, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            return false;
        }
    }

    kparams->n_hmx = 1;
    kparams->pipeline = pipeline ? 1 : 0;
    kparams->m_chunk = m_chunk;
    kparams->n_chunk = n_chunk;
    kparams->n_threads = n_threads;
    kparams->n_act_threads = act_threads_selected;
    kparams->tile_size = htp_mm_get_weight_tile_size(wtype);
    kparams->aligned_tile_size = aligned_tile_size;
    kparams->src1_row_size = (wtype == GGML_TYPE_Q4_1 || wtype == GGML_TYPE_Q4_K) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
    kparams->vtcm_size = vtcm_size;
    kparams->vtcm_src0_size = 0;
    kparams->div_n_act_threads = init_fastdiv_values(act_threads_selected);
    kparams->div_ne00_padded   = init_fastdiv_values(ne00_padded);
    kparams->vtcm_src1_size = 0;
    kparams->vtcm_src2_size = (int32_t) src2_size;
    kparams->vtcm_dst_size = 0;

    if (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_F16_BATCHED;
    } else {
        kparams->kernel_type = requested_kernel;
    }
    return true;

    GGML_UNUSED(src0);
}

static void ggml_hexagon_precompute_hvx_mm_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int wtype,
    int ne02,
    int ne03,
    int ne10,
    int ne11,
    int ne12,
    int ne13,
    bool is_matmul_id,
    const size_t src2_row_size,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    if (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        return;
    }

    kparams->n_hmx = 0;
    kparams->n_threads = sess->n_threads;

    const bool is_quant = (wtype != GGML_TYPE_F16 && wtype != GGML_TYPE_F32);
    const int src1_nrows = ne11 * ne12 * ne13;

    if (is_quant) {
        // Quantized HVX
        kparams->tile_size = htp_mm_get_weight_tile_size(wtype);
        kparams->aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);

        const bool k_align = (ne10 % 32 == 0);

        if (is_matmul_id) {
            kparams->kernel_type   = (src1_nrows < (int) sess->n_threads) ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->src1_row_size = (wtype == GGML_TYPE_Q4_1 || wtype == GGML_TYPE_Q4_K) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

            struct htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], kparams->src1_row_size, 0, d, true, false
                );
                if (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    break;
                }
            }
            if (L.total_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                return;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_dst_size  = L.dst_bytes;
            goto done_quant;
        } else {
            bool try_tiled = (k_align && opt_mm_select >= 1);
            if (try_tiled) {
                kparams->src1_row_size = (wtype == GGML_TYPE_Q4_1 || wtype == GGML_TYPE_Q4_K)
                                       ? htp_mm_q8_1_tiled_row_size(ne10)
                                       : htp_mm_q8_0_tiled_row_size(ne10);
                if (src1_nrows < (int) sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } else {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                struct htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, false, false
                    );
                    if (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        break;
                    }
                }

                uint32_t m_chunk = 0;
                if (htp_mm_hvx_solve_vtcm_params(
                        kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, best_n_prefetch, vtcm_budget,
                        &L, &m_chunk)) {
                    kparams->n_prefetch     = best_n_prefetch;
                    kparams->m_chunk        = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
                    kparams->vtcm_size      = L.total_bytes;
                    kparams->vtcm_src0_size = L.src0_bytes;
                    kparams->vtcm_src1_size = L.src1_bytes;
                    kparams->vtcm_src2_size = L.src2_bytes;
                    kparams->vtcm_dst_size  = L.dst_bytes;
                    goto done_quant;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            return;
        }

    done_quant:;
    } else if (wtype == GGML_TYPE_F16) {
        // F16 HVX
        struct htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        if (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->src1_row_size = hex_round_up(ne10 * 2, 128);
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_src2_size = L.src2_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
            return;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        return;
    } else {
        // F32 HVX
        struct htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        if (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->src1_row_size = hex_round_up(ne10 * 4, 128);
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_src2_size = L.src2_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
            return;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        return;
    }
}

static void ggml_hexagon_precompute_matmul_params_impl(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    const size_t src2_row_size,
    const size_t src2_size,
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);
    const int ne00_padded = (is_repack || wtype == GGML_TYPE_F16) ? hex_round_up(ne00, 32) : ne00;
    const int ne01_padded = is_repack ? hex_round_up(ne01, 32) : ne01;
    const int ne11_padded = hex_round_up(ne11, 32);

    const bool is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    const bool is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    const size_t vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    bool hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    if (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, ne01_padded, is_matmul_id, is_batched)) {
        if (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_padded, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
            goto finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_size, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_values(ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_values(ne12);
}

static void ggml_hexagon_precompute_matmul_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

static void ggml_hexagon_precompute_fused_matmul_add_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
) {
    const size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

static bool ggml_hexagon_precompute_binary_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_binary_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const size_t elem_size = ggml_type_size(src0->type);
    const size_t src0_row_size = src0->ne[0] * elem_size;
    const size_t src1_row_size = src1->ne[0] * elem_size;
    const size_t dst_row_size  = dst->ne[0]  * elem_size;

    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    const size_t src1_row_size_aligned = hex_round_up(src1_row_size, 128);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size,  128);

    const bool is_add_id = op == HTP_OP_ADD_ID;
    const bool is_scalar = !is_add_id && src1->ne[0] == 1;
    const bool is_transposed = src0->nb[1] < src0_row_size || src1->nb[1] < src1_row_size || dst->nb[1] < dst_row_size;
    const bool is_row_bcast = !is_add_id && !is_scalar && !is_transposed &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    const bool is_same_shape = !is_add_id && !is_scalar && !is_transposed &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    const bool is_complex   = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    const bool is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    const bool is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    if (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        const uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        const uint32_t n_threads = sess->n_threads;
        const uint32_t max_chunk_elems = 32768 / elem_size;
        const uint32_t min_chunk_elems = 256;
        const uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
        const uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        const uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

        kparams->kernel_type           = HTP_BINARY_KERNEL_CHUNKED;
        kparams->n_threads             = n_threads;
        kparams->rows_per_buffer       = 1;
        kparams->src0_row_size_aligned = src0_row_size_aligned;
        kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
        kparams->dst_row_size_aligned  = dst_row_size_aligned;
        kparams->src1_size             = 0;
        kparams->chunk_size            = chunk_size;
        kparams->chunk_bytes           = chunk_bytes;
        kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

        struct htp_binary_vtcm_layout L;
        htp_binary_vtcm_layout_build(&L, kparams, sess->vtcm_size);
        if (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
            return false;
        }

        kparams->vtcm_size = L.total_bytes;
        return true;
    }

    enum htp_binary_kernel_type kernel_type;
    size_t src1_size = 0;

    if (is_add_id) {
        kernel_type = HTP_BINARY_KERNEL_ADD_ID;
        src1_size = hex_round_up(src1->ne[1] * src1_row_size_aligned, 128);
    } else if (is_row_bcast) {
        kernel_type = HTP_BINARY_KERNEL_ROW_BCAST;
        src1_size = src1_row_size_aligned;
    } else if (is_scalar) {
        const bool is_scalar_static = (src1->ne[2] == 1 && src1->ne[3] == 1) &&
            (src1->ne[1] == 1 || src1->nb[1] == elem_size);
        if (is_scalar_static) {
            kernel_type = HTP_BINARY_KERNEL_SCALAR_DMA;
            src1_size = hex_round_up(src1->ne[1] * elem_size, 128);
        } else {
            kernel_type = HTP_BINARY_KERNEL_SCALAR;
        }
    } else if (is_same_shape) {
        kernel_type = HTP_BINARY_KERNEL_SAME_SHAPE;
    } else if (is_complex) {
        kernel_type = HTP_BINARY_KERNEL_COMPLEX;
    } else {
        kernel_type = HTP_BINARY_KERNEL_REPEAT;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_threads = sess->n_threads;
    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;
    kparams->src1_size             = src1_size;

    struct htp_binary_vtcm_layout L;
    htp_binary_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    if (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {
        if (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            const uint32_t total_elems = (uint32_t) ggml_nelements(src0);
            const uint32_t n_threads = sess->n_threads;
            const uint32_t max_chunk_elems = 32768 / elem_size;
            const uint32_t min_chunk_elems = 256;
            const uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
            const uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
            const uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

            kparams->kernel_type           = HTP_BINARY_KERNEL_CHUNKED;
            kparams->n_threads             = n_threads;
            kparams->rows_per_buffer       = 1;
            kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
            kparams->src1_size             = 0;
            kparams->chunk_size            = chunk_size;
            kparams->chunk_bytes           = chunk_bytes;
            kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

            htp_binary_vtcm_layout_build(&L, kparams, sess->vtcm_size);
            if (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
                return false;
            }

            kparams->vtcm_size = L.total_bytes;
            return true;
        }
        return false;
    }

    kparams->rows_per_buffer = L.rows_per_buffer;
    kparams->vtcm_size       = L.total_bytes;

    return true;
}

static void ggml_hexagon_precompute_unary_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_unary_kernel_params * kparams,
    bool                             tile_out
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    const size_t elem_size = ggml_type_size(src0->type);

    const size_t src0_data_row_size = src0->ne[0] * elem_size;
    const size_t dst_data_row_size  = dst->ne[0]  * ggml_type_size(dst->type);

    const size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    const size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = 0;
    bool broadcast_weight = false;

    if (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_data_row_size = src1->ne[0] * ggml_type_size(src1->type);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->broadcast_weight      = broadcast_weight;

    struct htp_unary_vtcm_layout L;
    uint32_t col_tile = 0;
    uint32_t vtcm_row_per_thread = 0;

    // Tile output (GGML_HEXAGON_ACT_TILE): fp16 staging (2 halves x one 32-row block per thread) is taken out of the VTCM
    // budget before the layout is built; the DSP places it at the layout total (src0 + src1 + dst).
    const size_t tile_per_thread = 2 * (size_t) HTP_MM_HMX_TILE_N_ROWS * (size_t) src0->ne[0] * sizeof(uint16_t);
    tile_out = tile_out && op == HTP_OP_RMS_NORM_MUL && src0->ne[2] == 1 && src0->ne[3] == 1 && dst->ne[2] == 1 &&
               dst->ne[3] == 1 && (size_t) n_threads * tile_per_thread < (size_t) sess->vtcm_size;
    const size_t layout_vtcm = tile_out ? (size_t) sess->vtcm_size - (size_t) n_threads * tile_per_thread
                                        : (size_t) sess->vtcm_size;
    // AE-07 (GGML_HEXAGON_ELT_SMALL bit 1): row blocks are capped, except tile output (its blocks must stay 32 rows).
    const uint32_t row_cap = (sess->elt_small & 2) != 0 && !tile_out ? GGML_HEXAGON_ELT_SMALL_ROW_CAP : 0;
    htp_unary_vtcm_layout_build(&L, op, src0->ne[0], dst->ne[0],
                                op == HTP_OP_RMS_NORM_MUL ? src1->ne[0] : 0,
                                broadcast_weight, n_threads, layout_vtcm, elem_size,
                                &col_tile, &vtcm_row_per_thread, row_cap);

    kparams->col_tile = col_tile;
    kparams->vtcm_row_per_thread = vtcm_row_per_thread;
    kparams->vtcm_size = L.total_bytes;

    kparams->vtcm_src0_size_per_thread = L.src0_bytes;
    kparams->vtcm_src1_size_per_thread = L.src1_bytes;
    kparams->vtcm_dst_size_per_thread  = L.dst_bytes;

    kparams->vtcm_src0_size = L.src0_bytes * n_threads;
    kparams->vtcm_src1_size = L.src1_bytes * n_threads;
    kparams->vtcm_dst_size  = L.dst_bytes * n_threads;

    kparams->block = col_tile ? 0 : ((L.src0_bytes / 2) / src0_row_size_aligned);
    if (tile_out && col_tile == 0 && vtcm_row_per_thread >= HTP_MM_HMX_TILE_N_ROWS) {
        // One DSP block = one 32-row tile block; the tile staging sits after the layout total.
        kparams->block                     = HTP_MM_HMX_TILE_N_ROWS;
        kparams->vtcm_tile_size_per_thread = (uint32_t) tile_per_thread;
        kparams->vtcm_size                += (uint32_t) (n_threads * tile_per_thread);
    }

    const uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_values(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_values(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_values(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_values(tiles_per_row);
}

static void ggml_hexagon_precompute_get_rows_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_get_rows_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];

    const uint32_t ne10 = src1->ne[0];
    const uint32_t ne11 = src1->ne[1];
    const uint32_t ne12 = src1->ne[2];
    const uint32_t nr = ne10 * ne11 * ne12;

    const ggml_tensor * src0_base = src0->view_src ? src0->view_src : src0;
    const auto * extra = src0_base->buffer && ggml_backend_buffer_is_hexagon(src0_base->buffer) ?
        (const ggml_hexagon_tensor_extra *) src0_base->extra : nullptr;
    const bool tiled = src0->type == GGML_TYPE_Q4_0 || (extra && (extra->flags & GGML_HEXAGON_TENSOR_REPACK) != 0) ||
                       sess->needs_repack.count(src0_base) || sess->needs_repack.count(src0);

    if (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } else if (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } else {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    const uint32_t chunks_per_row = 1;
    const uint32_t chunk_size = ne00;
    const uint32_t total_tasks = nr;

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, total_tasks);

    struct htp_get_rows_vtcm_layout vtcm_layout = {};
    while (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, src0->type, ne00, kparams->n_threads);
        if (vtcm_layout.total_bytes <= sess->vtcm_size) {
            break;
        }
        --kparams->n_threads;
    }

    if (kparams->n_threads == 0 && total_tasks > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, src0->type, ne00, 1);
    }

    kparams->vtcm_size = (total_tasks == 0) ? 0 : vtcm_layout.total_bytes;
    kparams->tasks_per_thread = kparams->n_threads > 0 ? (total_tasks + kparams->n_threads - 1) / kparams->n_threads : 0;

    kparams->chunks_per_row = chunks_per_row;
    kparams->chunk_size = chunk_size;
    kparams->total_tasks = total_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_values(ne10) : fastdiv_values{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_values(ne10 * ne11) : fastdiv_values{0, 0};
    kparams->div_chunks_per_row = chunks_per_row > 0 ? init_fastdiv_values(chunks_per_row) : fastdiv_values{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_values(ne02) : fastdiv_values{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_values(ne03) : fastdiv_values{0, 0};
}

static void ggml_hexagon_precompute_set_rows_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0, // values
    const struct ggml_tensor * src1, // indices
    const struct ggml_tensor * dst,  // destination
    struct htp_set_rows_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t nr = src0->ne[1];

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, nr);
    kparams->tasks_per_thread = (nr + kparams->n_threads - 1) / kparams->n_threads;
    kparams->total_tasks = nr;

    kparams->div_ne11 = init_fastdiv_values(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_values(src1->ne[2]);
    kparams->div_tasks_per_thread = init_fastdiv_values(kparams->tasks_per_thread);
    kparams->div_ne02 = init_fastdiv_values(src0->ne[2]);

    struct htp_set_rows_vtcm_layout vtcm_layout;
    htp_set_rows_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_threads);
    kparams->vtcm_size = vtcm_layout.total_bytes;
}

static void ggml_hexagon_precompute_softmax_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);

    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    &op->op_params[0], sizeof(float));
    memcpy(&max_bias, &op->op_params[1], sizeof(float));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    const uint32_t n_head = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    kparams->n_head      = n_head;
    kparams->n_head_log2 = n_head_log2;

    if (max_bias > 0.0f && n_head_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_head_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    } else {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne10 = src1 ? src1->ne[0] : 1;

    struct htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, ne00, ne10, kparams->use_src1 != 0, kparams->use_f16 != 0, n_threads);

    kparams->n_threads                 = n_threads;
    kparams->src0_nrows                = src0_nrows;
    kparams->src0_nrows_per_thread     = (src0_nrows + n_threads - 1) / n_threads;
    kparams->vtcm_size                 = (uint32_t) layout.total_bytes;
    kparams->vtcm_src0_size_per_thread = (uint32_t) layout.src0_bytes_per_thread;
    kparams->vtcm_src1_size_per_thread = (uint32_t) layout.src1_bytes_per_thread;
    kparams->vtcm_dst_size_per_thread  = (uint32_t) layout.dst_bytes_per_thread;
    kparams->src0_row_size_aligned     = (uint32_t) layout.src0_spad_half_size;
    kparams->src1_row_size_aligned     = (uint32_t) layout.src1_spad_half_size;
    kparams->dst_row_size_aligned      = (uint32_t) layout.dst_spad_half_size;
    kparams->src0_spad_half_size       = (uint32_t) layout.src0_spad_half_size;
    kparams->src1_spad_half_size       = (uint32_t) layout.src1_spad_half_size;
    kparams->dst_spad_half_size        = (uint32_t) layout.dst_spad_half_size;
    if (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } else if (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F16;
    } else {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    if (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_values(src0->ne[1]);
    if (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_values(src0->ne[2]);
    const uint32_t ne12 = src1 ? src1->ne[2] : 1;
    const uint32_t ne13 = src1 ? src1->ne[3] : 1;
    if (ne12 > 0) kparams->div_ne12 = init_fastdiv_values(ne12);
    if (ne13 > 0) kparams->div_ne13 = init_fastdiv_values(ne13);
}

static void ggml_hexagon_precompute_rope_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    const uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    struct htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);

    kparams->n_threads              = n_threads;
    kparams->src0_nrows             = src0_nrows;
    kparams->src0_nrows_per_thread  = (src0_nrows + n_threads - 1) / n_threads;
    kparams->vtcm_size              = (uint32_t) layout.total_bytes;
    kparams->spad_per_thread        = (uint32_t) layout.bytes_per_thread;
    kparams->theta_cache_offset     = (uint32_t) layout.theta_cache_size_aligned;
    kparams->src0_row_size_aligned  = (uint32_t) layout.src0_row_size_aligned;
    kparams->freq_factors_offset    = (uint32_t) (layout.bytes_per_thread * n_threads);
    kparams->freq_factors_size      = (uint32_t) layout.freq_factors_size_aligned;

    if (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_values(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_values(dst->ne[1]);
    }
}

static void ggml_hexagon_precompute_ssm_conv_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_ssm_conv_kernel_params * kparams,
    bool fused_concat,
    uint32_t n_state_rows
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t d_conv  = (uint32_t) src1->ne[0];
    const uint32_t d_inner = (uint32_t) src0->ne[1];
    const uint32_t n_t     = (uint32_t) dst->ne[1];
    const uint32_t n_s     = (uint32_t) dst->ne[2];
    const uint32_t ncs     = (uint32_t) src0->ne[0];
    const bool use_fused_prefetch = fused_concat && n_t > 1 && n_state_rows > 0 && n_state_rows < ncs;

    const uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;
    kparams->fused_concat = use_fused_prefetch ? 1 : 0;
    kparams->n_state_rows = use_fused_prefetch ? n_state_rows : 0;

    const uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
    const uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);
    kparams->d_inner_per_thread = d_inner_per_thread;

    kparams->src0_row_size_aligned = hex_round_up(ncs * sizeof(float), 128);
    kparams->src1_row_size_aligned = hex_round_up(d_conv * sizeof(float), 128);
    kparams->dst_row_size_aligned  = hex_round_up(d_inner * sizeof(float), 128);

    if (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;

        const uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * sizeof(float), 128) + 128;
        const uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 128);
        const uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

        const uint32_t src0_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * sizeof(float), 128) + 128;
        const uint32_t src0_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 128);
        const uint32_t vtcm_src0_per_thread = src0_raw_bytes + src0_T_bytes;

        const uint32_t vtcm_dst_per_thread = hex_round_up(d_inner_per_thread * sizeof(float), 128);

        kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
        kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
        kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;

        kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
        kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
        kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
        kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
    } else {
        const uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * sizeof(float), 128) + 128;
        const uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 128);
        const uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

        const size_t vtcm_budget = (sess->vtcm_size > 0 ? sess->vtcm_size / n_threads : (1024 * 1024));
        const size_t avail_for_src0 = vtcm_budget > vtcm_src1_per_thread ? vtcm_budget - vtcm_src1_per_thread : (128 * 1024);

        uint32_t d_inner_tile;
        const uint32_t pipe_chunk = (n_t < 128) ? n_t : 128;
        const bool use_pipe = opt_ssm_conv_pipe && use_fused_prefetch && d_conv == 4 && n_state_rows == d_conv - 1;
        if (use_pipe) {
            d_inner_tile = d_inner_per_thread;
            while (d_inner_tile > 32) {
                const size_t raw_bytes   = hex_round_up(d_inner_tile * n_state_rows * sizeof(float), 128) + 128;
                const size_t trans_bytes = hex_round_up((pipe_chunk + n_state_rows) * d_inner_tile * sizeof(float), 128);
                const size_t dst_bytes   = hex_round_up(pipe_chunk * d_inner_tile * sizeof(float), 128);
                if (2 * (raw_bytes + trans_bytes) + 2 * dst_bytes <= avail_for_src0) {
                    break;
                }
                d_inner_tile -= 32;
            }
        } else if (use_fused_prefetch) {
            d_inner_tile = d_inner_per_thread;
            while (d_inner_tile > 32) {
                const size_t raw_bytes = hex_round_up(d_inner_tile * n_state_rows * sizeof(float), 128) + 128;
                const size_t trans_bytes = hex_round_up(ncs * d_inner_tile * sizeof(float), 128);
                const size_t dst_bytes = hex_round_up(d_inner_tile * n_t * sizeof(float), 128);
                if (2 * (raw_bytes + trans_bytes) + dst_bytes <= avail_for_src0) {
                    break;
                }
                d_inner_tile -= 32;
            }
        } else {
            d_inner_tile = (uint32_t)((avail_for_src0 / 2) / (ncs * sizeof(float) + n_t * sizeof(float) + 1));
            d_inner_tile = (d_inner_tile / 32) * 32;
            if (d_inner_tile == 0) {
                d_inner_tile = 32;
            }
            if (d_inner_tile > d_inner_per_thread) {
                d_inner_tile = d_inner_per_thread;
            }
        }
        kparams->d_inner_tile = d_inner_tile;
        kparams->pipe       = use_pipe ? 1 : 0;
        kparams->pipe_chunk = use_pipe ? pipe_chunk : 0;

        const uint32_t src0_raw_rows = use_fused_prefetch ? n_state_rows : ncs;
        const uint32_t src0_slots = use_fused_prefetch ? 2 : 1;
        const uint32_t src0_tile_raw = hex_round_up(d_inner_tile * src0_raw_rows * sizeof(float), 128) + 128;
        const uint32_t src0_tile_T   = hex_round_up((use_pipe ? pipe_chunk + n_state_rows : ncs) * d_inner_tile * sizeof(float), 128);
        const uint32_t vtcm_src0_per_thread = src0_slots * (src0_tile_raw + src0_tile_T);

        const uint32_t vtcm_dst_per_thread = (use_pipe ? 2 : 1) * hex_round_up((use_pipe ? pipe_chunk : n_t) * d_inner_tile * sizeof(float), 128);

        kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
        kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
        kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;

        kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
        kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
        kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
        kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
    }

    kparams->div_n_threads = init_fastdiv_values(n_threads);
}

// VTCM per thread: src0 = conv state + n_buf x (input tile, state tile), src1 = weights, dst = n_buf output tiles.
// With qk_head_dim != 0 tiles and thread ranges are whole heads. Returns false if no tiling fits in VTCM.
static bool ggml_hexagon_precompute_ssm_conv_chain_params(
    const struct ggml_hexagon_session * sess,
    uint32_t d_conv,
    uint32_t d_inner,
    uint32_t n_t,
    uint32_t n_s,
    uint32_t st_in_seq_stride,
    uint32_t qkv_seq_stride,
    uint32_t dst_seq_stride,
    uint32_t st_out_seq_stride,
    uint32_t qk_head_dim,
    uint32_t n_qk,
    uint32_t n_tb,
    struct htp_ssm_conv_chain_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t n_state = d_conv - 1;

    const uint32_t align = qk_head_dim ? qk_head_dim : 32;
    if (qk_head_dim && (qk_head_dim % 32 != 0 || d_inner % qk_head_dim != 0)) {
        return false;
    }
    if (n_tb >= n_t) {
        n_tb = 0;
    }
    const uint32_t rows = n_tb ? n_tb : n_t;

    const uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);
    if (n_threads == 0) {
        return false;
    }

    uint32_t d_inner_per_thread = hex_round_up((d_inner + n_threads - 1) / n_threads, align);

    // With the epilogue a q/k head costs about 14/9 of a v head at prefill (decode keeps the even split), so split the
    // heads into contiguous ranges of the smallest maximum cost instead of evenly. Starts go in 32-channel units.
    uint8_t ch_start[HTP_SSM_CONV_CHAIN_MAX_BALANCED_THREADS - 1] = { 0 };
    bool    balanced = false;
    if (qk_head_dim && n_qk > 0 && n_qk < d_inner && n_qk % qk_head_dim == 0 && n_threads > 1 && n_t > 1 &&
        n_threads <= HTP_SSM_CONV_CHAIN_MAX_BALANCED_THREADS && d_inner / 32 <= 255) {
        const uint32_t n_units = d_inner / qk_head_dim;
        const uint32_t n_qk_u  = n_qk / qk_head_dim;
        auto cost = [&](uint32_t u) { return u < n_qk_u ? 14u : 9u; };
        // greedy fill under a cost cap, returns the number of ranges and their starts in head units
        auto fill = [&](uint32_t cap, uint32_t * starts) {
            uint32_t parts = 1, acc = 0;
            for (uint32_t u = 0; u < n_units; u++) {
                if (acc + cost(u) > cap) {
                    if (starts && parts < n_threads) {
                        starts[parts - 1] = u;
                    }
                    parts++;
                    acc = 0;
                }
                acc += cost(u);
            }
            return parts;
        };
        uint32_t lo = 14, hi = 14 * n_units;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (fill(mid, nullptr) <= n_threads) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        uint32_t starts[HTP_SSM_CONV_CHAIN_MAX_BALANCED_THREADS] = { 0 };
        const uint32_t parts = fill(lo, starts);
        for (uint32_t t = parts - 1; t + 1 < n_threads; t++) {
            starts[t] = n_units;  // trailing threads get an empty range
        }
        uint32_t widest = 0, prev = 0;
        for (uint32_t t = 0; t < n_threads; t++) {
            const uint32_t end = t + 1 < n_threads ? starts[t] : n_units;
            widest = (std::max)(widest, end - prev);
            prev   = end;
            if (t + 1 < n_threads) {
                ch_start[t] = (uint8_t) (starts[t] * qk_head_dim / 32);
            }
        }
        balanced           = true;
        d_inner_per_thread = widest * qk_head_dim;

        // GGML_HEXAGON_CONV_SPLIT32=1: v channels are split at 32-channel granularity. Optimal contiguous split over 32-channel units (q/k units are whole heads, cost 107/4 per unit; v units 62/4).
        // Measured per-head busy time at 512 tokens: q/k head ~107 us, v head ~62 us (+ ~35 us per thread of fixed cost), so whole-head ranges leave threads 4..7 idle ~20%.
        if (opt_conv_split32) {
            const uint32_t u_per_head = qk_head_dim / 32;
            const uint32_t n_u        = d_inner / 32;
            const uint32_t n_qk_u     = n_qk / 32;
            const uint32_t cq = opt_conv_split32_cq, cv = opt_conv_split32_cv;   // cost of one 32-channel unit x 4
            std::vector<uint32_t> pre(n_u + 1, 0);
            for (uint32_t u = 0; u < n_u; u++) {
                pre[u + 1] = pre[u] + (u < n_qk_u ? cq : cv);
            }
            // a range may only start/end inside the q/k region on a head boundary
            auto ok_cut = [&](uint32_t u) { return u >= n_qk_u || (u % u_per_head) == 0; };
            // greedy fill under a cost cap; returns the number of ranges, starts[] = first unit of range p + 1
            auto fill2 = [&](uint32_t cap, uint32_t * starts) {
                uint32_t parts = 1, begin = 0;
                while (pre[n_u] - pre[begin] > cap) {
                    uint32_t end = begin;
                    for (uint32_t e = begin + 1; e < n_u; e++) {
                        if (pre[e] - pre[begin] <= cap && ok_cut(e)) {
                            end = e;
                        }
                    }
                    if (end == begin) {
                        return 1000u;   // a single head exceeds the cap
                    }
                    if (starts && parts <= n_threads) {
                        starts[parts - 1] = end;
                    }
                    begin = end;
                    parts++;
                }
                return parts;
            };
            uint32_t lo = cq * u_per_head, hi = pre[n_u];
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if (fill2(mid, nullptr) <= n_threads) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            uint32_t starts2[HTP_SSM_CONV_CHAIN_MAX_BALANCED_THREADS] = { 0 };
            const uint32_t parts2 = fill2(lo, starts2);
            if (parts2 <= n_threads) {
                for (uint32_t t = parts2 - 1; t + 1 < n_threads; t++) {
                    starts2[t] = n_u;
                }
                uint32_t widest_u = 0, prev_u = 0;
                for (uint32_t t = 0; t < n_threads; t++) {
                    const uint32_t end = t + 1 < n_threads ? starts2[t] : n_u;
                    widest_u = (std::max)(widest_u, end - prev_u);
                    prev_u   = end;
                    if (t + 1 < n_threads) {
                        ch_start[t] = (uint8_t) starts2[t];
                    }
                }
                d_inner_per_thread = widest_u * 32;
            }
        }
    }

    if (opt_conv_dyn && balanced && n_threads == (uint32_t) sess->n_threads && n_t > 1) {
        // dynamic tile queue: one head per tile, conv state / weights of all channels in a shared VTCM region after the per-thread ones
        const uint32_t hd = qk_head_dim;
        const uint32_t xt_bytes        = hex_round_up((n_state + rows) * hd * sizeof(float), 128);
        const uint32_t state_out_bytes = hex_round_up(hd * n_state * sizeof(float), 128);
        const uint32_t buf_stride      = xt_bytes + state_out_bytes;
        const uint32_t dst_tile_bytes  = hex_round_up(rows * hd * sizeof(float), 128);
        const size_t   shared_bytes    = hex_round_up((size_t) n_s * n_state * d_inner * sizeof(float), 128) +
                                         hex_round_up((size_t) d_inner * d_conv * sizeof(float), 128) +
                                         hex_round_up((size_t) d_conv * d_inner * sizeof(float), 128) + 128;
        const size_t   total = (size_t) n_threads * (2 * buf_stride + 2 * dst_tile_bytes) + shared_bytes;
        if (total <= sess->vtcm_size) {
            kparams->n_threads          = n_threads;
            kparams->d_conv             = d_conv;
            kparams->n_state            = n_state;
            kparams->d_inner            = d_inner;
            kparams->n_t                = n_t;
            kparams->n_s                = n_s;
            kparams->d_inner_per_thread = hd;
            kparams->d_inner_tile       = hd;
            kparams->n_tb               = n_tb;
            kparams->qk_head_dim        = qk_head_dim;
            kparams->st_in_seq_stride   = st_in_seq_stride;
            kparams->qkv_seq_stride     = qkv_seq_stride;
            kparams->dst_seq_stride     = dst_seq_stride;
            kparams->st_out_seq_stride  = st_out_seq_stride;
            kparams->n_buf                  = 2;
            kparams->vtcm_xt_bytes          = xt_bytes;
            kparams->vtcm_state_stage_bytes = 0;
            kparams->vtcm_buf_stride        = buf_stride;
            kparams->vtcm_dst_tile_bytes    = dst_tile_bytes;
            kparams->vtcm_src0_size_per_thread = 2 * buf_stride;
            kparams->vtcm_src1_size_per_thread = 0;
            kparams->vtcm_dst_size_per_thread  = 2 * dst_tile_bytes;
            kparams->vtcm_size = (uint32_t) total;
            kparams->balanced  = 2;
            return true;
        }
    }

    const uint32_t w_raw_bytes          = hex_round_up(d_inner_per_thread * d_conv * sizeof(float), 128) + 128;
    const uint32_t w_T_bytes            = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 128);
    const uint32_t vtcm_src1_per_thread = w_raw_bytes + w_T_bytes;
    const uint32_t state_stage_bytes    = hex_round_up(n_s * n_state * d_inner_per_thread * sizeof(float), 128) + 128;

    const size_t vtcm_budget = sess->vtcm_size / n_threads;
    if (vtcm_budget <= (size_t) vtcm_src1_per_thread + state_stage_bytes) {
        return false;
    }
    const size_t avail = vtcm_budget - vtcm_src1_per_thread - state_stage_bytes;

    // decode with one sequence is a single tile; everything else is split into at least 4 tiles and double buffered
    constexpr uint32_t N_TILES_PIPELINE = 4;
    const uint32_t n_buf       = (n_t == 1 && n_s == 1) ? 1 : 2;
    const uint32_t per_channel = n_buf * (n_state + rows + rows + n_state) * sizeof(float);

    uint32_t d_inner_tile = 0;
    if (avail > 1024 * n_buf) {
        d_inner_tile = (uint32_t) ((avail - 1024 * n_buf) / per_channel);
    }
    d_inner_tile = (std::min)((d_inner_tile / align) * align, d_inner_per_thread);
    if (n_buf > 1 && d_inner_tile > 0) {
        const uint32_t n_tiles = (std::max)((d_inner_per_thread + d_inner_tile - 1) / d_inner_tile, N_TILES_PIPELINE);
        const uint32_t even    = hex_round_up((d_inner_per_thread + n_tiles - 1) / n_tiles, align);
        d_inner_tile = (std::min)(d_inner_tile, even);
    }

    for (; d_inner_tile >= align; d_inner_tile -= align) {
        // a single buffer only works when the thread's channels fit in one tile
        if (n_buf == 1 && d_inner_tile < d_inner_per_thread) {
            break;
        }

        const uint32_t xt_bytes        = hex_round_up((n_state + rows) * d_inner_tile * sizeof(float), 128);
        const uint32_t state_out_bytes = hex_round_up(d_inner_tile * n_state * sizeof(float), 128);
        const uint32_t buf_stride      = xt_bytes + state_out_bytes;
        const uint32_t dst_tile_bytes  = hex_round_up(rows * d_inner_tile * sizeof(float), 128);

        const uint32_t vtcm_src0_per_thread = state_stage_bytes + n_buf * buf_stride;
        const uint32_t vtcm_dst_per_thread  = n_buf * dst_tile_bytes;

        if ((size_t) vtcm_src0_per_thread + vtcm_src1_per_thread + vtcm_dst_per_thread > vtcm_budget) {
            continue;
        }

        kparams->n_threads          = n_threads;
        kparams->d_conv             = d_conv;
        kparams->n_state            = n_state;
        kparams->d_inner            = d_inner;
        kparams->n_t                = n_t;
        kparams->n_s                = n_s;
        kparams->d_inner_per_thread = d_inner_per_thread;
        kparams->d_inner_tile       = d_inner_tile;
        kparams->n_tb               = n_tb;
        kparams->qk_head_dim        = qk_head_dim;

        kparams->st_in_seq_stride  = st_in_seq_stride;
        kparams->qkv_seq_stride    = qkv_seq_stride;
        kparams->dst_seq_stride    = dst_seq_stride;
        kparams->st_out_seq_stride = st_out_seq_stride;

        kparams->n_buf                  = n_buf;
        kparams->vtcm_xt_bytes          = xt_bytes;
        kparams->vtcm_state_stage_bytes = state_stage_bytes;
        kparams->vtcm_buf_stride        = buf_stride;
        kparams->vtcm_dst_tile_bytes    = dst_tile_bytes;

        kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
        kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
        kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;

        kparams->vtcm_size = (vtcm_src0_per_thread + vtcm_src1_per_thread + vtcm_dst_per_thread) * n_threads;

        kparams->balanced = balanced;
        memcpy(kparams->ch_start, ch_start, sizeof(kparams->ch_start));

        return kparams->vtcm_size <= sess->vtcm_size;
    }

    return false;
}

static void ggml_hexagon_precompute_gated_delta_net_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->snap_mode = (uint8_t) opt_gdn_snap;
    kparams->opt      = opt_gdn_opt;
    kparams->fix      = (uint8_t) ((opt_gdn_fix & 0x0Fu) | ((uint32_t) (opt_gdn_colsplit & 3) << 4));
    kparams->skew_units = (uint8_t) opt_gdn_skew_units;

    const struct ggml_tensor * q     = op->src[0];
    const struct ggml_tensor * k     = op->src[1];
    const struct ggml_tensor * v     = op->src[2];
    const struct ggml_tensor * g     = op->src[3];
    const struct ggml_tensor * state = op->src[5];

    const uint32_t S_v      = (uint32_t) v->ne[0];
    const uint32_t H        = (uint32_t) v->ne[1];
    const uint32_t n_tokens = (uint32_t) v->ne[2];
    const uint32_t n_seqs   = (uint32_t) v->ne[3];
    const uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    const uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    const uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    const uint32_t total_rows = H * n_seqs;
    const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    const bool can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             ggml_hexagon_gdn_hmx_snap_ok(op, K) && opt_gdn_hmx != 0 && (state->type != GGML_TYPE_F16);

    struct htp_gdn_hmx_vtcm_layout hmx_layout;
    struct htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    if (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_SIZE, H, total_rows, sess->vtcm_size, n_threads, true, opt_gdn_opt,
                                                htp_gdn_skew_bytes((uint8_t) opt_gdn_fix, (uint8_t) opt_gdn_skew_units, n_tokens), &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_size      = HTP_GDN_CHUNK_SIZE;
        kparams->n_chunks        = (n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_size       = (uint32_t) hmx_layout.total_bytes;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_bytes;
        kparams->vtcm_per_thread = (uint32_t) (hmx_layout.total_bytes / (n_threads > 0 ? n_threads : 1));
        if (K > 1) {
            kparams->opt |= HTP_GDN_OPT_PROMPT_HMX; // the device runs K > 1 chunked HMX only with this bit (final-state range)
        }
    } else {
        htp_gdn_vtcm_layout_select(&hvx_layout, S_v, n_threads, kparams->opt);
        kparams->kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
        kparams->pipeline        = 0;
        kparams->n_heads_batch   = 1;
        kparams->state_aligned   = (uint32_t) hvx_layout.state_aligned;
        kparams->vtcm_per_thread = (uint32_t) hvx_layout.bytes_per_thread;
        kparams->vtcm_size       = (uint32_t) hvx_layout.total_bytes;
    }

    kparams->n_threads           = n_threads;
    kparams->S_v                 = S_v;
    kparams->H                   = H;
    kparams->n_tokens            = n_tokens;
    kparams->n_seqs              = n_seqs;
    kparams->K                   = K;
    kparams->total_rows          = total_rows;
    kparams->rows_per_thread     = (total_rows + kparams->n_threads - 1) / kparams->n_threads;
    kparams->kda                 = (g->ne[0] == S_v) ? 1 : 0;
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / sizeof(float));
    kparams->state_size_per_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((float) S_v);

    if (H > 0)                  kparams->div_H         = init_fastdiv_values(H);
    if (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_values((uint32_t) q->ne[1]);
    if (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_values((uint32_t) k->ne[1]);
    if (rq3 > 0)                kparams->div_rq3       = init_fastdiv_values(rq3);
    if (rk3 > 0)                kparams->div_rk3       = init_fastdiv_values(rk3);
    if (kparams->n_threads > 0) kparams->div_n_threads = init_fastdiv_values(kparams->n_threads);
}

static void ggml_hexagon_precompute_sort_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    bool is_top_k,
    struct htp_sort_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t ne00       = src0->ne[0];
    const uint32_t k          = dst->ne[0];

    int32_t order = GGML_SORT_ORDER_DESC;
    if (!is_top_k) {
        order = ((const int32_t *) op->op_params)[0];
    }

    const uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    const size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    struct htp_sort_vtcm_layout layout;
    bool ok = htp_sort_solve_layout(&layout, ne00, total_rows, k, n_threads_max, vtcm_budget, is_top_k);
    GGML_ASSERT(ok);

    kparams->n_threads         = (int32_t) layout.n_threads;
    kparams->total_rows        = (int32_t) total_rows;
    kparams->row_start         = 0;
    kparams->row_end           = (int32_t) total_rows;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->order             = order;
    kparams->is_top_k          = is_top_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_size         = (int32_t) layout.total_bytes;
    kparams->phase1_slot_size  = (int32_t) layout.phase1_slot_size;
    kparams->merge_values_off  = (int32_t) layout.merge_values_off;
    kparams->merge_indices_off = (int32_t) layout.merge_indices_off;
    kparams->merge_elems       = (int32_t) layout.merge_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

static void ggml_hexagon_precompute_fused_mmnx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0, // W0
    const struct ggml_tensor * src1, // x
    int32_t n_weights,
    size_t src2_size,   // bias bytes of the last weight (fused MUL_MAT+ADD, GGML_HEXAGON_NX_ALPHABETA); 0 = none
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->n_threads = sess->n_threads;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);
    const int ne00_padded = is_repack ? hex_round_up(ne00, 32) : ne00;
    const int ne01_padded = is_repack ? hex_round_up(ne01, 32) : ne01;
    const int ne11_padded = hex_round_up(ne11, 32);

    const size_t vtcm_budget = sess->vtcm_size;
    const bool is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    bool hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    if (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, nullptr, ne01_padded, false, is_batched)) {
        if (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, nullptr, wtype, ne00_padded, ne01_padded, ne02, ne11, ne12, ne11_padded, false, is_batched, src2_size, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            goto finalize;
        }
    }

    if (!is_repack) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        return;
    }

    {
        const int src1_nrows = ne11 * ne12 * ne13;
        const size_t src1_row_size = (wtype == GGML_TYPE_Q4_1 || wtype == GGML_TYPE_Q4_K) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
        const size_t src0_row_size = src0->nb[1];

        uint32_t best_n_prefetch = 16;

        if (is_repack) {
            const uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            best_n_prefetch = 2;
            for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                struct htp_mm_hvx_vtcm_layout L;
                htp_mm_hvx_vtcm_layout_build(
                    &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0_row_size, src1_row_size, 0, d, false, true
                );
                if (L.total_bytes <= sess->vtcm_size) {
                    best_n_prefetch = d;
                    break;
                }
            }
        }

        struct htp_mm_hvx_vtcm_layout L;
        bool try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, src1_nrows, sess->n_threads,
            0, src0_row_size, src1_row_size, 0, best_n_prefetch, false, true
        );

        if (try_tiled && L.total_bytes <= sess->vtcm_size) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_dst_size  = L.dst_bytes;
            kparams->vtcm_size      = L.total_bytes;
            kparams->n_prefetch     = best_n_prefetch;
            kparams->n_weights      = n_weights;
        } else {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            return;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_values(ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_values(ne12);
}

// Kernel params of the fused gate/up NX + SWIGLU output (GGML_HEXAGON_SWIGLU_OUT): HMX 2D, pipelined, chunk width
// HTP_MM_SWIGLU_NC, four fp16 output chunks (htp/matmul-ops.h). Q4_0 gate/up weights with prefill-sized activations only.
static bool ggml_hexagon_precompute_swiglu_nx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * w0,   // gate weight
    const struct ggml_tensor * x,    // activation
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->n_threads = sess->n_threads;

    if (sess->n_hmx == 0 || opt_mm_select < 2 || w0->type != GGML_TYPE_Q4_0) {
        return false;
    }
    if (w0->ne[2] != 1 || w0->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1 || w0->ne[1] % opt_swiglu_nc != 0) {
        return false;
    }
    // Row counts that are not whole 32-row tiles fail on device (213 and 336 rows: PPL and cabin decode diverge, see
    // impl/SwigluFix.md). Refuse the fusion for them: the GLU then runs on the fp32 path and no tile view is installed.
    if (x->ne[1] % 32 != 0) {
        return false;
    }
    const int ne00_padded = hex_round_up((int) w0->ne[0], 32);
    const int ne01_padded = hex_round_up((int) w0->ne[1], 32);
    const int ne11_padded = hex_round_up((int) x->ne[1], 32);
    if (!ggml_hexagon_matmul_is_hmx_eligible(w0, x, nullptr, ne01_padded, false, false)) {
        return false;
    }

    const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size((int) w0->type);
    size_t mc = 0;
    size_t vtcm_total = 0;
    if (!htp_mm_swiglu_solve_mc((uint32_t) ne00_padded, (size_t) ne11_padded, opt_swiglu_nc, aligned_tile_size,
                                (uint32_t) sess->n_threads, sess->vtcm_size, &mc, &vtcm_total)) {
        return false;
    }

    kparams->n_hmx             = 1;
    kparams->pipeline          = 1;
    kparams->kernel_type       = HTP_MM_KERNEL_HMX_2D;
    kparams->m_chunk           = (int32_t) mc;
    kparams->n_chunk           = opt_swiglu_nc;
    kparams->n_act_threads     = (int32_t) sess->n_threads;
    kparams->tile_size         = (int32_t) htp_mm_get_weight_tile_size((int) w0->type);
    kparams->aligned_tile_size = (int32_t) aligned_tile_size;
    kparams->src1_row_size     = (int32_t) htp_mm_q8_0_tiled_row_size((int) x->ne[0]);
    kparams->vtcm_size         = (int32_t) vtcm_total;
    kparams->n_weights         = 2;
    kparams->div_n_act_threads = init_fastdiv_values((uint32_t) sess->n_threads);
    kparams->div_ne00_padded   = init_fastdiv_values((uint32_t) ne00_padded);
    return true;
}

static void ggml_hexagon_precompute_fused_mmidnx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0, // W0
    const struct ggml_tensor * src1, // x
    const struct ggml_tensor * dst,  // dst0
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

static bool ggml_hexagon_tensor_is_host(const struct ggml_hexagon_session * sess, const struct ggml_tensor * t) {
    return t && t->buffer && ggml_backend_buft_is_host(t->buffer->buft);
    GGML_UNUSED(sess);
}

static bool ggml_hexagon_tensor_is_non_host(const struct ggml_hexagon_session * sess, const struct ggml_tensor * t) {
    return t && t->buffer && !ggml_backend_buft_is_host(t->buffer->buft);
    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_mul_mat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            if (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                return false;
            }

            if (src0->ne[0] % ((src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q4_K) ? QK_K : 32)) {
                return false;
            }

            if (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                return false;
            }
            if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                return false;
            }

            if (!src0->buffer) {
                sess->needs_repack.insert(src0);
            }
            break;

        case GGML_TYPE_F16:
            if (src0->nb[1] < src0->nb[0]) {
                return false;
            }
            if (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                return false;
            }
            if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                return false;
            }
            break;

        case GGML_TYPE_F32:
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }
            if (src0->nb[1] < src0->nb[0]) {
                return false;
            }
            if (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                return false;
            }
            if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                return false;
            }
            break;

        default:
            return false;
    }

    struct htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    if (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_mul_mat_id(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        return false;
    }

    if (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            if (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                return false;
            }

            if (src0->ne[0] % ((src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q4_K) ? QK_K : 32)) {
                return false;
            }

            if (!src0->buffer) {
                sess->needs_repack.insert(src0);
            }
            break;

        default:
            return false;
    }

    struct htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    if (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_binary(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type != GGML_TYPE_F32) {
            return false;
        }
        if (dst->type != GGML_TYPE_F32) {
            return false;
        }
    }
    else if (src0->type == GGML_TYPE_F16) {
        if (src1->type != GGML_TYPE_F16) {
            return false;
        }
        if (dst->type != GGML_TYPE_F16) {
            return false;
        }
    }
    else {
        return false;
    }

    if (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        return false;
    }
    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }
    if (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        return false;
    }

    struct htp_binary_kernel_params kparams;
    return ggml_hexagon_precompute_binary_params(sess, op_remap_to_htp(op), src0, src1, dst, &kparams);
}

static bool ggml_hexagon_supported_add_id(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (!src2) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        return false;
    }
    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }
    if (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        return false;
    }
    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || src2->nb[0] != sizeof(int32_t)) {
        return false;
    }

    // REVISIT: add support for non-contiguous tensors
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    struct htp_binary_kernel_params kparams;
    return ggml_hexagon_precompute_binary_params(sess, HTP_OP_ADD_ID, src0, src1, dst, &kparams);
}

static bool ggml_hexagon_supported_unary(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        return false;
    }
    if (dst->type != src0->type) {
        return false;
    }
    if (!ggml_is_contiguous_rows(src0)) {
        return false;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    if (src0->type == GGML_TYPE_F16) {
        switch (op->op) {
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_L2_NORM:
            case GGML_OP_SCALE:
            case GGML_OP_CLAMP:
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
                break;
            case GGML_OP_UNARY:
                if (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_STEP) {
                    return false;
                }
                break;
            default:
                return false;
        }
    }

    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }

    // dst must be contiguous; src0 may be non-contiguous
    if (!ggml_is_contiguous(dst)) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_sum(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_sum_rows(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    // TODO: add support for non-contigiuos tensors
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_argmax(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    if (sess->mdev.count > 1) return false;
    return false;
    GGML_UNUSED(op);
    return false;
}

static bool ggml_hexagon_supported_activations(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous_1(src0)) {
        return false;
    }
    if (!ggml_is_contiguous(dst)) {
        return false;
    }

    if (src1) {
        if (src1->type != GGML_TYPE_F32) {
            return false;
        }
        if (!ggml_are_same_shape(src0, src1)) {
            return false;
        }
        if (!ggml_is_contiguous_1(src1)) {
            return false;
        }
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_softmax(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (src2) {
        return false;  // FIXME: add support for sinks
    }

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[2] > 512) {
        return false;
    }

    if (src1) {
        if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            return false;
        }
        if (src0->ne[0] != src1->ne[0]) {
            return false;
        }
        if (src1->ne[1] < src0->ne[1]) {
            return false;
        }
        if (src0->ne[2] % src1->ne[2] != 0) {
            return false;
        }
        if (src0->ne[3] % src1->ne[3] != 0) {
            return false;
        }
    }

    if (src1) {
        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            return false;
        }
    } else {
        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            return false;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    const int64_t ne0 = src0->ne[0];
    if (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        return false;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    if (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        return false;
    }

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    struct htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != nullptr, src1 && src1->type == GGML_TYPE_F16, n_threads);
    if (layout.total_bytes > sess->vtcm_size) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_set_rows(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * src1 = op->src[1]; // indices
    const struct ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    if (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }

    if (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        return false;
    }

    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_get_rows(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * src1 = op->src[1]; // indices
    const struct ggml_tensor * dst  = op;

    if (src0->type == GGML_TYPE_Q4_0 && src0->view_src) {
        return false;
    }

    const ggml_tensor * src0_base = src0->view_src ? src0->view_src : src0;
    bool is_repacked = false;
    if (src0_base->buffer && ggml_backend_buffer_is_hexagon(src0_base->buffer) && src0_base->extra) {
        const auto * extra = (const ggml_hexagon_tensor_extra *) src0_base->extra;
        is_repacked = (extra->flags & GGML_HEXAGON_TENSOR_REPACK) != 0;
        if (is_repacked && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0) {
            return false;
        }
    }
    is_repacked = is_repacked || sess->needs_repack.count(src0_base) || sess->needs_repack.count(src0);

    // View offsets use the raw quantized layout and cannot address a tiled allocation.
    if (src0->view_src && is_repacked) {
        return false;
    }

    if (src0->type == GGML_TYPE_Q4_0 && src0->buffer && !is_repacked) {
        return false;
    }

    if (src0->type != dst->type && src0->ne[0] < 32) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
        src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_I32) {
        return false;
    }

    if (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        return false;
    }

    if (src0->type == dst->type) {
        if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            return false;
        }
    } else if (src0->type == GGML_TYPE_I32) {
        return false;
    } else if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    if (ggml_is_empty(op)) {
        return true;
    }

    struct htp_get_rows_kernel_params kparams;
    ggml_hexagon_precompute_get_rows_params(sess, src0, src1, dst, &kparams);
    if (kparams.n_threads == 0 || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        return false;
    }

    // Q4_0 has no raw fallback. Mark only accepted tensors for repacking.
    if (src0->type == GGML_TYPE_Q4_0 && !src0->buffer) {
        sess->needs_repack.insert(src0);
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_argsort(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * dst  = op;         // indices

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        return false;
    }

    const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    const size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    struct htp_sort_vtcm_layout layout;
    if (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, false)) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_top_k(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * dst  = op;         // indices

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        return false;
    }

    const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    const size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    struct htp_sort_vtcm_layout layout;
    if (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, true)) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_rope(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        return false;
    }

    if (src0->ne[0] <= 0) {
        return false;
    }

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    if (src0_nrows == 0) {
        return false;
    }

    const int32_t * op_params = &op->op_params[0];
    const int n_dims = op_params[1];
    const int mode   = op_params[2];
    const int n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    if (n_dims < 0 || n_dims % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    if (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        return false;
    }

    float freq_base;
    memcpy(&freq_base, op_params + 5, sizeof(float));
    if (freq_base < 0.0f) {
        return false;
    }

    if (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }

    const bool is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    if (mode == GGML_ROPE_TYPE_VISION) {
        if (n_dims != (int) (src0->ne[0] / 2) || n_offs != 0) {
            return false;
        }
    }

    if (is_mrope) {
        const int32_t * sections = op_params + 11;
        if (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            return false;
        }
    }

    const int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    if (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        return false;
    }

    if (src2) {
        if (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            return false;
        }
        if (src2->ne[0] < (n_dims / 2)) {
            return false;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    if (src0->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->nb[1] < src0->ne[0] * sizeof(float) || dst->nb[1] < dst->ne[0] * sizeof(float)) {
        return false;
    }

    const uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    const uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    struct htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    if (layout.total_bytes > sess->vtcm_size) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_ssm_conv(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    // Only support FP32 for now
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Check IO tensor shapes and dims
    if (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        return false; // src0 should be effectively 3D
    }

    const int d_conv  = src1->ne[0];
    const int d_inner = src0->ne[1];
    const int n_t = dst->ne[1];
    const int n_s = dst->ne[2];

    if (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        return false;
    }
    if (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        return false;
    }
    if (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        return false;
    }
    if (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        return false;
    }
    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->nb[1] != src0->ne[0] * sizeof(float) || src1->nb[1] != src1->ne[0] * sizeof(float)) {
        return false;
    }

    struct htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    if ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_im2col(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    // For now support F32->F32 and F32->F16 only.
    if (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        return false;
    }

    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    GGML_UNUSED(sess);
    return true;
}

static bool ggml_hexagon_supported_pad(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int32_t lp0 = ((const int32_t *) op->op_params)[0];
    const int32_t rp0 = ((const int32_t *) op->op_params)[1];
    const int32_t circular = ((const int32_t *) op->op_params)[8];

    if (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_cumsum(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_diag(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    // diag only supports F32 currently
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Input must have ne[1] == 1 (vector input)
    if (src0->ne[1] != 1) {
        return false;
    }

    // Output must be square in first two dimensions
    if (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_solve_tri(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // A
    const struct ggml_tensor * src1 = op->src[1]; // B
    const struct ggml_tensor * dst  = op;         // X

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[0] != src0->ne[1]) {
        return false;
    }

    if (src0->ne[1] != src1->ne[1]) {
        return false;
    }

    if (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        return false;
    }

    if (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_tri(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) { return false; }
    if (dst->type  != GGML_TYPE_F32) { return false; }
    if (!ggml_are_same_shape(src0, dst)) { return false; }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { return false; }

    return true;

    GGML_UNUSED(sess);
}

static const char * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    return sess->c_name();
}

static void ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

static htp_op_code op_remap_to_htp(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_FLASH_ATTN_EXT:  return HTP_OP_FLASH_ATTN_EXT;
        case GGML_OP_MUL_MAT:         return HTP_OP_MUL_MAT;
        case GGML_OP_MUL_MAT_ID:      return HTP_OP_MUL_MAT_ID;
        case GGML_OP_MUL:             return HTP_OP_MUL;
        case GGML_OP_ADD:             return HTP_OP_ADD;
        case GGML_OP_ADD_ID:          return HTP_OP_ADD_ID;
        case GGML_OP_SUB:             return HTP_OP_SUB;
        case GGML_OP_DIV:             return HTP_OP_DIV;
        case GGML_OP_CPY:             return HTP_OP_CPY;
        case GGML_OP_CONT:            return HTP_OP_CPY;
        case GGML_OP_GET_ROWS:        return HTP_OP_GET_ROWS;
        case GGML_OP_SET_ROWS:        return HTP_OP_SET_ROWS;
        case GGML_OP_SUM:             return HTP_OP_SUM;
        case GGML_OP_SUM_ROWS:        return HTP_OP_SUM_ROWS;
        case GGML_OP_ARGSORT:         return HTP_OP_ARGSORT;
        case GGML_OP_TOP_K:           return HTP_OP_TOP_K;
        case GGML_OP_ARGMAX:          return HTP_OP_ARGMAX;
        case GGML_OP_NORM:            return HTP_OP_NORM;
        case GGML_OP_L2_NORM:         return HTP_OP_L2_NORM;
        case GGML_OP_RMS_NORM:        return HTP_OP_RMS_NORM;
        case GGML_OP_CONCAT:          return HTP_OP_CONCAT;
        case GGML_OP_SCALE:           return HTP_OP_SCALE;
        case GGML_OP_CLAMP:           return HTP_OP_CLAMP;
        case GGML_OP_LEAKY_RELU:      return HTP_OP_LEAKY_RELU;
        case GGML_OP_SQR:             return HTP_OP_SQR;
        case GGML_OP_SQRT:            return HTP_OP_SQRT;
        case GGML_OP_LOG:             return HTP_OP_UNARY_LOG;
        case GGML_OP_SOFT_MAX:        return HTP_OP_SOFTMAX;
        case GGML_OP_SSM_CONV:        return HTP_OP_SSM_CONV;
        case GGML_OP_GATED_DELTA_NET: return HTP_OP_GATED_DELTA_NET;
        case GGML_OP_ROPE:            return HTP_OP_ROPE;
        case GGML_OP_REPEAT:          return HTP_OP_REPEAT;
        case GGML_OP_CUMSUM:          return HTP_OP_CUMSUM;
        case GGML_OP_FILL:            return HTP_OP_FILL;
        case GGML_OP_DIAG:            return HTP_OP_DIAG;
        case GGML_OP_SOLVE_TRI:       return HTP_OP_SOLVE_TRI;
        case GGML_OP_TRI:             return HTP_OP_TRI;
        case GGML_OP_PAD:             return HTP_OP_PAD;
        case GGML_OP_IM2COL:          return HTP_OP_IM2COL;
        case GGML_OP_ROLL:            return HTP_OP_ROLL;

        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_SILU:       return HTP_OP_UNARY_SILU;
                case GGML_UNARY_OP_GELU:       return HTP_OP_UNARY_GELU;
                case GGML_UNARY_OP_GELU_QUICK: return HTP_OP_UNARY_GELU;
                case GGML_UNARY_OP_SIGMOID:    return HTP_OP_UNARY_SIGMOID;
                case GGML_UNARY_OP_NEG:        return HTP_OP_UNARY_NEG;
                case GGML_UNARY_OP_EXP:        return HTP_OP_UNARY_EXP;
                case GGML_UNARY_OP_SOFTPLUS:   return HTP_OP_UNARY_SOFTPLUS;
                case GGML_UNARY_OP_TANH:       return HTP_OP_UNARY_TANH;
                case GGML_UNARY_OP_ABS:        return HTP_OP_UNARY_ABS;
                case GGML_UNARY_OP_RELU:       return HTP_OP_UNARY_RELU;
                case GGML_UNARY_OP_STEP:       return HTP_OP_UNARY_STEP;
            default:
                break;
            }
            break;

        case GGML_OP_GLU:
            switch (ggml_get_glu_op(t)) {
                case GGML_GLU_OP_SWIGLU:     return HTP_OP_GLU_SWIGLU;
                case GGML_GLU_OP_SWIGLU_OAI: return HTP_OP_GLU_SWIGLU_OAI;
                case GGML_GLU_OP_SWIGLU_CLAMP: return HTP_OP_GLU_SWIGLU_CLAMP;
                case GGML_GLU_OP_GEGLU:      return HTP_OP_GLU_GEGLU;
                case GGML_GLU_OP_GEGLU_QUICK: return HTP_OP_GLU_GEGLU_QUICK;
                default: break;
            }
            break;

        default:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    return HTP_OP_INVALID;
}

static inline bool op_is_compute(ggml_tensor *node)
{
    return !ggml_op_is_empty(node->op) && !ggml_is_empty(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE);
}

static inline void hex_hash_combine(uint64_t & h, uint64_t v) {
    // 64-bit variant of boost::hash_combine (splitmix64-style avalanche)
    v += 0x9e3779b97f4a7c15ULL;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    v = v ^ (v >> 31);
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
}

static inline void hex_hash_tensor(uint64_t & h, const ggml_tensor * t) {
    if (!t) {
        hex_hash_combine(h, 0xdeadbeefULL);
        return;
    }
    hex_hash_combine(h, (uint64_t) t->type);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        hex_hash_combine(h, (uint64_t) t->ne[i]);
        hex_hash_combine(h, (uint64_t) t->nb[i]);
    }
    // Weight-usage and repack/tiled state affect kernel selection (see e.g.
    // ggml_hexagon_precompute_get_rows_params's `tiled` check) -- hash them too so a re-layout of the same tensor address changes the key.
    const bool is_weight = t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    hex_hash_combine(h, is_weight ? 1 : 0);
    const auto * extra = (t->buffer && ggml_backend_buffer_is_hexagon(t->buffer)) ?
        (const ggml_hexagon_tensor_extra *) t->extra : nullptr;
    const bool is_repacked = extra && (extra->flags & GGML_HEXAGON_TENSOR_REPACK) != 0;
    hex_hash_combine(h, is_repacked ? 1 : 0);
}

// Pure per-node topology hash: depends only on op/type/shape/stride/ op_params/weight-usage of the node and its sources -- never on tensor data pointers or any ctx_compute-specific identity. elt_small (the graph's ELT_SMALL bits, ggml_hexagon_elt_small_graph_bits) is hashed too: a unary op's row cap follows the batch gate.
// Two nodes with an identical hash are guaranteed (barring a hash collision, astronomically unlikely at 64 bits for this cache's size) to need identical htp_op_desc.kernel_params, so a hash hit may skip the ggml_hexagon_precompute_*_params recomputation below -- but the htp_opnode itself (which carries a live ggml_tensor* pointer) is always built fresh against the CURRENT graph's node on every call; only the derived, pointer-free kernel_params blob is cached/reused.
static uint64_t ggml_hexagon_node_hash(const ggml_tensor * n, int elt_small) {
    uint64_t h = 1469598103934665603ULL; // FNV offset basis, arbitrary non-zero seed
    hex_hash_combine(h, (uint64_t) n->op);
    hex_hash_combine(h, (uint64_t) elt_small);
    hex_hash_tensor(h, n);
    for (int s = 0; s < GGML_MAX_SRC; s++) {
        hex_hash_tensor(h, n->src[s]);
    }
    static_assert(GGML_MAX_OP_PARAMS % sizeof(uint64_t) == 0, "op_params must be a multiple of 8 bytes");
    const uint64_t * op_params64 = (const uint64_t *) n->op_params;
    for (size_t p = 0; p < GGML_MAX_OP_PARAMS / sizeof(uint64_t); p++) {
        hex_hash_combine(h, op_params64[p]);
    }
    // 0 is reserved as "no cached entry" by kparams_cache lookups below;
    // avoid colliding with it (astronomically unlikely, but make it impossible).
    return h != 0 ? h : 1;
}

static bool mm_is_hmx_eligible(const ggml_tensor * t) {
    if (opt_nhmx == 0) { return false; }

    const ggml_tensor * src0 = t->src[0];
    const ggml_tensor * src1 = t->src[1];

    const int wtype = src0->type;
    const bool is_repack    = ggml_hexagon_is_repack_type((ggml_type) wtype);
    const bool is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    const bool is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    const int ne01_padded = is_repack ? hex_round_up(src0->ne[1], 32) : src0->ne[1];

    return ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, ne01_padded, is_matmul_id, is_batched);
}

static bool is_supported_mul_mat_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams) {
    if (kparams->n_hmx) {
        return kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    if (!ggml_hexagon_is_repack_type(src0->type) || src0->type == GGML_TYPE_Q6_K) {
        return false;  // Q6_K has no fused HVX kernel
    }

    return kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
}

static bool is_supported_mul_mat_id_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams) {
    if (kparams->n_hmx) {
        return kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    if (!ggml_hexagon_is_repack_type(src0->type)) {
        return false;
    }

    return kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK;
}

static bool is_mergeable_mul_mat(const ggml_tensor * t) {
    if (t->op != GGML_OP_MUL_MAT) return false;

    const ggml_tensor * src0 = t->src[0];
    const ggml_tensor * src1 = t->src[1];
    if (src1->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    // NX HMX dequant needs N % 32 == 0: beta/alpha (N=10/12) stay standalone.
    // GGML_HEXAGON_NX_ALPHABETA admits them on the HMX kernel for prefill batches only (>= GGML_HEXAGON_MM_PREFILL_MIN_ROWS rows): decode and draft batches keep the original graph, fusion and kernels (repacked weights are stored padded to 32 rows; the DSP crops the output to N).
    if (opt_fuse_s150 && (src0->ne[1] % 32) != 0 &&
        !(opt_nx_alphabeta && ggml_hexagon_src1_rows(src1) >= GGML_HEXAGON_MM_PREFILL_MIN_ROWS && mm_is_hmx_eligible(t))) {
        return false;
    }

    if (mm_is_hmx_eligible(t)) {
        return ggml_hexagon_is_hmx_weight_type(src0->type);
    }

    return ggml_hexagon_is_repack_type(src0->type) && src0->type != GGML_TYPE_Q6_K;
}

static bool is_mergeable_mul_mat_pair(const ggml_tensor * n1, const ggml_tensor * n2) {
    if (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        return false;
    }
    if (n1->src[1] != n2->src[1]) {
        return false;
    }
    if (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        return false;
    }
    if (n1->src[0]->type != n2->src[0]->type) {
        return false;
    }
    if (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        return false;
    }
    return true;
}

static bool is_mergeable_mul_mat_id(const ggml_tensor * t) {
    if (t->op != GGML_OP_MUL_MAT_ID) return false;

    const ggml_tensor * src0 = t->src[0];
    return ggml_hexagon_is_repack_type(src0->type);
}

static bool is_mergeable_mul_mat_id_pair(const ggml_tensor * n1, const ggml_tensor * n2) {
    if (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        return false;
    }
    if (n1->src[1] != n2->src[1]) {
        return false;
    }
    if (n1->src[2] != n2->src[2]) {
        return false;
    }
    if (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        return false;
    }
    if (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        return false;
    }
    if (n1->src[0]->type != n2->src[0]->type) {
        return false;
    }
    if (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        return false;
    }
    return true;
}

// Predicate for fusing two ROPE calls (e.g. Duplicated logic/comment from is_mergeable_mul_mat_pair's family above by design -- this is a distinct op class with its own eligibility rules (shared src[1]/src[2] rather than shared src[1] alone, exact op_params match rather than shared-weight-shape match).
// Q and K) that share the same position/ freq_factors inputs and op_params into one ROPE_NX dispatch, computing the theta_cache once instead of once per tensor.
static bool is_mergeable_rope_pair(const ggml_tensor * n1, const ggml_tensor * n2) {
    if (!n1 || !n2 || n1->op != GGML_OP_ROPE || n2->op != GGML_OP_ROPE) {
        return false;
    }
    if (n1->type != GGML_TYPE_F32 || n2->type != GGML_TYPE_F32) {
        return false;
    }
    if (n1->src[0]->type != GGML_TYPE_F32 || n2->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (n1->src[1] != n2->src[1]) { // inp_pos
        return false;
    }
    if (n1->src[2] != n2->src[2]) { // freq_factors (nullable, but must match)
        return false;
    }
    if (n1->ne[0] != n2->ne[0]) {
        return false;
    }
    if (memcmp(n1->op_params, n2->op_params, sizeof(n1->op_params)) != 0) {
        return false;
    }
    return true;
}

// Admission of an MC05 norm output for fp16 tile output (GGML_HEXAGON_GATHER_ACT_TILE): the output needs at least one reader, and every
// reader must be a MUL_MAT that takes it as src1 on the HMX 2D kernel (the ACT_TILE consumer rule), so no reader needs fp32 rows.
static bool gather_tile_readers_hmx(ggml_hexagon_session * sess, const ggml_cgraph * graph, const ggml_tensor * P) {
    int n_readers = 0;
    for (int j = 0; j < graph->n_nodes; j++) {
        const ggml_tensor * n = graph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = n->src[s];
            if (!src || (src != P && src->view_src != P)) {
                continue;
            }
            if (n->op != GGML_OP_MUL_MAT || s != 1 || src != P || n->src[0] == P) {
                return false;
            }
            struct htp_mm_kernel_params kp;
            memset(&kp, 0, sizeof(kp));
            ggml_hexagon_precompute_matmul_params(sess, n->src[0], n->src[1], n, &kp);
            if (!act_tile_mm_hmx(&kp)) {
                return false;
            }
            n_readers++;
        }
    }
    return n_readers > 0;
}

// Enqueues one MC05 norm gather deferred by the comm hook. With GGML_HEXAGON_GATHER_ACT_TILE every norm output of the boundary is an
// fp16 tile view when this session's output qualifies (shape, readers, VTCM). One format per boundary: all sessions must agree.
static void ggml_hexagon_enqueue_pending_rms_ag(ggml_hexagon_session * sess, const ggml_cgraph * graph,
                                                const ggml_hexagon_pending_rms_ag & p) {
    const ggml_tensor * P = p.norm_tensors[p.rank];
    const int64_t K = P->ne[0];
    const int64_t M = P->ne[1];
    bool tile = opt_gather_act_tile != 0 && P->type == GGML_TYPE_F32 && ggml_is_contiguous(P) &&
                P->ne[2] == 1 && P->ne[3] == 1 && K % 32 == 0 && M >= PREFILL_MIN_ROWS && M % 32 == 0 &&
                strncmp(P->name, "result_norm", 11) != 0 && strncmp(P->name, "h_nextn", 7) != 0;
    if (tile) {
        tile = gather_tile_readers_hmx(sess, graph, P);
    }
    if (tile) {
        struct htp_allreduce_kernel_params kp;
        tile = ggml_hexagon_precompute_rms_ag_params(sess, P, p.rank, p.n_ranks, true, &kp);
    }
    static std::unordered_map<uint32_t, bool> decided;  // boundary (fence sequence) -> format, to check the sessions against each other
    if (decided.size() > 4096) {
        decided.clear();
    }
    const auto ins = decided.emplace(p.fence_seq_entry, tile);
    if (!ins.second && ins.first->second != tile) {
        GGML_ABORT("ggml-hex: %s GATHER_ACT_TILE: sessions disagree on the norm gather format (boundary %u)", sess->c_name(), p.fence_seq_entry);
    }
    if (!tile) {
        sess->enqueue_allreduce_rms_ag(p.sync_tensors, p.residual, p.weight, p.norm_tensors, p.rank, p.n_ranks,
                                       p.fence_seq_entry, p.fence_seq_exit, p.epsilon, false);
        return;
    }
    std::vector<const ggml_tensor *> views(p.n_ranks);
    for (uint32_t r = 0; r < p.n_ranks; r++) {
        views[r] = sess->gather_tile_view(p.norm_tensors[r]);
    }
    sess->act_tile_live[P] = views[p.rank];  // this session's readers read the tiles
    sess->act_tile_skip_live.insert(P);      // no fp32 rows: a fused NX reader that is not HMX is refused
    sess->enqueue_allreduce_rms_ag(p.sync_tensors, p.residual, p.weight, views, p.rank, p.n_ranks,
                                   p.fence_seq_entry, p.fence_seq_exit, p.epsilon, true);
}

// AE-03: enqueues one fused owner-row boundary. Tile mode writes each norm output into its owner's norm tile buffer (norm_tile_view), never into a partial sum: a tile row lands on 16 fp32 rows of the buffer it is written to, and other ranks still read those rows as partial sums (impl/ArRms.md section 3).
// The output format follows the separate norm gather (ggml_hexagon_enqueue_pending_rms_ag): fp16 tiles when every reader is an HMX matmul and the tile layout fits VTCM, decided per session and checked for agreement.
static void ggml_hexagon_enqueue_pending_add_rms(ggml_hexagon_session * sess, const ggml_cgraph * graph,
                                                 const ggml_hexagon_pending_add_rms & p, uint32_t resid_mode, uint32_t resid_off) {
    GGML_ASSERT(p.owners.size() == p.n_ranks);
    const ggml_tensor * P = p.norm_tensors[p.rank];
    const int64_t K = P->ne[0];
    const int64_t M = P->ne[1];
    if (p.partial_f16) {
        resid_mode |= HTP_ALLREDUCE_PARTIAL_F16;
    }
    bool tile = opt_gather_act_tile != 0 && P->type == GGML_TYPE_F32 && ggml_is_contiguous(P) &&
                P->ne[2] == 1 && P->ne[3] == 1 && K % 32 == 0 && M >= PREFILL_MIN_ROWS && M % 32 == 0 &&
                strncmp(P->name, "result_norm", 11) != 0 && strncmp(P->name, "h_nextn", 7) != 0;
    if (tile) {
        tile = gather_tile_readers_hmx(sess, graph, P);
    }
    if (tile) {
        struct htp_allreduce_kernel_params kp;
        tile = ggml_hexagon_precompute_add_rms_params(sess, p.resid_out, p.rank, p.n_ranks, true, &kp);
    }
    if (tile) {
        // every owner holds one norm tile buffer; the check covers all owners, so every session decides the same way
        tile = K * M * (int64_t) sizeof(uint16_t) <= (int64_t) GGML_HEXAGON_NORM_TILE_BYTES;
        for (uint32_t r = 0; tile && r < p.n_ranks; r++) {
            tile = ggml_hexagon_norm_tile_reserve(p.owners[r]);
        }
    }
    static std::unordered_map<uint32_t, bool> decided;  // boundary (fence sequence) -> format, to check the sessions against each other
    if (decided.size() > 4096) {
        decided.clear();
    }
    const auto ins = decided.emplace(p.fence_seq_entry, tile);
    if (!ins.second && ins.first->second != tile) {
        GGML_ABORT("ggml-hex: %s AR_RMS_FUSE: sessions disagree on the norm output format (boundary %u)", sess->c_name(), p.fence_seq_entry);
    }
    if (!tile) {
        sess->enqueue_allreduce_add_rms(p.partials, p.sync_tensors, p.residual, p.weight, p.resid_out, p.norm_tensors, p.rank, p.n_ranks,
                                        p.fence_seq_entry, p.fence_seq_exit, p.epsilon, false, resid_mode, resid_off);
        return;
    }
    std::vector<const ggml_tensor *> views(p.n_ranks);
    for (uint32_t r = 0; r < p.n_ranks; r++) {
        views[r] = p.owners[r]->norm_tile_view(K, M);
    }
    sess->act_tile_live[P] = views[p.rank];  // this session's readers read the tiles
    sess->act_tile_skip_live.insert(P);      // no fp32 rows: a fused NX reader that is not HMX is refused
    sess->enqueue_allreduce_add_rms(p.partials, p.sync_tensors, p.residual, p.weight, p.resid_out, views, p.rank, p.n_ranks,
                                    p.fence_seq_entry, p.fence_seq_exit, p.epsilon, true, resid_mode, resid_off);
}

// VrAr step A (GGML_HEXAGON_VR_AR bit 4). The owner rows of this session's residual stream stay in VTCM at vr_a_res_off from the first fused ADD_RMS boundary of a forward to the boundary whose norm is result_norm. Anything else (a broken chain, a fused boundary missing while pinned, another reader) would read a DDR residual that is not written, so it aborts rather than produce wrong values. Readers are checked where they run: the residual of a chained boundary is read only by its own ADD, and the output of a pinned boundary only by its own RMS_NORM.
// first boundary (not pinned): the residual comes from DDR; the sum is written to the resident rows (RES_PIN | RES_OUT); later boundaries: the residual must be the previous resident output (same data): RES_IN | RES_OUT; last boundary (result_norm): RES_IN | RES_OUT | RES_FLUSH | RES_UNPIN: the rows also go to DDR and the budget is restored. The comm hook runs after each subgraph, so one graph compute carries one boundary (its consumer subgraph starts with the ADD_RMS): the state lives in the session.
static void ggml_hexagon_vr_ar_begin(ggml_hexagon_session * sess, const ggml_cgraph * graph,
                                     const std::vector<ggml_hexagon_pending_add_rms> & pend, std::vector<uint32_t> & modes) {
    modes.assign(pend.size(), 0u);
    if ((opt_vr_ar & 4u) == 0 || sess->vtcm_size_hw == 0) {
        return;
    }
    if (pend.empty()) {
        if (sess->vr_a_pinned) {
            GGML_ABORT("ggml-hex: %s VrAR A: a subgraph of a pinned forward has no fused boundary", sess->c_name());
        }
        return;
    }
    static int n_log = 0;   // the first decisions of the process, on stderr (device logs keep stderr)
    for (size_t k = 0; k < pend.size(); k++) {
        const ggml_hexagon_pending_add_rms & p = pend[k];
        const ggml_tensor * out = p.resid_out;
        const bool last = strncmp(p.norm_tensors[p.rank]->name, "result_norm", 11) == 0;
        if (!sess->vr_a_pinned) {
            const uint32_t ne0 = (uint32_t) out->ne[0];
            const uint32_t ne1 = (uint32_t) (out->ne[1] * out->ne[2] * out->ne[3]);
            const size_t   res_bytes = (size_t) ar_rs_row_chunk_rows(ne1, p.n_ranks) * ne0 * sizeof(float);
            if (res_bytes == 0 || res_bytes > sess->vtcm_size_hw / 4) {
                continue;   // plain path for this boundary (mode 0)
            }
            sess->vr_a_pinned  = true;
            sess->vr_a_res_off = (uint32_t) (sess->vtcm_size_hw - res_bytes);
            modes[k] = HTP_ALLREDUCE_RES_PIN | HTP_ALLREDUCE_RES_OUT;
        } else {
            if (p.residual->data != sess->vr_a_out_data) {
                GGML_ABORT("ggml-hex: %s VrAR A: residual chain broken (boundary %zu of %zu)", sess->c_name(), k, pend.size());
            }
            modes[k] = HTP_ALLREDUCE_RES_IN | HTP_ALLREDUCE_RES_OUT;
        }
        if (last) {
            modes[k] |= HTP_ALLREDUCE_RES_FLUSH | HTP_ALLREDUCE_RES_UNPIN;
            sess->vr_a_pinned = false;
        }
        sess->vr_a_out_data = out->data;
        if (n_log < 16) {
            n_log++;
            fprintf(stderr, "ggml-hex: %s VrAR A boundary %zu/%zu mode 0x%x res_off %u\n", sess->c_name(), k, pend.size(), modes[k],
                    sess->vr_a_res_off);
        }
    }
    for (size_t k = 0; k < pend.size(); k++) {
        if (modes[k] == 0) {
            continue;
        }
        const ggml_hexagon_pending_add_rms & p = pend[k];
        const ggml_tensor * out = p.resid_out;
        const bool chained = (modes[k] & HTP_ALLREDUCE_RES_IN) != 0;
        for (int i = 0; i < graph->n_nodes; i++) {
            const ggml_tensor * n = graph->nodes[i];
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                const ggml_tensor * src = n->src[s];
                if (!src) {
                    continue;
                }
                if (chained && (src == p.residual || src->view_src == p.residual) && n != out) {
                    GGML_ABORT("ggml-hex: %s VrAR A: resident residual read by another node (%s)", sess->c_name(), n->name);
                }
                if (src == out || src->view_src == out) {
                    bool ok = src == out && n->op == GGML_OP_RMS_NORM && i + 1 < graph->n_nodes;
                    if (ok) {
                        const ggml_tensor * mul = graph->nodes[i + 1];
                        ok = mul->op == GGML_OP_MUL && mul->src[0] == n &&
                             std::find(p.norm_tensors.begin(), p.norm_tensors.end(), mul) != p.norm_tensors.end();
                    }
                    if (!ok) {
                        GGML_ABORT("ggml-hex: %s VrAR A: resident output read by an unexpected node (%s)", sess->c_name(), n->name);
                    }
                }
            }
        }
    }
    if (sess->vtcm_size_hw != 0) {
        sess->vtcm_size = sess->vr_a_pinned ? sess->vr_a_res_off : sess->vtcm_size_hw;
    }
}

// Restores the host VTCM budget when a graph compute ends: the resident size while a forward is pinned, the full VTCM otherwise.
struct vr_ar_budget_guard {
    ggml_hexagon_session * sess;
    explicit vr_ar_budget_guard(ggml_hexagon_session * s) : sess(s) {}
    ~vr_ar_budget_guard() {
        if (sess->vtcm_size_hw != 0) {
            sess->vtcm_size = sess->vr_a_pinned ? sess->vr_a_res_off : sess->vtcm_size_hw;
        }
    }
    vr_ar_budget_guard(const vr_ar_budget_guard &)             = delete;
    vr_ar_budget_guard & operator=(const vr_ar_budget_guard &) = delete;
};

// Token count of a graph batch: the largest activation row count among its matmuls (MUL_MAT dst ne[1]). Used only for the
// prefill-only switch GGML_HEXAGON_DMA_CLEAN; it runs once per uncached graph (cache hits reuse the flags of the node list).
static int64_t ggml_hexagon_graph_tokens(const ggml_cgraph * graph) {
    int64_t n = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (graph->nodes[i]->op == GGML_OP_MUL_MAT) {
            n = std::max(n, graph->nodes[i]->ne[1]);
        }
    }
    return n;
}

// GGML_HEXAGON_EARLY_FLUSH_OPS=N (prefill, default 32, 0 = off): at every subgraph end where this session's batch holds at least N ops, the batch is sent now instead of at synchronize, so the DSP executes while the host enqueues the rest of the forward. Same-device peers (virt_peers) share one DSP queue, where an early batch could wait on the other session's unsent batch: refused.
// Called only at the end of graph compute (after plan record or replay), so plan capture never sees the flush, and the flush point depends only on op counts (recorded and replayed batches match). Arithmetic is unchanged (batch boundary only). Sessions reach N within about one subgraph of each other, so a cross-session AR fence waits about one subgraph of host enqueue at most.
static void ggml_hexagon_early_flush(ggml_hexagon_session * sess, const ggml_cgraph * graph) {
    if (opt_early_flush_ops <= 0 || sess->mdev.idx != 0) {
        return;
    }
    if (sess->op_batch->n_ops < (uint32_t) opt_early_flush_ops) {
        return;
    }
    if (!sess->virt_peers.empty()) {
        return;
    }
    if (ggml_hexagon_graph_tokens(graph) < 64) {
        return;
    }
    if (opt_verbose) {
        GGML_LOG_INFO("early flush %s: %u ops\n", sess->c_name(), sess->op_batch->n_ops);
    }
    sess->flush_batch();
}
using ggml_hexagon_tensor_map = std::unordered_map<const ggml_tensor *, const ggml_tensor *>;
static bool ggml_hexagon_gdn_gate_fold(const ggml_cgraph * graph, int i_gdn, std::vector<htp_opnode> & nodes, htp_opnode & gdn,
                                       const std::vector<bool> & consumed);
static bool ggml_hexagon_try_fuse_ssm_conv_chain(const ggml_hexagon_session * sess, const ggml_cgraph * graph, int i,
                                                 const std::vector<bool> & consumed, std::vector<int> & absorbed,
                                                 htp_opnode & out, ggml_hexagon_tensor_map & qk_redirect);
static bool ggml_hexagon_try_fuse_ssm_conv_chain_p4k(const ggml_hexagon_session * sess, const ggml_cgraph * graph, int i,
                                                     const std::vector<bool> & consumed, std::vector<int> & absorbed,
                                                     htp_opnode & out, ggml_hexagon_tensor_map & qk_redirect);

// AE-16 for the GDN conv state (GGML_HEXAGON_ELT_SMALL bit 4). Here the clear and the state GET_ROWS are skipped only when the fused SSM_CONV_CHAIN is certain to be emitted at the CONCAT (the same matcher runs on the graph first) and nothing else reads the gathered state: the chain then reads zeros.
// qwen35's build_conv_state marks the clear of a fresh single-sequence conv state like the SSM one.
static void ggml_hexagon_zero_state_plan_conv(ggml_hexagon_session * sess, const ggml_cgraph * graph) {
    auto root = [](const ggml_tensor * t) {
        while (t->view_src) {
            t = t->view_src;
        }
        return t;
    };
    const std::vector<bool> none(graph->n_nodes, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        // the chain is matched at the CONCAT, or at the state CPY with LLAMA_P4K_CONV_FUSE (same dispatch as the graph walk)
        const ggml_tensor * n = graph->nodes[i];
        const bool at_concat = n->op == GGML_OP_CONCAT && n->src[1] && n->src[1]->ne[0] >= SSM_CONV_CHAIN_MIN_ROWS;
        const bool at_cpy    = n->op == GGML_OP_CPY && n->src[0] && n->src[0]->op == GGML_OP_TRANSPOSE;
        if (opt_ssm_conv_chain < 1 || (!at_concat && !at_cpy)) {
            continue;
        }
        std::vector<int> absorbed;
        htp_opnode chain;
        ggml_hexagon_tensor_map qk_redirect;
        const bool ok = at_concat ? ggml_hexagon_try_fuse_ssm_conv_chain(sess, graph, i, none, absorbed, chain, qk_redirect)
                                  : ggml_hexagon_try_fuse_ssm_conv_chain_p4k(sess, graph, i, none, absorbed, chain, qk_redirect);
        if (!ok || chain.inputs.empty()) {
            continue;
        }
        const ggml_tensor * st = chain.inputs[0];   // the conv state the chain reads: RESHAPE(GET_ROWS(cache rows))
        const ggml_tensor * g  = st->src[0];
        if (st->op != GGML_OP_RESHAPE || !g || g->op != GGML_OP_GET_ROWS || !g->src[0] || st->ne[2] != 1) {
            HEX_VERBOSE("ggml-hex: %s zero-state conv (#%d) kept: state source is not reshape(get_rows) or n_seqs != 1\n", sess->c_name(), i);
            continue;
        }
        const ggml_tensor * clear = nullptr;
        for (int j = 0; j < i && !clear; j++) {
            const ggml_tensor * sc = graph->nodes[j];
            if (sc->op == GGML_OP_SCALE && sc->src[0] && ggml_get_op_params_i32(sc, 2) == 1 && root(sc->src[0]) == root(g->src[0])) {
                clear = sc;
            }
        }
        // the gathered state has exactly one reader (its reshape), the reshape only the chain's CONCAT (or nothing else but the p4k ordering hint)
        int n_g = 0, n_st = 0;
        for (int j = 0; j < graph->n_nodes; j++) {
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                n_g  += graph->nodes[j]->src[k] == g;
                n_st += graph->nodes[j]->src[k] == st;
            }
        }
        if (!clear || n_g != 1 || n_st != 1 || ggml_nbytes(st) > GGML_HEXAGON_ZERO_STATE_BYTES || !ggml_hexagon_zero_state_reserve(sess)) {
            HEX_VERBOSE("ggml-hex: %s zero-state conv (#%d) kept: clear %d n_g %d n_st %d\n", sess->c_name(), i, clear != nullptr, n_g, n_st);
            continue;
        }
        sess->zero_state_clear_skip.insert(clear);
        sess->zero_state_dead.insert(g);
        sess->zero_state_conv.insert(n);
        HEX_VERBOSE("ggml-hex: %s zero-state conv chain (#%d): fresh-state clear skipped\n", sess->c_name(), i);
    }
}

static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    // LLAMA_CYCLE_TIMING>=2 (hostovh diagnostics): enqueue time of every compute call, pass start = first compute after a sync
    struct ov_enq_scope {
        ggml_hexagon_session * s;
        uint64_t t0;
        ~ov_enq_scope() { if (t0) { s->ov_enq_us += ggml_time_us() - t0; s->ov_n_compute++; } }
    } ov_scope { sess, opt_cycle_timing >= 2 ? (uint64_t) ggml_time_us() : (uint64_t) 0 };
    if (ov_scope.t0 && sess->ov_n_compute == 0) { sess->ov_first_us = ov_scope.t0; }

    if (sess->last_error > HTP_STATUS_OK) {
        sess->pending_rms_ag.clear();
        sess->pending_add_rms.clear();
        return GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);
    // Residual ADDs executed by the gap-fused comm hook for this subgraph (ggml_backend_hexagon_comm_allreduce_add_tensor).
    std::vector<const ggml_tensor *> gap_skip;
    gap_skip.swap(sess->ar_gap_skip);
    const bool has_gap_skip = !gap_skip.empty();
    // fp16 activation-tile state is per graph: views of an earlier graph must never match this graph's pointers.
    sess->act_tile_live.clear();
    sess->act_tile_ok.clear();
    sess->act_tile_skip_ok.clear();
    sess->act_tile_skip_live.clear();
    sess->swiglu_ok.clear();
    sess->norm_rope_only.clear();
    sess->ngtile_ok.clear();
    sess->gatetile_ok.clear();
    // VrAr step A (GGML_HEXAGON_VR_AR bit 4): the residual rows of a pinned forward stay resident across its computes. The state is
    // advanced by ggml_hexagon_vr_ar_begin before the pending records are enqueued; the guard restores the host budget at the end.
    const vr_ar_budget_guard vr_budget(sess);
    std::vector<ggml_hexagon_pending_add_rms> pending_fused;
    pending_fused.swap(sess->pending_add_rms);   // AE-03: fused owner-row boundaries recorded by the predecessor subgraph go first
    std::vector<uint32_t> vr_modes;
    ggml_hexagon_vr_ar_begin(sess, graph, pending_fused, vr_modes);
    for (size_t k = 0; k < pending_fused.size(); k++) {
        ggml_hexagon_enqueue_pending_add_rms(sess, graph, pending_fused[k], vr_modes[k], sess->vr_a_res_off);
    }
    // GGML_HEXAGON_GATHER_ACT_TILE: MC05 norm gathers deferred by the comm hook are enqueued here, after the admission of their
    // readers and before any node of this subgraph. This runs before the cache and plan paths, so both see the same enqueue.
    if (!sess->pending_rms_ag.empty()) {
        std::vector<ggml_hexagon_pending_rms_ag> pending_ag;
        pending_ag.swap(sess->pending_rms_ag);
        for (const auto & p : pending_ag) {
            ggml_hexagon_enqueue_pending_rms_ag(sess, graph, p);
        }
    }
    sess->zero_state_clear_skip.clear();
    sess->zero_state_dead.clear();
    sess->zero_state_gdn.clear();
    sess->zero_state_conv.clear();
    sess->elt_small = ggml_hexagon_elt_small_graph_bits(graph);

    const std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit across multi-entry graph cache
    ggml_hexagon_session::htp_cached_graph * cached_entry = nullptr;
    if (graph->uid != 0) {
        for (auto it = sess->cached_graphs.rbegin(); it != sess->cached_graphs.rend(); ++it) {
            if (it->uid == graph->uid) {
                cached_entry = &(*it);
                nodes_ptr = &cached_entry->nodes;
                break;
            }
        }
    }
    bool cache_hit = (cached_entry != nullptr);
    if (cache_hit) {
        // fast op-enqueue: replay cached plan if valid
        if (opt_enqueue_fast && !sess->fast_enq_disabled && cached_entry->plan && !cached_entry->plan->dirty &&
            cached_entry->plan->gap_skip == has_gap_skip) {
            if (sess->plan_replay(*cached_entry->plan)) {
                sess->n_replay++;
                if (sess->n_replay == 1) {
                    GGML_LOG_INFO("ggml-hex: %s enqueue-fast: replaying pre-built subgraph plans\n", sess->c_name());
                }
                ggml_hexagon_early_flush(sess, graph);
                return GGML_STATUS_SUCCESS;
            }
        }
    } else {
        // Tag fusable tensors in graph
        for (int i = 0; i < graph->n_nodes; i++) {
            auto * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            if (!extra) continue;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            if (graph->nodes[i]->op == GGML_OP_RMS_NORM &&
                (ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }) ||
                 ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }))) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } else if (opt_norm_gated && graph->nodes[i]->op == GGML_OP_MUL && i + 1 < graph->n_nodes &&
                       graph->nodes[i + 1]->op == GGML_OP_GLU && ggml_get_glu_op(graph->nodes[i + 1]) == GGML_GLU_OP_SWIGLU &&
                       graph->nodes[i + 1]->src[1] == graph->nodes[i] && ggml_node_has_n_uses(graph, i, 1)) {
                // the normalized output of the gated output norm (RMS_NORM_MUL) whose only reader is the SWIGLU
                // (try_fuse_norm_gated; graph-level order comes from LLAMA_S150_GDN_SWIGLU in qwen35.cpp)
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } else if (graph->nodes[i]->op == GGML_OP_MUL_MAT || graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                if ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_node_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            } else if (graph->nodes[i]->op == GGML_OP_SSM_CONV &&
                       i + 1 < graph->n_nodes &&
                       graph->nodes[i + 1]->op == GGML_OP_UNARY &&
                       ggml_get_unary_op(graph->nodes[i + 1]) == GGML_UNARY_OP_SILU &&
                       graph->nodes[i + 1]->src[0] == graph->nodes[i] &&
                       ggml_node_has_n_uses(graph, i, 1)) {
                // try_fuse_ssm_conv_silu (see enqueue_op): SSM_CONV's dst and SILU's dst are same-shape here, but ggml_can_fuse's op-list form (unlike this explicit check) can't express "GGML_OP_UNARY specifically carrying GGML_UNARY_OP_SILU", only the raw ggml_op enum -- so this uses the same manual-adjacency-check pattern as the MUL_MAT/CONCAT tagging above, re-checked more precisely inside try_fuse_ssm_conv_silu itself.
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } else if (graph->nodes[i]->op == GGML_OP_CONCAT &&
                       i + 1 < graph->n_nodes &&
                       (graph->nodes[i + 1]->op == GGML_OP_SSM_CONV) &&
                       graph->nodes[i + 1]->src[0] == graph->nodes[i] &&
                       ggml_node_has_n_uses(graph, i, 1)) {
                // try_fuse_concat_ssm_conv (see enqueue_op): the reordered GDN conv-input build (delta-net-base.cpp's deferred_state_cpys) makes CONCAT land node-array-adjacent to its only real consumer, SSM_CONV, whenever the fusion's other shape preconditions hold (checked again, more precisely, inside try_fuse_concat_ssm_conv itself).
                // Not ggml_can_fuse: that helper also requires the two nodes be ggml_are_same_shape, which CONCAT (ne[0] = n_state_rows + n_tokens) and SSM_CONV (ne[0] = d_inner) never are.
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            }
        }

        if (opt_rope_norm) {
            // RMS_NORM+MUL outputs read by exactly one node and that node is a ROPE (the attention q/k norms): ROPE_NORM may fold them
            for (int i = 0; i + 1 < graph->n_nodes; i++) {
                const ggml_tensor * m = graph->nodes[i];
                if (m->op != GGML_OP_MUL || !m->src[0] || m->src[0]->op != GGML_OP_RMS_NORM || !ggml_node_has_n_uses(graph, i, 1)) {
                    continue;
                }
                for (int j = i + 1; j < graph->n_nodes && j < i + 8; j++) {
                    if (graph->nodes[j]->op == GGML_OP_ROPE && graph->nodes[j]->src[0] == m) {
                        sess->norm_rope_only.insert(m);
                        break;
                    }
                }
            }
        }
        if (opt_act_tile) {
            if (ggml_hexagon_graph_rows(graph) >= PREFILL_MIN_ROWS) {
                ggml_hexagon_act_tile_plan(sess, graph);
            }
        }
        if (opt_swiglu_out) {
            ggml_hexagon_swiglu_out_plan(sess, graph);
        }
        if (opt_gate_tile && opt_gate_mul && ggml_hexagon_graph_rows(graph) >= PREFILL_MIN_ROWS) {
            ggml_hexagon_gate_tile_plan(sess, graph);
        }
        if (opt_ng_tile && opt_norm_gated && ggml_hexagon_graph_rows(graph) >= PREFILL_MIN_ROWS) {
            ggml_hexagon_ng_tile_plan(sess, graph);
        }
        // GGML_HEXAGON_DMA_CLEAN is decided per batch: only batches of 64 tokens or more carry the flag (add_op)
        const bool dma_clean_graph = opt_dma_clean && ggml_hexagon_graph_tokens(graph) >= 64;
        if (sess->elt_small & 16) {
            ggml_hexagon_zero_state_plan(sess, graph);
            ggml_hexagon_zero_state_plan_conv(sess, graph);
        }
        computed_nodes.reserve(graph->n_nodes);

        // nodes absorbed by a fusion that looks ahead.
        // The look-ahead runs only with GGML_HEXAGON_SSM_CONV_CHAIN and for a CONCAT of at least SSM_CONV_CHAIN_MIN_ROWS tokens; the consumed vector is allocated on the first such CONCAT, so decode and small batches take the unfused path with no per-graph bookkeeping.
        std::vector<bool> consumed;
        ggml_hexagon_tensor_map qk_redirect;

        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            if (!op_is_compute(n) || (!consumed.empty() && consumed[i])) {
                continue;
            }
            if (sess->zero_state_clear_skip.count(n) || sess->zero_state_dead.count(n)) {
                continue;   // AE-16: a fresh-state clear, or the state gather that only feeds a GDN reading zeros (ggml_hexagon_zero_state_plan)
            }

            if (n->op == GGML_OP_CONCAT && opt_ssm_conv_chain >= 1 && n->src[1] &&
                n->src[1]->ne[0] >= SSM_CONV_CHAIN_MIN_ROWS) {
                if (consumed.empty()) {
                    consumed.assign(graph->n_nodes, false);
                }
                htp_opnode chain;
                std::vector<int> absorbed;
                const bool chain_ok = ggml_hexagon_try_fuse_ssm_conv_chain(sess, graph, i, consumed, absorbed, chain, qk_redirect);
                if (!chain_ok && sess->zero_state_conv.count(n)) {
                    GGML_ABORT("ggml-hex: %s zero-state conv chain at node %d was planned but not emitted\n", sess->c_name(), i);
                }
                if (chain_ok) {
                    if (sess->zero_state_conv.count(n)) {
                        // AE-16 for the conv state: the clear and the state gather are gone, the chain reads zeros
                        chain.inputs[0] = ggml_hexagon_zero_state_src(sess, chain.inputs[0], true);
                    }
                    HEX_VERBOSE("ggml-hex: %s fused %s at node %d\n", sess->c_name(), chain.name.c_str(), i);
                    for (int j : absorbed) {
                        consumed[j] = true;
                    }
                    computed_nodes.push_back(std::move(chain));
                    continue;
                }
            }

            if (n->op == GGML_OP_CPY && opt_ssm_conv_chain >= 1 && n->src[0] && n->src[0]->op == GGML_OP_TRANSPOSE) {
                if (consumed.empty()) {
                    consumed.assign(graph->n_nodes, false);
                }
                htp_opnode chain;
                std::vector<int> absorbed;
                const bool chain_ok = ggml_hexagon_try_fuse_ssm_conv_chain_p4k(sess, graph, i, consumed, absorbed, chain, qk_redirect);
                if (!chain_ok && sess->zero_state_conv.count(n)) {
                    GGML_ABORT("ggml-hex: %s zero-state conv chain at node %d was planned but not emitted\n", sess->c_name(), i);
                }
                if (chain_ok) {
                    if (sess->zero_state_conv.count(n)) {
                        chain.inputs[0] = ggml_hexagon_zero_state_src(sess, chain.inputs[0], true);
                    }
                    HEX_VERBOSE("ggml-hex: %s fused %s at node %d\n", sess->c_name(), chain.name.c_str(), i);
                    for (int j : absorbed) {
                        consumed[j] = true;
                    }
                    computed_nodes.push_back(std::move(chain));
                    continue;
                }
            }

            htp_opnode node(HTP_OP_INVALID, n);
            node.opcode = op_remap_to_htp(n);
            node.dma_clean = dma_clean_graph;

            if (sess->zero_state_gdn.count(n)) {
                // AE-16: this GDN reads zeros for its state (ggml_hexagon_zero_state_plan)
                node.inputs[5] = ggml_hexagon_zero_state_src(sess, n->src[5]);
            }

            // q and k written into silu by the fused q/k epilogue: the GATED_DELTA_NET reads them from the chain outputs
            if (!qk_redirect.empty() && node.opcode == HTP_OP_GATED_DELTA_NET) {
                for (int p = 0; p < 2; p++) {
                    auto it = qk_redirect.find(node.inputs[p]);
                    if (it != qk_redirect.end()) {
                        node.inputs[p] = it->second;
                    }
                }
            }

            if (opt_gdn_gate_fold && node.opcode == HTP_OP_GATED_DELTA_NET && ggml_hexagon_graph_tokens(graph) >= 64 && sess->mdev.count <= 1) {
                // folded gate: ggml_hexagon_gdn_gate_fold removes the SOFTPLUS/MUL/SIGMOID opnodes from computed_nodes
                if (ggml_hexagon_gdn_gate_fold(graph, i, computed_nodes, node, consumed)) {
                    HEX_VERBOSE("ggml-hex: %s folded GDN gate into %s\n", sess->c_name(), node.node->name);
                }
            }

            const bool has_kparams = node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID ||
                node.opcode == HTP_OP_MUL || node.opcode == HTP_OP_ADD || node.opcode == HTP_OP_ADD_ID ||
                node.opcode == HTP_OP_SUB || node.opcode == HTP_OP_DIV || node.opcode == HTP_OP_FLASH_ATTN_EXT ||
                htp_op_is_unary(node.opcode) || node.opcode == HTP_OP_GET_ROWS || node.opcode == HTP_OP_SET_ROWS ||
                node.opcode == HTP_OP_ROPE || node.opcode == HTP_OP_SSM_CONV || node.opcode == HTP_OP_SOFTMAX ||
                node.opcode == HTP_OP_GATED_DELTA_NET || node.opcode == HTP_OP_ARGSORT || node.opcode == HTP_OP_TOP_K;

            // kparams_cache hit: skip the (comparatively expensive) ggml_hexagon_precompute_*_params call below -- the cached blob is plain POD, never a pointer, so reuse across unrelated ggml_cgraph objects (which is the common case once the uid- keyed cached_graphs cache above almost always misses) is safe.
            // elt_small keys the cache too: a unary op's row cap follows the batch gate, which its shape alone does not carry.
            const uint64_t node_hash = has_kparams ? ggml_hexagon_node_hash(n, sess->elt_small) : 0;
            bool kparams_hit = has_kparams && sess->kparams_lookup(node_hash, node.kernel_params);

            if (!kparams_hit && (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID)) {
                ggml_hexagon_precompute_matmul_params(sess,
                    node.node->src[0], node.node->src[1], node.node,
                    (struct htp_mm_kernel_params *)node.kernel_params
                );
            } else if (!kparams_hit && (node.opcode == HTP_OP_MUL || node.opcode == HTP_OP_ADD || node.opcode == HTP_OP_ADD_ID || node.opcode == HTP_OP_SUB || node.opcode == HTP_OP_DIV)) {
                const ggml_tensor * src1 = node.node->src[1];
                GGML_ASSERT(ggml_hexagon_precompute_binary_params(sess,
                    node.opcode, node.node->src[0], src1, node.node,
                    (struct htp_binary_kernel_params *) node.kernel_params));
            } else if (!kparams_hit && node.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    node.node,
                    (struct htp_fa_kernel_params *)node.kernel_params
                );
            } else if (!kparams_hit && htp_op_is_unary(node.opcode)) {
                auto inputs = node.get_inputs();
                const struct ggml_tensor * src0 = inputs[0];
                const struct ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    node.opcode, src0, src1, node.dst(),
                    (struct htp_unary_kernel_params *)node.kernel_params
                );
            } else if (!kparams_hit && node.opcode == HTP_OP_GET_ROWS) {
                ggml_hexagon_precompute_get_rows_params(sess,
                    node.node->src[0], node.node->src[1], node.dst(),
                    (struct htp_get_rows_kernel_params *)node.kernel_params
                );
            } else if (!kparams_hit && node.opcode == HTP_OP_SET_ROWS) {
                ggml_hexagon_precompute_set_rows_params(sess,
                    node.node->src[0], node.node->src[1], node.dst(),
                    (struct htp_set_rows_kernel_params *)node.kernel_params
                );
            } else if (!kparams_hit && node.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    node.node,
                    (struct htp_rope_kernel_params *)node.kernel_params
                );
            } else if (!kparams_hit && node.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    node.node->src[0], node.node->src[1], node.dst(),
                    (struct htp_ssm_conv_kernel_params *)node.kernel_params
                );
            } else if (!kparams_hit && node.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    node.node,
                    (struct htp_softmax_kernel_params *)node.kernel_params
                );
            } else if (!kparams_hit && node.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    node.node,
                    (struct htp_gdn_kernel_params *)node.kernel_params
                );
            } else if (!kparams_hit && (node.opcode == HTP_OP_ARGSORT || node.opcode == HTP_OP_TOP_K)) {
                ggml_hexagon_precompute_sort_params(sess,
                    node.node,
                    node.opcode == HTP_OP_TOP_K,
                    (struct htp_sort_kernel_params *) node.kernel_params
                );
            }

            if (has_kparams && !kparams_hit) {
                sess->kparams_store(node_hash, node.kernel_params);
            }
            computed_nodes.push_back(std::move(node));
        }

        if (graph->uid != 0) {
            if (sess->cached_graphs.size() >= ggml_hexagon_session::HTP_GRAPH_CACHE_MAX) {
                sess->cached_graphs.erase(sess->cached_graphs.begin());
            }
            sess->cached_graphs.push_back({graph->uid, std::move(computed_nodes), nullptr});
            cached_entry = &sess->cached_graphs.back();
            nodes_ptr = &cached_entry->nodes;
        } else {
            nodes_ptr = &computed_nodes;
        }
    }
    // fast op-enqueue: this call took the slow path (cache miss, or a cache-hit whose plan replay above failed/was absent).
    // If this uid's cached plan is missing (no plan recorded yet, went dirty, or a different uid just evicted the single-slot cache above), record one from this run's batch-state trajectory so the NEXT repeat of this uid can replay instead of re-enqueuing.
    htp_graph_plan * record_plan = nullptr;
    if (opt_enqueue_fast && !sess->fast_enq_disabled && graph->uid != 0 && cached_entry) {
        if (cached_entry->plan && cached_entry->plan->dirty) {
            cached_entry->plan.reset();
        }
        if (!cached_entry->plan) {
            cached_entry->plan = std::make_unique<htp_graph_plan>();
            sess->plan_begin(*cached_entry->plan);
            cached_entry->plan->gap_skip = has_gap_skip;
            record_plan = cached_entry->plan.get();
        }
    }

    // Queue and execute
    for (const auto & node : *nodes_ptr) {
        // ADD consumed by the gap-fused ALLREDUCE+ADD of this subgraph (enqueued by the comm hook): not enqueued again.
        if (has_gap_skip && std::find(gap_skip.begin(), gap_skip.end(), node.node) != gap_skip.end()) {
            continue;
        }
        sess->enqueue_op(node);
    }

    if (record_plan) {
        if (!sess->plan_finish(*record_plan)) {
            if (cached_entry) cached_entry->plan.reset();
        } else if (opt_verbose) {
            HEX_VERBOSE("ggml-hex: %s enqueue-fast: recorded plan for uid %llu (%u ops, %u new tens, %u new bufs)\n",
                        sess->c_name(), (unsigned long long) graph->uid,
                        (unsigned) record_plan->ops.size(), (unsigned) record_plan->new_tens.size(),
                        (unsigned) record_plan->new_bufs.size());
        }
    }

    if (sess->last_error > HTP_STATUS_OK) {
        return GGML_STATUS_FAILED;
    }

    ggml_hexagon_early_flush(sess, graph);
    return GGML_STATUS_SUCCESS;
}

// LLAMA_CYCLE_TIMING>=2: prints the host timestamps of the batch writes since the last sync, in us from the first write
static void ggml_hexagon_dispatch_log_flush() {
    if (opt_disp_log.empty()) {
        return;
    }
    uint64_t base = opt_disp_log[0].t_w0;
    for (const auto & r : opt_disp_log) {
        base = std::min(base, r.t_w0);
    }
    for (const auto & r : opt_disp_log) {
        GGML_LOG_INFO("ggml-hex: dispatch %s seq %llu : begin %+lld us, write %+lld..%+lld us (copy %lld us)\n",
                r.name, (unsigned long long) r.seq,
                (long long) ((int64_t) r.t_begin - (int64_t) base), (long long) ((int64_t) r.t_w0 - (int64_t) base),
                (long long) ((int64_t) r.t_w1 - (int64_t) base), (long long) ((int64_t) r.t_w0 - (int64_t) r.t_begin));
    }
    opt_disp_log.clear();
}

static void ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    ggml_hexagon_dispatch_log_flush();
    if (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
}

enum ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

struct ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

struct ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    void reset() {
        ranges.clear();
    }

    void add(const ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    bool check(const ggml_hexagon_mem_range & mr) const {
        for (const auto & cmp : ranges) {
            if (mr.pb != cmp.pb) {
                continue;
            }
            if (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                continue;
            }
            if (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                return false;
            }
        }
        return true;
    }
};

static ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(const ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    const ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    if (tensor->buffer) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } else {
        mr = {
            /*.pb =*/ (uint64_t) base,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    return mr;
}

static void ggml_hexagon_mem_ranges_add_node(ggml_hexagon_mem_ranges & mrs, const htp_opnode & node) {
    if (node.is_empty()) return;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node.node->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(node.node->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    for (const auto * fused : node.fused) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(node.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

static bool ggml_hexagon_mem_ranges_check_node(const ggml_hexagon_mem_ranges & mrs, const htp_opnode & node) {
    if (node.is_empty()) return true;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node.node->src[i]) {
            if (!mrs.check(ggml_hexagon_mem_range_from_tensor(node.node->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                return false;
            }
        }
    }
    for (const auto * fused : node.fused) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (fused->src[i]) {
                if (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    return false;
                }
            }
        }
    }
    return mrs.check(ggml_hexagon_mem_range_from_tensor(node.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

// HTP_OP_SSM_CONV_CHAIN: CONCAT(conv_states, qkv^T), CPY of the new conv state, SSM_CONV and SILU in one op. The ops are not adjacent: the fused op runs at the CONCAT, so the ops in between must not touch what it writes.
// With LLAMA_P4K_CONV_FUSE=1 the state CPY reads a TRANSPOSE of qkv_mixed_orig (conv_tail_from_qkv) instead of a view of the CONCAT, so the matcher declines and the unfused path runs. Graph shape matched (src/models/delta-net-base.cpp build_conv_state, LLAMA_P4K_CONV_FUSE unset, n_rs_seq == 0): CONCAT(conv_states, TRANSPOSE(qkv_mixed)) -> VIEW (the state tail, view_offs = n_t * 4) -> CPY into the state view, CONCAT -> SSM_CONV(conv_input, conv1d) -> SILU.

static int32_t ggml_hexagon_tensor_use_count(const ggml_cgraph * graph, const ggml_tensor * t) {
    const size_t hash_pos = ggml_hash_find(&graph->visited_hash_set, t);
    if (hash_pos == GGML_HASHSET_FULL || !ggml_bitset_get(graph->visited_hash_set.used, hash_pos)) {
        return -1;
    }
    return graph->use_counts[hash_pos];
}

static inline bool ggml_hexagon_ssm_conv_chain_tensor_ok(const ggml_tensor * t) {
    return t && t->buffer && t->extra && t->data && t->type == GGML_TYPE_F32;
}


// GGML_HEXAGON_GDN_GATE_FOLD: the GATED_DELTA_NET of a prefill graph reads the raw gate input x = alpha + dt (the softplus input) and the raw beta (the sigmoid input) and applies softplus(x) * ssm_a and sigmoid itself (phase 1a), so the SOFTPLUS, MUL(A) and SIGMOID nodes ahead of it need no op.
// Matches GDN.src[3] = RESHAPE(MUL(SOFTPLUS(x), A)) and GDN.src[4] = RESHAPE(SIGMOID(RESHAPE(beta_mm))). Returns true and rewrites `gdn` (inputs 3/4/6, extra flag) on a match. The three nodes are removed from `nodes` (they are single-use); the GDN reads x and beta_mm where they were produced, so no node between the first removed node and the GDN may write them.
static bool ggml_hexagon_gdn_gate_fold(const ggml_cgraph * graph, int i_gdn, std::vector<htp_opnode> & nodes, htp_opnode & gdn,
                                       const std::vector<bool> & consumed) {
    const ggml_tensor * g_in = gdn.inputs.size() > 4 ? gdn.inputs[3] : nullptr;
    const ggml_tensor * b_in = gdn.inputs.size() > 4 ? gdn.inputs[4] : nullptr;
    if (!g_in || !b_in || gdn.inputs.size() != 6) return false;
    const ggml_tensor * mul = g_in;
    while (mul && mul->op == GGML_OP_RESHAPE) mul = mul->src[0];
    const ggml_tensor * sig = b_in;
    while (sig && sig->op == GGML_OP_RESHAPE) sig = sig->src[0];
    if (!mul || mul->op != GGML_OP_MUL || !sig || sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) return false;
    const ggml_tensor * sp = mul->src[0];
    const ggml_tensor * a  = mul->src[1];
    if (sp && !(sp->op == GGML_OP_UNARY && ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS)) { std::swap(sp, a); }
    if (!sp || !a || sp->op != GGML_OP_UNARY || ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS) return false;
    const ggml_tensor * x = sp->src[0];
    const ggml_tensor * braw = sig->src[0];
    while (braw && braw->op == GGML_OP_RESHAPE) braw = braw->src[0];
    if (!x || !braw) return false;
    for (const ggml_tensor * t : { mul, sp, sig }) {
        if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_hexagon_tensor_use_count(graph, t) != 1) return false;
    }
    // the reshapes between the nodes and the GDN are single-use views of them
    if (ggml_hexagon_tensor_use_count(graph, g_in) != 1 || ggml_hexagon_tensor_use_count(graph, b_in) != 1) return false;
    const int64_t H = gdn.node->src[2]->ne[1];
    const int64_t T = gdn.node->src[2]->ne[2];
    const int64_t S = gdn.node->src[2]->ne[3];
    if (T * S < 64 || H > 32 || H < 1) return false;
    if (x->type != GGML_TYPE_F32 || braw->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(braw) || x->ne[0] != H || braw->ne[0] != H || a->ne[0] != H || ggml_nelements(a) != H) return false;
    if (ggml_nelements(x) != H * T * S || ggml_nelements(braw) != H * T * S || !ggml_is_contiguous(a)) return false;
    if (!ggml_hexagon_ssm_conv_chain_tensor_ok(x) || !ggml_hexagon_ssm_conv_chain_tensor_ok(braw) || !ggml_hexagon_ssm_conv_chain_tensor_ok(a)) return false;
    // g/beta rows as the GDN reads them: [1, H, T, S] with row stride H floats
    if (g_in->ne[0] != 1 || g_in->ne[1] != H || g_in->nb[2] != (size_t) H * sizeof(float) || b_in->ne[0] != 1 || b_in->ne[1] != H ||
        b_in->nb[2] != (size_t) H * sizeof(float) || g_in->nb[3] != (size_t) H * T * sizeof(float) || b_in->nb[3] != g_in->nb[3]) return false;

    // positions of the three nodes in the graph (they precede the GDN)
    int first = i_gdn;
    int cnt = 0;
    for (int j = i_gdn - 1; j >= 0 && i_gdn - j < 64 && cnt < 3; j--) {
        const ggml_tensor * n = graph->nodes[j];
        if (n == mul || n == sp || n == sig) { first = j; cnt++; }
    }
    if (cnt != 3) return false;
    ggml_hexagon_mem_ranges mrs;
    mrs.add(ggml_hexagon_mem_range_from_tensor(x,    HEXAGON_MEM_RANGE_TYPE_SRC));
    mrs.add(ggml_hexagon_mem_range_from_tensor(braw, HEXAGON_MEM_RANGE_TYPE_SRC));
    for (int j = first + 1; j < i_gdn; j++) {
        const ggml_tensor * n = graph->nodes[j];
        if (n == mul || n == sp || n == sig || !op_is_compute(graph->nodes[j])) continue;
        if (!ggml_hexagon_mem_ranges_check_node(mrs, htp_opnode(HTP_OP_INVALID, graph->nodes[j]))) {
            HEX_VERBOSE("ggml-hex: skip GDN gate fold: %s aliases x/beta\n", n->name);
            return false;
        }
    }

    // erase the three computed nodes
    size_t removed = 0;
    for (size_t k = nodes.size(); k-- > 0 && removed < 3; ) {
        if (nodes[k].node == mul || nodes[k].node == sp || nodes[k].node == sig) { nodes.erase(nodes.begin() + k); removed++; }
    }
    if (removed != 3) GGML_ABORT("ggml-hex: GDN gate fold: gate nodes not found in the node list\n");
    (void) consumed;

    ggml_tensor gt = *g_in;
    gt.op = GGML_OP_NONE; gt.flags = 0; gt.view_src = nullptr; gt.view_offs = 0;
    gt.data = x->data; gt.buffer = x->buffer; gt.extra = x->extra;
    memset(gt.src, 0, sizeof(gt.src));
    ggml_tensor bt = *b_in;
    bt.op = GGML_OP_NONE; bt.flags = 0; bt.view_src = nullptr; bt.view_offs = 0;
    bt.data = braw->data; bt.buffer = braw->buffer; bt.extra = braw->extra;
    memset(bt.src, 0, sizeof(bt.src));
    snprintf(gt.name, sizeof(gt.name), "%s (raw)", x->name);
    snprintf(bt.name, sizeof(bt.name), "%s (raw)", braw->name);
    gdn.inputs[3] = gdn.add_dummy(gt);
    gdn.inputs[4] = gdn.add_dummy(bt);
    gdn.inputs.push_back(a);
    gdn.extra_flags |= HTP_OPFLAGS_GDN_GATE_FOLD;
    return true;
}

// q/k per-head l2-norm after the chain (build_gdn_l2_norm): SCALE(RMS_NORM(view(silu))) for q (channel 0) and k (n_q).
struct ggml_hexagon_ssm_conv_chain_qk {
    int           idx[4]   = { -1, -1, -1, -1 };     // RMS_NORM(q), SCALE(q), RMS_NORM(k), SCALE(k)
    ggml_tensor * out[2]   = { nullptr, nullptr };   // the two SCALE nodes, q and k
    int64_t       head_dim = 0;
    int64_t       n_qk     = 0;
    float         eps      = 0.0f;
    float         scale    = 0.0f;
    float         bias     = 0.0f;
};

static bool ggml_hexagon_match_ssm_conv_chain_qk(const ggml_cgraph * graph, int i_silu,
                                                 const std::vector<bool> & consumed,
                                                 const ggml_tensor * silu, ggml_hexagon_ssm_conv_chain_qk & qk) {
    constexpr int N_LOOKAHEAD = 8;

    const int64_t d_inner = silu->ne[0];
    const int64_t n_t     = silu->ne[1];
    const int64_t n_s     = silu->ne[2];

    // the q/k range of silu is no longer written, so silu must not be a graph output
    if (silu->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }

    ggml_tensor * rms[2] = { nullptr, nullptr };
    ggml_tensor * scl[2] = { nullptr, nullptr };
    int           idx[4] = { -1, -1, -1, -1 };
    int           n_pairs = 0;
    for (int j = i_silu + 1, k = 0; j < graph->n_nodes && k < N_LOOKAHEAD && n_pairs < 2; j++) {
        ggml_tensor * n = graph->nodes[j];
        if (!op_is_compute(n)) {
            continue;
        }
        k++;
        if (consumed[j]) {
            return false;
        }
        if (!rms[n_pairs] && n->op == GGML_OP_RMS_NORM && n->src[0] && n->src[0]->op == GGML_OP_VIEW &&
            n->src[0]->src[0] == silu) {
            rms[n_pairs]         = n;
            idx[2 * n_pairs]     = j;
        } else if (rms[n_pairs] && n->op == GGML_OP_SCALE && n->src[0] == rms[n_pairs]) {
            scl[n_pairs]         = n;
            idx[2 * n_pairs + 1] = j;
            n_pairs++;
        }
    }
    if (n_pairs != 2) {
        return false;
    }

    // q is the pair at channel 0
    const int q = rms[0]->src[0]->view_offs == 0 ? 0 : 1;
    const int k = 1 - q;

    const ggml_tensor * view[2] = { rms[q]->src[0], rms[k]->src[0] };
    const int64_t hd = view[0]->ne[0];
    if (hd < 32 || hd % 32 != 0 || d_inner % hd != 0 || view[1]->ne[0] != hd) {
        return false;
    }
    for (int p = 0; p < 2; p++) {
        const ggml_tensor * v = view[p];
        const ggml_tensor * r = p == 0 ? rms[q] : rms[k];
        const ggml_tensor * o = p == 0 ? scl[q] : scl[k];
        if (v->type != GGML_TYPE_F32 || v->ne[2] != n_t || v->ne[3] != n_s || v->nb[0] != sizeof(float) ||
            v->nb[1] != (size_t) hd * sizeof(float) || v->nb[2] != silu->nb[1] ||
            (n_s > 1 && v->nb[3] != silu->nb[2]) ||
            ggml_hexagon_tensor_use_count(graph, v) != 1) {
            return false;
        }
        if ((r->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_hexagon_tensor_use_count(graph, r) != 1 ||
            r->type != GGML_TYPE_F32) {
            return false;
        }
        if (!ggml_hexagon_ssm_conv_chain_tensor_ok(o) || !ggml_are_same_shape(o, v) || !ggml_is_contiguous(o)) {
            return false;
        }
    }
    const int64_t n_q = hd * view[0]->ne[1];
    const int64_t n_k = hd * view[1]->ne[1];
    if (view[0]->view_offs != 0 || view[1]->view_offs != (size_t) n_q * sizeof(float) || n_q + n_k > d_inner) {
        return false;
    }

    // one eps, scale and bias for both pairs
    const float eps   = ggml_get_op_params_f32(rms[0], 0);
    const float scale = ggml_get_op_params_f32(scl[0], 0);
    const float bias  = ggml_get_op_params_f32(scl[0], 1);
    if (ggml_get_op_params_f32(rms[1], 0) != eps || ggml_get_op_params_f32(scl[1], 0) != scale ||
        ggml_get_op_params_f32(scl[1], 1) != bias) {
        return false;
    }

    // every other reader of silu must stay above the q/k channels
    const size_t  row    = silu->nb[1];
    const int32_t n_uses = ggml_hexagon_tensor_use_count(graph, silu);
    int32_t       n_seen = 0;
    for (int j = i_silu + 1; j < graph->n_nodes; j++) {
        const ggml_tensor * n = graph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (n->src[s] != silu) {
                continue;
            }
            n_seen++;
            if (n == view[0] || n == view[1]) {
                continue;
            }
            if (n->op != GGML_OP_VIEW || n->view_src != silu ||
                n->view_offs % row < (size_t) (n_q + n_k) * sizeof(float)) {
                return false;
            }
            // dims below the row stride stay inside one row, dims at or above it move by whole rows
            size_t in_row = n->view_offs % row + sizeof(float);
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                if (n->ne[d] > 1 && n->nb[d] < row) {
                    in_row += (n->ne[d] - 1) * n->nb[d];
                } else if (n->ne[d] > 1 && n->nb[d] % row != 0) {
                    return false;
                }
            }
            if (in_row > row) {
                return false;
            }
        }
    }
    if (n_seen != n_uses) {
        return false;
    }

    qk.idx[0]   = idx[2 * q];
    qk.idx[1]   = idx[2 * q + 1];
    qk.idx[2]   = idx[2 * k];
    qk.idx[3]   = idx[2 * k + 1];
    qk.out[0]   = scl[q];
    qk.out[1]   = scl[k];
    qk.head_dim = hd;
    qk.n_qk     = n_q + n_k;
    qk.eps      = eps;
    qk.scale    = scale;
    qk.bias     = bias;
    return true;
}

// Index of the GATED_DELTA_NET that is the only reader of the q/k outputs, or -1.
static int ggml_hexagon_find_ssm_conv_chain_qk_gdn(const ggml_cgraph * graph,
                                                   const ggml_hexagon_ssm_conv_chain_qk & qk) {
    for (int p = 0; p < 2; p++) {
        if ((qk.out[p]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_hexagon_tensor_use_count(graph, qk.out[p]) != 1) {
            return -1;
        }
    }
    for (int j = *std::max_element(qk.idx, qk.idx + 4) + 1; j < graph->n_nodes; j++) {
        const ggml_tensor * n = graph->nodes[j];
        if (n->op == GGML_OP_GATED_DELTA_NET && n->src[0] == qk.out[0] && n->src[1] == qk.out[1]) {
            return j;
        }
    }
    return -1;
}

// q or k in its own channels of silu, laid out like the q/k output
static ggml_tensor ggml_hexagon_ssm_conv_chain_qk_view(const ggml_tensor * silu, const ggml_tensor * like,
                                                       size_t offs) {
    ggml_tensor v = *silu;
    v.op        = GGML_OP_VIEW;
    v.flags     = 0;
    v.view_src  = const_cast<ggml_tensor *>(silu);
    v.view_offs = offs;
    v.data      = (char *) silu->data + offs;
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        v.ne[d] = like->ne[d];
    }
    v.nb[1] = like->ne[0] * sizeof(float);
    v.nb[2] = silu->nb[1];
    v.nb[3] = silu->nb[2];
    memset(v.src, 0, sizeof(v.src));
    snprintf(v.name, sizeof(v.name), "%s (in silu)", like->name);
    return v;
}

// Emits SSM_CONV_CHAIN once a CONCAT / state CPY / SSM_CONV / SILU match is found.
// idx = {state CPY, SSM_CONV, SILU} graph indices, concat_idx = the CONCAT. The fused op is placed at node i (the CONCAT, or the state CPY in the LLAMA_P4K_CONV_FUSE graph) and absorbs the other three.
static bool ggml_hexagon_ssm_conv_chain_emit(const ggml_hexagon_session * sess, const ggml_cgraph * graph, int i,
                                             int concat_idx, const int (&idx)[3], const std::vector<bool> & consumed,
                                             std::vector<int> & absorbed, htp_opnode & out,
                                             ggml_hexagon_tensor_map & qk_redirect) {
    ggml_tensor * concat      = graph->nodes[concat_idx];
    ggml_tensor * cpy         = graph->nodes[idx[0]];
    ggml_tensor * conv        = graph->nodes[idx[1]];
    ggml_tensor * silu        = graph->nodes[idx[2]];
    ggml_tensor * conv_states = concat->src[0];
    ggml_tensor * qkv_t       = concat->src[1];
    ggml_tensor * conv1d      = conv->src[1];

    const int64_t n_state         = conv_states->ne[0];
    const int64_t d_inner         = conv_states->ne[1];
    const int64_t n_s             = conv_states->ne[2];
    const int64_t n_t             = qkv_t->ne[0];
    const int64_t d_conv          = conv1d->ne[0];
    const size_t  st_in_seq_stride = (size_t) n_state * d_inner * sizeof(float);

    const size_t qkv_seq_stride    = n_s > 1 ? qkv_t->nb[2] : 0;
    const size_t dst_seq_stride    = n_s > 1 ? silu->nb[2]  : 0;
    const size_t st_out_seq_stride = n_s > 1 ? cpy->nb[1]   : 0;
    if (qkv_seq_stride > UINT32_MAX || dst_seq_stride > UINT32_MAX || st_out_seq_stride > UINT32_MAX ||
        st_in_seq_stride > UINT32_MAX) {
        return false;
    }

    // An output on top of qkv is safe only as a backward whole-row shift (row t on qkv row t - k, already read).
    // An overlap with conv_states is safe: the kernel stages all of it before any output DMA.
    for (const ggml_tensor * t : { (const ggml_tensor *) silu, (const ggml_tensor *) cpy }) {
        ggml_hexagon_mem_ranges mr;
        mr.add(ggml_hexagon_mem_range_from_tensor(qkv_t, HEXAGON_MEM_RANGE_TYPE_SRC));
        if (mr.check(ggml_hexagon_mem_range_from_tensor(t, HEXAGON_MEM_RANGE_TYPE_DST))) {
            continue;
        }
        const size_t    row   = silu->nb[1];
        const ptrdiff_t delta = (const char *) t->data - (const char *) qkv_t->data;
        if (t != silu || row != qkv_t->nb[0] || (n_s > 1 && silu->nb[2] != qkv_t->nb[2]) ||
            delta > 0 || (size_t) (-delta) % row != 0) {
            HEX_VERBOSE("ggml-hex: %s skip SSM_CONV_CHAIN: %s overlaps qkv\n", sess->c_name(), t->name);
            return false;
        }
    }

    // q/k epilogue (switch value 2): if a GATED_DELTA_NET of this graph is the only q/k reader, q/k go to their silu
    // channels and it reads them there; else q/k must not overlap anything the chain reads or writes, except conv_states.
    ggml_hexagon_ssm_conv_chain_qk qk;
    bool fuse_qk = opt_ssm_conv_chain >= 2 && ggml_hexagon_match_ssm_conv_chain_qk(graph, idx[2], consumed, silu, qk);
    const bool qk_in_silu = fuse_qk && ggml_hexagon_find_ssm_conv_chain_qk_gdn(graph, qk) >= 0;
    ggml_tensor qk_view[2];
    if (qk_in_silu) {
        qk_view[0] = ggml_hexagon_ssm_conv_chain_qk_view(silu, qk.out[0], 0);
        const size_t k_offs = qk.out[0]->ne[0] * qk.out[0]->ne[1] * sizeof(float);
        qk_view[1] = ggml_hexagon_ssm_conv_chain_qk_view(silu, qk.out[1], k_offs);
    }
    if (fuse_qk && !qk_in_silu) {
        const ggml_tensor * others[] = { qkv_t, conv1d, silu, cpy, qk.out[0] };
        for (int p = 0; p < 2 && fuse_qk; p++) {
            for (const ggml_tensor * o : others) {
                if (o == qk.out[p] || (p == 0 && o == qk.out[0])) {
                    continue;
                }
                ggml_hexagon_mem_ranges mr;
                mr.add(ggml_hexagon_mem_range_from_tensor(o, HEXAGON_MEM_RANGE_TYPE_DST));
                if (!mr.check(ggml_hexagon_mem_range_from_tensor(qk.out[p], HEXAGON_MEM_RANGE_TYPE_DST))) {
                    HEX_VERBOSE("ggml-hex: %s skip SSM_CONV_CHAIN q/k l2-norm: %s overlaps %s\n", sess->c_name(),
                                qk.out[p]->name, o->name);
                    fuse_qk = false;
                    break;
                }
            }
        }
    }

    // The absorbed ops run earlier than in the graph: the ops in between must not write conv1d or read or
    // write any output. conv_input, the raw conv output and the norms are not touched by the fused op.
    auto window_ok = [&](bool with_qk) {
        std::vector<int> abs_idx(idx, idx + 3);
        abs_idx.push_back(concat_idx);
        ggml_hexagon_mem_ranges mrs;
        mrs.add(ggml_hexagon_mem_range_from_tensor(conv1d, HEXAGON_MEM_RANGE_TYPE_SRC));
        mrs.add(ggml_hexagon_mem_range_from_tensor(silu,   HEXAGON_MEM_RANGE_TYPE_DST));
        mrs.add(ggml_hexagon_mem_range_from_tensor(cpy,    HEXAGON_MEM_RANGE_TYPE_DST));
        if (with_qk) {
            abs_idx.insert(abs_idx.end(), qk.idx, qk.idx + 4);
            if (!qk_in_silu) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(qk.out[0], HEXAGON_MEM_RANGE_TYPE_DST));
                mrs.add(ggml_hexagon_mem_range_from_tensor(qk.out[1], HEXAGON_MEM_RANGE_TYPE_DST));
            }
        }
        const int last = *std::max_element(abs_idx.begin(), abs_idx.end());
        for (int j = i + 1; j < last; j++) {
            if (std::find(abs_idx.begin(), abs_idx.end(), j) != abs_idx.end() || !op_is_compute(graph->nodes[j])) {
                continue;
            }
            if (!ggml_hexagon_mem_ranges_check_node(mrs, htp_opnode(HTP_OP_INVALID, graph->nodes[j]))) {
                HEX_VERBOSE("ggml-hex: %s skip SSM_CONV_CHAIN%s: %s aliases the chain\n", sess->c_name(),
                            with_qk ? " q/k l2-norm" : "", graph->nodes[j]->name);
                return false;
            }
        }
        absorbed.clear();
        for (int a : abs_idx) {
            if (a != i) {
                absorbed.push_back(a);
            }
        }
        return true;
    };
    if (fuse_qk && !window_ok(true)) {
        fuse_qk = false;
    }
    if (!fuse_qk && !window_ok(false)) {
        return false;
    }

    // with the epilogue try all tokens at once, then token blocks, then drop the epilogue
    struct htp_ssm_conv_chain_kernel_params kparams;
    bool kp_ok = false;
    if (fuse_qk) {
        // GGML_HEXAGON_SSM_CONV_CHAIN_TB also applies to the q/k epilogue: token blocks overlap the DMA of one tile with the compute of its neighbour
        const uint32_t qk_tb = (opt_ssm_conv_chain_tb > 0 && (int64_t) opt_ssm_conv_chain_tb < (int64_t) n_t) ? opt_ssm_conv_chain_tb : 0;
        kp_ok = ggml_hexagon_precompute_ssm_conv_chain_params(sess, (uint32_t) d_conv, (uint32_t) d_inner,
                    (uint32_t) n_t, (uint32_t) n_s, (uint32_t) st_in_seq_stride, (uint32_t) qkv_seq_stride,
                    (uint32_t) dst_seq_stride, (uint32_t) st_out_seq_stride, (uint32_t) qk.head_dim, (uint32_t) qk.n_qk, qk_tb, &kparams);
        for (uint32_t n_blk = 2; !kp_ok && n_blk <= 64; n_blk++) {
            const uint32_t n_tb = ((uint32_t) n_t + n_blk - 1) / n_blk;
            if (n_tb < 16) {
                break;
            }
            kp_ok = ggml_hexagon_precompute_ssm_conv_chain_params(sess, (uint32_t) d_conv, (uint32_t) d_inner,
                        (uint32_t) n_t, (uint32_t) n_s, (uint32_t) st_in_seq_stride, (uint32_t) qkv_seq_stride,
                        (uint32_t) dst_seq_stride, (uint32_t) st_out_seq_stride, (uint32_t) qk.head_dim, (uint32_t) qk.n_qk, n_tb, &kparams);
        }
        if (kp_ok) {
            kparams.qk_eps   = qk.eps;
            kparams.qk_scale = qk.scale;
            kparams.qk_bias  = qk.bias;
        } else {
            HEX_VERBOSE("ggml-hex: %s skip SSM_CONV_CHAIN q/k l2-norm: no VTCM layout (n_t %d n_s %d)\n",
                        sess->c_name(), (int) n_t, (int) n_s);
            fuse_qk = false;
            if (!window_ok(false)) {
                return false;
            }
        }
    }
    if (!kp_ok && opt_ssm_conv_chain_tb > 0 && (int64_t) opt_ssm_conv_chain_tb < (int64_t) n_t) {
        // GGML_HEXAGON_SSM_CONV_CHAIN_TB (default 0): token blocks for the non-QK chain, so each thread runs several smaller
        // DMA tiles and the input and output DMA of one tile overlaps the compute of its neighbour
        kp_ok = ggml_hexagon_precompute_ssm_conv_chain_params(sess, (uint32_t) d_conv, (uint32_t) d_inner, (uint32_t) n_t,
                    (uint32_t) n_s, (uint32_t) st_in_seq_stride, (uint32_t) qkv_seq_stride, (uint32_t) dst_seq_stride,
                    (uint32_t) st_out_seq_stride, 0, 0, opt_ssm_conv_chain_tb, &kparams);
    }
    if (!kp_ok && !ggml_hexagon_precompute_ssm_conv_chain_params(sess, (uint32_t) d_conv, (uint32_t) d_inner, (uint32_t) n_t,
            (uint32_t) n_s, (uint32_t) st_in_seq_stride, (uint32_t) qkv_seq_stride, (uint32_t) dst_seq_stride,
            (uint32_t) st_out_seq_stride, 0, 0, 0, &kparams)) {
        HEX_VERBOSE("ggml-hex: %s skip SSM_CONV_CHAIN: no VTCM layout (d_inner %d n_t %d n_s %d)\n",
                    sess->c_name(), (int) d_inner, (int) n_t, (int) n_s);
        return false;
    }

    out = htp_opnode(HTP_OP_SSM_CONV_CHAIN, graph->nodes[i]);
    out.name    = "SSM_CONV_CHAIN";
    out.fused   = (i == concat_idx) ? std::vector<ggml_tensor *> { cpy, conv, silu }
                                    : std::vector<ggml_tensor *> { concat, conv, silu };
    out.inputs  = { conv_states, qkv_t, conv1d };
    out.outputs = { silu, cpy };
    if (fuse_qk) {
        out.name = "SSM_CONV_CHAIN+QK_NORM";
        for (int p = 0; p < 4; p++) {
            out.fused.push_back(graph->nodes[qk.idx[p]]);
        }
        for (int p = 0; p < 2; p++) {
            if (qk_in_silu) {
                const ggml_tensor * v = out.add_dummy(qk_view[p]);
                out.outputs.push_back(v);
                qk_redirect[qk.out[p]] = v;
            } else {
                out.outputs.push_back(qk.out[p]);
            }
        }
    }
    kparams.silu_variant = (uint32_t) (opt_c2_silu == 3 && opt_sig16_v2 ? 4 : opt_c2_silu);
    if (const char * v = getenv("GGML_HEXAGON_CONV_ABL")) kparams.silu_variant |= (uint32_t) atoi(v) << 8;
    memcpy(out.kernel_params, &kparams, sizeof(kparams));

    return true;
}
// Returns true and the graph indices of the absorbed nodes if nodes[i] starts a fusable chain.
// With the q/k epilogue feeding a GATED_DELTA_NET, qk_redirect maps the q/k outputs to where the fused op writes them.
static bool ggml_hexagon_try_fuse_ssm_conv_chain(const ggml_hexagon_session * sess, const ggml_cgraph * graph, int i,
                                                 const std::vector<bool> & consumed, std::vector<int> & absorbed,
                                                 htp_opnode & out, ggml_hexagon_tensor_map & qk_redirect) {
    if (sess->mdev.count > 1) {
        return false;
    }

    ggml_tensor * concat = graph->nodes[i];
    if (concat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(concat, 0) != 0) {
        return false;
    }
    // prefill only: a CONCAT below SSM_CONV_CHAIN_MIN_ROWS tokens keeps the unfused path (checked before any look-ahead)
    if (!concat->src[1] || concat->src[1]->ne[0] < SSM_CONV_CHAIN_MIN_ROWS) {
        return false;
    }

    // look a few compute nodes ahead for the CPY, the SSM_CONV and the SILU
    constexpr int N_LOOKAHEAD = 8;

    int idx[3] = { -1, -1, -1 };
    for (int j = i + 1, k = 0; j < graph->n_nodes && k < N_LOOKAHEAD && idx[2] < 0; j++) {
        ggml_tensor * n = graph->nodes[j];
        if (!op_is_compute(n)) {
            continue;
        }
        k++;
        if (consumed[j]) {
            return false;
        }
        if (idx[0] < 0 && n->op == GGML_OP_CPY && n->src[0] && n->src[0]->view_src == concat) {
            idx[0] = j;
        } else if (idx[1] < 0 && n->op == GGML_OP_SSM_CONV && n->src[0] == concat) {
            idx[1] = j;
        } else if (idx[1] >= 0 && n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == GGML_UNARY_OP_SILU &&
                   n->src[0] == graph->nodes[idx[1]]) {
            idx[2] = j;
        }
    }
    if (idx[0] < 0 || idx[1] < 0 || idx[2] < 0) {
        return false;
    }

    ggml_tensor * cpy  = graph->nodes[idx[0]];
    ggml_tensor * conv = graph->nodes[idx[1]];
    ggml_tensor * silu = graph->nodes[idx[2]];

    ggml_tensor * conv_states = concat->src[0];
    ggml_tensor * qkv_t       = concat->src[1];
    ggml_tensor * conv1d      = conv->src[1];

    for (const ggml_tensor * t : { concat, conv_states, qkv_t, cpy, conv, conv1d, silu }) {
        if (!ggml_hexagon_ssm_conv_chain_tensor_ok(t)) {
            return false;
        }
    }

    const int64_t n_state = conv_states->ne[0];
    const int64_t d_inner = conv_states->ne[1];
    const int64_t n_s     = conv_states->ne[2];
    const int64_t n_t     = qkv_t->ne[0];
    const int64_t ncs     = n_state + n_t;
    const int64_t d_conv  = conv1d->ne[0];

    if (n_state < 1 || n_state > 31 || d_inner < 32 || n_t < 1 || n_s < 1 || d_conv != n_state + 1) {
        return false;
    }
    if (conv_states->ne[3] != 1 || qkv_t->ne[1] != d_inner || qkv_t->ne[2] != n_s || qkv_t->ne[3] != 1 ||
        concat->ne[0] != ncs || concat->ne[1] != d_inner || concat->ne[2] != n_s || concat->ne[3] != 1) {
        return false;
    }

    // conv_states: contiguous channel-major rows; qkv_t: transposed view (channels contiguous)
    const size_t st_in_seq_stride = (size_t) n_state * d_inner * sizeof(float);
    if (conv_states->nb[0] != sizeof(float) || conv_states->nb[1] != (size_t) n_state * sizeof(float) ||
        (n_s > 1 && conv_states->nb[2] != st_in_seq_stride)) {
        return false;
    }
    if (qkv_t->nb[1] != sizeof(float) || qkv_t->nb[0] < (size_t) d_inner * sizeof(float)) {
        return false;
    }
    if (concat->nb[1] != (size_t) ncs * sizeof(float) || concat->nb[2] != (size_t) ncs * d_inner * sizeof(float)) {
        return false;
    }

    // conv_input feeds only the state view and the SSM_CONV (this also declines the n_rs_seq > 0 rollback form)
    if ((concat->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_hexagon_tensor_use_count(graph, concat) != 2) {
        return false;
    }

    const ggml_tensor * state_view = cpy->src[0];
    if (state_view->op != GGML_OP_VIEW || state_view->view_offs != (size_t) n_t * sizeof(float) ||
        state_view->ne[0] != n_state || state_view->ne[1] != d_inner || state_view->ne[2] != n_s ||
        state_view->ne[3] != 1 || state_view->nb[0] != sizeof(float) || state_view->nb[1] != concat->nb[1] ||
        (n_s > 1 && state_view->nb[2] != concat->nb[2]) ||
        ggml_hexagon_tensor_use_count(graph, state_view) != 1) {
        return false;
    }
    if (cpy->ne[0] != n_state * d_inner || cpy->ne[1] != n_s || cpy->ne[2] != 1 || cpy->ne[3] != 1 ||
        cpy->nb[0] != sizeof(float) || (n_s > 1 && cpy->nb[1] != st_in_seq_stride)) {
        return false;
    }

    if (conv1d->ne[1] != d_inner || conv1d->ne[2] != 1 || conv1d->ne[3] != 1 ||
        conv1d->nb[0] != sizeof(float) || conv1d->nb[1] != (size_t) d_conv * sizeof(float)) {
        return false;
    }
    if (conv->ne[0] != d_inner || conv->ne[1] != n_t || conv->ne[2] != n_s || conv->ne[3] != 1 ||
        (conv->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_hexagon_tensor_use_count(graph, conv) != 1) {
        return false;
    }
    if (silu->ne[0] != d_inner || silu->ne[1] != n_t || silu->ne[2] != n_s || silu->ne[3] != 1 ||
        silu->nb[0] != sizeof(float) || silu->nb[1] < (size_t) d_inner * sizeof(float)) {
        return false;
    }

    return ggml_hexagon_ssm_conv_chain_emit(sess, graph, i, i, idx, consumed, absorbed, out, qk_redirect);
}


// LLAMA_P4K_CONV_FUSE graph (build_conv_state, conv_tail_from_qkv): the state CPY reads the last n_state tokens of qkv_mixed through a TRANSPOSE view and is built before the CONCAT, so the chain is anchored at the CPY (the fused op takes its place) and absorbs the CONCAT, SSM_CONV and SILU.
// The state it writes is the same qkv tail the CONCAT-form chain writes.
static bool ggml_hexagon_try_fuse_ssm_conv_chain_p4k(const ggml_hexagon_session * sess, const ggml_cgraph * graph, int i,
                                                     const std::vector<bool> & consumed, std::vector<int> & absorbed,
                                                     htp_opnode & out, ggml_hexagon_tensor_map & qk_redirect) {
    if (sess->mdev.count > 1) {
        return false;
    }

    ggml_tensor * cpy = graph->nodes[i];
    if (cpy->op != GGML_OP_CPY || !cpy->src[0] || cpy->src[0]->op != GGML_OP_TRANSPOSE) {
        return false;
    }
    const ggml_tensor * tail = cpy->src[0]->src[0];
    if (!tail || tail->op != GGML_OP_VIEW || !tail->src[0]) {
        return false;
    }
    const ggml_tensor * qkv_mixed_orig = tail->src[0];
    const int64_t       n_t            = qkv_mixed_orig->ne[1];
    if (n_t < SSM_CONV_CHAIN_MIN_ROWS) {
        return false;
    }

    // a few compute nodes ahead: the CONCAT of this qkv tail, its SSM_CONV and the SILU
    constexpr int N_LOOKAHEAD = 12;

    int concat_idx = -1;
    int idx[3]     = { i, -1, -1 };
    for (int j = i + 1, k = 0; j < graph->n_nodes && k < N_LOOKAHEAD && idx[2] < 0; j++) {
        ggml_tensor * n = graph->nodes[j];
        if (!op_is_compute(n)) {
            continue;
        }
        k++;
        if (consumed[j]) {
            return false;
        }
        if (concat_idx < 0 && n->op == GGML_OP_CONCAT && n->src[1] && n->src[1]->op == GGML_OP_TRANSPOSE &&
            n->src[1]->src[0] == qkv_mixed_orig) {
            concat_idx = j;
        } else if (concat_idx >= 0 && idx[1] < 0 && n->op == GGML_OP_SSM_CONV && n->src[0] == graph->nodes[concat_idx]) {
            idx[1] = j;
        } else if (idx[1] >= 0 && n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == GGML_UNARY_OP_SILU &&
                   n->src[0] == graph->nodes[idx[1]]) {
            idx[2] = j;
        }
    }
    if (concat_idx < 0 || idx[1] < 0 || idx[2] < 0) {
        return false;
    }

    ggml_tensor * concat      = graph->nodes[concat_idx];
    ggml_tensor * conv        = graph->nodes[idx[1]];
    ggml_tensor * silu        = graph->nodes[idx[2]];
    ggml_tensor * conv_states = concat->src[0];
    ggml_tensor * qkv_t       = concat->src[1];
    ggml_tensor * conv1d      = conv->src[1];

    for (const ggml_tensor * t : { concat, conv_states, qkv_t, cpy, conv, conv1d, silu }) {
        if (!ggml_hexagon_ssm_conv_chain_tensor_ok(t)) {
            return false;
        }
    }

    const int64_t n_state = conv_states->ne[0];
    const int64_t d_inner = conv_states->ne[1];
    const int64_t n_s     = conv_states->ne[2];
    const int64_t ncs     = n_state + n_t;
    const int64_t d_conv  = conv1d->ne[0];

    if (n_state < 1 || n_state > 31 || d_inner < 32 || n_s < 1 || d_conv != n_state + 1) {
        return false;
    }
    if (conv_states->ne[3] != 1 || qkv_t->ne[0] != n_t || qkv_t->ne[1] != d_inner || qkv_t->ne[2] != n_s ||
        qkv_t->ne[3] != 1 || concat->ne[0] != ncs || concat->ne[1] != d_inner || concat->ne[2] != n_s ||
        concat->ne[3] != 1) {
        return false;
    }

    const size_t st_in_seq_stride = (size_t) n_state * d_inner * sizeof(float);
    if (conv_states->nb[0] != sizeof(float) || conv_states->nb[1] != (size_t) n_state * sizeof(float) ||
        (n_s > 1 && conv_states->nb[2] != st_in_seq_stride)) {
        return false;
    }
    if (qkv_t->src[0] != qkv_mixed_orig || qkv_t->nb[1] != sizeof(float) ||
        qkv_t->nb[0] < (size_t) d_inner * sizeof(float)) {
        return false;
    }
    if (concat->nb[1] != (size_t) ncs * sizeof(float) || concat->nb[2] != (size_t) ncs * d_inner * sizeof(float)) {
        return false;
    }
    if (qkv_mixed_orig->nb[0] != sizeof(float)) {
        return false;
    }

    // the state tail: the last n_state tokens of qkv_mixed, channel-major rows of the transposed view
    if (tail->ne[0] != d_inner || tail->ne[1] != n_state || tail->ne[2] != n_s || tail->nb[1] != qkv_mixed_orig->nb[1] ||
        tail->view_offs != (size_t) (n_t - n_state) * qkv_mixed_orig->nb[1] || cpy->src[0]->ne[0] != n_state ||
        cpy->src[0]->ne[1] != d_inner) {
        return false;
    }

    // conv_input feeds only the SSM_CONV (this also declines the n_rs_seq > 0 rollback form)
    if ((concat->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_hexagon_tensor_use_count(graph, concat) != 1) {
        return false;
    }

    if (cpy->ne[0] != n_state * d_inner || cpy->ne[1] != n_s || cpy->ne[2] != 1 || cpy->ne[3] != 1 ||
        cpy->nb[0] != sizeof(float) || (n_s > 1 && cpy->nb[1] != st_in_seq_stride)) {
        return false;
    }
    if (conv1d->ne[1] != d_inner || conv1d->ne[2] != 1 || conv1d->ne[3] != 1 || conv1d->nb[0] != sizeof(float) ||
        conv1d->nb[1] != (size_t) d_conv * sizeof(float)) {
        return false;
    }
    if (conv->ne[0] != d_inner || conv->ne[1] != n_t || conv->ne[2] != n_s || conv->ne[3] != 1 ||
        (conv->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_hexagon_tensor_use_count(graph, conv) != 1) {
        return false;
    }
    if (silu->ne[0] != d_inner || silu->ne[1] != n_t || silu->ne[2] != n_s || silu->ne[3] != 1 ||
        silu->nb[0] != sizeof(float) || silu->nb[1] < (size_t) d_inner * sizeof(float)) {
        return false;
    }

    return ggml_hexagon_ssm_conv_chain_emit(sess, graph, i, concat_idx, idx, consumed, absorbed, out, qk_redirect);
}

static std::vector<int> ggml_hexagon_graph_optimize_reorder(const std::vector<htp_opnode> & nodes) {
    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        if (!node0.stackable()) {
            res.push_back(i0);
            used[i0] = true;
            continue;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr int N_FORWARD = 16;

        std::vector<int> stack;
        stack.push_back(i0);

        mrs.reset();

        for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            if (used[i1]) {
                continue;
            }

            const auto & node1 = nodes[i1];

            if (node1.stackable() && node1.same_input(node0) && ggml_hexagon_mem_ranges_check_node(mrs, node1)) {
                stack.push_back(i1);
            } else {
                ggml_hexagon_mem_ranges_add_node(mrs, node1);
            }
        }

        for (int idx : stack) {
            res.push_back(idx);
            used[idx] = true;
        }
    }

    return res;
}

static void ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    const int n = gf->n_nodes;

    constexpr int MAX_FUSE = 16;

    enum ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    for (int i = 0; i < n; i++) {
        htp_opnode node(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        if (node.op() == GGML_OP_ADD ||
            node.op() == GGML_OP_NORM ||
            node.op() == GGML_OP_RMS_NORM) {
            ops[0] = node.op();

            int f = i + 1;
            while (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                if (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    break;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            for (; f > 1; f--) {
                if (ggml_can_fuse(gf, i, ops, f)) {
                    break;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            for (int k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                node.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(node));
    }

    const auto order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

static uint64_t ggml_hexagon_session_key(const ggml_hexagon_session * sess) {
    return ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

static bool ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    auto sbuf_dst = sess_dst->mmap_tensor(dst);

    if (!sess_src->clone_buffer(sbuf_dst)) { return false; }

    const uint64_t src_key = ggml_hexagon_session_key(sess_src);
    auto & fence_slot = sess_dst->cpy_fence_slots[src_key];
    if (!fence_slot) {
        fence_slot = (volatile uint32_t *) sess_dst->alloc_fence(1);
    }

    if (!sess_src->clone_buffer(sess_dst->fence_buf)) { return false; }

    if (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    static ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.buffer = &sess_dst->fence_buf->backend_buffer;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.data   = (void *) fence_slot;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = sizeof(int32_t);
    fence_tensor.nb[1]  = sizeof(int32_t);
    fence_tensor.nb[2]  = sizeof(int32_t);
    fence_tensor.nb[3]  = sizeof(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ true);

    sess_dst->add_peer(sess_src);

    return true;
}

static bool ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    auto sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    if (!sess_dst->clone_buffer(sbuf_src)) { return false; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    return true;
}

static bool ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        return false;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    if (!dst->extra) {
        ggml_backend_buffer_init_tensor(dst->buffer, dst);
    }

    auto * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    const auto * src_extra = static_cast<const ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    if (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        return true;
    }

    if (sess_src->phys_idx != sess_dst->phys_idx)
        return ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    return ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

static ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = new ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_slot = (volatile uint32_t *) sess->alloc_fence(1);

    static ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.buffer = &sess->fence_buf->backend_buffer;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.data   = (void *) hex_event->fence_slot;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = sizeof(int32_t);
    hex_event->fence_tensor.nb[1]  = sizeof(int32_t);
    hex_event->fence_tensor.nb[2]  = sizeof(int32_t);
    hex_event->fence_tensor.nb[3]  = sizeof(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (void *)hex_event, (void *)hex_event->fence_slot);

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

static void ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    if (hex_event->seq == 0) {
        return;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (void *)hex_event, hex_event->seq, (void *)hex_event->fence_slot);

    auto * fence = reinterpret_cast<const volatile std::atomic<uint32_t> *>(hex_event->fence_slot);

    if ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    while (true) {
        if ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            if (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            break;
        }
        std::this_thread::yield();
    }
}

static void ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (void *)hex_event);
    hex_event->fence_sess->free_fence((void *) hex_event->fence_slot, 1);
    delete hex_event;
    delete event;
}

static void ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

static void ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    if (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ false);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (void *)hex_event, hex_event->seq, (void *)hex_event->fence_slot);
}

static void ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    if (hex_event->seq == 0) {
        return;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (void *)hex_event, hex_event->seq, (void *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    if (sess->phys_idx == hex_event->sess->phys_idx) {
        if (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        return;
    }

    sess->clone_buffer(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ true);
}

static void ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (int) tensor->buffer->usage : -1);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

static void ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (int) tensor->buffer->usage : -1);
    sess->flush_sync();
    if (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

static void ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     struct ggml_tensor * tensor,
                                                     const void * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (int) tensor->buffer->usage : -1);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

static void ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     const struct ggml_tensor * tensor,
                                                     void * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (int) tensor->buffer->usage : -1);
    sess->flush_sync();
    if (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

static struct ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_record,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

static ggml_guid_t ggml_backend_hexagon_guid() {
    static ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    return &guid;
}

bool ggml_backend_is_hexagon(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

static ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, const char * params) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    return new ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

static const char * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    return dev_ctx->c_name();

    GGML_UNUSED(dev);
}

static const char * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    return "Hexagon";
    GGML_UNUSED(dev);
}

static void ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = *free;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    return &dev_ctx->buffer_type;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    if (!opt_hostbuf) {
        return NULL;
    }
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    return &dev_ctx->host_buffer_type;
}

static bool ggml_hexagon_supported_cpy(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
        src0->type != GGML_TYPE_I32) return false;
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 &&
        dst->type != GGML_TYPE_I32) return false;

    const bool is_scalar  = (ggml_nelements(src0) == 1 && ggml_nelements(dst) == 1);
    const bool sametype   = (src0->type == dst->type);
    const bool transposed = !is_scalar && (ggml_is_transposed(src0) || ggml_is_transposed(dst));
    const bool sameshape  = is_scalar || (!transposed && ggml_are_same_shape(src0, dst));

    if (src0->type == GGML_TYPE_I32 || dst->type == GGML_TYPE_I32) {
        if (!sameshape) return false;
        if (sametype) return true;
        if ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32)) {
            return true;
        }
        return false;
    }

    // can handle any shape and any same-type (pretty slow if reshaping is required)
    if (sametype) return true;

    // cannot handle re-shaping and type conversion at the same time
    if (!sameshape) return false;

    return true;
}

static bool ggml_hexagon_supported_cont(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);
    const struct ggml_tensor * src0 = op->src[0];

    // CONT is same-type only and supports F32, F16, and I32.
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
        src0->type != GGML_TYPE_I32) return false;

    return true;
}

static bool ggml_hexagon_supported_repeat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    // Support f32 and f16
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;

    // src and dst must be the same type
    if (src0->type != dst->type) return false;

    // dst dims must be multiples of src dims
    if (dst->ne[0] % src0->ne[0] != 0) return false;
    if (dst->ne[1] % src0->ne[1] != 0) return false;
    if (dst->ne[2] % src0->ne[2] != 0) return false;
    if (dst->ne[3] % src0->ne[3] != 0) return false;

    // require contiguous tensors (no transposition)
    if (ggml_is_transposed(src0) || ggml_is_transposed(dst)) return false;

    return true;
}

static bool ggml_hexagon_supported_concat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    int dim = ((const int32_t *) op->op_params)[0];
    if (dim < 0 || dim >= GGML_MAX_DIMS) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const struct ggml_tensor * src = op->src[i];
        if (!src) {
            continue;
        }
        if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_I32 && src->type != GGML_TYPE_F16) {
            return false;
        }
    }

    if (op->src[2]) {
        // conv ring offsets: F32 dim-0 concat with transposed src1 and plain src0 (htp concat transposed path only)
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        if (dim != 0 || op->src[2]->type != GGML_TYPE_I32 || op->type != GGML_TYPE_F32) {
            return false;
        }
        if (!(src1->nb[0] > src1->nb[1]) || src0->nb[0] > src0->nb[1] ||
            src0->nb[0] != sizeof(float) || src1->nb[1] != sizeof(float) || op->nb[0] != sizeof(float)) {
            return false;
        }
    }
    return true;
    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_fill(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * dst = op;

    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return false;
    }

    return true;
    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_roll(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }

    if (src0->nb[0] != ggml_type_size(src0->type) || dst->nb[0] != ggml_type_size(dst->type)) {
        return false;
    }

    if (!ggml_is_contiguous(dst)) {
        return false;
    }

    return true;
}

static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    // reject ops that match the filter
    if (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        return false;
    }

    bool supp = false;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            supp = true;
            break;

        case GGML_OP_MUL:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
            supp = ggml_hexagon_supported_binary(sess, op);
            break;

        case GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            break;

        case GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            break;

        case GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            break;

        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            break;

        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            break;

        case GGML_OP_SUM:
            supp = ggml_hexagon_supported_sum(sess, op);
            break;

        case GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            break;

        case GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            break;

        case GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            break;

        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_STEP:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    break;
                default:
                    supp = false;
                    break;
            }
            break;

        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_GEGLU_QUICK:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    break;
                default:
                    supp = false;
                    break;
            }
            break;

        case GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            break;

        case GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            break;

        case GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            break;

        case GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            break;

        case GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            break;

        case GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            break;

        case GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            break;

        case GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            break;

        case GGML_OP_TOP_K:
            supp = ggml_hexagon_supported_top_k(sess, op);
            break;

        case GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            break;

        case GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            break;

        case GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            break;

        case GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            break;

        case GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            break;

        case GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            break;

        case GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            break;

        case GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            break;

        case GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            break;

        case GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            break;

        case GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            break;

        default:
            break;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    return supp;
}

static bool ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    bool supp = (buft == &dev_ctx->host_buffer_type) || (buft == &dev_ctx->buffer_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    return supp;
}

static const struct ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ NULL,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    for (size_t i = 0; i < opt_ndev; i++) {
        const auto & cfg = opt_device_configs[i];
        if (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } else {
            std::string peers_str;
            for (const auto & p : cfg.mdev_group) {
                if (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = new ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    for (size_t i = 0; i < opt_ndev; i++) {
        auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        delete dev_ctx;
    }
}

static const char * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    return "HTP";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_hexagon_reg_get_device_count(ggml_backend_reg_t reg) {
    return opt_ndev;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    if (index >= opt_ndev || !hreg->devices[index].context) {
        return nullptr;
    }

    return &hreg->devices[index];
}

// ** communication context for tensor-split allreduce

static void * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (n_backends < 2 || n_backends > 4) {
        return nullptr;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_hexagon(backends[i])) {
            return nullptr;
        }
    }

    for (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        for (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            if (sess_i->phys_idx == sess_j->phys_idx) {
                return nullptr;
            }
        }
    }

    auto * ctx = new ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    static ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    for (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (volatile uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].buffer = &sess_i->fence_buf->backend_buffer;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].data   = (void *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = sizeof(int32_t);
        ctx->fence_tensors[i].nb[1]  = sizeof(int32_t);
        ctx->fence_tensors[i].nb[2]  = sizeof(int32_t);
        ctx->fence_tensors[i].nb[3]  = sizeof(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    return ctx;
}

static void ggml_backend_hexagon_comm_free(void * comm_ctx_v) {
    if (!comm_ctx_v) return;
    auto * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    for (size_t i = 0; i < ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((void *) ctx->fence_slots[i], 1);
    }
    delete ctx;
}

// Fence sequence numbers of one comm operation: entry and exit are both newer than every session's last sequence, and every session
// adopts the exit value.
static void ggml_backend_hexagon_comm_fence_seqs(ggml_backend_hexagon_comm_context * comm_ctx, uint32_t & fence_seq_entry,
                                                 uint32_t & fence_seq_exit) {
    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    for (size_t i = 1; i < comm_ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        if ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    if (++max_seq == 0) max_seq = 1;
    fence_seq_entry = max_seq;
    if (++max_seq == 0) max_seq = 1;
    fence_seq_exit  = max_seq;

    for (size_t i = 0; i < comm_ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }
}

static bool ggml_backend_hexagon_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (opt_ar_select == 0 || !comm_ctx_v) return false;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    const size_t n_backends = comm_ctx->n_backends;

    if (n_backends < 2 || n_backends > 4) return false;

    for (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        for (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            if (sess_i->phys_idx == sess_j->phys_idx) {
                return false;
            }
        }
    }

    for (size_t i = 0; i < n_backends; i++) {
        if (!tensors[i] || !tensors[i]->buffer || !ggml_backend_buffer_is_hexagon(tensors[i]->buffer)) {
            return false;
        }
        if (tensors[i]->type != tensors[0]->type) {
            return false;
        }
        if (!ggml_is_contiguous(tensors[i])) {
            return false;
        }
        if (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            return false;
        }
    }

    if (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        return false;
    }

    for (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        struct htp_allreduce_kernel_params kparams;
        if (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, false, false, &kparams)) {
            return false;
        }
    }

    uint32_t fence_seq_entry = 0;
    uint32_t fence_seq_exit  = 0;
    ggml_backend_hexagon_comm_fence_seqs(comm_ctx, fence_seq_entry, fence_seq_exit);

    std::vector<const ggml_tensor *> data_tensors(n_backends);
    std::vector<const ggml_tensor *> sync_tensors(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        data_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    for (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], data_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        for (size_t j = 0; j < n_backends; j++) {
            if (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    return true;
}

// Diagnostics for the gap-fused path: the first rejections are logged with their reason (at most 16 per process).
static void ggml_backend_hexagon_ar_gap_reject(const char * why) {
    static int n_logged = 0;
    if (n_logged < 16) {
        n_logged++;
        GGML_LOG_INFO("ggml-hex: gap-fused ALLREDUCE+ADD not enqueued (%s); the comm falls back to ALLREDUCE\n", why);
    }
}

// RS-ROW owner-only writes (see ggml-backend-meta.cpp): this backend's dst gets its own rows only, so the rows must be the row-chunked
// partition of precompute_allreduce_params (which needs ne0 to be a multiple of the 128-element block) and the batch must be prefill-sized.
static bool ar_rs_row_owner_ok(const ggml_tensor * d) {
    const uint32_t ne1 = (uint32_t) (d->ne[1] * d->ne[2] * d->ne[3]);
    return opt_rs_row != 0 && ar_gap_chunked() && ar_rs_row_rows(ne1) && d->ne[0] % 128 == 0 && ggml_is_contiguous(d);
}

// Gap-fused residual add (opt-in via GGML_HEXAGON_AR_GAP_FUSE; see ggml-backend-meta.cpp). The op is enqueued outside plan capture, so plan replay stays valid. Mode 1 reduce-scatters: rank r reduces its chunk with the per-element order ((s0+s1)+s2)+res and writes it to all n dsts (MC01).
// Mode 2 reduces the full range into the local dst only. The ALLREDUCE+ADD of one tensor-parallel boundary is enqueued here, between two subgraphs, and the residual ADD node it consumes is skipped in the next subgraph.
static bool ggml_backend_hexagon_comm_allreduce_add_impl(void * comm_ctx_v, struct ggml_tensor ** partials,
                                                         struct ggml_tensor ** residuals, struct ggml_tensor ** dsts,
                                                         bool owner_only, bool dry_run) {
    if (opt_ar_gap_fuse == 0 || opt_ar_select == 0 || !comm_ctx_v) return false;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    const size_t n_backends = comm_ctx->n_backends;

    if (n_backends < 2 || n_backends > 4) {
        ggml_backend_hexagon_ar_gap_reject("backend count");
        return false;
    }

    for (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        for (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            if (sess_i->phys_idx == sess_j->phys_idx) {
                ggml_backend_hexagon_ar_gap_reject("backends share a physical device");
                return false;
            }
        }
    }

    const ggml_tensor * ref = partials[0];
    if (!ref || (ref->type != GGML_TYPE_F16 && ref->type != GGML_TYPE_F32)) {
        ggml_backend_hexagon_ar_gap_reject("partial type");
        return false;
    }

    for (size_t i = 0; i < n_backends; i++) {
        const ggml_tensor * p = partials[i];
        const ggml_tensor * r = residuals[i];
        const ggml_tensor * d = dsts[i];
        if (!p || !r || !d || !p->buffer || !r->buffer || !d->buffer) {
            ggml_backend_hexagon_ar_gap_reject("missing tensor or buffer");
            return false;
        }
        if (!ggml_backend_buffer_is_hexagon(p->buffer) || !ggml_backend_buffer_is_hexagon(r->buffer) ||
            !ggml_backend_buffer_is_hexagon(d->buffer)) {
            ggml_backend_hexagon_ar_gap_reject("non-hexagon buffer");
            return false;
        }
        if (p->type != ref->type || r->type != ref->type || d->type != ref->type) {
            ggml_backend_hexagon_ar_gap_reject("type mismatch");
            return false;
        }
        if (!ggml_is_contiguous(p) || !ggml_is_contiguous(r) || !ggml_is_contiguous(d)) {
            ggml_backend_hexagon_ar_gap_reject("non-contiguous tensor");
            return false;
        }
        if (!ggml_are_same_shape(p, r) || !ggml_are_same_shape(p, d)) {
            ggml_backend_hexagon_ar_gap_reject("shape mismatch");
            return false;
        }
        for (size_t o = 0; o < n_backends; o++) {
            if (!ggml_hexagon_tensors_overlap(d, partials[o])) {
                continue;
            }
            // Chunked mode only: rank r reads chunk r of every partial and writes chunk r of every dst, so a dst that is exactly its own partial is safe (ggml-alloc reuses the buffer of a single-use partial for the residual ADD output).
            // A full-range reduction reads its own partial while writing it, so that case stays rejected.
            if (ar_gap_chunked() && o == i && d->data == p->data && ggml_nbytes(d) == ggml_nbytes(p)) {
                continue;
            }
            ggml_backend_hexagon_ar_gap_reject("dst overlaps a partial");
            return false;
        }
    }

    if (owner_only && !ar_rs_row_owner_ok(dsts[0])) {
        ggml_backend_hexagon_ar_gap_reject("RS-ROW owner-only needs GGML_HEXAGON_RS_ROW, chunked mode, rows >= GGML_HEXAGON_RS_ROW_MIN_ROWS and a 128-aligned row");
        return false;
    }

    // Every backend must fit the 1D kernel in VTCM before anything is enqueued.
    for (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        struct htp_allreduce_kernel_params kparams;
        if (!ggml_hexagon_precompute_allreduce_params(sess, dsts[r], (uint32_t) r, (uint32_t) n_backends, true, false,
                                                      &kparams, ar_gap_chunked()) ||
            kparams.kernel_type != HTP_ALLREDUCE_KERNEL_DMA_1D) {
            ggml_backend_hexagon_ar_gap_reject("no 1D kernel fits VTCM");
            return false;
        }
    }
    if (dry_run) return true; // validation only: nothing enqueued, no fence sequence consumed

    uint32_t fence_seq_entry = 0;
    uint32_t fence_seq_exit  = 0;
    ggml_backend_hexagon_comm_fence_seqs(comm_ctx, fence_seq_entry, fence_seq_exit);

    std::vector<const ggml_tensor *> data_tensors(n_backends);
    std::vector<const ggml_tensor *> sync_tensors(n_backends);
    std::vector<const ggml_tensor *> dst_tensors(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        data_tensors[i] = partials[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
        dst_tensors[i]  = dsts[i];
    }

    for (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce_add(data_tensors, sync_tensors, residuals[r], dst_tensors, (uint32_t) r, (uint32_t) n_backends,
                                    fence_seq_entry, fence_seq_exit, ar_gap_chunked(), owner_only);
        sess->ar_gap_skip.push_back(dsts[r]);
        for (size_t j = 0; j < n_backends; j++) {
            if (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    return true;
}

// Plain gap-fused ALLREDUCE+ADD (every dst written): the hook behind ggml_backend_comm_allreduce_add_tensor_t.
static bool ggml_backend_hexagon_comm_allreduce_add_tensor(void * comm_ctx_v, struct ggml_tensor ** partials,
                                                           struct ggml_tensor ** residuals, struct ggml_tensor ** dsts, bool dry_run) {
    return ggml_backend_hexagon_comm_allreduce_add_impl(comm_ctx_v, partials, residuals, dsts, false, dry_run);
}

// RS-ROW row threshold for ggml-backend-meta.cpp: batches below GGML_HEXAGON_RS_ROW_MIN_ROWS get no owner-only candidates.
static uint32_t ggml_backend_hexagon_comm_rs_row_min_rows(void) { return opt_rs_row_min_rows; }

// fp16 partial rows (GGML_HEXAGON_AR_PARTIAL_F16): the producer of each partial must be a plain HMX 2D MUL_MAT (two inputs, no fused src2) still in its session's op list.
// Descriptors are filled when an op is enqueued and copied out at synchronize, so the flag is set on the enqueued descriptor itself. Every producer is marked or none, so every owner reads the partials in one format.
static bool ggml_hexagon_mark_partial_f16(const std::vector<ggml_hexagon_session *> & owners, const std::vector<const ggml_tensor *> & partials) {
    std::vector<size_t> prod(owners.size(), SIZE_MAX);
    for (size_t r = 0; r < owners.size(); r++) {
        const ggml_hexagon_opbatch * b = owners[r]->op_batch;
        if (b->h_ops.size() != b->ops.size()) {
            return false;
        }
        // ops/h_ops are fixed-size arrays (n_ops_max) whose entries past n_ops are stale records of earlier batches: only the first n_ops entries are this batch's ops.
        // Scanning the whole array can hit a stale record of the same output tensor (the graph, hence the tensor pointers, repeats every ubatch) and mark the wrong descriptor, leaving the real producer writing fp32 rows while the consumer widens them as fp16.
        for (size_t k = b->n_ops; k-- > 0; ) {
            const htp_opnode & node = b->ops[k];
            if (node.outputs.size() != 1 || node.outputs[0] != partials[r]) {
                continue;
            }
            // the last op that writes this partial decides; it must be a plain HMX 2D MUL_MAT
            const htp_mm_kernel_params * kp = (const htp_mm_kernel_params *) node.kernel_params;
            if (node.opcode == HTP_OP_MUL_MAT && node.inputs.size() == 2 && kp->n_hmx != 0 && kp->kernel_type == HTP_MM_KERNEL_HMX_2D) {
                prod[r] = k;
            }
            break;
        }
        if (prod[r] == SIZE_MAX) {
            return false;
        }
    }
    for (size_t r = 0; r < owners.size(); r++) {
        owners[r]->op_batch->h_ops[prod[r]].flags |= HTP_OPFLAGS_MM_OUT_F16;
    }
    return true;
}

// AE-03 (GGML_HEXAGON_AR_RMS_FUSE=1): the owner-only MC05 boundary (RS-ROW) as one fused op per backend (op_allreduce_add_rms in the HTP). The fused ops are recorded per session and enqueued by the consuming subgraph before its first node (ggml_hexagon_enqueue_pending_add_rms), so no op that runs before them can reuse the partial sums.
// The AR part is validated by the plain gap-fused checks (ggml_backend_hexagon_comm_allreduce_add_impl, dry run). Nothing is recorded when a check fails: the caller then takes the separate path.
static bool ggml_backend_hexagon_comm_allreduce_add_rms_fused(void * comm_ctx_v, struct ggml_tensor ** partials,
                                                             struct ggml_tensor ** residuals, struct ggml_tensor ** dsts,
                                                             struct ggml_tensor ** rms_nodes, struct ggml_tensor ** mul_nodes, bool dry_run) {
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    const size_t n_backends = comm_ctx->n_backends;
    if (n_backends < 2 || n_backends > 3) {
        ggml_backend_hexagon_ar_gap_reject("fused ALLREDUCE+ADD+RMS needs 2 or 3 ranks");
        return false;
    }
    if (!ggml_backend_hexagon_comm_allreduce_add_impl(comm_ctx_v, partials, residuals, dsts, true, true)) {
        return false;
    }
    for (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        struct htp_allreduce_kernel_params kparams;
        if (!ggml_hexagon_precompute_add_rms_params(sess, dsts[r], (uint32_t) r, (uint32_t) n_backends, false, &kparams)) {
            ggml_backend_hexagon_ar_gap_reject("fused ALLREDUCE+ADD+RMS does not fit VTCM");
            return false;
        }
    }
    if (dry_run) return true;

    uint32_t fence_seq_entry = 0;
    uint32_t fence_seq_exit  = 0;
    ggml_backend_hexagon_comm_fence_seqs(comm_ctx, fence_seq_entry, fence_seq_exit);

    std::vector<ggml_hexagon_session *> owners(n_backends);
    for (size_t r = 0; r < n_backends; r++) {
        owners[r] = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
    }
    std::vector<const ggml_tensor *> partial_tensors(n_backends);
    std::vector<const ggml_tensor *> sync_tensors(n_backends);
    std::vector<const ggml_tensor *> norm_tensors(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        partial_tensors[i] = partials[i];
        sync_tensors[i]    = &comm_ctx->fence_tensors[i];
        norm_tensors[i]    = mul_nodes[i];
    }

    // fp16 partial rows (GGML_HEXAGON_AR_PARTIAL_F16): the producers are marked when every partial qualifies; the records carry the flag
    const bool partial_f16 = opt_ar_partial_f16 != 0 && dsts[0]->ne[1] >= 512 && ggml_hexagon_mark_partial_f16(owners, partial_tensors);
    if (opt_ar_partial_f16 != 0) {
        static int n_logged = 0;
        if (n_logged++ < 4) {
            GGML_LOG_INFO("ggml-hex: AR fp16 partials %s (boundary %zu ranks)\n", partial_f16 ? "ON" : "refused", n_backends);
        }
    }
    for (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        float epsilon = 0.0f;
        memcpy(&epsilon, rms_nodes[r]->op_params, sizeof(float));

        ggml_hexagon_pending_add_rms p;
        p.rank            = (uint32_t) r;
        p.n_ranks         = (uint32_t) n_backends;
        p.fence_seq_entry = fence_seq_entry;
        p.fence_seq_exit  = fence_seq_exit;
        p.epsilon         = epsilon;
        p.partials        = partial_tensors;
        p.sync_tensors    = sync_tensors;
        p.residual        = residuals[r];
        p.resid_out       = dsts[r];
        p.weight          = mul_nodes[r]->src[1];
        p.norm_tensors    = norm_tensors;
        p.owners          = owners;
        p.partial_f16     = partial_f16;
        sess->pending_add_rms.push_back(std::move(p));

        sess->ar_gap_skip.push_back(dsts[r]);
        sess->ar_gap_skip.push_back(rms_nodes[r]);
        sess->ar_gap_skip.push_back(mul_nodes[r]);
        for (size_t j = 0; j < n_backends; j++) {
            if (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }
    return true;
}

// MC05 (GGML_HEXAGON_AR_GAP_FUSE=3): the gap-fused ALLREDUCE+ADD above, followed by a row-chunked norm gather that replaces the RMS_NORM+MUL behind the residual ADD on every backend.
// Each rank normalizes its own token rows with the RMS_NORM_MUL row kernel and writes them to every norm dst. The backends skip the ADD, RMS_NORM and MUL nodes (ar_gap_skip).
static bool ggml_backend_hexagon_comm_allreduce_add_rms_tensor(void * comm_ctx_v, struct ggml_tensor ** partials,
                                                                struct ggml_tensor ** residuals, struct ggml_tensor ** dsts,
                                                                struct ggml_tensor ** rms_nodes, struct ggml_tensor ** mul_nodes,
                                                                bool owner_only, bool dry_run) {
    if (opt_ar_gap_fuse != 3 || opt_ar_select == 0 || !comm_ctx_v) return false;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    const size_t n_backends = comm_ctx->n_backends;
    if (n_backends < 2 || n_backends > 4) return false;

    for (size_t i = 0; i < n_backends; i++) {
        const ggml_tensor * rms = rms_nodes[i];
        const ggml_tensor * mul = mul_nodes[i];
        const ggml_tensor * d   = dsts[i];
        if (!rms || !mul || !d || !mul->src[1]) return false;
        const ggml_tensor * w = mul->src[1];
        if (rms->op != GGML_OP_RMS_NORM || rms->src[0] != d || mul->op != GGML_OP_MUL || mul->src[0] != rms) {
            ggml_backend_hexagon_ar_gap_reject("norm structure");
            return false;
        }
        if (d->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || !ggml_are_same_shape(mul, d) ||
            ggml_nelements(w) != d->ne[0] || !ggml_is_contiguous(d) || !ggml_is_contiguous(w) || !ggml_is_contiguous(mul)) {
            ggml_backend_hexagon_ar_gap_reject("norm operands");
            return false;
        }
        if (!ggml_backend_buffer_is_hexagon(mul->buffer) || !ggml_backend_buffer_is_hexagon(w->buffer)) {
            ggml_backend_hexagon_ar_gap_reject("norm buffers");
            return false;
        }
    }

    // AE-03: owner-only boundaries of prefill batches take the fused op (ggml_backend_hexagon_comm_allreduce_add_rms_fused). A failed check
    // returns false before anything is recorded, and the caller then takes the separate path.
    if (owner_only && opt_ar_rms_fuse && n_backends <= 3 &&
            (int64_t) dsts[0]->ne[1] * dsts[0]->ne[2] * dsts[0]->ne[3] >= (int64_t) PREFILL_MIN_ROWS) {
        return ggml_backend_hexagon_comm_allreduce_add_rms_fused(comm_ctx_v, partials, residuals, dsts, rms_nodes, mul_nodes, dry_run);
    }

    // The norm gather must fit in VTCM on every backend before anything is enqueued.
    for (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        struct htp_allreduce_kernel_params kparams;
        ggml_hexagon_precompute_rms_ag_params(sess, dsts[r], (uint32_t) r, (uint32_t) n_backends, false, &kparams);
        if ((size_t) kparams.vtcm_size > sess->vtcm_size) {
            ggml_backend_hexagon_ar_gap_reject("norm gather does not fit VTCM");
            return false;
        }
    }

    // (1) ALLREDUCE+ADD, rank-chunked (own rows only with owner_only). It validates first and enqueues nothing on failure.
    if (!ggml_backend_hexagon_comm_allreduce_add_impl(comm_ctx_v, partials, residuals, dsts, owner_only, dry_run)) return false;
    if (dry_run) return true; // validation only

    // (2) norm gather on every backend, enqueued after that backend's ALLREDUCE+ADD in the same batch.
    uint32_t fence_seq_entry = 0;
    uint32_t fence_seq_exit  = 0;
    ggml_backend_hexagon_comm_fence_seqs(comm_ctx, fence_seq_entry, fence_seq_exit);

    std::vector<const ggml_tensor *> sync_tensors(n_backends);
    std::vector<const ggml_tensor *> norm_tensors(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
        norm_tensors[i] = mul_nodes[i];
    }

    for (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        float epsilon = 0.0f;
        memcpy(&epsilon, rms_nodes[r]->op_params, sizeof(float));
        // Tile-capable (prefill-sized) norm outputs are deferred to the consuming subgraph; decode-sized ones are enqueued here as before.
        if (opt_gather_act_tile && dsts[r]->ne[1] >= PREFILL_MIN_ROWS) {
            // GGML_HEXAGON_GATHER_ACT_TILE: the norm gather is enqueued by the consuming subgraph (ggml_hexagon_enqueue_pending_rms_ag)
            ggml_hexagon_pending_rms_ag p;
            p.rank            = (uint32_t) r;
            p.n_ranks         = (uint32_t) n_backends;
            p.fence_seq_entry = fence_seq_entry;
            p.fence_seq_exit  = fence_seq_exit;
            p.epsilon         = epsilon;
            p.sync_tensors    = sync_tensors;
            p.residual        = dsts[r];
            p.weight          = mul_nodes[r]->src[1];
            p.norm_tensors    = norm_tensors;
            sess->pending_rms_ag.push_back(std::move(p));
        } else {
            sess->enqueue_allreduce_rms_ag(sync_tensors, dsts[r], mul_nodes[r]->src[1], norm_tensors, (uint32_t) r, (uint32_t) n_backends,
                                           fence_seq_entry, fence_seq_exit, epsilon, false);
        }
        sess->ar_gap_skip.push_back(rms_nodes[r]);
        sess->ar_gap_skip.push_back(mul_nodes[r]);
    }
    return true;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_split_buffer_type(int main_device, const float * tensor_split) {
    GGML_UNUSED(tensor_split);
    auto reg = ggml_backend_hexagon_reg();
    auto dev = ggml_backend_reg_dev_get(reg, main_device);
    if (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    if (!dev) return nullptr;
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    return &dev_ctx->buffer_type;
}

static void * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *) ggml_backend_hexagon_split_buffer_type;
    }
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *) ggml_backend_hexagon_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *) ggml_backend_hexagon_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_add_tensor") == 0 && opt_ar_gap_fuse != 0) {
        return (void *) ggml_backend_hexagon_comm_allreduce_add_tensor;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_add_rms_tensor") == 0 && opt_ar_gap_fuse == 3) {
        return (void *) ggml_backend_hexagon_comm_allreduce_add_rms_tensor;
    }
    // RS-ROW capability (owner-only residual writes): offered only when GGML_HEXAGON_RS_ROW is on, so the meta layer stays on the
    // full-write path otherwise.
    if (strcmp(name, "ggml_backend_comm_allreduce_add_rms_owner_tensor") == 0 && opt_ar_gap_fuse == 3 && opt_rs_row != 0) {
        return (void *) ggml_backend_hexagon_comm_allreduce_add_rms_tensor;
    }
    // RS-ROW row threshold, offered with the owner-only capability (ggml-backend-meta.cpp reads it at construction).
    if (strcmp(name, "ggml_backend_comm_rs_row_min_rows") == 0 && opt_ar_gap_fuse == 3 && opt_rs_row != 0) {
        return (void *) ggml_backend_hexagon_comm_rs_row_min_rows;
    }
    return NULL;
}

template<typename T> std::vector<T> str_to_vec(const char* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    while (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    return v;
}

template<typename T, int BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    for (auto i : v) { ss << i << ','; }
    auto str = ss.str(); str.pop_back(); // drop last comma
    return str;
}

static void ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, bool discovery_supported, const std::unordered_map<int, fastrpc_domain> & cdsp_map) {
    if (discovery_supported) {
        auto it = cdsp_map.find(cfg.physical_idx);
        if (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } else {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } else {
        switch (cfg.physical_idx) {
            case 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                break;
            case 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                break;
            default:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                break;
        }
    }
    for (auto & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
static void ggml_hexagon_discover_devices() {
    std::unordered_map<int, fastrpc_domain> cdsp_map;
    bool discovery_supported = false;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    int err = remote_system_request(&domain_info);
    if (err == AEE_SUCCESS && domain_info.sys.num_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.num_domains);
        domain_info.sys.domains     = domains.data();
        domain_info.sys.max_domains = (int) domains.size();

        err = remote_system_request(&domain_info);
        if (err == AEE_SUCCESS) {
            discovery_supported = true;
            const int n_domains = std::min(domain_info.sys.num_domains, (int) domains.size());
            for (int i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (int) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                if (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (int) domains[i].type);
                    continue;
                }
                if (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    continue;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } else {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (unsigned) err);
        }
    } else if (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (unsigned) err);
    }

    // Populate domain IDs and names for all configured devices
    for (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

static void ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned int) HTP_TYPE_Q4_0 == (unsigned int) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q4_1 == (unsigned int) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q8_0 == (unsigned int) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_MXFP4 == (unsigned int) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_IQ4_NL == (unsigned int) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q4_K == (unsigned int) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q5_K == (unsigned int) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q6_K == (unsigned int) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");

    const char * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    const char * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    const char * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    const char * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    const char * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    const char * str_enqueue_fast = getenv("GGML_HEXAGON_ENQUEUE_FAST");
    const char * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    const char * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    const char * str_etm      = getenv("GGML_HEXAGON_ETM");
    const char * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    const char * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    const char * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    const char * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    const char * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    const char * str_gdn_hmx  = getenv("GGML_HEXAGON_GDN_HMX");
    const char * str_gdn_prompt_hmx = getenv("GGML_HEXAGON_GDN_PROMPT_HMX");
    const char * str_gdn_snap = getenv("GGML_HEXAGON_GDN_SNAP");
    const char * str_gdn_hmx_batch   = getenv("GGML_HEXAGON_GDN_HMX_BATCH");
    const char * str_gdn_wide        = getenv("GGML_HEXAGON_GDN_WIDE");
    const char * str_gdn_qkshare     = getenv("GGML_HEXAGON_GDN_QKSHARE");
    const char * str_gdn_state_merge = getenv("GGML_HEXAGON_GDN_STATE_MERGE");
    const char * str_gdn_up          = getenv("GGML_HEXAGON_GDN_UP");
    const char * str_gdn_colsplit = getenv("GGML_HEXAGON_GDN_COLSPLIT");
    const char * str_concat_t_hvx = getenv("GGML_HEXAGON_CONCAT_T_HVX");
    const char * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    const char * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    const char * str_arch     = getenv("GGML_HEXAGON_ARCH");
    const char * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    const char * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    const char * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    const char * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    const char * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    const char * str_hmx_int  = getenv("GGML_HEXAGON_HMX_INT");
    const char * str_hmx_min_rows = getenv("GGML_HEXAGON_HMX_MIN_ROWS");

    // Init Arch first since it affects other defaults
    if (!str_arch) {
        int err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } else {
            if (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } else if (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } else {
        if (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, NULL, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? new std::regex(str_opfilter, RE_ICASE) : NULL;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    if (const char * v = getenv("GGML_HEXAGON_EARLY_FLUSH_OPS")) opt_early_flush_ops = atoi(v);
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, NULL, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, NULL, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, NULL, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  NULL, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    if (const char * v = getenv("GGML_HEXAGON_FUSE_S150"))  opt_fuse_s150 = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_GDN_GATE_PF"))  opt_gdn_gate_pf = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_GATE_MUL"))     opt_gate_mul = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_NORM_GATED"))  opt_norm_gated = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_NG_SIG16")) opt_ng_sig16 = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_NG_TILE")) opt_ng_tile = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_GATE_TILE")) opt_gate_tile = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_ROPE_KV")) opt_rope_kv = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_GDN_GATE_FOLD")) opt_gdn_gate_fold = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_LOSSY_GDN")) opt_lossy_gdn = (uint32_t) strtoul(v, NULL, 0);
    if (const char * v = getenv("GGML_HEXAGON_SIG16_V2")) opt_sig16_v2 = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_FA_EXP_V2")) opt_fa_exp_v2 = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_LM_NORM")) opt_lm_norm = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_LM_FA_SKIP")) opt_lm_fa_skip = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_CONV_DYN")) opt_conv_dyn = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_CONV_SPLIT32")) opt_conv_split32 = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_CONV_SPLIT32_COST")) { unsigned a = 0, b = 0; if (sscanf(v, "%u,%u", &a, &b) == 2 && a && b) { opt_conv_split32_cq = a; opt_conv_split32_cv = b; } }
    if (const char * v = getenv("GGML_HEXAGON_DMA_CLEAN"))  opt_dma_clean = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_ROPE_FAST")) opt_rope_fast = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_ROPE_NORM")) opt_rope_norm = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_FUSE_AR_DIAG")) opt_ar_diag = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_AR_GAP_FUSE")) {
        const int gap = atoi(v);
        opt_ar_gap_fuse = (gap >= 1 && gap <= 3) ? gap : 0;
    }
    // RS-ROW only exists on the MC05 path (GGML_HEXAGON_AR_GAP_FUSE=3); anything else keeps it off.
    if (const char * v = getenv("GGML_HEXAGON_RS_ROW")) opt_rs_row = (atoi(v) != 0 && opt_ar_gap_fuse == 3) ? 1 : 0;
    if (const char * v = getenv("GGML_HEXAGON_AR_RMS_FUSE")) opt_ar_rms_fuse = (atoi(v) != 0 && opt_ar_gap_fuse == 3 && opt_rs_row != 0) ? 1 : 0;
    if (const char * v = getenv("GGML_HEXAGON_AR_PARTIAL_F16")) opt_ar_partial_f16 = (atoi(v) != 0 && opt_ar_rms_fuse != 0) ? 1 : 0;
    if (const char * v = getenv("GGML_HEXAGON_RS_ROW_MIN_ROWS")) {
        const long n = atol(v);
        if (n >= 16 && n <= (1L << 20)) {
            opt_rs_row_min_rows = (uint32_t) n;
        } else {
            GGML_LOG_WARN("ggml-hex: GGML_HEXAGON_RS_ROW_MIN_ROWS=%s ignored (need an integer in [16, 1048576]); using 64\n", v);
        }
    }
    if (const char * v = getenv("GGML_HEXAGON_ACT_TILE")) opt_act_tile = atoi(v) >= 2 ? 2 : (atoi(v) > 0 ? 1 : 0);
    if (const char * v = getenv("GGML_HEXAGON_GATHER_ACT_TILE")) opt_gather_act_tile = (atoi(v) != 0 && opt_ar_gap_fuse == 3) ? 1 : 0;
    if (const char * v = getenv("GGML_HEXAGON_SWIGLU_OUT")) opt_swiglu_out = atoi(v) ? 1 : 0;
    if (const char * v = getenv("GGML_HEXAGON_SWIGLU_NC")) {
        const int n = atoi(v);
        if (n == 512 || n == HTP_MM_SWIGLU_NC) {
            opt_swiglu_nc = n;
        } else {
            GGML_LOG_WARN("ggml-hex: GGML_HEXAGON_SWIGLU_NC=%s ignored (need 512 or 768); using %d\n", v, HTP_MM_SWIGLU_NC);
        }
    }
    if (const char * v = getenv("GGML_HEXAGON_VR_AR")) opt_vr_ar = (uint32_t) atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_AR_BLOCK_ROWS")) opt_ar_block_rows = (atoi(v) == 8) ? 8u : 16u;
    if (const char * v = getenv("GGML_HEXAGON_ELT_SMALL")) opt_elt_small = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_SWIGLU_OUT_MIN_ROWS")) opt_swiglu_out_min_rows = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_LX_FUSE")) opt_lx_fuse = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_SWIGLU_SIG16")) opt_swiglu_sig16 = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_MM_PROLOGUE")) opt_mm_prologue_flags = mm_prologue_op_flags(atoi(v));
    if (const char * v = getenv("GGML_HEXAGON_AR_DMA_CLEAN")) opt_ar_dma_clean_flags = atoi(v) ? HTP_OPFLAGS_AR_DMA_CLEAN : 0;
    if (const char * v = getenv("GGML_HEXAGON_A8W8_PRE")) { opt_a8w8_pre = (uint32_t) atoi(v); }
    if (const char * v = getenv("GGML_HEXAGON_MM_A8W8")) { opt_mm_a8w8_ops = (uint32_t) atoi(v); opt_mm_a8w8_flags = opt_mm_a8w8_ops ? HTP_OPFLAGS_MM_A8W8 : 0; }
    if (const char * v = getenv("GGML_HEXAGON_MM_C1SMALL")) opt_mm_c1small_flags = atoi(v) ? HTP_OPFLAGS_MM_C1SMALL : 0;
    if (const char * v = getenv("GGML_HEXAGON_MM_EARLY_W")) opt_mm_early_w_flags = atoi(v) ? HTP_OPFLAGS_MM_EARLY_W : 0;
    if (const char * v = getenv("GGML_HEXAGON_MM_ACT_OVERLAP")) opt_mm_act_overlap_flags = atoi(v) ? HTP_OPFLAGS_MM_ACT_OVERLAP : 0;
    if (const char * v = getenv("GGML_HEXAGON_LANE_THREADS")) opt_lane_threads = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_LANE_A_MB"))    opt_lane_a_mb    = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_LANE_OPS")) {
        opt_lane_ops = 0;
        if (strstr(v, "ssm"))     opt_lane_ops |= 1;
        if (strstr(v, "normg"))   opt_lane_ops |= 2;
        if (strstr(v, "qk"))      opt_lane_ops |= 4;
        if (strstr(v, "rope"))    opt_lane_ops |= 8;
        if (strstr(v, "gatemul")) opt_lane_ops |= 16;
    }
    if (const char * v = getenv("GGML_HEXAGON_NX_ALPHABETA")) opt_nx_alphabeta = atoi(v) ? 1 : 0;
    if (const char * v = getenv("GGML_HEXAGON_ELT_SMALL_MIN_ROWS")) {
        const long n = atol(v);
        if (n >= 16 && n <= (1L << 20)) {
            opt_elt_small_min_rows = (uint32_t) n;
        } else {
            GGML_LOG_WARN("ggml-hex: GGML_HEXAGON_ELT_SMALL_MIN_ROWS=%s ignored (need an integer in [16, 1048576]); using 64\n", v);
        }
    }
    if (const char * v = getenv("GGML_HEXAGON_SSM_CONV_PIPE")) opt_ssm_conv_pipe = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_SSM_CONV_CHAIN")) opt_ssm_conv_chain = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_DISPATCH_1PASS")) opt_dispatch_1pass = atoi(v);
    if (const char * v = getenv("LLAMA_CYCLE_TIMING"))          opt_cycle_timing   = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_SSM_CONV_CHAIN_TB")) opt_ssm_conv_chain_tb = (uint32_t) atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_C2_NG")) opt_c2_ng = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_C2_RS")) opt_c2_rs = atoi(v);
    if (const char * v = getenv("GGML_HEXAGON_C2_SILU")) opt_c2_silu = atoi(v);
    opt_enqueue_fast = str_enqueue_fast ? atoi(str_enqueue_fast)          : opt_enqueue_fast;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, NULL, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_gdn_select = str_gdn_select ? atoi(str_gdn_select)                 : opt_gdn_select;
    opt_gdn_hmx  = str_gdn_hmx ? atoi(str_gdn_hmx) : opt_gdn_hmx;
    opt_gdn_prompt_hmx = str_gdn_prompt_hmx ? atoi(str_gdn_prompt_hmx) : opt_gdn_prompt_hmx;
    if (opt_gdn_prompt_hmx != 0) {
        GGML_LOG_INFO("ggml-hex: GDN prompt HMX on (K > 1 batches with the typed final-state snapshot range; needs LLAMA_P4K_PROMPT_SNAP=1)\n");
    }
    if (str_gdn_snap) {
        opt_gdn_snap = strcmp(str_gdn_snap, "skip") == 0        ? 1 :
                       strcmp(str_gdn_snap, "async") == 0       ? 2 :
                       strcmp(str_gdn_snap, "sync_ticket") == 0 ? 3 :
                       strcmp(str_gdn_snap, "version") == 0     ? 4 : 0;
    }
    opt_gdn_opt = (str_gdn_hmx_batch   && atoi(str_gdn_hmx_batch)   != 0 ? HTP_GDN_OPT_HMX_BATCH   : 0) |
                  (str_gdn_wide        && atoi(str_gdn_wide)        != 0 ? HTP_GDN_OPT_WIDE        : 0) |
                  (str_gdn_qkshare     && atoi(str_gdn_qkshare)     != 0 ? HTP_GDN_OPT_QKSHARE     : 0) |
                  (str_gdn_state_merge && atoi(str_gdn_state_merge) != 0 ? HTP_GDN_OPT_STATE_MERGE : 0);
    // GGML_HEXAGON_GDN_UP bitmask (default 0): bit 0 = DMA-for-all-tensors token-gen path + its VTCM layout (upstream 9b47b8ad4, c6c532daa; SYNC snapshots only), bit 1 = HMX chunk decays with f32 exp (revert of 268f3874d numerics), bit 2 = HMX phase 4 without the zero-decay mask (revert of 7d4b7c4d9 numerics)
    const uint32_t gdn_up = str_gdn_up ? (uint32_t) strtoul(str_gdn_up, NULL, 0) : 0;
    opt_gdn_opt |= ((gdn_up & 1u) ? HTP_GDN_OPT_UP_DMA : 0u) |
                   ((gdn_up & 2u) ? HTP_GDN_OPT_UP_LEGACY_EXP : 0u) |
                   ((gdn_up & 4u) ? HTP_GDN_OPT_UP_LEGACY_ACC : 0u);
    if (gdn_up != 0) {
        GGML_LOG_INFO("ggml-hex: GDN up bits 0x%x (dma-tg %d, legacy-exp %d, legacy-acc %d)\n", (unsigned) gdn_up,
                      !!(gdn_up & 1u), !!(gdn_up & 2u), !!(gdn_up & 4u));
    }
    if (opt_gdn_opt != 0) {
        GGML_LOG_INFO("ggml-hex: GDN opt bits 0x%x (hmx-batch %d, wide %d, qkshare %d, state-merge %d)\n", (unsigned) opt_gdn_opt,
                      !!(opt_gdn_opt & HTP_GDN_OPT_HMX_BATCH), !!(opt_gdn_opt & HTP_GDN_OPT_WIDE),
                      !!(opt_gdn_opt & HTP_GDN_OPT_QKSHARE), !!(opt_gdn_opt & HTP_GDN_OPT_STATE_MERGE));
    }
    // GGML_HEXAGON_GDN_FIX bitmask (default 0): HTP_GDN_FIX_PASS_SLICE, HTP_GDN_FIX_PROLOGUE (chunked HMX, n_tokens >= 64)
    if (const char * str_gdn_fix = getenv("GGML_HEXAGON_GDN_FIX")) {
        opt_gdn_fix = (uint32_t) strtoul(str_gdn_fix, NULL, 0);
        GGML_LOG_INFO("ggml-hex: GDN fix bits 0x%x (pass-slice %d, prologue %d)\n", (unsigned) opt_gdn_fix,
                      !!(opt_gdn_fix & 1u), !!(opt_gdn_fix & 2u));
    }
    // GGML_HEXAGON_GDN_SKEW=<bytes>: per-head pad of HTP_GDN_FIX_HEAD_SKEW (bit 2), rounded up to 128-byte units (one byte in kparams)
    if (const char * str_gdn_skew = getenv("GGML_HEXAGON_GDN_SKEW")) {
        const uint32_t skew_bytes = (uint32_t) strtoul(str_gdn_skew, NULL, 0);
        opt_gdn_skew_units = (skew_bytes + 127u) / 128u > 255u ? 255u : (skew_bytes + 127u) / 128u;
        GGML_LOG_INFO("ggml-hex: GDN head skew %u bytes (bit 2 of GGML_HEXAGON_GDN_FIX)\n", (unsigned) opt_gdn_skew_units * 128u);
    }
    if (str_gdn_colsplit) {
        opt_gdn_colsplit = (std::min)((std::max)(atoi(str_gdn_colsplit), 0), 3);
    }
    if (str_concat_t_hvx) {
        opt_concat_t_hvx = atoi(str_concat_t_hvx) != 0;
    }
    opt_ar_select = str_ar_select ? atoi(str_ar_select)                   : opt_ar_select;
    opt_mbuf      = str_mbuf     ? strtoul(str_mbuf, NULL, 0) * MiB       : opt_mbuf;
    opt_vmem      = str_vmem     ? strtoul(str_vmem, NULL, 0) * MiB       : opt_vmem;
    opt_hostbuf   = str_hostbuf  ? atoi(str_hostbuf) != 0                 : opt_hostbuf;
    opt_hmx_int   = str_hmx_int  ? atoi(str_hmx_int)                      : 0;
    opt_hmx_min_rows = str_hmx_min_rows ? atoi(str_hmx_min_rows)          : opt_hmx_min_rows;

    // Parse device configuration
    const char * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    if (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    if (str_devices && str_devices[0] != '\0') {
        bool is_single_number = true;
        for (int i = 0; str_devices[i] != '\0'; i++) {
            if (!isdigit((unsigned char)str_devices[i])) {
                is_single_number = false;
                break;
            }
        }
        if (is_single_number) {
            int n = atoi(str_devices);
            if (n < 1) n = 1;
            if (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            for (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (int)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.clear();
            }
        } else {
            std::string s_devices(str_devices);
            std::vector<std::string> items;
            std::string curr_item;
            int bracket_depth = 0;
            for (char ch : s_devices) {
                if (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } else if (ch == ']') {
                    if (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } else if (ch == ',' && bracket_depth == 0) {
                    size_t s = curr_item.find_first_not_of(" \t\r\n");
                    size_t e = curr_item.find_last_not_of(" \t\r\n");
                    if (s != std::string::npos) {
                        items.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.clear();
                } else {
                    curr_item += ch;
                }
            }
            size_t s = curr_item.find_first_not_of(" \t\r\n");
            size_t e = curr_item.find_last_not_of(" \t\r\n");
            if (s != std::string::npos) {
                items.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            for (const auto & item : items) {
                size_t b_open  = item.find('[');
                size_t b_close = item.rfind(']');

                if (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    int virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    if (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        try {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } catch (...) {
                            virt = 0;
                        }
                    } else {
                        size_t dev_colon = dev_name.find(':');
                        if (dev_colon != std::string::npos) {
                            try {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } catch (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<int> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    while (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        if (ps == std::string::npos) continue;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        if (dash_pos != std::string::npos) {
                            try {
                                int p_start = std::stoi(p_part.substr(0, dash_pos));
                                int p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                for (int p = p_start; p <= p_end; p++) {
                                    if (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } catch (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } else {
                            try {
                                int p = std::stoi(p_part);
                                if (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } catch (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    if (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    if (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        auto & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.clear();

                        for (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_group.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } else {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } else if (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    size_t colon_pos = rest.find(':');
                    int phys = 0;
                    int virt = 0;
                    try {
                        if (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } else {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } catch (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        continue;
                    }

                    if (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_group.clear();
                        opt_ndev++;
                    } else {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } else {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", item.c_str());
                }
            }
        }
    } else {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_group.clear();
    }

#if defined(__ANDROID__)
    if (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    if (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            auto v  = str_to_vec<uint32_t>(str_profile);
            switch (v.size()) {
                case 1:  opt_profile = v[0]; return opt_pmu_evt; // mode with default pmu events
                case 8:  opt_profile = 2;    return v;           // mode with custom  pmu events
                default: opt_profile = 0;    return {};          // garbage input
            }}();
        if (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = new ggml_hexagon_registry(reg);
}

static const struct ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(void) {
    static bool initialized = false;

    static ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ NULL };

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            auto nErr = htpdrv_init();
            if (nErr != AEE_SUCCESS) {
                return NULL;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
