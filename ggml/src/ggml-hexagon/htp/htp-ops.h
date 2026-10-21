#ifndef HTP_OPS_H
#define HTP_OPS_H

#include <assert.h>

// ggml-common.h must be included prio to this header

enum htp_status {
    HTP_STATUS_OK             = 1,
    HTP_STATUS_INTERNAL_ERR   = 2,
    HTP_STATUS_NO_SUPPORT     = 3,
    HTP_STATUS_INVAL_PARAMS   = 4,
    HTP_STATUS_VTCM_TOO_SMALL = 5,
};

// First set of values must match the ggml_type.
// Duplicated here because we can't include full ggml.h in the htp build.
// We have some static_asserts in the cpp code to ensure things are in sync.
enum htp_data_type {
    HTP_TYPE_F32    = 0,
    HTP_TYPE_F16    = 1,
    HTP_TYPE_Q4_0   = 2,
    HTP_TYPE_Q4_1   = 3,
    HTP_TYPE_Q8_0   = 8,
    HTP_TYPE_Q4_K   = 12,
    HTP_TYPE_Q5_K   = 13,
    HTP_TYPE_Q6_K   = 14,
    HTP_TYPE_IQ4_NL = 20,
    HTP_TYPE_I32    = 26,
    HTP_TYPE_I64    = 27,
    HTP_TYPE_MXFP4  = 39,

    // types used internally for repack, dyn.quant, etc
    HTP_TYPE_Q4_0_TILED = 200,
    HTP_TYPE_Q4_1_TILED,
    HTP_TYPE_Q8_0_TILED,
    HTP_TYPE_MXFP4_TILED,

    HTP_TYPE_INVALID
};

// Constats for internal types
#define QK_Q4_0_TILED  256  // 32x32 Q4_0 tiled layout
#define QK_Q8_0_TILED  128  // 32x32 Q8_0 tiled layout
#define QK_MXFP4_TILED 256  // 32x32 MXFP4 tiled layout



// Do not reorder first 4 (used as an index)
enum htp_op_code {
    HTP_OP_MUL = 0,
    HTP_OP_ADD = 1,
    HTP_OP_SUB = 2,
    HTP_OP_DIV = 3,
    HTP_OP_MUL_MAT,
    HTP_OP_MUL_MAT_ID,
    HTP_OP_MUL_MAT_NX,
    HTP_OP_MUL_MAT_ID_NX,
    HTP_OP_MUL_MAT_ADD,
    HTP_OP_RMS_NORM,
    HTP_OP_RMS_NORM_MUL,
    HTP_OP_RMS_NORM_SCALE,
    HTP_OP_UNARY_SILU,
    HTP_OP_UNARY_GELU,
    HTP_OP_UNARY_SIGMOID,
    HTP_OP_UNARY_EXP,
    HTP_OP_UNARY_NEG,
    HTP_OP_UNARY_SOFTPLUS,
    HTP_OP_UNARY_TANH,
    HTP_OP_UNARY_ABS,
    HTP_OP_UNARY_LOG,
    HTP_OP_UNARY_RELU,
    HTP_OP_UNARY_STEP,
    HTP_OP_GLU_SWIGLU,
    HTP_OP_GLU_SWIGLU_OAI,
    HTP_OP_GLU_GEGLU,
    HTP_OP_GLU_GEGLU_QUICK,
    HTP_OP_SOFTMAX,
    HTP_OP_ADD_ID,
    HTP_OP_ROPE,
    HTP_OP_ROPE_NX,
    HTP_OP_FLASH_ATTN_EXT,
    HTP_OP_SET_ROWS,
    HTP_OP_GET_ROWS,
    HTP_OP_SCALE,
    HTP_OP_CPY,
    HTP_OP_CPY_FENCE,
    HTP_OP_ARGSORT,
    HTP_OP_TOP_K,
    HTP_OP_SQR,
    HTP_OP_SQRT,
    HTP_OP_SUM,
    HTP_OP_SUM_ROWS,
    HTP_OP_SSM_CONV,
    HTP_OP_SSM_CONV_SILU,
    HTP_OP_REPEAT,
    HTP_OP_CUMSUM,
    HTP_OP_FILL,
    HTP_OP_DIAG,
    HTP_OP_SOLVE_TRI,
    HTP_OP_L2_NORM,
    HTP_OP_GATED_DELTA_NET,
    HTP_OP_TRI,
    HTP_OP_PAD,
    HTP_OP_NORM,
    HTP_OP_CONCAT,
    HTP_OP_CLAMP,
    HTP_OP_LEAKY_RELU,
    HTP_OP_IM2COL,
    HTP_OP_FENCE,
    HTP_OP_ALLREDUCE,
    HTP_OP_ALLREDUCE_ADD,
    HTP_OP_GLU_SWIGLU_CLAMP,
    HTP_OP_MDEV_GROUP,
    HTP_OP_ROLL,
    HTP_OP_ARGMAX,
    HTP_OP_GDN_GATE,   // fused GDN gate prep (softplus(x)*A, sigmoid(beta)); see unary-ops.c
    HTP_OP_RMS_NORM_MUL_SILU, // fused gated output norm RMS_NORM_MUL(x, w) * SILU(z); see unary-ops.c
    HTP_OP_SSM_CONV_CHAIN,  // fused CONCAT + CPY + SSM_CONV + SILU; see ssm-conv.c
    HTP_OP_GATE_MUL,        // fused attention gate MUL(x, SIGMOID(g)); see unary-ops.c

    HTP_OP_INVALID
};

#define HTP_OP_MAX_DIMS    4    // aka GGML_MAX_DIMS
#define HTP_OP_MAX_INPUTS  10   // aka GGML_MAX_SRCS
#define HTP_OP_MAX_OUTPUTS 4
#define HTP_OP_MAX_PARAMS  16   // aka GGML_MAX_OP_PARAMS
#define HTP_OP_MAX_KERN_PARAMS 32

#define HTP_OP_MAX_BUFS    16
#define HTP_OP_MAX_TENSORS 8192 // must stay under 64K (uint16)

#define HTP_FENCE_TIMEOUT  (1000000000ULL)
#define HTP_FENCE_SLOT_SIZE 128

#define HTP_OP_MAX_VMEM_DEFAULT (3355443200u)

#define HTP_MMAP_MAX_VMEM  (2147483648u)

#define HTP_A8W8_PE_TAIL 1024   // per column tile of a pre-expanded A8W8 blob: [n_k*1024 w8][sw f32[32] @+0][cs i32[32] @+128][pad]; keeps every tile base 1 KB aligned

enum htp_tensor_flags {
    HTP_TENSOR_WEIGHT  = (1U << 0), // Tensor buffer model weight data (not compute)
    HTP_TENSOR_REPACK  = (1U << 1), // Tensor is in repacked tiled format
    HTP_TENSOR_FENCE   = (1U << 2), // Tensor is synchronization fence (explicitly managed)
    HTP_TENSOR_ACT_TILE = (1U << 3), // Activation is stored in the fp16 HMX tile layout (htp/hmx-act-tiles.h)
    HTP_TENSOR_A8W8     = (1U << 4)  // Q4_0 tiled weight is followed (at round_up(nb[3]*ne[3], 128) from data) by the pre-expanded A8W8 int8 blob: per column tile n_k*1024 B w8 + HTP_A8W8_PE_TAIL tail (sw f32[32], cs i32[32] at +128)
};

enum htp_buf_flags {
    HTP_BUF_EXTENDED = (1U << 0),
};

// Tensor descriptor
struct htp_tensor {
    uint64_t data;                 // Buffer offset in the messages, and data pointer on the NPU
    uint32_t size;                 // Data size in bytes
    uint32_t flags;                // Buffer / tensor flags
    uint32_t type;                 // Data type
    uint16_t bi;                   // Buffer index
    uint16_t ti;                   // Tensor index
    uint32_t ne[HTP_OP_MAX_DIMS];  // Number of elements
    uint32_t nb[HTP_OP_MAX_DIMS];  // Stride in bytes (see ggml.h ggml_tensor)
};

// Buffer descriptor
struct htp_buf_desc {
    uint64_t base;     // base address
    uint64_t size;     // total size
    uint32_t flags;    // HTP_BUF_*
    uint32_t fd;       // file descriptor
};

// LossyMisc (GGML_HEXAGON_LM_*): per-opcode lossy mask in op flag bits 17.. (FLASH_ATTN_EXT: ablation/softmax, ALLREDUCE_ADD: norm mode). Bits 17..30 are
// per-opcode (GDN / SSM_CONV_CHAIN use them for their own ops).
#define HTP_OPFLAGS_LM_SHIFT 17

enum htp_op_flags {
    HTP_OPFLAGS_STUB  = (1U << 0),
    HTP_OPFLAGS_ACT_TILE_ONLY = (1U << 1), // RMS_NORM_MUL with dsts[1] tiles: the fp32 output is not written (all consumers are HMX matmuls)
    HTP_OPFLAGS_SWIGLU_OUT = (1U << 3), // NX gate/up writes silu(g)*u as fp16 act tiles (dsts[0]) for the FFN down (GGML_HEXAGON_SWIGLU_OUT)
    HTP_OPFLAGS_DMA_CLEAN = (1U << 4), // GGML_HEXAGON_DMA_CLEAN: an op that reads its inputs by DMA skips the flush of inputs only DMA wrote in this batch (htp-tensor.c)
    HTP_OPFLAGS_MM_SPLIT_EDGES = (1U << 5), // HMX MUL_MAT/NX: first and last weight chunk are half width (GGML_HEXAGON_MM_PROLOGUE)
    HTP_OPFLAGS_MM_CROSS_WEIGHT = (1U << 6), // HMX MUL_MAT_NX: one chunk pipeline over all weights (cross-weight prefetch)
    HTP_OPFLAGS_MM_ACT_OVERLAP = (1U << 7), // HMX MUL_MAT/NX: fp16-tile M blocks of >= 256 rows DMA their activation in 128-row groups, overlapped with the first chunk (GGML_HEXAGON_MM_ACT_OVERLAP)
    HTP_OPFLAGS_SWIGLU_NOBAR = (1U << 8), // SWIGLU_OUT epilogue without the value barriers (GGML_HEXAGON_LX_FUSE bit 0, LxFuseFree)
    HTP_OPFLAGS_MM_OUT_F16 = (1U << 9), // HMX MUL_MAT partial producer of the fused ALLREDUCE+ADD+RMS: compact fp16 output rows (GGML_HEXAGON_AR_PARTIAL_F16)
    HTP_OPFLAGS_LANE_B    = (1U << 10),  // Phase-1 lane experiment (GGML_HEXAGON_LANE_OPS): issue to HVX lane B (own work queue, VTCM partition, DMA queues); dispatch continues
    HTP_OPFLAGS_MM_C1SMALL = (1U << 15), // HMX MUL_MAT prefill with SPLIT_EDGES, K >= 3072: chunk 1 as narrow as chunk 0 (GGML_HEXAGON_MM_C1SMALL)
    HTP_OPFLAGS_MM_EARLY_W = (1U << 14), // HMX MUL_MAT/NX/SWIGLU prefill: first weight chunk DMA is pushed at op entry (GGML_HEXAGON_MM_EARLY_W)
    HTP_OPFLAGS_AR_DMA_CLEAN = (1U << 13), // ALLREDUCE*: DMA-only reader/writer (GGML_HEXAGON_AR_DMA_CLEAN; needs DMA_CLEAN)
    HTP_OPFLAGS_SWIGLU_SIG16 = (1U << 12), // SWIGLU_OUT epilogue with the fp16 sigmoid (GGML_HEXAGON_SWIGLU_SIG16, numerics-changing)
    HTP_OPFLAGS_ROPE_FAST = (1U << 13), // GGML_HEXAGON_ROPE_FAST: MROPE theta cache through the vectorised mod-3 path for qwen35's 11/11/10 sections (rounding of theta differs)
    HTP_OPFLAGS_ROPE_KV = (1U << 15), // ROPE_NX: also writes the rotated K rows and the V rows (src[7]) to the fp16 K/V cache rows given by the I64 index tensors src[6]/src[8] (dsts[2]/[3]) (GGML_HEXAGON_ROPE_KV)
    HTP_OPFLAGS_ROPE_NORM = (1U << 14), // ROPE_NX: the A (src[0], weight src[4]) and B (src[3], weight src[5]) RMS_NORM+MUL run first on the loaded rows; eps = kernel_params[0] (GGML_HEXAGON_ROPE_NORM)
    HTP_OPFLAGS_FA_EXP_V2 = (1U << 15), // FLASH_ATTN_EXT: softmax exp2 through hvx_vec_exp2_f16_v2 (GGML_HEXAGON_FA_EXP_V2, numerics-changing)
    HTP_OPFLAGS_GDN_GATE_FOLD = (1U << 16), // GATED_DELTA_NET: src[3] = alpha+dt rows (view of the GDN_GATE input), src[4] = raw beta rows, src[6] = ssm_a; softplus*A and sigmoid run in phase 1a (GGML_HEXAGON_GDN_GATE_FOLD)
    HTP_OPFLAGS_MM_A8W8 = (1U << 31), // GGML_HEXAGON_MM_A8W8: SWIGLU gate/up runs u8 x int8 HMX jobs (htp/hmx-a8w8.h): Q4M weights expanded per chunk, dynamic per-row u8 activations, 4-plane readout
    HTP_OPFLAGS_LANE_JOIN = (1U << 11), // join lane B (wait for its outstanding op) before this op is dispatched
};
// LossyGdn (GGML_HEXAGON_LOSSY_GDN, numerics-changing, default 0). Host mask bits 0..7 -> GATED_DELTA_NET flags 17..24,
// host mask bits 8..13 -> SSM_CONV_CHAIN flags 25..30 (bit n of the host mask = HTP_LOSSY_*).
#define HTP_OPFLAGS_LOSSY_GDN_SHIFT  17
#define HTP_OPFLAGS_LOSSY_GDN_MASK   (0xffu << HTP_OPFLAGS_LOSSY_GDN_SHIFT)
#define HTP_OPFLAGS_LOSSY_CONV_SHIFT 25
#define HTP_OPFLAGS_LOSSY_CONV_MASK  (0x3fu << HTP_OPFLAGS_LOSSY_CONV_SHIFT)
#define HTP_LOSSY_GDN_DECAY_V2    (1u << 0) // GDN phase 1b decay matrix: hvx_vec_exp2_f16_v2 (corrected polynomial) instead of the qf16 Horner exp2
#define HTP_LOSSY_GDN_SOLVE_B16   (1u << 1) // GDN phase 2: forward substitution of each 32x32 block keeps only the 16 rows before a group of 8 (band)
#define HTP_LOSSY_GDN_SOLVE_B8    (1u << 2) // same, 8 rows (wins over B16)
#define HTP_LOSSY_GDN_DLAST_K     (1u << 3) // GDN: decay_last[s] is applied to the K rows of the S-update operand (phase 1b) instead of to delta (phase 4)
#define HTP_LOSSY_GDN_STATE_F16   (1u << 4) // GDN phase 6: S*lambda + update in fp16 on the f16 state copy; the f32 state is rebuilt in the last chunk
#define HTP_LOSSY_GDN_VPRIME_F16  (1u << 5) // GDN phase 3: (v - v_inter) * beta in fp16
#define HTP_LOSSY_CONV_RSQRT2     (1u << 0) // chain q/k norm: 2 Newton iterations of the rsqrt instead of 3
#define HTP_LOSSY_CONV_RSQRT1     (1u << 1) // chain q/k norm: 1 Newton iteration (wins over RSQRT2)
#define HTP_LOSSY_CONV_NORM_FUSE  (1u << 2) // chain q/k norm: out = x * (rsqrt * scale) + bias (one multiply and one qf32 conversion fewer)
#define HTP_LOSSY_CONV_SILU_H16   (1u << 3) // chain SiLU (needs C2_SILU>=3): y * sigmoid(y) as one fp16 multiply on the packed vectors

// Op descriptor
struct htp_op_desc {
    uint32_t opcode;                    // GGML/HTP Op
    uint32_t flags;                     // Op flags
    int32_t  params[HTP_OP_MAX_PARAMS]; // Params for the op, e.g. epsilon of RMS norm
    int32_t  kernel_params[HTP_OP_MAX_KERN_PARAMS]; // generic blob for host-precomputed parameters
    uint16_t src[HTP_OP_MAX_INPUTS];    // Input tensors indices
    uint16_t dst[HTP_OP_MAX_OUTPUTS];   // Output tensor indices
    uint16_t pad[2];                    // padding to align to 64 bits
};

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_TRACE_MAX_EVENTS 256

enum htp_profiler_mode {
    HTP_PROF_DISABLED = 0,
    HTP_PROF_BASIC    = 1,
    HTP_PROF_PMU      = 2,
    HTP_PROF_TRACE    = 3,
};

enum htp_trace_event_id {
    HTP_TRACE_EVT_DMA                 = 0,
    HTP_TRACE_EVT_L2FLUSH             = 1,
    HTP_TRACE_EVT_INIT                = 2,
    HTP_TRACE_EVT_BUFF                = 3,
    HTP_TRACE_EVT_FENCE               = 4,

    HTP_TRACE_EVT_HVX_COMP            = 20,
    HTP_TRACE_EVT_HVX_A_QUANT         = 21,
    HTP_TRACE_EVT_HVX_A_PREP          = 22,
    HTP_TRACE_EVT_HVX_W_DEQUANT       = 23,
    HTP_TRACE_EVT_HVX_W_PREP          = 24,
    HTP_TRACE_EVT_HVX_O_PROC          = 25,
    HTP_TRACE_EVT_HVX_FA_QK           = 26,
    HTP_TRACE_EVT_HVX_FA_SFM          = 27,
    HTP_TRACE_EVT_HVX_FA_Q_PREP       = 28,
    HTP_TRACE_EVT_HVX_FA_K_PREP       = 29,
    HTP_TRACE_EVT_HVX_FA_V_PREP       = 30,

    HTP_TRACE_EVT_HVX_GDN_PREP        = 31,
    HTP_TRACE_EVT_HVX_GDN_SOLVE       = 32,
    HTP_TRACE_EVT_HVX_GDN_V_PREP      = 33,
    HTP_TRACE_EVT_HVX_GDN_D_PREP      = 34,
    HTP_TRACE_EVT_HVX_GDN_OUT         = 35,
    HTP_TRACE_EVT_HVX_GDN_STATE       = 36,
    HTP_TRACE_EVT_HVX_GDN_REM         = 37,

    HTP_TRACE_EVT_HMX_COMP            = 40,
};

struct htp_trace_desc {
    uint32_t cycles;  // lower 32-bits of cycle counter
    uint16_t id;      // Event ID
    uint16_t info;    // bit 15: is_stop. bits 14-0: tile/chunk index or other metadata.
};

#define HTP_PROF_PMU_NCNT 8

// Profile descriptor
struct htp_prof_desc {
    uint32_t opcode;                 // GGML/HTP Op
    uint32_t usecs;                  // Number of usec
    uint32_t cycles_start;           // Start cycle counter
    uint32_t cycles_stop;            // Stop cycle counter
    uint32_t pmu[HTP_PROF_PMU_NCNT]; // PMU counters
};

struct htp_opbatch_req {
    uint64_t seq;         // Sequence number
    uint32_t n_bufs;      // Number of buffers
    uint32_t n_tensors;   // Number of tensors
    uint32_t n_ops;       // Number of ops
    uint32_t n_traces;    // Number of trace descriptors per thread
    // struct htp_buf_desc  bufs[];    -- dspqueue buf 0
    // struct htp_tensor    tensors[]; -- dspqueue buf 0
    // struct htp_op_desc   ops[];     -- dspqueue buf 0
};

struct htp_opbatch_rsp {
    uint64_t seq;            // Sequence number
    uint64_t cycles_start;   // Start cycle counter
    uint64_t cycles_stop;    // Stop cycle counter
    uint32_t status;         // HTP_STATUS_...
    uint32_t n_bufs;         // Number of buffers
    uint32_t n_tensors;      // Number of tensors
    uint32_t n_ops;          // Number of op profile descriptors
    uint32_t usecs;          // Number of usec
    uint32_t n_traces[HTP_MAX_NTHREADS + 1];
    // struct htp_prof_desc profs[];  -- dspqueue buf 0
};

#endif /* HTP_OPS_H */
