# Hexagon VMEM Overclaim Fixes

## Overview

Two separate VMEM overclaim problems were identified and fixed in the Hexagon backend.
Both produced the same error message but from different allocation paths:

```
ggml-hex: HTP0 op does not fit into empty batch (vmem/tensor/buffer limit exceeded)
```

The Hexagon backend maps tensor backing allocations through shared-memory FDs. The key
distinction is between **tensor payload size** and **physical backing allocation size**.
If a 10 MiB tensor is a subrange of a 994 MiB allocation, mapping that tensor requires
the full 994 MiB FD. The admission check (`fit_op` in `ggml-hexagon.cpp`) correctly
accounts for the full backing allocation — which is the right behavior for ordinary
compute tensors, but was incorrect for the two special cases fixed here.

The operation VMEM limit is `HTP_OP_MAX_VMEM_DEFAULT = 3,355,443,200` bytes (~3.1 GB).

---

## Fix 1: Collective Communication Tensor Isolation

**Branch:** `dev/ebateni/kv-vmem-fix`  
**Toggle:** `GGML_META_COMM_ISOLATION` (unset or `1` = enabled, `0` = disabled)

### Problem

In 4-core tensor-split mode, the allreduce operation (`enqueue_allreduce`) maps peer
tensors from all 4 ranks simultaneously. The collective boundary tensors
(`linear_attn_out`, `ffn_out`) had small payloads (~10 MB) but lived inside the large
per-rank compute backing allocation (~994 MB). The admission check charged the full
backing size per rank:

```
4 ranks × ~994 MB = ~3.97 GB > 3.1 GB vmem limit → abort
```

This affected 4-core tensor-split at `ub=1024`. At `ub=512` the smaller batch reduced
the compute arena enough to fit. The abort was in `enqueue_allreduce` →
`enqueue_op` → `ggml_abort`.

### Investigation

Backtrace confirmed the abort site:
```
ggml_backend_meta_graph_compute
  → ggml_backend_hexagon_comm_allreduce_tensor
    → ggml_hexagon_session::enqueue_allreduce
      → ggml_hexagon_session::enqueue_op
        → empty-batch VMEM abort
```

COMM-DIAG instrumentation in `enqueue_allreduce` confirmed `sbuf_size=1042284544`
(~994 MB) for all 4 peer tensors, causing the overflow.

The existing `GGML_TENSOR_FLAG_COMMUNICATION` flag and communication-class chunk logic
in `ggml_dyn_tallocr_alloc` were present but ineffective: collective boundary tensors
have `tensor->data` set before gallocr runs (set to a dummy sentinel in
`ggml_backend_meta_alloc_ctx_tensors_from_buft`), so gallocr skips them entirely and
they always land in the large compute buffer.

### Fix

In `ggml_backend_meta_buffer_init_tensor_impl` (`ggml-backend-meta.cpp`), collective
boundary tensors are detected using `split_state_async` (`assume_sync=false`), which
returns `GGML_BACKEND_SPLIT_AXIS_PARTIAL` for allreduce boundary tensors — the same
detection used in `ggml_backend_meta_graph_compute`. For these tensors, instead of
assigning `bufs[j]` (the large compute buffer), a compact per-tensor per-rank buffer
is allocated on demand (`comm_bufs[j]`) sized to the actual tensor payload:

```cpp
// detect collective boundary (same as graph_compute does)
const ggml_backend_meta_split_state split_state_async =
    ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync=*/ false);
const bool is_collective =
    (split_state_async.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) &&
    (getenv("GGML_META_COMM_ISOLATION") == nullptr ||
     atoi(getenv("GGML_META_COMM_ISOLATION")) != 0);

if (is_collective) {
    // allocate compact buffer for this tensor on this rank
    size_t payload = GGML_PAD(ggml_backend_buft_get_alloc_size(simple_buft, t_ij), alignment);
    if (!buf_ctx->comm_bufs[j] || buf_ctx->comm_bufs[j].size() < payload)
        buf_ctx->comm_bufs[j] = ggml_backend_buft_alloc_buffer(simple_buft, payload);
    simple_buf = buf_ctx->comm_bufs[j].get();
}
t_ij->buffer = simple_buf;
t_ij->data   = ggml_backend_buffer_get_base(simple_buf);  // starts at base
```

`comm_bufs` is a new `std::vector<ggml_backend_buffer_ptr>` added to
`ggml_backend_meta_buffer_context`, resized lazily when a collective tensor is first
encountered.

### Result

- `sbuf_size` for `linear_attn_out` dropped from ~994 MB to ~12 MB
- 4-core tensor-split `ub=1024` runs without abort: `pp1024 = 2029 t/s` for Qwen3.5-4B
- `ub=512` fallback no longer needed

### Files changed

- `ggml/src/ggml-backend-meta.cpp`: `comm_bufs` field, `is_collective` detection,
  compact buffer allocation in `ggml_backend_meta_buffer_init_tensor_impl`

---

## Fix 2: KV Cache K/V Allocation Affinity

**Branch:** `dev/ebateni/kv-vmem-fix`  
**Toggle:** `LLAMA_KV_ALLOC_AFFINITY` (unset or `1` = enabled, `0` = disabled)

### Problem

For long prompts (32k+ tokens), the KV cache is split across multiple physical backing
buffers by the context range allocator (`ggml_backend_alloc_ctx_tensors_from_buft_impl`).
If the buffer split boundary fell **between** the K and V tensors of the same transformer
layer, the attention operation for that layer needed to map three buffers simultaneously:

```
compute buffer (~317 MB)
+ KV buffer N   (~2,099 MB)   ← contains cache_k_l12
+ KV buffer N+1 (~2,099 MB)   ← contains cache_v_l12
= ~4.5 GB > 3.1 GB vmem limit → abort
```

Earlier layers worked because their K and V were in the same physical KV buffer
(~2.4 GB total ≤ limit). The abort was in `enqueue_op` during the first attention
layer whose K/V pair straddled a buffer boundary.

### Investigation

Verbose logging confirmed the buffer layout:

```
KV backing FD 18:  2,099,249,152 bytes  (contains cache_k_l12, end of buffer)
KV backing FD 19:  2,099,249,152 bytes  (contains cache_v_l12, start of buffer)

Attention op maps: compute(316MB) + FD18(2099MB) + FD19(2099MB) = 4515MB > limit
```

The split occurred because the range allocator (`ggml_backend_alloc_ctx_tensors_from_buft_impl`)
treated tensors individually, placing a physical buffer boundary at a size threshold
without regard to which tensors belong together.

### Fix

Two changes:

**1. Tensor allocation metadata** (`ggml.h`): The existing 8-byte `padding` field in
`ggml_tensor` is reused (without changing `sizeof(ggml_tensor)`) for an anonymous union:

```c
struct ggml_tensor_alloc_metadata {
    uint32_t alloc_group;    // non-zero = atomic placement group
    uint32_t alloc_reserved;
};
union {
    char padding[8];
    struct ggml_tensor_alloc_metadata alloc;
};
```

Group `0` means ungrouped (historical behavior). Group `il+1` (layer index + 1) means
K and V of that layer must land in the same physical buffer.

**2. Per-layer K/V group assignment** (`src/llama-kv-cache.cpp`): After K and V tensors
are created for each layer:

```cpp
if (k != nullptr && v != nullptr) {
    k->alloc.alloc_group = il + 1;
    v->alloc.alloc_group = il + 1;
}
```

**3. Group-aware context range allocator** (`ggml-alloc.c`): The loop in
`ggml_backend_alloc_ctx_tensors_from_buft_impl` is changed from a simple per-tensor
iteration to a group-aware one. A non-zero `alloc_group` means consecutive same-group
tensors form an atomic unit — the physical buffer split point is pushed before or after
the complete group, never inside it:

```c
// accumulate full group size atomically
do {
    if (tensor->data == NULL && tensor->view_src == NULL)
        group_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
    tensor = ggml_get_next_tensor(ctx, tensor);
} while (alloc_group != 0 && tensor != NULL &&
         tensor->alloc.alloc_group == alloc_group);

// flush before complete group, never inside
if (cur_buf_size > 0 && cur_buf_size + group_size > max_size) {
    alloc_tensor_range(ctx, first, group_first, buft, cur_buf_size, ...);
    first = group_first;
    cur_buf_size = group_size;
} else {
    cur_buf_size += group_size;
}
```

### Result

- Single-core 32k prompt that previously aborted now completes successfully
- At 1k context: no buffer split occurs at all (all KV fits in one buffer) →
  zero behavioral change, zero throughput impact
- At 32k context: K and V of each layer guaranteed to share one physical backing FD →
  attention maps at most 2 KV buffers per op → fits within vmem limit

### A/B validation

| Config | 32k prompt |
|---|---|
| `LLAMA_KV_ALLOC_AFFINITY=0` | abort (vmem overflow) |
| `LLAMA_KV_ALLOC_AFFINITY=1` | completes |
| Collective knob variation | no effect on KV result |

### Files changed

- `ggml/include/ggml.h`: `ggml_tensor_alloc_metadata` struct, union replacing `padding`,
  `GGML_TENSOR_FLAG_COMMUNICATION = 32`
- `ggml/src/ggml-alloc.c`: group-aware loop in
  `ggml_backend_alloc_ctx_tensors_from_buft_impl`
- `src/llama-kv-cache.cpp`: K/V group assignment per layer

---

## Infrastructure: `alloc_buffer_n`

Both fixes build on upstream PR #23671 (`alloc_buffer_n` infrastructure) merged on this
branch. That PR added:

- `ggml_backend_buft_alloc_buffer_n` / `ggml_backend_buft_get_alloc_size_n` to the
  buffer type interface — a way to allocate a list of tensors into one or more physical
  buffers with size constraints
- `ggml_backend_buft_alloc_buffer_n_plan` in `ggml-backend.cpp` — planning logic that
  respects `alloc_group` when deciding split points

Fix 2 uses this infrastructure through the group-aware `alloc_tensor_range` loop.
Fix 1 uses a complementary approach in the meta backend's tensor initialization path,
allocating compact per-tensor buffers at a different level.

---

## Combined branch

`dev/ebateni/vmem-fix-plus-head-split` merges both fixes with the head-parallel
flash_attn optimization (`dev/ebateni/row-split-optimization-clean`):

```
upstream/master
    ↓
dee1ec9b3  hexagon: fix KV cache vmem overclaim using alloc_buffer_n infrastructure
    ↓
22ec847b8  hexagon: collective tensor compact buffers + KV affinity vmem fixes
    ↓  (merge)
5ca543de9  Merge dev/ebateni/row-split-optimization-clean
           (head-parallel flash_attn + results doc)
```

**Runtime toggles:**

| Variable | Default | Effect |
|---|---|---|
| `GGML_META_COMM_ISOLATION` | 1 (on) | Fix 1: compact collective buffers |
| `GGML_META_COMM_PROMOTE` | 0 (off) | Fix 3: reconcile Fix 1 with reduce-scatter (experimental) |
| `LLAMA_KV_ALLOC_AFFINITY` | 1 (on) | Fix 2: K/V affinity grouping |
| `GGML_HEXAGON_FA_HEAD_SPLIT` | 1 (on) | Head-parallel flash_attn (row-split) |

---

## Fix 3: Fix 1 silently disables reduce-scatter

**Toggle:** `GGML_META_COMM_PROMOTE` (unset or `0` = disabled, `1` = enabled)

### Symptom

After rebasing onto upstream `a4d880fd5c` ("safe scatter mode"),
`GGML_HEXAGON_AR_SCATTER=1` produced no measurable gain. Instrumenting
`try_fuse_allreduce_add` on device `1b531bdd` (4-core tensor, llama-3.2-3b,
440 fused sites) showed a 100% flip:

| `GGML_META_COMM_ISOLATION` | fused sites |
|---|---|
| 0 | 440/440 `mode=1 n_dsts=4` (SHARDED_FANOUT) |
| 1 | 440/440 `mode=0 n_dsts=1` (FULL) |

### Root cause

Upstream gates reduce-scatter on the fused ADD being in-place over the
ALLREDUCE result (`ggml-hexagon.cpp`):

```c
const bool is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;
```

Fix 1 moves the ALLREDUCE result into a compact side buffer while the ADD stays
in the ordinary compute arena, so the two addresses can never match. Fix 1 and
reduce-scatter were therefore mutually exclusive, and since Fix 1 is on by
default, reduce-scatter was dead everywhere.

Note: the `GGML_TENSOR_FLAG_COMMUNICATION` chunk-separation path in `ggml-alloc.c`
is inert in llama.cpp. The flag is set in `ggml_backend_meta_graph_compute`, which
runs *after* `ggml_backend_sched_alloc_graph`, and llama.cpp rebuilds the graph per
token, so the flag never survives into a `ggml_gallocr_reserve` pass. The compact
`comm_bufs` override is the entire memory fix, and the entire scatter blocker.

### Approach

`GGML_META_COMM_PROMOTE=1` follows the in-place consumer into the collective
backing, so both tensors share one address again. The rule is deliberately
narrow — a plain `GGML_OP_ADD`, not a view, with a source at the same address,
same type, same shape and same byte count whose split axis is `PARTIAL`. A
broader rule (any in-place consumer) deadlocks the DSP.

Because several collective tensors now coexist, the compact backing is divided
into `GGML_META_COMM_MAX_SLOTS` slots. A promoted ADD is keyed by the address of
the collective it aliases, so it resolves to the same slot, while unrelated
collectives get their own.

### Results (device `1b531bdd`, 4-core tensor, `ub=1024`, pp1024)

| model | promote=0 | promote=1 | gain | scatter | output |
|---|---|---|---|---|---|
| llama-3.2-3b-instruct-q4_0 | 2667.78 | **3385.91** | +26.9% | 440/440 | OK |
| Qwen3-4B-Instruct_Q4 | 2219.28 | **2583.00** | +16.4% | 568/568 | OK |
| Qwen3.5-4B-Q4 | 1909.03 | **2081.78** | +9.0% | 320/512 | OK |
| Phi-3.5-mini-instruct-Q4_0 | 2392.96 | **2533.22** | +5.9% | 248/504 | OK |
| gemma-2-2b_Q4_0 | 3301.46 | 3368.88 | +2.0% | none fused | unaffected |
| Qwen3.5-0.8B-Q4_0 | 4482.00 | 4966.94 | +10.8% | 240/384 | **CORRUPT** |

No `does not fit` aborts in either mode, so the Fix 1 memory saving is fully
preserved — the three models that abort without Fix 1 still run.

### Why it is off by default

Qwen3.5-0.8B (hybrid linear/full attention) produces garbage with
`GGML_META_COMM_PROMOTE=1`. The corruption is independent of
`GGML_HEXAGON_AR_SCATTER` and does not occur at 1 core, so it comes from the
relocation itself, not from reduce-scatter.

Relocating a tensor out of the gallocr arena while gallocr keeps reasoning about
arena addresses cannot be made sound by address matching alone: a tensor that
merely *overlaps* a relocated range, rather than sharing its exact address, is
not detected. Alternatives that were measured and rejected:

- **Per-slot buffers** (one exactly-sized buffer per collective, unlimited
  slots): stalls model load — each HTP buffer costs an rpcmem allocation plus a
  DSP mapping, and hundreds of them never finish.
- **Round-robin slots keyed by tensor name**: fixes Qwen3.5-0.8B but breaks
  llama-3.2-3b even with promotion disabled (`get-tensor-async failed`).
- **Declining the fusion** so the standalone ALLREDUCE (which always scatters)
  runs instead: correct, but the scatter gain is exactly cancelled by the loss of
  ADD fusion (pp 2904.50 vs 2901.14 baseline).

The structurally correct fix is to mark collectives with
`GGML_TENSOR_FLAG_COMMUNICATION` *before* `ggml_backend_sched_alloc_graph`, so
gallocr places them and their in-place consumers in a separate chunk itself and
no manual relocation is needed. That requires a marking hook ahead of
allocation and is not implemented here.

---

## Test results

### Fix 1: 4-core tensor-split Qwen3.5-4B Q4, device `1b531bdd`

| ubatch | before | after |
|---|---|---|
| 512 | PP=1857 t/s (fallback) | PP=1857 t/s |
| 1024 | abort | **PP=2029 t/s** |

### Fix 2: single-core long context Qwen3-0.6B Q4, device `1b531bdd`

| depth | before | after |
|---|---|---|
| 8k | PP=1435 t/s | PP=1435 t/s |
| 20k | PP=752 t/s | PP=752 t/s |
| 32k | abort | **PP=524 t/s** |

Note: 32k PP at `ub=512` (harness fallback from 1024) — the abort was in the vmem
admission check before any compute, not in flash_attn. Fix 2 resolves the admission
check; `ub=1024` at 32k hits a separate vmem limit from the large batch size itself.
