# Adaptive Per-Stage Axis Planner for Multi-NSP (AUTOSP-100680)

**Branch:** `dev/ebateni/seq-parallel-prototype` (`/local/mnt2/workspace2/llama.cpp.qualcomm`, base
`d2e54583c`). **4-core target device:** `1b531bdd`.

---

## Thesis

Instead of a fixed split *mode* (tensor / row / layer / sequence — each pins one axis everywhere),
choose at **each stage** the split axis that (a) keeps all N cores busy and (b) needs the least
cross-NSP communication, inserting a reshard only at the boundaries where the optimal axis must
change. This subsumes every fixed mode and, by matching the best axis at every stage simultaneously,
can beat each fixed mode (every fixed mode is forced into a suboptimal axis for *some* stage).

Motivating evidence: moving row-split's axis from **tokens → KV-head** gave a large prefill jump,
because head is the no-reshard axis for attention while token-split forces a K/V gather. Generalize
that choice to every stage.

**Unique gain (not reachable by any fixed mode):** minimum-communication prefill — attention on the
head axis (no gather), projections on the hidden axis, norm/residual/RoPE on the token axis (fully
local), paying only the cheapest reshards at transitions. Target: **≥ row on prefill, strictly
> tensor, at row's 1× weight memory** (weights stay sharded, never mirrored).

---

## Two phases

The planner is built in two phases because prefill and decode have **different optimal plans** (the
token axis is live in prefill, degenerate in decode) and different bottlenecks (prefill = compute +
reshard comms; decode = weight/KV bandwidth + collective latency). The backend already distinguishes
the two regimes via `GGML_META_COMM_DECODE_TOKENS` (`ggml-backend-meta.cpp`), so the planner can emit
a different axis assignment per regime.

- **Phase 1 — adaptive split for PREFILL** (`n_tokens > 1`). Token axis is live. Plan: attention on
  head, projections on hidden, norm/residual/RoPE on token (local), min-comm reshards. Target: ≥ row,
  > tensor, 1× weight memory. This is where the unique prefill win lives.

- **Phase 2 — adaptive split for DECODE** (`n_tokens == 1`). Token axis is dead (1 token can't split
  4 ways), so the planner selects among the **head / hidden** axes and — crucially — the **per-stage
  reduction style** (row-style on-chip reduce vs tensor-style cross-session allreduce). A fixed mode
  is locked into one reduction style for the whole block; the planner can use **row-reduce where row
  wins (attention/attn_output) and tensor-shard where tensor wins (FFN)**, reshard only at the one
  boundary that pays off → potentially **faster than both row and tensor**. Composes with the already
  validated decode-gated **collective isolation** TG gain (`GGML_META_COMM_ISOLATION`, +6–20% TG,
  commit `ca34eb0da6` / AUTOSP-111237): the few collectives the plan does emit run on the cheap
  decode-gated path, so the two levers multiply.

Phase 1 first (clear unique win, token axis); Phase 2 second (reduction-style switching + isolation).

---

## History of the existing splits (for context)

All current multi-NSP modes partition **one fixed axis** and keep exactly **one weight replica**
(partitioned across cores, 1× total memory):

| Mode | Enum | Weight axis | KV axis | Attention axis | Per-layer comm | 4c/1c PP |
|---|---|---|---|---|---|---|
| **layer** | `LLAMA_SPLIT_MODE_LAYER` | disjoint layers/core | per-layer | — | pipeline handoff | additive, pipelined |
| **row** | `LLAMA_SPLIT_MODE_ROW` | rows, 1 logical dev (bracket `HTP0[0-3]`) | head | head | on-chip reduce (cheap) | **>3× (best)** |
| **tensor** | `LLAMA_SPLIT_MODE_TENSOR` | heads (`AXIS_1`) | head (`AXIS_0`) | head | **2 allreduces/layer** (`attn_output`, `ffn_down`) | 2–3× |
| **tensor-scatter** | tensor + `GGML_HEXAGON_AR_SCATTER=1` | heads | head | head | reduce-scatter allreduce | ~tensor |
| **sequence (this branch, broken)** | `LLAMA_SPLIT_MODE_SEQUENCE` | **mirrored (N×)** | position (token) | Q-slice × full K/V | **2 K/V gathers/layer** | ~0.25× (regression) |
| **2D: layer-row / tensor-row** | 2 logical devs `HTP0[0-1:0],HTP1[2-3:0]` | outer layer/tensor × inner row | — | — | outer + inner | validated |

Key facts established while building sequence mode (see `sequence-context-parallelism.md`):
- tensor/row split KV **by head** (`cache_k/v → AXIS_0`); each core keeps **all positions** → KV/core
  does not shrink with context; long-context OOM-prone. Per-core KV is already 1/N (head-split), so
  position-split gives **no** KV-memory advantage over tensor.
- tensor pays an allreduce at the `AXIS_0 × AXIS_0 → PARTIAL` matmul boundary; this is the scaling
  wall (~14% of 4c prefill, DDR-bound; `MULTI-CORE-ALLREDUCE.md`).
- Pure sequence mirrors weights → **N× total weight memory** (strictly worse than the above); its only
  theoretical edge is removing the allreduce, which the broken 2-gather/layer implementation squanders.
- At 4 cores on mainstream GQA models (`n_kv_heads ≥ 4`) pure sequence has **no unique win** — the
  honest conclusion that motivated this pivot to an adaptive planner.

---

## Why the Meta backend is already 90% of the way there

The Meta backend (`ggml/src/ggml-backend-meta.cpp`) **already** assigns a per-tensor split axis and
inserts a collective only at axis-transition boundaries. Two pieces:

1. **Seed policy** — `llama_meta_device_get_split_state` (`src/llama-model.cpp:383`): a regex table
   that assigns the axis of each *named* weight/cache/input tensor. This is where a "mode" is encoded
   (tensor pins `attn_q→AXIS_1`, `cache→AXIS_0`, etc.). **This is the only thing that is fixed-per-mode.**
2. **Propagation handlers** — `ggml_backend_meta_get_split_state` dispatch (`:520–890`): for every op
   (`handle_mul_mat`, `handle_flash_attn_ext`, `handle_per_row`, `handle_rope`, `handle_reshape`,
   `handle_get_rows`/`set_rows`, …) they compute the output axis from input axes and encode **what is
   legal**. e.g. `handle_mul_mat`: `AXIS_1 weight × MIRRORED act → AXIS_0`; `MIRRORED × AXIS_1 →
   AXIS_1`; `AXIS_0 × AXIS_0 → PARTIAL` (needs allreduce). `handle_flash_attn_ext`: `Q AXIS_2 (head) +
   K/V AXIS_2 → AXIS_1`, or (added this branch) `Q AXIS_1 (token) + K/V MIRRORED → AXIS_1`.
3. **Collective insertion** — graph-compute boundary loop (`:~2208 boundary`, `:~2470 invoke`): starts
   a new subgraph at a `PARTIAL` node (→ `comm_allreduce`) or a `GATHER` node (→ `comm_allgather`,
   added this branch), computes the collective, continues.

So "adaptive per-stage axis" = **replace the fixed seed policy with a cost-driven axis assignment**,
reusing the existing legality (propagation) and reshard (collective) machinery. It is a *policy +
cost model* change, not a new backend.

---

## Design: min-communication axis assignment

### Axis vocabulary (already exists, `ggml-backend.h`)
`AXIS_0` (rows/hidden-in), `AXIS_1` (cols/hidden-out, also the token dim of activations), `AXIS_2`
(head/batch), `AXIS_3`, `MIRRORED`, `PARTIAL` (needs allreduce), `GATHER` (needs all-gather, added).

### Per-stage optimal axis (the target assignment, weights kept SHARDED = 1× memory)
| Stage | Optimal axis | Why minimal comm |
|---|---|---|
| token embedding / input | token (`AXIS_1` act) | seed; local |
| LayerNorm / RMSNorm / residual add / RoPE | token (`AXIS_1`) | fully local (`handle_per_row` allows AXIS_1, `handle_rope`/`bin_bcast` pass it) |
| QKV / FFN-up / gate projections | weight cols (`AXIS_1`) | activation stays token or mirrored; output head/col-sharded, no reduce |
| **attention (flash-attn)** | **head (`AXIS_2`)** | **no cross-head comm — the row insight** |
| attn_output / FFN-down projections | weight rows (`AXIS_0`) | output is `PARTIAL` → one reduce, OR keep token-split to defer |
| output / logits | token (`AXIS_1`) → gather at the very end | one final gather |

The planner's job: assign seeds so that **as many transitions as possible are free** (token↔token,
or axis preserved), and the unavoidable transitions pick the **cheapest** collective:
- token→head at attention entry: a reshard (transpose/gather of Q,K,V) — but K/V on head is the
  *cheap* side (small `n_embd_kv`), far less than hidden-state allreduce.
- head→token after attention: reshard back.
- The remaining reduce for FFN-down/attn_output: an allreduce of the hidden state — **unless** the
  activation is token-split there, in which case it is a no-reduce `MIRRORED×AXIS_1→AXIS_1`.

### Cost model
Attach a scalar comm cost to each possible axis-transition at each boundary:
- `token→token`, `axis preserved`, `MIRRORED×AXISk`: **0** (no collective).
- `GATHER` (all-gather) of tensor T: `≈ (N-1)/N · bytes(T)` one-way copy.
- `PARTIAL→MIRRORED` (allreduce) of tensor T: `≈ 2·(N-1)/N · bytes(T)` (reduce + broadcast; less with
  reduce-scatter).
- transpose/reshard within a device: cheap but nonzero.
Choose the seed assignment minimizing total boundary cost subject to "every op N-way parallel".

### Phase 1 MVP (PREFILL) — do this first
Rather than a full ILP solver, start with a **hand-authored optimal plan** that is already the known
min-comm assignment for a standard transformer block, exposed as a new mode
`LLAMA_SPLIT_MODE_AUTO` (or extend `sequence`):
- norm / residual / RoPE / token-wise ops → token (`AXIS_1`)
- projections → hidden axes (as tensor does)
- **attention → head (`AXIS_2`)** (reuse tensor's proven attention path — this is the big win vs
  pure sequence, which forced token-split attention + gather)
- FFN-down / attn_output → keep token-split output (no allreduce) where legal, else one allreduce
- weights **sharded, never mirrored** → 1× memory

This is essentially **tensor-split's attention + projections, but with norm/residual/token ops split
on the token axis instead of mirrored**, removing redundant per-core norm work and setting up
no-allreduce FFN. It is the smallest step that could beat tensor while matching its memory.

### Phase 2 (DECODE) — reduction-style switching + isolation
At `n_tokens == 1` the token axis is degenerate, so Phase 2 chooses among head/hidden axes and the
**reduction style per stage**:
- **Attention + attn_output region → row-style** (bracket / on-chip reduce) — this is why row scales
  >3× vs tensor's 2–3× at decode; keep that cheap reduction here.
- **FFN region → whichever of row/tensor reduction is cheaper** for that model's FFN shape; reshard
  (one transition) only if the switch pays for itself.
- All emitted collectives run under decode-gated **`GGML_META_COMM_ISOLATION`** (already validated,
  +6–20% TG) so each is cheap.
- GQA edge case: if `n_kv_heads < cores`, allow a **KV-position** split for attention so all cores
  participate (head-split would strand cores) — the one place a non-head axis helps decode.

MVP for Phase 2: a static decode plan = "row reduction for attention, tensor/row for FFN, isolation
on", gated on `ggml_nrows(tensor) <= GGML_META_COMM_DECODE_TOKENS`. Later: cost-model picks per-stage.

---

## Where the code changes go

1. **`src/llama-model.cpp` `llama_meta_device_get_split_state` (`:383`, policy `get_tensor_config`
   `:486`):** add an `AUTO`/adaptive branch that returns the per-stage optimal seed axes above
   (keyed off `ud->split_mode`). This is the primary change — the min-comm assignment lives here.
   Keep weights on their sharded axes (reuse tensor's `AXIS_0/1` seeds); set norm/residual/token
   tensors to `AXIS_1`; keep `cache_k/v` on head (`AXIS_0`) so attention stays head-split.
   **Phase 1 vs Phase 2 is one policy keyed on regime:** the prefill plan (token-live) and decode plan
   (reduction-style switching) are selected by the same decode gate the backend already uses
   (`ggml_nrows <= GGML_META_COMM_DECODE_TOKENS`). The seed policy here is static per tensor; the
   regime split actually happens in the backend (next item), since the same tensor can take different
   axes in the prefill vs decode graph.

2. **`ggml/src/ggml-backend-meta.cpp` propagation handlers (`:520–890`):** verify each op on the
   chosen plan has a legal transition; add branches only where a needed (axis-in → axis-out) pair is
   currently unhandled (e.g. a token-split residual meeting a head-split attention output needs a
   defined reshard). The `handle_flash_attn_ext` token/head branches and `handle_get_rows` gather
   branch added on this branch already cover the sequence pieces; the planner mostly reuses tensor's
   existing legal transitions.

3. **Cost model + planner (new, optional for MVP):** a small pass that, given the graph, assigns
   seeds by min-cost. For MVP this is a static table (the hand-authored plan); later a real
   cost-minimizing pass over the propagation legality.

4. **Collective path (`:~2208` boundary, `:~2470` invoke):** already handles `PARTIAL→allreduce` and
   `GATHER→allgather`. If a transition needs a **transpose/reshard** collective (token↔head), add a
   third kind (`COLLECTIVE_RESHARD`) mirroring the `collective_kind` tag added this branch, backed by
   a Hexagon-side reshard op (per the prefer-Hexagon directive).

5. **`include/llama.h` + `common/arg.cpp` + `tools/llama-bench/llama-bench.cpp`:** add the new mode
   name (`auto`) to the enum, the common CLI parser, and the bench CLI parser/`split_mode_str` (same
   three spots `sequence` was added).

6. **Harness `llama.cpp-build-and-test.sh`:** add `auto` to the `--split` allowlist (plain device
   list `HTP0,HTP1,…` + `-sm auto`), like `sequence`.

---

## Expected gains vs fixed modes (4-core)

### Phase 1 — PREFILL (PP)
| mode | 4c/1c PP | weight mem (total) | KV/core | note |
|---|---|---|---|---|
| row | >3× | 1× | 1/N (head) | best today, on-chip reduce |
| tensor | 2–3× | 1× | 1/N (head) | 2 allreduces/layer |
| pure sequence (broken) | ~0.25× | **N×** | 1/N (pos) | abandoned |
| **adaptive (target)** | **≥ row, > tensor** | **1×** | 1/N | min-comm: head attn + token norm/residual, fewest reshards |

### Phase 2 — DECODE (TG)
| mode | 4c/1c TG | note |
|---|---|---|
| row | best today | on-chip reduce in attention region |
| tensor | < row | cross-session allreduce |
| tensor + isolation | +6–20% over tensor | validated (`ca34eb0da6`) |
| **adaptive (target)** | **≥ row, > tensor+isolation** | per-stage reduction-style switch (row-reduce attn + best FFN) × decode-gated isolation on emitted collectives |

Token axis is dead at n_tokens=1, so the decode win is **not** from a new axis — it is from (a)
choosing the cheaper reduction style per stage instead of being locked to one, and (b) composing with
the isolation TG gain. GQA `n_kv_heads < cores` is the one case the planner also unlocks extra
decode parallelism (KV-position split) that head-split cannot.

- **Prefill:** the unique win — match the best axis at every stage, so no single stage is forced
  suboptimal; removes tensor's redundant mirrored-norm work and minimizes reshards. Target ≥ row.
- **Decode (TG):** ≈ 1-core / ≈ tensor; token axis offers nothing at 1 token. Not a goal.
- **Memory:** **1× weights** (sharded) — this is the correction that makes it viable vs pure
  sequence. KV 1/N (same as tensor).
- **Long context:** attention stays head-split (like tensor), so no new long-context *capacity* win
  over tensor; the win is purely the min-comm prefill speed. (A separate ring/context-parallel KV
  split remains the only path to long-context capacity beyond head-count, but it is orthogonal and
  not this plan.)

---

## Build / verify

- Build: `sh ./scripts/buid_in_docker.sh linux`; deploy via the fixed `do_deploy` (tar, preserves
  symlinks) to `pkg-bat`.
- A/B: harness `--split auto,row,tensor` at `CORES=1,4`, `PP_TOKENS=1024,4096,8192`, iq/cookie/needle
  gates. Success = `auto` 4c/1c ≥ tensor (ideally ≥ row) with identical quality and 1× weight memory.
- Smoke on `1b531bdd`: llama-3.2-3b, Qwen3-4B, Qwen3-VL-4B.

---

## Status of current branch (what to keep / revisit)

Keep (reusable): `AXIS_GATHER` sentinel + `HTP_OP_ALLGATHER` kernel + `comm_allgather` + meta
boundary/invoke; the `handle_flash_attn_ext` token branch; the mode plumbing; the deploy tar fix;
harness `sequence`/`--split` support; bench `-sm` parser fix.

Revisit: the pure-`sequence` seed policy (mirrored weights, token-split attention with per-layer K/V
gather) is the regressing path — superseded by the adaptive plan, which keeps weights sharded and
attention head-split. Do not ship pure sequence as-is.
