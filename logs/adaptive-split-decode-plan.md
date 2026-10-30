# Adaptive per-stage split for decode on Hexagon mdev — plan and work log

AUTOSP-100680, Phase 2 (decode; prefill is Phase 1, [`adaptive-split-plan.md`](adaptive-split-plan.md)).
Branch `ebateni/adaptive-split`. Device `1b531bdd`, 4 NSPs, row split
(`GGML_HEXAGON_DEVICES=HTP0[0-3]`).

This file is the living plan and history. It is updated and pushed after every step.
Code references are relative to `ggml/src/ggml-hexagon/` on this branch.

_Last updated: 2026-10-30 — initial plan from a code reading and the TG numbers already in `logs/`. Nothing in this file has been measured specifically for decode yet. D0 (decode profiling) comes first._

---

## 1. Goal and principles

**Goal:** 4-core decode (TG) scaling well above today's ~2.0x in row split. It must hold for deep
context, MoE models, special models (Gemma 4, Qwen3.5, Nemotron) and small batches (speculative
decoding verify, `-np`).

**Principles:** the same as Phase 1.
1. Change the Hexagon side. Change the Meta backend / `src/` only when necessary and approved.
2. Redundant execution of an op on all NSPs is acceptable when it is cheaper.
3. If a stage's best axis never depends on the scenario, hardcode it. If it does depend on the
   scenario, give it a knob. In the last stage, tune the knobs automatically.
4. Beat row split and stay robust.

**Scope:**
- **Primary:** `n_tokens == 1` (TG).
- **Secondary:** small batches, `n_tokens` 2–256. These come from spec-decode verify (the
  Nemotron SAGA requires the LiLiCorr drafter), parallel sequences, and the prefill tail.

**Workflow:** the same as Phase 1. Claude writes the code and pushes; the owner builds, runs the
device tests and pushes results (`logs/<ts>/`). Every change sits behind a knob that defaults off,
gets an A/B in one binary (knob off == upstream), and the default flips only after the A/B passes.

### Why decode is a different problem

| | prefill (Phase 1) | decode (Phase 2) |
|---|---|---|
| bound by | HMX compute, DDR for elementwise ops | **weight + KV bytes per token**, plus fixed per-op cost |
| matmul kernel | HMX, TOKEN split | HVX GEMV (`m <= HTP_MM_HMX_MIN_NROWS = 4`, `ggml-hexagon.cpp:5483`), **FEAT split** (32-row chunks, `htp/matmul-ops.c:1693`) |
| op time | 0.1–50 ms | **a few us to ~100 us**; a 2.5 us barrier is no longer noise |
| graph time | 180–1700 ms | ~10–60 ms per token |
| host time per graph | negligible | **a visible share**: the descriptors are re-encoded every token |
| collectives | none in row split | none in row split (allreduce is tensor-split only, `ggml-hexagon.cpp:9686`) |

Row split has no reduction step in decode either. Each GEMV writes a disjoint slice of output
rows to shared DDR, and the next op reads the whole vector after a barrier. Everything the 4 NSPs
lose therefore falls into four buckets: **sync** (barrier + flush), **SOLO ops** (NSP0 works, NSP1-3
wait), **imbalance** and **host time**.

---

## 2. What the data says (P2-0, from existing runs)

### Current row split, TG 4c vs 1c

TG is `llama-bench -p 0 -n 256` at depth `TG_DEPTH` (or `depth + PP_TOKENS` with `DEPTHS`).
Values are 1c → 4c t/s (x = 4c/1c). Sources are listed per row.

| model | TG depth ~1–2k | TG depth ~18k | source |
|---|---|---|---|
| llama-3.2-3b Q4_0 (8 KV heads) | 25.2 → 51.6 (**2.05**) | 11.2 → 21.4 (**1.92**) | 144105, 171354 |
| Qwen3.5-4B Q4_0 (4 KV, GDN) | 19.4 → 38.2 (**1.97**) | 15.6 → 33.5 (**2.15**) | 144105, 171354 |
| Gemma-4-E4B Q4_0 (2 KV), FA_SPLIT=0 | 17.7 → 32.9 (1.86) | 14.4 → 24.5 (**1.70**) | 223555 |
| Gemma-4-E4B, FA_SPLIT=1 (shipped M1) | 17.1 → 35.3 (2.07) | 14.9 → 29.6 (1.99) | 223555 |
| Gemma-4-E2B Q4_0 | 31.7 → 54.4 (**1.72**) | — | 104316 |
| gemma-2-2b Q4_0 | 26.1 → 46.1 (1.77) | — | 191941 |
| Gemma-4-12B QAT Q4_0 (FA_SPLIT=1) | 7.8 → 17.2 (2.19) | — | 104316 |
| Gemma-4-26B Q4_0 (MoE), FA_SPLIT=0 / 1 | 16.5 → 22.4 (**1.36**) / 25.9 (1.57) | — | 104316 |
| Qwen3.5-35B-A3B UD-Q4_K_M (MoE) | 15.0 → 24.1 (**1.61**) | 13.8 → 19.8 (**1.44**) | 181419 |

Noise: 1c TG moves by about 4% between identical configs (E4B 17.67 vs 17.05). Use `REPS=5`.

- **Dense models stop at ~2.0x.** Prefill on the same models reaches 3.0–3.7x.
- **Small models scale worst** (E2B, gemma-2-2b at ~1.7x). They have the most ops per weight byte,
  which points at a fixed per-op or per-token cost.
- **MoE scales worst of all** (1.36–1.61x, and lower at depth).
- **FA matters at depth for 2-KV-head models.** M1 lifted Gemma d18k from 1.70x to 1.99x and
  26B from 1.36x to 1.57x.

### Rough budget (llama-3.2-3b, d1k, estimate)

The model is 1.78 GiB, almost all of it read once per token.

| | t/s | ms/token | effective weight GB/s |
|---|---|---|---|
| 1c | 25.2 | 39.7 | ~48 (one NSP alone can stream ~80–95) |
| 4c | 51.6 | 19.4 | ~99 (the 4-NSP DDR cap is ~150–185) |
| 4c at a 170 GB/s cap | ~89 | ~11.2 | 170 |

**About 8 ms of each 19.4 ms 4c token is not weight streaming.** Candidate costs:
- Sync: ~600 ops × (2.5 us barrier + flush) ≈ 2–3 ms.
- SOLO ops.
- Imbalance.
- Host encode and submit.
- The completion round trip.

There is no decode profile yet, so D0 must attribute this before any optimization is ranked.
At 4c each NSP only needs ~25–40 GB/s, so per-NSP kernel efficiency (D9) matters less than in 1c.

### What the code does at decode (findings)

**F1 — GEMV is already FEAT-split, and the weights are split 4 ways.** HVX GEMV partitions output
rows in 32-row chunks (`htp/matmul-ops.c:1693`, NX at `:479`). One copy of the weights sits in DDR;
sub-sessions map the same fd (`clone_buffer`, `ggml-hexagon.cpp:4711`), and each NSP DMAs only its
tile range (`matmul-ops.c:497`).
- Every NSP DMAs and quantizes the **full** activation into its own VTCM (`matmul-ops.c:1935`,
  NX `:4686`). This is redundant but tiny at 1 token, and acceptable under principle 2.
- **The Phase 1 "FEAT for decode" item is already done for n_tokens ≤ 4.**

**F2 — small-M HMX splits TOKENS (spec-decode verify).** For `m > 4` the HMX path splits tokens
with chunk 1 (`matmul-ops.c:3115`, host solver `ggml-hexagon.cpp:5520`).
- With 8–16 tokens every NSP reads **all** weights: 4x the weight DDR traffic of a FEAT split.
- At small M the weight bytes are far larger than the activation bytes, which is the inverse of
  baseline Finding 3. FEAT is the right axis here.

**F3 — every op pays a 4-NSP barrier.** `proc_op_req` (`htp/main.c:1188`) does three things:
- flushes the dirty inputs (`:1214`);
- marks the **whole** dst dirty, although this NSP only wrote its slice (`:1230`);
- runs `htp_mdev_group_barrier` before every op (`:1232`). The barrier is an all-to-all spin over
  128B fence slots (`htp/htp-fence.h:33`).

Per batch there is a full cache flush/invalidate at start and end (`main.c:1288`, `:1345`).
There is no barrier-skip knob (M6 `MDEV_SYNC` was never implemented) and no REPL mode.

**F4 — 1-row ops run SOLO on NSP0.** The rule in `htp_tensor_mdev_partition`
(`htp/htp-tensor.h:98`) is: fewer chunks than NSPs means everything goes to NSP0. At 1 token:

| op | decode behaviour | where |
|---|---|---|
| RMS_NORM / RMS_NORM_MUL / NORM / L2 | 1 row → **SOLO** (per-head Q/K norm splits) | `htp/unary-ops.c:1658` |
| GLU (SWIGLU/GEGLU), f32 and f16 | **SOLO** | `htp/act-ops.c:737`, `:897` |
| ADD/MUL with broadcast (row path) | **SOLO** | `htp/binary-ops.c:1117` |
| ADD/MUL same shape (CHUNKED) | splits by elements | `htp/binary-ops.c:1047` |
| ROPE | Q splits by heads; K with n_kv < 4 → **SOLO** | `htp/rope-ops.c:722` |
| GET_ROWS (embedding) | **SOLO**, 1 thread | `htp/get-rows-ops.c:445` |
| SET_ROWS (KV-cache write) | **SOLO** | `htp/set-rows-ops.c:209` |
| ARGSORT / TOP_K (MoE router) | **SOLO** | `htp/argsort-ops.c:828` |
| CPY | 1D contiguous splits; row variant **SOLO** | `htp/cpy-ops.c:290` |
| SSM_CONV | splits by channels | `htp/ssm-conv.c:562` |
| GATED_DELTA_NET | splits by heads if H ≥ 4 | `htp/gated-delta-net-ops.c:2393` |

Each SOLO op costs its own runtime on NSP0, plus a barrier during which NSP1-3 do nothing.

**F5 — FA decode leaves NSPs and threads idle, and there is no split-KV.**
- Path choice at M=1 (`ggml-hexagon.cpp:5141`):
  - HMX if DK ≥ 256, or if S ≥ the crossover (512–2048).
  - Otherwise HVX with Br=1, Bc=64.
- **HMX** (`htp/flash-attn-ops.c:1902`): there is exactly 1 Q-block.
  - n_kv % 4 == 0: plain head split, all NSPs work.
  - **n_kv = 2: 2 teams × 2, and member 1 of each team returns** (`:1927`). Only NSP0 and NSP2 work.
  - n_kv = 1: NSP0 only.
- **HVX** (`htp/flash-attn-ops.c:2554`): with n_kv < N it splits q-rows, so two NSPs stream the same
  KV head. Within an NSP, threads split q-rows (`:2584`), so `G_local` falls to 1 and every thread
  DMAs the same K/V (`:314`). That can be up to G× redundant K/V traffic.
- No split-KV / flash-decoding / LSE combine exists anywhere. Every NSP and thread loops over all
  KV blocks.
- Why this matters at depth: at d18k, llama's F16 KV is ~75 MB per layer per token (2.1 GB per
  token), which is **more than the weights**. KV bytes, not weights, dominate deep-context decode.

**F6 — MoE decode is HVX MUL_MAT_ID with rows split inside each expert.** HMX is excluded for
`n_tokens <= 4` (`ggml-hexagon.cpp:5485`).
- `hvx_mv_id` (`matmul-ops.c:1360`) loops over all active experts. Each NSP takes a 32-row slice of
  each one (`:4433`), and threads round up to 32 rows (`:4451`).
- Small expert FFNs therefore give each NSP only a few chunks. Example: an expert with 512 rows gives
  128 rows per NSP, which is 4 thread-chunks for 6–8 HVX threads. The DMAs are short and threads sit
  idle.
- The expert/LPT splits (`GGML_HEXAGON_MMID_SPLIT`, moe-split `hmx_mmid_plan`) are HMX only and
  **never apply at decode**.
- The router chain (sigmoid/softmax → ARGSORT/TOP_K → GET_ROWS → SUM_ROWS → DIV) is all 1-row ops,
  so it is all SOLO with a barrier each (F4).

**F7 — host re-encodes the decode graph every token.**
- On a uid cache hit (`ggml-hexagon.cpp:8569`) only kparams and fusion tagging are skipped.
- Every node still goes through `enqueue_op` (`:4410`): clone_buffer per input × sub-session,
  `try_fuse` matching, `fit_op`, map inserts.
- The batch is rebuilt and memcpy'd once per NSP (`:4202`), then one `dspqueue_write` per NSP.
- `OPBATCH = 1280` (`:164`). A Gemma-4 or Nemotron decode graph (42 / 88 layers) may need **2
  batches**, which adds a second submit, a second full cache flush and an end-of-batch barrier.

---

## 3. Plan

| # | Item | Axis change? | Knob | Estimated 4c TG gain (estimate, pending D0) | Status |
|---|---|---|---|---|---|
| D0 | **Decode profiling.** Harness: `--hex-profile-tg` so the TG bench is profiled too (today only PP is). `mdev-profile-analyze.py --select decode` picks the last OPBATCH with `ne1 == 1` (today it always takes the last batch). Report host time per token (token time − DSP graph time), ops per graph, batches per graph, SOLO-op time, sync and imbalance. Baseline for llama-3.2-3b, Qwen3.5-4B, Gemma-4-E4B, Gemma-4-E2B, Qwen3.5-35B-A3B at d1k / d18k | — | — | prerequisite; ranks D1–D9 | **TODO (next)** |
| D1 | **Host: one batch per decode graph, plus a descriptor cache.** OPBATCH sized to the graph (or 2K, as in PR #30843). On a uid hit, reuse the encoded `htp_op_desc` / `htp_tensor` arrays per sub-session and patch only what changes per token (data pointers, KV view length, FA/ROPE params) | no | `GGML_HEXAGON_DESC_CACHE` | host ms/token (TBD by D0); most on small models (E2B, 2B) | planned |
| D2 | **Barrier elision for partition-affine chains.** The host flags an op `NOSYNC` when each NSP reads only what it wrote itself in the previous op: same partition and same unit size, e.g. GEMV slice → GLU / bias / scale on the same rows. The DSP skips the barrier and the flush; dirty ranges carry over to the next real barrier. Mark only this NSP's slice dirty, not the whole dst (F3) | no (sync only) | `GGML_HEXAGON_MDEV_SYNC` (the M6 name) | 3–8% on dense models (estimate: 2–3 ms of sync in 19 ms) | planned |
| D3 | **1-row ops off SOLO.** Split by columns in 128B lines, the way the CHUNKED binary path already works. RMS_NORM(+MUL): each NSP reads the whole row for the sum (redundant, principle 2) and writes its column slice. GLU, broadcast MUL/ADD, ROPE-K by (head, pair) units, SET_ROWS by columns. Ops too small to split go **REPL** (every NSP computes; NSP0 writes), so no NSP waits | **yes** (rows → columns) | `GGML_HEXAGON_DECODE_SPLIT` | with D2; mostly removes SOLO time on NSP0 | planned |
| D4 | **GEMV epilogue fusion at decode.** GLU into the gate/up `MUL_MAT_NX` writeback (the gate and up rows of a slice live on the same NSP, so no barrier is needed). Residual ADD into attn_out / ffn_down (extend `MUL_MAT_ADD`). ADD + RMS_NORM + MUL as one op. In decode, each fused op removes a full op + barrier. **Builds on fusion ticket F** (`ebateni/fusions`): F1 GLU epilogue (`8192`) is **HMX only**, so D4 adds the HVX GEMV / `hvx_mv_id_nx` epilogue. F2 post-norm residual (`16384`) is row-block only, which is 1 row → SOLO at decode until D3 | no | F1/F2 bits extended to HVX; new fusions from `32768` | −2 to −4 ops per layer | planned; after F builds and passes |
| D5 | **Head-affine attention block.** Align the QKV GEMV row chunks to KV-head groups, so that each NSP's Q/K/V slice → q/k-norm → ROPE → SET_ROWS → FA covers the **same** heads with no barrier until attn_out. One barrier per attention block instead of ~6–8. This is the adaptive planner's "attention on head" at decode, inside row split | **yes** (FEAT chunks → head-aligned) | part of `MDEV_SYNC` (level 2) | D2-class gain on the attention half of the layer | planned (after D2) |
| D6 | **Split-KV FA (flash-decoding) across NSPs and threads.** For n_kv < N (Gemma, Qwen3.5 attention, Nemotron: 2 KV heads) and deep context: each team member takes a KV-position range and writes (O, m, l). A combine step folds the partials (in the next op, or REPL'd by the attn_out consumer). HVX FA: assign threads by KV head and split KV blocks within a head, so the G query heads share one K/V stream (removes the up-to-G× K/V re-read, F5) | **yes** (KV position) | `GGML_HEXAGON_FA_SPLIT=3` | Gemma d18k: the 2 idle NSPs come back; dense d18k: KV bytes ÷ G on HVX | planned |
| D7 | **MoE decode.** HVX `MUL_MAT_ID` with a flattened partition: units = (active expert × 32-row chunk), dealt contiguously across NSPs and then threads, so every NSP gets the same bytes and long DMA runs regardless of expert size. Router chain fused into one op (sigmoid/softmax + top-k + gather + normalize), or REPL'd. Shared expert overlapped with the routed experts in one batch, with no barrier between them | **yes** | `GGML_HEXAGON_MMID_SPLIT` decode mode (moe-split branch) | MoE 1.36–1.61x → toward the dense ~2x+ (estimate) | planned. The prefill M5 lesson (deferred 2026-10-25): the MoE loss was summed-work inflation from memory stalls, not imbalance. So D0 must first show whether decode MoE loses to imbalance or idle threads (D7 helps) or to DDR stalls (D7 does not) |
| D8 | **Small-M matmul axis.** For 4 < M ≤ threshold, the HMX matmul splits output columns (FEAT) instead of tokens: each NSP streams 1/N of the weights. Auto rule from bytes: FEAT when `weight_bytes > M × K × 4` (activation f32 bytes). Applies to MUL_MAT, MUL_MAT_NX and MUL_MAT_ID with HMX | **yes** (token → feat) | `GGML_HEXAGON_MM_AXIS` 0 = token, 1 = feat, 2 = auto | spec-decode verify / ub 8–64: up to 4x less weight traffic | planned; key for the Nemotron spec-decode KPI |
| D9 | **Per-NSP GEMV efficiency.** 1c streams ~48 GB/s against an ~80–95 GB/s single-NSP ceiling. Check prefetch depth (`max_prefetch = 16`, `ggml-hexagon.cpp:5635`), thread count, and the DMA tile size. This sets the 4c ceiling once sync and SOLO are gone | no | — | investigate after D0 | later |
| D10 | Automatic tuning of whatever knobs remain scenario-dependent (as M7) | picks axes | `auto` | — | last |

Order:
1. D0.
2. D1, D2, D3: sync and host costs, which help every model.
3. D6 and D7: FA and MoE, the worst-scaling cases.
4. D8: spec decode.
5. D4, D5, D9.
6. D10.

**Dropped / out of scope for decode:**
- Tensor-split collectives and `GGML_META_COMM_ISOLATION`. Row split has no collectives to optimize,
  and tensor TG is below row (`adaptive-axis-planner.md`).
- Sequence split.
- Weight replication: decode is weight-bandwidth-bound, so a replica only adds bytes.

**Overlap with upstream PR #30843:**
- D1 can take its OPBATCH 2K default.
- D4 builds on fusion ticket F (`ebateni/fusions`, base contains #30843).
- Its GLU, RMS_NORM_SCALE and BINARY_NX fusions are still 1-row → SOLO at decode. They cut ops,
  but they do not use the other NSPs until D3.

**Nemotron (nemotron_h_moe), once enablement lands:**
- 2 KV heads, so D6 applies.
- MoE top-9 of ~368 latent experts, so D7 applies.
- Mamba2: SSM_SCAN is not in the backend yet; its partition (heads) must not go SOLO at decode.
- 88 layers: D1 (2 batches) and D2 (op count) apply.
- Spec-decode verify: D8 applies.

### Hardcode vs knob (current view)

| Stage | Decision |
|---|---|
| Dense GEMV (M ≤ 4) | hardcode FEAT (upstream, already) |
| Small-M matmul (4 < M ≤ ~256) | knob (D8) → auto by bytes |
| 1-row elementwise / norm / GLU / ROPE-K / SET_ROWS | hardcode column split or REPL (D3), once A/B passes |
| Barrier per op | knob (D2): 0 = upstream, 1 = affine chains, 2 = + head-affine attention block (D5) |
| FA, `n_kv % N == 0`, shallow context | hardcode KV-head split (upstream) |
| FA, `n_kv < N` or deep context | knob (D6), split-KV |
| MUL_MAT_ID at decode | knob (D7) |
| GATED_DELTA_NET / SSM_CONV | hardcode upstream (they split) |

---

## 4. Test runs (owner runs; commands are for the current HEAD)

Run from the `ebateni/adaptive-split` checkout without `REPO=`. TG uses `-n 256`. `DEPTHS=0,16384`
with `PP_TOKENS=2048` gives TG depths 2048 and 18432. Always use `REPS=5` (1c noise is ~4%).

```bash
# D0  decode profile (needs the harness --hex-profile-tg change; PROFILE=3 for FENCE/L2FLUSH)
ADB_SERIAL=<serial> CORES=1,4 UBATCH=1024 PP_TOKENS=2048 DEPTHS=0,16384 REPS=1 \
  MODELS="llama-3.2-3b-instruct-q4_0:4 Qwen3.5-4B-Q4:4 gemma-4-E4B_q4_0-it:4 gemma-4-E2B_q4_0-it:4 Qwen3.5-35B-A3B-UD-Q4_K_M:4" \
  ./logs/llama.cpp-build-and-test.sh --no-build --only-perf --split row --hex-profile-tg 3 --push-results

# T1  correctness for a knob: ops touched by it, 1/2/4 cores, knob off/on
ADB_SERIAL=<serial> CORES=1,2,4 AB_VAR=<knob> AB=0,1 \
  ./logs/llama.cpp-build-and-test.sh --test-ops <OP> --push-results

# T2  TG A/B, affected + control models
ADB_SERIAL=<serial> CORES=1,4 UBATCH=1024 PP_TOKENS=2048 DEPTHS=0,16384 REPS=5 \
  AB_VAR=<knob> AB=0,1 \
  MODELS="llama-3.2-3b-instruct-q4_0:4 Qwen3.5-4B-Q4:4 gemma-4-E4B_q4_0-it:4 gemma-4-E2B_q4_0-it:4" \
  ./logs/llama.cpp-build-and-test.sh --no-build --split row --push-results

# T3  MoE (D7)
ADB_SERIAL=<serial> CORES=1,4 UBATCH=1024 PP_TOKENS=2048 DEPTHS=0,16384 REPS=5 \
  AB_VAR=GGML_HEXAGON_MMID_SPLIT AB=<off>,<decode> \
  MODELS="Qwen3.5-35B-A3B-UD-Q4_K_M:4 gemma-4-26B_q4_0-it:4" \
  ./logs/llama.cpp-build-and-test.sh --no-build --split row --push-results

# T4  small-M (D8): the PP bench at small ubatches stands in for the spec-decode verify batch
ADB_SERIAL=<serial> CORES=1,4 UBATCH=8,16,32,64 PP_TOKENS=256 DEPTHS=0,4096 REPS=5 \
  AB_VAR=GGML_HEXAGON_MM_AXIS AB=0,2 \
  MODELS="llama-3.2-3b-instruct-q4_0:4 Qwen3.5-4B-Q4:4 gemma-4-E4B_q4_0-it:4" \
  ./logs/llama.cpp-build-and-test.sh --no-build --split row --push-results
```

**Pass criteria (every item):**
- test-ops pass count unchanged, zero FAIL lines.
- Needle, cookie and IQ unchanged.
- 4c TG ≥ knob-off at every depth, by more than the 4% noise to count as a gain.
- PP not lower than knob-off.
- 1c unchanged (±1%).

---

## 5. Results

_None yet. D0 will fill in the decode baseline profile:_

| model | depth | 1c ms/token | 4c ms/token | x | DSP graph ms (4c) | host ms (4c) | ops / batches | sync % | SOLO % | imbalance % |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b | 2k / 18k | | | | | | | | | |
| Qwen3.5-4B | 2k / 18k | | | | | | | | | |
| Gemma-4-E4B | 2k / 18k | | | | | | | | | |
| Gemma-4-E2B | 2k | | | | | | | | | |
| Qwen3.5-35B-A3B | 2k / 18k | | | | | | | | | |

---

## 6. Open issues

- **No decode profile exists.** All earlier profiles are PP-only (`-n 0`). This blocks the ranking
  of D1–D9, so the gains in §3 are estimates.
- **M6 (`MDEV_SYNC`) was never implemented** and the OPBATCH 2K default is not in this tree (it is
  in PR #30843). D1 and D2 start from scratch.
- **Qwen3.5-35B-A3B quality:** IQ 0/10 and cookie ERR in `20261018-213025`. 4c cookie is GARBAGE
  even with `MMID_SPLIT=0` (pre-existing, M5 notes). Must be fixed before D7 can be judged on
  quality.
- **Qwen3.5-35B 4c is killed under profile 3** (`HTP_OP_MEMORY_OVERCLAIM`, M5). Use profile 1 for
  the MoE part of D0.
- **Gemma-4-12B QAT returns garbage** (from Phase 1). Its numbers are perf-only.
- **MoE 4c host OOM** at ub1024 / d16k (from Phase 1). T3 may hit it during the PP part of the run.
- **Build drift:** llama d2048 1c TG went from 23.48 to 22.73 between `75360014b3` and `96ed45522`.
  Re-baseline at D0 on the current HEAD.
- **Nemotron** joins the model set once the enablement epics land (SSM_SCAN, IQ2_XXS / NVFP4).

---

## 7. Work log

- **2026-10-30** — Plan created from a code reading of `ebateni/adaptive-split` (F1–F7) and the
  TG numbers in runs `20261015-144105` … `20261021-104316`. Next: D0 (harness `--hex-profile-tg`
  plus the analyzer decode selector).
