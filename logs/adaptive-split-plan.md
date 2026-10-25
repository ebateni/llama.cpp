# Adaptive per-stage split for prefill on Hexagon mdev — plan and work log

AUTOSP-100680, Phase 1 (prefill; decode is Phase 2). Branch `ebateni/adaptive-split`, off
upstream `620d2225c`. Device `1b531bdd`, 4 NSPs, row split (`GGML_HEXAGON_DEVICES=HTP0[0-3]`).

This file is the living plan and history. It is updated and pushed after every step.
Detailed measurements live in [`adaptive-split-baseline.md`](adaptive-split-baseline.md).

_Last updated: 2026-10-22 — M1b zig-zag dropped after retry (no gain); M5 next. M1 shipped as a single `GGML_HEXAGON_FA_SPLIT` flag on upstream master. M1b (zig-zag Q-blocks) pushed for A/B. Plan replanned around upstream PR #30843 (fusions)._

---

## 1. Goal and principles

**Goal:** state-of-the-art 4-core prefill scaling, above today's row split, that stays robust
for long contexts, large models (MoE) and special models (Gemma 4, Qwen3.5).

**Principles** (from the project owner):
1. Change the Hexagon side. Change the Meta backend / `src/` only when necessary and approved.
2. Redundant execution of an op on all NSPs is acceptable when it is cheaper.
3. If a stage's best axis never depends on the scenario, hardcode it. If it does depend on the
   scenario, give it a knob. In the last stage, tune the knobs automatically.
4. Beat row split and stay robust.

**Scope:** prefill with ubatch ≥ 1024 (1024 / 2048). Anything smaller is decode, Phase 2.

**Workflow:** Claude writes the code and pushes; the owner builds, runs the device tests and
pushes results (`logs/<ts>/`) to the fork. Every change sits behind a knob that defaults off,
gets an A/B in one binary (knob off == upstream), and the default flips only after the A/B
passes.

---

## 2. What the data says (P0 baseline)

Details are in `adaptive-split-baseline.md`.

- **Sync is cheap.** A barrier costs ~2.5 us per op (~1% of the graph) and flush 2–3%, so
  eliding barriers is worth ≤ 1–2%.
- **Gemma FA runs SOLO on NSP0** at ub1024. With 2 KV heads (`n_kv % 4 != 0`) FA falls back to
  Q-blocks, and the single-core Br leaves fewer than 4 blocks. That is 15% of the graph. ub2048
  hides it (Gemma 2.31x -> 2.75x).
- **Elementwise ops are DDR-bound** (~150–185 GB/s across all 4 NSPs, so ~2x scaling). No axis
  choice helps; fusion does.
- **Matmul axis: TOKEN is right for prefill.** At ub ≥ 1024 the f32 activations are larger than
  the Q4 weights, so a FEAT split moves more bytes.
- **Qwen3.5 conv block** (CONCAT/transpose + SSM_CONV + SILU + state CPY) costs 16% of the 4c graph.
- **GLU ops** (SWIGLU/GEGLU) cost 7–11% of the 4c graph.
- **MoE (Qwen3.5-35B-A3B):** worst scaling, 2.30x at ub1024. OOM at ub2048, and at 4c / d16k
  (1c passes).

### Current row split vs 1c (pp2048)

| model | ub1024 d0 | ub2048 d0 | ub1024 d16k | ub2048 d16k |
|---|---|---|---|---|
| llama-3.2-3b | 2.99 | 3.18 | 3.63 | 3.67 |
| Qwen3.5-4B | 2.77 | 2.75 | 3.01 | 2.99 |
| Gemma-4-E4B | 2.31 | 2.75 | 2.48 | 2.87 |
| Qwen3.5-35B-A3B (MoE) | 2.30 | OOM | OOM (4c) | OOM |

---

## 3. Plan

| # | Item | Axis change? | Knob | Estimated 4c gain (upper bound from profiles) | Status |
|---|---|---|---|---|---|
| M1 | **FA adaptive team split** — shipped as `hexagon: KV-head team split for HMX flash attention in multicore` + merged flag (`8075d1d54`, qualcomm `ebateni/adaptive-split`). `FA_SPLIT=0` = token split only; `1` = head split when `n_kv % N == 0`, team split when `n_kv < N` | **yes** | `GGML_HEXAGON_FA_SPLIT=0/1` | — | **SHIPPED.** On the new base (run `20261021-104316`, pp1024 4c): E4B **+14.7%**, 12B **+14.9%**, 26B **+6.6%** (TG +15%), E2B −0.5% |
| M1b | **Zig-zag Q-blocks within a team.** Masked-block skipping makes per-Q-block cost grow with position under a causal mask, so the contiguous split leaves the last member with the most work: 4-way critical path 7/16 vs 4/16 ideal (E2B, 1 team × 4); 2-way 3/4 vs 1/2 (E4B / 12B / 26B, 2 teams × 2). Snake order {m, 2T−1−m, …} balances it. Any mask; no host cost | no (order only) | `FA_SPLIT=2` | E2B and Gemma global layers: FA up to ~1.5–1.75× faster at depth 0 | **DROPPED.** Run `20261021-213738` (1 vs 2, ub1024, d0/16k): no change on any model, all within ±1% (E2B 4c d0 5143→5111, d16k 3912→3932; E4B 3320→3322). Debug (`20261021-233331`, `fa_dbg.txt`) **confirmed** the bench shapes get Br=256 → 4 Q-blocks = team size, so zig-zag == contiguous (no-op). Causal critical path at d0 is 4/10 vs 2.5/10 ideal. Retry `bf0558252` (zig-zag + forced ≥ 2 Q-blocks per member): `20261022-000438` E4B 3309→3287 (−0.7%), `20261022-001604` E2B 5157→5165 (+0.2%) → no gain; causal FA imbalance is not material end-to-end |
| M2 | **Matmul-epilogue GLU**: MUL_MAT_NX(gate, up) + act·mul in the HMX writeback. Not covered by #30843 (it fuses UNARY+MUL into one elementwise GLU op, but gate/up still round-trip through DDR) | no | next free `OPFUSION` bit after #30843 (≥ 8192) | llama ~8%, Qwen ~7%, Gemma ~10% (upper bound; the elementwise part is already fused by #30843) | **blocked on #30843 landing** (it refactors htp-fusion / kparams); then rebase and build on its GLU op |
| M3 | Qwen3.5 conv block, **narrowed**: #30843 fuses SSM_CONV+SILU and GDN+CPY. What remains: the CONCAT (transpose) and conv-state CPY around SSM_CONV, and the channels → TOKEN + 3-token halo axis | **yes** | next free bit | Qwen3.5 ~5–8% (was ~13%) | after #30843; needs a Qwen3.5 profile on the post-#30843 base |
| M4 | **Cross-op residual ADD → RMS_NORM(+MUL)** (pre-norm of the next sublayer). #30843 covers same-op chains (BINARY_NX), RMS_NORM+MUL, RMS_NORM+SCALE and MUL+MUL scalar, but not ADD feeding a norm | no | next free bit | 3–6% | **agree ownership with Max first**; otherwise drop |
| M5 | **MoE**: token-count-balanced experts (LPT), expert vs token split; 4c host memory overhead. FA does not help here (35B-A3B: +2.1%, noise) and the scaling is 2.25×, the worst of all models | **probably** | `GGML_HEXAGON_MMID_SPLIT` | TBD | **NEXT**: needs the M5-0 MoE profile |
| M6 | Barrier elision + REPL for tiny ops, on the post-fusion graph | no (sync only) | `GGML_HEXAGON_MDEV_SYNC` | ≤ 2–3% | later |
| M7 | Automatic tuning of whatever knobs remain scenario-dependent | picks axes | `auto` | — | last |

**Qwen3.5 (4 KV heads → head split):** its FA partition is already balanced; every NSP does one head's full causal triangle, and masked-block skipping cuts the same work on every NSP. So no FA tweak helps it. Qwen3.5's remaining 4c gap is in GDN, the conv block (M3), GLU (M2) and, for 35B-A3B, MUL_MAT_ID (M5).

**Overlap with upstream PR #30843** (fusion refactor, not merged as of 2026-10-21). Our plan does not duplicate it:
- Fusions it adds: RMS_NORM+MUL (non-contiguous), RMS_NORM+SCALE, MUL_MAT+ADD, MUL_MAT_NX, MUL_MAT_ID_NX, GDN+CPY, SSM_CONV+SILU, REPEAT+GET_ROWS, UNARY+MUL → GLU (SWIGLU/GEGLU/…), MUL+MUL scalar, BINARY_NX (same-op chains), ALLREDUCE+ADD.
- `OPFUSION` bits 2…4096 are taken, so new fusions start at 8192.
- htp-fusion.cpp / htp-kparams are refactored, so do not write fusion code until it lands.

Dropped for prefill: FEAT matmul split (loses at ub ≥ 1024; revisit in decode, Phase 2).

**Ticketing** (decided after results): one umbrella ticket with a child per milestone, each a
separate PR. The axis-related items are M1, M3 and M5 (+ M7); M2 and M4 are fusion.

### Hardcode vs knob (current view)

| Stage | Decision |
|---|---|
| Dense matmuls (QKV, o, FFN, PLE) | hardcode TOKEN |
| Elementwise (norm, act, GLU, ADD, ROPE) | hardcode TOKEN/rows; fuse |
| FA, `n_kv % N == 0` | hardcode KV-head split (upstream) |
| FA | **shipped**: head split (`n_kv % N == 0`), team split (`n_kv < N`); zig-zag within a team tried and dropped (M1b, no gain) |
| SSM_CONV / conv block | TOKEN + halo once validated (M3) |
| GATED_DELTA_NET | hardcode V-heads (upstream) |
| MUL_MAT_ID | knob (M5) |

---

## 4. Test runs (owner runs; commands are for the current HEAD)

Run from the `ebateni/adaptive-split` checkout without `REPO=` (it defaults to the script's checkout).

**M1 closing set** (HEAD >= the solver-memo commit, see §7; T1 builds, the rest use `--no-build`):

```bash
# T1  correctness: FA ops, 1/2/4 cores, all three modes
ADB_SERIAL=<serial> CORES=1,2,4 AB_VAR=GGML_HEXAGON_FA_SPLIT AB=0,1,2   ./logs/llama.cpp-build-and-test.sh --test-ops FLASH_ATTN_EXT --push-results

# T2  perf + quality A/B, affected and control models
ADB_SERIAL=<serial> CORES=1,4 UBATCH=1024,2048 PP_TOKENS=2048 DEPTHS=0,16384   AB_VAR=GGML_HEXAGON_FA_SPLIT AB=0,2   MODELS="gemma-4-E4B_q4_0-it:4 Qwen3.5-4B-Q4:4 llama-3.2-3b-instruct-q4_0:4"   ./logs/llama.cpp-build-and-test.sh --no-build --split row --push-results

# T3  =1 vs =2 on Gemma
ADB_SERIAL=<serial> CORES=4 UBATCH=1024,2048 PP_TOKENS=2048 DEPTHS=0,16384   AB_VAR=GGML_HEXAGON_FA_SPLIT AB=1,2 MODELS="gemma-4-E4B_q4_0-it:4"   ./logs/llama.cpp-build-and-test.sh --no-build --split row --push-results

# T4  MoE (2 KV heads)
ADB_SERIAL=<serial> CORES=4 UBATCH=1024 PP_TOKENS=2048 DEPTHS=0   AB_VAR=GGML_HEXAGON_FA_SPLIT AB=0,2 MODELS="Qwen3.5-35B-A3B-UD-Q4_K_M:4"   ./logs/llama.cpp-build-and-test.sh --no-build --split row --push-results
```

**M1 close criteria:** T1 shows no per-case failure with `=1`/`=2` that passes with `=0`. In T2/T4, `=2` 4c PP is ≥ `=0` in every row, TG is no worse, and IQ / needle / cookie are unchanged. In T3, `=2` is ≥ `=1` at ub1024 and in TG, and ≥ `=0` at ub2048. Then flip the default to 2, write the PR, and close.

**Tracked separately:**

```bash
# R-1  llama d16k regression vs the old branch: FA head split on/off
ADB_SERIAL=<serial> CORES=4 UBATCH=1024,2048 PP_TOKENS=2048 DEPTHS=16384   AB_VAR=GGML_HEXAGON_FA_HEAD_SPLIT AB=1,0 MODELS="llama-3.2-3b-instruct-q4_0:4"   ./logs/llama.cpp-build-and-test.sh --no-build --only-perf --split row --push-results

# M5-0  MoE per-op profile
ADB_SERIAL=<serial> REPS=1 CORES=1,4 UBATCH=1024 PP_TOKENS=2048 DEPTHS=0   MODELS="Qwen3.5-35B-A3B-UD-Q4_K_M:4"   ./logs/llama.cpp-build-and-test.sh --no-build --only-perf --split row --hex-profile 1 --push-results
```

**Pass criteria for any item:** test-ops pass count unchanged with zero FAIL lines; needle,
cookie and IQ unchanged; 4c PP ≥ the knob-off result at every depth and ubatch; TG unchanged.

---

## 5. Measured results per item

| Item | Model | ub | depth | knob off | knob on | gain | quality |
|---|---|---|---|---|---|---|---|
Run `20261015-223555`, binaries `d3f87a011` (this branch), pp2048, row, 3 reps. "knob off" is upstream,
so this is also the **re-measured baseline**. Quality (IQ / needle / cookie) is identical with the knob off and on, in every row.

| Item | Model | ub | depth | knob off | knob on (`=1`) | gain | 4c x1c off -> on |
|---|---|---|---|---|---|---|---|
| M1 | Gemma-4-E4B | 1024 | 0 | 2857 | 3258 | **+14.0%** | 2.32 -> 2.66 |
| M1 | Gemma-4-E4B | 1024 | 16k | 2476 | 2808 | **+13.4%** | 2.48 -> 2.80 |
| M1 | Gemma-4-E4B | 2048 | 0 | 3358 | 3316 | −1.3% | 2.75 -> 2.71 |
| M1 | Gemma-4-E4B | 2048 | 16k | 2897 | 2822 | −2.6% | 2.95 -> 2.87 |
| M1 | Qwen3.5-4B | 1024 | 0 | 3094 | 3155 | +2.0% | 2.80 -> 2.86 |
| M1 | Qwen3.5-4B | 2048 | 0 | 3114 | 3213 | +3.2% | 2.86 -> 2.96 |
| M1 | Qwen3.5-4B | 1024/2048 | 16k | 2573 / 2603 | 2563 / 2605 | ±0.4% | 3.07 / 3.14 |
| M1 | llama-3.2-3b | all | all | — | — | ±0.6% (n_kv % 4 == 0, no change expected) | 3.01 / 3.15 / 3.29 / 3.28 |

- 1c is unchanged (±0.4%), as expected: the knob only acts at mdev > 1.
- **Decode bonus:** Gemma 4c TG goes from 32.9 / 31.6 / 24.5 / 23.8 to 35.3 / 32.9 / 29.6 / 27.5 (+4% to +21%). Decode FA has one Q-block, so upstream runs it SOLO; the team split puts it on 2 NSPs (one per KV head). That is a Phase 2 win for free.
- **The ub2048 loss** is small but in the same direction at both depths, so it is treated as real. The team split does the same number of block-head units per NSP as upstream when the block count divides evenly, so the cost model sees a tie, yet it loses. Hypothesis: the larger Br per NSP or the shared-KV DDR pattern. **Fix (`=2`, auto):** run the solver for both partitions and use the team split only when it is ≥ 5% cheaper per NSP. Otherwise use upstream's exact Br/Bc and partition, so a tie reproduces upstream bit for bit. ub1024 (upstream has < 4 blocks) and decode (1 block) still pick the team.
- **test-ops FLASH_ATTN_EXT** (run `20261015-200226`): 2577–2594 of 2640 pass in **every** configuration, including 1c where the knob does nothing (52 vs 56 FAIL lines between two identical 1c runs). So the failures are pre-existing and nondeterministic, not caused by M1. Their logs were never pushed (`*.log` is gitignored), so the harness now force-adds them; M1-4 re-runs it.


**M1 closing runs** (binaries `96ed45522`, 3 reps):
- `20261018-142527` (T1): test-ops FA at 1/2/4c × `FA_SPLIT` 0/1/2.
- `20261018-171424` (T2): `=0` vs `=2`.
- `20261018-191245` (T3): `=1` vs `=2`.
- `20261018-200106`: Gemma-4-12B, 0/1/2. T4 (MoE) was not run.

| Model | ub | depth | `=0` | `=1` (T3) | `=2` (T2 / T3) | `=2` vs `=0` |
|---|---|---|---|---|---|---|
| Gemma-4-E4B | 1024 | 0 | 2807 | 3213 | 3203 / 3191 | **+14.1%** |
| Gemma-4-E4B | 1024 | 16k | 2477 | 2809 | 2801 / 2769 | **+13.1%** |
| Gemma-4-E4B | 2048 | 0 | 3348 | 3283 | 3326 / 3297 | −0.7% |
| Gemma-4-E4B | 2048 | 16k | 2839 | 2793 | 2790 / 2805 | −1.7% |
| Qwen3.5-4B / llama-3.2-3b | all | all | — | — | — | within ±0.7% (Qwen ub2048 d16k +1.4%) |
| Gemma-4-12B | 1024 | 0 | 1762 | 1777 | 1758 | ±0.8% |

Gemma 4c TG: 32.9 / 31.5 / 24.4 / 23.8 with `=0`, 34.3 / 32.7 / 29.6 / 27.6 with `=2` (the same as `=1`).

**Decision (owner, 2026-10-18): `=2` is not a success.** It is equal to `=1` within noise everywhere (−1.4% … +0.5%) and did not remove the ub2048 dip, so the added code and host work buy nothing. Reverted to plain M1 and made the team split the default.
- The ub2048 −1–2% is within run-to-run noise: the same `=0` config measured 2897 and 2839 t/s (2%) in two runs.
- So M1 = `02f7ad0f5` + default on.

**test-ops** (T1, logs now pushed): 44–63 failing cases per configuration. Every one has `sinks=1`, and every one is a small NMSE overshoot (5.1e-4 … 1.4e-3 against a 5e-4 limit). The failing set changes from run to run, even between configs where the knob has no effect (1c ab0 / ab1 / ab2: 45 / 60 / 63). So this is pre-existing nondeterministic precision noise in the sinks path, not caused by M1. Upstream `ceefd16da` "hexagon: correct HMX FA sinks" (#30581, not on our base) may be the fix; check after the rebase.

Gemma-4-12B is still GARBAGE (IQ 2/10) at 1c and 4c, independent of the knob.

---

## 6. Open issues

- **Wrong source tree (confirmed).** `REPO="$PWD"` pointed to the `llama.cpp.ebateni` checkout on
  `dev/ebateni/seq-parallel-prototype` (+dirty, 520 commits behind upstream), while the harness
  and the results lived in the `ebateni/adaptive-split` checkout. Run `20261015-191941` stamps
  `device binaries: 75360014b3 (dev/ebateni/seq-parallel-prototype) +dirty`. So the P0
  baseline numbers come from that branch, not from upstream `620d2225c`. Fix (`686052437`): REPO
  now defaults to the script's own checkout, and a foreign REPO is refused. **The baseline has to
  be re-measured from this branch** (it is the knob-off half of every A/B, so the M1 A/B does
  that for free).
- **Gemma-4-12B garbage** (run `20261015-191941`, wrong-branch binaries): cookie GARBAGE and IQ
  1/10 at **1c and 4c, with fusion on and off**. So it is not the multi-core split and not the
  fusions; it is a single-core kernel or the model. Gemma-2-2B is fine. Re-check on this branch.
- **MoE 4c host OOM** at ub1024 / d16k while 1c passes; 4c uses extra host memory (M5).
- **Upstream PR in flight (Alan Zhu, 2026-10-18), affects M1 / M2 / M4 / M6.**
  - (a) FA HMX→HVX crossover for SWA models: Gemma's 25 SWA layers move to HVX FA (2–3× per op).
    - M1 only acts on the HMX path, so on top of this PR its gain shrinks to the global layers.
    - His crossover is tuned against upstream HMX, which on mdev runs mostly on NSP0 for n_kv=2. With M1 the 4c HMX side is faster, so the crossover must be re-measured with `FA_SPLIT=2`.
    - Expect a merge conflict in `ggml_hexagon_precompute_flash_attn_params` and the solver signature.
    - In M7, the kernel choice (HMX/HVX) and the partition choice belong in one cost model.
  - (b) Gemma4 op-sequence fusions (as Metal and CUDA do): overlap with M2 (GEGLU) and M4 (post-norm + ADD + layer-scale). Agree ownership and `OPFUSION` bits (we planned 128/256/512) before writing them.
  - (c) OPBATCH default 1280 → 2K: every baseline shifts, so re-measure knob-off after the rebase. This helps M6, since the whole graph fits in one batch and batch boundaries are full syncs.
  - Plan: share M1 with Alan before his merge; split fusion ownership; after the merge, rebase and re-run T2/T3 on Gemma.
- **Pre-existing FA test-ops failures**: all `sinks=1`, small NMSE overshoot, nondeterministic (see §5). Re-check after rebasing onto upstream, which includes `ceefd16da` (HMX FA sinks fix).
- **llama 4c at d16k is ~10% slower than the old branch.** Old binaries (`75360014b`, 520 upstream commits older): 1724 / 1817 t/s at ub1024 / 2048. This branch: 1567 / 1624. 1c is identical (476 / 495), so this is a multi-core regression in newer upstream, likely the FA KV-head split (#29974) at long KV (llama `n_kv=8`). Gemma and Qwen3.5 are not affected. Test R-1 checks it with `FA_HEAD_SPLIT=0`. If the regression is real, the head split vs Q-block split choice becomes a depth-dependent knob (principle 3) for the cost model.
- **Causal block skipping** in HMX FA: every Q-block streams the full KV, including fully
  masked blocks. Skipping them would roughly halve FA at depth 0, for 1c too. This is a
  candidate item outside the axis work.

---

## 7. Work log

### 2026-10-15

| Commit | What |
|---|---|
| `d6105d5c5` | Branch created off upstream `620d2225c`; build tooling, snapdragon notes, logs carried over |
| `e675caf18`, `32813a099`, `fbd8ff60e` | Harness: `--hex-profile`, timestamped OUTDIR, `--push-results` (commit + push run dir, gzip profiles) |
| `0f55e7b84` | Harness: profiled PP stderr goes straight to a file (the host had stalled on a grep/tee pipe) |
| `88979da5f` | Harness: unquoted `$ADB` in profile/memory pulls (profile dir was empty) |
| `003057f5b` | `pkg-snapdragon/` build output removed from the repo and ignored |
| `ca3c740be`, `04c6b80d6` | Harness: split `.gz` > 94 MB into parts; `--push-only <dir>` |
| owner runs | `20261015-144105` (row, d0/4096), `-152029` (profile 1), `-153217` (profile 3, FENCE/L2FLUSH) |
| `6193dab81` | **P0 baseline** doc + `scripts/snapdragon/mdev-profile-analyze.py` (per-op flush / barrier / imbalance / exec). Findings: sync ~3–4%, Gemma FA SOLO 15%, elementwise DDR-bound |
| owner runs | `20261015-163820` (profile 1, ub1024/2048), `-171354` (ub1024/2048 x d0/16k) |
| `ace42eb26` | Harness: depth column in results/summary, device `BUILD_INFO` stamp, accept `name.gguf` model names (the MoE run had failed on `.gguf.gguf`). Baseline: ub2048/16k update, Qwen3.5 conv block 16%, GLU 7–11% |
| owner run | `20261015-181419`: Qwen3.5-35B-A3B MoE, 2.30x at ub1024, OOMs |
| `02f7ad0f5` | **M1**: `GGML_HEXAGON_FA_SPLIT` team split + per-NSP Br cost; FA profile string shows br/bc/team; new test-backend-ops FA cases (n_kv 1/2/6); harness `--test-ops` |
| `b4c3373a2` | Baseline: MoE results |
| (this file) | Plan + work log |
| `2b0ce0c1e` | This plan + work log |
| owner run | `20261015-191941`: Gemma-4-12B / Gemma-2-2B, OPFUSION 1 vs 0 (fusion = +8% PP); Gemma-4-12B garbage at 1c and 4c |
| `686052437` | Harness: REPO defaults to the script's checkout; refuses a foreign REPO (the root cause of the wrong-branch binaries) |
| `d3f87a011` | Plan: wrong-branch finding, Gemma-4-12B triage |
| owner runs | `20261015-200226` (test-ops FA, 1/2/4c x FA_SPLIT 0/1: failures pre-existing), `20261015-223555` (**M1 A/B on this branch**: Gemma ub1024 +14%, ub2048 −1–3%, Gemma TG +4–21%) |

### 2026-10-18

| Commit | What |
|---|---|
| `f4a9b2801` | **M1 auto** `GGML_HEXAGON_FA_SPLIT=2`: cost-model choice between the team split and upstream, per FA op; `hmx_fa_find_chunk_size` returns its cost. Harness: `git add -f` so the run's `*.log` files are pushed. Plan: M1 results, re-measured baseline, llama d16k regression vs the old branch |
| (this commit) | M1 named "FA adaptive team split"; M1 closing test set T1–T4 and criteria |
| (this commit) | Host: memoize `hmx_fa_find_chunk_size` (32-entry cache on the full input tuple). FA precompute runs every prefill ubatch (KV length changes), and auto doubles the search; all layers share ~2 shapes, so this makes it ~free. Results are identical |
| (this commit) | Plan: dependency on Alan Zhu's in-flight PR (FA HMX/HVX SWA crossover, Gemma4 fusions, OPBATCH 2K): impact on M1/M2/M4/M6 |
| owner runs | `20261018-142527` (T1 test-ops), `-171424` (T2 0 vs 2), `-191245` (T3 1 vs 2), `-200106` (Gemma-4-12B) |
| (this commit) | **M1 closed**: reverted `FA_SPLIT=2` auto + solver memo (no gain over `=1`), team split default on (`opt_fa_split = 1`). Harness keeps `git add -f` for logs |
| owner run | `20261021-104316` (shipped flag, new base): E4B +14.7%, 12B +14.9%, 26B +6.6% / TG +15%, E2B −0.5% |
| qualcomm `e9ee0abec` | **M1b** zig-zag Q-block order in a team (`FA_SPLIT=2`), branch `ebateni/fa-team-zigzag` on the ebateni fork |
| (this commit) | Replan around PR #30843: M2 = matmul-epilogue GLU only; M3 narrowed; M4 = cross-op ADD→norm (coordinate); M5 next after M1b |
| owner run | `20261021-213738` (M1b 1 vs 2, stopped before 26B 4c): no change, within noise -> **M1b dropped** |
| ebateni `bf0558252` | M1b retry: zig-zag forces >= 2 Q-blocks per member (debug showed qblk == team -> no-op) |
| owner runs | `20261022-000438` (E4B), `20261022-001604` (E2B): zig-zag retry no gain -> **M1b closed** |
| owner `0694d6ae3` (ebateni `ebateni/moe-split`) | **M5 v1**: `GGML_HEXAGON_MMID_SPLIT` 0 = round-robin, 1 = LPT on 32-row tiles, 2 = token split. Run `20261023-183334` 4c pp1024: Gemma-4-26B +5.0%, Qwen3.5-35B-A3B +1.3%; 1c deltas -1.4/-2.2% are pure noise (same code path at 1c). Review: add per-expert weight cost to LPT, O(n) bucket sort, check bit-exactness (Qwen 4c ab1 iq/cookie changed); run the M5-0 profile before more tuning |
