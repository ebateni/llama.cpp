# Adaptive per-stage split for prefill on Hexagon mdev — plan and work log

AUTOSP-100680, Phase 1 (prefill; decode is Phase 2). Branch `ebateni/adaptive-split`, off
upstream `620d2225c`. Device `1b531bdd`, 4 NSPs, row split (`GGML_HEXAGON_DEVICES=HTP0[0-3]`).

This file is the living plan and history. It is updated and pushed after every step.
Detailed measurements live in [`adaptive-split-baseline.md`](adaptive-split-baseline.md).

_Last updated: 2026-10-15 — **all device runs so far were built from the wrong branch** (see §6); M1 awaiting a build from this branch._

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
| M1 | FA team split: KV heads -> gcd(n_kv, N) teams, Q-blocks split within each team, per-NSP Br cost | **yes** (Q-block -> KV-head team x Q-block) | `GGML_HEXAGON_FA_SPLIT=0/1` | Gemma ub1024 +15–18%; llama 0% | **pushed `02f7ad0f5`, awaiting test** |
| M2 | GLU epilogue fusion: MUL_MAT_NX(gate, up) + SWIGLU/GEGLU -> one op | no | `GGML_HEXAGON_OPFUSION` new bit 128 | llama ~8%, Qwen ~7%, Gemma ~10% | next |
| M3 | Qwen3.5 conv block fusion: CONCAT + SSM_CONV + SILU + state CPY | **yes** (channels -> TOKEN + 3-token halo) | OPFUSION bit 256 | Qwen3.5 ~13% | planned |
| M4 | Residual ADD + RMS_NORM(+MUL) fusion; Gemma post-norm + ADD + layer-scale MUL | no | OPFUSION bit 512 | 3–8% | planned |
| M5 | MoE: token-count-balanced experts (LPT), expert vs token split; fix the 4c memory overhead | **probably** | `GGML_HEXAGON_MMID_SPLIT` | TBD (needs profile) | waiting for MoE profile |
| M6 | Barrier elision + REPL for tiny ops, on the post-fusion graph | no (sync only) | `GGML_HEXAGON_MDEV_SYNC` | ≤ 2–3% | later |
| M7 | Automatic tuning of whatever knobs remain scenario-dependent | picks axes | `auto` | — | last |

Dropped for prefill: FEAT matmul split (loses at ub ≥ 1024; revisit in decode, Phase 2).

**Ticketing** (decided after results): one umbrella ticket with a child per milestone, each a
separate PR. The axis-related items are M1, M3 and M5 (+ M7); M2 and M4 are fusion.

### Hardcode vs knob (current view)

| Stage | Decision |
|---|---|
| Dense matmuls (QKV, o, FFN, PLE) | hardcode TOKEN |
| Elementwise (norm, act, GLU, ADD, ROPE) | hardcode TOKEN/rows; fuse |
| FA, `n_kv % N == 0` | hardcode KV-head split (upstream) |
| FA, `n_kv % N != 0` | `FA_SPLIT` knob -> default on once validated |
| SSM_CONV / conv block | TOKEN + halo once validated (M3) |
| GATED_DELTA_NET | hardcode V-heads (upstream) |
| MUL_MAT_ID | knob (M5) |

---

## 4. Test runs (owner runs; commands are for the current HEAD)

```bash
# M1-1  build + correctness (knob only matters at 4 cores)
REPO="$PWD" ADB_SERIAL=<serial> CORES=4 AB_VAR=GGML_HEXAGON_FA_SPLIT AB=0,1 \
  ./logs/llama.cpp-build-and-test.sh --test-ops FLASH_ATTN_EXT --push-results

# M1-2  perf A/B with quality checks
REPO="$PWD" ADB_SERIAL=<serial> CORES=1,4 UBATCH=1024,2048 PP_TOKENS=2048 DEPTHS=0,16384 \
  AB_VAR=GGML_HEXAGON_FA_SPLIT AB=0,1 \
  MODELS="gemma-4-E4B_q4_0-it:4 Qwen3.5-4B-Q4:4 llama-3.2-3b-instruct-q4_0:4" \
  ./logs/llama.cpp-build-and-test.sh --no-build --split row --push-results

# M5-0  MoE per-op profile
REPO="$PWD" ADB_SERIAL=<serial> REPS=1 CORES=1,4 UBATCH=1024 PP_TOKENS=2048 DEPTHS=0 \
  MODELS="Qwen3.5-35B-A3B-UD-Q4_K_M:4" \
  ./logs/llama.cpp-build-and-test.sh --no-build --only-perf --split row --hex-profile 1 --push-results
```

**Pass criteria for any item:** test-ops pass count unchanged with zero FAIL lines; needle,
cookie and IQ unchanged; 4c PP ≥ the knob-off result at every depth and ubatch; TG unchanged.

---

## 5. Measured results per item

| Item | Model | ub | depth | knob off | knob on | gain | quality |
|---|---|---|---|---|---|---|---|
| M1 | — | — | — | — | — | — | pending |

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
