# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `08dadb29c` (`ebateni/moe-split`)
- device binaries: `unknown (package built before BUILD_INFO stamping)`
- date: 2026-10-25T12:47:52-04:00
- reps: 5, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_MMID_SPLIT` = 0,1

## Results

| model | cores | split | GGML_HEXAGON_MMID_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | 0 | 1024 | 1024 | 0 | 153.30 | 1.00x | 16.69 | 1.00x | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | 1 | 1024 | 1024 | 0 | 162.53 | 1.00x | 16.41 | 1.00x | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 0 | 512* | 1024 | 0 | NA | NA | 28.95 | NA | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 1 | 512* | 1024 | 0 | NA | NA | 28.19 | NA | - | - | - |
| gemma-4-26B_q4_0-it | 1 | single | 0 | 1024 | 1024 | 0 | 235.65 | 1.00x | 17.10 | 1.00x | - | - | - |
| gemma-4-26B_q4_0-it | 1 | single | 1 | 1024 | 1024 | 0 | 248.01 | 1.00x | 17.10 | 1.00x | - | - | - |
| gemma-4-26B_q4_0-it | 4 | row | 0 | 1024 | 1024 | 0 | 280.18 | 1.19x | 26.95 | 1.58x | - | - | - |
| gemma-4-26B_q4_0-it | 4 | row | 1 | 1024 | 1024 | 0 | 297.98 | 1.20x | 27.04 | 1.58x | - | - | - |

## PP delta (GGML_HEXAGON_MMID_SPLIT=0 -> 1), per model / cores / ubatch (split=row)

If the percentage shrinks as ubatch grows, the change and ubatch are NOT additive.

| model | cores | ubatch | pp_tok | depth | PP GGML_HEXAGON_MMID_SPLIT=0 | PP GGML_HEXAGON_MMID_SPLIT=1 | delta |
|---|---|---|---|---|---|---|---|
| gemma-4-26B_q4_0-it | 1 | 1024 | 1024 | 0 | 235.7 | 248.0 | +5.2% |
| gemma-4-26B_q4_0-it | 4 | 1024 | 1024 | 0 | 280.2 | 298.0 | +6.4% |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | 1024 | 1024 | 0 | 153.3 | 162.5 | +6.0% |

## Cross-core scaling (GGML_HEXAGON_MMID_SPLIT=0=Base vs GGML_HEXAGON_MMID_SPLIT=1=Opt), per model (split=row)

Uses the largest ubatch tested for each model. "vs 1c" is that series's own
speedup over its 1-core result; PP gain is Opt vs Base at that core count.

### gemma-4-26B_q4_0-it (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 236 | 1.00x | 248 | 1.00x | +5.2% | 17.1 | 17.1 |
| 4 | 280 | 1.19x | 298 | 1.20x | +6.4% | 26.9 | 27.0 |

### Qwen3.5-35B-A3B-UD-Q4_K_M (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 153 | 1.00x | 163 | 1.00x | +6.0% | 16.7 | 16.4 |

## Quality gate

_Quality checks were skipped for this run (--only-perf) — performance numbers alone do NOT establish correctness._

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/moe-split/logs/20261025-121720/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/moe-split/logs/20261025-121720/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/moe-split/logs/20261025-121720/logs/cookie_*.txt`
