# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `1043941ed` (`ebateni/fa-team-zigzag`)
- device binaries: `1043941ed (ebateni/fa-team-zigzag) +dirty built 2026-10-22T18:57:22-04:00`
- date: 2026-10-22T21:21:46-04:00
- reps: 5, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_FA_SPLIT` = 1,2

## Results

| model | cores | split | GGML_HEXAGON_FA_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 1 | 1024 | 1024 | 0 | 987.05 | NA | 25.69 | NA | 1/10 | PASS | ERR |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 2 | 1024 | 1024 | 0 | 1067.37 | NA | 25.02 | NA | 1/10 | PASS | OK |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 1 | 1024 | 1024 | 16384 | NA | NA | 22.45 | NA | 2/10 | PASS | ERR |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 2 | 1024 | 1024 | 16384 | NA | NA | NA | NA | 2/10 | PASS | ERR |
| gemma-4-E4B_q4_0-it | 4 | row | 1 | 1024 | 1024 | 0 | 3356.01 | NA | 36.19 | NA | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 2 | 1024 | 1024 | 0 | 3356.34 | NA | 36.17 | NA | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 1 | 1024 | 1024 | 16384 | 2861.29 | NA | 30.29 | NA | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 2 | 1024 | 1024 | 16384 | 2788.71 | NA | 30.29 | NA | 7/10 | PASS | OK |
| gemma-4-E2B_q4_0-it | 4 | row | 1 | 1024 | 1024 | 0 | 5239.06 | NA | 55.80 | NA | 5/10 | PASS | OK |
| gemma-4-E2B_q4_0-it | 4 | row | 2 | 1024 | 1024 | 0 | 5229.39 | NA | 55.83 | NA | 5/10 | PASS | OK |
| gemma-4-E2B_q4_0-it | 4 | row | 1 | 1024 | 1024 | 16384 | 4027.97 | NA | 41.15 | NA | 5/10 | PASS | OK |
| gemma-4-E2B_q4_0-it | 4 | row | 2 | 1024 | 1024 | 16384 | 4046.98 | NA | 41.15 | NA | 5/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 1 | 1024 | 1024 | 0 | 3159.26 | NA | 36.75 | NA | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 2 | 1024 | 1024 | 0 | 3138.50 | NA | 36.72 | NA | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 1 | 1024 | 1024 | 16384 | 2482.27 | NA | 30.42 | NA | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 2 | 1024 | 1024 | 16384 | 2474.82 | NA | 30.41 | NA | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 1 | 1024 | 1024 | 0 | 5471.00 | NA | 51.33 | NA | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 2 | 1024 | 1024 | 0 | 5411.08 | NA | 51.53 | NA | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 1 | 1024 | 1024 | 16384 | 1770.84 | NA | 22.67 | NA | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 2 | 1024 | 1024 | 16384 | 1595.50 | NA | 21.88 | NA | 9/10 | PASS | OK |

## PP delta (GGML_HEXAGON_FA_SPLIT=1 -> 2), per model / cores / ubatch (split=row)

If the percentage shrinks as ubatch grows, the change and ubatch are NOT additive.

| model | cores | ubatch | pp_tok | depth | PP GGML_HEXAGON_FA_SPLIT=1 | PP GGML_HEXAGON_FA_SPLIT=2 | delta |
|---|---|---|---|---|---|---|---|
| gemma-4-E2B_q4_0-it | 4 | 1024 | 1024 | 0 | 5239.1 | 5229.4 | -0.2% |
| gemma-4-E2B_q4_0-it | 4 | 1024 | 1024 | 16384 | 4028.0 | 4047.0 | +0.5% |
| gemma-4-E4B_q4_0-it | 4 | 1024 | 1024 | 0 | 3356.0 | 3356.3 | +0.0% |
| gemma-4-E4B_q4_0-it | 4 | 1024 | 1024 | 16384 | 2861.3 | 2788.7 | -2.5% |
| llama-3.2-3b-instruct-q4_0 | 4 | 1024 | 1024 | 0 | 5471.0 | 5411.1 | -1.1% |
| llama-3.2-3b-instruct-q4_0 | 4 | 1024 | 1024 | 16384 | 1770.8 | 1595.5 | -9.9% |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | 1024 | 1024 | 0 | 987.0 | 1067.4 | +8.1% |
| Qwen3.5-4B-Q4 | 4 | 1024 | 1024 | 0 | 3159.3 | 3138.5 | -0.7% |
| Qwen3.5-4B-Q4 | 4 | 1024 | 1024 | 16384 | 2482.3 | 2474.8 | -0.3% |

## Cross-core scaling (GGML_HEXAGON_FA_SPLIT=1=Base vs GGML_HEXAGON_FA_SPLIT=2=Opt), per model (split=row)

Uses the largest ubatch tested for each model. "vs 1c" is that series's own
speedup over its 1-core result; PP gain is Opt vs Base at that core count.

### gemma-4-E2B_q4_0-it (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 4 | 4028 | NA | 4047 | NA | +0.5% | 41.1 | 41.1 |

### gemma-4-E4B_q4_0-it (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 4 | 2861 | NA | 2789 | NA | -2.5% | 30.3 | 30.3 |

### llama-3.2-3b-instruct-q4_0 (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 4 | 1771 | NA | 1596 | NA | -9.9% | 22.7 | 21.9 |

### Qwen3.5-35B-A3B-UD-Q4_K_M (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 4 | 987 | NA | 1067 | NA | +8.1% | 25.7 | 25.0 |

### Qwen3.5-4B-Q4 (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 4 | 2482 | NA | 2475 | NA | -0.3% | 30.4 | 30.4 |

## Quality gate

**3 configuration(s) FAILED quality checks — see table above.**

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/20261022-185551/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/20261022-185551/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/20261022-185551/logs/cookie_*.txt`
