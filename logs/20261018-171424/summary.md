# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `2a8e3240e` (`ebateni/adaptive-split`)
- device binaries: `96ed45522 (ebateni/adaptive-split) built 2026-10-18T14:27:49-04:00`
- date: 2026-10-18T19:01:16-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_FA_SPLIT` = 0,2

## Results

| model | cores | split | GGML_HEXAGON_FA_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct-q4_0 | 1 | single | 0 | 1024 | 2048 | 0 | 1696.66 | 1.00x | 22.73 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 0 | 2048 | 2048 | 0 | 1618.43 | 1.00x | 22.73 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 2 | 1024 | 2048 | 0 | 1695.21 | 1.00x | 22.73 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 2 | 2048 | 2048 | 0 | 1618.10 | 1.00x | 22.74 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 0 | 1024 | 2048 | 16384 | 473.01 | 1.00x | 10.98 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 0 | 2048 | 2048 | 16384 | 492.73 | 1.00x | 10.88 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 2 | 1024 | 2048 | 16384 | 472.49 | 1.00x | 10.98 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 2 | 2048 | 2048 | 16384 | 492.56 | 1.00x | 10.88 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 0 | 1024 | 2048 | 0 | 5067.95 | 2.99x | 45.54 | 2.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 0 | 2048 | 2048 | 0 | 5058.62 | 3.13x | 45.54 | 2.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 2 | 1024 | 2048 | 0 | 5070.16 | 2.99x | 45.51 | 2.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 2 | 2048 | 2048 | 0 | 5025.58 | 3.11x | 45.57 | 2.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 0 | 1024 | 2048 | 16384 | 1566.17 | 3.31x | 20.68 | 1.88x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 0 | 2048 | 2048 | 16384 | 1618.82 | 3.29x | 20.22 | 1.86x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 2 | 1024 | 2048 | 16384 | 1559.65 | 3.30x | 20.66 | 1.88x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 2 | 2048 | 2048 | 16384 | 1629.35 | 3.31x | 20.26 | 1.86x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 0 | 1024 | 2048 | 0 | 1104.36 | 1.00x | 18.24 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 0 | 2048 | 2048 | 0 | 1089.19 | 1.00x | 18.24 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 2 | 1024 | 2048 | 0 | 1104.76 | 1.00x | 18.24 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 2 | 2048 | 2048 | 0 | 1087.35 | 1.00x | 18.24 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 0 | 1024 | 2048 | 16384 | 854.56 | 1.00x | 15.63 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 0 | 2048 | 2048 | 16384 | 861.33 | 1.00x | 15.62 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 2 | 1024 | 2048 | 16384 | 851.61 | 1.00x | 15.62 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 2 | 2048 | 2048 | 16384 | 862.26 | 1.00x | 15.62 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 0 | 1024 | 2048 | 0 | 3106.06 | 2.81x | 37.47 | 2.05x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 0 | 2048 | 2048 | 0 | 3178.02 | 2.92x | 37.43 | 2.05x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 2 | 1024 | 2048 | 0 | 3108.05 | 2.81x | 37.50 | 2.06x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 2 | 2048 | 2048 | 0 | 3182.48 | 2.93x | 37.48 | 2.05x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 0 | 1024 | 2048 | 16384 | 2568.87 | 3.01x | 33.45 | 2.14x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 0 | 2048 | 2048 | 16384 | 2560.50 | 2.97x | 33.51 | 2.15x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 2 | 1024 | 2048 | 16384 | 2555.34 | 3.00x | 33.42 | 2.14x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 2 | 2048 | 2048 | 16384 | 2596.81 | 3.01x | 33.49 | 2.14x | 9/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 0 | 1024 | 2048 | 0 | 1221.97 | 1.00x | 17.05 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 0 | 2048 | 2048 | 0 | 1213.57 | 1.00x | 16.67 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 2 | 1024 | 2048 | 0 | 1221.96 | 1.00x | 17.05 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 2 | 2048 | 2048 | 0 | 1213.01 | 1.00x | 16.66 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 0 | 1024 | 2048 | 16384 | 1001.66 | 1.00x | 14.36 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 0 | 2048 | 2048 | 16384 | 979.48 | 1.00x | 14.65 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 2 | 1024 | 2048 | 16384 | 999.88 | 1.00x | 14.36 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 2 | 2048 | 2048 | 16384 | 980.33 | 1.00x | 14.11 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 0 | 1024 | 2048 | 0 | 2806.95 | 2.30x | 32.89 | 1.93x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 0 | 2048 | 2048 | 0 | 3347.92 | 2.76x | 31.53 | 1.89x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 2 | 1024 | 2048 | 0 | 3202.72 | 2.62x | 34.25 | 2.01x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 2 | 2048 | 2048 | 0 | 3325.64 | 2.74x | 32.71 | 1.96x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 0 | 1024 | 2048 | 16384 | 2476.85 | 2.47x | 24.44 | 1.70x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 0 | 2048 | 2048 | 16384 | 2839.30 | 2.90x | 23.78 | 1.62x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 2 | 1024 | 2048 | 16384 | 2801.33 | 2.80x | 29.62 | 2.06x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 2 | 2048 | 2048 | 16384 | 2790.27 | 2.85x | 27.56 | 1.95x | 7/10 | PASS | OK |

## PP delta (GGML_HEXAGON_FA_SPLIT=0 -> 2), per model / cores / ubatch (split=row)

If the percentage shrinks as ubatch grows, the change and ubatch are NOT additive.

| model | cores | ubatch | pp_tok | depth | PP GGML_HEXAGON_FA_SPLIT=0 | PP GGML_HEXAGON_FA_SPLIT=2 | delta |
|---|---|---|---|---|---|---|---|
| gemma-4-E4B_q4_0-it | 1 | 1024 | 2048 | 0 | 1222.0 | 1222.0 | -0.0% |
| gemma-4-E4B_q4_0-it | 1 | 1024 | 2048 | 16384 | 1001.7 | 999.9 | -0.2% |
| gemma-4-E4B_q4_0-it | 1 | 2048 | 2048 | 0 | 1213.6 | 1213.0 | -0.0% |
| gemma-4-E4B_q4_0-it | 1 | 2048 | 2048 | 16384 | 979.5 | 980.3 | +0.1% |
| gemma-4-E4B_q4_0-it | 4 | 1024 | 2048 | 0 | 2806.9 | 3202.7 | +14.1% |
| gemma-4-E4B_q4_0-it | 4 | 1024 | 2048 | 16384 | 2476.8 | 2801.3 | +13.1% |
| gemma-4-E4B_q4_0-it | 4 | 2048 | 2048 | 0 | 3347.9 | 3325.6 | -0.7% |
| gemma-4-E4B_q4_0-it | 4 | 2048 | 2048 | 16384 | 2839.3 | 2790.3 | -1.7% |
| llama-3.2-3b-instruct-q4_0 | 1 | 1024 | 2048 | 0 | 1696.7 | 1695.2 | -0.1% |
| llama-3.2-3b-instruct-q4_0 | 1 | 1024 | 2048 | 16384 | 473.0 | 472.5 | -0.1% |
| llama-3.2-3b-instruct-q4_0 | 1 | 2048 | 2048 | 0 | 1618.4 | 1618.1 | -0.0% |
| llama-3.2-3b-instruct-q4_0 | 1 | 2048 | 2048 | 16384 | 492.7 | 492.6 | -0.0% |
| llama-3.2-3b-instruct-q4_0 | 4 | 1024 | 2048 | 0 | 5067.9 | 5070.2 | +0.0% |
| llama-3.2-3b-instruct-q4_0 | 4 | 1024 | 2048 | 16384 | 1566.2 | 1559.7 | -0.4% |
| llama-3.2-3b-instruct-q4_0 | 4 | 2048 | 2048 | 0 | 5058.6 | 5025.6 | -0.7% |
| llama-3.2-3b-instruct-q4_0 | 4 | 2048 | 2048 | 16384 | 1618.8 | 1629.3 | +0.7% |
| Qwen3.5-4B-Q4 | 1 | 1024 | 2048 | 0 | 1104.4 | 1104.8 | +0.0% |
| Qwen3.5-4B-Q4 | 1 | 1024 | 2048 | 16384 | 854.6 | 851.6 | -0.3% |
| Qwen3.5-4B-Q4 | 1 | 2048 | 2048 | 0 | 1089.2 | 1087.3 | -0.2% |
| Qwen3.5-4B-Q4 | 1 | 2048 | 2048 | 16384 | 861.3 | 862.3 | +0.1% |
| Qwen3.5-4B-Q4 | 4 | 1024 | 2048 | 0 | 3106.1 | 3108.1 | +0.1% |
| Qwen3.5-4B-Q4 | 4 | 1024 | 2048 | 16384 | 2568.9 | 2555.3 | -0.5% |
| Qwen3.5-4B-Q4 | 4 | 2048 | 2048 | 0 | 3178.0 | 3182.5 | +0.1% |
| Qwen3.5-4B-Q4 | 4 | 2048 | 2048 | 16384 | 2560.5 | 2596.8 | +1.4% |

## Cross-core scaling (GGML_HEXAGON_FA_SPLIT=0=Base vs GGML_HEXAGON_FA_SPLIT=2=Opt), per model (split=row)

Uses the largest ubatch tested for each model. "vs 1c" is that series's own
speedup over its 1-core result; PP gain is Opt vs Base at that core count.

### gemma-4-E4B_q4_0-it (ubatch=2048)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 979 | 1.00x | 980 | 1.00x | +0.1% | 14.7 | 14.1 |
| 4 | 2839 | 2.90x | 2790 | 2.85x | -1.7% | 23.8 | 27.6 |

### llama-3.2-3b-instruct-q4_0 (ubatch=2048)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 493 | 1.00x | 493 | 1.00x | -0.0% | 10.9 | 10.9 |
| 4 | 1619 | 3.29x | 1629 | 3.31x | +0.7% | 20.2 | 20.3 |

### Qwen3.5-4B-Q4 (ubatch=2048)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 861 | 1.00x | 862 | 1.00x | +0.1% | 15.6 | 15.6 |
| 4 | 2560 | 2.97x | 2597 | 3.01x | +1.4% | 33.5 | 33.5 |

## Quality gate

All 48 checked configuration(s) passed needle + cookie (see table above for IQ score per configuration).

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-171424/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-171424/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-171424/logs/cookie_*.txt`
