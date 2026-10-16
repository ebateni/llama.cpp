# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `549ffa703` (`ebateni/adaptive-split`)
- device binaries: `d3f87a011 (ebateni/adaptive-split) built 2026-10-15T20:03:51-04:00`
- date: 2026-10-16T00:22:51-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_FA_SPLIT` = 0,1

## Results

| model | cores | split | GGML_HEXAGON_FA_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct-q4_0 | 1 | single | 0 | 1024 | 2048 | 0 | 1713.06 | 1.00x | 23.47 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 0 | 2048 | 2048 | 0 | 1635.20 | 1.00x | 23.50 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 1 | 1024 | 2048 | 0 | 1712.22 | 1.00x | 23.47 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 1 | 2048 | 2048 | 0 | 1633.56 | 1.00x | 23.50 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 0 | 1024 | 2048 | 16384 | 476.22 | 1.00x | 11.15 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 0 | 2048 | 2048 | 16384 | 495.37 | 1.00x | 11.05 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 1 | 1024 | 2048 | 16384 | 476.35 | 1.00x | 11.15 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 1 | single | 1 | 2048 | 2048 | 16384 | 495.03 | 1.00x | 11.05 | 1.00x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 0 | 1024 | 2048 | 0 | 5152.46 | 3.01x | 47.57 | 2.03x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 0 | 2048 | 2048 | 0 | 5154.33 | 3.15x | 47.50 | 2.02x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 1 | 1024 | 2048 | 0 | 5141.59 | 3.00x | 47.63 | 2.03x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 1 | 2048 | 2048 | 0 | 5124.01 | 3.14x | 47.46 | 2.02x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 0 | 1024 | 2048 | 16384 | 1566.93 | 3.29x | 20.68 | 1.85x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 0 | 2048 | 2048 | 16384 | 1623.96 | 3.28x | 20.43 | 1.85x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 1 | 1024 | 2048 | 16384 | 1563.30 | 3.28x | 20.69 | 1.86x | 9/10 | PASS | OK |
| llama-3.2-3b-instruct-q4_0 | 4 | row | 1 | 2048 | 2048 | 16384 | 1631.14 | 3.30x | 20.40 | 1.85x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 0 | 1024 | 2048 | 0 | 1104.90 | 1.00x | 18.23 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 0 | 2048 | 2048 | 0 | 1088.52 | 1.00x | 18.24 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 1 | 1024 | 2048 | 0 | 1103.88 | 1.00x | 18.24 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 1 | 2048 | 2048 | 0 | 1087.08 | 1.00x | 18.24 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 0 | 1024 | 2048 | 16384 | 836.75 | 1.00x | 14.67 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 0 | 2048 | 2048 | 16384 | 827.58 | 1.00x | 14.67 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 1 | 1024 | 2048 | 16384 | 837.35 | 1.00x | 14.67 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 1 | single | 1 | 2048 | 2048 | 16384 | 827.48 | 1.00x | 14.66 | 1.00x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 0 | 1024 | 2048 | 0 | 3094.25 | 2.80x | 35.99 | 1.97x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 0 | 2048 | 2048 | 0 | 3113.76 | 2.86x | 36.12 | 1.98x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 1 | 1024 | 2048 | 0 | 3155.15 | 2.86x | 37.62 | 2.06x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 1 | 2048 | 2048 | 0 | 3213.22 | 2.96x | 37.47 | 2.05x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 0 | 1024 | 2048 | 16384 | 2573.00 | 3.07x | 33.54 | 2.29x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 0 | 2048 | 2048 | 16384 | 2602.62 | 3.14x | 33.54 | 2.29x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 1 | 1024 | 2048 | 16384 | 2563.19 | 3.06x | 33.68 | 2.30x | 9/10 | PASS | OK |
| Qwen3.5-4B-Q4 | 4 | row | 1 | 2048 | 2048 | 16384 | 2605.20 | 3.15x | 33.57 | 2.29x | 9/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 0 | 1024 | 2048 | 0 | 1231.34 | 1.00x | 17.67 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 0 | 2048 | 2048 | 0 | 1221.65 | 1.00x | 17.26 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 1 | 1024 | 2048 | 0 | 1226.07 | 1.00x | 17.05 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 1 | 2048 | 2048 | 0 | 1222.06 | 1.00x | 16.66 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 0 | 1024 | 2048 | 16384 | 1000.21 | 1.00x | 14.36 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 0 | 2048 | 2048 | 16384 | 981.08 | 1.00x | 14.64 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 1 | 1024 | 2048 | 16384 | 1001.45 | 1.00x | 14.91 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 1 | 2048 | 2048 | 16384 | 984.02 | 1.00x | 14.11 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 0 | 1024 | 2048 | 0 | 2857.01 | 2.32x | 32.93 | 1.86x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 0 | 2048 | 2048 | 0 | 3358.01 | 2.75x | 31.58 | 1.83x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 1 | 1024 | 2048 | 0 | 3257.67 | 2.66x | 35.31 | 2.07x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 1 | 2048 | 2048 | 0 | 3315.58 | 2.71x | 32.92 | 1.98x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 0 | 1024 | 2048 | 16384 | 2475.75 | 2.48x | 24.47 | 1.70x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 0 | 2048 | 2048 | 16384 | 2897.11 | 2.95x | 23.77 | 1.62x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 1 | 1024 | 2048 | 16384 | 2808.05 | 2.80x | 29.63 | 1.99x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 1 | 2048 | 2048 | 16384 | 2822.13 | 2.87x | 27.54 | 1.95x | 7/10 | PASS | OK |

## PP delta (GGML_HEXAGON_FA_SPLIT=0 -> 1), per model / cores / ubatch (split=row)

If the percentage shrinks as ubatch grows, the change and ubatch are NOT additive.

| model | cores | ubatch | pp_tok | depth | PP GGML_HEXAGON_FA_SPLIT=0 | PP GGML_HEXAGON_FA_SPLIT=1 | delta |
|---|---|---|---|---|---|---|---|
| gemma-4-E4B_q4_0-it | 1 | 1024 | 2048 | 0 | 1231.3 | 1226.1 | -0.4% |
| gemma-4-E4B_q4_0-it | 1 | 1024 | 2048 | 16384 | 1000.2 | 1001.5 | +0.1% |
| gemma-4-E4B_q4_0-it | 1 | 2048 | 2048 | 0 | 1221.7 | 1222.1 | +0.0% |
| gemma-4-E4B_q4_0-it | 1 | 2048 | 2048 | 16384 | 981.1 | 984.0 | +0.3% |
| gemma-4-E4B_q4_0-it | 4 | 1024 | 2048 | 0 | 2857.0 | 3257.7 | +14.0% |
| gemma-4-E4B_q4_0-it | 4 | 1024 | 2048 | 16384 | 2475.8 | 2808.1 | +13.4% |
| gemma-4-E4B_q4_0-it | 4 | 2048 | 2048 | 0 | 3358.0 | 3315.6 | -1.3% |
| gemma-4-E4B_q4_0-it | 4 | 2048 | 2048 | 16384 | 2897.1 | 2822.1 | -2.6% |
| llama-3.2-3b-instruct-q4_0 | 1 | 1024 | 2048 | 0 | 1713.1 | 1712.2 | -0.0% |
| llama-3.2-3b-instruct-q4_0 | 1 | 1024 | 2048 | 16384 | 476.2 | 476.4 | +0.0% |
| llama-3.2-3b-instruct-q4_0 | 1 | 2048 | 2048 | 0 | 1635.2 | 1633.6 | -0.1% |
| llama-3.2-3b-instruct-q4_0 | 1 | 2048 | 2048 | 16384 | 495.4 | 495.0 | -0.1% |
| llama-3.2-3b-instruct-q4_0 | 4 | 1024 | 2048 | 0 | 5152.5 | 5141.6 | -0.2% |
| llama-3.2-3b-instruct-q4_0 | 4 | 1024 | 2048 | 16384 | 1566.9 | 1563.3 | -0.2% |
| llama-3.2-3b-instruct-q4_0 | 4 | 2048 | 2048 | 0 | 5154.3 | 5124.0 | -0.6% |
| llama-3.2-3b-instruct-q4_0 | 4 | 2048 | 2048 | 16384 | 1624.0 | 1631.1 | +0.4% |
| Qwen3.5-4B-Q4 | 1 | 1024 | 2048 | 0 | 1104.9 | 1103.9 | -0.1% |
| Qwen3.5-4B-Q4 | 1 | 1024 | 2048 | 16384 | 836.8 | 837.4 | +0.1% |
| Qwen3.5-4B-Q4 | 1 | 2048 | 2048 | 0 | 1088.5 | 1087.1 | -0.1% |
| Qwen3.5-4B-Q4 | 1 | 2048 | 2048 | 16384 | 827.6 | 827.5 | -0.0% |
| Qwen3.5-4B-Q4 | 4 | 1024 | 2048 | 0 | 3094.2 | 3155.2 | +2.0% |
| Qwen3.5-4B-Q4 | 4 | 1024 | 2048 | 16384 | 2573.0 | 2563.2 | -0.4% |
| Qwen3.5-4B-Q4 | 4 | 2048 | 2048 | 0 | 3113.8 | 3213.2 | +3.2% |
| Qwen3.5-4B-Q4 | 4 | 2048 | 2048 | 16384 | 2602.6 | 2605.2 | +0.1% |

## Cross-core scaling (GGML_HEXAGON_FA_SPLIT=0=Base vs GGML_HEXAGON_FA_SPLIT=1=Opt), per model (split=row)

Uses the largest ubatch tested for each model. "vs 1c" is that series's own
speedup over its 1-core result; PP gain is Opt vs Base at that core count.

### gemma-4-E4B_q4_0-it (ubatch=2048)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 981 | 1.00x | 984 | 1.00x | +0.3% | 14.6 | 14.1 |
| 4 | 2897 | 2.95x | 2822 | 2.87x | -2.6% | 23.8 | 27.5 |

### llama-3.2-3b-instruct-q4_0 (ubatch=2048)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 495 | 1.00x | 495 | 1.00x | -0.1% | 11.1 | 11.1 |
| 4 | 1624 | 3.28x | 1631 | 3.30x | +0.4% | 20.4 | 20.4 |

### Qwen3.5-4B-Q4 (ubatch=2048)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 828 | 1.00x | 827 | 1.00x | -0.0% | 14.7 | 14.7 |
| 4 | 2603 | 3.14x | 2605 | 3.15x | +0.1% | 33.5 | 33.6 |

## Quality gate

All 48 checked configuration(s) passed needle + cookie (see table above for IQ score per configuration).

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-223555/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-223555/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-223555/logs/cookie_*.txt`
