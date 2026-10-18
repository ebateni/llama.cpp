# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `3115eba1f` (`ebateni/adaptive-split`)
- device binaries: `96ed45522 (ebateni/adaptive-split) built 2026-10-18T14:27:49-04:00`
- date: 2026-10-18T19:54:52-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_FA_SPLIT` = 1,2

## Results

| model | cores | split | GGML_HEXAGON_FA_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-E4B_q4_0-it | 1 | single | 1 | 1024 | 2048 | 0 | 1232.28 | 1.00x | 17.67 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 1 | 2048 | 2048 | 0 | 1213.85 | 1.00x | 16.66 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 2 | 1024 | 2048 | 0 | 1231.83 | 1.00x | 17.66 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 2 | 2048 | 2048 | 0 | 1213.56 | 1.00x | 16.67 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 1 | 1024 | 2048 | 16384 | 1002.55 | 1.00x | 14.91 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 1 | 2048 | 2048 | 16384 | 985.63 | 1.00x | 14.67 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 2 | 1024 | 2048 | 16384 | 1007.14 | 1.00x | 14.35 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 2 | 2048 | 2048 | 16384 | 980.21 | 1.00x | 14.11 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 1 | 1024 | 2048 | 0 | 3212.86 | 2.61x | 34.24 | 1.94x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 1 | 2048 | 2048 | 0 | 3282.99 | 2.70x | 32.88 | 1.97x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 2 | 1024 | 2048 | 0 | 3191.05 | 2.59x | 34.28 | 1.94x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 2 | 2048 | 2048 | 0 | 3297.45 | 2.72x | 32.86 | 1.97x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 1 | 1024 | 2048 | 16384 | 2809.25 | 2.80x | 28.43 | 1.91x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 1 | 2048 | 2048 | 16384 | 2792.73 | 2.83x | 27.52 | 1.88x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 2 | 1024 | 2048 | 16384 | 2768.54 | 2.75x | 29.60 | 2.06x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 2 | 2048 | 2048 | 16384 | 2805.19 | 2.86x | 27.53 | 1.95x | 7/10 | PASS | OK |

## PP delta (GGML_HEXAGON_FA_SPLIT=1 -> 2), per model / cores / ubatch (split=row)

If the percentage shrinks as ubatch grows, the change and ubatch are NOT additive.

| model | cores | ubatch | pp_tok | depth | PP GGML_HEXAGON_FA_SPLIT=1 | PP GGML_HEXAGON_FA_SPLIT=2 | delta |
|---|---|---|---|---|---|---|---|
| gemma-4-E4B_q4_0-it | 1 | 1024 | 2048 | 0 | 1232.3 | 1231.8 | -0.0% |
| gemma-4-E4B_q4_0-it | 1 | 1024 | 2048 | 16384 | 1002.5 | 1007.1 | +0.5% |
| gemma-4-E4B_q4_0-it | 1 | 2048 | 2048 | 0 | 1213.8 | 1213.6 | -0.0% |
| gemma-4-E4B_q4_0-it | 1 | 2048 | 2048 | 16384 | 985.6 | 980.2 | -0.5% |
| gemma-4-E4B_q4_0-it | 4 | 1024 | 2048 | 0 | 3212.9 | 3191.1 | -0.7% |
| gemma-4-E4B_q4_0-it | 4 | 1024 | 2048 | 16384 | 2809.2 | 2768.5 | -1.4% |
| gemma-4-E4B_q4_0-it | 4 | 2048 | 2048 | 0 | 3283.0 | 3297.4 | +0.4% |
| gemma-4-E4B_q4_0-it | 4 | 2048 | 2048 | 16384 | 2792.7 | 2805.2 | +0.4% |

## Cross-core scaling (GGML_HEXAGON_FA_SPLIT=1=Base vs GGML_HEXAGON_FA_SPLIT=2=Opt), per model (split=row)

Uses the largest ubatch tested for each model. "vs 1c" is that series's own
speedup over its 1-core result; PP gain is Opt vs Base at that core count.

### gemma-4-E4B_q4_0-it (ubatch=2048)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 986 | 1.00x | 980 | 1.00x | -0.5% | 14.7 | 14.1 |
| 4 | 2793 | 2.83x | 2805 | 2.86x | +0.4% | 27.5 | 27.5 |

## Quality gate

All 16 checked configuration(s) passed needle + cookie (see table above for IQ score per configuration).

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-191245/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-191245/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-191245/logs/cookie_*.txt`
