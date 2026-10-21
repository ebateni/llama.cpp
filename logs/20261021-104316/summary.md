# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `` (``)
- device binaries: `ba0e946ad (ebateni/adaptive-split) built 2026-10-20T18:57:43-04:00`
- date: 2026-10-21T12:18:31-04:00
- reps: 5, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_FA_SPLIT` = 0,1

## Results

| model | cores | split | GGML_HEXAGON_FA_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-E4B_q4_0-it | 1 | single | 0 | 1024 | 1024 | 0 | 1240.75 | 1.00x | 17.36 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 1 | single | 1 | 1024 | 1024 | 0 | 1239.65 | 1.00x | 17.36 | 1.00x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 0 | 1024 | 1024 | 0 | 2922.26 | 2.36x | 33.93 | 1.95x | 7/10 | PASS | OK |
| gemma-4-E4B_q4_0-it | 4 | row | 1 | 1024 | 1024 | 0 | 3351.40 | 2.70x | 35.80 | 2.06x | 7/10 | PASS | OK |
| gemma-4-E2B_q4_0-it | 1 | single | 0 | 1024 | 1024 | 0 | 2062.93 | 1.00x | 31.69 | 1.00x | 5/10 | PASS | OK |
| gemma-4-E2B_q4_0-it | 1 | single | 1 | 1024 | 1024 | 0 | 2064.14 | 1.00x | 31.69 | 1.00x | 5/10 | PASS | OK |
| gemma-4-E2B_q4_0-it | 4 | row | 0 | 1024 | 1024 | 0 | 5237.94 | 2.54x | 54.04 | 1.71x | 5/10 | PASS | OK |
| gemma-4-E2B_q4_0-it | 4 | row | 1 | 1024 | 1024 | 0 | 5211.44 | 2.52x | 54.36 | 1.72x | 5/10 | PASS | OK |
| gemma-4-26B_q4_0-it | 1 | single | 0 | 1024 | 1024 | 0 | 558.24 | 1.00x | 16.49 | 1.00x | 9/10 | PASS | GARBAGE(glued) |
| gemma-4-26B_q4_0-it | 1 | single | 1 | 1024 | 1024 | 0 | 564.30 | 1.00x | 16.51 | 1.00x | 9/10 | PASS | GARBAGE(glued) |
| gemma-4-26B_q4_0-it | 4 | row | 0 | 1024 | 1024 | 0 | 1296.40 | 2.32x | 22.44 | 1.36x | 9/10 | PASS | GARBAGE(glued) |
| gemma-4-26B_q4_0-it | 4 | row | 1 | 1024 | 1024 | 0 | 1382.10 | 2.45x | 25.89 | 1.57x | 9/10 | PASS | GARBAGE(glued) |
| gemma-4-12b-it-qat-q4_0 | 1 | single | 0 | 1024 | 1024 | 0 | 586.28 | 1.00x | 7.84 | 1.00x | 2/10 | PASS | GARBAGE(phrase) |
| gemma-4-12b-it-qat-q4_0 | 1 | single | 1 | 1024 | 1024 | 0 | 590.37 | 1.00x | 7.84 | 1.00x | 2/10 | PASS | GARBAGE(phrase) |
| gemma-4-12b-it-qat-q4_0 | 4 | row | 0 | 1024 | 1024 | 0 | 1563.85 | 2.67x | 14.59 | 1.86x | 2/10 | PASS | GARBAGE(phrase) |
| gemma-4-12b-it-qat-q4_0 | 4 | row | 1 | 1024 | 1024 | 0 | 1796.44 | 3.04x | 17.15 | 2.19x | 2/10 | PASS | GARBAGE(phrase) |

## PP delta (GGML_HEXAGON_FA_SPLIT=0 -> 1), per model / cores / ubatch (split=row)

If the percentage shrinks as ubatch grows, the change and ubatch are NOT additive.

| model | cores | ubatch | pp_tok | depth | PP GGML_HEXAGON_FA_SPLIT=0 | PP GGML_HEXAGON_FA_SPLIT=1 | delta |
|---|---|---|---|---|---|---|---|
| gemma-4-12b-it-qat-q4_0 | 1 | 1024 | 1024 | 0 | 586.3 | 590.4 | +0.7% |
| gemma-4-12b-it-qat-q4_0 | 4 | 1024 | 1024 | 0 | 1563.8 | 1796.4 | +14.9% |
| gemma-4-26B_q4_0-it | 1 | 1024 | 1024 | 0 | 558.2 | 564.3 | +1.1% |
| gemma-4-26B_q4_0-it | 4 | 1024 | 1024 | 0 | 1296.4 | 1382.1 | +6.6% |
| gemma-4-E2B_q4_0-it | 1 | 1024 | 1024 | 0 | 2062.9 | 2064.1 | +0.1% |
| gemma-4-E2B_q4_0-it | 4 | 1024 | 1024 | 0 | 5237.9 | 5211.4 | -0.5% |
| gemma-4-E4B_q4_0-it | 1 | 1024 | 1024 | 0 | 1240.8 | 1239.7 | -0.1% |
| gemma-4-E4B_q4_0-it | 4 | 1024 | 1024 | 0 | 2922.3 | 3351.4 | +14.7% |

## Cross-core scaling (GGML_HEXAGON_FA_SPLIT=0=Base vs GGML_HEXAGON_FA_SPLIT=1=Opt), per model (split=row)

Uses the largest ubatch tested for each model. "vs 1c" is that series's own
speedup over its 1-core result; PP gain is Opt vs Base at that core count.

### gemma-4-12b-it-qat-q4_0 (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 586 | 1.00x | 590 | 1.00x | +0.7% | 7.8 | 7.8 |
| 4 | 1564 | 2.67x | 1796 | 3.04x | +14.9% | 14.6 | 17.1 |

### gemma-4-26B_q4_0-it (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 558 | 1.00x | 564 | 1.00x | +1.1% | 16.5 | 16.5 |
| 4 | 1296 | 2.32x | 1382 | 2.45x | +6.6% | 22.4 | 25.9 |

### gemma-4-E2B_q4_0-it (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 2063 | 1.00x | 2064 | 1.00x | +0.1% | 31.7 | 31.7 |
| 4 | 5238 | 2.54x | 5211 | 2.52x | -0.5% | 54.0 | 54.4 |

### gemma-4-E4B_q4_0-it (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 1241 | 1.00x | 1240 | 1.00x | -0.1% | 17.4 | 17.4 |
| 4 | 2922 | 2.36x | 3351 | 2.70x | +14.7% | 33.9 | 35.8 |

## Quality gate

**8 configuration(s) FAILED quality checks — see table above.**

Raw logs: `/local/mnt2/workspace2/llama.cpp_Seca/20261021-104316/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp_Seca/20261021-104316/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp_Seca/20261021-104316/logs/cookie_*.txt`
