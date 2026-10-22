# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `41a4fc7ef` (`ebateni/fa-team-zigzag`)
- device binaries: `41a4fc7ef (ebateni/fa-team-zigzag) built 2026-10-21T23:34:58-04:00`
- date: 2026-10-21T23:45:28-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_FA_SPLIT` = 1,2

## Results

| model | cores | split | GGML_HEXAGON_FA_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-E2B_q4_0-it | 1 | single | 1 | 1024 | 1024 | 0 | 2058.43 | 1.00x | 31.71 | 1.00x | 5/10 | PASS | OK |
| gemma-4-E2B_q4_0-it | 1 | single | 2 | 1024 | 1024 | 0 | 2045.90 | 1.00x | 32.24 | 1.00x | 5/10 | PASS | OK |
| gemma-4-E2B_q4_0-it | 4 | row | 1 | 1024 | 1024 | 0 | 5125.83 | 2.49x | 53.81 | 1.70x | 5/10 | PASS | OK |
| gemma-4-E2B_q4_0-it | 4 | row | 2 | 1024 | 1024 | 0 | 5132.79 | 2.51x | 54.12 | 1.68x | 5/10 | PASS | OK |

## PP delta (GGML_HEXAGON_FA_SPLIT=1 -> 2), per model / cores / ubatch (split=row)

If the percentage shrinks as ubatch grows, the change and ubatch are NOT additive.

| model | cores | ubatch | pp_tok | depth | PP GGML_HEXAGON_FA_SPLIT=1 | PP GGML_HEXAGON_FA_SPLIT=2 | delta |
|---|---|---|---|---|---|---|---|
| gemma-4-E2B_q4_0-it | 1 | 1024 | 1024 | 0 | 2058.4 | 2045.9 | -0.6% |
| gemma-4-E2B_q4_0-it | 4 | 1024 | 1024 | 0 | 5125.8 | 5132.8 | +0.1% |

## Cross-core scaling (GGML_HEXAGON_FA_SPLIT=1=Base vs GGML_HEXAGON_FA_SPLIT=2=Opt), per model (split=row)

Uses the largest ubatch tested for each model. "vs 1c" is that series's own
speedup over its 1-core result; PP gain is Opt vs Base at that core count.

### gemma-4-E2B_q4_0-it (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 2058 | 1.00x | 2046 | 1.00x | -0.6% | 31.7 | 32.2 |
| 4 | 5126 | 2.49x | 5133 | 2.51x | +0.1% | 53.8 | 54.1 |

## Quality gate

All 4 checked configuration(s) passed needle + cookie (see table above for IQ score per configuration).

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261021-233331/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261021-233331/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261021-233331/logs/cookie_*.txt`
