# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `e2a7f9f27` (`ebateni/fa-team-zigzag`)
- device binaries: `e2a7f9f27 (ebateni/fa-team-zigzag) built 2026-10-22T00:17:33-04:00`
- date: 2026-10-22T00:23:07-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_FA_SPLIT` = 1,2

## Results

| model | cores | split | GGML_HEXAGON_FA_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-E2B_q4_0-it | 4 | row | 1 | 1024 | 1024 | 0 | 5157.06 | NA | 54.68 | NA | 5/10 | PASS | OK |
| gemma-4-E2B_q4_0-it | 4 | row | 2 | 1024 | 1024 | 0 | 5164.84 | NA | 54.72 | NA | 5/10 | PASS | OK |

## PP delta (GGML_HEXAGON_FA_SPLIT=1 -> 2), per model / cores / ubatch (split=row)

If the percentage shrinks as ubatch grows, the change and ubatch are NOT additive.

| model | cores | ubatch | pp_tok | depth | PP GGML_HEXAGON_FA_SPLIT=1 | PP GGML_HEXAGON_FA_SPLIT=2 | delta |
|---|---|---|---|---|---|---|---|
| gemma-4-E2B_q4_0-it | 4 | 1024 | 1024 | 0 | 5157.1 | 5164.8 | +0.2% |

## Cross-core scaling (GGML_HEXAGON_FA_SPLIT=1=Base vs GGML_HEXAGON_FA_SPLIT=2=Opt), per model (split=row)

Uses the largest ubatch tested for each model. "vs 1c" is that series's own
speedup over its 1-core result; PP gain is Opt vs Base at that core count.

### gemma-4-E2B_q4_0-it (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 4 | 5157 | NA | 5165 | NA | +0.2% | 54.7 | 54.7 |

## Quality gate

All 2 checked configuration(s) passed needle + cookie (see table above for IQ score per configuration).

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261022-001604/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261022-001604/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261022-001604/logs/cookie_*.txt`
