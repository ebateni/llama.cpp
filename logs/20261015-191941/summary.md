# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `75360014b3` (`dev/ebateni/seq-parallel-prototype`)
- device binaries: `75360014b3 (dev/ebateni/seq-parallel-prototype) +dirty built 2026-10-15T19:20:55-04:00`
- date: 2026-10-15T19:51:22-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_OPFUSION` = 1,0

## Results

| model | cores | split | GGML_HEXAGON_OPFUSION | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-12b-it-qat-q4_0 | 1 | single | 1 | 1024 | 1024 | 0 | 574.47 | 1.00x | 7.53 | 1.00x | 1/10 | PASS | GARBAGE(phrase) |
| gemma-4-12b-it-qat-q4_0 | 1 | single | 0 | 1024 | 1024 | 0 | 528.91 | 1.00x | 7.48 | 1.00x | 1/10 | PASS | GARBAGE(phrase) |
| gemma-4-12b-it-qat-q4_0 | 4 | row | 1 | 1024 | 1024 | 0 | 1579.87 | 2.75x | 15.19 | 2.02x | 1/10 | PASS | GARBAGE(phrase) |
| gemma-4-12b-it-qat-q4_0 | 4 | row | 0 | 1024 | 1024 | 0 | 1455.33 | 2.75x | 14.69 | 1.96x | 1/10 | PASS | GARBAGE(phrase) |
| gemma-2-2b_Q4_0 | 1 | single | 1 | 1024 | 1024 | 0 | 2319.00 | 1.00x | 26.07 | 1.00x | 6/10 | PASS | OK |
| gemma-2-2b_Q4_0 | 1 | single | 0 | 1024 | 1024 | 0 | 2143.07 | 1.00x | 25.77 | 1.00x | 6/10 | PASS | OK |
| gemma-2-2b_Q4_0 | 4 | row | 1 | 1024 | 1024 | 0 | 5059.64 | 2.18x | 46.06 | 1.77x | 6/10 | PASS | OK |
| gemma-2-2b_Q4_0 | 4 | row | 0 | 1024 | 1024 | 0 | 4615.20 | 2.15x | 42.44 | 1.65x | 6/10 | PASS | OK |

## PP delta (GGML_HEXAGON_OPFUSION=1 -> 0), per model / cores / ubatch (split=row)

If the percentage shrinks as ubatch grows, the change and ubatch are NOT additive.

| model | cores | ubatch | pp_tok | depth | PP GGML_HEXAGON_OPFUSION=1 | PP GGML_HEXAGON_OPFUSION=0 | delta |
|---|---|---|---|---|---|---|---|
| gemma-2-2b_Q4_0 | 1 | 1024 | 1024 | 0 | 2319.0 | 2143.1 | -7.6% |
| gemma-2-2b_Q4_0 | 4 | 1024 | 1024 | 0 | 5059.6 | 4615.2 | -8.8% |
| gemma-4-12b-it-qat-q4_0 | 1 | 1024 | 1024 | 0 | 574.5 | 528.9 | -7.9% |
| gemma-4-12b-it-qat-q4_0 | 4 | 1024 | 1024 | 0 | 1579.9 | 1455.3 | -7.9% |

## Cross-core scaling (GGML_HEXAGON_OPFUSION=1=Base vs GGML_HEXAGON_OPFUSION=0=Opt), per model (split=row)

Uses the largest ubatch tested for each model. "vs 1c" is that series's own
speedup over its 1-core result; PP gain is Opt vs Base at that core count.

### gemma-2-2b_Q4_0 (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 2319 | 1.00x | 2143 | 1.00x | -7.6% | 26.1 | 25.8 |
| 4 | 5060 | 2.18x | 4615 | 2.15x | -8.8% | 46.1 | 42.4 |

### gemma-4-12b-it-qat-q4_0 (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 574 | 1.00x | 529 | 1.00x | -7.9% | 7.5 | 7.5 |
| 4 | 1580 | 2.75x | 1455 | 2.75x | -7.9% | 15.2 | 14.7 |

## Quality gate

**4 configuration(s) FAILED quality checks — see table above.**

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-191941/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-191941/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-191941/logs/cookie_*.txt`
