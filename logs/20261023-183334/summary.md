# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `08131f369` (`ebateni/moe-split`)
- device binaries: `unknown (package built before BUILD_INFO stamping)`
- date: 2026-10-23T19:58:42-04:00
- reps: 5, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_MMID_SPLIT` = 0,1

## Results

| model | cores | split | GGML_HEXAGON_MMID_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | 0 | 1024 | 1024 | 0 | 458.54 | 1.00x | 16.54 | 1.00x | 6/10 | PASS | OK |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | 1 | 1024 | 1024 | 0 | 448.52 | 1.00x | 16.54 | 1.00x | 6/10 | PASS | OK |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 0 | 1024 | 1024 | 0 | 1146.22 | 2.50x | 27.96 | 1.69x | 7/10 | PASS | OK |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 1 | 1024 | 1024 | 0 | 1161.48 | 2.59x | 27.97 | 1.69x | 10/10 | PASS | GARBAGE(<think>) |
| gemma-4-26B_q4_0-it | 1 | single | 0 | 1024 | 1024 | 0 | 635.08 | 1.00x | 16.95 | 1.00x | 10/10 | PASS | OK |
| gemma-4-26B_q4_0-it | 1 | single | 1 | 1024 | 1024 | 0 | 626.32 | 1.00x | 16.96 | 1.00x | 10/10 | PASS | OK |
| gemma-4-26B_q4_0-it | 4 | row | 0 | 1024 | 1024 | 0 | 1521.91 | 2.40x | 26.53 | 1.57x | 9/10 | PASS | OK |
| gemma-4-26B_q4_0-it | 4 | row | 1 | 1024 | 1024 | 0 | 1598.10 | 2.55x | 26.63 | 1.57x | 9/10 | PASS | OK |

## PP delta (GGML_HEXAGON_MMID_SPLIT=0 -> 1), per model / cores / ubatch (split=row)

If the percentage shrinks as ubatch grows, the change and ubatch are NOT additive.

| model | cores | ubatch | pp_tok | depth | PP GGML_HEXAGON_MMID_SPLIT=0 | PP GGML_HEXAGON_MMID_SPLIT=1 | delta |
|---|---|---|---|---|---|---|---|
| gemma-4-26B_q4_0-it | 1 | 1024 | 1024 | 0 | 635.1 | 626.3 | -1.4% |
| gemma-4-26B_q4_0-it | 4 | 1024 | 1024 | 0 | 1521.9 | 1598.1 | +5.0% |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | 1024 | 1024 | 0 | 458.5 | 448.5 | -2.2% |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | 1024 | 1024 | 0 | 1146.2 | 1161.5 | +1.3% |

## Cross-core scaling (GGML_HEXAGON_MMID_SPLIT=0=Base vs GGML_HEXAGON_MMID_SPLIT=1=Opt), per model (split=row)

Uses the largest ubatch tested for each model. "vs 1c" is that series's own
speedup over its 1-core result; PP gain is Opt vs Base at that core count.

### gemma-4-26B_q4_0-it (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 635 | 1.00x | 626 | 1.00x | -1.4% | 16.9 | 17.0 |
| 4 | 1522 | 2.40x | 1598 | 2.55x | +5.0% | 26.5 | 26.6 |

### Qwen3.5-35B-A3B-UD-Q4_K_M (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 1 | 459 | 1.00x | 449 | 1.00x | -2.2% | 16.5 | 16.5 |
| 4 | 1146 | 2.50x | 1161 | 2.59x | +1.3% | 28.0 | 28.0 |

## Quality gate

**1 configuration(s) FAILED quality checks — see table above.**

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/moe-split/logs/20261023-183334/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/moe-split/logs/20261023-183334/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/moe-split/logs/20261023-183334/logs/cookie_*.txt`
