# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `c3da2bb33` (`ebateni/fa-team-zigzag`)
- device binaries: `c3da2bb33 (ebateni/fa-team-zigzag) built 2026-10-22T00:32:48-04:00`
- date: 2026-10-22T02:15:04-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_FA_SPLIT` = 1,2

## Results

| model | cores | split | GGML_HEXAGON_FA_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 1 | 1024 | 1024 | 0 | 857.39 | NA | 25.05 | NA | 0/10 | PASS | ERR |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 2 | 1024 | 1024 | 0 | 1084.33 | NA | 25.87 | NA | 2/10 | PASS | OK |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 1 | 512* | 1024 | 16384 | 709.36 | NA | 21.56 | NA | 2/10 | PASS | ERR |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 2 | 512* | 1024 | 16384 | 686.56 | NA | 22.26 | NA | 1/10 | PASS | ERR |

## PP delta (GGML_HEXAGON_FA_SPLIT=1 -> 2), per model / cores / ubatch (split=row)

If the percentage shrinks as ubatch grows, the change and ubatch are NOT additive.

| model | cores | ubatch | pp_tok | depth | PP GGML_HEXAGON_FA_SPLIT=1 | PP GGML_HEXAGON_FA_SPLIT=2 | delta |
|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | 512* | 1024 | 16384 | 709.4 | 686.6 | -3.2% |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | 1024 | 1024 | 0 | 857.4 | 1084.3 | +26.5% |

## Cross-core scaling (GGML_HEXAGON_FA_SPLIT=1=Base vs GGML_HEXAGON_FA_SPLIT=2=Opt), per model (split=row)

Uses the largest ubatch tested for each model. "vs 1c" is that series's own
speedup over its 1-core result; PP gain is Opt vs Base at that core count.

### Qwen3.5-35B-A3B-UD-Q4_K_M (ubatch=1024)

| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |
|---|---|---|---|---|---|---|---|
| 4 | 857 | NA | 1084 | NA | +26.5% | 25.1 | 25.9 |

## Quality gate

**3 configuration(s) FAILED quality checks — see table above.**

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261022-003123/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261022-003123/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261022-003123/logs/cookie_*.txt`
