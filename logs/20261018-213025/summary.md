# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `630b24e4d` (`ebateni/adaptive-split`)
- device binaries: `96ed45522 (ebateni/adaptive-split) built 2026-10-18T14:27:49-04:00`
- date: 2026-10-18T23:48:08-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_FA_SPLIT` = 0,1,2

## Results

| model | cores | split | GGML_HEXAGON_FA_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | 0 | 1024 | 2048 | 0 | 427.61 | 1.00x | 15.72 | 1.00x | 0/10 | PASS | ERR |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | 1 | 1024 | 2048 | 0 | 428.18 | 1.00x | 16.17 | 1.00x | 0/10 | PASS | ERR |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | 2 | 1024 | 2048 | 0 | 434.52 | 1.00x | 15.73 | 1.00x | 0/10 | PASS | ERR |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 0 | 1024 | 2048 | 0 | 963.60 | 2.25x | 25.87 | 1.65x | 1/10 | PASS | ERR |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 1 | 1024 | 2048 | 0 | 983.69 | 2.30x | 25.46 | 1.57x | 0/10 | PASS | ERR |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 2 | 1024 | 2048 | 0 | 962.09 | 2.21x | 25.93 | 1.65x | 0/10 | PASS | ERR |

## Quality gate

**6 configuration(s) FAILED quality checks — see table above.**

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-213025/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-213025/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-213025/logs/cookie_*.txt`
