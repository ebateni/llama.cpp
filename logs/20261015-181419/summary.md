# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- commit: `75360014b3` (`dev/ebateni/seq-parallel-prototype`)
- date: 2026-10-15T19:15:35-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)

## Results

| model | cores | split | ubatch | pp_tok | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | - | 1024 | 2048 | 428.02 | 1.20x | 14.97 | 1.09x | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | - | 512* | 2048 | 319.98 | 1.13x | 15.01 | 1.09x | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | - | 1024 | 2048 | 355.73 | 1.00x | 13.77 | 1.00x | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | - | 512* | 2048 | 283.71 | 1.00x | 13.75 | 1.00x | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | - | 1024 | 2048 | 985.70 | 2.77x | 24.11 | 1.75x | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | - | 512* | 2048 | 768.21 | 2.71x | 23.27 | 1.69x | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | - | 512* | 2048 | 705.16 | 2.49x | 19.82 | 1.44x | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | - | 512* | 2048 | 714.41 | 2.52x | 19.80 | 1.44x | - | - | - |

## Quality gate

_Quality checks were skipped for this run (--only-perf) — performance numbers alone do NOT establish correctness._

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-181419/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-181419/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-181419/logs/cookie_*.txt`
