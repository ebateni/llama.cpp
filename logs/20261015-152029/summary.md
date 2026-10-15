# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- commit: `75360014b3` (`dev/ebateni/seq-parallel-prototype`)
- date: 2026-10-15T15:31:02-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)

## Results

| model | cores | split | ubatch | pp_tok | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 1024 | 1024 | 1781.14 | 1.59x | 24.35 | 1.29x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 1024 | 1024 | 1121.93 | 1.00x | 18.90 | 1.00x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 1024 | 1024 | 5074.22 | 4.52x | 50.23 | 2.66x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 1024 | 1024 | 3475.63 | 3.10x | 37.42 | 1.98x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 1024 | 1024 | 1107.35 | 1.08x | 18.65 | 1.07x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 1024 | 1024 | 1028.46 | 1.00x | 17.43 | 1.00x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 1024 | 1024 | 2929.15 | 2.85x | 36.69 | 2.10x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 1024 | 1024 | 2838.01 | 2.76x | 36.84 | 2.11x | - | - | - |
| gemma-4-E4B_q4_0-it | 1 | single | - | 1024 | 1024 | 1237.88 | 1.07x | 18.03 | 1.06x | - | - | - |
| gemma-4-E4B_q4_0-it | 1 | single | - | 1024 | 1024 | 1153.99 | 1.00x | 17.09 | 1.00x | - | - | - |
| gemma-4-E4B_q4_0-it | 4 | row | - | 1024 | 1024 | 2791.88 | 2.42x | 34.10 | 2.00x | - | - | - |
| gemma-4-E4B_q4_0-it | 4 | row | - | 1024 | 1024 | 2612.11 | 2.26x | 30.84 | 1.80x | - | - | - |

## Quality gate

_Quality checks were skipped for this run (--only-perf) — performance numbers alone do NOT establish correctness._

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-152029/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-152029/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-152029/logs/cookie_*.txt`
