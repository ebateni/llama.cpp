# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- commit: `0f55e7b84` (`ebateni/adaptive-split`)
- date: 2026-10-15T14:55:02-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)

## Results

| model | cores | split | ubatch | pp_tok | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 1024 | 1024 | 1788.96 | 1.58x | 25.18 | 1.29x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 1024 | 1024 | 1133.21 | 1.00x | 19.46 | 1.00x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 1024 | 1024 | 5475.07 | 4.83x | 51.64 | 2.65x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 1024 | 1024 | 3694.15 | 3.26x | 38.50 | 1.98x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 1024 | 1024 | 1110.59 | 1.05x | 19.37 | 1.06x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 1024 | 1024 | 1054.81 | 1.00x | 18.19 | 1.00x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 1024 | 1024 | 3130.64 | 2.97x | 38.17 | 2.10x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 1024 | 1024 | 3005.11 | 2.85x | 36.89 | 2.03x | - | - | - |
| gemma-4-12b-it-qat-q4_0 | 1 | single | - | 1024 | 1024 | 588.16 | 1.11x | 7.55 | 1.04x | - | - | - |
| gemma-4-12b-it-qat-q4_0 | 1 | single | - | 1024 | 1024 | 532.00 | 1.00x | 7.29 | 1.00x | - | - | - |
| gemma-4-12b-it-qat-q4_0 | 4 | row | - | 1024 | 1024 | 1818.10 | 3.42x | 17.59 | 2.41x | - | - | - |
| gemma-4-12b-it-qat-q4_0 | 4 | row | - | 1024 | 1024 | 1651.76 | 3.10x | 16.54 | 2.27x | - | - | - |

## Quality gate

_Quality checks were skipped for this run (--only-perf) — performance numbers alone do NOT establish correctness._

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-144105/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-144105/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-144105/logs/cookie_*.txt`
