# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- commit: `75360014b3` (`dev/ebateni/seq-parallel-prototype`)
- date: 2026-10-15T15:53:26-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)

## Results

| model | cores | split | ubatch | pp_tok | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 1024 | 1024 | 571.89 | 1.24x | 25.22 | 1.30x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 1024 | 1024 | 462.71 | 1.00x | 19.45 | 1.00x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 1024 | 1024 | 668.25 | 1.44x | 51.85 | 2.67x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 1024 | 1024 | 509.95 | 1.10x | 38.14 | 1.96x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 1024 | 1024 | 432.05 | 1.04x | 19.36 | 1.06x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 1024 | 1024 | 416.94 | 1.00x | 18.20 | 1.00x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 1024 | 1024 | 520.70 | 1.25x | 38.03 | 2.09x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 1024 | 1024 | 512.84 | 1.23x | 36.89 | 2.03x | - | - | - |
| gemma-4-E4B_q4_0-it | 1 | single | - | 1024 | 1024 | 449.52 | 0.96x | 18.04 | 1.06x | - | - | - |
| gemma-4-E4B_q4_0-it | 1 | single | - | 1024 | 1024 | 466.99 | 1.00x | 17.09 | 1.00x | - | - | - |
| gemma-4-E4B_q4_0-it | 4 | row | - | 1024 | 1024 | 559.44 | 1.20x | 34.27 | 2.01x | - | - | - |
| gemma-4-E4B_q4_0-it | 4 | row | - | 1024 | 1024 | 517.67 | 1.11x | 31.09 | 1.82x | - | - | - |

## Quality gate

_Quality checks were skipped for this run (--only-perf) — performance numbers alone do NOT establish correctness._

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-153217/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-153217/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-153217/logs/cookie_*.txt`
