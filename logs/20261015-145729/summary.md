# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- commit: `dc97cfd1a` (`ebateni/adaptive-split`)
- date: 2026-10-15T15:03:44-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)

## Results

| model | cores | split | ubatch | pp_tok | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 1024 | 1024 | 1798.76 | 1.59x | 25.22 | 1.30x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 1024 | 1024 | 1130.37 | 1.00x | 19.44 | 1.00x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 1024 | 1024 | 5249.87 | 4.64x | 51.50 | 2.65x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 1024 | 1024 | 3586.92 | 3.17x | 38.39 | 1.97x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 1024 | 1024 | 1134.77 | 1.08x | 19.36 | 1.06x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 1024 | 1024 | 1050.84 | 1.00x | 18.20 | 1.00x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 1024 | 1024 | 3002.20 | 2.86x | 38.08 | 2.09x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 1024 | 1024 | 2874.29 | 2.74x | 36.88 | 2.03x | - | - | - |
| gemma-4-E4B_q4_0 | 1 | single | - | 512* | 1024 | NA | NA | NA | NA | - | - | - |
| gemma-4-E4B_q4_0 | 1 | single | - | 512* | 1024 | NA | NA | NA | NA | - | - | - |
| gemma-4-E4B_q4_0 | 4 | row | - | 512* | 1024 | NA | NA | NA | NA | - | - | - |
| gemma-4-E4B_q4_0 | 4 | row | - | 512* | 1024 | NA | NA | NA | NA | - | - | - |

## Quality gate

_Quality checks were skipped for this run (--only-perf) — performance numbers alone do NOT establish correctness._

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-145729/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-145729/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-145729/logs/cookie_*.txt`
