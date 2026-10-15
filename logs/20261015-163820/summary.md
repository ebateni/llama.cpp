# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- commit: `75360014b3` (`dev/ebateni/seq-parallel-prototype`)
- date: 2026-10-15T16:48:24-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)

## Results

| model | cores | split | ubatch | pp_tok | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 1024 | 2048 | 1706.73 | 1.00x | 25.22 | 1.00x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 2048 | 2048 | 1630.67 | 1.00x | 25.23 | 1.00x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 1024 | 2048 | 4923.63 | 2.88x | 51.77 | 2.05x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 2048 | 2048 | 5021.83 | 3.08x | 51.68 | 2.05x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 1024 | 2048 | 1124.30 | 1.00x | 19.36 | 1.00x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 2048 | 2048 | 1146.14 | 1.00x | 19.35 | 1.00x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 1024 | 2048 | 2980.14 | 2.65x | 37.96 | 1.96x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 2048 | 2048 | 3127.20 | 2.73x | 37.94 | 1.96x | - | - | - |
| gemma-4-E4B_q4_0-it | 1 | single | - | 1024 | 2048 | 1223.96 | 1.00x | 18.04 | 1.00x | - | - | - |
| gemma-4-E4B_q4_0-it | 1 | single | - | 2048 | 2048 | 1218.85 | 1.00x | 18.03 | 1.00x | - | - | - |
| gemma-4-E4B_q4_0-it | 4 | row | - | 1024 | 2048 | 2717.17 | 2.22x | 34.37 | 1.91x | - | - | - |
| gemma-4-E4B_q4_0-it | 4 | row | - | 2048 | 2048 | 3241.93 | 2.66x | 34.35 | 1.91x | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M.gguf | 1 | single | - | 512* | 2048 | NA | NA | NA | NA | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M.gguf | 1 | single | - | 512* | 2048 | NA | NA | NA | NA | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M.gguf | 4 | row | - | 512* | 2048 | NA | NA | NA | NA | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M.gguf | 4 | row | - | 512* | 2048 | NA | NA | NA | NA | - | - | - |

## Quality gate

_Quality checks were skipped for this run (--only-perf) — performance numbers alone do NOT establish correctness._

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-163820/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-163820/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-163820/logs/cookie_*.txt`
