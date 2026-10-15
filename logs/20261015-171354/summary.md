# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- commit: `75360014b3` (`dev/ebateni/seq-parallel-prototype`)
- date: 2026-10-15T17:42:55-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)

## Results

| model | cores | split | ubatch | pp_tok | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 1024 | 2048 | 1714.13 | 3.60x | 23.48 | 2.11x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 2048 | 2048 | 1632.13 | 3.30x | 23.50 | 2.13x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 1024 | 2048 | 475.50 | 1.00x | 11.15 | 1.00x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 1 | single | - | 2048 | 2048 | 494.77 | 1.00x | 11.05 | 1.00x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 1024 | 2048 | 5121.46 | 10.77x | 47.66 | 4.27x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 2048 | 2048 | 5187.98 | 10.49x | 47.69 | 4.32x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 1024 | 2048 | 1723.86 | 3.63x | 21.43 | 1.92x | - | - | - |
| llama-3.2-3b-instruct-q4_0 | 4 | row | - | 2048 | 2048 | 1816.62 | 3.67x | 21.16 | 1.91x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 1024 | 2048 | 1129.36 | 1.33x | 19.00 | 1.22x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 2048 | 2048 | 1147.73 | 1.33x | 18.99 | 1.21x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 1024 | 2048 | 851.19 | 1.00x | 15.63 | 1.00x | - | - | - |
| Qwen3.5-4B-Q4 | 1 | single | - | 2048 | 2048 | 862.72 | 1.00x | 15.64 | 1.00x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 1024 | 2048 | 3131.13 | 3.68x | 37.64 | 2.41x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 2048 | 2048 | 3155.75 | 3.66x | 37.61 | 2.40x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 1024 | 2048 | 2561.49 | 3.01x | 33.54 | 2.15x | - | - | - |
| Qwen3.5-4B-Q4 | 4 | row | - | 2048 | 2048 | 2576.54 | 2.99x | 33.49 | 2.14x | - | - | - |
| gemma-4-E4B_q4_0-it | 1 | single | - | 1024 | 2048 | 1230.63 | 1.22x | 17.67 | 1.18x | - | - | - |
| gemma-4-E4B_q4_0-it | 1 | single | - | 2048 | 2048 | 1222.57 | 1.24x | 17.27 | 1.18x | - | - | - |
| gemma-4-E4B_q4_0-it | 1 | single | - | 1024 | 2048 | 1011.10 | 1.00x | 14.92 | 1.00x | - | - | - |
| gemma-4-E4B_q4_0-it | 1 | single | - | 2048 | 2048 | 986.94 | 1.00x | 14.67 | 1.00x | - | - | - |
| gemma-4-E4B_q4_0-it | 4 | row | - | 1024 | 2048 | 2842.22 | 2.81x | 33.09 | 2.22x | - | - | - |
| gemma-4-E4B_q4_0-it | 4 | row | - | 2048 | 2048 | 3365.69 | 3.41x | 31.74 | 2.16x | - | - | - |
| gemma-4-E4B_q4_0-it | 4 | row | - | 1024 | 2048 | 2505.40 | 2.48x | 24.55 | 1.65x | - | - | - |
| gemma-4-E4B_q4_0-it | 4 | row | - | 2048 | 2048 | 2835.57 | 2.87x | 23.88 | 1.63x | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M.gguf | 1 | single | - | 512* | 2048 | NA | NA | NA | NA | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M.gguf | 1 | single | - | 512* | 2048 | NA | NA | NA | NA | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M.gguf | 1 | single | - | 512* | 2048 | NA | NA | NA | NA | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M.gguf | 1 | single | - | 512* | 2048 | NA | NA | NA | NA | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M.gguf | 4 | row | - | 512* | 2048 | NA | NA | NA | NA | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M.gguf | 4 | row | - | 512* | 2048 | NA | NA | NA | NA | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M.gguf | 4 | row | - | 512* | 2048 | NA | NA | NA | NA | - | - | - |
| Qwen3.5-35B-A3B-UD-Q4_K_M.gguf | 4 | row | - | 512* | 2048 | NA | NA | NA | NA | - | - | - |

## Quality gate

_Quality checks were skipped for this run (--only-perf) — performance numbers alone do NOT establish correctness._

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-171354/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-171354/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-171354/logs/cookie_*.txt`
