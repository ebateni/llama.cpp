# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `48a0ed3f4` (`ebateni/moe-split`)
- device binaries: `48a0ed3f4 (ebateni/moe-split) built 2026-10-25T13:42:54-04:00`
- date: 2026-10-25T15:51:32-04:00
- reps: 5, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_MMID_SPLIT` = 0,1,3

## Results

| model | cores | split | GGML_HEXAGON_MMID_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | 0 | 1024 | 1024 | 0 | 442.21 | 1.00x | 16.73 | 1.00x | 6/10 | PASS | OK |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | 1 | 1024 | 1024 | 0 | 445.88 | 1.00x | 17.01 | 1.00x | 6/10 | PASS | OK |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 1 | single | 3 | 1024 | 1024 | 0 | 448.00 | 1.00x | 16.56 | 1.00x | 6/10 | PASS | OK |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 0 | 1024 | 1024 | 0 | 1053.17 | 2.38x | 28.69 | 1.71x | 8/10 | PASS | GARBAGE(<think>) |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 1 | 1024 | 1024 | 0 | 1068.13 | 2.40x | 27.97 | 1.64x | 7/10 | PASS | GARBAGE(<think>) |
| Qwen3.5-35B-A3B-UD-Q4_K_M | 4 | row | 3 | 1024 | 1024 | 0 | 1042.94 | 2.33x | 28.88 | 1.74x | 7/10 | PASS | GARBAGE(<think>) |
| gemma-4-26B_q4_0-it | 1 | single | 0 | 1024 | 1024 | 0 | 650.93 | 1.00x | 17.09 | 1.00x | 10/10 | PASS | OK |
| gemma-4-26B_q4_0-it | 1 | single | 1 | 1024 | 1024 | 0 | 645.92 | 1.00x | 17.56 | 1.00x | 10/10 | PASS | OK |
| gemma-4-26B_q4_0-it | 1 | single | 3 | 1024 | 1024 | 0 | 622.91 | 1.00x | 17.11 | 1.00x | 10/10 | PASS | OK |
| gemma-4-26B_q4_0-it | 4 | row | 0 | 1024 | 1024 | 0 | 1437.62 | 2.21x | 27.23 | 1.59x | 9/10 | PASS | OK |
| gemma-4-26B_q4_0-it | 4 | row | 1 | 1024 | 1024 | 0 | 1450.08 | 2.24x | 27.07 | 1.54x | 9/10 | PASS | OK |
| gemma-4-26B_q4_0-it | 4 | row | 3 | 1024 | 1024 | 0 | 1441.57 | 2.31x | 27.35 | 1.60x | 9/10 | PASS | OK |

## Quality gate

**3 configuration(s) FAILED quality checks — see table above.**

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/moe-split/logs/20261025-134128/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/moe-split/logs/20261025-134128/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/moe-split/logs/20261025-134128/logs/cookie_*.txt`
