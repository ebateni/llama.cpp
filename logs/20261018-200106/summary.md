# llama.cpp / ggml-hexagon build-and-test results

- device: `1b531bdd`
- repo HEAD: `dc4c260b3` (`ebateni/adaptive-split`)
- device binaries: `96ed45522 (ebateni/adaptive-split) built 2026-10-18T14:27:49-04:00`
- date: 2026-10-18T20:35:36-04:00
- reps: 3, ctx: 4096, pp-depth: 0, tg-depth: 1024
- split mode(s): row (1-core rows always run as "single" — nothing to split)
- A/B variable: `GGML_HEXAGON_FA_SPLIT` = 0,1,2

## Results

| model | cores | split | GGML_HEXAGON_FA_SPLIT | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-12b-it-qat-q4_0 | 1 | single | 0 | 1024 | 1024 | 0 | 583.82 | 1.00x | 7.53 | 1.00x | 2/10 | PASS | GARBAGE(phrase) |
| gemma-4-12b-it-qat-q4_0 | 1 | single | 1 | 1024 | 1024 | 0 | 588.06 | 1.00x | 7.53 | 1.00x | 2/10 | PASS | GARBAGE(phrase) |
| gemma-4-12b-it-qat-q4_0 | 1 | single | 2 | 1024 | 1024 | 0 | 584.11 | 1.00x | 7.36 | 1.00x | 2/10 | PASS | GARBAGE(phrase) |
| gemma-4-12b-it-qat-q4_0 | 4 | row | 0 | 1024 | 1024 | 0 | 1762.33 | 3.02x | 17.22 | 2.29x | 2/10 | PASS | GARBAGE(phrase) |
| gemma-4-12b-it-qat-q4_0 | 4 | row | 1 | 1024 | 1024 | 0 | 1777.09 | 3.02x | 17.22 | 2.29x | 2/10 | PASS | GARBAGE(phrase) |
| gemma-4-12b-it-qat-q4_0 | 4 | row | 2 | 1024 | 1024 | 0 | 1758.26 | 3.01x | 17.20 | 2.34x | 2/10 | PASS | GARBAGE(phrase) |

## Quality gate

**6 configuration(s) FAILED quality checks — see table above.**

Raw logs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-200106/logs`  ·  iq transcripts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-200106/logs/iq_*.txt`  ·  cookie texts: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-200106/logs/cookie_*.txt`
