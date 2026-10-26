# test-backend-ops results

- device: `1b531bdd`
- device binaries: `542f3d387 (ebateni/fusions) built 2026-10-25T20:00:34-04:00`
- date: 2026-10-25T20:29:34-04:00

| op | cores | GGML_HEXAGON_OPFUSION | passed | total | FAIL lines | rc |
|---|---|---|---|---|---|---|
| MUL_MAT_VEC_FUSION | 1 | 8191 | 915 | 915 | 0 | 0 |
| MUL_MAT_VEC_FUSION | 1 | 1 | 915 | 915 | 0 | 0 |
| MUL_MAT_VEC_FUSION | 4 | 8191 | 915 | 915 | 0 | 0 |
| MUL_MAT_VEC_FUSION | 4 | 1 | 915 | 915 | 0 | 0 |
| MMID_GATE_UP_GLU | 1 | 8191 | 16 | 16 | 0 | 0 |
| MMID_GATE_UP_GLU | 1 | 1 | 16 | 16 | 0 | 0 |
| MMID_GATE_UP_GLU | 4 | 8191 | 16 | 16 | 0 | 0 |
| MMID_GATE_UP_GLU | 4 | 1 | 16 | 16 | 0 | 0 |
| RMS_NORM_MUL_ADD | 1 | 8191 | 39 | 39 | 0 | 0 |
| RMS_NORM_MUL_ADD | 1 | 1 | 39 | 39 | 0 | 0 |
| RMS_NORM_MUL_ADD | 4 | 8191 | 39 | 39 | 0 | 0 |
| RMS_NORM_MUL_ADD | 4 | 1 | 39 | 39 | 0 | 0 |

Full outputs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261025-195904/logs/testops_*.log`
