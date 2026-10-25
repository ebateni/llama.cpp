# test-backend-ops results

- device: `1b531bdd`
- device binaries: `1ef46d769 (heads/origin/ebateni/fusions) built 2026-10-25T18:28:33-04:00`
- date: 2026-10-25T18:54:45-04:00

| op | cores | GGML_HEXAGON_OPFUSION | passed | total | FAIL lines | rc |
|---|---|---|---|---|---|---|
| MUL_MAT_VEC_FUSION | 1 | 8191 | 891 | 891 | 0 | 0 |
| MUL_MAT_VEC_FUSION | 1 | 1 | 891 | 891 | 0 | 0 |
| MUL_MAT_VEC_FUSION | 4 | 8191 | 891 | 891 | 0 | 0 |
| MUL_MAT_VEC_FUSION | 4 | 1 | 891 | 891 | 0 | 0 |
| MMID_GATE_UP_GLU | 1 | 8191 | 0 | 0 | 0 | 0 |
| MMID_GATE_UP_GLU | 1 | 1 | 0 | 0 | 0 | 0 |
| MMID_GATE_UP_GLU | 4 | 8191 | 0 | 0 | 0 | 0 |
| MMID_GATE_UP_GLU | 4 | 1 | 0 | 0 | 0 | 0 |
| RMS_NORM_MUL_ADD | 1 | 8191 | 37 | 37 | 0 | 0 |
| RMS_NORM_MUL_ADD | 1 | 1 | 37 | 37 | 0 | 0 |
| RMS_NORM_MUL_ADD | 4 | 8191 | 37 | 37 | 0 | 0 |
| RMS_NORM_MUL_ADD | 4 | 1 | 37 | 37 | 0 | 0 |

Full outputs: `/local/mnt2/workspace2/llama.cpp.ebateni/moe-split/logs/20261025-182701/logs/testops_*.log`
