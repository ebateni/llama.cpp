# test-backend-ops results

- device: `1b531bdd`
- device binaries: `d3f87a011 (ebateni/adaptive-split) built 2026-10-15T20:03:51-04:00`
- date: 2026-10-15T21:51:30-04:00

| op | cores | GGML_HEXAGON_FA_SPLIT | passed | total | FAIL lines | rc |
|---|---|---|---|---|---|---|
| FLASH_ATTN_EXT | 1 | 1 | 2590 | 2640 | 52 | 1 |
| FLASH_ATTN_EXT | 1 | 0 | 2586 | 2640 | 56 | 1 |
| FLASH_ATTN_EXT | 2 | 1 | 2594 | 2640 | 48 | 1 |
| FLASH_ATTN_EXT | 2 | 0 | 2577 | 2640 | 65 | 1 |
| FLASH_ATTN_EXT | 4 | 1 | 2582 | 2640 | 60 | 1 |
| FLASH_ATTN_EXT | 4 | 0 | 2584 | 2640 | 58 | 1 |

Full outputs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261015-200226/logs/testops_*.log`
