# test-backend-ops results

- device: `1b531bdd`
- device binaries: `96ed45522 (ebateni/adaptive-split) built 2026-10-18T14:27:49-04:00`
- date: 2026-10-18T17:09:20-04:00

| op | cores | GGML_HEXAGON_FA_SPLIT | passed | total | FAIL lines | rc |
|---|---|---|---|---|---|---|
| FLASH_ATTN_EXT | 1 | 0 | 2595 | 2640 | 47 | 1 |
| FLASH_ATTN_EXT | 1 | 1 | 2580 | 2640 | 62 | 1 |
| FLASH_ATTN_EXT | 1 | 2 | 2577 | 2640 | 65 | 1 |
| FLASH_ATTN_EXT | 2 | 0 | 2581 | 2640 | 61 | 1 |
| FLASH_ATTN_EXT | 2 | 1 | 2590 | 2640 | 52 | 1 |
| FLASH_ATTN_EXT | 2 | 2 | 2596 | 2640 | 46 | 1 |
| FLASH_ATTN_EXT | 4 | 0 | 2591 | 2640 | 51 | 1 |
| FLASH_ATTN_EXT | 4 | 1 | 2596 | 2640 | 46 | 1 |
| FLASH_ATTN_EXT | 4 | 2 | 2588 | 2640 | 54 | 1 |

Full outputs: `/local/mnt2/workspace2/llama.cpp.ebateni/llama.cpp/logs/20261018-142527/logs/testops_*.log`
