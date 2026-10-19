# hexagon: adaptive KV-head team split for flash attention on multi-NSP

Branch `ebateni/adaptive-split`, commit `7d6e771b7` on upstream master `31288c9c6`. Knob:
`GGML_HEXAGON_FA_SPLIT` (1 = team split, the default; 0 = previous partitioning).

## Problem

In multi-NSP mode (`GGML_HEXAGON_DEVICES=HTP0[0-3]`), HMX flash attention splits by KV heads only
when `n_kv_heads % N == 0`. Otherwise it falls back to splitting contiguous Q-blocks, and every
NSP processes all KV heads. The (Br, Bc) solver sizes Br for a single core, so it produces few,
large Q-blocks. With fewer Q-blocks than NSPs, the op runs on NSP0 while the other NSPs wait at
the barrier. Two cases hit this:

- **Prefill** with few KV heads: Gemma-4 (2), Qwen3.5 attention layers (2). For Gemma-4-E4B at
  ubatch 1024, the single-NSP FA was about 15% of the 4-core graph.
- **Decode** on the HMX path (DK > 256, or DK = 256 with fewer KV heads than threads, e.g.
  Gemma-4): there is a single Q-block, so FA runs on one NSP.

## Solution

- **Teams.** The NSPs form `gcd(n_kv_heads, N)` teams of `N / teams` NSPs each. Each team owns a
  contiguous group of KV heads, so an NSP only reads the K/V of its own heads. The team's Q-blocks
  are split evenly over its members, in contiguous ranges.
  - Gemma (2 KV heads, 4 NSPs) gets 2 teams × 2 NSPs.
  - When `n_kv % N == 0` this reduces exactly to upstream's KV-head split, so nothing changes.
- **Per-NSP Br cost.** `hmx_fa_find_chunk_size` takes `q_split`, the number of NSPs sharing the
  Q-blocks, and costs `ceil(q_blocks / q_split)`. That is the critical path of one NSP rather
  than of the whole op, so Br no longer collapses the work onto one NSP.
- **Scope.** The changes are on the Hexagon side only: the shared helper `htp_fa_mdev_teams()` in
  `htp/flash-attn-ops.h`, the HMX FA partition in `htp/flash-attn-ops.c`, and the FA precompute in
  `ggml-hexagon.cpp`. The FA kernel is unchanged; only the head and Q ranges per NSP differ. The
  HVX FA path is untouched.
- **Load balance.** The kernel skips fully masked KV blocks, so under a causal mask later Q-blocks
  cost more than earlier ones. With a shallow KV cache, the contiguous split within a team is
  therefore not perfectly balanced. Assigning Q-blocks zig-zag within the team would fix this.
- **Tests.** New test-backend-ops FA cases cover:
  - n_kv = 1, 2, 6;
  - GQA factor 4 and 8;
  - head_dim 128, 256, 512.

## Results

Run `20261015-223555`: device `1b531bdd`, 4 NSPs, row split, pp2048, ubatch 1024, depth 0, 3 reps.
Both settings were measured in the same binary. A repeat run (`20261018-171424`) gave Gemma 4c
prefill +14.1%. These numbers were measured on an upstream base without masked KV-block skipping.
Re-measurement on the current base is pending.

`ab=0` is upstream (`GGML_HEXAGON_FA_SPLIT=0`) and `ab=1` is the team split. The 4c/1c columns
divide each setting's 4c result by its own 1c result. "4c gain" is 4c `ab=1` over 4c `ab=0`.

**Prefill (t/s)**

| Model | 1c ab=0 | 1c ab=1 | 4c ab=0 | 4c ab=1 | 4c/1c ab=0 | 4c/1c ab=1 | 4c gain |
|---|---|---|---|---|---|---|---|
| Gemma-4-E4B Q4_0 | 1231.34 | 1226.07 | 2857.01 | **3257.67** | 2.32x | **2.66x** | **+14.0%** |
| Qwen3.5-4B Q4 | 1104.90 | 1103.88 | 3094.25 | **3155.15** | 2.80x | **2.86x** | **+2.0%** |

**Decode (t/s)**

| Model | 1c ab=0 | 1c ab=1 | 4c ab=0 | 4c ab=1 | 4c/1c ab=0 | 4c/1c ab=1 | 4c gain |
|---|---|---|---|---|---|---|---|
| Gemma-4-E4B Q4_0 | 17.67 | 17.05 | 32.93 | **35.31** | 1.86x | **2.07x** | **+7.2%** |
| Qwen3.5-4B Q4 | 18.23 | 18.24 | 35.99 | **37.62** | 1.97x | **2.06x** | **+4.5%** |

The gain on Qwen3.5 is smaller because only 1 in 4 of its layers is attention; the rest are
gated delta net.

## Notes

- IQ, needle and cookie are identical with the split off and on, for both models.
- 1-core results are unchanged (±0.4%).
- llama-3.2-3b (8 KV heads) is unchanged (±0.7%).
- Qwen3.5-35B-A3B (MoE, 2 KV heads): 4c prefill +2.1% (963.6 -> 983.7 t/s), within noise.
  Attention is a small share of the MoE graph.
