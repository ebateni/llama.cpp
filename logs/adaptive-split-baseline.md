# Adaptive split — P0 baseline (row/mdev, upstream 620d2225c)

Runs: `logs/20261015-152029` (GGML_HEXAGON_PROFILE=1) and `logs/20261015-153217`
(PROFILE=3, FENCE/L2FLUSH traces). 1 vs 4 NSP, row split, pp1024, ub1024, depth 0 / 4096.
Analyzer: `scripts/snapdragon/mdev-profile-analyze.py <1c.log> <4c.log>` (last graph of each log).

Level-3 tracing does not perturb DSP time (llama 4c graph 183.9 ms vs 181.5 ms at level 1);
it only slows the host (bench t/s drops ~8x from printing).

## Graph time and where the 4c time goes

Per-op time on each NSP = flush (L2FLUSH before the barrier) + barrier (min FENCE over NSPs)
+ imbalance (FENCE wait above the minimum, charged to the op that caused it) + exec.

| model | depth | 1c ms | 4c ms | x | ideal 1c/4 | lost | exec | flush | barrier | imbalance |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b | 0    | 563 | 184 | 3.06 | 141 | 43  | 96% | 2% | 1% | 1% |
| llama-3.2-3b | 4096 | 897 | 273 | 3.29 | 224 | 49  | 97% | 2% | 0% | 1% |
| Qwen3.5-4B   | 0    | 892 | 324 | 2.76 | 223 | 101 | 95% | 2% | 1% | 2% |
| Qwen3.5-4B   | 4096 | 967 | 335 | 2.89 | 242 | 93  | 96% | 2% | 1% | 1% |
| Gemma-4-E4B  | 0    | 807 | 335 | 2.41 | 202 | 133 | 80% | 3% | 1% | 16% |
| Gemma-4-E4B  | 4096 | 863 | 357 | 2.42 | 216 | 141 | 81% | 3% | 1% | 15% |

**A barrier costs ~2.5 us per op (~1% of the graph). Flush is 2-3%.** Eliminating every
barrier in the graph (plan P1) is worth at most ~1-2% plus whatever imbalance it hides.
The loss is in *exec*, not in sync — with one exception (Gemma FA, below).

## Finding 1 — Gemma FA runs SOLO on NSP0 (biggest single item)

Gemma-4-E4B: 8 Q heads, 2 KV heads (G=4), head_dim 256/512. `n_kv % 4 != 0` -> Q-block
fallback (`flash-attn-ops.c:1919`). The host picks `Br` for a single core
(`hmx_fa_find_chunk_size`, `ggml-hexagon.cpp:5237`), so 1024 tokens give fewer than 4 Q-blocks
and `htp_tensor_mdev_partition` puts the whole op on NSP0. NSP1-3 finish in ~27 us and wait.

| depth | FA 1c ms | FA 4c ms | x | exec max/mean | ops SOLO |
|---|---|---|---|---|---|
| 0    | 68.0  | 68.1 | 1.00 | 3.90 | 42/42 |
| 4096 | 128.3 | 84.3 | 1.52 | 2.53 | 35/42 |

Cost: ~51 ms of 335 ms (15%). Fixing it alone takes Gemma 4c from 2.41x to ~2.8x.
The same fallback hits Qwen3.5/Next attention layers (2 KV heads) whenever Br is large.

## Finding 2 — elementwise ops are DDR-bound and scale ~2x

All four NSPs stream DDR together; the aggregate tops out at ~150-185 GB/s (one NSP alone
gets ~80-95 GB/s). Every op that only moves bytes is capped near 1.6-2.3x:

| op (llama d0) | 1c ms | 4c ms | x | 4c GB/s |
|---|---|---|---|---|
| SWIGLU (8192x1024 f32, 2 in + 1 out) | 35.0 | 15.2 | 2.30 | ~185 |
| ADD residual (3072x1024, 2 in + 1 out) | 11.4 | 6.5 | 1.76 | ~163 |
| RMS_NORM+MUL | 10.5 | 4.9 | 2.14 | ~150 |
| ROPE Q | 11.6 | 5.8 | 1.98 | |

Loss vs ideal per class (ms, d0), top items:

| llama-3.2-3b | ms | Qwen3.5-4B | ms | Gemma-4-E4B | ms |
|---|---|---|---|---|---|
| SWIGLU | 6.5 | SWIGLU | 11.6 | **FA (SOLO)** | **51.1** |
| QKV NX matmul | 4.9 | SSM_CONV (1.59x) | 7.2 | GEGLU | 12.5 |
| ffn gate/up NX | 4.4 | ffn gate/up NX | 6.5 | attn_out/PLE matmuls | 9.6 |
| ADD l_out + ffn_inp | 7.5 | CONCAT conv_input | 6.3 | ffn gate/up NX | 7.2 |
| FA | 3.5 | MUL (gate) | 4.7 | 3x ADD (residual, PLE) | 12.0 |
| attn_out | 3.0 | 2x ADD residual | 8.4 | 6x RMS_NORM+MUL | 18.4 |
| ROPE Q+K | 3.5 | SILU x2 | 6.6 | ROPE Q | 3.4 |
| 2x RMS_NORM+MUL | 4.5 | linear_attn_out | 3.3 | MUL l_out (layer scale) | 3.2 |

No axis choice fixes a DDR-bound op: TOKEN, FEAT, or REPL all move the same or more bytes.
Only moving fewer bytes does — fusion.

## Finding 3 — matmul axis: TOKEN is right for prefill

Large FFN matmuls scale 3.4-3.6x (HMX-bound). Smaller ones (QKV, attn_out, 3072x3072-class)
scale ~3.0x. With ub=1024 the f32 activations are larger than the Q4_0 weights
(attn_out: 12.6 MB activations vs 5.3 MB weights), so FEAT (every NSP reads all activations)
moves *more* DDR bytes than TOKEN (every NSP reads all weights). FEAT only wins when
weight bytes > activation bytes, i.e. small ubatches — that is decode (Phase 2), not prefill.

## Consequences for the plan

| plan item | data says | recommendation |
|---|---|---|
| P3 FA fallback (QHEAD / QSTRIPE) | 15% on Gemma, also Qwen3.5 attention | **do first**; hardcode QHEAD when `n_head % N == 0`, else mdev-aware Br + zig-zag Q-blocks |
| P1 barrier elision | <= 1-2% | keep, but after the fusion work |
| P2 REPL for tiny ops | only 1-2 SOLO ops seen (SCALE) | low |
| P4 FEAT matmul axis | loses at ub>=256 | drop for prefill; revisit in decode phase |
| P5 MoE LPT | no MoE model measured yet | needs a MoE run first |
| **new: GLU epilogue fusion** (SWIGLU/GEGLU into the gate/up MUL_MAT_NX) | 6.5 / 11.6 / 12.5 ms + the gate/up writes | **second**; removes ~100 MB DDR per layer |
| **new: residual ADD + RMS_NORM(+MUL) fusion; post-norm + ADD (Gemma)** | 7-20 ms per model | third |
| **new: CONCAT into SSM_CONV (Qwen3.5)** | 6.3 ms + SSM_CONV 1.59x | with the GDN work |

All of it is Hexagon-side (host fusion matcher + DSP kernels); no Meta or `src/` change.
