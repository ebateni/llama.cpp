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

## Update — ub 1024/2048, depth 0/16k (`logs/20261015-171354`, profile `logs/20261015-163820`)

pp2048, row, 4c vs 1c (x1c). Binary provenance unknown for these runs: the harness recorded
repo HEAD `75360014b` (seq-parallel branch) with `--no-build`; the harness now stamps `BUILD_INFO`.

| model | ub1024 d0 | ub2048 d0 | ub1024 d16k | ub2048 d16k |
|---|---|---|---|---|
| llama-3.2-3b | 2.99 | 3.18 | 3.63 | 3.67 |
| Qwen3.5-4B   | 2.77 | 2.75 | 3.01 | 2.99 |
| Gemma-4-E4B  | 2.31 | **2.75** | 2.48 | **2.87** |

Gemma gains +18% from ub1024 -> ub2048 because 2048 tokens give >= 4 Q-blocks and the FA
fallback stops running SOLO (FA 3.46x at ub2048). This confirms Finding 1, and that the fix must
not depend on ubatch.

Top losses at ub2048 d0 (profile level 1, so no flush/barrier split; graph = pp2048):

| llama 3.27x (lost 69 ms of 377) | ms | Qwen3.5-4B 2.84x (lost 179 of 619) | ms | Gemma 2.88x (lost 159 of 566) | ms |
|---|---|---|---|---|---|
| SWIGLU 2.03x | 15.8 | **SSM_CONV 1.38x** | **24.9** | GEGLU 2.11x | 28.7 |
| ADD l_out + ffn_inp | 14.4 | SWIGLU 1.85x | 23.6 | MUL_MAT:node (attn_out+PLE) 2.75x | 14.8 |
| FA | 5.4 | **CONCAT conv_input** | **12.8** | 3x ADD (attn_out, pe_in, node) | 26.7 |
| ROPE Q | 4.6 | MUL (gate) | 9.0 | ROPE Q + q-norm | 13.7 |
| 2x RMS_NORM+MUL | 7.0 | SILU conv_output | 7.7 | MUL l_out | 6.4 |
| | | 2x ADD residual | 14.9 | FA 3.46x | 8.4 |
| | | CPY conv_state_update | 5.5 | | |

**Qwen3.5 conv block** (CONCAT[transpose qkv_mixed] -> SSM_CONV -> SILU, + CPY of the last 3
columns into conv_state): 28.8 + 38.1 + 20.8 + 11.5 = **99 ms of 619 (16%)** of the 4c graph.
The CONCAT is a transpose (`delta-net-base.cpp:466`); the CPY gathers 3 strided columns. A
fused kernel reading `qkv_mixed` token-major with a 3-token halo moves ~2 x 67 MB per layer
(~0.8 ms at 170 GB/s, ~19 ms total) -> estimate **-80 ms (~13%)**. With the transpose gone the
natural axis becomes TOKEN (contiguous rows plus a read-only halo) instead of channels.

**GLU ops are the whole cost, not just the loss:** SWIGLU/GEGLU take 8.5% (llama), 7% (Qwen)
and 11% (Gemma) of the 4c graph. Fusing them into the gate/up MUL_MAT_NX epilogue removes the op
and halves the NX output writes.

## MoE — Qwen3.5-35B-A3B-UD-Q4_K_M (`logs/20261015-181419`, row, pp2048)

| ub | depth | 1c PP | 4c PP | x |
|---|---|---|---|---|
| 1024 | 0     | 428.0 | 985.7 | **2.30** |
| 512  | 0     | 320.0 | 768.2 | 2.40 |
| 512  | 16384 | 283.7 | 714.4 | 2.52 |
| 1024 | 16384 | 355.7 | OOM (4c) | — |
| 2048 | any   | OOM (1c and 4c) | | — |

The worst 4c scaling of all the models. 4c at ub1024 / d16k is host-OOM-killed while 1c passes,
so the mdev path costs extra host memory: a robustness item for M5. Needs a level-1 profile at
ub1024 d0 to split MUL_MAT_ID vs router/top-k vs shared expert vs GDN.
