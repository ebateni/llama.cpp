# gemma4-E2B prefill optimization on Hexagon HTP (SC8480XP)

Model: `gemma-4-E2B-it-Q4_0-imatembd.gguf`
Device: SC8480XP (Snapdragon X2 Elite), Windows on Snapdragon, HTP v81, on AC power.
Tree: `llamacpp-gemme4-e2b` at `a25c9865f` plus uncommitted changes (16 files, +1824/-60).
The fusion round at the end of this document was measured on that state plus its own changes.

## Results

`llama-bench -m <gguf> --device HTP0 -ngl 99 -fa 1 -t 6 --poll 1000 -ub 512 --load-mode none -p 512 -n 128 -r 5`, measured on 10.92.137.16 back to back:

| Package | prefill pp512 (t/s) | decode tg128 (t/s) |
|---|---|---|
| `pkg-wos-g4-base` (upstream) | 1616.75 +- 12.98 | 48.29 +- 0.12 |
| `pkg-wos-g4-mb2` (all backend work, no row pruning) | 3617.31 +- 8.05 | 53.70 +- 0.10 |
| `pkg-wos-g4-prune` (current best) | 7579.33 +- 37.21 | 53.71 +- 0.06 |

- Device time for one 512-token ubatch (`GGML_HEXAGON_PROFILE`): 138.4 ms (`mb2`) -> 64.1 ms (`prune`).
- Correctness:
  - HTP PPL (wiki.test.raw, c512, 40 chunks) is 134.3645 for both packages.
  - 48-token greedy output is byte-identical across packages.
  - CPU PPL is 134.3625 with row pruning on and off.

## Fusion round: packed weight DMA, GEGLU, softcap, PLE GELU+MUL, rope, bookkeeping

Baseline: the row-pruning tree above. This round keeps the same graph and the same numerics; it changes the HTP
kernels, four fusions and the per-op bookkeeping only. Model, device and command line as above
(`gemma-4-E2B-it-Q4_0-imatembd.gguf`, SC8480XP / Snapdragon X2 Elite, HTP v81, Windows 11 ARM64, HTP0,
`llama-bench -p 512 -n 128 -r 5 -ub 512 -fa 1 -t 6 --poll 1000`). One warmup run first, then 3 interleaved rounds
(baseline, new) with a 30 s gap between the runs; the table is the median of the 3 rounds.

| Package | prefill pp512 (t/s) | decode tg128 (t/s) |
|---|---|---|
| baseline (row pruning) | 7449.09 +- 380.51 | 54.07 +- 0.51 |
| `pkg-fuse-final3` (this commit is final3 plus the restored lm_head softcap epilogue) | 7834.33 +- 305.00 | 65.99 +- 0.52 |

- Decode 54.07 -> 65.99 t/s (+22.0%), prefill 7449.09 -> 7834.33 t/s (+5.2%).
- Per round (baseline / new): pp512 7022.25 / 7834.33, 7449.09 / 7946.07, 7661.43 / 7825.48;
  tg128 54.06 / 65.99, 54.07 / 65.24, 54.20 / 66.55.
- Bit-identical: 48-token greedy PARITY_IDENTICAL and PPL 134.3955 +- 6.77407 (`llama-perplexity`, `wiki.test.raw`,
  `-c 512 -ch 40`, HTP0), the same gate as the row-pruning work above.
- The 3-round medians above are `pkg-fuse-final3`. The tree committed here is the `f3y` state: final3 plus the
  restored lm_head softcap epilogue in the packed m=1 q4_0 mat-vec kernel (`hvx_scale_offset(s1) -> hvx_tanh ->
  hvx_scale_offset(s2)` on the per-thread tile before the copy-out), see fuse-log.md "R2a fused m=1 mat-vec prologue"
  and "CURRENT BEST / MERGE BASE".
- f3y has the same gates as final3 (chk.bat PARITY_IDENTICAL and PPL 134.3955, plus the stronger 5-output gate:
  3 greedy prompts + 2 high-temperature sampled runs with `--ignore-eos`, SAME on all five) but only a 2-round tg128
  A/B (65.84 +- 0.50 / 65.21 +- 0.98 t/s) and no 3-round pp512 A/B of its own (out of budget). The table numbers are
  final3's; they are not a fresh 3-round measurement of f3y.
- A first run after an idle gap can be 20-40% slow (pp 3000-5500, tg 25-43) on either package; the warmup run plus
  the median of the interleaved rounds are needed for a stable comparison.

### Independent verification (separate session, not the author's measurement)

A second session rebuilt this change set from scratch in its own source copy and build directory (`C:\bverify`,
package `pkg-verify`; all 4 skels and the host binaries match the build dir byte for byte, `fc /b`) and measured it on
the same kind of device (SC8480XP, HTP v81, Windows 11 ARM64), same model and command line as above. The author's
table above is unchanged; the numbers below are the independent ones (analysis/STATUS.md sections 1 and 1.1,
analysis/verify-progress.md pieces C, D, E).

- Gates reproduced: the 48-token greedy output (`llama-completion`, `-n 48 --temp 0 --seed 1234`, HTP0) is
  byte-identical (`fc /b`) to the author's package in the same session and to the author's `par_new.txt`; PPL
  (`wiki.test.raw`, `-c 512 -b 512 --chunks 40`, HTP0) is 134.3955 +- 6.77407, identical to the author's package in the
  same session.
- Interleaved A/B (`llama-bench -p 512 -n 128 -r 5`, warmup run first, 30 s gaps), baseline `pkg-new` vs `pkg-verify`:

| Medians | prefill pp512 (t/s) | decode tg128 (t/s) |
|---|---|---|
| baseline, 3 rounds | 7423.25 | 54.23 |
| independent build, 3 rounds | 7701.94 (+3.75%) | 65.32 (+20.45%) |
| baseline, 5 rounds | 7538.90 | 54.16 |
| independent build, 5 rounds | 7865.12 (+4.33%) | 65.32 (+20.61%) |

- Per round (3-round set, independent build): pp512 6492.27 / 7701.94 / 7900.76, tg128 54.87 / 66.07 / 65.32. Round 1
  was a noisy outlier (in-run stdev 2982.67 for pp512 and 22.19 for tg128); it is kept in the medians, nothing was
  dropped.
- The independent medians are 1.0-1.7% below the author's own medians above (decode 65.32 vs 65.99, prefill 7701.94 vs
  7834.33), which is inside the round-to-round noise of this device.
- One `llama-bench` run of the independent package hung mid-round on the device (over 15 minutes, partial pp512 2760.14
  +- 3502.10). The cause was not found; the same package was healthy before and after, and the supplementary rounds
  were re-run alone and completed. The partial numbers are excluded from the table.

### 9. q4_0 packed weight DMA for the m=1 mat-vec

- The m=1 HVX q4_0 kernels (`hvx_mv_2d_repacked_pk_q4_0`, `hvx_mm_nx_2d_repacked_pk_q4_0`) fetched each 32-row
  weight tile with a 2D descriptor of 576-of-640 B rows. A diagnostic that skipped the dot compute showed that this
  strided fetch, not the prefetch depth and not the HVX math, was the decode bottleneck: DMA-only decode 56.4 t/s with
  the 2D descriptor, 64.0-65.2 t/s with one contiguous 1D block per output tile, 60.1 t/s with 2-tile rows; a
  prefetch-depth sweep (2/4/8/16/32/64) was flat between 52.4 and 54.6 t/s.
- Change: each 32-row output tile is DMAed as one contiguous block of `n_k_tiles * 576` B with a single 1D
  descriptor; `tiled_vec_dot_q4_0_32x1_pk` reads the odd tiles with unaligned vector loads (same math and order).
- Result: 59.55 +- 0.39 t/s against 54.1-54.6 baseline (+9.8%); PARITY_IDENTICAL, PPL 134.3955.

### 10. GEGLU fused into the decode gate+up mat-vec

- New HVX kernel `hvx_mm_nx_geglu_pk_q4_0` (op `HTP_OP_MUL_MAT_NX_GEGLU`): gate and up tiles are DMAed as contiguous
  blocks, two packed dots, then the f32 `gelu(x) * y` epilogue in the same op (`hvx_vec_geglu_f32` moved to
  `hvx-sigmoid.h`, same math as the standalone GEGLU). The host matcher accepts the HVX q4_0 m=1 NX path (no HMX) as
  well as the HMX path.
- The epilogue made the packed dot stop inlining on the first attempt (58.0 against 59.7 t/s); `always_inline` fixed it.
- Result: the 0.52 ms/token standalone GEGLU is gone and the decode batch drops 16262 -> 15889 us/token; interleaved
  A/B tg128 baseline 54.28/54.07/54.12, packed DMA 59.54/59.40/55.54, this step 61.13/61.73/60.71 t/s.

### 11. lm_head softcap fusion

- New op `HTP_OP_MUL_MAT_CAP` (host match `MUL_MAT + SCALE + TANH + SCALE`, flag 4096): the m=1 q4_0 mat-vec writes
  the tile into VTCM and applies `scale(s1) -> tanh -> scale(s2)` per thread before the copy-out, using the same HVX
  functions as the three standalone ops.
- Result: lm_head + SCALE + TANH + SCALE 1.92 + 0.04 + 0.24 + 0.04 = 2.24 -> 2.01 ms/token; the lm_head weight stream
  reaches 226 MB in 1.90 ms = 119 GB/s (91 GB/s before). PARITY_IDENTICAL, PPL 134.3955.

### 12. GELU + MUL fusion for the per-layer embedding gate

- New op `HTP_OP_GELU_MUL` (flag 8192, `gelu_mul_f32`): `gelu(inp_gate) * inp_per_layer` in one pass
  (`hvx_vec_gelu_f32` moved to `hvx-sigmoid.h`, the same function as the standalone GELU).
- In the output-pruned part of the graph a `GET_ROWS` of `inp_per_layer` sits between the GELU and the MUL; the node
  loop now moves that `GET_ROWS` in front of the GELU (it does not read the GELU result), so 34 of 35 layers fuse
  (layer 0 has a different shape). The same reorder helps decode: GET_ROWS 5.3 + GELU 4.5 + MUL 10 us -> GELU+MUL
  4.3 us.
- Result: prefill GELU 1.26 + MUL about 1.5 -> 3.8 ms/ubatch, about 0.9 ms saved; the profiled batch drops
  62.4 -> 60.4 ms. A/B pp512 baseline 7572/7512/7462 against 7957/7928 t/s (+3%); PARITY_IDENTICAL, PPL 134.3955.

### 13. Flat deep-pipeline rope

- New `rope_job_f32_flat` for contiguous src/dst: 16-row DMA blocks cross token boundaries (the old job stopped at
  head boundaries, 8 rows), 8 slots / 4 blocks in flight, and the per-token theta (cos/sin) cache is only recomputed
  when the token index changes (`rope_theta_update`, shared with the old path). That recompute was about 22 us per
  call per thread and dominated the small k calls.
- Result: q 3.18 -> 2.85 ms and k 0.98 -> 0.79 ms per ubatch (-0.5 ms); pp512 is flat within noise.
  PARITY_IDENTICAL, PPL 134.3955.

### 14. Cheaper dirty-range bookkeeping

- `merge_dirty_ranges` restarted its 64x64 scan from 0 after every merge and ran 2-3 times per op. Now one pass per
  range, restarting only the range that grew, and `dirty_all` merges only when a range grew or several outputs were
  added. `HTP_MAX_DIRTY_RANGES` 64 -> 16 (few live tensors per op chain; evicting an old range only flushes it
  earlier, which is conservative).
- Result: thread 0 `htp_tensor_dirty_all` 1287 -> 813 us and `htp_tensor_flush_all` 940 -> 877 us per token, and
  tg128 62.5 -> 64.2/65.1 t/s (a variant with more range slots measured 65.9/65.9 t/s). PARITY_IDENTICAL, PPL
  134.3955. The rest of the per-op bookkeeping is not removable: see "Measured and rejected".

## Optimization steps

Numbers per step are medians of 6-8 interleaved `llama-bench` pp512 rounds on 10.92.138.99. That device was later found throttled on battery, so only the relative change between adjacent builds is meaningful. The final table above is from 10.92.137.16.

### 0. Pre-existing patch (before this work)

- Store BF16 weights as F16.
- Fuse gate/up matmul with GEGLU: `HTP_OP_MUL_MAT_NX_GEGLU`.
- The GEGLU result is emitted as f16 in HMX activation-tile layout and DMAed straight into down-proj: `HTP_OP_MUL_MAT_NX_GEGLU_T` -> `HTP_OP_MUL_MAT_AT`.
- `opt_oppoll=1`.

### 1. ADD + RMS_NORM + MUL fusion

- New op `HTP_OP_ADD_RMS_NORM_MUL`. The sum stays in VTCM, is written out once (it is the next residual) and is never read back for the norm.
- Files:
  - `htp/htp-ops.h`
  - `htp/hvx-norm.h` (`hvx_fast_add_rms_norm_mul_f32`)
  - `htp/unary-ops.c/.h` (`unary_thread_add_rms_norm_mul_f32`, `htp_unary_fused_vtcm_layout_build`)
  - `htp/main.c`
  - `ggml-hexagon.cpp` (`try_merge_add_rms_norm_mul`)
- Fusion flag: `GGML_HEXAGON_FUSE_ADD_RMS_NORM_MUL` (512).
- Alias guard: an output that exactly reuses an input buffer is allowed; partial overlap, or overlap between the two outputs, is rejected. Without the guard, PPL on the folded model was 428.

### 2. Down-proj chunking for the tiled f16 activation

- `MUL_MAT_AT` reads an activation that is already f16 tiles, so it needs no f32 scratch.
- `htp_mm_hmx_solve_2d_tiled_act_params` (`htp/matmul-ops.h`) re-solves `mc`/`nc` without that scratch, which allows larger row chunks.

### 3. PLE graph simplification (`src/models/gemma4.cpp`)

- `per_layer_projection_scale` is folded into the RMS eps: `eps / s^2`.
- `per_layer_input_scale` is folded into the norm weight and the table scale.
- `gemma4_view_2d_slice` uses the real strides, so the `ggml_cont` after the permute is gone.
- New GGUF KV `<arch>.embedding_per_layer_scale` (`src/llama-arch.*`, `src/llama-hparams.h`; default 1.0, so old models are unchanged).
  - `/tmp/ple/fold_ple_scale.py` folds the PLE table scale into the Q4_0/Q8_0 block `d`, which removes a full-tensor SCALE.
  - The main embedding scale is not folded because it is tied to lm_head.

### 4. Flash attention: skip fully masked causal KV blocks (`htp/flash-attn-ops.c`)

- `fa_kv_blocks_live()` scans the mask once per q tile and trims trailing all-masked KV blocks.
- The DMA push/pop pairing is preserved: block 0 and the next-tile prefetch are unchanged.
- Result: 3230 -> 3284 t/s (+1.7%).

### 5. q/k RMS_NORM + MUL + ROPE fusion

- New op `HTP_OP_RMS_NORM_MUL_ROPE`. The norm runs in place on the rows rope has already staged in VTCM.
- Files:
  - `htp/rope-ops.c/.h` (norm weight is DMAed into VTCM)
  - `ggml-hexagon.cpp` (`try_fuse_rms_norm_mul_rope`)
- Fusion flag: `GGML_HEXAGON_FUSE_RMS_NORM_MUL_ROPE` (1024).
- Result: 3221 -> 3330 t/s (+3.4%).

### 6. Residual chain fusion: RMS_NORM + MUL + ADD [+ MUL scale] [+ RMS_NORM + MUL]

- New op `HTP_OP_RMS_NORM_MUL_ADD`. It covers the post-norm, the residual add, the optional `layer_output_scale` multiply and the next layer's pre-norm in one pass.
- Files:
  - `htp/hvx-norm.h` (`hvx_fast_rms_norm_mul_add_f32`)
  - `htp/unary-ops.c`
  - `ggml-hexagon.cpp` (`try_fuse_rms_norm_mul_add`, `try_fuse_norm_add_scale`, `try_merge_norm_add_norm`)
- Fusion flag: `GGML_HEXAGON_FUSE_RMS_NORM_MUL_ADD` (2048).
- The scalar scale is DMAed into VTCM. The first version read it from DDR on the device and hung prefill.
- Result: 3274 -> 3460 t/s (+5.7%). After removing per-element branches from the HVX loop: 3480 -> 3535 t/s (+1.6%).

### 7. PLE chain fusion + down-proj m-block balancing

- `gemma4.cpp` adds the scaled PLE norm weight to the graph first (`ggml_build_forward_expand`). The PLE RMS_NORM -> MUL -> ADD chain is then contiguous and fuses: 1126 us -> 464 us.
- `hmx_mm_2d_f32` (`htp/matmul-ops.c`) splits rows evenly across m-blocks. A small tail block is bound by the weight stream, not by HMX.
- Result: 3520 -> 3571 t/s.

### 8. Prune rows before the first KV-shared layer (`src/models/gemma4.cpp`)

Source: the QC Step-2 tutorial (`/wd/weihuan/gemm4-e2b-it-tutorials/model/Step-2/base`). Its prefill graph keeps only the KV-cache outputs and drops `logits`, lm_head and the KV-shared layers.

- Why it is valid: gemma4-E2B has 35 layers. Layers 15-34 (20 layers) reuse the KV of earlier layers and write nothing to the cache, so in prefill they only feed the output tokens.
- Change: before layer 15, keep only the rows in `inp_out_ids` for:
  - hidden state `inpL`
  - `inp_pos`
  - both KV masks (global and SWA), which the shared layers' FA uses
  - the per-layer PLE slice
- Layers 0-14 still run on all tokens and fill the KV cache, so decode sees the same cache.
- Fallback: pruning is skipped when every token's hidden state is needed:
  - `embeddings`
  - unmasked MTP nextn
  - layer-input export at layer >= 15
  - multiple streams
  - non-causal attention
- Effect: 20 layers of Q projection, FA, FFN and PLE go from 512 rows to 1 row in pp512.
  - Device time per ubatch: 138.4 -> 64.1 ms.
  - Prefill: 3617 -> 7579 t/s.
  - Decode is unchanged (1 token, nothing to prune).

## Measured and rejected

- int8 HMX: slower for k=1536 (0.79x). Down-proj gains 1.4-1.65x, but the end-to-end ceiling is about +4%.
- FP8 HMX: 0.35-0.72x of fp16.
- int32-accumulate single-pass readout: returns zeros.
- IQ4_NL / MXFP4 / Q8_0 weight formats: no prefill change.
- Flash attention Br tiling sweep: +-3%.
- Per-op flush/dirty bookkeeping removal (fuse-log.md, "R1 per-op bookkeeping: ceiling experiment"): ceiling only.
  Removing `htp_tensor_flush_all` and `htp_tensor_dirty_all`
  from the op prologue gives +0.41 ms/token decode (65.93/66.43 -> 67.76/68.28 t/s) and about +2.4 ms/ubatch prefill at
  best, and it breaks parity (PARITY_DIFF): those flushes are what makes parity hold. A host-side static plan cannot
  decide per opcode whether an output is written through the cache or whether a consumer reads through it, and a wrong
  guess is a data-dependent coherence bug. Not shipped.
- Fused m=1 mat-vec prologue (fuse-log.md, "R2a fused m=1 mat-vec prologue"): neutral, not shipped. Overlapping the
  activation DMA/quantize with the first weight-tile fetch (spin barrier on act-ready + quant-done, activation staging
  buffer moved out of the weight VTCM region) measured 65.79 +- 1.54 / 65.61 +- 0.48 t/s against 65.84 +- 0.50 /
  65.21 +- 0.98 for the same tree without it. It
  added VTCM pressure and a barrier for no gain; the first-tile latency is per thread, not serial. Earlier fast reads
  of this idea (69.34/69.14 t/s) came from a base that was missing the lm_head softcap epilogue in the mat-vec kernel
  and are discarded.
- PLE tail fusion (gate matmul -> GELU -> mul -> proj matmul in one op), fuse-log.md "R3 PLE tail" repriced by
  "B) PLE matmuls are already HMX and DDR-bound": ceiling about 0.5-0.7 ms/ubatch, not implemented. The per-layer PLE
  tail is 175.7 us/layer (about 6.1 ms/ubatch): inp_gate matmul 32.6 + GELU+MUL 105.4 + proj matmul 37.7 us. R3 first
  priced the upside at about 1.4 ms/ubatch (both matmuls against an ideal of about 1 ms); the later profile shows both
  matmuls are already on HMX (hmx-tiled mc 512 nc 256 and nc 1536) and are bound by DDR traffic: per-layer traffic is
  about 7 MB, and a chained VTCM op removes only about 1 MB of it (the GELU+MUL output write and the proj activation
  read), about 0.35 ms/ubatch, plus two ops' fixed overhead, about 0.3 ms/ubatch, hence the 0.5-0.7 ms/ubatch ceiling.
  The 105.4 us GELU+MUL is HVX-compute bound (tanh-approx gelu + mul over 131k elements) and would not shrink in a
  fused op. It needs chained HMX matmuls with a VTCM-resident 512x256 intermediate and a new kernel-parameter layout.

## Not done (estimates, not measured)

- Lower per-op L2 flush cost: about 2-5 ms per ubatch on the old graph (thread 0 spent about 3.7 ms flushing inside ops).
- Overlap the next matmul's activation prep with the previous HMX: matmul prologues summed to about 12.9 ms.
- Down-proj split-K so k=12288 weights stream once. Moot after step 8: the k=12288 layers are all KV-shared, so pp512 runs them on 1 row.

## Runtime switches

- `GGML_HEXAGON_OPFUSION` bitmask. All new fusions are on by default:
  - 128: GEGLU
  - 256: GEGLU_T chain
  - 512: ADD + RMS_NORM + MUL
  - 1024: RMS_NORM + MUL + ROPE
  - 2048: RMS_NORM + MUL + ADD chain
  - 4096: MUL_MAT + SCALE + TANH + SCALE (lm_head softcap)
  - 8192: GELU + MUL (per-layer embedding gate)
- `GGML_HEXAGON_PROFILE=1` (per-op time) or `=3` (per-thread trace).

## Build and deploy (WoS)

1. Build the skels on Linux: `cmake --build build-snapdragon -j 16`, producing `build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-v*.so`.
2. Copy the skels to `C:/bw/ggml/src/ggml-hexagon/catpkg/` and sign them with Inf2Cat + signtool (cert `GGML HTP Test`).
3. Build the host on the device: source in `C:\myfiles\weihuan\llama-g4`, build dir `C:\bw2`, script `C:\myfiles\weihuan\build-pkg.bat <pkg>`.
4. Verify: `chk.bat <pkg>` (parity + PPL + profile), `ab2.bat <A> <B> <rounds>` (interleaved A/B), `verify_pd.bat <pkg>` (pp512 + tg128).
5. To run on another device, trust the public `GGML HTP Test` cert: `certutil -addstore Root` and `certutil -addstore TrustedPublisher`.
