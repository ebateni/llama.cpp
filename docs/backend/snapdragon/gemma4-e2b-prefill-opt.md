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

## Merged round: int8 HMX matmuls with activation smoothing (DEVICE RUN 4)

This round adds the int8 HMX work on top of the fusion round above. It is off by default: with `GGML_HEXAGON_INT_MM=0`
the merged tree is bit-identical to the fusion round. Source of the numbers: analysis/merge-log.md "DEVICE RUN 4" and
analysis/hmx-int-log.md M90-M98. Tree: the fuse fast state plus the frozen int8 tree, lab plumbing removed, v73/v75
wrapper fix applied.

Same model and device as above (`gemma-4-E2B-it-Q4_0-imatembd.gguf`, SC8480XP, HTP v81, Windows 11 ARM64, HTP0).
`llama-bench -fa 1 -t 6 --poll 1000 -ub 512 --load-mode none`, prefill `-p 512 -n 0 -r 5`, decode `-p 512 -n 128 -r 5`.
Warmup of every arm first, 3 rounds, arm order rotated per round, 30 s gaps, a health check before every run (nothing
was flagged contaminated, no round excluded). Medians of the 3 rounds:

| Arm | prefill pp512 (t/s) | decode tg128 (t/s) |
|---|---|---|
| `pkg-new` (baseline) | 7604.54 | 54.38 |
| merged, `INT_MM=0` (fp16 path) | 7985.58 (+5.0%) | 66.47 (+22.2%) |
| merged, `INT_MM=1` (W4A8 expand, 11 layers) | 8135.77 (+7.0%) | 65.55 (+20.5%) |
| merged, config A (smoothed packed gate+up A8) | 8407.10 (+10.6%) | 66.16 (+21.7%) |

- Combined result over `pkg-new`: prefill +10.6% (config A), decode +22.2% (66.47 t/s with `INT_MM=0`; config A is
  within 0.2% of it, 66.16 t/s).
- Contribution split: the fuse side gives +5.0% prefill and +22.2% decode (the `INT_MM=0` arm isolates it); config A
  adds +3.5% prefill on top (paired mean 1.035, rounds 1.074 / 0.989 / 1.042) and leaves decode unchanged
  (0.998 of `INT_MM=0`), because m=1 never reaches the int8 code. In ms per 512-token ubatch: `pkg-new` 67.3,
  `INT_MM=0` 64.1, config A 60.9. Config A round 2 has a 16.6% own stddev and is kept; the median is robust to it.
- Config A: `GGML_HEXAGON_INT_MM=3 GGML_HEXAGON_INT_MM_MASK=4 GGML_HEXAGON_INT_MM_LAYERS=7ffffffff
  GGML_HEXAGON_INT_MM_REDUCE=2 GGML_HEXAGON_INT_MM_REDUCE_BIG=4 GGML_HEXAGON_INT_SMOOTH=<path>/smooth.bin
  GGML_HEXAGON_VMEM=5G` (gate+up as packed W8A8 on all 35 layers, smoothed; everything else stays fp16).

### Gates that passed

- Bit-identity of the fp16 path (`INT_MM=0`): `llama-perplexity` PPL exactly 134.3955 +/- 6.77407, 48-token greedy
  identical, 5-output gate SAME on all five (3 greedy + 2 high-temperature sampled).
- Int gate, config A, against the fp16 logits (20 chunks, c512): KLD 0.007160 +/- 0.001159 (limit 0.008), same-top-p
  96.941 +/- 0.241 % (limit 96.0), greedy identical. PPL is 132.4530 (-1.44% vs fp16, informational: int8 error lowers
  PPL on wiki, the KLD gate governs). `INT_MM=1` expand: KLD 0.006950, top-p 96.765 %, PPL 133.8129.
- Greedy parity is not a gate for the int path: a packed gate+up variant with reduce 4 gave PPL 404 / KLD 2.2 with
  identical 48 greedy tokens (hmx-int-log.md M77). Every int configuration must pass the KLD / top-p gate.

### KPI distance

- Prefill target 10000 t/s: NOT MET. Config A 8407.10 is 84.1% of it, 1593 t/s short (about 9.7 ms per ubatch to go from
  60.9 ms against the 51.2 ms target).
- Decode target 70 t/s: NOT MET. Best 66.47 t/s, 3.53 t/s short (about 15.0 ms/token now, 14.3 ms needed).
- The remaining int options cannot close the prefill gap: qkv / o-proj A8 pass the gate alone but save at most 0.7 ms
  and exhaust the KLD budget together with gate+up (M93); down (k=6144 and k=12288) fails A8 (KLD 0.16 / 0.02) and A16 is
  1.5-2x slower than fp16; the m-split overlap is slower and stays off (`GGML_HEXAGON_INT_SPLIT=0`).

### Why decode stops at about 66 t/s

The q4_0 decode dot spends about 79 nibble-unpack ops (26 vshuff + 21 valign + 8 vand + 8 vlsr + 16 vsub.b per two
tiles) per 16 vrmpy. HVX has 2 permute/shift slots, so the permutes alone need at least about 40 packets against 61-64
observed. That is about 11.8 MAC/cycle/thread, roughly 9% of the vrmpy peak (128 MAC per instruction). Pairing two output
tiles per pass cannot remove the unpack ops and, made bit-identical, costs more than it saves (gate+up +55%, MUL_MAT
+23%, lm_head +23%), so it was dropped. The next step is a load-time q4_0 weight repack, so that a plain vand / vlsr
yields the vrmpy operand order directly.

### Int8 env scheme (all default off, fp16)

- `GGML_HEXAGON_INT_MM`: 0 off | 1 W4A8 on-the-fly expand, gate+up only (no packing, no smoothing, reduce 4) | 2 packed +
  Hadamard rotation (diagnostic) | 3 packed W8A8, unrotated (the shipped mode) | 4, 5 diagnostics | 6 A16 for the plain
  classes. The int paths are packed W4A8 / W8A8 at load time, with a two-pass A16 form and a two-level bias for k=12288.
- `GGML_HEXAGON_INT_MM_MASK`: class bits, 1 qkv, 2 o-proj, 4 gate+up, 8 down k=6144, 0x10 lm_head (never packed), 0x20
  PLE model_proj, 0x40 PLE gate/proj (no op), 0x80 down k=12288; default 0xEF. `GGML_HEXAGON_INT_MM_MASK16`: classes
  that run A16 (two passes) instead of A8; gate+up has no A16.
- `GGML_HEXAGON_INT_MM_LAYERS` / `_LAYERS_GATE`: layer masks for the plain classes / the gate+up class (default all).
- `GGML_HEXAGON_INT_MM_REDUCE` (window headroom bits for the packed classes, 2 packed, 4 expand) and `_REDUCE_BIG`
  (k=12288 class, 4; 8 clips). `GGML_HEXAGON_INT_MM_SCHEME` (1 = load-time Q4.4 numerics), `GGML_HEXAGON_INT_SPLIT`
  (m-split overlap, 0), `GGML_HEXAGON_VMEM` (HTP VM cap, 5G for the packed copies, about 1.4 GB extra host RAM).

### SHIPPING REQUIREMENT: smooth.bin travels with the model

- `smooth.bin` (777299 bytes, sha256 prefix 26a4948d5bea2dac, 105 vectors keyed `blk.L.{attn_q,ffn_gate,attn_output}.weight`,
  layout `[u32 name_len][name][u32 k][f32 s[k]]`) MUST be shipped next to the model and pointed to with
  `GGML_HEXAGON_INT_SMOOTH`. It is generated by `smooth_calib.py` with the host dumper `dump2b` from 6 `wiki.test.raw`
  texts of about 9 KB (about 54 KB, about 3000 tokens), about 10 minutes on a host CPU.
- Exact fp16 fallback when it is absent: if the file is missing, unreadable or a vector is short, the tensors that need
  smoothing are NOT packed and run the exact fp16 path (one warning naming the tensor); the quality regression of a
  silent unsmoothed A8 is never shipped. Verified on the device: KLD 0.000000 / top-p 100.0 without the file or with a
  non-existent path (M92). `GGML_HEXAGON_INT_SMOOTH_OPTIONAL=1` forces unsmoothed A8 (diagnostic only, KLD 0.0229 on
  wiki). A16 classes (for example qkv in `MASK16`) do not need smoothing.
- Domain note: calibrated on wiki text only. Validated on code, docs and chat text (about 60 KB each, 20 chunks c512), where
  smoothing lowers KLD 2.6-3.4x versus unsmoothed A8 and config A passes the gate (0.003-0.006) on all three (M96).
  Those texts are low-perplexity, the high-entropy wiki regime is the pessimistic case. Not tested: other languages,
  long context (> 512), math. For another language or a very different distribution, regenerate `smooth.bin` with
  `smooth_calib.py` and re-validate the KLD gate on a sample of that text before enabling.

## Repack round: q4_0 tile layout v2, decode above 70 t/s (DEVICE RUN 5 and 6)

This round adds a load-time q4_0 weight repack (tile layout v2) to the merged tree of the previous round, plus a
merge-local guard for `INT_MM=1`. Source of the numbers: analysis/merge-log.md "DEVICE RUN 5" and "DEVICE RUN 6",
analysis/fuse-log.md (repack and DECODE-2 sections). The fp16 path stays bit-identical.

Same model, device and command lines as the previous round (`gemma-4-E2B-it-Q4_0-imatembd.gguf`, SC8480XP, HTP v81,
Windows 11 ARM64, HTP0; `llama-bench -fa 1 -t 6 --poll 1000 -ub 512 --load-mode none`, prefill `-p 512 -n 0 -r 5`,
decode `-p 512 -n 128 -r 5`). Warmup of every arm first, 3 rounds, arm order rotated per round, 30 s gaps, health check
before every run; both sessions were normal (pkg-new decode 53.7-54.4) and no round was contaminated or excluded.
Medians of the 3 rounds.

DEVICE RUN 5 (before the repack: final9 fuse state plus the frozen int tree):

| Arm | prefill pp512 (t/s) | decode tg128 (t/s) |
|---|---|---|
| `pkg-new` (baseline) | 7463.79 | 54.33 |
| merged, `INT_MM=0` | 7913.13 | 66.52 |
| merged, `INT_MM=1` (expand) | 7975.23 | 66.84 |
| merged, config A | 8231.58 | 65.74 |

DEVICE RUN 6 (with the repack and the `INT_MM=1` guard):

| Arm | prefill pp512 (t/s) | decode tg128 (t/s) |
|---|---|---|
| `pkg-new` (baseline) | 7482.80 | 54.15 |
| merged, `INT_MM=0` (fp16 path) | 8003.40 (+7.0%) | 71.53 (+32.1%) |
| merged, `INT_MM=1` (guard: falls back to fp16) | 7912.27 (+5.7%) | 71.29 (+31.7%) |
| merged, config A (packed int8 + smoothing, gate+up all 35 layers) | 8569.12 (+14.5%) | 70.20 (+29.6%) |

- Effect of the repack on the same int tree (run 5 -> run 6, same-session pkg-new references 54.33 -> 54.15): decode
  `INT_MM=0` 66.52 -> 71.53 (+7.5%) and config A 65.74 -> 70.20, so the decode gain is the repack. Its prefill effect is
  within run-to-run noise and is NOT claimed: `INT_MM=0` prefill is 7986 / 7913 / 8003 in runs 4 / 5 / 6 against
  `pkg-new` references of 7605 / 7464 / 7483, i.e. ratios vs `pkg-new` of 1.050 / 1.060 / 1.070; config A moved
  8407 -> 8232 -> 8569 with the same noise (own stddev up to 15% in single rounds). The run-5 and run-6 tables above are the
  measurements; do not read the config A prefill difference between them as an effect of the repack.
- Config A is 0.981 of `INT_MM=0` in decode (70.20 vs 71.53). m=1 decode never runs the int8 ops; the offset is probably
  the larger weight buffer (inference, not measured). One config A decode round (69.47) is below 70; the median clears
  70 by 0.20 t/s only.
- How the repack works: q4_0 weights are repacked once at load so that the vrmpy operand order comes out of a plain
  AND / shift. Unpack per plane drops from 9 ops (vand, vlsr, 3x vshuff, 2x valign, 2 vsub) to 4 (vand, vlsr, 2 vsub.b).
  The geglu dot loop goes from 63-64 packets (26 vshuff, 21 valign) to 34-35 packets per 2 k-tiles (16 vrmpy, 2 vshuff,
  5 valign per tile-dot), about 45% fewer. Op sum per token 14.23 -> 13.17 ms (NX+GEGLU 5.37 -> 4.95, MUL_MAT 4.95 ->
  4.57, lm_head 1.94 -> 1.76 ms); the loop is now partly DMA-bound again, so the packet cut shows as -8% op time.
  Files: host repack (`repack_q4_0_tiled`, v2 flag for Q4_0 only, IQ4_NL keeps the legacy layout) and the
  `INT_MM=1` guard in `ggml-hexagon.cpp`, the packed kernels in `htp/hvx-mm-kernels-tiled.h` and `htp/matmul-ops.c`,
  the HMX dequant consumers in `htp/hmx-mm-kernels-tiled.h` and `htp/get-rows-ops.c`; layout flag
  `HTP_Q4_0_TILE_LAYOUT_V2` in `htp/htp-ops.h`.

### Gates (DEVICE RUN 6)

- `INT_MM=0` bit-identity on the v2 layout: PARITY_IDENTICAL, PPL exactly 134.3955 +/- 6.77407, 5-output gate SAME on
  all five (g0 g1 g2 s2 s0).
- Config A against the fp16 logits: KLD 0.007160, top-p 96.941%, PPL 132.4530 (-1.44%, informational), greedy identical.
  Identical to runs 4 and 5: the int8 copy is packed from the GGUF bytes, independent of the tile layout.
- Static compile matrix: 31 objects / 0 errors on all 8 cells (SDK 6.6.0.0 and 6.7.0.0 x v73 v75 v79 v81); host
  `clang++ -std=c++17 -mavx512fp16 -fsyntax-only` on `ggml-hexagon.cpp` 0 errors. v73 / v75 / v79 are compile-only.

### INT_MM=1 is deprecated under the v2 layout

- `INT_MM=1` (on-the-fly int8 expand of Q4_0 tiles) reads the legacy tile layout, which no longer exists after the
  repack. Guard on the host: `try_fuse_mul_mat_geglu` sets `int_mm=0` for `INT_MM=1` with Q4_0 tensors and logs one
  warning (`INT_MM=1 (on-the-fly int8 expand of Q4_0 tiles) is disabled: the Q4_0 tile layout is v2 ...`). Guard on the
  DSP: `hmx_mm_nx_geglu_2d_int8` returns NO_SUPPORT under `HTP_Q4_0_TILE_LAYOUT_V2`, so the op falls through to the fp16
  HMX geglu.
- Measured: the `INT_MM=1` arm is bit-equal to fp16 (KLD 0, top-p 100.000%, PPL 134.3955, one warning line) and runs at
  fp16 speed (7912.27 / 71.29). Config A is unaffected.

### KPI verdict

- Decode target 70 t/s: MET. `INT_MM=0` 71.53 (+1.53), guard arm 71.29, config A 70.20 (+0.20 only, one round at
  69.47). Single device session, 3 rounds.
- Prefill target 10000 t/s: NOT MET. Best arm config A 8569.12 is 85.7% of it, 1431 t/s short (59.7 ms ubatch against
  51.2 ms, about 8.5 ms to go).
- Both targets with one configuration: NOT MET (`INT_MM=0` has decode 71.5 and prefill 8003, config A has prefill 8569
  and decode on the 70 line).
- Honest prefill ceiling, bit-identically: about 8500-8700 t/s, for two structural reasons. (1) The HMX matmuls are
  about 38 ms of the roughly 60 ms ubatch and are already near the fp16 HMX peak. (2) The rest is descriptor- and
  bandwidth-bound op work, and its largest single item is the host serial upload / download around each ubatch
  (about 3-4 ms), which sits outside the backend (a scheduler-level change); overlapping it from the backend is worth
  0 ms.

## Final state: multi-row SET_ROWS block DMA (DEVICE RUN 7)

The final tree adds one source change on top of the repack round: `htp/set-rows-ops.c` groups SET_ROWS into block DMAs
(new kernels `set_rows_thread_block_{f32,f16}_{int32,int64}`, consecutive-index detection, block size `SET_ROWS_BLOCK`
= 16, dispatched before the old kernel). Per op (PROFILE=1, last prefill ubatch, same session, repack vs repack plus
B=16): SET_ROWS 0.930 -> 0.533 ms per ubatch (-0.40 ms); with the B=8 first version it was 0.919 -> 0.553 ms. The
fp16 path stays bit-identical. Source of the numbers: analysis/merge-log.md "DEVICE RUN 7" and analysis/fuse-log.md
("PREFILL item 1 RESULT", "final13 RESULT"). Same model, device and command lines as the previous rounds.

DEVICE RUN 7 (final tree, v2 guard, frozen int tree), medians of the 3 clean interleaved rounds. Warmup of every arm
first, 30 s gaps, a health check before every run. Two rounds were flagged contaminated by the health rule (prefill
round 2, decode round 3), excluded and repeated, never averaged; the session was otherwise normal (pkg-new decode
53.8-54.3):

| Arm | prefill pp512 (t/s) | ratio vs `pkg-new` | decode tg128 (t/s) | ratio vs `pkg-new` |
|---|---|---|---|---|
| `pkg-new` (baseline) | 7391.58 | 1.000 | 54.13 | 1.000 |
| merged, `INT_MM=0` (fp16 path) | 8099.76 | 1.096 | 71.42 | 1.319 |
| merged, `INT_MM=1` (guard: falls back to fp16) | 8130.88 | 1.100 | 71.03 | 1.312 |
| merged, config A (packed int8 + smoothing) | 8424.51 | 1.140 | 71.07 | 1.313 |

Both runs side by side (same int tree, same configs; the earlier table is DEVICE RUN 6 above, this one is DEVICE RUN 7):

| Metric | run 6 | run 7 | run 7 / run 6 |
|---|---|---|---|
| pp512 `INT_MM=0` | 8003.40 | 8099.76 | 1.012 |
| pp512 config A | 8569.12 | 8424.51 | 0.983 |
| pp512 guard arm | 7912.27 | 8130.88 | 1.028 |
| tg128 `INT_MM=0` | 71.53 | 71.42 | 0.998 |
| tg128 config A | 70.20 | 71.07 | 1.012 |
| tg128 guard arm | 71.29 | 71.03 | 0.996 |
| pp512 `pkg-new` | 7482.80 | 7391.58 | 0.988 |

- SET_ROWS B=16 is NEUTRAL within noise; no gain is claimed. Decode (which never takes the multi-row SET_ROWS) differs
  by -0.2% / +1.2% / -0.4% with per-run stddev 0.3-1.7%. Prefill changes of +1.2%, -1.7% and +2.8% have mixed signs
  across the arms and sit inside the run-to-run spread (arm stddev up to 5.8%, `pkg-new` round stddev up to 14.8%).
  The expected effect, about -0.4 ms per ubatch (about +0.7% prefill), is below the resolution of a 3-round A/B.
- The run-6 vs run-7 spread on the same config (config A prefill 8569.12 vs 8424.51, decode 70.20 vs 71.07) is
  run-to-run noise, not an effect of the change. Read both tables as two samples of the same state.

### Gates (DEVICE RUN 7)

- `INT_MM=0` bit-identity (hard stop): PARITY_IDENTICAL, `llama-perplexity` PPL exactly 134.3955 +/- 6.77407, 5-output
  gate SAME on all five (g0 g1 g2 s2 s0).
- Config A against the fp16 logits: KLD 0.007160, top-p 96.941%, PPL 132.4530 (-1.44%, informational), greedy
  identical; identical to runs 4-6.
- `INT_MM=1` with the v2 guard: KLD 0, top-p 100.000%, PPL 134.3955 (equal to fp16), greedy identical, one guard
  warning (first tensor `blk.0.ffn_gate.weight`).
- Static compile matrix: 31 objects / 0 errors on all 8 cells (SDK 6.6.0.0 and 6.7.0.0 x v73 v75 v79 v81); host
  `clang++` syntax check 0 errors. v73 / v75 / v79 are compile-only.

### KPI distance (DEVICE RUN 7)

- Decode target 70 t/s: MET in all three merged arms (71.42 `INT_MM=0`, 71.03 guard arm, 71.07 config A). Config A
  margin is +1.07 t/s with one of three rounds below 70 (69.68).
- Prefill target 10000 t/s: NOT MET. Best arm config A 8424.51 (median) is 84.2% of it, 1575 t/s short; its best single
  round is 8463.50. `INT_MM=0` is 8099.76. Run 6 gave 8569.12 (85.7%) for the same configuration, see the noise note
  above.
- Both targets with one configuration: NOT MET (config A: decode 71.07, prefill 8424.51).
- The bit-identical prefill ceiling stays about 8500-8700 t/s, for the structural reasons given in the repack round.

## Current state: R32 rotation int8, L2 nodirty subset, FA head scan (RUN 9)

RUN 9 supersedes the DEVICE RUN 7 state above as the pushed state. It adds, on top of the SET_ROWS round: (1) the R32
rotation for the int8 path (an in-lane Walsh-Hadamard transform over 32 elements, applied on top of smoothing, in the
host packer and in the DSP activation quantize); with it A8 also fits qkv and o-proj under the KLD gate. (2) The L2
flush "nodirty" safe subset in the host op builder (`ggml-hexagon.cpp`, `std::max<uint64_t>` spelling) and the FA
head-scan fix in `htp/flash-attn-ops.c`, together about -1.0 ms per ubatch. (3) The GELU+MUL stage cap in
`htp/act-ops.c` (a thread's block is capped so each thread runs at least 4 pipeline stages). Source of the numbers:
analysis/merge-log.md "DEVICE RUN 9" and analysis/hmx-int-log.md M123 / M124. The fp16 path stays bit-identical.

Merged tree: fuse patch `fuse-fa1nd2` (2045 lines, 17 files) on the frozen int tree (3338 lines) with the v2 guard,
lab plumbing removed. Same model, device and command lines as the previous rounds. Medians of 3 clean interleaved rounds,
arm order rotated per round, 30 s gaps, a health check before every run; nothing was contaminated, excluded or repeated
(pkg-new decode 54.13-54.35):

| Arm | prefill pp512 (t/s) | ratio vs `pkg-new` | decode tg128 (t/s) | ratio vs `pkg-new` |
|---|---|---|---|---|
| `pkg-new` (baseline) | 7491.30 | 1.000 | 54.18 | 1.000 |
| merged, `INT_MM=0` (fp16 path) | 8081.90 | 1.079 | 71.45 | 1.319 |
| merged, int best-speed (R32, qkv + o + gate+up A8) | 8800.21 | 1.175 | 71.35 | 1.317 |
| merged, int full-coverage | 8144.31 | 1.087 | 70.80 | 1.307 |
| fuse-only `final17` (reference) | 8156.35 | 1.089 | 70.95 | 1.310 |

Per round prefill (best-speed arm): 8596.01 / 8891.33 / 8800.21, every round above every other arm's median.

- Recommended int configuration (best-speed):
  `GGML_HEXAGON_INT_MM=3 GGML_HEXAGON_INT_ROT=32 GGML_HEXAGON_INT_ROT_MASK=4 GGML_HEXAGON_INT_MM_MASK=7
  GGML_HEXAGON_INT_MM_LAYERS=7ffffffff GGML_HEXAGON_INT_MM_REDUCE=2 GGML_HEXAGON_INT_MM_REDUCE_BIG=4
  GGML_HEXAGON_INT_SMOOTH=<path>/smooth.bin GGML_HEXAGON_VMEM=5G`. Here MASK=7 is qkv + o-proj + gate+up as A8, and
  `ROT_MASK=4` applies R32 to the gate+up class (bit0 qkv, bit1 o-proj, bit2 gate+up). `smooth.bin` must still travel with
  the model and the exact fp16 fallback when it is absent still applies (see the shipping requirement above).
- Full-coverage arm (an option, not the recommendation):
  `GGML_HEXAGON_INT_MM=3 GGML_HEXAGON_INT_ROT=32 GGML_HEXAGON_INT_ROT_MASK=7 GGML_HEXAGON_INT_MM_MASK=cf
  GGML_HEXAGON_INT_MM_MASK16=c8 GGML_HEXAGON_INT_MM_LAYERS=7ffffffff GGML_HEXAGON_INT_MM_REDUCE=2
  GGML_HEXAGON_INT_MM_REDUCE_BIG=4 GGML_HEXAGON_INT_SMOOTH=<path>/smooth.bin GGML_HEXAGON_VMEM=5G` (adds down, k=12288 and
  the PLE gate / proj as A16 two-pass).
- R32 lifts pp512 to 8800.21 = +17.5% over `pkg-new` (+8.9% over the same-session `INT_MM=0` arm). The earlier config A
  (gate+up smoothed, no rotation) was +14.0% in run 7 and +9.0% in run 8, so R32 on qkv + o + gate+up is the first merged
  state with an end-to-end prefill gain well outside the run-to-run noise.
- The fp16-path changes (nodirty subset, FA head scan, GELU+MUL stage cap) did not produce a resolvable end-to-end change
  at 3 rounds: `INT_MM=0` 8081.90 against 8099.76 (run 7) and 8017.04 (run 8), and `final17` reads 8156.35 in this
  session, inside the arms' spread (stddev up to 8%). The expected 1 ms per ubatch is 1.7% of a 59-60 ms ubatch, below the
  A/B resolution. Nothing regressed.

### Gates (RUN 9)

- `INT_MM=0` bit-identity (hard stop): PARITY_IDENTICAL, `llama-perplexity` PPL exactly 134.3955 +/- 6.77407, 5-output
  gate SAME on all five (g0 g1 g2 s2 s0) against the fuse package.
- Int best-speed against the fp16 logits (20 chunks, c512): KLD 0.006124 +/- 0.000347, top-p 96.941%, PPL 133.1222
  (-0.94% vs 134.3955, informational), greedy identical.
- Int full-coverage: KLD 0.006942 +/- 0.001158, top-p 96.569%, PPL 132.1043 (-1.70%), greedy identical.
- Int PPL falls rather than rises, so the one-sided PPL rule holds; the KLD / top-p gate (0.008 / 96.0) governs.
- `INT_MM=1` guard arm: one guard warning (first tensor `blk.0.ffn_gate.weight`), as in the earlier runs.
- Static compile matrix: 31 objects / 0 errors on all 8 cells (SDK 6.6.0.0 and 6.7.0.0 x v73 v75 v79 v81); host
  `clang++ -fsyntax-only` on `ggml-hexagon.cpp` 0 errors. The measured package was built before the one-line
  `std::max<uint64_t>` spelling fix in the nodirty block (`std::max(..., 0ULL)` -> `std::max<uint64_t>(..., (uint64_t) 0)`);
  it is a type-only change (uint64_t is unsigned long long on the device compiler) and the results are unaffected. The
  committed tree has the fix and host clang++ is now clean. v73 / v75 / v79 are compile-only.

### KPI distance (RUN 9)

- Decode target 70 t/s: MET in all four arms (71.45 `INT_MM=0`, 71.35 best-speed, 70.95 `final17`, 70.80 full-coverage).
  The best-speed arm clears 70 by +1.35 t/s with all three rounds at 70.50 or above; the full-coverage arm has one round
  at 69.80.
- Prefill target 10000 t/s: NOT MET. Best arm (best-speed) 8800.21 is 88.0% of it, 1200 t/s short (about 58.2 ms per
  ubatch against 51.2 ms needed); its best single round is 8891.33.
- Both targets with one configuration: NOT MET (best-speed: decode 71.35, prefill 8800.21); decode is met by every arm.

### Two honest notes

- (a) The full-coverage arm buys down and PLE coverage (more int8 matmuls) at -7.5% prefill versus best-speed (8144.31
  vs 8800.21), which is essentially fp16-level speed (`INT_MM=0` 8081.90, `final17` 8156.35). It is an option for a
  broader int8 footprint, not a speed configuration.
- (b) Corrected accounting (hmx-int-log.md M123). The traced `prompt512.txt` is 721 tokens, i.e. two ubatches (512 + 209
  rows); the earlier 46.9 ms fp16 matmul total and its int8 saving summed both. The 512-row ubatch has 30.6 ms of fp16
  matmul. A perfect int8 version of every remaining fp16 matmul tops out around 62 ms per 512-row ubatch (about 8200
  t/s) on this HMX, so the 10000 t/s target cannot come from int8 matmuls alone: it needs the fuse side (non-matmul
  time, batch and DSP overhead) and int8 together. With the best-speed arm at 8800 and the fp16 state at about 8100, the
  remaining 1200 t/s (about 7 ms per ubatch) has to be found in the non-matmul time.

## RUN 10: KQ-mask fill fast path (host timeline)

RUN 10 adds one source change on top of the RUN 9 state: a fast path in `src/llama-kv-cache.cpp` for the KQ-mask fill
(for the cells that can still change, the cell positions are cached per sequence and only the position tests run, so
the mask row is written in one tight loop; alibi and 2D positions keep the old path). The tree is the one the RUN 10
package was built from, including the `std::max<uint64_t>` spelling of the nodirty block that the RUN 9 package predated.
Source of the numbers: analysis/merge-log.md "DEVICE RUN 10", analysis/fuse-log.md (host-cost round). Same model, device
and command lines as before; medians of 3 clean interleaved rounds (decode round 1 was flagged contaminated by the
health rule, excluded and repeated, never averaged; session otherwise normal, pkg-new decode 54.16-54.24):

| Arm | prefill pp512 (t/s) | ratio vs `pkg-new` | decode tg128 (t/s) | ratio vs `pkg-new` |
|---|---|---|---|---|
| `pkg-new` (baseline) | 7299.55 | 1.000 | 54.20 | 1.000 |
| merged, `INT_MM=0` (fp16 path) | 8144.79 | 1.116 | 70.93 | 1.309 |
| merged, int best-speed | 9004.30 | 1.234 | 71.56 | 1.320 |
| merged, int full-coverage | 8210.40 | 1.125 | 71.47 | 1.319 |

- The int best-speed arm has a second, same-session prefill sample from the decode runs: 8567.32 (vs 8212.78 for
  `INT_MM=0` there). The two samples of the same arm differ by 5%, so its level in this session is 8.6-9.0k t/s; do not
  read 9004 as a step over RUN 9's 8800.21 (inside the spread of the arm, round stddev 0.9-6.3%).
- Gates unchanged from RUN 9: `INT_MM=0` bit-identity PASS (PARITY_IDENTICAL, PPL exactly 134.3955 +/- 6.77407, 5/5
  SAME). Best-speed KLD 0.006124, top-p 96.941%, PPL 133.1222. Full-coverage KLD 0.006942, top-p 96.569%, PPL 132.1043.
  Static compile matrix 31 objects / 0 errors on 8 cells; host `clang++` syntax check 0 errors on `ggml-hexagon.cpp` and
  `src/llama-kv-cache.cpp`.
- KQ-mask fast path, host timeline (two instrumented builds of the same tree, `INT_MM=0`, `-p 512 -n 0 -r 5`, 8 runs per
  build): `set_inputs` of the KQ masks 645.5 -> 241.5 us per run, i.e. -0.40 ms of host timeline (all llama
  `set_inputs` 0.657 -> 0.251 ms). Byte-identical masks: 58/58 in-process fill compares and 162 layout cases, 0
  mismatches, plus unchanged end-to-end outputs on the real package. This is a host-timeline win, NOT a throughput claim:
  the total host time before the batch write (6.9-7.0 ms with the instrumented stderr output, 2-3 ms spread between
  runs) and the device wait (56.9 ms) do not resolve 0.4 ms, and the pp512 differences are inside the arms' spread.
- `GGML_HEXAGON_HOSTBUF=1` was measured and left off by default: it removes about 1.1 ms of input copies on the host but
  loses three norm-chain fusions through an allocator / overlap interaction (`ggml_hexagon_tensors_overlap` on the new
  intermediate layout), netting only +0.3..0.9%. Making it a win would need a separate allocator task.
- KPI: decode > 70 MET in all merged arms (70.93 `INT_MM=0`, 71.56 best-speed with all rounds >= 70.34, 71.47
  full-coverage). Prefill > 10000 NOT MET: best arm 9004.30 = 90.0% of it, 996 t/s short (honest band 8.6-9.0k); the
  remaining gap has no identified lever above 1 ms (the host headroom left is about 0.4 ms, now taken, and the matmul
  accounting is in the RUN 9 notes above).

## RUN 11: host-buffer inputs become the default

RUN 11 changes exactly one line on top of the RUN 10 state: `opt_hostbuf` in `ggml-hexagon.cpp` goes from `false` to
`true`, so `GGML_HEXAGON_HOSTBUF=1` is the default (CPU-side inputs live in a host-type buffer and the per-run input
copies disappear); `GGML_HEXAGON_HOSTBUF=0` restores the old behaviour. This supersedes the "left off by default" bullet
of RUN 10. The int8 path, the fp16 path and the KQ-mask fast path are otherwise unchanged. Source of the numbers:
analysis/merge-log.md "DEVICE RUN 11", analysis/fuse-log.md (HOSTBUF entries).

- Host timeline (single-run timelines, median of 14 runs): host time before the DSP batch write 3.78 -> 2.66 ms, device
  wait unchanged (57.2 -> 57.3 ms), run time 61.3 -> 60.1 ms.
- Corrected story: with HOSTBUF=1 the allocator lays out the intermediates differently and three norm-chain fusions are
  lost (`ggml_hexagon_tensors_overlap` fires on the new layout), leaving 16 un-fused ops (534 ops instead of the 518
  target). The RUN 10 note said this "eats the win"; the per-op profile does not show that. The 16 un-fused ops cost only
  about 0.4 ms, and the DSP op sum is LOWER with HOSTBUF=1 (57.05 ms vs 57.61 ms). The 3- to 5-round A/Bs (+0.3..0.9%)
  could not resolve this because their spread is 2-6%. A backend-local allocator fix was not found (root cause is
  best-fit reuse of freed 524288-byte PLE holes by 3 MB tensors in the shared `ggml_dyn_tallocr`, a policy change in
  shared ggml code), so the 16 ops stay un-fused.
- 80-sample alternating test (4 rounds x 3 arms x 2 settings, each cell one `llama-bench -p 512 -n 0 -r 20`, 10 s gaps,
  20 samples per cell, 80 samples per arm and setting; one sample is one timed pp512 run):

| Arm | HOSTBUF=0 median (t/s) | default (=1) median (t/s) | median delta | mean delta (z) | ms/run |
|---|---|---|---|---|---|
| null control: `pkg-new` (default already 0, so the same code) | 7589.0 | 7591.1 | +0.03% | +0.32% (z 0.6) | +0.02 |
| fuse: merged `INT_MM=0` | 8411.3 | 8487.5 | +0.91% | +0.97% (z 2.1) | +0.55 |
| int best-speed | 8979.8 | 9101.9 | +1.36% | +0.70% (z 1.8) | +0.76 |

  The null control shows the floor of the test (identical code, +0.03% median, z 0.6). Both real arms are positive, but
  each is only at the edge of significance (z about 2). The effect is real but small: 0.5-0.8 ms per run, 1-1.4%, below
  what a 3-round A/B resolves. The fuse side's own 80-sample test measured +1.43% / 1.2 ms for the fp16 arm; that full size
  is not confirmed here.

RUN 11 arm table (3 clean rounds, medians; no contaminated round, session normal, pkg-new decode 54.16-54.40):

| Arm | prefill pp512 (t/s) | ratio vs `pkg-new` | decode tg128 (t/s) | ratio vs `pkg-new` |
|---|---|---|---|---|
| `pkg-new` (baseline) | 7503.70 | 1.000 | 54.18 | 1.000 |
| merged, `INT_MM=0` (fp16 path) | 8310.35 | 1.108 | 71.55 | 1.321 |
| merged, int best-speed | 9024.11 | 1.203 | 71.33 | 1.317 |
| merged, int full-coverage | 8295.78 | 1.106 | 71.26 | 1.315 |

- The 3-round table does not resolve the hostbuf effect (ratios vs `pkg-new` are slightly lower than in RUN 10 because
  `pkg-new` was 2.8% higher this session; arm stddev is up to 8.9%). The in-run pp512 sample of the best-speed arm in the
  decode runs is 8615.28, so its level with hostbuf in this session is 8.6-9.1k t/s.
- Gates with the new default: `INT_MM=0` bit-identity PASS (PARITY_IDENTICAL, PPL exactly 134.3955 +/- 6.77407, 5/5
  SAME). Best-speed KLD 0.006124, top-p 96.941%, PPL 133.1222. Full-coverage KLD 0.006942, top-p 96.569%, PPL 132.1043.
  `INT_MM=1` guard: one warning, as before. Stress (HOSTBUF 0 vs 1, 4 prompts x greedy / temperature 1.5): 8/8
  byte-identical (fuse side, on its own tree). Static compile matrix 31 objects / 0 errors on 8 cells; host `clang++`
  syntax check 0 errors.
- KPI: decode > 70 MET in all merged arms (71.55 `INT_MM=0` with one round at 69.77, 71.33 best-speed with every round >=
  71.02, 71.26 full-coverage). Prefill > 10000 NOT MET: best arm 9024.11 (median of the prefill phase; 80-sample median
  9101.9) is 90.2% of the target, 976 t/s short; the arm is at about 56 ms per ubatch at 9.1k against the 51.2 ms needed.
  The one named lever not in this state, the int side's fused quantizer (about 1 ms estimated), has not been built; nothing
  else identified is above 1 ms.

## RUN 12: FA O-store via DMA, PLE int8 model_proj and PLE chunk caps

RUN 12 adds two gated deltas on top of the RUN 11 state. Source of the numbers: analysis/merge-log.md "DEVICE RUN 12",
analysis/hmx-int-log.md (M129 / M130), analysis/fuse-log.md. Same model, device and command lines as before.

- (A) Flash attention writes its output O through DMA from VTCM instead of cached stores (`htp/flash-attn-ops.c`, kernel
  flag 2). Default on; `GGML_HEXAGON_FA_OSTORE=0` disables it. Bit-identical on the fp16 path (stress 8/8 on the fuse
  side).
- (B) PLE delta: a BF16 -> int8 packer for the per-layer `model_proj` (`hex-int8-pack.h`, opt-in through the class bit
  0x20 of `GGML_HEXAGON_INT_MM_MASK`, so `0x27` = qkv + o-proj + gate+up + model_proj), and n-chunk caps for the PLE
  matmuls in `htp/matmul-ops.c`: `GGML_HEXAGON_PLE_NC=512` (proj, n = 1536) and `GGML_HEXAGON_PLE_NCG=64` (inp_gate,
  n = 256). The caps are on by default; 0 disables them. Per-op (int owner): `model_proj` 962 -> 412 us per 512 rows, PLE
  class 2.12 -> 1.72 ms.

Merged tree: RUN 11 state plus both deltas (`INT_MM=0` stays the default and stays bit-identical). Medians of 3 clean
interleaved rounds, arm order rotated per round, 30 s gaps, a health check before every run, nothing contaminated or
excluded (pkg-new decode 54.13-54.50):

| Arm | prefill pp512 (t/s) | ratio vs `pkg-new` | decode tg128 (t/s) | ratio vs `pkg-new` |
|---|---|---|---|---|
| `pkg-new` (baseline) | 7624.97 | 1.000 | 54.26 | 1.000 |
| merged12, `INT_MM=0` (fp16 path) | 8599.71 | 1.128 | 71.72 | 1.322 |
| merged12, int best-speed (`MASK=0x27`) | 9078.18 | 1.191 | 70.19 | 1.294 |
| merged12, int full-coverage | 8555.14 | 1.122 | 70.68 | 1.303 |

- Best-speed env line: `GGML_HEXAGON_INT_MM=3 GGML_HEXAGON_INT_ROT=32 GGML_HEXAGON_INT_ROT_MASK=4
  GGML_HEXAGON_INT_MM_MASK=0x27 GGML_HEXAGON_INT_MM_LAYERS=7ffffffff GGML_HEXAGON_INT_MM_REDUCE=2
  GGML_HEXAGON_INT_MM_REDUCE_BIG=4 GGML_HEXAGON_INT_SMOOTH=<path>/smooth.bin GGML_HEXAGON_VMEM=5G`, with the PLE caps at
  their defaults (`PLE_NC=512`, `PLE_NCG=64`). The RUN 9 line with `MASK=7` is the same configuration without the
  `model_proj` class. The full-coverage line is unchanged (`MASK=cf`, `MASK16=c8`, `ROT_MASK=7`; it does not include the
  `model_proj` bit).
- 80-sample attribution (4 rounds x 7 cells, each cell one `llama-bench -p 512 -n 0 -r 20`, 10 s gaps, 80 samples per
  cell; one sample is one timed pp512 run):

| Comparison | median delta | mean delta (z) | ms/run (median) |
|---|---|---|---|
| FA O-store, fp16 (`INT_MM=0`, on vs `FA_OSTORE=0`) | +1.08% | +1.30% (z 2.8), 4/4 rounds positive | +0.64 |
| FA O-store, int best-speed (on vs `FA_OSTORE=0`) | +2.10% | +1.35% (z 2.1) | +1.15 |
| PLE delta (caps + `model_proj` int8), int best-speed | +1.90% | +1.45% (z 2.4), 4/4 rounds positive | +1.04 |
| FA + PLE caps, fp16 (merged12 `INT_MM=0` vs the RUN 11 build) | +2.05% | +2.17% (z 4.7), 4/4 rounds positive | +1.21 |
| null control (same function, two builds) | -0.65% | -0.37% (z -0.6) | -0.36 |

  The build-to-build floor is about 0.4% (the null control reads one build higher than the other for identical
  function), so the cross-build FA comparison (+0.97% mean, z 1.4) is the weakest and the same-build toggles are the
  evidence. Each delta is about +1% (0.6-1.2 ms per run), significant at z of about 2-3 on its own; the fp16 sum is the
  only comparison above z 4. The best arm (int best-speed, `MASK=0x27`, FA on) has an 80-sample median of 9380.6 t/s
  (trimmed mean 9321.6, stddev 408.9); its in-run decode-phase sample is 9279.55 and the 3-round prefill median 9078.18,
  so the honest band is 9.1-9.4k t/s.
- Decode check (the main table has the best-speed arm at 70.19 with round 1 at 69.82). Extra check, 4 clean rounds,
  health before every run: tg128 medians `INT_MM=0` 71.73, best-speed `MASK=7` 71.04, best-speed `MASK=0x27` 71.15
  (rounds 71.20 / 71.09 / 71.51 / 70.20), PLE tree without the FA change 70.82. The `model_proj` int8 does not cost decode
  (`0x27` vs `7`: +0.11, inside noise); the 69.8-70.7 values of the main table are the lower tail of a 70.2-71.5 band
  that the int arms occupy in this session.

### Gates (RUN 12)

- `INT_MM=0` bit-identity (hard stop): PARITY_IDENTICAL, `llama-perplexity` PPL exactly 134.3955 +/- 6.77407, 5-output
  gate SAME on all five (g0 g1 g2 s2 s0). The FA O-store, the PLE caps (default on) and the packer do not change fp16
  behaviour.
- Int best-speed (`MASK=0x27`) against the fp16 logits (20 chunks, c512): KLD 0.006692 +/- 0.000328, top-p 96.647%, PPL
  132.5813 (-1.34%), greedy identical. The step from the RUN 11 best-speed numbers (KLD 0.006124, top-p 96.941%) is the
  price of the `model_proj` class; the gate (0.008 / 96.0) holds. Matches the int owner's M130 numbers exactly.
- Int full-coverage: KLD 0.006942, top-p 96.569%, PPL 132.1043 (-1.70%), greedy identical; unchanged from RUNs 9-11.
- `INT_MM=1` guard: one warning, as before.
- Static compile matrix 31 objects / 0 errors on all 8 cells (SDK 6.6.0.0 and 6.7.0.0 x v73 v75 v79 v81); host `clang++`
  syntax check on `ggml-hexagon.cpp` 0 errors. v73 / v75 / v79 are compile-only.

### KPI distance (RUN 12)

- Decode target 70 t/s: MET in all three merged arms of the main table (71.72 / 70.19 / 70.68). The best-speed margin is
  thin (+0.19 in the main table, rounds 69.82 and 70.19 at or below the line); the 4-round extra check gives 71.15 median.
- Prefill target 10000 t/s: NOT MET. Best arm: 9078.18 (3-round median), 9256.02 (best round); 80-sample median 9380.6 =
  93.8% of the target, about 3.4 ms per ubatch short of the 51.2 ms needed (the arm is at about 54.6 ms at 9.38k).
- Both targets with one configuration: NOT MET (best-speed: decode 70.2-71.2, prefill 9.1-9.4k).

## RUN 13: interleaved GELU for the PLE path

RUN 13 adds a two-file, bit-identical change on top of the RUN 12 state (`htp/hvx-sigmoid.h`, `htp/act-ops.c`). Source of
the numbers: analysis/merge-log.md "DEVICE RUN 13" and the int owner's per-op measurements. Same model, device and
command lines as before.

- Change: `hvx_vec_gelu_f32_xn` (`HVX_GELU_X` = 4) pushes four independent vectors through exactly the intrinsic
  sequence of `hvx_vec_gelu_f32`. The GELU body is latency-bound (one long dependent chain per vector), so interleaving
  four chains lets them overlap. It is used for the 4-aligned part of the `gelu_mul` rows (the remainder runs through the
  old single-vector loop). The activation pipeline depth constants become `ACT_STAGES` 8 (a thread's block is
  `rows_per_thread / ACT_STAGES`, was `/ 4`) and `ACT_MINBLK` 4 (minimum block rows, was 8).
- Why it is bit-identical: every lane goes through the same intrinsic sequence with the same rounding; only the
  scheduling of independent vectors changes, and the block size only changes how rows are grouped, not the per-element
  arithmetic. This is backed by a full 20-chunk logit byte-compare on both the fp16 and the int paths (int owner) and by
  the gates below.
- Per op (int owner): GELU+MUL 85.6 -> 55.4 us at 512 rows and 43.9 -> 32.4 us at 209 rows, about -0.42 ms per ubatch
  (about 0.75% of the ubatch).

DEVICE RUN 13 arm table (3 clean rounds, medians; arm order rotated per round, 30 s gaps, a health check before every
run, no contaminated A/B round; the session level was low, `pkg-new` prefill 7.35-7.42k, about 2% below RUN 12):

| Arm | prefill pp512 (t/s) | ratio vs `pkg-new` | decode tg128 (t/s) | ratio vs `pkg-new` |
|---|---|---|---|---|
| `pkg-new` (baseline) | 7387.87 | 1.000 | 54.25 | 1.000 |
| merged13, `INT_MM=0` (fp16 path) | 8613.57 | 1.166 | 71.27 | 1.314 |
| merged13, int best-speed (`MASK=0x27`) | 9389.12 | 1.271 | 70.79 | 1.305 |
| merged13, int full-coverage | 8720.57 | 1.180 | 70.39 | 1.298 |

- The best-speed arm: rounds 8974.07 (stddev 13.1% inside the run) / 9389.12 / 9508.44, so its best round is 9508. The
  ratios vs `pkg-new` are inflated by the low `pkg-new` level of this session (7388 vs 7625 in RUN 12); on absolute
  numbers the 3-round medians are +3.4% (best-speed), +0.2% (fp16) and +1.9% (full-coverage) over RUN 12, which is more
  than the 0.75% the change can explain, so the 3-round table is not an attribution of this change. The in-run pp512
  sample of the best-speed arm in the decode runs is 8853.51, so the honest band is 8.9-9.5k t/s, central about 9.4k.
- 80-sample alternating test, merged13 vs merged12 (`llama-bench -p 512 -n 0 -r 10` with per-sample times, pair order
  swapped every round, 7 s gaps, health check at every round start):

| Comparison | median (t/s) | median delta | mean delta (z) | ms/run | round pairs |
|---|---|---|---|---|---|
| int best-speed, merged13 vs merged12 (80 / 80 samples) | 9462.6 vs 9386.1 | +0.81% | +0.95% (z 1.3) | +0.44 | 6 of 8 positive |
| same without one round flagged contaminated (70 / 70) | - | +0.88% | +0.71% (z 0.9) | +0.47 | 5 of 7 positive |
| fp16 `INT_MM=0`, merged13 vs merged12 (40 / 40 samples) | 8704.5 vs 8618.5 | +1.00% | +0.70% (z 0.9) | +0.59 | 4 of 4 positive |

  Both pairs point in the predicted direction and size (+0.8..1.0% median = 0.44-0.59 ms per ubatch, the per-op figure is
  -0.42 ms), and 10 of 12 round pairs favour merged13. The unpaired z is only 0.9-1.3 (per-sample stddev 250-460 t/s
  against an effect of 60-90 t/s), so the test confirms consistency; it does not resolve the 0.8% at 2 sigma. One round
  start (health pp128 2784.85 = 68.2% of the best seen) fell under the 70% line and is flagged; the result is given with
  and without it and does not change. Not a regression anywhere.

### Gates (RUN 13)

- `INT_MM=0` bit-identity (hard stop): PARITY_IDENTICAL, `llama-perplexity` PPL exactly 134.3955 +/- 6.77407, 5-output
  gate SAME on all five (g0 g1 g2 s2 s0).
- Int best-speed (`MASK=0x27`): KLD 0.006692 +/- 0.000328, top-p 96.647%, PPL 132.5813 (-1.34%), greedy identical; the
  same numbers to the last digit as RUN 12, so the GELU change does not move a logit statistic.
- Int full-coverage: KLD 0.006942, top-p 96.569%, PPL 132.1043 (-1.70%), greedy identical; unchanged from RUNs 9-12.
- `INT_MM=1` guard: one warning, as before.
- Static compile matrix 31 objects / 0 errors on all 8 cells (SDK 6.6.0.0 and 6.7.0.0 x v73 v75 v79 v81); host `clang++`
  syntax check 0 errors. v73 / v75 / v79 are compile-only.

### KPI distance (RUN 13)

- Decode target 70 t/s: MET in all three merged arms (71.27 `INT_MM=0`, 70.79 best-speed, 70.39 full-coverage). The
  best-speed margin is +0.79 by median (rounds 70.34-71.28), thin as before; full-coverage has one round at 69.77.
- Prefill target 10000 t/s: NOT MET. Best arm (best-speed): 9389.12 (3-round median), 9508.44 (best round); 80-sample
  median 9462.6 = 94.6% of the target, 537 t/s short (the 3-round median is 611 short, 93.9%). The arm is at about 54.1
  ms per ubatch at 9.46k against the 51.2 ms needed.
- Both targets with one configuration: NOT MET (best-speed: decode 70.4-71.3, prefill about 9.4k).

## RUN 14: interleaved ROPE and FA softmax row pairs

RUN 14 adds a three-file, bit-identical change on top of the RUN 13 state (`htp/flash-attn-ops.c`, `htp/hvx-norm.h`,
`htp/rope-ops.c`). Source of the numbers: analysis/merge-log.md "DEVICE RUN 14" and the owner's per-op measurements.
Same model, device and command lines as before.

- ROPE and the fused RMS_NORM+MUL+ROPE: the sin/cos blocks are processed 4 at a time, and the new
  `hvx_fast_rms_norm_mul_f32_xn` normalizes 4 rows at a time in the fused spad loops (`rope-ops.c`, `hvx-norm.h`). The
  tails keep the old single-row paths.
- FA softmax: `fa_sm_pairs_kv128` / `fa_sm_pairs_kv64` run two row pairs per pass as straight-line code
  (`flash-attn-ops.c`). They are taken only for the plain-mask case (mask present, no softcap, no alibi), `kv_rows` of
  128 or 64 with `Bc >= kv_rows`, and whole row pairs; every other case, and the tail, keeps the old path.
- Why it is bit-identical: each row and each lane goes through the same arithmetic in the same order; only independent
  rows / vectors are interleaved to hide latency. This is backed by a full 20-chunk logit byte-compare on both the fp16 and
  the int paths (owner) and by the gates below.
- Per op (owner): FA_EXT 386.5 -> 338.3 us on the 15 big ops (-0.72 ms per ubatch) and RMS_NORM_MUL_ROPE 116.5 ->
  104.6 us (-0.36 ms), about -1.1 ms per ubatch together (about 2.0% of the ubatch).

DEVICE RUN 14 arm table (3 clean rounds, medians; arm order rotated per round, 30 s gaps, a health check before every
run, no contaminated A/B round; `pkg-new` prefill 6.9-7.6k, a low sample of 6922.91 inside round 2):

| Arm | prefill pp512 (t/s) | ratio vs `pkg-new` | decode tg128 (t/s) | ratio vs `pkg-new` |
|---|---|---|---|---|
| `pkg-new` (baseline) | 7519.36 | 1.000 | 54.24 | 1.000 |
| merged14, `INT_MM=0` (fp16 path) | 8717.14 | 1.159 | 70.55 | 1.301 |
| merged14, int best-speed (`MASK=0x27`) | 9524.93 | 1.267 | 70.44 | 1.299 |
| merged14, int full-coverage | 8801.36 | 1.170 | 70.45 | 1.299 |

- The best-speed arm: rounds 9524.93 / 9620.43 / 9409.10, best round 9620. Against RUN 13 the absolute medians are +1.2%
  (`INT_MM=0`), +1.4% (best-speed) and +0.9% (full-coverage) while `pkg-new` is +1.8%, so the ratios vs `pkg-new` are
  flat; the 3-round table is not an attribution of this change. The in-run pp512 sample of the best-speed arm in the
  decode runs is 9601.48.
- 80-sample alternating test, merged14 vs merged13 (`llama-bench -p 512 -n 0 -r 10` with per-sample times, pair order
  swapped every round, 7 s gaps, health at every round start, no round flagged):

| Comparison | median (t/s) | median delta | mean delta (z) | ms/run | round pairs |
|---|---|---|---|---|---|
| int best-speed, merged14 vs merged13 (80 / 80) | 9641.6 vs 9408.7 | +2.48% | +4.42% (z 4.2) | +1.31 | 8 of 8 positive |
| same without the one bad merged13 round (70 / 70) | - | +2.06% | +2.66% (z 2.6) | +1.10 | 7 of 7 positive |
| fp16 `INT_MM=0`, merged14 vs merged13 (40 / 40) | 8873.9 vs 8720.0 | +1.77% | +1.65% (z 1.6) | +1.02 | 4 of 4 positive |

  The +4.42% mean is inflated by one bad merged13 round (cell mean 8177, stddev 754.8 t/s, with a clean health check at
  its start). Use the conservative range: the effect is +2.1% to +2.5% (1.1-1.3 ms per ubatch) on the int best-speed arm,
  and +1.8% (median) on fp16 with only 40 samples. The conservative end-to-end figure (+2.06%, 1.10 ms) matches the
  owner's op-level estimate of -1.1 ms (about 2.0%). The GELU change of RUN 13 is not part of this comparison (both
  builds contain it).

### Gates (RUN 14)

- `INT_MM=0` bit-identity (hard stop): PARITY_IDENTICAL, `llama-perplexity` PPL exactly 134.3955 +/- 6.77407, 5-output
  gate SAME on all five (g0 g1 g2 s2 s0).
- Int best-speed (`MASK=0x27`): KLD 0.006692 +/- 0.000328, top-p 96.647%, PPL 132.5813 (-1.34%), greedy identical; the
  same numbers to the last digit as RUNs 12 and 13.
- Int full-coverage: KLD 0.006942, top-p 96.569%, PPL 132.1043 (-1.70%), greedy identical; unchanged from RUNs 9-13.
- `INT_MM=1` guard: one warning, as before.
- Static compile matrix 31 objects / 0 errors on all 8 cells (SDK 6.6.0.0 and 6.7.0.0 x v73 v75 v79 v81); host `clang++`
  syntax check 0 errors (the host file is unchanged by this patch). v73 / v75 / v79 are compile-only.

### Decode watch item (RUN 14)

- Decode target 70 t/s is MET by median in all three merged arms (70.55 `INT_MM=0`, 70.44 best-speed, 70.45
  full-coverage), but the margin is only +0.4..+0.6 t/s and three rounds are below 70: best-speed round 2 = 68.83,
  full-coverage round 2 = 69.34, `INT_MM=0` round 1 = 69.85. Decode is about 0.5 t/s lower than in RUNs 12 and 13
  (71.2-71.7 for `INT_MM=0`) while `pkg-new` decode is unchanged (54.24).
- It is not resolved whether this is session level (all arms moved together in round 2) or a small decode cost: the patch
  touches FA / ROPE / RMS paths that decode also uses, but with one row, where the old tail paths run. It is recorded as a
  watch item, not a gate failure: no gate failed and no decode regression has been demonstrated.

### KPI distance (RUN 14)

- Decode target 70 t/s: MET by median, thin, see the watch item above.
- Prefill target 10000 t/s: NOT MET. Best arm (best-speed): 9524.93 (3-round median), 9620.43 (best round); 80-sample
  median 9641.6 = 96.4% of the target, 358 t/s short (the 3-round median is 475 short, 95.2%). The arm is at about 53.1 ms
  per ubatch at 9.64k against the 51.2 ms needed, 1.9 ms away. Honest band 9.5-9.65k.
- Both targets with one configuration: NOT MET (best-speed: decode 70.4, prefill 9.5-9.65k).

## RUN 15: host batch path and fused qkv quantize

RUN 15 adds five changes on top of the RUN 14 state: four host-side changes ("host stack") and one DSP change
(RMS_NORM_MUL_Q8, "q8f"). Source of the numbers: analysis/merge-log.md "HOST STACK" and "DEVICE RUN 15". Same model,
device and command lines as before. All five are bit-identical by the gates below.

1. CPU gather (`ggml-cpu.c`): GET_ROWS is planned with up to 8 tasks when it gathers at least 64 rows and at least 8 MB
   of output from a quantized table (the PLE table gather). Every other GET_ROWS stays single-task. It was single-task
   by design in ggml, so the 18.35 MB q4_0 -> f32 gather ran on one core. The kernel is already row-range parallel, so only
   the task count changed; an offline check of the same dequantize loop gave byte-identical output with 1 / 2 / 4 / 8
   threads. Scheduler split 0 on the device: 854 -> 460 us.
2. Descriptor cache (`ggml-hexagon.cpp`): the finished op batch of a reused graph (534 ops, 824 tensors, 7 buffers,
   after buffer sort and nodirty marking) is replayed with three memcpy instead of being rebuilt. The key is a
   fingerprint of the graph uid plus every node's opcode, op_params and kernel_params (first 16 words) and every
   tensor's data address, type, ne / nb, shared-buffer fd / base and extra flags, so any change of address, shape or
   buffer forces a normal build and a new capture. It is off under profiling, with multiple devices or peers, and when
   the batch already holds ops. `GGML_HEXAGON_DESC_CACHE=0` disables it; `GGML_HEXAGON_DESC_VERIFY=1` rebuilds normally
   every rep and memcmp's the three descriptor arrays against the snapshot. The replay is byte-identical to a rebuild
   (DESC_VERIFY_OK, n_ops 534, n_tens 824, n_bufs 7).
3. KV-cache `prepare()` (`src/llama-kv-cache.cpp`): when a single ubatch lands in empty cells the apply / restore round
   trip is skipped, because for that case it is exact and unobserved (no cell is removed). Anything else (an occupied
   cell, SWA overwrite, more than one ubatch) takes the generic path. Decode entry to init_batch 440 -> 182 us.
4. KQ mask (`src/llama-kv-cache.cpp`): for consecutive cells with strictly ascending positions and causal or standard
   SWA masks, a mask row is three fills around two bisections instead of a per-cell loop; chunked and symmetric SWA,
   ALiBi, M-RoPE and non-ascending cells fall through to the per-cell code. Verified against the per-cell predicate on
   400000 random cases (0 mismatches). `set_inputs` 241 -> 81 us. This builds on the KQ-mask fast path of RUN 10.
5. RMS_NORM_MUL_Q8 (DSP, `htp/unary-ops.c`, `htp/matmul-ops.c`, `htp/hmx-mm-kernels-int8.h`, `htp/main.c` and headers):
   with `GGML_HEXAGON_NORM_Q8=1` the attn_norm RMS_NORM+MUL that feeds a 3-weight qkv MUL_MAT_NX writes the int8 payload
   and the f32 row scales itself, with the same smoothing multiply, row max and rounding as the matmul's own activation
   quantize, and the qkv matmul skips its A_QUANT. The host only does this when it proves: 3 uses and all of them
   MUL_MAT src1, `n_weights` 3, `int_mm` 3, `int8_split` 0, class qkv, no A16, m 64..1024 and a multiple of 32, k a
   multiple of 64, single device. Every other DSP route for a flagged op returns INVAL_PARAMS. **Default off.**

The `src/` part is now a single patch: `src/llama-kv-cache.cpp` is the only `src/` change since the RUN 10 patch
(analysis/merged10-src.patch), and items 3 and 4 build on it, so the two are merged into one patch against the base
(analysis/merged15-src.patch, 117 lines).

Host timeline before the DSP write (probe build, int best-speed arm, 7 reps x 2 runs, medians, us):

| Stage | RUN 14 state | host stack | delta |
|---|---|---|---|
| decode entry to init_batch done (item 3) | 440 | 182 | -258 |
| output_reserve + apply + graph_params | 62 | 76 | +14 (noise) |
| `set_inputs` (item 4) | 240 | 81 | -160 |
| scheduler split 0 CPU gather (item 1) | 854 | 460 | -394 |
| hexagon enqueue of 534 ops (item 2) | 392 | 76 | -316 |
| logits `get_async` entry to `flush_batch` (item 2) | 102 | 2 | -100 |
| `flush_batch` (sort / nodirty / push, item 2) | 132 | 70 | -62 |
| host before the DSP write | 2232 | 991 | -1240 |

  With `GGML_HEXAGON_DESC_CACHE=0` on the same build the host window is 1344 us, so the cache is worth 0.31-0.39 ms of
  it. The DSP op sum is unchanged by the host stack (8 alternating pairs: 50.504 vs 50.485 ms, -0.02 ms). With q8f in the
  RUN 15 package the host window measured 1036 us (range 906-1723) and 1233 us in a noisier second run: about 1.0-1.2 ms,
  not demonstrably below 1.0 ms, because q8f does not touch the host timeline.

### Gates (RUN 15)

- `INT_MM=0` bit-identity (hard stop): PARITY_IDENTICAL, `llama-perplexity` PPL exactly 134.3955 +/- 6.77407, 5-output
  gate SAME on all five (g0 g1 g2 s2 s0).
- 20-chunk logits against the RUN 14 build, `fc /b`: no differences on the fp16 path (md5
  0e36db18db77650597d4c4088025835f) and on the int best-speed path (md5 b76ef14e158447c224903a9807a704e2), with
  `NORM_Q8` on and off (the q8f path is byte-equal to the unfused path).
- Int best-speed (`MASK=0x27`), `NORM_Q8` 1 and 0: KLD 0.006692, top-p 96.647%, PPL 132.5813, greedy identical. Int
  full-coverage (`MASK=cf`): KLD 0.006942, top-p 96.569%, PPL 132.1043, greedy identical. `INT_MM=1` guard: one warning.
- `GGML_HEXAGON_DESC_VERIFY=1`: DESC_VERIFY_OK on 6 of 6 reps with `NORM_Q8=1` and 6 of 6 with `NORM_Q8=0`, including the
  RMS_NORM_MUL_Q8 producer rewrite (the fused op reuses an existing tensor as src[2], so the tensor count stays 824).
- Static compile matrix 31 objects / 0 errors on all 8 cells (SDK 6.6.0.0 and 6.7.0.0 x v73 v75 v79 v81); host `clang++
  -fsyntax-only` 0 errors on `ggml-hexagon.cpp`, `llama-kv-cache.cpp` and `llama-context.cpp`. v73 / v75 / v79 are
  compile-only.

### Environment (RUN 15 best-speed)

`GGML_HEXAGON_INT_MM=3 GGML_HEXAGON_INT_MM_MASK=27 GGML_HEXAGON_INT_MM_LAYERS=7ffffffff GGML_HEXAGON_INT_MM_REDUCE=2
GGML_HEXAGON_INT_MM_REDUCE_BIG=4 GGML_HEXAGON_INT_MM_MASK16=0 GGML_HEXAGON_INT_ROT=32 GGML_HEXAGON_INT_ROT_MASK=4
GGML_HEXAGON_INT_SMOOTH=<path>/smooth.bin GGML_HEXAGON_VMEM=5G GGML_HEXAGON_NORM_Q8=1`. `NORM_Q8` defaults to 0. The
descriptor cache is on by default (`GGML_HEXAGON_DESC_CACHE=0` disables it, `GGML_HEXAGON_DESC_VERIFY=1` checks it). The
gather change has no switch (it is a ggml-cpu task count).

### Arm table (RUN 15)

3 clean interleaved rounds, medians, `llama-bench -p 512 -r 5` per cell; `NORM_Q8=1` is on the best-speed arm only
unless noted. Prefill health was clean in every round; two decode rounds (round 1: health pp128 2875 under 70% of the
best 4145; round 4: tg32 45.4 under 90% of 53.5) were flagged contaminated, excluded and repeated, never averaged.

| Arm | prefill pp512 (t/s), rounds | median | decode tg128 (t/s), clean rounds | median |
|---|---|---|---|---|
| `pkg-new` (baseline) | 7502.22 / 7570.46 / 7186.26 | 7502.22 | 54.41 / 54.32 / 54.26 | 54.32 |
| merged15, `INT_MM=0` (fp16 path) | 8813.86 / 8716.93 / 8750.22 | 8750.22 | 71.83 / 72.36 / 71.55 | 71.83 |
| merged15, int best-speed, `NORM_Q8=1` | 10042.68 / 9540.13 / 9787.68 | 9787.68 | 70.73 / 71.15 / 72.65 | 71.15 |
| merged15, int best-speed, `NORM_Q8=0` | 9664.81 / 9748.76 / 9441.59 | 9664.81 | 71.51 / 71.28 / 70.87 / 72.53 | 71.40 |
| merged15, int full-coverage | 8932.14 / 8589.12 / 8688.61 | 8688.61 | 71.12 / 70.84 / 71.31 | 71.12 |

- The best-speed arm with `NORM_Q8=1` is 1.305x `pkg-new` in prefill (9788 vs 7502) and the fp16 arm 1.166x. Round 1 of
  that arm read 10042.68 (stddev 0.7%), the first 5-rep cell above 10000, but the other two rounds were 9540 and 9788.
  `NORM_Q8=1` vs 0 in place: +1.3% on the medians (9788 vs 9665), 2 of 3 rounds.
- Decode is unchanged by these changes (decode graphs do not hit the fused qkv or the 512-row paths): every merged arm
  stays above 70 t/s.

### 120-sample alternation (RUN 15)

Int best-speed (`MASK=0x27`), merged15 (`NORM_Q8=1`) vs the RUN 14 build, 12 rounds x 10 samples, order swapped every
round, health at every round start (lowest pp128 3270 = 78.9% of the best seen, none flagged, nothing excluded):

| Pair | merged15 | RUN 14 build | median delta | mean delta (z) | ms/run | round pairs |
|---|---|---|---|---|---|---|
| int best-speed, 120 / 120 | median 9928.3, mean 9723.2, stddev 491 | median 9636.9, mean 9607.4, stddev 345 | +3.02% | +1.97% (+187.6 t/s, z 3.4) | +1.56 | 11 of 12 positive |
| fp16, 40 / 40 | median 9009.3 | median 8796.4 | +2.42% | +2.64% (z 2.5) | +1.38 | 4 of 4 positive |

- Bootstrap 95% CI of the int median: 9866-9965 (the upper end is below 10000). 39 of 120 samples (32.5%) are at or
  above 10000, maximum 10173, against 0 of 120 for the RUN 14 build. The bootstrap 95% CI of the median difference is
  +227..+353 t/s. The single losing round (round 12, mean 9276) had a clean health check; without it the median is 9951
  (+3.43%).
- Median 9928.3 t/s is 51.57 ms per ubatch.

### Op-sum caveat (RUN 15)

- The whole-sum drop expected from q8f was not reproduced. `PROFILE=3`, int best-speed, 6 rotating rounds, 534 ops per
  ubatch, medians: `NORM_Q8=1` 50.50 ms, `NORM_Q8=0` 51.01 ms, RUN 14 build 50.29 ms. Without the first (cold) round of
  each arm `NORM_Q8=1` is 50.24 vs 50.30 ms, i.e. -0.06 ms. One trace varies by 0.5-1 ms from run to run, larger than the
  0.47-0.70 ms that was claimed.
- The per-class effect of q8f is real: MUL_MAT_NX 3.44 -> 2.96 ms (-0.48 ms, the removed qkv A_QUANT). RMS_NORM+MUL
  1.41 -> 0.45 ms plus the new RMS_NORM+MUL+Q8 1.14 ms on the 15 producers add +0.18 ms, so the net on the 15 pairs is
  about -0.3 ms. The end-to-end gain of the stack is therefore mostly the host-side one measured above.

### KPI line (RUN 15)

- Prefill target 10000 t/s: NOT MET. Best arm (int best-speed, `NORM_Q8=1`): 120-sample median 9928.3 t/s, 72 t/s (0.7%,
  0.37 ms per ubatch) short; 3-round median 9787.68; best single 5-rep cell 10042.68.
- Decode target 70 t/s: MET in every merged arm (71.12-71.83).

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
