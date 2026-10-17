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
