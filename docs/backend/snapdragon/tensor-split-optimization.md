# Hexagon tensor-split scaling optimizations

Experimental optimizations for the multi-NPU (tensor-split) allreduce path.
When a model is tensor-split across 4 distinct physical NPUs, measured speedup
plateaus around 2x instead of the expected ~4x. These changes target the
sources of that gap. Every optimization is off by default and enabled with an
environment variable, so existing single-device and multi-device behavior is
unchanged unless explicitly requested.

## Problem

Tensor-split parallelizes the row dimension of large GEMMs across NPUs and
combines the partial results with an allreduce. Amdahl's law shows the serial
fraction must be below ~4% to reach 3.6x on 4 cores. The current allreduce
introduces serial and N-dependent cost in four places:

1. Two spin barriers per allreduce (entry and exit). For decode-sized tensors
   the barrier rendezvous costs more than the actual reduction.
2. Four NPU cores spin on shared DDR-backed peer fences at the same time,
   causing bus contention that grows faster than N.
3. Allreduce VTCM usage grows with rank count (`n_bufs = n_ranks + 1`), so the
   solver shrinks `block_elems` as ranks are added, reducing DMA efficiency
   precisely when more cores are used.
4. Rank chunks are rounded up per rank, so the last rank often holds far fewer
   elements than the others, leaving it idle at the barrier.

## Changes

All changes are confined to the tensor-split allreduce primitive. The general
`FENCE` op and the inter-session copy paths are not modified.

### `GGML_HEXAGON_AR_SINGLE_FENCE=1`

Replaces the entry + exit barrier pair with a single arrival barrier that uses
persistent comm-owned fence words. The fence sequence rises monotonically, so
the arrival check accepts any value at or past the expected sequence. The exit
barrier is removed because the next allreduce's arrival barrier covers
completion, and only rank-local work happens between allreduces.

- `ggml-hexagon.cpp`: `ggml_backend_hexagon_comm_context` allocates one
  persistent fence buffer per rank in `comm_init` and frees it in `comm_free`.
  `comm_allreduce_tensor` submits a single-fence allreduce when enabled.
- `ggml-hexagon.cpp`: `enqueue_allreduce` gained a `single_fence` parameter
  that sets `kparams->fence_mode`. The fused graph path never sets it.
- `htp/allreduce-ops.c`: the entry barrier accepts a rising sequence in single
  fence mode; the exit barrier is skipped.
- `htp/allreduce-ops.h`: added `fence_mode` to `htp_allreduce_kernel_params`.

### `GGML_HEXAGON_FENCE_BACKOFF=1`

Adds backoff to the allreduce barrier spin loops so cores do not hammer shared
bus lines in lockstep while waiting on peers.

- `htp/hex-utils.h`: new `hex_pause_backoff()` helper, backs off every 1024
  spins.
- `htp/allreduce-ops.c`: entry and exit barrier spins call the backoff helper
  when `kparams->fence_backoff` is set.
- `htp/allreduce-ops.h`: added `fence_backoff` to `htp_allreduce_kernel_params`.

### `GGML_HEXAGON_AR_STREAM=1`

Adds a streaming 1D accumulation kernel. Instead of reserving one VTCM buffer
per rank, it keeps one destination block plus one temporary block per thread
and accumulates ranks in place. VTCM use is now independent of rank count, so
`block_elems` no longer shrinks as ranks are added. Applies to the contiguous
non-add (pure allreduce) 1D path only.

- `htp/allreduce-ops.c`: new `allreduce_thread_dma_1d_stream_f16/f32` kernels,
  selected when `kparams->stream_accum` is set; VTCM layout for the stream
  path reserves dst + tmp blocks only.
- `htp/allreduce-ops.h`: added `stream_accum` to `htp_allreduce_kernel_params`.
- `ggml-hexagon.cpp`: `ggml_hexagon_precompute_allreduce_params` gained a
  stream solver branch that computes rank chunks, block size and VTCM usage for
  the streaming path.

### `GGML_HEXAGON_AR_BALANCED=1`

Spreads the 1D allreduce tail so the slowest rank holds at most 64 elements
more than the fastest, instead of rounding the whole chunk up per rank.

- `ggml-hexagon.cpp`: new `ggml_hexagon_ar_chunk_1d()` used by both the default
  and stream solvers. Balanced mode partitions with 64-element granularity.

## Files changed

- `ggml/src/ggml-hexagon/ggml-hexagon.cpp` - env vars, persistent fence
  buffers, single-fence submit, stream solver, balanced partition.
- `ggml/src/ggml-hexagon/htp/allreduce-ops.c` - single fence barrier, spin
  backoff, streaming accumulation kernel.
- `ggml/src/ggml-hexagon/htp/allreduce-ops.h` - new kernel params.
- `ggml/src/ggml-hexagon/htp/hex-utils.h` - `hex_pause_backoff()` helper.
- `scripts/snapdragon/run.py` - `--hex-*` flags for each env var.
- `docs/backend/snapdragon/README.md` - env var reference.

## Usage

Run with distinct physical NPUs, for example:

```
GGML_HEXAGON_DEVICES=HTP0:0,HTP1:0,HTP2:0,HTP3:0 \
GGML_HEXAGON_AR_SINGLE_FENCE=1 \
GGML_HEXAGON_FENCE_BACKOFF=1 \
GGML_HEXAGON_AR_STREAM=1 \
GGML_HEXAGON_AR_BALANCED=1 \
./scripts/snapdragon/run.py --target adb -- llama-cli ...
```

Suggested trial order (each independently measurable with
`GGML_HEXAGON_PROFILE` / `GGML_HEXAGON_OPTRACE` and the trace post-processor):

1. `FENCE_BACKOFF=1` and `AR_BALANCED=1` (low risk), check `FENCE` time.
2. Add `AR_STREAM=1`, compare per-rank `block_elems` and DMA time at 4 ranks.
3. Add `AR_SINGLE_FENCE=1`, verify correctness on a short run first, then measure.
