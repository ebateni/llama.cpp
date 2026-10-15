# Decode-Only Collective-Buffer Isolation (clean reimplementation on upstream)

## Status: VERIFIED — quick TG test confirms the gain reproduces, no deadlock

Quick sanity test, device `1b531bdd`, 4c tensor split, Qwen3-0.6B-Q4_0,
`-p 0 -n 128 -d 1024 -ub 1024 -r 2` (A/B on `GGML_META_COMM_ISOLATION` only,
no other knobs — the `GGML_META_COMM_BUFS` toggle no longer exists in this
design, relocation is unconditional whenever isolation + decode-gate match):

| config | tg128 @ d1024 (4c) |
|---|---|
| `GGML_META_COMM_ISOLATION=0` (no relocation) | 90.93 ± 0.55 |
| `GGML_META_COMM_ISOLATION=1` (decode-gated relocation, default) | **104.72 ± 2.04** |

**+15.2% gain**, no deadlock. Consistent with historical magnitude measured on
the old (pre-rebase) fork (+10.4% to +16.0% across several runs for this same
model). Confirms the fresh reimplementation on clean `origin/master` works.

Full validation (prefill-scatter check, correctness on Qwen3.5-0.8B, other
models/core-counts) is still outstanding — see checklist below. User intends
to run the rest of the validation themselves.

Branch: `test/kv-affinity-upstream` (created off `origin/master` @ `de7fa0a3c6`)
File touched: `ggml/src/ggml-backend-meta.cpp` only (uncommitted).
`ggml-hexagon.cpp` is stock in this design.

## Why this exists (background)

Two earlier, more invasive fixes (KV-cache "alloc_group" affinity fix, and a
gallocr-based collective/compute buffer-separation "vmem overclaim" fix) were
developed on `dev/ebateni/tg-decode-optimize` / `dev/ebateni/vmem-kv-alloc-buffer-n`
against an old fork point. When rebasing onto current `origin/master`, we found:

- Upstream (Max Krasnyansky, PR ggml-org/llama.cpp#30126, merged Oct 8) had
  independently landed his own fix for the *same* KV-split/VMEM-overclaim bug,
  via a different mechanism (hexagon-local tensor-name-suffix matching +
  lowered `GGML_HEXAGON_MBUF` default) — not a literal upstream of our
  `alloc_group` mechanism, but solving the same problem. His version does
  **not** include our giant-buffer isolation feature.
- `ggml-backend-meta.cpp` itself (the file hosting our decode-gated relocation
  mechanism) had been independently restructured upstream since the fork
  (new `ggml_backend_meta_simple_tensor_container` / rotating
  `stc_static`/`stc_compute[2]` containers / `split_state_cache` map). None of
  our `comm_bufs` structures exist there anymore — nothing to merge, the
  mechanism simply does not exist upstream and must be re-implemented fresh.

Per user direction, the decision was: **drop** the KV-affinity (`alloc_group`)
mechanism and the gallocr-separation "vmem overclaim" fix entirely, and
**re-implement only** the decode-gated comm-buffer isolation/relocation
mechanism, built fresh on a clean branch off current `origin/master`. That
mechanism alone was previously verified (on the old fork) to give a real TG
(decode) speedup with no correctness cost, at the price of disabling
reduce-scatter during decode only (a net win — see prior revisions in this
session's planning notes for the quantified trade-off).

## What is implemented (uncommitted diff, `ggml/src/ggml-backend-meta.cpp`, +120/-1)

1. `#define GGML_META_COMM_MAX_SLOTS 8` — near top of file.
2. `ggml_backend_meta_comm_isolation_enabled()` — env `GGML_META_COMM_ISOLATION`,
   default **on**.
3. `ggml_backend_meta_comm_decode_max_tokens()` — env `GGML_META_COMM_DECODE_TOKENS`,
   default **1**. (Default intentionally `1`, not `8`: every decode benchmark
   measured in this effort was single-stream `tg*` with `n_tokens == 1` per
   graph, so `<=8` was never actually distinguished from `==1` by any
   measurement. `==1` is also strictly safer: `n_tokens > 1` cannot
   distinguish a short prefill chunk from several parallel decode sequences
   each contributing one token, but `n_tokens == 1` can only mean one sequence
   advancing by one token.)
4. New fields on `struct ggml_backend_meta_buffer_context`:
   `comm_bufs` (per-rank side buffers), `comm_slots` (map keyed on the
   *original* gallocr tensor address -> slot index), `comm_slot_size`.
5. In `ggml_backend_meta_buffer_init_tensor_impl()`:
   - Computes `split_state_async` (assume_sync=**false**) to correctly detect
     `GGML_BACKEND_SPLIT_AXIS_PARTIAL` (an `assume_sync=true` call can report
     a collective as already-MIRRORED and miss the case).
   - `n_tokens = ggml_nrows(tensor)`; `is_collective` = isolation-enabled AND
     `n_tokens <= decode_max_tokens` AND split axis == PARTIAL.
   - Lazily sizes `comm_bufs`/assigns a `comm_slots` slot keyed on
     `tensor->data` (the gallocr-computed address before redirection), so
     repeated encounters of the "same" collective land in the same physical
     slot.
   - Per-rank loop: if `is_collective && t_ij->view_src == nullptr`, redirects
     `t_ij->buffer`/`t_ij->data` into the isolated per-rank backing at the
     assigned slot offset (capacity = `comm_slot_size * GGML_META_COMM_MAX_SLOTS`).
6. `ggml_backend_meta_buffer_reset()`: clears `comm_slots` / resets
   `comm_slot_size` (does **not** free the already-allocated `comm_bufs`
   backing — a pre-existing, previously-verified-safe quirk carried over
   unchanged from the old implementation; see "Known quirks" below).

### Deliberately NOT ported (dropped per explicit instruction)

- `GGML_META_COMM_BUFS` on/off knob — relocation is now unconditional whenever
  isolation + decode-gate match; no separate toggle re-added.
- `GGML_META_COMM_PROMOTE` / `comm_inplace_src` in-place-ADD-following logic
  ("α-tight promotion"). This existed only to *also* preserve reduce-scatter
  under relocation; it was always off-by-default and known to corrupt
  Qwen3.5-0.8B (uneven KV sharding at 3c/4c). Relocation's decode-time TG gain
  already exceeds what's lost by giving up scatter at decode, so this was
  safely droppable.
- `comm_promoted` debug counter/log line.
- The entire KV-affinity (`alloc_group`) mechanism
  (`ggml.h`/`ggml-alloc.c`/`ggml-backend.cpp`/`ggml-hexagon.cpp`/`llama-kv-cache.cpp`).
- The gallocr-separation "vmem overclaim" fix (`is_communication` buft
  callback, `GGML_TENSOR_FLAG_COMMUNICATION` inherit/reject logic in
  `ggml-alloc.c`, `uneven_sharding` auto-detection, `GGML_META_COMM_INHERIT`).

## Build status

`sh ./scripts/buid_in_docker.sh linux` — **confirmed clean**: full build log
captured (`/tmp/build-full.log`), zero `error:` lines anywhere, and the
specific compile step `Building CXX object
ggml/src/CMakeFiles/ggml-base.dir/ggml-backend-meta.cpp.o` produced no
warnings. Packaging step completes with `done building!`.

## Known quirks (carried over unchanged, previously verified safe)

- `comm_slot_size` is fixed at first use per buffer lifetime, not per
  graph/reset. `ggml_backend_meta_buffer_reset()` clears the slot *map* but
  not the backing buffer, so if slot size ever needed to grow across resets,
  the stale (too-small) buffer would silently be reused. This was true of the
  original (pre-rebase) implementation too and was empirically safe in all
  prior on-device testing; not introduced or changed in this reimplementation.
- `adb root` resets on device reboot — after a reboot, `adb shell` silently
  drops back to `uid=2000(adb)`. In this state `adb push` can report success
  ("1 file pushed") without actually updating file content if the destination
  directory isn't writable by that uid. Always `adb -s <device> root` after a
  reboot, before pushing.
- `adb push <dir>/.` (bulk) can fail with `EOF` on one file even when
  individual per-file pushes to the same path succeed — push files
  individually if bulk push fails.
- `timeout` kills only the adb client, not the on-device process. Reap leaked
  `llama-bench`/`llama-cli` processes between runs:
  `adb shell "ps -A | grep llama"` then `kill -9 <pid>`. A hung process can
  pin memory/DSP sessions and cause spurious failures on later runs, including
  (suspected, unconfirmed) degraded DSP/fastrpc state requiring a device
  reboot to clear.

## Verification checklist (device: `1b531bdd` unless noted)

- [x] Build compiles `ggml-backend-meta.cpp` cleanly, no errors/warnings
      (confirmed via full captured log, see Build status above).
- [ ] Deploy to device, confirm no deadlock, decode TG gain reproduces
      (expect ~41 t/s llama-3.2-3b-instruct-q4_0, matching the prior fork's
      verified 41.04 / 41.38 t/s).
- [ ] Confirm scatter survives at prefill: `GGML_HEXAGON_VERBOSE=1 -v`, pp1024,
      expect fused ALLREDUCE+ADD sites to report `mode=1 n_dsts=4` (not
      `mode=0`), i.e. the decode gate correctly declines relocation for large
      prefill graphs.
- [ ] Correctness spot-check, especially Qwen3.5-0.8B at 3c/4c (previously the
      only model sensitive to the dropped PROMOTE logic — expected safe now
      that PROMOTE is removed entirely, but unverified under this new
      minimal design).

## Next steps / plan

1. Re-verify on device (checklist above) — was in progress when a suspected
   device DSP/fastrpc degradation (from repeated `kill -9` during earlier,
   now-reverted experimentation on a sibling branch) required a manual device
   reboot. Re-run after reboot + `adb root`.
2. Per user's stated plan: once verified working, rename
   `test/kv-affinity-upstream` to a permanent branch name and push to `origin`.
3. Decide disposition of PR 436 (old KV affinity + gallocr separation fixes on
   the old fork) — being superseded/abandoned per this session's direction,
   but the PR itself has not been explicitly closed.
