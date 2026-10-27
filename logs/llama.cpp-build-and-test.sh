#!/usr/bin/env bash
#==============================================================================
# llama.cpp-build-and-test.sh — build, deploy and validate ANY change to this
#                  llama.cpp fork on a multi-core Snapdragon device.
#
# WHAT IT DOES
#   This is a general build -> deploy -> test harness for the Hexagon backend,
#   not a script tied to one specific optimization. It exists so that every
#   change to ggml-hexagon (fusion, allreduce strategy, batching, scheduling,
#   whatever comes next) is checked the same disciplined way, for every
#   configuration you care about:
#
#     1. bench   — llama-bench PP (prompt) and TG (decode) throughput
#     2. iq      — 10 deterministic short questions, substring-scored  (x/10)
#     3. cookie  — open-ended greedy generation, coherence vs baseline
#     4. needle  — needle-in-haystack recall over a 1k-token prompt
#
#   Rationale: more than one past optimization on this fork produced *plausible*
#   text and fully normal llama-bench numbers while silently being wrong (stale
#   cross-core data, garbage from invalid op fusion). Performance numbers alone
#   never catch that. A change is only good if it is FASTER *and* still passes
#   iq + cookie + needle.
#
#   Optionally sweeps two independent axes on top of the above:
#     - AB_VAR/AB   : an env var toggle you want to A/B (e.g. an optimization's
#                     on/off switch). Any GGML_HEXAGON_* env var works here —
#                     this is not specific to any one feature.
#     - UBATCH      : the -ub sweep, useful whenever a change might interact
#                     with batch size.
#   Both are optional knobs, not requirements: set AB_VAR="" for a single
#   build/config test, or UBATCH to one value if batch size is irrelevant to
#   what you're testing.
#
# QUICK START
#   ADB_SERIAL=192.168.1.1:5555 ./llama.cpp-build-and-test.sh
#                                                     # build + deploy + full lagvm matrix
#   ADB_SERIAL=192.168.1.1:5555 ./llama.cpp-build-and-test.sh --android
#                                                     # Android build + direct ethernet deploy + test
#   ADB_SERIAL=192.168.1.1:5555 ./llama.cpp-build-and-test.sh --no-build
#                                                     # reuse existing pkg-snapdragon
#   ADB_SERIAL=192.168.1.1:5555 ./llama.cpp-build-and-test.sh --clean
#                                                     # wipe device pkg dir, then full deploy
#   ADB_SERIAL=192.168.1.1:5555 ./llama.cpp-build-and-test.sh --android --reboot-before-test
#                                                     # reboot/recover the physical target, then test
#   ADB_SERIAL=192.168.1.1:5555 ./llama.cpp-build-and-test.sh --no-build --no-deploy --quick
#                                                     # fastest re-test
#   ./llama.cpp-build-and-test.sh --allow-concurrent   # do NOT kill other jobs (unsafe)
#   REPS=1 ./llama.cpp-build-and-test.sh --no-build --only-perf --split row --hex-profile 3
#                                                     # per-NSP op profile + FENCE/L2FLUSH
#                                                     # trace of the PP bench -> $OUTDIR/profile/
#   CORES=1,4 AB_VAR=GGML_HEXAGON_FA_SPLIT AB=0,1 ./llama.cpp-build-and-test.sh --test-ops FLASH_ATTN_EXT
#                                                     # test-backend-ops -b HTP0 on the device per cores x A/B
#                                                     # value (row split); no model runs -> testops.tsv
#   ./llama.cpp-build-and-test.sh --push-only logs/<ts>
#                                                     # no device: split >94 MB .gz into parts,
#                                                     # then commit + push that run dir only
#   AB_VAR= UBATCH=1024 ./llama.cpp-build-and-test.sh --no-build   # single config, no sweep
#   AB_VAR=GGML_HEXAGON_FUSE_MUL_MAT_ADD AB=0,1 ./llama.cpp-build-and-test.sh --no-build
#
#   Example (the reduce-scatter/ubatch-additivity investigation this was
#   originally written for is just one instance of the above):
#   AB_VAR=GGML_HEXAGON_AR_SCATTER AB=0,1 CORES=1,2,4 UBATCH=512,1024,2048 \
#     ./llama.cpp-build-and-test.sh --no-build
#
# OUTPUT
#   terminal            live progress + final summary tables
#   $OUTDIR/summary.md  same tables, markdown
#   $OUTDIR/results.tsv machine-readable, one row per configuration
#   $OUTDIR/logs/       raw llama-bench / llama-completion output per run
#
# WHERE IT RUNS
#   Run this from your normal HOST terminal (x86_64 Ubuntu). It is a host-side
#   driver; nothing needs to be copied to the device by hand.
#
#   The build phase is the one exception and it is handled for you: this host
#   CANNOT build llama.cpp natively. Verified:
#     - host cmake is 3.22 and cannot even parse CMakePresets.json
#       ("Unrecognized version field"); the presets need cmake >= 3.31
#     - there is no Hexagon SDK and no arm64 cross toolchain on the host
#   So `build` just calls the repo's own scripts/buid_in_docker.sh, which picks
#   the toolchain image for the platform and runs scripts/build.sh inside it
#   (cmake 3.31.6 there). The image is already cached locally, so it works
#   offline. Nothing about the docker invocation is duplicated here.
#
#   Conversely, deploy and test MUST run on the host: the container has no adb
#   and no access to the USB device. Hence the split:
#       build   -> docker
#       deploy  -> host (adb)
#       test    -> host (adb) driving a small script on the device
#
# REQUIREMENTS
#   host   : adb + docker  (docker only for --build; use --no-build to skip)
#   device : models + prompt_1k.txt already present (see MODELS_DEV / PROMPT_DEV)
#
#==============================================================================
set -uo pipefail

ORIGINAL_ARGV=("$@")

#------------------------------------------------------------------------------
# Configuration — every value can be overridden from the environment.
#------------------------------------------------------------------------------
: "${REPO:=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)}"
                                                         # llama.cpp source tree to build; default:
                                                         # the checkout this script lives in
: "${ADB_SERIAL:=}"                                      # required target serial
: "${ADB_SERVER_SOCKET:=}"                               # optional host ADB server override
                                                         #   format: tcp:<host>:<port>
                                                         #   e.g.  tcp:10.46.148.187:5037
: "${PLATFORM:=linux}"                                   # linux | android
                                                         # (image selection is handled by
                                                         #  scripts/buid_in_docker.sh)
#
# Seca preset — uncomment or set these in your environment:
#   export ADB_SERIAL=e834120e
#   export ADB_SERVER_SOCKET=tcp:10.46.148.187:5037
: "${PKG_HOST:=$REPO/pkg-snapdragon}"                    # host package dir
: "${PKG_DEV:=/data/local/tmp/pkg-bat}"                # device package dir
: "${MODELS_DEV:=/data/local/tmp/models}"                # device model dir
: "${PROMPT_DEV:=/data/local/tmp/prompt_1k.txt}"         # 1k needle prompt
: "${NEEDLE:=CTX-1k-DEPTH-50PCT-1024T}"                  # string a correct run echoes
# Default: a timestamped run dir next to this script (i.e. <repo>/logs/<ts>/).
# --push-results commits that dir and pushes it (the fork is the only channel
# between the test machine and the analysis side).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${OUTDIR:=$SCRIPT_DIR/$(date +%Y%m%d-%H%M%S)}"
# Optional per-test diagnostics. Continuous samples are streamed to the host,
# never accumulated in /tmp or /data/local/tmp on the device.
: "${COLLECT_MEMORY:=0}"              # 1 with --collect-memory
: "${MEMORY_INTERVAL:=1}"             # sampling period in seconds
: "${MEMORY_PROCESS_MAP:=1}"          # capture status/smaps/maps once per test
: "${AUTO_REBOOT_ON_CLEANUP_FAILURE:=1}"
# PDU mapping for the two lab targets (8ee1cb0 + its lagvm guest, and 1b531bdd)
# is built in below: host, outlet and username need no configuration. Only the
# password is not stored in this script; it is taken from PDU_PASS, or asked for
# once per run. Any OTHER target must supply PDU_HOST, PDU_OUTLET, PDU_USER and
# PDU_PASS explicitly, otherwise it falls back to a manual reboot prompt.
: "${PDU_USER:=}"                     # auto-set to the lab default for mapped targets
: "${PDU_PASS:=}"                     # export once in your shell to run fully unattended
: "${PDU_HOST:=}"                     # optional override for an unmapped target
: "${PDU_OUTLET:=}"                   # optional override for an unmapped target
: "${PDU_LAB_USER:=admin}"            # username used for the mapped lab PDUs
: "${PDU_PROMPT:=1}"                  # 0 disables the one-time password prompt
: "${PDU_OFFLINE_WAIT:=240}"          # maximum time to observe power-off
: "${REBOOT_POLL:=5}"

# Test matrix.
: "${AB_VAR=}"                          # env var to A/B-sweep; ANY GGML_HEXAGON_*
                                        # var works here (fusion toggles, allreduce
                                        # strategy, etc.) — disabled by default
                                        # (empty = single configuration, no sweep).
                                        # e.g. AB_VAR=GGML_HEXAGON_AR_SCATTER to
                                        # A/B a var supported by your build.
: "${AB:=0,1}"                         # values of $AB_VAR to compare; ignored if AB_VAR=""
: "${CORES:=1,2,4}"                    # core counts
: "${SPLITS:=tensor}"                  # multi-core split mode(s) to test when
                                        # cores>1: tensor | tensor-scatter | layer | row | layer-row | tensor-row | sequence
                                        # (comma list, e.g. "tensor,tensor-scatter,row").
                                        # tensor-scatter = tensor layout with AR_SCATTER=1.
                                        # sequence = token-dim split, mirrored weights/KV (no allreduce).
                                        # Meaningless at 1 core (always "single").
: "${UBATCH:=1024}"                     # -ub value(s) to test; comma list to sweep
                                        # multiple (e.g. "512,1024,2048"), catching
                                        # interactions between a change and batch
                                        # size. Default: single value, no sweep.
: "${BATCH:=}"                          # llama-bench -b (batch size); empty = llama-bench default
: "${REPS:=5}"                 # llama-bench -r (matches llama-bench's native default)
: "${QTMO:=90}"                # iq/cookie timeout (s); +30s per 4GB of model, needle +60s
: "${PUSH_RESULTS:=0}"         # --push-results: git commit $OUTDIR + push at the end
: "${PUSH_ONLY:=}"             # --push-only <dir>: only commit + push an existing run dir (no device)
: "${TEST_OPS:=}"              # --test-ops OP[,OP..]: run test-backend-ops (-o OP) instead of the model matrix
: "${TEST_OPS_PATTERN:=}"      # optional test-backend-ops -p <regex> filter for --test-ops
: "${HEX_PROFILE:=0}"          # --hex-profile N: GGML_HEXAGON_PROFILE for the PP
                               # bench only (1=op usec, 2=+PMU, 3=+trace events
                               # incl. FENCE/L2FLUSH). Adds -v and keeps the FULL
                               # PP stderr in $OUTDIR/profile/<label>.log (not the
                               # 4000-line tail). TG runs unprofiled. Profiling
                               # distorts PP t/s: use REPS=1 and take clean
                               # numbers from a separate run with HEX_PROFILE=0.
# Accept the actual runtime environment-variable names. The shorter aliases
# remain supported for backward compatibility.
#
# Precedence: GGML_META_COMM_ISOLATION > COMM_ISOLATION > default 1.
# Kept for reference/A-B only; not passed to the binaries unless swept via AB_VAR.
: "${COMM_ISOLATION:=${GGML_META_COMM_ISOLATION:-1}}"
GGML_META_COMM_ISOLATION="$COMM_ISOLATION"
if [ -n "$ADB_SERVER_SOCKET" ]; then
  export ADB_SERVER_SOCKET
fi
: "${PP_DEPTH:=0}"             # llama-bench -d for the PP measurement (KV-cache
                               # depth prompt processing starts from; 0 = empty cache)
: "${TG_DEPTH:=1024}"
: "${PP_TOKENS:=1024}"         # llama-bench -p; comma list to sweep multiple
                               # prompt lengths (e.g. "1024,4096,8192"). Each
                               # value produces its own bench rows (tagged _ppN
                               # in labels). TG depth adjusts per PP length.
: "${DEPTHS:=}"
: "${RETRY_ON_FAIL:=1}"          # llama-bench -d for the TG measurement (KV-cache
                               # depth decode starts from, i.e. tokens already in
                               # context before the 256 generated/measured)
: "${CTX:=4096}"               # MUST stay well below the model max: the default
                               # (model max) makes llama.cpp allocate a
                               # multi-buffer KV cache that the tensor-split meta
                               # backend aborts on. Upstream issue #22197.
                               # llama-bench is unaffected. 4096 covers the
                               # bench (-d 1024 -n 256) and needle (1k prompt +
                               # 128 gen) workloads with headroom.

# Models: <device-basename-without-.gguf>:<max-cores>
#   Gemma-4-E4B is capped at 2 cores: it uses GQA with only 2 KV heads, so a
#   4-way KV split leaves 2 cores with zero heads and trips an assertion in
#   ggml-backend-meta.cpp. The GGML_HEXAGON_KV_MIRROR workaround does NOT exist
#   on this branch (verified: not among the env vars ggml-hexagon.cpp reads),
#   so 4-core Gemma would simply abort.
: "${MODELS:=llama-3.2-3b-instruct-q4_0:4 Qwen3.5-4B-Q4:4 gemma-4-E4B_q4_0-it:2}"
# Accept "name.gguf[:N]" too: the runner always appends .gguf itself.
MODELS=$(for spec in $MODELS; do printf '%s ' "$(printf '%s' "$spec" | sed -E 's/\.gguf(:|$)/\1/')"; done)
MODELS=${MODELS% }

# Phase switches.
DO_BUILD=1; DO_DEPLOY=1; DO_BENCH=1; DO_IQ=1; DO_COOKIE=1; DO_NEEDLE=1; DO_RUN=1
REBOOT_BEFORE_TEST=0

# Wipe $PKG_DEV on the device before pushing. Off by default because a full
# re-push costs ~100 files; see do_deploy() for when you actually need it.
DO_CLEAN=0

# Kill every other llama.cpp workload (host and device) before starting.
# On by default — see ensure_exclusive() for why.
KILL_CONCURRENT=1

while [ $# -gt 0 ]; do
  case "$1" in
    --no-build)   DO_BUILD=0 ;;
    --no-deploy)  DO_DEPLOY=0 ;;
    --no-bench)   DO_BENCH=0 ;;
    --no-iq)      DO_IQ=0 ;;
    --no-cookie)  DO_COOKIE=0 ;;
    --no-needle)  DO_NEEDLE=0 ;;
    --no-run)     DO_DEPLOY=0; DO_BENCH=0; DO_IQ=0; DO_COOKIE=0; DO_NEEDLE=0 ;;
    --clean)      DO_CLEAN=1 ;;
    --android)    PLATFORM=android ;;
    --device)     DEVICE=$2; shift ;;
    --device=*)   DEVICE=${1#--device=} ;;
    --reboot-before-test) REBOOT_BEFORE_TEST=1 ;;
    --quick)      UBATCH=1024; CORES=1,4; REPS=2 ;;
    --only-perf)  DO_IQ=0; DO_COOKIE=0; DO_NEEDLE=0 ;;
    --collect-memory) COLLECT_MEMORY=1 ;;
    --push-results) PUSH_RESULTS=1 ;;
    --push-only)  PUSH_ONLY=$2; shift ;;
    --push-only=*) PUSH_ONLY=${1#--push-only=} ;;
    --test-ops)   TEST_OPS=$2; shift ;;
    --test-ops=*) TEST_OPS=${1#--test-ops=} ;;
    --hex-profile) HEX_PROFILE=$2; shift ;;
    --hex-profile=*) HEX_PROFILE=${1#--hex-profile=} ;;
    --memory-interval) MEMORY_INTERVAL=$2; shift ;;
    --memory-interval=*) MEMORY_INTERVAL=${1#--memory-interval=} ;;
    --allow-concurrent) KILL_CONCURRENT=0 ;;
    --split)      SPLITS=$2; shift ;;
    --split=*)    SPLITS=${1#--split=} ;;
    --pp-depth)   PP_DEPTH=$2; shift ;;
    --pp-depth=*) PP_DEPTH=${1#--pp-depth=} ;;
    --tg-depth)   TG_DEPTH=$2; shift ;;
    --tg-depth=*) TG_DEPTH=${1#--tg-depth=} ;;
    --pp-tokens)  PP_TOKENS=$2; shift ;;
    --pp-tokens=*) PP_TOKENS=${1#--pp-tokens=} ;;
    --batch)      BATCH=$2; shift ;;
    --batch=*)    BATCH=${1#--batch=} ;;
    -h|--help)    awk 'NR>1 && /^#/{sub(/^# ?/,"");print;next} NR>1{exit}' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# --device preset: sets ADB target, ADB server socket, SDK, and HTP arch.
# Any value already set in the environment takes precedence (: "${VAR:=}" pattern).
: "${DEVICE:=}"
: "${HTP_ARCH:=}"   # auto-set by --device; can also be overridden directly
case "$DEVICE" in
  Seca|seca|SECA)
    : "${ADB_SERIAL:=e834120e}"
    : "${ADB_SERVER_SOCKET:=tcp:10.46.148.187:5037}"
    : "${HTP_ARCH:=v85}"
    : "${CORES:=1,2}"
    ;;
  Nord|nord|NORD)
    : "${HTP_ARCH:=v81}"
    ;;
  "")
    : "${HTP_ARCH:=v81}"   # default when no --device given
    ;;
  *)
    echo "unknown --device: $DEVICE (supported: Nord, Seca)" >&2; exit 2 ;;
esac

# --clean only has meaning if the deploy phase actually runs; silently ignoring
# it would leave the user believing the device was wiped when it was not.
if [ "$DO_CLEAN" = 1 ] && [ "$DO_DEPLOY" = 0 ]; then
  echo "--clean requires the deploy phase (drop --no-deploy)" >&2; exit 2
fi

# --push-only <dir>: no device work at all — just commit + push an existing run dir.
if [ -n "$PUSH_ONLY" ]; then
  [ -d "$PUSH_ONLY" ] || { echo "--push-only: no such directory: $PUSH_ONLY" >&2; exit 2; }
  OUTDIR=$(cd "$PUSH_ONLY" && pwd)
  PUSH_RESULTS=1
fi

[ -n "$ADB_SERIAL" ] || [ -n "$PUSH_ONLY" ] ||
  { echo "ADB_SERIAL is required (for lagvm: ADB_SERIAL=192.168.1.1:5555)" >&2; exit 2; }

for _sp in $(printf '%s' "$SPLITS" | tr ',' ' '); do
  case "$_sp" in
    tensor|tensor-scatter|layer|row|layer-row|tensor-row|sequence) ;;
    *) echo "invalid --split value: $_sp (must be tensor, tensor-scatter, layer, row, layer-row, tensor-row, or sequence)" >&2; exit 2 ;;
  esac
done

ADB=""
LIB="$PKG_DEV/lib"
DEV_DRIVER=/data/local/tmp/llama.cpp-build-and-test-dev.sh
MEM_SAMPLER_DEV=/data/local/tmp/llama.cpp-mem-sample.sh
TSV="$OUTDIR/results.tsv"
SUMMARY="$OUTDIR/summary.md"
LOGS="$OUTDIR/logs"

[ -n "$PUSH_ONLY" ] || mkdir -p "$LOGS"

RUN_CONFIG_LOG="$OUTDIR/run-config.log"
DEVICE_INVOCATIONS_LOG="$OUTDIR/device-invocations.log"
[ -n "$PUSH_ONLY" ] || : > "$DEVICE_INVOCATIONS_LOG"
MEMORY_LOGS="$OUTDIR/memory"
MEMORY_TSV="$OUTDIR/memory-summary.tsv"
REBOOT_LOG="$OUTDIR/reboots.log"
ROW_RECOVERY_DEFER=0
ROW_RECOVERY_STATE="$OUTDIR/.row-recovery-reasons"
if [ "$COLLECT_MEMORY" = 1 ]; then
  mkdir -p "$MEMORY_LOGS"
  printf 'label\texit_code\tpeak_system_used_kB\tmin_available_kB\tmin_free_kB\tpeak_rss_kB\tpeak_anon_kB\tpeak_pagetables_kB\tpeak_dmabuf_bytes\tpeak_dmabuf_objects\tpeak_psi_full_avg10\toom_lines\toom_source\n' > "$MEMORY_TSV"
fi

# ANSI helpers (disabled when not a tty).
if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; C=$'\033[36m'; N=$'\033[0m'
else B=""; G=""; R=""; Y=""; C=""; N=""; fi

say()  { printf '%s\n' "$*"; }
head1() { printf '\n%s=== %s ===%s\n' "$B" "$*" "$N"; }
die()  { printf '%sERROR:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }
# TCP-style serials (IP:port, e.g. 192.168.1.1:5555) are never auto-visible to
# the host adb server the way USB serials are — they must be re-registered
# with `adb connect` any time the server was (re)started or the transport was
# dropped (power cycle, kill-server, etc). USB serials (no ':') are a no-op.
ensure_adb_connected() {
  case "$ADB_SERIAL" in
    *:*) command adb connect "$ADB_SERIAL" >/dev/null 2>&1 ;;
  esac
}

setup_adb_transport() {
  if [ -n "$ADB_SERVER_SOCKET" ]; then
    ADB="adb -L $ADB_SERVER_SOCKET -s $ADB_SERIAL"
  else
    ADB="adb -s $ADB_SERIAL"
  fi
  ensure_adb_connected
}
[ -n "$PUSH_ONLY" ] || setup_adb_transport

#------------------------------------------------------------------------------
# Phase 1 — build
#   Delegates to the repo's existing scripts/buid_in_docker.sh (note the typo in
#   the upstream filename), which selects the right toolchain image for the
#   platform and runs scripts/build.sh inside it. build.sh itself is pure cmake
#   and contains no docker; the container is required because this host has
#   cmake 3.22 (too old to parse CMakePresets.json) and no Hexagon SDK.
#
#   Inside the container build.sh does:
#     cmake --preset -> build -> install -> SWIV-sign libggml-htp-v81.so -> zip
#   The signing step is mandatory; an unsigned skel will not load on a
#   production-fused part.
#
#   Deliberately NOT reimplemented here: image selection and the docker flags
#   live in one place so a toolchain bump only has to be made once.
#------------------------------------------------------------------------------
do_build() {
  head1 "BUILD ($PLATFORM, HTP=$HTP_ARCH) — via scripts/buid_in_docker.sh"
  cd "$REPO" || die "no such repo: $REPO"
  [ -f scripts/buid_in_docker.sh ] || die "missing $REPO/scripts/buid_in_docker.sh"

  DEVICE="$DEVICE" bash ./scripts/buid_in_docker.sh "$PLATFORM" || die "docker build failed"

  # buid_in_docker.sh ends its container command with `echo done building!`,
  # which masks build.sh's exit status, so the check above cannot be trusted.
  # Verify the two artifacts that actually matter instead.
  [ -f "$PKG_HOST/bin/llama-bench" ]                       || die "build produced no llama-bench"
  [ -f "$PKG_HOST/lib/libggml-htp-${HTP_ARCH}.so" ] || die "build produced no signed HTP skel (libggml-htp-${HTP_ARCH}.so)"

  # Stamp the package so results record which source the device binaries came from
  # (with --no-build the repo HEAD can differ from what is deployed).
  printf '%s (%s)%s built %s
' "$(git rev-parse --short HEAD)" "$(git rev-parse --abbrev-ref HEAD)"     "$(git diff --quiet HEAD -- ggml src common tools 2>/dev/null || echo ' +dirty')" "$(date -Iseconds)"     > "$PKG_HOST/BUILD_INFO"
  say "${G}build ok${N}: $PKG_HOST  ($(cat "$PKG_HOST/BUILD_INFO"))"
}

device_build_info() {
  $ADB shell "cat $PKG_DEV/BUILD_INFO 2>/dev/null" 2>/dev/null | tr -d '
' | grep . || echo "unknown (package built before BUILD_INFO stamping)"
}

#------------------------------------------------------------------------------
# Phase 2 — deploy
#   Pushes the package and the device-side driver, then sanity-checks that every
#   model and the prompt file actually exist. Failing here is far cheaper than
#   discovering a missing model 40 minutes into the matrix.
#------------------------------------------------------------------------------
do_deploy() {
  head1 "DEPLOY -> $ADB_SERIAL:$PKG_DEV"
  $ADB get-state >/dev/null 2>&1 || die "device $ADB_SERIAL not available"
  [ -d "$PKG_HOST" ] || die "no package at $PKG_HOST (run without --no-build)"

  # --clean: wipe the device pkg dir first so no stale libs (renamed/removed
  # in a new build) persist and shadow the fresh ones. Use tar for deployment
  # so symlinks (libfoo.so -> libfoo.so.0 etc.) are preserved on the device;
  # adb push resolves symlinks on the host and sends regular file copies,
  # which means the .so soname links point to whichever version was current at
  # copy time and get out of sync after a rebuild.
  if [ "$DO_CLEAN" = 1 ]; then
    say "wiping $PKG_DEV"
    $ADB shell "rm -rf $PKG_DEV" || die "cannot wipe $PKG_DEV"
  fi

  $ADB shell "mkdir -p $PKG_DEV" || die "cannot create $PKG_DEV"

  tar czf - -C "$PKG_HOST" . \
    | $ADB exec-in "tar xzf - -C $PKG_DEV" \
    || die "tar deploy failed"

  $ADB shell "chmod -R 755 $PKG_DEV/bin $PKG_DEV/lib"
  $ADB shell "[ -f $PKG_DEV/bin/llama-bench ]" || die "deploy incomplete: llama-bench missing on device"

  for spec in $MODELS; do
    local m=${spec%%:*}
    $ADB shell "[ -f $MODELS_DEV/$m.gguf ]" \
      || die "missing model on device: $MODELS_DEV/$m.gguf"
  done
  $ADB shell "[ -f $PROMPT_DEV ]" || die "missing prompt on device: $PROMPT_DEV"

  say "${G}deploy ok${N}"
}

#------------------------------------------------------------------------------
# Device-side driver
#   Generated on the host and pushed, rather than inlined into `adb shell "..."`.
#   Nested quoting across adb is the single biggest source of silent breakage in
#   scripts like this; a real file on the device removes that class of bug and
#   makes each run independently reproducible by hand.
#
#   Usage on device:  sh llama.cpp-build-and-test-dev.sh <bench|iq|cookie|needle> <args...>
#------------------------------------------------------------------------------
write_dev_driver() {
  # Seca has only 2 NSPs (cdsp0/cdsp1); Nord and others have 4. Build the
  # cdsp2/cdsp3 env lines here so the heredoc stays simple — the \$LIB inside
  # must reach the device script literally (expanded on-device, not at gen time).
  local CDSP23_LINES=''
  case "$DEVICE" in
    Seca|seca|SECA) CDSP23_LINES='' ;;
    *) CDSP23_LINES='CDSP2_LIBRARY_PATH="$LIB${CDSP_PATH_SEP}/vendor/dsp/cdsp2"
CDSP3_LIBRARY_PATH="$LIB${CDSP_PATH_SEP}/vendor/dsp/cdsp3"' ;;
  esac
  cat > "$OUTDIR/llama.cpp-build-and-test-dev.sh" <<DEVEOF
#!/bin/sh
# Auto-generated by llama.cpp-build-and-test.sh — device-side measurement driver.
# noglob: the GGML_HEXAGON_DEVICES row-split value (e.g. HTP0[0-3]) must survive
# word-splitting of the env-prefix command without pathname expansion — a stray
# file like HTP00 in cwd would otherwise expand the bracket and corrupt it.
set -f
PKG=$PKG_DEV
LIB=\$PKG/lib
BENCH=\$PKG/bin/llama-bench
COMP=\$PKG/bin/llama-completion
MODELS=$MODELS_DEV
PROMPT=$PROMPT_DEV
CTX=$CTX
PP_TOKENS=$PP_TOKENS
BATCH=$BATCH
AB_VAR="$AB_VAR"   # name of the env var being A/B-swept this run (may be empty)
HEX_PROFILE=$HEX_PROFILE   # GGML_HEXAGON_PROFILE for the PP bench (0 = off)
PROF_FILE=/data/local/tmp/bench_pp_prof.txt
# profenv — extra env for the PP bench only (TG stays unprofiled). Skipped when
# GGML_HEXAGON_PROFILE is itself the A/B variable (hexenv emits it then).
profenv() {
  [ "\$HEX_PROFILE" != 0 ] && [ "\$AB_VAR" != "GGML_HEXAGON_PROFILE" ] && \
    printf '%s=%s ' GGML_HEXAGON_PROFILE "\$HEX_PROFILE"
}
# profiling is on if either --hex-profile or the A/B value enables it
prof_on() {
  [ "\$HEX_PROFILE" != 0 ] && return 0
  [ "\$AB_VAR" = "GGML_HEXAGON_PROFILE" ] && [ "\${AB_VAL:-0}" != 0 ] && return 0
  return 1
}

# --- Hexagon runtime environment -------------------------------------------
# LD_LIBRARY_PATH        : host-side .so (libggml-hexagon, libllama, ...)
# GGML_HEXAGON_ARCH=81   : v81 HTP; selects libggml-htp-v81.so
# CDSPn_LIBRARY_PATH     : per-core FastRPC search path. Entry separator is
#                          platform-dependent: ';' on OE Linux, ':' on
#                          Android. CDSP_LIBRARY_PATH for core 0, then
#                          CDSP1/2/3_LIBRARY_PATH — NOT CDSP0_LIBRARY_PATH.
# ADSP_LIBRARY_PATH      : Android DSP search path; set to the package's full
#                          absolute lib path (not the bare "/lib").
# GGML_HEXAGON_DEVICES   : which physical cores the backend may open.
# NOTE: none of these are \`export\`ed. They are passed inline to every
# binary invocation via \`env \$(hexenv) <cmd> ...\` (see hexenv() below)
# instead, so they can't be silently lost if an adb shell layer
# mangles/collapses earlier \`export\` statements onto one line.
LD_LIBRARY_PATH=\$LIB
if [ "$PLATFORM" = "android" ]; then
  ADSP_LIBRARY_PATH=\$LIB
  CDSP_PATH_SEP=":"
else
  CDSP_PATH_SEP=";"
fi
GGML_HEXAGON_ARCH=${HTP_ARCH#v}
GGML_HEXAGON_VERBOSE=${HEXAGON_VERBOSE:-0}
GGML_HEXAGON_NHVX=${GGML_HEXAGON_NHVX:-}
GGML_HEXAGON_VTCM_SIZE=${GGML_HEXAGON_VTCM_SIZE:-}
GGML_META_COMM_ISOLATION=$COMM_ISOLATION
CDSP_LIBRARY_PATH="\$LIB\${CDSP_PATH_SEP}/vendor/dsp/cdsp0"
CDSP1_LIBRARY_PATH="\$LIB\${CDSP_PATH_SEP}/vendor/dsp/cdsp1"
${CDSP23_LINES}

# devlist <n> -> "HTP0,HTP1,..."   (llama-completion -dev, GGML_HEXAGON_DEVICES)
devlist() { awk -v n="\$1" 'BEGIN{s="";for(i=0;i<n;i++)s=s (i?",":"") "HTP" i;print s}'; }
# devslash <n> -> "HTP0/HTP1/..."  (llama-bench --device)
devslash() { awk -v n="\$1" 'BEGIN{s="";for(i=0;i<n;i++)s=s (i?"/":"") "HTP" i;print s}'; }
# split mode: only meaningful with >1 core
# tensor-scatter uses the same device layout as tensor but with AR_SCATTER=1 (set by set_scatter)
# layer-row is a 2D split on 4 NSPs: HTP0=cores 0-1, HTP1=cores 2-3, outer layer split
#   is implicit (two logical devices) — NO -sm flag (and -sm layer must NOT be passed).
# tensor-row is the same physical layout as layer-row but outer split is tensor (-sm tensor).
smode() {
  [ "\$1" -gt 1 ] && [ "\$2" != "layer-row" ] && echo "-sm \${2%%-*}" || echo ""
}
# hexdevices <n> <split> -> value for GGML_HEXAGON_DEVICES.
#   row-split groups all n physical cores into one logical device via bracket
#   syntax (native to ggml-hexagon, see docs/backend/snapdragon/README.md);
#   layer-row/tensor-row map two logical devices onto two disjoint physical core
#   pairs (HTP0=cores 0-1, HTP1=cores 2-3); tensor/layer need the plain list.
hexdevices() {
  if [ "\$1" -gt 1 ] && [ "\$2" = "row" ]; then echo "HTP0[0-\$((\$1-1))]"
  elif [ "\$1" -gt 1 ] && { [ "\$2" = "layer-row" ] || [ "\$2" = "tensor-row" ]; }; then echo "HTP0[0-1:0],HTP1[2-3:0]"
  else devlist "\$1"; fi
}
# clidev <n> <split> -> value for llama-completion's -dev (comma form)
clidev() {
  if [ "\$1" -gt 1 ] && [ "\$2" = "row" ]; then echo "HTP0"
  elif [ "\$1" -gt 1 ] && { [ "\$2" = "layer-row" ] || [ "\$2" = "tensor-row" ]; }; then echo "HTP0,HTP1"
  else devlist "\$1"; fi
}
# clidevslash <n> <split> -> value for llama-bench's --device (slash form)
clidevslash() {
  if [ "\$1" -gt 1 ] && [ "\$2" = "row" ]; then echo "HTP0"
  elif [ "\$1" -gt 1 ] && { [ "\$2" = "layer-row" ] || [ "\$2" = "tensor-row" ]; }; then echo "HTP0/HTP1"
  else devslash "\$1"; fi
}
# devarg <n> <split> -> "-dev X" or "" (llama-completion)
devarg() { local d=\$(clidev "\$1" "\$2"); [ -n "\$d" ] && echo "-dev \$d" || echo ""; }
# devargslash <n> <split> -> "--device X" or "" (llama-bench)
devargslash() { local d=\$(clidevslash "\$1" "\$2"); [ -n "\$d" ] && echo "--device \$d" || echo ""; }
# set_scatter <split> — set AR_SCATTER=1 for tensor-scatter, 0 otherwise
# (plain variable; picked up by hexenv() below, not exported)
set_scatter() {
  if [ "\${1%%-scatter}" != "\$1" ]; then GGML_HEXAGON_AR_SCATTER=1
  else GGML_HEXAGON_AR_SCATTER=0; fi
}
# set_ab <value> — record the swept env var's value, if one is configured
# (plain variable; hexenv() emits "\$AB_VAR=\$AB_VAL" inline, not exported)
set_ab() { AB_VAL="\$1"; }
# qtmo <model> — quality-check timeout scaled by model size (load time dominates)
qtmo() { local sz=\$(stat -c %s \$MODELS/\$1.gguf 2>/dev/null || echo 0); echo \$(( $QTMO + sz / 4294967296 * 30 )); }
# hexenv — prints the full Hexagon runtime env as "VAR=value" tokens, for
# use as: env \$(hexenv) <command> ...   (values never contain whitespace,
# so unquoted word-splitting here is safe). Passing these inline rather than
# exporting avoids losing them if an adb shell layer mangles earlier
# multi-line \`export\` statements onto one line.
hexenv() {
  printf '%s=%s ' \
    LD_LIBRARY_PATH "\$LIB" \
    GGML_HEXAGON_ARCH "\$GGML_HEXAGON_ARCH" \
    GGML_HEXAGON_VERBOSE "\$GGML_HEXAGON_VERBOSE" \
    CDSP_LIBRARY_PATH "\$CDSP_LIBRARY_PATH" \
    CDSP1_LIBRARY_PATH "\$CDSP1_LIBRARY_PATH"
  [ -n "\${CDSP2_LIBRARY_PATH:-}" ] && printf '%s=%s ' CDSP2_LIBRARY_PATH "\$CDSP2_LIBRARY_PATH"
  [ -n "\${CDSP3_LIBRARY_PATH:-}" ] && printf '%s=%s ' CDSP3_LIBRARY_PATH "\$CDSP3_LIBRARY_PATH"
  printf '%s=%s ' \
    GGML_HEXAGON_DEVICES "\${GGML_HEXAGON_DEVICES:-}" \
    GGML_HEXAGON_AR_SCATTER "\${GGML_HEXAGON_AR_SCATTER:-0}"
  [ -n "\${GGML_HEXAGON_NHVX:-}" ] && printf '%s=%s ' GGML_HEXAGON_NHVX "\$GGML_HEXAGON_NHVX"
  [ -n "\${GGML_HEXAGON_VTCM_SIZE:-}" ] && printf '%s=%s ' GGML_HEXAGON_VTCM_SIZE "\$GGML_HEXAGON_VTCM_SIZE"
  [ "$PLATFORM" = "android" ] && printf '%s=%s ' ADSP_LIBRARY_PATH "\$LIB"
  [ -n "\$AB_VAR" ] && printf '%s=%s ' "\$AB_VAR" "\$AB_VAL"
}

CMD=\$1; shift

case "\$CMD" in

  # bench <model> <cores> <split> <abval> <ubatch>
  #   Emits: "PP=<t/s> TG=<t/s>"
  #   PP is measured from cache-depth $PP_DEPTH (0 = empty cache by default).
  #   TG is measured after a $TG_DEPTH-token prompt (1024 by default) so decode
  #   throughput is checked past the empty-cache case; it is ubatch-independent
  #   (1 token per step) but still re-measured per config to catch decode
  #   regressions.
  bench)
    m=\$1
    n=\$2
    sp=\$3
    ab=\$4
    ub=\$5
    pp_depth=\$6
    tg_depth=\$7
    pp_tokens=\${8:-\$PP_TOKENS}

    GGML_HEXAGON_DEVICES=\$(hexdevices \$n \$sp)
    set_scatter "\$sp"
    set_ab "\$ab"
    echo "HARNESS_ENV split=\$sp ab=\$ab" \
         "GGML_HEXAGON_AR_SCATTER=\${GGML_HEXAGON_AR_SCATTER:-unset}" \
         "GGML_META_COMM_ISOLATION=\${GGML_META_COMM_ISOLATION:-unset}" \
         "GGML_HEXAGON_DEVICES=\${GGML_HEXAGON_DEVICES:-unset}"



    pp="NA"
    tg="NA"
    pp_rc=0
    tg_rc=0
    pp_reason="PASS"
    pp_error=""

    pp_out=/data/local/tmp/bench_pp_out.txt
    pp_err=/data/local/tmp/bench_pp_err.txt
    pp_fifo=/data/local/tmp/bench_pp_err.fifo

    rm -f "\$pp_out" "\$pp_err" "\$pp_fifo" "\$PROF_FILE"

    mkfifo "\$pp_fifo" || {
      echo \
        "PP=NA TG=NA PP_RC=1 TG_RC=SKIPPED" \
        "PP_REASON=FIFO_CREATE_FAILED PP_ERROR=mkfifo_failed"
      exit 1
    }

    # Drain stderr continuously. COMM-DIAG is excluded because retaining it
    # previously exhausted device memory. Only the final useful lines remain.
    # With profiling, stderr goes straight to \$PROF_FILE (device storage) with
    # no pipe chain: the host prints profile lines while popping DSP responses,
    # so any pipe backpressure stalls the NSPs. Filtered after the run.
    if prof_on; then
      cat < "\$pp_fifo" > "\$PROF_FILE" &
    else
      grep -v 'COMM-DIAG' < "\$pp_fifo" |
        tee "\$pp_err.verbose" |
        tail -n 4000 > "\$pp_err" &
    fi

    pp_filter_pid=\$!

    # Exit 137 is ambiguous: it can be an OOM kill or an unrelated SIGKILL.
    # Record journal and dmesg boundaries before PP so only an OOM emitted by
    # this invocation can change its failure classification. Journal is the
    # primary source: the ring-buffer delta can omit records under pressure.
    pp_journal_cursor=""
    if command -v journalctl >/dev/null 2>&1; then
      pp_journal_cursor=\$(
        journalctl --show-cursor -n 0 --no-pager 2>/dev/null |
          sed -n 's/^-- cursor: //p' |
          tail -1
      )
    fi

    pp_dmesg_available=0
    pp_dmesg_lines=0
    if dmesg >/dev/null 2>&1; then
      pp_dmesg_available=1
      pp_dmesg_lines=\$(dmesg 2>/dev/null | wc -l)
    fi

    pp_logcat_available=0
    pp_logcat_lines=0
    if [ "$PLATFORM" = "android" ] && logcat -b kernel -d >/dev/null 2>&1; then
      pp_logcat_available=1
      pp_logcat_lines=\$(logcat -b kernel -d -v brief 2>/dev/null | wc -l)
    fi

    env \$(hexenv) \$(profenv) setsid -w "\$BENCH" \
      -m "\$MODELS/\$m.gguf" \
      \$(devargslash \$n \$sp) \
      \$(smode \$n \$sp) \
      \$(prof_on && echo "-v") \
      -ngl 99 \
      -p "\$pp_tokens" \
      -n 0 \
      -d "\$pp_depth" \
      -ub "\$ub" \
      \$([ -n "\$BATCH" ] && echo "-b \$BATCH") \
      -r $REPS \
      > "\$pp_out" \
      2> "\$pp_fifo"

    pp_rc=\$?

    wait "\$pp_filter_pid" 2>/dev/null || true
    rm -f "\$pp_fifo"
    if prof_on; then
      grep -v -e 'COMM-DIAG' -e 'profile-op' -e 'trace-evt' "\$PROF_FILE" |
        tail -n 4000 > "\$pp_err"
    fi

    pp=\$(
      awk -F'|' -v pat="pp\$pp_tokens" '
        index(\$0, pat) {
          field = \$(NF - 1)
          gsub(/^[[:space:]]+/, "", field)
          gsub(/[[:space:]]+\$/, "", field)
          split(field, value, "±")
          gsub(/[[:space:]]/, "", value[1])
          print value[1]
          exit
        }
      ' "\$pp_out"
    )

    [ -n "\$pp" ] || pp="NA"

    pp_oom_error=""
    if [ "\$pp_rc" -ne 0 ] || [ "\$pp" = "NA" ]; then
      if [ "\$pp_logcat_available" -eq 1 ]; then
        pp_oom_error=\$(
          logcat -b kernel -d -v brief 2>/dev/null |
            sed -n "\$((pp_logcat_lines + 1)),\\\$p" |
            grep -iE 'oom-kill:|out of memory: killed process' |
            tail -1 |
            tr '\t\r\n ' '_____'
        )
      fi

      if [ -z "\$pp_oom_error" ] && [ -n "\$pp_journal_cursor" ]; then
        pp_oom_error=\$(
          journalctl --after-cursor="\$pp_journal_cursor" --no-pager -o short-precise 2>/dev/null |
            grep -iE 'oom-kill:|out of memory: killed process' |
            tail -1 |
            tr '\t\r\n ' '_____'
        )
      fi

      # Journal may be unavailable on stripped images; retain dmesg as a
      # fallback rather than degrading a confirmed OOM to generic exit 137.
      if [ -z "\$pp_oom_error" ] && [ "\$pp_dmesg_available" -eq 1 ]; then
      pp_oom_error=\$(
        dmesg 2>/dev/null |
          sed -n "\$((pp_dmesg_lines + 1)),\\\$p" |
          grep -iE 'oom-kill:|out of memory: killed process' |
          tail -1 |
          tr '\t\r\n ' '_____'
      )
      fi
    fi

    # FastRPC/HTP is named in normal backend startup diagnostics. A completed
    # benchmark with a parsed PP rate is therefore authoritative; only inspect
    # failure signatures when PP did not succeed. A concurrent OOM is only a
    # fallback: it may kill an unrelated process while this benchmark succeeds.
    if [ "\$pp_rc" -eq 0 ] && [ "\$pp" != "NA" ]; then
      pp_reason="PASS"
      pp_error=""
    elif grep -qiE \
      'op does not fit into empty batch|exceeds max_vmem' \
      "\$pp_err"
    then
      pp_reason="HTP_OP_MEMORY_OVERCLAIM"

      pp_error=\$(
        grep -iE \
          'op does not fit into empty batch|exceeds max_vmem' \
          "\$pp_err" |
        tail -1 |
        tr '\t\r\n ' '_____'
      )
    elif grep -qiE \
      'out of memory|ENOMEM|cannot allocate memory|allocation failed|failed to allocate' \
      "\$pp_err"
    then
      pp_reason="ALLOCATION_FAILURE"

      pp_error=\$(
        grep -iE \
          'out of memory|ENOMEM|cannot allocate memory|allocation failed|failed to allocate' \
          "\$pp_err" |
        tail -1 |
        tr '\t\r\n ' '_____'
      )
    elif grep -qiE \
      'fastrpc.*(fail|error|denied|timeout)|HAB.*(fail|error|denied|timeout)|vchan.*failed|remote.*failed|CDSP.*failed' \
      "\$pp_err"
    then
      pp_reason="FASTRPC_ERROR"

      pp_error=\$(
        grep -iE \
          'fastrpc.*(fail|error|denied|timeout)|HAB.*(fail|error|denied|timeout)|vchan.*failed|remote.*failed|CDSP.*failed' \
          "\$pp_err" |
        tail -1 |
        tr '\t\r\n ' '_____'
      )
    elif [ "\$pp_rc" -eq 124 ]; then
      pp_reason="TIMEOUT"
    elif [ -n "\$pp_oom_error" ]; then
      pp_reason="HOST_OOM_KILL"
      pp_error="\$pp_oom_error"
    elif [ "\$pp_rc" -eq 137 ]; then
      pp_reason="SIGKILL_OR_OOM"
    elif [ "\$pp_rc" -ne 0 ]; then
      pp_reason="PROCESS_EXIT_\$pp_rc"
    elif [ "\$pp" = "NA" ]; then
      pp_reason="PARSE_MISSING"
    fi

    tg_out=/data/local/tmp/bench_tg_out.txt
    tg_err=/data/local/tmp/bench_tg_err.txt
    tg_fifo=/data/local/tmp/bench_tg_err.fifo

    rm -f "\$tg_out" "\$tg_err" "\$tg_fifo"

    mkfifo "\$tg_fifo" || {
      echo \
        "PP=\$pp TG=NA PP_RC=\$pp_rc TG_RC=1" \
        "PP_REASON=PASS PP_ERROR=none"
      exit 1
    }

    grep -v 'COMM-DIAG' < "\$tg_fifo" |
      tee "\$tg_err.verbose" |
      tail -n 4000 > "\$tg_err" &

    tg_filter_pid=\$!

    env \$(hexenv) setsid -w "\$BENCH" \
      -m "\$MODELS/\$m.gguf" \
      \$(devargslash \$n \$sp) \
      \$(smode \$n \$sp) \
      -ngl 99 \
      -p 0 \
      -n 256 \
      -ub "\$ub" \
      -d "\$tg_depth" \
      \$([ -n "\$BATCH" ] && echo "-b \$BATCH") \
      -r $REPS \
      > "\$tg_out" \
      2> "\$tg_fifo"

    tg_rc=\$?

    wait "\$tg_filter_pid" 2>/dev/null || true
    rm -f "\$tg_fifo"

    tg=\$(
      awk -F'|' '
        /tg256/ {
          field = \$(NF - 1)
          gsub(/^[[:space:]]+/, "", field)
          gsub(/[[:space:]]+\$/, "", field)
          split(field, value, "±")
          gsub(/[[:space:]]/, "", value[1])
          print value[1]
          exit
        }
      ' "\$tg_out"
    )

    [ -n "\$tg" ] || tg="NA"

    [ -f "\$pp_err" ] && grep '^MMGLU ' "\$pp_err" | sort -u
    echo \
      "PP=\$pp TG=\$tg" \
      "PP_RC=\$pp_rc TG_RC=\$tg_rc" \
      "PP_REASON=\$pp_reason" \
      "PP_ERROR=\${pp_error:-none}"

    [ "\$tg_rc" -eq 0 ] || exit "\$tg_rc"
    ;;


  # iq <model> <cores> <split> <abval>
  #   Emits: "IQ=<score>/10"
  #   Ten short questions with unambiguous answers, greedy (--temp 0), scored by
  #   case-insensitive substring match. Three use a one-shot example to pin the
  #   output format. Absolute scores run low because the scorer is strict (it
  #   does not strip <think> blocks or accept word-form numbers) — what matters
  #   is that a candidate config scores >= the baseline at the same core count.
  #   Also emits one "IQD=..." line per question (expected/got/match/prompt) so
  #   the caller can save the full transcript, not just the tally.
  iq)
    m=\$1; n=\$2; sp=\$3; ab=\$4
    GGML_HEXAGON_DEVICES=\$(hexdevices \$n \$sp)
    set_scatter "\$sp"
    set_ab "\$ab"
    SYS="You are a helpful assistant. Answer concisely."
    i=1; correct=0
    while [ \$i -le 10 ]; do
      case \$i in
        1) P="What is 17 + 26? Answer with just the number."; E="43" ;;
        2) P="What is 8 multiplied by 7? Answer with just the number."; E="56" ;;
        3) P="What is 100 minus 37? Answer with just the number."; E="63" ;;
        4) P="The capital of Italy is Rome.
The capital of Japan is"; E="Tokyo" ;;
        5) P="The chemical symbol for oxygen is O.
The chemical symbol for gold is"; E="Au" ;;
        6) P="Complete the sequence: 2, 4, 6, 8,"; E="10" ;;
        7) P="Roses are red, violets are"; E="blue" ;;
        8) P="The largest planet in our solar system is"; E="Jupiter" ;;
        9) P="How many days are in a week? Answer with just the number."; E="7" ;;
       10) P="The opposite of \\"big\\" is small.
The opposite of \\"hot\\" is"; E="cold" ;;
      esac
      raw=\$( { timeout \$(qtmo \$m) env \$(hexenv) setsid -w \$COMP -m \$MODELS/\$m.gguf \$(devarg \$n \$sp) \$(smode \$n \$sp) \\
              -c \$CTX -ngl 99 -no-cnv --temp 0 -sys "\$SYS" -p "\$P" -n 32; } \\
              2>>$([ "${HEXAGON_VERBOSE:-0}" = 1 ] && echo /data/local/tmp/iq_verbose.log || echo /dev/null) \\
            | tr -d '\r' | tr '\n' ' ')
      case "\$raw" in *"</think>"*) ans=\${raw##*</think>} ;; *) ans=\$raw ;; esac
      out=\$(printf '%s' "\$ans" | tr 'A-Z' 'a-z')
      el=\$(printf '%s' "\$E" | tr 'A-Z' 'a-z')
      m_flag="NO"
      case "\$out" in *"\$el"*) correct=\$((correct+1)); m_flag="YES" ;; esac
      pflat=\$(printf '%s' "\$P" | tr '\n' ' ')
      echo "IQD=\$i|EXPECTED=\$E|GOT=\$raw|MATCH=\$m_flag|PROMPT=\$pflat"
      i=\$((i+1))
    done
    echo "IQ=\$correct/10"
    [ "\$GGML_HEXAGON_VERBOSE" = 1 ] && journalctl -b --no-pager -o short-precise 2>/dev/null | grep -iE 'farf|adsprpc|fastrpc|cdsp|hexagon|htp|hab:' > /data/local/tmp/iq_journal.log || true
    ;;

  # cookie <model> <cores> <split> <abval>
  #   Emits the raw generation on one line, prefixed "COOKIE=".
  #   Open-ended greedy decode. This is the check that catches partially-stale
  #   allreduce output or other subtle corruption: the text stays grammatical
  #   but drifts off-topic or collapses into repetition. Judged by comparing
  #   against the baseline config at the same model/core count.
  cookie)
    m=\$1; n=\$2; sp=\$3; ab=\$4
    GGML_HEXAGON_DEVICES=\$(hexdevices \$n \$sp)
    set_scatter "\$sp"
    set_ab "\$ab"
    printf '%s\n\n%s\n' 'You are a helpful assistant. Answer concisely.' 'What is the best cookie in the world?' > /data/local/tmp/cookie.txt
    rm -f /data/local/tmp/cookie_rc
    out=\$( { timeout \$(qtmo \$m) env \$(hexenv) setsid -w \$COMP -m \$MODELS/\$m.gguf \$(devarg \$n \$sp) \$(smode \$n \$sp) \\
            -c \$CTX -ngl 99 -no-cnv --temp 0 -f /data/local/tmp/cookie.txt -n 128 \\
            2>>$([ "${HEXAGON_VERBOSE:-0}" = 1 ] && echo /data/local/tmp/cookie_verbose.log || echo /dev/null); echo \$? > /data/local/tmp/cookie_rc; } \\
          | tr -d '\r' | tr '\n' ' ')
    echo "COOKIE=\$out"
    echo "COOKIE_RC=\$(cat /data/local/tmp/cookie_rc 2>/dev/null || echo 1)"
    [ "\$GGML_HEXAGON_VERBOSE" = 1 ] && journalctl -b --no-pager -o short-precise 2>/dev/null | grep -iE 'farf|adsprpc|fastrpc|cdsp|hexagon|htp|hab:' > /data/local/tmp/cookie_journal.log || true
    ;;

  # needle <model> <cores> <split> <abval>
  #   Emits: "NEEDLE=PASS|FAIL"
  #   Long-context recall: a 1k-token prompt with an identifier buried at 50%
  #   depth. A correct run echoes it. Directly exercises the attention/KV path,
  #   so it is the most sensitive of the checks to a bad cross-core reduction or
  #   any change touching context handling.
  needle)
    m=\$1; n=\$2; sp=\$3; ab=\$4
    GGML_HEXAGON_DEVICES=\$(hexdevices \$n \$sp)
    set_scatter "\$sp"
    set_ab "\$ab"
    out=\$(timeout \$(( \$(qtmo \$m) + 60 )) env \$(hexenv) setsid -w \$COMP -m \$MODELS/\$m.gguf \$(devarg \$n \$sp) \$(smode \$n \$sp) \\
            -c \$CTX -ngl 99 -no-cnv -f \$PROMPT -n 128 2>/dev/null | tr -d '\r')
    case "\$out" in *"$NEEDLE"*) echo "NEEDLE=PASS" ;; *) echo "NEEDLE=FAIL" ;; esac
    ;;

  # testops <cores> <abval> <op> [pattern]
  #   Runs test-backend-ops against the (row-grouped) HTP0 device vs CPU.
  #   Emits the tool's output plus "TESTOPS_RC=<rc>".
  testops)
    n=\$1; ab=\$2; op=\$3; pat=\${4:-}
    GGML_HEXAGON_DEVICES=\$(hexdevices \$n row)
    set_ab "\$ab"
    echo "HARNESS_ENV testops cores=\$n ab=\$ab op=\$op GGML_HEXAGON_DEVICES=\$GGML_HEXAGON_DEVICES"
    (
      if [ -n "\$pat" ]; then
        timeout 3600 env \$(hexenv) setsid -w \$PKG/bin/test-backend-ops test -b HTP0 -o "\$op" -p "\$pat"
      else
        timeout 3600 env \$(hexenv) setsid -w \$PKG/bin/test-backend-ops test -b HTP0 -o "\$op"
      fi
      echo "TESTOPS_RC=\$?"
    ) 2>&1 | tr -d '\r'
    ;;

  *) echo "usage: llama.cpp-build-and-test-dev.sh <bench|iq|cookie|needle> <model> <cores> <split> <abval> [ubatch] | testops <cores> <abval> <op> [pattern]" >&2; exit 2 ;;
esac
DEVEOF

  $ADB push "$OUTDIR/llama.cpp-build-and-test-dev.sh" "$DEV_DRIVER" >/dev/null || die "driver push failed"
  $ADB shell "chmod 755 $DEV_DRIVER"
}

# Optional per-test memory and failure diagnostics. The sampler output is stored
# on the host. This avoids the device page-cache OOM caused by large device logs.
mem_safe_label() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'; }

# Defense-in-depth against row-label collisions: every per-row log filename is
# built from the loop variables (model/cores/split/ab/ubatch/depth) and is
# expected to already be unique, but if a future sweep dimension is ever added
# without updating every label site (see the ub512-fallback bug this fixed),
# a silent overwrite would otherwise destroy an earlier row's only copy of its
# diagnostics. Given "$1", print back a path that doesn't yet exist, appending
# ".2", ".3", ... before the final extension if needed.
unique_log_path() {
  local path=$1
  [ -e "$path" ] || { printf '%s\n' "$path"; return; }
  local base=$path ext=""
  case "$path" in
    *.*) base=${path%.*}; ext=".${path##*.}" ;;
  esac
  local n=2
  while [ -e "${base}.${n}${ext}" ]; do n=$((n + 1)); done
  printf '%s\n' "${base}.${n}${ext}"
}

mem_capture_snapshot() {
  local out=$1
  $ADB shell '
    echo "===== date ====="; date -Ins 2>/dev/null || date
    echo "===== uptime ====="; cat /proc/uptime
    echo "===== free ====="; free -b 2>/dev/null || free -k
    echo "===== meminfo ====="; cat /proc/meminfo
    echo "===== pressure ====="; cat /proc/pressure/memory 2>/dev/null || true
    echo "===== vmstat ====="; cat /proc/vmstat
    echo "===== buddyinfo ====="; cat /proc/buddyinfo 2>/dev/null || true
    echo "===== dma-buf total ====="; tail -1 /sys/kernel/debug/dma_buf/bufinfo 2>/dev/null || true
    echo "===== large dma-bufs >= 100 MiB ====="
    awk '\''/^[0-9]/ && $1 >= 104857600 { print }'\'' /sys/kernel/debug/dma_buf/bufinfo 2>/dev/null || true
    echo "===== cgroup ====="; cat /proc/self/cgroup 2>/dev/null || true
  ' > "$out" 2>&1 || true
}

# Pushed once per run (lazily, on first use) and invoked with a short
# `sh $MEM_SAMPLER_DEV <process_name>` each sampling tick. The sampler logic
# used to be inlined directly into the per-tick `adb shell "..."` command,
# but that string exceeded adb's shell service-name length limit — every
# single sample silently failed with
# `error: service name too long`, which is why peak-used/min-avail/etc always
# printed as -1/-1/0/0 regardless of how long a case actually ran. A real
# on-device file has no such length limit, since the command invoking it is
# always short.
#
# Guarded by a file on disk, not a shell variable: every caller
# (monitored_dev -> mem_monitor_start -> write_mem_sampler) runs inside a
# `$(...)` command substitution, i.e. a subshell, so a plain variable flag
# set here never survives back to the parent shell and would silently push
# on every single case instead of once per run.
MEM_SAMPLER_PUSHED_MARKER="$OUTDIR/.mem-sampler-pushed"
write_mem_sampler() {
  [ -f "$MEM_SAMPLER_PUSHED_MARKER" ] && return 0
  cat > "$OUTDIR/llama.cpp-mem-sample.sh" <<'SAMPEOF'
#!/bin/sh
# Auto-generated by llama.cpp-build-and-test.sh — device-side memory sampler.
# usage: sh llama.cpp-mem-sample.sh <process_name>
process_name=$1
printf 'ts='; date +%s.%N 2>/dev/null || date +%s
awk '
  /^MemTotal:/     { printf " MemTotal_kB=%s", $2 }
  /^MemFree:/      { printf " MemFree_kB=%s", $2 }
  /^MemAvailable:/ { printf " MemAvailable_kB=%s", $2 }
  /^Buffers:/      { printf " Buffers_kB=%s", $2 }
  /^Cached:/       { printf " Cached_kB=%s", $2 }
  /^AnonPages:/    { printf " AnonPages_kB=%s", $2 }
  /^Mapped:/       { printf " Mapped_kB=%s", $2 }
  /^Slab:/         { printf " Slab_kB=%s", $2 }
  /^SUnreclaim:/   { printf " SUnreclaim_kB=%s", $2 }
  /^PageTables:/   { printf " PageTables_kB=%s", $2 }
  /^CmaFree:/      { printf " CmaFree_kB=%s", $2 }
' /proc/meminfo
awk '
  /^some / { printf " PSI_some_avg10=%s", substr($2,7) }
  /^full / { printf " PSI_full_avg10=%s", substr($2,7) }
' /proc/pressure/memory 2>/dev/null || true
set -- $(tail -1 /sys/kernel/debug/dma_buf/bufinfo 2>/dev/null)
[ "$1" = Total ] && printf ' Dmabuf_objects=%s Dmabuf_bytes=%s' "$2" "$4"
pid=$(pidof "$process_name" 2>/dev/null | awk '{print $1}')
if [ -n "$pid" ] && [ -r /proc/$pid/status ]; then
  printf ' pid=%s' "$pid"
  awk '
    /^VmPeak:/   { printf " VmPeak_kB=%s", $2 }
    /^VmSize:/   { printf " VmSize_kB=%s", $2 }
    /^VmRSS:/    { printf " VmRSS_kB=%s", $2 }
    /^RssAnon:/  { printf " RssAnon_kB=%s", $2 }
    /^RssFile:/  { printf " RssFile_kB=%s", $2 }
    /^RssShmem:/ { printf " RssShmem_kB=%s", $2 }
    /^VmPTE:/    { printf " VmPTE_kB=%s", $2 }
    /^VmSwap:/   { printf " VmSwap_kB=%s", $2 }
  ' /proc/$pid/status
fi
echo
SAMPEOF
  $ADB push "$OUTDIR/llama.cpp-mem-sample.sh" "$MEM_SAMPLER_DEV" >/dev/null \
    || die "mem sampler push failed"
  $ADB shell "chmod 755 $MEM_SAMPLER_DEV"
  : > "$MEM_SAMPLER_PUSHED_MARKER"
}

mem_monitor_start() {
  local label=$1 process_name=$2
  [ "$COLLECT_MEMORY" = 1 ] || return 0

  write_mem_sampler
  MEM_CUR_LABEL=$(mem_safe_label "$label")
  MEM_CUR_DIR="$MEMORY_LOGS/$MEM_CUR_LABEL"
  # Same collision guard as unique_log_path(): if this label's memory/ dir
  # already exists and has prior artifacts in it (e.g. a sweep dimension
  # collided, or a label was reused), append a numeric suffix instead of
  # silently mixing/overwriting this row's files into the earlier row's dir.
  if [ -d "$MEM_CUR_DIR" ] && [ -n "$(ls -A "$MEM_CUR_DIR" 2>/dev/null)" ]; then
    local n=2
    while [ -d "${MEM_CUR_DIR}.${n}" ] && [ -n "$(ls -A "${MEM_CUR_DIR}.${n}" 2>/dev/null)" ]; do
      n=$((n + 1))
    done
    MEM_CUR_LABEL="${MEM_CUR_LABEL}.${n}"
    MEM_CUR_DIR="${MEM_CUR_DIR}.${n}"
  fi
  MEM_CUR_PROCESS=$process_name
  mkdir -p "$MEM_CUR_DIR"

  MEM_DMESG_LINES=$($ADB shell 'dmesg 2>/dev/null | wc -l' 2>/dev/null | tr -d '\r[:space:]')
  case "$MEM_DMESG_LINES" in ''|*[!0-9]*) MEM_DMESG_LINES=0 ;; esac

  MEM_LOGCAT_KERNEL_LINES=0
  if [ "$PLATFORM" = "android" ]; then
    MEM_LOGCAT_KERNEL_LINES=$($ADB shell 'logcat -b kernel -d -v brief 2>/dev/null | wc -l' 2>/dev/null | tr -d '\r[:space:]')
    case "$MEM_LOGCAT_KERNEL_LINES" in ''|*[!0-9]*) MEM_LOGCAT_KERNEL_LINES=0 ;; esac
  fi

  MEM_JOURNAL_CURSOR=$($ADB shell 'journalctl --show-cursor -n 0 --no-pager 2>/dev/null | sed -n "s/^-- cursor: //p"' 2>/dev/null | tr -d '\r' | tail -1)

  mem_capture_snapshot "$MEM_CUR_DIR/before.txt"

  # One-time process map collector. It waits until the target process appears.
  if [ "$MEMORY_PROCESS_MAP" = 1 ]; then
    (
      local tries=0
      while [ "$tries" -lt 600 ]; do
        $ADB shell "
          pid=\$(pidof '$process_name' 2>/dev/null | awk '{print \$1}')
          if [ -n \"\$pid\" ] && [ -r /proc/\$pid/status ]; then
            echo pid=\$pid
            echo '===== status ====='; cat /proc/\$pid/status
            echo '===== smaps_rollup ====='; cat /proc/\$pid/smaps_rollup 2>/dev/null || true
            echo '===== maps ====='; cat /proc/\$pid/maps 2>/dev/null || true
            echo '===== fdinfo ====='; grep -H . /proc/\$pid/fdinfo/* 2>/dev/null || true
            exit 0
          fi
          exit 1
        " 2>/dev/null | tr -d '\r' && exit 0
        tries=$((tries + 1))
        sleep 0.1
      done
    ) > "$MEM_CUR_DIR/process-map.txt" 2>&1 &
    MEM_MAP_PID=$!
  else
    MEM_MAP_PID=""
  fi

  (
    while :; do
      $ADB shell "sh $MEM_SAMPLER_DEV '$process_name'" 2>/dev/null | tr -d '\r'
      sleep "$MEMORY_INTERVAL"
    done
  ) > "$MEM_CUR_DIR/timeline.log" 2>&1 &
  MEM_MON_PID=$!
}

mem_monitor_analyze() {
  local dir=$1
  local label=$2
  local rc=$3
  local summary="$dir/summary.txt"
  awk '
    function val(field, i,a) {
      for (i=1; i<=NF; i++) if ($i ~ ("^" field "=")) { split($i,a,"="); return a[2]+0 }
      return -1
    }
    BEGIN { minavail=-1; minfree=-1 }
    {
      total=val("MemTotal_kB"); avail=val("MemAvailable_kB"); freev=val("MemFree_kB")
      rss=val("VmRSS_kB"); anon=val("RssAnon_kB"); pte=val("PageTables_kB")
      dmab=val("Dmabuf_bytes"); dmao=val("Dmabuf_objects"); psi=val("PSI_full_avg10")
      if (total > maxtotal) maxtotal=total
      if (avail >= 0 && (minavail < 0 || avail < minavail)) minavail=avail
      if (freev >= 0 && (minfree < 0 || freev < minfree)) minfree=freev
      if (rss > maxrss) maxrss=rss
      if (anon > maxanon) maxanon=anon
      if (pte > maxpte) maxpte=pte
      if (dmab > maxdmab) maxdmab=dmab
      if (dmao > maxdmao) maxdmao=dmao
      if (psi > maxpsi) maxpsi=psi
    }
    END {
      peakused=(maxtotal > 0 && minavail >= 0) ? maxtotal-minavail : -1
      printf "peak_system_used_kB=%d\n", peakused
      printf "min_MemAvailable_kB=%d\n", minavail
      printf "min_MemFree_kB=%d\n", minfree
      printf "peak_process_VmRSS_kB=%d\n", maxrss
      printf "peak_process_RssAnon_kB=%d\n", maxanon
      printf "peak_PageTables_kB=%d\n", maxpte
      printf "peak_Dmabuf_bytes=%.0f\n", maxdmab
      printf "peak_Dmabuf_objects=%d\n", maxdmao
      printf "peak_PSI_full_avg10=%.2f\n", maxpsi
    }
  ' "$dir/timeline.log" > "$summary"
  # The kernel ring-buffer delta can lose OOM lines under extreme pressure.
  # Android's logcat kernel buffer and OE Linux's journal cursor are captured
  # before each case and are therefore authoritative when they contain OOMs.
  local oom oom_source=none oom_log=""
  if [ "$PLATFORM" = "android" ] &&
       grep -qiE 'oom-kill:|out of memory: killed process' "$dir/logcat-kernel-new.txt" 2>/dev/null; then
    oom_source=logcat-kernel
    oom_log="$dir/logcat-kernel-new.txt"
  elif grep -qiE 'oom-kill:|out of memory: killed process' "$dir/journal.txt" 2>/dev/null; then
    oom_source=journal
    oom_log="$dir/journal.txt"
  elif grep -qiE 'oom-kill:|out of memory: killed process' "$dir/kernel-focused.txt" 2>/dev/null; then
    oom_source=dmesg
    oom_log="$dir/kernel-focused.txt"
  fi
  if [ -n "$oom_log" ]; then
    oom=$(grep -ciE 'oom-kill:|out of memory: killed process' "$oom_log" 2>/dev/null || true)
  else
    oom=0
  fi
  printf 'oom_kernel_lines=%s\noom_source=%s\nexit_code=%s\n' "${oom:-0}" "$oom_source" "$rc" >> "$summary"

  local peak used avail freev rss anon pte dmab dmao psi
  peak=$(sed -n 's/^peak_system_used_kB=//p' "$summary")
  avail=$(sed -n 's/^min_MemAvailable_kB=//p' "$summary")
  freev=$(sed -n 's/^min_MemFree_kB=//p' "$summary")
  rss=$(sed -n 's/^peak_process_VmRSS_kB=//p' "$summary")
  anon=$(sed -n 's/^peak_process_RssAnon_kB=//p' "$summary")
  pte=$(sed -n 's/^peak_PageTables_kB=//p' "$summary")
  dmab=$(sed -n 's/^peak_Dmabuf_bytes=//p' "$summary")
  dmao=$(sed -n 's/^peak_Dmabuf_objects=//p' "$summary")
  psi=$(sed -n 's/^peak_PSI_full_avg10=//p' "$summary")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$rc" "$peak" "$avail" "$freev" "$rss" "$anon" "$pte" "$dmab" "$dmao" "$psi" "${oom:-0}" "$oom_source" >> "$MEMORY_TSV"

  say "    ${C}memory[$label]: peak-used=${peak}kB min-avail=${avail}kB rss-peak=${rss}kB dma-peak=${dmab}B oom-lines=${oom:-0}${N}" >&2
}

mem_monitor_stop() {
  local rc=${1:-0}
  [ "$COLLECT_MEMORY" = 1 ] || return 0
  kill "${MEM_MON_PID:-}" 2>/dev/null || true
  wait "${MEM_MON_PID:-}" 2>/dev/null || true
  if [ -n "${MEM_MAP_PID:-}" ]; then
    kill "$MEM_MAP_PID" 2>/dev/null || true
    wait "$MEM_MAP_PID" 2>/dev/null || true
  fi
  mem_capture_snapshot "$MEM_CUR_DIR/after.txt"

  local from=$((MEM_DMESG_LINES + 1))
  $ADB shell "dmesg 2>/dev/null | sed -n '${from},\$p'" > "$MEM_CUR_DIR/dmesg-new.txt" 2>&1 || true
  grep -iE 'oom|out of memory|killed process|page allocation failure|cma|dma|ion|fastrpc|cdsp|hab:|vhost|smmu|iommu|blocked for more|rcu_preempt|BUG:|WARNING:' \
    "$MEM_CUR_DIR/dmesg-new.txt" > "$MEM_CUR_DIR/kernel-focused.txt" 2>/dev/null || true

  if [ "$PLATFORM" = "android" ]; then
    local logcat_from=$((MEM_LOGCAT_KERNEL_LINES + 1))
    $ADB shell "logcat -b kernel -d -v brief 2>/dev/null | sed -n '${logcat_from},\$p'" \
      > "$MEM_CUR_DIR/logcat-kernel-new.txt" 2>&1 || true
    grep -iE 'oom|out of memory|killed process|page allocation failure|fastrpc|cdsp|hexagon|htp|hab:|smmu|iommu|BUG:|WARNING:' \
      "$MEM_CUR_DIR/logcat-kernel-new.txt" > "$MEM_CUR_DIR/logcat-kernel-focused.txt" 2>/dev/null || true
  elif [ -n "${MEM_JOURNAL_CURSOR:-}" ]; then
    $ADB shell "journalctl --after-cursor='${MEM_JOURNAL_CURSOR}' --no-pager -o short-precise 2>/dev/null" > "$MEM_CUR_DIR/journal.txt" 2>&1 || true
  else
    $ADB shell 'journalctl -b -n 1000 --no-pager -o short-precise 2>/dev/null' > "$MEM_CUR_DIR/journal.txt" 2>&1 || true
  fi
  grep -iE 'farf|adsprpc|fastrpc|cdsp|hexagon|htp|hab:' "$MEM_CUR_DIR/journal.txt" > "$MEM_CUR_DIR/farf-focused.txt" 2>/dev/null || true

  # Core-dump metadata only; never copy large core images automatically.
  $ADB shell "coredumpctl --no-pager list --since '@$(date +%s -d '1 hour ago' 2>/dev/null || echo 0)' 2>/dev/null || true" > "$MEM_CUR_DIR/coredump-list.txt" 2>&1 || true
  $ADB shell 'find /data/local/tmp /var/lib/systemd/coredump -maxdepth 3 -type f \( -name "core*" -o -name "*.core" -o -name "*.zst" \) -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" 2>/dev/null | tail -100' > "$MEM_CUR_DIR/core-files.txt" 2>&1 || true

  mem_monitor_analyze "$MEM_CUR_DIR" "$MEM_CUR_LABEL" "$rc"
}

# pull_profile <label> <action> — with profiling on, copy the full PP stderr of a
# bench case to $OUTDIR/profile/<label>.log and delete it on the device (it can
# be tens of MB). Profiling is on for --hex-profile N, or when the case's A/B
# value of GGML_HEXAGON_PROFILE is non-zero (the device driver decides; we just
# pull whatever file it left behind).
pull_verbose() {
  local label=$1 action=${2:-}
  [ "${HEXAGON_VERBOSE:-0}" != 1 ] && return 0
  case "$action" in
    bench)
      $ADB pull /data/local/tmp/bench_pp_err.verbose "$LOGS/${label}-pp.verbose.log" >/dev/null 2>&1 || true
      $ADB pull /data/local/tmp/bench_tg_err.verbose "$LOGS/${label}-tg.verbose.log" >/dev/null 2>&1 || true
      ;;
    iq)
      $ADB pull /data/local/tmp/iq_verbose.log  "$LOGS/${label}.verbose.log"  >/dev/null 2>&1 || true
      $ADB pull /data/local/tmp/iq_journal.log  "$LOGS/${label}.journal.log"  >/dev/null 2>&1 || true
      ;;
    cookie)
      $ADB pull /data/local/tmp/cookie_verbose.log  "$LOGS/${label}.verbose.log"  >/dev/null 2>&1 || true
      $ADB pull /data/local/tmp/cookie_journal.log  "$LOGS/${label}.journal.log"  >/dev/null 2>&1 || true
      ;;
  esac
}

pull_profile() {
  local label=$1 action=$2
  [ "$action" = "bench" ] || return 0
  [ "$HEX_PROFILE" != 0 ] || [ "$AB_VAR" = "GGML_HEXAGON_PROFILE" ] || return 0
  local dir="$OUTDIR/profile" dst
  mkdir -p "$dir"
  dst="$dir/$(mem_safe_label "$label").log"
  if $ADB pull /data/local/tmp/bench_pp_prof.txt "$dst" >/dev/null 2>&1; then
    local nl; nl=$(wc -l < "$dst" 2>/dev/null | tr -d ' ')
    # gzip so profiles stay far below GitHub's 100 MB per-file limit
    gzip -9f "$dst" && dst="$dst.gz"
    # split anything still too big into 90 MB parts (cat <file>.gz.part* to rejoin)
    if [ "$(wc -c < "$dst")" -gt 94000000 ]; then
      split -b 90m -d -a 2 "$dst" "$dst.part" && rm -f "$dst" && dst="$dst.part*"
    fi
    say "    ${C}profile[$label]: $nl lines -> $dst${N}" >&2
  fi
  $ADB shell "rm -f /data/local/tmp/bench_pp_prof.txt" >/dev/null 2>&1 || true
}

monitored_dev() {
  local label=$1 process_name=$2; shift 2
  if [ "$COLLECT_MEMORY" != 1 ]; then
    local out rc
    out=$(run_dev_streamed "$label" "$@"); rc=$?
    pull_profile "$label" "${1:-}"
    pull_verbose "$label" "${1:-}"
    cleanup_device_workloads "$label" || return 1
    if case_had_kernel_oom "$out"; then
      recover_cleanup_failure "case=$label; confirmed kernel OOM requires a clean memory state" || return 1
    fi
    printf '%s\n' "$out"
    return "$rc"
  fi
  mem_monitor_start "$label" "$process_name"
  local out rc
  out=$(run_dev_streamed "$label" "$@"); rc=$?

  local action=${1:-}

  if [ "$action" = "bench" ]; then
    printf '%s
' "$out"       > "$MEM_CUR_DIR/device-handler-output.txt"

    $ADB pull       /data/local/tmp/bench_pp_out.txt       "$MEM_CUR_DIR/benchmark-pp-stdout.txt"       >/dev/null 2>&1 || true

    $ADB pull       /data/local/tmp/bench_pp_err.txt       "$MEM_CUR_DIR/benchmark-pp-stderr.txt"       >/dev/null 2>&1 || true

    $ADB pull       /data/local/tmp/bench_tg_out.txt       "$MEM_CUR_DIR/benchmark-tg-stdout.txt"       >/dev/null 2>&1 || true

    $ADB pull       /data/local/tmp/bench_tg_err.txt       "$MEM_CUR_DIR/benchmark-tg-stderr.txt"       >/dev/null 2>&1 || true
  fi

  pull_profile "$label" "$action"
  pull_verbose "$label" "$action"
  mem_monitor_stop "$rc"
  cleanup_device_workloads "$label" || return 1
  if case_had_kernel_oom "$out"; then
    recover_cleanup_failure "case=$label; confirmed kernel OOM requires a clean memory state" || return 1
  fi
  printf '%s\n' "$out"
  return "$rc"
}

write_run_config() {
  {
    echo "===== RUN IDENTITY ====="

    printf 'timestamp='
    date -Ins 2>/dev/null || date

    printf 'host='
    hostname 2>/dev/null || true

    printf 'user='
    id -un 2>/dev/null || true

    printf 'working_directory='
    pwd

    echo
    echo "===== EXACT SCRIPT INVOCATION ====="

    printf 'command='
    printf '%q ' "$0" "${ORIGINAL_ARGV[@]}"
    echo

    echo
    echo "===== GIT ====="

    printf 'branch='
    git branch --show-current 2>/dev/null || true

    printf 'commit='
    git rev-parse HEAD 2>/dev/null || true

    printf 'commit_short='
    git rev-parse --short HEAD 2>/dev/null || true

    printf 'describe='
    git describe --always --dirty 2>/dev/null || true

    echo
    echo "===== RESOLVED MATRIX CONFIGURATION ====="

    printf 'ADB_SERIAL=%q\n' "$ADB_SERIAL"
    printf 'ADB_SERVER_SOCKET=%q\n' "${ADB_SERVER_SOCKET:-}"
    printf 'PLATFORM=%q\n' "$PLATFORM"
    printf 'PDU_OFFLINE_WAIT=%q\n' "$PDU_OFFLINE_WAIT"
    printf 'CORES=%q\n' "$CORES"
    printf 'SPLITS=%q\n' "$SPLITS"
    printf 'UBATCH=%q\n' "$UBATCH"
    printf 'BATCH=%q\n' "$BATCH"
    printf 'MODELS=%q\n' "$MODELS"
    printf 'REPS=%q\n' "$REPS"
    printf 'HEX_PROFILE=%q\n' "$HEX_PROFILE"
    printf 'PP_DEPTH=%q\n' "$PP_DEPTH"
    printf 'TG_DEPTH=%q\n' "$TG_DEPTH"
    printf 'PP_TOKENS=%q\n' "$PP_TOKENS"
    printf 'CTX=%q\n' "$CTX"
    printf 'AB_VAR=%q\n' "$AB_VAR"
    printf 'AB=%q\n' "$AB"

    printf 'DO_BUILD=%q\n' "$DO_BUILD"
    printf 'DO_DEPLOY=%q\n' "$DO_DEPLOY"
    printf 'REBOOT_BEFORE_TEST=%q\n' "$REBOOT_BEFORE_TEST"
    printf 'DO_CLEAN=%q\n' "$DO_CLEAN"
    printf 'DO_BENCH=%q\n' "$DO_BENCH"
    printf 'DO_IQ=%q\n' "$DO_IQ"
    printf 'DO_COOKIE=%q\n' "$DO_COOKIE"
    printf 'DO_NEEDLE=%q\n' "$DO_NEEDLE"
    printf 'DO_PP=%q\n' "${DO_PP:-1}"
    printf 'DO_TG=%q\n' "${DO_TG:-1}"

    printf 'COLLECT_MEMORY=%q\n' "${COLLECT_MEMORY:-0}"
    printf 'MEMORY_INTERVAL=%q\n' "${MEMORY_INTERVAL:-1}"
    printf 'MEMORY_PROCESS_MAP=%q\n' \
      "${MEMORY_PROCESS_MAP:-1}"

    printf 'PKG_HOST=%q\n' "$PKG_HOST"
    printf 'PKG_DEV=%q\n' "$PKG_DEV"
    printf 'MODELS_DEV=%q\n' "$MODELS_DEV"
    printf 'PROMPT_DEV=%q\n' "$PROMPT_DEV"
    printf 'OUTDIR=%q\n' "$OUTDIR"

    echo
    echo "===== EFFECTIVE TEST ENVIRONMENT ====="

    env |
      LC_ALL=C sort |
      grep -E \
        '^(ADB_SERIAL|AB|AB_VAR|CORES|SPLITS|UBATCH|MODELS|REPS|PP_DEPTH|TG_DEPTH|CTX|COLLECT_MEMORY|MEMORY_INTERVAL|HEX_PROFILE|GGML_|LLAMA_|HEXAGON_|CDSP|ADSP_)=' \
      || true

    echo
    echo "===== ADB COMMAND ====="

    printf 'ADB='
    printf '%q ' "${ADB[@]}"
    echo

    echo
    echo "===== DEVICE IDENTITY ====="

    $ADB shell '
      printf "hostname="
      hostname 2>/dev/null || true

      printf "serial="
      getprop ro.serialno 2>/dev/null || true

      printf "kernel="
      uname -a 2>/dev/null || true

      printf "uptime="
      cat /proc/uptime 2>/dev/null || true
    ' 2>/dev/null |
      tr -d '\r' || true

    echo
    echo "===== INSTALLED DEVICE DRIVER ENVIRONMENT ====="

    $ADB shell "
      if [ -r '$DEV_DRIVER' ]; then
        grep -n -E \
          '^(export )?(LD_LIBRARY_PATH|ADSP_LIBRARY_PATH|CDSP[0-9]*_LIBRARY_PATH|GGML_|LLAMA_|HEXAGON_)' \
          '$DEV_DRIVER' || true
      else
        echo 'device driver not installed'
      fi
    " 2>/dev/null |
      tr -d '\r' || true
  } > "$RUN_CONFIG_LOG"
}

log_device_invocation() {
  {
    echo "===== DEVICE INVOCATION ====="
    printf 'timestamp='
    date -Ins 2>/dev/null || date

    printf 'host_command='
    printf '%q ' "$ADB" "shell" "sh $DEV_DRIVER $*"
    echo

    printf 'device_args='
    printf '%q ' "$@"
    echo

    echo
  } >> "$DEVICE_INVOCATIONS_LOG"
}

dev() {
  log_device_invocation "$@"

  $ADB shell "sh $DEV_DRIVER $*" |
    tr -d '\r'
}

run_dev_streamed() {
  local label=$1
  shift
  say "    running[$label]..." >&2
  # Everything stays on stdout (parsed by the caller and kept in the per-case
  # log). The live view drops HARNESS_ENV/PP= plus the detailed per-question
  # IQ (IQD=) and raw cookie generation (COOKIE=) lines — those are saved in
  # full to $LOGS/iq_*.txt / $LOGS/cookie_*.txt, and the formatted result row
  # printed right after carries the summarized fields (iq=N/10, cookie=OK, etc).
  dev "$@" | awk '{
      print
      fflush()
      if ($0 !~ /^HARNESS_ENV / && $0 !~ /^PP=/ && \
          $0 !~ /^IQD=/ && $0 !~ /^COOKIE=/) {
        print > "/dev/stderr"
        fflush("/dev/stderr")
      }
    }'
}

#------------------------------------------------------------------------------
# Per-case cleanup and recovery
#
# Only the two harness workload executables are considered here.  The cleanup
# runs after per-case artifacts are captured, so an OOM/failure still retains
# its logs before an automatic recovery makes the next matrix row possible.
#------------------------------------------------------------------------------
reboot_event() {
  local status=$1 reason=$2
  printf '%s\t%s\t%s\n' "$(date -Ins 2>/dev/null || date)" "$status" "$reason" \
    >> "$REBOOT_LOG"
}

device_workload_pids() {
  $ADB shell 'ps -A -o PID,ARGS 2>/dev/null |
    awk '"'"'$2 ~ /(^|\/)llama-(bench|completion)([[:space:]]|$)/ { print $1 }'"'"'' \
    2>/dev/null |
    tr -d '\r' |
    awk '/^[0-9]+$/ { print }'
}

device_signal_pids() {
  local signal=$1
  shift
  [ "$#" -gt 0 ] || return 0
  $ADB shell "for pid in $*; do kill -$signal \"\$pid\" || exit 1; done" \
    >/dev/null 2>&1
}

# A hard power cycle can leave an earlier adb server holding the target's USB
# node: the device re-enumerates, but never reappears to this client. Refreshing
# the server releases that claim so the device can be re-acquired.
refresh_host_adb_server() {
  say "    recovery: refreshing the host adb server"
  command adb kill-server >/dev/null 2>&1 || true
  sleep 2
  command adb start-server >/dev/null 2>&1 || true
  sleep 2
  # kill-server drops any TCP transport (e.g. ADB_SERIAL=192.168.1.1:5555) —
  # USB serials re-enumerate on their own, TCP ones need an explicit reconnect.
  ensure_adb_connected
}

wait_for_current_target() {
  local limit=$1 waited=0
  while [ "$waited" -lt "$limit" ]; do
    [ "$($ADB get-state 2>/dev/null)" = "device" ] && return 0
    ensure_adb_connected
    sleep "$REBOOT_POLL"
    waited=$((waited + REBOOT_POLL))
  done
  return 1
}

# Waits for the target, refreshing the host adb server once before giving up.
wait_for_current_target_resilient() {
  local limit=$1
  wait_for_current_target "$limit" && return 0
  refresh_host_adb_server
  wait_for_current_target "$limit"
}

wait_for_reboot_outage() {
  local waited=0 state
  while [ "$waited" -lt "$PDU_OFFLINE_WAIT" ]; do
    state=$($ADB get-state 2>/dev/null)
    [ "$state" != "device" ] && return 0
    # A PDU's off interval may be shorter than the normal recovery poll.
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

restore_device_root() {
  local uid attempts=0
  uid=$($ADB shell id -u 2>/dev/null | tr -d '\r\n')
  [ "$uid" = "0" ] && return 0

  $ADB root >/dev/null 2>&1 || return 1
  while [ "$attempts" -lt 5 ]; do
    sleep 1
    uid=$($ADB shell id -u 2>/dev/null | tr -d '\r\n')
    [ "$uid" = "0" ] && return 0
    attempts=$((attempts + 1))
  done
  return 1
}

recover_after_reboot() {
  RECOVERY_FAILURE_STAGE=""
  wait_for_current_target_resilient "$PDU_OFFLINE_WAIT" ||
    { RECOVERY_FAILURE_STAGE="target did not return to ADB"; return 1; }

  restore_device_root ||
    { RECOVERY_FAILURE_STAGE="could not restore adb root"; return 1; }
  write_dev_driver ||
    { RECOVERY_FAILURE_STAGE="could not reinstall the device driver"; return 1; }
}

pdu_cycle() {
  local host=$1 outlet=$2 index http
  index=$((outlet - 1))
  [ -n "$PDU_USER" ] && [ -n "$PDU_PASS" ] || return 2

  http=$(timeout 25 curl -s -o /dev/null -w '%{http_code}' -X POST \
    -H 'X-CSRF: x' --digest -u "$PDU_USER:$PDU_PASS" \
    "http://$host/restapi/relay/outlets/$index/cycle/" 2>/dev/null) || return 1
  [ "$http" = "200" ] || [ "$http" = "207" ]
}

PDU_CREDS_PRIMED=0

# Resolves PDU host/outlet/user for a target. Prints "host outlet user" and
# returns 0 when the target is reachable by PDU, 1 otherwise.
# The two lab targets are mapped unconditionally; everything else must be fully
# specified by the caller via PDU_HOST/PDU_OUTLET/PDU_USER/PDU_PASS.
pdu_mapping_for_target() {
  local target=$1
  case "$target" in
    # Same physical board, three ways of naming it: the USB serial (8ee1cb0),
    # the board's own br0 over the direct ethernet link (192.168.1.1), and its
    # lagvm guest on that bridge (192.168.1.3). A power cycle hits the board,
    # so all three map to the one outlet.
    8ee1cb0|192.168.1.1:5555|192.168.1.3:5555)
      printf '%s %s %s\n' "${PDU_HOST:-10.238.2.49}" "${PDU_OUTLET:-1}" "${PDU_USER:-$PDU_LAB_USER}"
      return 0
      ;;
    1b531bdd)
      printf '%s %s %s\n' "${PDU_HOST:-10.238.2.94}" "${PDU_OUTLET:-2}" "${PDU_USER:-$PDU_LAB_USER}"
      return 0
      ;;
    *)
      if [ -n "$PDU_HOST" ] && [ -n "$PDU_OUTLET" ] && [ -n "$PDU_USER" ]; then
        printf '%s %s %s\n' "$PDU_HOST" "$PDU_OUTLET" "$PDU_USER"
        return 0
      fi
      return 1
      ;;
  esac
}

# Asks once, at startup, for the PDU password of a mapped lab target so the rest
# of the run is unattended. Never logged, never written to disk.
prime_pdu_credentials() {
  local target
  [ "$PDU_CREDS_PRIMED" = 1 ] && return 0
  PDU_CREDS_PRIMED=1
  [ "$AUTO_REBOOT_ON_CLEANUP_FAILURE" = 1 ] || return 0
  [ -n "$PDU_PASS" ] && return 0
  [ "$PDU_PROMPT" = 1 ] || return 0
  [ -t 0 ] || return 0

  target=$ADB_SERIAL
  pdu_mapping_for_target "$target" >/dev/null || return 0

  say "${Y}PDU password for $target is not set; automated reboot needs it once.${N}"
  say "${Y}Press Enter to skip (reboots will then prompt you to power-cycle by hand).${N}"
  say "${Y}Tip: add 'export PDU_PASS=...' to your shell rc to never be asked again.${N}"
  IFS= read -rs -p "PDU password: " PDU_PASS
  printf '\n'
  export PDU_PASS
}

reboot_reason_brief() {
  local reason=$1
  case "$reason" in
    *--reboot-before-test*) printf '%s' "user request" ;;
    *"confirmed kernel OOM"*) printf '%s' "previous kernel OOM" ;;
    *"stale PIDs survived KILL"*) printf '%s' "stale workload processes" ;;
    *"unable to enumerate"*) printf '%s' "previous cleanup verification failure" ;;
    *) printf '%s' "previous test recovery request" ;;
  esac
}

manual_reboot_required() {
  local reason=$1
  reboot_event "MANUAL_REQUIRED" "$reason; reboot the target, then press Enter to continue"
  if [ -t 0 ]; then
    IFS= read -r -p "Reboot required. Press Enter after the target has rebooted, or Ctrl-C to stop: "
    return 0
  fi
  return 1
}

recover_cleanup_failure() {
  local reason=$1 target host="" outlet="" use_pdu=0 automated=0 mapping=""
  if [ "$ROW_RECOVERY_DEFER" = 1 ]; then
    if [ ! -s "$ROW_RECOVERY_STATE" ]; then
      reboot_event "DEFERRED" "$reason; recovery deferred until this row finishes"
    fi
    printf '%s\n' "$reason" >> "$ROW_RECOVERY_STATE"
    return 0
  fi

  target=$ADB_SERIAL

  if mapping=$(pdu_mapping_for_target "$target"); then
    read -r host outlet PDU_USER <<<"$mapping"
    use_pdu=1
  fi

  if [ "$use_pdu" = 1 ] && [ "$AUTO_REBOOT_ON_CLEANUP_FAILURE" = 1 ] &&
       [ -n "$PDU_USER" ] && [ -n "$PDU_PASS" ]; then
    reboot_event "REQUESTED" "$reason; PDU=$host outlet=$outlet"
    if ! pdu_cycle "$host" "$outlet"; then
      reboot_event "FAILED" "$reason; PDU cycle failed for $host outlet=$outlet"
      say "    ${R}device reboot failed: PDU cycle request rejected${N}"
      return 1
    fi
    automated=1
    if ! wait_for_reboot_outage; then
      reboot_event "FAILED" "$reason; target never went offline after PDU cycle"
      say "    ${R}device reboot failed: target never went offline${N}"
      return 1
    fi
  else
    manual_reboot_required "$reason; no usable PDU credentials/mapping (set PDU_HOST/PDU_OUTLET/PDU_USER/PDU_PASS for an unmapped target)" || return 1
  fi

  if recover_after_reboot; then
    reboot_event "RECOVERED" "$reason; device online and driver reinstalled"
    if [ "$automated" = 1 ]; then
      say "    ${Y}device rebooted: $(reboot_reason_brief "$reason")${N}"
    fi
    return 0
  fi
  reboot_event "FAILED" "$reason; ${RECOVERY_FAILURE_STAGE:-recovery failed}"
  say "    ${R}device reboot failed: ${RECOVERY_FAILURE_STAGE:-recovery failed}${N}"
  return 1
}

begin_row_recovery_window() {
  ROW_RECOVERY_DEFER=1
  : > "$ROW_RECOVERY_STATE"
}

finish_row_recovery_window() {
  local reason=""
  if [ -s "$ROW_RECOVERY_STATE" ]; then
    reason=$(awk 'NR == 1 { printf "%s", $0 } NR > 1 { printf " | %s", $0 }' \
      "$ROW_RECOVERY_STATE")
  fi
  rm -f "$ROW_RECOVERY_STATE"
  ROW_RECOVERY_DEFER=0
  [ -n "$reason" ] || return 0
  recover_cleanup_failure "$reason; row complete"
}

cleanup_device_workloads() {
  local label=$1 pids survivors waited=0
  [ "$KILL_CONCURRENT" = 1 ] || return 0

  if ! pids=$(device_workload_pids); then
    recover_cleanup_failure "case=$label; unable to enumerate llama-bench/llama-completion PIDs"
    return $?
  fi
  [ -n "$pids" ] || return 0

  say "    cleanup[$label]: TERM pids $(printf '%s' "$pids" | tr '\n' ' ')"
  if ! device_signal_pids TERM $pids; then
    recover_cleanup_failure "case=$label; TERM failed for PIDs $(printf '%s' "$pids" | tr '\n' ',')"
    return $?
  fi

  while [ "$waited" -lt 5 ]; do
    sleep 1
    if ! survivors=$(device_workload_pids); then
      recover_cleanup_failure "case=$label; unable to verify TERM cleanup"
      return $?
    fi
    [ -z "$survivors" ] && return 0
    waited=$((waited + 1))
  done

  say "    cleanup[$label]: KILL pids $(printf '%s' "$survivors" | tr '\n' ' ')"
  if ! device_signal_pids KILL $survivors; then
    recover_cleanup_failure "case=$label; KILL failed for PIDs $(printf '%s' "$survivors" | tr '\n' ',')"
    return $?
  fi
  sleep 1
  if ! survivors=$(device_workload_pids); then
    recover_cleanup_failure "case=$label; unable to verify KILL cleanup"
    return $?
  fi
  [ -z "$survivors" ] && return 0

  recover_cleanup_failure "case=$label; stale PIDs survived KILL: $(printf '%s' "$survivors" | tr '\n' ',')"
}

case_had_kernel_oom() {
  local out=$1 oom_lines
  if [ "$COLLECT_MEMORY" = 1 ] && [ -n "${MEM_CUR_DIR:-}" ]; then
    oom_lines=$(sed -n 's/^oom_kernel_lines=//p' "$MEM_CUR_DIR/summary.txt" 2>/dev/null | tail -1)
    case "$oom_lines" in
      ''|*[!0-9]*) ;;
      *) [ "$oom_lines" -gt 0 ] && return 0 ;;
    esac
  fi

  # Without memory collection, retain the device driver's confirmed OOM
  # classification as the only available per-case signal.
  printf '%s\n' "$out" | grep -q 'PP_REASON=HOST_OOM_KILL'
}

#------------------------------------------------------------------------------
# Phase 3 — run the matrix
#------------------------------------------------------------------------------
run_matrix() {
  head1 "TEST MATRIX"
  say "device   : $ADB_SERIAL"
  say "commit   : $(cd "$REPO" && git rev-parse --short HEAD) ($(cd "$REPO" && git rev-parse --abbrev-ref HEAD))"
  DEV_BUILD_INFO=$(device_build_info)
  say "binaries : $DEV_BUILD_INFO"
  if [ -n "$AB_VAR" ]; then
    say "A/B var  : $AB_VAR = $AB"
  fi
  say "cores    : $CORES      splits: $SPLITS      ubatch: $UBATCH      reps: $REPS      $(if [ -n "$DEPTHS" ]; then
echo "depths: $DEPTHS (pp-depth=depth, tg-depth=depth+PP_TOKENS=$PP_TOKENS)"
else
echo "pp-depth: $PP_DEPTH      tg-depth: $TG_DEPTH"
fi)"
  if [ "$HEX_PROFILE" != 0 ]; then
    say "profile  : GGML_HEXAGON_PROFILE=$HEX_PROFILE on PP bench (+ -v); full logs -> $OUTDIR/profile/  (PP t/s distorted; TG unprofiled)"
  fi
  say "outdir   : $OUTDIR"
  say ""

  printf 'model\tcores\tsplit\tab\tubatch\tpp_tokens\tpp_ts\ttg_ts\tiq\tneedle\tcookie_ok\tdepth\n' > "$TSV"

  # If no A/B var is configured, run the loop body once with a placeholder value.
  local ab_values="$AB"; [ -z "$AB_VAR" ] && ab_values="-"

  local IFSB=$IFS
  for spec in $MODELS; do
    local model=${spec%%:*} maxc=${spec##*:}

    # Hard, non-overridable per-model tensor-split core cap. The per-model
    # cap in $MODELS (above) is only a *default* and can be raised by a
    # caller-supplied --models/MODELS override (e.g. "gemma-4-E4B_q4_0-it:4"),
    # which defeats the GQA KV-head-limit protection this cap exists for.
    # Enforce a hard ceiling here that MODELS can lower but never raise.
    case "$model" in
      gemma-4-E4B_q4_0-it) [ "$maxc" -gt 2 ] && maxc=2 ;;
      # Qwen3.5-4B has no KV-head limit up to 4 cores; a MODELS override such as
      # "Qwen3.5-4B-Q4:1" must not make tensor split skip 2/4 cores.
      Qwen3.5-4B-Q4) [ "$maxc" -lt 4 ] && maxc=4 ;;
      gemma-4-E2B_q4_0-it|gemma-4-12b-it-qat-q4_0) [ "$maxc" -gt 1 ] && maxc=1 ;;
    esac

    IFS=,; for n in $CORES; do IFS=$IFSB

      # Split mode is only meaningful with >1 core: at 1 core there is nothing to
      # split, so run it once labelled "single" instead of once per $SPLITS entry.
      local split_values="$SPLITS"; [ "$n" -eq 1 ] && split_values="single"

      IFS=,; for sp in $split_values; do IFS=$IFSB
        # The per-model core cap in $MODELS (e.g. Gemma-4-E4B's GQA 2-KV-head
        # limit) is a tensor-split-specific constraint: tensor split divides
        # attention heads across cores, so it's the only mode that can strand a
        # core with zero KV heads. Both tensor and tensor-scatter use tensor split,
        # so both are subject to the cap. tensor-row's outer split is also tensor,
        # so it's capped too. Layer- and row-split partition by layer/row instead.
        case "$sp" in
          tensor|tensor-scatter|tensor-row)
            if [ "$n" -gt "$maxc" ]; then
              say "${Y}skip${N} $model ${n}core $sp (max ${maxc}: GQA KV-head limit)"
              continue
            fi ;;
        esac
        # 2D splits (layer-row, tensor-row) require exactly 4 physical cores:
        # HTP0[0-1] + HTP1[2-3]. Skip at any other core count.
        case "$sp" in
          layer-row|tensor-row)
            if [ "$n" -ne 4 ]; then
              say "${Y}skip${N} $model ${n}core $sp (2D splits require exactly 4 cores)"
              continue
            fi ;;
        esac

        # smode()/hexdevices()/clidev() on the device understand
        # tensor|tensor-scatter|layer|row; "single" (1-core) just needs any valid
        # value since smode() short-circuits to "" for n=1.
        local sp_dev="$sp"; [ "$sp" = "single" ] && sp_dev="tensor"
       # --- performance: swept over ubatch ---------------------------------
        local depth_values
        if [ -n "$DEPTHS" ]; then
          depth_values="$DEPTHS"
        else
          depth_values="fixed"
        fi

        IFS=,; for depth in $depth_values; do IFS=$IFSB

          if [ "$depth" != "fixed" ]; then
            PP_DEPTH_RUN="$depth"
            # TG is measured right after PP would have written $PP_TOKENS
            # fresh tokens into the KV cache starting at $depth, so TG picks
            # up where PP left off instead of re-measuring at the same depth.
            TG_DEPTH_RUN=$((depth + PP_TOKENS))
          else
            PP_DEPTH_RUN="$PP_DEPTH"
            TG_DEPTH_RUN="$TG_DEPTH"
          fi

          # Distinguish per-depth labels/log filenames when DEPTHS sweeps
          # multiple values — otherwise every depth in the sweep reuses the
          # same label, so each iteration overwrites the previous depth's
          # log file and memory/<label>/ artifacts, leaving only the last
          # depth's diagnostics on disk (the earlier depths' actual error
          # text becomes unrecoverable after the run finishes).
          local depth_tag=""
          [ "$depth" != "fixed" ] && depth_tag="_d${depth}"

        IFS=,; for abv in $ab_values; do IFS=$IFSB

          # --- quality checks: independent of ubatch, so measured once -------
          local iq="-" needle="-" cookie_ok="-" cookie_txt=""
          begin_row_recovery_window
          if [ "$DO_IQ" = 1 ]; then
            local iq_raw; iq_raw=$(monitored_dev "iq_${model}_${n}c_${sp}_ab${abv}${depth_tag}" llama-completion iq "$model" "$n" "$sp_dev" "$abv")
            iq=$(printf '%s\n' "$iq_raw" | sed -n 's/^IQ=//p')
            [ -z "$iq" ] && iq="ERR"
            {
              printf 'Q#\tEXPECTED\tGOT\tMATCH\tPROMPT\n'
              printf '%s\n' "$iq_raw" | sed -n 's/^IQD=//p' \
                | awk -F'|' 'BEGIN{OFS="\t"}{
                    q=$1; e=$2; g=$3; m=$4; p=$5
                    sub(/^EXPECTED=/,"",e); sub(/^GOT=/,"",g)
                    sub(/^MATCH=/,"",m); sub(/^PROMPT=/,"",p)
                    print q,e,g,m,p
                  }'
              printf '\nSCORE: %s\n' "$iq"
            } > "$(unique_log_path "$LOGS/iq_${model}_${n}c_${sp}_ab${abv}${depth_tag}.txt")"
          fi
          if [ "$DO_NEEDLE" = 1 ]; then
            needle=$(monitored_dev "needle_${model}_${n}c_${sp}_ab${abv}${depth_tag}" llama-completion needle "$model" "$n" "$sp_dev" "$abv" | sed -n 's/^NEEDLE=//p')
            [ -z "$needle" ] && needle="ERR"
          fi
          if [ "$DO_COOKIE" = 1 ]; then
            local cookie_raw cookie_rc
            cookie_raw=$(monitored_dev "cookie_${model}_${n}c_${sp}_ab${abv}${depth_tag}" llama-completion cookie "$model" "$n" "$sp_dev" "$abv")
            cookie_txt=$(printf '%s\n' "$cookie_raw" | sed -n 's/^COOKIE=//p')
            cookie_rc=$(printf '%s\n' "$cookie_raw" | sed -n 's/^COOKIE_RC=//p')
            printf '%s\n' "$cookie_txt" > "$(unique_log_path "$LOGS/cookie_${model}_${n}c_${sp}_ab${abv}${depth_tag}.txt")"
            # Heuristic gate only — the saved text is the real artifact for review.
            # GARBAGE(short)/GARBAGE(repeat): word-level (space-separated) checks.
            # GARBAGE(phrase): a multi-word phrase looped BACK-TO-BACK (e.g. "You have a
            #   dataset of labeled data. You have a dataset of labeled data." x14) — the
            #   signature of stale-output collapse. Must be *consecutive* repeats: coherent
            #   list-style prose reuses short phrases scattered around, which is NOT garbage,
            #   so a plain "appears N times anywhere" test false-positives on good answers.
            # GARBAGE(glued): character-level repeat (e.g. "tütütü...") with no
            #   spaces, invisible to a whitespace split.
            # Score only the answer: drop everything up to the last </think>.
            local cookie_ans
            cookie_ans=$(printf '%s' "$cookie_txt" | perl -0777 -pe 's/.*<\/think>//s; s/\[end of text\]//g; s/^\s+|\s+$//g' 2>/dev/null)
            cookie_ok=$(printf '%s' "$cookie_ans" | awk '{
                len=length($0); if (len<20) { print "GARBAGE(short)"; exit }
                if ($0 ~ /\[New LWP|backtrace|ggml_abort|libthread_db|Thread debugging|Inferior.*detached|from \/|#[0-9]+ 0x/) { print "GARBAGE(crash)"; exit }
                n=split($0,w," "); if (n<8) { print "GARBAGE(short)"; exit }
                for(i=1;i<=n;i++) c[w[i]]++
                top=0; distinct=0; for(k in c){ distinct++; if(c[k]>top) top=c[k] }
                if (top/n > 0.5) { print "GARBAGE(repeat)"; exit }
                # Consecutive phrase loop: for window g, count max run of back-to-back
                # identical g-grams. A real loop repeats immediately (run>=3); coherent
                # prose does not. Require the loop to also cover a big share of the output.
                for (g=1; g<=8; g++) {
                  run=1; maxrun=1
                  for (i=1; i+2*g-1<=n; i++) {
                    same=1
                    for (j=0;j<g;j++) if (w[i+j]!=w[i+g+j]) { same=0; break }
                    if (same) { run++; if (run>maxrun) maxrun=run } else run=1
                  }
                  # maxrun consecutive repeats span maxrun*g words; flag if that is a large
                  # fraction of the whole output (a genuine collapse, not incidental reuse).
                  if (maxrun>=3 && (maxrun*g) > n*0.4) { print "GARBAGE(phrase)"; exit }
                }
                # Extreme low lexical diversity over a long output = collapse from the other
                # side (few distinct words, many tokens). Tightened to avoid coherent prose.
                if (n>=60 && distinct/n < 0.15) { print "GARBAGE(phrase)"; exit }
                print "OK"
              }')
            if [ "$cookie_ok" = "OK" ] && printf '%s' "$cookie_ans" \
                 | perl -0777 -CS -ne 'exit(/(.{2,10})\1{4,}/s ? 0 : 1)' 2>/dev/null; then
              cookie_ok="GARBAGE(glued)"
            fi
            [ -z "$cookie_ok" ] && cookie_ok="ERR"
            if [ "${cookie_rc:-1}" != 0 ] || ! printf '%s\n' "$cookie_raw" | grep -q '^COOKIE='; then
              cookie_ok="ERR"
            elif [ -z "$(printf '%s' "$cookie_txt" | tr -d ' \t')" ]; then
              cookie_ok="GARBAGE(empty)"
            elif printf '%s' "$cookie_txt" | grep -q '<think>' && { [ -z "$cookie_ans" ] || ! printf '%s' "$cookie_txt" | grep -q '</think>'; }; then
              cookie_ok="GARBAGE(<think>)"
            fi
          fi

   
          IFS=,; for ppt in $PP_TOKENS; do IFS=$IFSB
          # TG is measured right after PP would have written $ppt fresh tokens
          # into the KV cache starting at the current depth, so recompute the TG
          # depth per PP length when sweeping DEPTHS.
          local TG_DEPTH_PP="$TG_DEPTH_RUN"
          [ "$depth" != "fixed" ] && TG_DEPTH_PP=$((depth + ppt))
          local pp_tag=""
          [ "$(printf '%s' "$PP_TOKENS" | tr ',' ' ' | wc -w)" -gt 1 ] && pp_tag="_pp${ppt}"
          IFS=,; for ub in $UBATCH; do IFS=$IFSB
            [ "$ROW_RECOVERY_DEFER" = 1 ] || begin_row_recovery_window
            local pp="-" tg="-" ub_used="$ub" ub_note=""
            if [ "$DO_BENCH" = 1 ]; then
              local out; out=$(monitored_dev "bench_${model}_${n}c_${sp}_ab${abv}_ub${ub}${pp_tag}${depth_tag}" llama-bench bench "$model" "$n" "$sp_dev" "$abv" "$ub" "$PP_DEPTH_RUN" "$TG_DEPTH_PP" "$ppt")



              printf '%s\n' "$out" > "$(unique_log_path "$LOGS/bench_${model}_${n}c_${sp}_ab${abv}_ub${ub}${pp_tag}${depth_tag}.log")"
              pp=$(printf '%s' "$out" | sed -n 's/.*PP=\([^ ]*\).*/\1/p')
              tg=$(printf '%s' "$out" | sed -n 's/.*TG=\([^ ]*\).*/\1/p')
              [ -z "$pp" ] && pp="NA"; [ -z "$tg" ] && tg="NA"

              pp_reason=$(
                printf '%s' "$out" |
                sed -n \
                  's/.*PP_REASON=\([^ ]*\).*/\1/p' |
                tail -1
              )

              [ -n "$pp_reason" ] ||
                pp_reason="UNKNOWN"


              # Some multi-core configs overflow the allreduce VMEM budget at large
              # ubatch (upstream regression: silent abort -> PP=NA). Retry the whole row
              # once at ub=512 so the config still yields a number, and flag it.
              if [ "${RETRY_ON_FAIL:-1}" = 1 ] &&
                   [ "$pp" = "NA" ] &&
                   [ "$ub" -gt 512 ] &&
                   [ "$pp_reason" = "HTP_OP_MEMORY_OVERCLAIM" ]; then
                say "    ${Y}$model ${n}c $sp ab=$abv ub=$ub pp=$ppt failed reason=$pp_reason — retrying at ub=512${N}"
                # NOTE: the fallback label must include the *original* $ub (not just
                # the literal "512" it fell back to) — otherwise two different
                # UBATCH sweep values that both overflow and fall back collide on
                # the exact same log/memory label and silently overwrite each other.
                out=$(monitored_dev "bench_${model}_${n}c_${sp}_ab${abv}_ub${ub}-fallback512${pp_tag}${depth_tag}" llama-bench bench "$model" "$n" "$sp_dev" "$abv" 512 "$PP_DEPTH_RUN" "$TG_DEPTH_PP" "$ppt")
                printf '%s\n' "$out" > "$(unique_log_path "$LOGS/bench_${model}_${n}c_${sp}_ab${abv}_ub${ub}-fallback512${pp_tag}${depth_tag}.log")"
                pp=$(printf '%s' "$out" | sed -n 's/.*PP=\([^ ]*\).*/\1/p')
                tg=$(printf '%s' "$out" | sed -n 's/.*TG=\([^ ]*\).*/\1/p')
                [ -z "$pp" ] && pp="NA"; [ -z "$tg" ] && tg="NA"
                ub_used="512*"; ub_note=" (ub reduced 1024->512)"
              fi
            fi

            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                   "$model" "$n" "$sp" "$abv" "$ub_used" "$ppt" "$pp" "$tg" "$iq" "$needle" "$cookie_ok" "$PP_DEPTH_RUN" >> "$TSV"

            local flag="$G"; case "$needle$cookie_ok" in *FAIL*|*GARBAGE*|*ERR*) flag="$R" ;; esac
            printf '  %-32s %dc %-6s ab=%-4s pp=%-6s ppd=%-6s tgd=%-6s ub=%-5s pp_ts=%-9s tg=%-7s %siq=%s needle=%s cookie=%s%s%s\n' \
                   "$model" "$n" "$sp" "$abv" "$ppt" "$PP_DEPTH_RUN" "$TG_DEPTH_PP" "$ub_used" "$pp" "$tg" "$flag" "$iq" "$needle" "$cookie_ok" "$N" "$ub_note"
            finish_row_recovery_window || return 1
          done
          done
        done
      done
    done
  done
    done
  IFS=$IFSB
}

#------------------------------------------------------------------------------
# Phase 4 — summarise
#   Emits the results a reviewer actually needs: an absolute results table, and
#   — only when an A/B var was swept over exactly two values — a delta table
#   per (model, cores, ubatch). That delta table is what answers questions like
#   "does this change's gain hold up / shrink as ubatch grows", for whatever
#   change AB_VAR represents.
#------------------------------------------------------------------------------
summarise() {
  {
    echo "# llama.cpp / ggml-hexagon build-and-test results"
    echo
    echo "- device: \`$ADB_SERIAL\`"
    echo "- repo HEAD: \`$(cd "$REPO" && git rev-parse --short HEAD)\` (\`$(cd "$REPO" && git rev-parse --abbrev-ref HEAD)\`)"
    echo "- device binaries: \`${DEV_BUILD_INFO:-unknown}\`"
    echo "- date: $(date -Iseconds)"
    echo "- reps: $REPS, ctx: $CTX, pp-depth: $PP_DEPTH, tg-depth: $TG_DEPTH"
    echo "- split mode(s): $SPLITS (1-core rows always run as \"single\" — nothing to split)"
    if [ -n "$AB_VAR" ]; then echo "- A/B variable: \`$AB_VAR\` = $AB"; fi
    echo
    echo '## Results'
    echo
    echo "| model | cores | split | ${AB_VAR:-ab} | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |"
    echo '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|'
    # "x1c" = speedup vs this model's 1-core ("single") result at the same
    # ab — computed here (not stored in the TSV) so it stays correct regardless
    # of which cores/splits were run. Keyed on model+ab only (NOT ubatch): the
    # 1-core baseline runs at the default ubatch, while a multi-core row may have
    # fallen back to a smaller ubatch (marked "512*") — we still want its x1c vs
    # the same 1-core reference.
    awk -F'\t' '
      FNR==1 { next }
      FNR==NR {
        if ($2+0==1) {
          if ($7!="NA" && $7!="-") bp[$1,$4,$5,$6,$12]=$7
          if ($8!="NA" && $8!="-") bt[$1,$4,$5,$6,$12]=$8
        }
        next
      }
      {
        k = $1 SUBSEP $4 SUBSEP $5 SUBSEP $6 SUBSEP $12
        ppx = ((k in bp) && $7!="NA" && $7!="-" && bp[k]+0!=0) ? sprintf("%.2fx", $7/bp[k]) : "NA"
        tgx = ((k in bt) && $8!="NA" && $8!="-" && bt[k]+0!=0) ? sprintf("%.2fx", $8/bt[k]) : "NA"
        printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",$1,$2,$3,$4,$5,$6,$12,$7,ppx,$8,tgx,$9,$10,$11
      }' "$TSV" "$TSV"
    echo

    # Split-mode comparison — only meaningful when more than one split was
    # actually tested and there's a single AB value (else the table above
    # already needs a 3rd dimension to read).
    local split_n; split_n=$(printf '%s' "$SPLITS" | tr ',' '\n' | sort -u | wc -l)
    if [ "$split_n" -gt 1 ]; then
      echo '## Split-mode comparison (per model / cores / ubatch)'
      echo
      echo '| model | cores | split | ab | ubatch | pp_tok | depth | PP t/s | PP x1c | TG t/s | TG x1c | IQ | needle | cookie |'
      echo '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|'
      awk -F'\t' '
        FNR==1 { next }
        FNR==NR {
          if ($2+0==1) {
            if ($7!="NA" && $7!="-") bp[$1,$4,$5,$6,$12]=$7
            if ($8!="NA" && $8!="-") bt[$1,$4,$5,$6,$12]=$8
          }
          next
        }
        $2+0>1 {
          k = $1 SUBSEP $4 SUBSEP $5 SUBSEP $6 SUBSEP $12
          ppx = ((k in bp) && $7!="NA" && $7!="-" && bp[k]+0!=0) ? sprintf("%.2fx", $7/bp[k]) : "NA"
          tgx = ((k in bt) && $8!="NA" && $8!="-" && bt[k]+0!=0) ? sprintf("%.2fx", $8/bt[k]) : "NA"
          printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",$1,$2,$3,$4,$5,$6,$12,$7,ppx,$8,tgx,$9,$10,$11
        }' "$TSV" "$TSV" \
        | sort -t'|' -k2,2 -k3,3n
      echo
    fi

    # Delta table only makes sense with an A/B var swept over exactly two values.
    # Restricted to one split mode (the first one listed in $SPLITS) so the
    # model/cores/ubatch key stays unambiguous; 1-core rows ("single") always
    # match since split doesn't apply there.
    local ab_n; ab_n=$(printf '%s' "$AB" | tr ',' '\n' | sort -u | wc -l)
    local primary_split; primary_split=$(printf '%s' "$SPLITS" | cut -d, -f1)
    if [ -n "$AB_VAR" ] && [ "$ab_n" = 2 ]; then
      local v0 v1; v0=$(printf '%s' "$AB" | cut -d, -f1); v1=$(printf '%s' "$AB" | cut -d, -f2)
      echo "## PP delta ($AB_VAR=$v0 -> $v1), per model / cores / ubatch (split=$primary_split)"
      echo
      echo 'If the percentage shrinks as ubatch grows, the change and ubatch are NOT additive.'
      echo
      echo "| model | cores | ubatch | pp_tok | depth | PP $AB_VAR=$v0 | PP $AB_VAR=$v1 | delta |"
      echo '|---|---|---|---|---|---|---|---|'
      tail -n +2 "$TSV" | awk -F'\t' -v v0="$v0" -v v1="$v1" -v ps="$primary_split" '
        $3!=ps && $3!="single" { next }
        $4==v0 { base[$1"\t"$2"\t"$5"\t"$6"\t"$12]=$7 }
        $4==v1 { opt[$1"\t"$2"\t"$5"\t"$6"\t"$12]=$7 }
        END {
          for (k in opt) {
            b=base[k]; o=opt[k]
            if (b=="" || b=="NA" || b=="-" || o=="NA" || o=="-") continue
            split(k,p,"\t")
            printf "| %s | %s | %s | %s | %s | %.1f | %.1f | %+.1f%% |\n", p[1],p[2],p[3],p[4],p[5],b,o,(o-b)/b*100
          }
      }' | sort -t'|' -k2,2 -k3,3n -k4,4n
      echo

      echo "## Cross-core scaling ($AB_VAR=$v0=Base vs $AB_VAR=$v1=Opt), per model (split=$primary_split)"
      echo
      echo 'Uses the largest ubatch tested for each model. "vs 1c" is that series'"'"'s own'
      echo 'speedup over its 1-core result; PP gain is Opt vs Base at that core count.'
      echo
      awk -F'\t' -v v0="$v0" -v v1="$v1" -v ps="$primary_split" '
        FNR==1 { next }
        $3!=ps && $3!="single" { next }
        FNR==NR {
          if ($5+0 > maxub[$1]+0) maxub[$1]=$5+0
          next
        }
        {
          if ($5+0 != maxub[$1]+0) next
          if ($7=="NA" || $7=="-" || $8=="NA" || $8=="-") next
          if ($4==v0) { bpp[$1,$2]=$7; btg[$1,$2]=$8 }
          if ($4==v1) { opp[$1,$2]=$7; otg[$1,$2]=$8 }
          seen[$1"\t"$2]=1
        }
        END {
          for (key in seen) {
            split(key,p,"\t"); m=p[1]; c=p[2]
            if (!((m,c) in bpp) || !((m,c) in opp)) continue
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", m, c, maxub[m], bpp[m,c], btg[m,c], opp[m,c], otg[m,c]
          }
        }' "$TSV" "$TSV" \
      | sort -t"$(printf '\t')" -k1,1 -k2,2n \
      | awk -F'\t' '
          {
            model=$1; cores=$2+0; ub=$3; bp=$4+0; bt=$5+0; op=$6+0; ot=$7+0
            if (model != prevmodel) {
              if (NR>1) print ""
              print "### " model " (ubatch=" ub ")"
              print ""
              print "| Cores | Base PP | vs 1c | Opt PP | vs 1c | PP gain | Base TG | Opt TG |"
              print "|---|---|---|---|---|---|---|---|"
              prevmodel=model; refbp=""; refop=""
            }
            if (cores==1) { refbp=bp; refop=op }
            bpr = (refbp!="" && refbp!=0) ? sprintf("%.2fx", bp/refbp) : "NA"
            opr = (refop!="" && refop!=0) ? sprintf("%.2fx", op/refop) : "NA"
            gain = (bp!=0) ? sprintf("%+.1f%%", (op-bp)/bp*100) : "NA"
            printf "| %d | %.0f | %s | %.0f | %s | %s | %.1f | %.1f |\n", cores, bp, bpr, op, opr, gain, bt, ot
          }'
      echo
    fi
    echo '## Quality gate'
    echo
    awk -F'\t' 'NR>1 {
                  if ($10=="FAIL"||$11~/^GARBAGE/||$10=="ERR"||$11=="ERR") bad++
                  else if ($9=="-" && $10=="-" && $11=="-") skip++
                  else ok++
                }
                END {
                  if (bad)       printf "**%d configuration(s) FAILED quality checks — see table above.**\n", bad
                  else if (ok)   printf "All %d checked configuration(s) passed needle + cookie (see table above for IQ score per configuration).\n", ok
                  else           print "_Quality checks were skipped for this run (--only-perf) — performance numbers alone do NOT establish correctness._"
                }' "$TSV"
    echo
    echo "Raw logs: \`$LOGS\`  ·  iq transcripts: \`$LOGS/iq_*.txt\`  ·  cookie texts: \`$LOGS/cookie_*.txt\`"
    if [ "$COLLECT_MEMORY" = 1 ]; then echo "Memory diagnostics: \`$MEMORY_LOGS\`  ·  peak summary: \`$MEMORY_TSV\`"; fi
  } > "$SUMMARY"

  head1 "SUMMARY"
  cat "$SUMMARY"
  say ""
  say "${B}summary${N} : $SUMMARY"
  say "${B}tsv${N}     : $TSV"
  say "${B}logs${N}    : $LOGS"
  if [ "$COLLECT_MEMORY" = 1 ]; then say "${B}memory${N}  : $MEMORY_LOGS  (summary: $MEMORY_TSV)"; fi

  zip_run_artifacts
}

# Zip up logs/ and memory/ at the end of a run so the (potentially large;
# memory/ especially, with a timeline.log + several dmesg/journal/logcat
# captures per row) raw-artifact tree is easy to archive/copy off-box.
# Non-destructive: the original directories are left in place.
zip_run_artifacts() {
  if [ -d "$LOGS" ] && [ -n "$(ls -A "$LOGS" 2>/dev/null)" ]; then
    rm -f "$OUTDIR/logs.zip"
    ( cd "$OUTDIR" && zip -rq logs.zip logs )
    say "${B}logs.zip${N}   : $OUTDIR/logs.zip"
  fi
  if [ -d "$MEMORY_LOGS" ] && [ -n "$(ls -A "$MEMORY_LOGS" 2>/dev/null)" ]; then
    rm -f "$OUTDIR/memory.zip"
    ( cd "$OUTDIR" && zip -rq memory.zip memory )
    say "${B}memory.zip${N} : $OUTDIR/memory.zip"
  fi
}


#------------------------------------------------------------------------------
# Exclusivity — make sure we are the only llama.cpp workload on this device.
#
#   WHY THIS EXISTS
#   Benchmark numbers from this harness are only meaningful if nothing else is
#   driving the cores at the same time. This has bitten us for real: a run left
#   over from 17:44 was still hammering the cores while a fresh A/B started at
#   18:07, so both sets of numbers were measured under unknown contention and
#   the A/B comparison was worthless. Worse, the stale run was wedged inside a
#   single multi-hour llama-bench call, so it was invisible unless you thought
#   to look at `ps`.
#
#   Rather than ask the user to remember, take the device by force at startup.
#   A benchmark harness that silently produces contaminated numbers is more
#   dangerous than one that is a bit heavy-handed about claiming the device.
#
#   Deliberately kills by explicit PID rather than pkill/killall: this host is
#   shared, and a name pattern that is merely *close* to right would take out
#   another user's job. We also never kill ourselves or any of our own
#   ancestors, which would otherwise happen immediately since this script's own
#   command line matches the pattern.
#
#   Escalates TERM -> KILL: wedged runs are typically blocked in an
#   uninterruptible adb read and ignore SIGTERM entirely.
#------------------------------------------------------------------------------
ensure_exclusive() {
  head1 "EXCLUSIVITY"

  if [ "$KILL_CONCURRENT" != 1 ]; then
    say "${Y}--allow-concurrent given: NOT clearing other jobs${N}"
    say "${Y}any numbers from this run may be contaminated by device contention${N}"
    return 0
  fi

  # PIDs we must never touch: ourselves plus our entire ancestor chain.
  local safe=" $$ " p=$$ ppid
  while [ -n "$p" ] && [ "$p" != 1 ]; do
    ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$ppid" ] || break
    safe="$safe$ppid "
    p=$ppid
  done

  # Only ever consider our own processes on this host.
  local pat='llama-(bench|cli|completion|server|perplexity|run)|llama\.cpp-build-and-test'
  local victims="" pid cmd
  while read -r pid cmd; do
    # An empty match set still feeds one blank line through the here-doc.
    case "$pid" in '' | *[!0-9]*) continue ;; esac
    case "$safe" in *" $pid "*) continue ;; esac
    victims="$victims $pid"
    say "  killing host pid $pid: $(printf '%.90s' "$cmd")"
  done <<EOF
$(ps -u "$(id -un)" -o pid=,args= 2>/dev/null | grep -E "$pat" | grep -v grep | grep "$ADB_SERIAL")
EOF

  if [ -n "$victims" ]; then
    for pid in $victims; do kill "$pid" 2>/dev/null; done
    # Give them a moment to exit cleanly, then be blunt about it.
    local waited=0
    while [ "$waited" -lt 5 ]; do
      local alive=0
      for pid in $victims; do kill -0 "$pid" 2>/dev/null && alive=1; done
      [ "$alive" = 0 ] && break
      sleep 1; waited=$((waited + 1))
    done
    for pid in $victims; do
      if kill -0 "$pid" 2>/dev/null; then
        say "  ${Y}pid $pid ignored SIGTERM (likely wedged in adb) — SIGKILL${N}"
        kill -9 "$pid" 2>/dev/null
      fi
    done
    say "${G}host cleared${N}"
  else
    say "host     : no other llama.cpp jobs"
  fi

  # Device side: a killed host process can leave its on-device workload running,
  # still holding the cores. Kill by PID there too, for the same reason.
  local devkilled
  devkilled=$($ADB shell 'ps -A -o PID,ARGS 2>/dev/null | grep llama- | grep -v grep | while read pid rest; do kill -9 "$pid" 2>/dev/null && echo "$pid"; done' 2>/dev/null | tr -d '\r' | tr '\n' ' ')
  if [ -n "${devkilled// /}" ]; then
    say "${G}device cleared${N}: killed on-device pids $devkilled"
  else
    say "device   : no stale llama.cpp processes"
  fi
}

#------------------------------------------------------------------------------
# Preflight — fail in seconds rather than 40 minutes in.
#   Checks only what the selected phases actually need.
#------------------------------------------------------------------------------
preflight() {
  # Building a different checkout than the one this script (and the pushed results)
  # live in silently mixes branches: results get filed under this branch while the
  # binaries come from elsewhere. Refuse unless explicitly allowed.
  local self_top repo_top
  self_top=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
  repo_top=$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null)
  if [ -n "$self_top" ] && [ "$self_top" != "$repo_top" ] && [ "${ALLOW_FOREIGN_REPO:-0}" != 1 ]; then
    die "REPO ($REPO, branch $(git -C "$REPO" branch --show-current 2>/dev/null)) is not this script's checkout ($self_top, branch $(git -C "$self_top" branch --show-current)). Drop REPO= (defaults to this checkout) or set ALLOW_FOREIGN_REPO=1."
  fi
  head1 "PREFLIGHT"
  local bad=0

  if [ "$DO_BUILD" = 1 ]; then
    command -v docker >/dev/null || { say "${R}missing:${N} docker (needed for --build; use --no-build)"; bad=1; }
    docker info >/dev/null 2>&1  || { say "${R}docker daemon not reachable as $(id -un)${N}"; bad=1; }
    [ -d "$REPO" ] || { say "${R}missing repo:${N} $REPO"; bad=1; }
    [ -f "$REPO/scripts/buid_in_docker.sh" ] \
      || { say "${R}missing:${N} $REPO/scripts/buid_in_docker.sh"; bad=1; }
    say "build    : scripts/buid_in_docker.sh $PLATFORM (host cmake cannot parse the presets)"
  else
    say "build    : skipped"
  fi

  command -v adb >/dev/null || { say "${R}missing:${N} adb"; bad=1; }
  if [ "$DO_DEPLOY" = 1 ] || [ "$DO_BENCH" = 1 ] || [ "$DO_IQ" = 1 ]; then
    if [ "$($ADB get-state 2>/dev/null)" = "device" ]; then
      say "device   : $ADB_SERIAL online"
    else
      say "${R}device $ADB_SERIAL not online through configured transport${N}"
      bad=1
    fi
  else
    say "device   : check skipped (build-only)"
  fi

  if [ "$bad" = 0 ] && { [ "$DO_DEPLOY" = 1 ] || [ "$DO_BENCH" = 1 ] || [ "$DO_IQ" = 1 ]; }; then
    # adbd on this device silently drops back to non-root (uid=2000) after any
    # reboot, which then fails the driver-script push with "Permission
    # denied" (the file on-device is root-owned from a prior root session).
    # Restore it here rather than making the user notice and fix it by hand.
    local uid; uid=$($ADB shell id -u 2>/dev/null | tr -d '\r\n')
    if [ "$uid" != "0" ]; then
      say "${Y}adbd not root (uid=$uid) — restoring root...${N}"
      $ADB root >/dev/null 2>&1
      for _ in 1 2 3 4 5; do
        sleep 1
        # adbd restart drops TCP transports (USB reattaches on its own) —
        # reconnect before every poll or we spin until the retry budget runs out.
        ensure_adb_connected
        uid=$($ADB shell id -u 2>/dev/null | tr -d '\r\n')
        [ "$uid" = "0" ] && break
      done
      [ "$uid" = "0" ] || { say "${R}failed to restore adb root (uid=$uid)${N}"; bad=1; }
    fi
    [ "$uid" = "0" ] && say "device   : adb root ok"
  fi

  [ "$bad" = 0 ] || die "preflight failed — nothing was run"
  say "${G}preflight ok${N}"
}

#------------------------------------------------------------------------------
# push_results — commit $OUTDIR to the current branch and push it. Opt-in via
# --push-results. Only $OUTDIR is committed (other local changes are left
# alone); files over 95 MB are skipped (GitHub rejects >100 MB).
push_results() {
  [ "$PUSH_RESULTS" = 1 ] || return 0
  head1 "PUSH RESULTS"
  local repo branch rel
  repo=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null) ||
    { say "${R}not a git checkout — results not pushed${N}"; return 0; }
  branch=$(git -C "$repo" branch --show-current)
  # let git resolve paths (shell path spellings can differ, e.g. symlinks/mounts)
  if [ "$(git -C "$OUTDIR" rev-parse --show-toplevel 2>/dev/null)" != "$repo" ]; then
    say "${R}OUTDIR is outside the repo ($OUTDIR) — results not pushed${N}"; return 0
  fi
  rel=$(git -C "$OUTDIR" rev-parse --show-prefix); rel=${rel%/}

  # split oversized profile logs into 90 MB parts (cat <file>.gz.part* to rejoin)
  find "$OUTDIR" -type f -name '*.gz' -size +94M | while IFS= read -r f; do
    say "${Y}splitting (>94 MB): $f${N}"
    split -b 90m -d -a 2 "$f" "$f.part" && rm -f "$f"
  done

  # -f: the repo .gitignore has *.log, and the run's logs/*.log are the evidence
  git -C "$repo" add -f -- "$rel"
  (cd "$OUTDIR" && find . -type f -size +95M) | while IFS= read -r f; do
    say "${Y}skipping (>95 MB): $OUTDIR/${f#./}${N}"
    git -C "$OUTDIR" reset -q -- "${f#./}"
  done

  local msg="results: $(basename "$OUTDIR") (splits=$SPLITS cores=$CORES pp=$PP_TOKENS profile=$HEX_PROFILE)"
  [ -n "$TEST_OPS" ] && msg="results: $(basename "$OUTDIR") (test-ops=$TEST_OPS cores=$CORES ${AB_VAR:+$AB_VAR=$AB})"
  [ -n "$PUSH_ONLY" ] && msg="results: $(basename "$OUTDIR") (push-only)"
  if git -C "$repo" commit -q -m "$msg" -- "$rel"; then
    git -C "$repo" pull -q --rebase --autostash origin "$branch" &&
      git -C "$repo" push -q origin "$branch" &&
      say "${G}results pushed: $rel -> origin/$branch${N}" ||
      say "${R}push failed — results are committed locally; push manually${N}"
  else
    say "${Y}nothing to commit for $rel${N}"
  fi
}

#------------------------------------------------------------------------------
# --test-ops: test-backend-ops per op x cores x A/B value -> $OUTDIR/testops.tsv
#------------------------------------------------------------------------------
run_testops() {
  head1 "TEST-BACKEND-OPS ($TEST_OPS)"
  DEV_BUILD_INFO=$(device_build_info)
  say "binaries : $DEV_BUILD_INFO"
  local tsv="$OUTDIR/testops.tsv" IFSB=$IFS
  printf 'op\tcores\tab\tpassed\ttotal\tfail_lines\trc\n' > "$tsv"
  local ab_values="$AB"; [ -z "$AB_VAR" ] && ab_values="-"
  IFS=,; for op in $TEST_OPS; do IFS=$IFSB
    IFS=,; for n in $CORES; do IFS=$IFSB
      IFS=,; for abv in $ab_values; do IFS=$IFSB
        local label="testops_${op}_${n}c_ab${abv}" out
        out=$(monitored_dev "$label" test-backend-ops testops "$n" "$abv" "$op" ${TEST_OPS_PATTERN:+"$TEST_OPS_PATTERN"})
        printf '%s\n' "$out" > "$(unique_log_path "$LOGS/$label.log")"
        # "  1234/1234 tests passed" is printed per backend
        local pt passed total fails rc
        pt=$(printf '%s\n' "$out" | sed -n 's/^ *\([0-9]*\)\/\([0-9]*\) tests passed.*/\1 \2/p' | tail -1)
        passed=${pt%% *}; total=${pt##* }
        [ -n "$pt" ] || { passed=NA; total=NA; }
        fails=$(printf '%s\n' "$out" | grep -c 'FAIL' || true)
        rc=$(printf '%s\n' "$out" | sed -n 's/^TESTOPS_RC=//p' | tail -1)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$op" "$n" "$abv" "$passed" "$total" "$fails" "${rc:-NA}" >> "$tsv"
        local flag="$G"; { [ "$passed" != "$total" ] || [ "$fails" != 0 ] || [ "${rc:-1}" != 0 ]; } && flag="$R"
        say "  ${flag}$op ${n}c ${AB_VAR:-ab}=$abv: $passed/$total passed, FAIL lines $fails, rc ${rc:-NA}${N}"
      done
    done
  done
  IFS=$IFSB
  {
    echo "# test-backend-ops results"
    echo
    echo "- device: \`$ADB_SERIAL\`"
    echo "- device binaries: \`${DEV_BUILD_INFO:-unknown}\`"
    echo "- date: $(date -Iseconds)"
    [ -n "$TEST_OPS_PATTERN" ] && echo "- pattern: \`$TEST_OPS_PATTERN\`"
    echo
    echo "| op | cores | ${AB_VAR:-ab} | passed | total | FAIL lines | rc |"
    echo '|---|---|---|---|---|---|---|'
    tail -n +2 "$tsv" | awk -F'\t' '{printf "| %s | %s | %s | %s | %s | %s | %s |\n",$1,$2,$3,$4,$5,$6,$7}'
    echo
    echo "Full outputs: \`$LOGS/testops_*.log\`"
  } > "$SUMMARY"
  cat "$SUMMARY"
}

#------------------------------------------------------------------------------
main() {
  if [ -n "$PUSH_ONLY" ]; then push_results; return; fi
  preflight
  ensure_exclusive
  [ "$DO_BUILD"  = 1 ] && do_build
  [ "$DO_DEPLOY" = 1 ] && do_deploy
  write_dev_driver
  write_run_config
  prime_pdu_credentials
  if [ "$REBOOT_BEFORE_TEST" = 1 ]; then
    head1 "PRE-TEST REBOOT"
    recover_cleanup_failure "requested by --reboot-before-test" ||
      die "pre-test reboot/recovery failed — no tests were run"
  fi
  if [ -n "$TEST_OPS" ]; then
    run_testops
  else
    run_matrix
    summarise
  fi
  push_results
}

main "$@"
