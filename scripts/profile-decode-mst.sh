#!/usr/bin/env bash
# scripts/profile-decode-mst.sh
#
# ADR-015 iter8c-prep — production-mode per-kernel attribution via
# `xctrace record --template "Metal System Trace"` on hf2q + llama-cli.
#
# Why this exists (per ADR-015 §iter8c-prep changelog 2026-04-27):
#   iter8b reported 14-37x per-kernel ratio gaps to a candle baseline using
#   HF2Q_MLX_KERNEL_PROFILE=1 mode (242 sessions/token vs 1 in production).
#   Side-by-side audit showed the kernels are byte-equivalent to llama.cpp's,
#   so the iter8b numbers are kprofile-mode artifacts, not production gaps.
#   Before launching iter8c shader work (NAX routing per ADR-015 P3c) we
#   need same-day production-mode per-kernel attribution against llama.cpp.
#
# Methodology:
#   - Single trial per binary (Metal System Trace overhead is significant;
#     multi-trial median doesn't help when we're after kernel ATTRIBUTION
#     not bench numbers; the bench ratio comes from bench-baseline.sh).
#   - Same model fixture for both binaries (gemma-4-26B-A4B-it-ara-abliterated-dwq
#     is the iter8b reference; it's the smaller of the two ADR-015 targets and
#     fits comfortably in unified memory alongside llama-cli's load).
#   - Same n_gen (32 default — short enough to keep trace size manageable,
#     long enough for steady-state per-kernel signal).
#   - Cold SoC enforced (pmset -g therm gate + RAM headroom precheck).
#   - Run hf2q FIRST, settle 60s, then llama-cli — order matters because
#     llama-cli's Metal cache warmup may bias the second run otherwise.
#
# Outputs:
#   ${OUT_DIR}/hf2q-decode-${DATE}.trace
#   ${OUT_DIR}/llama-decode-${DATE}.trace
#   ${OUT_DIR}/hf2q-decode-${DATE}.{stdout,stderr,metadata.json}
#   ${OUT_DIR}/llama-decode-${DATE}.{stdout,stderr,metadata.json}
#   ${OUT_DIR}/run-${DATE}.metadata.json   (hardware + git head)
#
# Aggregation: pipe traces through scripts/aggregate_decode_mst.py
# (separate script — produces per-kernel µs/dispatch + count + total).
#
# Usage:
#   scripts/profile-decode-mst.sh                 # gemma fixture, n_gen 32
#   FIXTURE=/path/to/q4_0.gguf scripts/profile-decode-mst.sh
#   N_TOKENS=64 scripts/profile-decode-mst.sh     # longer
#   SKIP_THERMAL_GATE=1 scripts/profile-decode-mst.sh
#   SKIP_RAM_GATE=1 scripts/profile-decode-mst.sh # smoke only
#   ONLY=hf2q scripts/profile-decode-mst.sh       # capture hf2q only
#   ONLY=llama scripts/profile-decode-mst.sh      # capture llama only

set -euo pipefail

HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"
LLAMA_BIN="${LLAMA_BIN:-/opt/homebrew/bin/llama-cli}"
FIXTURE="${FIXTURE:-/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf}"
PROMPT="${PROMPT:-Hello, my name is}"
N_TOKENS="${N_TOKENS:-32}"
THERMAL_SETTLE_SEC="${THERMAL_SETTLE_SEC:-60}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
OUT_DIR="${OUT_DIR:-/tmp/adr015-iter8c-prep}"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
XCTRACE="${XCTRACE:-/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace}"
ONLY="${ONLY:-both}"

mkdir -p "$OUT_DIR"

if [[ ! -x "$HF2Q_BIN" ]]; then
  echo "ERROR: hf2q binary not found at $HF2Q_BIN" >&2
  exit 1
fi
if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "ERROR: llama-cli not found at $LLAMA_BIN" >&2
  exit 1
fi
if [[ ! -f "$FIXTURE" ]]; then
  echo "ERROR: fixture not found at $FIXTURE" >&2
  exit 1
fi
if [[ ! -x "$XCTRACE" ]]; then
  echo "ERROR: xctrace not found at $XCTRACE" >&2
  exit 1
fi

echo "=== ADR-015 iter8c-prep — metal-system-trace decode capture ==="
echo "hf2q   : $HF2Q_BIN"
echo "llama  : $LLAMA_BIN"
echo "fixture: $FIXTURE"
echo "prompt : $PROMPT"
echo "n_tok  : $N_TOKENS"
echo "out    : $OUT_DIR"
echo "only   : $ONLY"
echo

if [[ "${SKIP_RAM_GATE:-0}" != "1" ]]; then
  PAGE_SIZE=$(vm_stat | head -1 | awk '{print $8}' | tr -d '.')
  FREE_PAGES=$(vm_stat | awk '/Pages free/ {print $3}' | tr -d '.')
  INACT_PAGES=$(vm_stat | awk '/Pages inactive/ {print $3}' | tr -d '.')
  SPEC_PAGES=$(vm_stat | awk '/Pages speculative/ {print $3}' | tr -d '.')
  AVAIL_PAGES=$((FREE_PAGES + INACT_PAGES + SPEC_PAGES))
  AVAIL_GB=$((AVAIL_PAGES * PAGE_SIZE / 1024 / 1024 / 1024))
  echo "RAM available: ${AVAIL_GB} GB (min: ${MIN_FREE_GB} GB)"
  if (( AVAIL_GB < MIN_FREE_GB )); then
    echo "ERROR: insufficient RAM headroom (need ${MIN_FREE_GB} GB, have ${AVAIL_GB} GB)" >&2
    exit 3
  fi
fi

if [[ "${SKIP_THERMAL_GATE:-0}" != "1" ]]; then
  THERM="$(pmset -g therm 2>&1 || true)"
  if echo "$THERM" | grep -q "CPU_Speed_Limit"; then
    LIMIT="$(echo "$THERM" | awk -F'=' '/CPU_Speed_Limit/ {print $2}' | tr -d ' ')"
    if [[ -n "$LIMIT" && "$LIMIT" != "100" ]]; then
      echo "ERROR: thermal throttle (CPU_Speed_Limit=$LIMIT)" >&2
      exit 2
    fi
  fi
fi

RUN_META="$OUT_DIR/run-${DATE_TAG}.metadata.json"
{
  echo "{"
  echo "  \"date_utc\": \"$DATE_TAG\","
  echo "  \"hostname\": \"$(hostname -s)\","
  echo "  \"macos_version\": \"$(sw_vers -productVersion)\","
  echo "  \"macos_build\": \"$(sw_vers -buildVersion)\","
  echo "  \"chip\": \"$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)\","
  echo "  \"hf2q_bin\": \"$HF2Q_BIN\","
  echo "  \"llama_bin\": \"$LLAMA_BIN\","
  echo "  \"fixture\": \"$FIXTURE\","
  echo "  \"prompt\": \"$PROMPT\","
  echo "  \"n_tokens\": $N_TOKENS,"
  echo "  \"xctrace_version\": \"$($XCTRACE version 2>&1 | head -1)\","
  echo "  \"hf2q_git_head\": \"$(git -C /opt/hf2q rev-parse HEAD)\""
  echo "}"
} > "$RUN_META"
echo "wrote run metadata: $RUN_META"

run_hf2q_capture() {
  echo
  echo "--- hf2q decode capture ---"
  local TRACE="$OUT_DIR/hf2q-decode-${DATE_TAG}.trace"
  local SOUT="$OUT_DIR/hf2q-decode-${DATE_TAG}.stdout"
  local SERR="$OUT_DIR/hf2q-decode-${DATE_TAG}.stderr"
  echo "trace -> $TRACE"
  set +e
  "$XCTRACE" record \
    --template "Metal System Trace" \
    --output "$TRACE" \
    --launch "$HF2Q_BIN" \
    -- generate \
       --model "$FIXTURE" \
       --prompt "$PROMPT" \
       --max-tokens "$N_TOKENS" \
       --temperature 0 \
    > "$SOUT" 2> "$SERR"
  local RC=$?
  set -e
  echo "exit code: $RC"
  if (( RC != 0 )); then
    echo "WARN: hf2q capture exit non-zero; tail of stderr:" >&2
    tail -20 "$SERR" >&2 || true
  fi
}

run_llama_capture() {
  echo
  echo "--- llama-cli decode capture ---"
  local TRACE="$OUT_DIR/llama-decode-${DATE_TAG}.trace"
  local SOUT="$OUT_DIR/llama-decode-${DATE_TAG}.stdout"
  local SERR="$OUT_DIR/llama-decode-${DATE_TAG}.stderr"
  echo "trace -> $TRACE"
  set +e
  "$XCTRACE" record \
    --template "Metal System Trace" \
    --output "$TRACE" \
    --launch "$LLAMA_BIN" \
    -- --model "$FIXTURE" \
       --prompt "$PROMPT" \
       --n-predict "$N_TOKENS" \
       --temp 0 \
       --no-conversation \
       --n-gpu-layers 999 \
    > "$SOUT" 2> "$SERR"
  local RC=$?
  set -e
  echo "exit code: $RC"
  if (( RC != 0 )); then
    echo "WARN: llama capture exit non-zero; tail of stderr:" >&2
    tail -20 "$SERR" >&2 || true
  fi
}

case "$ONLY" in
  hf2q)
    run_hf2q_capture
    ;;
  llama)
    run_llama_capture
    ;;
  both)
    run_hf2q_capture
    echo "settling ${THERMAL_SETTLE_SEC}s before llama capture..."
    sleep "$THERMAL_SETTLE_SEC"
    run_llama_capture
    ;;
  *)
    echo "ERROR: unknown ONLY=$ONLY (want hf2q|llama|both)" >&2
    exit 1
    ;;
esac

echo
echo "=== captures complete ==="
ls -la "$OUT_DIR"/*.trace 2>/dev/null || true
echo
echo "Next: aggregate per-kernel times (script TODO)."
