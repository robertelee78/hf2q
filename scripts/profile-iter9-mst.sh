#!/usr/bin/env bash
# scripts/profile-iter9-mst.sh
#
# ADR-015 iter9 (cfa-20260427-adr015-iter9-q4_0-localize) — 5-cold-trial
# Metal System Trace capture wrapper for qwen3.6-35b-a3b dwq46 apex.
#
# Why this exists:
#   iter9 must LOCALIZE the qwen35-dwq46 D4 first-bullet 9.2% gap
#   (hf2q 110.5 t/s vs llama 121.36 t/s) at the kernel level. iter8c-prep
#   shipped a single-trial gemma-fixture per-dispatch capture
#   (scripts/profile-decode-mst.sh + aggregate_decode_mst.py); iter9 needs
#   5 cold-SoC trials × 64 decode tokens × dwq46 apex × kernel-name
#   attribution. Per spec S1: extend, do not rebuild — this script wraps
#   the existing capture.
#
# Methodology:
#   - 5 cold-SoC trials per binary (this team: hf2q only; codex: llama-cli only).
#   - 120s thermal settle between trials (per spec; 60s in iter8c-prep was
#     too tight for back-to-back same-binary captures).
#   - Pre-bench process audit per trial: ps top-10 by CPU, recorded to
#     hf2q-trial-N.process-audit. Per feedback_bench_process_audit, sibling
#     CPU >5% retries the trial.
#   - pmset -g therm gate per trial — bail on thermal warning (cold-SoC GATE).
#   - vm_stat ≥ MIN_FREE_GB before each trial.
#   - Underlying capture: xctrace record --template "Metal System Trace"
#     --launch hf2q -- generate --model dwq46 --prompt "Hello, my name is"
#     --max-tokens 64 --temperature 0
#
# Output layout (OUT_DIR=/tmp/adr015-iter9):
#   /tmp/adr015-iter9/hf2q-trial-{1..5}.trace
#   /tmp/adr015-iter9/hf2q-trial-{1..5}.{stdout,stderr,metadata.json}
#   /tmp/adr015-iter9/hf2q-trial-{1..5}.process-audit
#   /tmp/adr015-iter9/hf2q-trial-{1..5}.thermal-pre
#   /tmp/adr015-iter9/run-{DATE}.metadata.json
#
# Usage (claude side):
#   scripts/profile-iter9-mst.sh                        # 5 trials hf2q
#   ONLY=hf2q scripts/profile-iter9-mst.sh              # explicit
#   ONLY=llama scripts/profile-iter9-mst.sh             # codex side
#   N_TRIALS=2 scripts/profile-iter9-mst.sh             # smoke
#   N_TOKENS=32 scripts/profile-iter9-mst.sh            # shorter
#   THERMAL_SETTLE_SEC=180 scripts/profile-iter9-mst.sh # longer cool-down
#   SKIP_THERMAL_GATE=1 SKIP_RAM_GATE=1 scripts/profile-iter9-mst.sh

set -euo pipefail

HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"
LLAMA_BIN="${LLAMA_BIN:-/opt/homebrew/bin/llama-cli}"
FIXTURE="${FIXTURE:-/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf}"
PROMPT="${PROMPT:-Hello, my name is}"
N_TOKENS="${N_TOKENS:-64}"
N_TRIALS="${N_TRIALS:-5}"
THERMAL_SETTLE_SEC="${THERMAL_SETTLE_SEC:-120}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
OUT_DIR="${OUT_DIR:-/tmp/adr015-iter9}"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
XCTRACE="${XCTRACE:-/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace}"
ONLY="${ONLY:-hf2q}"
SIBLING_CPU_THRESHOLD="${SIBLING_CPU_THRESHOLD:-5}"

mkdir -p "$OUT_DIR"

# ---- pre-flight ----
if [[ ! -x "$HF2Q_BIN" ]]; then
  echo "ERROR: hf2q binary not found at $HF2Q_BIN" >&2
  exit 1
fi
if [[ "$ONLY" == "llama" || "$ONLY" == "both" ]]; then
  if [[ ! -x "$LLAMA_BIN" ]]; then
    echo "ERROR: llama-cli not found at $LLAMA_BIN" >&2
    exit 1
  fi
fi
if [[ ! -f "$FIXTURE" ]]; then
  echo "ERROR: fixture not found at $FIXTURE" >&2
  exit 1
fi
if [[ ! -x "$XCTRACE" ]]; then
  echo "ERROR: xctrace not found at $XCTRACE" >&2
  exit 1
fi

echo "=== ADR-015 iter9 — 5-cold-trial MST capture (qwen35-dwq46 apex) ==="
echo "binary  : $ONLY"
echo "hf2q    : $HF2Q_BIN"
echo "llama   : $LLAMA_BIN"
echo "fixture : $FIXTURE"
echo "prompt  : $PROMPT"
echo "n_tok   : $N_TOKENS"
echo "trials  : $N_TRIALS"
echo "settle  : ${THERMAL_SETTLE_SEC}s"
echo "out     : $OUT_DIR"
echo "date    : $DATE_TAG"
echo

# ---- run-level metadata ----
RUN_META="$OUT_DIR/run-${ONLY}-${DATE_TAG}.metadata.json"
{
  echo "{"
  echo "  \"date_utc\": \"$DATE_TAG\","
  echo "  \"binary\": \"$ONLY\","
  echo "  \"hostname\": \"$(hostname -s)\","
  echo "  \"macos_version\": \"$(sw_vers -productVersion)\","
  echo "  \"macos_build\": \"$(sw_vers -buildVersion)\","
  echo "  \"chip\": \"$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)\","
  echo "  \"hf2q_bin\": \"$HF2Q_BIN\","
  echo "  \"llama_bin\": \"$LLAMA_BIN\","
  echo "  \"fixture\": \"$FIXTURE\","
  echo "  \"prompt\": \"$PROMPT\","
  echo "  \"n_tokens\": $N_TOKENS,"
  echo "  \"n_trials\": $N_TRIALS,"
  echo "  \"thermal_settle_sec\": $THERMAL_SETTLE_SEC,"
  echo "  \"xctrace_version\": \"$($XCTRACE version 2>&1 | head -1)\","
  echo "  \"hf2q_git_head\": \"$(git -C /opt/hf2q rev-parse HEAD)\","
  echo "  \"hf2q_git_branch\": \"$(git -C /opt/hf2q rev-parse --abbrev-ref HEAD)\""
  echo "}"
} > "$RUN_META"
echo "wrote run metadata: $RUN_META"
echo

# ---- per-trial gates ----
ram_gate() {
  local label="$1"
  local out="$2"
  if [[ "${SKIP_RAM_GATE:-0}" == "1" ]]; then
    echo "RAM gate SKIPPED" > "$out"
    return 0
  fi
  local PAGE_SIZE FREE_PAGES INACT_PAGES SPEC_PAGES AVAIL_PAGES AVAIL_GB
  PAGE_SIZE=$(vm_stat | head -1 | awk '{print $8}' | tr -d '.')
  FREE_PAGES=$(vm_stat | awk '/Pages free/ {print $3}' | tr -d '.')
  INACT_PAGES=$(vm_stat | awk '/Pages inactive/ {print $3}' | tr -d '.')
  SPEC_PAGES=$(vm_stat | awk '/Pages speculative/ {print $3}' | tr -d '.')
  AVAIL_PAGES=$((FREE_PAGES + INACT_PAGES + SPEC_PAGES))
  AVAIL_GB=$((AVAIL_PAGES * PAGE_SIZE / 1024 / 1024 / 1024))
  echo "RAM available: ${AVAIL_GB} GB (min ${MIN_FREE_GB})" | tee "$out"
  if (( AVAIL_GB < MIN_FREE_GB )); then
    echo "FAIL: ${label} insufficient RAM (${AVAIL_GB} < ${MIN_FREE_GB})" >&2
    return 3
  fi
  return 0
}

thermal_gate() {
  local label="$1"
  local out="$2"
  if [[ "${SKIP_THERMAL_GATE:-0}" == "1" ]]; then
    echo "thermal gate SKIPPED" > "$out"
    return 0
  fi
  local THERM
  THERM="$(pmset -g therm 2>&1 || true)"
  echo "$THERM" > "$out"
  if echo "$THERM" | grep -q "CPU_Speed_Limit"; then
    local LIMIT
    LIMIT="$(echo "$THERM" | awk -F'=' '/CPU_Speed_Limit/ {print $2}' | tr -d ' ')"
    if [[ -n "$LIMIT" && "$LIMIT" != "100" ]]; then
      echo "FAIL: ${label} thermal throttle (CPU_Speed_Limit=$LIMIT)" >&2
      return 2
    fi
  fi
  # Only fail on POSITIVE thermal warning ("CPU_Scheduler_Limit", "warning level: X" with X != 0).
  # Default pmset output contains the string "No thermal warning level has been recorded" — that
  # is the cold/no-warning state, NOT a fail signal. Match `warning level=` with non-zero value.
  if echo "$THERM" | grep -qE "(CPU_Scheduler_Limit|CPU_Available_CPUs).*=[^0]"; then
    echo "FAIL: ${label} thermal/scheduler limit:" >&2
    echo "$THERM" >&2
    return 2
  fi
  if echo "$THERM" | grep -qE "(thermal|performance) warning level\s*=\s*[1-9]"; then
    echo "FAIL: ${label} thermal warning level non-zero:" >&2
    echo "$THERM" >&2
    return 2
  fi
  return 0
}

process_audit() {
  local label="$1"
  local out="$2"
  local self_pid="$$"
  ps -Ao pid,pcpu,pmem,rss,comm | sort -k2 -nr | head -10 > "$out"
  echo "--- audit ${label} ---"
  cat "$out"
  # Sibling-claude / mcp / llama / hf2q hot processes other than self
  local hot
  hot=$(ps -Ao pid,pcpu,comm | awk -v self="$self_pid" -v thr="$SIBLING_CPU_THRESHOLD" '
    NR > 1 && $1 != self && $2+0 > thr+0 && $3 !~ /xctrace/ {print}')
  if [[ -n "$hot" ]]; then
    echo "WARN: sibling process(es) over ${SIBLING_CPU_THRESHOLD}% CPU:" >&2
    echo "$hot" >&2
    echo "(retry recommended; see feedback_bench_process_audit)" >&2
    return 4
  fi
  return 0
}

# ---- single-binary capture ----
run_capture() {
  local binary="$1"   # hf2q | llama
  local trial="$2"    # 1..N
  local trace="$OUT_DIR/${binary}-trial-${trial}.trace"
  local sout="$OUT_DIR/${binary}-trial-${trial}.stdout"
  local serr="$OUT_DIR/${binary}-trial-${trial}.stderr"
  local meta="$OUT_DIR/${binary}-trial-${trial}.metadata.json"

  # Wipe any prior trace bundle (xctrace refuses to overwrite).
  rm -rf "$trace"

  local started_utc
  started_utc="$(date -u +%Y%m%dT%H%M%SZ)"

  echo "trace -> $trace"
  set +e
  if [[ "$binary" == "hf2q" ]]; then
    "$XCTRACE" record \
      --template "Metal System Trace" \
      --output "$trace" \
      --launch "$HF2Q_BIN" \
      -- generate \
         --model "$FIXTURE" \
         --prompt "$PROMPT" \
         --max-tokens "$N_TOKENS" \
         --temperature 0 \
      > "$sout" 2> "$serr"
  else
    "$XCTRACE" record \
      --template "Metal System Trace" \
      --output "$trace" \
      --launch "$LLAMA_BIN" \
      -- --model "$FIXTURE" \
         --prompt "$PROMPT" \
         --n-predict "$N_TOKENS" \
         --temp 0 \
         --no-conversation \
         --n-gpu-layers 999 \
      > "$sout" 2> "$serr"
  fi
  local rc=$?
  set -e

  local finished_utc
  finished_utc="$(date -u +%Y%m%dT%H%M%SZ)"

  {
    echo "{"
    echo "  \"binary\": \"$binary\","
    echo "  \"trial\": $trial,"
    echo "  \"started_utc\": \"$started_utc\","
    echo "  \"finished_utc\": \"$finished_utc\","
    echo "  \"trace\": \"$trace\","
    echo "  \"exit_code\": $rc,"
    echo "  \"n_tokens\": $N_TOKENS,"
    echo "  \"prompt\": \"$PROMPT\","
    echo "  \"fixture\": \"$FIXTURE\""
    echo "}"
  } > "$meta"

  if (( rc != 0 )); then
    echo "WARN: ${binary} trial ${trial} exit ${rc}; tail of stderr:" >&2
    tail -20 "$serr" >&2 || true
  fi
  return $rc
}

# ---- top-level loop ----
case "$ONLY" in
  hf2q|llama|both) ;;
  *) echo "ERROR: unknown ONLY=$ONLY (want hf2q|llama|both)" >&2; exit 1;;
esac

binaries=()
case "$ONLY" in
  hf2q)  binaries=(hf2q) ;;
  llama) binaries=(llama) ;;
  both)  binaries=(hf2q llama) ;;
esac

for binary in "${binaries[@]}"; do
  for trial in $(seq 1 "$N_TRIALS"); do
    echo
    echo "=== ${binary} trial ${trial}/${N_TRIALS} ==="

    if (( trial > 1 )); then
      echo "thermal settle ${THERMAL_SETTLE_SEC}s..."
      sleep "$THERMAL_SETTLE_SEC"
    fi

    thermal_pre="$OUT_DIR/${binary}-trial-${trial}.thermal-pre"
    audit_pre="$OUT_DIR/${binary}-trial-${trial}.process-audit"
    ram_log="$OUT_DIR/${binary}-trial-${trial}.ram"

    if ! ram_gate "${binary}-trial-${trial}" "$ram_log"; then exit 3; fi
    if ! thermal_gate "${binary}-trial-${trial}" "$thermal_pre"; then exit 2; fi
    process_audit "${binary}-trial-${trial}" "$audit_pre" || true

    run_capture "$binary" "$trial" || true
  done
done

echo
echo "=== iter9 captures complete ==="
ls -la "$OUT_DIR"/*.trace 2>/dev/null || true
echo
echo "Next: aggregate via scripts/aggregate-q4_0-mst.py"
