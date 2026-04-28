#!/bin/bash
# Wave 5b.11 sub-task: HF2Q_QWEN35_FA_LEGACY=1 sunset parity audit.
#
# Per W-5b.10 closure plan: 5 cold model loads × 3 cold prefills × 2 paths
# (legacy + new) = 30 runs. All 30 must produce token id 11 (`,`) — if any
# run drifts, the legacy escape hatch is KEPT and the failure documented.
#
# Each iteration of the outer loop is a fresh process (cold first-forward).
# We do NOT reuse model loads across paths — every "load" is a new process.
#
# Outputs: /tmp/w5b11-sunset/{new,legacy}-load{N}-trial{T}.log

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
LOADS=${LOADS:-5}
TRIALS_PER_LOAD=${TRIALS_PER_LOAD:-3}

if [ ! -f "$MODEL" ]; then echo "ERROR: model not found"; exit 1; fi
if [ ! -f "$PROMPT" ]; then echo "ERROR: prompt not found"; exit 1; fi

mkdir -p /tmp/w5b11-sunset

run_one() {
  local label="$1"
  local outfile="$2"
  shift 2
  /usr/bin/time -p env "$@" \
    /opt/hf2q/target/release/hf2q -v generate --model "$MODEL" \
    --prompt-file "$PROMPT" \
    --chat-template '{{ messages[0]["content"] }}' \
    --max-tokens 2 --temperature 0.0 > "$outfile" 2>&1
  local prefill_ms token real_s
  prefill_ms=$(grep -oE 'prefill: [0-9]+ tok in [0-9]+ms' "$outfile" | sed -E 's/.* in ([0-9]+)ms/\1/' | head -1)
  token=$(grep -oE 'first decoded token: [0-9]+' "$outfile" | sed -E 's/.*: ([0-9]+)/\1/' | head -1)
  real_s=$(awk '/^real/ {print $2}' "$outfile")
  printf '%-50s prefill=%6sms tok=%5s real=%5ss\n' \
    "$label" "${prefill_ms:-???}" "${token:-???}" "${real_s:-???}"
}

echo "=== Wave 5b.11 sunset parity audit (HF2Q_QWEN35_FA_LEGACY=1 removal pre-flight) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo "Loads: $LOADS, Trials per load: $TRIALS_PER_LOAD, Paths: 2 (new + legacy)"
echo "Total runs: $((LOADS * TRIALS_PER_LOAD * 2))"
echo

PASS_NEW=0
FAIL_NEW=0
PASS_LEGACY=0
FAIL_LEGACY=0

for L in $(seq 1 "$LOADS"); do
  for T in $(seq 1 "$TRIALS_PER_LOAD"); do
    run_one "[L${L}T${T}] CHUNK new" "/tmp/w5b11-sunset/new-L${L}-T${T}.log" \
      HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
    tok=$(grep -oE 'first decoded token: [0-9]+' "/tmp/w5b11-sunset/new-L${L}-T${T}.log" | sed -E 's/.*: ([0-9]+)/\1/' | head -1)
    if [ "$tok" = "11" ]; then PASS_NEW=$((PASS_NEW+1)); else FAIL_NEW=$((FAIL_NEW+1)); fi

    run_one "[L${L}T${T}] CHUNK legacy" "/tmp/w5b11-sunset/legacy-L${L}-T${T}.log" \
      HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1 \
      HF2Q_QWEN35_FA_LEGACY=1
    tok=$(grep -oE 'first decoded token: [0-9]+' "/tmp/w5b11-sunset/legacy-L${L}-T${T}.log" | sed -E 's/.*: ([0-9]+)/\1/' | head -1)
    if [ "$tok" = "11" ]; then PASS_LEGACY=$((PASS_LEGACY+1)); else FAIL_LEGACY=$((FAIL_LEGACY+1)); fi
  done
done

echo
echo "=== SUNSET PARITY AUDIT SUMMARY ==="
echo "NEW path:    PASS=$PASS_NEW / FAIL=$FAIL_NEW"
echo "LEGACY path: PASS=$PASS_LEGACY / FAIL=$FAIL_LEGACY"
TOTAL_PASS=$((PASS_NEW + PASS_LEGACY))
TOTAL_RUNS=$((LOADS * TRIALS_PER_LOAD * 2))
echo "Total: $TOTAL_PASS / $TOTAL_RUNS"
if [ "$TOTAL_PASS" = "$TOTAL_RUNS" ]; then
  echo "VERDICT: PARITY HOLDS — legacy gate is safe to remove."
  exit 0
else
  echo "VERDICT: PARITY BROKEN — KEEP legacy gate; document failure."
  exit 1
fi
