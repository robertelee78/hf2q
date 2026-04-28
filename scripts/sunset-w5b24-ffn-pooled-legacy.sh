#!/bin/bash
# Wave 5b.25 Phase A — HF2Q_FFN_POOLED_LEGACY=1 sunset parity audit.
#
# Per W-5b.24 closure plan (mirrors W-5b.16/19A/21A cadence):
#   5 cold model loads × 3 cold prefills × 2 paths (NEW + LEGACY) = 30 runs
#   at PP4106 with Qwen3.6 27B DWQ46. All 30 must produce token id 11 (`,`).
#
# NEW addition relative to prior sunsets: one apex 35B-A3B PP65536 cross-path
# verification (LEGACY + NEW each run once at the apex shape) — catches edge
# cases the smaller 27B 4106-prefill audit might miss. Apex skipped if model
# absent (documented; does NOT block the 27B sunset on its own).
#
# Each iteration of the outer loop is a fresh process (cold first-forward).
# We do NOT reuse model loads across paths — every "load" is a new process.
#
# Outputs: /tmp/w5b25-sunset/{new,legacy}-loadN-trialT.log
#          /tmp/w5b25-sunset/apex-{new,legacy}.log
#
# NOTE: foreground sequential only. No `&`, no `nohup`, no parallel.

set -u

MODEL_27B=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
MODEL_APEX=/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf
PROMPT_27B=/tmp/walkbar-pp4096-prompt.txt
PROMPT_APEX=/tmp/walkbar-pp65536-prompt.txt
LOADS=${LOADS:-5}
TRIALS_PER_LOAD=${TRIALS_PER_LOAD:-3}
SKIP_APEX=${SKIP_APEX:-0}

if [ ! -f "$MODEL_27B" ]; then echo "ERROR: 27B model not found: $MODEL_27B"; exit 1; fi
if [ ! -f "$PROMPT_27B" ]; then echo "ERROR: pp4096 prompt not found: $PROMPT_27B"; exit 1; fi

mkdir -p /tmp/w5b25-sunset

run_one() {
  local label="$1"
  local outfile="$2"
  local model="$3"
  local prompt="$4"
  shift 4
  /usr/bin/time -p env "$@" \
    /opt/hf2q/target/release/hf2q -v generate --model "$model" \
    --prompt-file "$prompt" \
    --chat-template '{{ messages[0]["content"] }}' \
    --max-tokens 2 --temperature 0.0 > "$outfile" 2>&1
  local prefill_ms token real_s
  prefill_ms=$(grep -oE 'prefill: [0-9]+ tok in [0-9]+ms' "$outfile" | sed -E 's/.* in ([0-9]+)ms/\1/' | head -1)
  token=$(grep -oE 'first decoded token: [0-9]+' "$outfile" | sed -E 's/.*: ([0-9]+)/\1/' | head -1)
  real_s=$(awk '/^real/ {print $2}' "$outfile")
  printf '%-50s prefill=%6sms tok=%5s real=%5ss\n' \
    "$label" "${prefill_ms:-???}" "${token:-???}" "${real_s:-???}"
}

echo "=== Wave 5b.25 sunset parity audit (HF2Q_FFN_POOLED_LEGACY=1 removal pre-flight) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "27B Prompt SHA: $(shasum "$PROMPT_27B" | awk '{print $1}')"
echo "Loads: $LOADS, Trials per load: $TRIALS_PER_LOAD, Paths: 2 (new + legacy)"
echo "Total 27B runs: $((LOADS * TRIALS_PER_LOAD * 2))"
echo

PASS_NEW=0
FAIL_NEW=0
PASS_LEGACY=0
FAIL_LEGACY=0

for L in $(seq 1 "$LOADS"); do
  for T in $(seq 1 "$TRIALS_PER_LOAD"); do
    run_one "[L${L}T${T}] FFN_POOLED new (default)" "/tmp/w5b25-sunset/new-L${L}-T${T}.log" \
      "$MODEL_27B" "$PROMPT_27B" \
      HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
    tok=$(grep -oE 'first decoded token: [0-9]+' "/tmp/w5b25-sunset/new-L${L}-T${T}.log" | sed -E 's/.*: ([0-9]+)/\1/' | head -1)
    if [ "$tok" = "11" ]; then PASS_NEW=$((PASS_NEW+1)); else FAIL_NEW=$((FAIL_NEW+1)); fi

    run_one "[L${L}T${T}] FFN_POOLED legacy" "/tmp/w5b25-sunset/legacy-L${L}-T${T}.log" \
      "$MODEL_27B" "$PROMPT_27B" \
      HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1 \
      HF2Q_FFN_POOLED_LEGACY=1
    tok=$(grep -oE 'first decoded token: [0-9]+' "/tmp/w5b25-sunset/legacy-L${L}-T${T}.log" | sed -E 's/.*: ([0-9]+)/\1/' | head -1)
    if [ "$tok" = "11" ]; then PASS_LEGACY=$((PASS_LEGACY+1)); else FAIL_LEGACY=$((FAIL_LEGACY+1)); fi
  done
done

echo
echo "=== 27B SUNSET PARITY AUDIT SUMMARY ==="
echo "NEW path:    PASS=$PASS_NEW / FAIL=$FAIL_NEW"
echo "LEGACY path: PASS=$PASS_LEGACY / FAIL=$FAIL_LEGACY"
TOTAL_PASS=$((PASS_NEW + PASS_LEGACY))
TOTAL_RUNS=$((LOADS * TRIALS_PER_LOAD * 2))
echo "Total: $TOTAL_PASS / $TOTAL_RUNS"
echo

# === Apex 35B-A3B PP65536 cross-path verification (NEW for W-5b.25) ===
APEX_NEW_TOK="(skipped)"
APEX_LEGACY_TOK="(skipped)"
APEX_NEW_PREFILL="(skipped)"
APEX_LEGACY_PREFILL="(skipped)"

if [ "$SKIP_APEX" = "1" ]; then
  echo "=== APEX VERIFICATION SKIPPED (SKIP_APEX=1) ==="
elif [ ! -f "$MODEL_APEX" ]; then
  echo "=== APEX VERIFICATION SKIPPED (apex 35B-A3B model not present at $MODEL_APEX) ==="
elif [ ! -f "$PROMPT_APEX" ]; then
  echo "=== APEX VERIFICATION SKIPPED (PP65536 prompt not present at $PROMPT_APEX) ==="
  echo "    Generate via: e.g. hash-of-w5b24 procedure; not auto-built by this script."
else
  echo "=== APEX 35B-A3B PP65536 cross-path verification (1 cold prefill per path) ==="
  echo "Apex prompt SHA: $(shasum "$PROMPT_APEX" | awk '{print $1}')"
  run_one "[APEX] FFN_POOLED new (default)" "/tmp/w5b25-sunset/apex-new.log" \
    "$MODEL_APEX" "$PROMPT_APEX" \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
  APEX_NEW_TOK=$(grep -oE 'first decoded token: [0-9]+' /tmp/w5b25-sunset/apex-new.log | sed -E 's/.*: ([0-9]+)/\1/' | head -1)
  APEX_NEW_PREFILL=$(grep -oE 'prefill: [0-9]+ tok in [0-9]+ms' /tmp/w5b25-sunset/apex-new.log | sed -E 's/.* in ([0-9]+)ms/\1/' | head -1)

  run_one "[APEX] FFN_POOLED legacy" "/tmp/w5b25-sunset/apex-legacy.log" \
    "$MODEL_APEX" "$PROMPT_APEX" \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1 \
    HF2Q_FFN_POOLED_LEGACY=1
  APEX_LEGACY_TOK=$(grep -oE 'first decoded token: [0-9]+' /tmp/w5b25-sunset/apex-legacy.log | sed -E 's/.*: ([0-9]+)/\1/' | head -1)
  APEX_LEGACY_PREFILL=$(grep -oE 'prefill: [0-9]+ tok in [0-9]+ms' /tmp/w5b25-sunset/apex-legacy.log | sed -E 's/.* in ([0-9]+)ms/\1/' | head -1)

  echo
  echo "APEX NEW    : tok=${APEX_NEW_TOK:-???} prefill=${APEX_NEW_PREFILL:-???}ms"
  echo "APEX LEGACY : tok=${APEX_LEGACY_TOK:-???} prefill=${APEX_LEGACY_PREFILL:-???}ms"
  if [ -n "${APEX_NEW_TOK:-}" ] && [ -n "${APEX_LEGACY_TOK:-}" ] && [ "$APEX_NEW_TOK" = "$APEX_LEGACY_TOK" ]; then
    echo "APEX cross-path token-id parity: PASS (id=$APEX_NEW_TOK)"
  else
    echo "APEX cross-path token-id parity: FAIL (new=$APEX_NEW_TOK legacy=$APEX_LEGACY_TOK)"
  fi
fi

echo
echo "=== FINAL VERDICT ==="
if [ "$TOTAL_PASS" = "$TOTAL_RUNS" ]; then
  echo "27B 30/30 PARITY: PASS — legacy gate is safe to remove."
  exit 0
else
  echo "27B PARITY BROKEN — KEEP legacy gate; document failure mode."
  exit 1
fi
