#!/bin/bash
# Wave 5b.7 iter 2 — residency-set adoption perf measurement
# Runs T=256 prompt × 2 paths (chunk, autoreg) × 2 conditions (residency, no-residency)
# × N trials. Each trial is a fresh process (cold first-forward).
#
# Token id 561 expected at T=256 (W-5b.5 baseline).

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/diag-prompt.txt
TRIALS=${TRIALS:-3}

if [ ! -f "$MODEL" ]; then
  echo "ERROR: model not found at $MODEL"; exit 1
fi
if [ ! -f "$PROMPT" ]; then
  echo "ERROR: prompt not found at $PROMPT"; exit 1
fi

run_one() {
  local label="$1"
  shift
  local out
  out=$( /usr/bin/time -p env "$@" \
    ./target/release/hf2q -v generate --model "$MODEL" \
    --prompt-file "$PROMPT" \
    --chat-template '{{ messages[0]["content"] }}' \
    --max-tokens 2 --temperature 0.0 2>&1 )
  local prefill_ms decode_tps token real_s loaded_s
  prefill_ms=$(echo "$out" | sed -nE 's/.*prefill: 256 tok in ([0-9]+)ms.*/\1/p' | head -1)
  decode_tps=$(echo "$out" | sed -nE 's/.*: 2 tokens in [0-9.]+s \(([0-9.]+) tok\/s\).*/\1/p' | head -1)
  token=$(echo "$out" | sed -nE 's/.*first decoded token: ([0-9]+).*/\1/p' | head -1)
  real_s=$(echo "$out" | awk '/^real/ {print $2}')
  loaded_s=$(echo "$out" | sed -nE 's/.*loaded in ([0-9.]+)s.*/\1/p' | head -1)
  printf '%-40s prefill=%6sms decode=%6s tok/s tok=%5s loaded=%5ss real=%5ss\n' \
    "$label" "${prefill_ms:-???}" "${decode_tps:-???}" "${token:-???}" "${loaded_s:-???}" "${real_s:-???}"
}

echo "=== Wave 5b.7 iter 2 residency-set perf bench ($TRIALS trials per condition) ==="
echo "HEAD: $(git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | head -1; vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo

for t in $(seq 1 "$TRIALS"); do
  echo "--- Trial $t ---"
  run_one "[T${t}] CHUNK   pre-residency  " HF2Q_NO_RESIDENCY=1 HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
  run_one "[T${t}] CHUNK   post-residency " HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
  run_one "[T${t}] AUTOREG pre-residency  " HF2Q_NO_RESIDENCY=1 HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1
  run_one "[T${t}] AUTOREG post-residency " HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1
done

echo
echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
