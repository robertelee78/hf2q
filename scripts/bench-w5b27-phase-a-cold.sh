#!/bin/bash
# Wave 5b.27 Phase A — cold-SoC retro-validation (per
# `feedback_perf_gate_thermal_methodology`: thermal pre-loading from
# repeated benches fakes ~5-10% regression on M5 Max).
#
# Inserts 60s sleep between trials to ensure each is cold-SoC. 6 trials.

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
TRIALS=${TRIALS:-6}
COOLDOWN_S=${COOLDOWN_S:-60}

if [ ! -f "$MODEL" ]; then echo "ERROR: model not found: $MODEL"; exit 1; fi
if [ ! -f "$PROMPT" ]; then echo "ERROR: prompt not found: $PROMPT"; exit 1; fi

mkdir -p /tmp/w5b27

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
  printf '%-44s prefill=%6sms tok=%5s real=%5ss\n' \
    "$label" "${prefill_ms:-???}" "${token:-???}" "${real_s:-???}"
}

echo "=== Wave 5b.27 Phase A cold-SoC retro-validation ($TRIALS cold trials × ${COOLDOWN_S}s cooldown) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo

# Trials 1,3,5 hf2q  → pause 60s  → trials 2,4,6 hf2q (alternates with llama? no — full 60s gap)
echo "--- hf2q chunk path current HEAD ($TRIALS cold trials, ${COOLDOWN_S}s cooldown each) ---"
for t in $(seq 1 "$TRIALS"); do
  if [ "$t" -gt 1 ]; then
    echo "  [cooldown ${COOLDOWN_S}s before T${t}]"
    sleep "$COOLDOWN_S"
  fi
  run_one "[T${t}] W-5b.27 phaseA HEAD" "/tmp/w5b27/coldA-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1 HF2Q_PROFILE_W5B22=1 \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

echo "--- llama baseline (3 cold trials, ${COOLDOWN_S}s cooldown each) ---"
for t in 1 2 3; do
  echo "  [cooldown ${COOLDOWN_S}s before llama T${t}]"
  sleep "$COOLDOWN_S"
  /usr/bin/time -p /opt/homebrew/bin/llama-completion \
    --model "$MODEL" \
    --batch-size 4096 --ubatch-size 4096 --n-predict 1 --no-warmup -no-cnv --perf \
    -f "$PROMPT" > "/tmp/w5b27/coldA-llama-T${t}.log" 2>&1
  prefill_ms=$(grep -oE 'prompt eval time = +[0-9.]+ ms' "/tmp/w5b27/coldA-llama-T${t}.log" | sed -E 's/.*= +([0-9.]+) ms/\1/' | head -1)
  printf '[llama T%d] prompt_eval=%sms\n' "$t" "${prefill_ms:-???}"
done

echo
echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Logs in /tmp/w5b27/coldA-*.log"
