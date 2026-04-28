#!/bin/bash
# Wave 5b.27 Phase B — DenseFfnOutputCache A/B bench at PP4106.
#
# Compares hf2q NEW path (default, lifted DenseFfnOutputCache) vs
# hf2q LEGACY path (HF2Q_FFN_DENSE_LIFT_LEGACY=1, per-layer
# pooled_alloc_buffer) at PP4106 with W-5b.8/W-5b.17/W-5b.22
# instrumentation enabled.  60s cooldown between trials per cold-SoC
# methodology.
#
# Pre-flight (per worker prompt):
#  - vm_stat free × 16 KB > 32 GB
#  - mcp-brain-server / WebKit content paused via `kill -STOP <pid>`
#
# Closure rule: NEW must drop FFN bucket by >= 50 ms vs Phase A median
# (3,265.5 ms, NOT W-5b.21's 3,316 ms or W-5b.24's 3,229.6 ms).
# Target: NEW median <= 3,215.5 ms.

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
TRIALS=${TRIALS:-3}
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

echo "=== Wave 5b.27 Phase B DenseFfnOutputCache A/B bench ($TRIALS × 2 paths × ${COOLDOWN_S}s cooldown) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo

# (1) hf2q NEW path — default, lifted DenseFfnOutputCache.
echo "--- hf2q chunk path NEW (lifted DenseFfnOutputCache, $TRIALS cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  if [ "$t" -gt 1 ]; then
    echo "  [cooldown ${COOLDOWN_S}s before NEW T${t}]"
    sleep "$COOLDOWN_S"
  fi
  run_one "[T${t}] W-5b.27 NEW (lifted)" "/tmp/w5b27/B-new-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1 HF2Q_PROFILE_W5B22=1 \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

# (2) hf2q LEGACY path — HF2Q_FFN_DENSE_LIFT_LEGACY=1, per-layer pool alloc.
echo "  [cooldown ${COOLDOWN_S}s before LEGACY trials]"
sleep "$COOLDOWN_S"
echo
echo "--- hf2q chunk path LEGACY (per-layer pool alloc, $TRIALS cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  if [ "$t" -gt 1 ]; then
    echo "  [cooldown ${COOLDOWN_S}s before LEGACY T${t}]"
    sleep "$COOLDOWN_S"
  fi
  run_one "[T${t}] W-5b.27 LEGACY (per-layer pool)" "/tmp/w5b27/B-legacy-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1 HF2Q_PROFILE_W5B22=1 \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1 \
    HF2Q_FFN_DENSE_LIFT_LEGACY=1
done
echo

# (3) llama drift gate (1 trial — Phase A already ran 3, this confirms day-of stability).
echo "  [cooldown ${COOLDOWN_S}s before llama drift]"
sleep "$COOLDOWN_S"
echo
echo "--- llama same-day drift gate (1 trial) ---"
/usr/bin/time -p /opt/homebrew/bin/llama-completion \
  --model "$MODEL" \
  --batch-size 4096 --ubatch-size 4096 --n-predict 1 --no-warmup -no-cnv --perf \
  -f "$PROMPT" > "/tmp/w5b27/B-llama-T1.log" 2>&1
prefill_ms=$(grep -oE 'prompt eval time = +[0-9.]+ ms' "/tmp/w5b27/B-llama-T1.log" | sed -E 's/.*= +([0-9.]+) ms/\1/' | head -1)
printf '[llama T1] prompt_eval=%sms\n' "${prefill_ms:-???}"

echo
echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Logs in /tmp/w5b27/B-*.log"
