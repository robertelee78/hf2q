#!/bin/bash
# Wave 5b.21 Phase B — fresh re-audit post-W-5b.20 mega-encoder landing.
#
# Single hf2q path (NEW: repeat_tiled folded into mega-encoder, no env gates
# — measures the production behaviour after Phase A's sunset removal). 3
# cold trials + 3 cold llama at PP4106 with full W-5b.8 + W-5b.17
# instrumentation. Goal: identify NEW top-3 contributors to the
# 15,859 ms whole-prefill wall (vs same-day llama for the 9,222 ms apples-
# to-apples gap).
#
# Pre-flight (per worker prompt):
#  - vm_stat free × 16KB > 32 GB
#  - mcp-brain-server paused via `kill -STOP <pid>`
#  - llama drift gate ≤7,300 ms (10 % above W-5b.20's 6,637 ms)

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
TRIALS=${TRIALS:-3}

if [ ! -f "$MODEL" ]; then echo "ERROR: model not found: $MODEL"; exit 1; fi
if [ ! -f "$PROMPT" ]; then echo "ERROR: prompt not found: $PROMPT"; exit 1; fi

mkdir -p /tmp/w5b21-reaudit

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
  printf '%-40s prefill=%6sms tok=%5s real=%5ss\n' \
    "$label" "${prefill_ms:-???}" "${token:-???}" "${real_s:-???}"
}

echo "=== Wave 5b.21 Phase B re-audit ($TRIALS trials hf2q + $TRIALS llama at PP4106) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo

# (1) Same-day llama baseline (drift gate ≤10 % vs W-5b.20's 6,637 ms = 7,300 ms ceiling).
echo "--- llama baseline ($TRIALS cold trials at pp4106) ---"
for t in $(seq 1 "$TRIALS"); do
  /usr/bin/time -p /opt/homebrew/bin/llama-completion \
    --model "$MODEL" \
    --batch-size 4096 --ubatch-size 4096 --n-predict 1 --no-warmup -no-cnv --perf \
    -f "$PROMPT" > "/tmp/w5b21-reaudit/llama-T${t}.log" 2>&1
  prefill_ms=$(grep -oE 'prompt eval time = +[0-9.]+ ms' "/tmp/w5b21-reaudit/llama-T${t}.log" | sed -E 's/.*= +([0-9.]+) ms/\1/' | head -1)
  printf '[llama T%d] prompt_eval=%sms\n' "$t" "${prefill_ms:-???}"
done
echo

# (2) hf2q NEW path (production post-W-5b.20 + Phase A sunset).
echo "--- hf2q chunk path post-W-5b.20 ($TRIALS cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] W-5b.21 reaudit" "/tmp/w5b21-reaudit/hf2q-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1 \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Logs in /tmp/w5b21-reaudit/"
