#!/bin/bash
# Wave 5b.22 — instrument the W-5b.21 `layer.linear_total` 3,318 ms
# unaccounted residual via DN-only outer-choreography sister buckets.
#
# Single hf2q path (production post-W-5b.21 sunset; no env A/B).
# 3 cold trials hf2q + 3 cold llama at PP4106 with full W-5b.8 +
# W-5b.17 + W-5b.22 instrumentation. Goal: confirm whether the
# 3,318 ms residual is the DN portion of `layer.ffn_dispatch` +
# `layer.post_attn_fused_norm` (already-named buckets aggregated
# across DN+FA layers).
#
# Pre-flight (per worker prompt + feedback_bench_process_audit):
#  - vm_stat free × 16KB > 32 GB
#  - mcp-brain-server paused via `kill -STOP <pid>` BEFORE running
#  - llama drift gate ≤7,437 ms (10 % above W-5b.21's 6,761 ms)

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
TRIALS=${TRIALS:-3}

if [ ! -f "$MODEL" ]; then echo "ERROR: model not found: $MODEL"; exit 1; fi
if [ ! -f "$PROMPT" ]; then echo "ERROR: prompt not found: $PROMPT"; exit 1; fi

mkdir -p /tmp/w5b22-residual

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

echo "=== Wave 5b.22 residual audit ($TRIALS trials hf2q + $TRIALS llama at PP4106) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo

# (1) Same-day llama baseline (drift gate ≤10 % vs W-5b.21's 6,761 ms = 7,437 ms ceiling).
echo "--- llama baseline ($TRIALS cold trials at pp4106) ---"
for t in $(seq 1 "$TRIALS"); do
  /usr/bin/time -p /opt/homebrew/bin/llama-completion \
    --model "$MODEL" \
    --batch-size 4096 --ubatch-size 4096 --n-predict 1 --no-warmup -no-cnv --perf \
    -f "$PROMPT" > "/tmp/w5b22-residual/llama-T${t}.log" 2>&1
  prefill_ms=$(grep -oE 'prompt eval time = +[0-9.]+ ms' "/tmp/w5b22-residual/llama-T${t}.log" | sed -E 's/.*= +([0-9.]+) ms/\1/' | head -1)
  printf '[llama T%d] prompt_eval=%sms\n' "$t" "${prefill_ms:-???}"
done
echo

# (2) hf2q production path with full W-5b.8 + W-5b.17 + W-5b.22 instrumentation.
echo "--- hf2q chunk path with W-5b.22 residual instrumentation ($TRIALS cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] W-5b.22 residual" "/tmp/w5b22-residual/hf2q-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1 HF2Q_PROFILE_W5B22=1 \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Logs in /tmp/w5b22-residual/"
