#!/bin/bash
# Wave 5b.11 — post-attention per-section bench at PP4096.
#
# Measures the unprofiled ~203 ms/layer in `layer.linear_total` revealed by
# W-5b.8 (chunk-path wrapper-internal sub-buckets sum to ~199 ms/layer
# matching `layer.chunk_call` exactly, leaving the rest to live in the
# post-attention path: fused residual+norm encoder + FFN MoE dispatch +
# post-FFN residual).
#
# 3 cold trials × 2 paths (chunk + autoreg) — autoreg measured for
# cross-check that post-attn cost is path-independent (it should be: same
# fused_residual_norm + MoeQ FFN code).

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
TRIALS=${TRIALS:-3}

if [ ! -f "$MODEL" ]; then echo "ERROR: model not found"; exit 1; fi
if [ ! -f "$PROMPT" ]; then echo "ERROR: prompt not found"; exit 1; fi

mkdir -p /tmp/w5b11

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
  printf '%-35s prefill=%6sms tok=%5s real=%5ss\n' \
    "$label" "${prefill_ms:-???}" "${token:-???}" "${real_s:-???}"
}

echo "=== Wave 5b.11 post-attention per-section bench ($TRIALS trials × 2 paths × 1 llama) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo

# (1) Same-day llama baseline (drift gate ≤ 10 % vs W-5b.10's 6 765 ms).
echo "--- llama baseline (3 cold trials at pp4096) ---"
for t in $(seq 1 "$TRIALS"); do
  /usr/bin/time -p /opt/homebrew/bin/llama-completion \
    --model "$MODEL" \
    --batch-size 4096 --ubatch-size 4096 --n-predict 1 --no-warmup -no-cnv --perf \
    -f "$PROMPT" > "/tmp/w5b11/llama-T${t}.log" 2>&1
  prefill_ms=$(grep -oE 'prompt eval time = +[0-9.]+ ms' "/tmp/w5b11/llama-T${t}.log" | sed -E 's/.*= +([0-9.]+) ms/\1/' | head -1)
  printf '[llama T%d] prompt_eval=%sms\n' "$t" "${prefill_ms:-???}"
done
echo

# (2) hf2q chunk path (3 cold trials).
echo "--- hf2q CHUNK path (3 cold trials, HF2Q_PROFILE_W5B8=1) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] CHUNK new" "/tmp/w5b11/chunk-new-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

# (3) hf2q autoreg path (3 cold trials).
echo "--- hf2q AUTOREG path (3 cold trials, HF2Q_PROFILE_W5B8=1) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] AUTOREG" "/tmp/w5b11/autoreg-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1
done
echo

echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Logs in /tmp/w5b11/"
