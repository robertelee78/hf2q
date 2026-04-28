#!/bin/bash
# Wave 5b.24 — pooled `quantized_matmul_id_ggml_pooled` wire-up bench.
# Lands the W-5b.23 audit's #1 recommendation: switch
# `gpu_ffn.rs:1492-1560`'s 3 `quantized_matmul_id_ggml` calls to the
# pooled variant (eliminates 286 of 288 per-prefill device allocs by
# caching `IdMmScratch` thread-locally across all 48 layers).
#
# Compares hf2q NEW path (default, pooled) vs hf2q LEGACY path
# (HF2Q_FFN_POOLED_LEGACY=1, un-pooled) at PP4106 with W-5b.8/W-5b.17/
# W-5b.22 instrumentation enabled, plus a 3-trial llama drift gate.
#
# Pre-flight (per worker prompt):
#  - vm_stat free × 16KB > 32 GB
#  - mcp-brain-server paused via `kill -STOP <pid>`
#
# Reuses the W-5b.18/19/20 walk-bar prompt + measurement protocol so
# wall-clock numbers are directly comparable across iters.

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
TRIALS=${TRIALS:-3}

if [ ! -f "$MODEL" ]; then echo "ERROR: model not found: $MODEL"; exit 1; fi
if [ ! -f "$PROMPT" ]; then echo "ERROR: prompt not found: $PROMPT"; exit 1; fi

mkdir -p /tmp/w5b24

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

echo "=== Wave 5b.24 pooled mul_mm_id wire-up ($TRIALS trials × 2 hf2q paths × $TRIALS llama) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo

# (1) Same-day llama baseline (drift gate <= 10% vs W-5b.22's 6,734 ms = 7,407 ceiling).
echo "--- llama baseline ($TRIALS cold trials at pp4106) ---"
for t in $(seq 1 "$TRIALS"); do
  /usr/bin/time -p /opt/homebrew/bin/llama-completion \
    --model "$MODEL" \
    --batch-size 4096 --ubatch-size 4096 --n-predict 1 --no-warmup -no-cnv --perf \
    -f "$PROMPT" > "/tmp/w5b24/llama-T${t}.log" 2>&1
  prefill_ms=$(grep -oE 'prompt eval time = +[0-9.]+ ms' "/tmp/w5b24/llama-T${t}.log" | sed -E 's/.*= +([0-9.]+) ms/\1/' | head -1)
  printf '[llama T%d] prompt_eval=%sms\n' "$t" "${prefill_ms:-???}"
done
echo

# (2) hf2q NEW path — default, pooled IdMmScratch reused across all 48 MoE layers.
echo "--- hf2q chunk path NEW (pooled mul_mm_id, $TRIALS cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] W-5b.24 NEW (pooled)" "/tmp/w5b24/new-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1 HF2Q_PROFILE_W5B22=1 \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

# (3) hf2q LEGACY path — HF2Q_FFN_POOLED_LEGACY=1 fires the un-pooled variant
#     (288 device allocs per prefill, 6 per FFN layer × 48 layers).
echo "--- hf2q chunk path LEGACY (un-pooled mul_mm_id, $TRIALS cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] W-5b.24 LEGACY (un-pooled)" "/tmp/w5b24/legacy-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1 HF2Q_PROFILE_W5B22=1 \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1 \
    HF2Q_FFN_POOLED_LEGACY=1
done
echo

echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Logs in /tmp/w5b24/"
