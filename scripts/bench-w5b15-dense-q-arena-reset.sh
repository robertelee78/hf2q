#!/bin/bash
# Wave 5b.15 — dense_q per-layer prefill arena-reset bench at PP4106.
#
# Closes the W-5b.14 architectural-limit STOP-AND-REPORT: with the new
# `decode_pool::reset_for_prefill_chunk()` call site at every prefill
# layer iteration in `forward_gpu_impl`, dense_q's `_into_pooled` variant
# can route its internal scratches through the pool unconditionally
# (FINAL output stays device-alloc'd at prefill so it survives the reset
# and becomes the next layer's `hidden`).  Forensic A/B retained behind
# `HF2Q_DENSE_Q_ARENA_RESET=0` (W-5b.14 device-alloc-prefill behavior).
#
# Same protocol as W-5b.14's bench: 3 cold trials × 2 paths (LEGACY +
# NEW) × 1 same-day llama at PP4106 (qwen3.6-27b-dwq46).

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
TRIALS=${TRIALS:-3}

if [ ! -f "$MODEL" ]; then echo "ERROR: model not found"; exit 1; fi
if [ ! -f "$PROMPT" ]; then echo "ERROR: prompt not found"; exit 1; fi

mkdir -p /tmp/w5b15

run_one() {
  local label="$1"
  local outfile="$2"
  shift 2
  /usr/bin/time -p env "$@" \
    /opt/hf2q/target/release/hf2q -v generate --model "$MODEL" \
    --prompt-file "$PROMPT" \
    --chat-template '{{ messages[0]["content"] }}' \
    --max-tokens 2 --temperature 0.0 > "$outfile" 2>&1
  local prefill_ms token real_s ffn_ms
  prefill_ms=$(grep -oE 'prefill: [0-9]+ tok in [0-9]+ms' "$outfile" | sed -E 's/.* in ([0-9]+)ms/\1/' | head -1)
  token=$(grep -oE 'first decoded token: [0-9]+' "$outfile" | sed -E 's/.*: ([0-9]+)/\1/' | head -1)
  real_s=$(awk '/^real/ {print $2}' "$outfile")
  ffn_ms=$(grep -oE 'layer.ffn_dispatch[^|]*\| *[0-9.]+' "$outfile" | sed -E 's/.*\| *([0-9.]+).*/\1/' | head -1)
  printf '%-40s prefill=%6sms tok=%5s real=%5ss ffn_dispatch=%sms\n' \
    "$label" "${prefill_ms:-???}" "${token:-???}" "${real_s:-???}" "${ffn_ms:-???}"
}

echo "=== Wave 5b.15 dense_q arena-reset bench ($TRIALS trials × 2 paths × 1 llama) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo

# (1) Same-day llama baseline (drift gate ≤ 10 % vs W-5b.14's 6 927 ms).
echo "--- llama baseline (3 cold trials at pp4106) ---"
for t in $(seq 1 "$TRIALS"); do
  /usr/bin/time -p /opt/homebrew/bin/llama-completion \
    --model "$MODEL" \
    --batch-size 4096 --ubatch-size 4096 --n-predict 1 --no-warmup -no-cnv --perf \
    -f "$PROMPT" > "/tmp/w5b15/llama-T${t}.log" 2>&1
  prefill_ms=$(grep -oE 'prompt eval time = +[0-9.]+ ms' "/tmp/w5b15/llama-T${t}.log" | sed -E 's/.*= +([0-9.]+) ms/\1/' | head -1)
  printf '[llama T%d] prompt_eval=%sms\n' "$t" "${prefill_ms:-???}"
done
echo

# (2) hf2q LEGACY path (HF2Q_DENSE_Q_ARENA_RESET=0: W-5b.14 device-alloc prefill).
echo "--- hf2q LEGACY path (HF2Q_DENSE_Q_ARENA_RESET=0, 3 cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] LEGACY (W-5b.14)" "/tmp/w5b15/legacy-T${t}.log" \
    HF2Q_DENSE_Q_ARENA_RESET=0 \
    HF2Q_PROFILE_W5B8=1 HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

# (3) hf2q NEW path (default: pooled prefill scratches + per-layer reset).
echo "--- hf2q NEW path (default = arena reset ON, 3 cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] NEW (W-5b.15 reset)" "/tmp/w5b15/new-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Logs in /tmp/w5b15/"
