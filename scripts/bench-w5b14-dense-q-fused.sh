#!/bin/bash
# Wave 5b.14 — dense_q wrapper-opt fused-CB bench at PP4106.
#
# Mirror of the MoE-Q `_into` analog landed in W-5b.13/14: external encoder +
# decode_pool::pooled_alloc_buffer scratches + fused single-CB residual_norm
# into the dense_q FFN forward.  Forensic A/B retained behind
# HF2Q_DENSE_Q_LEGACY=1 (pre-W-5b.14 device-alloc + own-encoder + 2-encoder).
#
# 3 cold trials × 2 paths (LEGACY + NEW) on AUTOREG (fused path is greedy
# decode + autoreg prefill — the chunk-prefill code path is not yet fused
# for dense_q in W-5b.14; that remains future work for W-5b.15+).

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
TRIALS=${TRIALS:-3}

if [ ! -f "$MODEL" ]; then echo "ERROR: model not found"; exit 1; fi
if [ ! -f "$PROMPT" ]; then echo "ERROR: prompt not found"; exit 1; fi

mkdir -p /tmp/w5b14

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
  printf '%-35s prefill=%6sms tok=%5s real=%5ss ffn_dispatch=%sms\n' \
    "$label" "${prefill_ms:-???}" "${token:-???}" "${real_s:-???}" "${ffn_ms:-???}"
}

echo "=== Wave 5b.14 dense_q wrapper-opt bench ($TRIALS trials × 2 paths × 1 llama) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo

# (1) Same-day llama baseline (drift gate ≤ 10 % vs W-5b.12's 6 765 ms).
echo "--- llama baseline (3 cold trials at pp4106) ---"
for t in $(seq 1 "$TRIALS"); do
  /usr/bin/time -p /opt/homebrew/bin/llama-completion \
    --model "$MODEL" \
    --batch-size 4096 --ubatch-size 4096 --n-predict 1 --no-warmup -no-cnv --perf \
    -f "$PROMPT" > "/tmp/w5b14/llama-T${t}.log" 2>&1
  prefill_ms=$(grep -oE 'prompt eval time = +[0-9.]+ ms' "/tmp/w5b14/llama-T${t}.log" | sed -E 's/.*= +([0-9.]+) ms/\1/' | head -1)
  printf '[llama T%d] prompt_eval=%sms\n' "$t" "${prefill_ms:-???}"
done
echo

# (2) hf2q LEGACY path (HF2Q_DENSE_Q_LEGACY=1: pre-W-5b.14 2-encoder).
echo "--- hf2q LEGACY path (HF2Q_DENSE_Q_LEGACY=1, 3 cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] LEGACY" "/tmp/w5b14/legacy-T${t}.log" \
    HF2Q_DENSE_Q_LEGACY=1 \
    HF2Q_PROFILE_W5B8=1 HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

# (3) hf2q NEW path (default: pooled scratches + fused single-CB).
echo "--- hf2q NEW path (default, 3 cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] NEW   " "/tmp/w5b14/new-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Logs in /tmp/w5b14/"
