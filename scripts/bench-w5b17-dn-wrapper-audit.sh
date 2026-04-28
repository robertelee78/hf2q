#!/bin/bash
# Wave 5b.17 — DN wrapper-overhead audit at PP4106.
#
# Reuses the W-5b.15/16 bench harness (3 cold trials × 1 hf2q path × 3
# llama trials at PP4106) with `HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1`
# enabled so the existing per-DN-layer buckets and the four NEW W-5b.17
# sub-buckets (`dn.qkv_download`, `dn.qkv_cpu_loop`, `dn.qkv_uploads`,
# `dn.state_pingpong_memcpy`) emit together.
#
# Prerequisites: hf2q built --release with W-5b.17 instrumentation; mlx-
# native HEAD pinned at 6875c925 (no upstream drift across this iter);
# mcp-brain-server paused via `kill -STOP <pid>` per
# `feedback_bench_process_audit`.

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
TRIALS=${TRIALS:-3}

if [ ! -f "$MODEL" ]; then echo "ERROR: model not found"; exit 1; fi
if [ ! -f "$PROMPT" ]; then echo "ERROR: prompt not found"; exit 1; fi

mkdir -p /tmp/w5b17

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

echo "=== Wave 5b.17 DN wrapper-overhead audit ($TRIALS trials × 1 hf2q × 3 llama) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo

# (1) Same-day llama baseline (drift gate ≤ 10 % vs W-5b.16's 6 650 ms).
echo "--- llama baseline (3 cold trials at pp4106) ---"
for t in $(seq 1 "$TRIALS"); do
  /usr/bin/time -p /opt/homebrew/bin/llama-completion \
    --model "$MODEL" \
    --batch-size 4096 --ubatch-size 4096 --n-predict 1 --no-warmup -no-cnv --perf \
    -f "$PROMPT" > "/tmp/w5b17/llama-T${t}.log" 2>&1
  prefill_ms=$(grep -oE 'prompt eval time = +[0-9.]+ ms' "/tmp/w5b17/llama-T${t}.log" | sed -E 's/.*= +([0-9.]+) ms/\1/' | head -1)
  printf '[llama T%d] prompt_eval=%sms\n' "$t" "${prefill_ms:-???}"
done
echo

# (2) hf2q chunk path with W-5b.8 + W-5b.17 instrumentation.
echo "--- hf2q chunk path (W-5b.8 + W-5b.17 instrumentation, 3 cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] W-5b.17 chunk" "/tmp/w5b17/chunk-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1 \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

# (3) hf2q autoregressive path (no HF2Q_CHUNK_SCAN_PREFILL) for comparison
#     — surfaces whether DN wrapper overhead is chunk-specific or shared.
echo "--- hf2q autoreg path (W-5b.8 + W-5b.17 instrumentation, 3 cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] W-5b.17 autoreg" "/tmp/w5b17/autoreg-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1 \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1
done
echo

echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Logs in /tmp/w5b17/"
