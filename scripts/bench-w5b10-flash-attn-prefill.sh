#!/bin/bash
# Wave 5b.10 — flash_attn_prefill wire-up perf measurement.
# Compares the new flash_attn_prefill_bf16_d256 path (default) vs the
# legacy mlx-native sdpa path (HF2Q_QWEN35_FA_LEGACY=1) at PP4096.
#
# Reads /tmp/walkbar-pp4096-prompt.txt (SHA 62e66013996f725c794d53fa9136f43c1b9eca0e),
# the W-5b.4/8/9 walkbar prompt, against qwen3.6-27b-dwq46.gguf.
#
# Each trial is a fresh process — cold first-forward.

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
TRIALS=${TRIALS:-3}

if [ ! -f "$MODEL" ]; then
  echo "ERROR: model not found at $MODEL"; exit 1
fi
if [ ! -f "$PROMPT" ]; then
  echo "ERROR: prompt not found at $PROMPT"; exit 1
fi

run_one() {
  local label="$1"
  local outfile="$2"
  shift 2
  /usr/bin/time -p env "$@" \
    ./target/release/hf2q -v generate --model "$MODEL" \
    --prompt-file "$PROMPT" \
    --chat-template '{{ messages[0]["content"] }}' \
    --max-tokens 2 --temperature 0.0 > "$outfile" 2>&1
  local prefill_ms decode_tps token real_s loaded_s
  prefill_ms=$(grep -oE 'prefill: [0-9]+ tok in [0-9]+ms' "$outfile" | sed -E 's/.* in ([0-9]+)ms/\1/' | head -1)
  decode_tps=$(grep -oE '[0-9.]+ tok/s' "$outfile" | sed -E 's/ tok\/s//' | tail -1)
  token=$(grep -oE 'first decoded token: [0-9]+' "$outfile" | sed -E 's/.*: ([0-9]+)/\1/' | head -1)
  real_s=$(awk '/^real/ {print $2}' "$outfile")
  loaded_s=$(grep -oE 'loaded in [0-9.]+s' "$outfile" | sed -E 's/loaded in ([0-9.]+)s/\1/' | head -1)
  printf '%-50s prefill=%6sms decode=%6s tok/s tok=%5s loaded=%5ss real=%5ss\n' \
    "$label" "${prefill_ms:-???}" "${decode_tps:-???}" "${token:-???}" "${loaded_s:-???}" "${real_s:-???}"
}

echo "=== Wave 5b.10 flash_attn_prefill perf bench ($TRIALS trials per condition) ==="
echo "HEAD: $(git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | head -1; vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo

mkdir -p /tmp/w5b10
for t in $(seq 1 "$TRIALS"); do
  echo "--- Trial $t ---"
  run_one "[T${t}] CHUNK new (flash_attn_prefill, default) " "/tmp/w5b10/chunk-new-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
  run_one "[T${t}] CHUNK legacy (sdpa, HF2Q_QWEN35_FA_LEGACY=1)" "/tmp/w5b10/chunk-legacy-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1 \
    HF2Q_QWEN35_FA_LEGACY=1
done

echo
echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo
echo "Logs in /tmp/w5b10/"
