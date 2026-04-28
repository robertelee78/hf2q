#!/bin/bash
# Wave 5b.26 — lift FFN pooled-output buffers to prefill scope + add
# 11 DnFfn* sub-bucket instrumentation.
#
# 2026-04-28 drift note: this script is historical. Current qwen35 source no
# longer contains `HF2Q_PROFILE_W5B26` or `HF2Q_FFN_OUTPUT_LIFT_LEGACY`, so the
# env vars below do not prove a live NEW-vs-LEGACY A/B. Revalidate source before
# using these results for architectural decisions.
#
# Lands the W-5b.25 audit's #1 candidate: cache the 9 per-FFN-call
# `pooled_alloc_buffer` calls (`ids` / `weights` / `gate_all` / `up_all`
# / `h_all` / `y_all` / `h_s` / `out` / `silu_params`) in a thread-local
# `FfnOutputCache` so 432 per-prefill device alloc lookups become 9
# first-call allocs + 423 cache hits.  Phase B bundles the W-5b.23
# audit's #2 candidate (sub-bucket instrumentation) at near-zero
# marginal cost via `HF2Q_PROFILE_W5B26=1`.
#
# Compares hf2q NEW path (default, lifted) vs hf2q LEGACY path
# (HF2Q_FFN_OUTPUT_LIFT_LEGACY=1, per-layer pool-alloc) at PP4106 with
# W-5b.8/W-5b.17/W-5b.22/W-5b.26 instrumentation enabled, plus a
# 3-trial llama drift gate.
#
# Pre-flight (per worker prompt):
#  - vm_stat free × 16 KB > 32 GB
#  - mcp-brain-server / WebKit content paused via `kill -STOP <pid>`
#
# Reuses the W-5b.18/19/20/24 walk-bar prompt + measurement protocol so
# wall-clock numbers are directly comparable across iters.

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
TRIALS=${TRIALS:-3}

if [ ! -f "$MODEL" ]; then echo "ERROR: model not found: $MODEL"; exit 1; fi
if [ ! -f "$PROMPT" ]; then echo "ERROR: prompt not found: $PROMPT"; exit 1; fi

mkdir -p /tmp/w5b26

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

echo "=== Wave 5b.26 FFN-output lift + sub-bucket instrumentation ($TRIALS trials × 2 hf2q paths × $TRIALS llama) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo

# (1) Same-day llama baseline (drift gate <= 10% vs W-5b.24's 6,756 ms = 7,431 ceiling).
echo "--- llama baseline ($TRIALS cold trials at pp4106) ---"
for t in $(seq 1 "$TRIALS"); do
  /usr/bin/time -p /opt/homebrew/bin/llama-completion \
    --model "$MODEL" \
    --batch-size 4096 --ubatch-size 4096 --n-predict 1 --no-warmup -no-cnv --perf \
    -f "$PROMPT" > "/tmp/w5b26/llama-T${t}.log" 2>&1
  prefill_ms=$(grep -oE 'prompt eval time = +[0-9.]+ ms' "/tmp/w5b26/llama-T${t}.log" | sed -E 's/.*= +([0-9.]+) ms/\1/' | head -1)
  printf '[llama T%d] prompt_eval=%sms\n' "$t" "${prefill_ms:-???}"
done
echo

# (2) hf2q NEW path — default, lifted FfnOutputCache reused across all 48 MoE layers.
echo "--- hf2q chunk path NEW (lifted FfnOutputCache, $TRIALS cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] W-5b.26 NEW (lifted)" "/tmp/w5b26/new-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1 HF2Q_PROFILE_W5B22=1 HF2Q_PROFILE_W5B26=1 \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

# (3) hf2q LEGACY path — HF2Q_FFN_OUTPUT_LIFT_LEGACY=1 fires the per-layer
#     pool-alloc variant (9 pool allocs per FFN layer × 48 layers = 432
#     per-prefill pool ops vs 9 first-call cache misses + 423 cache hits in NEW).
echo "--- hf2q chunk path LEGACY (per-layer pool alloc, $TRIALS cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] W-5b.26 LEGACY (per-layer pool)" "/tmp/w5b26/legacy-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1 HF2Q_PROFILE_W5B22=1 HF2Q_PROFILE_W5B26=1 \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1 \
    HF2Q_FFN_OUTPUT_LIFT_LEGACY=1
done
echo

echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Logs in /tmp/w5b26/"
