#!/bin/bash
# Wave 5b.27 Phase A — retrospective validation of W-5b.24's -86ms claim.
#
# Runs 6 cold trials at PP4106 on the current main HEAD (post-W-5b.26
# reverts) and captures `dn.outer_ffn_dispatch` median + whole-prefill
# wall + 95% CI. Compares to W-5b.21 baseline (3,316ms) per the
# retrospective spec at:
#   /Users/robert/.claude/projects/-opt-hf2q/memory/project_w5b26_n_expert_zero_target_inert.md
#
# Decision matrix:
#  - median <= 3,266ms AND 95% CI doesn't overlap 3,316ms: W-5b.24 VALIDATED
#  - median in 3,266-3,366ms (W-5b.21 noise band): W-5b.24 FALSIFIED, ratio 2.343x
#  - median > 3,366ms: regression → STOP
#
# Same-day llama 3 cold trials (within 10% of W-5b.24's 6,756ms = 7,431ms ceiling).

set -u

MODEL=/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf
PROMPT=/tmp/walkbar-pp4096-prompt.txt
TRIALS=${TRIALS:-6}
LLAMA_TRIALS=${LLAMA_TRIALS:-3}

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

echo "=== Wave 5b.27 Phase A retro-validation ($TRIALS cold trials hf2q + $LLAMA_TRIALS llama) ==="
echo "HEAD: $(cd /opt/hf2q && git rev-parse --short HEAD) | mlx-native: $(cd /opt/mlx-native && git rev-parse --short HEAD)"
echo "Pre-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Prompt SHA: $(shasum "$PROMPT" | awk '{print $1}')"
echo

# (1) Same-day llama baseline (drift gate <= 10% vs W-5b.24's 6,756 ms = 7,431 ceiling).
echo "--- llama baseline ($LLAMA_TRIALS cold trials at pp4106) ---"
for t in $(seq 1 "$LLAMA_TRIALS"); do
  /usr/bin/time -p /opt/homebrew/bin/llama-completion \
    --model "$MODEL" \
    --batch-size 4096 --ubatch-size 4096 --n-predict 1 --no-warmup -no-cnv --perf \
    -f "$PROMPT" > "/tmp/w5b27/llama-T${t}.log" 2>&1
  prefill_ms=$(grep -oE 'prompt eval time = +[0-9.]+ ms' "/tmp/w5b27/llama-T${t}.log" | sed -E 's/.*= +([0-9.]+) ms/\1/' | head -1)
  printf '[llama T%d] prompt_eval=%sms\n' "$t" "${prefill_ms:-???}"
done
echo

# (2) hf2q current main HEAD — measures post-revert dn.outer_ffn_dispatch.
echo "--- hf2q chunk path current HEAD ($TRIALS cold trials) ---"
for t in $(seq 1 "$TRIALS"); do
  run_one "[T${t}] W-5b.27 phaseA HEAD" "/tmp/w5b27/phaseA-T${t}.log" \
    HF2Q_PROFILE_W5B8=1 HF2Q_PROFILE_W5B17=1 HF2Q_PROFILE_W5B22=1 \
    HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1
done
echo

echo "Post-bench: $(vm_stat | awk '/Pages free/ {printf "free=%s pages\n", $3}')"
echo "Logs in /tmp/w5b27/"
