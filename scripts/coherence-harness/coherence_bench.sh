#!/bin/bash
# coherence_bench.sh — Run hf2q AND llama.cpp on the same prompt, capture both
# outputs and tok/s, and report a side-by-side. Used to confirm whether a
# given symptom (garbage / repetition / abort) is isolated to hf2q or shared.
#
# 2026-05-03 — built per user directive. Pairs with render_parity.sh +
# logits_parity.sh as the third leg of the regression-isolation harness.
#
# Usage: ./coherence_bench.sh <gguf> "<user-prompt>" [-n=<max>] [--temp=<t>]
#                             [--seed=<s>] [thinking|no-thinking]
set -euo pipefail
MODEL="${1:?model gguf required}"
PROMPT="${2:?prompt required}"
shift 2
MAX=300
TEMP=0
SEED=42
MODE=thinking
for arg in "$@"; do
  case "$arg" in
    -n=*) MAX="${arg#*=}" ;;
    --temp=*) TEMP="${arg#*=}" ;;
    --seed=*) SEED="${arg#*=}" ;;
    thinking|no-thinking) MODE="$arg" ;;
  esac
done

HF2Q="${HF2Q:-/opt/hf2q/target/release/hf2q}"
LLAMA_COMPLETION="${LLAMA_COMPLETION:-/opt/homebrew/bin/llama-completion}"
EXTRACT_PY="${EXTRACT_PY:-/opt/hf2q/scripts/coherence-harness/extract_template.py}"

WORKDIR="$(mktemp -d)"
trap "rm -rf '$WORKDIR'" EXIT

# Render prompt via hf2q.
case "$MODE" in
  thinking) HF2Q_FLAG="--enable-thinking" ;;
  no-thinking) HF2Q_FLAG="--no-thinking" ;;
esac
HF2Q_DUMP_RENDERED_PROMPT="$WORKDIR/rendered.txt" \
  "$HF2Q" generate --model "$MODEL" --prompt "$PROMPT" --max-tokens 1 \
  $HF2Q_FLAG >/dev/null 2>&1 || true

# hf2q decode.
echo "=== hf2q (temp=$TEMP, max=$MAX, mode=$MODE) ==="
"$HF2Q" generate --model "$MODEL" --prompt "$PROMPT" \
  --max-tokens $MAX --temperature $TEMP $HF2Q_FLAG \
  2>"$WORKDIR/hf2q.stderr" | tee "$WORKDIR/hf2q.out" \
  | head -50

# llama.cpp decode on the EXACT same rendered prompt.
echo ""
echo "=== llama.cpp on hf2q-rendered prompt (temp=$TEMP, max=$MAX, mode=$MODE) ==="
"$LLAMA_COMPLETION" -m "$MODEL" -f "$WORKDIR/rendered.txt" -n $MAX \
  --temp $TEMP --seed $SEED --no-display-prompt --special \
  2>"$WORKDIR/llama.stderr" | tee "$WORKDIR/llama.out" \
  | head -50

# Pull tok/s from each.
hf_dec=$(grep -oE 'mlx-native \(qwen35\): [0-9]+ tokens in [0-9.]+s \([0-9.]+ tok/s\)' "$WORKDIR/hf2q.stderr" | head -1)
ll_dec=$(grep -oE 'eval time = [0-9.]+ ms /[ ]+[0-9]+ runs.*tokens per second' "$WORKDIR/llama.stderr" | head -1)
echo ""
echo "--- summary ---"
echo "hf2q     : $hf_dec"
echo "llama.cpp: $ll_dec"
