#!/usr/bin/env bash
# BUG-gemma-cli divergence diagnostic — 2026-05-16
#
# Runs llama-cli and hf2q on the SAME prompt (read from $1), with the SAME
# chat-template wrapping (Gemma GGUF's tokenizer.chat_template), at greedy
# temp=0. Goal: locate the first token where the two runtimes diverge.
#
# Usage:
#   ./scripts/bug-gemma-cli-diverge-2026-05-16.sh <prompt-file>
#
# Outputs preserved at /tmp/bug-gemma-<epoch>/ — never auto-deleted.
# Prompt content is NEVER printed by this script; only its byte+sha
# fingerprint, so you can verify both runtimes saw identical bytes.
set -euo pipefail

PROMPT_FILE="${1:-}"
if [[ -z "$PROMPT_FILE" || ! -f "$PROMPT_FILE" ]]; then
  echo "usage: $0 <prompt-file>" >&2
  exit 64
fi

MODEL="/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf"
NPRED="${NPRED:-500}"
OUTDIR="/tmp/bug-gemma-$(date +%s)"
mkdir -p "$OUTDIR"

bytes=$(wc -c <"$PROMPT_FILE" | tr -d ' ')
sha=$(shasum -a 256 "$PROMPT_FILE" | awk '{print $1}')
echo "=== prompt: $bytes bytes, sha256=$sha ==="
echo "=== outputs: $OUTDIR (preserved) ==="
echo

LLAMA_OUT="$OUTDIR/llama.stdout"
LLAMA_ERR="$OUTDIR/llama.stderr"
HF2Q_OUT="$OUTDIR/hf2q.stdout"
HF2Q_ERR="$OUTDIR/hf2q.stderr"

echo "--- running llama-cli (--jinja, --temp 0, --n-predict $NPRED) ---"
/opt/llama.cpp/build/bin/llama-cli \
  --model "$MODEL" \
  --jinja \
  --file "$PROMPT_FILE" \
  --temp 0 \
  --n-predict "$NPRED" \
  --single-turn \
  >"$LLAMA_OUT" 2>"$LLAMA_ERR" || echo "(llama-cli exit $?)"
echo "  stdout: $(wc -l <"$LLAMA_OUT") lines, $(wc -c <"$LLAMA_OUT") bytes"
echo "  stderr: $(wc -l <"$LLAMA_ERR") lines, $(wc -c <"$LLAMA_ERR") bytes"
echo

echo "--- running hf2q generate (--temperature 0.0, --max-tokens $NPRED) ---"
( cd /opt/hf2q && ./target/release/hf2q generate \
    --model "$MODEL" \
    --prompt-file "$PROMPT_FILE" \
    --temperature 0.0 \
    --max-tokens "$NPRED" \
    >"$HF2Q_OUT" 2>"$HF2Q_ERR" ) || echo "(hf2q exit $?)"
echo "  stdout: $(wc -l <"$HF2Q_OUT") lines, $(wc -c <"$HF2Q_OUT") bytes"
echo "  stderr: $(wc -l <"$HF2Q_ERR") lines, $(wc -c <"$HF2Q_ERR") bytes"
echo

echo "=========================================================="
echo "  RAW LLAMA-CLI OUTPUT (stdout — last 50 lines)"
echo "=========================================================="
tail -n 50 "$LLAMA_OUT"
echo
echo "=========================================================="
echo "  RAW HF2Q OUTPUT (stdout — last 50 lines)"
echo "=========================================================="
tail -n 50 "$HF2Q_OUT"
echo
echo "=========================================================="
echo "  Files preserved for inspection:"
echo "    $LLAMA_OUT"
echo "    $LLAMA_ERR"
echo "    $HF2Q_OUT"
echo "    $HF2Q_ERR"
echo "=========================================================="
