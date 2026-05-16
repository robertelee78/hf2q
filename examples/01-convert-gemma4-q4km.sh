#!/usr/bin/env bash
# 01-convert-gemma4-q4km.sh
#
# Convert a HuggingFace Gemma-4 26B model to a Q4_K_M GGUF.
#
# Estimated time:  10–20 min on M5 Max (depends on HF download throughput).
# Estimated disk:  ~60 GB during conversion, ~16 GB final GGUF.
# Output:          $OUT_DIR/out.gguf  + sidecar tokenizer/config files.

set -euo pipefail

REPO="${REPO:-google/gemma-4-26b-it}"
QUANT="${QUANT:-q4_k_m}"
OUT_DIR="${OUT_DIR:-$HOME/.cache/hf2q/examples/gemma-4-26b-it-$QUANT}"
HF2Q="${HF2Q:-$(dirname "$0")/../target/release/hf2q}"

[[ -x "$HF2Q" ]] || { echo "hf2q binary not at $HF2Q — run 'cargo build --release' first" >&2; exit 1; }

mkdir -p "$OUT_DIR"

echo "Converting $REPO → $QUANT GGUF"
echo "  out → $OUT_DIR/out.gguf"
echo

"$HF2Q" convert \
  --repo "$REPO" \
  --format gguf \
  --quant "$QUANT" \
  --output "$OUT_DIR/out.gguf"

echo
echo "Done. Try it:"
echo "  $HF2Q generate --model $OUT_DIR/out.gguf --prompt 'Hello, world!'"
echo "  $HF2Q serve --model $OUT_DIR/out.gguf --port 8080"
