#!/usr/bin/env bash
# 04-embeddings-bert.sh
#
# Convert a BERT-family encoder (text-embedding-only arch) to GGUF, serve
# it, and call /v1/embeddings.  hf2q supports BERT + Nomic-BERT today.

set -euo pipefail

REPO="${REPO:-sentence-transformers/all-MiniLM-L6-v2}"
QUANT="${QUANT:-q8_0}"
OUT_DIR="${OUT_DIR:-$HOME/.cache/hf2q/examples/$(basename "$REPO")-$QUANT}"
PORT="${PORT:-18081}"
HF2Q="${HF2Q:-$(dirname "$0")/../target/release/hf2q}"

[[ -x "$HF2Q" ]] || { echo "hf2q binary not at $HF2Q — run 'cargo build --release' first" >&2; exit 1; }

mkdir -p "$OUT_DIR"

if [[ ! -f "$OUT_DIR/out.gguf" ]]; then
    echo "Converting $REPO → $QUANT GGUF (one-time, ~2–5 min)"
    "$HF2Q" convert --repo "$REPO" --format gguf --quant "$QUANT" \
        --output "$OUT_DIR/out.gguf"
fi

echo "Starting hf2q serve on :$PORT"
"$HF2Q" serve --model "$OUT_DIR/out.gguf" --host 127.0.0.1 --port "$PORT" &
PID=$!
trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; exit' EXIT INT TERM

for _ in {1..60}; do
    curl -fsS "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1 && break
    sleep 1
done

echo "Computing embeddings for 3 short strings…"
curl -sS -X POST "http://127.0.0.1:$PORT/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "default",
      "input": [
        "Sourdough fermentation is slower than baker'\''s yeast.",
        "TurboQuant compresses KV cache by 2× without quality loss.",
        "Apple Silicon unified memory is great for ML inference."
      ]
    }' | python3 -c '
import json, sys
data = json.load(sys.stdin)["data"]
for i, item in enumerate(data):
    vec = item["embedding"]
    n = len(vec)
    norm = sum(v*v for v in vec) ** 0.5
    print(f"  embedding[{i}]: dim={n}, L2-norm={norm:.4f}, first 5 = {vec[:5]}")
'
