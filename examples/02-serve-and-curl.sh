#!/usr/bin/env bash
# 02-serve-and-curl.sh
#
# Launch `hf2q serve` in the background, wait for /readyz, send one
# chat-completion via curl, then shut down.
#
# Demonstrates the OpenAI-compatible HTTP API surface on localhost.

set -euo pipefail

MODEL="${MODEL:-$HOME/.cache/hf2q/examples/gemma-4-26b-it-q4_k_m/out.gguf}"
PORT="${PORT:-18080}"
HF2Q="${HF2Q:-$(dirname "$0")/../target/release/hf2q}"

[[ -f "$MODEL" ]] || { echo "Model not found: $MODEL — run 01-convert-gemma4-q4km.sh first" >&2; exit 1; }
[[ -x "$HF2Q" ]]  || { echo "hf2q binary not at $HF2Q — run 'cargo build --release' first" >&2; exit 1; }

echo "Starting hf2q serve on :$PORT (model: $MODEL)"
"$HF2Q" serve --model "$MODEL" --host 127.0.0.1 --port "$PORT" &
PID=$!
trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; exit' EXIT INT TERM

# Poll /readyz until the model has finished loading.
for _ in {1..120}; do
    if curl -fsS "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; then break; fi
    sleep 1
done

echo "Sending chat completion request…"
curl -sS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "default",
      "messages": [{"role": "user", "content": "Write a haiku about sourdough bread."}],
      "max_tokens": 80,
      "temperature": 0
    }' | python3 -m json.tool
echo

echo "Shutting down…"
