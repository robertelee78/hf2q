#!/usr/bin/env bash
# 05-qwen-vision-image.sh
#
# Chat with Qwen3-VL using an image input.  Requires the vision model
# GGUF + paired mmproj GGUF (vision projector).
#
# Demonstrates: vision input via the OpenAI image_url shape, the
# `qwen3vl` arch + `--mmproj` flag plumbing.

set -euo pipefail

MODEL="${MODEL:-$HOME/.cache/hf2q/examples/qwen3vl/text.gguf}"
MMPROJ="${MMPROJ:-$HOME/.cache/hf2q/examples/qwen3vl/mmproj.gguf}"
IMAGE_PATH="${IMAGE_PATH:-$(dirname "$0")/sample-image.png}"
PORT="${PORT:-18082}"
HF2Q="${HF2Q:-$(dirname "$0")/../target/release/hf2q}"

[[ -f "$MODEL" ]]  || { echo "Model GGUF not found: $MODEL"; exit 1; }
[[ -f "$MMPROJ" ]] || { echo "mmproj GGUF not found: $MMPROJ"; exit 1; }
[[ -x "$HF2Q" ]]   || { echo "hf2q binary not at $HF2Q — run 'cargo build --release' first" >&2; exit 1; }

# Make a 256×256 solid-red PNG fixture if the operator didn't supply one.
if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "Generating sample image: $IMAGE_PATH"
    python3 -c "
from PIL import Image
Image.new('RGB', (256, 256), (220, 30, 30)).save('$IMAGE_PATH')
" 2>/dev/null || { echo "Need either PIL (pip install pillow) or a pre-existing image at $IMAGE_PATH"; exit 1; }
fi

# Base64-encode the image so it goes inline in the JSON request.
IMG_DATA=$(python3 -c "
import base64, mimetypes, sys
p = '$IMAGE_PATH'
mime = mimetypes.guess_type(p)[0] or 'image/png'
data = base64.b64encode(open(p, 'rb').read()).decode()
print(f'data:{mime};base64,{data}')
")

echo "Starting hf2q serve on :$PORT (vision)"
"$HF2Q" serve --model "$MODEL" --mmproj "$MMPROJ" --host 127.0.0.1 --port "$PORT" &
PID=$!
trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; exit' EXIT INT TERM

for _ in {1..180}; do
    curl -fsS "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1 && break
    sleep 1
done

echo "Asking the model what's in the image…"
curl -sS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c "
import json, sys
print(json.dumps({
  'model': 'default',
  'messages': [{
    'role': 'user',
    'content': [
      {'type': 'text', 'text': 'What color is the dominant region of this image?'},
      {'type': 'image_url', 'image_url': {'url': '$IMG_DATA'}}
    ]
  }],
  'max_tokens': 80,
  'temperature': 0
}))
")" | python3 -m json.tool
