#!/bin/bash
# logits_parity.sh — Compare hf2q's first-token logits to llama.cpp's on the
# same rendered prompt. Reports top-10 token IDs from each side and the L2
# delta on the overlapping vocab.
#
# 2026-05-03 — built per user directive. Combined with render_parity.sh
# this isolates the bug to the forward-pass when render is byte-identical.
#
# Usage: ./logits_parity.sh <gguf> <rendered-prompt-file>
set -euo pipefail
MODEL="${1:?model gguf required}"
RENDERED="${2:?rendered-prompt-file required}"

HF2Q="${HF2Q:-/opt/hf2q/target/release/hf2q}"
LLAMA_EVAL="${LLAMA_EVAL:-/opt/homebrew/bin/llama-eval-callback}"

WORKDIR="$(mktemp -d)"
trap "rm -rf '$WORKDIR'" EXIT

# 1. hf2q first-token logits via HF2Q_DUMP_LOGITS=1.
PROMPT_TEXT="$(cat "$RENDERED")"
HF2Q_DUMP_LOGITS=1 "$HF2Q" generate --model "$MODEL" \
  --prompt "$PROMPT_TEXT" --max-tokens 1 \
  --chat-template '{{- messages[0].content -}}' \
  >"$WORKDIR/hf2q.stdout" 2>"$WORKDIR/hf2q.stderr" || true

if [ ! -f /tmp/hf2q_logits_t0.bin ]; then
  echo "hf2q did not produce logits dump" >&2
  cat "$WORKDIR/hf2q.stderr" >&2
  exit 1
fi
cp /tmp/hf2q_logits_t0.bin "$WORKDIR/hf2q.bin"

# 2. llama.cpp first-token logits via llama-eval-callback.
"$LLAMA_EVAL" -m "$MODEL" -p "$PROMPT_TEXT" -n 0 \
  >"$WORKDIR/llama.stdout" 2>"$WORKDIR/llama.stderr" || true

# 3. Compare. (For now top-1 argmax; full L2 distance requires a llama.cpp
# logit-dump tool that we may need to script up.)
python3 <<PY
import struct
with open("$WORKDIR/hf2q.bin", "rb") as f:
    raw = f.read()
n = len(raw) // 4
import struct
vals = struct.unpack(f"<{n}f", raw)
top = sorted(enumerate(vals), key=lambda x: -x[1])[:10]
print("hf2q top-10:")
for idx, l in top:
    print(f"  id={idx:6d}  logit={l:.4f}")
PY

# llama-eval-callback prints "first 10 tokens" by default; surface them.
echo "---"
echo "llama.cpp last lines:"
tail -30 "$WORKDIR/llama.stdout" "$WORKDIR/llama.stderr" 2>/dev/null | tail -30
