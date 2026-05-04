#!/bin/bash
# render_parity.sh — Byte-compare hf2q's chat-template render vs llama.cpp's.
#
# 2026-05-03 — built per user directive after hf2q started emitting garbage
# on the wedding-cake prompt. Output is the same model + same prompt rendered
# through both stacks; if SHA256s differ, the chat-template path is the
# regression site. If SHA256s match, the bug is downstream of render.
#
# Usage: ./render_parity.sh <gguf> "<user-prompt>" [thinking|no-thinking]
set -euo pipefail
MODEL="${1:?model gguf required}"
PROMPT="${2:?prompt required}"
MODE="${3:-thinking}"

HF2Q="${HF2Q:-/opt/hf2q/target/release/hf2q}"
LLAMA_CLI="${LLAMA_CLI:-/opt/homebrew/bin/llama-cli}"
EXTRACT_PY="${EXTRACT_PY:-/opt/hf2q/scripts/coherence-harness/extract_template.py}"

WORKDIR="$(mktemp -d)"
trap "rm -rf '$WORKDIR'" EXIT

case "$MODE" in
  thinking) HF2Q_FLAG="--enable-thinking" ;;
  no-thinking) HF2Q_FLAG="--no-thinking" ;;
  *) echo "mode must be thinking|no-thinking" >&2; exit 2 ;;
esac

# 1. hf2q render via HF2Q_DUMP_RENDERED_PROMPT.
HF2Q_DUMP_RENDERED_PROMPT="$WORKDIR/hf2q.txt" \
  "$HF2Q" generate --model "$MODEL" --prompt "$PROMPT" --max-tokens 1 $HF2Q_FLAG \
  >/dev/null 2>"$WORKDIR/hf2q.stderr" || true

# 2. llama.cpp render — extract jinja from gguf, render via debug-template-parser.
python3 "$EXTRACT_PY" "$MODEL" >"$WORKDIR/template.jinja"
case "$MODE" in
  thinking) REASONING_FLAG="--enable-reasoning=1" ;;
  no-thinking) REASONING_FLAG="--enable-reasoning=0" ;;
esac

# llama-debug-template-parser doesn't take a prompt; it ships fixed test messages.
# To get a render against OUR exact user prompt we use a Python jinja render
# that uses the same values hf2q feeds.
python3 <<PY > "$WORKDIR/llama.txt"
import sys, json
import jinja2
tmpl = open("$WORKDIR/template.jinja").read()
env = jinja2.Environment(undefined=jinja2.DebugUndefined)
env.globals["raise_exception"] = lambda m: ""
template = env.from_string(tmpl)
ctx = dict(
    messages=[{"role": "user", "content": $(printf '%s' "$PROMPT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}],
    add_generation_prompt=True,
    bos_token="<|im_start|>",
    eos_token="<|im_end|>",
    enable_thinking=$([ "$MODE" = "thinking" ] && echo True || echo False),
)
out = template.render(**ctx)
sys.stdout.write(out)
PY

# 3. Compare.
hf_sum=$(shasum -a 256 "$WORKDIR/hf2q.txt" | awk '{print $1}')
ll_sum=$(shasum -a 256 "$WORKDIR/llama.txt" | awk '{print $1}')
hf_bytes=$(wc -c < "$WORKDIR/hf2q.txt" | tr -d ' ')
ll_bytes=$(wc -c < "$WORKDIR/llama.txt" | tr -d ' ')

echo "hf2q     bytes=$hf_bytes  sha256=$hf_sum"
echo "llama.cpp bytes=$ll_bytes  sha256=$ll_sum"
if [ "$hf_sum" = "$ll_sum" ]; then
  echo "RENDER PARITY: PASS"
  exit 0
else
  echo "RENDER PARITY: FAIL"
  diff "$WORKDIR/hf2q.txt" "$WORKDIR/llama.txt" || true
  exit 1
fi
