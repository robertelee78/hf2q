#!/usr/bin/env bash
# ADR-022 Phase 1 P1.8 byte-equal sub-AC.
#
# Verifies hf2q's mlx-native runtime and llama.cpp's llama-completion produce
# byte-identical generated text on the original ADR-022 motivating file
# (gemma4-ara-2pass-APEX-Q5_K_M.gguf, with Q5_1 + IQ4_NL + Q6_K + Q8_0 + F32
# weights) under matched greedy sampling.
#
# Run from /opt/hf2q. Requires:
#   - cargo build --release --bin hf2q (already done if PATH already includes ./target/release)
#   - llama-completion in PATH (Homebrew ggml package)
#
# Falsifier: any non-zero exit means the byte-equal AC has regressed.

set -euo pipefail

MODEL=/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf
PROMPT_FILE=/tmp/adr022_p18_prompt.txt
HF2Q_OUT=/tmp/adr022_p18_hf2q.out
LLAMA_OUT=/tmp/adr022_p18_llama.out
PROMPT_TEXT="What is 2+2?"

if [[ ! -f "$MODEL" ]]; then
  echo "FAIL: model not found at $MODEL" >&2
  exit 2
fi
if ! command -v llama-completion >/dev/null 2>&1; then
  echo "FAIL: llama-completion not in PATH" >&2
  exit 2
fi
if [[ ! -x ./target/release/hf2q ]]; then
  echo "FAIL: ./target/release/hf2q not found; run 'cargo build --release --bin hf2q' first" >&2
  exit 2
fi

# Step 1: hf2q dumps the rendered prompt + generates 32 tokens at temp=0.
HF2Q_DUMP_RENDERED_PROMPT="$PROMPT_FILE" \
  ./target/release/hf2q generate \
    --model "$MODEL" \
    --prompt "$PROMPT_TEXT" \
    --max-tokens 1 --temperature 0 \
  >/dev/null 2>&1

if [[ ! -s "$PROMPT_FILE" ]]; then
  echo "FAIL: HF2Q_DUMP_RENDERED_PROMPT did not produce $PROMPT_FILE" >&2
  exit 1
fi

# Generate the actual text output from hf2q (separate run, no dump).
./target/release/hf2q generate \
  --model "$MODEL" \
  --prompt "$PROMPT_TEXT" \
  --max-tokens 32 --temperature 0 \
  2>/dev/null \
| awk '/^prefill: /{flag=1; next} /^--- mlx-native:/{flag=0} flag {print}' \
| tr -d '\n' \
| sed 's/\[HF2Q_TQ_CODEBOOK_BITS\] [^ ]* [^ ]* [^ ]* [^ ]* [^ ]*//' \
> "$HF2Q_OUT"

# Step 2: feed hf2q's exact rendered prompt to llama-completion under greedy.
# llama-completion echoes the prompt then the generation; we want only the
# generated continuation. The rendered prompt ends with `<channel|>` (the
# Gemma4 chat-template's thought-channel-close marker), so we delete from
# the start of llama's output through that marker, then strip ` [end of text]`.
llama-completion \
  -m "$MODEL" \
  -f "$PROMPT_FILE" \
  -n 32 --temp 0 --top-p 1.0 --top-k 1 --repeat-penalty 1.0 \
  --no-warmup -sp --no-perf -no-cnv --jinja \
  2>/dev/null \
| sed -n '/<channel|>/,$p' \
| python3 -c 'import sys; s=sys.stdin.read(); i=s.find("<channel|>"); s=s[i+len("<channel|>"):] if i>=0 else s; s=s.replace(" [end of text]", "").rstrip("\n"); sys.stdout.write(s)' \
> "$LLAMA_OUT"

if [[ ! -s "$HF2Q_OUT" || ! -s "$LLAMA_OUT" ]]; then
  echo "FAIL: empty hf2q ($(wc -c <"$HF2Q_OUT") B) or llama ($(wc -c <"$LLAMA_OUT") B) output" >&2
  echo "--- hf2q ---" >&2; cat "$HF2Q_OUT" >&2; echo >&2
  echo "--- llama ---" >&2; cat "$LLAMA_OUT" >&2; echo >&2
  exit 1
fi

if diff -q "$HF2Q_OUT" "$LLAMA_OUT" >/dev/null 2>&1; then
  echo "PASS: byte-equal text output between hf2q and llama-completion"
  echo "  output: $(cat "$HF2Q_OUT")"
  exit 0
fi

echo "FAIL: hf2q output differs from llama-completion" >&2
echo "--- hf2q ---" >&2; cat "$HF2Q_OUT" >&2; echo >&2
echo "--- llama ---" >&2; cat "$LLAMA_OUT" >&2; echo >&2
diff <(od -c "$HF2Q_OUT" | head) <(od -c "$LLAMA_OUT" | head) || true
exit 1
