#!/usr/bin/env bash
# qwen35_tokenizer_parity.sh — gate-time parity check for hf2q's
# GGUF-driven Qwen3.5 tokenizer vs llama-tokenize on the same GGUF.
#
# Per ADR-013 §"Sovereignty", llama-tokenize is invoked here as an
# external black-box reference. Never linked at build/test/CI time;
# this script is operator-run on demand to verify the contract.
#
# The contract: hf2q's `build_tokenizer_from_gguf` (cmd_generate_qwen35
# path) must produce token streams byte-equivalent to llama-tokenize on
# the same GGUF, INCLUDING BOS. Both sides honour
# `tokenizer.ggml.add_bos_token=true`:
#  * llama-tokenize prepends BOS by default (use `--no-bos` to suppress).
#  * hf2q's `tokenize_rendered_prompt_llama_style` explicitly prepends
#    BOS via `resolve_token_id(..., "tokenizer.ggml.bos_token_id")`,
#    falling back to `llama_cpp_special_token_id_for_model("gpt2", ...)`
#    → 11 (mirrors llama-vocab.cpp:1838-1839 GPT-2 default
#    `special_bos_id = 11`) when GGUF lacks an explicit
#    `tokenizer.ggml.bos_token_id`.
# 2026-05-03 — earlier versions of this script asserted hf2q does NOT
# auto-prepend and stripped llama's leading BOS. That premise was stale
# after ADR-015 iter42 (commit 0122100, 2026-04-29) which added
# `add_bos_token` honour repo-wide. Both streams now compared with-BOS.
#
# Usage:
#   scripts/qwen35_tokenizer_parity.sh <gguf_path>
#
# Exit codes:
#   0  parity verified on all fixture prompts
#   1  usage / env error
#   2  hf2q and llama-tokenize disagree on some fixture
#   3  tool invocation failure

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HF2Q_BIN="target/release/hf2q"
[[ -x "$HF2Q_BIN" ]] || { echo "error: build hf2q first (cargo build --release)" >&2; exit 1; }

GGUF_PATH="${1:?usage: $0 <gguf_path>}"
[[ -f "$GGUF_PATH" ]] || { echo "error: GGUF not found: $GGUF_PATH" >&2; exit 1; }

if [[ -x "/opt/llama.cpp/build/bin/llama-tokenize" ]]; then
  LLAMA_TOK="/opt/llama.cpp/build/bin/llama-tokenize"
elif command -v llama-tokenize >/dev/null 2>&1; then
  LLAMA_TOK="$(command -v llama-tokenize)"
else
  echo "error: llama-tokenize not found on PATH or at /opt/llama.cpp/build/bin/" >&2
  exit 1
fi

# llama-tokenize stdout format: lines like `   248045 -> '<|im_start|>'`
# We extract the leading numeric id from each `-> '` line.
extract_llama_ids() {
  local prompt="$1"
  "$LLAMA_TOK" --model "$GGUF_PATH" -p "$prompt" 2>/dev/null \
    | awk '/ -> /{print $1}' | tr '\n' ' '
}

# hf2q's `generate --debug-tokenize-only` (added in this commit) renders
# the chat template (or, with `--chat-template-file`, an override),
# tokenizes via the GGUF-driven path, and prints the token IDs on
# stdout — one line, space-separated. Exits without loading the model.
extract_hf2q_ids() {
  local prompt="$1"
  HF2Q_DEBUG_TOKENIZE_ONLY=1 "$HF2Q_BIN" generate \
    --model "$GGUF_PATH" \
    --chat-template-file "$REPO_ROOT/scripts/qwen35_raw_passthrough.jinja" \
    --prompt "$prompt" --max-tokens 1 --temperature 0 2>/dev/null \
    | awk '/^TOKENIZE_DEBUG_IDS/{$1=""; print}' | sed 's/^ //'
}

# 2026-05-03 — `strip_bos` removed: both sides auto-prepend BOS now,
# so the streams compare directly. See the contract block at the top
# of the file for why the prior strip-BOS premise is stale.
trim_ws() {
  echo "$1" | sed 's/[[:space:]]*$//'
}

PROMPTS=(
  '<|im_start|>user
How to make bread?<|im_end|>
<|im_start|>assistant'
  'How to make bread?'
  'Hello, world! Café — Москва'
)

failures=0
for prompt in "${PROMPTS[@]}"; do
  printf '\n--- fixture: %q ---\n' "$prompt"
  llama_ids="$(extract_llama_ids "$prompt")" || { echo "llama-tokenize failed"; exit 3; }
  hf2q_ids="$(extract_hf2q_ids "$prompt")"  || { echo "hf2q debug-tokenize failed"; exit 3; }
  echo "llama-tokenize: $llama_ids"
  echo "hf2q:           $hf2q_ids"

  llama_ids="$(trim_ws "$llama_ids")"
  hf2q_ids="$(trim_ws "$hf2q_ids")"

  if [[ "$llama_ids" == "$hf2q_ids" ]]; then
    echo "PARITY: PASS"
  else
    echo "PARITY: FAIL — token streams differ"
    failures=$((failures + 1))
  fi
done

if [[ $failures -gt 0 ]]; then
  echo
  echo "$failures fixture(s) failed parity"
  exit 2
fi

echo
echo "All fixtures: PARITY PASS"
exit 0
