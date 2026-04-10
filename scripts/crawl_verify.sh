#!/usr/bin/env bash
# crawl_verify.sh — ADR-005 Phase 1b Design C correctness fixture builder.
# Runs llama-cli and hf2q on the same GGUF + prompt at T=0 greedy, compares
# their outputs by longest common byte prefix, classifies divergence, and
# (with --commit) writes tests/fixtures/{crawl_baseline.tokens,
# llama_cpp_reference.tokens, crawl_verified.meta}.
#
# Usage: scripts/crawl_verify.sh <gguf_path> [--commit] [--prompt-file PATH]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_PROMPT="tests/bench_prompt_128.txt"
HF2Q_BIN="target/release/hf2q"
OUT_HF2Q="/tmp/crawl_hf2q.txt"; LOG_HF2Q="/tmp/crawl_hf2q.log"
OUT_LLAMA="/tmp/crawl_llama.txt"; LOG_LLAMA="/tmp/crawl_llama.log"

usage() {
  cat <<EOF
Usage: scripts/crawl_verify.sh <gguf_path> [--commit] [--prompt-file PATH]
  <gguf_path>         Path to the Gemma 4 GGUF model (required)
  --commit            Write fixtures to tests/fixtures/ on success
  --prompt-file PATH  Override the canonical prompt (default: $DEFAULT_PROMPT)
EOF
}
err() { echo "error: $*" >&2; exit 1; }

COMMIT=0; PROMPT_FILE="$DEFAULT_PROMPT"; GGUF_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --commit) COMMIT=1; shift ;;
    --prompt-file)
      [[ $# -ge 2 ]] || err "--prompt-file requires an argument"
      PROMPT_FILE="$2"; shift 2 ;;
    -*) usage >&2; err "unknown flag: $1" ;;
    *)  [[ -z "$GGUF_PATH" ]] || err "unexpected positional arg: $1"
        GGUF_PATH="$1"; shift ;;
  esac
done
[[ -n "$GGUF_PATH" ]] || { usage >&2; exit 1; }

# Validate inputs and locate tools ------------------------------------------
[[ -f "$GGUF_PATH"   ]] || err "GGUF file not found: $GGUF_PATH"
[[ -f "$PROMPT_FILE" ]] || err "Prompt file not found: $PROMPT_FILE"
[[ -x "$HF2Q_BIN"    ]] || err "hf2q binary not found at $HF2Q_BIN (run: cargo build --release)"

# llama-completion is the headless single-shot binary; llama-cli is the chat
# REPL and will hang waiting for stdin even with prompt-from-file. We MUST use
# llama-completion to get a deterministic 128-token output and then exit.
if command -v llama-completion >/dev/null 2>&1; then
  LLAMA_BIN="$(command -v llama-completion)"
elif [[ -x "/opt/llama.cpp/build/bin/llama-completion" ]]; then
  LLAMA_BIN="/opt/llama.cpp/build/bin/llama-completion"
else
  err "llama-completion not found on PATH or at /opt/llama.cpp/build/bin/llama-completion"
fi

if   command -v shasum    >/dev/null 2>&1; then SHA256_CMD=(shasum -a 256)
elif command -v sha256sum >/dev/null 2>&1; then SHA256_CMD=(sha256sum)
else err "Neither shasum nor sha256sum is available"
fi
GGUF_SHA256="$("${SHA256_CMD[@]}" "$GGUF_PATH" | awk '{print $1}')"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "=== Crawl Verify — ADR-005 Phase 1b Design C ==="
echo "GGUF:        $GGUF_PATH"
echo "GGUF sha256: $GGUF_SHA256"
echo "Prompt:      $PROMPT_FILE"
echo "hf2q:        $HF2Q_BIN"
echo "llama-comp:  $LLAMA_BIN"
echo "git HEAD:    $GIT_HEAD"
echo
echo "--- Chat template probe (best-effort) ---"
if command -v strings >/dev/null 2>&1; then
  # `set -o pipefail` + `grep -m1` produces SIGPIPE on `strings` once grep exits,
  # which trips the `||` even when grep matched. Capture into a variable first.
  CHAT_PROBE="$(strings "$GGUF_PATH" 2>/dev/null | grep -m1 -i "chat_template" || true)"
  if [[ -n "$CHAT_PROBE" ]]; then
    echo "$CHAT_PROBE"
  else
    echo "(no tokenizer.chat_template string found via 'strings')"
  fi
else
  echo "(strings not available; skipping template probe)"
fi
echo

# Run llama-completion and hf2q ---------------------------------------------
# Notes:
#   -st / --single-turn   : exit after one turn (otherwise waits for stdin)
#   </dev/null            : explicit EOF on stdin so no interactive prompt
#   --jinja               : apply the GGUF's tokenizer.chat_template (matches hf2q)
#   --no-display-prompt   : stdout = generated text only, no echo
echo "--- Running llama-completion (T=0 greedy, 128 tokens) ---"
if ! "$LLAMA_BIN" --model "$GGUF_PATH" --file "$PROMPT_FILE" \
      --predict 128 --temp 0 --seed 42 \
      --no-display-prompt --jinja \
      -st -ngl 999 \
      </dev/null >"$OUT_LLAMA" 2>"$LOG_LLAMA"; then
  echo "llama-completion failed. See $LOG_LLAMA" >&2; exit 3
fi

echo "--- Running hf2q (T=0 greedy, 128 tokens) ---"
if ! "$HF2Q_BIN" generate --model "$GGUF_PATH" --prompt-file "$PROMPT_FILE" \
      --max-tokens 128 --temperature 0 \
      >"$OUT_HF2Q" 2>"$LOG_HF2Q"; then
  echo "hf2q failed. See $LOG_HF2Q" >&2; exit 3
fi

# Compare via portable python one-shot --------------------------------------
command -v python3 >/dev/null 2>&1 || err "python3 is required for byte-prefix comparison"
read -r LLAMA_BYTES HF2Q_BYTES COMMON_BYTES DIVERGE_A DIVERGE_B < <(
  python3 - "$OUT_LLAMA" "$OUT_HF2Q" <<'PY'
import sys, json
a = open(sys.argv[1], "rb").read()
b = open(sys.argv[2], "rb").read()
n = min(len(a), len(b)); i = 0
while i < n and a[i] == b[i]: i += 1
def snip(buf, start, length=200):
    s = buf[start:start+length].decode("utf-8", errors="replace")
    if len(buf) - start > length: s += "..."
    return json.dumps(s)
print(len(a), len(b), i, snip(a, i), snip(b, i))
PY
)
APPROX_TOKENS=$(( COMMON_BYTES / 4 ))

echo
echo "--- Comparison ---"
echo "llama-cli output: ${LLAMA_BYTES} bytes"
echo "hf2q output:      ${HF2Q_BYTES} bytes"
echo "Common prefix:    ${COMMON_BYTES} bytes (~${APPROX_TOKENS} tokens, ~4 chars/tok)"

# Classify -------------------------------------------------------------------
if   [[ "$COMMON_BYTES" -eq "$LLAMA_BYTES" && "$COMMON_BYTES" -eq "$HF2Q_BYTES" ]]; then
  CLASS="PERFECT"; CLASS_MSG="Token-for-token match with llama.cpp."
elif [[ "$COMMON_BYTES" -lt 20  ]]; then
  CLASS="RED";     CLASS_MSG="Chat template mismatch or major math bug. hf2q is NOT porting llama.cpp's pipeline correctly."
elif [[ "$COMMON_BYTES" -lt 200 ]]; then
  CLASS="YELLOW";  CLASS_MSG="Early FP drift. Partial port correctness."
else
  CLASS="GREEN";   CLASS_MSG="Normal FP-associativity drift; Crawl is real."
fi
echo
echo "Classification: $CLASS"
echo "  $CLASS_MSG"
echo
echo "--- Divergence point (first 200 chars after byte $COMMON_BYTES) ---"
echo "llama-cli: $DIVERGE_A"
echo "hf2q:      $DIVERGE_B"
echo

# Optionally commit fixtures -------------------------------------------------
if [[ "$COMMIT" -eq 1 ]]; then
  FIX_DIR="tests/fixtures"; mkdir -p "$FIX_DIR"
  cp "$OUT_HF2Q"  "$FIX_DIR/crawl_baseline.tokens"
  cp "$OUT_LLAMA" "$FIX_DIR/llama_cpp_reference.tokens"
  CLEAN=$(git diff --quiet && git diff --cached --quiet && echo yes || echo no)
  {
    echo "# ADR-005 Phase 1b Design C — crawl verification metadata"
    echo "timestamp_utc:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_head:           $GIT_HEAD"
    echo "git_status_clean:   $CLEAN"
    echo "gguf_path:          $GGUF_PATH"
    echo "gguf_sha256:        $GGUF_SHA256"
    echo "prompt_file:        $PROMPT_FILE"
    echo "llama_cli:          $LLAMA_BIN"
    echo "llama_bytes:        $LLAMA_BYTES"
    echo "hf2q_bytes:         $HF2Q_BYTES"
    echo "common_prefix:      $COMMON_BYTES"
    echo "approx_tokens:      $APPROX_TOKENS"
    echo "classification:     $CLASS"
    echo "classification_msg: $CLASS_MSG"
  } > "$FIX_DIR/crawl_verified.meta"
  echo "Fixtures committed to $FIX_DIR/. Run 'git add $FIX_DIR && git commit' to persist."
fi

[[ "$CLASS" == "RED" ]] && exit 2 || exit 0
