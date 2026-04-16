#!/usr/bin/env bash
# long_decode.sh — Long-decode coherence smoke (Tier 1 "DP-class long-decode
# drift on non-gate prompts" reproducer, made parameterized).
#
# Runs hf2q on a long-decode prompt at T=0 greedy, up to max-tokens. Exits
# non-zero on crash/hang and reports completion byte count + decode tok/s.
# Does NOT assert against a locked reference (there isn't one at 20k tokens
# for non-gate prompts; drift in this regime is documented but un-locked).
# This is a smoke gate — catches crashes, hangs, obvious regressions, and
# perf cliffs, not drift byte-exactness.
#
# Relationship to scripts/release-check.sh: long_decode.sh is *opt-in*, not
# part of the default release-check — a 20k-token decode takes ~3 minutes
# on the Gemma-4 26B DWQ and is too slow for every merge. Run this for
# sliding-window-wrap investigations, 20k+ context work, and nightly-style
# coherence sweeps.
#
# Replaces the bare command that used to live at docs/coherence-test.txt.
#
# Usage:
#   scripts/long_decode.sh <gguf_path>
#   scripts/long_decode.sh <gguf_path> --max-tokens 20000 --min-decode-tps 85
#   scripts/long_decode.sh <gguf_path> --prompt "..." --prompt-file path
#
# Exit codes:
#   0  completed cleanly, perf floor met (if enforced)
#   1  usage / env error
#   2  perf floor violated
#   3  tool invocation failure (hf2q crashed / timed out)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HF2Q_BIN="target/release/hf2q"

# The canonical long-decode prompt. Note the "Comlprehensive" typo is
# deliberate — the exact token sequence matters for reproducibility;
# different typos tokenize differently and produce different trajectories.
# This typo is distinct from the sourdough_gate prompt ("Complrehensive"),
# on purpose: the two gates exercise independent decode paths.
DEFAULT_PROMPT="Comlprehensive instructions for making sourdough bread."

MAX_TOKENS="20000"
MIN_DECODE_TPS=""          # empty = don't enforce; set to a number to enforce
PROMPT=""
PROMPT_FILE=""
GGUF_PATH=""
SHOW_OUTPUT=0
OUT="/tmp/long_decode_out.txt"
LOG="/tmp/long_decode.log"

usage() {
  cat <<EOF
Usage: scripts/long_decode.sh <gguf_path> [options]

Options:
  --max-tokens N      Max decode tokens (default: 20000)
  --min-decode-tps N  Enforce decode tok/s floor (default: off)
  --prompt "..."      Override the long-decode prompt
  --prompt-file PATH  Read prompt from a file
  --show              Print the generated text to stdout after the gate

Exit codes:
  0  gate passed
  1  usage / env error
  2  perf floor violated
  3  tool invocation failure
EOF
}
err() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --max-tokens)
      [[ $# -ge 2 ]] || err "--max-tokens requires an argument"
      MAX_TOKENS="$2"; shift 2 ;;
    --min-decode-tps)
      [[ $# -ge 2 ]] || err "--min-decode-tps requires an argument"
      MIN_DECODE_TPS="$2"; shift 2 ;;
    --prompt)
      [[ $# -ge 2 ]] || err "--prompt requires an argument"
      PROMPT="$2"; shift 2 ;;
    --prompt-file)
      [[ $# -ge 2 ]] || err "--prompt-file requires an argument"
      PROMPT_FILE="$2"; shift 2 ;;
    --show)
      SHOW_OUTPUT=1; shift ;;
    -*) usage >&2; err "unknown flag: $1" ;;
    *)  [[ -z "$GGUF_PATH" ]] || err "unexpected positional arg: $1"
        GGUF_PATH="$1"; shift ;;
  esac
done
[[ -n "$GGUF_PATH" ]] || { usage >&2; exit 1; }
[[ -f "$GGUF_PATH" ]] || err "GGUF file not found: $GGUF_PATH"
[[ -x "$HF2Q_BIN" ]] || err "hf2q binary not found at $HF2Q_BIN (run: cargo build --release)"

if [[ -n "$PROMPT_FILE" ]]; then
  [[ -f "$PROMPT_FILE" ]] || err "prompt file not found: $PROMPT_FILE"
  [[ -z "$PROMPT" ]] || err "--prompt and --prompt-file are mutually exclusive"
fi
if [[ -z "$PROMPT" && -z "$PROMPT_FILE" ]]; then
  PROMPT="$DEFAULT_PROMPT"
fi

GIT_HEAD="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "=== hf2q long-decode coherence smoke ==="
echo "GGUF:           $GGUF_PATH"
echo "hf2q:           $HF2Q_BIN"
echo "git HEAD:       $GIT_HEAD"
echo "max tokens:     $MAX_TOKENS"
if [[ -n "$MIN_DECODE_TPS" ]]; then
  echo "perf floor:     $MIN_DECODE_TPS tok/s"
else
  echo "perf floor:     (not enforced)"
fi
if [[ -n "$PROMPT_FILE" ]]; then
  echo "prompt file:    $PROMPT_FILE"
else
  echo "prompt:         $PROMPT"
fi
echo

echo "--- Running hf2q (T=0 greedy, $MAX_TOKENS tokens) ---"
set +e
if [[ -n "$PROMPT_FILE" ]]; then
  "$HF2Q_BIN" generate --model "$GGUF_PATH" --prompt-file "$PROMPT_FILE" \
      --max-tokens "$MAX_TOKENS" --temperature 0 \
      >"$OUT" 2>"$LOG"
else
  "$HF2Q_BIN" generate --model "$GGUF_PATH" --prompt "$PROMPT" \
      --max-tokens "$MAX_TOKENS" --temperature 0 \
      >"$OUT" 2>"$LOG"
fi
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "FAIL: hf2q exited with status $RC; see $LOG" >&2
  exit 3
fi

OUT_BYTES="$(wc -c < "$OUT" | tr -d ' ')"
echo "output: $OUT_BYTES bytes"

TPS="$(grep -oE '[0-9]+\.[0-9]+ tok/s' "$LOG" | tail -1 | awk '{print $1}')"
if [[ -n "$TPS" ]]; then
  echo "decode: $TPS tok/s"
else
  echo "decode: (no tok/s reported)"
fi

if [[ -n "$MIN_DECODE_TPS" ]]; then
  if [[ -z "$TPS" ]]; then
    echo "FAIL: --min-decode-tps set to $MIN_DECODE_TPS but no tok/s was parsed." >&2
    exit 2
  fi
  PASS_PERF="$(awk -v t="$TPS" -v m="$MIN_DECODE_TPS" 'BEGIN { print (t+0 >= m+0) ? "1" : "0" }')"
  if [[ "$PASS_PERF" != "1" ]]; then
    echo "FAIL: decode $TPS tok/s is below floor $MIN_DECODE_TPS." >&2
    exit 2
  fi
fi

echo
echo "=== long_decode PASS ==="

if [[ "$SHOW_OUTPUT" -eq 1 ]]; then
  echo
  echo "--- generated output ($OUT_BYTES bytes) ---"
  cat "$OUT"
  # Trailing newline if hf2q didn't emit one (prevents PS1 clobber).
  [[ -n "$(tail -c1 "$OUT" 2>/dev/null)" ]] && echo
fi

exit 0
