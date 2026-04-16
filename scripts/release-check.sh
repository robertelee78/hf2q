#!/usr/bin/env bash
# release-check.sh — Hardening v1 reproducible gate runner.
#
# Runs the merge-gating checks defined in docs/shipping-contract.md in
# sequence and exits non-zero on first fail:
#
#   1. parity suite   — short_hello, sourdough (>=3094), sliding_wrap (>=700)
#   2. perf sanity    — decode tok/s on the sourdough prompt >= floor
#
# The parity suite wraps `hf2q parity check` via scripts/parity_check.sh.
# The perf sanity runs hf2q on the canonical sourdough prompt and parses
# tok/s from stderr, matching the "X.X tok/s" shape already emitted by
# the decode loop.
#
# Usage:
#   scripts/release-check.sh <gguf_path>
#   scripts/release-check.sh <gguf_path> --min-decode-tps 95 --max-tokens 1000
#
# Exit codes:
#   0  all gates passed
#   1  usage / env error
#   2  a gate failed
#   3  tool invocation failure

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HF2Q_BIN="target/release/hf2q"
MIN_DECODE_TPS="95"
MAX_TOKENS="1000"
PERF_PROMPT="Complrehensive instructions for making sourdough bread."
PERF_LOG="/tmp/release_check_perf.log"
GGUF_PATH=""

usage() {
  cat <<EOF
Usage: scripts/release-check.sh <gguf_path> [--min-decode-tps N] [--max-tokens N]
  <gguf_path>         Path to the Gemma 4 GGUF model (required)
  --min-decode-tps N  Decode tok/s floor for perf sanity (default: 95)
  --max-tokens N      Max tokens for perf sanity run (default: 1000)

Exit codes:
  0  all gates passed
  1  usage / env error
  2  a gate failed
  3  tool invocation failure
EOF
}
err() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --min-decode-tps)
      [[ $# -ge 2 ]] || err "--min-decode-tps requires an argument"
      MIN_DECODE_TPS="$2"; shift 2 ;;
    --max-tokens)
      [[ $# -ge 2 ]] || err "--max-tokens requires an argument"
      MAX_TOKENS="$2"; shift 2 ;;
    -*) usage >&2; err "unknown flag: $1" ;;
    *)  [[ -z "$GGUF_PATH" ]] || err "unexpected positional arg: $1"
        GGUF_PATH="$1"; shift ;;
  esac
done
[[ -n "$GGUF_PATH" ]] || { usage >&2; exit 1; }
[[ -f "$GGUF_PATH" ]] || err "GGUF file not found: $GGUF_PATH"
[[ -x "$HF2Q_BIN"  ]] || err "hf2q binary not found at $HF2Q_BIN (run: cargo build --release --features metal)"
[[ -x "$SCRIPT_DIR/parity_check.sh" ]] || err "scripts/parity_check.sh not found or not executable"

GIT_HEAD="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "=== hf2q release-check (hardening v1) ==="
echo "GGUF:            $GGUF_PATH"
echo "hf2q:            $HF2Q_BIN"
echo "git HEAD:        $GIT_HEAD"
echo "min decode tps:  $MIN_DECODE_TPS"
echo "perf max-tokens: $MAX_TOKENS"
echo

# --- Gate 1: parity suite (short_hello + sourdough + sliding_wrap) ---
echo "--- Gate 1/2: parity suite ---"
if ! "$SCRIPT_DIR/parity_check.sh" "$GGUF_PATH"; then
  echo "FAIL: parity gate tripped. See output above." >&2
  exit 2
fi

# --- Gate 2: perf sanity on the sourdough prompt ---
echo
echo "--- Gate 2/2: perf sanity (decode tok/s >= $MIN_DECODE_TPS on sourdough prompt) ---"
if ! "$HF2Q_BIN" generate --model "$GGUF_PATH" --prompt "$PERF_PROMPT" \
      --max-tokens "$MAX_TOKENS" --temperature 0 \
      >/dev/null 2>"$PERF_LOG"; then
  echo "perf sanity run crashed; see $PERF_LOG" >&2
  exit 3
fi

# Parse decode tok/s. The decode loop emits a line like
# "--- N tokens in X.XXs (Y.Y tok/s) ---". Take the last match (decode
# rate, not prefill rate if both are emitted).
TPS="$(grep -oE '[0-9]+\.[0-9]+ tok/s' "$PERF_LOG" | tail -1 | awk '{print $1}')"
if [[ -z "$TPS" ]]; then
  echo "FAIL: no 'tok/s' line found in $PERF_LOG — cannot enforce perf sanity." >&2
  echo "      hf2q's decode timing output format may have changed; update this script." >&2
  exit 2
fi

# awk handles float comparison portably.
PASS_PERF="$(awk -v t="$TPS" -v m="$MIN_DECODE_TPS" 'BEGIN { print (t+0 >= m+0) ? "1" : "0" }')"
echo "decode: $TPS tok/s (floor: $MIN_DECODE_TPS)"
if [[ "$PASS_PERF" != "1" ]]; then
  echo "FAIL: decode $TPS tok/s is below floor $MIN_DECODE_TPS." >&2
  echo "      Bisect recent changes to the forward pass, KV cache, SDPA, MoE," >&2
  echo "      or lm_head before landing." >&2
  exit 2
fi

echo
echo "=== release-check PASS — all gates green ==="
exit 0
