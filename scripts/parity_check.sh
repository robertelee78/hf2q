#!/usr/bin/env bash
# parity_check.sh — ADR-009 Phase 2: quick parity validation suite.
#
# Runs the sourdough coherence gate and a short deterministic sanity check.
# Used for routine validation during development and CI.
#
# Usage:
#   scripts/parity_check.sh <gguf_path>
#
# Exit codes:
#   0  all checks passed
#   1  usage / env error
#   2  parity check failed
#   3  tool invocation failure

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HF2Q_BIN="target/release/hf2q"

err() { echo "error: $*" >&2; exit 1; }

GGUF_PATH="${1:-}"
[[ -n "$GGUF_PATH" ]] || { echo "Usage: scripts/parity_check.sh <gguf_path>"; exit 1; }
[[ -f "$GGUF_PATH" ]] || err "GGUF not found: $GGUF_PATH"
[[ -x "$HF2Q_BIN"  ]] || err "hf2q binary not found (run: cargo build --release)"

GIT_HEAD="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
PASS=0
FAIL=0

echo "=== ADR-009 Phase 2: Parity Validation Suite ==="
echo "GGUF: $GGUF_PATH"
echo "hf2q: $HF2Q_BIN"
echo "git:  $GIT_HEAD"
echo

# --- Check 1: Short deterministic sanity ---
echo "--- Check 1: Short deterministic output ---"
SHORT_OUT=$("$HF2Q_BIN" generate --model "$GGUF_PATH" \
    --prompt "What is 2+2?" --max-tokens 10 --temperature 0 2>/dev/null)
if echo "$SHORT_OUT" | grep -qi "4"; then
    echo "PASS: short prompt produces coherent output containing '4'"
    echo "  Output: $SHORT_OUT"
    ((PASS++))
else
    echo "FAIL: short prompt output does not contain '4'"
    echo "  Output: $SHORT_OUT"
    ((FAIL++))
fi
echo

# --- Check 2: Sourdough coherence gate ---
echo "--- Check 2: Sourdough coherence gate (min-prefix 3094) ---"
if scripts/sourdough_gate.sh "$GGUF_PATH" --min-prefix 3094; then
    ((PASS++))
else
    ((FAIL++))
fi
echo

# --- Summary ---
echo "=== Parity Summary: $PASS passed, $FAIL failed ==="
if [[ $FAIL -gt 0 ]]; then
    exit 2
fi
exit 0
