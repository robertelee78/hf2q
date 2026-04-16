#!/usr/bin/env bash
# parity_check.sh — ADR-009 Phase 2: parity validation suite.
#
# Runs hf2q parity checks against locked llama.cpp reference outputs.
# Uses the `hf2q parity check` CLI for structured comparison.
#
# Usage:
#   scripts/parity_check.sh <gguf_path>
#
# Exit codes:
#   0  all checks passed
#   1  usage / env error
#   2  parity check failed

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

# --- Check 1: Short deterministic — exact byte comparison ---
echo "--- Check 1: short_hello (exact byte comparison) ---"
if $HF2Q_BIN parity check --model "$GGUF_PATH" --prompt short_hello --min-prefix 29 2>/dev/null; then
    echo "  PASS"
    ((PASS++))
else
    echo "  FAIL"
    ((FAIL++))
fi
echo

# --- Check 2: Sourdough coherence gate ---
echo "--- Check 2: sourdough (min-prefix 3094) ---"
if $HF2Q_BIN parity check --model "$GGUF_PATH" --prompt sourdough --min-prefix 3094 2>/dev/null; then
    echo "  PASS"
    ((PASS++))
else
    echo "  FAIL"
    ((FAIL++))
fi
echo

# --- Check 3: Sliding wrap prompt ---
echo "--- Check 3: sliding_wrap (min-prefix 700) ---"
if $HF2Q_BIN parity check --model "$GGUF_PATH" --prompt sliding_wrap --min-prefix 700 2>/dev/null; then
    echo "  PASS"
    ((PASS++))
else
    echo "  FAIL"
    ((FAIL++))
fi
echo

# --- Summary ---
TOTAL=$((PASS + FAIL))
echo "=== Parity Summary: $PASS/$TOTAL passed ==="
if [[ $FAIL -gt 0 ]]; then
    exit 2
fi
exit 0
