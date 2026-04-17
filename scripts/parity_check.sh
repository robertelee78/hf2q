#!/usr/bin/env bash
# parity_check.sh — ADR-009 Phase 2 + ADR-005 Gates C/E/F.
#
# Runs hf2q parity checks against locked llama.cpp reference outputs.
# Each prompt is run N times (default 3) at T=0; every run must pass its
# declared min-prefix threshold. This encodes ADR-005 Gate F (deterministic
# reproducibility): a single flaky run fails the whole gate. Gate E is
# partially covered by the min-prefix floors; full divergence-point
# tracking is a follow-up.
#
# Usage:
#   scripts/parity_check.sh <gguf_path>
#   scripts/parity_check.sh <gguf_path> --n-runs 3
#
# Exit codes:
#   0  all checks passed (every run of every prompt met its threshold)
#   1  usage / env error
#   2  parity check failed (at least one run of one prompt failed)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HF2Q_BIN="target/release/hf2q"
N_RUNS="3"

err() { echo "error: $*" >&2; exit 1; }

GGUF_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-runs)
      [[ $# -ge 2 ]] || err "--n-runs requires an argument"
      N_RUNS="$2"; shift 2 ;;
    -*) err "unknown flag: $1" ;;
    *)  [[ -z "$GGUF_PATH" ]] || err "unexpected positional arg: $1"
        GGUF_PATH="$1"; shift ;;
  esac
done

[[ -n "$GGUF_PATH" ]] || { echo "Usage: scripts/parity_check.sh <gguf_path> [--n-runs N]"; exit 1; }
[[ -f "$GGUF_PATH" ]] || err "GGUF not found: $GGUF_PATH"
[[ -x "$HF2Q_BIN"  ]] || err "hf2q binary not found (run: cargo build --release)"

GIT_HEAD="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
PASS=0
FAIL=0

# Runs `hf2q parity check` N times for one prompt; every run must pass.
# Encodes Gate F (N≥3 determinism): a single flaky run fails the gate.
run_parity_n_times() {
  local prompt="$1"
  local min_prefix="$2"
  local label="$3"

  echo "--- $label (prompt=$prompt, min-prefix=$min_prefix, n_runs=$N_RUNS) ---"
  local run_pass=0
  local run_fail=0
  for run in $(seq 1 "$N_RUNS"); do
    if $HF2Q_BIN parity check --model "$GGUF_PATH" --prompt "$prompt" \
        --min-prefix "$min_prefix" 2>/dev/null >/dev/null; then
      run_pass=$((run_pass + 1))
    else
      run_fail=$((run_fail + 1))
      echo "  run $run: FAIL"
    fi
  done
  if (( run_fail == 0 )); then
    echo "  $run_pass/$N_RUNS PASS"
    ((PASS++))
  else
    echo "  $run_pass/$N_RUNS PASS, $run_fail/$N_RUNS FAIL (Gate F: determinism violated)"
    ((FAIL++))
  fi
  echo
}

echo "=== ADR-005 Parity Suite (Gates C/E + F via N=$N_RUNS rerun) ==="
echo "GGUF: $GGUF_PATH"
echo "hf2q: $HF2Q_BIN"
echo "git:  $GIT_HEAD"
echo

# Short deterministic — exact byte comparison
run_parity_n_times "short_hello"  29   "Check 1: short_hello (exact)"
# Sourdough coherence gate — mid-length
run_parity_n_times "sourdough"    3094 "Check 2: sourdough (exact-parity prompt)"
# Sliding wrap — long, ADR-010 Deferred exact parity (floor only)
run_parity_n_times "sliding_wrap" 700  "Check 3: sliding_wrap (ADR-010-deferred, floor)"

# --- Summary ---
TOTAL=$((PASS + FAIL))
echo "=== Parity Summary: $PASS/$TOTAL prompts fully deterministic ==="
if [[ $FAIL -gt 0 ]]; then
    exit 2
fi
exit 0
