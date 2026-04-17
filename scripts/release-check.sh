#!/usr/bin/env bash
# release-check.sh — ADR-005 Closeout Amendment reproducible gate runner.
#
# Runs the merge-gating checks defined in the ADR and in docs/shipping-
# contract.md in sequence, exiting non-zero on first fail:
#
#   1. parity suite (Gates C/E + F) — short_hello / sourdough / sliding_wrap
#      each run 3× at T=0; every run must pass its min-prefix threshold.
#   2. perf sanity (Gate B) — median-of-3 decode tok/s on sourdough ≥ floor.
#   3. prefill perf (Gate A) — batched prefill tok/s on a ≥2048-token prompt
#      ≥ floor. Requires prefill_2048.txt fixture (skipped if absent).
#
# Gates not yet wired here: D (frozen hf2q self-baseline), G (mlx-native
# dispatch counter thresholds). Both are follow-ups; see ADR-005 Closeout
# Amendment for scope.
#
# Parity suite wraps `hf2q parity check` via scripts/parity_check.sh.
# Perf gates parse tok/s from stderr of `hf2q generate`.
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
# Gate B floor: hf2q decode median 102.9-103.4 tok/s on M5 Max (HEAD post
# mlx-native 0.3.1 race fix); peer llama.cpp 102.01 tok/s median on identical
# setup. Floor at 100 encodes "within measurement variance of peer" — tight
# enough to flag actual regressions, loose enough to tolerate minor thermal
# jitter. Previous floor of 95 was set when hf2q was 17 tok/s below peer
# (Walk-era) and is now stale. See ADR-005 Closeout Amendment Gate B.
MIN_DECODE_TPS="100"
MAX_TOKENS="1000"
PERF_PROMPT="Complrehensive instructions for making sourdough bread."
PERF_LOG="/tmp/release_check_perf.log"
GGUF_PATH=""

usage() {
  cat <<EOF
Usage: scripts/release-check.sh <gguf_path> [--min-decode-tps N] [--max-tokens N]
  <gguf_path>         Path to the Gemma 4 GGUF model (required)
  --min-decode-tps N  Decode tok/s floor for perf sanity (default: 100,
                      = within variance of peer llama.cpp ~102 tok/s)
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

# --- Gate 1: parity suite (Gates C/E + F) ---
echo "--- Gate 1/3: parity suite ---"
if ! "$SCRIPT_DIR/parity_check.sh" "$GGUF_PATH"; then
  echo "FAIL: parity gate tripped. See output above." >&2
  exit 2
fi

# --- Gate 2: perf sanity on the sourdough prompt (median of 3 runs) ---
# Single-sample decode tok/s is thermal-jitter sensitive on M5 Max: a
# 1000-token decode sustains GPU load long enough that the chip throttles
# intermittently, dropping tok/s ~5 below steady-state. Median of 3 runs
# encodes the ADR Gate B intent "within measurement variance of peer
# median" — resilient to a single thermal dip, still tight enough to flag
# real regressions.
echo
echo "--- Gate 2/3: perf sanity (median-of-3 decode tok/s >= $MIN_DECODE_TPS) ---"
PERF_SAMPLES=()
for run in 1 2 3; do
  if ! "$HF2Q_BIN" generate --model "$GGUF_PATH" --prompt "$PERF_PROMPT" \
        --max-tokens "$MAX_TOKENS" --temperature 0 \
        >/dev/null 2>"$PERF_LOG"; then
    echo "perf sanity run $run crashed; see $PERF_LOG" >&2
    exit 3
  fi
  SAMPLE="$(grep -oE '[0-9]+\.[0-9]+ tok/s' "$PERF_LOG" | tail -1 | awk '{print $1}')"
  if [[ -z "$SAMPLE" ]]; then
    echo "FAIL: no 'tok/s' line found in $PERF_LOG (run $run)." >&2
    echo "      hf2q's decode timing output format may have changed; update this script." >&2
    exit 2
  fi
  PERF_SAMPLES+=("$SAMPLE")
  echo "  run $run: $SAMPLE tok/s"
done

# Sort ascending, take middle (index 1 of 3).
TPS="$(printf '%s\n' "${PERF_SAMPLES[@]}" | sort -n | awk 'NR==2')"

# awk handles float comparison portably.
PASS_PERF="$(awk -v t="$TPS" -v m="$MIN_DECODE_TPS" 'BEGIN { print (t+0 >= m+0) ? "1" : "0" }')"
echo "median: $TPS tok/s (floor: $MIN_DECODE_TPS)"
if [[ "$PASS_PERF" != "1" ]]; then
  echo "FAIL: median-of-3 decode $TPS tok/s is below floor $MIN_DECODE_TPS." >&2
  echo "      Samples: ${PERF_SAMPLES[*]}." >&2
  echo "      Bisect recent changes to the forward pass, KV cache, SDPA, MoE," >&2
  echo "      or lm_head before landing." >&2
  exit 2
fi


# --- Gate 3: prefill tok/s on a ≥2048-token prompt (batched path) ---
# ADR-005 Closeout Amendment Gate A: prefill tok/s parity vs llama.cpp on
# a ≥2048-token prompt. llama.cpp is the reference at ~3260 tok/s on M5 Max.
# hf2q batched prefill (post mlx-native 0.3.1 race fix + sdpa_sliding
# re-enable) sits at ~152-155 tok/s on the same prompt. True peer-parity
# is Run-scope (needs a flash-attn-style tiled kernel, not a one-liner).
# For now, floor at 150 tok/s catches genuine regressions in batched-
# prefill throughput without pretending the peer gap is closed.
PREFILL_2048_PROMPT="tests/evals/prompts/prefill_2048.txt"
# Floor at 140 tok/s: cold-start batched prefill hits 152-155 tok/s on
# M5 Max, but Gate 3 runs AFTER Gate 2's ~30s of sustained decode load,
# so thermal throttling can pull it into the 145-150 range on the third
# gate. 140 gives ~5-10 tok/s headroom below hot steady-state — tight
# enough to flag real regressions (pre-fix batched prefill was broken
# at seq_len>1024, not slow; a real regression would drop far below 140).
MIN_PREFILL_TPS="140"
if [[ -f "$PREFILL_2048_PROMPT" ]]; then
  echo
  echo "--- Gate 3/3: prefill perf on ≥2048-token prompt (batched) ---"
  PREFILL_LOG="/tmp/release_check_prefill.log"
  if ! HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_BATCHED_PREFILL=1 \
      "$HF2Q_BIN" generate --model "$GGUF_PATH" --prompt-file "$PREFILL_2048_PROMPT" \
        --max-tokens 1 --temperature 0 \
        >/dev/null 2>"$PREFILL_LOG"; then
    echo "prefill sanity run crashed; see $PREFILL_LOG" >&2
    exit 3
  fi
  # hf2q emits the prefill summary to stdout but also logs a stderr line
  # "Batched prefill complete: N tokens in X.X ms (Y.Y tok/s)" at the end
  # of the batched prefill session. We parse the stderr line to avoid
  # needing stdout capture (which would mix decoded text with timing).
  PREFILL_TPS="$(grep -oE 'Batched prefill complete: [0-9]+ tokens in [0-9.]+ ms \(([0-9.]+) tok/s\)' "$PREFILL_LOG" \
      | tail -1 | grep -oE '\([0-9.]+ tok/s\)' | grep -oE '[0-9.]+' | head -1)"
  if [[ -z "$PREFILL_TPS" ]]; then
    echo "FAIL: no prefill tok/s line found in $PREFILL_LOG — format may have changed." >&2
    exit 2
  fi
  PASS_PREFILL="$(awk -v t="$PREFILL_TPS" -v m="$MIN_PREFILL_TPS" 'BEGIN { print (t+0 >= m+0) ? "1" : "0" }')"
  echo "prefill: $PREFILL_TPS tok/s on 2455-token prompt (floor: $MIN_PREFILL_TPS)"
  if [[ "$PASS_PREFILL" != "1" ]]; then
    echo "FAIL: batched prefill $PREFILL_TPS tok/s is below floor $MIN_PREFILL_TPS." >&2
    echo "      Bisect recent changes to fused_head_norm_rope, sdpa_sliding," >&2
    echo "      or the batched prefill session structure." >&2
    exit 2
  fi
else
  echo
  echo "(Gate 3 skipped: $PREFILL_2048_PROMPT not found)" >&2
fi

echo
echo "=== release-check PASS — all gates green ==="
exit 0
