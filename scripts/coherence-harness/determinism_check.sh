#!/bin/bash
# determinism_check.sh — 5× cold-process byte-equality check on the
# first-token logit dump. Catches forward-pass Heisenbugs (the class that
# slipped through ADR-019 phase-1's AC-PA2 5× check on a different fixture
# and shipped the missing memory_barrier in `apply_output_head_gpu_into`,
# fixed at hf2q@f591a66 2026-05-03).
#
# A passing run prints:
#   DETERMINISM: PASS — 5/5 cold runs produced byte-identical top-3
# A failing run prints each run's top-3 line and exits non-zero, so the
# operator can see whether the argmax flips, the logit values drift, or
# both.
#
# Usage: ./determinism_check.sh <gguf> "<prompt>" [thinking|no-thinking]
set -euo pipefail
MODEL="${1:?model gguf required}"
PROMPT="${2:?prompt required}"
MODE="${3:-thinking}"

HF2Q="${HF2Q:-/opt/hf2q/target/release/hf2q}"

case "$MODE" in
  thinking) FLAG="--enable-thinking" ;;
  no-thinking) FLAG="--no-thinking" ;;
  *) echo "mode must be thinking|no-thinking" >&2; exit 2 ;;
esac

WORKDIR="$(mktemp -d)"
trap "rm -rf '$WORKDIR'" EXIT

declare -a RUNS
for i in 1 2 3 4 5; do
  out="$WORKDIR/run-$i.txt"
  HF2Q_DUMP_LOGITS=1 "$HF2Q" generate --model "$MODEL" --prompt "$PROMPT" \
    --max-tokens 1 --temperature 0 $FLAG \
    >/dev/null 2>"$out" || true
  line=$(grep "top-3" "$out" | head -1 | sed 's/^[[:space:]]*//')
  RUNS+=("$line")
  echo "  run $i: $line"
done

# All runs equal?
all_equal=1
for ((i=1; i<5; i++)); do
  if [ "${RUNS[i]}" != "${RUNS[0]}" ]; then
    all_equal=0
    break
  fi
done

if [ "$all_equal" = "1" ]; then
  echo "DETERMINISM: PASS — 5/5 cold runs produced byte-identical top-3"
  exit 0
else
  echo "DETERMINISM: FAIL — at least two runs differ. See per-run top-3 above."
  exit 2
fi
