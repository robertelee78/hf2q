#!/usr/bin/env bash
# ADR-015 iter61a-3: per-op bisection driver.
#
# Runs two cold processes with HF2Q_DUMP_LAYER=ALL, on identical inputs,
# then byte-diffs every dumped step×layer×op file to find the earliest
# divergence.  Output: sorted DIFF list to stdout.
set -euo pipefail

WORKTREE="${WORKTREE:-/opt/hf2q/.cfa-worktrees/adr015-iter61a-3-bisection-scaffold}"
BIN="${WORKTREE}/target/release/hf2q"
MODEL="${MODEL:-/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf}"
PROMPT="${PROMPT:-Hello}"
MAX_TOKENS="${MAX_TOKENS:-2}"

DUMP_ROOT=/tmp/hf2q-dump
rm -rf "${DUMP_ROOT}/run1" "${DUMP_ROOT}/run2"
mkdir -p "${DUMP_ROOT}"

echo "[bisect] running cold-process run1..."
HF2Q_DUMP_LAYER=ALL HF2Q_DUMP_RUN_ID=run1 \
  "${BIN}" generate \
  --model "${MODEL}" \
  --prompt "${PROMPT}" \
  --temperature 0 \
  --top-k 1 \
  --max-tokens "${MAX_TOKENS}" \
  > "${DUMP_ROOT}/run1.stdout" 2> "${DUMP_ROOT}/run1.stderr" || {
    echo "run1 failed; tail of stderr:"
    tail -40 "${DUMP_ROOT}/run1.stderr"
    exit 1
  }

echo "[bisect] running cold-process run2..."
HF2Q_DUMP_LAYER=ALL HF2Q_DUMP_RUN_ID=run2 \
  "${BIN}" generate \
  --model "${MODEL}" \
  --prompt "${PROMPT}" \
  --temperature 0 \
  --top-k 1 \
  --max-tokens "${MAX_TOKENS}" \
  > "${DUMP_ROOT}/run2.stdout" 2> "${DUMP_ROOT}/run2.stderr" || {
    echo "run2 failed; tail of stderr:"
    tail -40 "${DUMP_ROOT}/run2.stderr"
    exit 1
  }

echo "[bisect] generated tokens (run1):"
grep -E "Generated|tok" "${DUMP_ROOT}/run1.stdout" | head -5 || true
cat "${DUMP_ROOT}/run1.stdout" | head -3
echo "---"
echo "[bisect] generated tokens (run2):"
cat "${DUMP_ROOT}/run2.stdout" | head -3

echo "[bisect] comparing dump files..."
ls "${DUMP_ROOT}/run1"/*.f32 2>/dev/null | wc -l | xargs -I{} echo "run1 dumps: {} files"
ls "${DUMP_ROOT}/run2"/*.f32 2>/dev/null | wc -l | xargs -I{} echo "run2 dumps: {} files"

echo "[bisect] DIFF report (sorted by step,layer,op):"
diffs=()
for f in "${DUMP_ROOT}/run1"/*.f32; do
  name=$(basename "$f")
  if [ ! -f "${DUMP_ROOT}/run2/$name" ]; then
    diffs+=("MISSING_RUN2: $name")
    continue
  fi
  if ! cmp -s "$f" "${DUMP_ROOT}/run2/$name"; then
    # Count first differing byte offset
    first_diff=$(cmp "$f" "${DUMP_ROOT}/run2/$name" 2>/dev/null | head -1 || echo "")
    diffs+=("DIFF: $name | $first_diff")
  fi
done

if [ ${#diffs[@]} -eq 0 ]; then
  echo "[bisect] NO DIFFERENCES — runs are byte-identical."
else
  printf '%s\n' "${diffs[@]}" | sort
  echo "[bisect] First-diverging op:"
  printf '%s\n' "${diffs[@]}" | sort | head -1
fi
