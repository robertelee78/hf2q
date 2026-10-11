#!/bin/sh
# Judge every pass of one gate34 run dir against the APEX judge.
# Usage: judge_all.sh <run_dir> <pass_label> [<pass_label> ...]
# Env: BASE_URL (default http://127.0.0.1:8081), JUDGE_MODEL
# (default qwen36-abliterix-t63-APEX). The judge server must already be
# listening; start/stop it outside this script (one runtime at a time).
set -x
RUN_DIR="$1"
shift
for p in "$@"; do
  RESULTS="$RUN_DIR/$p/results.jsonl" \
  OUT="$RUN_DIR/$p/verdicts.jsonl" \
  BASE_URL="${BASE_URL:-http://127.0.0.1:8081}" \
  JUDGE_MODEL="${JUDGE_MODEL:-qwen36-abliterix-t63-APEX}" \
  python3 "$(dirname "$0")/judge_driver.py"
done
echo "JUDGE_ALL_DONE"
