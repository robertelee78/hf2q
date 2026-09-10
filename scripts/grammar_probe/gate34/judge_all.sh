#!/bin/sh
set -x
for p in baseline 0.5 1.0 2.0 4.0; do
  RESULTS="scripts/grammar_probe/gate34/$p/results.jsonl" \
  OUT="scripts/grammar_probe/gate34/$p/verdicts.jsonl" \
  BASE_URL="http://127.0.0.1:8081" \
  JUDGE_MODEL="qwen36-abliterix-t63-APEX" \
  python3 scripts/grammar_probe/gate34/judge_driver.py
done
echo "JUDGE_ALL_DONE"
