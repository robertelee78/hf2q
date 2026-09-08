#!/bin/bash
# supervise judge.py: restart until all rows judged (idempotent, skips done)
RESULTS_FILE="$1"; OUT_FILE="$2"
TOTAL=$(jq -r 'select(.error | not)' "$RESULTS_FILE" | grep -c .)
for i in $(seq 1 40); do
  DONE_N=$(grep -c . "$OUT_FILE" 2>/dev/null || echo 0)
  if [ "$DONE_N" -ge "$TOTAL" ]; then echo "ALL JUDGED ($DONE_N/$TOTAL)"; exit 0; fi
  JUDGE_MODEL="qwen36-abliterix-t63-APEX" RESULTS="$RESULTS_FILE" OUT="$OUT_FILE" python3 judge.py >> /tmp/judge_supervised.log 2>&1
  echo "pass $i: $DONE_N/$TOTAL judged" >> /tmp/judge_supervised.log
  sleep 2
done
echo "GAVE UP after 40 passes" >> /tmp/judge_supervised.log
