#!/bin/bash
# overnight-pipeline.sh — drives ALL remaining measurement work to completion.
# Resume-safe at every stage; one large model resident at a time; every
# server stopped at the end. Log: /tmp/overnight-pipeline.log
set -u
PROBE=/opt/hf2q/scripts/grammar_probe
LOG=/tmp/overnight-pipeline.log
JUDGE_MODEL=qwen36-abliterix-t63-APEX
cd "$PROBE"

log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

# --- Stage 1: wait for the in-flight slice runner (LIMIT=512) to exit ---
while pgrep -f "baseline_run.py" > /dev/null; do sleep 60; done
log "stage 1: slice runner exited at $(wc -l < paired_w1_baseline_slice.jsonl) rows"

# --- Stage 2: continue the SAME file to the full 4096-row run ---
# (LIMIT is not part of the config hash; resume skips the done rows)
BASE_URL=http://127.0.0.1:18085 \
SERVER_IDENTITY="hf2q main 294907cd, DeepSeek-V4-Flash-0731-agentic-q2, started WITHOUT --gcd/--glp" \
BUDGET_LADDER=800,1600 OUT=paired_w1_baseline_slice.jsonl \
python3 baseline_run.py >> "$LOG" 2>&1
log "stage 2: full paired run complete at $(wc -l < paired_w1_baseline_slice.jsonl) rows"

# --- Stage 3: stop DeepSeek; verify dead before the judge loads ---
P=$(lsof -ti :18085 -sTCP:LISTEN); [ -n "$P" ] && kill -9 "$P"
sleep 10
if pgrep -x hf2q >/dev/null; then log "FATAL: hf2q still alive after DeepSeek stop"; exit 1; fi
log "stage 3: DeepSeek stopped"

# --- Stage 4: judge up ---
cd /opt/hf2q
nohup ./target/release/hf2q serve --model /opt/hf2q/models/qwen3.6/APEX-Q5_K_M.gguf --port 8081 > /tmp/judge-overnight.log 2>&1 &
for i in $(seq 1 90); do curl -s -o /dev/null http://127.0.0.1:8081/v1/models && break; sleep 10; done
sleep 20
cd "$PROBE"
log "stage 4: judge up"

# --- Stage 5: concurrent judge pool (independent processes, distinct outputs;
#     the judge server interleaves them across its slots) ---
python3 rejudge.py --results full_results_w1.jsonl --old-verdicts full_verdicts_w1.jsonl \
  --label w1_deepseek --base-url http://127.0.0.1:8081 --judge-model "$JUDGE_MODEL" \
  --sample 40 --sample-seed 7 > /tmp/rj-ds.log 2>&1 &
python3 rejudge.py --results spike_results_w1_gemma.jsonl --old-verdicts spike_verdicts_w1_gemma.jsonl \
  --label w1_gemma --base-url http://127.0.0.1:8081 --judge-model "$JUDGE_MODEL" \
  --sample 40 --sample-seed 7 > /tmp/rj-gemma.log 2>&1 &
python3 rejudge.py --results spike_results_w1_qwen38.jsonl --old-verdicts spike_verdicts_w1_qwen38.jsonl \
  --label w1_qwen38 --base-url http://127.0.0.1:8081 --judge-model "$JUDGE_MODEL" \
  --sample 40 --sample-seed 7 > /tmp/rj-qwen38.log 2>&1 &
PROMPTS=prompts_512.tsv RESULTS=paired_w1_baseline_slice.jsonl \
  OUT=paired_w1_baseline_slice_verdicts.jsonl BASE_URL=http://127.0.0.1:8081 \
  JUDGE_MODEL="$JUDGE_MODEL" python3 judge.py > /tmp/judge-paired.log 2>&1 &
wait
log "stage 5: judge pool complete"

# --- Stage 6: stop the judge; verify dead ---
P=$(lsof -ti :8081 -sTCP:LISTEN); [ -n "$P" ] && kill -9 "$P"
sleep 5
if pgrep -x hf2q >/dev/null; then log "FATAL: hf2q still alive after judge stop"; exit 1; fi
log "stage 6: judge stopped"

# --- Stage 7: reports ---
CONTROL=BASE VERDICTS=paired_w1_baseline_slice_verdicts.jsonl \
  RESULTS=paired_w1_baseline_slice.jsonl \
  REPORT_JSON=paired_w1_baseline_slice_report.json \
  python3 report.py > /tmp/report-paired.log 2>&1
log "stage 7: paired report written (report exit $?)"
for L in w1_deepseek w1_gemma w1_qwen38; do
  log "rejudge $L: $(wc -l < rejudge_v2_$L/verdicts_v2.jsonl 2>/dev/null || echo 0) verdicts"
done
log "PIPELINE COMPLETE — all servers stopped"
