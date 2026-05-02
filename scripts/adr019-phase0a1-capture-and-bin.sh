#!/bin/bash
# ADR-019 Phase 0a.1 — xctrace residual attribution.
#
# Purpose: bin the 530 ms `wall - GPU` residual at chunk-engaged pp4096
# into four classes:
#   1. CPU encoder-build (per-CB)
#   2. Driver commit overhead (per-CB)
#   3. Inter-CB GPU pipeline-bubble
#   4. Residency-set add/remove churn
#
# Acceptance: if any single class >= 70% of residual, that class is the
# primary D3 lever. If uniformly distributed, D4 microprototype goes
# load-bearing.
#
# Cross-check: Phase 0a.3 H3 floor = 13.3 us/CB driver commit;
# at ~96 CBs that bin must sum to ~1.3 ms (= 0.25% of 530 ms). Materially
# larger driver-commit numbers from xctrace mean window-cut is wrong.
#
# Phase 0a.2 + 0a.3 receipts:
#   /tmp/cfa-adr019-phase0a3/results.md
#   /tmp/cfa-adr019-phase0a2-research/design.md
#
# Methodology constraints (from MEMORY.md):
#   - Pre-bench process audit (mandatory; 0a.3 runs 1-3 contaminated)
#   - Cold-first per perf-gate methodology
#   - Measure 3x cut once (3 trials min, 5 ideal)
#
# Author: ADR-019 Phase 0a.1 implementation, 2026-05-02.

set -euo pipefail

LOGDIR=/tmp/cfa-adr019-phase0a1
COOLDOWN_S=30
N_TRIALS=${N_TRIALS:-3}

MODEL_DIR=/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat
HF2Q=/opt/hf2q/target/release/hf2q
PROMPT=/tmp/walkbar-pp4096-prompt.txt

# Pre-flight: model + binary + fixture must exist.
if [ ! -x "$HF2Q" ]; then echo "ERROR: hf2q binary missing at $HF2Q" >&2; exit 1; fi
if [ ! -f "$MODEL_DIR/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat.gguf" ]; then
  echo "ERROR: model gguf missing under $MODEL_DIR" >&2; exit 1
fi
if [ ! -f "$PROMPT" ]; then echo "ERROR: prompt fixture missing at $PROMPT" >&2; exit 1; fi
if ! command -v xctrace >/dev/null 2>&1; then echo "ERROR: xctrace not on PATH" >&2; exit 1; fi

mkdir -p "$LOGDIR"

# Pre-flight: process audit (memory/feedback_bench_process_audit.md).
# Note: `head -10` SIGPIPE-kills the upstream pipe. Use process substitution
# to avoid pipefail abort.
echo "=== Pre-bench process audit at $(date '+%H:%M:%S') ===" | tee "$LOGDIR/process-audit.txt"
ps aux > "$LOGDIR/.ps-tmp" 2>/dev/null || true
sort -k 3 -nr "$LOGDIR/.ps-tmp" 2>/dev/null | awk 'NR<=10' | tee -a "$LOGDIR/process-audit.txt"
rm -f "$LOGDIR/.ps-tmp"

run_xctrace() {
    local label="$1"
    local trace_out="$LOGDIR/${label}.trace"
    local stdout_log="$LOGDIR/${label}.log"

    # remove any leftover trace dir from prior trial (xctrace refuses to overwrite)
    rm -rf "$trace_out"

    echo "--- xctrace record trial: $label at $(date '+%H:%M:%S') ---"
    /usr/bin/xctrace record \
        --template "Metal System Trace" \
        --no-prompt \
        --output "$trace_out" \
        --env HF2Q_PROFILE_GPU_TS=1 \
        --env HF2Q_PROFILE_W5B8=1 \
        --env HF2Q_QWEN36_AUTOREG=1 \
        --env HF2Q_UNSAFE_EXPERIMENTS=1 \
        --env HF2Q_CHUNK_SCAN_PREFILL=1 \
        --target-stdout "$stdout_log" \
        --launch -- "$HF2Q" generate \
            --model "$MODEL_DIR/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat.gguf" \
            --tokenizer "$MODEL_DIR/tokenizer.json" \
            --config "$MODEL_DIR/config.json" \
            --prompt-file "$PROMPT" \
            --max-tokens 4 \
            --temperature 0.0 \
        > "${stdout_log}.xctrace.log" 2>&1
    local rc=$?
    echo "  xctrace exit=$rc"
    if [ -f "$stdout_log" ]; then
        grep "^prefill:" "$stdout_log" | tail -1 || echo "  (no prefill: line in stdout)"
    else
        echo "  WARN: target stdout $stdout_log not produced"
    fi
    if [ ! -d "$trace_out" ] && [ ! -f "$trace_out" ]; then
        echo "  WARN: trace output $trace_out did not materialize"
    fi
}

# Run trials
for i in $(seq 1 "$N_TRIALS"); do
    run_xctrace "trial${i}"
    if [ "$i" -lt "$N_TRIALS" ]; then
        sleep $COOLDOWN_S
    fi
done

echo "=== ALL DONE at $(date '+%H:%M:%S') ==="
ls -la "$LOGDIR"
