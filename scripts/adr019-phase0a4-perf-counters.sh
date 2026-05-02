#!/bin/bash
# ADR-019 Phase 0a.4 — Metal performance-counter attribution.
#
# Purpose: localize the 352 ms in-process prefill delta (hf2q 1538 ms vs
# llama.cpp 1186 ms) measured by Phase 0a.1 cross-protocol on apex
# Q4_0-flat pp4096. Three competing mechanism hypotheses:
#
#   M1 — Per-CB GPU-side dispatch cost (kernel launch overhead, scheduler
#        reset, cache flush per CB-boundary). 235 CBs × ~1.5 ms/CB ≈ 352 ms.
#        D3 (CB consolidation) directly captures this.
#   M2 — Per-CB occupancy gap (wavefront ramp-up/drain at CB boundaries).
#        D3 also captures this.
#   M3 — Per-DISPATCH overhead (not per-CB). If hf2q has 235× more
#        *dispatches* than llama.cpp (not just 235× more CBs), CB
#        consolidation does not help; D3 has no recovery path.
#
# H-0a.4-3 (the only directly testable hypothesis from this seat):
#   compare per-side dispatch count via:
#     - hf2q internal: HF2Q_PROFILE_SYNC=1 → "[P19 H9] ... dispatch_count=" stderr
#     - both sides external: count metal-gpu-execution-points event pairs
#       (each pair = one GPU sub-work item; see 0a.1 trace schema)
#
# H-0a.4-1 (ALU active / total cycles) and H-0a.4-2 (memory stall counters)
# are NOT testable from this seat. The xctrace 16.0 CLI cannot select a
# Metal Counter Set on M5 Max — the GUI Counter Set picker must be driven
# from Xcode Instruments app. The "Metal GPU Counters" instrument added
# via `xctrace --instrument` returns "Selected counter profile is not
# supported on target device" because no profile is selected. Game
# Performance template captures only one counter (RT Unit Active) by
# default, irrelevant to ALU/stall/occupancy. This script DOCUMENTS that
# limitation; it does NOT fabricate counter values.
#
# Companion to /opt/hf2q/scripts/adr019-phase0a1-{capture-and-bin,llamacpp-capture}.sh
# (do not modify those — committed at 42dd6c6 / 7af36ac / 21ddcab).
#
# Methodology (mirrored from 0a.1):
#   - Pre-bench process audit (mandatory)
#   - Cold-first per perf-gate methodology
#   - Measure 3x cut once (3 trials min)
#
# Author: ADR-019 Phase 0a.4 implementation, 2026-05-02 evening.

set -euo pipefail

LOGDIR=/tmp/cfa-adr019-phase0a4
COOLDOWN_S=30
N_TRIALS=${N_TRIALS:-3}

MODEL_DIR=/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat
MODEL_GGUF="$MODEL_DIR/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat.gguf"
HF2Q=/opt/hf2q/target/release/hf2q
LLAMA=/opt/homebrew/bin/llama-completion
PROMPT=/tmp/walkbar-pp4096-prompt.txt

# Pre-flight.
[ -x "$HF2Q" ]  || { echo "ERROR: hf2q binary missing at $HF2Q" >&2; exit 1; }
[ -x "$LLAMA" ] || { echo "ERROR: llama-completion missing at $LLAMA" >&2; exit 1; }
[ -f "$MODEL_GGUF" ] || { echo "ERROR: model gguf missing at $MODEL_GGUF" >&2; exit 1; }
[ -f "$PROMPT" ] || { echo "ERROR: prompt fixture missing at $PROMPT" >&2; exit 1; }
command -v xctrace >/dev/null 2>&1 || { echo "ERROR: xctrace not on PATH" >&2; exit 1; }

mkdir -p "$LOGDIR"

# Pre-flight process audit (memory/feedback_bench_process_audit.md).
echo "=== Pre-bench process audit at $(date '+%H:%M:%S') ===" | tee "$LOGDIR/process-audit.txt"
ps aux > "$LOGDIR/.ps-tmp" 2>/dev/null || true
sort -k 3 -nr "$LOGDIR/.ps-tmp" 2>/dev/null | awk 'NR<=10' | tee -a "$LOGDIR/process-audit.txt"
rm -f "$LOGDIR/.ps-tmp"

LLAMA_ARGS=(
    --model "$MODEL_GGUF"
    -f "$PROMPT"
    --batch-size 4096
    --ubatch-size 4096
    --n-predict 1
    --no-warmup
    -no-cnv
    --perf
    --temp 0
    --top-k 1
    --seed 0
)

# (1) hf2q with HF2Q_PROFILE_SYNC=1 — gives [P19 H9] line with dispatch_count.
#     Run UNDER xctrace Metal System Trace so we also get
#     metal-gpu-execution-points for cross-validation.
run_hf2q_trial() {
    local label="$1"
    local trace_out="$LOGDIR/hf2q-${label}.trace"
    local stdout_log="$LOGDIR/hf2q-${label}.stdout"
    local stderr_log="$LOGDIR/hf2q-${label}.stderr"

    rm -rf "$trace_out"

    echo "--- hf2q trial: $label at $(date '+%H:%M:%S') ---"
    /usr/bin/xctrace record \
        --template "Metal System Trace" \
        --no-prompt \
        --output "$trace_out" \
        --env HF2Q_PROFILE_GPU_TS=1 \
        --env HF2Q_PROFILE_W5B8=1 \
        --env HF2Q_QWEN36_AUTOREG=1 \
        --env HF2Q_UNSAFE_EXPERIMENTS=1 \
        --env HF2Q_CHUNK_SCAN_PREFILL=1 \
        --env HF2Q_PROFILE_SYNC=1 \
        --target-stdout "$stdout_log" \
        --launch -- "$HF2Q" generate \
            --model "$MODEL_GGUF" \
            --tokenizer "$MODEL_DIR/tokenizer.json" \
            --config "$MODEL_DIR/config.json" \
            --prompt-file "$PROMPT" \
            --max-tokens 4 \
            --temperature 0.0 \
        > "$stderr_log" 2>&1
    local rc=$?
    echo "  xctrace exit=$rc"
    if [ -f "$stdout_log" ]; then
        grep "^prefill:" "$stdout_log" | tail -1 || echo "  (no prefill: line)"
    fi
    if [ -f "$stderr_log" ]; then
        grep "P19 H9" "$stderr_log" | tail -1 || echo "  (no P19 H9 line)"
    fi
}

# (2) llama.cpp under same template — get metal-gpu-execution-points + --perf.
run_llama_trial() {
    local label="$1"
    local trace_out="$LOGDIR/llama-${label}.trace"
    local stdout_log="$LOGDIR/llama-${label}.stdout"
    local stderr_log="$LOGDIR/llama-${label}.stderr"

    rm -rf "$trace_out"

    echo "--- llama trial: $label at $(date '+%H:%M:%S') ---"
    /usr/bin/xctrace record \
        --template "Metal System Trace" \
        --no-prompt \
        --output "$trace_out" \
        --target-stdout "$stdout_log" \
        --launch -- "$LLAMA" "${LLAMA_ARGS[@]}" \
        > "$stderr_log" 2>&1
    local rc=$?
    echo "  xctrace exit=$rc"
    if [ -f "$stdout_log" ]; then
        grep -E "prompt eval|llama_perf" "$stdout_log" | head -3 || echo "  (no perf lines)"
    fi
}

echo
echo "=== hf2q xctrace+P19-H9 — $N_TRIALS cold trials ==="
for i in $(seq 1 "$N_TRIALS"); do
    run_hf2q_trial "trial${i}"
    if [ "$i" -lt "$N_TRIALS" ]; then sleep "$COOLDOWN_S"; fi
done

echo
echo "=== llama.cpp xctrace — $N_TRIALS cold trials ==="
for i in $(seq 1 "$N_TRIALS"); do
    run_llama_trial "trial${i}"
    if [ "$i" -lt "$N_TRIALS" ]; then sleep "$COOLDOWN_S"; fi
done

# (3) Export metal-gpu-execution-points for every trial so we can count
#     dispatches per side independently of stderr instrumentation.
echo
echo "=== Exporting metal-gpu-execution-points per trial ==="
for side in hf2q llama; do
    for i in $(seq 1 "$N_TRIALS"); do
        in="$LOGDIR/${side}-trial${i}.trace"
        out="$LOGDIR/${side}-trial${i}.exec-points.xml"
        if [ -d "$in" ] || [ -f "$in" ]; then
            /usr/bin/xctrace export --input "$in" \
                --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-execution-points"]' \
                --output "$out" 2>&1 | tail -1 || echo "  export $side trial$i failed"
            wc -l "$out" 2>&1 | head -1
        else
            echo "  WARN: $in missing"
        fi
    done
done

# (4) Also export cb-submissions to confirm CB count parity vs 0a.1.
echo
echo "=== Exporting metal-application-command-buffer-submissions ==="
for side in hf2q llama; do
    for i in $(seq 1 "$N_TRIALS"); do
        in="$LOGDIR/${side}-trial${i}.trace"
        out="$LOGDIR/${side}-trial${i}.cb-submissions.xml"
        if [ -d "$in" ] || [ -f "$in" ]; then
            /usr/bin/xctrace export --input "$in" \
                --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-application-command-buffer-submissions"]' \
                --output "$out" 2>&1 | tail -1 || echo "  export $side trial$i failed"
        fi
    done
done

echo
echo "=== ALL DONE at $(date '+%H:%M:%S') ==="
ls -la "$LOGDIR" | head -40
