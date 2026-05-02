#!/bin/bash
# ADR-019 Phase 0a.1 — cross-protocol llama.cpp companion capture.
#
# Purpose: capture llama.cpp on the SAME apex Q4_0-flat pp4096 fixture
# via xctrace Metal System Trace, then bin into the same four classes
# that 0a.1 (hf2q-side, commit 42dd6c6) used. Localizes the 305-616 ms
# peer gap into one of:
#   (i)   llama-bench wrap (model-load + warmup) — would show llama.cpp
#         in-process wall ≈ hf2q in-process wall (≤ 50 ms delta).
#   (ii)  GPU-active kernel-level difference — would show llama.cpp wall
#         materially less AND GPU-active dominant in the delta.
#   (iii) Post-iter88a perf landings — gap already absorbed.
#
# Companion to /opt/hf2q/scripts/adr019-phase0a1-capture-and-bin.sh
# (hf2q-side; do not modify that script — it's already committed at 42dd6c6).
# This script targets `llama-completion` instead of `hf2q generate`, but
# uses the exact same xctrace template + xpath bin queries.
#
# Methodology constraints (mirrored from hf2q-side script):
#   - Pre-bench process audit (mandatory)
#   - Cold-first per perf-gate methodology
#   - Measure 3x cut once (3 trials min)
#
# Author: ADR-019 Phase 0a.1 cross-protocol capture, 2026-05-02.

set -euo pipefail

LOGDIR=/tmp/cfa-adr019-phase0a1-llamacpp
COOLDOWN_S=30
N_TRIALS=${N_TRIALS:-3}

# Use homebrew llama-completion (8680, 15f786e65). The /opt/llama.cpp/build
# variant has a stale dyld link to libllama 0.0.8999 (symbol mismatch on
# llama_memory_breakdown_print). Homebrew is the binary the existing hf2q
# bench scripts (bench-w5b11-post-attn.sh etc.) already use; cross-comparable.
LLAMA=/opt/homebrew/bin/llama-completion

MODEL_DIR=/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat
MODEL_GGUF="$MODEL_DIR/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat.gguf"
PROMPT=/tmp/walkbar-pp4096-prompt.txt

# Pre-flight: binary + model + fixture must exist.
if [ ! -x "$LLAMA" ]; then echo "ERROR: llama-completion missing at $LLAMA" >&2; exit 1; fi
if [ ! -f "$MODEL_GGUF" ]; then echo "ERROR: model gguf missing at $MODEL_GGUF" >&2; exit 1; fi
if [ ! -f "$PROMPT" ]; then echo "ERROR: prompt fixture missing at $PROMPT" >&2; exit 1; fi
if ! command -v xctrace >/dev/null 2>&1; then echo "ERROR: xctrace not on PATH" >&2; exit 1; fi

mkdir -p "$LOGDIR"

# Pre-flight: process audit.
echo "=== Pre-bench process audit at $(date '+%H:%M:%S') ===" | tee "$LOGDIR/process-audit.txt"
ps aux > "$LOGDIR/.ps-tmp" 2>/dev/null || true
sort -k 3 -nr "$LOGDIR/.ps-tmp" 2>/dev/null | awk 'NR<=10' | tee -a "$LOGDIR/process-audit.txt"
rm -f "$LOGDIR/.ps-tmp"

echo "=== llama-completion version ==="
"$LLAMA" --version 2>&1 | grep -E "^version:|^built" | tee -a "$LOGDIR/process-audit.txt" || true

# Match the canonical hf2q chunk-engaged bench invocation
# (scripts/bench-w5b11-post-attn.sh:51-56) — the production-comparable args.
# Add --temp 0 --top-k 1 --seed 0 for deterministic decode (the 1 token after
# prefill); --perf for prompt eval timing in stderr.
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

# (1) Wall-time baseline WITHOUT xctrace (3 cold trials).
# This anchors the unperturbed llama.cpp in-process wall, comparable to
# ADR-019's 1233 ms peer-baseline number AND to 0a.1's no-xctrace baseline
# (1534 ± 13 ms hf2q).
echo "=== Wall-time baseline (no xctrace) — 3 trials ==="
for i in $(seq 1 "$N_TRIALS"); do
    OUTLOG="$LOGDIR/baseline-trial${i}.log"
    echo "--- baseline trial ${i} at $(date '+%H:%M:%S') ---"
    /usr/bin/time -p "$LLAMA" "${LLAMA_ARGS[@]}" > "$OUTLOG" 2>&1 || true
    # llama --perf prints to stderr (captured along with stdout via 2>&1).
    PROMPT_EVAL_MS=$(grep -oE 'prompt eval time = +[0-9.]+ ms' "$OUTLOG" \
        | sed -E 's/.*= +([0-9.]+) ms/\1/' | head -1)
    REAL_S=$(awk '/^real/ {print $2}' "$OUTLOG")
    printf '[T%d] prompt_eval=%s ms | real=%s s\n' "$i" "${PROMPT_EVAL_MS:-???}" "${REAL_S:-???}"
    if [ "$i" -lt "$N_TRIALS" ]; then sleep "$COOLDOWN_S"; fi
done
echo

# (2) xctrace-on (3 cold trials; same args).
run_xctrace() {
    local label="$1"
    local trace_out="$LOGDIR/${label}.trace"
    local stdout_log="$LOGDIR/${label}.log"

    rm -rf "$trace_out"

    echo "--- xctrace record trial: $label at $(date '+%H:%M:%S') ---"
    /usr/bin/xctrace record \
        --template "Metal System Trace" \
        --no-prompt \
        --output "$trace_out" \
        --target-stdout "$stdout_log" \
        --launch -- "$LLAMA" "${LLAMA_ARGS[@]}" \
        > "${stdout_log}.xctrace.log" 2>&1
    local rc=$?
    echo "  xctrace exit=$rc"
    if [ -f "$stdout_log" ]; then
        grep -E "prompt eval time|tokens generated|^real" "$stdout_log" | head -5 \
            || echo "  (no llama.cpp timing line in stdout)"
    else
        echo "  WARN: target stdout $stdout_log not produced"
    fi
    if [ ! -d "$trace_out" ] && [ ! -f "$trace_out" ]; then
        echo "  WARN: trace output $trace_out did not materialize"
    fi
}

echo "=== xctrace Metal System Trace — $N_TRIALS cold trials ==="
for i in $(seq 1 "$N_TRIALS"); do
    run_xctrace "trial${i}"
    if [ "$i" -lt "$N_TRIALS" ]; then
        sleep "$COOLDOWN_S"
    fi
done

echo "=== ALL DONE at $(date '+%H:%M:%S') ==="
ls -la "$LOGDIR" | head -40
