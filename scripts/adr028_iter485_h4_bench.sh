#!/usr/bin/env bash
# ADR-028 iter-485 (Phase 7d / H4) — A/B bench for fused 4-bit K+V TQ encoder.
#
# Compares HEAD baseline (2-dispatch K then V encode) vs
# HF2Q_TQ_FAST_FUSED_KV=1 (single Z-dim-split dispatch).
#
# Expected: ~30 KV-write dispatches/decode-token saved at gemma4 30L
# (~14 µs each = ~0.4 ms/tok; ~3% theoretical). Ship gate: +3% median.
#
# Usage:
#   bash scripts/adr028_iter485_h4_bench.sh [PROMPT] [MAX_TOKENS] [N_RUNS]
set -euo pipefail

MODEL="${HF2Q_MODEL:-/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf}"
PROMPT="${1:-Write a long story about a sentient telescope}"
MAX_TOKENS="${2:-200}"
N_RUNS="${3:-5}"
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"

if [[ ! -x "$HF2Q_BIN" ]]; then
    echo "[fatal] hf2q binary not found at $HF2Q_BIN" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "[fatal] model not found at $MODEL" >&2
    exit 1
fi

run_stack() {
    local label="$1"; shift
    local env_vars="$*"
    echo "=== $label ==="
    for (( i=1; i<=N_RUNS; i++ )); do
        env $env_vars "$HF2Q_BIN" generate \
            --model "$MODEL" \
            --prompt "$PROMPT" \
            --max-tokens "$MAX_TOKENS" 2>&1 \
            | grep -E "tokens in|tok/s ---" \
            | tail -1
    done
}

echo "ADR-028 iter-485 H4 — fused 4-bit K+V TQ encoder A/B"
echo "  Model: $(basename "$MODEL")"
echo "  Prompt: $PROMPT"
echo "  Max tokens: $MAX_TOKENS"
echo "  Runs/stack: $N_RUNS"
echo "  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

run_stack "OFF (HEAD baseline, 2-dispatch K+V)" ""
run_stack "ON  (HF2Q_TQ_FAST_FUSED_KV=1, fused dual)" "HF2Q_TQ_FAST_FUSED_KV=1"
