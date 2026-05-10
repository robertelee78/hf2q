#!/usr/bin/env bash
# ADR-028 iter-231 — full-stack regression bench script.
# Single-command operator-runner that produces the standard 4-stack
# comparison table at gemma4 + peer baseline. Useful for verifying
# iter-216 / iter-221 measurements remain stable post-future-changes.
#
# Usage:
#   bash scripts/adr028_full_stack_bench.sh [PROMPT] [MAX_TOKENS] [N_RUNS]
#
# Defaults: long-form telescope prompt, 200 tokens, 3 runs per stack.
#
# Output: comparison table (default / Path E / Path E+G / Path E+F+G /
# llama.cpp peer) with median tok/s + std-dev hint.
#
# Requires: gemma-4-26B-A4B-it-DFlash-not-required model file at
#   /opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf
# AND llama-bench at /opt/llama.cpp/build/bin/llama-bench (optional).

set -euo pipefail

MODEL="${HF2Q_MODEL:-/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf}"
PROMPT="${1:-Write a long story about a sentient telescope}"
# iter-233: default raised from 200 to 1000 tokens; HF2Q_F16_KV=1 produces
# degraded output ("<pad>") at long context but appears coherent at 200-tok.
# Long-form runs are required to catch this class of regression.
MAX_TOKENS="${2:-1000}"
N_RUNS="${3:-3}"
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"
LLAMA_BENCH="${LLAMA_BENCH:-/opt/llama.cpp/build/bin/llama-bench}"

if [[ ! -x "$HF2Q_BIN" ]]; then
    echo "[fatal] hf2q binary not found at $HF2Q_BIN" >&2
    echo "[fatal] run: cd /opt/hf2q && cargo build --release" >&2
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

echo "ADR-028 full-stack regression bench"
echo "  Model: $(basename "$MODEL")"
echo "  Prompt: $PROMPT"
echo "  Max tokens: $MAX_TOKENS"
echo "  Runs/stack: $N_RUNS"
echo "  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

run_stack "Default (current)" ""
run_stack "Path E (USE_DENSE)" "HF2Q_USE_DENSE=1"
run_stack "Path E+G (+LMHEAD_Q6K) ★ safe flip" "HF2Q_USE_DENSE=1 HF2Q_LMHEAD_Q6K=1"
run_stack "Path E+G + FUSED_END_OF_LAYER" "HF2Q_USE_DENSE=1 HF2Q_LMHEAD_Q6K=1 HF2Q_FUSED_END_OF_LAYER=1 HF2Q_UNSAFE_EXPERIMENTS=1"
# iter-233: Path E+F+G (HF2Q_F16_KV=1) produces "<pad>" at 1000-tok output —
# DO NOT recommend as default. Kept here for regression-tracking only.
run_stack "Path E+F+G (F16 KV — DEGRADED >200 tok ✗)" "HF2Q_USE_DENSE=1 HF2Q_F16_KV=1 HF2Q_LMHEAD_Q6K=1 HF2Q_UNSAFE_EXPERIMENTS=1"

if [[ -x "$LLAMA_BENCH" ]]; then
    echo "=== llama.cpp peer (tg128) ==="
    "$LLAMA_BENCH" -m "$MODEL" -p 0 -n 128 -t 8 2>&1 | grep -E "tg128 *\|" | tail -1
else
    echo "[skip] llama.cpp peer bench: $LLAMA_BENCH not found"
fi

echo
echo "Reference (iter-216/iter-233 — long-form 1000-tok measurements):"
echo "  Default:       68.4 tok/s   (0.666x peer)   ✓ coherent"
echo "  Path E:        69.2 tok/s   (0.674x peer)   ✓ coherent"
echo "  Path E+G:      70.9 tok/s   (0.690x peer)   ✓ coherent ★ SAFE FLIP"
echo "  Path E+G+FUS:  ~73 tok/s    (0.71x peer)    ✓ coherent (UNSAFE_EXP)"
echo "  Path E+F+G:    ~73 tok/s    -- F16 KV DEGRADES at 1000 tok ✗"
echo "  llama.cpp:    102.7 tok/s   (1.000x peer baseline)"
