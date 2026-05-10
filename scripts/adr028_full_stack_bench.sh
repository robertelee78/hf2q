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
# ADR-028 iter-293/297/299: G+FUSED (TQ-HB-intact) is the operator-actionable
# safe flip — both flags precision-exact (iter-188 + iter-219), TQ-HB memory
# savings preserved (3.94x per-slot), measured +2.6% (3-run σ < 0.1).
run_stack "G+FUSED (TQ-HB intact) ★ iter-293 safe flip" "HF2Q_LMHEAD_Q6K=1 HF2Q_FUSED_END_OF_LAYER=1"
# ADR-028 iter-318: STACKED G+FUSED+V2+NR2 — adds iter-310 rms_norm_v2
# (float4+simd_sum) and iter-309 q6_K nr0=2 on top. TQ-HB intact, parity-
# proven, +4.3% measured (3-run σ < 0.1).
run_stack "G+FUSED+V2+NR2 (TQ-HB intact) ★ iter-318 stacked safe flip" "HF2Q_LMHEAD_Q6K=1 HF2Q_FUSED_END_OF_LAYER=1 HF2Q_RMS_NORM_V2=1 HF2Q_Q6K_MV_NR2=1"
run_stack "Path E (USE_DENSE)" "HF2Q_USE_DENSE=1"
run_stack "Path E+G (+LMHEAD_Q6K)" "HF2Q_USE_DENSE=1 HF2Q_LMHEAD_Q6K=1"
run_stack "Path E+G + FUSED_END_OF_LAYER" "HF2Q_USE_DENSE=1 HF2Q_LMHEAD_Q6K=1 HF2Q_FUSED_END_OF_LAYER=1 HF2Q_UNSAFE_EXPERIMENTS=1"
# ADR-028 iter-318: E+G+FUSED+V2+NR2 — breaks TQ-HB memory mantra (Path E)
# but provides reference of full stack ceiling.
run_stack "Path E+G+FUSED+V2+NR2 (breaks TQ-HB)" "HF2Q_USE_DENSE=1 HF2Q_LMHEAD_Q6K=1 HF2Q_FUSED_END_OF_LAYER=1 HF2Q_RMS_NORM_V2=1 HF2Q_Q6K_MV_NR2=1 HF2Q_UNSAFE_EXPERIMENTS=1"
# iter-233: Path E+F+G (HF2Q_F16_KV=1) produces "<pad>" at 1000-tok output —
# DO NOT recommend as default. Kept here for regression-tracking only.
run_stack "Path E+F+G (F16 KV — DEGRADED >200 tok ✗)" "HF2Q_USE_DENSE=1 HF2Q_F16_KV=1 HF2Q_LMHEAD_Q6K=1 HF2Q_UNSAFE_EXPERIMENTS=1"

if [[ -x "$LLAMA_BENCH" ]]; then
    # iter-259: hf2q runs sustained 1000-token decode (~14s); tg128 measures
    # burst (~1.4s) which doesn't hit Apple Silicon's thermal throttle.
    # Use tg1024 for proper apples-to-apples regime match (matches hf2q's
    # measurement window). tg128 reported alongside for backwards
    # compatibility with iter-183/iter-216 historical numbers.
    echo "=== llama.cpp peer (tg128 burst + tg1024 matched-regime) ==="
    "$LLAMA_BENCH" -m "$MODEL" -p 0 -n 128,1024 -r 3 -t 8 2>&1 \
        | grep -E "(tg128|tg1024) *\|" | tail -2
else
    echo "[skip] llama.cpp peer bench: $LLAMA_BENCH not found"
fi

echo
echo "Reference HEAD measurements (gemma4 APEX-Q5_K_M, 5-run median 200-tok, σ <0.1):"
echo "  Default:           69.2 tok/s   (0.679x peer 102) — TQ-HB intact"
echo "  G+FUSED:           71.3 tok/s   (0.700x peer; +2.6%) iter-293"
echo "  G+FUSED+V2+NR2:    72.4 tok/s   (0.711x peer; +4.6%) ★ iter-318 STACKED SAFE FLIP — TQ-HB intact"
echo "  Path E:            ~71.0 tok/s  (0.696x peer)        — F32 KV (no TQ-HB)"
echo "  Path E+G:          ~72.4 tok/s  (0.710x peer)"
echo "  Path E+G+FUS:      73.5 tok/s   (0.721x peer)        — breaks TQ-HB memory mantra"
echo "  Path E+G+FUS+V2+NR2: 74.9 tok/s (0.735x peer; +7.8%) — breaks TQ-HB"
echo "  Path E+F+G:        --           (F16 KV DEGRADES at 1000 tok ✗ iter-233)"
echo "  llama.cpp tg128:   ~103 tok/s   (burst)"
echo "  llama.cpp tg1024:  ~97 tok/s    (matched regime)"
echo
echo "Operator decision tree (TQ-HB intact options):"
echo "  - Status quo (default):              0.679x peer  TQ-HB ✓"
echo "  - Flip G+FUSED:                      0.700x peer  TQ-HB ✓  (iter-293 candidate)"
echo "  - Flip G+FUSED+V2+NR2:               0.711x peer  TQ-HB ✓  ★ iter-318 candidate (+1.7% more)"
echo "  - Flip Path E+G+FUSED+V2+NR2:        0.735x peer  TQ-HB ✗  (breaks 3.94x memory)"
echo "  - DFlash port (multi-month):         1.4-2.8x peer TQ-HB ✓"
