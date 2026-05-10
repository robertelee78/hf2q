#!/usr/bin/env bash
# ADR-028 iter-241+iter-242 — long-context coherence regression gate.
#
# Guards against silent regression of the iter-233 F16 KV finding +
# extends to BOTH production models (gemma4 + qwen3.6) per iter-242.
# For each (model, stack), runs hf2q generate at MAX_TOKENS=1000 and
# checks the output text for known degradation signatures:
#   - "<pad>"      → F16 KV / argmax-killed-by-noise pattern
#   - "<unk>"      → tokenizer fallback / weight corruption
#   - empty body   → early-EOS at tok-0 (per project_iter40 memory)
#   - all-same-char repeat-loop (>150 chars of same byte)
#   - non-zero exit (binary crash)
#
# Stack semantics differ by model (per iter-234 cross-model finding):
#   - gemma4: F16 KV is BROKEN at long context (random <pad>) → EXPECTED-FAIL
#   - qwen3.6: F16 KV is COHERENT but no-op perf-wise (MoE not KV-bandwidth-bound)
#
# Output: PASS/FAIL per (model, stack).  Exit non-zero if any of the
# SAFE stacks fails on EITHER model.
#
# Usage:
#   bash scripts/adr028_coherence_gate.sh [PROMPT] [MAX_TOKENS]
#
# Defaults: telescope prompt, 1000 tokens.
#
# Requires: hf2q built at target/release/.  Model paths below (or
# overrideable via HF2Q_GEMMA4_MODEL / HF2Q_QWEN36_MODEL env).

set -euo pipefail

GEMMA4_MODEL="${HF2Q_GEMMA4_MODEL:-/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf}"
QWEN36_MODEL="${HF2Q_QWEN36_MODEL:-/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf}"
PROMPT="${1:-Write a comprehensive multi-chapter story about a sentient telescope that observes the universe over centuries}"
MAX_TOKENS="${2:-1000}"
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"

if [[ ! -x "$HF2Q_BIN" ]]; then
    echo "[fatal] hf2q binary not found at $HF2Q_BIN" >&2
    exit 1
fi
# Per-model existence checks happen at gate-time so missing one model
# only skips that section, not the whole gate.

# hf2q stdout layout (verified iter-241 against gemma4 APEX-Q5_K_M):
#   "hf2q load: ..." banner lines (multiple)
#   "prefill: NN tok in MMms (RR tok/s)"
#   <blank line>
#   <model body — what we care about>
# (timing line "--- mlx-native: ..." is on STDERR, not STDOUT.)
extract_body() {
    local out_file="$1"
    # Strip ANSI escapes, then take everything AFTER the "prefill:" line.
    sed $'s/\x1b\\[[0-9;]*[a-zA-Z]//g' "$out_file" \
      | awk '/^prefill: /{found=1; next} found{print}'
}

# Returns 0 (coherent) or 1 (degraded). Prints OK or FAIL[reason].
check_coherence() {
    local body_file="$1"
    local body_len
    body_len=$(wc -c < "$body_file" | tr -d ' ')
    if (( body_len < 50 )); then
        echo "FAIL[short:$body_len]"; return 1
    fi
    if grep -qE '<pad>|<unk>' "$body_file"; then
        echo "FAIL[sentinel]"; return 1
    fi
    # Detect repeat-loop: any single byte repeated >150 times in a row.
    local max_repeat
    max_repeat=$(tr -d '\n' < "$body_file" \
        | awk '{
            n = length($0); maxr = 0; i = 1;
            while (i <= n) {
                c = substr($0, i, 1); j = i;
                while (j <= n && substr($0, j, 1) == c) j++;
                if (j - i > maxr) maxr = j - i;
                i = j;
            }
            print maxr;
        }')
    if (( max_repeat > 150 )); then
        echo "FAIL[repeat:$max_repeat]"; return 1
    fi
    echo "OK"; return 0
}

run_stack_with_check() {
    local model="$1"; shift
    local label="$1"; shift
    local env_vars="$*"
    local tmpfile errfile bodyfile
    tmpfile=$(mktemp)
    errfile=$(mktemp)
    bodyfile=$(mktemp)
    # shellcheck disable=SC2086
    # env requires word-split of "KEY=v1 KEY2=v2" to pass each as a
    # separate argv element. Quoting collapses to one arg, breaking env.
    # stdout = model body; stderr = diagnostic banner (separate).
    if ! env $env_vars "$HF2Q_BIN" generate \
            --model "$model" \
            --prompt "$PROMPT" \
            --max-tokens "$MAX_TOKENS" >"$tmpfile" 2>"$errfile"; then
        echo "  $label: FAIL[crash] (stderr:$(tail -1 "$errfile" 2>/dev/null | head -c 60))"
        rm -f "$tmpfile" "$errfile" "$bodyfile"
        return 1
    fi
    extract_body "$tmpfile" >"$bodyfile"
    local result
    result=$(check_coherence "$bodyfile") || true
    local preview
    preview=$(head -c 80 "$bodyfile" | tr -d '\n' || true)
    echo "  $label: $result  | \"${preview}…\""
    rm -f "$tmpfile" "$errfile" "$bodyfile"
    [[ "$result" == "OK" ]]
}

echo "ADR-028 coherence regression gate (iter-241+iter-242 multi-model)"
echo "  Prompt: ${PROMPT:0:60}..."
echo "  Max tokens: $MAX_TOKENS"
echo "  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

failures=0
expected_failures=0

# ============================================================================
# Section 1 — gemma4 26B-A4B (full stack matrix; F16 KV is EXPECTED-FAIL)
# ============================================================================
if [[ -f "$GEMMA4_MODEL" ]]; then
    echo "================================================================"
    echo "Model 1: gemma4 — $(basename "$GEMMA4_MODEL")"
    echo "================================================================"
    echo "[SAFE] stacks (gate-break if any fails):"
    run_stack_with_check "$GEMMA4_MODEL" "Default                        " "" || failures=$((failures + 1))
    run_stack_with_check "$GEMMA4_MODEL" "Path E                         " "HF2Q_USE_DENSE=1" || failures=$((failures + 1))
    run_stack_with_check "$GEMMA4_MODEL" "Path E+G ★ safe flip           " "HF2Q_USE_DENSE=1 HF2Q_LMHEAD_Q6K=1" || failures=$((failures + 1))
    run_stack_with_check "$GEMMA4_MODEL" "Path E+G + FUSED               " "HF2Q_USE_DENSE=1 HF2Q_LMHEAD_Q6K=1 HF2Q_FUSED_END_OF_LAYER=1 HF2Q_UNSAFE_EXPERIMENTS=1" || failures=$((failures + 1))

    echo
    echo "[EXPECTED-FAIL] stacks (deprecated on gemma4; failure is correctness signal):"
    # iter-233/iter-234: HF2Q_F16_KV=1 produces random `<pad>` on gemma4 only.
    if run_stack_with_check "$GEMMA4_MODEL" "Path E+F+G (deprecated F16 KV)" "HF2Q_USE_DENSE=1 HF2Q_F16_KV=1 HF2Q_LMHEAD_Q6K=1 HF2Q_UNSAFE_EXPERIMENTS=1"; then
        echo "  WARN: Path E+F+G PASSED coherence on gemma4 — sampling-luck or F16 KV behavior changed."
        echo "  WARN: Re-run several times.  Per iter-234 sweep, failure rate is non-deterministic."
    else
        expected_failures=$((expected_failures + 1))
    fi
    echo
else
    echo "[SKIP] gemma4 model not found at $GEMMA4_MODEL (override via HF2Q_GEMMA4_MODEL)"
    echo
fi

# ============================================================================
# Section 2 — qwen3.6 35B-A3B (default-stack only; F16 KV is no-op-but-coherent
# per iter-234, so SAFE list is identical and Path E+F+G is also SAFE there)
# ============================================================================
if [[ -f "$QWEN36_MODEL" ]]; then
    echo "================================================================"
    echo "Model 2: qwen3.6 — $(basename "$QWEN36_MODEL")"
    echo "================================================================"
    echo "[SAFE] stacks (gate-break if any fails):"
    run_stack_with_check "$QWEN36_MODEL" "Default                        " "" || failures=$((failures + 1))
    # Per iter-234: F16 KV on qwen3.6 is COHERENT (1000-tok identical to default)
    # but provides ZERO perf gain.  Listed under SAFE because correctness-wise
    # it is safe on qwen3.6 — the deprecation banner still fires for operator
    # awareness, but the gate does not break.  This is the cross-model
    # asymmetry from iter-234.
    run_stack_with_check "$QWEN36_MODEL" "Default + F16 KV (no-op on qwen3.6)" "HF2Q_F16_KV=1 HF2Q_UNSAFE_EXPERIMENTS=1" || failures=$((failures + 1))
    echo
else
    echo "[SKIP] qwen3.6 model not found at $QWEN36_MODEL (override via HF2Q_QWEN36_MODEL)"
    echo
fi

echo "================================================================"
echo "Summary:"
echo "  SAFE stack failures (across all models): $failures (gate-break threshold = 0)"
echo "  EXPECTED-FAIL confirmed (gemma4 F16 KV): $expected_failures (= 1 means iter-233 deprecation still load-bearing)"
if (( failures > 0 )); then
    echo "  GATE: FAIL — $failures SAFE stack(s) regressed coherence."
    exit 1
fi
echo "  GATE: PASS"
