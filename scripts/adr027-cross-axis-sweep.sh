#!/usr/bin/env bash
# ADR-027 Phase B cross-axis sweep harness.
#
# Verifies coherence + byte-equivalence across the 4-cell matrix:
#   {HF2Q_TQ_KV ∈ 0, 1} × {HF2Q_KV_PERSIST ∈ unset, /tmp/cache}
#
# Built for iter-19 (initial empirical pass) + packaged in iter-21 as a
# reusable operator-runnable harness so future ADR-027 changes (iter-22+
# F32 drop, future kernel updates, etc.) can be re-validated against the
# same regression matrix with one command.
#
# # Usage
#
# bash scripts/adr027-cross-axis-sweep.sh [MODEL_PATH] [PROMPT] [MAX_TOKENS]
#
# Defaults:
#   MODEL_PATH  = /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf
#   PROMPT      = "Explain in three sentences why the speed of light is constant in a vacuum, citing the relevant principle."
#   MAX_TOKENS  = 32
#
# # Exit codes
#
# 0 = all 4 cells produced byte-identical output (after filtering dynamic
#     banner lines like `[hf2q lcp] byte_budget=...`).
# 1 = at least one cell diverged from the F32 baseline.
# 2 = generate command failed for at least one cell.
#
# # ADR-027 LANDED commitments this harness defends
#
# - iter-15: HF2Q_TQ_KV=1 produces BYTE-IDENTICAL output to F32 baseline
# - iter-16: live coherence on qwen36 35B-A3B-APEX-Q5_K_M
# - iter-17: load-banner surfaces tq_kv_active state
# - iter-19: cross-axis {TQ × persist} all 4 cells byte-identical
#
# Any future change that breaks one of these commitments will fail this
# harness. Run after every meaningful change to the qwen35 forward path
# OR the TQ encode/SDPA chain OR the LCP/persist subsystem.

set -euo pipefail

MODEL="${1:-/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf}"
PROMPT="${2:-Explain in three sentences why the speed of light is constant in a vacuum, citing the relevant principle.}"
MAX_TOKENS="${3:-32}"
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"

if [[ ! -x "$HF2Q_BIN" ]]; then
    echo "ERROR: $HF2Q_BIN not found or not executable."
    echo "Build with: cargo build --bin hf2q --release"
    exit 2
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found: $MODEL"
    exit 2
fi

PERSIST_DIR="$(mktemp -d -t hf2q-adr027-sweep-XXXXXX)"
trap 'rm -rf "$PERSIST_DIR"' EXIT

echo "=== ADR-027 Phase B cross-axis sweep ==="
echo "Model:        $MODEL"
echo "Prompt:       \"$PROMPT\""
echo "Max tokens:   $MAX_TOKENS"
echo "Persist dir:  $PERSIST_DIR"
echo ""

# Strip dynamic / system-state banner lines so byte-equivalence holds
# across cells regardless of momentary system memory or wall-clock
# timing. Specifically we drop:
#   - hf2q load: ...           (load banner; varies on quant_label timing)
#   - prefill: ...             (perf line; absolute ms varies)
#   - --- mlx-native ...       (perf footer; varies)
#   - [hf2q lcp] byte_budget=  (sysinfo-derived; varies with available memory)
extract_output() {
    grep -v '^hf2q load\|^prefill:\|^---\|hf2q lcp\] byte_budget=' || true
}

run_cell() {
    local label="$1"
    local env_pre="$2"
    local out
    if ! out=$(eval "$env_pre $HF2Q_BIN generate --model $MODEL --prompt \"$PROMPT\" --max-tokens $MAX_TOKENS --temperature 0" 2>&1); then
        echo "ERROR: cell $label failed to generate" >&2
        echo "$out" | tail -10 >&2
        exit 2
    fi
    # Display info → stderr (so capturing stdout returns only the
    # filtered comparison body for byte-equivalence checks).
    local prefill_line
    local decode_line
    local tq_banner
    prefill_line=$(echo "$out" | grep '^prefill:' | head -1)
    decode_line=$(echo "$out" | grep 'tokens in' | tail -1)
    tq_banner=$(echo "$out" | grep '^hf2q load: tq_kv' | head -1)
    {
        printf "  [%s]\n" "$label"
        printf "    %s\n" "$tq_banner"
        printf "    %s\n" "$prefill_line"
        printf "    %s\n" "$decode_line"
    } >&2
    # Filtered comparison body → stdout (consumed by check_pair diff).
    echo "$out" | extract_output
}

OUT_A=$(run_cell "A: F32 baseline (no persist)" "")
OUT_B=$(run_cell "B: F32 + persist"             "HF2Q_KV_PERSIST=$PERSIST_DIR")
OUT_C=$(run_cell "C: TQ-on (no persist)"         "HF2Q_TQ_KV=1")
OUT_D=$(run_cell "D: TQ-on + persist"            "HF2Q_TQ_KV=1 HF2Q_KV_PERSIST=$PERSIST_DIR")

echo ""
echo "=== Byte-equivalence matrix (filtered) ==="
FAILED=0
check_pair() {
    local label="$1"
    local left="$2"
    local right="$3"
    if diff <(echo "$left") <(echo "$right") > /dev/null; then
        printf "  %-30s  BYTE-IDENTICAL ✓\n" "$label"
    else
        printf "  %-30s  DIFF FOUND ✗\n" "$label"
        echo "    --- diff ---"
        diff <(echo "$left") <(echo "$right") | head -20 | sed 's/^/    /'
        FAILED=1
    fi
}
check_pair "A vs B (F32 ± persist)"       "$OUT_A" "$OUT_B"
check_pair "A vs C (F32 vs TQ, no persist)" "$OUT_A" "$OUT_C"
check_pair "A vs D (F32 vs TQ, persist)"   "$OUT_A" "$OUT_D"
check_pair "C vs D (TQ ± persist)"         "$OUT_C" "$OUT_D"

echo ""
if [[ "$FAILED" -ne 0 ]]; then
    echo "FAILED: at least one cell diverged from F32 baseline." >&2
    echo "ADR-027 Phase B coherence commitments may have regressed." >&2
    exit 1
fi

echo "PASSED: all 4 cells byte-identical."
echo "ADR-027 Phase B coherence commitments hold."
exit 0
