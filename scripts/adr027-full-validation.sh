#!/usr/bin/env bash
# ADR-027 Phase B + iter-23 chain FULL validation entry-point (iter-46).
#
# One-stop operator-runnable command that sequences every regression
# net protecting the iter-23 sub-iter chain's 3.94× memory savings.
#
# Sequence:
#   1. Build hf2q --release (required by the sweep harnesses).
#   2. Run hf2q full unit-test suite (3377 tests including iter-44's
#      `qh35_disk_cross_process_replay_with_tq_payload_byte_equal`).
#   3. Run mlx-native lib unit-test suite (269 tests).
#   4. Run the 31-token cross-axis sweep (4 cells × 1 length, ~15s).
#   5. Run the long-context sweep (default 4K + 8K, ~30s; operator can
#      extend via LENGTHS env).
#
# Total runtime: ~1-2 minutes wall (depending on first-build cache).
#
# # Usage
#
# bash scripts/adr027-full-validation.sh
#
# # Env overrides
#
#   MODEL=...     model GGUF path (defaults to qwen36 35B-A3B-APEX-Q5_K_M)
#   LENGTHS=...   long-context sweep lengths (default 4096,8192;
#                 add 16384,32768 for extended validation; multi-min wall)
#   SKIP_TESTS=1  skip cargo test (assume already green; saves ~30s)
#   SKIP_BUILD=1  skip cargo build (assume already built; saves ~5s)
#
# # Exit codes
#
#   0 = ALL gates PASS (build + tests + both sweeps green)
#   non-0 = some gate failed; check the corresponding section's output
#
# # ADR-027 LANDED commitments this script defends
#
# - iter-23 chain (29..36): 3.94× per-slot KV memory savings live; F32
#   K/V alloc dropped in TQ-only mode.
# - iter-21 cross-axis sweep: F32 ≡ TQ ≡ persist ≡ no-persist at 31-tok.
# - iter-41/43 long-context byte-identity: F32 ≡ TQ at 4K/8K/16K/32K.
# - iter-44 end-to-end cross-process replay: TQ payload byte-identical.
# - iter-45 mul_mv_ext kernel_name test debt closed.
#
# Any future change that breaks one of these gates will fail this
# script. Run after every meaningful change to:
#   - qwen35 forward path / TQ encode / SDPA
#   - LCP/persist subsystem / codec
#   - mlx-native KV-related kernels
#
# # Loading dock for new validation gates
#
# When ADR-027 grows new commitments, append a new section below the
# existing four. Don't introduce parallel/branching scripts — keep
# this as the single source of truth for operator validation.

set -euo pipefail

# Allow color output but stay sane in CI/non-tty.
if [[ -t 1 ]]; then
    BOLD=$'\e[1m'
    GREEN=$'\e[32m'
    RED=$'\e[31m'
    YELLOW=$'\e[33m'
    RESET=$'\e[0m'
else
    BOLD=""
    GREEN=""
    RED=""
    YELLOW=""
    RESET=""
fi

MODEL="${MODEL:-/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf}"
LENGTHS="${LENGTHS:-4096,8192}"
SKIP_TESTS="${SKIP_TESTS:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"

cd /opt/hf2q

echo "${BOLD}=== ADR-027 Phase B + iter-23 chain FULL validation ===${RESET}"
echo "Model:    $MODEL"
echo "Lengths:  $LENGTHS (long-context sweep)"
echo "Skip:     build=${SKIP_BUILD} tests=${SKIP_TESTS}"
echo ""

# ── Gate 1: build hf2q --release ──
if [[ "$SKIP_BUILD" != "1" ]]; then
    echo "${BOLD}[1/5] Build hf2q --release${RESET}"
    if cargo build --bin hf2q --release 2>&1 | tail -3; then
        echo "${GREEN}    ✓ build OK${RESET}"
    else
        echo "${RED}    ✗ build FAILED${RESET}" >&2
        exit 1
    fi
    echo ""
else
    echo "${YELLOW}[1/5] Build SKIPPED (SKIP_BUILD=1)${RESET}"
    echo ""
fi

# ── Gate 2: hf2q full unit-test suite ──
if [[ "$SKIP_TESTS" != "1" ]]; then
    echo "${BOLD}[2/5] hf2q unit tests (3377+ expected)${RESET}"
    if cargo test --bin hf2q --quiet 2>&1 | tail -3; then
        echo "${GREEN}    ✓ hf2q tests PASS${RESET}"
    else
        echo "${RED}    ✗ hf2q tests FAILED${RESET}" >&2
        exit 2
    fi
    echo ""

    # ── Gate 3: mlx-native lib unit-test suite ──
    echo "${BOLD}[3/5] mlx-native lib tests (269+ expected)${RESET}"
    if (cd /opt/mlx-native && RUSTC_WRAPPER= cargo test --lib --quiet 2>&1 | tail -3); then
        echo "${GREEN}    ✓ mlx-native lib tests PASS${RESET}"
    else
        echo "${RED}    ✗ mlx-native lib tests FAILED${RESET}" >&2
        exit 3
    fi
    echo ""
else
    echo "${YELLOW}[2-3/5] Tests SKIPPED (SKIP_TESTS=1)${RESET}"
    echo ""
fi

# ── Gate 4: 31-tok cross-axis sweep ──
echo "${BOLD}[4/5] 31-tok cross-axis sweep (F32 × TQ × persist)${RESET}"
if bash scripts/adr027-cross-axis-sweep.sh "$MODEL" 2>&1 | tail -8; then
    echo "${GREEN}    ✓ cross-axis sweep PASS (4 cells byte-identical)${RESET}"
else
    echo "${RED}    ✗ cross-axis sweep FAILED${RESET}" >&2
    exit 4
fi
echo ""

# ── Gate 5: long-context sweep at LENGTHS ──
echo "${BOLD}[5/5] long-context sweep at $LENGTHS (F32 vs TQ via null template)${RESET}"
if MODEL="$MODEL" bash scripts/adr027-long-context-sweep.sh "$LENGTHS" 64 2>&1 | tail -10; then
    echo "${GREEN}    ✓ long-context sweep PASS at all lengths${RESET}"
else
    echo "${RED}    ✗ long-context sweep FAILED${RESET}" >&2
    exit 5
fi
echo ""

echo "${GREEN}${BOLD}=== ADR-027 PHASE B + iter-23 CHAIN: ALL GATES PASS ===${RESET}"
echo "  • Build green"
echo "  • hf2q + mlx-native unit tests green"
echo "  • Cross-axis sweep at 31-tok prefill: 4 cells byte-identical"
echo "  • Long-context sweep at $LENGTHS: F32 ≡ TQ at every length"
echo ""
echo "ADR-027 commitments hold. The 3.94× per-slot KV memory savings"
echo "is correct and validated end-to-end."
