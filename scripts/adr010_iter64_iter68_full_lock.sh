#!/usr/bin/env bash
# ADR-010 iter-64 + iter-68 wins lock-in chain.
#
# Single operator-runnable command that runs ALL the regression gates
# protecting the iter-64 (forward_prefill_batched.rs leg_hb_encoded fix,
# 22-29× speedup over per-token default) AND iter-68 (mlx-native one-
# character typo fix unlocking tensor mm_id, BEATS llama.cpp at pp1024)
# wins.
#
# Sequence:
#   1. mlx-native shader-compile gate (iter-73) — every .metal file
#      compiles via xcrun. Catches iter-68-style silent typos that
#      cause runtime fallback to slower kernel paths.
#   2. hf2q batched-prefill coherence + perf gate (iter-65/74/75):
#      - per-token vs batched byte-identity at short prompt
#      - pp1024 batched perf >= 1500 t/s floor (iter-68 baseline 1942)
#
# Total runtime: ~1 minute. Run after any change to forward_prefill_
# batched.rs, the mm_id Metal kernels, or the dispatcher routing.
#
# # Usage
#
#   bash scripts/adr010_iter64_iter68_full_lock.sh
#
# # Env overrides
#
#   MODEL=...                     model path (default: gemma APEX-Q5_K_M)
#   HF2Q_GATE_PERF_FLOOR=...      lower the pp1024 floor (default 1500)
#   HF2Q_GATE_SKIP_PERF=1         skip perf-sanity (slower hardware)
#
# # Exit codes
#
#   0 = ALL gates PASS
#   1 = shader-compile gate failed (mlx-native side)
#   2 = batched coherence/perf gate failed (hf2q side)

set -euo pipefail

if [[ -t 1 ]]; then
    BOLD=$'\e[1m'
    GREEN=$'\e[32m'
    RED=$'\e[31m'
    RESET=$'\e[0m'
else
    BOLD=""; GREEN=""; RED=""; RESET=""
fi

echo "${BOLD}=== ADR-010 iter-64 + iter-68 wins LOCK-IN check ===${RESET}"
echo

echo "${BOLD}[1/2] mlx-native shader-compile gate (iter-73)${RESET}"
if (cd /opt/mlx-native && RUSTC_WRAPPER= cargo test --test test_all_shaders_compile --quiet 2>&1 | tail -3); then
    echo "${GREEN}  ✓ shader-compile gate PASS${RESET}"
else
    echo "${RED}  ✗ shader-compile gate FAILED${RESET}" >&2
    exit 1
fi
echo

echo "${BOLD}[2/2] hf2q batched-prefill coherence + perf gate (iter-65/74/75)${RESET}"
if bash /opt/hf2q/scripts/adr010_iter64_batched_coherence_gate.sh 2>&1 | tail -10; then
    echo "${GREEN}  ✓ batched coherence + perf gate PASS${RESET}"
else
    echo "${RED}  ✗ batched coherence + perf gate FAILED${RESET}" >&2
    exit 2
fi
echo

echo "${GREEN}${BOLD}=== ALL ADR-010 iter-64/68 LOCK-IN GATES PASS ===${RESET}"
echo "  • Every Metal shader compiles (no silent typos)"
echo "  • Per-token ≡ batched on short prompt (no compute-path divergence)"
echo "  • pp1024 batched ≥ 1500 t/s floor (iter-68 tensor mm_id active)"
echo
echo "iter-64 (forward_prefill_batched.rs HB-encode fix, 91 LOC at 133722d)"
echo "and iter-68 (mlx-native typo fix at b6b8e79) wins are protected."
