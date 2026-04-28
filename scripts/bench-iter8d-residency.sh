#!/usr/bin/env bash
# scripts/bench-iter8d-residency.sh
#
# ADR-015 iter8d residency-set bench runner.
#
# Wraps bench-baseline.sh with the exact parameters from the iter8d spec:
#   - 5 trials (N_TRIALS=5)
#   - 256-token generation (NGEN=256)
#   - 120-second thermal settle between trials (THERMAL_SETTLE_SEC=120)
#   - SKIP_THERMAL_GATE=1 (thermal pre-check skipped; SoC must be cold before
#     invoking this script)
#   - model: apex dwq46 35B-A3B
#   - label: apex-dwq46-iter8d-residency
#
# Phase 3 merge orchestrator runs this AFTER the worktree is merged into main
# and a fresh release build is produced.  Do NOT run inside the worktree.
#
# Usage:
#   cd /opt/hf2q
#   ./scripts/bench-iter8d-residency.sh
#
# To override model path:
#   MODEL=/path/to/model.gguf ./scripts/bench-iter8d-residency.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-${REPO_ROOT}/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf}"

if [[ ! -f "${MODEL}" ]]; then
    echo "ERROR: model not found at ${MODEL}" >&2
    echo "Set MODEL env var to override." >&2
    exit 1
fi

exec env \
    SKIP_THERMAL_GATE=1 \
    N_TRIALS=5 \
    THERMAL_SETTLE_SEC=120 \
    NGEN=256 \
    "${SCRIPT_DIR}/bench-baseline.sh" \
        --model "${MODEL}" \
        --label "apex-dwq46-iter8d-residency" \
        "$@"
