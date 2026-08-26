#!/usr/bin/env bash

# Resolve the canonical Q5_K narrow-width route once, before any model or
# binary preflight. Canonical launchers make the accepted policy explicit;
# direct `hf2q serve` does not source this file and retains mlx-native's typed
# default.
hf2q_resolve_q5k_canonical_policy() {
    local value=${HF2Q_Q5K_CANONICAL_Q4X4-1}
    case "$value" in
        0|1) ;;
        *)
            echo "HF2Q_Q5K_CANONICAL_Q4X4 must be 0 or 1 (got: $value)" >&2
            return 3
            ;;
    esac
    HF2Q_Q5K_CANONICAL_Q4X4=$value
    export HF2Q_Q5K_CANONICAL_Q4X4
}
