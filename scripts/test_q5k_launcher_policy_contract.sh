#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
HELPER="$ROOT_DIR/scripts/hf2q_q5_policy.sh"
LAUNCHERS=(
    "$ROOT_DIR/scripts/serve_qwen38_opencode.sh"
    "$ROOT_DIR/scripts/serve_qwen36_opencode.sh"
    "$ROOT_DIR/scripts/serve_gemma4_opencode.sh"
    "$ROOT_DIR/scripts/serve_deepseek4_opencode.sh"
)

# shellcheck source=scripts/hf2q_q5_policy.sh
source "$HELPER"

unset HF2Q_Q5K_CANONICAL_Q4X4
hf2q_resolve_q5k_canonical_policy
[[ "$HF2Q_Q5K_CANONICAL_Q4X4" == 1 ]]

HF2Q_Q5K_CANONICAL_Q4X4=0
hf2q_resolve_q5k_canonical_policy
[[ "$HF2Q_Q5K_CANONICAL_Q4X4" == 0 ]]

HF2Q_Q5K_CANONICAL_Q4X4=invalid
if hf2q_resolve_q5k_canonical_policy >/dev/null 2>&1; then
    echo "invalid Q5 policy unexpectedly resolved" >&2
    exit 1
fi

for launcher in "${LAUNCHERS[@]}"; do
    bash -n "$launcher"
    grep -Fq 'hf2q_q5_policy.sh' "$launcher"
    grep -Fq 'hf2q_resolve_q5k_canonical_policy' "$launcher"
    grep -Fq 'HF2Q_Q5K_CANONICAL_Q4X4=' "$launcher"

    set +e
    error=$(HF2Q_Q5K_CANONICAL_Q4X4=invalid \
        MODEL=/definitely/missing/q5-policy-contract.gguf \
        HF2Q_BIN=/definitely/missing/hf2q \
            "$launcher" 2>&1 >/dev/null)
    status=$?
    set -e
    [[ "$status" == 3 ]] || {
        echo "$(basename "$launcher") returned $status instead of 3 for invalid Q5 policy" >&2
        exit 1
    }
    grep -Fq 'HF2Q_Q5K_CANONICAL_Q4X4 must be 0 or 1' <<<"$error" || {
        echo "$(basename "$launcher") did not reject invalid Q5 policy before model preflight" >&2
        exit 1
    }
    if grep -Eq 'model not found|binary not found' <<<"$error"; then
        echo "$(basename "$launcher") reached model/binary preflight before Q5 policy rejection" >&2
        exit 1
    fi
done

# The server implementation resolves the native typed policy. It must not
# duplicate the launcher's environment default in Rust.
grep -Fq 'mlx_native::ggml_routing_policy_from_environment()' \
    "$ROOT_DIR/src/serve/process_policy.rs"
if grep -Eq 'set_var\("HF2Q_Q5K_CANONICAL_Q4X4"|HF2Q_Q5K_CANONICAL_Q4X4.*unwrap_or' \
    "$ROOT_DIR/src/serve/process_policy.rs"; then
    echo "direct serve must retain the mlx-native typed default" >&2
    exit 1
fi

echo "canonical Q5 launcher policy contract passed"
