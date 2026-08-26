#!/usr/bin/env bash
# Canonical Qwen3.8-27B launcher for OpenAI-compatible coding clients.
#
# Qwen3.8 is multimodal. The default requires both the text GGUF and its
# source-matched projector; startup fails closed if either side is missing or
# incompatible. Set QWEN38_VISION=off only for an intentional text-only run.
# Exact target-verified speculation is considered by default on eligible
# requests. `auto` first measures ordinary decode, tries request-history lookup
# before an available MTP head, and independently drops either proposer for the
# generation after two consecutive measured four-round windows are not better
# than ordinary decode. Unsupported request semantics remain on ordinary target
# decode.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/hf2q_q5_policy.sh
source "$script_dir/hf2q_q5_policy.sh"
hf2q_resolve_q5k_canonical_policy

export MODEL="${MODEL:-/opt/hf2q/models/qwen3.8/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf}"
export MMPROJ="${MMPROJ:-${MODEL%.gguf}-mmproj.gguf}"
export VISION_MODE="${QWEN38_VISION:-required}"
export HF2Q_QWEN_SPECULATION="${QWEN38_SPECULATION:-auto}"
# Qwen3.8 K=3 verifies four target positions at once. The verifier uses the
# shared coherent K-quant routing policy; a model label must not select a
# width-dependent reduction tree. The canonical Q5 policy is resolved and
# exported above so this launcher and every later model swap share one value.
# Nonzero MV_EXT remains an explicit experimental/inexact override.

case "$HF2Q_QWEN_SPECULATION" in
    off|auto) ;;
    *)
        echo "QWEN38_SPECULATION must be off or auto (got: $HF2Q_QWEN_SPECULATION)" >&2
        exit 3
        ;;
esac
for route_var in HF2Q_DECODE_MVN HF2Q_DECODE_MV_EXT; do
    route_value=${!route_var-}
    if [[ -n "$route_value" && "$route_value" != 0 && "$route_value" != 1 ]]; then
        echo "$route_var must be 0 or 1 when explicitly set" >&2
        exit 3
    fi
done
if [[ "$HF2Q_QWEN_SPECULATION" == auto && "${HF2Q_DECODE_MV_EXT:-0}" == 1 ]]; then
    echo "HF2Q_DECODE_MV_EXT=1 is not exact across scalar and width-four target routes; set QWEN38_SPECULATION=off for that experiment" >&2
    exit 3
fi

echo "Qwen3.8 exact speculation policy: $HF2Q_QWEN_SPECULATION" >&2
echo "Qwen3.8 K-quant routing: q5_canonical=$HF2Q_Q5K_CANONICAL_Q4X4 mvN=${HF2Q_DECODE_MVN:-native-default} mv_ext=${HF2Q_DECODE_MV_EXT:-native-default}" >&2

exec env HF2Q_Q5K_CANONICAL_Q4X4="$HF2Q_Q5K_CANONICAL_Q4X4" \
    "$script_dir/serve_qwen36_opencode.sh"
