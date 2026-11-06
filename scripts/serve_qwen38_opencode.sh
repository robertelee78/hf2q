#!/usr/bin/env bash
# Canonical Qwen3.8-27B launcher for OpenAI-compatible coding clients.
#
# Qwen3.8 is multimodal. The default requires both the text GGUF and its
# source-matched projector; startup fails closed if either side is missing or
# incompatible. Set QWEN38_VISION=off only for an intentional text-only run.
# Exact target-verified speculation is off by default at MAX_SLOTS>=4 (the
# serve_qwen36_opencode.sh default is 4) and on below that. `auto` tries
# request-history lookup before an available MTP head and drops either
# proposer for the generation after two consecutive four-round windows that
# are not better than ordinary decode. Under the default repeatable route
# (HF2Q_QWEN_SPECULATION_ROUTE) those windows are judged against a fixed cost
# model and a speculating slot never joins a joint step, so identical
# requests take identical routes. Unsupported request semantics remain on
# ordinary target decode.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-/opt/hf2q/models/qwen3.8/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf}"
export MMPROJ="${MMPROJ:-${MODEL%.gguf}-mmproj.gguf}"
export VISION_MODE="${QWEN38_VISION:-required}"
# An unset HF2Q_QWEN_SPECULATION resolves per worker in the engine: `off` on
# joint-decode workers at MAX_SLOTS>=4 (Qwen3.8: ADR-063 D1 measured a
# four-agent makespan loss for `auto`; Qwen3.6 likewise, for repeatable
# four-agent serving), `auto` below four slots. QWEN38_SPECULATION=auto opts
# back in and QWEN38_SPECULATION=off forces it off; the explicit value reaches
# the engine only when the operator sets QWEN38_SPECULATION (ADR-062 D1).
if [[ -n "${QWEN38_SPECULATION:-}" ]]; then
    export HF2Q_QWEN_SPECULATION="$QWEN38_SPECULATION"
fi
# Qwen3.8 K=3 verifies four target positions at once. The native qL4
# decision/cache gate and matched ABBA receipt qualify the weight-amortized
# K-quant width-four route here. These two are genuinely non-default
# mlx-native routing choices (MVN default-on, MV_EXT default-off), so the
# launcher keeps setting them; process-wide defaults remain unchanged
# outside this family launcher.
export HF2Q_DECODE_MVN="${HF2Q_DECODE_MVN:-0}"
export HF2Q_DECODE_MV_EXT="${HF2Q_DECODE_MV_EXT:-1}"

case "${QWEN38_SPECULATION:-}" in
    ''|off|auto) ;;
    *)
        echo "QWEN38_SPECULATION must be off or auto (got: $QWEN38_SPECULATION)" >&2
        exit 3
        ;;
esac
case "$HF2Q_DECODE_MVN:$HF2Q_DECODE_MV_EXT" in
    0:1|0:0|1:0|1:1) ;;
    *)
        echo "HF2Q_DECODE_MVN and HF2Q_DECODE_MV_EXT must each be 0 or 1" >&2
        exit 3
        ;;
esac

echo "Qwen3.8 exact speculation policy: ${QWEN38_SPECULATION:-engine default (off at MAX_SLOTS>=4, auto below)}" >&2
echo "Qwen3.8 K-quant width routing: mvN=$HF2Q_DECODE_MVN mv_ext=$HF2Q_DECODE_MV_EXT" >&2

exec "$script_dir/serve_qwen36_opencode.sh"
