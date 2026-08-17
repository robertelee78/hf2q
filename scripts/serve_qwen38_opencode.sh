#!/usr/bin/env bash
# Canonical Qwen3.8-27B launcher for OpenAI-compatible coding clients.
#
# Qwen3.8 is multimodal. The default requires both the text GGUF and its
# source-matched projector; startup fails closed if either side is missing or
# incompatible. Set QWEN38_VISION=off only for an intentional text-only run.
# Speculative decode remains disabled until the optional MTP path has its own
# real-artifact acceptance gate; ordinary autoregressive decode is supported.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-/opt/hf2q/models/qwen3.8/Qwen3.8-27B-Q4_K_M.gguf}"
export MMPROJ="${MMPROJ:-/opt/hf2q/models/qwen3.8/Qwen3.8-27B-mmproj-F16.gguf}"
export VISION_MODE="${QWEN38_VISION:-required}"
export HF2Q_SPEC_DECODE="${HF2Q_SPEC_DECODE:-0}"

exec "$script_dir/serve_qwen36_opencode.sh"
