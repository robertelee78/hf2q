#!/usr/bin/env bash
# Canonical text-only Qwen3.8-27B launcher for OpenAI-compatible coding clients.
#
# Qwen3.8 uses hf2q's native dense Qwen execution family and the same bounded,
# four-slot scheduling contract as the established Qwen launcher. Speculative
# decode remains disabled until the model's optional MTP path has its own
# real-artifact acceptance gate; ordinary autoregressive decode is supported.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-/opt/hf2q/models/qwen3.8/Qwen3.8-27B-Q4_K_M.gguf}"
export HF2Q_SPEC_DECODE="${HF2Q_SPEC_DECODE:-0}"

exec "$script_dir/serve_qwen36_opencode.sh"
