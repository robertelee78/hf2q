#!/usr/bin/env bash
set -euo pipefail

: "${HF2Q_QWEN38_GGUF:?set HF2Q_QWEN38_GGUF to the Q5-admitted Qwen3.8 GGUF}"
: "${HF2Q_ADR065_E_Q5_CASE:?set HF2Q_ADR065_E_Q5_CASE to the frozen Q5 case JSON}"
: "${HF2Q_ADR065_E_OUTPUT:?set HF2Q_ADR065_E_OUTPUT to the retained report path}"

export HF2Q_ADR065_E_GRID="${HF2Q_ADR065_E_GRID:-smoke}"
export HF2Q_ADR065_E_SOURCE_COMMIT="${HF2Q_ADR065_E_SOURCE_COMMIT:-$(git rev-parse HEAD)}"

cargo test --release --locked --bin hf2q \
  qwen38_joint_verify_diagnostic_grid -- --ignored --nocapture --test-threads=1
