#!/usr/bin/env bash
# The literal shell expressions below are source-contract needles, not values
# that should expand while this test is running.
# shellcheck disable=SC2016
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
spec_runner="$root_dir/scripts/qwen38_speculation_ab.sh"
long_runner="$root_dir/scripts/qwen38_long_decode_ab.sh"
workflow="$root_dir/.github/workflows/cache-lifecycle.yml"

readonly paired_model_path='/opt/hf2q/models/qwen3.8/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf'
readonly paired_model_sha256='1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a'

grep -Fq "MODEL_PATH:-$paired_model_path" "$spec_runner" || {
  echo "Qwen3.8 speculation gate does not default to the canonical paired artifact" >&2
  exit 1
}
grep -Fq 'MODEL_SHA256=${MODEL_SHA256:?MODEL_SHA256 is required}' "$spec_runner" || {
  echo "Qwen3.8 speculation gate does not require an explicit artifact digest" >&2
  exit 1
}
if grep -Fq "$paired_model_sha256" "$spec_runner" "$long_runner"; then
  echo "Qwen3.8 runner bakes in a digest that will drift on rotation" >&2
  exit 1
fi
grep -Fq "ACCEPTED_QWEN38_MODEL_SHA256: \"$paired_model_sha256\"" "$workflow" || {
  echo "Cache lifecycle does not accept the canonical paired artifact digest" >&2
  exit 1
}
grep -Fq 'MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}' "$long_runner" || {
  echo "Qwen3.8 long-decode gate does not require an explicit artifact path" >&2
  exit 1
}
grep -Fq 'MODEL_SHA256=${MODEL_SHA256:?MODEL_SHA256 is required}' "$long_runner" || {
  echo "Qwen3.8 long-decode gate does not require an explicit artifact digest" >&2
  exit 1
}

for runner in "$spec_runner" "$long_runner"; do
  grep -Fq 'source "$script_dir/qwen36_watchdog_validate.sh"' "$runner" || {
    echo "Qwen3.8 runner bypasses the shared model-verification contract: $runner" >&2
    exit 1
  }
  grep -Fq 'hf2q_release_prepare_model_verification' "$runner" || {
    echo "Qwen3.8 standalone runner cannot reuse an unchanged-file receipt: $runner" >&2
    exit 1
  }
  [[ "$(grep -cF 'hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256"' "$runner")" -ge 2 ]] || {
    echo "Qwen3.8 runner does not verify artifact identity before and after inference: $runner" >&2
    exit 1
  }
  if grep -Fq 'sha256_file "$MODEL_PATH"' "$runner"; then
    echo "Qwen3.8 runner rereads the complete model instead of using its receipt: $runner" >&2
    exit 1
  fi
done

grep -Fq 'model_verification_mode=provided_receipt' "$spec_runner" || {
  echo "Qwen3.8 speculation receipt mislabels parent-provided verification" >&2
  exit 1
}

for artifact_consumer in \
  "$spec_runner" \
  "$long_runner" \
  "$root_dir/src/inference/models/qwen35/spec_decode.rs"; do
  if grep -Fq 'Qwen3.8-27B-Q4_K_M.gguf' "$artifact_consumer"; then
    echo "Qwen3.8 execution path still names the removed vanilla artifact: $artifact_consumer" >&2
    exit 1
  fi
done

printf 'qwen38 paired-artifact binding contract passed\n'
