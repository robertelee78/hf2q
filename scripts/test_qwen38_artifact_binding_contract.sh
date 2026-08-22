#!/usr/bin/env bash
# The literal shell expressions below are source-contract needles, not values
# that should expand while this test is running.
# shellcheck disable=SC2016
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
spec_runner="$root_dir/scripts/qwen38_speculation_ab.sh"
long_runner="$root_dir/scripts/qwen38_long_decode_ab.sh"
matched_runner="$root_dir/scripts/qwen38_matched_reference_abba.sh"
matched_contract="$root_dir/scripts/qwen38_matched_reference_contract.sh"
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
if ! grep -Fq 'MODEL_ID=${MODEL_ID:-}' "$spec_runner" \
  || ! grep -Fq 'resolve_loaded_model_id' "$spec_runner"; then
  echo "Qwen3.8 speculation runner still assumes a stale display model id" >&2
  exit 1
fi

for needle in \
  'HF2Q_SHA256=${HF2Q_SHA256:?HF2Q_SHA256 is required}' \
  'MIN_HF2Q_RATIO must be >= 1.0' \
  "readonly QUALIFIED_MODEL_SHA256='4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e'" \
  'readonly THERMAL_SETTLE_SECONDS=30' \
  'verify_executable_identity hf2q' \
  'verify_executable_identity reference' \
  'for reference_trial in 2 3' \
  'run_stream_ttft' \
  'contract_sha256:$contract_sha' \
  'request_manifest_sha256' \
  'evidence_manifest_sha256' \
  'required_process_state:"quiet"'; do
  grep -Fq "$needle" "$matched_runner" || {
    echo "matched Q5_K_M runner lost required evidence contract: $needle" >&2
    exit 1
  }
done
grep -Fq '.status.value == "loaded"' "$matched_contract" || {
  echo "matched Q5_K_M contract lost the reference loaded-state parser" >&2
  exit 1
}
[[ "$(grep -cF 'hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256"' \
  "$matched_runner")" -ge 2 ]] || {
  echo "matched Q5_K_M runner does not revalidate model identity" >&2
  exit 1
}
if grep -Eq '^THERMAL_(SETTLE|SAMPLE).*\$\{THERMAL_' "$matched_runner"; then
  echo "matched Q5_K_M calibration timings became caller-overridable" >&2
  exit 1
fi

for artifact_consumer in \
  "$spec_runner" \
  "$long_runner" \
  "$matched_runner" \
  "$root_dir/src/inference/models/qwen35/spec_decode.rs"; do
  if grep -Fq 'Qwen3.8-27B-Q4_K_M.gguf' "$artifact_consumer"; then
    echo "Qwen3.8 execution path still names the removed vanilla artifact: $artifact_consumer" >&2
    exit 1
  fi
done

printf 'qwen38 paired-artifact binding contract passed\n'
