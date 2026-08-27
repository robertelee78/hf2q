#!/usr/bin/env bash
# The literal shell expressions below are source-contract needles, not values
# that should expand while this test is running.
# shellcheck disable=SC2016
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
spec_runner="$root_dir/scripts/qwen38_speculation_ab.sh"
long_runner="$root_dir/scripts/qwen38_long_decode_ab.sh"
matched_runner="$root_dir/scripts/qwen38_matched_reference_abba.sh"
physical_runner="$root_dir/scripts/qwen38_physical_multislot_gate.sh"
physical_matrix_runner="$root_dir/scripts/qwen38_physical_multislot_matrix.sh"
exact_swap_runner="$root_dir/scripts/run_qwen38_exact_swap_matrix.sh"
matched_contract="$root_dir/scripts/qwen38_matched_reference_contract.sh"
artifact_contract="$root_dir/scripts/qwen38_artifact_contract.sh"
workflow="$root_dir/.github/workflows/cache-lifecycle.yml"
ci_workflow="$root_dir/.github/workflows/ci.yml"

readonly paired_model_path='/opt/hf2q/models/qwen3.8/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf'
readonly paired_model_sha256='1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a'

verify_blocking_contract_suites() {
  local workflow_path=$1 suite count
  for suite in \
    test_qwen38_artifact_matrix_contract.sh \
    test_qwen38_physical_multislot_gate_contract.sh \
    test_qwen38_matched_physical_contract.sh \
    test_generative_swap_matrix_contract.sh; do
    count=$(sed 's/^[[:space:]]*//' "$workflow_path" \
      | grep -cxF "bash scripts/$suite" || true)
    [[ "$count" == 1 ]] || {
      echo "blocking CI must execute exactly one $suite" >&2
      return 1
    }
  done
}

verify_blocking_contract_suites "$ci_workflow"
ci_fixture_dir=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-ci-contract-matrix.XXXXXX")
cleanup_ci_fixture() {
  case "$ci_fixture_dir" in
    "${TMPDIR:-/tmp}"/hf2q-ci-contract-matrix.*)
      rm -rf -- "$ci_fixture_dir"
      ;;
    *)
      echo "refusing unsafe CI fixture cleanup: $ci_fixture_dir" >&2
      ;;
  esac
}
trap cleanup_ci_fixture EXIT
for suite in \
  test_qwen38_artifact_matrix_contract.sh \
  test_qwen38_physical_multislot_gate_contract.sh \
  test_qwen38_matched_physical_contract.sh \
  test_generative_swap_matrix_contract.sh; do
  mutated_ci="$ci_fixture_dir/missing-$suite.yml"
  grep -Fv "bash scripts/$suite" "$ci_workflow" > "$mutated_ci"
  if verify_blocking_contract_suites "$mutated_ci" >/dev/null 2>&1; then
    echo "blocking-CI mutation survived removal of $suite" >&2
    exit 1
  fi
done

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
  grep -Fq 'HF2Q_MODEL_VERIFICATION_BINARY="$BINARY_PATH"' "$runner" || {
    echo "Qwen3.8 runner does not make the sealed binary the v2 receipt authority: $runner" >&2
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

grep -Fq 'hf2q_release_materialize_model_verification' "$spec_runner" || {
  echo "Qwen3.8 speculation runner can pass a v1 receipt through to startup" >&2
  exit 1
}
if ! grep -Fq 'MODEL_ID=${MODEL_ID:-}' "$spec_runner" \
  || ! grep -Fq 'resolve_loaded_model_id' "$spec_runner"; then
  echo "Qwen3.8 speculation runner still assumes a stale display model id" >&2
  exit 1
fi
grep -Fq 'HF2Q_Q5K_CANONICAL_Q4X4="$Q5K_CANONICAL_Q4X4"' "$spec_runner" || {
  echo "Qwen3.8 speculation gate does not bind its recorded Q5_K route" >&2
  exit 1
}
grep -Fq 'dense_q5k_canonical_q4x4:$q5k_canonical_q4x4' "$spec_runner" || {
  echo "Qwen3.8 speculation receipt omits the Q5_K route" >&2
  exit 1
}
grep -Fq 'dense_q5k_canonical_q4x4=(true|false)' "$spec_runner" || {
  echo "Qwen3.8 speculation gate does not reopen the frozen Q5_K route" >&2
  exit 1
}

for needle in \
  'HF2Q_SHA256=${HF2Q_SHA256:?HF2Q_SHA256 is required}' \
  'MIN_HF2Q_RATIO must be >= 1.0' \
  'readonly MIN_SUSTAINED_WARMUP_TOKENS=512' \
  'readonly MAX_WITHIN_ENGINE_GROUP_SPREAD_PERCENT=5' \
  'readonly MAX_WITHIN_ENGINE_CASE_SPREAD_PERCENT=10' \
  'MODEL_FORMAT=${MODEL_FORMAT:?MODEL_FORMAT is required}' \
  'source "$SCRIPT_DIR/qwen38_artifact_contract.sh"' \
  'qwen38_validate_artifact_identity "$MODEL_FORMAT"' \
  'qwen38_validate_pinned_peer_commit "$REFERENCE_COMMIT"' \
  'readonly THERMAL_SETTLE_SECONDS=120' \
  'verify_executable_identity hf2q' \
  'verify_executable_identity reference' \
  'for reference_trial in 2 3' \
  'run_stream_ttft' \
  'contract_sha256:$contract_sha' \
  'artifact_contract_sha256:$artifact_contract_sha' \
  'request_manifest_sha256' \
  'evidence_manifest_sha256' \
  'required_energy_mode:"automatic-or-high"' \
  'required_process_state:"quiet"' \
  'matched_measurement_stability_json "$rows_file"' \
  '.observed_band_dominance == true' \
  'sustained_warmup:$warmup_diagnostics'; do
  grep -Fq "$needle" "$matched_runner" || {
    echo "matched Q5_K_M runner lost required evidence contract: $needle" >&2
    exit 1
  }
done

# Physical and matched matrix gates pass one v2 receipt per artifact directly
# to the server.  A second hidden recorder or a raw catalog `shasum` would
# make the startup/swap improvement illusory on the largest GGUFs.
for runner in "$physical_runner" "$physical_matrix_runner"; do
  grep -Fq 'HF2Q_MODEL_VERIFICATION_BINARY="$BINARY_PATH"' "$runner" || {
    echo "physical Qwen3.8 runner lacks a v2 recorder authority: $runner" >&2
    exit 1
  }
done
grep -Fq 'HF2Q_MODEL_VERIFICATION_RECEIPT="$model_verification_receipt"' \
  "$physical_runner" || {
  echo "physical Qwen3.8 gate does not pass its single v2 receipt to the server" >&2
  exit 1
}
if grep -Fq 'model-verification-runtime-v2.json' "$physical_runner" \
  || grep -Fq '__record-model-verification' "$physical_runner"; then
  echo "physical Qwen3.8 gate still creates a duplicate runtime receipt" >&2
  exit 1
fi
if grep -Fq 'actual_sha256=$(shasum -a 256 "$model_path"' "$physical_matrix_runner"; then
  echo "physical Qwen3.8 matrix still shell-hashes catalog artifacts before v2 recording" >&2
  exit 1
fi
for needle in \
  'source "$script_dir/qwen36_watchdog_validate.sh"' \
  'hf2q_release_prepare_model_verification "$model_path" "$expected_sha"' \
  'HF2Q_MODEL_VERIFICATION_RECEIPT_DIR="$OUT_DIR/preflight"'; do
  grep -Fq "$needle" "$exact_swap_runner" || {
    echo "exact-swap runner lost preverified-directory contract: $needle" >&2
    exit 1
  }
done
if grep -Fq 'actual_sha=$(shasum -a 256 "$model_path"' "$exact_swap_runner"; then
  echo "exact-swap runner still hashes each complete artifact outside the v2 authority" >&2
  exit 1
fi
for format in BF16 Q4_K_M Q5_K_M Q6_K Q8_0; do
  qwen38_record=$(bash -c \
    'source "$1"; qwen38_artifact_record "$2"' _ \
    "$artifact_contract" "$format")
  [[ "$qwen38_record" == "$format"$'\t'* ]] || {
    echo "shared artifact contract is missing $format" >&2
    exit 1
  }
done
grep -Fq '.status.value == "loaded"' "$matched_contract" || {
  echo "matched artifact contract lost the reference loaded-state parser" >&2
  exit 1
}
[[ "$(grep -cF 'hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256"' \
  "$matched_runner")" -ge 2 ]] || {
  echo "matched artifact runner does not revalidate model identity" >&2
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
