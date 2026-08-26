#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/agentic_cache_lifecycle_contract.sh
source "$script_dir/agentic_cache_lifecycle_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

verify_exact_sidecar() {
  local payload=$1
  local sidecar=$2
  local payload_dir payload_name expected

  [[ -f "$payload" && -r "$payload" && ! -L "$payload" \
    && -f "$sidecar" && -r "$sidecar" && ! -L "$sidecar" ]] || return 1
  payload_dir=$(cd "$(dirname "$payload")" && pwd) || return 1
  payload_name=$(basename "$payload")
  [[ "$sidecar" == "$payload.sha256" \
    && "$(awk 'END { print NR }' "$sidecar")" == 1 ]] || return 1
  expected="$(sha256_file "$payload")  $payload_name"
  [[ "$(sed -n '1p' "$sidecar")" == "$expected" ]] || return 1
  (cd "$payload_dir" && shasum -a 256 -c "$payload_name.sha256" >/dev/null)
}

qwen38_validate_agentic_lifecycle_release_receipt() {
  local manifest=$1
  local manifest_sidecar=$2
  local lifecycle=$3
  local lifecycle_sidecar=$4
  local dependency_receipt=$5
  local expected_dependency_receipt_sha256=$6
  local model_verification_receipt=$7
  local expected_source_sha=$8
  local expected_crate_sha256=$9
  local expected_binary=${10}
  local expected_binary_sha256=${11}
  local expected_model=${12}
  local expected_model_sha256=${13}
  local lifecycle_sha dependency_sha model_bytes expected_run_id

  [[ "$expected_source_sha" =~ ^[0-9a-f]{40}$ \
    && "$expected_crate_sha256" =~ ^[0-9a-f]{64}$ \
    && "$expected_dependency_receipt_sha256" =~ ^[0-9a-f]{64}$ \
    && "$expected_binary_sha256" =~ ^[0-9a-f]{64}$ \
    && "$expected_model_sha256" =~ ^[0-9a-f]{64}$ \
    && -f "$expected_binary" && -x "$expected_binary" \
    && ! -L "$expected_binary" \
    && -f "$expected_model" && -r "$expected_model" \
    && -f "$model_verification_receipt" \
    && -r "$model_verification_receipt" \
    && ! -L "$model_verification_receipt" \
    && -f "$dependency_receipt" && -r "$dependency_receipt" \
    && ! -L "$dependency_receipt" ]] || return 1
  [[ "$(sha256_file "$expected_binary")" == "$expected_binary_sha256" ]] \
    || return 1
  hf2q_release_verify_model "$expected_model" "$expected_model_sha256" \
    "$model_verification_receipt" || return 1
  [[ "$(jq -s 'length' "$manifest")" == 1 \
    && "$(jq -s 'length' "$lifecycle")" == 1 \
    && "$(jq -s 'length' "$dependency_receipt")" == 1 ]] || return 1

  verify_exact_sidecar "$manifest" "$manifest_sidecar" || return 1
  verify_exact_sidecar "$lifecycle" "$lifecycle_sidecar" || return 1
  expected_run_id="release-${expected_source_sha:0:12}-qwen38"
  agentic_lifecycle_validate_summary "$lifecycle" "$expected_run_id" 2800 \
    "$expected_model_sha256" qwen35 qwen35 16 false || return 1

  lifecycle_sha=$(sha256_file "$lifecycle") || return 1
  dependency_sha=$(sha256_file "$dependency_receipt") || return 1
  [[ "$dependency_sha" == "$expected_dependency_receipt_sha256" ]] || return 1
  model_bytes=$(wc -c <"$expected_model" | tr -d '[:space:]') || return 1
  [[ "$model_bytes" =~ ^[1-9][0-9]*$ ]] || return 1

  jq -e \
    --arg source_sha "$expected_source_sha" \
    --arg crate_sha256 "$expected_crate_sha256" \
    --arg binary_sha256 "$expected_binary_sha256" \
    --arg model_path "$expected_model" \
    --arg model_sha256 "$expected_model_sha256" \
    --argjson model_bytes "$model_bytes" \
    --arg lifecycle_sha256 "$lifecycle_sha" \
    --arg dependency_sha256 "$dependency_sha" \
    --slurpfile lifecycle "$lifecycle" \
    --slurpfile dependency "$dependency_receipt" '
      .status == "pass"
      and .source_sha == $source_sha
      and .crate_sha256 == $crate_sha256
      and .binary_sha256 == $binary_sha256
      and .power_guarded_ac == true
      and (.power_event_snapshots_sha256 | test("^[0-9a-f]{64}$"))
      and .dependency_provenance == $dependency[0]
      and .models.qwen38
        == {path:$model_path,bytes:$model_bytes,sha256:$model_sha256}
      and .receipt_sha256.provenance.dependency == $dependency_sha256
      and .receipt_sha256.qwen38.lifecycle == $lifecycle_sha256
      and (.receipt_sha256.qwen38.long_decode | test("^[0-9a-f]{64}$"))
      and .families.qwen38.status == "pass"
      and .families.qwen38.lifecycle == $lifecycle[0]
      and .families.qwen38.long_decode.status == "pass"
      and .dependency_provenance.schema_version == 1
      and .dependency_provenance.status == "pass"
      and .dependency_provenance.package.source == "packed-crate"
      and .dependency_provenance.package.crate_sha256 == $crate_sha256
      and .dependency_provenance.build.cargo_target_checkout_disjoint == true
      and .dependency_provenance.build.rust_build_override_env_cleared == true
      and .dependency_provenance.dependency.name == "mlx-native"
      and (.dependency_provenance.dependency.version
        | test("^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"))
      and .dependency_provenance.dependency.requirement
        == ("=" + .dependency_provenance.dependency.version)
      and .dependency_provenance.dependency.source
        == "registry+https://github.com/rust-lang/crates.io-index"
      and (.dependency_provenance.dependency.checksum
        | test("^[0-9a-f]{64}$"))
    ' "$manifest" >/dev/null
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  if (($# != 13)); then
    echo "usage: $0 MANIFEST MANIFEST_SHA LIFECYCLE LIFECYCLE_SHA DEPENDENCY_RECEIPT DEPENDENCY_RECEIPT_SHA MODEL_VERIFICATION_RECEIPT SOURCE_SHA CRATE_SHA BINARY BINARY_SHA MODEL MODEL_SHA" >&2
    exit 2
  fi
  qwen38_validate_agentic_lifecycle_release_receipt "$@"
fi
