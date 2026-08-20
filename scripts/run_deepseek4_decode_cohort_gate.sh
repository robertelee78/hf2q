#!/usr/bin/env bash
set -euo pipefail

test_binary=${1:?prebuilt test binary is required}
model=${2:?DeepSeek model path is required}
out_dir=${3:?output directory is required}
expected_source_sha=${4:?expected source SHA is required}
expected_model_sha=${5:?expected model SHA-256 is required}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=scripts/macos_thermal_guard.sh
source "$ROOT_DIR/scripts/macos_thermal_guard.sh"

if [[ ${HF2Q_THERMAL_SWIFTC_BIN+x} || ${HF2Q_THERMAL_PROBE_BIN+x} \
  || ${HF2Q_THERMAL_PROBE_SOURCE+x} ]]; then
  echo "thermal probe overrides are reserved for isolated contract tests" >&2
  exit 2
fi
readonly HF2Q_THERMAL_SWIFTC_BIN=/usr/bin/swiftc
[[ -x "$HF2Q_THERMAL_SWIFTC_BIN" ]] || {
  echo "required system Swift compiler is unavailable: $HF2Q_THERMAL_SWIFTC_BIN" >&2
  exit 2
}

[[ -x "$test_binary" ]]
[[ -f "$model" ]]
[[ "$expected_source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$expected_model_sha" =~ ^[0-9a-f]{64}$ ]]
mkdir -p "$out_dir"
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

raw="$out_dir/raw.json"
test_log="$out_dir/test.log"
measurement_log="$out_dir/thermal.log"
settle_log="$out_dir/settle.log"
rm -f "$raw" "$test_log" "$measurement_log" "$settle_log"

test_pid=""
cleanup() {
  local cleanup_rc=0
  if [[ -n "$test_pid" ]]; then
    kill -TERM "$test_pid" 2>/dev/null || true
    wait "$test_pid" 2>/dev/null || true
  fi
  thermal_cleanup_probe || cleanup_rc=1
  return "$cleanup_rc"
}
on_exit() {
  local original_rc=$?
  trap - EXIT
  if ! cleanup && ((original_rc == 0)); then
    original_rc=1
  fi
  exit "$original_rc"
}
trap on_exit EXIT
trap 'exit 1' INT TERM

thermal_prepare_probe
thermal_probe_source_sha=$(sha256_file "$THERMAL_PROBE_SOURCE")
thermal_probe_compiler_sha=$(sha256_file "$THERMAL_PROBE_COMPILER")
thermal_probe_binary_sha=$(sha256_file "$THERMAL_PROBE_BIN")
thermal_wait_for_nominal "$settle_log" decode-cohort-settle 60 900 5
: >"$measurement_log"
thermal_sample "$measurement_log" decode-cohort-measurement-start
test "$THERMAL_STATE" = nominal

env -i \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  TMPDIR="${TMPDIR:-/tmp}" \
  HF2Q_DEEPSEEK4_GGUF="$model" \
  HF2Q_DEEPSEEK4_DECODE_COHORT_RECEIPT="$raw" \
  "$test_binary" official_artifact_b4_decode_body_is_exact_and_measured \
    --ignored --test-threads=1 --nocapture >"$test_log" 2>&1 &
test_pid=$!
set +e
thermal_rc=0
thermal_monitor_fair_or_better_while_pid "$measurement_log" \
  decode-cohort-measurement "$test_pid" 2
thermal_rc=$?
if ((thermal_rc != 0)); then
  kill -TERM "$test_pid" 2>/dev/null || true
fi
wait "$test_pid"
test_rc=$?
test_pid=""
set -e
test "$test_rc" = 0
test "$thermal_rc" = 0
thermal_sample "$measurement_log" decode-cohort-measurement-end
[[ "$THERMAL_STATE" == nominal || "$THERMAL_STATE" == fair ]]

thermal_validate_fair_or_better_measurement_log "$measurement_log" 5
measurement_samples=$THERMAL_LOG_SAMPLES
measurement_duration_seconds=$THERMAL_LOG_DURATION_SECONDS
non_nominal_measurement_samples=$THERMAL_LOG_NON_NOMINAL_SAMPLES
fair_measurement_samples=$THERMAL_LOG_FAIR_SAMPLES
over_limit_measurement_samples=$THERMAL_LOG_OVER_LIMIT_SAMPLES
measurement_gaps=$THERMAL_LOG_GAPS
thermal_validate_settle_log "$settle_log" 60 8
settle_samples=$THERMAL_LOG_SAMPLES
settle_duration_seconds=$THERMAL_LOG_DURATION_SECONDS
settle_gaps=$THERMAL_LOG_GAPS

jq --arg source_sha "$expected_source_sha" \
  --arg model_sha256 "$expected_model_sha" \
  --arg raw_sha256 "$(sha256_file "$raw")" \
  --arg test_log_sha256 "$(sha256_file "$test_log")" \
  --arg measurement_log_sha256 "$(sha256_file "$measurement_log")" \
  --arg settle_log_sha256 "$(sha256_file "$settle_log")" \
  --arg thermal_probe_source_sha256 "$thermal_probe_source_sha" \
  --arg thermal_probe_compiler_path "$THERMAL_PROBE_COMPILER" \
  --arg thermal_probe_compiler_sha256 "$thermal_probe_compiler_sha" \
  --arg thermal_probe_compiler_version "$THERMAL_PROBE_COMPILER_VERSION" \
  --arg thermal_probe_binary_sha256 "$thermal_probe_binary_sha" \
  --argjson settle_samples "$settle_samples" \
  --argjson settle_duration_seconds "$settle_duration_seconds" \
  --argjson settle_telemetry_gaps "$settle_gaps" \
  --argjson measurement_samples "$measurement_samples" \
  --argjson measurement_duration_seconds "$measurement_duration_seconds" \
  --argjson non_nominal_measurement_samples "$non_nominal_measurement_samples" \
  --argjson fair_measurement_samples "$fair_measurement_samples" \
  --argjson over_limit_measurement_samples "$over_limit_measurement_samples" \
  --argjson telemetry_gaps "$measurement_gaps" '
  . + {schema_version:2,source_sha:$source_sha,model_sha256:$model_sha256,
    mlx_native_version:"0.10.12",raw_sha256:$raw_sha256,
    test_log_sha256:$test_log_sha256,thermal_status:"fair_or_better",
    required_start_state:"nominal",maximum_measurement_state:"fair",
    measurement_log_sha256:$measurement_log_sha256,
    settle_log_sha256:$settle_log_sha256,settle_seconds:60,
    thermal_probe:{implementation:"compiled-foundation-helper",
      source_path:"scripts/macos_thermal_probe.swift",
      source_sha256:$thermal_probe_source_sha256,
      compiler_path:$thermal_probe_compiler_path,
      compiler_sha256:$thermal_probe_compiler_sha256,
      compiler_version:$thermal_probe_compiler_version,
      binary_sha256:$thermal_probe_binary_sha256},
    settle_samples:$settle_samples,
    settle_duration_seconds:$settle_duration_seconds,
    settle_sample_interval_seconds:5,maximum_settle_sample_gap_seconds:8,
    settle_telemetry_gaps:$settle_telemetry_gaps,
    measurement_samples:$measurement_samples,
    measurement_duration_seconds:$measurement_duration_seconds,
    sample_interval_seconds:2,maximum_sample_gap_seconds:5,
    non_nominal_measurement_samples:$non_nominal_measurement_samples,
    fair_measurement_samples:$fair_measurement_samples,
    over_limit_measurement_samples:$over_limit_measurement_samples,
    telemetry_gaps:$telemetry_gaps}
' "$raw" >"$out_dir/summary.json.tmp"
mv "$out_dir/summary.json.tmp" "$out_dir/summary.json"
bash "$ROOT_DIR/scripts/verify_deepseek4_decode_cohort_receipt.sh" \
  "$out_dir/summary.json" "$raw" "$test_log" "$measurement_log" \
  "$settle_log" "$expected_source_sha" "$expected_model_sha"
sha256_file "$out_dir/summary.json" >"$out_dir/summary.json.sha256"
