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
# shellcheck source=scripts/macos_memory_guard.sh
source "$ROOT_DIR/scripts/macos_memory_guard.sh"

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
contention_measurement_log="$out_dir/measurement-contention.log"
contention_settle_log="$out_dir/settle-contention.log"
memory_log="$out_dir/memory-pressure.log"
rm -f "$raw" "$test_log" "$measurement_log" "$settle_log" \
  "$contention_measurement_log" "$contention_settle_log" "$memory_log"

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
memory_guard_source_sha=$(sha256_file "$ROOT_DIR/scripts/macos_memory_guard.sh")
thermal_wait_for_nominal "$settle_log" decode-cohort-settle 60 900 5 \
  "$contention_settle_log" "$$"
: >"$measurement_log"
: >"$contention_measurement_log"
thermal_sample "$measurement_log" decode-cohort-measurement-start
test "$THERMAL_STATE" = nominal
host_contention_sample "$contention_measurement_log" \
  decode-cohort-measurement-start "$$" "$THERMAL_SAMPLED_AT"
host_contention_require_quiet decode-cohort-measurement-start
: >"$memory_log"
memory_sample "$memory_log" decode-cohort-measurement-start
initial_swapouts=$MEMORY_SWAPOUTS

monitor_decode_measurement() {
  local producer_pid=$1
  local producer_state

  while :; do
    thermal_read_process_state "$producer_pid" || return 1
    producer_state=$THERMAL_PROCESS_STATE
    if [[ -z "$producer_state" || "$producer_state" == Z* ]]; then
      return 0
    fi
    thermal_sample "$measurement_log" decode-cohort-measurement || return 1
    host_contention_sample "$contention_measurement_log" \
      decode-cohort-measurement "$$" "$THERMAL_SAMPLED_AT" || return 1
    host_contention_require_quiet decode-cohort-measurement || return 1
    case "$THERMAL_STATE" in
      nominal|fair) ;;
      *)
        echo "decode-cohort measurement exceeded fair thermal state: $THERMAL_STATE" >&2
        return 1
        ;;
    esac
    memory_sample "$memory_log" decode-cohort-measurement \
      "$initial_swapouts" || return 1
    sleep 2
  done
}

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
monitor_decode_measurement "$test_pid"
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
host_contention_sample "$contention_measurement_log" \
  decode-cohort-measurement-end "$$" "$THERMAL_SAMPLED_AT"
host_contention_require_quiet decode-cohort-measurement-end
memory_sample "$memory_log" decode-cohort-measurement-end "$initial_swapouts"

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
host_contention_validate_measurement_log "$contention_measurement_log" 5
contention_measurement_samples=$HOST_CONTENTION_LOG_SAMPLES
contention_measurement_duration_seconds=$HOST_CONTENTION_LOG_DURATION_SECONDS
contention_measurement_contended_samples=$HOST_CONTENTION_LOG_CONTENDED_SAMPLES
contention_measurement_gaps=$HOST_CONTENTION_LOG_GAPS
host_contention_validate_settle_log "$contention_settle_log" 60 8
contention_settle_samples=$HOST_CONTENTION_LOG_SAMPLES
contention_settle_duration_seconds=$HOST_CONTENTION_LOG_DURATION_SECONDS
contention_settle_contended_samples=$HOST_CONTENTION_LOG_CONTENDED_SAMPLES
contention_settle_gaps=$HOST_CONTENTION_LOG_GAPS
host_contention_validate_thermal_alignment "$measurement_log" \
  "$contention_measurement_log"
host_contention_validate_thermal_alignment "$settle_log" \
  "$contention_settle_log"
memory_validate_normal_no_swapout_log "$memory_log" 5
memory_validate_measurement_coverage "$memory_log" "$measurement_log" 5
memory_samples=$MEMORY_LOG_SAMPLES
memory_duration_seconds=$MEMORY_LOG_DURATION_SECONDS
memory_initial_swapouts=$MEMORY_LOG_INITIAL_SWAPOUTS
memory_final_swapouts=$MEMORY_LOG_FINAL_SWAPOUTS
memory_swapout_delta=$MEMORY_LOG_SWAPOUT_DELTA
memory_min_free_percentage=$MEMORY_LOG_MIN_FREE_PERCENTAGE
memory_max_pressure_level=$MEMORY_LOG_MAX_PRESSURE_LEVEL
memory_max_throttled_pages=$MEMORY_LOG_MAX_THROTTLED_PAGES

jq --arg source_sha "$expected_source_sha" \
  --arg model_sha256 "$expected_model_sha" \
  --arg raw_sha256 "$(sha256_file "$raw")" \
  --arg test_log_sha256 "$(sha256_file "$test_log")" \
  --arg measurement_log_sha256 "$(sha256_file "$measurement_log")" \
  --arg settle_log_sha256 "$(sha256_file "$settle_log")" \
  --arg contention_policy "$HOST_CONTENTION_POLICY" \
  --arg contention_measurement_log_sha256 \
    "$(sha256_file "$contention_measurement_log")" \
  --arg contention_settle_log_sha256 \
    "$(sha256_file "$contention_settle_log")" \
  --arg memory_policy "$MEMORY_PRESSURE_POLICY" \
  --arg memory_log_sha256 "$(sha256_file "$memory_log")" \
  --arg memory_guard_source_sha256 "$memory_guard_source_sha" \
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
  --argjson telemetry_gaps "$measurement_gaps" \
  --argjson contention_settle_samples "$contention_settle_samples" \
  --argjson contention_settle_duration_seconds \
    "$contention_settle_duration_seconds" \
  --argjson contention_settle_contended_samples \
    "$contention_settle_contended_samples" \
  --argjson contention_settle_gaps "$contention_settle_gaps" \
  --argjson contention_measurement_samples "$contention_measurement_samples" \
  --argjson contention_measurement_duration_seconds \
    "$contention_measurement_duration_seconds" \
  --argjson contention_measurement_contended_samples \
    "$contention_measurement_contended_samples" \
  --argjson contention_measurement_gaps "$contention_measurement_gaps" \
  --argjson memory_normal_level "$MEMORY_PRESSURE_NORMAL_LEVEL" \
  --argjson memory_samples "$memory_samples" \
  --argjson memory_duration_seconds "$memory_duration_seconds" \
  --argjson memory_initial_swapouts "$memory_initial_swapouts" \
  --argjson memory_final_swapouts "$memory_final_swapouts" \
  --argjson memory_swapout_delta "$memory_swapout_delta" \
  --argjson memory_min_free_percentage "$memory_min_free_percentage" \
  --argjson memory_max_pressure_level "$memory_max_pressure_level" \
  --argjson memory_max_throttled_pages "$memory_max_throttled_pages" '
  . + {schema_version:4,source_sha:$source_sha,model_sha256:$model_sha256,
    mlx_native_version:"0.11.0",raw_sha256:$raw_sha256,
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
    telemetry_gaps:$telemetry_gaps,
    host_contention:{policy:$contention_policy,
      settle:{log_sha256:$contention_settle_log_sha256,
        samples:$contention_settle_samples,
        duration_seconds:$contention_settle_duration_seconds,
        contended_samples:$contention_settle_contended_samples,
        telemetry_gaps:$contention_settle_gaps},
      measurement:{log_sha256:$contention_measurement_log_sha256,
        samples:$contention_measurement_samples,
        duration_seconds:$contention_measurement_duration_seconds,
        contended_samples:$contention_measurement_contended_samples,
        telemetry_gaps:$contention_measurement_gaps}},
    memory_pressure:{policy:$memory_policy,normal_level:$memory_normal_level,
      log_sha256:$memory_log_sha256,
      guard_source_path:"scripts/macos_memory_guard.sh",
      guard_source_sha256:$memory_guard_source_sha256,
      sample_interval_seconds:2,maximum_sample_gap_seconds:5,
      samples:$memory_samples,duration_seconds:$memory_duration_seconds,
      initial_swapouts:$memory_initial_swapouts,
      final_swapouts:$memory_final_swapouts,
      swapout_delta:$memory_swapout_delta,
      min_free_percentage:$memory_min_free_percentage,
      max_pressure_level:$memory_max_pressure_level,
      max_throttled_pages:$memory_max_throttled_pages}}
' "$raw" >"$out_dir/summary.json.tmp"
mv "$out_dir/summary.json.tmp" "$out_dir/summary.json"
bash "$ROOT_DIR/scripts/verify_deepseek4_decode_cohort_receipt.sh" \
  "$out_dir/summary.json" "$raw" "$test_log" "$measurement_log" \
  "$settle_log" "$expected_source_sha" "$expected_model_sha" \
  "$contention_measurement_log" "$contention_settle_log" "$memory_log"
sha256_file "$out_dir/summary.json" >"$out_dir/summary.json.sha256"
