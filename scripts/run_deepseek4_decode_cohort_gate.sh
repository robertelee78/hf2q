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

[[ -x "$test_binary" ]]
[[ -f "$model" ]]
[[ "$expected_source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$expected_model_sha" =~ ^[0-9a-f]{64}$ ]]
mkdir -p "$out_dir"

raw="$out_dir/raw.json"
test_log="$out_dir/test.log"
measurement_log="$out_dir/thermal.log"
settle_log="$out_dir/settle.log"
stop_file="$out_dir/thermal.stop"
rm -f "$raw" "$test_log" "$measurement_log" "$settle_log" "$stop_file"

monitor_pid=""
cleanup() {
  touch "$stop_file" 2>/dev/null || true
  if [[ -n "$monitor_pid" ]]; then
    kill -TERM "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

thermal_wait_for_nominal "$settle_log" decode-cohort-settle 60 900 5
: >"$measurement_log"
thermal_sample "$measurement_log" decode-cohort-measurement-start
thermal_monitor_nominal "$measurement_log" decode-cohort-measurement \
  "$stop_file" 2 &
monitor_pid=$!

test_rc=0
env -i \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  TMPDIR="${TMPDIR:-/tmp}" \
  HF2Q_DEEPSEEK4_GGUF="$model" \
  HF2Q_DEEPSEEK4_DECODE_COHORT_RECEIPT="$raw" \
  "$test_binary" official_artifact_b4_decode_body_is_exact_and_measured \
    --ignored --test-threads=1 --nocapture >"$test_log" 2>&1 || test_rc=$?
touch "$stop_file"
monitor_rc=0
wait "$monitor_pid" || monitor_rc=$?
monitor_pid=""
thermal_sample "$measurement_log" decode-cohort-measurement-end
test "$test_rc" = 0
test "$monitor_rc" = 0

thermal_validate_measurement_log "$measurement_log" 5
measurement_samples=$THERMAL_LOG_SAMPLES
measurement_duration_seconds=$THERMAL_LOG_DURATION_SECONDS
non_nominal_measurement_samples=$THERMAL_LOG_NON_NOMINAL_SAMPLES
measurement_gaps=$THERMAL_LOG_GAPS
thermal_validate_settle_log "$settle_log" 60 8
settle_samples=$THERMAL_LOG_SAMPLES
settle_duration_seconds=$THERMAL_LOG_DURATION_SECONDS
settle_gaps=$THERMAL_LOG_GAPS

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
jq --arg source_sha "$expected_source_sha" \
  --arg model_sha256 "$expected_model_sha" \
  --arg raw_sha256 "$(sha256_file "$raw")" \
  --arg test_log_sha256 "$(sha256_file "$test_log")" \
  --arg measurement_log_sha256 "$(sha256_file "$measurement_log")" \
  --arg settle_log_sha256 "$(sha256_file "$settle_log")" \
  --argjson settle_samples "$settle_samples" \
  --argjson settle_duration_seconds "$settle_duration_seconds" \
  --argjson settle_telemetry_gaps "$settle_gaps" \
  --argjson measurement_samples "$measurement_samples" \
  --argjson measurement_duration_seconds "$measurement_duration_seconds" \
  --argjson non_nominal_measurement_samples "$non_nominal_measurement_samples" \
  --argjson telemetry_gaps "$measurement_gaps" '
  . + {source_sha:$source_sha,model_sha256:$model_sha256,
    mlx_native_version:"0.10.12",raw_sha256:$raw_sha256,
    test_log_sha256:$test_log_sha256,thermal_status:"nominal",
    measurement_log_sha256:$measurement_log_sha256,
    settle_log_sha256:$settle_log_sha256,settle_seconds:60,
    settle_samples:$settle_samples,
    settle_duration_seconds:$settle_duration_seconds,
    settle_sample_interval_seconds:5,maximum_settle_sample_gap_seconds:8,
    settle_telemetry_gaps:$settle_telemetry_gaps,
    measurement_samples:$measurement_samples,
    measurement_duration_seconds:$measurement_duration_seconds,
    sample_interval_seconds:2,maximum_sample_gap_seconds:5,
    non_nominal_measurement_samples:$non_nominal_measurement_samples,
    telemetry_gaps:$telemetry_gaps}
' "$raw" >"$out_dir/summary.json.tmp"
mv "$out_dir/summary.json.tmp" "$out_dir/summary.json"
bash "$ROOT_DIR/scripts/verify_deepseek4_decode_cohort_receipt.sh" \
  "$out_dir/summary.json" "$raw" "$test_log" "$measurement_log" \
  "$settle_log" "$expected_source_sha" "$expected_model_sha"
sha256_file "$out_dir/summary.json" >"$out_dir/summary.json.sha256"
trap - EXIT INT TERM
cleanup
