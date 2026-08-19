#!/usr/bin/env bash
set -euo pipefail

summary=${1:?summary path is required}
raw=${2:?raw receipt path is required}
test_log=${3:?test log path is required}
measurement_log=${4:?measurement log path is required}
settle_log=${5:?settle log path is required}
expected_source_sha=${6:?expected source SHA is required}
expected_model_sha=${7:?expected model SHA-256 is required}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=scripts/macos_thermal_guard.sh
source "$ROOT_DIR/scripts/macos_thermal_guard.sh"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

for path in "$summary" "$raw" "$test_log" "$measurement_log" "$settle_log"; do
  [[ -s "$path" ]] || {
    echo "decode-cohort receipt input is missing or empty: $path" >&2
    exit 1
  }
done
[[ "$expected_source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$expected_model_sha" =~ ^[0-9a-f]{64}$ ]]

test "$(sha256_file "$raw")" = "$(jq -er .raw_sha256 "$summary")"
test "$(sha256_file "$test_log")" = "$(jq -er .test_log_sha256 "$summary")"
test "$(sha256_file "$measurement_log")" = \
  "$(jq -er .measurement_log_sha256 "$summary")"
test "$(sha256_file "$settle_log")" = \
  "$(jq -er .settle_log_sha256 "$summary")"
jq -s -e 'length == 1' "$summary" >/dev/null

jq -e --slurpfile raw "$raw" \
  --arg source_sha "$expected_source_sha" \
  --arg model_sha256 "$expected_model_sha" '
  def abs: if . < 0 then -. else . end;
  . as $summary
  | $raw[0] as $receipt
  | ($receipt.benchmark.serial_ms | sort | ((.[4] + .[5]) / 2)) as $serial_median
  | ($receipt.benchmark.cohort_ms | sort | ((.[4] + .[5]) / 2)) as $cohort_median
  | ($raw | length) == 1
    and ($summary | del(
      .source_sha,.model_sha256,.mlx_native_version,.raw_sha256,
      .test_log_sha256,.thermal_status,.measurement_log_sha256,
      .settle_log_sha256,.settle_seconds,.settle_samples,
      .settle_duration_seconds,.settle_sample_interval_seconds,
      .maximum_settle_sample_gap_seconds,.settle_telemetry_gaps,
      .measurement_samples,.measurement_duration_seconds,
      .sample_interval_seconds,.maximum_sample_gap_seconds,
      .non_nominal_measurement_samples,.telemetry_gaps
    )) == $receipt
    and .schema_version == 1 and .status == "pass"
    and .source_sha == $source_sha and .model_sha256 == $model_sha256
    and .mlx_native_version == "0.10.12"
    and .artifact_bytes == 107431343168 and .layers == 43 and .lanes == 4
    and .parity == {
      prefix_rows:148,steps:132,final_position:280,final_mod_4:0,
      final_mod_128:24,physical_to_logical:[2,0,3,1],
      exact_state_logits_cache_recurrent:true
    }
    and .benchmark.position == 6676
    and .benchmark.logical_capacity == 131072
    and .benchmark.loaded_idle_seconds == 45
    and .benchmark.pairs == 10 and .benchmark.order == "alternating"
    and (.benchmark.serial_ms | length) == 10
    and (.benchmark.cohort_ms | length) == 10
    and all(.benchmark.serial_ms[],.benchmark.cohort_ms[];
      type == "number" and isfinite and . > 0)
    and .benchmark.serial_median_ms == $serial_median
    and .benchmark.cohort_median_ms == $cohort_median
    and ((.benchmark.speedup - ($serial_median / $cohort_median)) | abs) < 1e-12
    and $serial_median > $cohort_median and .benchmark.speedup > 1
    and .benchmark.serial_command_buffers_per_pair == 92
    and .benchmark.cohort_command_buffers_per_pair == 23
    and .benchmark.serial_synchronizations_per_pair == 4
    and .benchmark.cohort_synchronizations_per_pair == 1
    and (.benchmark.serial_counters | length) == 10
    and (.benchmark.cohort_counters | length) == 10
    and all(.benchmark.serial_counters[];
      .command_buffers == 92 and .synchronizations == 4
      and .dispatches > 0 and .barriers > 0)
    and all(.benchmark.cohort_counters[];
      .command_buffers == 23 and .synchronizations == 1
      and .dispatches > 0 and .barriers > 0)
    and .benchmark_environment == {
      profile:"clean-hf2q-mlx-metal-v1",override_variables_absent:true,
      unexpected_override_variables:[]
    }
    and .thermal_status == "nominal"
    and .settle_seconds == 60
    and .settle_sample_interval_seconds == 5
    and .maximum_settle_sample_gap_seconds == 8
    and .sample_interval_seconds == 2
    and .maximum_sample_gap_seconds == 5
    and (.settle_samples | type) == "number" and .settle_samples > 0
    and (.settle_duration_seconds | type) == "number"
    and .settle_duration_seconds >= .settle_seconds
    and (.measurement_samples | type) == "number" and .measurement_samples >= 2
    and (.measurement_duration_seconds | type) == "number"
    and .measurement_duration_seconds > 0
    and .non_nominal_measurement_samples == 0
    and .settle_telemetry_gaps == 0 and .telemetry_gaps == 0
  ' "$summary" >/dev/null

rg -Fq 'official_artifact_b4_decode_body_is_exact_and_measured ... ok' "$test_log"
rg -Fq 'exact_state_logits_cache_recurrent=true' "$test_log"

thermal_validate_measurement_log "$measurement_log" 5
test "$THERMAL_LOG_SAMPLES" = "$(jq -er .measurement_samples "$summary")"
test "$THERMAL_LOG_DURATION_SECONDS" = \
  "$(jq -er .measurement_duration_seconds "$summary")"
test "$THERMAL_LOG_NON_NOMINAL_SAMPLES" = \
  "$(jq -er .non_nominal_measurement_samples "$summary")"
test "$THERMAL_LOG_GAPS" = "$(jq -er .telemetry_gaps "$summary")"
test "$(head -1 "$measurement_log" | awk -F '\t' '{print $3}')" = \
  decode-cohort-measurement-start
test "$(tail -1 "$measurement_log" | awk -F '\t' '{print $3}')" = \
  decode-cohort-measurement-end
awk -F '\t' 'NR > 1 && $3 != "decode-cohort-measurement" && \
  $3 != "decode-cohort-measurement-end" { exit 1 }' "$measurement_log"

thermal_validate_settle_log "$settle_log" 60 8
test "$THERMAL_LOG_SAMPLES" = "$(jq -er .settle_samples "$summary")"
test "$THERMAL_LOG_DURATION_SECONDS" = \
  "$(jq -er .settle_duration_seconds "$summary")"
test "$THERMAL_LOG_GAPS" = "$(jq -er .settle_telemetry_gaps "$summary")"
awk -F '\t' '$3 != "decode-cohort-settle" { exit 1 }' "$settle_log"

echo "DeepSeek-V4 decode-cohort receipt verified" >&2
