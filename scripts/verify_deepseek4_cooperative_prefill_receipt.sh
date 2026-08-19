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
    echo "cooperative prefill receipt input is missing or empty: $path" >&2
    exit 1
  }
done
[[ "$expected_source_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "invalid expected source SHA: $expected_source_sha" >&2
  exit 2
}
[[ "$expected_model_sha" =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid expected model SHA-256: $expected_model_sha" >&2
  exit 2
}

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
  | $raw[0] as $raw_receipt
  | ($raw_receipt.benchmark.serial_ms | sort | .[2]) as $serial_median
  | ($raw_receipt.benchmark.cohort_ms | sort | .[2]) as $cohort_median
  | ($raw | length) == 1
    and ($summary | del(
      .source_sha,
      .model_sha256,
      .mlx_native_version,
      .raw_sha256,
      .test_log_sha256,
      .thermal_status,
      .measurement_log_sha256,
      .settle_log_sha256,
      .settle_seconds,
      .settle_samples,
      .settle_duration_seconds,
      .settle_sample_interval_seconds,
      .maximum_settle_sample_gap_seconds,
      .settle_telemetry_gaps,
      .measurement_samples,
      .measurement_duration_seconds,
      .sample_interval_seconds,
      .maximum_sample_gap_seconds,
      .non_nominal_measurement_samples,
      .telemetry_gaps
    )) == $raw_receipt
    and .schema_version == 1 and .status == "pass"
    and .source_sha == $source_sha and .model_sha256 == $model_sha256
    and .mlx_native_version == "0.10.12"
    and .artifact_bytes == 107431343168 and .layers == 43
    and .prefix_rows == 148 and .prefix_mod_128 == 20 and .prefix_mod_4 == 0
    and [.parity_shapes[] | [.sequences,.rows_per_lane,.aggregate_rows]]
      == [[2,1024,2048],[3,640,1920],[4,512,2048]]
    and all(.parity_shapes[]; .exact_state_logits_decode == true)
    and .benchmark.sequences == 4 and .benchmark.rows_per_lane == 512
    and .benchmark.aggregate_rows == 2048 and .benchmark.pairs == 5
    and .benchmark.order == "alternating"
    and (.benchmark.serial_ms | length) == 5
    and (.benchmark.cohort_ms | length) == 5
    and all(.benchmark.serial_ms[], .benchmark.cohort_ms[];
      type == "number" and . > 0)
    and .benchmark.serial_median_ms == $serial_median
    and .benchmark.cohort_median_ms == $cohort_median
    and ((.benchmark.speedup - ($serial_median / $cohort_median)) | abs) < 1e-12
    and $serial_median > $cohort_median and .benchmark.speedup > 1
    and .benchmark.process_lifetime_peak_rss_bytes > 0
    and .benchmark_environment == {
      profile:"clean-hf2q-mlx-metal-v1",
      override_variables_absent:true,
      unexpected_override_variables:[],
      pairs:5
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

thermal_validate_measurement_log "$measurement_log" 5
test "$THERMAL_LOG_SAMPLES" = "$(jq -er .measurement_samples "$summary")"
test "$THERMAL_LOG_DURATION_SECONDS" = \
  "$(jq -er .measurement_duration_seconds "$summary")"
test "$THERMAL_LOG_NON_NOMINAL_SAMPLES" = \
  "$(jq -er .non_nominal_measurement_samples "$summary")"
test "$THERMAL_LOG_GAPS" = "$(jq -er .telemetry_gaps "$summary")"
test "$(head -1 "$measurement_log" | awk -F '\t' '{print $3}')" = \
  cooperative-prefill-measurement-start
test "$(tail -1 "$measurement_log" | awk -F '\t' '{print $3}')" = \
  cooperative-prefill-measurement-end
awk -F '\t' 'NR > 1 && $3 != "cooperative-prefill-measurement" && \
  $3 != "cooperative-prefill-measurement-end" { exit 1 }' "$measurement_log"

thermal_validate_settle_log "$settle_log" 60 8
test "$THERMAL_LOG_SAMPLES" = "$(jq -er .settle_samples "$summary")"
test "$THERMAL_LOG_DURATION_SECONDS" = \
  "$(jq -er .settle_duration_seconds "$summary")"
test "$THERMAL_LOG_GAPS" = "$(jq -er .settle_telemetry_gaps "$summary")"
awk -F '\t' '$3 != "cooperative-prefill-settle" { exit 1 }' "$settle_log"

echo "DeepSeek-V4 cooperative prefill receipt verified" >&2
