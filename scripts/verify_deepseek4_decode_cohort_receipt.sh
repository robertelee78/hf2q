#!/usr/bin/env bash
set -euo pipefail

summary=${1:?summary path is required}
raw=${2:?raw receipt path is required}
test_log=${3:?test log path is required}
measurement_log=${4:?measurement log path is required}
settle_log=${5:?settle log path is required}
expected_source_sha=${6:?expected source SHA is required}
expected_model_sha=${7:?expected model SHA-256 is required}
contention_measurement_log=${8:?contention measurement log is required}
contention_settle_log=${9:?contention settle log is required}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=scripts/macos_thermal_guard.sh
source "$ROOT_DIR/scripts/macos_thermal_guard.sh"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

for path in "$summary" "$raw" "$test_log" "$measurement_log" "$settle_log" \
  "$contention_measurement_log" "$contention_settle_log"; do
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
test "$(sha256_file "$contention_measurement_log")" = \
  "$(jq -er .host_contention.measurement.log_sha256 "$summary")"
test "$(sha256_file "$contention_settle_log")" = \
  "$(jq -er .host_contention.settle.log_sha256 "$summary")"
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
      .schema_version,
      .source_sha,.model_sha256,.mlx_native_version,.raw_sha256,
      .test_log_sha256,.thermal_status,.required_start_state,
      .maximum_measurement_state,.measurement_log_sha256,
      .settle_log_sha256,.settle_seconds,.settle_samples,
      .thermal_probe,
      .settle_duration_seconds,.settle_sample_interval_seconds,
      .maximum_settle_sample_gap_seconds,.settle_telemetry_gaps,
      .measurement_samples,.measurement_duration_seconds,
      .sample_interval_seconds,.maximum_sample_gap_seconds,
      .non_nominal_measurement_samples,.fair_measurement_samples,
      .over_limit_measurement_samples,.telemetry_gaps,.host_contention
    )) == ($receipt | del(.schema_version))
    and $receipt.schema_version == 1
    and .schema_version == 3 and .status == "pass"
    and .source_sha == $source_sha and .model_sha256 == $model_sha256
    and .mlx_native_version == "0.11.2"
    and .thermal_probe.implementation == "compiled-foundation-helper"
    and .thermal_probe.source_path == "scripts/macos_thermal_probe.swift"
    and (.thermal_probe.source_sha256 | test("^[0-9a-f]{64}$"))
    and .thermal_probe.compiler_path == "/usr/bin/swiftc"
    and (.thermal_probe.compiler_sha256 | test("^[0-9a-f]{64}$"))
    and (.thermal_probe.compiler_version | type) == "string"
    and (.thermal_probe.compiler_version | length) > 0
    and (.thermal_probe.binary_sha256 | test("^[0-9a-f]{64}$"))
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
    and .thermal_status == "fair_or_better"
    and .required_start_state == "nominal"
    and .maximum_measurement_state == "fair"
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
    and (.non_nominal_measurement_samples | type) == "number"
    and (.fair_measurement_samples | type) == "number"
    and (.over_limit_measurement_samples | type) == "number"
    and .non_nominal_measurement_samples >= 0
    and .fair_measurement_samples >= 0
    and .over_limit_measurement_samples == 0
    and .non_nominal_measurement_samples == .fair_measurement_samples
    and .settle_telemetry_gaps == 0 and .telemetry_gaps == 0
    and .host_contention.policy == "process-group-v1"
    and (.host_contention.settle.log_sha256 | test("^[0-9a-f]{64}$"))
    and (.host_contention.settle.samples | type) == "number"
    and .host_contention.settle.samples > 0
    and (.host_contention.settle.duration_seconds | type) == "number"
    and .host_contention.settle.duration_seconds >= 60
    and (.host_contention.settle.contended_samples | type) == "number"
    and .host_contention.settle.contended_samples >= 0
    and .host_contention.settle.telemetry_gaps == 0
    and (.host_contention.measurement.log_sha256 | test("^[0-9a-f]{64}$"))
    and (.host_contention.measurement.samples | type) == "number"
    and .host_contention.measurement.samples >= 2
    and (.host_contention.measurement.duration_seconds | type) == "number"
    and .host_contention.measurement.duration_seconds > 0
    and .host_contention.measurement.contended_samples == 0
    and .host_contention.measurement.telemetry_gaps == 0
  ' "$summary" >/dev/null

if [[ "$(sha256_file "$ROOT_DIR/scripts/macos_thermal_probe.swift")" != \
  "$(jq -er .thermal_probe.source_sha256 "$summary")" ]]; then
  echo "decode-cohort receipt thermal-probe source digest mismatch" >&2
  exit 1
fi

if ! grep -Fq \
    'test inference::models::deepseek4::real_artifact_decode_cohort_tests::official_artifact_b4_decode_body_is_exact_and_measured ... ' \
    "$test_log"; then
  echo "decode-cohort test log is missing the exact named test" >&2
  exit 1
fi
test_result_lines=$(awk '/^test result: / { count++ } END { print count + 0 }' \
  "$test_log")
if [[ "$test_result_lines" != 1 ]]; then
  echo "decode-cohort test log must contain exactly one libtest result line" >&2
  exit 1
fi
if ! grep -Eq \
    '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out; finished in [0-9]+(\.[0-9]+)?s$' \
    "$test_log"; then
  echo "decode-cohort named test did not finish with one pass and zero failures" >&2
  exit 1
fi
grep -Fq 'exact_state_logits_cache_recurrent=true' "$test_log"

thermal_validate_fair_or_better_measurement_log "$measurement_log" 5
test "$THERMAL_LOG_SAMPLES" = "$(jq -er .measurement_samples "$summary")"
test "$THERMAL_LOG_DURATION_SECONDS" = \
  "$(jq -er .measurement_duration_seconds "$summary")"
test "$THERMAL_LOG_NON_NOMINAL_SAMPLES" = \
  "$(jq -er .non_nominal_measurement_samples "$summary")"
test "$THERMAL_LOG_FAIR_SAMPLES" = \
  "$(jq -er .fair_measurement_samples "$summary")"
test "$THERMAL_LOG_OVER_LIMIT_SAMPLES" = \
  "$(jq -er .over_limit_measurement_samples "$summary")"
test "$THERMAL_LOG_GAPS" = "$(jq -er .telemetry_gaps "$summary")"
test "$(head -1 "$measurement_log" | awk -F '\t' '{print $3}')" = \
  decode-cohort-measurement-start
test "$(head -1 "$measurement_log" | awk -F '\t' '{print $2}')" = nominal
test "$(tail -1 "$measurement_log" | awk -F '\t' '{print $3}')" = \
  decode-cohort-measurement-end
case "$(tail -1 "$measurement_log" | awk -F '\t' '{print $2}')" in
  nominal|fair) ;;
  *) exit 1 ;;
esac
awk -F '\t' 'NR > 1 && $3 != "decode-cohort-measurement" && \
  $3 != "decode-cohort-measurement-end" { exit 1 }' "$measurement_log"

thermal_validate_settle_log "$settle_log" 60 8
test "$THERMAL_LOG_SAMPLES" = "$(jq -er .settle_samples "$summary")"
test "$THERMAL_LOG_DURATION_SECONDS" = \
  "$(jq -er .settle_duration_seconds "$summary")"
test "$THERMAL_LOG_GAPS" = "$(jq -er .settle_telemetry_gaps "$summary")"
awk -F '\t' '$3 != "decode-cohort-settle" { exit 1 }' "$settle_log"

host_contention_validate_measurement_log "$contention_measurement_log" 5
test "$HOST_CONTENTION_LOG_SAMPLES" = \
  "$(jq -er .host_contention.measurement.samples "$summary")"
test "$HOST_CONTENTION_LOG_DURATION_SECONDS" = \
  "$(jq -er .host_contention.measurement.duration_seconds "$summary")"
test "$HOST_CONTENTION_LOG_CONTENDED_SAMPLES" = \
  "$(jq -er .host_contention.measurement.contended_samples "$summary")"
test "$HOST_CONTENTION_LOG_GAPS" = \
  "$(jq -er .host_contention.measurement.telemetry_gaps "$summary")"
host_contention_validate_settle_log "$contention_settle_log" 60 8
test "$HOST_CONTENTION_LOG_SAMPLES" = \
  "$(jq -er .host_contention.settle.samples "$summary")"
test "$HOST_CONTENTION_LOG_DURATION_SECONDS" = \
  "$(jq -er .host_contention.settle.duration_seconds "$summary")"
test "$HOST_CONTENTION_LOG_CONTENDED_SAMPLES" = \
  "$(jq -er .host_contention.settle.contended_samples "$summary")"
test "$HOST_CONTENTION_LOG_GAPS" = \
  "$(jq -er .host_contention.settle.telemetry_gaps "$summary")"
host_contention_validate_thermal_alignment "$measurement_log" \
  "$contention_measurement_log"
host_contention_validate_thermal_alignment "$settle_log" \
  "$contention_settle_log"
awk -F '\t' 'NR == 1 && $3 != "decode-cohort-measurement-start" { exit 1 }
  NR > 1 && $3 != "decode-cohort-measurement" && \
    $3 != "decode-cohort-measurement-end" { exit 1 }' \
  "$contention_measurement_log"
awk -F '\t' '$3 != "decode-cohort-settle" { exit 1 }' \
  "$contention_settle_log"

echo "DeepSeek-V4 decode-cohort receipt verified" >&2
