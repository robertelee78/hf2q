#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERIFY="$ROOT_DIR/scripts/verify_deepseek4_decode_cohort_receipt.sh"
tmp_dir=$(mktemp -d -t hf2q-decode-cohort-receipt.XXXXXX)
trap 'rm -rf "$tmp_dir"' EXIT

source_sha=0123456789abcdef0123456789abcdef01234567
model_sha=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
raw="$tmp_dir/raw.json"
summary="$tmp_dir/summary.json"
test_log="$tmp_dir/test.log"
measurement="$tmp_dir/measurement.log"
settle="$tmp_dir/settle.log"
contention_measurement="$tmp_dir/measurement-contention.log"
contention_settle="$tmp_dir/settle-contention.log"

jq -n '
  def counter($cb;$sync):
    {command_buffers:$cb,synchronizations:$sync,dispatches:100,barriers:50};
  {
    schema_version:1,status:"pass",artifact_bytes:107431343168,layers:43,lanes:4,
    parity:{prefix_rows:148,steps:132,final_position:280,final_mod_4:0,
      final_mod_128:24,physical_to_logical:[2,0,3,1],
      exact_state_logits_cache_recurrent:true},
    benchmark:{position:6676,logical_capacity:131072,loaded_idle_seconds:45,
      pairs:10,order:"alternating",
      serial_ms:[20,20,20,20,20,20,20,20,20,20],
      cohort_ms:[10,10,10,10,10,10,10,10,10,10],
      serial_median_ms:20,cohort_median_ms:10,speedup:2,
      serial_counters:[range(0;10)|counter(92;4)],
      cohort_counters:[range(0;10)|counter(23;1)],
      serial_command_buffers_per_pair:92,cohort_command_buffers_per_pair:23,
      serial_synchronizations_per_pair:4,cohort_synchronizations_per_pair:1},
    benchmark_environment:{profile:"clean-hf2q-mlx-metal-v1",
      override_variables_absent:true,unexpected_override_variables:[]}
  }
' >"$raw"
printf '%s\n' \
  'test inference::models::deepseek4::real_artifact_decode_cohort_tests::official_artifact_b4_decode_body_is_exact_and_measured ... DeepSeek-V4 B=4 benchmark loaded-idle settle: position=6676 logical_capacity=131072 seconds=45' \
  'DeepSeek-V4 B=4 decode spike: exact_state_logits_cache_recurrent=true' \
  'ok' \
  '' \
  'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 4653 filtered out; finished in 202.59s' >"$test_log"
printf '2000\tnominal\tdecode-cohort-measurement-start\n' >"$measurement"
printf '2002\tfair\tdecode-cohort-measurement\n' >>"$measurement"
printf '2004\tfair\tdecode-cohort-measurement-end\n' >>"$measurement"
printf '2000\tquiet\tdecode-cohort-measurement-start\t100\t-\n' \
  >"$contention_measurement"
printf '2002\tquiet\tdecode-cohort-measurement\t100\t-\n' \
  >>"$contention_measurement"
printf '2004\tquiet\tdecode-cohort-measurement-end\t100\t-\n' \
  >>"$contention_measurement"
for timestamp in $(seq 1000 5 1060); do
  printf '%s\tnominal\tdecode-cohort-settle\n' "$timestamp" >>"$settle"
  printf '%s\tquiet\tdecode-cohort-settle\t100\t-\n' "$timestamp" \
    >>"$contention_settle"
done

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
thermal_probe_source_sha=$(sha256_file "$ROOT_DIR/scripts/macos_thermal_probe.swift")
thermal_probe_compiler_sha=abababababababababababababababababababababababababababababababab
thermal_probe_binary_sha=cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd
write_summary() {
  local output=$1
  local measurement_path=${2:-$measurement}
  local settle_path=${3:-$settle}
  local test_log_path=${4:-$test_log}
  local measurement_samples measurement_duration_seconds
  local non_nominal_measurement_samples fair_measurement_samples
  local over_limit_measurement_samples telemetry_gaps
  IFS=$'\t' read -r measurement_samples measurement_duration_seconds \
    non_nominal_measurement_samples fair_measurement_samples \
    over_limit_measurement_samples telemetry_gaps <<<"$(awk -F '\t' '
      BEGIN { non_nominal = 0; fair = 0; over = 0; gaps = 0 }
      {
        samples++
        if ($2 != "nominal") non_nominal++
        if ($2 == "fair") fair++
        if ($2 == "serious" || $2 == "critical") over++
        if (samples == 1) first = $1
        if (samples > 1 && ($1 < previous || $1 - previous > 5)) gaps++
        previous = $1
        last = $1
      }
      END { print samples "\t" last-first "\t" non_nominal "\t" fair \
        "\t" over "\t" gaps }
    ' "$measurement_path")"
  jq --arg source_sha "$source_sha" --arg model_sha256 "$model_sha" \
    --arg raw_sha256 "$(sha256_file "$raw")" \
    --arg test_log_sha256 "$(sha256_file "$test_log_path")" \
    --arg measurement_log_sha256 "$(sha256_file "$measurement_path")" \
    --arg settle_log_sha256 "$(sha256_file "$settle_path")" \
    --arg contention_measurement_log_sha256 \
      "$(sha256_file "$contention_measurement")" \
    --arg contention_settle_log_sha256 \
      "$(sha256_file "$contention_settle")" \
    --arg thermal_probe_source_sha256 "$thermal_probe_source_sha" \
    --arg thermal_probe_compiler_sha256 "$thermal_probe_compiler_sha" \
    --arg thermal_probe_binary_sha256 "$thermal_probe_binary_sha" \
    --argjson measurement_samples "$measurement_samples" \
    --argjson measurement_duration_seconds "$measurement_duration_seconds" \
    --argjson non_nominal_measurement_samples \
      "$non_nominal_measurement_samples" \
    --argjson fair_measurement_samples "$fair_measurement_samples" \
    --argjson over_limit_measurement_samples \
      "$over_limit_measurement_samples" \
    --argjson telemetry_gaps "$telemetry_gaps" '
    . + {schema_version:3,source_sha:$source_sha,model_sha256:$model_sha256,
      mlx_native_version:"0.11.1",raw_sha256:$raw_sha256,
      test_log_sha256:$test_log_sha256,thermal_status:"fair_or_better",
      required_start_state:"nominal",maximum_measurement_state:"fair",
      measurement_log_sha256:$measurement_log_sha256,
      settle_log_sha256:$settle_log_sha256,settle_seconds:60,
      thermal_probe:{implementation:"compiled-foundation-helper",
        source_path:"scripts/macos_thermal_probe.swift",
        source_sha256:$thermal_probe_source_sha256,
        compiler_path:"/usr/bin/swiftc",
        compiler_sha256:$thermal_probe_compiler_sha256,
        compiler_version:"Apple Swift version 6.2 (synthetic)",
        binary_sha256:$thermal_probe_binary_sha256},
      settle_samples:13,settle_duration_seconds:60,
      settle_sample_interval_seconds:5,maximum_settle_sample_gap_seconds:8,
      settle_telemetry_gaps:0,measurement_samples:$measurement_samples,
      measurement_duration_seconds:$measurement_duration_seconds,
      sample_interval_seconds:2,maximum_sample_gap_seconds:5,
      non_nominal_measurement_samples:$non_nominal_measurement_samples,
      fair_measurement_samples:$fair_measurement_samples,
      over_limit_measurement_samples:$over_limit_measurement_samples,
      telemetry_gaps:$telemetry_gaps,
      host_contention:{policy:"process-group-v1",
        settle:{log_sha256:$contention_settle_log_sha256,samples:13,
          duration_seconds:60,contended_samples:0,telemetry_gaps:0},
        measurement:{log_sha256:$contention_measurement_log_sha256,samples:3,
          duration_seconds:4,contended_samples:0,telemetry_gaps:0}}}
  ' "$raw" >"$output"
}

write_summary "$summary"

bash "$VERIFY" "$summary" "$raw" "$test_log" "$measurement" "$settle" \
  "$source_sha" "$model_sha" "$contention_measurement" "$contention_settle"

expect_reject() {
  local label=$1
  local mutated=$2
  local measurement_path=${3:-$measurement}
  local settle_path=${4:-$settle}
  local test_log_path=${5:-$test_log}
  if bash "$VERIFY" "$mutated" "$raw" "$test_log_path" "$measurement_path" \
      "$settle_path" \
      "$source_sha" "$model_sha" "$contention_measurement" \
      "$contention_settle" >/dev/null 2>&1; then
    echo "decode-cohort verifier accepted invalid case: $label" >&2
    exit 1
  fi
}

jq '.benchmark.speedup = 1' "$summary" >"$tmp_dir/bad-speedup.json"
expect_reject bad-speedup "$tmp_dir/bad-speedup.json"
jq '.benchmark.cohort_command_buffers_per_pair = 24' "$summary" \
  >"$tmp_dir/bad-topology.json"
expect_reject bad-topology "$tmp_dir/bad-topology.json"
jq '.raw_sha256 = ("0" * 64)' "$summary" >"$tmp_dir/bad-hash.json"
expect_reject bad-hash "$tmp_dir/bad-hash.json"
jq '.non_nominal_measurement_samples = 1' "$summary" \
  >"$tmp_dir/bad-non-nominal-count.json"
expect_reject bad-non-nominal-count "$tmp_dir/bad-non-nominal-count.json"
jq '.fair_measurement_samples = 1' "$summary" \
  >"$tmp_dir/bad-fair-count.json"
expect_reject bad-fair-count "$tmp_dir/bad-fair-count.json"
jq '.over_limit_measurement_samples = 1' "$summary" \
  >"$tmp_dir/bad-over-limit-count.json"
expect_reject bad-over-limit-count "$tmp_dir/bad-over-limit-count.json"
jq '.required_start_state = "fair"' "$summary" \
  >"$tmp_dir/bad-required-start.json"
expect_reject bad-required-start "$tmp_dir/bad-required-start.json"
jq '.maximum_measurement_state = "nominal"' "$summary" \
  >"$tmp_dir/bad-maximum-state.json"
expect_reject bad-maximum-state "$tmp_dir/bad-maximum-state.json"
jq '.thermal_probe.source_sha256 = ("0" * 64)' "$summary" \
  >"$tmp_dir/bad-probe-source.json"
expect_reject bad-probe-source "$tmp_dir/bad-probe-source.json"
jq '.schema_version = 2' "$summary" >"$tmp_dir/stale-summary-schema.json"
expect_reject stale-summary-schema "$tmp_dir/stale-summary-schema.json"
jq 'del(.thermal_probe.binary_sha256)' "$summary" \
  >"$tmp_dir/missing-probe-binary.json"
expect_reject missing-probe-binary "$tmp_dir/missing-probe-binary.json"

printf '%s\n' \
  'DeepSeek-V4 B=4 decode spike: exact_state_logits_cache_recurrent=true' \
  'ok' \
  '' \
  'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 4653 filtered out; finished in 202.59s' \
  >"$tmp_dir/missing-test-name.log"
write_summary "$tmp_dir/missing-test-name-summary.json" "$measurement" \
  "$settle" "$tmp_dir/missing-test-name.log"
expect_reject missing-test-name "$tmp_dir/missing-test-name-summary.json" \
  "$measurement" "$settle" "$tmp_dir/missing-test-name.log"

printf '%s\n' \
  'test inference::models::deepseek4::real_artifact_decode_cohort_tests::official_artifact_b4_decode_body_is_exact_and_measured ... DeepSeek-V4 B=4 benchmark loaded-idle settle: position=6676 logical_capacity=131072 seconds=45' \
  'DeepSeek-V4 B=4 decode spike: exact_state_logits_cache_recurrent=true' \
  'FAILED' \
  '' \
  'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 4653 filtered out; finished in 202.59s' \
  >"$tmp_dir/failed-test-result.log"
write_summary "$tmp_dir/failed-test-result-summary.json" "$measurement" \
  "$settle" "$tmp_dir/failed-test-result.log"
expect_reject failed-test-result "$tmp_dir/failed-test-result-summary.json" \
  "$measurement" "$settle" "$tmp_dir/failed-test-result.log"

printf '%s\n' \
  'test inference::models::deepseek4::real_artifact_decode_cohort_tests::official_artifact_b4_decode_body_is_exact_and_measured ... DeepSeek-V4 B=4 benchmark loaded-idle settle: position=6676 logical_capacity=131072 seconds=45' \
  'DeepSeek-V4 B=4 decode spike: exact_state_logits_cache_recurrent=true' \
  'FAILED' \
  '' \
  'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 4653 filtered out; finished in 202.59s' \
  'test unrelated_test ... ok' \
  '' \
  'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 4653 filtered out; finished in 0.01s' \
  >"$tmp_dir/concatenated-results.log"
write_summary "$tmp_dir/concatenated-results-summary.json" "$measurement" \
  "$settle" "$tmp_dir/concatenated-results.log"
expect_reject concatenated-results "$tmp_dir/concatenated-results-summary.json" \
  "$measurement" "$settle" "$tmp_dir/concatenated-results.log"

printf '2000\tnominal\tdecode-cohort-measurement-start\n' \
  >"$tmp_dir/gapped-measurement.log"
printf '2010\tnominal\tdecode-cohort-measurement-end\n' \
  >>"$tmp_dir/gapped-measurement.log"
write_summary "$tmp_dir/gapped-summary.json" "$tmp_dir/gapped-measurement.log"
expect_reject measurement-gap "$tmp_dir/gapped-summary.json" \
  "$tmp_dir/gapped-measurement.log"

for over_limit_state in serious critical; do
  printf '2000\tnominal\tdecode-cohort-measurement-start\n' \
    >"$tmp_dir/$over_limit_state-measurement.log"
  printf '2002\t%s\tdecode-cohort-measurement\n' "$over_limit_state" \
    >>"$tmp_dir/$over_limit_state-measurement.log"
  printf '2004\tfair\tdecode-cohort-measurement-end\n' \
    >>"$tmp_dir/$over_limit_state-measurement.log"
  write_summary "$tmp_dir/$over_limit_state-summary.json" \
    "$tmp_dir/$over_limit_state-measurement.log"
  expect_reject "$over_limit_state-state" \
    "$tmp_dir/$over_limit_state-summary.json" \
    "$tmp_dir/$over_limit_state-measurement.log"
done

printf '2000\tfair\tdecode-cohort-measurement-start\n' \
  >"$tmp_dir/fair-start-measurement.log"
printf '2002\tfair\tdecode-cohort-measurement-end\n' \
  >>"$tmp_dir/fair-start-measurement.log"
write_summary "$tmp_dir/fair-start-summary.json" \
  "$tmp_dir/fair-start-measurement.log"
expect_reject non-nominal-start "$tmp_dir/fair-start-summary.json" \
  "$tmp_dir/fair-start-measurement.log"

cp "$settle" "$tmp_dir/non-nominal-settle.log"
printf '1065\tfair\tdecode-cohort-settle\n' \
  >>"$tmp_dir/non-nominal-settle.log"
write_summary "$tmp_dir/non-nominal-settle-summary.json" "$measurement" \
  "$tmp_dir/non-nominal-settle.log"
expect_reject non-nominal-settle "$tmp_dir/non-nominal-settle-summary.json" \
  "$measurement" "$tmp_dir/non-nominal-settle.log"

awk -F '\t' 'BEGIN { OFS="\t" }
  NR == 2 { $2="contended"; $5="200:200:rustc" }
  { print }
' "$contention_measurement" >"$tmp_dir/contended-host.log"
jq --arg sha "$(sha256_file "$tmp_dir/contended-host.log")" \
  '.host_contention.measurement.log_sha256 = $sha
   | .host_contention.measurement.contended_samples = 1' \
  "$summary" >"$tmp_dir/contended-host-summary.json"
if bash "$VERIFY" "$tmp_dir/contended-host-summary.json" "$raw" "$test_log" \
    "$measurement" "$settle" "$source_sha" "$model_sha" \
    "$tmp_dir/contended-host.log" "$contention_settle" >/dev/null 2>&1; then
  echo "decode-cohort verifier accepted host contention" >&2
  exit 1
fi

echo "DeepSeek-V4 decode-cohort receipt contract: pass"
