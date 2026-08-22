#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERIFY="$ROOT_DIR/scripts/verify_deepseek4_cooperative_prefill_receipt.sh"
SOURCE_SHA=0123456789abcdef0123456789abcdef01234567
MODEL_SHA=936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d

tmp_dir=$(mktemp -d -t hf2q-cooperative-receipt.XXXXXX)
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

if MLX_NATIVE_SKIP_METALLIB=1 \
  EXPECTED_SHA="$SOURCE_SHA" \
  CRATE_SHA256="$MODEL_SHA" \
  DEPENDENCY_PROVENANCE_DIR="$tmp_dir/dummy-dependency-provenance" \
  HF2Q_BIN=/bin/true \
  EXPECTED_BINARY_SHA256="$MODEL_SHA" \
  DEEPSEEK_MODEL=/dev/null GEMMA_MODEL=/dev/null \
  QWEN_MODEL=/dev/null QWEN38_MODEL=/dev/null \
  DEEPSEEK_MODEL_SHA256="$MODEL_SHA" GEMMA_MODEL_SHA256="$MODEL_SHA" \
  QWEN_MODEL_SHA256="$MODEL_SHA" QWEN38_MODEL_SHA256="$MODEL_SHA" \
  OUT_ROOT="$tmp_dir/forbidden-build-env" \
  bash "$ROOT_DIR/scripts/run_agentic_cache_release_gate.sh" \
    >"$tmp_dir/forbidden.stdout" 2>"$tmp_dir/forbidden.stderr"; then
  echo "release gate accepted MLX_NATIVE_SKIP_METALLIB" >&2
  exit 1
fi
grep -F 'MLX_NATIVE_SKIP_METALLIB is forbidden' \
  "$tmp_dir/forbidden.stderr" >/dev/null

raw="$tmp_dir/raw.json"
test_log="$tmp_dir/test.log"
measurement="$tmp_dir/thermal.log"
settle="$tmp_dir/settle.log"
contention_measurement="$tmp_dir/measurement-contention.log"
contention_settle="$tmp_dir/settle-contention.log"
summary="$tmp_dir/summary.json"

jq -n '{
  schema_version:1,status:"pass",artifact_bytes:107431343168,layers:43,
  prefix_rows:148,prefix_mod_128:20,prefix_mod_4:0,
  parity_shapes:[
    {sequences:2,rows_per_lane:1024,aggregate_rows:2048,exact_state_logits_decode:true},
    {sequences:3,rows_per_lane:640,aggregate_rows:1920,exact_state_logits_decode:true},
    {sequences:4,rows_per_lane:512,aggregate_rows:2048,exact_state_logits_decode:true}
  ],
  benchmark:{sequences:4,rows_per_lane:512,aggregate_rows:2048,pairs:5,
    order:"alternating",serial_ms:[11,12,13,14,15],cohort_ms:[8,9,10,11,12],
    serial_median_ms:13,cohort_median_ms:10,speedup:1.3,
    process_lifetime_peak_rss_bytes:123456},
  benchmark_environment:{profile:"clean-hf2q-mlx-metal-v1",
    override_variables_absent:true,unexpected_override_variables:[],pairs:5}
}' >"$raw"
printf 'cooperative hardware test passed\n' >"$test_log"
printf '2000\tnominal\tcooperative-prefill-measurement-start\n' >"$measurement"
printf '2002\tfair\tcooperative-prefill-measurement\n' >>"$measurement"
printf '2004\tfair\tcooperative-prefill-measurement-end\n' >>"$measurement"
printf '2000\tquiet\tcooperative-prefill-measurement-start\t100\t-\n' \
  >"$contention_measurement"
printf '2002\tquiet\tcooperative-prefill-measurement\t100\t-\n' \
  >>"$contention_measurement"
printf '2004\tquiet\tcooperative-prefill-measurement-end\t100\t-\n' \
  >>"$contention_measurement"
for timestamp in 1000 1005 1010 1015 1020 1025 1030 1035 1040 1045 1050 1055 1060; do
  printf '%s\tnominal\tcooperative-prefill-settle\n' "$timestamp" >>"$settle"
  printf '%s\tquiet\tcooperative-prefill-settle\t100\t-\n' "$timestamp" \
    >>"$contention_settle"
done

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
write_summary() {
  local output=$1
  local measurement_path=${2:-$measurement}
  local settle_path=${3:-$settle}
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
  jq --arg source_sha "$SOURCE_SHA" --arg model_sha256 "$MODEL_SHA" \
    --arg raw_sha256 "$(sha256_file "$raw")" \
    --arg test_log_sha256 "$(sha256_file "$test_log")" \
    --arg measurement_log_sha256 "$(sha256_file "$measurement_path")" \
    --arg settle_log_sha256 "$(sha256_file "$settle_path")" \
    --arg contention_measurement_log_sha256 \
      "$(sha256_file "$contention_measurement")" \
    --arg contention_settle_log_sha256 \
      "$(sha256_file "$contention_settle")" \
    --argjson measurement_samples "$measurement_samples" \
    --argjson measurement_duration_seconds "$measurement_duration_seconds" \
    --argjson non_nominal_measurement_samples \
      "$non_nominal_measurement_samples" \
    --argjson fair_measurement_samples "$fair_measurement_samples" \
    --argjson over_limit_measurement_samples \
      "$over_limit_measurement_samples" \
    --argjson telemetry_gaps "$telemetry_gaps" '
    . + {schema_version:2,source_sha:$source_sha,model_sha256:$model_sha256,
      mlx_native_version:"0.11.2",raw_sha256:$raw_sha256,
      test_log_sha256:$test_log_sha256,thermal_status:"fair_or_better",
      required_start_state:"nominal",maximum_measurement_state:"fair",
      measurement_log_sha256:$measurement_log_sha256,
      settle_log_sha256:$settle_log_sha256,settle_seconds:60,
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
  "$SOURCE_SHA" "$MODEL_SHA" "$contention_measurement" \
  "$contention_settle" >/dev/null

expect_rejected() {
  local label=$1
  shift
  if bash "$VERIFY" "$@" "$contention_measurement" \
      "$contention_settle" >/dev/null 2>&1; then
    echo "cooperative receipt verifier accepted invalid case: $label" >&2
    exit 1
  fi
}

jq '.benchmark.speedup = 9' "$summary" >"$tmp_dir/bad-speedup.json"
expect_rejected derived-speedup "$tmp_dir/bad-speedup.json" "$raw" "$test_log" \
  "$measurement" "$settle" "$SOURCE_SHA" "$MODEL_SHA"

cp "$raw" "$tmp_dir/mutated-raw.json"
printf '\n' >>"$tmp_dir/mutated-raw.json"
expect_rejected raw-hash "$summary" "$tmp_dir/mutated-raw.json" "$test_log" \
  "$measurement" "$settle" "$SOURCE_SHA" "$MODEL_SHA"

jq 'del(.parity_shapes)' "$raw" >"$tmp_dir/missing-raw-field.json"
jq --arg raw_sha256 "$(sha256_file "$tmp_dir/missing-raw-field.json")" \
  '.raw_sha256 = $raw_sha256' "$summary" >"$tmp_dir/missing-raw-field-summary.json"
expect_rejected missing-raw-field "$tmp_dir/missing-raw-field-summary.json" \
  "$tmp_dir/missing-raw-field.json" "$test_log" "$measurement" "$settle" \
  "$SOURCE_SHA" "$MODEL_SHA"

cp "$raw" "$tmp_dir/multiple-raw.json"
printf '\n{}\n' >>"$tmp_dir/multiple-raw.json"
jq --arg raw_sha256 "$(sha256_file "$tmp_dir/multiple-raw.json")" \
  '.raw_sha256 = $raw_sha256' "$summary" >"$tmp_dir/multiple-raw-summary.json"
expect_rejected multiple-raw-documents "$tmp_dir/multiple-raw-summary.json" \
  "$tmp_dir/multiple-raw.json" "$test_log" "$measurement" "$settle" \
  "$SOURCE_SHA" "$MODEL_SHA"

cp "$summary" "$tmp_dir/multiple-summary.json"
printf '\n{}\n' >>"$tmp_dir/multiple-summary.json"
expect_rejected multiple-summary-documents "$tmp_dir/multiple-summary.json" \
  "$raw" "$test_log" "$measurement" "$settle" "$SOURCE_SHA" "$MODEL_SHA"

printf '2000\tnominal\tcooperative-prefill-measurement-start\n' \
  >"$tmp_dir/gapped-measurement.log"
printf '2010\tnominal\tcooperative-prefill-measurement-end\n' \
  >>"$tmp_dir/gapped-measurement.log"
write_summary "$tmp_dir/gapped-summary.json" "$tmp_dir/gapped-measurement.log"
expect_rejected measurement-gap "$tmp_dir/gapped-summary.json" "$raw" \
  "$test_log" "$tmp_dir/gapped-measurement.log" "$settle" "$SOURCE_SHA" "$MODEL_SHA"

for over_limit_state in serious critical; do
  printf '2000\tnominal\tcooperative-prefill-measurement-start\n' \
    >"$tmp_dir/$over_limit_state-measurement.log"
  printf '2002\t%s\tcooperative-prefill-measurement\n' "$over_limit_state" \
    >>"$tmp_dir/$over_limit_state-measurement.log"
  printf '2004\tfair\tcooperative-prefill-measurement-end\n' \
    >>"$tmp_dir/$over_limit_state-measurement.log"
  write_summary "$tmp_dir/$over_limit_state-summary.json" \
    "$tmp_dir/$over_limit_state-measurement.log"
  expect_rejected "$over_limit_state-state" \
    "$tmp_dir/$over_limit_state-summary.json" "$raw" "$test_log" \
    "$tmp_dir/$over_limit_state-measurement.log" "$settle" \
    "$SOURCE_SHA" "$MODEL_SHA"
done

printf '2000\tfair\tcooperative-prefill-measurement-start\n' \
  >"$tmp_dir/fair-start-measurement.log"
printf '2002\tfair\tcooperative-prefill-measurement-end\n' \
  >>"$tmp_dir/fair-start-measurement.log"
write_summary "$tmp_dir/fair-start-summary.json" \
  "$tmp_dir/fair-start-measurement.log"
expect_rejected non-nominal-start "$tmp_dir/fair-start-summary.json" "$raw" \
  "$test_log" "$tmp_dir/fair-start-measurement.log" "$settle" \
  "$SOURCE_SHA" "$MODEL_SHA"

jq '.required_start_state = "fair"' "$summary" \
  >"$tmp_dir/bad-required-start.json"
expect_rejected required-start-policy "$tmp_dir/bad-required-start.json" "$raw" \
  "$test_log" "$measurement" "$settle" "$SOURCE_SHA" "$MODEL_SHA"

jq '.maximum_measurement_state = "nominal"' "$summary" \
  >"$tmp_dir/bad-maximum-state.json"
expect_rejected maximum-state-policy "$tmp_dir/bad-maximum-state.json" "$raw" \
  "$test_log" "$measurement" "$settle" "$SOURCE_SHA" "$MODEL_SHA"

jq '.fair_measurement_samples = 1' "$summary" >"$tmp_dir/bad-fair-count.json"
expect_rejected fair-count "$tmp_dir/bad-fair-count.json" "$raw" "$test_log" \
  "$measurement" "$settle" "$SOURCE_SHA" "$MODEL_SHA"

jq '.over_limit_measurement_samples = 1' "$summary" \
  >"$tmp_dir/bad-over-limit-count.json"
expect_rejected over-limit-count "$tmp_dir/bad-over-limit-count.json" "$raw" \
  "$test_log" "$measurement" "$settle" "$SOURCE_SHA" "$MODEL_SHA"

cp "$settle" "$tmp_dir/mutated-settle.log"
printf '1065\tfair\tcooperative-prefill-settle\n' >>"$tmp_dir/mutated-settle.log"
write_summary "$tmp_dir/mutated-settle-summary.json" "$measurement" \
  "$tmp_dir/mutated-settle.log"
jq '.settle_samples = 14 | .settle_duration_seconds = 60' \
  "$tmp_dir/mutated-settle-summary.json" >"$tmp_dir/bad-settle-summary.json"
expect_rejected settle-state "$tmp_dir/bad-settle-summary.json" "$raw" "$test_log" \
  "$measurement" "$tmp_dir/mutated-settle.log" "$SOURCE_SHA" "$MODEL_SHA"

cp "$contention_measurement" "$tmp_dir/contended-host.log"
sed -i.bak '2s/quiet/contended/;2s/-$/200:200:cargo/' \
  "$tmp_dir/contended-host.log"
rm -f "$tmp_dir/contended-host.log.bak"
jq --arg sha "$(sha256_file "$tmp_dir/contended-host.log")" \
  '.host_contention.measurement.log_sha256 = $sha
   | .host_contention.measurement.contended_samples = 1' \
  "$summary" >"$tmp_dir/contended-host-summary.json"
if bash "$VERIFY" "$tmp_dir/contended-host-summary.json" "$raw" "$test_log" \
    "$measurement" "$settle" "$SOURCE_SHA" "$MODEL_SHA" \
    "$tmp_dir/contended-host.log" "$contention_settle" >/dev/null 2>&1; then
  echo "cooperative receipt verifier accepted host contention" >&2
  exit 1
fi

echo "DeepSeek-V4 cooperative prefill receipt contract: pass"
