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
  'test inference::models::deepseek4::real_artifact_decode_cohort_tests::official_artifact_b4_decode_body_is_exact_and_measured ... ok' \
  'DeepSeek-V4 B=4 decode spike: exact_state_logits_cache_recurrent=true' >"$test_log"
printf '2000\tnominal\tdecode-cohort-measurement-start\n' >"$measurement"
printf '2002\tnominal\tdecode-cohort-measurement-end\n' >>"$measurement"
for timestamp in $(seq 1000 5 1060); do
  printf '%s\tnominal\tdecode-cohort-settle\n' "$timestamp" >>"$settle"
done

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
jq --arg source_sha "$source_sha" --arg model_sha256 "$model_sha" \
  --arg raw_sha256 "$(sha256_file "$raw")" \
  --arg test_log_sha256 "$(sha256_file "$test_log")" \
  --arg measurement_log_sha256 "$(sha256_file "$measurement")" \
  --arg settle_log_sha256 "$(sha256_file "$settle")" '
  . + {source_sha:$source_sha,model_sha256:$model_sha256,
    mlx_native_version:"0.10.12",raw_sha256:$raw_sha256,
    test_log_sha256:$test_log_sha256,thermal_status:"nominal",
    measurement_log_sha256:$measurement_log_sha256,
    settle_log_sha256:$settle_log_sha256,settle_seconds:60,
    settle_samples:13,settle_duration_seconds:60,
    settle_sample_interval_seconds:5,maximum_settle_sample_gap_seconds:8,
    settle_telemetry_gaps:0,measurement_samples:2,
    measurement_duration_seconds:2,sample_interval_seconds:2,
    maximum_sample_gap_seconds:5,non_nominal_measurement_samples:0,
    telemetry_gaps:0}
' "$raw" >"$summary"

bash "$VERIFY" "$summary" "$raw" "$test_log" "$measurement" "$settle" \
  "$source_sha" "$model_sha"

expect_reject() {
  local label=$1
  local mutated=$2
  if bash "$VERIFY" "$mutated" "$raw" "$test_log" "$measurement" "$settle" \
      "$source_sha" "$model_sha" >/dev/null 2>&1; then
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
  >"$tmp_dir/bad-thermal.json"
expect_reject bad-thermal "$tmp_dir/bad-thermal.json"

echo "DeepSeek-V4 decode-cohort receipt contract: pass"
