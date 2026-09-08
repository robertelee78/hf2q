#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERIFY="$ROOT_DIR/scripts/verify_deepseek4_decode_cohort_receipt.sh"
# shellcheck source=scripts/macos_thermal_guard.sh
source "$ROOT_DIR/scripts/macos_thermal_guard.sh"
# shellcheck source=scripts/macos_memory_guard.sh
source "$ROOT_DIR/scripts/macos_memory_guard.sh"
tmp_dir=$(mktemp -d -t hf2q-decode-cohort-receipt.XXXXXX)
trap 'rm -rf "$tmp_dir"' EXIT

source_sha=0123456789abcdef0123456789abcdef01234567
model_sha=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
run_uuid=01234567-89ab-cdef-0123-456789abcdef
producer_pid=4242
raw="$tmp_dir/raw.json"
summary="$tmp_dir/summary.json"
test_log="$tmp_dir/test.log"
measurement="$tmp_dir/measurement.log"
settle="$tmp_dir/settle.log"
contention_measurement="$tmp_dir/measurement-contention.log"
contention_settle="$tmp_dir/settle-contention.log"
memory_log="$tmp_dir/memory-pressure.log"
phase_log="$tmp_dir/phases.jsonl"
setup_thermal="$tmp_dir/loaded-setup-thermal.log"
setup_contention="$tmp_dir/loaded-setup-contention.log"
setup_memory="$tmp_dir/loaded-setup-memory.log"
loaded_idle_memory="$tmp_dir/loaded-idle-memory.log"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
thermal_probe_source_sha=$(sha256_file "$ROOT_DIR/scripts/macos_thermal_probe.swift")
thermal_probe_compiler_sha=abababababababababababababababababababababababababababababababab
thermal_probe_binary_sha=cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd
memory_guard_source_sha=$(sha256_file "$ROOT_DIR/scripts/macos_memory_guard.sh")

jq -n --arg uuid "$run_uuid" --argjson pid "$producer_pid" '
  def counter($cb;$sync):
    {command_buffers:$cb,synchronizations:$sync,dispatches:100,barriers:50};
  def marker($seq;$phase;$mono;$wall):
    {run_uuid:$uuid,sequence:$seq,phase:$phase,pid:$pid,
      monotonic_ns:$mono,wall_ns:$wall};
  def vm:
    {boot_time_seconds:1786056685,page_size:16384,pressure_level:2,
      pageins:10000,pageouts:100,swapins:20,swapouts:30,
      compressions:400,decompressions:300,purges:200,reactivations:500,
      throttled_pages:0,wired_pages:1000,compressor_pages:2000,
      uncompressed_compressor_pages:3000,process_pageins:7};
  {
    schema_version:3,status:"pass",artifact_bytes:107431343168,layers:43,lanes:4,
    parity:{prefix_rows:148,steps:132,final_position:280,final_mod_4:0,
      final_mod_128:24,physical_to_logical:[2,0,3,1],
      exact_state_logits_cache_recurrent:true},
    residency:{effective_weight_mode:"mmap-file-backed",weight_bytes:100,
      weight_file_backed_bytes:90,weight_anonymous_bytes:10,
      weight_mapped_segment_count:8,shape_pass:true,
      serial_live_cache_bytes:20,cohort_live_cache_bytes:20,
      serial_snapshot_bytes:10,cohort_snapshot_bytes:10,tracked_total_bytes:160},
    phase_contract:{policy:"fsynced-run-bound-markers-v1",run_uuid:$uuid,pid:$pid,
      ack_timeout_seconds:300,
      markers:[marker(0;"process-start";0;1940000000000),
        marker(1;"loaded-settle-start";10000000000;1950000000000),
        marker(2;"measurement-ready";55000000000;1980000000000),
        marker(3;"measurement-complete";59000000000;2004000000000)]},
    darwin_vm_window:{policy:"darwin25-phase-bound-process-residency-v3",
      claim_scope:"within-run-paired-only",start:vm,
      end:(vm + {pageins:10005,pageouts:102,compressions:440,
        decompressions:330,purges:203,reactivations:505,wired_pages:1001}),
      deltas:{pageins:5,pageouts:2,swapins:0,swapouts:0,compressions:40,
        decompressions:30,purges:3,reactivations:5,process_pageins:0},
      gated_zero_deltas:["swapins","swapouts","process_pageins"],
      diagnostic_deltas:["pageins","pageouts","compressions","decompressions",
        "purges","reactivations"],
      monotonic_counters:true,pressure_boundary_pass:true,environment_pass:true},
    benchmark:{position:6676,anchor_exact_state_logits_cache_recurrent:true,
      logical_capacity:131072,loaded_idle_seconds:45,pairs:20,order:"alternating",
      serial_ms:[range(0;20)|if (. % 2) == 0 then 5 else 14 end],
      cohort_ms:[range(0;20)|10],serial_median_ms:9.5,
      cohort_median_ms:10,speedup:0.95,
      unconditioned_order_signature:{
        historical_signature:"even_delta_negative_odd_delta_positive",
        even_delta_median_ms:-5,odd_delta_median_ms:4,observed:true,gating:false},
      conditioned:{protocol:"same-topology-prime-restore-measure",
        primes_per_measurement:1,
        serial_prime_ms:[range(0;20)|21],cohort_prime_ms:[range(0;20)|11],
        serial_ms:[range(0;20)|20],cohort_ms:[range(0;20)|10],
        serial_median_ms:20,cohort_median_ms:10,speedup:2,
        even_order_speedup:2,odd_order_speedup:2,
        even_order_paired_delta_median_ms:10,
        odd_order_paired_delta_median_ms:10,performance_pass:true,
        serial_prime_counters:[range(0;20)|counter(92;4)],
        cohort_prime_counters:[range(0;20)|counter(23;1)],
        serial_counters:[range(0;20)|counter(92;4)],
        cohort_counters:[range(0;20)|counter(23;1)]},
      serial_counters:[range(0;20)|counter(92;4)],
      cohort_counters:[range(0;20)|counter(23;1)],
      serial_command_buffers_per_pair:92,cohort_command_buffers_per_pair:23,
      serial_synchronizations_per_pair:4,cohort_synchronizations_per_pair:1,
      topology_pass:true,topology_errors:[]},
    benchmark_environment:{profile:"clean-hf2q-mlx-metal-v1",
      override_variables_absent:true,unexpected_override_variables:[]}
  }
' >"$raw"
jq -cS '.phase_contract.markers[]' "$raw" >"$phase_log"

printf '2000\tnominal\tdecode-cohort-measurement-start\n' >"$measurement"
printf '2002\tfair\tdecode-cohort-measurement\n' >>"$measurement"
printf '2004\tfair\tdecode-cohort-measurement-end\n' >>"$measurement"
printf '2000\tquiet\tdecode-cohort-measurement-start\t100\t-\n' \
  >"$contention_measurement"
printf '2002\tquiet\tdecode-cohort-measurement\t100\t-\n' \
  >>"$contention_measurement"
printf '2004\tquiet\tdecode-cohort-measurement-end\t100\t-\n' \
  >>"$contention_measurement"

for timestamp in $(seq 1940 2 2000); do
  phase=decode-cohort-loaded-setup
  [[ "$timestamp" == 1940 ]] && phase=decode-cohort-loaded-setup-start
  [[ "$timestamp" == 2000 ]] && phase=decode-cohort-loaded-setup-end
  state=nominal
  ((timestamp >= 1950 && timestamp < 1970)) && state=fair
  printf '%s\t%s\t%s\n' "$timestamp" "$state" "$phase" >>"$setup_thermal"
  printf '%s\tquiet\t%s\t100\t-\n' "$timestamp" "$phase" \
    >>"$setup_contention"
done

memory_row() {
  local timestamp=$1 pressure=$2 free=$3 phase=$4 pageins=$5 swapouts=$6
  local reactivations=${7:-500}
  local pageouts=${8:-100} swapins=${9:-20} compressions=${10:-400}
  local decompressions=${11:-300} purges=${12:-200}
  printf '%s\t1786056685\t16384\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t0\t1000\t2000\t3000\t%s\n' \
    "$timestamp" "$pressure" "$free" "$pageins" "$pageouts" "$swapins" \
    "$swapouts" "$compressions" "$decompressions" "$purges" \
    "$reactivations" "$phase"
}
for timestamp in $(seq 1940 2 2000); do
  phase=decode-cohort-loaded-setup
  [[ "$timestamp" == 1940 ]] && phase=decode-cohort-loaded-setup-start
  [[ "$timestamp" == 2000 ]] && phase=decode-cohort-loaded-setup-end
  pressure=1
  ((timestamp >= 1980)) && pressure=2
  memory_row "$timestamp" "$pressure" 8 "$phase" \
    "$((9000 + timestamp - 1940))" 30 >>"$setup_memory"
done
awk -F '\t' '$1 > 1980' "$setup_memory" >"$loaded_idle_memory"
memory_row 2000 2 8 decode-cohort-measurement-start 10000 30 >"$memory_log"
memory_row 2002 2 8 decode-cohort-measurement 10003 30 502 101 20 425 318 201 \
  >>"$memory_log"
memory_row 2004 2 8 decode-cohort-measurement-end 10005 30 505 102 20 440 330 203 \
  >>"$memory_log"

for timestamp in $(seq 1000 5 1060); do
  printf '%s\tnominal\tdecode-cohort-settle\n' "$timestamp" >>"$settle"
  printf '%s\tquiet\tdecode-cohort-settle\t100\t-\n' "$timestamp" \
    >>"$contention_settle"
done

write_test_log() {
  local phases=$1 output=$2 duration=${3:-60.00}
  {
  printf '%s\n' \
    'test inference::models::deepseek4::real_artifact_decode_cohort_tests::official_artifact_b4_decode_body_is_exact_and_measured ... '
  while IFS= read -r marker; do
    printf 'HF2Q_DEEPSEEK4_PHASE %s\n' "$marker"
    done <"$phases"
  printf '%s\n' \
    'DeepSeek-V4 B=4 decode spike: exact_state_logits_cache_recurrent=true' \
    'ok' \
      ''
    printf 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 4653 filtered out; finished in %ss\n' \
      "$duration"
  } >"$output"
}
write_test_log "$phase_log" "$test_log"

write_summary() {
  local output=$1
  local raw_path=${2:-$raw}
  local phase_path=${3:-$phase_log}
  local test_log_path=${4:-$test_log}
  local measurement_path=${5:-$measurement}
  local memory_path=${6:-$memory_log}
  local setup_memory_path=${7:-$setup_memory}
  local setup_thermal_path=${8:-$setup_thermal}
  local loaded_idle_memory_path=${9:-$loaded_idle_memory}

  thermal_validate_fair_or_better_measurement_log "$measurement_path" 5 || return 1
  local measurement_samples=$THERMAL_LOG_SAMPLES
  local measurement_duration=$THERMAL_LOG_DURATION_SECONDS
  local measurement_non_nominal=$THERMAL_LOG_NON_NOMINAL_SAMPLES
  local measurement_fair=$THERMAL_LOG_FAIR_SAMPLES
  local measurement_over=$THERMAL_LOG_OVER_LIMIT_SAMPLES
  local measurement_gaps=$THERMAL_LOG_GAPS
  thermal_validate_fair_or_better_measurement_log "$setup_thermal_path" 5 || return 1
  local setup_samples=$THERMAL_LOG_SAMPLES
  local setup_duration=$THERMAL_LOG_DURATION_SECONDS
  local setup_fair=$THERMAL_LOG_FAIR_SAMPLES
  local setup_gaps=$THERMAL_LOG_GAPS
  thermal_validate_settle_log "$setup_thermal_path" 0 5 || return 1
  local setup_nominal_tail=$THERMAL_LOG_DURATION_SECONDS
  memory_validate_warning_log "$setup_memory_path" 5 1 || return 1
  local setup_memory_json
  setup_memory_json=$(memory_log_summary_json)
  memory_validate_warning_log "$loaded_idle_memory_path" 5 1 || return 1
  local loaded_idle_memory_json
  loaded_idle_memory_json=$(memory_log_summary_json)
  memory_validate_warning_log "$memory_path" 5 0 || return 1
  local measurement_memory_json
  measurement_memory_json=$(memory_log_summary_json)

  jq --arg source_sha "$source_sha" --arg model_sha256 "$model_sha" \
    --arg raw_sha256 "$(sha256_file "$raw_path")" \
    --arg test_log_sha256 "$(sha256_file "$test_log_path")" \
    --arg phase_log_sha256 "$(sha256_file "$phase_path")" \
    --arg run_uuid "$(jq -er .phase_contract.run_uuid "$raw_path")" \
    --arg measurement_log_sha256 "$(sha256_file "$measurement_path")" \
    --arg settle_log_sha256 "$(sha256_file "$settle")" \
    --arg setup_thermal_log_sha256 "$(sha256_file "$setup_thermal_path")" \
    --arg contention_measurement_sha "$(sha256_file "$contention_measurement")" \
    --arg contention_settle_sha "$(sha256_file "$contention_settle")" \
    --arg setup_contention_sha "$(sha256_file "$setup_contention")" \
    --arg memory_log_sha256 "$(sha256_file "$memory_path")" \
    --arg setup_memory_sha "$(sha256_file "$setup_memory_path")" \
    --arg loaded_idle_memory_sha "$(sha256_file "$loaded_idle_memory_path")" \
    --arg memory_guard_sha "$memory_guard_source_sha" \
    --arg thermal_source_sha "$thermal_probe_source_sha" \
    --arg thermal_compiler_sha "$thermal_probe_compiler_sha" \
    --arg thermal_binary_sha "$thermal_probe_binary_sha" \
    --argjson producer_pid "$(jq -er .phase_contract.pid "$raw_path")" \
    --argjson measurement_samples "$measurement_samples" \
    --argjson measurement_duration "$measurement_duration" \
    --argjson measurement_non_nominal "$measurement_non_nominal" \
    --argjson measurement_fair "$measurement_fair" \
    --argjson measurement_over "$measurement_over" \
    --argjson measurement_gaps "$measurement_gaps" \
    --argjson setup_samples "$setup_samples" \
    --argjson setup_duration "$setup_duration" \
    --argjson setup_fair "$setup_fair" --argjson setup_gaps "$setup_gaps" \
    --argjson setup_nominal_tail "$setup_nominal_tail" \
    --argjson setup_memory_json "$setup_memory_json" \
    --argjson loaded_idle_memory_json "$loaded_idle_memory_json" \
    --argjson measurement_memory_json "$measurement_memory_json" '
    . + {schema_version:6,source_sha:$source_sha,model_sha256:$model_sha256,
      mlx_native_version:"0.15.1",producer_exit_code:0,raw_sha256:$raw_sha256,
      test_log_sha256:$test_log_sha256,
      phase_evidence:{policy:"fsynced-run-bound-markers-v1",run_uuid:$run_uuid,
        producer_pid:$producer_pid,test_spawned_at:1940,log_sha256:$phase_log_sha256},
      thermal_status:"fair_or_better",required_start_state:"nominal",
      maximum_measurement_state:"fair",measurement_log_sha256:$measurement_log_sha256,
      settle_log_sha256:$settle_log_sha256,settle_seconds:60,
      thermal_probe:{implementation:"compiled-foundation-helper",
        source_path:"scripts/macos_thermal_probe.swift",source_sha256:$thermal_source_sha,
        compiler_path:"/usr/bin/swiftc",compiler_sha256:$thermal_compiler_sha,
        compiler_version:"Apple Swift version 6.2 (synthetic)",
        binary_sha256:$thermal_binary_sha},
      settle_samples:13,settle_duration_seconds:60,settle_sample_interval_seconds:5,
      maximum_settle_sample_gap_seconds:8,settle_telemetry_gaps:0,
      loaded_setup:{thermal:{log_sha256:$setup_thermal_log_sha256,
          samples:$setup_samples,duration_seconds:$setup_duration,
          fair_samples:$setup_fair,telemetry_gaps:$setup_gaps,
          required_nominal_tail_seconds:30,
          nominal_tail_seconds:$setup_nominal_tail,
          nominal_wait_timeout_seconds:240},
        host_contention:{log_sha256:$setup_contention_sha,samples:$setup_samples,
          duration_seconds:$setup_duration,contended_samples:0,telemetry_gaps:0}},
      measurement_samples:$measurement_samples,
      measurement_duration_seconds:$measurement_duration,sample_interval_seconds:2,
      maximum_sample_gap_seconds:5,
      non_nominal_measurement_samples:$measurement_non_nominal,
      fair_measurement_samples:$measurement_fair,
      over_limit_measurement_samples:$measurement_over,telemetry_gaps:$measurement_gaps,
      host_contention:{policy:"process-group-v1",
        settle:{log_sha256:$contention_settle_sha,samples:13,duration_seconds:60,
          contended_samples:0,telemetry_gaps:0},
        measurement:{log_sha256:$contention_measurement_sha,
          samples:$measurement_samples,duration_seconds:$measurement_duration,
          contended_samples:0,telemetry_gaps:0}},
      memory_pressure:{policy:"darwin25-phase-bound-process-residency-v3",
        normal_level:1,warning_level:2,critical_level:4,
        claim_scope:"within-run-paired-only",
        guard_source_path:"scripts/macos_memory_guard.sh",
        guard_source_sha256:$memory_guard_sha,sample_interval_seconds:2,
        maximum_sample_gap_seconds:5,
        setup:($setup_memory_json + {log_sha256:$setup_memory_sha}),
        loaded_idle:($loaded_idle_memory_json + {
          log_sha256:$loaded_idle_memory_sha,
          phase:"post-ready-pre-ack",gating:false}),
        measurement:($measurement_memory_json + {log_sha256:$memory_log_sha256}),
        exact_window:.darwin_vm_window}}
  ' "$raw_path" >"$output"
}

verify() {
  local summary_path=$1 raw_path=${2:-$raw} test_path=${3:-$test_log}
  local measurement_path=${4:-$measurement} memory_path=${5:-$memory_log}
  local phase_path=${6:-$phase_log} setup_memory_path=${7:-$setup_memory}
  local setup_thermal_path=${8:-$setup_thermal}
  local loaded_idle_memory_path=${9:-$loaded_idle_memory}
  bash "$VERIFY" "$summary_path" "$raw_path" "$test_path" \
    "$measurement_path" "$settle" "$source_sha" "$model_sha" \
    "$contention_measurement" "$contention_settle" "$memory_path" \
    "$phase_path" "$setup_thermal_path" "$setup_contention" \
    "$setup_memory_path" "$loaded_idle_memory_path"
}

write_summary "$summary"
verify "$summary"

expect_reject() {
  local label=$1 summary_path=$2 raw_path=${3:-$raw} test_path=${4:-$test_log}
  local measurement_path=${5:-$measurement} memory_path=${6:-$memory_log}
  local phase_path=${7:-$phase_log} setup_memory_path=${8:-$setup_memory}
  local setup_thermal_path=${9:-$setup_thermal}
  local loaded_idle_memory_path=${10:-$loaded_idle_memory}
  if verify "$summary_path" "$raw_path" "$test_path" "$measurement_path" \
      "$memory_path" "$phase_path" "$setup_memory_path" \
      "$setup_thermal_path" "$loaded_idle_memory_path" >/dev/null 2>&1; then
    echo "decode-cohort verifier accepted invalid case: $label" >&2
    exit 1
  fi
}

for mutation in \
  '.schema_version = 5' \
  '.benchmark.speedup = 1' \
  '.producer_exit_code = 1' \
  '.measurement_samples += 1' \
  '.raw_sha256 = ("0" * 64)' \
  '.phase_evidence.log_sha256 = ("0" * 64)' \
  '.memory_pressure.policy = "darwin25-normal-no-swapout-v1"' \
  'del(.memory_pressure.loaded_idle)' \
  '.required_start_state = "fair"' \
  '.maximum_measurement_state = "nominal"' \
  '.loaded_setup.thermal.nominal_tail_seconds += 1' \
  '.loaded_setup.thermal.required_nominal_tail_seconds = 20' \
  '.loaded_setup.thermal.nominal_wait_timeout_seconds = 300' \
  'del(.loaded_setup.thermal.nominal_tail_seconds)' \
  '.thermal_probe.source_sha256 = ("0" * 64)' \
  'del(.thermal_probe.binary_sha256)'; do
  label=$(printf '%s' "$mutation" | shasum | cut -c1-8)
  jq "$mutation" "$summary" >"$tmp_dir/$label-summary.json"
  expect_reject "$label" "$tmp_dir/$label-summary.json"
done

consistent_raw_reject() {
  local label=$1 filter=$2
  local changed_raw="$tmp_dir/$label-raw.json"
  local changed_phase="$tmp_dir/$label-phases.jsonl"
  local changed_test_log="$tmp_dir/$label-test.log"
  local changed_summary="$tmp_dir/$label-summary.json"
  jq "$filter" "$raw" >"$changed_raw"
  jq -cS '.phase_contract.markers[]' "$changed_raw" >"$changed_phase"
  write_test_log "$changed_phase" "$changed_test_log"
  write_summary "$changed_summary" "$changed_raw" "$changed_phase" \
    "$changed_test_log"
  expect_reject "$label" "$changed_summary" "$changed_raw" \
    "$changed_test_log" \
    "$measurement" "$memory_log" "$changed_phase"
}

consistent_vm_churn_reject() {
  local label=$1 field=$2 memory_column=$3
  local changed_raw="$tmp_dir/$label-raw.json"
  local changed_phase="$tmp_dir/$label-phases.jsonl"
  local changed_memory="$tmp_dir/$label-memory.log"
  local changed_summary="$tmp_dir/$label-summary.json"
  jq --arg field "$field" '
    .darwin_vm_window.end[$field] += 1
    | .darwin_vm_window.deltas[$field] = 1
  ' "$raw" >"$changed_raw"
  jq -cS '.phase_contract.markers[]' "$changed_raw" >"$changed_phase"
  awk -F '\t' -v column="$memory_column" 'BEGIN { OFS="\t" }
    { if (NR == 3) $column += 1; print }
  ' "$memory_log" >"$changed_memory"
  write_summary "$changed_summary" "$changed_raw" "$changed_phase" \
    "$test_log" "$measurement" "$changed_memory"
  expect_reject "$label" "$changed_summary" "$changed_raw" "$test_log" \
    "$measurement" "$changed_memory" "$changed_phase"
}

# Keep status/environment_pass positive and update the enclosing sampled
# counter. Retained zero gates must still be rejected by independent replay.
consistent_vm_churn_reject vm-swapin-churn swapins 8
consistent_vm_churn_reject vm-swapout-churn swapouts 9
consistent_raw_reject process-pagein-churn '
  .darwin_vm_window.end.process_pageins += 1
  | .darwin_vm_window.deltas.process_pageins = 1'
consistent_raw_reject missing-diagnostic-delta \
  'del(.darwin_vm_window.deltas.compressions)'
consistent_raw_reject legacy-v2-policy '
  .schema_version = 2
  | .darwin_vm_window.policy = "darwin25-phase-bound-no-vm-churn-v2"'
consistent_raw_reject changed-boot-epoch '
  .darwin_vm_window.end.boot_time_seconds += 1'
consistent_raw_reject changed-page-size '
  .darwin_vm_window.end.page_size = 4096'
consistent_raw_reject critical-window-boundary '
  .darwin_vm_window.start.pressure_level = 4'
consistent_raw_reject marker-ready-after-start-sample '
  .phase_contract.markers[2].wall_ns = 2001000000000'
consistent_raw_reject marker-complete-after-end-sample '
  .phase_contract.markers[3].wall_ns = 2005000000000'
consistent_raw_reject process-marker-before-spawn '
  .phase_contract.markers[0].wall_ns = 1939000000000'
consistent_raw_reject short-loaded-settle \
  '.phase_contract.markers[2].monotonic_ns = 54000000000'
consistent_raw_reject wrong-marker-run \
  '.phase_contract.markers[2].run_uuid = "ffffffff-ffff-ffff-ffff-ffffffffffff"'
consistent_raw_reject wrong-marker-process \
  '.phase_contract.markers[2].pid = 4243'
consistent_raw_reject anonymous-residency \
  '.residency.weight_file_backed_bytes = 40
   | .residency.weight_anonymous_bytes = 60'
consistent_raw_reject residency-sum-mismatch \
  '.residency.tracked_total_bytes = 159'
consistent_raw_reject topology-counter-lie \
  '.benchmark.conditioned.serial_counters[0].command_buffers = 91'
consistent_raw_reject topology-verdict-failure '
  .benchmark.topology_pass = false
  | .benchmark.topology_errors = ["synthetic topology failure"]'
consistent_raw_reject failed-producer-status '.status = "fail"'
consistent_raw_reject conditioned-even-stratum-failure '
  .benchmark.conditioned.serial_ms = [range(0;20)
    | if (. % 2) == 0 then 9 else 20 end]
  | .benchmark.conditioned.serial_median_ms = 14.5
  | .benchmark.conditioned.speedup = 1.45
  | .benchmark.conditioned.even_order_speedup = 0.9
  | .benchmark.conditioned.even_order_paired_delta_median_ms = -1'

head -3 "$phase_log" >"$tmp_dir/missing-phase.jsonl"
write_summary "$tmp_dir/missing-phase-summary.json" "$raw" \
  "$tmp_dir/missing-phase.jsonl"
expect_reject missing-phase "$tmp_dir/missing-phase-summary.json" "$raw" \
  "$test_log" "$measurement" "$memory_log" "$tmp_dir/missing-phase.jsonl"

jq -cS '.phase_contract.markers | .[0],.[2],.[1],.[3]' "$raw" \
  >"$tmp_dir/reordered-phases.jsonl"
write_summary "$tmp_dir/reordered-phases-summary.json" "$raw" \
  "$tmp_dir/reordered-phases.jsonl"
expect_reject reordered-phases "$tmp_dir/reordered-phases-summary.json" \
  "$raw" "$test_log" "$measurement" "$memory_log" \
  "$tmp_dir/reordered-phases.jsonl"

jq -cS '.phase_contract.markers | .[0],.[1],.[1],.[3]' "$raw" \
  >"$tmp_dir/duplicate-phases.jsonl"
write_summary "$tmp_dir/duplicate-phases-summary.json" "$raw" \
  "$tmp_dir/duplicate-phases.jsonl"
expect_reject duplicate-phases "$tmp_dir/duplicate-phases-summary.json" \
  "$raw" "$test_log" "$measurement" "$memory_log" \
  "$tmp_dir/duplicate-phases.jsonl"

write_test_log "$phase_log" "$tmp_dir/short-duration-test.log" 58.00
write_summary "$tmp_dir/short-duration-summary.json" "$raw" "$phase_log" \
  "$tmp_dir/short-duration-test.log"
expect_reject short-test-duration "$tmp_dir/short-duration-summary.json" \
  "$raw" "$tmp_dir/short-duration-test.log"

head -2 "$measurement" >"$tmp_dir/truncated-measurement.log"
write_summary "$tmp_dir/truncated-measurement-summary.json" "$raw" "$phase_log" \
  "$test_log" "$tmp_dir/truncated-measurement.log"
expect_reject truncated-measurement \
  "$tmp_dir/truncated-measurement-summary.json" "$raw" "$test_log" \
  "$tmp_dir/truncated-measurement.log"

grep -v 'measurement-complete' "$test_log" >"$tmp_dir/missing-test-marker.log"
write_summary "$tmp_dir/missing-test-marker-summary.json" "$raw" "$phase_log" \
  "$tmp_dir/missing-test-marker.log"
expect_reject missing-test-marker "$tmp_dir/missing-test-marker-summary.json" \
  "$raw" "$tmp_dir/missing-test-marker.log"

awk 'BEGIN { changed=0 }
  !changed && index($0, "\"pid\":4242") {
    sub(/\"pid\":4242/, "\"pid\":4243"); changed=1
  }
  { print }
' "$test_log" >"$tmp_dir/mismatched-test-marker.log"
write_summary "$tmp_dir/mismatched-test-marker-summary.json" "$raw" "$phase_log" \
  "$tmp_dir/mismatched-test-marker.log"
expect_reject mismatched-test-marker \
  "$tmp_dir/mismatched-test-marker-summary.json" "$raw" \
  "$tmp_dir/mismatched-test-marker.log"

sed 's/^test result: ok\./test result: FAILED./' "$test_log" \
  >"$tmp_dir/failed-test.log"
write_summary "$tmp_dir/failed-test-summary.json" "$raw" "$phase_log" \
  "$tmp_dir/failed-test.log"
expect_reject failed-test "$tmp_dir/failed-test-summary.json" "$raw" \
  "$tmp_dir/failed-test.log"

{
  cat "$test_log"
  printf '%s\n' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s'
} >"$tmp_dir/concatenated-results.log"
write_summary "$tmp_dir/concatenated-results-summary.json" "$raw" \
  "$phase_log" "$tmp_dir/concatenated-results.log"
expect_reject concatenated-results \
  "$tmp_dir/concatenated-results-summary.json" "$raw" \
  "$tmp_dir/concatenated-results.log"

grep -v '^test inference::models::deepseek4::real_artifact_decode_cohort_tests::official_artifact_b4_decode_body_is_exact_and_measured' \
  "$test_log" >"$tmp_dir/missing-test-name.log"
write_summary "$tmp_dir/missing-test-name-summary.json" "$raw" "$phase_log" \
  "$tmp_dir/missing-test-name.log"
expect_reject missing-test-name "$tmp_dir/missing-test-name-summary.json" \
  "$raw" "$tmp_dir/missing-test-name.log"

awk -F '\t' 'BEGIN { OFS="\t" } NR == 1 { $2="fair" } { print }' \
  "$setup_thermal" >"$tmp_dir/fair-setup-start.log"
write_summary "$tmp_dir/fair-setup-start-summary.json" "$raw" "$phase_log" \
  "$test_log" "$measurement" "$memory_log" "$setup_memory" \
  "$tmp_dir/fair-setup-start.log"
expect_reject fair-setup-start "$tmp_dir/fair-setup-start-summary.json" \
  "$raw" "$test_log" "$measurement" "$memory_log" "$phase_log" \
  "$setup_memory" "$tmp_dir/fair-setup-start.log"

awk -F '\t' 'BEGIN { OFS="\t" } NR == 1 { $2="fair" } { print }' \
  "$measurement" >"$tmp_dir/fair-measurement-start.log"
write_summary "$tmp_dir/fair-measurement-start-summary.json" "$raw" \
  "$phase_log" "$test_log" "$tmp_dir/fair-measurement-start.log"
expect_reject fair-measurement-start \
  "$tmp_dir/fair-measurement-start-summary.json" "$raw" "$test_log" \
  "$tmp_dir/fair-measurement-start.log"

# A nominal start sample alone is insufficient. The loaded producer must be
# held at the readiness barrier until the setup log proves a continuously
# nominal 30-second tail immediately before measurement is armed.
awk -F '\t' 'BEGIN { OFS="\t" } $1 == 1980 { $2="fair" } { print }' \
  "$setup_thermal" >"$tmp_dir/short-loaded-nominal-tail.log"
write_summary "$tmp_dir/short-loaded-nominal-tail-summary.json" "$raw" \
  "$phase_log" "$test_log" "$measurement" "$memory_log" "$setup_memory" \
  "$tmp_dir/short-loaded-nominal-tail.log"
expect_reject short-loaded-nominal-tail \
  "$tmp_dir/short-loaded-nominal-tail-summary.json" "$raw" "$test_log" \
  "$measurement" "$memory_log" "$phase_log" "$setup_memory" \
  "$tmp_dir/short-loaded-nominal-tail.log"

# Even a valid 30-second nominal tail is not the same run if it is separated
# from measurement arming by an unobserved telemetry gap.
awk -F '\t' 'BEGIN { OFS="\t" } { $1 += 6; print }' "$measurement" \
  >"$tmp_dir/delayed-measurement.log"
awk -F '\t' 'BEGIN { OFS="\t" } { $1 += 6; print }' \
  "$contention_measurement" >"$tmp_dir/delayed-contention.log"
awk -F '\t' 'BEGIN { OFS="\t" } { $1 += 6; print }' "$memory_log" \
  >"$tmp_dir/delayed-memory.log"
write_summary "$tmp_dir/delayed-summary-base.json" "$raw" "$phase_log" \
  "$test_log" "$tmp_dir/delayed-measurement.log" \
  "$tmp_dir/delayed-memory.log"
jq --arg sha "$(sha256_file "$tmp_dir/delayed-contention.log")" \
  '.host_contention.measurement.log_sha256 = $sha' \
  "$tmp_dir/delayed-summary-base.json" >"$tmp_dir/delayed-summary.json"
if bash "$VERIFY" "$tmp_dir/delayed-summary.json" "$raw" "$test_log" \
    "$tmp_dir/delayed-measurement.log" "$settle" "$source_sha" \
    "$model_sha" "$tmp_dir/delayed-contention.log" \
    "$contention_settle" "$tmp_dir/delayed-memory.log" "$phase_log" \
    "$setup_thermal" "$setup_contention" "$setup_memory" \
    "$loaded_idle_memory" \
    >/dev/null 2>&1; then
  echo "decode-cohort verifier accepted a detached loaded nominal tail" >&2
  exit 1
fi

awk -F '\t' 'BEGIN { OFS="\t" } NR > 1 { $9=31 } { print }' \
  "$setup_memory" >"$tmp_dir/setup-swapout-growth.log"
if write_summary "$tmp_dir/setup-swapout-growth-summary.json" "$raw" \
    "$phase_log" "$test_log" "$measurement" "$memory_log" \
    "$tmp_dir/setup-swapout-growth.log" >/dev/null 2>&1; then
  echo "summary builder accepted setup swapout growth" >&2
  exit 1
fi

# Diagnostic host-global counters remain mandatory and must enclose the
# in-process exact window even though their nonzero values do not fail v3.
awk -F '\t' 'BEGIN { OFS="\t" } NR == 3 { $10=439 } { print }' \
  "$memory_log" >"$tmp_dir/nonenclosing-compression.log"
write_summary "$tmp_dir/nonenclosing-compression-summary.json" "$raw" \
  "$phase_log" "$test_log" "$measurement" \
  "$tmp_dir/nonenclosing-compression.log"
expect_reject nonenclosing-compression \
  "$tmp_dir/nonenclosing-compression-summary.json" "$raw" "$test_log" \
  "$measurement" "$tmp_dir/nonenclosing-compression.log"

# The permanent idle control is the exact post-ready suffix of setup telemetry,
# not any other internally valid selection of samples.
tail -9 "$loaded_idle_memory" >"$tmp_dir/detached-loaded-idle.log"
write_summary "$tmp_dir/detached-loaded-idle-summary.json" "$raw" \
  "$phase_log" "$test_log" "$measurement" "$memory_log" "$setup_memory" \
  "$setup_thermal" "$tmp_dir/detached-loaded-idle.log"
expect_reject detached-loaded-idle \
  "$tmp_dir/detached-loaded-idle-summary.json" "$raw" "$test_log" \
  "$measurement" "$memory_log" "$phase_log" "$setup_memory" \
  "$setup_thermal" "$tmp_dir/detached-loaded-idle.log"

for memory_mutation in nonmonotonic-pageins changed-log-boot throttled-page; do
  case "$memory_mutation" in
    nonmonotonic-pageins)
      awk -F '\t' 'BEGIN { OFS="\t" } NR == 2 { $6=9999 } { print }' \
        "$memory_log" >"$tmp_dir/$memory_mutation.log"
      ;;
    changed-log-boot)
      awk -F '\t' 'BEGIN { OFS="\t" } NR == 2 { $2+=1 } { print }' \
        "$memory_log" >"$tmp_dir/$memory_mutation.log"
      ;;
    throttled-page)
      awk -F '\t' 'BEGIN { OFS="\t" } NR == 2 { $14=1 } { print }' \
        "$memory_log" >"$tmp_dir/$memory_mutation.log"
      ;;
  esac
  if write_summary "$tmp_dir/$memory_mutation-summary.json" "$raw" \
      "$phase_log" "$test_log" "$measurement" \
      "$tmp_dir/$memory_mutation.log" "$setup_memory" >/dev/null 2>&1; then
    echo "summary builder accepted invalid memory log: $memory_mutation" >&2
    exit 1
  fi
done

awk -F '\t' 'BEGIN { OFS="\t" } NR == 2 { $4=4 } { print }' \
  "$memory_log" >"$tmp_dir/critical-memory.log"
if write_summary "$tmp_dir/critical-memory-summary.json" "$raw" "$phase_log" \
    "$test_log" "$measurement" "$tmp_dir/critical-memory.log" \
    "$setup_memory" >/dev/null 2>&1; then
  echo "summary builder accepted critical memory pressure" >&2
  exit 1
fi
jq --arg sha "$(sha256_file "$tmp_dir/critical-memory.log")" \
  '.memory_pressure.measurement.log_sha256 = $sha' "$summary" \
  >"$tmp_dir/critical-verifier-summary.json"
expect_reject critical-memory-verifier \
  "$tmp_dir/critical-verifier-summary.json" "$raw" "$test_log" \
  "$measurement" "$tmp_dir/critical-memory.log"

awk -F '\t' 'BEGIN { OFS="\t" } NR == 2 { $2="contended"; $5="9:9:rustc" }
  { print }' "$contention_measurement" >"$tmp_dir/contended.log"
jq --arg sha "$(sha256_file "$tmp_dir/contended.log")" \
  '.host_contention.measurement.log_sha256=$sha
   | .host_contention.measurement.contended_samples=1' \
  "$summary" >"$tmp_dir/contended-summary.json"
if bash "$VERIFY" "$tmp_dir/contended-summary.json" "$raw" "$test_log" \
    "$measurement" "$settle" "$source_sha" "$model_sha" \
    "$tmp_dir/contended.log" "$contention_settle" "$memory_log" "$phase_log" \
    "$setup_thermal" "$setup_contention" "$setup_memory" \
    "$loaded_idle_memory" >/dev/null 2>&1; then
  echo "decode-cohort verifier accepted host contention" >&2
  exit 1
fi

echo "DeepSeek-V4 decode-cohort receipt contract: pass"
