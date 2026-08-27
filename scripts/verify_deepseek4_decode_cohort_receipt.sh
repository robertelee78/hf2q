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
memory_log=${10:?memory-pressure log is required}
phase_log=${11:?phase-marker log is required}
setup_thermal_log=${12:?loaded-setup thermal log is required}
setup_contention_log=${13:?loaded-setup contention log is required}
setup_memory_log=${14:?loaded-setup memory log is required}
loaded_idle_memory_log=${15:?loaded-idle memory log is required}
dependency_receipt=${16:?verified dependency receipt is required}
expected_dependency_receipt_sha=${17:?verified dependency receipt SHA-256 is required}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=scripts/macos_thermal_guard.sh
source "$ROOT_DIR/scripts/macos_thermal_guard.sh"
# shellcheck source=scripts/macos_memory_guard.sh
source "$ROOT_DIR/scripts/macos_memory_guard.sh"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

for path in "$summary" "$raw" "$test_log" "$measurement_log" "$settle_log" \
  "$contention_measurement_log" "$contention_settle_log" "$memory_log" \
  "$phase_log" "$setup_thermal_log" "$setup_contention_log" \
  "$setup_memory_log" "$loaded_idle_memory_log" "$dependency_receipt"; do
  [[ -s "$path" ]] || {
    echo "decode-cohort receipt input is missing or empty: $path" >&2
    exit 1
  }
done
[[ "$expected_source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$expected_model_sha" =~ ^[0-9a-f]{64}$ ]]
[[ ! -L "$dependency_receipt" \
  && "$expected_dependency_receipt_sha" =~ ^[0-9a-f]{64}$ ]]
test "$(sha256_file "$dependency_receipt")" = \
  "$expected_dependency_receipt_sha"
jq -e '
  .schema_version == 1 and .status == "pass"
  and .dependency.name == "mlx-native"
  and (.dependency.version
    | test("^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"))
  and .dependency.requirement == ("=" + .dependency.version)
  and .dependency.source
    == "registry+https://github.com/rust-lang/crates.io-index"
  and (.dependency.checksum | test("^[0-9a-f]{64}$"))
' "$dependency_receipt" >/dev/null

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
test "$(sha256_file "$memory_log")" = \
  "$(jq -er .memory_pressure.measurement.log_sha256 "$summary")"
test "$(sha256_file "$phase_log")" = \
  "$(jq -er .phase_evidence.log_sha256 "$summary")"
test "$(sha256_file "$setup_thermal_log")" = \
  "$(jq -er .loaded_setup.thermal.log_sha256 "$summary")"
test "$(sha256_file "$setup_contention_log")" = \
  "$(jq -er .loaded_setup.host_contention.log_sha256 "$summary")"
test "$(sha256_file "$setup_memory_log")" = \
  "$(jq -er .memory_pressure.setup.log_sha256 "$summary")"
test "$(sha256_file "$loaded_idle_memory_log")" = \
  "$(jq -er .memory_pressure.loaded_idle.log_sha256 "$summary")"
jq -s -e 'length == 1' "$summary" >/dev/null

jq -e --slurpfile raw "$raw" \
  --slurpfile dependency "$dependency_receipt" \
  --arg source_sha "$expected_source_sha" \
  --arg model_sha256 "$expected_model_sha" '
  def abs: if . < 0 then -. else . end;
  def median:
    sort as $sorted
    | ($sorted | length) as $length
    | if ($length % 2) == 0
      then (($sorted[$length / 2 - 1] + $sorted[$length / 2]) / 2)
      else $sorted[($length / 2 | floor)]
      end;
  def stratum($parity):
    to_entries | map(select((.key % 2) == $parity) | .value);
  def deltas($serial; $cohort):
    [range(0; ($serial | length)) | $serial[.] - $cohort[.]];
  . as $summary
  | $raw[0] as $receipt
  | ($receipt.benchmark.serial_ms | median) as $serial_median
  | ($receipt.benchmark.cohort_ms | median) as $cohort_median
  | ($receipt.benchmark.conditioned.serial_ms | median) as $conditioned_serial_median
  | ($receipt.benchmark.conditioned.cohort_ms | median) as $conditioned_cohort_median
  | ($receipt.benchmark.conditioned.serial_ms | stratum(0) | median) as $conditioned_serial_even_median
  | ($receipt.benchmark.conditioned.cohort_ms | stratum(0) | median) as $conditioned_cohort_even_median
  | ($receipt.benchmark.conditioned.serial_ms | stratum(1) | median) as $conditioned_serial_odd_median
  | ($receipt.benchmark.conditioned.cohort_ms | stratum(1) | median) as $conditioned_cohort_odd_median
  | deltas($receipt.benchmark.serial_ms;
      $receipt.benchmark.cohort_ms) as $unconditioned_deltas
  | ($unconditioned_deltas | stratum(0) | median) as $unconditioned_even_delta
  | ($unconditioned_deltas | stratum(1) | median) as $unconditioned_odd_delta
  | deltas($receipt.benchmark.conditioned.serial_ms;
      $receipt.benchmark.conditioned.cohort_ms) as $conditioned_deltas
  | ($conditioned_deltas | stratum(0) | median) as $conditioned_even_delta
  | ($conditioned_deltas | stratum(1) | median) as $conditioned_odd_delta
  | ($raw | length) == 1
    and ($summary | del(
      .schema_version,
      .source_sha,.model_sha256,.mlx_native_version,.producer_exit_code,.raw_sha256,
      .test_log_sha256,.thermal_status,.required_start_state,
      .maximum_measurement_state,.measurement_log_sha256,
      .settle_log_sha256,.settle_seconds,.settle_samples,
      .thermal_probe,
      .settle_duration_seconds,.settle_sample_interval_seconds,
      .maximum_settle_sample_gap_seconds,.settle_telemetry_gaps,
      .measurement_samples,.measurement_duration_seconds,
      .sample_interval_seconds,.maximum_sample_gap_seconds,
      .non_nominal_measurement_samples,.fair_measurement_samples,
      .over_limit_measurement_samples,.telemetry_gaps,.host_contention,.loaded_setup,
      .phase_evidence,
      .memory_pressure
    )) == ($receipt | del(.schema_version))
    and $receipt.schema_version == 3
    and .schema_version == 6 and .status == "pass"
    and .source_sha == $source_sha and .model_sha256 == $model_sha256
    and .mlx_native_version == $dependency[0].dependency.version
    and .producer_exit_code == 0
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
    and .residency.effective_weight_mode == "mmap-file-backed"
    and .residency.weight_bytes > 0
    and .residency.weight_file_backed_bytes > .residency.weight_anonymous_bytes
    and .residency.weight_anonymous_bytes >= 0
    and .residency.weight_mapped_segment_count > 0
    and .residency.weight_bytes == (
      .residency.weight_file_backed_bytes + .residency.weight_anonymous_bytes)
    and .residency.shape_pass == true
    and .residency.serial_live_cache_bytes > 0
    and .residency.cohort_live_cache_bytes > 0
    and .residency.serial_snapshot_bytes > 0
    and .residency.cohort_snapshot_bytes > 0
    and .residency.tracked_total_bytes == (
      .residency.weight_bytes + .residency.serial_live_cache_bytes
      + .residency.cohort_live_cache_bytes + .residency.serial_snapshot_bytes
      + .residency.cohort_snapshot_bytes)
    and .phase_contract.policy == "fsynced-run-bound-markers-v1"
    and (.phase_contract.run_uuid
      | test("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"))
    and (.phase_contract.pid | type) == "number" and .phase_contract.pid > 0
    and .phase_contract.ack_timeout_seconds == 300
    and (.phase_contract.markers | length) == 4
    and all(.phase_contract.markers[];
      .run_uuid == $receipt.phase_contract.run_uuid
      and .pid == $receipt.phase_contract.pid
      and (.monotonic_ns | type) == "number" and .monotonic_ns >= 0
      and (.wall_ns | type) == "number" and .wall_ns > 0)
    and [.phase_contract.markers[].sequence] == [0,1,2,3]
    and [.phase_contract.markers[].phase] == ["process-start",
      "loaded-settle-start","measurement-ready","measurement-complete"]
    and .phase_contract.markers[0].monotonic_ns
      < .phase_contract.markers[1].monotonic_ns
    and .phase_contract.markers[1].monotonic_ns
      < .phase_contract.markers[2].monotonic_ns
    and .phase_contract.markers[2].monotonic_ns
      < .phase_contract.markers[3].monotonic_ns
    and (.phase_contract.markers[2].monotonic_ns
      - .phase_contract.markers[1].monotonic_ns) >= 45000000000
    and .darwin_vm_window.policy == "darwin25-phase-bound-process-residency-v3"
    and .darwin_vm_window.claim_scope == "within-run-paired-only"
    and .darwin_vm_window.monotonic_counters == true
    and .darwin_vm_window.pressure_boundary_pass == true
    and .darwin_vm_window.gated_zero_deltas
      == ["swapins","swapouts","process_pageins"]
    and .darwin_vm_window.diagnostic_deltas
      == ["pageins","pageouts","compressions","decompressions",
          "purges","reactivations"]
    and .darwin_vm_window.environment_pass == true
    and .darwin_vm_window.start.boot_time_seconds
      == .darwin_vm_window.end.boot_time_seconds
    and .darwin_vm_window.start.page_size == .darwin_vm_window.end.page_size
    and .darwin_vm_window.start.page_size > 0
    and (.darwin_vm_window.start.pressure_level == 1
      or .darwin_vm_window.start.pressure_level == 2)
    and (.darwin_vm_window.end.pressure_level == 1
      or .darwin_vm_window.end.pressure_level == 2)
    and .darwin_vm_window.start.throttled_pages == 0
    and .darwin_vm_window.end.throttled_pages == 0
    and .darwin_vm_window.deltas.pageins
      == (.darwin_vm_window.end.pageins - .darwin_vm_window.start.pageins)
    and .darwin_vm_window.deltas.pageouts
      == (.darwin_vm_window.end.pageouts - .darwin_vm_window.start.pageouts)
    and .darwin_vm_window.deltas.swapins
      == (.darwin_vm_window.end.swapins - .darwin_vm_window.start.swapins)
    and .darwin_vm_window.deltas.swapouts
      == (.darwin_vm_window.end.swapouts - .darwin_vm_window.start.swapouts)
    and .darwin_vm_window.deltas.compressions
      == (.darwin_vm_window.end.compressions - .darwin_vm_window.start.compressions)
    and .darwin_vm_window.deltas.decompressions
      == (.darwin_vm_window.end.decompressions - .darwin_vm_window.start.decompressions)
    and .darwin_vm_window.deltas.purges
      == (.darwin_vm_window.end.purges - .darwin_vm_window.start.purges)
    and .darwin_vm_window.deltas.reactivations
      == (.darwin_vm_window.end.reactivations - .darwin_vm_window.start.reactivations)
    and .darwin_vm_window.deltas.process_pageins
      == (.darwin_vm_window.end.process_pageins
        - .darwin_vm_window.start.process_pageins)
    and all(.darwin_vm_window.deltas.swapins,
      .darwin_vm_window.deltas.swapouts,
      .darwin_vm_window.deltas.process_pageins; . == 0)
    and all(.darwin_vm_window.deltas.pageins,
      .darwin_vm_window.deltas.pageouts,.darwin_vm_window.deltas.compressions,
      .darwin_vm_window.deltas.decompressions,.darwin_vm_window.deltas.purges,
      .darwin_vm_window.deltas.reactivations; . >= 0)
    and .benchmark.position == 6676
    and .benchmark.anchor_exact_state_logits_cache_recurrent == true
    and .benchmark.logical_capacity == 131072
    and .benchmark.loaded_idle_seconds == 45
    and .benchmark.pairs == 20 and .benchmark.order == "alternating"
    and (.benchmark.serial_ms | length) == 20
    and (.benchmark.cohort_ms | length) == 20
    and all(.benchmark.serial_ms[],.benchmark.cohort_ms[];
      type == "number" and isfinite and . > 0)
    and .benchmark.serial_median_ms == $serial_median
    and .benchmark.cohort_median_ms == $cohort_median
    and ((.benchmark.speedup - ($serial_median / $cohort_median)) | abs) < 1e-12
    and .benchmark.unconditioned_order_signature == {
      historical_signature:"even_delta_negative_odd_delta_positive",
      even_delta_median_ms:$unconditioned_even_delta,
      odd_delta_median_ms:$unconditioned_odd_delta,
      observed:($unconditioned_even_delta < 0 and $unconditioned_odd_delta > 0),
      gating:false
    }
    and .benchmark.conditioned.protocol == "same-topology-prime-restore-measure"
    and .benchmark.conditioned.primes_per_measurement == 1
    and (.benchmark.conditioned.serial_prime_ms | length) == 20
    and (.benchmark.conditioned.cohort_prime_ms | length) == 20
    and (.benchmark.conditioned.serial_ms | length) == 20
    and (.benchmark.conditioned.cohort_ms | length) == 20
    and all(
      .benchmark.conditioned.serial_prime_ms[],
      .benchmark.conditioned.cohort_prime_ms[],
      .benchmark.conditioned.serial_ms[],
      .benchmark.conditioned.cohort_ms[];
      type == "number" and isfinite and . > 0)
    and .benchmark.conditioned.serial_median_ms == $conditioned_serial_median
    and .benchmark.conditioned.cohort_median_ms == $conditioned_cohort_median
    and ((.benchmark.conditioned.speedup
      - ($conditioned_serial_median / $conditioned_cohort_median)) | abs) < 1e-12
    and ((.benchmark.conditioned.even_order_speedup
      - ($conditioned_serial_even_median / $conditioned_cohort_even_median)) | abs) < 1e-12
    and ((.benchmark.conditioned.odd_order_speedup
      - ($conditioned_serial_odd_median / $conditioned_cohort_odd_median)) | abs) < 1e-12
    and .benchmark.conditioned.even_order_paired_delta_median_ms
      == $conditioned_even_delta
    and .benchmark.conditioned.odd_order_paired_delta_median_ms
      == $conditioned_odd_delta
    and .benchmark.conditioned.performance_pass == true
    and $conditioned_serial_median > $conditioned_cohort_median
    and .benchmark.conditioned.speedup > 1
    and .benchmark.conditioned.even_order_speedup > 1
    and .benchmark.conditioned.odd_order_speedup > 1
    and $conditioned_even_delta > 0 and $conditioned_odd_delta > 0
    and .benchmark.serial_command_buffers_per_pair == 92
    and .benchmark.cohort_command_buffers_per_pair == 23
    and .benchmark.serial_synchronizations_per_pair == 4
    and .benchmark.cohort_synchronizations_per_pair == 1
    and .benchmark.topology_pass == true
    and .benchmark.topology_errors == []
    and (.benchmark.serial_counters | length) == 20
    and (.benchmark.cohort_counters | length) == 20
    and all(.benchmark.serial_counters[];
      .command_buffers == 92 and .synchronizations == 4
      and .dispatches > 0 and .barriers > 0)
    and all(.benchmark.cohort_counters[];
      .command_buffers == 23 and .synchronizations == 1
      and .dispatches > 0 and .barriers > 0)
    and (.benchmark.conditioned.serial_prime_counters | length) == 20
    and (.benchmark.conditioned.cohort_prime_counters | length) == 20
    and (.benchmark.conditioned.serial_counters | length) == 20
    and (.benchmark.conditioned.cohort_counters | length) == 20
    and all(
      .benchmark.conditioned.serial_prime_counters[],
      .benchmark.conditioned.serial_counters[];
      .command_buffers == 92 and .synchronizations == 4
      and .dispatches > 0 and .barriers > 0)
    and all(
      .benchmark.conditioned.cohort_prime_counters[],
      .benchmark.conditioned.cohort_counters[];
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
    and .host_contention.policy == "process-group-cpu-v2"
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
    and .loaded_setup.thermal.samples >= 2
    and .loaded_setup.thermal.duration_seconds > 0
    and .loaded_setup.thermal.fair_samples >= 0
    and .loaded_setup.thermal.telemetry_gaps == 0
    and .loaded_setup.host_contention.samples >= 2
    and .loaded_setup.host_contention.duration_seconds > 0
    and .loaded_setup.host_contention.contended_samples == 0
    and .loaded_setup.host_contention.telemetry_gaps == 0
    and .phase_evidence.policy == "fsynced-run-bound-markers-v1"
    and .phase_evidence.run_uuid == $receipt.phase_contract.run_uuid
    and .phase_evidence.producer_pid == $receipt.phase_contract.pid
    and .phase_evidence.producer_pid > 0
    and .phase_evidence.test_spawned_at > 0
    and (.phase_evidence.log_sha256 | test("^[0-9a-f]{64}$"))
    and .memory_pressure.policy == "darwin25-phase-bound-process-residency-v3"
    and .memory_pressure.normal_level == 1
    and .memory_pressure.warning_level == 2
    and .memory_pressure.critical_level == 4
    and .memory_pressure.claim_scope == "within-run-paired-only"
    and .memory_pressure.guard_source_path == "scripts/macos_memory_guard.sh"
    and (.memory_pressure.guard_source_sha256 | test("^[0-9a-f]{64}$"))
    and .memory_pressure.sample_interval_seconds == 2
    and .memory_pressure.maximum_sample_gap_seconds == 5
    and .memory_pressure.setup.samples >= 2
    and .memory_pressure.setup.duration_seconds > 0
    and (.memory_pressure.setup.normal_samples
      + .memory_pressure.setup.warning_samples) == .memory_pressure.setup.samples
    and .memory_pressure.setup.min_free_percentage >= 0
    and .memory_pressure.setup.min_free_percentage <= 100
    and (.memory_pressure.setup.max_pressure_level == 1
      or .memory_pressure.setup.max_pressure_level == 2)
    and .memory_pressure.setup.max_throttled_pages == 0
    and .memory_pressure.setup.observed_deltas.swapouts == 0
    and .memory_pressure.loaded_idle.phase == "post-ready-pre-ack"
    and .memory_pressure.loaded_idle.gating == false
    and .memory_pressure.loaded_idle.samples >= 2
    and .memory_pressure.loaded_idle.duration_seconds > 0
    and (.memory_pressure.loaded_idle.normal_samples
      + .memory_pressure.loaded_idle.warning_samples)
      == .memory_pressure.loaded_idle.samples
    and .memory_pressure.loaded_idle.min_free_percentage >= 0
    and .memory_pressure.loaded_idle.min_free_percentage <= 100
    and (.memory_pressure.loaded_idle.max_pressure_level == 1
      or .memory_pressure.loaded_idle.max_pressure_level == 2)
    and .memory_pressure.loaded_idle.max_throttled_pages == 0
    and .memory_pressure.loaded_idle.observed_deltas.swapouts == 0
    and .memory_pressure.measurement.samples >= 2
    and .memory_pressure.measurement.duration_seconds > 0
    and (.memory_pressure.measurement.normal_samples
      + .memory_pressure.measurement.warning_samples)
      == .memory_pressure.measurement.samples
    and .memory_pressure.measurement.min_free_percentage >= 0
    and .memory_pressure.measurement.min_free_percentage <= 100
    and (.memory_pressure.measurement.max_pressure_level == 1
      or .memory_pressure.measurement.max_pressure_level == 2)
    and .memory_pressure.measurement.max_throttled_pages == 0
    and .memory_pressure.setup.boot_time_seconds
      == .memory_pressure.loaded_idle.boot_time_seconds
    and .memory_pressure.loaded_idle.boot_time_seconds
      == .memory_pressure.measurement.boot_time_seconds
    and .memory_pressure.measurement.boot_time_seconds
      == $receipt.darwin_vm_window.start.boot_time_seconds
    and .memory_pressure.setup.page_size == .memory_pressure.measurement.page_size
    and .memory_pressure.loaded_idle.page_size
      == .memory_pressure.measurement.page_size
    and .memory_pressure.measurement.page_size
      == $receipt.darwin_vm_window.start.page_size
    and .memory_pressure.exact_window == $receipt.darwin_vm_window
  ' "$summary" >/dev/null

if [[ "$(sha256_file "$ROOT_DIR/scripts/macos_thermal_probe.swift")" != \
  "$(jq -er .thermal_probe.source_sha256 "$summary")" ]]; then
  echo "decode-cohort receipt thermal-probe source digest mismatch" >&2
  exit 1
fi
if [[ "$(sha256_file "$ROOT_DIR/scripts/macos_memory_guard.sh")" != \
  "$(jq -er .memory_pressure.guard_source_sha256 "$summary")" ]]; then
  echo "decode-cohort receipt memory-guard source digest mismatch" >&2
  exit 1
fi
memory_validate_warning_log "$setup_memory_log" 5 1
setup_memory_actual=$(memory_log_summary_json)
jq -e --argjson actual "$setup_memory_actual" \
  '(.memory_pressure.setup | del(.log_sha256)) == $actual' \
  "$summary" >/dev/null
setup_boot_time=$MEMORY_LOG_BOOT_TIME_SECONDS
setup_page_size=$MEMORY_LOG_PAGE_SIZE
measurement_ready_wall=$(jq -er \
  '.phase_contract.markers[] | select(.sequence == 2)
    | (.wall_ns / 1000000000 | floor)' "$raw")
awk -F '\t' -v ready="$measurement_ready_wall" '$1 > ready' \
  "$setup_memory_log" | cmp -s - "$loaded_idle_memory_log"
memory_validate_warning_log "$loaded_idle_memory_log" 5 1
loaded_idle_memory_actual=$(memory_log_summary_json)
jq -e --argjson actual "$loaded_idle_memory_actual" \
  '(.memory_pressure.loaded_idle
    | del(.log_sha256,.phase,.gating)) == $actual' \
  "$summary" >/dev/null
test "$setup_boot_time" = "$MEMORY_LOG_BOOT_TIME_SECONDS"
test "$setup_page_size" = "$MEMORY_LOG_PAGE_SIZE"
memory_validate_warning_log "$memory_log" 5 0
measurement_memory_actual=$(memory_log_summary_json)
jq -e --argjson actual "$measurement_memory_actual" \
  '(.memory_pressure.measurement | del(.log_sha256)) == $actual' \
  "$summary" >/dev/null
test "$setup_boot_time" = "$MEMORY_LOG_BOOT_TIME_SECONDS"
test "$setup_page_size" = "$MEMORY_LOG_PAGE_SIZE"

# The runner samples immediately before acknowledging marker 2 and immediately
# after observing marker 3. Those sampled system counters must enclose the
# stronger in-process endpoint snapshots; this prevents a receipt from binding
# an unrelated but internally self-consistent VM window.
for field_column in \
  pageins:6 pageouts:7 swapins:8 swapouts:9 compressions:10 \
  decompressions:11 purges:12 reactivations:13; do
  field=${field_column%%:*}
  column=${field_column##*:}
  sampled_start=$(head -1 "$memory_log" | awk -F '\t' -v column="$column" \
    '{ print $column }')
  sampled_end=$(tail -1 "$memory_log" | awk -F '\t' -v column="$column" \
    '{ print $column }')
  exact_start=$(jq -er --arg field "$field" \
    '.darwin_vm_window.start[$field]' "$raw")
  exact_end=$(jq -er --arg field "$field" \
    '.darwin_vm_window.end[$field]' "$raw")
  ((sampled_start <= exact_start && exact_start <= exact_end \
    && exact_end <= sampled_end)) || {
    echo "sampled VM counter does not enclose exact window: $field" >&2
    exit 1
  }
done
test "$(head -1 "$setup_memory_log" | awk -F '\t' '{print $18}')" = \
  decode-cohort-loaded-setup-start
test "$(tail -1 "$setup_memory_log" | awk -F '\t' '{print $18}')" = \
  decode-cohort-loaded-setup-end
awk -F '\t' 'NR > 1 && $18 != "decode-cohort-loaded-setup" &&
  $18 != "decode-cohort-loaded-setup-end" { exit 1 }' "$setup_memory_log"
test "$(head -1 "$loaded_idle_memory_log" | awk -F '\t' '{print $18}')" = \
  decode-cohort-loaded-setup
test "$(tail -1 "$loaded_idle_memory_log" | awk -F '\t' '{print $18}')" = \
  decode-cohort-loaded-setup-end
awk -F '\t' '$18 != "decode-cohort-loaded-setup" &&
  $18 != "decode-cohort-loaded-setup-end" { exit 1 }' \
  "$loaded_idle_memory_log"
test "$(head -1 "$memory_log" | awk -F '\t' '{print $18}')" = \
  decode-cohort-measurement-start
test "$(tail -1 "$memory_log" | awk -F '\t' '{print $18}')" = \
  decode-cohort-measurement-end
awk -F '\t' 'NR > 1 && $18 != "decode-cohort-measurement" &&
  $18 != "decode-cohort-measurement-end" { exit 1 }' "$memory_log"

jq -s -e --slurpfile raw "$raw" \
  '. == $raw[0].phase_contract.markers' "$phase_log" >/dev/null
jq -s -e --slurpfile phases "$phase_log" '. == $phases' \
  <(sed -n 's/^.*HF2Q_DEEPSEEK4_PHASE //p' "$test_log") >/dev/null

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
test_duration_seconds=$(sed -nE \
  's/^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out; finished in ([0-9]+(\.[0-9]+)?)s$/\1/p' \
  "$test_log")
[[ "$test_duration_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]
grep -Fq 'exact_state_logits_cache_recurrent=true' "$test_log"

thermal_validate_fair_or_better_measurement_log "$measurement_log" 5
process_start_ns=$(jq -er 'select(.sequence == 0).monotonic_ns' "$phase_log")
measurement_ready_ns=$(jq -er 'select(.sequence == 2).monotonic_ns' "$phase_log")
measurement_complete_ns=$(jq -er 'select(.sequence == 3).monotonic_ns' "$phase_log")
process_start_wall=$(jq -er \
  'select(.sequence == 0) | (.wall_ns / 1000000000 | floor)' "$phase_log")
measurement_ready_wall=$(jq -er \
  'select(.sequence == 2) | (.wall_ns / 1000000000 | floor)' "$phase_log")
measurement_complete_wall=$(jq -er \
  'select(.sequence == 3) | (.wall_ns / 1000000000 | floor)' "$phase_log")
test_spawned_at=$(jq -er .phase_evidence.test_spawned_at "$summary")
thermal_first=$(head -1 "$measurement_log" | awk -F '\t' '{print $1}')
thermal_last=$(tail -1 "$measurement_log" | awk -F '\t' '{print $1}')
memory_first=$(head -1 "$memory_log" | awk -F '\t' '{print $1}')
memory_last=$(tail -1 "$memory_log" | awk -F '\t' '{print $1}')
setup_thermal_last=$(tail -1 "$setup_thermal_log" | awk -F '\t' '{print $1}')
awk -v test_duration="$test_duration_seconds" \
  -v process_start_ns="$process_start_ns" \
  -v measurement_ready_ns="$measurement_ready_ns" \
  -v measurement_complete_ns="$measurement_complete_ns" \
  -v spawned="$test_spawned_at" -v process_wall="$process_start_wall" \
  -v ready_wall="$measurement_ready_wall" \
  -v complete_wall="$measurement_complete_wall" \
  -v thermal_first="$thermal_first" -v thermal_last="$thermal_last" \
  -v setup_thermal_last="$setup_thermal_last" \
  -v memory_first="$memory_first" -v memory_last="$memory_last" '
  BEGIN {
    phase_span=(measurement_complete_ns-process_start_ns)/1000000000
    if (test_duration <= 0 || phase_span <= 0 || test_duration < phase_span)
      exit 1
    if (measurement_ready_ns <= process_start_ns ||
        measurement_complete_ns <= measurement_ready_ns) exit 1
    if (process_wall < spawned) exit 1
    if (thermal_first < ready_wall || thermal_last < complete_wall) exit 1
    if (setup_thermal_last > thermal_first ||
        thermal_first - setup_thermal_last > 5) exit 1
    if (memory_first < ready_wall || memory_last < complete_wall) exit 1
  }
' || {
  echo "phase-bound telemetry does not cover the named test runtime/window" >&2
  exit 1
}
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

thermal_validate_fair_or_better_measurement_log "$setup_thermal_log" 5
test "$THERMAL_LOG_SAMPLES" = \
  "$(jq -er .loaded_setup.thermal.samples "$summary")"
test "$THERMAL_LOG_DURATION_SECONDS" = \
  "$(jq -er .loaded_setup.thermal.duration_seconds "$summary")"
test "$THERMAL_LOG_FAIR_SAMPLES" = \
  "$(jq -er .loaded_setup.thermal.fair_samples "$summary")"
test "$THERMAL_LOG_GAPS" = \
  "$(jq -er .loaded_setup.thermal.telemetry_gaps "$summary")"
test "$(jq -er .loaded_setup.thermal.required_nominal_tail_seconds \
  "$summary")" = 30
test "$(jq -er .loaded_setup.thermal.nominal_wait_timeout_seconds \
  "$summary")" = 240
test "$(head -1 "$setup_thermal_log" | awk -F '\t' '{print $3}')" = \
  decode-cohort-loaded-setup-start
test "$(head -1 "$setup_thermal_log" | awk -F '\t' '{print $2}')" = nominal
test "$(tail -1 "$setup_thermal_log" | awk -F '\t' '{print $3}')" = \
  decode-cohort-loaded-setup-end
awk -F '\t' 'NR > 1 && $3 != "decode-cohort-loaded-setup" &&
  $3 != "decode-cohort-loaded-setup-end" { exit 1 }' "$setup_thermal_log"
thermal_validate_settle_log "$setup_thermal_log" 30 5
test "$THERMAL_LOG_DURATION_SECONDS" = \
  "$(jq -er .loaded_setup.thermal.nominal_tail_seconds "$summary")"

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
host_contention_validate_measurement_log "$setup_contention_log" 5
test "$HOST_CONTENTION_LOG_SAMPLES" = \
  "$(jq -er .loaded_setup.host_contention.samples "$summary")"
test "$HOST_CONTENTION_LOG_DURATION_SECONDS" = \
  "$(jq -er .loaded_setup.host_contention.duration_seconds "$summary")"
test "$HOST_CONTENTION_LOG_CONTENDED_SAMPLES" = \
  "$(jq -er .loaded_setup.host_contention.contended_samples "$summary")"
test "$HOST_CONTENTION_LOG_GAPS" = \
  "$(jq -er .loaded_setup.host_contention.telemetry_gaps "$summary")"
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
host_contention_validate_thermal_alignment "$setup_thermal_log" \
  "$setup_contention_log"
host_contention_validate_thermal_alignment "$settle_log" \
  "$contention_settle_log"
awk -F '\t' 'NR == 1 && $3 != "decode-cohort-measurement-start" { exit 1 }
  NR > 1 && $3 != "decode-cohort-measurement" && \
    $3 != "decode-cohort-measurement-end" { exit 1 }' \
  "$contention_measurement_log"
awk -F '\t' 'NR == 1 && $3 != "decode-cohort-loaded-setup-start" { exit 1 }
  NR > 1 && $3 != "decode-cohort-loaded-setup" &&
    $3 != "decode-cohort-loaded-setup-end" { exit 1 }' \
  "$setup_contention_log"
awk -F '\t' '$3 != "decode-cohort-settle" { exit 1 }' \
  "$contention_settle_log"

echo "DeepSeek-V4 decode-cohort receipt verified" >&2
