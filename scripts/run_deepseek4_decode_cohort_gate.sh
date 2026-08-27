#!/usr/bin/env bash
set -euo pipefail

test_binary=${1:?prebuilt test binary is required}
model=${2:?DeepSeek model path is required}
out_dir=${3:?output directory is required}
expected_source_sha=${4:?expected source SHA is required}
expected_model_sha=${5:?expected model SHA-256 is required}
dependency_receipt=${6:?verified dependency receipt is required}
expected_dependency_receipt_sha=${7:?verified dependency receipt SHA-256 is required}

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
readonly loaded_nominal_settle_seconds=30
# The producer's acknowledgement timeout is 300 seconds. Stop at 240 seconds
# from first observation, leaving roughly 57 seconds after marker-observation
# lag so cleanup remains fail-closed rather than racing the producer panic.
readonly loaded_nominal_timeout_seconds=240
[[ -x "$HF2Q_THERMAL_SWIFTC_BIN" ]] || {
  echo "required system Swift compiler is unavailable: $HF2Q_THERMAL_SWIFTC_BIN" >&2
  exit 2
}
[[ -x /usr/bin/uuidgen ]] || {
  echo "required system UUID generator is unavailable" >&2
  exit 2
}

[[ -x "$test_binary" ]]
[[ -f "$model" ]]
[[ "$expected_source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$expected_model_sha" =~ ^[0-9a-f]{64}$ ]]
[[ -f "$dependency_receipt" && -r "$dependency_receipt" \
  && ! -L "$dependency_receipt" ]]
[[ "$expected_dependency_receipt_sha" =~ ^[0-9a-f]{64}$ ]]
mkdir -p "$out_dir"
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
[[ "$(sha256_file "$dependency_receipt")" == \
  "$expected_dependency_receipt_sha" ]]
mlx_native_version=$(jq -er '
  select(
    .schema_version == 1 and .status == "pass"
    and .dependency.name == "mlx-native"
    and (.dependency.version
      | test("^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"))
    and .dependency.requirement == ("=" + .dependency.version)
    and .dependency.source
      == "registry+https://github.com/rust-lang/crates.io-index"
    and (.dependency.checksum | test("^[0-9a-f]{64}$"))
  ) | .dependency.version
' "$dependency_receipt")

raw="$out_dir/raw.json"
test_log="$out_dir/test.log"
measurement_log="$out_dir/thermal.log"
settle_log="$out_dir/settle.log"
contention_measurement_log="$out_dir/measurement-contention.log"
contention_settle_log="$out_dir/settle-contention.log"
memory_log="$out_dir/memory-pressure.log"
setup_thermal_log="$out_dir/loaded-setup-thermal.log"
setup_contention_log="$out_dir/loaded-setup-contention.log"
setup_memory_log="$out_dir/loaded-setup-memory.log"
loaded_idle_memory_log="$out_dir/loaded-idle-memory.log"
phase_log="$out_dir/phases.jsonl"
phase_dir="$out_dir/phase-markers"
run_uuid=$(/usr/bin/uuidgen | /usr/bin/tr '[:upper:]' '[:lower:]')
[[ "$run_uuid" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]

rm -f "$raw" "$test_log" "$measurement_log" "$settle_log" \
  "$contention_measurement_log" "$contention_settle_log" "$memory_log" \
  "$setup_thermal_log" "$setup_contention_log" "$setup_memory_log" \
  "$loaded_idle_memory_log" \
  "$phase_log" "${raw}.tmp" "$out_dir/summary.json" \
  "$out_dir/summary.json.tmp" \
  "$out_dir/summary.json.sha256"
if [[ -e "$phase_dir" ]]; then
  if find "$phase_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "phase-marker directory is not empty: $phase_dir" >&2
    exit 2
  fi
else
  mkdir "$phase_dir"
fi

process_start_marker="$phase_dir/000-process-start.json"
loaded_marker="$phase_dir/001-loaded-settle-start.json"
ready_marker="$phase_dir/002-measurement-ready.json"
complete_marker="$phase_dir/003-measurement-complete.json"
ack_file="$phase_dir/measurement-armed.ack"
ack_tmp="$phase_dir/.measurement-armed.ack.tmp"
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

phase_marker_matches() {
  local marker=$1
  local sequence=$2
  local phase=$3
  local pid=$4
  jq -e --arg uuid "$run_uuid" --arg phase "$phase" \
    --argjson sequence "$sequence" --argjson pid "$pid" '
      type == "object" and length == 6 and
      .run_uuid == $uuid and .sequence == $sequence and .phase == $phase and
      .pid == $pid and (.monotonic_ns | type) == "number" and
      .monotonic_ns >= 0 and (.wall_ns | type) == "number" and .wall_ns > 0
    ' "$marker" >/dev/null
}

thermal_prepare_probe
thermal_probe_source_sha=$(sha256_file "$THERMAL_PROBE_SOURCE")
thermal_probe_compiler_sha=$(sha256_file "$THERMAL_PROBE_COMPILER")
thermal_probe_binary_sha=$(sha256_file "$THERMAL_PROBE_BIN")
memory_guard_source_sha=$(sha256_file "$ROOT_DIR/scripts/macos_memory_guard.sh")
thermal_wait_for_nominal "$settle_log" decode-cohort-settle 60 900 5 \
  "$contention_settle_log" "$$"

: >"$setup_thermal_log"
: >"$setup_contention_log"
: >"$setup_memory_log"
: >"$measurement_log"
: >"$contention_measurement_log"
: >"$memory_log"

thermal_sample "$setup_thermal_log" decode-cohort-loaded-setup-start
test "$THERMAL_STATE" = nominal
host_contention_sample "$setup_contention_log" \
  decode-cohort-loaded-setup-start "$$" "$THERMAL_SAMPLED_AT"
host_contention_require_quiet decode-cohort-loaded-setup-start
memory_sample "$setup_memory_log" decode-cohort-loaded-setup-start
setup_initial_swapouts=$MEMORY_SWAPOUTS

test_spawned_at=$(/bin/date +%s)
env -i \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  TMPDIR="${TMPDIR:-/tmp}" \
  HF2Q_DEEPSEEK4_GGUF="$model" \
  HF2Q_DEEPSEEK4_DECODE_COHORT_RECEIPT="$raw" \
  HF2Q_DEEPSEEK4_DECODE_COHORT_PHASE_DIR="$phase_dir" \
  HF2Q_DEEPSEEK4_DECODE_COHORT_RUN_UUID="$run_uuid" \
  "$test_binary" official_artifact_b4_decode_body_is_exact_and_measured \
    --ignored --test-threads=1 --nocapture >"$test_log" 2>&1 &
test_pid=$!

declare -a buffered_thermal=()
declare -a buffered_contention=()
declare -a buffered_memory=()

capture_buffered_measurement_sample() {
  local phase=$1

  thermal_sample /dev/null "$phase" || return 1
  case "$THERMAL_STATE" in
    nominal|fair) ;;
    *)
      echo "decode-cohort measurement exceeded fair thermal state: $THERMAL_STATE" >&2
      return 1
      ;;
  esac
  host_contention_sample /dev/null "$phase" "$$" "$THERMAL_SAMPLED_AT" || return 1
  host_contention_require_quiet "$phase" || return 1
  memory_capture_line "$phase" || return 1
  memory_require_warning_or_better "$phase" || return 1
  buffered_thermal+=("$THERMAL_SAMPLED_AT"$'\t'"$THERMAL_STATE"$'\t'"$phase")
  buffered_contention+=("$THERMAL_SAMPLED_AT"$'\t'"$HOST_CONTENTION_STATE"$'\t'"$phase"$'\t'"$HOST_CONTENTION_OWNER_PGID"$'\t'"$HOST_CONTENTION_OFFENDERS")
  buffered_memory+=("$MEMORY_SAMPLE_LINE")
}

flush_measurement_buffers() {
  local minimum_samples=${1:-2}
  local samples=${#buffered_thermal[@]}

  [[ "$minimum_samples" =~ ^[1-9][0-9]*$ ]] || return 2
  ((samples >= minimum_samples)) || return 1
  ((samples == ${#buffered_contention[@]} \
    && samples == ${#buffered_memory[@]})) || return 1
  printf '%s\n' "${buffered_thermal[@]}" >"$measurement_log" || return 1
  printf '%s\n' "${buffered_contention[@]}" \
    >"$contention_measurement_log" || return 1
  printf '%s\n' "${buffered_memory[@]}" >"$memory_log" || return 1
}

monitor_decode_run() {
  local producer_pid=$1
  local producer_state
  local measurement_active=0
  local ready_validated=0
  local loaded_nominal_since=-1
  local loaded_ready_deadline=-1

  while :; do
    thermal_read_process_state "$producer_pid" || return 1
    producer_state=$THERMAL_PROCESS_STATE
    if ((measurement_active == 1)) && [[ -e "$complete_marker" ]]; then
      capture_buffered_measurement_sample \
        decode-cohort-measurement-end || return 1
      break
    fi
    if [[ -z "$producer_state" || "$producer_state" == Z* ]]; then
      if [[ ! -e "$complete_marker" ]]; then
        echo "decode-cohort producer exited before measurement-complete" >&2
        return 1
      fi
      break
    fi

    if ((measurement_active == 0)); then
      if [[ -e "$ready_marker" ]]; then
        if ((ready_validated == 0)); then
          phase_marker_matches "$process_start_marker" 0 process-start \
            "$producer_pid" || return 1
          phase_marker_matches "$loaded_marker" 1 loaded-settle-start \
            "$producer_pid" || return 1
          phase_marker_matches "$ready_marker" 2 measurement-ready \
            "$producer_pid" || return 1
          ready_validated=1
          loaded_ready_deadline=$((SECONDS + loaded_nominal_timeout_seconds))
        fi

        thermal_sample "$setup_thermal_log" \
          decode-cohort-loaded-setup || return 1
        case "$THERMAL_STATE" in nominal|fair) ;; *) return 1 ;; esac
        host_contention_sample "$setup_contention_log" \
          decode-cohort-loaded-setup "$$" \
          "$THERMAL_SAMPLED_AT" || return 1
        host_contention_require_quiet \
          decode-cohort-loaded-setup || return 1
        memory_sample "$setup_memory_log" decode-cohort-loaded-setup \
          "$setup_initial_swapouts" || return 1

        if [[ "$THERMAL_STATE" == nominal ]]; then
          if ((loaded_nominal_since < 0)); then
            loaded_nominal_since=$THERMAL_SAMPLED_AT
          fi
        else
          loaded_nominal_since=-1
        fi

        if ((loaded_nominal_since >= 0 \
          && THERMAL_SAMPLED_AT - loaded_nominal_since \
            >= loaded_nominal_settle_seconds)); then
          thermal_sample "$setup_thermal_log" \
            decode-cohort-loaded-setup-end || return 1
          test "$THERMAL_STATE" = nominal || return 1
          host_contention_sample "$setup_contention_log" \
            decode-cohort-loaded-setup-end "$$" \
            "$THERMAL_SAMPLED_AT" || return 1
          host_contention_require_quiet \
            decode-cohort-loaded-setup-end || return 1
          memory_sample "$setup_memory_log" decode-cohort-loaded-setup-end \
            "$setup_initial_swapouts" || return 1

          capture_buffered_measurement_sample \
            decode-cohort-measurement-start || return 1
          test "$THERMAL_STATE" = nominal || return 1
          printf '%s\n' "$run_uuid" >"$ack_tmp" || return 1
          mv "$ack_tmp" "$ack_file" || return 1
          measurement_active=1
          /bin/sleep 2 || return 1
          continue
        fi

        if ((SECONDS >= loaded_ready_deadline)); then
          echo "loaded nominal cooldown did not remain calibrated for ${loaded_nominal_settle_seconds}s within ${loaded_nominal_timeout_seconds}s" >&2
          return 1
        fi
        /bin/sleep 2 || return 1
        continue

      fi

      thermal_sample "$setup_thermal_log" decode-cohort-loaded-setup || return 1
      case "$THERMAL_STATE" in nominal|fair) ;; *) return 1 ;; esac
      host_contention_sample "$setup_contention_log" \
        decode-cohort-loaded-setup "$$" "$THERMAL_SAMPLED_AT" || return 1
      host_contention_require_quiet decode-cohort-loaded-setup || return 1
      memory_sample "$setup_memory_log" decode-cohort-loaded-setup \
        "$setup_initial_swapouts" || return 1
    else
      capture_buffered_measurement_sample decode-cohort-measurement || return 1
    fi
    /bin/sleep 2
  done
  ((measurement_active == 1)) || return 1
  phase_marker_matches "$complete_marker" 3 measurement-complete \
    "$producer_pid" || return 1
  flush_measurement_buffers || return 1
}

set +e
monitor_rc=0
monitor_decode_run "$test_pid"
monitor_rc=$?
if ((monitor_rc != 0)); then
  # Measurement telemetry is normally flushed only after marker 3 so no
  # receipt I/O enters the exact Rust window. Once the run has already failed,
  # preserve any complete buffered sample tuples for diagnosis before cleanup.
  flush_measurement_buffers 1 || true
  kill -TERM "$test_pid" 2>/dev/null || true
fi
wait "$test_pid"
test_rc=$?
producer_pid=$test_pid
test_pid=""
set -e
test "$monitor_rc" = 0
test -s "$raw"

for marker in "$process_start_marker" "$loaded_marker" "$ready_marker" \
  "$complete_marker"; do
  jq -cS . "$marker" >>"$phase_log"
done
test "$(wc -l <"$phase_log" | tr -d '[:space:]')" = 4

# Persist the already-loaded, post-ready interval as the adjacent idle control.
# Samples in the marker's wall-clock second are excluded because second-level
# wrapper timestamps cannot prove that they occurred after the nanosecond marker.
measurement_ready_wall_seconds=$(jq -er \
  '.wall_ns / 1000000000 | floor' "$ready_marker")
awk -F '\t' -v ready="$measurement_ready_wall_seconds" '$1 > ready' \
  "$setup_memory_log" >"$loaded_idle_memory_log"
test "$(wc -l <"$loaded_idle_memory_log" | tr -d '[:space:]')" -ge 2

thermal_validate_fair_or_better_measurement_log "$setup_thermal_log" 5
setup_thermal_samples=$THERMAL_LOG_SAMPLES
setup_thermal_duration=$THERMAL_LOG_DURATION_SECONDS
setup_thermal_fair=$THERMAL_LOG_FAIR_SAMPLES
setup_thermal_gaps=$THERMAL_LOG_GAPS
thermal_validate_settle_log "$setup_thermal_log" \
  "$loaded_nominal_settle_seconds" 5
setup_nominal_tail_seconds=$THERMAL_LOG_DURATION_SECONDS
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

host_contention_validate_measurement_log "$setup_contention_log" 5
setup_contention_samples=$HOST_CONTENTION_LOG_SAMPLES
setup_contention_duration=$HOST_CONTENTION_LOG_DURATION_SECONDS
setup_contention_gaps=$HOST_CONTENTION_LOG_GAPS
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
host_contention_validate_thermal_alignment "$setup_thermal_log" "$setup_contention_log"
host_contention_validate_thermal_alignment "$measurement_log" \
  "$contention_measurement_log"
host_contention_validate_thermal_alignment "$settle_log" "$contention_settle_log"

memory_validate_warning_log "$setup_memory_log" 5 1
setup_memory_summary=$(memory_log_summary_json)
memory_validate_warning_log "$loaded_idle_memory_log" 5 1
loaded_idle_memory_summary=$(memory_log_summary_json)
memory_validate_warning_log "$memory_log" 5 0
measurement_memory_summary=$(memory_log_summary_json)

jq --arg source_sha "$expected_source_sha" \
  --arg model_sha256 "$expected_model_sha" \
  --arg raw_sha256 "$(sha256_file "$raw")" \
  --arg test_log_sha256 "$(sha256_file "$test_log")" \
  --arg phase_log_sha256 "$(sha256_file "$phase_log")" \
  --arg run_uuid "$run_uuid" \
  --arg measurement_log_sha256 "$(sha256_file "$measurement_log")" \
  --arg settle_log_sha256 "$(sha256_file "$settle_log")" \
  --arg setup_thermal_log_sha256 "$(sha256_file "$setup_thermal_log")" \
  --arg contention_policy "$HOST_CONTENTION_POLICY" \
  --arg contention_measurement_log_sha256 \
    "$(sha256_file "$contention_measurement_log")" \
  --arg contention_settle_log_sha256 \
    "$(sha256_file "$contention_settle_log")" \
  --arg setup_contention_log_sha256 "$(sha256_file "$setup_contention_log")" \
  --arg memory_policy "$MEMORY_PRESSURE_POLICY" \
  --arg memory_log_sha256 "$(sha256_file "$memory_log")" \
  --arg setup_memory_log_sha256 "$(sha256_file "$setup_memory_log")" \
  --arg loaded_idle_memory_log_sha256 \
    "$(sha256_file "$loaded_idle_memory_log")" \
  --arg memory_guard_source_sha256 "$memory_guard_source_sha" \
  --arg thermal_probe_source_sha256 "$thermal_probe_source_sha" \
  --arg thermal_probe_compiler_path "$THERMAL_PROBE_COMPILER" \
  --arg thermal_probe_compiler_sha256 "$thermal_probe_compiler_sha" \
  --arg thermal_probe_compiler_version "$THERMAL_PROBE_COMPILER_VERSION" \
  --arg thermal_probe_binary_sha256 "$thermal_probe_binary_sha" \
  --argjson producer_exit_code "$test_rc" \
  --argjson producer_pid "$producer_pid" \
  --argjson test_spawned_at "$test_spawned_at" \
  --argjson settle_samples "$settle_samples" \
  --argjson settle_duration_seconds "$settle_duration_seconds" \
  --argjson settle_telemetry_gaps "$settle_gaps" \
  --argjson setup_thermal_samples "$setup_thermal_samples" \
  --argjson setup_thermal_duration "$setup_thermal_duration" \
  --argjson setup_thermal_fair "$setup_thermal_fair" \
  --argjson setup_thermal_gaps "$setup_thermal_gaps" \
  --argjson setup_nominal_required_seconds \
    "$loaded_nominal_settle_seconds" \
  --argjson setup_nominal_tail_seconds "$setup_nominal_tail_seconds" \
  --argjson setup_nominal_timeout_seconds \
    "$loaded_nominal_timeout_seconds" \
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
  --argjson setup_contention_samples "$setup_contention_samples" \
  --argjson setup_contention_duration "$setup_contention_duration" \
  --argjson setup_contention_gaps "$setup_contention_gaps" \
  --argjson contention_measurement_samples "$contention_measurement_samples" \
  --argjson contention_measurement_duration_seconds \
    "$contention_measurement_duration_seconds" \
  --argjson contention_measurement_contended_samples \
    "$contention_measurement_contended_samples" \
  --argjson contention_measurement_gaps "$contention_measurement_gaps" \
  --arg mlx_native_version "$mlx_native_version" \
  --argjson setup_memory "$setup_memory_summary" \
  --argjson loaded_idle_memory "$loaded_idle_memory_summary" \
  --argjson measurement_memory "$measurement_memory_summary" '
  . + {schema_version:6,source_sha:$source_sha,model_sha256:$model_sha256,
    mlx_native_version:$mlx_native_version,producer_exit_code:$producer_exit_code,
    raw_sha256:$raw_sha256,test_log_sha256:$test_log_sha256,
    phase_evidence:{policy:"fsynced-run-bound-markers-v1",run_uuid:$run_uuid,
      producer_pid:$producer_pid,test_spawned_at:$test_spawned_at,
      log_sha256:$phase_log_sha256},
    thermal_status:"fair_or_better",required_start_state:"nominal",
    maximum_measurement_state:"fair",measurement_log_sha256:$measurement_log_sha256,
    settle_log_sha256:$settle_log_sha256,settle_seconds:60,
    thermal_probe:{implementation:"compiled-foundation-helper",
      source_path:"scripts/macos_thermal_probe.swift",
      source_sha256:$thermal_probe_source_sha256,
      compiler_path:$thermal_probe_compiler_path,
      compiler_sha256:$thermal_probe_compiler_sha256,
      compiler_version:$thermal_probe_compiler_version,
      binary_sha256:$thermal_probe_binary_sha256},
    settle_samples:$settle_samples,settle_duration_seconds:$settle_duration_seconds,
    settle_sample_interval_seconds:5,maximum_settle_sample_gap_seconds:8,
    settle_telemetry_gaps:$settle_telemetry_gaps,
    loaded_setup:{thermal:{log_sha256:$setup_thermal_log_sha256,
        samples:$setup_thermal_samples,duration_seconds:$setup_thermal_duration,
        fair_samples:$setup_thermal_fair,telemetry_gaps:$setup_thermal_gaps,
        required_nominal_tail_seconds:$setup_nominal_required_seconds,
        nominal_tail_seconds:$setup_nominal_tail_seconds,
        nominal_wait_timeout_seconds:$setup_nominal_timeout_seconds},
      host_contention:{log_sha256:$setup_contention_log_sha256,
        samples:$setup_contention_samples,duration_seconds:$setup_contention_duration,
        contended_samples:0,telemetry_gaps:$setup_contention_gaps}},
    measurement_samples:$measurement_samples,
    measurement_duration_seconds:$measurement_duration_seconds,
    sample_interval_seconds:2,maximum_sample_gap_seconds:5,
    non_nominal_measurement_samples:$non_nominal_measurement_samples,
    fair_measurement_samples:$fair_measurement_samples,
    over_limit_measurement_samples:$over_limit_measurement_samples,
    telemetry_gaps:$telemetry_gaps,
    host_contention:{policy:$contention_policy,
      settle:{log_sha256:$contention_settle_log_sha256,samples:$contention_settle_samples,
        duration_seconds:$contention_settle_duration_seconds,
        contended_samples:$contention_settle_contended_samples,
        telemetry_gaps:$contention_settle_gaps},
      measurement:{log_sha256:$contention_measurement_log_sha256,
        samples:$contention_measurement_samples,
        duration_seconds:$contention_measurement_duration_seconds,
        contended_samples:$contention_measurement_contended_samples,
        telemetry_gaps:$contention_measurement_gaps}},
    memory_pressure:{policy:$memory_policy,normal_level:1,warning_level:2,
      critical_level:4,claim_scope:"within-run-paired-only",
      guard_source_path:"scripts/macos_memory_guard.sh",
      guard_source_sha256:$memory_guard_source_sha256,
      sample_interval_seconds:2,maximum_sample_gap_seconds:5,
      setup:($setup_memory + {log_sha256:$setup_memory_log_sha256}),
      loaded_idle:($loaded_idle_memory + {
        log_sha256:$loaded_idle_memory_log_sha256,
        phase:"post-ready-pre-ack",gating:false}),
      measurement:($measurement_memory + {log_sha256:$memory_log_sha256}),
      exact_window:.darwin_vm_window}}
' "$raw" >"$out_dir/summary.json.tmp"
mv "$out_dir/summary.json.tmp" "$out_dir/summary.json"

bash "$ROOT_DIR/scripts/verify_deepseek4_decode_cohort_receipt.sh" \
  "$out_dir/summary.json" "$raw" "$test_log" "$measurement_log" \
  "$settle_log" "$expected_source_sha" "$expected_model_sha" \
  "$contention_measurement_log" "$contention_settle_log" "$memory_log" \
  "$phase_log" "$setup_thermal_log" "$setup_contention_log" \
  "$setup_memory_log" "$loaded_idle_memory_log" "$dependency_receipt" \
  "$expected_dependency_receipt_sha"
test "$test_rc" = 0
sha256_file "$out_dir/summary.json" >"$out_dir/summary.json.sha256"
