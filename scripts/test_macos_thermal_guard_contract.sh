#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=scripts/macos_thermal_guard.sh
source "$ROOT_DIR/scripts/macos_thermal_guard.sh"

tmp_dir=$(mktemp -d -t hf2q-thermal-guard.XXXXXX)
cleanup() {
  thermal_cleanup_probe || true
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

probe_fixture_dir="$tmp_dir/probe-fixture"
mkdir -p "$probe_fixture_dir"
fake_swiftc="$probe_fixture_dir/swiftc"
compile_count="$probe_fixture_dir/compile-count"
# The single-quoted fragments are the literal body of the generated fake
# compiler; expansion must happen only when that fixture executes.
# shellcheck disable=SC2016
printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' \
  'if [[ "${1:-}" == --version ]]; then printf "fake swiftc 1.0\\n"; exit 0; fi' \
  'output=""' \
  'while (($#)); do' \
  '  if [[ "$1" == -o ]]; then output=$2; shift 2; else shift; fi' \
  'done' \
  '[[ -n "$output" ]]' \
  'printf "compile\\n" >>"$HF2Q_FAKE_COMPILE_COUNT"' \
  'printf "%s\\n" "#!/usr/bin/env bash" "printf '\''nominal\\n'\''" >"$output"' \
  'chmod +x "$output"' >"$fake_swiftc"
chmod +x "$fake_swiftc"

# The helper is compiled once, reused for every sample, and removed only from
# its exact private directory. This is the regression contract for the prior
# per-sample `swift -e` compiler launch.
HF2Q_THERMAL_SWIFTC_BIN=$fake_swiftc
HF2Q_FAKE_COMPILE_COUNT=$compile_count
export HF2Q_FAKE_COMPILE_COUNT
thermal_prepare_probe
prepared_probe=$THERMAL_PROBE_BIN
prepared_probe_dir=$THERMAL_PROBE_OWNED_DIR
test -x "$prepared_probe"
thermal_read_state
test "$THERMAL_STATE" = nominal
thermal_read_state
test "$THERMAL_STATE" = nominal
test "$(wc -l <"$compile_count" | tr -d '[:space:]')" = 1
test "$THERMAL_PROBE_BIN" = "$prepared_probe"
thermal_cleanup_probe
test ! -e "$prepared_probe"
test ! -e "$prepared_probe_dir"
unset HF2Q_THERMAL_SWIFTC_BIN HF2Q_FAKE_COMPILE_COUNT

# Every preparation boundary fails closed and removes its private directory.
if HF2Q_THERMAL_SWIFTC_BIN="$fake_swiftc" \
  HF2Q_THERMAL_PROBE_SOURCE="$probe_fixture_dir/missing.swift" \
  thermal_prepare_probe; then
  echo "thermal guard accepted a missing probe source" >&2
  exit 1
fi
test -z "$THERMAL_PROBE_BIN"

failing_swiftc="$probe_fixture_dir/failing-swiftc"
# shellcheck disable=SC2016
printf '%s\n' '#!/usr/bin/env bash' \
  'if [[ "${1:-}" == --version ]]; then printf "fake swiftc 1.0\\n"; exit 0; fi' \
  'echo "synthetic compiler failure" >&2' 'exit 1' >"$failing_swiftc"
chmod +x "$failing_swiftc"
if TMPDIR="$probe_fixture_dir" \
  HF2Q_THERMAL_SWIFTC_BIN="$failing_swiftc" \
  thermal_prepare_probe >/dev/null 2>&1; then
  echo "thermal guard accepted a compiler failure" >&2
  exit 1
fi
test -z "$THERMAL_PROBE_BIN"
test -z "$(find "$probe_fixture_dir" -maxdepth 1 -type d \
  -name 'hf2q-thermal-probe.*' -print -quit)"

malformed_probe="$probe_fixture_dir/malformed-probe"
printf '%s\n' '#!/usr/bin/env bash' 'printf "unknown\\n"' >"$malformed_probe"
chmod +x "$malformed_probe"
if HF2Q_THERMAL_PROBE_BIN="$malformed_probe" thermal_prepare_probe; then
  echo "thermal guard accepted malformed probe output" >&2
  exit 1
fi
test -z "$THERMAL_PROBE_BIN"

failing_probe="$probe_fixture_dir/failing-probe"
printf '%s\n' '#!/usr/bin/env bash' 'exit 1' >"$failing_probe"
chmod +x "$failing_probe"
if HF2Q_THERMAL_PROBE_BIN="$failing_probe" thermal_prepare_probe; then
  echo "thermal guard accepted probe execution failure" >&2
  exit 1
fi
test -z "$THERMAL_PROBE_BIN"

external_probe="$probe_fixture_dir/external-probe"
printf '%s\n' '#!/usr/bin/env bash' 'printf "nominal\\n"' >"$external_probe"
chmod +x "$external_probe"
HF2Q_THERMAL_PROBE_BIN="$external_probe" thermal_prepare_probe
test -z "$THERMAL_PROBE_OWNED_DIR"
thermal_cleanup_probe
test -x "$external_probe"

mismatched_owned_dir="$probe_fixture_dir/mismatched-owned"
mkdir -p "$mismatched_owned_dir"
THERMAL_PROBE_BIN="$probe_fixture_dir/not-the-owned-probe"
THERMAL_PROBE_OWNED_DIR=$mismatched_owned_dir
if thermal_cleanup_probe; then
  echo "thermal cleanup accepted a mismatched owned path" >&2
  exit 1
fi
test -d "$mismatched_owned_dir"
THERMAL_PROBE_BIN=""
THERMAL_PROBE_OWNED_DIR=""

if [[ $(uname -s) == Darwin ]]; then
  thermal_read_state
  [[ "$THERMAL_STATE" =~ ^(nominal|fair|serious|critical)$ ]]
  thermal_cleanup_probe
fi

sequence_index=0
sequence=()
read_sequence_state() {
  local last_index=$(( ${#sequence[@]} - 1 ))
  local index=$sequence_index
  ((index <= last_index)) || index=$last_index
  THERMAL_STATE=${sequence[$index]}
  sequence_index=$((sequence_index + 1))
}

# A non-nominal host must wait until the probe reports Nominal.
sequence=(serious fair nominal)
sequence_index=0
thermal_read_state() { read_sequence_state; }
thermal_wait_for_nominal "$tmp_dir/wait.log" wait-test 0 5 0
test "$(tail -1 "$tmp_dir/wait.log" | awk -F '\t' '{print $2}')" = nominal
test "$(wc -l <"$tmp_dir/wait.log" | tr -d '[:space:]')" = 3

# A non-Nominal transition resets the settle window. Advance Bash's monotonic
# timer without sleeping so the contract proves a trailing continuous window.
sequence=(nominal serious nominal nominal nominal)
sequence_index=0
thermal_read_state() {
  read_sequence_state
  SECONDS=$((SECONDS + 1))
}
thermal_wait_for_nominal "$tmp_dir/reset.log" reset-test 2 10 0
test "$(wc -l <"$tmp_dir/reset.log" | tr -d '[:space:]')" = 5
test "$(tail -1 "$tmp_dir/reset.log" | awk -F '\t' '{print $2}')" = nominal

# A host that never becomes Nominal must fail at the timeout.
sequence=(serious)
sequence_index=0
if thermal_wait_for_nominal "$tmp_dir/timeout.log" timeout-test 0 0 0; then
  echo "thermal guard accepted a serious host at timeout" >&2
  exit 1
fi

# Any in-wave transition away from Nominal invalidates the measurement.
sequence=(nominal serious)
sequence_index=0
if thermal_monitor_nominal "$tmp_dir/monitor.log" monitor-test \
  "$tmp_dir/never-stop" 0; then
  echo "thermal monitor accepted a mid-wave serious state" >&2
  exit 1
fi
test "$(tail -1 "$tmp_dir/monitor.log" | awk -F '\t' '{print $2}')" = serious

# A normal stop file exits cleanly, while a probe failure exits nonzero.
normal_stop="$tmp_dir/normal-stop"
thermal_read_state() {
  THERMAL_STATE=nominal
  : >"$normal_stop"
}
thermal_monitor_nominal "$tmp_dir/normal-monitor.log" normal-monitor \
  "$normal_stop" 0
test "$(wc -l <"$tmp_dir/normal-monitor.log" | tr -d '[:space:]')" = 1

thermal_read_state() { return 1; }
if thermal_monitor_nominal "$tmp_dir/probe-failure.log" probe-failure \
  "$tmp_dir/never-stop-probe" 0; then
  echo "thermal monitor accepted probe failure" >&2
  exit 1
fi

# Foreground process supervision exits cleanly after its producer and fails
# immediately on a non-Nominal transition without a background stop/join
# handoff.
thermal_read_state() { THERMAL_STATE=nominal; }
(sleep 0.1) &
supervised_pid=$!
thermal_monitor_nominal_while_pid "$tmp_dir/supervised.log" \
  supervised "$supervised_pid" 0
wait "$supervised_pid"
test -s "$tmp_dir/supervised.log"

(sleep 5) &
supervised_pid=$!
if (
  thermal_read_process_state() { return 1; }
  thermal_monitor_nominal_while_pid "$tmp_dir/supervised-ps-failure.log" \
    supervised-ps-failure "$supervised_pid" 0
); then
  echo "foreground thermal supervision accepted process-state probe failure" >&2
  kill -TERM "$supervised_pid" 2>/dev/null || true
  wait "$supervised_pid" 2>/dev/null || true
  exit 1
fi
kill -TERM "$supervised_pid" 2>/dev/null || true
wait "$supervised_pid" 2>/dev/null || true
test ! -s "$tmp_dir/supervised-ps-failure.log"

(sleep 5) &
supervised_pid=$!
thermal_read_process_state "$supervised_pid"
test -n "$THERMAL_PROCESS_STATE"
kill -TERM "$supervised_pid" 2>/dev/null || true
wait "$supervised_pid" 2>/dev/null || true
thermal_read_process_state "$supervised_pid"
test -z "$THERMAL_PROCESS_STATE"

sequence=(nominal fair)
sequence_index=0
thermal_read_state() { read_sequence_state; }
(sleep 5) &
supervised_pid=$!
if thermal_monitor_nominal_while_pid "$tmp_dir/supervised-fair.log" \
  supervised-fair "$supervised_pid" 0; then
  echo "foreground thermal supervision accepted a fair state" >&2
  kill -TERM "$supervised_pid" 2>/dev/null || true
  wait "$supervised_pid" 2>/dev/null || true
  exit 1
fi
kill -TERM "$supervised_pid" 2>/dev/null || true
wait "$supervised_pid" 2>/dev/null || true
test "$(tail -1 "$tmp_dir/supervised-fair.log" | awk -F '\t' '{print $2}')" = fair

# Long calibrated workloads may reach Fair after a Nominal start. The bounded
# monitor accepts Fair but still fails closed on Serious/Critical.
sequence=(nominal fair fair)
sequence_index=0
thermal_read_state() { read_sequence_state; }
(sleep 0.1) &
supervised_pid=$!
thermal_monitor_fair_or_better_while_pid "$tmp_dir/supervised-bounded.log" \
  supervised-bounded "$supervised_pid" 0
wait "$supervised_pid"
test -s "$tmp_dir/supervised-bounded.log"

sequence=(fair serious)
sequence_index=0
thermal_read_state() { read_sequence_state; }
(sleep 5) &
supervised_pid=$!
if thermal_monitor_fair_or_better_while_pid \
  "$tmp_dir/supervised-bounded-serious.log" supervised-bounded-serious \
  "$supervised_pid" 0; then
  echo "bounded thermal monitor accepted a serious state" >&2
  kill -TERM "$supervised_pid" 2>/dev/null || true
  wait "$supervised_pid" 2>/dev/null || true
  exit 1
fi
kill -TERM "$supervised_pid" 2>/dev/null || true
wait "$supervised_pid" 2>/dev/null || true
test "$(tail -1 "$tmp_dir/supervised-bounded-serious.log" | awk -F '\t' '{print $2}')" = serious

# Cold-cohort measurement ends only after the exact number of non-empty,
# atomically published cold receipts exists. Later functional phases are not
# part of the calibrated thermal envelope.
cold_prepared="$tmp_dir/cold-prepared"
thermal_prepare_cold_receipt_dir "$cold_prepared"
test -d "$cold_prepared"
printf '{}\n' >"$cold_prepared/agent-1.cold.json"
if thermal_prepare_cold_receipt_dir "$cold_prepared"; then
  echo "cold thermal setup accepted stale receipt evidence" >&2
  exit 1
fi

cold_receipts="$tmp_dir/cold-receipts"
mkdir -p "$cold_receipts"
printf '{}\n' >"$cold_receipts/agent-1.cold.json"
printf '{}\n' >"$cold_receipts/agent-2.cold.json"
thermal_read_state() { THERMAL_STATE=nominal; }
thermal_monitor_nominal_until_cold_receipts \
  "$tmp_dir/cold-complete.log" cold-complete "$cold_receipts" 2 0 1
test "$(tail -1 "$tmp_dir/cold-complete.log" | awk -F '\t' '{print $3}')" = \
  cold-complete-end

cold_partial="$tmp_dir/cold-partial"
mkdir -p "$cold_partial"
printf '{}\n' >"$cold_partial/agent-1.cold.json"
if thermal_monitor_nominal_until_cold_receipts \
  "$tmp_dir/cold-partial.log" cold-partial "$cold_partial" 2 0 0; then
  echo "cold thermal monitor accepted an incomplete receipt cohort" >&2
  exit 1
fi

cold_excess="$tmp_dir/cold-excess"
mkdir -p "$cold_excess"
for agent in 1 2 3; do
  printf '{}\n' >"$cold_excess/agent-${agent}.cold.json"
done
if thermal_monitor_nominal_until_cold_receipts \
  "$tmp_dir/cold-excess.log" cold-excess "$cold_excess" 2 0 1; then
  echo "cold thermal monitor accepted excess receipts" >&2
  exit 1
fi

cold_wrong_set="$tmp_dir/cold-wrong-set"
mkdir -p "$cold_wrong_set"
printf '{}\n' >"$cold_wrong_set/agent-1.cold.json"
printf '{}\n' >"$cold_wrong_set/agent-3.cold.json"
if thermal_monitor_nominal_until_cold_receipts \
  "$tmp_dir/cold-wrong-set.log" cold-wrong-set "$cold_wrong_set" 2 0 1; then
  echo "cold thermal monitor accepted the wrong exact receipt set" >&2
  exit 1
fi

thermal_read_state() { THERMAL_STATE=fair; }
if thermal_monitor_nominal_until_cold_receipts \
  "$tmp_dir/cold-fair.log" cold-fair "$cold_receipts" 2 0 1; then
  echo "cold thermal monitor accepted non-Nominal terminal telemetry" >&2
  exit 1
fi

thermal_read_state() { THERMAL_STATE=nominal; }
(exit 0) &
dead_producer=$!
wait "$dead_producer"
if thermal_monitor_nominal_until_cold_receipts \
  "$tmp_dir/cold-dead.log" cold-dead "$cold_partial" 2 0 1 \
  "$dead_producer"; then
  echo "cold thermal monitor waited after its receipt producer exited" >&2
  exit 1
fi

# Malformed telemetry and probe failure are fail-closed.
thermal_read_state() { THERMAL_STATE=unknown; return 1; }
if thermal_wait_for_nominal "$tmp_dir/malformed.log" malformed-test 0 1 0; then
  echo "thermal guard accepted malformed telemetry" >&2
  exit 1
fi

# The receipt validators reject missing cadence, backwards time, non-Nominal
# measurement state, and a nominal-looking settle interval with a 60s hole.
printf '100\tnominal\tstart\n102\tnominal\tmiddle\n104\tnominal\tend\n' \
  >"$tmp_dir/valid-measurement.log"
thermal_validate_measurement_log "$tmp_dir/valid-measurement.log" 5
test "$THERMAL_LOG_SAMPLES" = 3
test "$THERMAL_LOG_DURATION_SECONDS" = 4

printf '100\tnominal\tstart\n99\tnominal\tend\n' \
  >"$tmp_dir/backwards.log"
if thermal_validate_measurement_log "$tmp_dir/backwards.log" 5; then
  echo "thermal validator accepted decreasing timestamps" >&2
  exit 1
fi
printf '100\tnominal\tstart\n106\tnominal\tend\n' >"$tmp_dir/gap.log"
if thermal_validate_measurement_log "$tmp_dir/gap.log" 5; then
  echo "thermal validator accepted an in-wave telemetry gap" >&2
  exit 1
fi
printf '100\tnominal\tstart\n102\tfair\tend\n' >"$tmp_dir/non-nominal.log"
if thermal_validate_measurement_log "$tmp_dir/non-nominal.log" 5; then
  echo "thermal validator accepted a non-Nominal measurement" >&2
  exit 1
fi
printf '100\tnominal\tstart\n102\tfair\tmiddle\n104\tfair\tend\n' \
  >"$tmp_dir/valid-bounded-measurement.log"
thermal_validate_fair_or_better_measurement_log \
  "$tmp_dir/valid-bounded-measurement.log" 5
test "$THERMAL_LOG_SAMPLES" = 3
test "$THERMAL_LOG_FAIR_SAMPLES" = 2
test "$THERMAL_LOG_OVER_LIMIT_SAMPLES" = 0
printf '100\tfair\tstart\n102\tserious\tend\n' \
  >"$tmp_dir/invalid-bounded-measurement.log"
if thermal_validate_fair_or_better_measurement_log \
  "$tmp_dir/invalid-bounded-measurement.log" 5; then
  echo "bounded thermal validator accepted a serious measurement" >&2
  exit 1
fi
printf '100\tnominal\tsettle\n160\tnominal\tsettle\n' >"$tmp_dir/settle-gap.log"
if thermal_validate_settle_log "$tmp_dir/settle-gap.log" 60 8; then
  echo "thermal validator accepted a 60-second settle telemetry hole" >&2
  exit 1
fi
printf '100\tnominal\tsettle\n99\tnominal\tsettle\n' \
  >"$tmp_dir/settle-backwards.log"
if thermal_validate_settle_log "$tmp_dir/settle-backwards.log" 0 8; then
  echo "thermal settle validator accepted decreasing timestamps" >&2
  exit 1
fi

# A complete receipt binds path, wave, summary object, hashes, schema, and log
# phase columns. Stale or cross-wave objects must fail even when re-hashed.
receipt_dir="$tmp_dir/receipt"
mkdir -p "$receipt_dir/agents"
for agent in 1 2 3 4; do
  printf '{"agent":%s,"status":"pass"}\n' "$agent" \
    >"$receipt_dir/agents/agent-${agent}.cold.json"
done
for epoch in $(seq 100 5 160); do
  printf '%s\tnominal\tdeepseek-wave-1-settle\n' "$epoch"
done >"$receipt_dir/settle.log"
printf '200\tnominal\tdeepseek-wave-1-measurement-start\n' \
  >"$receipt_dir/measurement.log"
printf '202\tnominal\tdeepseek-wave-1-measurement\n' \
  >>"$receipt_dir/measurement.log"
printf '204\tnominal\tdeepseek-wave-1-measurement-end\n' \
  >>"$receipt_dir/measurement.log"
settle_sha=$(shasum -a 256 "$receipt_dir/settle.log" | awk '{print $1}')
measurement_sha=$(shasum -a 256 "$receipt_dir/measurement.log" | awk '{print $1}')
cold_receipts=$(
  for receipt in "$receipt_dir"/agents/agent-*.cold.json; do
    jq -n --arg name "$(basename "$receipt")" \
      --arg sha256 "$(shasum -a 256 "$receipt" | awk '{print $1}')" \
      '{name:$name,sha256:$sha256}'
  done | jq -s 'sort_by(.name)'
)
jq -n --arg settle_sha "$settle_sha" --arg measurement_sha "$measurement_sha" \
  --argjson cold_receipts "$cold_receipts" '
  {
    status:"pass", phase:"deepseek-wave-1", required_state:"nominal",
    runtime_preflight:"pass", measurement_scope:"cold-cohort",
    cold_receipts:$cold_receipts,
    settle_seconds:60, settle_duration_seconds:60, settle_samples:13,
    measurement_samples:3, measurement_duration_seconds:4,
    sample_interval_seconds:2, maximum_sample_gap_seconds:5,
    settle_sample_interval_seconds:5, maximum_settle_sample_gap_seconds:8,
    non_nominal_measurement_samples:0, settle_telemetry_gaps:0,
    telemetry_gaps:0, settle_log_sha256:$settle_sha,
    measurement_log_sha256:$measurement_sha
  }
' >"$receipt_dir/summary.json"
jq -n --argjson wave 1 --slurpfile thermal "$receipt_dir/summary.json" \
  '{wave:$wave,thermal:$thermal[0]}' >"$receipt_dir/envelope.json"
bash "$ROOT_DIR/scripts/verify_macos_thermal_receipt.sh" 1 \
  "$receipt_dir/envelope.json" "$receipt_dir/summary.json" \
  "$receipt_dir/measurement.log" "$receipt_dir/settle.log" \
  "$receipt_dir/agents" >/dev/null

jq '.wave = 2' "$receipt_dir/envelope.json" >"$receipt_dir/wrong-wave.json"
if bash "$ROOT_DIR/scripts/verify_macos_thermal_receipt.sh" 1 \
  "$receipt_dir/wrong-wave.json" "$receipt_dir/summary.json" \
  "$receipt_dir/measurement.log" "$receipt_dir/settle.log" \
  "$receipt_dir/agents" >/dev/null 2>&1; then
  echo "thermal receipt accepted the wrong wave" >&2
  exit 1
fi
jq 'del(.maximum_sample_gap_seconds)' "$receipt_dir/summary.json" \
  >"$receipt_dir/stale-summary.json"
jq --slurpfile thermal "$receipt_dir/stale-summary.json" \
  '.thermal = $thermal[0]' "$receipt_dir/envelope.json" \
  >"$receipt_dir/stale-envelope.json"
if bash "$ROOT_DIR/scripts/verify_macos_thermal_receipt.sh" 1 \
  "$receipt_dir/stale-envelope.json" "$receipt_dir/stale-summary.json" \
  "$receipt_dir/measurement.log" "$receipt_dir/settle.log" \
  "$receipt_dir/agents" >/dev/null 2>&1; then
  echo "thermal receipt accepted a stale schema" >&2
  exit 1
fi
jq 'del(.measurement_scope)' "$receipt_dir/summary.json" \
  >"$receipt_dir/no-scope-summary.json"
jq --slurpfile thermal "$receipt_dir/no-scope-summary.json" \
  '.thermal = $thermal[0]' "$receipt_dir/envelope.json" \
  >"$receipt_dir/no-scope-envelope.json"
if bash "$ROOT_DIR/scripts/verify_macos_thermal_receipt.sh" 1 \
  "$receipt_dir/no-scope-envelope.json" "$receipt_dir/no-scope-summary.json" \
  "$receipt_dir/measurement.log" "$receipt_dir/settle.log" \
  "$receipt_dir/agents" >/dev/null 2>&1; then
  echo "thermal receipt accepted a missing cold-cohort scope" >&2
  exit 1
fi
printf '{"agent":1,"status":"mutated"}\n' \
  >"$receipt_dir/agents/agent-1.cold.json"
if bash "$ROOT_DIR/scripts/verify_macos_thermal_receipt.sh" 1 \
  "$receipt_dir/envelope.json" "$receipt_dir/summary.json" \
  "$receipt_dir/measurement.log" "$receipt_dir/settle.log" \
  "$receipt_dir/agents" >/dev/null 2>&1; then
  echo "thermal receipt accepted mutated cold evidence" >&2
  exit 1
fi
# Restore the valid cold receipt for the remaining phase-column mutation.
printf '{"agent":1,"status":"pass"}\n' \
  >"$receipt_dir/agents/agent-1.cold.json"
awk -F '\t' 'BEGIN { OFS = "\t" } NR == 2 { $3 = "deepseek-wave-2-measurement" } { print }' \
  "$receipt_dir/measurement.log" >"$receipt_dir/wrong-phase.log"
wrong_phase_sha=$(shasum -a 256 "$receipt_dir/wrong-phase.log" | awk '{print $1}')
jq --arg sha "$wrong_phase_sha" '.measurement_log_sha256 = $sha' \
  "$receipt_dir/summary.json" >"$receipt_dir/wrong-phase-summary.json"
jq --slurpfile thermal "$receipt_dir/wrong-phase-summary.json" \
  '.thermal = $thermal[0]' "$receipt_dir/envelope.json" \
  >"$receipt_dir/wrong-phase-envelope.json"
if bash "$ROOT_DIR/scripts/verify_macos_thermal_receipt.sh" 1 \
  "$receipt_dir/wrong-phase-envelope.json" "$receipt_dir/wrong-phase-summary.json" \
  "$receipt_dir/wrong-phase.log" "$receipt_dir/settle.log" \
  "$receipt_dir/agents" >/dev/null 2>&1; then
  echo "thermal receipt accepted a cross-wave log phase" >&2
  exit 1
fi

echo "macOS thermal guard contract: pass"
