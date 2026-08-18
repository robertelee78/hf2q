#!/usr/bin/env bash
# Model-free negative tests for the exact Qwen watchdog receipt parser.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

for command in jq awk cmp diff sed wc tr mktemp seq shasum stat find grep date sleep cat; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done

test_dir=$(mktemp -d -t hf2q-qwen36-validator.XXXXXX)
trap 'rm -rf "$test_dir"' EXIT

expect_fail() {
  if "$@" >/dev/null 2>&1; then
    echo "expected validator failure: $*" >&2
    exit 1
  fi
}

expect_fail qwen36_bind_server_process \
  'https://127.0.0.1:18081' 1 /bin/false /dev/null 4
expect_fail qwen36_bind_server_process \
  'http://127.0.0.1:18081' not-a-pid /bin/false /dev/null 4
mkdir "$test_dir/empty-receipt"
qwen36_require_empty_receipt_dir "$test_dir/empty-receipt"
printf 'stale\n' >"$test_dir/empty-receipt/stale-summary.json"
expect_fail qwen36_require_empty_receipt_dir "$test_dir/empty-receipt"
qwen36_validate_power_event_counts 12 12
expect_fail qwen36_validate_power_event_counts 12 13
expect_fail qwen36_validate_power_event_counts not-a-number 12

# Exact cancellation triggering must return the first observed transaction
# boundary without running heavyweight power-log work in the hot poll. The
# release harness brackets this helper with the shared power guard and permits
# at most one already-submitted transaction to finish after disconnect.
progress_sequence="$test_dir/progress-sequence"
progress_index="$test_dir/progress-index"
printf '%s\n' 10 11 12 13 14 >"$progress_sequence"
printf '1\n' >"$progress_index"
# Invoked indirectly by name through qwen36_wait_for_exact_progress_count.
# shellcheck disable=SC2329
fixture_progress_count() {
  local index value
  index=$(cat "$progress_index")
  value=$(sed -n "${index}p" "$progress_sequence")
  [[ -n "$value" ]] || return 1
  printf '%s\n' "$((index + 1))" >"$progress_index"
  printf '%s\n' "$value"
}
exact_progress=$(qwen36_wait_for_exact_progress_count \
  fixture_progress_count 10 3 $$ "$(( $(date +%s) + 5 ))")
[[ "$exact_progress" == 3 ]]
[[ "$(cat "$progress_index")" == 5 ]]

printf '%s\n' 10 12 14 >"$progress_sequence"
printf '1\n' >"$progress_index"
expect_fail qwen36_wait_for_exact_progress_count \
  fixture_progress_count 10 3 $$ "$(( $(date +%s) + 5 ))"
printf 'not-a-count\n' >"$progress_sequence"
printf '1\n' >"$progress_index"
expect_fail qwen36_wait_for_exact_progress_count \
  fixture_progress_count 10 3 $$ "$(( $(date +%s) + 5 ))"

printf '9\n' >"$progress_sequence"
printf '1\n' >"$progress_index"
expect_fail qwen36_wait_for_exact_progress_count \
  fixture_progress_count 10 3 $$ "$(( $(date +%s) + 5 ))"

# A count callback may emit plausible output and still fail. Its nonzero
# status must remain load-bearing even through Bash conditional contexts.
# shellcheck disable=SC2329
fixture_progress_error() {
  printf '13\n'
  return 42
}
expect_fail qwen36_wait_for_exact_progress_count \
  fixture_progress_error 10 3 $$ "$(( $(date +%s) + 5 ))"

# Neither an incidental request exit nor a stale target observed after the
# deadline may be relabeled as the harness's intentional disconnect.
true &
dead_progress_pid=$!
wait "$dead_progress_pid"
if kill -0 "$dead_progress_pid" 2>/dev/null; then
  echo "dead progress fixture PID unexpectedly remained live" >&2
  exit 1
fi
printf '13\n' >"$progress_sequence"
printf '1\n' >"$progress_index"
expect_fail qwen36_wait_for_exact_progress_count \
  fixture_progress_count 10 3 "$dead_progress_pid" "$(( $(date +%s) + 5 ))"
printf '1\n' >"$progress_index"
if expired_progress=$(qwen36_wait_for_exact_progress_count \
  fixture_progress_count 10 3 $$ "$(( $(date +%s) - 1 ))" 2>/dev/null); then
  echo "expired Qwen progress target was accepted: $expired_progress" >&2
  exit 1
fi
unset -f fixture_progress_count
unset -f fixture_progress_error

progress_log="$test_dir/progress.log"
printf '%s\n' 'chunk complete' 'other' 'chunk complete' >"$progress_log"
[[ "$(qwen36_count_matching_lines 'chunk complete' "$progress_log")" == 2 ]]
[[ "$(qwen36_count_matching_lines 'no such chunk' "$progress_log")" == 0 ]]
# A scanner that emits a plausible count and then fails must not be accepted.
# shellcheck disable=SC2329
rg() {
  printf '2\n'
  return 42
}
expect_fail qwen36_count_matching_lines 'chunk complete' "$progress_log"
unset -f rg

cancellation_counts="$test_dir/cancellation-counts.json"
jq -n '{chunks_at_disconnect:3,chunks_after_cancel:4,chunks_after_stability:4}' \
  >"$cancellation_counts"
qwen36_validate_cancellation_transaction_counts "$cancellation_counts"
jq '.chunks_after_cancel = 3.5 | .chunks_after_stability = 3.5' \
  "$cancellation_counts" >"$cancellation_counts.fractional"
expect_fail qwen36_validate_cancellation_transaction_counts \
  "$cancellation_counts.fractional"
jq '.chunks_at_disconnect = "3"' "$cancellation_counts" \
  >"$cancellation_counts.string"
expect_fail qwen36_validate_cancellation_transaction_counts \
  "$cancellation_counts.string"
jq 'del(.chunks_after_stability)' "$cancellation_counts" \
  >"$cancellation_counts.missing"
expect_fail qwen36_validate_cancellation_transaction_counts \
  "$cancellation_counts.missing"
jq '.chunks_after_cancel = 3.5 | .chunks_after_stability = 3.5' \
  "$cancellation_counts" >"$cancellation_counts.invalid-document"
cat "$cancellation_counts.invalid-document" "$cancellation_counts" \
  >"$cancellation_counts.invalid-then-valid"
expect_fail qwen36_validate_cancellation_transaction_counts \
  "$cancellation_counts.invalid-then-valid"
cat "$cancellation_counts" "$cancellation_counts.invalid-document" \
  >"$cancellation_counts.valid-then-invalid"
expect_fail qwen36_validate_cancellation_transaction_counts \
  "$cancellation_counts.valid-then-invalid"

power_baseline="$test_dir/power-events.baseline"
power_final="$test_dir/power-events.final"
power_new="$test_dir/power-events.new"
printf '%s\n' \
  '2026-08-12 05:23:10 -0700 Sleep Entering Sleep state old-a' \
  '2026-08-12 06:08:39 -0700 Sleep Entering Sleep state old-b' \
  >"$power_baseline"
# A rolling log may prune the oldest baseline row without recording a new
# event. That is not an interruption and must not create a false positive.
printf '%s\n' \
  '2026-08-12 06:08:39 -0700 Sleep Entering Sleep state old-b' \
  >"$power_final"
qwen36_extract_new_power_events "$power_baseline" "$power_final" "$power_new"
[[ ! -s "$power_new" ]]

# A newly appended event remains visible even when another baseline row was
# pruned, which is the exact counterexample from hardware run 31755150906.
printf '%s\n' \
  '2026-08-12 06:08:39 -0700 Sleep Entering Sleep state old-b' \
  '2026-08-13 22:58:38 -0700 Sleep Entering Sleep state new-clamshell' \
  >"$power_final"
qwen36_extract_new_power_events "$power_baseline" "$power_final" "$power_new"
[[ "$(wc -l <"$power_new" | tr -d ' ')" == 1 ]]
grep -Fq 'new-clamshell' "$power_new"
expect_fail qwen36_validate_power_event_counts 12 13

# Preserve multiset semantics if two byte-identical rows ever appear.
printf '%s\n' \
  '2026-08-12 06:08:39 -0700 Sleep Entering Sleep state old-b' \
  '2026-08-12 06:08:39 -0700 Sleep Entering Sleep state old-b' \
  >"$power_final"
qwen36_extract_new_power_events "$power_baseline" "$power_final" "$power_new"
[[ "$(wc -l <"$power_new" | tr -d ' ')" == 1 ]]

# An empty baseline must not cause awk's usual NR==FNR empty-file trap.
: >"$power_baseline"
printf '%s\n' \
  '2026-08-13 22:58:38 -0700 Sleep Entering Sleep state first-event' \
  >"$power_final"
qwen36_extract_new_power_events "$power_baseline" "$power_final" "$power_new"
[[ "$(wc -l <"$power_new" | tr -d ' ')" == 1 ]]
expect_fail qwen36_extract_new_power_events \
  "$test_dir/missing-power-events" "$power_final" "$power_new"

pmset() {
  [[ "${QWEN36_TEST_PMSET_FAIL:-0}" == 0 ]] || return 1
  printf '%s\n' \
    '2026-08-13 22:58:38 -0700 Sleep Entering Sleep state captured-event' \
    "2026-08-13 22:58:33 -0700 Sleep Entering DarkWake state due to 'Clamshell Sleep':TCPKeepAlive=active Using AC (Charge:63%)" \
    'irrelevant power-log row'
}
qwen36_capture_power_events "$test_dir/captured-power-events"
[[ "$(wc -l <"$test_dir/captured-power-events" | tr -d ' ')" == 2 ]]
grep -Fq 'captured-event' "$test_dir/captured-power-events"
grep -Fq 'Entering DarkWake state' "$test_dir/captured-power-events"
QWEN36_TEST_PMSET_FAIL=1 \
  expect_fail qwen36_capture_power_events "$test_dir/failed-power-events"
unset -f pmset

# Bash 3.2 suppresses `errexit` when a function is evaluated by `if` or a
# command substitution. A capture error must return explicitly instead of
# reusing a pre-existing clean final snapshot and reporting a zero delta.
printf 'old sleep row\n' >"$test_dir/stale-power-events.baseline"
cp "$test_dir/stale-power-events.baseline" \
  "$test_dir/stale-power-events.final"
: >"$test_dir/stale-power-events.new"
QWEN36_POWER_GUARD_TARGET_PID=$$
QWEN36_POWER_GUARD_PID=$$
QWEN36_POWER_EVENT_BASELINE=1
QWEN36_POWER_EVENT_BASELINE_PATH="$test_dir/stale-power-events.baseline"
QWEN36_POWER_EVENT_FINAL_PATH="$test_dir/stale-power-events.final"
QWEN36_POWER_EVENT_NEW_PATH="$test_dir/stale-power-events.new"
qwen36_bound_caffeinate_pid() { printf '%s\n' "$$"; }
pmset() {
  case "$*" in
    '-g assertions') printf 'mock assertion\n' ;;
    '-g batt') printf "Now drawing from 'AC Power'\n" ;;
    '-g log') return 1 ;;
    *) return 1 ;;
  esac
}
expect_fail qwen36_assert_power_guard
wait_like_command_substitution() {
  qwen36_assert_power_guard || return 1
  printf 'unreachable\n'
}
if stale_result=$(wait_like_command_substitution 2>/dev/null); then
  echo "power capture failure was hidden by command substitution: $stale_result" >&2
  exit 1
fi
unset -f pmset qwen36_bound_caffeinate_pid wait_like_command_substitution

snapshot_root="$test_dir/power-snapshot-root"
snapshot_manifest="$snapshot_root/power-event-snapshots.sha256"
snapshot_prefixes=(family-a/caffeinate.log family-b/caffeinate.log)
release_snapshot_prefixes=()
while IFS= read -r prefix; do
  release_snapshot_prefixes+=("$prefix")
done < <(qwen36_release_power_snapshot_prefixes)
[[ "${#release_snapshot_prefixes[@]}" == 5 ]]
[[ "$(qwen36_power_snapshot_paths "${release_snapshot_prefixes[@]}" | wc -l | tr -d ' ')" == 15 ]]
for prefix in "${snapshot_prefixes[@]}"; do
  mkdir -p "$snapshot_root/$(dirname "$prefix")"
  printf '%s\n' "old event for $prefix" \
    >"$snapshot_root/${prefix}.power-events.baseline"
  cp "$snapshot_root/${prefix}.power-events.baseline" \
    "$snapshot_root/${prefix}.power-events.final"
  : >"$snapshot_root/${prefix}.power-events.new"
done
qwen36_write_power_snapshot_manifest \
  "$snapshot_root" "$snapshot_manifest" "${snapshot_prefixes[@]}"
qwen36_verify_power_snapshot_manifest \
  "$snapshot_root" "$snapshot_manifest" "${snapshot_prefixes[@]}"
[[ "$(wc -l <"$snapshot_manifest" | tr -d ' ')" == 6 ]]

# A producer that emits a complete-looking list and then fails must not be
# accepted in Bash conditional contexts where `errexit` is suppressed.
# shellcheck disable=SC2329
find() {
  command find "$@"
  return 42
}
expect_fail qwen36_validate_power_snapshot_inventory \
  "$snapshot_root" "${snapshot_prefixes[@]}"
unset -f find
# shellcheck disable=SC2329
qwen36_power_snapshot_paths() {
  local injected_prefix
  for injected_prefix in "$@"; do
    printf '%s\n' \
      "${injected_prefix}.power-events.baseline" \
      "${injected_prefix}.power-events.final" \
      "${injected_prefix}.power-events.new"
  done
  return 43
}
expect_fail qwen36_validate_power_snapshot_inventory \
  "$snapshot_root" "${snapshot_prefixes[@]}"
# Restore the authoritative helper definitions for the remaining tests.
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

# Download-side verification must reject changed bytes, an unrecorded delta,
# missing/extra inventory, and a nonempty committed `.new` file.
printf 'tamper\n' >>"$snapshot_root/family-a/caffeinate.log.power-events.final"
expect_fail qwen36_verify_power_snapshot_manifest \
  "$snapshot_root" "$snapshot_manifest" "${snapshot_prefixes[@]}"
cp "$snapshot_root/family-a/caffeinate.log.power-events.baseline" \
  "$snapshot_root/family-a/caffeinate.log.power-events.final"
qwen36_write_power_snapshot_manifest \
  "$snapshot_root" "$snapshot_manifest" "${snapshot_prefixes[@]}"
printf 'new dark wake\n' >>"$snapshot_root/family-a/caffeinate.log.power-events.final"
qwen36_write_power_snapshot_manifest \
  "$snapshot_root" "$snapshot_manifest" "${snapshot_prefixes[@]}"
expect_fail qwen36_verify_power_snapshot_manifest \
  "$snapshot_root" "$snapshot_manifest" "${snapshot_prefixes[@]}"
cp "$snapshot_root/family-a/caffeinate.log.power-events.baseline" \
  "$snapshot_root/family-a/caffeinate.log.power-events.final"
: >"$snapshot_root/family-a/caffeinate.log.power-events.new"
qwen36_write_power_snapshot_manifest \
  "$snapshot_root" "$snapshot_manifest" "${snapshot_prefixes[@]}"
printf 'unbound\n' >"$snapshot_root/extra.power-events.baseline"
expect_fail qwen36_verify_power_snapshot_manifest \
  "$snapshot_root" "$snapshot_manifest" "${snapshot_prefixes[@]}"
rm "$snapshot_root/extra.power-events.baseline"
rm "$snapshot_root/family-b/caffeinate.log.power-events.final"
expect_fail qwen36_verify_power_snapshot_manifest \
  "$snapshot_root" "$snapshot_manifest" "${snapshot_prefixes[@]}"
cp "$snapshot_root/family-b/caffeinate.log.power-events.baseline" \
  "$snapshot_root/family-b/caffeinate.log.power-events.final"
printf 'new event\n' >"$snapshot_root/family-b/caffeinate.log.power-events.new"
expect_fail qwen36_write_power_snapshot_manifest \
  "$snapshot_root" "$snapshot_manifest" "${snapshot_prefixes[@]}"

# Every leaf powered receipt must take its terminal snapshot after validating
# the temporary JSON and immediately before the atomic receipt commit.
for powered_script in \
  test_qwen36_cumulative_release.sh \
  test_qwen36_prefill_watchdog.sh \
  test_qwen36_prefill_cancellation.sh \
  test_deepseek4_interactive_overlap.sh \
  test_gemma4_long_short_overlap.sh; do
  awk '
    previous == "qwen36_assert_power_guard" && /^mv .*summary/ { found = 1 }
    { previous = $0 }
    END { exit(found ? 0 : 1) }
  ' "$script_dir/$powered_script" || {
    echo "powered receipt lacks a terminal assertion: $powered_script" >&2
    exit 1
  }
done

grep -Fq 'qwen36_wait_for_exact_progress_count' \
  "$script_dir/test_qwen36_prefill_cancellation.sh"
grep -Fq 'qwen36_count_matching_lines' \
  "$script_dir/test_qwen36_prefill_cancellation.sh"
# Literal shell source; `$cancel_pid` must not expand in this contract.
# shellcheck disable=SC2016
grep -Fq 'kill "$cancel_pid" 2>/dev/null || {' \
  "$script_dir/test_qwen36_prefill_cancellation.sh"
grep -Fq '.chunks_after_cancel <= (.chunks_at_disconnect + 1)' \
  "$script_dir/test_qwen36_prefill_cancellation.sh"
grep -Fq '.chunks_after_cancel == (.chunks_after_cancel | floor)' \
  "$script_dir/test_qwen36_prefill_cancellation.sh"
# Literal jq source; `$q` is the release verifier binding, not a shell value.
# shellcheck disable=SC2016
grep -Fq '$q.cancellation.chunks_after_cancel <= ($q.cancellation.chunks_at_disconnect + 1)' \
  "$script_dir/../.github/workflows/release.yml"
# Literal jq source; `$q` is the release verifier binding, not a shell value.
# shellcheck disable=SC2016
grep -Fq '$q.cancellation.chunks_after_cancel == ($q.cancellation.chunks_after_cancel | floor)' \
  "$script_dir/../.github/workflows/release.yml"
grep -Fq 'qwen36_validate_cancellation_transaction_counts' \
  "$script_dir/../.github/workflows/release.yml"
grep -Fq 'fixture_sha256 == "3558d4f4b251ed833ee7da1b037fa3f241a4309590d45930b525b690f543a31e"' \
  "$script_dir/../.github/workflows/release.yml"
if grep -Fq 'fixture_sha256 == "6671a0c89b8d4935caa4b87bee08361c5b8727ec557e9edb05947ad90c94c13d"' \
  "$script_dir/../.github/workflows/release.yml"; then
  echo "release verifier still accepts the historical key-sorted fixture" >&2
  exit 1
fi
# Literal jq source; `$qwen_cancellation` must not expand in this contract.
# shellcheck disable=SC2016
grep -Fq '($qwen_cancellation | length) != 1' \
  "$script_dir/run_agentic_cache_release_gate.sh"
awk '
  /qwen36_wait_for_exact_progress_count/ { wait_line = NR }
  /qwen36_assert_power_guard/ {
    if (!wait_line) pre_assert = NR
    else if (!post_assert) post_assert = NR
  }
  /kill "\$cancel_pid"/ && wait_line && !disconnect { disconnect = NR }
  END {
    ok = pre_assert && wait_line && disconnect && post_assert
    ok = ok && pre_assert < wait_line && wait_line < disconnect
    ok = ok && disconnect < post_assert
    exit(ok ? 0 : 1)
  }
' "$script_dir/test_qwen36_prefill_cancellation.sh" || {
  echo "Qwen cancellation trigger is not bracketed by power assertions" >&2
  exit 1
}

grep -Fq 'power_event_snapshots_sha256' \
  "$script_dir/run_agentic_cache_release_gate.sh"
grep -Fq 'qwen36_verify_power_snapshot_manifest' \
  "$script_dir/../.github/workflows/release.yml"

clean_log="$test_dir/clean.log"
printf 'INFO bounded transaction complete\n' >"$clean_log"
qwen36_reject_fatal_log "$clean_log"
expect_fail qwen36_reject_fatal_log "$test_dir/missing.log"
qwen36_write_log_baseline "$clean_log" "$test_dir/log-baseline.json"
printf 'INFO later bounded transaction complete\n' >>"$clean_log"
qwen36_extract_append_only_log_delta "$clean_log" "$test_dir/log-baseline.json" \
  "$test_dir/log-delta.log"
[[ "$(wc -l <"$test_dir/log-delta.log" | tr -d ' ')" == 1 ]]
printf 'CHANGED baseline\nINFO later bounded transaction complete\n' >"$clean_log"
expect_fail qwen36_extract_append_only_log_delta "$clean_log" \
  "$test_dir/log-baseline.json" "$test_dir/changed-log-delta.log"
for signature in \
  'GPU Timeout' \
  'SubmissionsIgnored' \
  'Command buffer error' \
  'Generation error' \
  'engine_unhealthy' \
  'panicked at' \
  'worker-fatal'; do
  printf '%s\n' "$signature" >"$test_dir/fatal.log"
  expect_fail qwen36_reject_fatal_log "$test_dir/fatal.log"
done

keepalive_sse="$test_dir/keepalive-only.sse"
printf ': keepalive\n\n: keepalive\n\n' >"$keepalive_sse"
[[ "$(qwen36_sse_data_bytes "$keepalive_sse")" == 0 ]]
printf 'data: {"id":"progress"}\n\n' >>"$keepalive_sse"
data_bytes=$(qwen36_sse_data_bytes "$keepalive_sse")
[[ "$data_bytes" -gt 0 ]]
printf ': keepalive\n\n: keepalive\n\n' >>"$keepalive_sse"
[[ "$(qwen36_sse_data_bytes "$keepalive_sse")" == "$data_bytes" ]] || {
  echo "SSE keepalive comments must not count as semantic progress" >&2
  exit 1
}

cancelled_sse="$test_dir/cancelled.sse"
printf ': keepalive\n\ndata: {"id":"cancelled","object":"chat.completion.chunk","choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n' >"$cancelled_sse"
qwen36_reject_successful_terminal_sse "$cancelled_sse"
cp "$cancelled_sse" "$test_dir/cancelled-done.sse"
printf 'data: [DONE]\n\n' >>"$test_dir/cancelled-done.sse"
expect_fail qwen36_reject_successful_terminal_sse "$test_dir/cancelled-done.sse"
cp "$cancelled_sse" "$test_dir/cancelled-finish.sse"
printf 'data: {"id":"cancelled","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n' \
  >>"$test_dir/cancelled-finish.sse"
expect_fail qwen36_reject_successful_terminal_sse "$test_dir/cancelled-finish.sse"
printf 'data: {not-json}\n\n' >"$test_dir/cancelled-malformed.sse"
expect_fail qwen36_reject_successful_terminal_sse "$test_dir/cancelled-malformed.sse"
printf ': keepalive\n\n' >"$test_dir/cancelled-keepalive-only.sse"
qwen36_reject_successful_terminal_sse "$test_dir/cancelled-keepalive-only.sse"
printf 'data: {}\n\n' >"$test_dir/cancelled-empty-object.sse"
expect_fail qwen36_reject_successful_terminal_sse "$test_dir/cancelled-empty-object.sse"
printf 'data: {"id":"cancelled","object":"chat.completion.chunk","choices":[]}\n\n' \
  >"$test_dir/cancelled-empty-choices.sse"
expect_fail qwen36_reject_successful_terminal_sse "$test_dir/cancelled-empty-choices.sse"
printf 'data: {"error":{"message":"engine_unhealthy"}}\n\n' \
  >"$test_dir/cancelled-error-object.sse"
expect_fail qwen36_reject_successful_terminal_sse "$test_dir/cancelled-error-object.sse"
cp "$cancelled_sse" "$test_dir/cancelled-mixed-ids.sse"
printf 'data: {"id":"different","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":null}]}\n\n' \
  >>"$test_dir/cancelled-mixed-ids.sse"
expect_fail qwen36_reject_successful_terminal_sse "$test_dir/cancelled-mixed-ids.sse"
printf 'data: {"id":"cancelled","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":null},{"delta":{},"finish_reason":null}]}\n\n' \
  >"$test_dir/cancelled-multiple-choices.sse"
expect_fail qwen36_reject_successful_terminal_sse "$test_dir/cancelled-multiple-choices.sse"
printf 'data: {"id":"cancelled","object":"chat.completion.chunk","choices":[{"finish_reason":null}]}\n\n' \
  >"$test_dir/cancelled-missing-delta.sse"
expect_fail qwen36_reject_successful_terminal_sse "$test_dir/cancelled-missing-delta.sse"

write_heap_fixture() {
  local path="$1"
  local cfstrings="$2"
  local command_buffers="$3"
  local command_buffer_impls="$4"
  local pool_pages="$5"
  cat >"$path" <<EOF
   COUNT      BYTES       AVG   CLASS_NAME                                        TYPE    BINARY
   =====      =====       ===   ==========                                        ====    ======
  $cfstrings      4096      48.0   CFString                                         ObjC    CoreFoundation
       7       1024     146.3   CFString (Storage)                               C       CoreFoundation
  $command_buffers       896     896.0   AGXG17XFamilyCommandBuffer                       ObjC    AGXMetalG17X
  $command_buffer_impls       640     640.0   AGXG17XFamilyCommandBuffer._impl                 malloc  AGXMetalG17X
  $pool_pages      4096    4096.0   @autoreleasepool content                        C       libobjc.A.dylib
EOF
}

for phase in baseline overlap warmup wave1 wave2; do
  case "$phase" in
    baseline) cf=100; pool=10 ;;
    overlap) cf=120; pool=12 ;;
    warmup) cf=140; pool=14 ;;
    wave1) cf=150; pool=16 ;;
    wave2) cf=160; pool=18 ;;
  esac
  write_heap_fixture "$test_dir/$phase.heap" "$cf" 0 0 "$pool"
  qwen36_parse_heap_summary "$test_dir/$phase.heap" "$test_dir/$phase.json"
done
qwen36_validate_heap_series \
  "$test_dir/baseline.json" "$test_dir/overlap.json" "$test_dir/warmup.json" \
  "$test_dir/wave1.json" "$test_dir/wave2.json"
write_heap_fixture "$test_dir/leaked-cb.heap" 160 1 1 18
qwen36_parse_heap_summary "$test_dir/leaked-cb.heap" "$test_dir/leaked-cb.json"
expect_fail qwen36_validate_heap_series \
  "$test_dir/baseline.json" "$test_dir/overlap.json" "$test_dir/warmup.json" \
  "$test_dir/wave1.json" "$test_dir/leaked-cb.json"
write_heap_fixture "$test_dir/leaked-label.heap" 10000 0 0 100
qwen36_parse_heap_summary "$test_dir/leaked-label.heap" "$test_dir/leaked-label.json"
expect_fail qwen36_validate_heap_series \
  "$test_dir/baseline.json" "$test_dir/overlap.json" "$test_dir/warmup.json" \
  "$test_dir/wave1.json" "$test_dir/leaked-label.json"
cp "$test_dir/baseline.heap" "$test_dir/duplicate-cfstring.heap"
printf '       1         48      48.0   CFString                                         ObjC    CoreFoundation\n' \
  >>"$test_dir/duplicate-cfstring.heap"
expect_fail qwen36_parse_heap_summary \
  "$test_dir/duplicate-cfstring.heap" "$test_dir/duplicate-cfstring.json"
cp "$test_dir/baseline.heap" "$test_dir/malformed-heap-count.heap"
sed -i '' 's/^  100      4096/  x      4096/' "$test_dir/malformed-heap-count.heap"
expect_fail qwen36_parse_heap_summary \
  "$test_dir/malformed-heap-count.heap" "$test_dir/malformed-heap-count.json"
cp "$test_dir/baseline.heap" "$test_dir/malformed-heap-bytes.heap"
sed -i '' 's/^  100      4096/  100      12x/' "$test_dir/malformed-heap-bytes.heap"
expect_fail qwen36_parse_heap_summary \
  "$test_dir/malformed-heap-bytes.heap" "$test_dir/malformed-heap-bytes.json"
write_heap_fixture "$test_dir/warmup-spike.heap" 10000 0 0 14
qwen36_parse_heap_summary "$test_dir/warmup-spike.heap" "$test_dir/warmup-spike.json"
expect_fail qwen36_validate_heap_series \
  "$test_dir/baseline.json" "$test_dir/overlap.json" "$test_dir/warmup-spike.json" \
  "$test_dir/wave1.json" "$test_dir/wave2.json"
write_heap_fixture "$test_dir/overlap-pool-spike.heap" 120 0 0 10000
qwen36_parse_heap_summary "$test_dir/overlap-pool-spike.heap" \
  "$test_dir/overlap-pool-spike.json"
expect_fail qwen36_validate_heap_series \
  "$test_dir/baseline.json" "$test_dir/overlap-pool-spike.json" \
  "$test_dir/warmup.json" "$test_dir/wave1.json" "$test_dir/wave2.json"

valid_chunks="$test_dir/valid-chunks.log"
for ordinal in $(seq 0 42); do
  start=$((ordinal * 2048))
  tokens=2048
  [[ "$ordinal" == 42 ]] && tokens=1956
  end=$((start + tokens))
  printf 'INFO Qwen35 bounded prefill chunk complete slot=0 chunk_start=%s chunk_end=%s chunk_tokens=%s prompt_tokens=87972\n' \
    "$start" "$end" "$tokens" >>"$valid_chunks"
done
qwen36_validate_chunk_lines "$valid_chunks"

stable_chunks="$test_dir/stable-chunks.log"
for ordinal in $(seq 0 41); do
  start=$((ordinal * 2048))
  end=$((start + 2048))
  printf 'INFO Qwen35 bounded prefill chunk complete slot=0 chunk_start=%s chunk_end=%s chunk_tokens=2048 prompt_tokens=87972\n' \
    "$start" "$end" >>"$stable_chunks"
done
printf 'INFO Qwen35 bounded prefill chunk complete slot=0 chunk_start=86016 chunk_end=87965 chunk_tokens=1949 prompt_tokens=87972\n' \
  >>"$stable_chunks"
printf 'INFO Qwen35 bounded prefill chunk complete slot=0 chunk_start=87965 chunk_end=87972 chunk_tokens=7 prompt_tokens=87972\n' \
  >>"$stable_chunks"
qwen36_validate_stable_boundary_chunk_lines "$stable_chunks" 87965
[[ "$(qwen36_count_chunks_with_tokens "$stable_chunks" 2048)" == 42 ]]
[[ "$(qwen36_count_chunks_with_tokens "$stable_chunks" 7)" == 1 ]]
expect_fail qwen36_count_chunks_with_tokens "$stable_chunks" not-a-number
expect_fail qwen36_validate_stable_boundary_chunk_lines "$stable_chunks" 87964
cache_hit_line='INFO cache hit slot=1 prompt_tokens=88040 cached_tokens=87965 suffix_tokens=75'
[[ "$(qwen36_log_uint_field "$cache_hit_line" prompt_tokens)" == 88040 ]]
expect_fail qwen36_log_uint_field "$cache_hit_line" missing_field
expect_fail qwen36_log_uint_field \
  "$cache_hit_line prompt_tokens=1" prompt_tokens
expect_fail qwen36_log_uint_field \
  'INFO cache hit prompt_tokens=not-a-number' prompt_tokens
cp "$stable_chunks" "$test_dir/stable-cross-slot.log"
sed -i '' '43s/slot=0/slot=1/' "$test_dir/stable-cross-slot.log"
expect_fail qwen36_validate_stable_boundary_chunk_lines \
  "$test_dir/stable-cross-slot.log" 87965

head -42 "$valid_chunks" >"$test_dir/missing-tail.log"
expect_fail qwen36_validate_chunk_lines "$test_dir/missing-tail.log"
cp "$valid_chunks" "$test_dir/discontinuous.log"
sed -i '' '22s/chunk_start=43008/chunk_start=43009/' "$test_dir/discontinuous.log"
expect_fail qwen36_validate_chunk_lines "$test_dir/discontinuous.log"
cp "$valid_chunks" "$test_dir/wrong-tail.log"
sed -i '' '$s/chunk_tokens=1956/chunk_tokens=1955/' "$test_dir/wrong-tail.log"
expect_fail qwen36_validate_chunk_lines "$test_dir/wrong-tail.log"
cp "$valid_chunks" "$test_dir/cross-slot.log"
sed -i '' '22s/slot=0/slot=1/' "$test_dir/cross-slot.log"
expect_fail qwen36_validate_chunk_lines "$test_dir/cross-slot.log"
cp "$valid_chunks" "$test_dir/missing-slot.log"
sed -i '' '1s/slot=0 //' "$test_dir/missing-slot.log"
expect_fail qwen36_validate_chunk_lines "$test_dir/missing-slot.log"
cp "$valid_chunks" "$test_dir/missing-start.log"
sed -i '' '1s/chunk_start=0 //' "$test_dir/missing-start.log"
expect_fail qwen36_validate_chunk_lines "$test_dir/missing-start.log"
cp "$valid_chunks" "$test_dir/non-numeric-slot.log"
sed -i '' '1s/slot=0/slot=not-a-number/' "$test_dir/non-numeric-slot.log"
expect_fail qwen36_validate_chunk_lines "$test_dir/non-numeric-slot.log"
cp "$valid_chunks" "$test_dir/non-numeric-start.log"
sed -i '' '1s/chunk_start=0/chunk_start=not-a-number/' "$test_dir/non-numeric-start.log"
expect_fail qwen36_validate_chunk_lines "$test_dir/non-numeric-start.log"
cp "$valid_chunks" "$test_dir/duplicate-slot.log"
sed -i '' '1s/slot=0/slot=0 slot=0/' "$test_dir/duplicate-slot.log"
expect_fail qwen36_validate_chunk_lines "$test_dir/duplicate-slot.log"

short_sse="$test_dir/short.sse"
cat >"$short_sse" <<'EOF'
data: {"id":"chat-short","object":"chat.completion.chunk","choices":[{"delta":{"content":"OK"},"finish_reason":null}]}

: keepalive

data: {"id":"chat-short","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]

EOF
qwen36_extract_and_validate_sse short "$short_sse" "$test_dir/short.jsonl"
qwen36_validate_short_events "$test_dir/short.jsonl"

cp "$short_sse" "$test_dir/duplicate-done.sse"
printf 'data: [DONE]\n\n' >>"$test_dir/duplicate-done.sse"
expect_fail qwen36_extract_and_validate_sse short "$test_dir/duplicate-done.sse" \
  "$test_dir/duplicate-done.jsonl"
cp "$short_sse" "$test_dir/nonterminal-done.sse"
printf 'data: {"id":"late","object":"chat.completion.chunk","choices":[],"usage":{}}\n\n' \
  >>"$test_dir/nonterminal-done.sse"
expect_fail qwen36_extract_and_validate_sse short "$test_dir/nonterminal-done.sse" \
  "$test_dir/nonterminal-done.jsonl"
cp "$short_sse" "$test_dir/non-data-line.sse"
printf 'event: surprise\n' >>"$test_dir/non-data-line.sse"
expect_fail qwen36_extract_and_validate_sse short "$test_dir/non-data-line.sse" \
  "$test_dir/non-data-line.jsonl"
cp "$test_dir/short.jsonl" "$test_dir/wrong-short.jsonl"
sed -i '' 's/"OK"/"NO"/' "$test_dir/wrong-short.jsonl"
expect_fail qwen36_validate_short_events "$test_dir/wrong-short.jsonl"
cp "$test_dir/short.jsonl" "$test_dir/wrong-short-finish.jsonl"
sed -i '' 's/"stop"/"length"/' "$test_dir/wrong-short-finish.jsonl"
expect_fail qwen36_validate_short_events "$test_dir/wrong-short-finish.jsonl"
jq 'if .choices[0].delta.content then .choices[0].delta.content = "O\nK" else . end' \
  "$test_dir/short.jsonl" >"$test_dir/newline-short.jsonl"
expect_fail qwen36_validate_short_events "$test_dir/newline-short.jsonl"
cp "$test_dir/short.jsonl" "$test_dir/empty-short-event.jsonl"
printf '{}\n' >>"$test_dir/empty-short-event.jsonl"
expect_fail qwen36_validate_short_events "$test_dir/empty-short-event.jsonl"
jq 'if .choices[0].finish_reason == "stop" then .id = "different-id" else . end' \
  "$test_dir/short.jsonl" >"$test_dir/split-short-id.jsonl"
expect_fail qwen36_validate_short_events "$test_dir/split-short-id.jsonl"

long_sse="$test_dir/long.sse"
cat >"$long_sse" <<'EOF'
data: {"id":"chat-long","object":"chat.completion.chunk","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"fixture_tool_346","arguments":"{\"path\":\"src/serve/api/engine.rs\"}"}}]},"finish_reason":null}]}

data: {"id":"chat-long","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

EOF
qwen36_extract_and_validate_sse long "$long_sse" "$test_dir/long.jsonl"
qwen36_validate_long_events "$test_dir/long.jsonl"

jq 'if .choices[0].delta.tool_calls then
      .choices[0].delta.tool_calls[0].function.arguments =
        "{\"path\": \"src/serve/api/engine.rs\"}"
    else . end' "$test_dir/long.jsonl" >"$test_dir/spaced-args.jsonl"
qwen36_validate_long_events "$test_dir/spaced-args.jsonl"

jq 'if .choices[0].delta.tool_calls then .choices[0].delta.tool_calls[0].index = 1 else . end' \
  "$test_dir/long.jsonl" >"$test_dir/second-index.jsonl"
expect_fail qwen36_validate_long_events "$test_dir/second-index.jsonl"
jq 'if .choices[0].delta.tool_calls then del(.choices[0].delta.tool_calls[0].id) else . end' \
  "$test_dir/long.jsonl" >"$test_dir/missing-id.jsonl"
expect_fail qwen36_validate_long_events "$test_dir/missing-id.jsonl"
jq 'if .choices[0].delta.tool_calls then del(.choices[0].delta.tool_calls[0].type) else . end' \
  "$test_dir/long.jsonl" >"$test_dir/missing-type.jsonl"
expect_fail qwen36_validate_long_events "$test_dir/missing-type.jsonl"
jq 'if .choices[0].delta.tool_calls then .choices[0].delta.content = "leak" else . end' \
  "$test_dir/long.jsonl" >"$test_dir/content-leak.jsonl"
expect_fail qwen36_validate_long_events "$test_dir/content-leak.jsonl"
jq 'if .choices[0].finish_reason == "tool_calls" then .choices[0].finish_reason = "stop" else . end' \
  "$test_dir/long.jsonl" >"$test_dir/wrong-long-finish.jsonl"
expect_fail qwen36_validate_long_events "$test_dir/wrong-long-finish.jsonl"
jq 'if .choices[0].delta.tool_calls then
      .choices[0].delta.tool_calls[0].function.arguments =
        "{\"path\":\"src/serve/api/\nengine.rs\"}"
    else . end' "$test_dir/long.jsonl" >"$test_dir/newline-args.jsonl"
expect_fail qwen36_validate_long_events "$test_dir/newline-args.jsonl"
cp "$test_dir/long.jsonl" "$test_dir/empty-long-event.jsonl"
printf '{}\n' >>"$test_dir/empty-long-event.jsonl"
expect_fail qwen36_validate_long_events "$test_dir/empty-long-event.jsonl"

printf 'qwen36 watchdog harness contract: pass\n'
