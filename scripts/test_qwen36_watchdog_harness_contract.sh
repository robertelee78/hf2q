#!/usr/bin/env bash
# Model-free negative tests for the exact Qwen watchdog receipt parser.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

for command in jq awk sed wc tr mktemp seq shasum stat find; do
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
