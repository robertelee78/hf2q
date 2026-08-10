#!/usr/bin/env bash
set -euo pipefail

# Focused DeepSeek-V4 SlotAware gate for the two properties that ordinary
# agentic smoke tests do not exercise:
#   1. a cached tool-result suffix yields between transactions so an already
#      decoding SSE peer continues to make semantic progress; and
#   2. dropping a longer cached suffix at a committed transaction boundary
#      stops within one additional transaction, increments cancellation once,
#      and leaves the engine ready and reusable.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:8080}
MODEL=${MODEL:-Deepseek v4 Flash 0731 Source}
SERVER_LOG=${SERVER_LOG:?SERVER_LOG must name the current server log}
SERVER_PID=${SERVER_PID:?SERVER_PID must name the current hf2q process}
OUT_DIR=${OUT_DIR:?OUT_DIR must name a fresh receipt directory}
EXPECTED_PATH=${EXPECTED_PATH:-$ROOT_DIR/README.md}
OVERLAP_TOOL_RESULT_PATH=${OVERLAP_TOOL_RESULT_PATH:-$EXPECTED_PATH}
CANCEL_TOOL_RESULT_PATH=${CANCEL_TOOL_RESULT_PATH:-$ROOT_DIR/docs/ADR-042-deepseek-v4-flash-rust-native.md}
CURL_CONNECT_TIMEOUT_SECONDS=${CURL_CONNECT_TIMEOUT_SECONDS:-5}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-180}
PREFILL_CHUNKS_BEFORE_CANCEL=${PREFILL_CHUNKS_BEFORE_CANCEL:-3}
CANCEL_SETTLE_SECONDS=${CANCEL_SETTLE_SECONDS:-15}
CANCEL_STABILITY_SECONDS=${CANCEL_STABILITY_SECONDS:-10}

for command in awk curl date jq ps rg sed shasum; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
for path in "$SERVER_LOG" "$EXPECTED_PATH" "$OVERLAP_TOOL_RESULT_PATH" \
  "$CANCEL_TOOL_RESULT_PATH"; do
  [[ -r "$path" ]] || {
    echo "required path is not readable: $path" >&2
    exit 2
  }
done
for setting in SERVER_PID CURL_CONNECT_TIMEOUT_SECONDS CURL_MAX_TIME_SECONDS \
  PREFILL_CHUNKS_BEFORE_CANCEL CANCEL_SETTLE_SECONDS CANCEL_STABILITY_SECONDS; do
  value=${!setting}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$setting must be a positive integer (got: $value)" >&2
    exit 2
  }
done
if [[ -e "$OUT_DIR" ]]; then
  echo "OUT_DIR must not already exist: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

server_command=$(ps -ww -p "$SERVER_PID" -o command=)
[[ -n "$server_command" && "$server_command" == *" serve "* ]] || {
  echo "SERVER_PID is not the expected live hf2q server: $SERVER_PID" >&2
  exit 2
}
if [[ "$server_command" != *"--max-slots 4"* ]]; then
  echo "DeepSeek cached-suffix gate requires --max-slots 4" >&2
  exit 2
fi
binary_path=${server_command%% *}
[[ -x "$binary_path" ]] || {
  echo "server binary is not executable: $binary_path" >&2
  exit 2
}
binary_sha256=$(shasum -a 256 "$binary_path" | awk '{print $1}')
server_log_inode_before=$(stat -f '%i' "$SERVER_LOG")
server_log_lines_before=$(wc -l <"$SERVER_LOG" | tr -d ' ')
server_log_prefix_sha256=$(sed -n "1,${server_log_lines_before}p" "$SERVER_LOG" | shasum -a 256 | awk '{print $1}')

request_a="$OUT_DIR/base-a.request.json"
response_a="$OUT_DIR/base-a.response.json"
response_a_cached="$OUT_DIR/base-a-cached.response.json"
continuation_a="$OUT_DIR/overlap.request.json"
continuation_a_response="$OUT_DIR/overlap.response.json"
peer_request="$OUT_DIR/peer.request.json"
peer_warm_response="$OUT_DIR/peer-warm.response.json"
peer_stream_request="$OUT_DIR/peer-stream.request.json"
peer_sse="$OUT_DIR/peer.sse"
peer_err="$OUT_DIR/peer.stderr"
request_b="$OUT_DIR/base-b.request.json"
response_b="$OUT_DIR/base-b.response.json"
cancel_request="$OUT_DIR/cancel.request.json"
cancel_sse="$OUT_DIR/cancel.sse"
cancel_err="$OUT_DIR/cancel.stderr"
late_cancel_sse="$OUT_DIR/late-cancel.sse"
late_cancel_err="$OUT_DIR/late-cancel.stderr"
decode_seed_request="$OUT_DIR/decode-seed.request.json"
decode_seed_response="$OUT_DIR/decode-seed.response.json"
decode_cancel_request="$OUT_DIR/decode-cancel.request.json"
decode_cancel_sse="$OUT_DIR/decode-cancel.sse"
decode_cancel_err="$OUT_DIR/decode-cancel.stderr"
decode_recovery_request="$OUT_DIR/decode-recovery.request.json"
decode_recovery_response="$OUT_DIR/decode-recovery.response.json"
control_response="$OUT_DIR/control.response.json"
summary_tmp="$OUT_DIR/summary.json.tmp"
summary="$OUT_DIR/summary.json"
peer_pid=
cancel_pid=
late_cancel_pid=
decode_cancel_pid=
cleanup() {
  if [[ -n "$peer_pid" ]] && kill -0 "$peer_pid" 2>/dev/null; then
    kill -TERM "$peer_pid" 2>/dev/null || true
    wait "$peer_pid" 2>/dev/null || true
  fi
  if [[ -n "$cancel_pid" ]] && kill -0 "$cancel_pid" 2>/dev/null; then
    kill -TERM "$cancel_pid" 2>/dev/null || true
    wait "$cancel_pid" 2>/dev/null || true
  fi
  if [[ -n "$late_cancel_pid" ]] && kill -0 "$late_cancel_pid" 2>/dev/null; then
    kill -TERM "$late_cancel_pid" 2>/dev/null || true
    wait "$late_cancel_pid" 2>/dev/null || true
  fi
  if [[ -n "$decode_cancel_pid" ]] && kill -0 "$decode_cancel_pid" 2>/dev/null; then
    kill -TERM "$decode_cancel_pid" 2>/dev/null || true
    wait "$decode_cancel_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

epoch_ms() {
  echo $(( $(date +%s) * 1000 ))
}

metric_value() {
  local metric=$1
  curl --fail-with-body --silent --show-error \
    --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
    --max-time 10 "$BASE_URL/metrics" |
    awk -v metric="$metric" '$1 == metric { value=$2 } END { if (value == "") exit 1; print value }'
}

post_json() {
  local input=$1
  local output=$2
  curl --fail-with-body --silent --show-error --no-buffer \
    --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
    --max-time "$CURL_MAX_TIME_SECONDS" \
    -H 'Content-Type: application/json' \
    --data-binary "@$input" \
    "$BASE_URL/v1/chat/completions" >"$output"
}

assert_tool_path() {
  local response=$1
  local expected=$2
  jq -e --arg expected "$expected" '
    (.choices | length) == 1
    and .choices[0].finish_reason == "tool_calls"
    and ((.choices[0].message.tool_calls // []) | length) == 1
    and .choices[0].message.tool_calls[0].function.name == "read_file"
    and ((.choices[0].message.tool_calls[0].function.arguments | fromjson).path == $expected)
  ' "$response" >/dev/null
}

build_base_request() {
  local output=$1
  local run_id=$2
  local expected=$3
  jq -n --rawfile repo "$ROOT_DIR/README.md" \
    --arg model "$MODEL" --arg run_id "$run_id" --arg expected "$expected" '{
      model: $model,
      messages: [
        {role: "system", content: "You are an agentic coding assistant. Use read_file before answering."},
        {role: "user", content: ("Cached suffix gate " + $run_id + ". Inspect the repository context, then call read_file exactly once with path " + $expected + ". After the tool succeeds, reply with exactly DEEPSEEK_CACHED_SUFFIX_OK and nothing else. Context follows:\n\n" + $repo)}
      ],
      tools: [{type: "function", function: {
        name: "read_file",
        description: "Read one UTF-8 workspace file",
        parameters: {type: "object", properties: {path: {type: "string"}}, required: ["path"], additionalProperties: false}
      }}],
      tool_choice: "required",
      temperature: 0,
      max_tokens: 128,
      stream: false
    }' >"$output"
}

build_continuation() {
  local base=$1
  local prior=$2
  local tool_result_path=$3
  local output=$4
  local stream=$5
  jq -n --slurpfile base "$base" --slurpfile prior "$prior" \
    --rawfile tool_result "$tool_result_path" --argjson stream "$stream" '
      $base[0]
      | .messages += [
          {role: "assistant", content: $prior[0].choices[0].message.content, tool_calls: $prior[0].choices[0].message.tool_calls},
          {role: "tool", tool_call_id: $prior[0].choices[0].message.tool_calls[0].id, content: ("Successful read_file result. File follows:\n" + $tool_result)}
        ]
      | .tool_choice = "auto"
      | .stream = $stream
      | if $stream then .stream_options = {include_usage: true} else . end
    ' >"$output"
}

sse_has_semantic_content() {
  local path=$1
  [[ -s "$path" ]] || return 1
  sed -n 's/^data: //p' "$path" | rg -v '^\[DONE\]$' |
    jq -e -s 'any(.[]; any(.choices[]?; ((.delta.content // "") | length) > 0))' >/dev/null 2>&1
}

wait_for_semantic_content() {
  local path=$1
  local pid=$2
  local deadline=$(( $(date +%s) + 90 ))
  while (( $(date +%s) < deadline )); do
    if sse_has_semantic_content "$path"; then
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  echo "SSE peer emitted no semantic content before exit/deadline" >&2
  return 1
}

wait_for_request_id() {
  local from_line=$1
  local mode=$2
  local deadline=$(( $(date +%s) + 30 ))
  local found
  while (( $(date +%s) < deadline )); do
    found=$(sed -n "$((from_line + 1)),\$p" "$SERVER_LOG" |
      awk -v mode="$mode" 'index($0, "mode=\"" mode "\"") {
        if (match($0, /request_id=[0-9]+/)) {
          print substr($0, RSTART + 11, RLENGTH - 11)
          exit
        }
      }')
    if [[ -n "$found" ]]; then
      printf '%s\n' "$found"
      return 0
    fi
    sleep 0.2
  done
  echo "server log did not expose request id for mode=$mode" >&2
  return 1
}

prefill_progress_count() {
  local request_id=$1
  rg -c "prefill progress request_id=${request_id}( |$)" "$SERVER_LOG" || true
}

wait_for_prefill_chunks() {
  local request_id=$1
  local wanted=$2
  local pid=$3
  local deadline=$(( $(date +%s) + 120 ))
  local count
  while (( $(date +%s) < deadline )); do
    count=$(prefill_progress_count "$request_id")
    if (( count >= wanted )); then
      printf '%s\n' "$count"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  echo "request $request_id reached only $(prefill_progress_count "$request_id")/$wanted prefill chunks" >&2
  return 1
}

wait_for_recovery_anchor_capture() {
  local request_id=$1
  local pid=$2
  local deadline=$(( $(date +%s) + 180 ))
  while (( $(date +%s) < deadline )); do
    if rg -q "request recovery anchor captured request_id=${request_id}( |$)" "$SERVER_LOG"; then
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  echo "request $request_id did not capture its request-local recovery anchor" >&2
  return 1
}

decode_progress_count() {
  local request_id=$1
  rg -c "decode progress request_id=${request_id}( |$)" "$SERVER_LOG" || true
}

wait_for_decode_progress() {
  local request_id=$1
  local pid=$2
  local deadline=$(( $(date +%s) + 120 ))
  while (( $(date +%s) < deadline )); do
    if (( $(decode_progress_count "$request_id") >= 1 )); then
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  echo "request $request_id emitted no decode progress before exit/deadline" >&2
  return 1
}

wait_for_cancellation_delta() {
  local baseline=$1
  local deadline=$(( $(date +%s) + 30 ))
  local current
  while (( $(date +%s) < deadline )); do
    current=$(metric_value hf2q_sse_cancellations)
    if (( current == baseline + 1 )); then
      printf '%s\n' "$current"
      return 0
    fi
    if (( current > baseline + 1 )); then
      echo "cancellation counter advanced by more than one: $baseline -> $current" >&2
      return 1
    fi
    sleep 0.2
  done
  echo "cancellation counter did not advance exactly once from $baseline" >&2
  return 1
}

cd "$ROOT_DIR"
ready_before=$(curl --fail-with-body --silent --show-error "$BASE_URL/readyz" | jq -r '.ready')
[[ "$ready_before" == "true" ]]

# Prime a separate long-output conversation first. The measured peer is the
# cached replay, not the cold turn: cold work belongs to DeepSeek's bounded
# cohort and is deliberately drained before unrelated warm admission.
jq -n --arg model "$MODEL" --arg run_id "peer-$$-$(date +%s)" '{
  model: $model,
  messages: [{role: "user", content: ("Peer run " + $run_id + ". Write at least 400 tokens explaining why bounded GPU transactions preserve multi-request liveness. End with PEER_DONE.")}],
  temperature: 0,
  max_tokens: 512,
  stream: false
}' >"$peer_request"
post_json "$peer_request" "$peer_warm_response"
peer_warm_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$peer_warm_response")
(( peer_warm_cached == 0 )) || {
  echo "peer warmup unexpectedly reused cache ($peer_warm_cached tokens)" >&2
  exit 1
}
jq '.stream = true | .stream_options = {include_usage: true}' "$peer_request" >"$peer_stream_request"

# Establish and immediately replay the retained prefix used by the overlap
# continuation. The replay is load-bearing: it proves the recovery anchor is
# selectable before the peer is launched.
build_base_request "$request_a" "overlap-$$-$(date +%s)" "$EXPECTED_PATH"
post_json "$request_a" "$response_a"
assert_tool_path "$response_a" "$EXPECTED_PATH"
base_a_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$response_a")
(( base_a_cached == 0 )) || {
  echo "overlap base unexpectedly reused cache ($base_a_cached tokens)" >&2
  exit 1
}
post_json "$request_a" "$response_a_cached"
assert_tool_path "$response_a_cached" "$EXPECTED_PATH"
base_a_replay_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$response_a_cached")
(( base_a_replay_cached > 0 )) || {
  echo "overlap base replay did not prove a selectable retained prefix" >&2
  exit 1
}
build_continuation "$request_a" "$response_a_cached" "$OVERLAP_TOOL_RESULT_PATH" "$continuation_a" false

# Start the cached semantic SSE peer first, then submit the cached continuation.
overlap_log_start=$(wc -l <"$SERVER_LOG" | tr -d ' ')
curl --fail-with-body --silent --show-error --no-buffer \
  --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
  --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary "@$peer_stream_request" \
  "$BASE_URL/v1/chat/completions" >"$peer_sse" 2>"$peer_err" &
peer_pid=$!
wait_for_semantic_content "$peer_sse" "$peer_pid"
peer_bytes_before=$(wc -c <"$peer_sse" | tr -d ' ')
overlap_started_ms=$(epoch_ms)
post_json "$continuation_a" "$continuation_a_response"
overlap_response_ms=$(( $(epoch_ms) - overlap_started_ms ))
continuation_content=$(jq -r '.choices[0].message.content // empty' "$continuation_a_response")
[[ "$continuation_content" == "DEEPSEEK_CACHED_SUFFIX_OK" ]] || {
  echo "cached overlap continuation returned unexpected content" >&2
  jq '.choices[0]' "$continuation_a_response" >&2
  exit 1
}
overlap_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$continuation_a_response")
(( overlap_cached > 0 )) || {
  echo "cached overlap continuation did not reuse a retained prefix" >&2
  exit 1
}
wait "$peer_pid"
peer_pid=
peer_bytes_after=$(wc -c <"$peer_sse" | tr -d ' ')
(( peer_bytes_after > peer_bytes_before )) || {
  echo "SSE peer made no byte progress during cached suffix prefill" >&2
  exit 1
}
[[ $(rg -c '^data: \[DONE\]$' "$peer_sse" || true) == 1 ]] || {
  echo "SSE peer did not end with exactly one [DONE]" >&2
  exit 1
}

peer_request_id=$(wait_for_request_id "$overlap_log_start" slot-stream-yielding-cached-prefill)
overlap_request_id=$(wait_for_request_id "$overlap_log_start" slot-unary-yielding-cached-prefill)
overlap_chunks=$(prefill_progress_count "$overlap_request_id")
(( overlap_chunks >= 3 )) || {
  echo "cached overlap suffix used only $overlap_chunks transactions" >&2
  exit 1
}
first_overlap_line=$(rg -n "prefill progress request_id=${overlap_request_id}( |$)" "$SERVER_LOG" | head -1 | cut -d: -f1)
last_overlap_line=$(rg -n "prefill progress request_id=${overlap_request_id}( |$)" "$SERVER_LOG" | tail -1 | cut -d: -f1)
peer_decode_between=$(sed -n "$((first_overlap_line + 1)),$((last_overlap_line - 1))p" "$SERVER_LOG" |
  rg -c "decode progress request_id=${peer_request_id}( |$)" || true)
(( peer_decode_between >= 1 )) || {
  echo "SSE peer made no logged decode progress between cached suffix transactions" >&2
  exit 1
}

# Establish a distinct prefix, then cancel a much longer cached continuation.
build_base_request "$request_b" "cancel-$$-$(date +%s)" "$EXPECTED_PATH"
post_json "$request_b" "$response_b"
assert_tool_path "$response_b" "$EXPECTED_PATH"
post_json "$request_b" "$control_response"
assert_tool_path "$control_response" "$EXPECTED_PATH"
base_b_replay_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$control_response")
(( base_b_replay_cached > 0 )) || {
  echo "cancellation base replay did not prove a selectable retained prefix" >&2
  exit 1
}
build_continuation "$request_b" "$control_response" "$CANCEL_TOOL_RESULT_PATH" "$cancel_request" true
cancel_counter_before=$(metric_value hf2q_sse_cancellations)
cancel_log_start=$(wc -l <"$SERVER_LOG" | tr -d ' ')
curl --fail-with-body --silent --show-error --no-buffer \
  --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
  --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary "@$cancel_request" \
  "$BASE_URL/v1/chat/completions" >"$cancel_sse" 2>"$cancel_err" &
cancel_pid=$!
cancel_request_id=$(wait_for_request_id "$cancel_log_start" slot-stream-yielding-cached-prefill)
chunks_at_disconnect=$(wait_for_prefill_chunks "$cancel_request_id" "$PREFILL_CHUNKS_BEFORE_CANCEL" "$cancel_pid")
(( chunks_at_disconnect == PREFILL_CHUNKS_BEFORE_CANCEL )) || {
  echo "cancellation boundary was not exact: expected $PREFILL_CHUNKS_BEFORE_CANCEL, observed $chunks_at_disconnect" >&2
  exit 1
}
kill -TERM "$cancel_pid"
wait "$cancel_pid" 2>/dev/null || true
cancel_pid=
cancel_counter_after=$(wait_for_cancellation_delta "$cancel_counter_before")
sleep "$CANCEL_SETTLE_SECONDS"
chunks_after_settle=$(prefill_progress_count "$cancel_request_id")
(( chunks_after_settle <= chunks_at_disconnect + 1 )) || {
  echo "cancelled suffix advanced too far: $chunks_at_disconnect -> $chunks_after_settle chunks" >&2
  exit 1
}
sleep "$CANCEL_STABILITY_SECONDS"
chunks_after_stability=$(prefill_progress_count "$cancel_request_id")
(( chunks_after_stability == chunks_after_settle )) || {
  echo "cancelled suffix kept advancing after stability window: $chunks_after_settle -> $chunks_after_stability" >&2
  exit 1
}
cancel_done_count=$(rg -c '^data: \[DONE\]$' "$cancel_sse" || true)
cancel_done_count=${cancel_done_count:-0}
[[ "$cancel_done_count" == 0 ]] || {
  echo "cancelled stream emitted [DONE]" >&2
  exit 1
}

# The original retained prefix must remain usable after cancellation.
post_json "$request_b" "$control_response"
assert_tool_path "$control_response" "$EXPECTED_PATH"
control_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$control_response")
(( control_cached > 0 )) || {
  echo "post-cancellation control did not reuse the original prefix" >&2
  exit 1
}

# Repeat the same cancelled continuation, but this time wait until the new
# request-local recovery anchor has actually been captured. Cancelling here
# proves that the candidate is discarded and the pre-request committed anchor
# is restored; an early three-chunk cancellation cannot exercise that edge.
late_cancel_counter_before=$(metric_value hf2q_sse_cancellations)
late_cancel_log_start=$(wc -l <"$SERVER_LOG" | tr -d ' ')
curl --fail-with-body --silent --show-error --no-buffer \
  --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
  --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary "@$cancel_request" \
  "$BASE_URL/v1/chat/completions" >"$late_cancel_sse" 2>"$late_cancel_err" &
late_cancel_pid=$!
late_cancel_request_id=$(wait_for_request_id "$late_cancel_log_start" slot-stream-yielding-cached-prefill)
wait_for_recovery_anchor_capture "$late_cancel_request_id" "$late_cancel_pid"
late_chunks_at_disconnect=$(prefill_progress_count "$late_cancel_request_id")
kill -TERM "$late_cancel_pid" 2>/dev/null || true
wait "$late_cancel_pid" 2>/dev/null || true
late_cancel_pid=
late_cancel_counter_after=$(wait_for_cancellation_delta "$late_cancel_counter_before")
sleep "$CANCEL_SETTLE_SECONDS"
late_chunks_after_settle=$(prefill_progress_count "$late_cancel_request_id")
(( late_chunks_after_settle <= late_chunks_at_disconnect + 1 )) || {
  echo "late-cancelled suffix advanced too far: $late_chunks_at_disconnect -> $late_chunks_after_settle chunks" >&2
  exit 1
}
sleep "$CANCEL_STABILITY_SECONDS"
late_chunks_after_stability=$(prefill_progress_count "$late_cancel_request_id")
(( late_chunks_after_stability == late_chunks_after_settle )) || {
  echo "late-cancelled suffix kept advancing after stability window: $late_chunks_after_settle -> $late_chunks_after_stability" >&2
  exit 1
}
late_cancel_done_count=$(rg -c '^data: \[DONE\]$' "$late_cancel_sse" || true)
late_cancel_done_count=${late_cancel_done_count:-0}
[[ "$late_cancel_done_count" == 0 ]] || {
  echo "late-cancelled stream emitted [DONE]" >&2
  exit 1
}

post_json "$request_b" "$control_response"
assert_tool_path "$control_response" "$EXPECTED_PATH"
late_control_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$control_response")
(( late_control_cached > 0 )) || {
  echo "post-late-cancellation control did not reuse the original prefix" >&2
  exit 1
}

# Complete one cached continuation so its prompt-boundary checkpoint becomes
# the committed pre-request anchor, then cancel a long following turn after
# decode has made observable progress. Reissuing that cancelled prompt with a
# one-token budget must reuse the committed checkpoint instead of retaining
# the cancelled decode tail or going cold.
jq '.stream = false | del(.stream_options)' "$cancel_request" >"$decode_seed_request"
post_json "$decode_seed_request" "$decode_seed_response"
decode_seed_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$decode_seed_response")
(( decode_seed_cached > 0 )) || {
  echo "decode-cancellation seed did not reuse the original retained prefix" >&2
  exit 1
}
jq -n --slurpfile base "$decode_seed_request" --slurpfile prior "$decode_seed_response" '
  $base[0]
  | .messages += [
      {role: "assistant", content: $prior[0].choices[0].message.content},
      {role: "user", content: "Write at least 600 tokens explaining cache-coherent cancellation recovery. Do not call tools. End with DECODE_CANCEL_DONE."}
    ]
  | del(.tools, .tool_choice)
  | .max_tokens = 768
  | .stream = true
  | .stream_options = {include_usage: true}
' >"$decode_cancel_request"

decode_cancel_counter_before=$(metric_value hf2q_sse_cancellations)
decode_cancel_log_start=$(wc -l <"$SERVER_LOG" | tr -d ' ')
curl --fail-with-body --silent --show-error --no-buffer \
  --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
  --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary "@$decode_cancel_request" \
  "$BASE_URL/v1/chat/completions" >"$decode_cancel_sse" 2>"$decode_cancel_err" &
decode_cancel_pid=$!
decode_cancel_request_id=$(wait_for_request_id "$decode_cancel_log_start" slot-stream-yielding-cached-prefill)
wait_for_decode_progress "$decode_cancel_request_id" "$decode_cancel_pid"
decode_events_at_disconnect=$(decode_progress_count "$decode_cancel_request_id")
kill -TERM "$decode_cancel_pid" 2>/dev/null || true
wait "$decode_cancel_pid" 2>/dev/null || true
decode_cancel_pid=
decode_cancel_counter_after=$(wait_for_cancellation_delta "$decode_cancel_counter_before")
sleep "$CANCEL_SETTLE_SECONDS"
decode_events_after_settle=$(decode_progress_count "$decode_cancel_request_id")
(( decode_events_after_settle <= decode_events_at_disconnect + 1 )) || {
  echo "cancelled decode advanced too far: $decode_events_at_disconnect -> $decode_events_after_settle events" >&2
  exit 1
}
sleep "$CANCEL_STABILITY_SECONDS"
decode_events_after_stability=$(decode_progress_count "$decode_cancel_request_id")
(( decode_events_after_stability == decode_events_after_settle )) || {
  echo "cancelled decode kept advancing after stability window: $decode_events_after_settle -> $decode_events_after_stability" >&2
  exit 1
}
decode_cancel_done_count=$(rg -c '^data: \[DONE\]$' "$decode_cancel_sse" || true)
decode_cancel_done_count=${decode_cancel_done_count:-0}
[[ "$decode_cancel_done_count" == 0 ]] || {
  echo "decode-cancelled stream emitted [DONE]" >&2
  exit 1
}

jq '.stream = false | del(.stream_options) | .max_tokens = 1' \
  "$decode_cancel_request" >"$decode_recovery_request"
post_json "$decode_recovery_request" "$decode_recovery_response"
decode_recovery_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$decode_recovery_response")
(( decode_recovery_cached > 0 )) || {
  echo "post-decode-cancellation probe did not reuse the committed checkpoint" >&2
  exit 1
}
ready_after=$(curl --fail-with-body --silent --show-error "$BASE_URL/readyz" | jq -r '.ready')
[[ "$ready_after" == "true" ]]

server_log_inode_after=$(stat -f '%i' "$SERVER_LOG")
server_log_lines_after=$(wc -l <"$SERVER_LOG" | tr -d ' ')
[[ "$server_log_inode_after" == "$server_log_inode_before" ]]
(( server_log_lines_after >= server_log_lines_before ))
[[ $(sed -n "1,${server_log_lines_before}p" "$SERVER_LOG" | shasum -a 256 | awk '{print $1}') == "$server_log_prefix_sha256" ]]
sed -n "$((server_log_lines_before + 1)),${server_log_lines_after}p" "$SERVER_LOG" >"$OUT_DIR/server.delta.log"
if rg -i 'command.?buffer.*(error|timeout)|submissions.?ignored|engine_unhealthy|worker.*fatal|panic' "$OUT_DIR/server.delta.log"; then
  echo "fatal/unhealthy signature found in server log delta" >&2
  exit 1
fi

jq -n \
  --arg binary_path "$binary_path" --arg binary_sha256 "$binary_sha256" \
  --arg model "$MODEL" --argjson server_pid "$SERVER_PID" \
  --argjson overlap_request_id "$overlap_request_id" --argjson peer_request_id "$peer_request_id" \
  --argjson overlap_chunks "$overlap_chunks" --argjson peer_decode_between "$peer_decode_between" \
  --argjson overlap_cached "$overlap_cached" --argjson overlap_response_ms "$overlap_response_ms" \
  --argjson peer_bytes_before "$peer_bytes_before" --argjson peer_bytes_after "$peer_bytes_after" \
  --argjson cancel_request_id "$cancel_request_id" \
  --argjson chunks_at_disconnect "$chunks_at_disconnect" \
  --argjson chunks_after_settle "$chunks_after_settle" \
  --argjson chunks_after_stability "$chunks_after_stability" \
  --argjson cancel_counter_before "$cancel_counter_before" \
  --argjson cancel_counter_after "$cancel_counter_after" \
  --argjson control_cached "$control_cached" \
  --argjson late_cancel_request_id "$late_cancel_request_id" \
  --argjson late_chunks_at_disconnect "$late_chunks_at_disconnect" \
  --argjson late_chunks_after_settle "$late_chunks_after_settle" \
  --argjson late_chunks_after_stability "$late_chunks_after_stability" \
  --argjson late_cancel_counter_before "$late_cancel_counter_before" \
  --argjson late_cancel_counter_after "$late_cancel_counter_after" \
  --argjson late_control_cached "$late_control_cached" \
  --argjson decode_cancel_request_id "$decode_cancel_request_id" \
  --argjson decode_events_at_disconnect "$decode_events_at_disconnect" \
  --argjson decode_events_after_settle "$decode_events_after_settle" \
  --argjson decode_events_after_stability "$decode_events_after_stability" \
  --argjson decode_cancel_counter_before "$decode_cancel_counter_before" \
  --argjson decode_cancel_counter_after "$decode_cancel_counter_after" \
  --argjson decode_recovery_cached "$decode_recovery_cached" '{
    status: "pass",
    authority: "focused-hardware-probe",
    server: {pid: $server_pid, binary_path: $binary_path, binary_sha256: $binary_sha256, model: $model, max_slots: 4},
    overlap: {
      cached_request_id: $overlap_request_id,
      peer_request_id: $peer_request_id,
      cached_prefill_transactions: $overlap_chunks,
      peer_decode_progress_events_between_transactions: $peer_decode_between,
      cached_tokens: $overlap_cached,
      response_ms: $overlap_response_ms,
      peer_bytes_before: $peer_bytes_before,
      peer_bytes_after: $peer_bytes_after,
      peer_terminal_done_exactly_once: true
    },
    cancellation: {
      request_id: $cancel_request_id,
      chunks_at_disconnect: $chunks_at_disconnect,
      chunks_after_settle: $chunks_after_settle,
      chunks_after_stability: $chunks_after_stability,
      counter_before: $cancel_counter_before,
      counter_after: $cancel_counter_after,
      emitted_done: false,
      post_cancel_cached_tokens: $control_cached
    },
    cancellation_after_candidate_anchor: {
      request_id: $late_cancel_request_id,
      chunks_at_disconnect: $late_chunks_at_disconnect,
      chunks_after_settle: $late_chunks_after_settle,
      chunks_after_stability: $late_chunks_after_stability,
      counter_before: $late_cancel_counter_before,
      counter_after: $late_cancel_counter_after,
      emitted_done: false,
      post_cancel_cached_tokens: $late_control_cached
    },
    cancellation_during_decode: {
      request_id: $decode_cancel_request_id,
      progress_events_at_disconnect: $decode_events_at_disconnect,
      progress_events_after_settle: $decode_events_after_settle,
      progress_events_after_stability: $decode_events_after_stability,
      counter_before: $decode_cancel_counter_before,
      counter_after: $decode_cancel_counter_after,
      emitted_done: false,
      post_cancel_cached_tokens: $decode_recovery_cached
    },
    ready_before: true,
    ready_after: true,
    fatal_log_signatures: 0
  }' >"$summary_tmp"
jq -e '
  .status == "pass"
  and .server.max_slots == 4
  and .overlap.cached_prefill_transactions >= 3
  and .overlap.peer_decode_progress_events_between_transactions >= 1
  and .overlap.peer_terminal_done_exactly_once == true
  and .cancellation.chunks_after_settle <= (.cancellation.chunks_at_disconnect + 1)
  and .cancellation.chunks_after_stability == .cancellation.chunks_after_settle
  and .cancellation.counter_after == (.cancellation.counter_before + 1)
  and .cancellation.emitted_done == false
  and .cancellation_after_candidate_anchor.chunks_after_settle <= (.cancellation_after_candidate_anchor.chunks_at_disconnect + 1)
  and .cancellation_after_candidate_anchor.chunks_after_stability == .cancellation_after_candidate_anchor.chunks_after_settle
  and .cancellation_after_candidate_anchor.counter_after == (.cancellation_after_candidate_anchor.counter_before + 1)
  and .cancellation_after_candidate_anchor.emitted_done == false
  and .cancellation_after_candidate_anchor.post_cancel_cached_tokens > 0
  and .cancellation_during_decode.progress_events_at_disconnect >= 1
  and .cancellation_during_decode.progress_events_after_settle <= (.cancellation_during_decode.progress_events_at_disconnect + 1)
  and .cancellation_during_decode.progress_events_after_stability == .cancellation_during_decode.progress_events_after_settle
  and .cancellation_during_decode.counter_after == (.cancellation_during_decode.counter_before + 1)
  and .cancellation_during_decode.emitted_done == false
  and .cancellation_during_decode.post_cancel_cached_tokens > 0
  and .ready_before == true and .ready_after == true
  and .fatal_log_signatures == 0
' "$summary_tmp" >/dev/null
mv "$summary_tmp" "$summary"
shasum -a 256 "$summary" >"$summary.sha256"
shasum -a 256 -c "$summary.sha256" >/dev/null
cat "$summary"
