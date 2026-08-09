#!/usr/bin/env bash
# Reproduce the exact public equivalent of the Qwen 3.6 incident: enqueue a
# 552-token SSE request, then enqueue the 87,972-token/347-tool SSE request
# before the short lane emits semantic content. The bounded driver must decode
# the short lane before every long-prefill transaction and complete the long
# prompt as exactly 42 * 2,048 + 1,956 tokens.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:18081}"
FIXTURE_JSON="${FIXTURE_JSON:-}"
SHORT_FIXTURE_JSON="${SHORT_FIXTURE_JSON:-}"
FIXTURE_MODEL="${FIXTURE_MODEL:-/opt/hf2q/models/qwen3.6/APEX-Q5_K_M.gguf}"
SERVER_LOG="${SERVER_LOG:?set SERVER_LOG to the regular-file server log}"
OUT_DIR="${OUT_DIR:-$(mktemp -d -t hf2q-qwen36-watchdog.XXXXXX)}"
SERVER_PID="${SERVER_PID:-}"
NO_PROGRESS_SECONDS="${NO_PROGRESS_SECONDS:-30}"
BINARY_PATH="${BINARY_PATH:-}"
BINARY_SHA256="${BINARY_SHA256:-unknown}"
MODEL_SHA256="${MODEL_SHA256:-unknown}"
REQUIRE_PROVENANCE="${REQUIRE_PROVENANCE:-1}"
MAX_SLOTS="${MAX_SLOTS:-4}"
LONG_RENDER_HEADSTART_SECONDS="${LONG_RENDER_HEADSTART_SECONDS:-0}"

LONG_FIXTURE_SHA256="6671a0c89b8d4935caa4b87bee08361c5b8727ec557e9edb05947ad90c94c13d"
LONG_RUNTIME_SHA256="ec53ffc6f71028484dbded593bcdbbbfa905b07a44051c222a44c25d8c9f39e2"
SHORT_FIXTURE_SHA256="7aeddea35e6363c698ea0bcb4934b9f2cf1e0c48fb2045fa9db3272461e54004"
TOOLS_SHA256="586e09658c8d4d69b1ad451c8218199e405eeb72de4e550741730e83ed653766"
TEMPLATE_SHA256="e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
LONG_PATTERN='Qwen35 bounded prefill chunk complete.*prompt_tokens=87972'
LONG_RECEIVED_PATTERN='chat completion request received.*messages=2 tools=347'
SHORT_SUBMIT_PATTERN='streaming request submitted to inference channel.*prompt_tokens=552'
LONG_SUBMIT_PATTERN='streaming request submitted to inference channel.*prompt_tokens=87972'
SHORT_SEMANTIC_PATTERN='Qwen35 semantic fragment ready.*prompt_tokens=552'

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

for command in curl jq rg sed awk date shasum sample wc ps cut lsof sort seq caffeinate pmset stat find; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ -f "$SERVER_LOG" ]] || { echo "server log not found: $SERVER_LOG" >&2; exit 2; }
qwen36_require_empty_receipt_dir "$OUT_DIR"
sha256_file() {
  shasum -a 256 "$1" | awk '{ print $1 }'
}
[[ "$REQUIRE_PROVENANCE" == 1 ]] || {
  echo "Qwen overlap gate never emits release authority without exact provenance" >&2
  exit 2
}
[[ -n "$SERVER_PID" && -n "$BINARY_PATH" && -x "$BINARY_PATH" ]] || {
  echo "release gate requires SERVER_PID and executable BINARY_PATH" >&2
  exit 2
}
[[ "$BINARY_SHA256" =~ ^[0-9a-f]{64}$ && "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "release gate requires exact BINARY_SHA256 and MODEL_SHA256" >&2
  exit 2
}
[[ "$(sha256_file "$BINARY_PATH")" == "$BINARY_SHA256" ]] || {
  echo "BINARY_SHA256 does not match BINARY_PATH" >&2
  exit 2
}
[[ "$(sha256_file "$FIXTURE_MODEL")" == "$MODEL_SHA256" ]] || {
  echo "MODEL_SHA256 does not match FIXTURE_MODEL" >&2
  exit 2
}
[[ "$MAX_SLOTS" == 4 ]] || {
  echo "Qwen overlap release gate requires MAX_SLOTS=4" >&2
  exit 2
}
qwen36_bind_server_process "$BASE_URL" "$SERVER_PID" "$BINARY_PATH" \
  "$FIXTURE_MODEL" "$MAX_SLOTS"

cleanup() {
  for pid in "${short_pid:-}" "${long_pid:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  qwen36_stop_power_guard
}
trap cleanup EXIT

qwen36_start_power_guard "${SERVER_PID:-$$}" "$OUT_DIR/caffeinate.log"

count_log() {
  local count
  count=$(rg -c "$1" "$SERVER_LOG" 2>/dev/null || true)
  printf '%s\n' "${count:-0}"
}

has_short_semantic_content() {
  sed -n 's/^data: //p' "$1" 2>/dev/null \
    | awk '$0 != "[DONE]" && $0 != ""' \
    | jq -se 'any(.[]; ((.choices[0].delta.content // "") | length) > 0)' \
      >/dev/null 2>&1
}

capture_no_progress() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    sample "$SERVER_PID" 5 2 -file "$OUT_DIR/no-progress.sample" >/dev/null 2>&1 || true
  fi
  curl --silent --show-error --max-time 2 "$BASE_URL/health" \
    >"$OUT_DIR/no-progress-health.txt" 2>&1 || true
  curl --silent --show-error --max-time 2 "$BASE_URL/readyz" \
    >"$OUT_DIR/no-progress-readyz.txt" 2>&1 || true
}

if [[ -z "$FIXTURE_JSON" || -z "$SHORT_FIXTURE_JSON" ]]; then
  repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  FIXTURE_JSON="$OUT_DIR/generated-long-request.json"
  SHORT_FIXTURE_JSON="$OUT_DIR/generated-short-request.json"
  (
    cd "$repo_root"
    HF2Q_QWEN36_WATCHDOG_FIXTURE_MODEL="$FIXTURE_MODEL" \
      HF2Q_QWEN36_WATCHDOG_FIXTURE_OUTPUT="$FIXTURE_JSON" \
      HF2Q_QWEN36_WATCHDOG_SHORT_FIXTURE_OUTPUT="$SHORT_FIXTURE_JSON" \
      cargo test --locked --bin hf2q \
        public_347_tool_fixture_renders_to_exact_87972_tokens -- \
        --ignored --test-threads=1
  ) >"$OUT_DIR/fixture-generation.log" 2>&1
fi

[[ -f "$FIXTURE_JSON" ]] || { echo "long fixture not found: $FIXTURE_JSON" >&2; exit 2; }
[[ -f "$SHORT_FIXTURE_JSON" ]] || { echo "short fixture not found: $SHORT_FIXTURE_JSON" >&2; exit 2; }
[[ "$(sha256_file "$FIXTURE_JSON")" == "$LONG_FIXTURE_SHA256" ]] || {
  echo "long fixture hash does not match the canonical public request" >&2
  exit 2
}
[[ "$(sha256_file "$SHORT_FIXTURE_JSON")" == "$SHORT_FIXTURE_SHA256" ]] || {
  echo "short fixture hash does not match the canonical public request" >&2
  exit 2
}

long_request="$OUT_DIR/long-request.json"
short_request="$OUT_DIR/short-request.json"
jq '.max_tokens = 64' "$FIXTURE_JSON" >"$long_request"
cp "$SHORT_FIXTURE_JSON" "$short_request"
[[ "$(sha256_file "$long_request")" == "$LONG_RUNTIME_SHA256" ]] || {
  echo "runtime long-request hash drifted" >&2
  exit 2
}
[[ "$(jq '.tools | length' "$long_request")" == 347 ]] || {
  echo "watchdog fixture must contain exactly 347 tools" >&2
  exit 2
}
[[ "$(jq -r '.stream' "$long_request")" == true ]] || exit 2
[[ "$(jq -r '.stream' "$short_request")" == true ]] || exit 2

baseline_chunks=$(count_log "$LONG_PATTERN")
baseline_long_received=$(count_log "$LONG_RECEIVED_PATTERN")
baseline_short_submits=$(count_log "$SHORT_SUBMIT_PATTERN")
baseline_long_submits=$(count_log "$LONG_SUBMIT_PATTERN")
qwen36_write_log_baseline "$SERVER_LOG" "$OUT_DIR/server-log-baseline.json"
baseline_log_lines=$(jq -er '.line_count' "$OUT_DIR/server-log-baseline.json")
short_metrics="$OUT_DIR/short-curl.metrics"
short_sse="$OUT_DIR/short-response.sse"
long_metrics="$OUT_DIR/long-curl.metrics"
long_sse="$OUT_DIR/long-response.sse"

# Give the expensive long request's render/tokenize phase a head start, then
# let the small request reach the worker first. The load-bearing order is
# short worker enqueue < long worker enqueue < short semantic output; this is
# the original poisoned-peer topology without relying on handler CPU timing.
curl --fail-with-body --silent --show-error --no-buffer \
  --connect-timeout 5 --max-time 300 \
  -H 'Content-Type: application/json' --data-binary "@$long_request" \
  -o "$long_sse" -w 'http_code=%{http_code}\ntotal_seconds=%{time_total}\n' \
  "$BASE_URL/v1/chat/completions" >"$long_metrics" 2>"$OUT_DIR/long-curl.err" &
long_pid=$!

received_deadline=$(( $(date +%s) + 10 ))
while (( $(count_log "$LONG_RECEIVED_PATTERN") <= baseline_long_received )); do
  kill -0 "$long_pid" 2>/dev/null || {
    wait "$long_pid" || true
    echo "long request ended before the server acknowledged receipt" >&2
    exit 1
  }
  (( $(date +%s) < received_deadline )) || {
    echo "long request was not received within 10 seconds" >&2
    exit 1
  }
  sleep 0.01
done

# Rendering/tokenizing 347 schemas is intentionally CPU-expensive. Give that
# handler a bounded head start so the short request still enters the worker
# first, while the long request becomes worker-visible before the short 552-row
# GPU transaction can emit a semantic token. The log-order assertions below,
# not this timing hint, are the acceptance authority.
sleep "$LONG_RENDER_HEADSTART_SECONDS"

curl --fail-with-body --silent --show-error --no-buffer \
  --connect-timeout 5 --max-time 300 \
  -H 'Content-Type: application/json' --data-binary "@$short_request" \
  -o "$short_sse" -w 'http_code=%{http_code}\ntotal_seconds=%{time_total}\n' \
  "$BASE_URL/v1/chat/completions" >"$short_metrics" 2>"$OUT_DIR/short-curl.err" &
short_pid=$!

enqueue_deadline=$(( $(date +%s) + 10 ))
while (( $(count_log "$SHORT_SUBMIT_PATTERN") <= baseline_short_submits )); do
  if (( $(count_log "$LONG_SUBMIT_PATTERN") > baseline_long_submits )); then
    echo "long request reached the worker before the short request" >&2
    exit 1
  fi
  kill -0 "$short_pid" 2>/dev/null || {
    wait "$short_pid" || true
    echo "short request ended before the server acknowledged its enqueue" >&2
    exit 1
  }
  (( $(date +%s) < enqueue_deadline )) || {
    echo "short request was not enqueued within 10 seconds" >&2
    exit 1
  }
  sleep 0.01
done

long_submit_deadline=$(( $(date +%s) + 10 ))
while (( $(count_log "$LONG_SUBMIT_PATTERN") <= baseline_long_submits )); do
  if has_short_semantic_content "$short_sse"; then
    echo "short semantic content arrived before the long request was submitted" >&2
    exit 1
  fi
  kill -0 "$long_pid" 2>/dev/null || {
    wait "$long_pid" || true
    echo "long request ended before submission to the inference channel" >&2
    exit 1
  }
  (( $(date +%s) < long_submit_deadline )) || {
    echo "long request was not submitted within 10 seconds" >&2
    exit 1
  }
  sleep 0.02
done

# This is the live form of the scheduler contract: the already-enqueued short
# lane must publish semantic content before the first 2,048-token long chunk
# completes. Poll quickly enough that one-second chunk boundaries cannot hide
# an ordering failure.
semantic_deadline=$(( $(date +%s) + 15 ))
while ! has_short_semantic_content "$short_sse"; do
  if (( $(count_log "$LONG_PATTERN") > baseline_chunks )); then
    echo "long prefill completed a chunk before the short lane made semantic progress" >&2
    exit 1
  fi
  (( $(date +%s) < semantic_deadline )) || {
    echo "short lane made no semantic progress within 15 seconds" >&2
    exit 1
  }
  sleep 0.02
done
short_semantic_before_long_chunk=true

last_chunks=$baseline_chunks
last_short_bytes=$(qwen36_sse_data_bytes "$short_sse")
last_long_bytes=$(qwen36_sse_data_bytes "$long_sse")
last_progress=$(date +%s)
while kill -0 "$short_pid" 2>/dev/null || kill -0 "$long_pid" 2>/dev/null; do
  sleep 1
  qwen36_assert_power_guard
  chunks=$(count_log "$LONG_PATTERN")
  short_bytes=$(qwen36_sse_data_bytes "$short_sse")
  long_bytes=$(qwen36_sse_data_bytes "$long_sse")
  if (( chunks > last_chunks || short_bytes > last_short_bytes || long_bytes > last_long_bytes )); then
    last_chunks=$chunks
    last_short_bytes=$short_bytes
    last_long_bytes=$long_bytes
    last_progress=$(date +%s)
  fi
  if (( $(date +%s) - last_progress > NO_PROGRESS_SECONDS )); then
    capture_no_progress
    echo "Qwen overlap gate made no bounded-prefill or SSE progress for ${NO_PROGRESS_SECONDS}s" >&2
    exit 1
  fi
done
wait "$short_pid"
short_pid=
wait "$long_pid"
long_pid=
qwen36_assert_power_guard

grep -qx 'http_code=200' "$short_metrics"
grep -qx 'http_code=200' "$long_metrics"
new_log="$OUT_DIR/server-log-delta.log"
qwen36_extract_append_only_log_delta "$SERVER_LOG" \
  "$OUT_DIR/server-log-baseline.json" "$new_log"
qwen36_reject_fatal_log "$new_log"
short_shape_count=$(rg -c 'chat completion request received.*messages=3 tools=0' "$new_log" || true)
long_shape_count=$(rg -c 'chat completion request received.*messages=2 tools=347' "$new_log" || true)
[[ "$short_shape_count" == 1 ]] || {
  echo "expected exactly one pinned 3-message/0-tool short request, observed $short_shape_count" >&2
  exit 1
}
[[ "$long_shape_count" == 1 ]] || {
  echo "expected exactly one pinned 2-message/347-tool long request, observed $long_shape_count" >&2
  exit 1
}
long_submit_line=$(rg -n -m1 "$LONG_SUBMIT_PATTERN" "$new_log" | cut -d: -f1)
short_submit_line=$(rg -n -m1 "$SHORT_SUBMIT_PATTERN" "$new_log" | cut -d: -f1)
short_semantic_line=$(rg -n -m1 "$SHORT_SEMANTIC_PATTERN" "$new_log" | cut -d: -f1)
first_long_chunk_line=$(rg -n -m1 "$LONG_PATTERN" "$new_log" | cut -d: -f1)
[[ -n "$short_submit_line" && -n "$long_submit_line" && -n "$short_semantic_line" && -n "$first_long_chunk_line" ]] || {
  echo "missing ordered enqueue/semantic/chunk evidence in the server log" >&2
  exit 1
}
(( short_submit_line < long_submit_line && long_submit_line < short_semantic_line && short_semantic_line < first_long_chunk_line )) || {
  echo "required order is short submit < long submit < short semantic < first long chunk; observed $short_submit_line/$long_submit_line/$short_semantic_line/$first_long_chunk_line" >&2
  exit 1
}

chunk_lines="$OUT_DIR/long-chunks.log"
rg "$LONG_PATTERN" "$SERVER_LOG" | sed -n "$((baseline_chunks + 1)),\$p" >"$chunk_lines"
[[ "$(wc -l <"$chunk_lines" | tr -d ' ')" == 43 ]] || {
  echo "expected exactly 43 request-scoped long chunks" >&2
  exit 1
}
qwen36_validate_chunk_lines "$chunk_lines"

qwen36_extract_and_validate_sse short "$short_sse" "$OUT_DIR/short-events.jsonl"
qwen36_extract_and_validate_sse long "$long_sse" "$OUT_DIR/long-events.jsonl"

short_content=$(jq -j '.choices[0].delta.content // empty' "$OUT_DIR/short-events.jsonl")
short_finish_count=$(jq -r '.choices[0].finish_reason // empty' "$OUT_DIR/short-events.jsonl" | awk '$0 == "stop" { count++ } END { print count + 0 }')
short_tool_count=$(jq -s '[.[] | .choices[0].delta.tool_calls[]?] | length' "$OUT_DIR/short-events.jsonl")
[[ "$short_content" == OK ]] || { echo "short lane content was not exact OK: $short_content" >&2; exit 1; }
[[ "$short_finish_count" == 1 && "$short_tool_count" == 0 ]] || {
  echo "short lane must have one stop finish and no tool calls" >&2
  exit 1
}
qwen36_validate_short_events "$OUT_DIR/short-events.jsonl"
tool_name=$(jq -j '.choices[0].delta.tool_calls[]?.function.name // empty' "$OUT_DIR/long-events.jsonl")
tool_args=$(jq -j '.choices[0].delta.tool_calls[]?.function.arguments // empty' "$OUT_DIR/long-events.jsonl")
finish_count=$(jq -r '.choices[0].finish_reason // empty' "$OUT_DIR/long-events.jsonl" | awk '$0 == "tool_calls" { count++ } END { print count + 0 }')
[[ "$tool_name" == fixture_tool_346 ]] || { echo "unexpected tool name: $tool_name" >&2; exit 1; }
[[ "$tool_args" == '{"path":"src/serve/api/engine.rs"}' ]] || {
  echo "unexpected tool arguments: $tool_args" >&2
  exit 1
}
[[ "$finish_count" == 1 ]] || { echo "expected one tool_calls finish, observed $finish_count" >&2; exit 1; }
qwen36_validate_long_events "$OUT_DIR/long-events.jsonl"

# Prove the worker remains ready and the same four-slot process can serve a
# fresh exact control after the cumulative long-prefill transaction stream.
curl --fail-with-body --silent --show-error --no-buffer \
  --connect-timeout 5 --max-time 60 \
  -H 'Content-Type: application/json' --data-binary "@$short_request" \
  -o "$OUT_DIR/post-response.sse" \
  -w 'http_code=%{http_code}\ntotal_seconds=%{time_total}\n' \
  "$BASE_URL/v1/chat/completions" >"$OUT_DIR/post-curl.metrics" \
  2>"$OUT_DIR/post-curl.err"
grep -qx 'http_code=200' "$OUT_DIR/post-curl.metrics"
qwen36_extract_and_validate_sse post "$OUT_DIR/post-response.sse" \
  "$OUT_DIR/post-events.jsonl"
qwen36_validate_short_events "$OUT_DIR/post-events.jsonl"
post_content=$(jq -j '.choices[0].delta.content // empty' "$OUT_DIR/post-events.jsonl")
[[ "$post_content" == OK ]] || {
  echo "post-overlap control did not return exact OK" >&2
  exit 1
}
ready_code=$(curl --silent --show-error --max-time 3 -o "$OUT_DIR/readyz.txt" \
  -w '%{http_code}' "$BASE_URL/readyz")
[[ "$ready_code" == 200 ]] || {
  echo "readyz failed after the cumulative overlap gate" >&2
  exit 1
}
qwen36_assert_power_guard

# Rebind the final log delta after the post-workload control. The workload-only
# counters above still prove exactly one incident-shaped pair; this final view
# proves exactly one additional control and rejects a delayed worker failure.
qwen36_extract_append_only_log_delta "$SERVER_LOG" \
  "$OUT_DIR/server-log-baseline.json" "$new_log"
final_short_shape_count=$(rg -c 'chat completion request received.*messages=3 tools=0' "$new_log" || true)
final_long_shape_count=$(rg -c 'chat completion request received.*messages=2 tools=347' "$new_log" || true)
[[ "$final_short_shape_count" == 2 && "$final_long_shape_count" == 1 ]] || {
  echo "unexpected final request topology: short=$final_short_shape_count long=$final_long_shape_count" >&2
  exit 1
}
qwen36_reject_fatal_log "$new_log"

short_sse_sha256=$(sha256_file "$short_sse")
long_sse_sha256=$(sha256_file "$long_sse")
post_sse_sha256=$(sha256_file "$OUT_DIR/post-response.sse")
server_log_delta_sha256=$(sha256_file "$new_log")
chunk_log_sha256=$(sha256_file "$chunk_lines")

summary="$OUT_DIR/long-watchdog-summary.json"
jq -n \
  --arg status pass \
  --arg out_dir "$OUT_DIR" \
  --arg long_fixture_sha256 "$LONG_FIXTURE_SHA256" \
  --arg long_runtime_sha256 "$LONG_RUNTIME_SHA256" \
  --arg short_fixture_sha256 "$SHORT_FIXTURE_SHA256" \
  --arg tools_sha256 "$TOOLS_SHA256" \
  --arg template_sha256 "$TEMPLATE_SHA256" \
  --arg binary_sha256 "$BINARY_SHA256" \
  --arg binary_path "$BINARY_PATH" \
  --arg model_sha256 "$MODEL_SHA256" \
  --arg short_sse_sha256 "$short_sse_sha256" \
  --arg long_sse_sha256 "$long_sse_sha256" \
  --arg post_sse_sha256 "$post_sse_sha256" \
  --arg server_log_delta_sha256 "$server_log_delta_sha256" \
  --arg chunk_log_sha256 "$chunk_log_sha256" \
  --argjson server_pid "${SERVER_PID:-0}" \
  --argjson max_slots "$MAX_SLOTS" \
  --arg long_render_headstart_seconds "$LONG_RENDER_HEADSTART_SECONDS" \
  --argjson baseline_log_lines "$baseline_log_lines" \
  --arg short_content "$short_content" \
  --argjson short_semantic_before_long_chunk "$short_semantic_before_long_chunk" \
  --argjson long_submit_line "$long_submit_line" \
  --argjson short_submit_line "$short_submit_line" \
  --argjson short_semantic_line "$short_semantic_line" \
  --argjson first_long_chunk_line "$first_long_chunk_line" \
  --argjson prompt_tokens 87972 \
  --argjson short_prompt_tokens 552 \
  --argjson short_shape_count "$short_shape_count" \
  --argjson long_shape_count "$long_shape_count" \
  --argjson final_short_shape_count "$final_short_shape_count" \
  --argjson final_long_shape_count "$final_long_shape_count" \
  --argjson ready_http "$ready_code" \
  --argjson power_event_baseline "$QWEN36_POWER_EVENT_BASELINE" \
  --argjson power_event_final "$QWEN36_POWER_EVENT_FINAL" \
  --argjson power_event_delta "$((QWEN36_POWER_EVENT_FINAL - QWEN36_POWER_EVENT_BASELINE))" \
  --arg post_content "$post_content" \
  --argjson tools 347 \
  --argjson chunks 43 \
  --argjson full_chunks 42 \
  --argjson tail_tokens 1956 \
  --arg tool_name "$tool_name" \
  --arg tool_args "$tool_args" \
  --argjson short_total_seconds "$(awk -F= '$1 == "total_seconds" { print $2 }' "$short_metrics")" \
  --argjson long_total_seconds "$(awk -F= '$1 == "total_seconds" { print $2 }' "$long_metrics")" \
  --argjson post_total_seconds "$(awk -F= '$1 == "total_seconds" { print $2 }' "$OUT_DIR/post-curl.metrics")" \
  '{status:$status,out_dir:$out_dir,binary_path:$binary_path,binary_sha256:$binary_sha256,model_sha256:$model_sha256,server_pid:$server_pid,max_slots:$max_slots,long_render_headstart_seconds:$long_render_headstart_seconds,baseline_log_lines:$baseline_log_lines,long_fixture_sha256:$long_fixture_sha256,long_runtime_sha256:$long_runtime_sha256,short_fixture_sha256:$short_fixture_sha256,tools_sha256:$tools_sha256,template_sha256:$template_sha256,short_sse_sha256:$short_sse_sha256,long_sse_sha256:$long_sse_sha256,post_sse_sha256:$post_sse_sha256,server_log_delta_sha256:$server_log_delta_sha256,chunk_log_sha256:$chunk_log_sha256,short_prompt_tokens:$short_prompt_tokens,prompt_tokens:$prompt_tokens,short_shape_count:$short_shape_count,long_shape_count:$long_shape_count,final_short_shape_count:$final_short_shape_count,final_long_shape_count:$final_long_shape_count,tools:$tools,chunks:$chunks,full_chunks:$full_chunks,tail_tokens:$tail_tokens,short_semantic_before_long_chunk:$short_semantic_before_long_chunk,short_submit_line:$short_submit_line,long_submit_line:$long_submit_line,short_semantic_line:$short_semantic_line,first_long_chunk_line:$first_long_chunk_line,short_content:$short_content,post_content:$post_content,ready_http:$ready_http,power_event_baseline:$power_event_baseline,power_event_final:$power_event_final,power_event_delta:$power_event_delta,tool_name:$tool_name,tool_args:$tool_args,short_total_seconds:$short_total_seconds,long_total_seconds:$long_total_seconds,post_total_seconds:$post_total_seconds}' \
  >"$summary.tmp"
jq -e '.status == "pass" and .max_slots == 4 and .short_semantic_before_long_chunk and .chunks == 43 and .short_shape_count == 1 and .long_shape_count == 1 and .final_short_shape_count == 2 and .final_long_shape_count == 1 and .post_content == "OK" and .ready_http == 200 and .power_event_delta == 0' "$summary.tmp" >/dev/null
mv "$summary.tmp" "$summary"
shasum -a 256 "$summary" >"$summary.sha256"
shasum -c "$summary.sha256" >/dev/null
cat "$summary"
