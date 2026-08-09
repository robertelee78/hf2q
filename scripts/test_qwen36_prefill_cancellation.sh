#!/usr/bin/env bash
# Abort the public 87,972-token Qwen SSE request after three committed prefill
# transactions. Cancellation must become visible at the next atomic boundary,
# leave readiness healthy, and permit a subsequent 552-token SSE request.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:18081}"
FIXTURE_JSON="${FIXTURE_JSON:?set FIXTURE_JSON to the canonical long fixture}"
SHORT_FIXTURE_JSON="${SHORT_FIXTURE_JSON:?set SHORT_FIXTURE_JSON to the canonical short fixture}"
SERVER_LOG="${SERVER_LOG:?set SERVER_LOG to the regular-file server log}"
OUT_DIR="${OUT_DIR:-$(mktemp -d -t hf2q-qwen36-cancel.XXXXXX)}"
SERVER_PID="${SERVER_PID:-}"
NO_PROGRESS_SECONDS="${NO_PROGRESS_SECONDS:-30}"
MAX_SLOTS="${MAX_SLOTS:?set MAX_SLOTS=1 so recovery proves reuse of the cancelled slot}"
BINARY_PATH="${BINARY_PATH:-}"
BINARY_SHA256="${BINARY_SHA256:-unknown}"
MODEL_PATH="${MODEL_PATH:-/opt/hf2q/models/qwen3.6/APEX-Q5_K_M.gguf}"
MODEL_SHA256="${MODEL_SHA256:-unknown}"
REQUIRE_PROVENANCE="${REQUIRE_PROVENANCE:-1}"

LONG_FIXTURE_SHA256="6671a0c89b8d4935caa4b87bee08361c5b8727ec557e9edb05947ad90c94c13d"
LONG_RUNTIME_SHA256="ec53ffc6f71028484dbded593bcdbbbfa905b07a44051c222a44c25d8c9f39e2"
SHORT_FIXTURE_SHA256="7aeddea35e6363c698ea0bcb4934b9f2cf1e0c48fb2045fa9db3272461e54004"
TOOLS_SHA256="586e09658c8d4d69b1ad451c8218199e405eeb72de4e550741730e83ed653766"
TEMPLATE_SHA256="e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
LONG_PATTERN='Qwen35 bounded prefill chunk complete.*prompt_tokens=87972'

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

for command in curl jq rg sed awk date shasum sample wc ps lsof sort seq caffeinate pmset stat find; do
  command -v "$command" >/dev/null || { echo "missing required command: $command" >&2; exit 2; }
done
[[ -f "$FIXTURE_JSON" && -f "$SHORT_FIXTURE_JSON" && -f "$SERVER_LOG" ]] || exit 2
qwen36_require_empty_receipt_dir "$OUT_DIR"
[[ "$MAX_SLOTS" == 1 ]] || {
  echo "cancellation gate requires MAX_SLOTS=1, got $MAX_SLOTS" >&2
  exit 2
}
sha256_file() { shasum -a 256 "$1" | awk '{ print $1 }'; }
[[ "$REQUIRE_PROVENANCE" == 1 ]] || {
  echo "Qwen cancellation gate never emits release authority without exact provenance" >&2
  exit 2
}
[[ -n "$SERVER_PID" && -n "$BINARY_PATH" && -x "$BINARY_PATH" && -f "$MODEL_PATH" ]] || {
  echo "release gate requires SERVER_PID, BINARY_PATH, and MODEL_PATH" >&2
  exit 2
}
[[ "$BINARY_SHA256" =~ ^[0-9a-f]{64}$ && "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "release gate requires exact binary and model SHA-256 values" >&2
  exit 2
}
[[ "$(sha256_file "$BINARY_PATH")" == "$BINARY_SHA256" ]] || exit 2
[[ "$(sha256_file "$MODEL_PATH")" == "$MODEL_SHA256" ]] || exit 2
qwen36_bind_server_process "$BASE_URL" "$SERVER_PID" "$BINARY_PATH" \
  "$MODEL_PATH" "$MAX_SLOTS"
count_chunks() {
  local count
  count=$(rg -c "$LONG_PATTERN" "$SERVER_LOG" 2>/dev/null || true)
  printf '%s\n' "${count:-0}"
}
cancellation_metric() {
  curl --fail --silent --show-error --max-time 2 "$BASE_URL/metrics" \
    | awk '$1 == "hf2q_sse_cancellations" { print $2; found=1 } END { if (!found) exit 1 }'
}
cleanup() {
  [[ -n "${cancel_pid:-}" ]] && kill "$cancel_pid" 2>/dev/null || true
  [[ -n "${tiny_pid:-}" ]] && kill "$tiny_pid" 2>/dev/null || true
  qwen36_stop_power_guard
}
trap cleanup EXIT

qwen36_start_power_guard "${SERVER_PID:-$$}" "$OUT_DIR/caffeinate.log"

[[ "$(sha256_file "$FIXTURE_JSON")" == "$LONG_FIXTURE_SHA256" ]] || exit 2
[[ "$(sha256_file "$SHORT_FIXTURE_JSON")" == "$SHORT_FIXTURE_SHA256" ]] || exit 2
jq '.max_tokens = 64' "$FIXTURE_JSON" >"$OUT_DIR/cancel-request.json"
cp "$SHORT_FIXTURE_JSON" "$OUT_DIR/tiny-request.json"
[[ "$(sha256_file "$OUT_DIR/cancel-request.json")" == "$LONG_RUNTIME_SHA256" ]] || exit 2

baseline_chunks=$(count_chunks)
baseline_cancel=$(cancellation_metric)
qwen36_write_log_baseline "$SERVER_LOG" "$OUT_DIR/server-log-baseline.json"
curl --silent --show-error --no-buffer --connect-timeout 5 --max-time 300 \
  -H 'Content-Type: application/json' --data-binary "@$OUT_DIR/cancel-request.json" \
  -o "$OUT_DIR/cancel-response.sse" \
  "$BASE_URL/v1/chat/completions" >"$OUT_DIR/cancel-curl.stdout" 2>"$OUT_DIR/cancel-curl.err" &
cancel_pid=$!

progress_deadline=$(( $(date +%s) + NO_PROGRESS_SECONDS ))
while (( $(count_chunks) - baseline_chunks < 3 )); do
  qwen36_assert_power_guard
  kill -0 "$cancel_pid" 2>/dev/null || {
    wait "$cancel_pid" || true
    echo "long cancellation request ended before three chunks" >&2
    exit 1
  }
  if (( $(date +%s) >= progress_deadline )); then
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
      sample "$SERVER_PID" 5 2 -file "$OUT_DIR/no-progress.sample" >/dev/null 2>&1 || true
    fi
    echo "no long-prefill progress before cancellation" >&2
    exit 1
  fi
  sleep 0.05
done

chunks_at_disconnect=$(( $(count_chunks) - baseline_chunks ))
[[ "$chunks_at_disconnect" == 3 ]] || {
  echo "cancellation did not occur at the exact third committed boundary: $chunks_at_disconnect" >&2
  exit 1
}
kill "$cancel_pid" 2>/dev/null || true
wait "$cancel_pid" 2>/dev/null || true
cancel_pid=

cancel_deadline=$(( $(date +%s) + 15 ))
while (( $(cancellation_metric) <= baseline_cancel )); do
  (( $(date +%s) < cancel_deadline )) || {
    echo "cancellation metric did not advance within 15 seconds" >&2
    exit 1
  }
  sleep 0.1
done
chunks_after_cancel=$(( $(count_chunks) - baseline_chunks ))
(( chunks_after_cancel <= chunks_at_disconnect + 1 )) || {
  echo "more than the one in-flight atomic chunk completed after disconnect" >&2
  exit 1
}
sleep 5
chunks_after_stability=$(( $(count_chunks) - baseline_chunks ))
[[ "$chunks_after_stability" == "$chunks_after_cancel" ]] || {
  echo "cancelled Qwen prefill continued after its transaction boundary" >&2
  exit 1
}
qwen36_reject_successful_terminal_sse "$OUT_DIR/cancel-response.sse"

pre_tiny_ready_code=$(curl --silent --show-error --max-time 3 -o "$OUT_DIR/readyz-before-tiny.txt" \
  -w '%{http_code}' "$BASE_URL/readyz")
[[ "$pre_tiny_ready_code" == 200 ]] || { echo "readyz failed after cancellation" >&2; exit 1; }

curl --fail-with-body --silent --show-error --no-buffer --connect-timeout 5 --max-time 60 \
  -H 'Content-Type: application/json' --data-binary "@$OUT_DIR/tiny-request.json" \
  -o "$OUT_DIR/tiny-response.sse" \
  -w 'http_code=%{http_code}\ntotal_seconds=%{time_total}\n' \
  "$BASE_URL/v1/chat/completions" >"$OUT_DIR/tiny-curl.metrics" 2>"$OUT_DIR/tiny-curl.err"
grep -qx 'http_code=200' "$OUT_DIR/tiny-curl.metrics"
done_count=$(sed -n 's/^data: //p' "$OUT_DIR/tiny-response.sse" \
  | awk '$0 == "[DONE]" { count++ } END { print count + 0 }')
[[ "$done_count" == 1 ]] || exit 1
qwen36_extract_and_validate_sse tiny "$OUT_DIR/tiny-response.sse" \
  "$OUT_DIR/tiny-events.jsonl"
tiny_content=$(jq -j '.choices[0].delta.content // empty' "$OUT_DIR/tiny-events.jsonl")
tiny_finish_count=$(jq -r '.choices[0].finish_reason // empty' "$OUT_DIR/tiny-events.jsonl" \
  | awk '$0 == "stop" { count++ } END { print count + 0 }')
tiny_tool_count=$(jq -s '[.[] | .choices[0].delta.tool_calls[]?] | length' "$OUT_DIR/tiny-events.jsonl")
[[ "$tiny_content" == OK && "$tiny_finish_count" == 1 && "$tiny_tool_count" == 0 ]] || {
  echo "post-cancel one-slot reuse did not return exact OK/stop/no-tools" >&2
  exit 1
}
qwen36_validate_short_events "$OUT_DIR/tiny-events.jsonl"
final_ready_code=$(curl --silent --show-error --max-time 3 -o "$OUT_DIR/readyz-after-tiny.txt" \
  -w '%{http_code}' "$BASE_URL/readyz")
[[ "$final_ready_code" == 200 ]] || { echo "readyz failed after one-slot reuse" >&2; exit 1; }

new_log="$OUT_DIR/server-log-delta.log"
qwen36_extract_append_only_log_delta "$SERVER_LOG" \
  "$OUT_DIR/server-log-baseline.json" "$new_log"
qwen36_reject_fatal_log "$new_log"
short_shape_count=$(rg -c 'chat completion request received.*messages=3 tools=0' "$new_log" || true)
long_shape_count=$(rg -c 'chat completion request received.*messages=2 tools=347' "$new_log" || true)
[[ "$short_shape_count" == 1 ]] || {
  echo "expected exactly one post-cancel 3-message/0-tool control, observed $short_shape_count" >&2
  exit 1
}
[[ "$long_shape_count" == 1 ]] || {
  echo "expected exactly one cancelled 2-message/347-tool request, observed $long_shape_count" >&2
  exit 1
}

chunk_lines="$OUT_DIR/cancelled-long-chunks.log"
rg "$LONG_PATTERN" "$SERVER_LOG" | sed -n "$((baseline_chunks + 1)),\$p" >"$chunk_lines"
qwen36_validate_chunk_prefix_lines "$chunk_lines" "$chunks_after_stability"

final_cancel=$(cancellation_metric)
qwen36_assert_power_guard
[[ "$((final_cancel - baseline_cancel))" == 1 ]] || {
  echo "expected cancellation counter delta 1, observed $((final_cancel - baseline_cancel))" >&2
  exit 1
}
tiny_sse_sha256=$(sha256_file "$OUT_DIR/tiny-response.sse")
canceled_sse_sha256=$(sha256_file "$OUT_DIR/cancel-response.sse")
server_log_delta_sha256=$(sha256_file "$new_log")
chunk_log_sha256=$(sha256_file "$chunk_lines")
summary="$OUT_DIR/cancellation-summary.json"
jq -n \
  --arg status pass \
  --arg out_dir "$OUT_DIR" \
  --argjson chunks_at_disconnect "$chunks_at_disconnect" \
  --argjson chunks_after_cancel "$chunks_after_cancel" \
  --argjson chunks_after_stability "$chunks_after_stability" \
  --argjson baseline_cancellation_counter "$baseline_cancel" \
  --argjson final_cancellation_counter "$final_cancel" \
  --argjson cancellation_delta "$((final_cancel - baseline_cancel))" \
  --argjson max_slots "$MAX_SLOTS" \
  --argjson pre_tiny_ready_http "$pre_tiny_ready_code" \
  --argjson ready_http "$final_ready_code" \
  --argjson power_event_baseline "$QWEN36_POWER_EVENT_BASELINE" \
  --argjson power_event_final "$QWEN36_POWER_EVENT_FINAL" \
  --argjson power_event_delta "$((QWEN36_POWER_EVENT_FINAL - QWEN36_POWER_EVENT_BASELINE))" \
  --arg tiny_content "$tiny_content" \
  --arg tiny_sse_sha256 "$tiny_sse_sha256" \
  --arg canceled_sse_sha256 "$canceled_sse_sha256" \
  --arg server_log_delta_sha256 "$server_log_delta_sha256" \
  --arg chunk_log_sha256 "$chunk_log_sha256" \
  --arg binary_path "$BINARY_PATH" \
  --arg binary_sha256 "$BINARY_SHA256" \
  --arg model_path "$MODEL_PATH" \
  --arg model_sha256 "$MODEL_SHA256" \
  --arg long_fixture_sha256 "$LONG_FIXTURE_SHA256" \
  --arg long_runtime_sha256 "$LONG_RUNTIME_SHA256" \
  --arg short_fixture_sha256 "$SHORT_FIXTURE_SHA256" \
  --arg tools_sha256 "$TOOLS_SHA256" \
  --arg template_sha256 "$TEMPLATE_SHA256" \
  --argjson server_pid "${SERVER_PID:-0}" \
  --argjson short_shape_count "$short_shape_count" \
  --argjson long_shape_count "$long_shape_count" \
  --argjson cancelled_success_terminal false \
  --argjson same_slot_reuse true \
  --argjson tiny_total_seconds "$(awk -F= '$1 == "total_seconds" { print $2 }' "$OUT_DIR/tiny-curl.metrics")" \
  '{status:$status,out_dir:$out_dir,binary_path:$binary_path,binary_sha256:$binary_sha256,model_path:$model_path,model_sha256:$model_sha256,server_pid:$server_pid,long_fixture_sha256:$long_fixture_sha256,long_runtime_sha256:$long_runtime_sha256,short_fixture_sha256:$short_fixture_sha256,tools_sha256:$tools_sha256,template_sha256:$template_sha256,max_slots:$max_slots,chunks_at_disconnect:$chunks_at_disconnect,chunks_after_cancel:$chunks_after_cancel,chunks_after_stability:$chunks_after_stability,baseline_cancellation_counter:$baseline_cancellation_counter,final_cancellation_counter:$final_cancellation_counter,cancellation_delta:$cancellation_delta,short_shape_count:$short_shape_count,long_shape_count:$long_shape_count,cancelled_success_terminal:$cancelled_success_terminal,same_slot_reuse:$same_slot_reuse,pre_tiny_ready_http:$pre_tiny_ready_http,ready_http:$ready_http,power_event_baseline:$power_event_baseline,power_event_final:$power_event_final,power_event_delta:$power_event_delta,tiny_content:$tiny_content,tiny_sse_sha256:$tiny_sse_sha256,canceled_sse_sha256:$canceled_sse_sha256,server_log_delta_sha256:$server_log_delta_sha256,chunk_log_sha256:$chunk_log_sha256,tiny_total_seconds:$tiny_total_seconds}' \
  >"$summary.tmp"
jq -e '.status == "pass" and .server_pid > 0 and .max_slots == 1 and (.binary_sha256 | test("^[0-9a-f]{64}$")) and (.model_sha256 | test("^[0-9a-f]{64}$")) and .chunks_at_disconnect == 3 and .cancellation_delta == 1 and .cancelled_success_terminal == false and .same_slot_reuse == true and .tiny_content == "OK" and .pre_tiny_ready_http == 200 and .ready_http == 200 and .short_shape_count == 1 and .long_shape_count == 1 and .chunks_after_cancel == .chunks_after_stability and .power_event_delta == 0' "$summary.tmp" >/dev/null
mv "$summary.tmp" "$summary"
shasum -a 256 "$summary" >"$summary.sha256"
shasum -c "$summary.sha256" >/dev/null
cat "$summary"
