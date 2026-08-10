#!/usr/bin/env bash
# Reproduce the user-visible DeepSeek-V4 lopsided overlap: a small SSE lane is
# already decoding when one public 347-tool cold prompt begins prefill. The
# interactive lane must receive a full decode quantum before the cold lane can
# consume one legacy 2,048-token transaction.
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:8080}
MODEL=${MODEL:-Deepseek v4 Flash 0731 Source}
MODEL_PATH=${MODEL_PATH:-/opt/hf2q/artifacts/DeepSeek-V4-Flash-0731-agentic-q2-repro.gguf}
FIXTURE_JSON=${FIXTURE_JSON:?FIXTURE_JSON must be the canonical public 347-tool request}
SERVER_LOG=${SERVER_LOG:?SERVER_LOG must name the current server log}
SERVER_PID=${SERVER_PID:?SERVER_PID must name the current hf2q process}
OUT_DIR=${OUT_DIR:?OUT_DIR must name a fresh receipt directory}
BINARY_PATH=${BINARY_PATH:?BINARY_PATH must name the running hf2q binary}
BINARY_SHA256=${BINARY_SHA256:?BINARY_SHA256 is required}
MODEL_SHA256=${MODEL_SHA256:?MODEL_SHA256 is required}
MAX_SLOTS=${MAX_SLOTS:-4}
NO_PROGRESS_SECONDS=${NO_PROGRESS_SECONDS:-30}

CANONICAL_FIXTURE_SHA256=6671a0c89b8d4935caa4b87bee08361c5b8727ec557e9edb05947ad90c94c13d

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

for command in awk curl date jq lsof ps rg sed seq shasum sort stat wc caffeinate pmset; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
for path in "$FIXTURE_JSON" "$SERVER_LOG" "$MODEL_PATH" "$BINARY_PATH"; do
  [[ -r "$path" ]] || { echo "required path is not readable: $path" >&2; exit 2; }
done
[[ ! -e "$OUT_DIR" ]] || { echo "OUT_DIR must be fresh: $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"
[[ "$SERVER_PID" =~ ^[1-9][0-9]*$ && "$MAX_SLOTS" == 4 ]] || exit 2
[[ "$BINARY_SHA256" =~ ^[0-9a-f]{64}$ && "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
[[ "$(sha256_file "$FIXTURE_JSON")" == "$CANONICAL_FIXTURE_SHA256" ]] || {
  echo "347-tool fixture is not the canonical public request" >&2
  exit 2
}
[[ "$(sha256_file "$BINARY_PATH")" == "$BINARY_SHA256" ]] || exit 2
[[ "$(sha256_file "$MODEL_PATH")" == "$MODEL_SHA256" ]] || exit 2

qwen36_bind_server_process \
  "$BASE_URL" "$SERVER_PID" "$BINARY_PATH" "$MODEL_PATH" "$MAX_SLOTS"
if rg -q 'DeepSeek-V4 request started' "$SERVER_LOG"; then
  echo "DeepSeek overlap authority requires a fresh process with no prior requests" >&2
  exit 2
fi

short_pid=
long_pid=
cleanup() {
  for pid in "$short_pid" "$long_pid"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  qwen36_stop_power_guard
}
trap cleanup EXIT
qwen36_start_power_guard "$SERVER_PID" "$OUT_DIR/caffeinate.log"

baseline_lines=$(wc -l <"$SERVER_LOG" | tr -d ' ')
baseline_inode=$(stat -f '%i' "$SERVER_LOG")
baseline_prefix_sha256=$(sed -n "1,${baseline_lines}p" "$SERVER_LOG" | shasum -a 256 | awk '{print $1}')

short_request="$OUT_DIR/short.request.json"
long_request="$OUT_DIR/long.request.json"
short_sse="$OUT_DIR/short.sse"
long_sse="$OUT_DIR/long.sse"
jq -n --arg model "$MODEL" '{
  model: $model,
  messages: [
    {role:"system", content:"You are the interactive lane in a deterministic scheduling test."},
    {role:"assistant", content:"Ready."},
    {role:"user", content: (([range(0; 450) | "x"] | join(" ")) + "\nWrite the integers one through sixty-four in words, separated by spaces, then write DONE.")}
  ],
  temperature: 0,
  max_tokens: 128,
  stream: true,
  stream_options: {include_usage: true}
}' >"$short_request"
jq --arg model "$MODEL" '.model = $model | .max_tokens = 64 | .stream_options = {include_usage:true}' \
  "$FIXTURE_JSON" >"$long_request"
[[ "$(jq '.tools | length' "$long_request")" == 347 ]] || exit 2

wait_for_log() {
  local pattern=$1
  local deadline=$(( $(date +%s) + ${2:-30} ))
  local first_line=${3:-$((baseline_lines + 1))}
  local found
  while (( $(date +%s) < deadline )); do
    found=$(sed -n "${first_line},\$p" "$SERVER_LOG" | rg -m1 "$pattern" || true)
    if [[ -n "$found" ]]; then printf '%s\n' "$found"; return 0; fi
    qwen36_assert_power_guard
    sleep 0.1
  done
  echo "timed out waiting for log pattern: $pattern" >&2
  return 1
}

curl --fail-with-body --silent --show-error --no-buffer \
  --connect-timeout 5 --max-time 600 -H 'Content-Type: application/json' \
  --data-binary "@$short_request" "$BASE_URL/v1/chat/completions" \
  >"$short_sse" 2>"$OUT_DIR/short.stderr" &
short_pid=$!
# Begin rendering the public 347-tool peer immediately. Waiting for a worker
# log first gives the fast 488-token request enough time to finish before the
# large HTTP body has even reached admission, which destroys the overlap this
# gate is meant to prove. The small shape gets a deterministic head start and
# the request IDs are recovered from their distinct max-token budgets.
sleep 0.05
curl --fail-with-body --silent --show-error --no-buffer \
  --connect-timeout 5 --max-time 600 -H 'Content-Type: application/json' \
  --data-binary "@$long_request" "$BASE_URL/v1/chat/completions" \
  >"$long_sse" 2>"$OUT_DIR/long.stderr" &
long_pid=$!
short_started=$(wait_for_log 'DeepSeek-V4 request started.*max_tokens=128( |$)')
short_request_id=$(sed -n 's/.*request_id=\([0-9][0-9]*\).*/\1/p' <<<"$short_started")
long_started=$(wait_for_log 'DeepSeek-V4 request started.*max_tokens=64( |$)')
long_request_id=$(sed -n 's/.*request_id=\([0-9][0-9]*\).*/\1/p' <<<"$long_started")
[[ "$long_request_id" != "$short_request_id" ]] || exit 1
wait_for_log "DeepSeek-V4 decode started.*request_id=${short_request_id}( |$)" 60 >/dev/null

# The ordinary progress row is cumulative over five seconds and can include
# solo work before the short lane reaches Decode. Bind the incident fix to the
# first actual mixed transaction instead: two native windows are 256 tokens.
first_mixed_slice=$(wait_for_log "DeepSeek-V4 mixed prefill slice.*request_id=${long_request_id}( |$)" 30)
first_mixed_chunk_tokens=$(sed -n 's/.*chunk_tokens=\([0-9][0-9]*\).*/\1/p' <<<"$first_mixed_slice")
first_mixed_window_cap=$(sed -n 's/.*window_cap=\([0-9][0-9]*\).*/\1/p' <<<"$first_mixed_slice")
[[ "$first_mixed_chunk_tokens" =~ ^[0-9]+$ && "$first_mixed_window_cap" == 2 ]] || exit 1
(( first_mixed_chunk_tokens > 0 && first_mixed_chunk_tokens <= 256 )) || {
  echo "interactive overlap used an oversized mixed prefill slice: $first_mixed_chunk_tokens" >&2
  exit 1
}

first_long_progress=$(wait_for_log "DeepSeek-V4 prefill progress.*request_id=${long_request_id}( |$)" 30)
first_long_processed=$(sed -n 's/.*processed_tokens=\([0-9][0-9]*\).*/\1/p' <<<"$first_long_progress")
[[ "$first_long_processed" =~ ^[0-9]+$ ]] || exit 1
(( first_long_processed > 0 )) || exit 1

decode_deadline=$(( $(date +%s) + 15 ))
short_generated=0
while (( $(date +%s) < decode_deadline )); do
  short_generated=$(sed -n "$((baseline_lines + 1)),\$p" "$SERVER_LOG" |
    sed -n "s/.*DeepSeek-V4 decode progress.*request_id=${short_request_id} .*generated_tokens=\([0-9][0-9]*\).*/\1/p" |
    tail -1)
  short_generated=${short_generated:-0}
  (( short_generated >= 8 )) && break
  sleep 0.1
done
(( short_generated >= 8 )) || {
  echo "interactive lane generated only $short_generated tokens during the first cold-prefill reporting window" >&2
  exit 1
}

last_progress=$(date +%s)
last_short_bytes=$(wc -c <"$short_sse" | tr -d ' ')
last_long_processed=$first_long_processed
while kill -0 "$short_pid" 2>/dev/null || kill -0 "$long_pid" 2>/dev/null; do
  sleep 1
  qwen36_assert_power_guard
  short_bytes=$(wc -c <"$short_sse" | tr -d ' ')
  current_long_processed=$(sed -n "$((baseline_lines + 1)),\$p" "$SERVER_LOG" |
    sed -n "s/.*DeepSeek-V4 prefill progress.*request_id=${long_request_id} .*processed_tokens=\([0-9][0-9]*\).*/\1/p" |
    tail -1)
  current_long_processed=${current_long_processed:-$last_long_processed}
  if (( short_bytes > last_short_bytes || current_long_processed > last_long_processed )); then
    last_short_bytes=$short_bytes
    last_long_processed=$current_long_processed
    last_progress=$(date +%s)
  fi
  if (( $(date +%s) - last_progress > NO_PROGRESS_SECONDS )); then
    echo "DeepSeek overlap made no semantic or prefill progress for ${NO_PROGRESS_SECONDS}s" >&2
    exit 1
  fi
done
wait "$short_pid"; short_pid=
wait "$long_pid"; long_pid=

qwen36_extract_and_validate_sse deep-short "$short_sse" "$OUT_DIR/short.events.jsonl"
qwen36_extract_and_validate_sse deep-long "$long_sse" "$OUT_DIR/long.events.jsonl"
qwen36_validate_long_events "$OUT_DIR/long.events.jsonl"
short_content=$(jq -j '.choices[0].delta.content // empty' "$OUT_DIR/short.events.jsonl")
[[ "$short_content" == *DONE* ]] || {
  echo "interactive lane did not finish its requested semantic response" >&2
  exit 1
}

final_lines=$(wc -l <"$SERVER_LOG" | tr -d ' ')
[[ "$(stat -f '%i' "$SERVER_LOG")" == "$baseline_inode" ]] || exit 1
(( final_lines >= baseline_lines )) || exit 1
[[ "$(sed -n "1,${baseline_lines}p" "$SERVER_LOG" | shasum -a 256 | awk '{print $1}')" == "$baseline_prefix_sha256" ]] || exit 1
sed -n "$((baseline_lines + 1)),${final_lines}p" "$SERVER_LOG" >"$OUT_DIR/server.delta.log"
qwen36_reject_fatal_log "$OUT_DIR/server.delta.log"
ready_http=$(curl --silent --show-error --max-time 3 -o "$OUT_DIR/readyz.json" -w '%{http_code}' "$BASE_URL/readyz")
[[ "$ready_http" == 200 ]] || exit 1
qwen36_assert_power_guard

long_prompt_tokens=$(sed -n 's/.*DeepSeek-V4 request started.*request_id='"$long_request_id"'.*prompt_tokens=\([0-9][0-9]*\).*/\1/p' \
  "$OUT_DIR/server.delta.log" | head -1)
summary="$OUT_DIR/summary.json"
jq -n \
  --arg binary_sha256 "$BINARY_SHA256" --arg model_sha256 "$MODEL_SHA256" \
  --arg fixture_sha256 "$CANONICAL_FIXTURE_SHA256" \
  --arg short_sse_sha256 "$(sha256_file "$short_sse")" \
  --arg long_sse_sha256 "$(sha256_file "$long_sse")" \
  --arg server_delta_sha256 "$(sha256_file "$OUT_DIR/server.delta.log")" \
  --argjson server_pid "$SERVER_PID" --argjson max_slots "$MAX_SLOTS" \
  --argjson short_request_id "$short_request_id" --argjson long_request_id "$long_request_id" \
  --argjson long_prompt_tokens "$long_prompt_tokens" \
  --argjson first_long_processed "$first_long_processed" \
  --argjson first_mixed_chunk_tokens "$first_mixed_chunk_tokens" \
  --argjson first_mixed_window_cap "$first_mixed_window_cap" \
  --argjson short_generated_at_first_window "$short_generated" \
  --argjson ready_http "$ready_http" \
  --argjson power_event_delta "$((QWEN36_POWER_EVENT_FINAL - QWEN36_POWER_EVENT_BASELINE))" '{
    status:"pass",
    server:{pid:$server_pid,max_slots:$max_slots,binary_sha256:$binary_sha256,model_sha256:$model_sha256},
    fixture_sha256:$fixture_sha256,
    short_request_id:$short_request_id,
    long_request_id:$long_request_id,
    long_prompt_tokens:$long_prompt_tokens,
    first_long_prefill_report_tokens:$first_long_processed,
    first_mixed_prefill_chunk_tokens:$first_mixed_chunk_tokens,
    first_mixed_prefill_window_cap:$first_mixed_window_cap,
    short_generated_tokens_at_first_window:$short_generated_at_first_window,
    short_sse_sha256:$short_sse_sha256,
    long_sse_sha256:$long_sse_sha256,
    server_delta_sha256:$server_delta_sha256,
    ready_http:$ready_http,
    power_event_delta:$power_event_delta
  }' >"$summary.tmp"
jq -e '.status=="pass" and .server.max_slots==4 and .long_prompt_tokens > 80000 and .first_long_prefill_report_tokens > 0 and .first_mixed_prefill_chunk_tokens > 0 and .first_mixed_prefill_chunk_tokens <= 256 and .first_mixed_prefill_window_cap == 2 and .short_generated_tokens_at_first_window >= 8 and .ready_http==200 and .power_event_delta==0' \
  "$summary.tmp" >/dev/null
mv "$summary.tmp" "$summary"
shasum -a 256 "$summary" >"$summary.sha256"
shasum -c "$summary.sha256" >/dev/null
cat "$summary"
