#!/usr/bin/env bash
# Prove Gemma's bounded long prefill yields to an interactive SSE lane and
# that disconnecting a second long prefill stops it at a transaction boundary.
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:18082}
SERVER_PID=${SERVER_PID:?SERVER_PID is required}
SERVER_LOG=${SERVER_LOG:?SERVER_LOG is required}
BINARY_PATH=${BINARY_PATH:?BINARY_PATH is required}
BINARY_SHA256=${BINARY_SHA256:?BINARY_SHA256 is required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
MODEL_SHA256=${MODEL_SHA256:?MODEL_SHA256 is required}
MAX_SLOTS=${MAX_SLOTS:-4}
OUT_DIR=${OUT_DIR:?OUT_DIR is required}
CONTEXT_LINES=${CONTEXT_LINES:-7000}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-900}
CANCELLATION_WAIT_SECONDS=${CANCELLATION_WAIT_SECONDS:-120}
PRIMARY_CONTEXT_SHA256=07b147e9c6ac26a0c9c4a719391c0772b2d27b9d77499479014b9ace88b6b11e
CANCELLATION_CONTEXT_SHA256=f0b264eedae315618941d8fa6fb16454c4eac03b5793e213b613d66ccb7b6e4a

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

[[ "$SERVER_PID" =~ ^[1-9][0-9]*$ && "$MAX_SLOTS" == 4 ]] || exit 2
[[ "$CURL_MAX_TIME_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "CURL_MAX_TIME_SECONDS must be a positive integer" >&2
  exit 2
}
[[ "$CANCELLATION_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "CANCELLATION_WAIT_SECONDS must be a positive integer" >&2
  exit 2
}
for command in awk curl jq rg sed shasum stat wc caffeinate pmset; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ "$CONTEXT_LINES" == 7000 ]] || {
  echo "release-authority Gemma overlap requires CONTEXT_LINES=7000" >&2
  exit 2
}
[[ "$BINARY_SHA256" =~ ^[0-9a-f]{64}$ && "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ -x "$BINARY_PATH" && -r "$MODEL_PATH" && -r "$SERVER_LOG" ]] || exit 2
qwen36_require_empty_receipt_dir "$OUT_DIR"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
[[ "$(sha256_file "$BINARY_PATH")" == "$BINARY_SHA256" ]] || exit 2
hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256"
qwen36_bind_server_process "$BASE_URL" "$SERVER_PID" "$BINARY_PATH" \
  "$MODEL_PATH" "$MAX_SLOTS"

model=$(curl --fail-with-body --silent --show-error --max-time 10 \
  "$BASE_URL/v1/models" | jq -er \
  '[.data[] | select((.arch // "") != "")] | if length == 1 then .[0].id else error("expected one model") end')

long_pid=""
short_pid=""
cancel_pid=""
cleanup() {
  for pid in "$long_pid" "$short_pid" "$cancel_pid"; do
    [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
  done
  qwen36_stop_power_guard
}
trap cleanup EXIT INT TERM
qwen36_start_power_guard "$SERVER_PID" "$OUT_DIR/caffeinate.log"
qwen36_write_log_baseline "$SERVER_LOG" "$OUT_DIR/server-log-baseline.json"

wait_for_log() {
  local first_line=$1
  local pattern=$2
  local timeout=${3:-120}
  local deadline=$((SECONDS + timeout))
  local row
  while ((SECONDS < deadline)); do
    row=$(sed -n "${first_line},\$p" "$SERVER_LOG" | rg -m1 "$pattern" || true)
    if [[ -n "$row" ]]; then
      printf '%s\n' "$row"
      return 0
    fi
    qwen36_assert_power_guard || return 1
    sleep 0.1
  done
  echo "timed out waiting for Gemma log: $pattern" >&2
  return 1
}

semantic_events() {
  local path=$1
  awk '/^data: \{/' "$path" 2>/dev/null | sed 's/^data: //' | jq -s '
    [.[] | select(type == "object") | .choices[0].delta
      | select(((.content // "") | length) > 0
          or ((.reasoning_content // "") | length) > 0
          or ((.tool_calls // []) | length) > 0)] | length
  ' 2>/dev/null || printf '0\n'
}

cancellation_metric() {
  curl --fail --silent --show-error --max-time 3 "$BASE_URL/metrics" |
    awk '$1 == "hf2q_sse_cancellations" {print $2; found=1} END {if (!found) exit 1}'
}

make_context() {
  local label=$1
  local path=$2
  local lines=${3:-$CONTEXT_LINES}
  awk -v lines="$lines" -v label="$label" 'BEGIN {
    for (i = 1; i <= lines; i++) {
      printf "gemma-overlap-%s line %05d: cache ownership bounded prefill cancellation and interactive fairness.\n", label, i
    }
  }' >"$path"
}

make_context primary "$OUT_DIR/long-context.txt"
[[ "$(sha256_file "$OUT_DIR/long-context.txt")" == "$PRIMARY_CONTEXT_SHA256" ]] || exit 2
jq -n --arg model "$model" --rawfile context "$OUT_DIR/long-context.txt" '{
  model:$model,
  messages:[
    {role:"system",content:"You are a deterministic Gemma transaction-boundary test."},
    {role:"user",content:("Read this unique context, then answer with a concise summary.\n\n" + $context)}
  ],
  temperature:0,max_tokens:32,stream:true,stream_options:{include_usage:true}
}' >"$OUT_DIR/long.request.json"
jq -n --arg model "$model" '{
  model:$model,
  messages:[
    {role:"system",content:"You are the interactive lane in a scheduler test."},
    {role:"assistant",content:"Ready."},
    {role:"user",content:"Write one short sentence confirming interactive progress."}
  ],
  temperature:0,max_tokens:16,stream:true,stream_options:{include_usage:true}
}' >"$OUT_DIR/short.request.json"

overlap_log_start=$(( $(wc -l <"$SERVER_LOG") + 1 ))
curl --fail-with-body --silent --show-error --no-buffer --connect-timeout 5 \
  --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary @"$OUT_DIR/long.request.json" \
  "$BASE_URL/v1/chat/completions" >"$OUT_DIR/long.sse" 2>"$OUT_DIR/long.stderr" &
long_pid=$!
prepared=$(wait_for_log "$overlap_log_start" 'chat completion prepared; dispatching to inference worker.*prompt_tokens=[0-9]+' 180)
long_prompt_tokens=$(sed -n 's/.*prompt_tokens=\([0-9][0-9]*\).*/\1/p' <<<"$prepared")
if [[ ! "$long_prompt_tokens" =~ ^[0-9]+$ ]] || ((long_prompt_tokens <= 80000)); then
  echo "Gemma long overlap prompt was not >80K tokens: $long_prompt_tokens" >&2
  exit 1
fi
wait_for_log "$overlap_log_start" \
  "Gemma4 bounded prefill transaction complete.*prompt_tokens=${long_prompt_tokens}( |$)" 180 >/dev/null

curl --fail-with-body --silent --show-error --no-buffer --connect-timeout 5 \
  --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary @"$OUT_DIR/short.request.json" \
  "$BASE_URL/v1/chat/completions" >"$OUT_DIR/short.sse" 2>"$OUT_DIR/short.stderr" &
short_pid=$!

semantic_deadline=$((SECONDS + 180))
short_semantic=0
while ((SECONDS < semantic_deadline)); do
  short_semantic=$(semantic_events "$OUT_DIR/short.sse")
  ((short_semantic > 0)) && break
  kill -0 "$long_pid" 2>/dev/null || {
    echo "long Gemma request completed before interactive semantic progress" >&2
    exit 1
  }
  qwen36_assert_power_guard
  sleep 0.1
done
((short_semantic > 0)) || { echo "no interactive Gemma semantic progress" >&2; exit 1; }
latest_long_transaction=$(sed -n "${overlap_log_start},\$p" "$SERVER_LOG" |
  rg "Gemma4 bounded prefill transaction complete.*prompt_tokens=${long_prompt_tokens}( |$)" |
  tail -1)
committed_during_short=$(sed -n 's/.*committed_tokens=\([0-9][0-9]*\).*/\1/p' \
  <<<"$latest_long_transaction")
if [[ ! "$committed_during_short" =~ ^[0-9]+$ ]] || \
  ((committed_during_short >= long_prompt_tokens)); then
  echo "short Gemma lane did not make semantic progress during an incomplete long prefill" >&2
  exit 1
fi
kill -0 "$long_pid" 2>/dev/null || exit 1

wait "$short_pid"; short_pid=""
wait "$long_pid"; long_pid=""
for name in short long; do
  [[ "$(rg -c '^data: \[DONE\]$' "$OUT_DIR/$name.sse" || true)" == 1 ]] || exit 1
  qwen36_extract_and_validate_sse "$name" "$OUT_DIR/$name.sse" "$OUT_DIR/$name.events.jsonl"
done
(( $(semantic_events "$OUT_DIR/long.sse") > 0 )) || exit 1
long_content=$(jq -j '.choices[0].delta.content // empty' "$OUT_DIR/long.events.jsonl")
long_reasoning=$(jq -j '.choices[0].delta.reasoning_content // empty' "$OUT_DIR/long.events.jsonl")

# A second, distinct long prompt must stop at the first safe boundary after
# the client disconnects, restore the completed first turn's checkpoint, and
# leave the worker ready for later work.
make_context cancellation "$OUT_DIR/cancel-context.txt" 1200
[[ "$(sha256_file "$OUT_DIR/cancel-context.txt")" == "$CANCELLATION_CONTEXT_SHA256" ]] || exit 2
jq -n --slurpfile base "$OUT_DIR/long.request.json" \
  --arg content "$long_content" --arg reasoning "$long_reasoning" \
  --rawfile context "$OUT_DIR/cancel-context.txt" '
  $base[0]
  | .messages += [
    ({role:"assistant",content:$content}
      + (if ($reasoning | length) > 0 then {reasoning_content:$reasoning} else {} end)),
    {role:"user",content:("Continue from the prior turn after reading this distinct suffix.\n\n" + $context)}
  ]
  | .max_tokens = 32
  | .stream = true
  | .stream_options = {include_usage:true}
  | .temperature = 0
  | {
  model:.model,
  messages:.messages,
  temperature:0,max_tokens:32,stream:true,stream_options:{include_usage:true}
}' >"$OUT_DIR/cancel.request.json"
cancel_log_start=$(( $(wc -l <"$SERVER_LOG") + 1 ))
cancel_before=$(cancellation_metric)
curl --silent --show-error --no-buffer --connect-timeout 5 \
  --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary @"$OUT_DIR/cancel.request.json" \
  "$BASE_URL/v1/chat/completions" >"$OUT_DIR/cancel.sse" 2>"$OUT_DIR/cancel.stderr" &
cancel_pid=$!
cancel_prepared=$(wait_for_log "$cancel_log_start" 'chat completion prepared; dispatching to inference worker.*prompt_tokens=[0-9]+' 180)
cancel_prompt_tokens=$(sed -n 's/.*prompt_tokens=\([0-9][0-9]*\).*/\1/p' <<<"$cancel_prepared")
[[ "$cancel_prompt_tokens" =~ ^[0-9]+$ ]] && ((cancel_prompt_tokens > 80000)) || exit 1
cancel_chunks=0
cancel_progress_deadline=$((SECONDS + 180))
while ((cancel_chunks < 2 && SECONDS < cancel_progress_deadline)); do
  cancel_chunks=$(sed -n "${cancel_log_start},\$p" "$SERVER_LOG" |
    rg -c "Gemma4 bounded prefill transaction complete.*prompt_tokens=${cancel_prompt_tokens}( |$)" || true)
  qwen36_assert_power_guard
  sleep 0.05
done
((cancel_chunks >= 2)) || { echo "Gemma cancellation prompt made no bounded progress" >&2; exit 1; }
kill "$cancel_pid" 2>/dev/null || true
wait "$cancel_pid" 2>/dev/null || true
cancel_pid=""
cancel_deadline=$((SECONDS + CANCELLATION_WAIT_SECONDS))
while (( $(cancellation_metric) <= cancel_before && SECONDS < cancel_deadline )); do sleep 0.1; done
cancel_after=$(cancellation_metric)
[[ "$((cancel_after - cancel_before))" == 1 ]] || exit 1
chunks_after_cancel=$(sed -n "${cancel_log_start},\$p" "$SERVER_LOG" |
  rg -c "Gemma4 bounded prefill transaction complete.*prompt_tokens=${cancel_prompt_tokens}( |$)" || true)
sleep 5
chunks_after_stability=$(sed -n "${cancel_log_start},\$p" "$SERVER_LOG" |
  rg -c "Gemma4 bounded prefill transaction complete.*prompt_tokens=${cancel_prompt_tokens}( |$)" || true)
[[ "$chunks_after_cancel" == "$chunks_after_stability" ]] || {
  echo "cancelled Gemma prefill continued past its transaction boundary" >&2
  exit 1
}
qwen36_reject_successful_terminal_sse "$OUT_DIR/cancel.sse"
rollback_restores=$(sed -n "${cancel_log_start},\$p" "$SERVER_LOG" |
  rg -c 'Gemma4 cancellation restored verified prompt checkpoint' || true)
((rollback_restores >= 1)) || {
  echo "Gemma cancellation did not restore the prior verified checkpoint" >&2
  exit 1
}

ready_http=$(curl --silent --show-error --max-time 3 -o "$OUT_DIR/readyz.json" \
  -w '%{http_code}' "$BASE_URL/readyz")
[[ "$ready_http" == 200 ]] || exit 1
qwen36_assert_power_guard
qwen36_extract_append_only_log_delta "$SERVER_LOG" \
  "$OUT_DIR/server-log-baseline.json" "$OUT_DIR/server-log-delta.log"
qwen36_reject_fatal_log "$OUT_DIR/server-log-delta.log"

bounded_log="$OUT_DIR/bounded-transactions.log"
rg 'Gemma4 (bounded prefill transaction complete|installed prefills advanced in one aggregate-bounded transaction|stable agent suffixes prefilled in one multi-slot body)' \
  "$OUT_DIR/server-log-delta.log" >"$bounded_log"
[[ -s "$bounded_log" ]] || exit 1
if awk '
  { for (i=1; i<=NF; i++) if ($i ~ /^(advanced_tokens|suffix_tokens)=/) {
      split($i, a, "="); if (a[2] !~ /^[0-9]+$/ || a[2] > 4096) exit 1
    }
  }
' "$bounded_log"; then :; else
  echo "Gemma transaction exceeded the 4,096-row cap" >&2
  exit 1
fi

jq -n \
  --arg status pass --arg binary_sha256 "$BINARY_SHA256" --arg model_sha256 "$MODEL_SHA256" \
  --arg primary_context_sha256 "$PRIMARY_CONTEXT_SHA256" \
  --arg cancellation_context_sha256 "$CANCELLATION_CONTEXT_SHA256" \
  --arg server_log_sha256 "$(sha256_file "$OUT_DIR/server-log-delta.log")" \
  --argjson server_pid "$SERVER_PID" --argjson max_slots "$MAX_SLOTS" \
  --argjson long_prompt_tokens "$long_prompt_tokens" \
  --argjson committed_tokens_when_short_progressed "$committed_during_short" \
  --argjson short_semantic_events "$short_semantic" \
  --argjson cancel_prompt_tokens "$cancel_prompt_tokens" \
  --argjson cancellation_delta "$((cancel_after-cancel_before))" \
  --argjson chunks_after_cancel "$chunks_after_cancel" \
  --argjson chunks_after_stability "$chunks_after_stability" \
  --argjson rollback_restores "$rollback_restores" \
  --argjson ready_http "$ready_http" \
  --argjson power_event_delta "$((QWEN36_POWER_EVENT_FINAL-QWEN36_POWER_EVENT_BASELINE))" \
  '{status:$status,binary_sha256:$binary_sha256,model_sha256:$model_sha256,primary_context_sha256:$primary_context_sha256,cancellation_context_sha256:$cancellation_context_sha256,server_pid:$server_pid,max_slots:$max_slots,long_prompt_tokens:$long_prompt_tokens,short_semantic_during_long_prefill:true,committed_tokens_when_short_progressed:$committed_tokens_when_short_progressed,short_semantic_events:$short_semantic_events,cancel_prompt_tokens:$cancel_prompt_tokens,cancellation_delta:$cancellation_delta,chunks_after_cancel:$chunks_after_cancel,chunks_after_stability:$chunks_after_stability,rollback_restores:$rollback_restores,transaction_cap_tokens:4096,ready_http:$ready_http,power_event_delta:$power_event_delta,server_log_sha256:$server_log_sha256}' \
  >"$OUT_DIR/summary.json.tmp"
jq -e '.status == "pass" and .server_pid > 0 and .max_slots == 4 and .long_prompt_tokens > 80000 and .short_semantic_during_long_prefill == true and .short_semantic_events > 0 and .committed_tokens_when_short_progressed < .long_prompt_tokens and .cancel_prompt_tokens > 80000 and .cancellation_delta == 1 and .chunks_after_cancel == .chunks_after_stability and .rollback_restores >= 1 and .transaction_cap_tokens == 4096 and .ready_http == 200 and .power_event_delta == 0' \
  "$OUT_DIR/summary.json.tmp" >/dev/null
qwen36_assert_power_guard
mv "$OUT_DIR/summary.json.tmp" "$OUT_DIR/summary.json"
shasum -a 256 "$OUT_DIR/summary.json" >"$OUT_DIR/summary.json.sha256"
shasum -c "$OUT_DIR/summary.json.sha256" >/dev/null
cat "$OUT_DIR/summary.json"
