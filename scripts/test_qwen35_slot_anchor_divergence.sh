#!/usr/bin/env bash
set -euo pipefail

# ADR-049 Lane A fail-closed hardware gate for an already-running Qwen
# SlotAware server. Run once per served artifact. The script proves the server
# process really uses `--scheduler inflight-batched --max-slots 4`, builds an
# A->B->C checkpoint lineage, rewinds to A for branch X, and rejects reuse of
# stale B/C descendants. It also overlaps four clients, cancels the owner,
# verifies rollback reuse, exercises a rejected oversized prefill, and requires
# speculative cached-state reuse to remain live.

BASE_URL=${BASE_URL:-http://127.0.0.1:8081}
MODEL=${MODEL:-}
SERVER_PID=${SERVER_PID:-}
OUT_DIR=${OUT_DIR:-$(mktemp -d /var/tmp/hf2q-qwen-anchor.XXXXXX)}
RUN_ID=${RUN_ID:-"$$-$(date +%s)"}
CONTEXT_LINES=${CONTEXT_LINES:-4096}
ACTIVE_MAX_TOKENS=${ACTIVE_MAX_TOKENS:-2048}
OVERFLOW_WORDS=${OVERFLOW_WORDS:-300000}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-900}
SEMANTIC_WAIT_SECONDS=${SEMANTIC_WAIT_SECONDS:-180}

for command in awk curl jq lsof ps rg sed; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
for setting in CONTEXT_LINES ACTIVE_MAX_TOKENS OVERFLOW_WORDS \
  CURL_MAX_TIME_SECONDS SEMANTIC_WAIT_SECONDS; do
  value=${!setting}
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$setting must be a positive integer (got: $value)" >&2
    exit 2
  fi
done

mkdir -p "$OUT_DIR"

base_port=${BASE_URL##*:}
base_port=${base_port%%/*}
if [[ -z "$SERVER_PID" ]]; then
  SERVER_PID=$(lsof -nP -iTCP:"$base_port" -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true)
fi
[[ "$SERVER_PID" =~ ^[1-9][0-9]*$ ]] || {
  echo "cannot identify the hf2q listener PID for $BASE_URL; set SERVER_PID" >&2
  exit 2
}
server_command=$(ps -p "$SERVER_PID" -o command=)
[[ "$server_command" == *"--scheduler inflight-batched"* ]] || {
  echo "server is not provably using --scheduler inflight-batched: $server_command" >&2
  exit 2
}
[[ "$server_command" == *"--max-slots 4"* ]] || {
  echo "server is not provably using --max-slots 4: $server_command" >&2
  exit 2
}

if [[ -z "$MODEL" ]]; then
  MODEL=$(curl --fail-with-body --silent --show-error "$BASE_URL/v1/models" |
    jq -er '[.data[] | select((.arch // "") != "")] | if length == 1 then .[0].id else error("expected one inference model") end')
fi

post_json() {
  local input=$1
  local output=$2
  curl --fail-with-body --silent --show-error --no-buffer \
    --connect-timeout 5 --max-time "$CURL_MAX_TIME_SECONDS" \
    -H 'Content-Type: application/json' --data-binary "@$input" \
    "$BASE_URL/v1/chat/completions" >"$output"
  jq -e '(.choices | length) == 1 and (.choices[0].message | type) == "object"' \
    "$output" >/dev/null || {
      echo "invalid completion: $output" >&2
      exit 1
    }
}

cached_tokens() {
  jq -er '.usage.prompt_tokens_details.cached_tokens // 0' "$1"
}

metric() {
  local name=$1
  curl --fail-with-body --silent --show-error "$BASE_URL/metrics" |
    awk -v name="$name" '$1 == name { print $2; found=1 } END { if (!found) exit 1 }'
}

context_file="$OUT_DIR/context.txt"
awk -v lines="$CONTEXT_LINES" -v run="$RUN_ID" 'BEGIN {
  for (i = 1; i <= lines; i++)
    printf "anchor-%s line %05d: mutable KV lineage, exact tool history, and Rust cache coherence.\n", run, i
}' >"$context_file"

request_a="$OUT_DIR/a.request.json"
response_a="$OUT_DIR/a.response.json"
request_b="$OUT_DIR/b.request.json"
response_b="$OUT_DIR/b.response.json"
request_b_equal="$OUT_DIR/b-equal.request.json"
response_b_equal="$OUT_DIR/b-equal.response.json"
request_c="$OUT_DIR/c.request.json"
response_c="$OUT_DIR/c.response.json"
request_x="$OUT_DIR/x.request.json"
response_x="$OUT_DIR/x.response.json"
request_old_c="$OUT_DIR/old-c.request.json"
response_old_c="$OUT_DIR/old-c.response.json"

captures_before=$(metric hf2q_qwen_anchor_captures_total)
hits_before=$(metric hf2q_qwen_anchor_restore_hits_total)
pruned_before=$(metric hf2q_qwen_anchor_descendants_pruned_total)
spec_cached_before=$(metric hf2q_qwen_speculation_cached_tokens_total)

jq -n --arg model "$MODEL" --arg run "$RUN_ID" --rawfile context "$context_file" '{
  model: $model,
  messages: [
    {role: "system", content: ("ADR-049 anchor lineage gate " + $run + ". Keep answers concise and deterministic.")},
    {role: "user", content: ("Remember this source context and answer with a short acknowledgement.\n" + $context)}
  ],
  temperature: 0,
  max_tokens: 32,
  stream: false
}' >"$request_a"
post_json "$request_a" "$response_a"
cold_cached=$(cached_tokens "$response_a")
(( cold_cached == 0 )) || {
  echo "lineage gate requires a fresh prompt; A reported $cold_cached cached tokens" >&2
  exit 1
}

jq -n --slurpfile request "$request_a" --slurpfile response "$response_a" '{
  model: $request[0].model,
  messages: ($request[0].messages + [$response[0].choices[0].message,
    {role: "user", content: "Turn B: identify one invariant from the retained source."}]),
  temperature: 0, max_tokens: 32, stream: false
}' >"$request_b"
post_json "$request_b" "$response_b"
b_cached=$(cached_tokens "$response_b")
(( b_cached > 0 )) || { echo "B did not reuse anchor A" >&2; exit 1; }

# Same rendered B prompt with a different response budget bypasses the exact
# generation-result cache while preserving prompt tokens. This is the equality
# restore arm and must reuse deeper than A.
jq '.max_tokens = 33' "$request_b" >"$request_b_equal"
post_json "$request_b_equal" "$response_b_equal"
b_equal_cached=$(cached_tokens "$response_b_equal")
(( b_equal_cached > b_cached )) || {
  echo "equal B prompt did not restore the full B anchor ($b_equal_cached <= $b_cached)" >&2
  exit 1
}

jq -n --slurpfile request "$request_b" --slurpfile response "$response_b" '{
  model: $request[0].model,
  messages: ($request[0].messages + [$response[0].choices[0].message,
    {role: "user", content: "Turn C: give a second distinct invariant."}]),
  temperature: 0, max_tokens: 32, stream: false
}' >"$request_c"
post_json "$request_c" "$response_c"
c_cached=$(cached_tokens "$response_c")
(( c_cached > b_cached )) || {
  echo "C did not reuse a boundary deeper than A ($c_cached <= $b_cached)" >&2
  exit 1
}

# Rewind to A, diverge, and commit branch X. This must prune B and C before X
# writes the physical suffix.
jq -n --slurpfile request "$request_a" --slurpfile response "$response_a" '{
  model: $request[0].model,
  messages: ($request[0].messages + [$response[0].choices[0].message,
    {role: "user", content: "Branch X replaces turns B and C. State the lineage rule."}]),
  temperature: 0, max_tokens: 34, stream: false
}' >"$request_x"
post_json "$request_x" "$response_x"
x_cached=$(cached_tokens "$response_x")
(( x_cached > 0 && x_cached < c_cached )) || {
  echo "branch X did not rewind to a shallower valid anchor (cached=$x_cached old-deep=$c_cached)" >&2
  exit 1
}

request_x_equality_probe="$OUT_DIR/x-equality-probe.request.json"
response_x_equality_probe="$OUT_DIR/x-equality-probe.response.json"
jq '.max_tokens = 36' "$request_x" >"$request_x_equality_probe"
post_json "$request_x_equality_probe" "$response_x_equality_probe"
x_equality_cached=$(cached_tokens "$response_x_equality_probe")
(( x_equality_cached > x_cached )) || {
  echo "X equality restore did not reuse the full divergent boundary" >&2
  exit 1
}

# Retry old C with a changed generation budget. B/C token text still matches,
# but those physical checkpoints are stale after branch X and must not restore.
jq '.max_tokens = 35' "$request_c" >"$request_old_c"
post_json "$request_old_c" "$response_old_c"
old_c_cached=$(cached_tokens "$response_old_c")
(( old_c_cached > 0 && old_c_cached < c_cached )) || {
  echo "stale old-C descendant was reused (retry=$old_c_cached prior-deep=$c_cached)" >&2
  exit 1
}

# Re-establish X after the old-C probe, then start one long owner plus three
# exact siblings. All four clients are submitted concurrently. The siblings
# must remain pending behind the stronger active prefix, survive owner
# cancellation, and complete through the restored pre-request boundary.
request_x_reestablish="$OUT_DIR/x-reestablish.request.json"
response_x_reestablish="$OUT_DIR/x-reestablish.response.json"
jq '.max_tokens = 37' "$request_x" >"$request_x_reestablish"
post_json "$request_x_reestablish" "$response_x_reestablish"
x_reestablish_cached=$(cached_tokens "$response_x_reestablish")
(( x_reestablish_cached > 0 && x_reestablish_cached < c_cached )) || {
  echo "X could not be rebuilt from the surviving A anchor" >&2
  exit 1
}

active_request="$OUT_DIR/active.request.json"
active_stream="$OUT_DIR/active.stream.sse"
jq -n --slurpfile request "$request_x" --slurpfile response "$response_x" \
  --argjson max_tokens "$ACTIVE_MAX_TOKENS" '{
  model: $request[0].model,
  messages: ($request[0].messages + [$response[0].choices[0].message,
    {role: "user", content: "Begin with ACTIVE_ANCHOR_STREAM, then write a long technical analysis of rollback correctness."}]),
  temperature: 0, max_tokens: $max_tokens, stream: true,
  stream_options: {include_usage: true}
}' >"$active_request"

active_pid=""
sibling_pids=()
cleanup() {
  [[ -z "$active_pid" ]] || kill -TERM "$active_pid" 2>/dev/null || true
  for pid in "${sibling_pids[@]:-}"; do
    [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

curl --fail-with-body --silent --show-error --no-buffer \
  --connect-timeout 5 --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary "@$active_request" \
  "$BASE_URL/v1/chat/completions" >"$active_stream" &
active_pid=$!

semantic_ready=0
for ((second = 0; second < SEMANTIC_WAIT_SECONDS; second++)); do
  if sed -n 's/^data: //p' "$active_stream" | jq -e -s '
    any(.[]; type == "object" and (
      ((.choices[0].delta.content // "") | length) > 0 or
      ((.choices[0].delta.reasoning_content // "") | length) > 0))' >/dev/null 2>&1; then
    semantic_ready=1
    break
  fi
  kill -0 "$active_pid" 2>/dev/null || break
  sleep 1
done
(( semantic_ready == 1 )) || { echo "active owner produced no semantic SSE event" >&2; exit 1; }
kill -0 "$active_pid" 2>/dev/null || { echo "active owner ended before overlap" >&2; exit 1; }

for index in 1 2 3; do
  sibling_request="$OUT_DIR/sibling-$index.request.json"
  sibling_response="$OUT_DIR/sibling-$index.response.json"
  jq --argjson max_tokens "$((96 + index))" \
    '.stream = false | del(.stream_options) | .max_tokens = $max_tokens' \
    "$active_request" >"$sibling_request"
  post_json "$sibling_request" "$sibling_response" &
  sibling_pids+=("$!")
done
sleep 1
for pid in "${sibling_pids[@]}"; do
  kill -0 "$pid" 2>/dev/null || {
    echo "a concurrent sibling completed before owner cancellation" >&2
    exit 1
  }
done

kill -TERM "$active_pid" 2>/dev/null || true
wait "$active_pid" 2>/dev/null || true
active_pid=""
rg -q '^data: \[DONE\]$' "$active_stream" && {
  echo "cancelled owner emitted a terminal DONE event" >&2
  exit 1
}
for index in 0 1 2; do
  pid=${sibling_pids[$index]}
  wait "$pid" || { echo "concurrent sibling $((index + 1)) failed" >&2; exit 1; }
  sibling_response="$OUT_DIR/sibling-$((index + 1)).response.json"
  sibling_cached=$(cached_tokens "$sibling_response")
  (( sibling_cached >= x_equality_cached )) || {
    echo "sibling $((index + 1)) went cold after cancellation ($sibling_cached < $x_equality_cached)" >&2
    exit 1
  }
done
sibling_pids=()

# A request too large for the served context must be rejected and must not
# disturb the committed lineage. This is a public-API failed-prefill guard;
# internal GPU-prefill fault injection remains covered by model-free reset
# tests and the worker's full-store-clear path.
overflow_text="$OUT_DIR/overflow.txt"
awk -v words="$OVERFLOW_WORDS" 'BEGIN { for (i = 0; i < words; i++) printf "overflow " }' >"$overflow_text"
overflow_request="$OUT_DIR/overflow.request.json"
overflow_response="$OUT_DIR/overflow.response.json"
jq -n --arg model "$MODEL" --rawfile text "$overflow_text" '{
  model: $model, messages: [{role: "user", content: $text}],
  temperature: 0, max_tokens: 32, stream: false
}' >"$overflow_request"
overflow_status=$(curl --silent --show-error --output "$overflow_response" --write-out '%{http_code}' \
  --connect-timeout 5 --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary "@$overflow_request" \
  "$BASE_URL/v1/chat/completions")
[[ "$overflow_status" =~ ^4[0-9][0-9]$ ]] || {
  echo "oversized prefill did not fail closed (HTTP $overflow_status)" >&2
  exit 1
}

post_failure_request="$OUT_DIR/post-failure.request.json"
post_failure_response="$OUT_DIR/post-failure.response.json"
jq '.max_tokens = 38' "$request_x" >"$post_failure_request"
post_json "$post_failure_request" "$post_failure_response"
post_failure_cached=$(cached_tokens "$post_failure_response")
(( post_failure_cached > 0 )) || {
  echo "committed anchor was lost after rejected oversized prefill" >&2
  exit 1
}

captures_after=$(metric hf2q_qwen_anchor_captures_total)
hits_after=$(metric hf2q_qwen_anchor_restore_hits_total)
pruned_after=$(metric hf2q_qwen_anchor_descendants_pruned_total)
spec_cached_after=$(metric hf2q_qwen_speculation_cached_tokens_total)

(( captures_after > captures_before )) || { echo "anchor capture telemetry did not move" >&2; exit 1; }
(( hits_after >= hits_before + 4 )) || { echo "too few anchor restore hits" >&2; exit 1; }
(( pruned_after >= pruned_before + 2 )) || { echo "descendant-prune telemetry did not prove B/C invalidation" >&2; exit 1; }
(( spec_cached_after > spec_cached_before )) || {
  echo "speculative cached-state carry did not execute" >&2
  exit 1
}

summary="$OUT_DIR/summary.json"
jq -n \
  --arg status pass --arg model "$MODEL" --arg run_id "$RUN_ID" \
  --argjson a_to_b_cached "$b_cached" \
  --argjson b_equality_cached "$b_equal_cached" \
  --argjson b_to_c_cached "$c_cached" \
  --argjson branch_x_cached "$x_cached" \
  --argjson stale_old_c_cached "$old_c_cached" \
  --argjson cancellation_siblings 3 \
  --argjson rejected_prefill_http "$overflow_status" \
  --argjson restore_hits_delta "$((hits_after - hits_before))" \
  --argjson descendants_pruned_delta "$((pruned_after - pruned_before))" \
  --argjson speculative_cached_tokens_delta "$((spec_cached_after - spec_cached_before))" '{
    status: $status, model: $model, run_id: $run_id,
    a_to_b_cached: $a_to_b_cached,
    b_equality_cached: $b_equality_cached,
    b_to_c_cached: $b_to_c_cached,
    branch_x_cached: $branch_x_cached,
    stale_old_c_cached: $stale_old_c_cached,
    concurrent_clients: 4,
    cancellation_siblings: $cancellation_siblings,
    rejected_prefill_http: $rejected_prefill_http,
    restore_hits_delta: $restore_hits_delta,
    descendants_pruned_delta: $descendants_pruned_delta,
    speculative_cached_tokens_delta: $speculative_cached_tokens_delta
  }' >"$summary"

jq . "$summary"
echo "Qwen SlotAware anchor gate artifacts: $OUT_DIR" >&2
