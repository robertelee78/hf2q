#!/usr/bin/env bash
set -euo pipefail

# ADR-049 Lane A fail-closed real-model gate for the model-neutral anchor
# contract implemented by Gemma 4 and DeepSeek-V4. Run against an already
# running four-slot server started with:
#
#   HF2Q_UNSAFE_EXPERIMENTS=1
#   HF2Q_TEST_ANCHOR_RESTORE_FAILURE_MAX_TOKENS=47
#
# The gate proves multi-depth reuse, equality reuse, stale-descendant pruning,
# four-client cancellation recovery, fail-closed restore rollback, exact
# semantic recovery, and byte-budget observability. Family-specific payload
# layouts remain covered by their Rust invariant batteries.

BASE_URL=${BASE_URL:-http://127.0.0.1:8082}
FAMILY=${FAMILY:-}
MODEL=${MODEL:-}
SERVER_PID=${SERVER_PID:-}
OUT_DIR=${OUT_DIR:-$(mktemp -d /var/tmp/hf2q-family-anchor.XXXXXX)}
RUN_ID=${RUN_ID:-"$$-$(date +%s)"}
EXPECTED_MAX_SLOTS=${EXPECTED_MAX_SLOTS:-4}
CONTEXT_LINES=${CONTEXT_LINES:-512}
ACTIVE_MAX_TOKENS=${ACTIVE_MAX_TOKENS:-512}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-900}
SEMANTIC_WAIT_SECONDS=${SEMANTIC_WAIT_SECONDS:-180}
RESTORE_FAILURE_MAX_TOKENS=${RESTORE_FAILURE_MAX_TOKENS:-47}

for command in awk curl jq lsof ps rg sed; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
for setting in EXPECTED_MAX_SLOTS CONTEXT_LINES ACTIVE_MAX_TOKENS \
  CURL_MAX_TIME_SECONDS SEMANTIC_WAIT_SECONDS RESTORE_FAILURE_MAX_TOKENS; do
  value=${!setting}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$setting must be a positive integer (got: $value)" >&2
    exit 2
  }
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
[[ " $server_command " == *" --scheduler inflight-batched "* ]] || {
  echo "server is not using --scheduler inflight-batched: $server_command" >&2
  exit 2
}
[[ " $server_command " == *" --max-slots $EXPECTED_MAX_SLOTS "* ]] || {
  echo "server is not using exact --max-slots $EXPECTED_MAX_SLOTS: $server_command" >&2
  exit 2
}

models_response=$(curl --fail-with-body --silent --show-error "$BASE_URL/v1/models")
if [[ -z "$MODEL" ]]; then
  MODEL=$(jq -er \
    '[.data[] | select((.arch // "") != "" and .loaded == true)] | if length == 1 then .[0].id else error("expected one loaded inference model") end' \
    <<<"$models_response")
fi
served_arch=$(jq -er --arg model "$MODEL" \
  '[.data[] | select(.id == $model and .loaded == true)] | if length == 1 then .[0].arch else error("expected one matching loaded model") end' \
  <<<"$models_response")
if [[ -z "$FAMILY" ]]; then
  FAMILY=$served_arch
fi
case "$FAMILY" in
  gemma4)
    metric_prefix=hf2q_gemma4_anchor
    ;;
  deepseek4)
    metric_prefix=hf2q_deepseek4_anchor
    ;;
  *)
    echo "FAMILY must be gemma4 or deepseek4 (got: $FAMILY)" >&2
    exit 2
    ;;
esac
[[ "$served_arch" == "$FAMILY" ]] || {
  echo "served architecture $served_arch does not match requested family $FAMILY" >&2
  exit 2
}

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
    printf "anchor-%s line %05d: exact cache lineage, tool history, and transactional state.\n", run, i
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

captures_before=$(metric "${metric_prefix}_captures_total")
hits_before=$(metric "${metric_prefix}_restore_hits_total")
misses_before=$(metric "${metric_prefix}_restore_misses_total")
pruned_before=$(metric "${metric_prefix}_descendants_pruned_total")
cancellations_before=$(metric "${metric_prefix}_cancellations_total")
lineage_clears_before=$(metric "${metric_prefix}_lineage_clears_total")
sse_cancellations_before=$(metric hf2q_sse_cancellations)

jq -n --arg model "$MODEL" --arg run "$RUN_ID" --rawfile context "$context_file" '{
  model: $model,
  messages: [
    {role: "system", content: ("ADR-049 family anchor gate " + $run + ". Be concise and deterministic.")},
    {role: "user", content: ("Retain this source context and acknowledge it briefly.\n" + $context)}
  ],
  temperature: 0, max_tokens: 32, stream: false
}' >"$request_a"
post_json "$request_a" "$response_a"
a_cached=$(cached_tokens "$response_a")
(( a_cached == 0 )) || { echo "A was not a fresh prompt ($a_cached cached tokens)" >&2; exit 1; }

jq -n --slurpfile request "$request_a" --slurpfile response "$response_a" '{
  model: $request[0].model,
  messages: ($request[0].messages + [$response[0].choices[0].message,
    {role: "user", content: "Turn B: state one invariant from the retained source."}]),
  temperature: 0, max_tokens: 32, stream: false
}' >"$request_b"
post_json "$request_b" "$response_b"
b_cached=$(cached_tokens "$response_b")
(( b_cached > 0 )) || { echo "B did not reuse anchor A" >&2; exit 1; }

jq '.max_tokens = 33' "$request_b" >"$request_b_equal"
post_json "$request_b_equal" "$response_b_equal"
b_equal_cached=$(cached_tokens "$response_b_equal")
(( b_equal_cached > b_cached )) || {
  echo "equal B prompt did not reuse the full B boundary ($b_equal_cached <= $b_cached)" >&2
  exit 1
}

jq -n --slurpfile request "$request_b" --slurpfile response "$response_b" '{
  model: $request[0].model,
  messages: ($request[0].messages + [$response[0].choices[0].message,
    {role: "user", content: "Turn C: state a second distinct invariant."}]),
  temperature: 0, max_tokens: 32, stream: false
}' >"$request_c"
post_json "$request_c" "$response_c"
c_cached=$(cached_tokens "$response_c")
(( c_cached > b_cached )) || {
  echo "C did not reuse a boundary deeper than A ($c_cached <= $b_cached)" >&2
  exit 1
}

jq -n --slurpfile request "$request_a" --slurpfile response "$response_a" '{
  model: $request[0].model,
  messages: ($request[0].messages + [$response[0].choices[0].message,
    {role: "user", content: "Branch X replaces turns B and C. State the lineage rule."}]),
  temperature: 0, max_tokens: 34, stream: false
}' >"$request_x"
post_json "$request_x" "$response_x"
x_cached=$(cached_tokens "$response_x")
(( x_cached > 0 && x_cached < c_cached )) || {
  echo "X did not rewind to a shallower valid anchor (cached=$x_cached old-deep=$c_cached)" >&2
  exit 1
}

jq '.max_tokens = 35' "$request_c" >"$request_old_c"
post_json "$request_old_c" "$response_old_c"
old_c_cached=$(cached_tokens "$response_old_c")
(( old_c_cached == 0 || old_c_cached == x_cached )) || {
  echo "stale old-C descendant was reused (retry=$old_c_cached surviving-A=$x_cached prior-deep=$c_cached)" >&2
  exit 1
}

# Rebuild X, then overlap one streaming owner with three exact siblings. The
# owner is cancelled only after a semantic event, and every sibling must
# complete with a restored stable boundary.
request_x_rebuild="$OUT_DIR/x-rebuild.request.json"
response_x_rebuild="$OUT_DIR/x-rebuild.response.json"
jq '.max_tokens = 36' "$request_x" >"$request_x_rebuild"
post_json "$request_x_rebuild" "$response_x_rebuild"
x_rebuild_cached=$(cached_tokens "$response_x_rebuild")
(( x_rebuild_cached > 0 )) || { echo "X could not be rebuilt" >&2; exit 1; }

active_request="$OUT_DIR/active.request.json"
active_stream="$OUT_DIR/active.stream.sse"
jq -n --slurpfile request "$request_x" --slurpfile response "$response_x" \
  --argjson max_tokens "$ACTIVE_MAX_TOKENS" '{
  model: $request[0].model,
  messages: ($request[0].messages + [$response[0].choices[0].message,
    {role: "user", content: "Begin with ACTIVE_ANCHOR_STREAM, then analyze transactional rollback at length."}]),
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
kill -TERM "$active_pid" 2>/dev/null || true
wait "$active_pid" 2>/dev/null || true
active_pid=""
rg -q '^data: \[DONE\]$' "$active_stream" && {
  echo "cancelled owner emitted a terminal DONE event" >&2
  exit 1
}
affinity_siblings=0
cold_siblings=0
for index in 0 1 2; do
  pid=${sibling_pids[$index]}
  wait "$pid" || { echo "concurrent sibling $((index + 1)) failed" >&2; exit 1; }
  sibling_cached=$(cached_tokens "$OUT_DIR/sibling-$((index + 1)).response.json")
  if (( sibling_cached >= x_cached )); then
    affinity_siblings=$((affinity_siblings + 1))
  else
    cold_siblings=$((cold_siblings + 1))
  fi
done
sibling_pids=()
(( affinity_siblings >= 1 )) || {
  echo "no sibling inherited the cancelled owner's stable boundary" >&2
  exit 1
}

# Establish a fresh boundary, diverge from it, then trigger the one-shot fault
# on a second branch. The family must hard-reset, clear every anchor, stay
# ready, reproduce the same semantic response cold, and rebuild reuse.
failure_base_request="$OUT_DIR/failure-base.request.json"
failure_base_response="$OUT_DIR/failure-base.response.json"
jq -n --arg model "$MODEL" --arg run "$RUN_ID" '{
  model: $model,
  messages: [{role: "user", content: ("Recovery base " + $run + ". Reply exactly RECOVERY_BASE.")}],
  temperature: 0, max_tokens: 48, stream: false
}' >"$failure_base_request"
post_json "$failure_base_request" "$failure_base_response"
failure_base_content=$(jq -er '.choices[0].message.content' "$failure_base_response")

failure_branch_request="$OUT_DIR/failure-branch.request.json"
failure_branch_response="$OUT_DIR/failure-branch.response.json"
jq -n --slurpfile request "$failure_base_request" --slurpfile response "$failure_base_response" '{
  model: $request[0].model,
  messages: ($request[0].messages + [$response[0].choices[0].message,
    {role: "user", content: "First branch. Reply exactly FIRST_BRANCH."}]),
  temperature: 0, max_tokens: 46, stream: false
}' >"$failure_branch_request"
post_json "$failure_branch_request" "$failure_branch_response"

failure_request="$OUT_DIR/restore-failure.request.json"
failure_response="$OUT_DIR/restore-failure.response.json"
jq -n --slurpfile request "$failure_base_request" --slurpfile response "$failure_base_response" \
  --argjson max_tokens "$RESTORE_FAILURE_MAX_TOKENS" '{
  model: $request[0].model,
  messages: ($request[0].messages + [$response[0].choices[0].message,
    {role: "user", content: "Second branch. Reply exactly RECOVERY_STABLE."}]),
  temperature: 0, max_tokens: $max_tokens, stream: false
}' >"$failure_request"
failure_status=$(curl --silent --show-error --output "$failure_response" --write-out '%{http_code}' \
  --connect-timeout 5 --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary "@$failure_request" \
  "$BASE_URL/v1/chat/completions")
[[ "$failure_status" =~ ^5[0-9][0-9]$ ]] || {
  echo "restore fault did not return 5xx (HTTP $failure_status)" >&2
  exit 1
}
curl --fail-with-body --silent --show-error "$BASE_URL/readyz" >/dev/null || {
  echo "$FAMILY worker was not ready after restore failure" >&2
  exit 1
}

cold_recovery_request="$OUT_DIR/cold-recovery.request.json"
cold_recovery_response="$OUT_DIR/cold-recovery.response.json"
jq '.max_tokens = 49' "$failure_request" >"$cold_recovery_request"
post_json "$cold_recovery_request" "$cold_recovery_response"
cold_recovery_cached=$(cached_tokens "$cold_recovery_response")
(( cold_recovery_cached == 0 )) || {
  echo "failed restore retained stale cache state ($cold_recovery_cached cached tokens)" >&2
  exit 1
}
cold_recovery_content=$(jq -er '.choices[0].message.content' "$cold_recovery_response")

reuse_recovery_request="$OUT_DIR/reuse-recovery.request.json"
reuse_recovery_response="$OUT_DIR/reuse-recovery.response.json"
jq '.max_tokens = 50' "$failure_request" >"$reuse_recovery_request"
post_json "$reuse_recovery_request" "$reuse_recovery_response"
reuse_recovery_cached=$(cached_tokens "$reuse_recovery_response")
(( reuse_recovery_cached > 0 )) || { echo "recovery boundary did not rebuild reuse" >&2; exit 1; }
reuse_recovery_content=$(jq -er '.choices[0].message.content' "$reuse_recovery_response")
[[ "$cold_recovery_content" == "$reuse_recovery_content" ]] || {
  echo "cold and rebuilt recovery responses differ" >&2
  exit 1
}

captures_after=$(metric "${metric_prefix}_captures_total")
hits_after=$(metric "${metric_prefix}_restore_hits_total")
misses_after=$(metric "${metric_prefix}_restore_misses_total")
pruned_after=$(metric "${metric_prefix}_descendants_pruned_total")
cancellations_after=$(metric "${metric_prefix}_cancellations_total")
lineage_clears_after=$(metric "${metric_prefix}_lineage_clears_total")
sse_cancellations_after=$(metric hf2q_sse_cancellations)
effective_depth=$(metric "${metric_prefix}_effective_committed_depth")
pending_capacity=$(metric "${metric_prefix}_simultaneous_pending_capacity_slots")
configured_slots=$(metric "${metric_prefix}_configured_slots")
aggregate_budget=$(metric "${metric_prefix}_aggregate_budget_bytes")
aggregate_peak=$(metric "${metric_prefix}_aggregate_peak_committed_pending_bytes")

(( captures_after > captures_before )) || { echo "capture telemetry did not move" >&2; exit 1; }
(( hits_after >= hits_before + 5 )) || { echo "too few restore hits" >&2; exit 1; }
(( misses_after >= misses_before + 1 )) || { echo "fault did not record a restore miss" >&2; exit 1; }
(( pruned_after >= pruned_before + 2 )) || { echo "stale descendants were not pruned" >&2; exit 1; }
(( cancellations_after >= cancellations_before + 1 )) || { echo "anchor cancellation did not advance" >&2; exit 1; }
(( sse_cancellations_after >= sse_cancellations_before + 1 )) || { echo "SSE cancellation did not advance" >&2; exit 1; }
(( lineage_clears_after >= lineage_clears_before + 1 )) || { echo "failed restore did not clear lineage" >&2; exit 1; }
(( configured_slots == EXPECTED_MAX_SLOTS )) || { echo "configured slot gauge mismatch" >&2; exit 1; }
(( effective_depth > 0 && effective_depth <= 4 )) || { echo "invalid effective depth: $effective_depth" >&2; exit 1; }
(( pending_capacity >= 0 && pending_capacity <= configured_slots )) || { echo "invalid pending capacity: $pending_capacity" >&2; exit 1; }
(( aggregate_peak <= aggregate_budget )) || { echo "anchor peak exceeded aggregate budget" >&2; exit 1; }

summary="$OUT_DIR/summary.json"
jq -n \
  --arg status pass --arg family "$FAMILY" --arg model "$MODEL" --arg run_id "$RUN_ID" \
  --arg failure_base_content "$failure_base_content" \
  --arg recovery_content "$cold_recovery_content" \
  --argjson a_to_b_cached "$b_cached" \
  --argjson b_equality_cached "$b_equal_cached" \
  --argjson b_to_c_cached "$c_cached" \
  --argjson branch_x_cached "$x_cached" \
  --argjson stale_old_c_cached "$old_c_cached" \
  --argjson affinity_siblings "$affinity_siblings" \
  --argjson cold_siblings "$cold_siblings" \
  --argjson restore_failure_http "$failure_status" \
  --argjson cold_recovery_cached "$cold_recovery_cached" \
  --argjson rebuilt_recovery_cached "$reuse_recovery_cached" \
  --argjson restore_hits_delta "$((hits_after - hits_before))" \
  --argjson restore_misses_delta "$((misses_after - misses_before))" \
  --argjson descendants_pruned_delta "$((pruned_after - pruned_before))" \
  --argjson cancellations_delta "$((cancellations_after - cancellations_before))" \
  --argjson effective_committed_depth "$effective_depth" \
  --argjson simultaneous_pending_capacity_slots "$pending_capacity" \
  --argjson configured_slots "$configured_slots" \
  --argjson aggregate_anchor_budget_bytes "$aggregate_budget" \
  --argjson aggregate_anchor_peak_bytes "$aggregate_peak" '{
    status: $status, family: $family, model: $model, run_id: $run_id,
    a_to_b_cached: $a_to_b_cached,
    b_equality_cached: $b_equality_cached,
    b_to_c_cached: $b_to_c_cached,
    branch_x_cached: $branch_x_cached,
    stale_old_c_cached: $stale_old_c_cached,
    concurrent_clients: 4,
    cancellation_siblings: 3,
    affinity_siblings: $affinity_siblings,
    independent_cold_siblings: $cold_siblings,
    restore_failure_http: $restore_failure_http,
    cold_recovery_cached_tokens: $cold_recovery_cached,
    rebuilt_recovery_cached_tokens: $rebuilt_recovery_cached,
    semantic_recovery_equal: true,
    failure_base_content: $failure_base_content,
    recovery_content: $recovery_content,
    restore_hits_delta: $restore_hits_delta,
    restore_misses_delta: $restore_misses_delta,
    descendants_pruned_delta: $descendants_pruned_delta,
    cancellations_delta: $cancellations_delta,
    configured_slots: $configured_slots,
    effective_committed_depth: $effective_committed_depth,
    simultaneous_pending_capacity_slots: $simultaneous_pending_capacity_slots,
    full_pending_concurrency_available: ($simultaneous_pending_capacity_slots == $configured_slots),
    full_depth_available: ($effective_committed_depth == 4),
    aggregate_anchor_budget_bytes: $aggregate_anchor_budget_bytes,
    aggregate_anchor_peak_bytes: $aggregate_anchor_peak_bytes
  }' >"$summary"

jq . "$summary"
echo "$FAMILY SlotAware anchor gate artifacts: $OUT_DIR" >&2
