#!/usr/bin/env bash
set -euo pipefail

# ADR-049 Lane A fail-closed hardware gate for an already-running Qwen
# SlotAware server. Run once per served artifact. The script proves the server
# process really uses `--scheduler inflight-batched` and the expected slot
# count, builds an A->B->C checkpoint lineage, rewinds to A for branch X, and rejects reuse of
# stale B/C descendants. It also overlaps four clients, cancels the owner,
# verifies rollback reuse, and requires speculative cached-state reuse when
# the served artifact exposes an MTP boundary payload. Start the server with both
# `HF2Q_UNSAFE_EXPERIMENTS=1` and
# `HF2Q_TEST_QWEN_POST_ADMISSION_PREFILL_FAILURE_MAX_TOKENS=39` (or the
# matching value below): the gate consumes that one-shot fault only after a
# real, non-empty GPU prefill slice succeeds.

BASE_URL=${BASE_URL:-http://127.0.0.1:8081}
MODEL=${MODEL:-}
SERVER_PID=${SERVER_PID:-}
OUT_DIR=${OUT_DIR:-$(mktemp -d /var/tmp/hf2q-qwen-anchor.XXXXXX)}
RUN_ID=${RUN_ID:-"$$-$(date +%s)"}
CONTEXT_LINES=${CONTEXT_LINES:-512}
ACTIVE_MAX_TOKENS=${ACTIVE_MAX_TOKENS:-512}
OVERFLOW_WORDS=${OVERFLOW_WORDS:-300000}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-900}
SEMANTIC_WAIT_SECONDS=${SEMANTIC_WAIT_SECONDS:-180}
EXPECTED_MAX_SLOTS=${EXPECTED_MAX_SLOTS:-4}
POST_ADMISSION_FAILURE_MAX_TOKENS=${POST_ADMISSION_FAILURE_MAX_TOKENS:-39}

for command in awk curl jq lsof ps rg sed; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
for setting in CONTEXT_LINES ACTIVE_MAX_TOKENS OVERFLOW_WORDS \
  CURL_MAX_TIME_SECONDS SEMANTIC_WAIT_SECONDS EXPECTED_MAX_SLOTS \
  POST_ADMISSION_FAILURE_MAX_TOKENS; do
  value=${!setting}
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$setting must be a positive integer (got: $value)" >&2
    exit 2
  fi
done
(( POST_ADMISSION_FAILURE_MAX_TOKENS >= 39 && POST_ADMISSION_FAILURE_MAX_TOKENS <= 96 )) || {
  echo "POST_ADMISSION_FAILURE_MAX_TOKENS must be in [39, 96] to avoid earlier gate requests" >&2
  exit 2
}
(( POST_ADMISSION_FAILURE_MAX_TOKENS != ACTIVE_MAX_TOKENS )) || {
  echo "POST_ADMISSION_FAILURE_MAX_TOKENS must differ from ACTIVE_MAX_TOKENS" >&2
  exit 2
}

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
  echo "server is not provably using --scheduler inflight-batched: $server_command" >&2
  exit 2
}
[[ " $server_command " == *" --max-slots $EXPECTED_MAX_SLOTS "* ]] || {
  echo "server is not provably using exact --max-slots $EXPECTED_MAX_SLOTS: $server_command" >&2
  exit 2
}

models_response=$(curl --fail-with-body --silent --show-error "$BASE_URL/v1/models")
if [[ -z "$MODEL" ]]; then
  MODEL=$(jq -er \
    '[.data[] | select((.arch // "") != "" and .loaded == true)] | if length == 1 then .[0].id else error("expected one loaded inference model") end' \
    <<<"$models_response")
fi
model_context=$(jq -er --arg model "$MODEL" \
  '[.data[] | select(.id == $model and .loaded == true)] | if length == 1 then (.[0].context_length // error("loaded model has no effective context length")) else error("expected one matching loaded model") end' \
  <<<"$models_response")
(( OVERFLOW_WORDS > model_context )) || {
  echo "OVERFLOW_WORDS=$OVERFLOW_WORDS cannot prove an oversized request for model context_length=$model_context" >&2
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
spec_anchor_cached_before=$(metric hf2q_qwen_anchor_spec_boundary_restore_tokens_total)
spec_anchor_capable=$(metric hf2q_qwen_anchor_spec_boundary_capable)
(( spec_anchor_capable == 0 || spec_anchor_capable == 1 )) || {
  echo "invalid speculative anchor capability gauge: $spec_anchor_capable" >&2
  exit 1
}
post_admission_failures_before=$(metric hf2q_qwen_anchor_post_admission_prefill_failures_total)

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
(( old_c_cached == 0 || old_c_cached == x_cached )) || {
  echo "stale old-C descendant was reused (retry=$old_c_cached surviving-A=$x_cached prior-deep=$c_cached)" >&2
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
# disturb the committed lineage. This is an admission-isolation check only;
# it deliberately does not claim to exercise a post-admission GPU failure.
overflow_text="$OUT_DIR/overflow.txt"
# Keep the transport body well below Axum's default JSON limit so this proves
# the model's tokenized context rejection rather than an earlier HTTP 413.
# `x ` is one token for the Qwen artifacts covered by this gate and is much
# denser than the former `overflow ` fixture.
awk -v words="$OVERFLOW_WORDS" 'BEGIN { for (i = 0; i < words; i++) printf "x " }' >"$overflow_text"
overflow_request="$OUT_DIR/overflow.request.json"
overflow_response="$OUT_DIR/overflow.response.json"
jq -n --arg model "$MODEL" --rawfile text "$overflow_text" '{
  model: $model, messages: [{role: "user", content: $text}],
  temperature: 0, max_tokens: 32, stream: false
}' >"$overflow_request"
overflow_request_bytes=$(wc -c <"$overflow_request")
(( overflow_request_bytes < 1048576 )) || {
  echo "overflow fixture is too large to isolate model context admission (${overflow_request_bytes} bytes)" >&2
  exit 2
}
overflow_status=$(curl --silent --show-error --output "$overflow_response" --write-out '%{http_code}' \
  --connect-timeout 5 --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary "@$overflow_request" \
  "$BASE_URL/v1/chat/completions")
[[ "$overflow_status" == 400 ]] || {
  echo "oversized prefill did not return the exact context rejection (HTTP $overflow_status)" >&2
  exit 1
}
jq -e --argjson context "$model_context" '
  .error.type == "invalid_request_error"
  and .error.code == "context_length_exceeded"
  and .error.param == "messages"
  and (.error.message
    | capture("^This model\u0027s maximum context length is (?<max>[0-9]+) tokens[.] However, your messages resulted in (?<actual>[0-9]+) tokens[.]$")
    | ((.max | tonumber) == $context and (.actual | tonumber) >= $context))
' "$overflow_response" >/dev/null || {
  echo "oversized request was not rejected after exact tokenized context accounting" >&2
  cat "$overflow_response" >&2
  exit 1
}

post_rejected_admission_request="$OUT_DIR/post-rejected-admission.request.json"
post_rejected_admission_response="$OUT_DIR/post-rejected-admission.response.json"
jq '.max_tokens = 38' "$request_x" >"$post_rejected_admission_request"
post_json "$post_rejected_admission_request" "$post_rejected_admission_response"
post_rejected_admission_cached=$(cached_tokens "$post_rejected_admission_response")
(( post_rejected_admission_cached > 0 )) || {
  echo "committed anchor was lost after rejected oversized prefill" >&2
  exit 1
}

# Establish a boundary unique to one slot, then extend it with the one-shot
# failure budget. The injected error occurs only after state.advance() returns
# a successful, non-empty GPU slice and before scheduler publication. The
# ordinary failed-slice path must therefore clear both the retained ledger and
# the complete AnchorStore, hard-reset the physical slot, release it, and keep
# the worker ready. A cold retry of the unique prior boundary proves stale
# anchors did not survive; a second retry proves the recovered slot can anchor
# and reuse again.
failure_base_request="$OUT_DIR/post-admission-base.request.json"
failure_base_response="$OUT_DIR/post-admission-base.response.json"
jq -n --slurpfile request "$request_x" --slurpfile response "$response_x" \
  --argjson max_tokens "$((POST_ADMISSION_FAILURE_MAX_TOKENS + 1))" '{
  model: $request[0].model,
  messages: ($request[0].messages + [$response[0].choices[0].message,
    {role: "user", content: "ADR-049 unique failed-prefill recovery base. Reply briefly."}]),
  temperature: 0, max_tokens: $max_tokens, stream: false
}' >"$failure_base_request"
post_json "$failure_base_request" "$failure_base_response"
failure_base_cached=$(cached_tokens "$failure_base_response")
(( failure_base_cached > 0 )) || {
  echo "post-admission failure base did not reuse branch X" >&2
  exit 1
}

post_admission_failure_request="$OUT_DIR/post-admission-failure.request.json"
post_admission_failure_response="$OUT_DIR/post-admission-failure.response.json"
jq -n --slurpfile request "$failure_base_request" \
  --slurpfile response "$failure_base_response" \
  --argjson max_tokens "$POST_ADMISSION_FAILURE_MAX_TOKENS" '{
  model: $request[0].model,
  messages: ($request[0].messages + [$response[0].choices[0].message,
    {role: "user", content: "This admitted suffix must fail only after its GPU prefill slice succeeds."}]),
  temperature: 0, max_tokens: $max_tokens, stream: false
}' >"$post_admission_failure_request"
compound_prefills_before_fault=$(
  metric hf2q_qwen_anchor_stable_boundary_compound_prefills_total
)
post_admission_failure_status=$(curl --silent --show-error \
  --output "$post_admission_failure_response" --write-out '%{http_code}' \
  --connect-timeout 5 --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary "@$post_admission_failure_request" \
  "$BASE_URL/v1/chat/completions")
[[ "$post_admission_failure_status" =~ ^5[0-9][0-9]$ ]] || {
  echo "post-admission prefill fault did not return 5xx (HTTP $post_admission_failure_status)" >&2
  exit 1
}
post_admission_failures_after_fault=$(
  metric hf2q_qwen_anchor_post_admission_prefill_failures_total
)
(( post_admission_failures_after_fault == post_admission_failures_before + 1 )) || {
  echo "post-admission prefill fault counter did not advance exactly once" >&2
  exit 1
}
compound_prefills_after_fault=$(
  metric hf2q_qwen_anchor_stable_boundary_compound_prefills_total
)
(( compound_prefills_after_fault == compound_prefills_before_fault + 1 )) || {
  echo "post-admission fault did not follow exactly one successful compound prefill" >&2
  exit 1
}
curl --fail-with-body --silent --show-error "$BASE_URL/readyz" >/dev/null || {
  echo "Qwen worker was not ready after recoverable post-admission prefill failure" >&2
  exit 1
}

post_failure_cold_request="$OUT_DIR/post-admission-cold-recovery.request.json"
post_failure_cold_response="$OUT_DIR/post-admission-cold-recovery.response.json"
jq --argjson max_tokens "$((POST_ADMISSION_FAILURE_MAX_TOKENS + 2))" \
  '.max_tokens = $max_tokens' "$failure_base_request" >"$post_failure_cold_request"
post_json "$post_failure_cold_request" "$post_failure_cold_response"
post_failure_cold_cached=$(cached_tokens "$post_failure_cold_response")
(( post_failure_cold_cached == 0 )) || {
  echo "failed-prefill slot retained stale cache state ($post_failure_cold_cached cached tokens)" >&2
  exit 1
}

post_failure_reuse_request="$OUT_DIR/post-admission-reuse.request.json"
post_failure_reuse_response="$OUT_DIR/post-admission-reuse.response.json"
jq --argjson max_tokens "$((POST_ADMISSION_FAILURE_MAX_TOKENS + 3))" \
  '.max_tokens = $max_tokens' "$failure_base_request" >"$post_failure_reuse_request"
post_json "$post_failure_reuse_request" "$post_failure_reuse_response"
post_failure_reuse_cached=$(cached_tokens "$post_failure_reuse_response")
(( post_failure_reuse_cached > 0 )) || {
  echo "slot did not rebuild and reuse an anchor after failed-prefill reset" >&2
  exit 1
}

captures_after=$(metric hf2q_qwen_anchor_captures_total)
hits_after=$(metric hf2q_qwen_anchor_restore_hits_total)
pruned_after=$(metric hf2q_qwen_anchor_descendants_pruned_total)
spec_cached_after=$(metric hf2q_qwen_speculation_cached_tokens_total)
spec_anchor_cached_after=$(metric hf2q_qwen_anchor_spec_boundary_restore_tokens_total)
effective_depth=$(metric hf2q_qwen_anchor_effective_committed_depth)
pending_capacity_slots=$(metric hf2q_qwen_anchor_simultaneous_pending_capacity_slots)
configured_slots=$(metric hf2q_qwen_anchor_configured_slots)
aggregate_anchor_budget=$(metric hf2q_qwen_anchor_aggregate_budget_bytes)
aggregate_anchor_peak=$(metric hf2q_qwen_anchor_aggregate_peak_committed_pending_bytes)
post_admission_failures_after=$(metric hf2q_qwen_anchor_post_admission_prefill_failures_total)

(( captures_after > captures_before )) || { echo "anchor capture telemetry did not move" >&2; exit 1; }
(( hits_after >= hits_before + 4 )) || { echo "too few anchor restore hits" >&2; exit 1; }
(( pruned_after >= pruned_before + 2 )) || { echo "descendant-prune telemetry did not prove B/C invalidation" >&2; exit 1; }
if (( spec_anchor_capable == 1 )); then
  (( spec_anchor_cached_after > spec_anchor_cached_before )) || {
    echo "MTP-capable artifact did not execute SlotAware speculative anchor-state carry" >&2
    exit 1
  }
else
  (( spec_anchor_cached_after == spec_anchor_cached_before )) || {
    echo "non-MTP artifact reported impossible speculative anchor-state carry" >&2
    exit 1
  }
fi
(( configured_slots == EXPECTED_MAX_SLOTS )) || {
  echo "anchor telemetry slot count $configured_slots != expected $EXPECTED_MAX_SLOTS" >&2
  exit 1
}
(( effective_depth > 0 && effective_depth <= 4 )) || {
  echo "anchor effective committed depth is invalid: $effective_depth" >&2
  exit 1
}
(( pending_capacity_slots >= 0 && pending_capacity_slots <= EXPECTED_MAX_SLOTS )) || {
  echo "anchor simultaneous pending capacity is invalid: $pending_capacity_slots" >&2
  exit 1
}
(( aggregate_anchor_peak <= aggregate_anchor_budget )) || {
  echo "aggregate anchor peak exceeded budget ($aggregate_anchor_peak > $aggregate_anchor_budget)" >&2
  exit 1
}
(( post_admission_failures_after == post_admission_failures_before + 1 )) || {
  echo "post-admission prefill fault fired more than once" >&2
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
  --argjson post_admission_failed_prefill_http "$post_admission_failure_status" \
  --argjson post_admission_failed_prefill_faults_delta "$((post_admission_failures_after - post_admission_failures_before))" \
  --argjson post_admission_compound_prefills_delta "$((compound_prefills_after_fault - compound_prefills_before_fault))" \
  --argjson post_failure_cold_cached_tokens "$post_failure_cold_cached" \
  --argjson post_failure_reuse_cached_tokens "$post_failure_reuse_cached" \
  --argjson restore_hits_delta "$((hits_after - hits_before))" \
  --argjson descendants_pruned_delta "$((pruned_after - pruned_before))" \
  --argjson speculative_cached_tokens_delta "$((spec_cached_after - spec_cached_before))" \
  --argjson speculative_anchor_capable "$spec_anchor_capable" \
  --argjson slotaware_spec_anchor_tokens_delta "$((spec_anchor_cached_after - spec_anchor_cached_before))" \
  --argjson configured_slots "$configured_slots" \
  --argjson effective_committed_depth "$effective_depth" \
  --argjson pending_capacity_slots "$pending_capacity_slots" \
  --argjson aggregate_anchor_budget_bytes "$aggregate_anchor_budget" \
  --argjson aggregate_anchor_peak_bytes "$aggregate_anchor_peak" '{
    status: $status, model: $model, run_id: $run_id,
    a_to_b_cached: $a_to_b_cached,
    b_equality_cached: $b_equality_cached,
    b_to_c_cached: $b_to_c_cached,
    branch_x_cached: $branch_x_cached,
    stale_old_c_cached: $stale_old_c_cached,
    concurrent_clients: 4,
    cancellation_siblings: $cancellation_siblings,
    rejected_admission_http: $rejected_prefill_http,
    post_admission_failed_prefill_http: $post_admission_failed_prefill_http,
    post_admission_failed_prefill_faults_delta: $post_admission_failed_prefill_faults_delta,
    post_admission_compound_prefills_delta: $post_admission_compound_prefills_delta,
    post_failure_cold_cached_tokens: $post_failure_cold_cached_tokens,
    post_failure_reuse_cached_tokens: $post_failure_reuse_cached_tokens,
    restore_hits_delta: $restore_hits_delta,
    descendants_pruned_delta: $descendants_pruned_delta,
    speculative_cached_tokens_delta: $speculative_cached_tokens_delta,
    speculative_anchor_capable: ($speculative_anchor_capable == 1),
    slotaware_spec_anchor_tokens_delta: $slotaware_spec_anchor_tokens_delta,
    configured_slots: $configured_slots,
    effective_committed_depth: $effective_committed_depth,
    simultaneous_pending_capacity_slots: $pending_capacity_slots,
    full_pending_concurrency_available: ($pending_capacity_slots == $configured_slots),
    full_depth_available: ($effective_committed_depth == 4),
    aggregate_anchor_budget_bytes: $aggregate_anchor_budget_bytes,
    aggregate_anchor_peak_bytes: $aggregate_anchor_peak_bytes
  }' >"$summary"

jq . "$summary"
echo "Qwen SlotAware anchor gate artifacts: $OUT_DIR" >&2
