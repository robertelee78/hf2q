#!/usr/bin/env bash
set -euo pipefail

# Cross-family, user-shaped KV lifecycle gate for an already-running hf2q
# server. Run it once per model process; never co-reside the large models.
#
# The gate reproduces the failure mode seen in OpenCode:
#   1. establish one long agentic tool conversation and a cached continuation;
#   2. start a long streamed turn on that retained prefix;
#   3. queue an exact retry while the strongest prefix is still active;
#   4. cancel the active turn and require the queued turn to reuse the restored
#      pre-request checkpoint rather than cold-prefilling in another slot;
#   5. reuse a slot for an unrelated conversation and prove private history did
#      not leak into its response or cached-token accounting.

BASE_URL=${BASE_URL:-http://127.0.0.1:8080}
MODEL=${MODEL:-}
OUT_DIR=${OUT_DIR:-$(mktemp -d /var/tmp/hf2q-cache-lifecycle.XXXXXX)}
RUN_ID=${RUN_ID:-"$$-$(date +%s)"}
CONTEXT_LINES=${CONTEXT_LINES:-6000}
ACTIVE_MAX_TOKENS=${ACTIVE_MAX_TOKENS:-2048}
CONTINUATION_THINKING_TOKEN_BUDGET=${CONTINUATION_THINKING_TOKEN_BUDGET:-}
ISOLATION_THINKING_DISABLED=${ISOLATION_THINKING_DISABLED:-false}
CURL_CONNECT_TIMEOUT_SECONDS=${CURL_CONNECT_TIMEOUT_SECONDS:-5}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-900}
SEMANTIC_WAIT_SECONDS=${SEMANTIC_WAIT_SECONDS:-180}
SIBLING_SETTLE_SECONDS=${SIBLING_SETTLE_SECONDS:-1}
EXPECTED_EXECUTION_ARTIFACT_SHA256=${EXPECTED_EXECUTION_ARTIFACT_SHA256:-}
EXPECTED_EXECUTION_ARCH_FAMILY=${EXPECTED_EXECUTION_ARCH_FAMILY:-}
EXPECTED_EXECUTION_ARCHITECTURE=${EXPECTED_EXECUTION_ARCHITECTURE:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/agentic_cache_lifecycle_contract.sh
source "$script_dir/agentic_cache_lifecycle_contract.sh"

for command in awk curl date grep jq sed; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
execution_identity_fields=0
for value in "$EXPECTED_EXECUTION_ARTIFACT_SHA256" \
  "$EXPECTED_EXECUTION_ARCH_FAMILY" "$EXPECTED_EXECUTION_ARCHITECTURE"; do
  [[ -z "$value" ]] || execution_identity_fields=$((execution_identity_fields + 1))
done
if ((execution_identity_fields != 0 && execution_identity_fields != 3)); then
  echo "execution identity requires artifact SHA-256, family, and architecture together" >&2
  exit 2
fi
if ((execution_identity_fields == 3)) \
  && ! [[ "$EXPECTED_EXECUTION_ARTIFACT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_EXECUTION_ARTIFACT_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
fi
for setting in CONTEXT_LINES ACTIVE_MAX_TOKENS CURL_CONNECT_TIMEOUT_SECONDS \
  CURL_MAX_TIME_SECONDS SEMANTIC_WAIT_SECONDS SIBLING_SETTLE_SECONDS; do
  value=${!setting}
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$setting must be a positive integer (got: $value)" >&2
    exit 2
  fi
done
if [[ -n "$CONTINUATION_THINKING_TOKEN_BUDGET" ]] \
  && ! [[ "$CONTINUATION_THINKING_TOKEN_BUDGET" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONTINUATION_THINKING_TOKEN_BUDGET must be a positive integer when set (got: $CONTINUATION_THINKING_TOKEN_BUDGET)" >&2
  exit 2
fi
[[ "$ISOLATION_THINKING_DISABLED" == true \
  || "$ISOLATION_THINKING_DISABLED" == false ]] || {
  echo "ISOLATION_THINKING_DISABLED must be true or false (got: $ISOLATION_THINKING_DISABLED)" >&2
  exit 2
}
unrelated_conversation_thinking_enabled=true
[[ "$ISOLATION_THINKING_DISABLED" == true ]] \
  && unrelated_conversation_thinking_enabled=false

mkdir -p "$OUT_DIR"

if [[ -z "$MODEL" ]]; then
  MODEL=$(curl --fail-with-body --silent --show-error \
    --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
    --max-time 10 "$BASE_URL/v1/models" | jq -er '
      [.data[] | select((.arch // "") != "")] as $models
      | if ($models | length) == 1
        then $models[0].id
        else error("expected exactly one architecture-bearing inference model")
        end
    ')
fi

context_file="$OUT_DIR/context.txt"
base_request="$OUT_DIR/base.request.json"
base_response="$OUT_DIR/base.response.json"
seed_request="$OUT_DIR/seed.request.json"
seed_response="$OUT_DIR/seed.response.json"
active_request="$OUT_DIR/active.request.json"
active_stream="$OUT_DIR/active.stream.sse"
sibling_request="$OUT_DIR/sibling.request.json"
sibling_response="$OUT_DIR/sibling.response.json"
isolation_request="$OUT_DIR/isolation.request.json"
isolation_response="$OUT_DIR/isolation.response.json"
summary="$OUT_DIR/summary.json"

capture_execution_receipt() {
  local phase=$1
  local stream=$2
  local headers=$3
  local receipt="$OUT_DIR/$phase.execution.json"

  ((execution_identity_fields == 3)) || return 0
  agentic_lifecycle_execution_receipt_json "$headers" \
    "$EXPECTED_EXECUTION_ARTIFACT_SHA256" \
    "$EXPECTED_EXECUTION_ARCH_FAMILY" \
    "$EXPECTED_EXECUTION_ARCHITECTURE" \
    | jq --arg phase "$phase" --argjson stream "$stream" \
      '. + {phase:$phase,stream:$stream}' >"$receipt.tmp"
  mv "$receipt.tmp" "$receipt"
}

active_pid=""
sibling_pid=""
cleanup() {
  if [[ -n "$active_pid" ]] && kill -0 "$active_pid" 2>/dev/null; then
    kill -TERM "$active_pid" 2>/dev/null || true
    wait "$active_pid" 2>/dev/null || true
  fi
  if [[ -n "$sibling_pid" ]] && kill -0 "$sibling_pid" 2>/dev/null; then
    kill -TERM "$sibling_pid" 2>/dev/null || true
    wait "$sibling_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

awk -v lines="$CONTEXT_LINES" -v run="$RUN_ID" 'BEGIN {
  for (i = 1; i <= lines; i++) {
    printf "cache-lifecycle-%s line %05d: Rust ownership, retained prefix, tool result, cancellation rollback, and isolation.\n", run, i
  }
}' >"$context_file"

jq -n --rawfile context "$context_file" --arg model "$MODEL" --arg run "$RUN_ID" '{
  model: $model,
  messages: [
    {
      role: "system",
      content: "You are a coding agent. Follow the requested tool and final-output contracts exactly."
    },
    {
      role: "user",
      content: ("Cache lifecycle run " + $run + ". Read the supplied context, then call lifecycle_probe exactly once with nonce " + $run + ".\n\n" + $context)
    }
  ],
  tools: [{
    type: "function",
    function: {
      name: "lifecycle_probe",
      description: "Record one deterministic cache-lifecycle probe",
      parameters: {
        type: "object",
        properties: {nonce: {type: "string"}},
        required: ["nonce"],
        additionalProperties: false
      }
    }
  }],
  tool_choice: "required",
  temperature: 0,
  max_tokens: 128,
  stream: false
}' >"$base_request"

post_json() {
  local input=$1
  local output=$2
  local phase=$3
  local headers="$OUT_DIR/$phase.response.headers"
  curl --fail-with-body --silent --show-error --no-buffer \
    --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
    --max-time "$CURL_MAX_TIME_SECONDS" \
    --dump-header "$headers" \
    -H 'Content-Type: application/json' \
    --data-binary "@$input" \
    "$BASE_URL/v1/chat/completions" >"$output"
  capture_execution_receipt "$phase" false "$headers"
}

post_json "$base_request" "$base_response" base
if ! jq -e --arg run "$RUN_ID" '
  (.choices | length) == 1
  and .choices[0].finish_reason == "tool_calls"
  and ((.choices[0].message.tool_calls // []) | length) == 1
  and .choices[0].message.tool_calls[0].function.name == "lifecycle_probe"
  and ((.choices[0].message.tool_calls[0].function.arguments | fromjson).nonce == $run)
' "$base_response" >/dev/null; then
  echo "cache lifecycle gate failed: cold base did not emit the required tool call" >&2
  jq '{choice: .choices[0], usage, timing: .x_hf2q_timing}' "$base_response" >&2
  exit 1
fi

base_cached=$(jq -er '.usage.prompt_tokens_details.cached_tokens // 0' "$base_response")
base_prompt=$(jq -er '.usage.prompt_tokens' "$base_response")
(( base_cached == 0 )) || {
  echo "cache lifecycle gate requires a fresh conversation; cold base reported $base_cached cached tokens" >&2
  exit 1
}

jq -n --slurpfile request "$base_request" --slurpfile response "$base_response" \
  --arg run "$RUN_ID" --argjson thinking_token_budget "${CONTINUATION_THINKING_TOKEN_BUDGET:-null}" '({
  model: $request[0].model,
  messages: ($request[0].messages + [
    $response[0].choices[0].message,
    {
      role: "tool",
      tool_call_id: $response[0].choices[0].message.tool_calls[0].id,
      content: ("probe accepted for " + $run)
    },
    {role: "user", content: "Reply with exactly CACHE_SEED_READY and nothing else."}
  ]),
  tools: $request[0].tools,
  # Keep the tool surface rendered exactly as an OpenCode continuation does.
  # Switching to `none` can remove the tool schema from family templates and
  # legitimately invalidate the otherwise unchanged long prefix.
  tool_choice: "auto",
  temperature: 0,
  max_tokens: 64,
  stream: false
} + if $thinking_token_budget == null then {}
     else {thinking_token_budget:$thinking_token_budget} end)' >"$seed_request"

post_json "$seed_request" "$seed_response" seed
seed_content=$(jq -r '.choices[0].message.content // empty' "$seed_response")
[[ "$seed_content" == "CACHE_SEED_READY" ]] || {
  echo "cache lifecycle gate failed: seed continuation returned unexpected content" >&2
  jq '{choice: .choices[0], usage, timing: .x_hf2q_timing}' "$seed_response" >&2
  exit 1
}
seed_cached=$(jq -er '.usage.prompt_tokens_details.cached_tokens // 0' "$seed_response")
minimum_seed_cached=$(( base_prompt > 64 ? base_prompt - 64 : 1 ))
(( seed_cached >= minimum_seed_cached )) || {
  echo "cache lifecycle gate failed: seed continuation reused only $seed_cached tokens; expected at least $minimum_seed_cached" >&2
  exit 1
}

jq -n --slurpfile seed "$seed_request" --argjson max_tokens "$ACTIVE_MAX_TOKENS" \
  --argjson thinking_token_budget "${CONTINUATION_THINKING_TOKEN_BUDGET:-null}" '({
  model: $seed[0].model,
  messages: ($seed[0].messages + [
    {role: "assistant", content: "CACHE_SEED_READY"},
    {
      role: "user",
      content: "Begin with exactly ACTIVE_STREAM_STARTED, then write a detailed explanation of cache-coherent cancellation recovery. Produce at least 1200 words, do not call tools, and end with ACTIVE_STREAM_COMPLETE."
    }
  ]),
  tools: $seed[0].tools,
  tool_choice: "auto",
  temperature: 0,
  max_tokens: $max_tokens,
  stream: true,
  stream_options: {include_usage: true}
} + if $thinking_token_budget == null then {}
     else {thinking_token_budget:$thinking_token_budget} end)' >"$active_request"

# Retry the exact same rendered prompt while its first execution still owns
# the strongest prefix. Stream/max-token controls do not change prompt tokens.
jq '.stream = false | del(.stream_options) | .max_tokens = 256' \
  "$active_request" >"$sibling_request"

# Fail before another 100K prefill if a future edit changes the rendered tool
# surface between turns. Qwen places tool definitions before the long message
# body, so `none` would make the exact LCP only the three-token system header.
jq -e -s --argjson thinking_token_budget "${CONTINUATION_THINKING_TOKEN_BUDGET:-null}" '
  .[0].tool_choice == "required"
  and .[1].tool_choice == "auto"
  and .[2].tool_choice == "auto"
  and .[3].tool_choice == "auto"
  and .[0].tools == .[1].tools
  and .[1].tools == .[2].tools
  and .[2].tools == .[3].tools
  and (.[0] | has("thinking_token_budget") | not)
  and (if $thinking_token_budget == null
    then all(.[1:][]; has("thinking_token_budget") | not)
    else all(.[1:][]; .thinking_token_budget == $thinking_token_budget)
  end)
' "$base_request" "$seed_request" "$active_request" "$sibling_request" >/dev/null || {
  echo "cache lifecycle gate misconfigured: tool surface changed across the agentic turn" >&2
  exit 2
}

curl --fail-with-body --silent --show-error --no-buffer \
  --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
  --max-time "$CURL_MAX_TIME_SECONDS" \
  --dump-header "$OUT_DIR/active_sse.response.headers" \
  -H 'Content-Type: application/json' \
  --data-binary "@$active_request" \
  "$BASE_URL/v1/chat/completions" >"$active_stream" &
active_pid=$!

semantic_ready=0
for ((second = 0; second < SEMANTIC_WAIT_SECONDS; second++)); do
  if ! kill -0 "$active_pid" 2>/dev/null; then
    break
  fi
  if sed -n 's/^data: //p' "$active_stream" | jq -e -s '
    any(.[];
      type == "object"
      and ((.choices[0].delta.content // "") | length) > 0
      or (type == "object" and ((.choices[0].delta.reasoning_content // "") | length) > 0)
      or (type == "object" and ((.choices[0].delta.tool_calls // []) | length) > 0)
    )
  ' >/dev/null 2>&1; then
    semantic_ready=1
    break
  fi
  sleep 1
done
(( semantic_ready == 1 )) || {
  echo "cache lifecycle gate failed: active stream produced no semantic event while still running" >&2
  sed -n '1,80p' "$active_stream" >&2
  exit 1
}
capture_execution_receipt active_sse true \
  "$OUT_DIR/active_sse.response.headers"
kill -0 "$active_pid" 2>/dev/null || {
  echo "cache lifecycle gate failed: active stream completed before the sibling could be queued" >&2
  exit 1
}
if grep -q '^data: \[DONE\]$' "$active_stream"; then
  echo "cache lifecycle gate failed: active stream was already terminal" >&2
  exit 1
fi

post_json "$sibling_request" "$sibling_response" sibling &
sibling_pid=$!
sleep "$SIBLING_SETTLE_SECONDS"
kill -0 "$sibling_pid" 2>/dev/null || {
  echo "cache lifecycle gate failed: exact retry finished before owner cancellation" >&2
  jq '{choice: .choices[0], usage, timing: .x_hf2q_timing}' "$sibling_response" >&2 || true
  exit 1
}

# The exact retry is now submitted while the retained-prefix owner is active.
# Cancelling that owner exercises rollback to the pre-request anchor.
kill -TERM "$active_pid" 2>/dev/null || true
wait "$active_pid" 2>/dev/null || true
active_pid=""

if ! wait "$sibling_pid"; then
  echo "cache lifecycle gate failed: queued exact retry did not complete" >&2
  sed -n '1,120p' "$sibling_response" >&2
  exit 1
fi
sibling_pid=""

if grep -q '^data: \[DONE\]$' "$active_stream"; then
  echo "cache lifecycle gate failed: cancelled active stream emitted [DONE]" >&2
  exit 1
fi
sibling_content=$(jq -r '.choices[0].message.content // empty' "$sibling_response")
[[ "$sibling_content" == ACTIVE_STREAM_STARTED* ]] || {
  echo "cache lifecycle gate failed: queued exact retry returned unexpected content" >&2
  jq '{choice: .choices[0], usage, timing: .x_hf2q_timing}' "$sibling_response" >&2
  exit 1
}
sibling_cached=$(jq -er '.usage.prompt_tokens_details.cached_tokens // 0' "$sibling_response")
(( sibling_cached >= minimum_seed_cached )) || {
  echo "cache lifecycle gate failed: active-prefix sibling went cold ($sibling_cached cached; expected at least $minimum_seed_cached)" >&2
  exit 1
}

jq -n --arg model "$MODEL" --arg run "$RUN_ID" \
  --argjson thinking_token_budget "${CONTINUATION_THINKING_TOKEN_BUDGET:-null}" \
  --argjson isolation_thinking_disabled "$ISOLATION_THINKING_DISABLED" '({
  model: $model,
  messages: [
    {
      role: "system",
      content: ("This is an unrelated isolation conversation " + $run + ". Do not mention any cache lifecycle probe, tool result, or prior conversation.")
    },
    {role: "user", content: "Reply with exactly ISOLATION_OK and nothing else."}
  ],
  temperature: 0,
  max_tokens: 64,
  stream: false
} + if $isolation_thinking_disabled then {hf2q_enable_thinking:false}
     elif $thinking_token_budget == null then {}
     else {thinking_token_budget:$thinking_token_budget} end)' >"$isolation_request"

jq -e --argjson thinking_token_budget "${CONTINUATION_THINKING_TOKEN_BUDGET:-null}" \
  --argjson isolation_thinking_disabled "$ISOLATION_THINKING_DISABLED" '
  if $isolation_thinking_disabled then
    .hf2q_enable_thinking == false
    and (has("thinking_token_budget") | not)
    and ((.chat_template_kwargs // {}) | has("enable_thinking") | not)
  elif $thinking_token_budget == null then
    (has("thinking_token_budget") | not)
    and (has("hf2q_enable_thinking") | not)
  else
    .thinking_token_budget == $thinking_token_budget
    and (has("hf2q_enable_thinking") | not)
  end
' "$isolation_request" >/dev/null || {
  echo "cache lifecycle gate misconfigured: isolation thinking contract changed" >&2
  exit 2
}

post_json "$isolation_request" "$isolation_response" isolation
isolation_content=$(jq -r '.choices[0].message.content // empty' "$isolation_response")
[[ "$isolation_content" == "ISOLATION_OK" ]] || {
  echo "cache lifecycle gate failed: unrelated conversation returned unexpected content" >&2
  jq '{choice: .choices[0], usage, timing: .x_hf2q_timing}' "$isolation_response" >&2
  exit 1
}
if grep -F -q \
  -e "$RUN_ID" \
  -e CACHE_SEED_READY \
  -e ACTIVE_STREAM_STARTED \
  -e ACTIVE_STREAM_COMPLETE \
  -e lifecycle_probe \
  "$isolation_response"; then
  echo "cache lifecycle gate failed: unrelated conversation leaked private prior-turn material" >&2
  exit 1
fi
isolation_cached=$(jq -er '.usage.prompt_tokens_details.cached_tokens // 0' "$isolation_response")
(( isolation_cached <= 64 )) || {
  echo "cache lifecycle gate failed: unrelated conversation reused more than a template-sized prefix ($isolation_cached tokens)" >&2
  exit 1
}

execution_receipts='[]'
if ((execution_identity_fields == 3)); then
  execution_receipts=$(jq -s '.' \
    "$OUT_DIR/base.execution.json" \
    "$OUT_DIR/seed.execution.json" \
    "$OUT_DIR/active_sse.execution.json" \
    "$OUT_DIR/sibling.execution.json" \
    "$OUT_DIR/isolation.execution.json")
fi

jq -n \
  --argjson schema_version 3 \
  --arg status pass \
  --arg model "$MODEL" \
  --arg base_url "$BASE_URL" \
  --arg run_id "$RUN_ID" \
  --argjson context_lines "$CONTEXT_LINES" \
  --argjson continuation_thinking_token_budget "${CONTINUATION_THINKING_TOKEN_BUDGET:-null}" \
  --argjson unrelated_conversation_thinking_enabled "$unrelated_conversation_thinking_enabled" \
  --argjson base_prompt_tokens "$base_prompt" \
  --argjson seed_cached_tokens "$seed_cached" \
  --argjson sibling_cached_tokens "$sibling_cached" \
  --argjson isolation_cached_tokens "$isolation_cached" \
  --argjson execution_receipts "$execution_receipts" '{
    schema_version: $schema_version,
    status: $status,
    model: $model,
    base_url: $base_url,
    run_id: $run_id,
    context_lines: $context_lines,
    continuation_thinking_token_budget: $continuation_thinking_token_budget,
    unrelated_conversation_thinking_enabled: $unrelated_conversation_thinking_enabled,
    base_prompt_tokens: $base_prompt_tokens,
    seed_cached_tokens: $seed_cached_tokens,
    active_stream_cancelled_without_done: true,
    queued_exact_retry_cached_tokens: $sibling_cached_tokens,
    unrelated_conversation_cached_tokens: $isolation_cached_tokens,
    unrelated_conversation_content: "ISOLATION_OK",
    execution_receipts: $execution_receipts
  }' >"$summary.tmp"
mv "$summary.tmp" "$summary"

if ((execution_identity_fields == 3)); then
  agentic_lifecycle_validate_summary "$summary" "$RUN_ID" "$CONTEXT_LINES" \
    "$EXPECTED_EXECUTION_ARTIFACT_SHA256" \
    "$EXPECTED_EXECUTION_ARCH_FAMILY" \
    "$EXPECTED_EXECUTION_ARCHITECTURE" \
    "${CONTINUATION_THINKING_TOKEN_BUDGET:-null}" \
    "$unrelated_conversation_thinking_enabled"
fi

jq . "$summary"
echo "cache lifecycle artifacts: $OUT_DIR" >&2
