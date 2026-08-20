#!/usr/bin/env bash
set -euo pipefail

# Realistic OpenCode acceptance gate for an already-running hf2q server.
# The historical filename is retained for compatibility; Gemma and Qwen use
# family wrappers that bind their own endpoint/model while sharing the same
# behavioral contract. It intentionally fails when tool semantics, prefix
# reuse, TTFT, SSE, or tool-result continuation are not agentic-usable.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:8080}
MODEL=${MODEL:-Deepseek v4 Flash 0731 Source}
EXPECTED_PATH=${EXPECTED_PATH:-$ROOT_DIR/Cargo.toml}
TOOL_RESULT_PATH=${TOOL_RESULT_PATH:-$EXPECTED_PATH}
TOOL_RESULT_SUCCESS_PREFIX=${TOOL_RESULT_SUCCESS_PREFIX:-$'Successful read_file result. File follows:\n'}
AGENTIC_SYSTEM_PROMPT=${AGENTIC_SYSTEM_PROMPT:-You are an agentic coding assistant. Use the provided tool to inspect files before answering.}
RUN_ID=${RUN_ID:-"$$-$(date +%s)"}
REQUIRE_COLD_FIRST=${REQUIRE_COLD_FIRST:-1}
MAX_COLD_TTFT_MS=${MAX_COLD_TTFT_MS:-30000}
MAX_CACHED_TTFT_MS=${MAX_CACHED_TTFT_MS:-5000}
MAX_COLD_RESPONSE_MS=${MAX_COLD_RESPONSE_MS:-40000}
MAX_CACHED_RESPONSE_MS=${MAX_CACHED_RESPONSE_MS:-10000}
MAX_CACHED_SEMANTIC_MS=${MAX_CACHED_SEMANTIC_MS:-10000}
MAX_TOOL_RESULT_RESPONSE_MS=${MAX_TOOL_RESULT_RESPONSE_MS:-$MAX_CACHED_RESPONSE_MS}
AGENTIC_CONTEXT_FIXTURE=${AGENTIC_CONTEXT_FIXTURE:-$ROOT_DIR/scripts/fixtures/deepseek4-agentic-repo-context.txt}
AGENTIC_CONTEXT_FIXTURE_SHA256=${AGENTIC_CONTEXT_FIXTURE_SHA256:-2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef}
AGENTIC_FIXTURE_ID=${AGENTIC_FIXTURE_ID:-full-context-agentic-v1}
EXPECTED_PROMPT_TOKENS=${EXPECTED_PROMPT_TOKENS:-0}
AGENTIC_PROMPT_CONTRACT=${AGENTIC_PROMPT_CONTRACT:-}
AGENTIC_PROMPT_CONTRACT_SHA256=${AGENTIC_PROMPT_CONTRACT_SHA256:-}
AGENT_INDEX=${AGENT_INDEX:-0}
PROMPT_PROVENANCE_SHA256=${PROMPT_PROVENANCE_SHA256:-}
CURL_CONNECT_TIMEOUT_SECONDS=${CURL_CONNECT_TIMEOUT_SECONDS:-5}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-60}
MAX_TOKENS=${MAX_TOKENS:-128}
SOURCE_MAX_TOKENS=${SOURCE_MAX_TOKENS:-256}
SENTINEL=${SENTINEL:-SENTINEL_CARGO_HF2Q_AGENTIC}
EXPECTED_SOURCE=${EXPECTED_SOURCE:-"fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result"}
HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT=${HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT:-}
COLD_RESULT_PATH=${COLD_RESULT_PATH:-}
HF2Q_AGENTIC_TIME_TOTAL_INPUT=${HF2Q_AGENTIC_TIME_TOTAL_INPUT:-}

for command in awk curl date grep jq shasum; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ -x /usr/bin/perl ]] || {
  echo "required monotonic timer is unavailable: /usr/bin/perl" >&2
  exit 2
}
parse_curl_time_total_ms() {
  local timing_file=$1

  awk '
    NR == 1 && $0 ~ /^[0-9]+([.][0-9]+)?$/ {
      milliseconds = $1 * 1000
      rounded = int(milliseconds)
      if (milliseconds > rounded) rounded++
      print rounded
      valid = 1
      next
    }
    { invalid = 1 }
    END { if (NR != 1 || !valid || invalid) exit 1 }
  ' "$timing_file"
}
if [[ -n "$HF2Q_AGENTIC_TIME_TOTAL_INPUT" ]]; then
  parse_curl_time_total_ms "$HF2Q_AGENTIC_TIME_TOTAL_INPUT"
  exit 0
fi
[[ -r "$TOOL_RESULT_PATH" ]] || {
  echo "tool result fixture is not readable: $TOOL_RESULT_PATH" >&2
  exit 2
}
for setting in MAX_TOKENS SOURCE_MAX_TOKENS CURL_CONNECT_TIMEOUT_SECONDS CURL_MAX_TIME_SECONDS; do
  value=${!setting}
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$setting must be a positive integer (got: $value)" >&2
    exit 2
  fi
done
[[ "$EXPECTED_PROMPT_TOKENS" == 0 || "$EXPECTED_PROMPT_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
  echo "EXPECTED_PROMPT_TOKENS must be zero or a positive integer (got: $EXPECTED_PROMPT_TOKENS)" >&2
  exit 2
}
[[ "$AGENTIC_FIXTURE_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "AGENTIC_FIXTURE_ID contains unsupported characters: $AGENTIC_FIXTURE_ID" >&2
  exit 2
}
[[ "$REQUIRE_COLD_FIRST" == 0 || "$REQUIRE_COLD_FIRST" == 1 ]] || {
  echo "REQUIRE_COLD_FIRST must be 0 or 1 (got: $REQUIRE_COLD_FIRST)" >&2
  exit 2
}
[[ -r "$AGENTIC_CONTEXT_FIXTURE" ]] || {
  echo "agentic context fixture is not readable: $AGENTIC_CONTEXT_FIXTURE" >&2
  exit 2
}
[[ "$AGENTIC_CONTEXT_FIXTURE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "AGENTIC_CONTEXT_FIXTURE_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
}
actual_context_fixture_sha256=$(shasum -a 256 "$AGENTIC_CONTEXT_FIXTURE" | awk '{print $1}')
if [[ "$actual_context_fixture_sha256" != "$AGENTIC_CONTEXT_FIXTURE_SHA256" ]]; then
  echo "agentic context fixture SHA-256 mismatch: expected $AGENTIC_CONTEXT_FIXTURE_SHA256, got $actual_context_fixture_sha256" >&2
  exit 2
fi
agentic_context_fixture_bytes=$(wc -c <"$AGENTIC_CONTEXT_FIXTURE" | tr -d '[:space:]')
agentic_system_prompt_sha256=$(printf '%s' "$AGENTIC_SYSTEM_PROMPT" | shasum -a 256 | awk '{print $1}')
tool_result_success_prefix_sha256=$(printf '%s' "$TOOL_RESULT_SUCCESS_PREFIX" | shasum -a 256 | awk '{print $1}')

prompt_contract_sha256=""
serialization_policy=""
request_sha256=""
request_bytes=0
rendered_prompt_sha256=""
prompt_token_ids_sha256=""
tool_result_fixture_sha256=$(shasum -a 256 "$TOOL_RESULT_PATH" | awk '{print $1}')
tool_result_fixture_bytes=$(wc -c <"$TOOL_RESULT_PATH" | tr -d '[:space:]')
tool_result_fixture_chars=$(jq -Rs 'length' "$TOOL_RESULT_PATH")
tool_result_success_prefix_bytes=$(printf '%s' "$TOOL_RESULT_SUCCESS_PREFIX" | wc -c | tr -d '[:space:]')
tool_result_payload_bytes=$((tool_result_success_prefix_bytes + tool_result_fixture_bytes))
tool_result_payload_sha256=$(
  { printf '%s' "$TOOL_RESULT_SUCCESS_PREFIX"; cat "$TOOL_RESULT_PATH"; } |
    shasum -a 256 | awk '{print $1}'
)

if [[ -n "$AGENTIC_PROMPT_CONTRACT" ]]; then
  [[ -r "$AGENTIC_PROMPT_CONTRACT" ]] || {
    echo "agentic prompt contract is not readable: $AGENTIC_PROMPT_CONTRACT" >&2
    exit 2
  }
  [[ "$AGENT_INDEX" =~ ^[1-4]$ ]] || {
    echo "AGENT_INDEX must be 1..4 when AGENTIC_PROMPT_CONTRACT is set" >&2
    exit 2
  }
  jq -e -f "$ROOT_DIR/scripts/deepseek4_agentic_prompt_contract.jq" \
    "$AGENTIC_PROMPT_CONTRACT" >/dev/null
  prompt_contract_sha256=$(shasum -a 256 "$AGENTIC_PROMPT_CONTRACT" | awk '{print $1}')
  if [[ -n "$AGENTIC_PROMPT_CONTRACT_SHA256" &&
        "$prompt_contract_sha256" != "$AGENTIC_PROMPT_CONTRACT_SHA256" ]]; then
    echo "agentic prompt contract SHA-256 mismatch" >&2
    exit 2
  fi
  AGENTIC_FIXTURE_ID=$(jq -er '.fixture_id' "$AGENTIC_PROMPT_CONTRACT")
  serialization_policy=$(jq -er '.serialization.policy' "$AGENTIC_PROMPT_CONTRACT")
  EXPECTED_PROMPT_TOKENS=$(jq -er '.serialization.expected_prompt_tokens' "$AGENTIC_PROMPT_CONTRACT")
  [[ "$MODEL" == "$(jq -er '.request.model' "$AGENTIC_PROMPT_CONTRACT")" &&
      "$MAX_TOKENS" == "$(jq -er '.request.max_tokens' "$AGENTIC_PROMPT_CONTRACT")" &&
      "$agentic_system_prompt_sha256" == "$(jq -er '.request.system_prompt_sha256' "$AGENTIC_PROMPT_CONTRACT")" ]] || {
    echo "agentic request settings disagree with prompt contract" >&2
    exit 2
  }
  expected_contract_path=$(jq -er '.request.expected_path' "$AGENTIC_PROMPT_CONTRACT")
  [[ "$EXPECTED_PATH" == "$expected_contract_path" ]] || {
    echo "agentic expected path disagrees with prompt contract" >&2
    exit 2
  }
  expected_run_id=$(jq -er --argjson agent "$AGENT_INDEX" \
    '.agents[] | select(.agent == $agent) | .run_id' "$AGENTIC_PROMPT_CONTRACT")
  expected_sentinel=$(jq -er --argjson agent "$AGENT_INDEX" \
    '.agents[] | select(.agent == $agent) | .sentinel' "$AGENTIC_PROMPT_CONTRACT")
  [[ "$RUN_ID" == "$expected_run_id" && "$SENTINEL" == "$expected_sentinel" ]] || {
    echo "agentic agent identity disagrees with prompt contract" >&2
    exit 2
  }
  expected_context_sha=$(jq -er '.repository_context.sha256' "$AGENTIC_PROMPT_CONTRACT")
  expected_context_bytes=$(jq -er '.repository_context.bytes' "$AGENTIC_PROMPT_CONTRACT")
  [[ "$actual_context_fixture_sha256" == "$expected_context_sha" &&
      "$agentic_context_fixture_bytes" == "$expected_context_bytes" ]] || {
    echo "agentic context input disagrees with prompt contract" >&2
    exit 2
  }
  expected_tool_sha=$(jq -er '.tool_result.sha256' "$AGENTIC_PROMPT_CONTRACT")
  expected_tool_bytes=$(jq -er '.tool_result.bytes' "$AGENTIC_PROMPT_CONTRACT")
  expected_tool_chars=$(jq -er '.tool_result.chars' "$AGENTIC_PROMPT_CONTRACT")
  expected_prefix_bytes=$(jq -er '.tool_result.success_prefix_bytes' "$AGENTIC_PROMPT_CONTRACT")
  expected_payload_bytes=$(jq -er '.tool_result.combined_payload_bytes' "$AGENTIC_PROMPT_CONTRACT")
  expected_payload_sha=$(jq -er '.tool_result.combined_payload_sha256' "$AGENTIC_PROMPT_CONTRACT")
  [[ "$tool_result_fixture_sha256" == "$expected_tool_sha" &&
      "$tool_result_fixture_bytes" == "$expected_tool_bytes" &&
      "$tool_result_fixture_chars" == "$expected_tool_chars" &&
      "$tool_result_success_prefix_bytes" == "$expected_prefix_bytes" &&
      "$tool_result_payload_bytes" == "$expected_payload_bytes" &&
      "$tool_result_payload_sha256" == "$expected_payload_sha" ]] || {
    echo "agentic tool-result input disagrees with prompt contract" >&2
    exit 2
  }
  [[ "$tool_result_success_prefix_sha256" == \
      "$(jq -er '.tool_result.success_prefix_sha256' "$AGENTIC_PROMPT_CONTRACT")" ]] || {
    echo "agentic tool-result prefix disagrees with prompt contract" >&2
    exit 2
  }
  canonical_tool_result="$ROOT_DIR/$(jq -er '.tool_result.path' "$AGENTIC_PROMPT_CONTRACT")"
  [[ "$TOOL_RESULT_PATH" == "$canonical_tool_result" ]] || {
    echo "agentic tool-result path is not the contract fixture" >&2
    exit 2
  }
  for input in request_builder chat_template; do
    input_path=$(jq -er --arg input "$input" '.[$input].path' "$AGENTIC_PROMPT_CONTRACT")
    input_sha=$(jq -er --arg input "$input" '.[$input].sha256' "$AGENTIC_PROMPT_CONTRACT")
    input_bytes=$(jq -er --arg input "$input" '.[$input].bytes' "$AGENTIC_PROMPT_CONTRACT")
    [[ "$(shasum -a 256 "$ROOT_DIR/$input_path" | awk '{print $1}')" == "$input_sha" &&
        "$(wc -c <"$ROOT_DIR/$input_path" | tr -d '[:space:]')" == "$input_bytes" ]] || {
      echo "agentic $input input disagrees with prompt contract" >&2
      exit 2
    }
  done
fi

request_file=$(mktemp -t hf2q-deepseek-agentic-request.XXXXXX)
first_file=$(mktemp -t hf2q-deepseek-agentic-first.XXXXXX)
second_file=$(mktemp -t hf2q-deepseek-agentic-second.XXXXXX)
auto_request_file=$(mktemp -t hf2q-deepseek-agentic-auto-request.XXXXXX)
auto_file=$(mktemp -t hf2q-deepseek-agentic-auto.XXXXXX)
stream_file=$(mktemp -t hf2q-deepseek-agentic-stream.XXXXXX)
stream_json_file=$(mktemp -t hf2q-deepseek-agentic-stream-json.XXXXXX)
stream_timing_file=$(mktemp -t hf2q-deepseek-agentic-stream-timing.XXXXXX)
continuation_file=$(mktemp -t hf2q-deepseek-agentic-continuation.XXXXXX)
continuation_response=$(mktemp -t hf2q-deepseek-agentic-continuation-response.XXXXXX)
source_request_file=$(mktemp -t hf2q-deepseek-agentic-source-request.XXXXXX)
source_response_file=$(mktemp -t hf2q-deepseek-agentic-source-response.XXXXXX)
post_timing_file=$(mktemp -t hf2q-deepseek-agentic-post-timing.XXXXXX)
cleanup() {
  rm -f "$request_file" "$first_file" "$second_file" "$auto_request_file" \
    "$auto_file" "$stream_file" \
    "$stream_json_file" "$stream_timing_file" "$continuation_file" \
    "$continuation_response" "$source_request_file" "$source_response_file"
  rm -f "$post_timing_file"
}
trap cleanup EXIT

cd "$ROOT_DIR"

repository_context_chars=$(jq -Rs 'length' "$AGENTIC_CONTEXT_FIXTURE")

jq -n --rawfile repo "$AGENTIC_CONTEXT_FIXTURE" \
  --argjson max_tokens "$MAX_TOKENS" \
  --arg model "$MODEL" --arg expected_path "$EXPECTED_PATH" --arg run_id "$RUN_ID" \
  --arg sentinel "$SENTINEL" --arg system_prompt "$AGENTIC_SYSTEM_PROMPT" \
  -f scripts/deepseek4_agentic_request.jq >"$request_file"
request_sha256=$(shasum -a 256 "$request_file" | awk '{print $1}')
request_bytes=$(wc -c <"$request_file" | tr -d '[:space:]')
if [[ -n "$AGENTIC_PROMPT_CONTRACT" ]]; then
  expected_request_sha=$(jq -er --argjson agent "$AGENT_INDEX" \
    '.agents[] | select(.agent == $agent) | .request_sha256' "$AGENTIC_PROMPT_CONTRACT")
  expected_request_bytes=$(jq -er --argjson agent "$AGENT_INDEX" \
    '.agents[] | select(.agent == $agent) | .request_bytes' "$AGENTIC_PROMPT_CONTRACT")
  [[ "$request_sha256" == "$expected_request_sha" &&
      "$request_bytes" == "$expected_request_bytes" ]] || {
    echo "agentic request bytes disagree with prompt contract" >&2
    exit 2
  }
  rendered_prompt_sha256=$(jq -er --argjson agent "$AGENT_INDEX" \
    '.agents[] | select(.agent == $agent) | .rendered_prompt_sha256' "$AGENTIC_PROMPT_CONTRACT")
  prompt_token_ids_sha256=$(jq -er --argjson agent "$AGENT_INDEX" \
    '.agents[] | select(.agent == $agent) | .prompt_token_ids_sha256' "$AGENTIC_PROMPT_CONTRACT")
fi
if [[ -n "$HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT" ]]; then
  cp "$request_file" "$HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT"
  exit 0
fi
if [[ -n "$AGENTIC_PROMPT_CONTRACT" &&
      ! "$PROMPT_PROVENANCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "PROMPT_PROVENANCE_SHA256 must bind the exact renderer/tokenizer receipt" >&2
  exit 2
fi

post_json() {
  local input=$1
  local output=$2
  if ! curl --fail-with-body --silent --show-error --no-buffer \
    --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
    --max-time "$CURL_MAX_TIME_SECONDS" \
    -H 'Content-Type: application/json' \
    --data-binary "@$input" \
    --output "$output" --write-out '%{time_total}\n' \
    "$BASE_URL/v1/chat/completions" >"$post_timing_file"; then
    echo "chat completion request failed; response body:" >&2
    sed -n '1,120p' "$output" >&2
    return 1
  fi
  POST_JSON_TIME_MS=$(parse_curl_time_total_ms "$post_timing_file")
  [[ "$POST_JSON_TIME_MS" =~ ^[0-9]+$ ]]
}

monotonic_us() {
  /usr/bin/perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
    -e 'printf "%.0f\n", 1000000 * clock_gettime(CLOCK_MONOTONIC)'
}

assert_tool_path() {
  local response=$1
  if ! jq -e --arg expected "$EXPECTED_PATH" '
    (.choices | length) == 1
    and .choices[0].finish_reason == "tool_calls"
    and ((.choices[0].message.content // "") | type == "string")
    and ((.choices[0].message.content // "") | contains("<｜DSML｜") | not)
    and ((.choices[0].message.content // "") | contains("<think>") | not)
    and ((.choices[0].message.tool_calls // []) | length) == 1
    and .choices[0].message.tool_calls[0].type == "function"
    and .choices[0].message.tool_calls[0].function.name == "read_file"
    and ((.choices[0].message.tool_calls[0].function.arguments | fromjson).path == $expected)
  ' "$response" >/dev/null; then
    echo "agentic gate failed: expected exactly one read_file tool call for $EXPECTED_PATH" >&2
    jq '{choice: .choices[0], usage, timing: .x_hf2q_timing}' "$response" >&2
    exit 1
  fi
}

post_json "$request_file" "$first_file"
cold_response_ms=$POST_JSON_TIME_MS
assert_tool_path "$first_file"

cold_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$first_file")
prompt_tokens=$(jq -r '.usage.prompt_tokens' "$first_file")
if (( EXPECTED_PROMPT_TOKENS > 0 && prompt_tokens != EXPECTED_PROMPT_TOKENS )); then
  echo "agentic gate failed: rendered prompt has $prompt_tokens tokens; expected exactly $EXPECTED_PROMPT_TOKENS" >&2
  exit 1
fi
if [[ "$REQUIRE_COLD_FIRST" == 1 ]] && (( cold_cached != 0 )); then
  echo "agentic gate failed: first request was not cold (cached_tokens=$cold_cached)" >&2
  exit 1
fi
cold_ttft=$(jq -r '.x_hf2q_timing.time_to_first_token_ms // -1' "$first_file")
awk -v actual="$cold_ttft" -v limit="$MAX_COLD_TTFT_MS" 'BEGIN { exit !(actual >= 0 && actual <= limit) }' || {
  echo "agentic gate failed: cold TTFT ${cold_ttft}ms exceeds ${MAX_COLD_TTFT_MS}ms" >&2
  exit 1
}
if (( cold_response_ms > MAX_COLD_RESPONSE_MS )); then
  echo "agentic gate failed: cold semantic response took ${cold_response_ms}ms; limit is ${MAX_COLD_RESPONSE_MS}ms" >&2
  exit 1
fi
if [[ -n "$COLD_RESULT_PATH" ]]; then
  cold_result_tmp="${COLD_RESULT_PATH}.tmp.$$"
  jq -n \
    --argjson agent "$AGENT_INDEX" \
    --arg run_id "$RUN_ID" \
    --arg fixture_id "$AGENTIC_FIXTURE_ID" \
    --arg prompt_contract_sha256 "$prompt_contract_sha256" \
    --arg prompt_provenance_sha256 "$PROMPT_PROVENANCE_SHA256" \
    --arg serialization_policy "$serialization_policy" \
    --arg request_sha256 "$request_sha256" \
    --argjson request_bytes "$request_bytes" \
    --arg rendered_prompt_sha256 "$rendered_prompt_sha256" \
    --arg prompt_token_ids_sha256 "$prompt_token_ids_sha256" \
    --arg fixture_sha256 "$actual_context_fixture_sha256" \
    --argjson fixture_bytes "$agentic_context_fixture_bytes" \
    --argjson prompt_tokens "$prompt_tokens" \
    --argjson cold_cached_tokens "$cold_cached" \
    --argjson cold_ttft_ms "$cold_ttft" \
    --argjson cold_semantic_response_ms "$cold_response_ms" \
    '{status:"pass",agent:$agent,run_id:$run_id,fixture_id:$fixture_id,
      prompt_contract_sha256:$prompt_contract_sha256,
      prompt_provenance_sha256:$prompt_provenance_sha256,
      serialization_policy:$serialization_policy,
      request_sha256:$request_sha256,request_bytes:$request_bytes,
      rendered_prompt_sha256:$rendered_prompt_sha256,
      prompt_token_ids_sha256:$prompt_token_ids_sha256,
      fixture_sha256:$fixture_sha256,
      fixture_bytes:$fixture_bytes,prompt_tokens:$prompt_tokens,
      cold_cached_tokens:$cold_cached_tokens,cold_ttft_ms:$cold_ttft_ms,
      cold_semantic_response_ms:$cold_semantic_response_ms}' \
    >"$cold_result_tmp"
  mv "$cold_result_tmp" "$COLD_RESULT_PATH"
fi

post_json "$request_file" "$second_file"
cached_response_ms=$POST_JSON_TIME_MS
assert_tool_path "$second_file"

# Temperature-zero recovery from the cached anchor must be semantically
# identical to the cold turn. Normalize only the synthesized call id, which is
# intentionally unique per response.
if ! jq -e -n --slurpfile cold "$first_file" --slurpfile cached "$second_file" '
  def normalized_choice:
    .choices[0] | (.message.tool_calls[]?.id = "<normalized>");
  ($cold[0] | normalized_choice) == ($cached[0] | normalized_choice)
  and $cold[0].usage.completion_tokens == $cached[0].usage.completion_tokens
' >/dev/null; then
  echo "agentic gate failed: cached temperature-zero tool turn diverged from cold output" >&2
  jq -n --slurpfile cold "$first_file" --slurpfile cached "$second_file" \
    '{cold: $cold[0].choices[0], cached: $cached[0].choices[0]}' >&2
  exit 1
fi

cached_prompt_tokens=$(jq -r '.usage.prompt_tokens' "$second_file")
if (( cached_prompt_tokens != prompt_tokens )); then
  echo "agentic gate failed: repeated request token count changed from $prompt_tokens to $cached_prompt_tokens" >&2
  exit 1
fi
cached_tokens=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$second_file")
cached_ttft=$(jq -r '.x_hf2q_timing.time_to_first_token_ms // -1' "$second_file")
minimum_cached=$((prompt_tokens - 32))
if (( cached_tokens < minimum_cached )); then
  echo "agentic gate failed: reused $cached_tokens/$prompt_tokens tokens; expected at least $minimum_cached" >&2
  exit 1
fi
awk -v actual="$cached_ttft" -v limit="$MAX_CACHED_TTFT_MS" 'BEGIN { exit !(actual >= 0 && actual <= limit) }' || {
  echo "agentic gate failed: cached TTFT ${cached_ttft}ms exceeds ${MAX_CACHED_TTFT_MS}ms" >&2
  exit 1
}
if (( cached_response_ms > MAX_CACHED_RESPONSE_MS )); then
  echo "agentic gate failed: cached semantic response took ${cached_response_ms}ms; limit is ${MAX_CACHED_RESPONSE_MS}ms" >&2
  exit 1
fi

# OpenCode's normal agent loop uses tool_choice=auto. Required-mode grammar
# proves the structured body, but only Auto proves that the model recognizes
# the task and elects to open DeepSeek's DSML tool block on its own.
jq '.tool_choice = "auto"' "$request_file" >"$auto_request_file"
post_json "$auto_request_file" "$auto_file"
auto_response_ms=$POST_JSON_TIME_MS
assert_tool_path "$auto_file"
auto_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$auto_file")
if (( auto_cached < minimum_cached )); then
  echo "agentic gate failed: tool_choice=auto reused only $auto_cached/$prompt_tokens prompt tokens" >&2
  exit 1
fi
if (( auto_response_ms > MAX_CACHED_RESPONSE_MS )); then
  echo "agentic gate failed: tool_choice=auto response took ${auto_response_ms}ms; limit is ${MAX_CACHED_RESPONSE_MS}ms" >&2
  exit 1
fi

stream_started_us=$(monotonic_us)
jq '.stream = true | .stream_options = {include_usage: true}' "$request_file" |
  curl --fail-with-body --silent --show-error --no-buffer \
    --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
    --max-time "$CURL_MAX_TIME_SECONDS" \
    -H 'Content-Type: application/json' --data-binary @- \
    "$BASE_URL/v1/chat/completions" |
  while IFS= read -r line; do
    printf '%s\n' "$line" >>"$stream_file"
    payload=${line#data: }
    if [[ "$payload" != "$line" && "$payload" != "[DONE]" ]] &&
      printf '%s\n' "$payload" | jq -e \
        'any(.choices[]?; ((.delta.tool_calls // []) | length) > 0)' >/dev/null; then
      if [[ ! -s "$stream_timing_file" ]]; then
        stream_finished_us=$(monotonic_us)
        echo $(( (stream_finished_us - stream_started_us + 999) / 1000 )) \
          >"$stream_timing_file"
      fi
    fi
  done

done_count=$(grep -c '^data: \[DONE\]$' "$stream_file" || true)
last_event=$(grep '^data: ' "$stream_file" | tail -1 || true)
if [[ "$done_count" != "1" || "$last_event" != "data: [DONE]" ]]; then
  echo "agentic gate failed: SSE stream did not end with exactly one [DONE]" >&2
  exit 1
fi
sed -n 's/^data: //p' "$stream_file" | grep -v '^\[DONE\]$' >"$stream_json_file"
if ! jq -s 'all(.[]; type == "object")' "$stream_json_file" >/dev/null; then
  echo "agentic gate failed: SSE contained a malformed JSON event" >&2
  exit 1
fi
if ! jq -e -s '[.[] | .choices[]?.delta.tool_calls[]?.index] | unique == [0]' \
  "$stream_json_file" >/dev/null; then
  echo "agentic gate failed: SSE did not contain exactly one tool-call index" >&2
  exit 1
fi
stream_content=$(jq -r -s '[.[] | .choices[]?.delta.content? | select(. != null)] | join("")' "$stream_json_file")
expected_stream_content=$(jq -r '.choices[0].message.content // ""' "$second_file")
if [[ "$stream_content" != "$expected_stream_content" ||
      "$stream_content" == *"<｜DSML｜"* || "$stream_content" == *"<think>"* ]]; then
  echo "agentic gate failed: SSE content diverged from the cached non-streaming tool turn" >&2
  printf 'expected content: %q\nactual content: %q\n' \
    "$expected_stream_content" "$stream_content" >&2
  exit 1
fi
stream_id=$(jq -r -s '[.[] | .choices[]?.delta.tool_calls[]? | select(.index == 0) | .id // empty] | join("")' "$stream_json_file")
stream_type=$(jq -r -s '[.[] | .choices[]?.delta.tool_calls[]? | select(.index == 0) | .type // empty] | join("")' "$stream_json_file")
stream_name=$(jq -r -s '[.[] | .choices[]?.delta.tool_calls[]? | select(.index == 0) | .function.name // empty] | join("")' "$stream_json_file")
stream_arguments=$(jq -r -s '[.[] | .choices[]?.delta.tool_calls[]? | select(.index == 0) | .function.arguments // empty] | join("")' "$stream_json_file")
stream_finish=$(jq -r -s '[.[] | .choices[]? | .finish_reason // empty] | last // empty' "$stream_json_file")
stream_cached=$(jq -r -s '[.[] | .usage.prompt_tokens_details.cached_tokens? // empty] | last // 0' "$stream_json_file")
if [[ -z "$stream_id" || "$stream_type" != "function" || "$stream_name" != "read_file" ||
      "$stream_finish" != "tool_calls" ]]; then
  echo "agentic gate failed: SSE tool-call identity or finish reason was invalid" >&2
  exit 1
fi
if ! jq -e --arg expected "$EXPECTED_PATH" '.path == $expected' \
  <<<"$stream_arguments" >/dev/null; then
  echo "agentic gate failed: SSE tool arguments were invalid or used the wrong path" >&2
  printf 'reconstructed SSE arguments: %s\n' "$stream_arguments" >&2
  exit 1
fi
if (( stream_cached < minimum_cached )); then
  echo "agentic gate failed: SSE reused only $stream_cached/$prompt_tokens prompt tokens" >&2
  exit 1
fi
if [[ ! -s "$stream_timing_file" ]]; then
  echo "agentic gate failed: SSE emitted no semantic tool-call event" >&2
  exit 1
fi
stream_semantic_ms=$(<"$stream_timing_file")
if (( stream_semantic_ms > MAX_CACHED_SEMANTIC_MS )); then
  echo "agentic gate failed: cached SSE tool call took ${stream_semantic_ms}ms; limit is ${MAX_CACHED_SEMANTIC_MS}ms" >&2
  exit 1
fi

jq -n --slurpfile base "$request_file" --slurpfile prior "$second_file" \
  --rawfile tool_result "$TOOL_RESULT_PATH" \
  --arg tool_result_success_prefix "$TOOL_RESULT_SUCCESS_PREFIX" '
    $base[0]
    | .messages += [
        {
          role: "assistant",
          content: $prior[0].choices[0].message.content,
          tool_calls: $prior[0].choices[0].message.tool_calls
        },
        {
          role: "tool",
          tool_call_id: $prior[0].choices[0].message.tool_calls[0].id,
          content: ($tool_result_success_prefix + $tool_result)
        }
      ]
    | .tool_choice = "auto"
    | .stream = false
  ' >"$continuation_file"

post_json "$continuation_file" "$continuation_response"
continuation_response_ms=$POST_JSON_TIME_MS
continuation_prompt_tokens=$(jq -r '.usage.prompt_tokens' "$continuation_response")
continuation_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$continuation_response")
continuation_uncached=$((continuation_prompt_tokens - continuation_cached))
continuation_content=$(jq -r '.choices[0].message.content // empty' "$continuation_response")
if (( continuation_cached < minimum_cached )); then
  echo "agentic gate failed: tool-result turn reused only $continuation_cached prefix tokens" >&2
  exit 1
fi
if [[ "$continuation_content" != "$SENTINEL" ]] || ! jq -e --arg sentinel "$SENTINEL" '
  (.choices | length) == 1
  and .choices[0].finish_reason == "stop"
  and .choices[0].message.content == $sentinel
  and ((.choices[0].message.tool_calls // []) | length) == 0
' "$continuation_response" >/dev/null; then
  echo "agentic gate failed: tool-result continuation was not one terminal sentinel response" >&2
  jq '.choices[0]' "$continuation_response" >&2
  exit 1
fi
if (( continuation_response_ms > MAX_TOOL_RESULT_RESPONSE_MS )); then
  echo "agentic gate failed: tool-result response took ${continuation_response_ms}ms; limit is ${MAX_TOOL_RESULT_RESPONSE_MS}ms" >&2
  exit 1
fi
if [[ -n "$AGENTIC_PROMPT_CONTRACT" ]]; then
  expected_anchor=$(jq -er '.prompt.cached_anchor_tokens' "$AGENTIC_PROMPT_CONTRACT")
  expected_suffix=$(jq -er '.prompt.tool_result_uncached_suffix_tokens' "$AGENTIC_PROMPT_CONTRACT")
  if (( continuation_cached != expected_anchor || continuation_uncached != expected_suffix )); then
    echo "agentic gate failed: continuation cache shape ${continuation_cached}+${continuation_uncached} disagrees with contract ${expected_anchor}+${expected_suffix}" >&2
    exit 1
  fi
  if (( cached_tokens != expected_anchor || auto_cached != expected_anchor ||
        stream_cached != expected_anchor || continuation_cached != expected_anchor ||
        continuation_prompt_tokens != continuation_cached + continuation_uncached )); then
    echo "agentic gate failed: cached-turn identities disagree with exact retained-prefix contract" >&2
    exit 1
  fi
fi

# Regression for a real OpenCode failure: the DeepSeek DSML grammar used to
# reject every `<` inside string parameters, so Rust lifetimes/generics forced
# an otherwise valid tool call to close at `fmt::Formatter`. Exercise the real
# model and require byte-exact source preservation, not merely valid JSON.
jq -n \
  --arg model "$MODEL" \
  --arg expected_source "$EXPECTED_SOURCE" \
  --argjson max_tokens "$SOURCE_MAX_TOKENS" '{
    model: $model,
    messages: [
      {
        role: "system",
        content: "You are a coding agent. Use the provided tool exactly once and preserve source code byte for byte."
      },
      {
        role: "user",
        content: ("Call emit_source exactly once with content equal to the following line. Do not explain.\n" + $expected_source)
      }
    ],
    tools: [{
      type: "function",
      function: {
        name: "emit_source",
        description: "Emit source code without changing it",
        parameters: {
          type: "object",
          properties: {content: {type: "string"}},
          required: ["content"],
          additionalProperties: false
        }
      }
    }],
    tool_choice: "required",
    temperature: 0,
    max_tokens: $max_tokens,
    stream: false
  }' >"$source_request_file"

post_json "$source_request_file" "$source_response_file"
source_response_ms=$POST_JSON_TIME_MS
if ! jq -e --arg expected "$EXPECTED_SOURCE" '
  (.choices | length) == 1
  and .choices[0].finish_reason == "tool_calls"
  and ((.choices[0].message.tool_calls // []) | length) == 1
  and .choices[0].message.tool_calls[0].function.name == "emit_source"
  and ((.choices[0].message.tool_calls[0].function.arguments | fromjson).content == $expected)
' "$source_response_file" >/dev/null; then
  echo "agentic gate failed: source tool argument lost or changed angle-bracket syntax" >&2
  jq '.choices[0]' "$source_response_file" >&2
  exit 1
fi

jq -n \
  --argjson agent "$AGENT_INDEX" \
  --arg run_id "$RUN_ID" \
  --arg fixture_id "$AGENTIC_FIXTURE_ID" \
  --arg prompt_contract_sha256 "$prompt_contract_sha256" \
  --arg prompt_provenance_sha256 "$PROMPT_PROVENANCE_SHA256" \
  --arg serialization_policy "$serialization_policy" \
  --arg request_sha256 "$request_sha256" \
  --argjson request_bytes "$request_bytes" \
  --arg rendered_prompt_sha256 "$rendered_prompt_sha256" \
  --arg prompt_token_ids_sha256 "$prompt_token_ids_sha256" \
  --arg tool_result_fixture_sha256 "$tool_result_fixture_sha256" \
  --argjson tool_result_fixture_bytes "$tool_result_fixture_bytes" \
  --arg tool_result_payload_sha256 "$tool_result_payload_sha256" \
  --arg agentic_context_fixture_sha256 "$actual_context_fixture_sha256" \
  --argjson agentic_context_fixture_bytes "$agentic_context_fixture_bytes" \
  --argjson repository_context_chars "$repository_context_chars" \
  --arg expected_path "$EXPECTED_PATH" \
  --arg agentic_system_prompt_sha256 "$agentic_system_prompt_sha256" \
  --arg tool_result_success_prefix_sha256 "$tool_result_success_prefix_sha256" \
  --argjson expected_prompt_tokens "$EXPECTED_PROMPT_TOKENS" \
  --argjson cold_cached_tokens "$cold_cached" \
  --argjson require_cold_first "$REQUIRE_COLD_FIRST" \
  --argjson prompt_tokens "$prompt_tokens" \
  --argjson cached_tokens "$cached_tokens" \
  --argjson continuation_prompt_tokens "$continuation_prompt_tokens" \
  --argjson continuation_cached "$continuation_cached" \
  --argjson continuation_uncached "$continuation_uncached" \
  --argjson auto_cached "$auto_cached" \
  --argjson stream_cached "$stream_cached" \
  --argjson cold_ttft_ms "$cold_ttft" \
  --argjson cached_ttft_ms "$cached_ttft" \
  --argjson cold_response_ms "$cold_response_ms" \
  --argjson cached_response_ms "$cached_response_ms" \
  --argjson auto_response_ms "$auto_response_ms" \
  --argjson stream_semantic_ms "$stream_semantic_ms" \
  --argjson continuation_response_ms "$continuation_response_ms" \
  --argjson source_response_ms "$source_response_ms" '{
    status: "pass",
    agent: $agent,
    run_id: $run_id,
    fixture_id: $fixture_id,
    prompt_contract_sha256: $prompt_contract_sha256,
    prompt_provenance_sha256: $prompt_provenance_sha256,
    serialization_policy: $serialization_policy,
    request_sha256: $request_sha256,
    request_bytes: $request_bytes,
    rendered_prompt_sha256: $rendered_prompt_sha256,
    prompt_token_ids_sha256: $prompt_token_ids_sha256,
    tool_result_fixture_sha256: $tool_result_fixture_sha256,
    tool_result_fixture_bytes: $tool_result_fixture_bytes,
    tool_result_payload_sha256: $tool_result_payload_sha256,
    agentic_context_fixture_sha256: $agentic_context_fixture_sha256,
    agentic_context_fixture_bytes: $agentic_context_fixture_bytes,
    repository_context_chars: $repository_context_chars,
    expected_path: $expected_path,
    agentic_system_prompt_sha256: $agentic_system_prompt_sha256,
    tool_result_success_prefix_sha256: $tool_result_success_prefix_sha256,
    expected_prompt_tokens: $expected_prompt_tokens,
    tool_semantics_pass: true,
    cached_replay_equal: true,
    automatic_tool_call_pass: true,
    sse_tool_call_pass: true,
    tool_result_continuation_pass: true,
    source_tool_syntax_pass: true,
    cold_cached_tokens: $cold_cached_tokens,
    require_cold_first: $require_cold_first,
    prompt_tokens: $prompt_tokens,
    cached_tokens: $cached_tokens,
    auto_cached_tokens: $auto_cached,
    stream_cached_tokens: $stream_cached,
    continuation_prompt_tokens: $continuation_prompt_tokens,
    continuation_cached_tokens: $continuation_cached,
    continuation_uncached_tokens: $continuation_uncached,
    cold_ttft_ms: $cold_ttft_ms,
    cached_ttft_ms: $cached_ttft_ms,
    cold_semantic_response_ms: $cold_response_ms,
    cached_semantic_response_ms: $cached_response_ms,
    auto_semantic_response_ms: $auto_response_ms,
    cached_sse_tool_call_ms: $stream_semantic_ms,
    tool_result_response_ms: $continuation_response_ms,
    source_tool_response_ms: $source_response_ms
  }'
