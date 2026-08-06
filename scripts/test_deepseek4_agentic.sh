#!/usr/bin/env bash
set -euo pipefail

# Realistic DeepSeek/OpenCode acceptance gate for an already-running hf2q
# server. It intentionally fails when tool semantics, prefix reuse, TTFT, SSE,
# or tool-result continuation are not usable for an agentic coding turn.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:8080}
MODEL=${MODEL:-Deepseek v4 Flash 0731 Source}
EXPECTED_PATH=${EXPECTED_PATH:-$ROOT_DIR/Cargo.toml}
RUN_ID=${RUN_ID:-"$$-$(date +%s)"}
MAX_COLD_TTFT_MS=${MAX_COLD_TTFT_MS:-30000}
MAX_CACHED_TTFT_MS=${MAX_CACHED_TTFT_MS:-5000}
MAX_COLD_RESPONSE_MS=${MAX_COLD_RESPONSE_MS:-40000}
MAX_CACHED_RESPONSE_MS=${MAX_CACHED_RESPONSE_MS:-10000}
MAX_CACHED_SEMANTIC_MS=${MAX_CACHED_SEMANTIC_MS:-10000}
CURL_CONNECT_TIMEOUT_SECONDS=${CURL_CONNECT_TIMEOUT_SECONDS:-5}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-60}
MAX_TOKENS=${MAX_TOKENS:-128}
SOURCE_MAX_TOKENS=${SOURCE_MAX_TOKENS:-256}
SENTINEL=${SENTINEL:-SENTINEL_CARGO_HF2Q_AGENTIC}
EXPECTED_SOURCE=${EXPECTED_SOURCE:-"fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result"}

for command in curl date jq rg; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
for setting in MAX_TOKENS SOURCE_MAX_TOKENS CURL_CONNECT_TIMEOUT_SECONDS CURL_MAX_TIME_SECONDS; do
  value=${!setting}
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$setting must be a positive integer (got: $value)" >&2
    exit 2
  fi
done

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
cleanup() {
  rm -f "$request_file" "$first_file" "$second_file" "$auto_request_file" \
    "$auto_file" "$stream_file" \
    "$stream_json_file" "$stream_timing_file" "$continuation_file" \
    "$continuation_response" "$source_request_file" "$source_response_file"
}
trap cleanup EXIT

cd "$ROOT_DIR"

jq -n --rawfile repo README.md \
  --argjson max_tokens "$MAX_TOKENS" \
  --arg model "$MODEL" --arg expected_path "$EXPECTED_PATH" --arg run_id "$RUN_ID" '{
    model: $model,
    messages: [
      {
        role: "system",
        content: "You are an agentic coding assistant. Use the provided tool to inspect files before answering."
      },
      {
        role: "user",
        content: ("Agentic acceptance run " + $run_id + ". Inspect this Rust repository context and read " + $expected_path + " before making any recommendation. The requested manifest is intentionally not embedded; use read_file with exactly that absolute path. Repository context follows:\n\n" + $repo)
      }
    ],
    tools: [{
      type: "function",
      function: {
        name: "read_file",
        description: "Read a UTF-8 text file from the local workspace",
        parameters: {
          type: "object",
          properties: {path: {type: "string", description: "Absolute file path"}},
          required: ["path"],
          additionalProperties: false
        }
      }
    }],
    tool_choice: "required",
    temperature: 0,
    max_tokens: $max_tokens,
    stream: false
  }' >"$request_file"

post_json() {
  local input=$1
  local output=$2
  if ! curl --fail-with-body --silent --show-error --no-buffer \
    --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
    --max-time "$CURL_MAX_TIME_SECONDS" \
    -H 'Content-Type: application/json' \
    --data-binary "@$input" \
    "$BASE_URL/v1/chat/completions" >"$output"; then
    echo "chat completion request failed; response body:" >&2
    sed -n '1,120p' "$output" >&2
    return 1
  fi
}

epoch_ms() {
  echo $(( $(date +%s) * 1000 ))
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

cold_started=$(epoch_ms)
post_json "$request_file" "$first_file"
cold_response_ms=$(( $(epoch_ms) - cold_started ))
assert_tool_path "$first_file"

cold_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$first_file")
if (( cold_cached != 0 )); then
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

cached_started=$(epoch_ms)
post_json "$request_file" "$second_file"
cached_response_ms=$(( $(epoch_ms) - cached_started ))
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

prompt_tokens=$(jq -r '.usage.prompt_tokens' "$second_file")
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
auto_started=$(epoch_ms)
post_json "$auto_request_file" "$auto_file"
auto_response_ms=$(( $(epoch_ms) - auto_started ))
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

stream_started=$(epoch_ms)
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
        echo $(( $(epoch_ms) - stream_started )) >"$stream_timing_file"
      fi
    fi
  done

done_count=$(rg -c '^data: \[DONE\]$' "$stream_file" || true)
last_event=$(rg '^data: ' "$stream_file" | tail -1 || true)
if [[ "$done_count" != "1" || "$last_event" != "data: [DONE]" ]]; then
  echo "agentic gate failed: SSE stream did not end with exactly one [DONE]" >&2
  exit 1
fi
sed -n 's/^data: //p' "$stream_file" | rg -v '^\[DONE\]$' >"$stream_json_file"
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
  --rawfile cargo Cargo.toml --arg sentinel "$SENTINEL" '
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
          content: ("Successful read_file result. Cargo.toml follows:\n" + $cargo + "\nReply with exactly this sentinel and nothing else: " + $sentinel)
        }
      ]
    | .tool_choice = "auto"
    | .stream = false
  ' >"$continuation_file"

continuation_started=$(epoch_ms)
post_json "$continuation_file" "$continuation_response"
continuation_response_ms=$(( $(epoch_ms) - continuation_started ))
continuation_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$continuation_response")
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
if (( continuation_response_ms > MAX_CACHED_RESPONSE_MS )); then
  echo "agentic gate failed: tool-result response took ${continuation_response_ms}ms; limit is ${MAX_CACHED_RESPONSE_MS}ms" >&2
  exit 1
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

source_started=$(epoch_ms)
post_json "$source_request_file" "$source_response_file"
source_response_ms=$(( $(epoch_ms) - source_started ))
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
  --argjson prompt_tokens "$prompt_tokens" \
  --argjson cached_tokens "$cached_tokens" \
  --argjson continuation_cached "$continuation_cached" \
  --argjson auto_cached "$auto_cached" \
  --argjson cold_ttft_ms "$cold_ttft" \
  --argjson cached_ttft_ms "$cached_ttft" \
  --argjson cold_response_ms "$cold_response_ms" \
  --argjson cached_response_ms "$cached_response_ms" \
  --argjson auto_response_ms "$auto_response_ms" \
  --argjson stream_semantic_ms "$stream_semantic_ms" \
  --argjson continuation_response_ms "$continuation_response_ms" \
  --argjson source_response_ms "$source_response_ms" '{
    status: "pass",
    prompt_tokens: $prompt_tokens,
    cached_tokens: $cached_tokens,
    auto_cached_tokens: $auto_cached,
    continuation_cached_tokens: $continuation_cached,
    cold_ttft_ms: $cold_ttft_ms,
    cached_ttft_ms: $cached_ttft_ms,
    cold_semantic_response_ms: $cold_response_ms,
    cached_semantic_response_ms: $cached_response_ms,
    auto_semantic_response_ms: $auto_response_ms,
    cached_sse_tool_call_ms: $stream_semantic_ms,
    tool_result_response_ms: $continuation_response_ms,
    source_tool_response_ms: $source_response_ms
  }'
