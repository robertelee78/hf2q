#!/usr/bin/env bash
set -euo pipefail

# Long-context DeepSeek/OpenCode cache gate for an already-running hf2q
# server.  A short calibration prompt is extended to approximately 120K
# actual tokenizer tokens.  That calibrated prompt is intentionally a cache
# miss; a tool result is then appended as the next agentic turn.  The final
# request reserves enough generation headroom to cross the initial 131,072
# physical-cache boundary and must still reuse nearly the whole long prefix
# instead of prefilling the conversation again. A final tiny user turn proves
# that a retained 9..32-token suffix takes incremental replay instead of the
# empty-chunk failure that escaped in v0.1.5.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:8080}
MODEL=${MODEL:-Deepseek v4 Flash 0731 Source}
EXPECTED_PATH=${EXPECTED_PATH:-$ROOT_DIR/Cargo.toml}
RUN_ID=${RUN_ID:-"$$-$(date +%s)"}
TARGET_PROMPT_TOKENS=${TARGET_PROMPT_TOKENS:-121000}
MIN_LONG_PROMPT_TOKENS=${MIN_LONG_PROMPT_TOKENS:-116000}
MAX_LONG_PROMPT_TOKENS=${MAX_LONG_PROMPT_TOKENS:-125000}
CALIBRATION_CHARS=${CALIBRATION_CHARS:-131072}
# Historical source-bound M5 Max acceptance: llama.cpp (b10293-a1f96d4fc)
# processes the same 121K-class prompt in roughly 556 s (217.5 tok/s).
# Keep a small variance margin while still failing any regression to the old
# hf2q path, which returned no bytes before its 600 s client timeout.
MAX_GROWTH_TTFT_MS=${MAX_GROWTH_TTFT_MS:-540000}
MAX_CACHED_TTFT_MS=${MAX_CACHED_TTFT_MS:-15000}
MAX_CACHED_RESPONSE_MS=${MAX_CACHED_RESPONSE_MS:-30000}
CURL_CONNECT_TIMEOUT_SECONDS=${CURL_CONNECT_TIMEOUT_SECONDS:-5}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-600}
MAX_TOKENS=${MAX_TOKENS:-128}
# The long prompt itself fits the initial 131,072-token cache. Reserving this
# continuation budget deliberately crosses that physical boundary so the gate
# proves lossless cache growth rather than only same-capacity prefix reuse.
CONTINUATION_MAX_TOKENS=${CONTINUATION_MAX_TOKENS:-16384}
SENTINEL=${SENTINEL:-HF2Q_DEEPSEEK_LONG_CONTEXT_CACHE_OK}
SHORT_TAIL_SENTINEL=${SHORT_TAIL_SENTINEL:-HF2Q_DEEPSEEK_SHORT_TAIL_OK}

for command in awk curl date dirname jq mktemp rm sed; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
for setting in TARGET_PROMPT_TOKENS MIN_LONG_PROMPT_TOKENS \
  MAX_LONG_PROMPT_TOKENS CALIBRATION_CHARS MAX_GROWTH_TTFT_MS \
  MAX_CACHED_TTFT_MS MAX_CACHED_RESPONSE_MS CURL_CONNECT_TIMEOUT_SECONDS \
  CURL_MAX_TIME_SECONDS MAX_TOKENS CONTINUATION_MAX_TOKENS; do
  value=${!setting}
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$setting must be a positive integer (got: $value)" >&2
    exit 2
  fi
done
if (( MIN_LONG_PROMPT_TOKENS >= TARGET_PROMPT_TOKENS ||
      TARGET_PROMPT_TOKENS >= MAX_LONG_PROMPT_TOKENS )); then
  echo "expected MIN_LONG_PROMPT_TOKENS < TARGET_PROMPT_TOKENS < MAX_LONG_PROMPT_TOKENS" >&2
  exit 2
fi

work_dir=$(mktemp -d -t hf2q-deepseek-long-cache.XXXXXX)
cleanup() {
  rm -rf "$work_dir"
}
trap cleanup EXIT

calibration_context="$work_dir/calibration.txt"
long_context="$work_dir/long.txt"
calibration_request="$work_dir/calibration-request.json"
calibration_response="$work_dir/calibration-response.json"
long_request="$work_dir/long-request.json"
long_response="$work_dir/long-response.json"
continuation_request="$work_dir/continuation-request.json"
continuation_response="$work_dir/continuation-response.json"
short_tail_request="$work_dir/short-tail-request.json"
short_tail_response="$work_dir/short-tail-response.json"

epoch_ms() {
  echo $(( $(date +%s) * 1000 ))
}

generate_context() {
  local chars=$1
  local output=$2
  awk -v target="$chars" 'BEGIN {
    unit = "The quick brown fox reviews safe Rust code and verifies every test. "
    unit_len = length(unit)
    written = 0
    while (written < target) {
      remaining = target - written
      if (remaining < unit_len) {
        printf "%s", substr(unit, 1, remaining)
        written += remaining
      } else {
        printf "%s", unit
        written += unit_len
      }
    }
  }' >"$output"
}

make_tool_request() {
  local context_file=$1
  local output=$2
  local phase=$3
  jq -n --rawfile context "$context_file" \
    --arg model "$MODEL" --arg path "$EXPECTED_PATH" --arg run_id "$RUN_ID" \
    --arg phase "$phase" --arg sentinel "$SENTINEL" \
    --argjson max_tokens "$MAX_TOKENS" '{
      model: $model,
      messages: [
        {
          role: "system",
          content: "You are an agentic coding assistant. Use the supplied tool exactly once."
        },
        {
          role: "user",
          content: (
            $phase + " run " + $run_id +
            ". After reading the repository context below, call inspect_file for exactly " +
            $path + ". After the tool result confirms success, reply with exactly " +
            $sentinel + " and nothing else. Repository context follows:\n\n" + $context
          )
        }
      ],
      tools: [{
        type: "function",
        function: {
          name: "inspect_file",
          description: "Inspect one UTF-8 source file",
          parameters: {
            type: "object",
            properties: {path: {type: "string"}},
            required: ["path"],
            additionalProperties: false
          }
        }
      }],
      tool_choice: "required",
      temperature: 0,
      max_tokens: $max_tokens,
      stream: false
    }' >"$output"
}

post_json() {
  local input=$1
  local output=$2
  if ! curl --fail-with-body --silent --show-error --no-buffer \
    --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
    --max-time "$CURL_MAX_TIME_SECONDS" \
    -H 'Content-Type: application/json' --data-binary "@$input" \
    "$BASE_URL/v1/chat/completions" >"$output"; then
    echo "chat completion failed; response body:" >&2
    sed -n '1,120p' "$output" >&2
    return 1
  fi
}

assert_tool_path() {
  local response=$1
  if ! jq -e --arg expected "$EXPECTED_PATH" '
    (.choices | length) == 1
    and .choices[0].finish_reason == "tool_calls"
    and ((.choices[0].message.tool_calls // []) | length) == 1
    and .choices[0].message.tool_calls[0].function.name == "inspect_file"
    and ((.choices[0].message.tool_calls[0].function.arguments | fromjson).path == $expected)
  ' "$response" >/dev/null; then
    echo "long-context gate failed: expected one inspect_file call for $EXPECTED_PATH" >&2
    jq '{choice: .choices[0], usage, timing: .x_hf2q_timing}' "$response" >&2
    exit 1
  fi
}

# Calibrate from actual server tokenization instead of assuming a byte/token
# ratio.  The long request uses a different phrase at the start of the user
# message, before the repeated context, so neither the live sequence nor its
# recovery anchor can contaminate the measured cold prefill.
generate_context "$CALIBRATION_CHARS" "$calibration_context"
make_tool_request "$calibration_context" "$calibration_request" \
  "Tokenizer-size calibration"
post_json "$calibration_request" "$calibration_response"
assert_tool_path "$calibration_response"
calibration_tokens=$(jq -r '.usage.prompt_tokens // 0' "$calibration_response")
calibration_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$calibration_response")
if (( calibration_tokens < 1024 || calibration_cached != 0 )); then
  echo "long-context gate failed: invalid cold calibration usage (prompt=$calibration_tokens cached=$calibration_cached)" >&2
  exit 1
fi

long_chars=$(( CALIBRATION_CHARS * TARGET_PROMPT_TOKENS / calibration_tokens ))
if (( long_chars <= CALIBRATION_CHARS )); then
  echo "long-context gate failed: calibration did not produce a growing prompt" >&2
  exit 1
fi
generate_context "$long_chars" "$long_context"
make_tool_request "$long_context" "$long_request" \
  "Long-context cache acceptance"

growth_started=$(epoch_ms)
post_json "$long_request" "$long_response"
growth_response_ms=$(( $(epoch_ms) - growth_started ))
assert_tool_path "$long_response"
long_prompt_tokens=$(jq -r '.usage.prompt_tokens // 0' "$long_response")
long_cached_tokens=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$long_response")
growth_ttft_ms=$(jq -r '.x_hf2q_timing.time_to_first_token_ms // -1' "$long_response")
if (( long_prompt_tokens < MIN_LONG_PROMPT_TOKENS ||
      long_prompt_tokens > MAX_LONG_PROMPT_TOKENS )); then
  echo "long-context gate failed: calibrated prompt has $long_prompt_tokens tokens; expected $MIN_LONG_PROMPT_TOKENS..$MAX_LONG_PROMPT_TOKENS" >&2
  exit 1
fi
if (( long_cached_tokens != 0 )); then
  echo "long-context gate failed: calibrated long prompt was not cold (cached_tokens=$long_cached_tokens)" >&2
  exit 1
fi
if ! awk -v actual="$growth_ttft_ms" -v limit="$MAX_GROWTH_TTFT_MS" \
  'BEGIN { exit !(actual >= 0 && actual <= limit) }'; then
  echo "long-context gate failed: growing-prompt TTFT ${growth_ttft_ms}ms exceeds ${MAX_GROWTH_TTFT_MS}ms" >&2
  exit 1
fi

# Append the model's real tool call and a small tool result.  This is the
# normal second half of an OpenCode turn, and must reuse the ~120K prefix.
jq -n --slurpfile base "$long_request" --slurpfile prior "$long_response" \
  --argjson continuation_max_tokens "$CONTINUATION_MAX_TOKENS" '
    $base[0]
    | .messages += [
        {
          role: "assistant",
          content: null,
          tool_calls: $prior[0].choices[0].message.tool_calls
        },
        {
          role: "tool",
          tool_call_id: $prior[0].choices[0].message.tool_calls[0].id,
          content: "File inspection succeeded."
        }
      ]
    | .tool_choice = "auto"
    | .max_tokens = $continuation_max_tokens
  ' >"$continuation_request"

continuation_started=$(epoch_ms)
post_json "$continuation_request" "$continuation_response"
continuation_response_ms=$(( $(epoch_ms) - continuation_started ))
continuation_prompt_tokens=$(jq -r '.usage.prompt_tokens // 0' "$continuation_response")
continuation_cached_tokens=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$continuation_response")
continuation_ttft_ms=$(jq -r '.x_hf2q_timing.time_to_first_token_ms // -1' "$continuation_response")
continuation_content=$(jq -r '.choices[0].message.content // empty' "$continuation_response")
minimum_continuation_cached=$((long_prompt_tokens - 64))
if (( continuation_cached_tokens < minimum_continuation_cached )); then
  echo "long-context gate failed: tool-result turn reused $continuation_cached_tokens/$continuation_prompt_tokens tokens; expected at least $minimum_continuation_cached" >&2
  exit 1
fi
if ! awk -v actual="$continuation_ttft_ms" -v limit="$MAX_CACHED_TTFT_MS" \
  'BEGIN { exit !(actual >= 0 && actual <= limit) }'; then
  echo "long-context gate failed: suffix-only TTFT ${continuation_ttft_ms}ms exceeds ${MAX_CACHED_TTFT_MS}ms" >&2
  exit 1
fi
if (( continuation_response_ms > MAX_CACHED_RESPONSE_MS )); then
  echo "long-context gate failed: cached continuation took ${continuation_response_ms}ms; limit is ${MAX_CACHED_RESPONSE_MS}ms" >&2
  exit 1
fi
if [[ "$continuation_content" != "$SENTINEL" ]] || ! jq -e --arg sentinel "$SENTINEL" '
  (.choices | length) == 1
  and .choices[0].finish_reason == "stop"
  and .choices[0].message.content == $sentinel
  and ((.choices[0].message.tool_calls // []) | length) == 0
' "$continuation_response" >/dev/null; then
  echo "long-context gate failed: continuation was not one terminal sentinel response" >&2
  jq '.choices[0]' "$continuation_response" >&2
  exit 1
fi

# Follow the retained 100K+ conversation with one deliberately tiny user turn.
# DeepSeek matrix append requires at least 33 cached-suffix tokens, while the
# recovery checkpoint covers the final eight prompt tokens. This request must
# therefore exercise a 9..32-token incremental segment before or across that
# boundary instead of selecting an empty resumable-prefill chunk.
jq -n --slurpfile base "$continuation_request" --slurpfile prior "$continuation_response" \
  --arg sentinel "$SHORT_TAIL_SENTINEL" '
    $base[0]
    | .messages += [
        {
          role: "assistant",
          content: $prior[0].choices[0].message.content
        },
        {
          role: "user",
          content: ("Reply exactly " + $sentinel + ".")
        }
      ]
    | .tool_choice = "auto"
    | .max_tokens = 32
  ' >"$short_tail_request"

short_tail_started=$(epoch_ms)
post_json "$short_tail_request" "$short_tail_response"
short_tail_response_ms=$(( $(epoch_ms) - short_tail_started ))
short_tail_prompt_tokens=$(jq -r '.usage.prompt_tokens // 0' "$short_tail_response")
short_tail_cached_tokens=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$short_tail_response")
short_tail_suffix_tokens=$((short_tail_prompt_tokens - short_tail_cached_tokens))
short_tail_content=$(jq -r '.choices[0].message.content // empty' "$short_tail_response")
if (( short_tail_cached_tokens < minimum_continuation_cached )); then
  echo "long-context gate failed: short follow-up reused only $short_tail_cached_tokens/$short_tail_prompt_tokens tokens" >&2
  exit 1
fi
if (( short_tail_suffix_tokens < 9 || short_tail_suffix_tokens >= 33 )); then
  echo "long-context gate failed: short follow-up suffix was $short_tail_suffix_tokens tokens; expected 9..32" >&2
  exit 1
fi
if [[ "$short_tail_content" != "$SHORT_TAIL_SENTINEL" ]] || ! jq -e --arg sentinel "$SHORT_TAIL_SENTINEL" '
  (.choices | length) == 1
  and .choices[0].finish_reason == "stop"
  and .choices[0].message.content == $sentinel
  and ((.choices[0].message.tool_calls // []) | length) == 0
' "$short_tail_response" >/dev/null; then
  echo "long-context gate failed: short cached continuation was not one terminal sentinel response" >&2
  jq '.choices[0]' "$short_tail_response" >&2
  exit 1
fi

jq -n \
  --argjson calibration_tokens "$calibration_tokens" \
  --argjson long_prompt_tokens "$long_prompt_tokens" \
  --argjson long_cached_tokens "$long_cached_tokens" \
  --argjson growth_ttft_ms "$growth_ttft_ms" \
  --argjson growth_response_ms "$growth_response_ms" \
  --argjson continuation_prompt_tokens "$continuation_prompt_tokens" \
  --argjson continuation_cached_tokens "$continuation_cached_tokens" \
  --argjson continuation_max_tokens "$CONTINUATION_MAX_TOKENS" \
  --argjson continuation_ttft_ms "$continuation_ttft_ms" \
  --argjson continuation_response_ms "$continuation_response_ms" \
  --argjson short_tail_prompt_tokens "$short_tail_prompt_tokens" \
  --argjson short_tail_cached_tokens "$short_tail_cached_tokens" \
  --argjson short_tail_suffix_tokens "$short_tail_suffix_tokens" \
  --argjson short_tail_response_ms "$short_tail_response_ms" '{
    status: "pass",
    calibration_tokens: $calibration_tokens,
    growing_prompt: {
      prompt_tokens: $long_prompt_tokens,
      cached_tokens: $long_cached_tokens,
      ttft_ms: $growth_ttft_ms,
      response_ms: $growth_response_ms
    },
    tool_result_continuation: {
      prompt_tokens: $continuation_prompt_tokens,
      cached_tokens: $continuation_cached_tokens,
      requested_max_tokens: $continuation_max_tokens,
      suffix_tokens: ($continuation_prompt_tokens - $continuation_cached_tokens),
      cached_percent: (
        if $continuation_prompt_tokens == 0 then 0
        else (10000 * $continuation_cached_tokens / $continuation_prompt_tokens | floor) / 100
        end
      ),
      ttft_ms: $continuation_ttft_ms,
      response_ms: $continuation_response_ms
    },
    short_tail_continuation: {
      prompt_tokens: $short_tail_prompt_tokens,
      cached_tokens: $short_tail_cached_tokens,
      suffix_tokens: $short_tail_suffix_tokens,
      response_ms: $short_tail_response_ms
    }
  }'
