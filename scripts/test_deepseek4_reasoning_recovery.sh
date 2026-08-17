#!/usr/bin/env bash
set -euo pipefail

# Incident-class DeepSeek-V4 recovery replay for an already-running hf2q
# server. This is a deterministic sanitized reconstruction, not the original
# private OpenCode transcript: it calibrates a reasoning_content-heavy history
# to 168K..178K actual server tokens, makes one cold tool call, appends its tool
# result, then appends a tiny new tool turn. Both continuations must reuse the
# long prefix and preserve native reasoning + tool-call semantics.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:8081}
MODEL=${MODEL:-}
EXPECTED_PATH=${EXPECTED_PATH:-$ROOT_DIR/Cargo.toml}
TARGET_PROMPT_TOKENS=${TARGET_PROMPT_TOKENS:-173000}
MIN_PROMPT_TOKENS=${MIN_PROMPT_TOKENS:-168000}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-178000}
CALIBRATION_CHARS=${CALIBRATION_CHARS:-131072}
MAX_TOKENS=${MAX_TOKENS:-256}
RECOVERY_MAX_TOKENS=${RECOVERY_MAX_TOKENS:-512}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-1200}
MIN_CACHE_MARGIN_TOKENS=${MIN_CACHE_MARGIN_TOKENS:-128}
KEEP_WORK_DIR=${KEEP_WORK_DIR:-0}
RUN_ID=${RUN_ID:-"$$-$(date +%s)"}
SENTINEL=${SENTINEL:-HF2Q_DEEPSEEK_REASONING_RECOVERY_OK}

for command in awk curl date jq mktemp rm sed shasum; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
for setting in TARGET_PROMPT_TOKENS MIN_PROMPT_TOKENS MAX_PROMPT_TOKENS \
  CALIBRATION_CHARS MAX_TOKENS RECOVERY_MAX_TOKENS CURL_MAX_TIME_SECONDS \
  MIN_CACHE_MARGIN_TOKENS; do
  value=${!setting}
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$setting must be a positive integer (got: $value)" >&2
    exit 2
  fi
done
if (( MIN_PROMPT_TOKENS >= TARGET_PROMPT_TOKENS ||
      TARGET_PROMPT_TOKENS >= MAX_PROMPT_TOKENS )); then
  echo "expected MIN_PROMPT_TOKENS < TARGET_PROMPT_TOKENS < MAX_PROMPT_TOKENS" >&2
  exit 2
fi
if [[ "$KEEP_WORK_DIR" != 0 && "$KEEP_WORK_DIR" != 1 ]]; then
  echo "KEEP_WORK_DIR must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$BASE_URL" =~ ^http://(127\.0\.0\.1|localhost):[0-9]+$ ]]; then
  echo "BASE_URL must be a loopback endpoint without /v1" >&2
  exit 2
fi

if [[ -z "$MODEL" ]]; then
  MODEL=$(curl --fail-with-body --silent --show-error \
    "$BASE_URL/v1/models" | jq -er '[.data[] | select(.loaded == true and .arch == "deepseek4")][0].id')
fi
[[ -n "$MODEL" ]] || { echo "no loaded DeepSeek-V4 model found" >&2; exit 1; }

work_dir=$(mktemp -d -t hf2q-deepseek-recovery.XXXXXX)
cleanup() {
  local status=$?
  if (( KEEP_WORK_DIR == 1 || status != 0 )); then
    echo "reasoning-recovery workspace retained at $work_dir" >&2
  else
    rm -rf "$work_dir"
  fi
}
trap cleanup EXIT

calibration_archive="$work_dir/calibration.txt"
long_archive="$work_dir/long.txt"
calibration_request="$work_dir/calibration-request.json"
calibration_response="$work_dir/calibration-response.json"
long_request="$work_dir/long-request.json"
long_response="$work_dir/long-response.json"
continuation_request="$work_dir/continuation-request.json"
continuation_response="$work_dir/continuation-response.json"
recovery_request="$work_dir/recovery-request.json"
recovery_response="$work_dir/recovery-response.json"

generate_archive() {
  local chars=$1
  local output=$2
  awk -v target="$chars" 'BEGIN {
    unit = "Reasoning archive: inspect evidence, preserve exact tool arguments, verify the result, and never repeat a completed call. "
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

make_request() {
  local archive=$1
  local output=$2
  local phase=$3
  jq -n --rawfile archive "$archive" \
    --arg model "$MODEL" --arg path "$EXPECTED_PATH" \
    --arg run_id "$RUN_ID" --arg phase "$phase" --arg sentinel "$SENTINEL" \
    --argjson max_tokens "$MAX_TOKENS" '{
      model: $model,
      messages: [
        {
          role: "system",
          content: ($phase + " sanitized replay " + $run_id + ". Preserve reasoning_content and use tools exactly as requested.")
        },
        {role: "user", content: "Review the archived reasoning before continuing."},
        {
          role: "assistant",
          reasoning_content: $archive,
          content: null,
          tool_calls: [{
            id: "archive_call",
            type: "function",
            function: {name: "inspect_file", arguments: "{\"path\":\"archive.md\"}"}
          }]
        },
        {role: "tool", tool_call_id: "archive_call", content: "Archived inspection completed successfully."},
        {
          role: "user",
          content: ("Call inspect_file exactly once for " + $path + ". After its successful tool result, reply exactly " + $sentinel + ".")
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
      top_p: 0.95,
      reasoning_effort: "max",
      max_tokens: $max_tokens,
      stream: false
    }' >"$output"
}

post_json() {
  local input=$1
  local output=$2
  if ! curl --fail-with-body --silent --show-error --no-buffer \
    --connect-timeout 5 --max-time "$CURL_MAX_TIME_SECONDS" \
    -H 'content-type: application/json' --data-binary "@$input" \
    "$BASE_URL/v1/chat/completions" >"$output"; then
    echo "chat completion failed; response body:" >&2
    sed -n '1,120p' "$output" >&2
    return 1
  fi
}

assert_tool_call() {
  local response=$1
  if ! jq -e --arg path "$EXPECTED_PATH" '
    (.choices | length) == 1
    and .choices[0].finish_reason == "tool_calls"
    and (.choices[0].message.reasoning_content | type == "string" and length > 0)
    and ((.choices[0].message.tool_calls // []) | length) == 1
    and .choices[0].message.tool_calls[0].function.name == "inspect_file"
    and ((.choices[0].message.tool_calls[0].function.arguments | fromjson).path == $path)
  ' "$response" >/dev/null; then
    echo "reasoning-recovery gate failed: expected one reasoned inspect_file call for $EXPECTED_PATH" >&2
    jq '{choice: .choices[0], usage, timing: .x_hf2q_timing}' "$response" >&2
    exit 1
  fi
}

generate_archive "$CALIBRATION_CHARS" "$calibration_archive"
make_request "$calibration_archive" "$calibration_request" "Calibration"
post_json "$calibration_request" "$calibration_response"
assert_tool_call "$calibration_response"
calibration_tokens=$(jq -r '.usage.prompt_tokens // 0' "$calibration_response")
if (( calibration_tokens < 1024 )); then
  echo "reasoning-recovery gate failed: invalid calibration token count $calibration_tokens" >&2
  exit 1
fi

long_chars=$((CALIBRATION_CHARS * TARGET_PROMPT_TOKENS / calibration_tokens))
generate_archive "$long_chars" "$long_archive"
make_request "$long_archive" "$long_request" "Cold incident-class"
post_json "$long_request" "$long_response"
assert_tool_call "$long_response"
long_prompt_tokens=$(jq -r '.usage.prompt_tokens // 0' "$long_response")
long_cached_tokens=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$long_response")
if (( long_prompt_tokens < MIN_PROMPT_TOKENS || long_prompt_tokens > MAX_PROMPT_TOKENS )); then
  echo "reasoning-recovery gate failed: prompt=$long_prompt_tokens, expected $MIN_PROMPT_TOKENS..$MAX_PROMPT_TOKENS" >&2
  exit 1
fi
if (( long_cached_tokens != 0 )); then
  echo "reasoning-recovery gate failed: cold prompt unexpectedly reused $long_cached_tokens tokens" >&2
  exit 1
fi

jq -n --slurpfile base "$long_request" --slurpfile prior "$long_response" \
  --arg sentinel "$SENTINEL" '
    $base[0]
    | .messages += [
        {
          role: "assistant",
          reasoning_content: $prior[0].choices[0].message.reasoning_content,
          content: $prior[0].choices[0].message.content,
          tool_calls: $prior[0].choices[0].message.tool_calls
        },
        {
          role: "tool",
          tool_call_id: $prior[0].choices[0].message.tool_calls[0].id,
          content: "File inspection succeeded."
        }
      ]
    | .tool_choice = "auto"
    | .max_tokens = 128
  ' >"$continuation_request"
post_json "$continuation_request" "$continuation_response"
continuation_prompt_tokens=$(jq -r '.usage.prompt_tokens // 0' "$continuation_response")
continuation_cached_tokens=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$continuation_response")
minimum_cached=$((long_prompt_tokens - MIN_CACHE_MARGIN_TOKENS))
if (( continuation_cached_tokens < minimum_cached )) || ! jq -e --arg sentinel "$SENTINEL" '
  .choices[0].finish_reason == "stop"
  and .choices[0].message.content == $sentinel
  and ((.choices[0].message.tool_calls // []) | length) == 0
' "$continuation_response" >/dev/null; then
  echo "reasoning-recovery gate failed: cached tool-result continuation was invalid" >&2
  jq '{choice: .choices[0], usage, timing: .x_hf2q_timing}' "$continuation_response" >&2
  exit 1
fi

jq -n --slurpfile base "$continuation_request" --slurpfile prior "$continuation_response" \
  --arg path "$EXPECTED_PATH" --argjson recovery_max_tokens "$RECOVERY_MAX_TOKENS" '
    $base[0]
    | .messages += [
        {
          role: "assistant",
          reasoning_content: $prior[0].choices[0].message.reasoning_content,
          content: $prior[0].choices[0].message.content
        },
        {role: "user", content: ("Recovery check: call inspect_file once for " + $path + ".")}
      ]
    | .tool_choice = "required"
    | .max_tokens = $recovery_max_tokens
  ' >"$recovery_request"
post_json "$recovery_request" "$recovery_response"
assert_tool_call "$recovery_response"
recovery_prompt_tokens=$(jq -r '.usage.prompt_tokens // 0' "$recovery_response")
recovery_cached_tokens=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$recovery_response")
if (( recovery_cached_tokens < continuation_cached_tokens )); then
  echo "reasoning-recovery gate failed: replay reused only $recovery_cached_tokens tokens after $continuation_cached_tokens" >&2
  exit 1
fi

fixture_sha256=$(shasum -a 256 "$long_archive" | awk '{print $1}')
request_sha256=$(shasum -a 256 "$long_request" | awk '{print $1}')
response_sha256=$(jq -Sc '{choices,usage,x_hf2q_timing}' "$recovery_response" | shasum -a 256 | awk '{print $1}')
jq -n \
  --arg model "$MODEL" --arg fixture_sha256 "$fixture_sha256" \
  --arg request_sha256 "$request_sha256" --arg response_sha256 "$response_sha256" \
  --argjson calibration_tokens "$calibration_tokens" \
  --argjson max_tokens "$MAX_TOKENS" \
  --argjson recovery_max_tokens "$RECOVERY_MAX_TOKENS" \
  --argjson long_prompt_tokens "$long_prompt_tokens" \
  --argjson long_cached_tokens "$long_cached_tokens" \
  --argjson continuation_prompt_tokens "$continuation_prompt_tokens" \
  --argjson continuation_cached_tokens "$continuation_cached_tokens" \
  --argjson recovery_prompt_tokens "$recovery_prompt_tokens" \
  --argjson recovery_cached_tokens "$recovery_cached_tokens" '{
    status: "pass",
    replay_kind: "sanitized-incident-class-not-historical-byte-replay",
    model: $model,
    fixture_sha256: $fixture_sha256,
    request_sha256: $request_sha256,
    normalized_recovery_response_sha256: $response_sha256,
    sampling_profile: {temperature: 0, top_p: 0.95, reasoning_effort: "max"},
    completion_budgets: {
      cold: $max_tokens,
      continuation: 128,
      recovery: $recovery_max_tokens
    },
    calibration_tokens: $calibration_tokens,
    cold: {prompt_tokens: $long_prompt_tokens, cached_tokens: $long_cached_tokens},
    tool_result_continuation: {
      prompt_tokens: $continuation_prompt_tokens,
      cached_tokens: $continuation_cached_tokens
    },
    recovery_turn: {
      prompt_tokens: $recovery_prompt_tokens,
      cached_tokens: $recovery_cached_tokens
    },
    reasoning_content_preserved: true,
    tool_semantics_valid: true
  }'
