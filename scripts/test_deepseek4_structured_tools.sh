#!/usr/bin/env bash
# Real-model regression for the nested structured tools used by stock coding
# clients. The server must already be running; this script never loads or stops
# a model.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8081}"
MODEL="${MODEL:-}"
MAX_TOKENS="${MAX_TOKENS:-768}"
REPEATS="${REPEATS:-3}"
TEMPERATURE="${TEMPERATURE:-0.55}"
TOP_P="${TOP_P:-0.95}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
KEEP_WORK_DIR="${KEEP_WORK_DIR:-0}"

if ! [[ "$BASE_URL" =~ ^http://(127\.0\.0\.1|localhost):[0-9]+$ ]]; then
    echo "BASE_URL must be a loopback endpoint without /v1" >&2
    exit 2
fi
if ! [[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ && "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_TOKENS and REPEATS must be positive integers" >&2
    exit 2
fi
if ! [[ "$TEMPERATURE" =~ ^[0-9]+([.][0-9]+)?$ && "$TOP_P" =~ ^[0-9]+([.][0-9]+)?$ && "$KEEP_WORK_DIR" =~ ^[01]$ ]]; then
    echo "TEMPERATURE/TOP_P must be nonnegative numbers and KEEP_WORK_DIR must be 0 or 1" >&2
    exit 2
fi
if ! jq -en --argjson top_p "$TOP_P" '$top_p > 0 and $top_p <= 1' >/dev/null; then
    echo "TOP_P must be greater than 0 and at most 1" >&2
    exit 2
fi
case "$REASONING_EFFORT" in
    low|high|max) ;;
    *) echo "REASONING_EFFORT must be low, high, or max" >&2; exit 2 ;;
esac

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-deepseek-structured.XXXXXX")
cleanup() {
    local status=$?
    if (( KEEP_WORK_DIR == 1 || status != 0 )); then
        echo "structured-tool workspace retained at $work_dir" >&2
    else
        rm -rf "$work_dir"
    fi
}
trap cleanup EXIT

if [[ -z "$MODEL" ]]; then
    MODEL=$(curl --fail-with-body --silent --show-error "$BASE_URL/v1/models" |
        jq -er '.data[0].id')
fi

question_tool=$(jq -cn '{
  type: "function",
  function: {
    name: "question",
    description: "Ask the user one or more questions",
    parameters: {
      type: "object",
      properties: {
        questions: {
          type: "array",
          items: {
            type: "object",
            properties: {
              header: {type: "string"},
              question: {type: "string"},
              options: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    label: {type: "string"},
                    description: {type: "string"}
                  },
                  required: ["label", "description"],
                  additionalProperties: false
                }
              },
              multiple: {type: "boolean"}
            },
            required: ["header", "question", "options"],
            additionalProperties: false
          }
        }
      },
      required: ["questions"],
      additionalProperties: false
    }
  }
}')

todo_tool=$(jq -cn '{
  type: "function",
  function: {
    name: "todowrite",
    description: "Replace the current todo list",
    parameters: {
      type: "object",
      properties: {
        todos: {
          type: "array",
          items: {
            type: "object",
            properties: {
              content: {type: "string"},
              status: {type: "string"},
              priority: {type: "string"}
            },
            required: ["content", "status", "priority"],
            additionalProperties: false
          }
        }
      },
      required: ["todos"],
      additionalProperties: false
    }
  }
}')

post_json() {
    local request_file=$1
    local response_file=$2
    curl --fail-with-body --silent --show-error \
        -H 'content-type: application/json' \
        --data-binary "@$request_file" \
        "$BASE_URL/v1/chat/completions" >"$response_file"
}

validate_question_arguments() {
    jq -e '
      def meaningful: type == "string" and test("[[:alnum:]]");
      (.questions | type == "array" and length == 1)
      and (.questions[0].header | meaningful)
      and (.questions[0].question | meaningful)
      and (.questions[0].options | type == "array" and length >= 2)
      and all(.questions[0].options[];
        (.label | meaningful)
        and (.description | meaningful))
    ' >/dev/null
}

validate_todo_arguments() {
    jq -e '
      def meaningful: type == "string" and test("[[:alnum:]]");
      (.todos | type == "array" and length >= 2)
      and all(.todos[];
        (.content | meaningful)
        and (.status | meaningful)
        and (.priority | meaningful))
    ' >/dev/null
}

build_request() {
    local tool_name=$1
    local tool_json=$2
    local request_file=$3
    local recovery=${4:-0}
    local tool_choice=${5:-required}
    local prompt
    local bad_arguments
    local final_recovery_instruction

    if [[ "$tool_name" == "question" ]]; then
        prompt='Call question exactly once. Ask which output format to use. Supply one concise header, one complete question, and exactly two options with nonempty labels and descriptions.'
        bad_arguments='{"questions":[{"header":null,"question":"Which output format?","options":[{"label":"JSON","description":"Machine-readable"},{"label":"Text","description":"Human-readable"}]}]}'
        final_recovery_instruction='Tool validation failed again: required strings cannot be null. Make one corrected question call now, preserving the original requirement of exactly one question with exactly two options.'
    else
        prompt='Call todowrite exactly once. Create two todos with nonempty content strings. Use pending then in_progress status, and high then medium priority.'
        bad_arguments='{"todos":[{"content":null,"status":"pending","priority":"high"}]}'
        final_recovery_instruction='Tool validation failed again: required strings cannot be null. Make one corrected todowrite call now, preserving the original requirement of exactly two todos with pending/high then in_progress/medium.'
    fi

    jq -n \
      --arg model "$MODEL" \
      --arg tool_name "$tool_name" \
      --arg prompt "$prompt" \
      --arg bad_arguments "$bad_arguments" \
      --arg final_recovery_instruction "$final_recovery_instruction" \
      --arg tool_choice "$tool_choice" \
      --arg reasoning_effort "$REASONING_EFFORT" \
      --argjson tool "$tool_json" \
      --argjson recovery "$recovery" \
      --argjson temperature "$TEMPERATURE" \
      --argjson top_p "$TOP_P" \
      --argjson max_tokens "$MAX_TOKENS" '
      {
        model: $model,
        messages: [
          {
            role: "system",
            content: "You are a coding agent. Use the requested tool exactly once. Every required string must contain real text; never emit null. After a successful tool result, respond exactly ACK."
          },
          {role: "user", content: $prompt}
        ],
        tools: [$tool],
        tool_choice: $tool_choice,
        temperature: $temperature,
        top_p: $top_p,
        reasoning_effort: $reasoning_effort,
        max_tokens: $max_tokens,
        stream: false
      }
      | if $recovery == 1 then
          .messages += [
            {
              role: "assistant",
              content: "I used null for a required string. I will correct it.",
              tool_calls: [{
                id: "failed_call_1",
                type: "function",
                function: {name: $tool_name, arguments: $bad_arguments}
              }]
            },
            {
              role: "tool",
              tool_call_id: "failed_call_1",
              content: "Tool validation failed: a required string was null. Correct every required field and try once more."
            },
            {
              role: "assistant",
              content: "I repeated the invalid call. I will now supply actual strings.",
              tool_calls: [{
                id: "failed_call_2",
                type: "function",
                function: {name: $tool_name, arguments: $bad_arguments}
              }]
            },
            {
              role: "tool",
              tool_call_id: "failed_call_2",
              content: $final_recovery_instruction
            }
          ]
        else . end
    ' >"$request_file"
}

validate_response() {
    local tool_name=$1
    local response_file=$2
    local arguments

    if ! jq -e --arg name "$tool_name" '
      (.choices | length) == 1
      and .choices[0].finish_reason == "tool_calls"
      and ((.choices[0].message.tool_calls // []) | length) == 1
      and .choices[0].message.tool_calls[0].type == "function"
      and .choices[0].message.tool_calls[0].function.name == $name
      and (.choices[0].message.tool_calls[0].function.arguments | type == "string")
    ' "$response_file" >/dev/null; then
        echo "structured-tool gate failed: invalid $tool_name response envelope" >&2
        jq '.choices[0] // .' "$response_file" >&2
        exit 1
    fi

    arguments=$(jq -r '.choices[0].message.tool_calls[0].function.arguments' "$response_file")
    if [[ "$tool_name" == "question" ]]; then
        if ! printf '%s' "$arguments" | validate_question_arguments; then
            echo "structured-tool gate failed: question arguments were null, empty, or malformed" >&2
            printf '%s\n' "$arguments" >&2
            exit 1
        fi
    elif ! printf '%s' "$arguments" | validate_todo_arguments; then
        echo "structured-tool gate failed: todowrite arguments were null, empty, or malformed" >&2
        printf '%s\n' "$arguments" >&2
        exit 1
    fi
}

question_request="$work_dir/question-request.json"
question_response="$work_dir/question-response.json"
todo_request="$work_dir/todo-request.json"
todo_response="$work_dir/todo-response.json"
question_auto_request="$work_dir/question-auto-request.json"
question_auto_response="$work_dir/question-auto-response.json"
todo_auto_request="$work_dir/todo-auto-request.json"
todo_auto_response="$work_dir/todo-auto-response.json"
build_request question "$question_tool" "$question_request"
build_request todowrite "$todo_tool" "$todo_request"
build_request question "$question_tool" "$question_auto_request" 0 auto
build_request todowrite "$todo_tool" "$todo_auto_request" 0 auto

question_cached=0
todo_cached=0
question_auto_cached=0
todo_auto_cached=0
for ((iteration = 1; iteration <= REPEATS; iteration++)); do
    post_json "$question_request" "$question_response"
    validate_response question "$question_response"
    question_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$question_response")
done
for ((iteration = 1; iteration <= REPEATS; iteration++)); do
    post_json "$todo_request" "$todo_response"
    validate_response todowrite "$todo_response"
    todo_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$todo_response")
done
for ((iteration = 1; iteration <= REPEATS; iteration++)); do
    post_json "$question_auto_request" "$question_auto_response"
    validate_response question "$question_auto_response"
    question_auto_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$question_auto_response")
done
for ((iteration = 1; iteration <= REPEATS; iteration++)); do
    post_json "$todo_auto_request" "$todo_auto_response"
    validate_response todowrite "$todo_auto_response"
    todo_auto_cached=$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$todo_auto_response")
done
if (( REPEATS > 1 && (question_cached == 0 || todo_cached == 0 || question_auto_cached == 0 || todo_auto_cached == 0) )); then
    echo "structured-tool gate failed: repeated requests did not reuse prompt cache" >&2
    exit 1
fi

question_recovery_request="$work_dir/question-recovery-request.json"
question_recovery_response="$work_dir/question-recovery-response.json"
todo_recovery_request="$work_dir/todo-recovery-request.json"
todo_recovery_response="$work_dir/todo-recovery-response.json"
question_auto_recovery_request="$work_dir/question-auto-recovery-request.json"
question_auto_recovery_response="$work_dir/question-auto-recovery-response.json"
todo_auto_recovery_request="$work_dir/todo-auto-recovery-request.json"
todo_auto_recovery_response="$work_dir/todo-auto-recovery-response.json"
build_request question "$question_tool" "$question_recovery_request" 1
build_request todowrite "$todo_tool" "$todo_recovery_request" 1
build_request question "$question_tool" "$question_auto_recovery_request" 1 auto
build_request todowrite "$todo_tool" "$todo_auto_recovery_request" 1 auto
post_json "$question_recovery_request" "$question_recovery_response"
validate_response question "$question_recovery_response"
post_json "$todo_recovery_request" "$todo_recovery_response"
validate_response todowrite "$todo_recovery_response"
post_json "$question_auto_recovery_request" "$question_auto_recovery_response"
validate_response question "$question_auto_recovery_response"
post_json "$todo_auto_recovery_request" "$todo_auto_recovery_response"
validate_response todowrite "$todo_auto_recovery_response"

stream_file="$work_dir/question-stream.sse"
stream_json="$work_dir/question-stream.jsonl"
jq '.stream = true | .stream_options = {include_usage: true}' "$question_auto_request" |
    curl --fail-with-body --silent --show-error --no-buffer \
      -H 'content-type: application/json' \
      --data-binary @- "$BASE_URL/v1/chat/completions" >"$stream_file"
if [[ $(grep -c '^data: \[DONE\]$' "$stream_file" || true) != 1 ]]; then
    echo "structured-tool gate failed: SSE stream did not terminate exactly once" >&2
    exit 1
fi
sed -n 's/^data: //p' "$stream_file" | grep -v '^\[DONE\]$' >"$stream_json"
stream_name=$(jq -r -s '[.[] | .choices[]?.delta.tool_calls[]? | select(.index == 0) | .function.name // empty] | join("")' "$stream_json")
stream_arguments=$(jq -r -s '[.[] | .choices[]?.delta.tool_calls[]? | select(.index == 0) | .function.arguments // empty] | join("")' "$stream_json")
stream_finish=$(jq -r -s '[.[] | .choices[]?.finish_reason // empty] | last // empty' "$stream_json")
if [[ "$stream_name" != "question" || "$stream_finish" != "tool_calls" ]] ||
   ! printf '%s' "$stream_arguments" | validate_question_arguments; then
    echo "structured-tool gate failed: reconstructed SSE question call was invalid" >&2
    printf '%s\n' "$stream_arguments" >&2
    exit 1
fi

continuation_request="$work_dir/continuation-request.json"
continuation_response="$work_dir/continuation-response.json"
selected_label=$(jq -er '
  .choices[0].message.tool_calls[0].function.arguments
  | fromjson
  | .questions[0].options[0].label
  | select(type == "string" and length > 0)
' "$question_response")
jq -n \
  --arg selected_label "$selected_label" \
  --slurpfile base "$question_request" \
  --slurpfile prior "$question_response" '
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
        content: ("User selected " + $selected_label + ".")
      }
    ]
  | .tool_choice = "auto"
  | .stream = false
' >"$continuation_request"
post_json "$continuation_request" "$continuation_response"
if ! jq -e '
  .choices[0].finish_reason == "stop"
  and .choices[0].message.content == "ACK"
  and ((.choices[0].message.tool_calls // []) | length) == 0
' "$continuation_response" >/dev/null; then
    echo "structured-tool gate failed: tool-result continuation was not terminal ACK" >&2
    jq '.choices[0]' "$continuation_response" >&2
    exit 1
fi

jq -n \
  --arg model "$MODEL" \
  --argjson repeats "$REPEATS" \
  --argjson temperature "$TEMPERATURE" \
  --argjson top_p "$TOP_P" \
  --arg reasoning_effort "$REASONING_EFFORT" \
  --argjson question_cached_tokens "$question_cached" \
  --argjson todo_cached_tokens "$todo_cached" \
  --argjson question_auto_cached_tokens "$question_auto_cached" \
  --argjson todo_auto_cached_tokens "$todo_auto_cached" \
  --argjson stream_cached_tokens "$(jq -r -s '[.[] | .usage.prompt_tokens_details.cached_tokens? // empty] | last // 0' "$stream_json")" \
  --argjson continuation_cached_tokens "$(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$continuation_response")" '{
    status: "pass",
    model: $model,
    repeats: $repeats,
    sampling_profile: {
      temperature: $temperature,
      top_p: $top_p,
      reasoning_effort: $reasoning_effort
    },
    question_arguments_valid: true,
    todo_arguments_valid: true,
    auto_question_arguments_valid: true,
    auto_todo_arguments_valid: true,
    repeated_null_recovery_valid: true,
    auto_repeated_null_recovery_valid: true,
    sse_question_valid: true,
    tool_result_continuation_valid: true,
    question_cached_tokens: $question_cached_tokens,
    todo_cached_tokens: $todo_cached_tokens,
    question_auto_cached_tokens: $question_auto_cached_tokens,
    todo_auto_cached_tokens: $todo_auto_cached_tokens,
    stream_cached_tokens: $stream_cached_tokens,
    continuation_cached_tokens: $continuation_cached_tokens
  }'
