#!/usr/bin/env bash
set -euo pipefail

# Qwen3.6/3.8 native tool-call wire gate for an already-running hf2q server
# (#662). The Qwen chat template encodes a call as
#   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
# joins later calls with `\n<tool_call>\n<function=`, and renders a
# zero-argument call as `<function=NAME>\n</function>`. The tool grammar must
# let the model emit exactly that.
#
# 1. Runs the shared OpenCode acceptance contract (scripts/test_qwen36_agentic.sh):
#    tool semantics, cached repeat, Auto, SSE tool deltas + [DONE], tool-result
#    continuation and prefix reuse. It needs a cold server, so it runs first.
# 2. Three-turn coding session under tool_choice=auto with logprobs: turn 1
#    reads a file, turn 2 receives the result and must call the tool again,
#    turn 3 receives the second result and must answer with the sentinel. Each
#    tool turn's generated tokens must contain the template-native call. The
#    replayed assistant turns carry `reasoning_content`, so the history is
#    token-identical to what the model generated; each follow-up must reuse at
#    least the previous prompt (REQUIRE_GENERATED_REUSE=1: also the generated
#    tokens, which only line up when the generated wire is template-native).
#    Turn 2 is also streamed and checked as valid SSE.
# 3. Auto multi-tool parallel request (two tools offered,
#    parallel_tool_calls=true): two read_file calls joined by
#    `</tool_call>\n<tool_call>\n<function=`, unary and SSE (indexes 0 and 1).
# 4. Zero-argument tool under tool_choice=auto and tool_choice=required: wire
#    `<function=git_status>\n</function>`, parsed arguments `{}`.
# Every tool turn fails clearly on finish_reason=length (raise TOOL_MAX_TOKENS).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
export BASE_URL=${BASE_URL:-http://127.0.0.1:8081}
export MODEL=${MODEL:-qwen36-abliterix-t63-APEX}
SENTINEL=${SENTINEL:-HF2Q_QWEN_NATIVE_WIRE_OK}
FIRST_PATH=${FIRST_PATH:-$ROOT_DIR/Cargo.toml}
SECOND_PATH=${SECOND_PATH:-$ROOT_DIR/src/main.rs}
# Thinking-enabled Qwen turns reason for ~1k characters before calling.
TOOL_MAX_TOKENS=${TOOL_MAX_TOKENS:-4096}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-300}
OUT_DIR=${OUT_DIR:-$(mktemp -d -t hf2q-qwen-native-wire.XXXXXX)}
SKIP_SHARED_GATE=${SKIP_SHARED_GATE:-0}
REQUIRE_GENERATED_REUSE=${REQUIRE_GENERATED_REUSE:-0}
GENERATED_REUSE_SLACK=${GENERATED_REUSE_SLACK:-2}

for command in curl jq; do
  command -v "$command" >/dev/null || { echo "missing required command: $command" >&2; exit 2; }
done
[[ -r "$FIRST_PATH" && -r "$SECOND_PATH" ]] || { echo "tool fixtures unreadable" >&2; exit 2; }
[[ "$TOOL_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "TOOL_MAX_TOKENS must be a positive integer" >&2; exit 2; }
mkdir -p "$OUT_DIR"

if [[ "$SKIP_SHARED_GATE" != 1 ]]; then
  "$SCRIPT_DIR/test_qwen36_agentic.sh" | tee "$OUT_DIR/shared-gate.json"
fi

post() {
  curl --fail-with-body --silent --show-error --no-buffer \
    --max-time "$CURL_MAX_TIME_SECONDS" -H 'Content-Type: application/json' \
    --data-binary "@$1" -o "$2" "$BASE_URL/v1/chat/completions" || {
    echo "request failed; body:" >&2; sed -n '1,80p' "$2" >&2; exit 1; }
}

post_sse() {
  local request=$1 label=$2
  jq '.stream = true | .stream_options = {include_usage: true} | del(.logprobs, .top_logprobs)' \
    "$request" >"$OUT_DIR/$label.stream.request.json"
  curl --fail-with-body --silent --show-error --no-buffer --max-time "$CURL_MAX_TIME_SECONDS" \
    -H 'Content-Type: application/json' --data-binary "@$OUT_DIR/$label.stream.request.json" \
    "$BASE_URL/v1/chat/completions" >"$OUT_DIR/$label.sse.txt"
  [[ $(grep -c '^data: \[DONE\]$' "$OUT_DIR/$label.sse.txt") == 1 &&
     $(grep '^data: ' "$OUT_DIR/$label.sse.txt" | tail -1) == 'data: [DONE]' ]] || {
    echo "$label SSE: not exactly one trailing [DONE]" >&2; exit 1; }
  sed -n 's/^data: //p' "$OUT_DIR/$label.sse.txt" | grep -v '^\[DONE\]$' >"$OUT_DIR/$label.sse.jsonl"
  jq -e -s 'all(.[]; type == "object")' "$OUT_DIR/$label.sse.jsonl" >/dev/null || {
    echo "$label SSE: non-object chunk" >&2; exit 1; }
  if [[ $(jq -r -s '[.[] | .choices[]? | .finish_reason // empty] | last // empty' \
      "$OUT_DIR/$label.sse.jsonl") == length ]]; then
    echo "$label SSE: finish_reason=length; raise TOOL_MAX_TOKENS (now $TOOL_MAX_TOKENS)" >&2; exit 1
  fi
}

# Generated token text of one unary response. Logprobs carry every generated
# token, including the native tool-call bytes the parser turns into tool_calls.
generated_text() { jq -r '[.choices[0].logprobs.content[]?.token] | join("")' "$1"; }
cached() { jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$1"; }
prompt_tokens() { jq -r '.usage.prompt_tokens' "$1"; }

assert_not_truncated() {
  local response=$1 label=$2
  if [[ $(jq -r '.choices[0].finish_reason' "$response") == length ]]; then
    echo "$label: finish_reason=length after $(jq -r '.usage.completion_tokens' "$response") tokens; raise TOOL_MAX_TOKENS (now $TOOL_MAX_TOKENS)" >&2
    exit 1
  fi
}

# assert_wire RESPONSE LABEL NATIVE_SUBSTRING
assert_wire() {
  local response=$1 label=$2 native=$3 wire
  wire=$(generated_text "$response")
  printf '%s' "$wire" >"$OUT_DIR/$label.wire.txt"
  if [[ "$wire" != *"$native"* ]]; then
    echo "$label: generated wire is not the template-native encoding" >&2
    printf 'wanted substring: %q\ngenerated: %q\n' "$native" "$wire" >&2
    exit 1
  fi
}

read_file_wire() {
  printf '<tool_call>\n<function=read_file>\n<parameter=path>\n%s\n</parameter>\n</function>\n</tool_call>' "$1"
}

assert_read_file_call() {
  local response=$1 expected=$2 label=$3
  assert_not_truncated "$response" "$label"
  jq -e --arg expected "$expected" '
    .choices[0].finish_reason == "tool_calls"
    and ((.choices[0].message.tool_calls // []) | length) == 1
    and .choices[0].message.tool_calls[0].function.name == "read_file"
    and ((.choices[0].message.tool_calls[0].function.arguments | fromjson).path == $expected)
    and ((.choices[0].message.content // "") | contains("<tool_call>") | not)
  ' "$response" >/dev/null || {
    echo "$label: not one read_file($expected) tool call" >&2; jq '.choices[0]' "$response" >&2; exit 1; }
  assert_wire "$response" "$label" "$(read_file_wire "$expected")"
}

# Append the assistant turn exactly as generated (reasoning, content, calls)
# plus its tool result, so the re-rendered history is token-identical.
continue_after() {
  local request=$1 response=$2 result_path=$3 out=$4
  jq --slurpfile prior "$response" --rawfile result "$result_path" '
    $prior[0].choices[0].message as $m
    | .messages += [
        ({role: "assistant", content: $m.content, tool_calls: $m.tool_calls}
         + (if $m.reasoning_content then {reasoning_content: $m.reasoning_content} else {} end)),
        {role: "tool", tool_call_id: $m.tool_calls[0].id, content: $result}
      ]' "$request" >"$out"
}

assert_reuse() {
  local previous=$1 response=$2 label=$3 previous_prompt previous_total got
  previous_prompt=$(prompt_tokens "$previous")
  previous_total=$(( previous_prompt + $(jq -r '.usage.completion_tokens' "$previous") ))
  got=$(cached "$response")
  echo "$label: cached_tokens=$got prompt_tokens=$(prompt_tokens "$response") previous_prompt=$previous_prompt previous_prompt+completion=$previous_total" |
    tee -a "$OUT_DIR/reuse.txt"
  if (( got < previous_prompt )); then
    echo "$label: prefix reuse lost the previous prompt ($got < $previous_prompt)" >&2; exit 1
  fi
  if [[ "$REQUIRE_GENERATED_REUSE" == 1 ]] && (( got + GENERATED_REUSE_SLACK < previous_total )); then
    echo "$label: generated tokens were not reused ($got + $GENERATED_REUSE_SLACK < $previous_total)" >&2; exit 1
  fi
}

read_file_tool='{"type":"function","function":{"name":"read_file","description":"Read a UTF-8 text file from the local workspace","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Absolute file path"}},"required":["path"],"additionalProperties":false}}}'
git_status_tool='{"type":"function","function":{"name":"git_status","description":"Show the working-tree status of the repository. Takes no arguments.","parameters":{"type":"object","properties":{}}}}'
system='You are an agentic coding assistant. Use the provided tools to inspect the workspace before answering.'

# request LABEL USER TOOLS_JSON TOOL_CHOICE_JSON PARALLEL
request() {
  jq -n --arg model "$MODEL" --arg system "$system" --arg user "$2" \
    --argjson tools "$3" --argjson tool_choice "$4" --argjson parallel "$5" \
    --argjson max_tokens "$TOOL_MAX_TOKENS" '{
      model: $model,
      messages: [{role: "system", content: $system}, {role: "user", content: $user}],
      tools: $tools, tool_choice: $tool_choice, parallel_tool_calls: $parallel,
      temperature: 0, max_tokens: $max_tokens,
      logprobs: true, top_logprobs: 1, stream: false
    }' >"$OUT_DIR/$1.request.json"
}

run_id="$$-$(date +%s)"

# --- 2. three-turn session -------------------------------------------------
request turn1 "Native-wire run $run_id. Read $FIRST_PATH with read_file, then read $SECOND_PATH with read_file, one call per turn. After both reads succeed, reply with exactly $SENTINEL and nothing else." \
  "[$read_file_tool]" '"auto"' false
post "$OUT_DIR/turn1.request.json" "$OUT_DIR/turn1.json"
assert_read_file_call "$OUT_DIR/turn1.json" "$FIRST_PATH" turn1

continue_after "$OUT_DIR/turn1.request.json" "$OUT_DIR/turn1.json" "$FIRST_PATH" "$OUT_DIR/turn2.request.json"
post "$OUT_DIR/turn2.request.json" "$OUT_DIR/turn2.json"
assert_read_file_call "$OUT_DIR/turn2.json" "$SECOND_PATH" turn2
assert_reuse "$OUT_DIR/turn1.json" "$OUT_DIR/turn2.json" turn2

post_sse "$OUT_DIR/turn2.request.json" turn2
jq -e -s --arg expected "$SECOND_PATH" '
  ([.[] | .choices[]?.delta.tool_calls[]?.index] | unique == [0])
  and ([.[] | .choices[]?.delta.tool_calls[]? | .function.name // empty] | join("")) == "read_file"
  and (([.[] | .choices[]?.delta.tool_calls[]? | .function.arguments // empty] | join("") | fromjson).path == $expected)
  and ([.[] | .choices[]? | .finish_reason // empty] | last) == "tool_calls"
  and ([.[] | .choices[]?.delta.content? // empty] | join("") | contains("<tool_call>") | not)
' "$OUT_DIR/turn2.sse.jsonl" >/dev/null || { echo "turn2 SSE tool delta invalid" >&2; exit 1; }

continue_after "$OUT_DIR/turn2.request.json" "$OUT_DIR/turn2.json" "$SECOND_PATH" "$OUT_DIR/turn3.request.json"
post "$OUT_DIR/turn3.request.json" "$OUT_DIR/turn3.json"
assert_not_truncated "$OUT_DIR/turn3.json" turn3
jq -e --arg sentinel "$SENTINEL" '
  .choices[0].finish_reason == "stop"
  and ((.choices[0].message.tool_calls // []) | length) == 0
  and ((.choices[0].message.content // "") | contains($sentinel))
' "$OUT_DIR/turn3.json" >/dev/null || {
  echo "turn3: tool-result continuation did not finish with $SENTINEL" >&2
  jq '.choices[0]' "$OUT_DIR/turn3.json" >&2; exit 1; }
assert_reuse "$OUT_DIR/turn2.json" "$OUT_DIR/turn3.json" turn3

# --- 3. Auto multi-tool parallel -----------------------------------------
request parallel "Parallel run $run_id. In this single turn, issue two read_file calls in parallel: one for $FIRST_PATH and one for $SECOND_PATH. Do not call git_status." \
  "[$read_file_tool,$git_status_tool]" '"auto"' true
post "$OUT_DIR/parallel.request.json" "$OUT_DIR/parallel.json"
assert_not_truncated "$OUT_DIR/parallel.json" parallel
jq -e --arg a "$FIRST_PATH" --arg b "$SECOND_PATH" '
  .choices[0].finish_reason == "tool_calls"
  and ((.choices[0].message.tool_calls // []) | length) == 2
  and all(.choices[0].message.tool_calls[]; .function.name == "read_file")
  and ([.choices[0].message.tool_calls[].function.arguments | fromjson | .path] | sort) == ([$a, $b] | sort)
' "$OUT_DIR/parallel.json" >/dev/null || {
  echo "parallel: not two read_file calls for both paths" >&2; jq '.choices[0]' "$OUT_DIR/parallel.json" >&2; exit 1; }
assert_wire "$OUT_DIR/parallel.json" parallel $'</tool_call>\n<tool_call>\n<function=read_file>\n<parameter=path>\n'
post_sse "$OUT_DIR/parallel.request.json" parallel
jq -e -s --arg a "$FIRST_PATH" --arg b "$SECOND_PATH" '
  [.[] | .choices[]?.delta.tool_calls[]?] as $d
  | ([$d[].index] | unique == [0, 1])
  and all([0, 1][]; . as $i
      | ([$d[] | select(.index == $i) | .function.name // empty] | join("")) == "read_file")
  and ([[0, 1][] as $i
      | [$d[] | select(.index == $i) | .function.arguments // empty] | join("") | fromjson | .path]
      | sort) == ([$a, $b] | sort)
  and ([.[] | .choices[]? | .finish_reason // empty] | last) == "tool_calls"
' "$OUT_DIR/parallel.sse.jsonl" >/dev/null || { echo "parallel SSE: tool deltas are not indexes 0 and 1 with both paths" >&2; exit 1; }

# --- 4. zero-argument tool under auto and required -----------------------
zero_arg_wire=$'<tool_call>\n<function=git_status>\n</function>\n</tool_call>'
for choice in auto required; do
  if [[ $choice == auto ]]; then tools="[$read_file_tool,$git_status_tool]"; else tools="[$git_status_tool]"; fi
  request "zero-$choice" "Zero-argument run $run_id ($choice). Call git_status now; it takes no arguments." \
    "$tools" "\"$choice\"" false
  post "$OUT_DIR/zero-$choice.request.json" "$OUT_DIR/zero-$choice.json"
  assert_not_truncated "$OUT_DIR/zero-$choice.json" "zero-$choice"
  jq -e '
    .choices[0].finish_reason == "tool_calls"
    and ((.choices[0].message.tool_calls // []) | length) == 1
    and .choices[0].message.tool_calls[0].function.name == "git_status"
    and (.choices[0].message.tool_calls[0].function.arguments | fromjson) == {}
  ' "$OUT_DIR/zero-$choice.json" >/dev/null || {
    echo "zero-$choice: not one git_status call with {} arguments" >&2
    jq '.choices[0]' "$OUT_DIR/zero-$choice.json" >&2; exit 1; }
  assert_wire "$OUT_DIR/zero-$choice.json" "zero-$choice" "$zero_arg_wire"
done

jq -n --arg out "$OUT_DIR" --arg model "$MODEL" --argjson shared "$([[ $SKIP_SHARED_GATE == 1 ]] && echo false || echo true)" \
  --argjson t2 "$(cached "$OUT_DIR/turn2.json")" --argjson t3 "$(cached "$OUT_DIR/turn3.json")" \
  '{qwen_native_tool_wire_pass: true, model: $model, evidence: $out,
    shared_gate_run: $shared, turn2_cached_tokens: $t2, turn3_cached_tokens: $t3,
    checks: ((if $shared then ["shared-gate"] else [] end)
             + ["three-turn-native-wire", "sse", "prefix-reuse",
                "auto-multi-tool-parallel", "zero-argument-auto", "zero-argument-required"])}'
