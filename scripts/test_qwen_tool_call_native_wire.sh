#!/usr/bin/env bash
set -euo pipefail

# Qwen3.6/3.8 native tool-call wire gate for an already-running hf2q server
# (#662). The Qwen chat template encodes a call as
#   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
# and the Auto-lazy tool grammar must let the model emit exactly that.
#
# 1. Runs the shared OpenCode acceptance contract (scripts/test_qwen36_agentic.sh):
#    tool semantics, cached repeat, Auto, SSE tool deltas + [DONE], tool-result
#    continuation and prefix reuse. It needs a cold server, so it runs first.
# 2. Runs a three-turn coding session under tool_choice=auto with logprobs:
#    turn 1 reads a file, turn 2 receives the tool result and must call the
#    tool again, turn 3 receives the second result and must answer with the
#    sentinel. Each tool turn's generated tokens must spell the template-native
#    `<tool_call>\n<function=read_file>\n<parameter=path>\n...` call, and each
#    follow-up turn must reuse at least the previous prompt (opt-in: also the
#    generated tool-call tokens, which only line up with the re-rendered
#    assistant turn when the generated wire is template-native).
# 3. Streams turn 2 and checks valid SSE: JSON chunks, one trailing [DONE],
#    one read_file tool delta whose arguments carry the expected path.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
export BASE_URL=${BASE_URL:-http://127.0.0.1:8081}
export MODEL=${MODEL:-qwen36-abliterix-t63-APEX}
SENTINEL=${SENTINEL:-HF2Q_QWEN_NATIVE_WIRE_OK}
FIRST_PATH=${FIRST_PATH:-$ROOT_DIR/Cargo.toml}
SECOND_PATH=${SECOND_PATH:-$ROOT_DIR/src/main.rs}
MAX_TOKENS=${MAX_TOKENS:-256}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-120}
OUT_DIR=${OUT_DIR:-$(mktemp -d -t hf2q-qwen-native-wire.XXXXXX)}
SKIP_SHARED_GATE=${SKIP_SHARED_GATE:-0}
REQUIRE_GENERATED_REUSE=${REQUIRE_GENERATED_REUSE:-0}
GENERATED_REUSE_SLACK=${GENERATED_REUSE_SLACK:-2}

for command in curl jq; do
  command -v "$command" >/dev/null || { echo "missing required command: $command" >&2; exit 2; }
done
[[ -r "$FIRST_PATH" && -r "$SECOND_PATH" ]] || { echo "tool fixtures unreadable" >&2; exit 2; }
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

# Generated token text of one unary response. Logprobs carry every generated
# token, including the native tool-call bytes the parser turns into tool_calls.
generated_text() { jq -r '[.choices[0].logprobs.content[]?.token] | join("")' "$1"; }
cached() { jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$1"; }
prompt_tokens() { jq -r '.usage.prompt_tokens' "$1"; }

assert_native_call() {
  local response=$1 expected=$2 turn=$3 wire native
  jq -e --arg expected "$expected" '
    .choices[0].finish_reason == "tool_calls"
    and ((.choices[0].message.tool_calls // []) | length) == 1
    and .choices[0].message.tool_calls[0].function.name == "read_file"
    and ((.choices[0].message.tool_calls[0].function.arguments | fromjson).path == $expected)
    and ((.choices[0].message.content // "") | contains("<tool_call>") | not)
  ' "$response" >/dev/null || {
    echo "$turn: not one read_file($expected) tool call" >&2; jq '.choices[0]' "$response" >&2; exit 1; }
  wire=$(generated_text "$response")
  printf '%s' "$wire" >"$OUT_DIR/$turn.wire.txt"
  native=$'<tool_call>\n<function=read_file>\n<parameter=path>\n'"$expected"$'\n</parameter>\n</function>\n</tool_call>'
  if [[ "$wire" != *"$native"* ]]; then
    echo "$turn: generated wire is not the template-native encoding" >&2
    printf 'wanted substring: %q\ngenerated: %q\n' "$native" "$wire" >&2
    exit 1
  fi
}

# Append the assistant call (re-rendered by the template) plus its result.
continue_after() {
  local request=$1 response=$2 result_path=$3 out=$4
  jq --slurpfile prior "$response" --rawfile result "$result_path" '
    .messages += [
      {role: "assistant", content: $prior[0].choices[0].message.content,
       tool_calls: $prior[0].choices[0].message.tool_calls},
      {role: "tool", tool_call_id: $prior[0].choices[0].message.tool_calls[0].id,
       content: $result}
    ]' "$request" >"$out"
}

assert_reuse() {
  local previous=$1 response=$2 turn=$3 previous_prompt previous_total got
  previous_prompt=$(prompt_tokens "$previous")
  previous_total=$(( previous_prompt + $(jq -r '.usage.completion_tokens' "$previous") ))
  got=$(cached "$response")
  echo "$turn: cached_tokens=$got prompt_tokens=$(prompt_tokens "$response") previous_prompt=$previous_prompt previous_prompt+completion=$previous_total" |
    tee -a "$OUT_DIR/reuse.txt"
  if (( got < previous_prompt )); then
    echo "$turn: prefix reuse lost the previous prompt ($got < $previous_prompt)" >&2; exit 1
  fi
  if [[ "$REQUIRE_GENERATED_REUSE" == 1 ]] && (( got + GENERATED_REUSE_SLACK < previous_total )); then
    echo "$turn: generated tool-call tokens were not reused ($got + $GENERATED_REUSE_SLACK < $previous_total)" >&2; exit 1
  fi
}

tools='[{"type":"function","function":{"name":"read_file","description":"Read a UTF-8 text file from the local workspace","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Absolute file path"}},"required":["path"],"additionalProperties":false}}}]'
system='You are an agentic coding assistant. Use the provided tool to inspect files before answering.'
user="Native-wire run $$-$(date +%s). Read $FIRST_PATH with read_file, then read $SECOND_PATH with read_file, one call per turn. After both reads succeed, reply with exactly $SENTINEL and nothing else."

jq -n --arg model "$MODEL" --arg system "$system" --arg user "$user" \
  --argjson tools "$tools" --argjson max_tokens "$MAX_TOKENS" '{
    model: $model,
    messages: [{role: "system", content: $system}, {role: "user", content: $user}],
    tools: $tools, tool_choice: "auto", temperature: 0, max_tokens: $max_tokens,
    logprobs: true, top_logprobs: 1, stream: false
  }' >"$OUT_DIR/turn1.request.json"
post "$OUT_DIR/turn1.request.json" "$OUT_DIR/turn1.json"
assert_native_call "$OUT_DIR/turn1.json" "$FIRST_PATH" turn1

continue_after "$OUT_DIR/turn1.request.json" "$OUT_DIR/turn1.json" "$FIRST_PATH" "$OUT_DIR/turn2.request.json"
post "$OUT_DIR/turn2.request.json" "$OUT_DIR/turn2.json"
assert_native_call "$OUT_DIR/turn2.json" "$SECOND_PATH" turn2
assert_reuse "$OUT_DIR/turn1.json" "$OUT_DIR/turn2.json" turn2

# SSE twin of turn 2.
jq '.stream = true | .stream_options = {include_usage: true} | del(.logprobs, .top_logprobs)' \
  "$OUT_DIR/turn2.request.json" >"$OUT_DIR/turn2.stream.request.json"
curl --fail-with-body --silent --show-error --no-buffer --max-time "$CURL_MAX_TIME_SECONDS" \
  -H 'Content-Type: application/json' --data-binary "@$OUT_DIR/turn2.stream.request.json" \
  "$BASE_URL/v1/chat/completions" >"$OUT_DIR/turn2.sse.txt"
[[ $(grep -c '^data: \[DONE\]$' "$OUT_DIR/turn2.sse.txt") == 1 &&
   $(grep '^data: ' "$OUT_DIR/turn2.sse.txt" | tail -1) == 'data: [DONE]' ]] || {
  echo "turn2 SSE: not exactly one trailing [DONE]" >&2; exit 1; }
sed -n 's/^data: //p' "$OUT_DIR/turn2.sse.txt" | grep -v '^\[DONE\]$' >"$OUT_DIR/turn2.sse.jsonl"
jq -e -s --arg expected "$SECOND_PATH" '
  all(.[]; type == "object")
  and ([.[] | .choices[]?.delta.tool_calls[]?.index] | unique == [0])
  and ([.[] | .choices[]?.delta.tool_calls[]? | .function.name // empty] | join("")) == "read_file"
  and (([.[] | .choices[]?.delta.tool_calls[]? | .function.arguments // empty] | join("") | fromjson).path == $expected)
  and ([.[] | .choices[]? | .finish_reason // empty] | last) == "tool_calls"
  and ([.[] | .choices[]?.delta.content? // empty] | join("") | contains("<tool_call>") | not)
' "$OUT_DIR/turn2.sse.jsonl" >/dev/null || { echo "turn2 SSE tool delta invalid" >&2; exit 1; }

continue_after "$OUT_DIR/turn2.request.json" "$OUT_DIR/turn2.json" "$SECOND_PATH" "$OUT_DIR/turn3.request.json"
post "$OUT_DIR/turn3.request.json" "$OUT_DIR/turn3.json"
jq -e --arg sentinel "$SENTINEL" '
  .choices[0].finish_reason == "stop"
  and ((.choices[0].message.tool_calls // []) | length) == 0
  and ((.choices[0].message.content // "") | contains($sentinel))
' "$OUT_DIR/turn3.json" >/dev/null || {
  echo "turn3: tool-result continuation did not finish with $SENTINEL" >&2
  jq '.choices[0]' "$OUT_DIR/turn3.json" >&2; exit 1; }
assert_reuse "$OUT_DIR/turn2.json" "$OUT_DIR/turn3.json" turn3

jq -n --arg out "$OUT_DIR" --arg model "$MODEL" \
  --argjson t2 "$(cached "$OUT_DIR/turn2.json")" --argjson t3 "$(cached "$OUT_DIR/turn3.json")" \
  '{qwen_native_tool_wire_pass: true, model: $model, evidence: $out,
    turn2_cached_tokens: $t2, turn3_cached_tokens: $t3}'
