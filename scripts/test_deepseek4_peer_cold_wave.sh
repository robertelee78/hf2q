#!/usr/bin/env bash
set -euo pipefail

# Cold-only, same-input llama.cpp comparator for the DeepSeek four-agent gate.
# The peer intentionally has no hf2q cache assertions: it proves request shape,
# tool semantics, prompt-token identity, and precise end-to-end cold timing.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:18080}
MODEL=${MODEL:-Deepseek v4 Flash 0731 Source}
AGENTS=${AGENTS:-4}
WAVE_ID=${WAVE_ID:-peer-default}
OUT_DIR=${OUT_DIR:-$(mktemp -d -t hf2q-deepseek-peer-wave.XXXXXX)}
EXPECTED_PATH=${EXPECTED_PATH:-/opt/hf2q-worktrees/full-context-slots/Cargo.toml}
TOOL_RESULT_PATH=${TOOL_RESULT_PATH:-$ROOT_DIR/Cargo.toml}
AGENTIC_CONTEXT_FIXTURE=${AGENTIC_CONTEXT_FIXTURE:-$ROOT_DIR/scripts/fixtures/deepseek4-agentic-repo-context.txt}
AGENTIC_CONTEXT_FIXTURE_SHA256=${AGENTIC_CONTEXT_FIXTURE_SHA256:-2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef}
# The request JSON is byte-bound to hf2q's 6,685-token fixture. Pinned
# llama.cpp build 10326 renders those same bytes as 6,695 prompt tokens; bind
# that runtime-specific count so a template/tokenizer drift cannot hide.
EXPECTED_PROMPT_TOKENS=${EXPECTED_PROMPT_TOKENS:-6695}
CURL_CONNECT_TIMEOUT_SECONDS=${CURL_CONNECT_TIMEOUT_SECONDS:-5}
CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-120}
MAX_COLD_RESPONSE_MS=${MAX_COLD_RESPONSE_MS:-0}
CURL_BIN=${CURL_BIN:-$(command -v curl)}

for command in awk jq shasum; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ -x /usr/bin/perl ]] || {
  echo "required monotonic timer is unavailable: /usr/bin/perl" >&2
  exit 2
}
[[ -x "$CURL_BIN" ]] || {
  echo "curl executable is unavailable: $CURL_BIN" >&2
  exit 2
}
for setting in AGENTS EXPECTED_PROMPT_TOKENS CURL_CONNECT_TIMEOUT_SECONDS \
  CURL_MAX_TIME_SECONDS MAX_COLD_RESPONSE_MS; do
  value=${!setting}
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "$setting must be a non-negative integer (got: $value)" >&2
    exit 2
  }
done
((AGENTS > 0 && EXPECTED_PROMPT_TOKENS > 0 \
  && CURL_CONNECT_TIMEOUT_SECONDS > 0 && CURL_MAX_TIME_SECONDS > 0)) || {
  echo "agent count, prompt tokens, and curl timeouts must be positive" >&2
  exit 2
}
[[ "$WAVE_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "WAVE_ID contains unsupported characters: $WAVE_ID" >&2
  exit 2
}
[[ -r "$TOOL_RESULT_PATH" ]] || {
  echo "tool-result fixture is not readable: $TOOL_RESULT_PATH" >&2
  exit 2
}
[[ -r "$AGENTIC_CONTEXT_FIXTURE" ]] || {
  echo "agentic context fixture is not readable: $AGENTIC_CONTEXT_FIXTURE" >&2
  exit 2
}
actual_fixture_sha=$(shasum -a 256 "$AGENTIC_CONTEXT_FIXTURE" | awk '{print $1}')
[[ "$actual_fixture_sha" == "$AGENTIC_CONTEXT_FIXTURE_SHA256" ]] || {
  echo "agentic context fixture SHA-256 mismatch" >&2
  exit 2
}

mkdir -p "$OUT_DIR"
pids=()
monotonic_us() {
  /usr/bin/perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
    -e 'printf "%.0f\n", 1000000 * clock_gettime(CLOCK_MONOTONIC)'
}
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM EXIT

run_agent() {
  local agent=$1
  local request="$OUT_DIR/agent-${agent}.request.json"
  local response="$OUT_DIR/agent-${agent}.response.json"
  local timing="$OUT_DIR/agent-${agent}.time"
  local receipt="$OUT_DIR/agent-${agent}.json"
  local elapsed_ms
  local prompt_tokens
  local cached_tokens

  if ! "$CURL_BIN" --fail-with-body --silent --show-error \
    --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" \
    --max-time "$CURL_MAX_TIME_SECONDS" \
    -H 'Content-Type: application/json' \
    --data-binary "@$request" \
    --output "$response" --write-out '%{time_total}\n' \
    "$BASE_URL/v1/chat/completions" >"$timing"; then
    echo "peer agent $agent HTTP request failed" >&2
    sed -n '1,120p' "$response" >&2 2>/dev/null || true
    return 1
  fi
  elapsed_ms=$(
    HF2Q_AGENTIC_TIME_TOTAL_INPUT="$timing" \
      "$ROOT_DIR/scripts/test_deepseek4_agentic.sh"
  )
  [[ "$elapsed_ms" =~ ^[0-9]+$ ]]
  if ((MAX_COLD_RESPONSE_MS > 0 && elapsed_ms > MAX_COLD_RESPONSE_MS)); then
    echo "peer agent $agent took ${elapsed_ms}ms; limit is ${MAX_COLD_RESPONSE_MS}ms" >&2
    return 1
  fi
  if ! jq -e --arg expected "$EXPECTED_PATH" '
    (.choices | length) == 1
    and .choices[0].finish_reason == "tool_calls"
    and ((.choices[0].message.tool_calls // []) | length) == 1
    and .choices[0].message.tool_calls[0].type == "function"
    and .choices[0].message.tool_calls[0].function.name == "read_file"
    and ((.choices[0].message.tool_calls[0].function.arguments
      | if type == "string" then fromjson else . end).path == $expected)
  ' "$response" >/dev/null; then
    echo "peer agent $agent did not return the exact read_file tool call" >&2
    jq '{choice:.choices[0],usage,error}' "$response" >&2 || true
    return 1
  fi
  prompt_tokens=$(jq -er '.usage.prompt_tokens | numbers' "$response")
  cached_tokens=$(jq -er \
    '.usage.prompt_tokens_details.cached_tokens // 0 | numbers' "$response")
  if ((prompt_tokens != EXPECTED_PROMPT_TOKENS)); then
    echo "peer agent $agent rendered $prompt_tokens prompt tokens; expected $EXPECTED_PROMPT_TOKENS" >&2
    return 1
  fi
  if ((cached_tokens != 0)); then
    echo "peer agent $agent was not cold (cached_tokens=$cached_tokens)" >&2
    return 1
  fi
  jq -n --argjson agent "$agent" --arg status pass \
    --arg request_sha256 "$(sha256_file "$request")" \
    --arg response_sha256 "$(sha256_file "$response")" \
    --argjson prompt_tokens "$prompt_tokens" \
    --argjson cached_tokens "$cached_tokens" \
    --argjson cold_semantic_response_ms "$elapsed_ms" \
    '{status:$status,agent:$agent,prompt_tokens:$prompt_tokens,
      cached_tokens:$cached_tokens,
      cold_semantic_response_ms:$cold_semantic_response_ms,
      request_sha256:$request_sha256,response_sha256:$response_sha256,
      tool_call:{name:"read_file",arguments_match:true}}' >"$receipt.tmp"
  mv "$receipt.tmp" "$receipt"
}

for ((agent = 1; agent <= AGENTS; agent++)); do
  RUN_ID="matched-peer-${WAVE_ID}-agent-${agent}" \
  SENTINEL="HF2Q_DEEPSEEK4_PEER_${WAVE_ID}_${agent}_OK" \
  MODEL="$MODEL" EXPECTED_PATH="$EXPECTED_PATH" \
  TOOL_RESULT_PATH="$TOOL_RESULT_PATH" \
  AGENTIC_CONTEXT_FIXTURE="$AGENTIC_CONTEXT_FIXTURE" \
  AGENTIC_CONTEXT_FIXTURE_SHA256="$AGENTIC_CONTEXT_FIXTURE_SHA256" \
  EXPECTED_PROMPT_TOKENS="$EXPECTED_PROMPT_TOKENS" \
  HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT="$OUT_DIR/agent-${agent}.request.json" \
    "$ROOT_DIR/scripts/test_deepseek4_agentic.sh"
done

cohort_started_us=$(monotonic_us)
for ((agent = 1; agent <= AGENTS; agent++)); do
  run_agent "$agent" >"$OUT_DIR/agent-${agent}.stdout" \
    2>"$OUT_DIR/agent-${agent}.stderr" &
  pids+=("$!")
done
failed=0
for ((agent = 1; agent <= AGENTS; agent++)); do
  if ! wait "${pids[$((agent - 1))]}"; then
    failed=1
    echo "peer agent $agent failed; stderr follows:" >&2
    sed -n '1,160p' "$OUT_DIR/agent-${agent}.stderr" >&2
  fi
done
pids=()
cohort_finished_us=$(monotonic_us)
trap - INT TERM EXIT
((failed == 0)) || {
  echo "matched peer cold wave failed; receipts: $OUT_DIR" >&2
  exit 1
}
cohort_wall_ms=$(((cohort_finished_us - cohort_started_us + 999) / 1000))

jq -s --arg status pass --arg runtime llama.cpp --arg wave_id "$WAVE_ID" \
  --arg fixture_sha256 "$actual_fixture_sha" \
  --arg expected_path "$EXPECTED_PATH" \
  --argjson concurrent_agents "$AGENTS" \
  --argjson expected_prompt_tokens "$EXPECTED_PROMPT_TOKENS" \
  --argjson cohort_cold_wall_ms "$cohort_wall_ms" '
  if length != $concurrent_agents or any(.[]; .status != "pass") then
    error("one or more peer agents did not pass")
  else {
    status:$status,runtime:$runtime,wave_id:$wave_id,
    concurrent_agents:$concurrent_agents,
    fixture_id:"full-context-agentic-v1",
    agentic_context_fixture_sha256:$fixture_sha256,
    expected_path:$expected_path,
    expected_prompt_tokens:$expected_prompt_tokens,
    prompt_tokens:(map(.prompt_tokens)|unique
      | if length == 1 then .[0] else error("peer prompt counts differ") end),
    cohort_cold_wall_ms:$cohort_cold_wall_ms,
    maximum_cold_semantic_response_ms:(map(.cold_semantic_response_ms)|max),
    minimum_cold_semantic_response_ms:(map(.cold_semantic_response_ms)|min),
    agents:.
  } end
' "$OUT_DIR"/agent-?.json >"$OUT_DIR/summary.json.tmp"
mv "$OUT_DIR/summary.json.tmp" "$OUT_DIR/summary.json"
printf '%s  %s\n' "$(sha256_file "$OUT_DIR/summary.json")" \
  "$OUT_DIR/summary.json" >"$OUT_DIR/summary.json.sha256"
cat "$OUT_DIR/summary.json"
echo "matched peer cold-wave receipts: $OUT_DIR" >&2
