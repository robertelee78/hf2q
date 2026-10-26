#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for command in jq mktemp awk cmp; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done

test_dir=$(mktemp -d -t hf2q-deepseek-cache-contract.XXXXXX)
trap 'rm -rf "$test_dir"' EXIT

jq -n '{
  model: "Deepseek v4 Flash 0731 Source",
  messages: [
    {role: "system", content: "Use tools when required."},
    {role: "user", content: "Read the requested path."}
  ],
  tools: [{type: "function", function: {name: "read", description: "Read a path", parameters: {type: "object"}}}],
  tool_choice: "auto",
  temperature: 0,
  max_tokens: 128,
  stream: false
}' >"$test_dir/seed.request.json"
jq -n '{
  choices: [{
    finish_reason: "stop",
    message: {role: "assistant", content: "DEEPSEEK_CACHED_SUFFIX_OK"}
  }]
}' >"$test_dir/seed.response.json"

jq -n \
  --slurpfile base "$test_dir/seed.request.json" \
  --slurpfile prior "$test_dir/seed.response.json" \
  -f "$script_dir/build_deepseek4_decode_cancel_request.jq" \
  >"$test_dir/decode-cancel.request.json"
jq -e \
  --slurpfile seed "$test_dir/seed.request.json" \
  -f "$script_dir/validate_deepseek4_decode_cancel_request.jq" \
  "$test_dir/decode-cancel.request.json" >/dev/null

jq 'del(.tools, .tool_choice)' "$test_dir/decode-cancel.request.json" \
  >"$test_dir/invalid.request.json"
if jq -e \
  --slurpfile seed "$test_dir/seed.request.json" \
  -f "$script_dir/validate_deepseek4_decode_cancel_request.jq" \
  "$test_dir/invalid.request.json" >/dev/null; then
  echo "decode-cancel validator accepted a template-rewriting request" >&2
  exit 1
fi


# Exercise the actual required-tool builder and its real positive-integer guard,
# without sourcing the live script's HTTP/process main.
render_required_base() (
  local budget=$1
  local output=$2
  ROOT_DIR=$(cd "$script_dir/.." && pwd)
  MODEL=fixture-model
  SERVER_PID=1
  CURL_CONNECT_TIMEOUT_SECONDS=1
  CURL_MAX_TIME_SECONDS=1
  PREFILL_CHUNKS_BEFORE_CANCEL=1
  CANCEL_SETTLE_SECONDS=1
  CANCEL_STABILITY_SECONDS=1
  PEER_MAX_TOKENS=512
  if [[ "$budget" == default ]]; then unset MAX_TOKENS; else MAX_TOKENS=$budget; fi
  eval "$(awk '/^MAX_TOKENS=/{print; exit}' "$script_dir/test_deepseek4_cached_suffix.sh")"
  eval "$(awk '/^for setting in SERVER_PID /{copy=1} copy{print} copy && /^done$/{exit}' \
    "$script_dir/test_deepseek4_cached_suffix.sh")"
  eval "$(awk '/^build_base_request\(\) \{/{copy=1} copy{print} copy && /^}/{exit}' \
    "$script_dir/test_deepseek4_cached_suffix.sh")"
  build_base_request "$output" fixture-run "$test_dir/fixture.rs"
)
render_required_base default "$test_dir/default-base.request.json"
render_required_base 1024 "$test_dir/large-base.request.json"
jq -e '.max_tokens == 128 and .tool_choice == "required" and .temperature == 0' \
  "$test_dir/default-base.request.json" >/dev/null
jq -e '.max_tokens == 1024 and .tool_choice == "required" and .temperature == 0' \
  "$test_dir/large-base.request.json" >/dev/null
jq -S 'del(.max_tokens)' "$test_dir/default-base.request.json" >"$test_dir/default-other.json"
jq -S 'del(.max_tokens)' "$test_dir/large-base.request.json" >"$test_dir/large-other.json"
cmp "$test_dir/default-other.json" "$test_dir/large-other.json"
for invalid in 0 -1 invalid 1.5 01; do
  if render_required_base "$invalid" "$test_dir/invalid-base.request.json"; then
    echo "required-tool budget accepted invalid value: $invalid" >&2
    exit 1
  fi
  test ! -e "$test_dir/invalid-base.request.json"
done

# Changing only the required base allowance must not inflate the existing
# cancellation/semantic continuation allowance inherited from that base.
(
  eval "$(awk '/^build_continuation\(\) \{/{copy=1} copy{print} copy && /^}/{exit}' \
    "$script_dir/test_deepseek4_cached_suffix.sh")"
  printf 'source fixture\n' >"$test_dir/tool-result.txt"
  jq -n '{choices:[{message:{content:null,tool_calls:[{id:"call-fixture"}]}}]}' \
    >"$test_dir/tool-prior.json"
  build_continuation "$test_dir/large-base.request.json" "$test_dir/tool-prior.json" \
    "$test_dir/tool-result.txt" "$test_dir/continuation.request.json" true
)
jq -e '.max_tokens == 128 and .tool_choice == "auto" and .stream == true' \
  "$test_dir/continuation.request.json" >/dev/null

# The actual separate peer builder changes only its token allowance. Both cold
# warmup and its SSE replay inherit that same request, without changing thinking.
render_peer() (
  local budget=$1
  local output=$2
  MODEL=fixture-model
  SERVER_PID=1
  CURL_CONNECT_TIMEOUT_SECONDS=1
  CURL_MAX_TIME_SECONDS=1
  PREFILL_CHUNKS_BEFORE_CANCEL=1
  CANCEL_SETTLE_SECONDS=1
  CANCEL_STABILITY_SECONDS=1
  MAX_TOKENS=1024
  if [[ "$budget" == default ]]; then unset PEER_MAX_TOKENS; else PEER_MAX_TOKENS=$budget; fi
  eval "$(awk '/^PEER_MAX_TOKENS=/{print; exit}' "$script_dir/test_deepseek4_cached_suffix.sh")"
  eval "$(awk '/^for setting in SERVER_PID /{copy=1} copy{print} copy && /^done$/{exit}' \
    "$script_dir/test_deepseek4_cached_suffix.sh")"
  eval "$(awk '/^build_peer_request\(\) \{/{copy=1} copy{print} copy && /^}/{exit}' \
    "$script_dir/test_deepseek4_cached_suffix.sh")"
  build_peer_request "$output" fixture-peer
)
render_peer default "$test_dir/default-peer.request.json"
render_peer 8192 "$test_dir/large-peer.request.json"
jq -e '.max_tokens == 512 and .temperature == 0 and .stream == false' \
  "$test_dir/default-peer.request.json" >/dev/null
jq -e '.max_tokens == 8192 and .temperature == 0 and .stream == false' \
  "$test_dir/large-peer.request.json" >/dev/null
jq -S 'del(.max_tokens)' "$test_dir/default-peer.request.json" >"$test_dir/default-peer-other.json"
jq -S 'del(.max_tokens)' "$test_dir/large-peer.request.json" >"$test_dir/large-peer-other.json"
cmp "$test_dir/default-peer-other.json" "$test_dir/large-peer-other.json"
for invalid in 0 -1 invalid 1.5 01; do
  if render_peer "$invalid" "$test_dir/invalid-peer.request.json"; then
    echo "peer budget accepted invalid value: $invalid" >&2
    exit 1
  fi
  test ! -e "$test_dir/invalid-peer.request.json"
done

# Exercise the long-cache gate's actual short-tail builder and validation.
render_short_tail() (
  local budget=$1
  local output=$2
  TARGET_PROMPT_TOKENS=127000
  MIN_LONG_PROMPT_TOKENS=124000
  MAX_LONG_PROMPT_TOKENS=130000
  CALIBRATION_CHARS=1
  MAX_GROWTH_TTFT_MS=1
  MAX_CACHED_TTFT_MS=1
  MAX_CACHED_RESPONSE_MS=1
  CURL_CONNECT_TIMEOUT_SECONDS=1
  CURL_MAX_TIME_SECONDS=1
  MAX_TOKENS=1024
  CONTINUATION_MAX_TOKENS=8192
  SHORT_TAIL_SENTINEL=HF2Q_DEEPSEEK_SHORT_TAIL_OK
  if [[ "$budget" == default ]]; then unset SHORT_TAIL_MAX_TOKENS; else SHORT_TAIL_MAX_TOKENS=$budget; fi
  eval "$(awk '/^SHORT_TAIL_MAX_TOKENS=/{print; exit}' "$script_dir/test_deepseek4_long_context_cache.sh")"
  eval "$(awk '/^for setting in TARGET_PROMPT_TOKENS /{copy=1} copy{print} copy && /^done$/{exit}' \
    "$script_dir/test_deepseek4_long_context_cache.sh")"
  eval "$(awk '/^make_short_tail_request\(\) \{/{copy=1} copy{print} copy && /^}/{exit}' \
    "$script_dir/test_deepseek4_long_context_cache.sh")"
  make_short_tail_request "$test_dir/seed.request.json" "$test_dir/seed.response.json" "$output"
)
render_short_tail default "$test_dir/default-short-tail.request.json"
render_short_tail 1024 "$test_dir/large-short-tail.request.json"
jq -e '.max_tokens == 32 and .tool_choice == "auto"' "$test_dir/default-short-tail.request.json" >/dev/null
jq -e '.max_tokens == 1024 and .tool_choice == "auto"' "$test_dir/large-short-tail.request.json" >/dev/null
jq -S 'del(.max_tokens)' "$test_dir/default-short-tail.request.json" >"$test_dir/default-short-other.json"
jq -S 'del(.max_tokens)' "$test_dir/large-short-tail.request.json" >"$test_dir/large-short-other.json"
cmp "$test_dir/default-short-other.json" "$test_dir/large-short-other.json"
for invalid in 0 -1 invalid 1.5 01; do
  if render_short_tail "$invalid" "$test_dir/invalid-short-tail.request.json"; then
    echo "short-tail budget accepted invalid value: $invalid" >&2
    exit 1
  fi
  test ! -e "$test_dir/invalid-short-tail.request.json"
done

# The actual allocation must honor the owned packet TMPDIR on macOS, where
# mktemp -t may choose a different directory despite TMPDIR.
(
  TMPDIR="$test_dir/owned-growth-tmp"
  mkdir "$TMPDIR"
  eval "$(awk '/^work_dir=\$\(mktemp /{print; exit}' "$script_dir/test_deepseek4_long_context_cache.sh")"
  [[ -d "$work_dir" && "$work_dir" == "$TMPDIR"/hf2q-deepseek-long-cache.* ]]
  [[ "$(cd "$(dirname "$work_dir")" && pwd -P)" == "$(cd "$TMPDIR" && pwd -P)" ]]
  rmdir "$work_dir"
)
