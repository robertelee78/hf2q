#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for command in jq mktemp; do
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
