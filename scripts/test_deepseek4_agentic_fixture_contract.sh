#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILDER="$ROOT_DIR/scripts/deepseek4_agentic_request.jq"
FIXTURE="$ROOT_DIR/scripts/fixtures/deepseek4-agentic-repo-context.txt"
FIXTURE_SHA256=2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef
FIXTURE_CHARS=20584

for command in cmp grep head jq ln shasum tail; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ -r "$BUILDER" ]] || {
  echo "DeepSeek agentic request builder is not readable: $BUILDER" >&2
  exit 2
}
[[ -r "$FIXTURE" ]] || {
  echo "DeepSeek agentic context fixture is not readable: $FIXTURE" >&2
  exit 2
}
test "$(shasum -a 256 "$FIXTURE" | awk '{print $1}')" = "$FIXTURE_SHA256"
test "$(jq -Rs 'length' "$FIXTURE")" = "$FIXTURE_CHARS"

tmp_dir=$(mktemp -d -t hf2q-deepseek-agentic-contract.XXXXXX)
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

no_rg_bin="$tmp_dir/no-rg-bin"
mkdir -p "$no_rg_bin"
ln -s "$(command -v jq)" "$no_rg_bin/jq"
no_rg_path="$no_rg_bin:/usr/bin:/bin"
if PATH="$no_rg_path" command -v rg >/dev/null; then
  echo "no-rg contract PATH unexpectedly resolves rg" >&2
  exit 1
fi

request_file="$tmp_dir/request.json"
rendered_context="$tmp_dir/rendered-context.txt"
invalid_stderr="$tmp_dir/invalid.stderr"
positive_receipt="$tmp_dir/positive-receipt.json"
mutated_receipt="$tmp_dir/mutated-receipt.json"
isolated_root="$tmp_dir/isolated"

PATH="$no_rg_path" HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT="$request_file" \
AGENTIC_CONTEXT_FIXTURE="$FIXTURE" \
AGENTIC_CONTEXT_FIXTURE_SHA256="$FIXTURE_SHA256" \
EXPECTED_PATH="$ROOT_DIR/Cargo.toml" \
TOOL_RESULT_PATH="$ROOT_DIR/Cargo.toml" \
RUN_ID=fixture-contract SENTINEL=FIXTURE_OK MAX_TOKENS=128 \
  bash "$ROOT_DIR/scripts/test_deepseek4_agentic.sh"

jq -e --argjson expected_chars "$FIXTURE_CHARS" \
  --arg expected_path "$ROOT_DIR/Cargo.toml" '
    .model == "Deepseek v4 Flash 0731 Source"
    and .max_tokens == 128
    and .stream == false
    and .tool_choice == "required"
    and (.tools | length) == 1
    and .tools[0].function.name == "read_file"
    and .tools[0].function.parameters.additionalProperties == false
    and .tools[0].function.parameters.properties.path.type == "string"
    and (.messages | length) == 2
    and (.messages[1].content | contains($expected_path))
    and (.messages[1].content
      | split("Repository context follows:\n\n")
      | length == 2)
    and ((.messages[1].content
      | split("Repository context follows:\n\n")[1]) as $context
      | ($context | length) == $expected_chars)
  ' "$request_file" >/dev/null
jq -j '.messages[1].content | split("Repository context follows:\n\n")[1]' \
  "$request_file" >"$rendered_context"
cmp -s "$FIXTURE" "$rendered_context"

mkdir -p "$isolated_root/scripts/fixtures"
cp "$ROOT_DIR/scripts/test_deepseek4_agentic.sh" "$isolated_root/scripts/"
cp "$BUILDER" "$isolated_root/scripts/"
cp "$FIXTURE" "$isolated_root/scripts/fixtures/"
cp "$ROOT_DIR/Cargo.toml" "$isolated_root/"
printf 'first mutable README\n' >"$isolated_root/README.md"
HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT="$tmp_dir/readme-before.json" \
EXPECTED_PATH=/opt/hf2q/Cargo.toml \
TOOL_RESULT_PATH="$isolated_root/Cargo.toml" \
RUN_ID=full-context-deepseek4-agent-1 \
SENTINEL=HF2Q_DEEPSEEK4_AGENT_1_OK \
  bash "$isolated_root/scripts/test_deepseek4_agentic.sh"
printf 'completely different mutable README content that must be ignored\n' \
  >"$isolated_root/README.md"
HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT="$tmp_dir/readme-after.json" \
EXPECTED_PATH=/opt/hf2q/Cargo.toml \
TOOL_RESULT_PATH="$isolated_root/Cargo.toml" \
RUN_ID=full-context-deepseek4-agent-1 \
SENTINEL=HF2Q_DEEPSEEK4_AGENT_1_OK \
  bash "$isolated_root/scripts/test_deepseek4_agentic.sh"
cmp -s "$tmp_dir/readme-before.json" "$tmp_dir/readme-after.json"

if AGENTIC_CONTEXT_FIXTURE_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  BASE_URL=http://127.0.0.1:1 \
  TOOL_RESULT_PATH="$ROOT_DIR/Cargo.toml" \
  bash "$ROOT_DIR/scripts/test_deepseek4_agentic.sh" \
    >"$tmp_dir/invalid.stdout" 2>"$invalid_stderr"; then
  echo "DeepSeek agentic gate accepted a mismatched fixture digest" >&2
  exit 1
fi
grep -F 'agentic context fixture SHA-256 mismatch' \
  "$invalid_stderr" >/dev/null

expect_fixture_rejected() {
  local label=$1
  local fixture=$2
  if HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT="$tmp_dir/$label-request.json" \
    AGENTIC_CONTEXT_FIXTURE="$fixture" \
    TOOL_RESULT_PATH="$ROOT_DIR/Cargo.toml" \
    bash "$ROOT_DIR/scripts/test_deepseek4_agentic.sh" \
      >"$tmp_dir/$label.stdout" 2>"$tmp_dir/$label.stderr"; then
    echo "DeepSeek agentic gate accepted mutated fixture: $label" >&2
    exit 1
  fi
  grep -F 'agentic context fixture SHA-256 mismatch' \
    "$tmp_dir/$label.stderr" >/dev/null
}

cp "$FIXTURE" "$tmp_dir/appended.txt"
printf 'x' >>"$tmp_dir/appended.txt"
head -c 21203 "$FIXTURE" >"$tmp_dir/truncated.txt"
{
  printf 'X'
  tail -c +2 "$FIXTURE"
} >"$tmp_dir/mutated.txt"
expect_fixture_rejected appended "$tmp_dir/appended.txt"
expect_fixture_rejected truncated "$tmp_dir/truncated.txt"
expect_fixture_rejected mutated "$tmp_dir/mutated.txt"

jq -n '
  def agent: {
    status: "pass",
    fixture_id: "full-context-agentic-v1",
    agentic_context_fixture_sha256: "2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef",
    agentic_context_fixture_bytes: 21204,
    repository_context_chars: 20584,
    expected_path: "/opt/hf2q/Cargo.toml",
    expected_prompt_tokens: 6685,
    prompt_tokens: 6685,
    cold_cached_tokens: 0,
    cold_ttft_ms: 55000,
    cold_semantic_response_ms: 55000,
    cached_tokens: 6677,
    auto_cached_tokens: 6677,
    continuation_cached_tokens: 6677,
    cached_ttft_ms: 5000,
    cached_semantic_response_ms: 15000,
    auto_semantic_response_ms: 15000,
    cached_sse_tool_call_ms: 15000,
    tool_result_response_ms: 35000,
    tool_semantics_pass: true,
    cached_replay_equal: true,
    automatic_tool_call_pass: true,
    sse_tool_call_pass: true,
    tool_result_continuation_pass: true,
    source_tool_syntax_pass: true
  };
  {
    status: "pass",
    family: "deepseek4",
    require_cold_first: 1,
    concurrent_agents: 4,
    fixture_id: "full-context-agentic-v1",
    agentic_context_fixture_sha256: "2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef",
    agentic_context_fixture_bytes: 21204,
    repository_context_chars: 20584,
    expected_prompt_tokens: 6685,
    prompt_tokens: 6685,
    maximum_cold_ttft_ms: 55000,
    maximum_cold_semantic_response_ms: 55000,
    cohort_cold_wall_ms: 55000,
    agents: [range(0; 4) | agent]
  }
' >"$positive_receipt"
jq -e -f "$ROOT_DIR/scripts/deepseek4_full_context_receipt.jq" \
  "$positive_receipt" >/dev/null

expect_receipt_failure() {
  local label=$1
  local mutation=$2
  jq "$mutation" "$positive_receipt" >"$mutated_receipt"
  if jq -e -f "$ROOT_DIR/scripts/deepseek4_full_context_receipt.jq" \
    "$mutated_receipt" >/dev/null; then
    echo "DeepSeek receipt validator accepted invalid case: $label" >&2
    exit 1
  fi
}

expect_receipt_failure missing_prompt 'del(.agents[0].prompt_tokens)'
expect_receipt_failure null_prompt '.agents[0].prompt_tokens = null'
expect_receipt_failure string_prompt '.agents[0].prompt_tokens = "6685"'
expect_receipt_failure zero_prompt '.agents[0].prompt_tokens = 0'
expect_receipt_failure short_prompt '.agents[0].prompt_tokens = 6684'
expect_receipt_failure long_prompt '.agents[0].prompt_tokens = 6686'
expect_receipt_failure fixture_id '.agents[0].fixture_id = "mutable"'
expect_receipt_failure fixture_digest '.agents[0].agentic_context_fixture_sha256 = ("a" * 64)'
expect_receipt_failure fixture_bytes '.agents[0].agentic_context_fixture_bytes = 21205'
expect_receipt_failure mixed_agent_fixture '.agents[3].repository_context_chars = 20583'
expect_receipt_failure cold_cache '.agents[2].cold_cached_tokens = 1'
expect_receipt_failure negative_timing '.agents[1].cold_ttft_ms = -1'
expect_receipt_failure over_boundary '.agents[1].cold_semantic_response_ms = 55001'
expect_receipt_failure cohort_over_boundary '.cohort_cold_wall_ms = 55001'
expect_receipt_failure wrong_agent_count '.agents = .agents[0:3]'
expect_receipt_failure missing_semantics 'del(.agents[0].sse_tool_call_pass)'

echo "DeepSeek agentic fixture contract: pass"
