#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILDER="$ROOT_DIR/scripts/deepseek4_agentic_request.jq"
FIXTURE="$ROOT_DIR/scripts/fixtures/deepseek4-agentic-repo-context.txt"
FIXTURE_SHA256=2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef
FIXTURE_CHARS=20584
# shellcheck source=scripts/macos_thermal_guard.sh
source "$ROOT_DIR/scripts/macos_thermal_guard.sh"

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
aggregate_root="$tmp_dir/aggregate"
early_exit_root="$tmp_dir/early-exit"

grep -F -- "--write-out '%{time_total}\\n'" \
  "$ROOT_DIR/scripts/test_deepseek4_agentic.sh" >/dev/null

printf '54.999999\n' >"$tmp_dir/time-below.txt"
test "$(HF2Q_AGENTIC_TIME_TOTAL_INPUT="$tmp_dir/time-below.txt" \
  bash "$ROOT_DIR/scripts/test_deepseek4_agentic.sh")" = 55000
printf '55.000001\n' >"$tmp_dir/time-above.txt"
test "$(HF2Q_AGENTIC_TIME_TOTAL_INPUT="$tmp_dir/time-above.txt" \
  bash "$ROOT_DIR/scripts/test_deepseek4_agentic.sh")" = 55001
printf 'not-a-time\n' >"$tmp_dir/time-invalid.txt"
if HF2Q_AGENTIC_TIME_TOTAL_INPUT="$tmp_dir/time-invalid.txt" \
  bash "$ROOT_DIR/scripts/test_deepseek4_agentic.sh" >/dev/null 2>&1; then
  echo "DeepSeek timing parser accepted malformed curl output" >&2
  exit 1
fi
printf '1.0\n2.0\n' >"$tmp_dir/time-multiple.txt"
if HF2Q_AGENTIC_TIME_TOTAL_INPUT="$tmp_dir/time-multiple.txt" \
  bash "$ROOT_DIR/scripts/test_deepseek4_agentic.sh" >/dev/null 2>&1; then
  echo "DeepSeek timing parser accepted multiple curl timing rows" >&2
  exit 1
fi

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
EXPECTED_PATH=/opt/hf2q-worktrees/full-context-slots/Cargo.toml \
TOOL_RESULT_PATH="$isolated_root/Cargo.toml" \
RUN_ID=full-context-deepseek4-agent-1 \
SENTINEL=HF2Q_DEEPSEEK4_AGENT_1_OK \
  bash "$isolated_root/scripts/test_deepseek4_agentic.sh"
printf 'completely different mutable README content that must be ignored\n' \
  >"$isolated_root/README.md"
HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT="$tmp_dir/readme-after.json" \
EXPECTED_PATH=/opt/hf2q-worktrees/full-context-slots/Cargo.toml \
TOOL_RESULT_PATH="$isolated_root/Cargo.toml" \
RUN_ID=full-context-deepseek4-agent-1 \
SENTINEL=HF2Q_DEEPSEEK4_AGENT_1_OK \
  bash "$isolated_root/scripts/test_deepseek4_agentic.sh"
cmp -s "$tmp_dir/readme-before.json" "$tmp_dir/readme-after.json"

# Qwen's automatic-tool gate uses the canonical operator path and explicitly
# tells the coding agent to invoke tools rather than imitate them in Markdown.
# The release gate runs from an ephemeral extracted Cargo package, so the
# simulated tool result still comes from the exact packed candidate.
HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT="$tmp_dir/qwen-stable-path.json" \
EXPECTED_PATH=/opt/hf2q/Cargo.toml \
TOOL_RESULT_PATH="$ROOT_DIR/Cargo.toml" \
TOOL_RESULT_SUCCESS_PREFIX=$'Result from the completed read_file call. The call succeeded; use this result to answer the user without calling read_file again. File follows:\n' \
AGENTIC_SYSTEM_PROMPT='You are an agentic coding assistant. Use the provided tools directly whenever they are needed. Never describe, imitate, or wrap a tool call in Markdown or a code fence.' \
RUN_ID=full-context-qwen36-np4-warmup-agent-1 \
SENTINEL=HF2Q_QWEN36_AGENT_1_OK \
  bash "$ROOT_DIR/scripts/test_qwen36_agentic.sh"
jq -e '
  .model == "qwen36-abliterix-t63-APEX"
  and .messages[0].content == "You are an agentic coding assistant. Use the provided tools directly whenever they are needed. Never describe, imitate, or wrap a tool call in Markdown or a code fence."
  and (.messages[1].content | contains("/opt/hf2q/Cargo.toml"))
  and (.messages[1].content | contains("/private/var/tmp/") | not)
' "$tmp_dir/qwen-stable-path.json" >/dev/null

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
    expected_path: "/opt/hf2q-worktrees/full-context-slots/Cargo.toml",
    expected_prompt_tokens: 6685,
    prompt_tokens: 6685,
    cold_cached_tokens: 0,
    cold_ttft_ms: 60000,
    cold_semantic_response_ms: 60000,
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
    maximum_cold_ttft_ms: 60000,
    maximum_cold_semantic_response_ms: 60000,
    cohort_cold_wall_ms: 60000,
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
expect_receipt_failure over_boundary '.agents[1].cold_semantic_response_ms = 60001'
expect_receipt_failure cohort_over_boundary '.cohort_cold_wall_ms = 60001'
expect_receipt_failure wrong_agent_count '.agents = .agents[0:3]'
expect_receipt_failure missing_semantics 'del(.agents[0].sse_tool_call_pass)'

# The live gate writes both `agent-N.json` and `agent-N.cold.json`. Prove the
# aggregate summary consumes only the four final receipts rather than matching
# both shapes through an over-broad glob.
mkdir -p "$aggregate_root/scripts"
cp "$ROOT_DIR/scripts/test_full_context_agent_slots.sh" "$aggregate_root/scripts/"
cat >"$aggregate_root/scripts/test_deepseek4_agentic.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
jq -n '{status:"pass", prompt_tokens:6685, cold_cached_tokens:0}' \
  >"$COLD_RESULT_PATH"
sleep 2
jq -n --arg run_id "$RUN_ID" '{
  status:"pass",
  fixture_id:"full-context-agentic-v1",
  agentic_context_fixture_sha256:"2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef",
  agentic_context_fixture_bytes:21204,
  repository_context_chars:20584,
  expected_prompt_tokens:6685,
  prompt_tokens:6685,
  cold_ttft_ms:1,
  cold_semantic_response_ms:1,
  cached_tokens:6677,
  tool_result_response_ms:1,
  run_id:$run_id
}'
EOF
chmod +x "$aggregate_root/scripts/test_deepseek4_agentic.sh"
thermal_read_state() { THERMAL_STATE=nominal; }
mkdir -p "$aggregate_root/receipts"
FAMILY=deepseek4 AGENTS=4 OUT_DIR="$aggregate_root/receipts" \
  bash "$aggregate_root/scripts/test_full_context_agent_slots.sh" \
  >"$aggregate_root/summary.json" &
aggregate_pid=$!
thermal_monitor_nominal_until_cold_receipts \
  "$aggregate_root/cold-measurement.log" fixture-cold \
  "$aggregate_root/receipts" 4 0 5 "$aggregate_pid"
if ! kill -0 "$aggregate_pid" 2>/dev/null; then
  echo "cold thermal scope did not end before functional receipts" >&2
  exit 1
fi
wait "$aggregate_pid"
test "$(tail -1 "$aggregate_root/cold-measurement.log" \
  | awk -F '\t' '{print $3}')" = fixture-cold-end
jq -e '
  .status == "pass"
  and .concurrent_agents == 4
  and (.agents | length) == 4
  and ([.agents[].run_id] | unique | length) == 4
' "$aggregate_root/summary.json" >/dev/null

# A child that fails the cold bound exits before writing agent-N.cold.json.
# The parent must report that exit immediately rather than waiting for the
# unrelated curl deadline plus its former 30-second receipt grace period.
mkdir -p "$early_exit_root/scripts"
cp "$ROOT_DIR/scripts/test_full_context_agent_slots.sh" "$early_exit_root/scripts/"
cat >"$early_exit_root/scripts/test_deepseek4_agentic.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${SENTINEL:-} == *_1_OK ]]; then
  exit 1
fi
exec sleep 30
EOF
chmod +x "$early_exit_root/scripts/test_deepseek4_agentic.sh"
early_exit_started_us=$(/usr/bin/perl \
  -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
  -e 'printf "%.0f\n", 1000000 * clock_gettime(CLOCK_MONOTONIC)')
if FAMILY=deepseek4 AGENTS=2 CURL_MAX_TIME_SECONDS=5 \
  OUT_DIR="$early_exit_root/receipts" \
  bash "$early_exit_root/scripts/test_full_context_agent_slots.sh" \
    >"$early_exit_root/stdout" 2>"$early_exit_root/stderr"; then
  echo "full-context parent accepted a child without a cold receipt" >&2
  exit 1
fi
early_exit_finished_us=$(/usr/bin/perl \
  -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
  -e 'printf "%.0f\n", 1000000 * clock_gettime(CLOCK_MONOTONIC)')
early_exit_elapsed_ms=$(( \
  (early_exit_finished_us - early_exit_started_us + 999) / 1000 \
))
((early_exit_elapsed_ms < 3000)) || {
  echo "full-context parent took ${early_exit_elapsed_ms}ms to notice child exit" >&2
  exit 1
}
grep -F 'exited before publishing its cold receipt' \
  "$early_exit_root/stderr" >/dev/null

echo "DeepSeek agentic fixture contract: pass"
