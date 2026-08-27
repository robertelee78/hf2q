#!/usr/bin/env bash

# Model-free validators shared by the live agentic cache-lifecycle fixture and
# its release receipt.  Execution identity comes from the LoadedEngine lease
# that actually handled the request; an HTTP success or a separately sampled
# model row is not sufficient proof.

agentic_lifecycle_header_value() {
  local headers=$1
  local wanted=$2
  awk -v wanted="$wanted" '
    {
      sub(/\r$/, "")
      separator = index($0, ":")
      if (separator == 0) next
      name = substr($0, 1, separator - 1)
      if (tolower(name) != tolower(wanted)) next
      value = substr($0, separator + 1)
      sub(/^[[:space:]]+/, "", value)
      if (++matches == 1) result = value
    }
    END {
      if (matches != 1 || result == "") exit 1
      print result
    }
  ' "$headers"
}

agentic_lifecycle_execution_receipt_json() {
  local headers=$1
  local expected_artifact_sha256=$2
  local expected_arch_family=$3
  local expected_architecture=$4
  local pool_key_b64 generation artifact_sha256 arch_family architecture

  [[ -f "$headers" && -r "$headers" && ! -L "$headers" ]] || return 1
  [[ "$expected_artifact_sha256" =~ ^[0-9a-f]{64}$ \
    && -n "$expected_arch_family" && -n "$expected_architecture" ]] || return 1

  pool_key_b64=$(agentic_lifecycle_header_value \
    "$headers" x-hf2q-execution-pool-key-b64) || return 1
  generation=$(agentic_lifecycle_header_value \
    "$headers" x-hf2q-execution-generation) || return 1
  artifact_sha256=$(agentic_lifecycle_header_value \
    "$headers" x-hf2q-execution-artifact-sha256) || return 1
  arch_family=$(agentic_lifecycle_header_value \
    "$headers" x-hf2q-execution-arch-family) || return 1
  architecture=$(agentic_lifecycle_header_value \
    "$headers" x-hf2q-execution-architecture) || return 1

  [[ "$pool_key_b64" =~ ^[A-Za-z0-9+/]+={0,2}$ \
    && "$generation" =~ ^[1-9][0-9]*$ \
    && "$artifact_sha256" == "$expected_artifact_sha256" \
    && "$arch_family" == "$expected_arch_family" \
    && "$architecture" == "$expected_architecture" ]] || return 1

  jq -n \
    --arg pool_key_b64 "$pool_key_b64" \
    --argjson generation "$generation" \
    --arg artifact_sha256 "$artifact_sha256" \
    --arg arch_family "$arch_family" \
    --arg architecture "$architecture" \
    '{pool_key_b64:$pool_key_b64,generation:$generation,
      artifact_sha256:$artifact_sha256,arch_family:$arch_family,
      architecture:$architecture}'
}

agentic_lifecycle_validate_summary() {
  local summary=$1
  local expected_run_id=$2
  local expected_context_lines=$3
  local expected_artifact_sha256=$4
  local expected_arch_family=$5
  local expected_architecture=$6
  local expected_continuation_thinking_token_budget=$7
  local expected_unrelated_conversation_thinking_enabled=${8:-true}

  [[ -f "$summary" && -r "$summary" && ! -L "$summary" \
    && "$expected_context_lines" =~ ^[1-9][0-9]*$ \
    && ("$expected_continuation_thinking_token_budget" == null \
      || "$expected_continuation_thinking_token_budget" =~ ^[1-9][0-9]*$) \
    && ("$expected_unrelated_conversation_thinking_enabled" == true \
      || "$expected_unrelated_conversation_thinking_enabled" == false) \
    && "$expected_artifact_sha256" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ "$(jq -s 'length' "$summary")" == 1 ]] || return 1
  jq -e \
    --arg run_id "$expected_run_id" \
    --argjson context_lines "$expected_context_lines" \
    --argjson continuation_thinking_token_budget "$expected_continuation_thinking_token_budget" \
    --argjson unrelated_conversation_thinking_enabled "$expected_unrelated_conversation_thinking_enabled" \
    --arg artifact_sha256 "$expected_artifact_sha256" \
    --arg arch_family "$expected_arch_family" \
    --arg architecture "$expected_architecture" '
      .schema_version == 3 and .status == "pass"
      and (.model | type == "string" and length > 0)
      and (.base_url | type == "string" and startswith("http://127.0.0.1:"))
      and .run_id == $run_id and .context_lines == $context_lines
      and .continuation_thinking_token_budget
        == $continuation_thinking_token_budget
      and .unrelated_conversation_thinking_enabled
        == $unrelated_conversation_thinking_enabled
      and .base_prompt_tokens > 0
      and .seed_cached_tokens
        >= (if .base_prompt_tokens > 64 then .base_prompt_tokens - 64 else 1 end)
      and .queued_exact_retry_cached_tokens
        >= (if .base_prompt_tokens > 64 then .base_prompt_tokens - 64 else 1 end)
      and .active_stream_cancelled_without_done == true
      and .unrelated_conversation_cached_tokens >= 0
      and .unrelated_conversation_cached_tokens <= 64
      and .unrelated_conversation_content == "ISOLATION_OK"
      and (.execution_receipts | length) == 5
      and (.execution_receipts | map(.phase))
        == ["base","seed","active_sse","sibling","isolation"]
      and (.execution_receipts | map(.stream))
        == [false,false,true,false,false]
      and all(.execution_receipts[];
        (.pool_key_b64 | test("^[A-Za-z0-9+/]+={0,2}$"))
        and (.generation | type == "number" and . > 0 and floor == .)
        and .artifact_sha256 == $artifact_sha256
        and .arch_family == $arch_family
        and .architecture == $architecture)
      and ([.execution_receipts[].pool_key_b64] | unique | length) == 1
      and ([.execution_receipts[].generation] | unique | length) == 1
    ' "$summary" >/dev/null
}
