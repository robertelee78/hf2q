def number_between($minimum; $maximum):
  type == "number" and . >= $minimum and . <= $maximum;

($contract |
  if length == 1 then .[0]
  else error("exactly one DeepSeek prompt contract is required")
  end) as $c
|
def valid_agent:
  . as $actual
  | ([$c.agents[] | select(.agent == $actual.agent)]
      | if length == 1 then .[0]
        else error("agent receipt is absent from prompt contract")
        end) as $expected
  | .status == "pass"
  and .agent == $expected.agent
  and .run_id == $expected.run_id
  and .fixture_id == $c.fixture_id
  and .prompt_contract_sha256 == $prompt_contract_sha256
  and .prompt_provenance_sha256 == $prompt_provenance_sha256
  and .serialization_policy == $c.serialization.policy
  and .request_sha256 == $expected.request_sha256
  and .request_bytes == $expected.request_bytes
  and .rendered_prompt_sha256 == $expected.rendered_prompt_sha256
  and .prompt_token_ids_sha256 == $expected.prompt_token_ids_sha256
  and .agentic_context_fixture_sha256 == $c.repository_context.sha256
  and .agentic_context_fixture_bytes == $c.repository_context.bytes
  and .repository_context_chars == $c.repository_context.chars
  and .expected_path == $c.request.expected_path
  and .agentic_system_prompt_sha256 == $c.request.system_prompt_sha256
  and .tool_result_success_prefix_sha256 == $c.tool_result.success_prefix_sha256
  and .tool_result_fixture_sha256 == $c.tool_result.sha256
  and .tool_result_fixture_bytes == $c.tool_result.bytes
  and .tool_result_payload_sha256 == $c.tool_result.combined_payload_sha256
  and .expected_prompt_tokens == $c.serialization.expected_prompt_tokens
  and .prompt_tokens == $c.serialization.expected_prompt_tokens
  and .prompt_tokens != $c.serialization.legacy_rejected_prompt_tokens
  and .cold_cached_tokens == 0
  and .cached_tokens == $c.prompt.cached_anchor_tokens
  and .auto_cached_tokens == $c.prompt.cached_anchor_tokens
  and .stream_cached_tokens == $c.prompt.cached_anchor_tokens
  and .continuation_cached_tokens == $c.prompt.cached_anchor_tokens
  and .continuation_uncached_tokens == $c.prompt.tool_result_uncached_suffix_tokens
  and .continuation_prompt_tokens ==
    (.continuation_cached_tokens + .continuation_uncached_tokens)
  and (.cold_ttft_ms | number_between(0; 60000))
  and (.cold_semantic_response_ms | number_between(0; 60000))
  and (.cached_ttft_ms | number_between(0; 5000))
  and (.cached_semantic_response_ms | number_between(0; 15000))
  and (.auto_semantic_response_ms | number_between(0; 15000))
  and (.cached_sse_tool_call_ms | number_between(0; 15000))
  and (.tool_result_response_ms | number_between(0; 35000))
  and .tool_semantics_pass == true
  and .cached_replay_equal == true
  and .automatic_tool_call_pass == true
  and .sse_tool_call_pass == true
  and .tool_result_continuation_pass == true
  and .source_tool_syntax_pass == true;

.status == "pass"
and .family == "deepseek4"
and .require_cold_first == 1
and .concurrent_agents == 4
and .fixture_id == $c.fixture_id
and .prompt_contract_sha256 == $prompt_contract_sha256
and .prompt_provenance_sha256 == $prompt_provenance_sha256
and .serialization_policy == $c.serialization.policy
and .agentic_context_fixture_sha256 == $c.repository_context.sha256
and .agentic_context_fixture_bytes == $c.repository_context.bytes
and .repository_context_chars == $c.repository_context.chars
and .agentic_system_prompt_sha256 == $c.request.system_prompt_sha256
and .tool_result_success_prefix_sha256 == $c.tool_result.success_prefix_sha256
and .tool_result_fixture_sha256 == $c.tool_result.sha256
and .tool_result_fixture_bytes == $c.tool_result.bytes
and .tool_result_payload_sha256 == $c.tool_result.combined_payload_sha256
and .expected_prompt_tokens == $c.serialization.expected_prompt_tokens
and .prompt_tokens == $c.serialization.expected_prompt_tokens
and .prompt_tokens != $c.serialization.legacy_rejected_prompt_tokens
and (.maximum_cold_ttft_ms | number_between(0; 60000))
and (.maximum_cold_semantic_response_ms | number_between(0; 60000))
and (.cohort_cold_wall_ms | number_between(0; 60000))
and (.agents | type == "array" and length == 4)
and ([.agents[].agent] == [1, 2, 3, 4])
and all(.agents[]; valid_agent)
