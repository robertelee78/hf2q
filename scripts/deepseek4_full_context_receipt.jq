def number_between($minimum; $maximum):
  type == "number" and . >= $minimum and . <= $maximum;

def valid_agent:
  .status == "pass"
  and .fixture_id == "full-context-agentic-v1"
  and .agentic_context_fixture_sha256 == "2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef"
  and .agentic_context_fixture_bytes == 21204
  and .repository_context_chars == 20584
  and .expected_path == "/opt/hf2q/Cargo.toml"
  and .expected_prompt_tokens == 6685
  and .prompt_tokens == 6685
  and .cold_cached_tokens == 0
  and (.cold_ttft_ms | number_between(0; 55000))
  and (.cold_semantic_response_ms | number_between(0; 55000))
  and (.cached_tokens | type == "number" and . >= 6653)
  and (.auto_cached_tokens | type == "number" and . >= 6653)
  and (.continuation_cached_tokens | type == "number" and . >= 6653)
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
and .fixture_id == "full-context-agentic-v1"
and .agentic_context_fixture_sha256 == "2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef"
and .agentic_context_fixture_bytes == 21204
and .repository_context_chars == 20584
and .expected_prompt_tokens == 6685
and .prompt_tokens == 6685
and (.maximum_cold_ttft_ms | number_between(0; 55000))
and (.maximum_cold_semantic_response_ms | number_between(0; 55000))
and (.cohort_cold_wall_ms | number_between(0; 55000))
and (.agents | type == "array" and length == 4)
and all(.agents[]; valid_agent)
