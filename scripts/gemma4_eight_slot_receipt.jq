def numeric_nonnegative:
  type == "number" and . >= 0;
def positive_number:
  type == "number" and . > 0;
def nonnegative_integer:
  type == "number" and floor == . and . >= 0;

.status == "pass"
and .family == "gemma4"
and .wave_id == "eight-slots"
and .concurrent_agents == 8
and .require_cold_first == 1
and (.agents | type) == "array"
and (.agents | length) == 8
and all(.agents[];
  .status == "pass"
  and (.prompt_tokens | positive_number)
  and (.cold_cached_tokens | nonnegative_integer)
  and .cold_cached_tokens == 0
  and (.cached_tokens | nonnegative_integer)
  and (.auto_cached_tokens | nonnegative_integer)
  and (.continuation_cached_tokens | nonnegative_integer)
  and .cached_tokens >= (.prompt_tokens - 32)
  and .auto_cached_tokens >= (.prompt_tokens - 32)
  and .continuation_cached_tokens >= (.prompt_tokens - 32)
  and (.cold_ttft_ms | numeric_nonnegative)
  and .cold_ttft_ms <= 40000
  and (.cold_semantic_response_ms | numeric_nonnegative)
  and .cold_semantic_response_ms <= 60000
  and (.tool_result_response_ms | numeric_nonnegative)
  and .tool_result_response_ms <= 30000)
and (.maximum_cold_ttft_ms | numeric_nonnegative)
and .maximum_cold_ttft_ms == ([.agents[].cold_ttft_ms] | max)
and .maximum_cold_ttft_ms <= 40000
and (.maximum_cold_semantic_response_ms | numeric_nonnegative)
and .maximum_cold_semantic_response_ms ==
  ([.agents[].cold_semantic_response_ms] | max)
and .maximum_cold_semantic_response_ms <= 60000
and (.maximum_tool_result_ms | numeric_nonnegative)
and .maximum_tool_result_ms == ([.agents[].tool_result_response_ms] | max)
and .maximum_tool_result_ms <= 30000
and .thermal.status == "pass"
and .thermal.phase == "gemma-eight-slots"
and .thermal.concurrent_agents == 8
and .thermal.required_state == "nominal"
and .thermal.measurement_scope == "full-agent-wave"
and (.thermal.settle_seconds | type) == "number"
and .thermal.settle_seconds == 60
and (.thermal.settle_duration_seconds | type) == "number"
and .thermal.settle_duration_seconds >= 60
and (.thermal.settle_samples | positive_number)
and (.thermal.measurement_samples | type) == "number"
and .thermal.measurement_samples >= 2
and (.thermal.measurement_duration_seconds | positive_number)
and (.thermal.sample_interval_seconds | type) == "number"
and .thermal.sample_interval_seconds == 2
and (.thermal.maximum_sample_gap_seconds | type) == "number"
and .thermal.maximum_sample_gap_seconds == 5
and (.thermal.settle_sample_interval_seconds | type) == "number"
and .thermal.settle_sample_interval_seconds == 5
and (.thermal.maximum_settle_sample_gap_seconds | type) == "number"
and .thermal.maximum_settle_sample_gap_seconds == 8
and (.thermal.non_nominal_measurement_samples | nonnegative_integer)
and .thermal.non_nominal_measurement_samples == 0
and (.thermal.settle_telemetry_gaps | nonnegative_integer)
and .thermal.settle_telemetry_gaps == 0
and (.thermal.telemetry_gaps | nonnegative_integer)
and .thermal.telemetry_gaps == 0
and (.thermal.cold_receipts | type) == "array"
and (.thermal.cold_receipts | length) == 8
