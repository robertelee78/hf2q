def sha256: type == "string" and test("^[0-9a-f]{64}$");

.schema_version == 2
and .fixture_id == "full-context-agentic-v2"
and .serialization.policy == "client-json-insertion-order"
and .serialization.legacy_rejected_policy == "recursive-lexicographic-key-order"
and .serialization.tool_function_keys == ["name", "description", "parameters"]
and .serialization.parameters_keys == ["type", "properties", "required", "additionalProperties"]
and .serialization.path_schema_keys == ["type", "description"]
and .serialization.token_id_digest_encoding == "hf2q-u32le-v1"
and .serialization.expected_prompt_tokens == 6684
and .serialization.legacy_rejected_prompt_tokens == 6685
and .model.artifact_sha256 == "936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d"
and .model.bytes == 107431343168
and .model.source_revision == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
and .request.model == "Deepseek v4 Flash 0731 Source"
and .request.expected_path == "/opt/hf2q-worktrees/full-context-slots/Cargo.toml"
and .request.max_tokens == 128
and .request.temperature == 0
and .request.tool_choice == "required"
and .request.system_prompt_sha256 == "8c704cd51ae982749b3e6abca22c32ab73716f6d21689eb2d3d4c5031c9405ef"
and .request.system_prompt_bytes == 93
and .request_builder.path == "scripts/deepseek4_agentic_request.jq"
and .request_builder.bytes == 1019
and .request_builder.sha256 == "d1441ec3f957aa1865edd30ce1573a17daa1809ec466f58ee0f1ccee3216e551"
and .chat_template.path == "src/core/chat_templates/deepseek-v4-flash-0731.jinja"
and .chat_template.bytes == 7646
and .chat_template.sha256 == "aec3d906bce80d43a9e6e89922c04e8e752c05f71f10040ba3f61b746ebacd6d"
and .repository_context.path == "scripts/fixtures/deepseek4-agentic-repo-context.txt"
and .repository_context.bytes == 21204
and .repository_context.chars == 20584
and .repository_context.sha256 == "2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef"
and .tool_result.path == "scripts/fixtures/deepseek4-agentic-tool-result-863ea423.toml"
and .tool_result.origin_commit == "863ea423a4ec4a4e46fc4bcce41ef2f439214a83"
and .tool_result.origin_repository_path == "Cargo.toml"
and .tool_result.prompt_visible_path == .request.expected_path
and .tool_result.bytes == 8912
and .tool_result.chars == 8892
and .tool_result.sha256 == "10d0410c76313d1783e491e17760a0946704e35c1566a93363dcb009f396bbbd"
and .tool_result.success_prefix == "Successful read_file result. File follows:\n"
and .tool_result.success_prefix_bytes == 43
and .tool_result.success_prefix_sha256 == "36425f21a24161ef516f4a80a8913ebcd956132cadfa5e3c3a940bd8245d8464"
and .tool_result.combined_payload_bytes == 8955
and .tool_result.combined_payload_sha256 == "34826d9e2ced0f41f4d57bf873ac6c1d8c294955893ae5f4d0815457e28b3c3c"
and .prompt.tokens == .serialization.expected_prompt_tokens
and .prompt.cached_anchor_tokens == 6676
and .prompt.tool_result_uncached_suffix_tokens == 2798
and .prompt.rejected_legacy_key_sorted_tokens == .serialization.legacy_rejected_prompt_tokens
and (.agents | type == "array" and length == 4)
and ([.agents[].agent] == [1, 2, 3, 4])
and ([.agents[].run_id] == [
  "full-context-deepseek4-agent-1",
  "full-context-deepseek4-agent-2",
  "full-context-deepseek4-agent-3",
  "full-context-deepseek4-agent-4"
])
and ([.agents[].sentinel] == [
  "HF2Q_DEEPSEEK4_AGENT_1_OK",
  "HF2Q_DEEPSEEK4_AGENT_2_OK",
  "HF2Q_DEEPSEEK4_AGENT_3_OK",
  "HF2Q_DEEPSEEK4_AGENT_4_OK"
])
and all(.agents[];
  .request_bytes == 22955
  and (.request_sha256 | sha256)
  and (.rendered_prompt_sha256 | sha256)
  and (.prompt_token_ids_sha256 | sha256)
  and (.legacy_key_sorted_rendered_prompt_sha256 | sha256)
  and (.legacy_key_sorted_prompt_token_ids_sha256 | sha256))
and ([.agents[].request_sha256] | unique | length == 4)
and ([.agents[].rendered_prompt_sha256] | unique | length == 4)
and ([.agents[].prompt_token_ids_sha256] | unique | length == 4)
and ([.agents[].request_sha256] == [
  "f70f24bb875e0d99a8f1f6e3e15be3c8c69f55e09f9e0cc251a98dd24bf11f5e",
  "6b6892f56dc256ebc4388d39be905fa6dcf7c1533a3c7b3c0f7e5315d8301693",
  "0d7f2b983b24cdc1e7c634ba958d8ceba7dff8beb57f91e2b4b0bc1ccabafb0e",
  "bec7e1161537279a7b85c9b7e28fd5546bdc048df650c1f53e2ed21b96dfd9d4"
])
and ([.agents[].rendered_prompt_sha256] == [
  "ff031a247908832feefec530813161aee4debd2f22c20641f8afd5d2c6bdb9c2",
  "2ed766812723f87adaf8633e1c0c265e6c3a029c643d39e774a16dc485f6b851",
  "934db33f83e64e434ea7136dbb87429803f89e42ebbf84ed7842dda903187a71",
  "cb1a369a0dc99054b290841618071ad2e485e4a7adf8cd74b0b0dc38157cbd98"
])
and ([.agents[].prompt_token_ids_sha256] == [
  "daaada048f48d613f7e98181eae6c3849253147fed0181c3831a4cfba3de9a86",
  "566df8f92d7eb2076898a5163ace03224c2fe7c562dc962fedacb08755d5b0eb",
  "367da8c22c59e326fac37d1851dbfd728c8ba06daf8312559d42fe7579ea5411",
  "3356019bef7e6cd323db78ec64264feba1e7daa50fc9a5a1bf398172164d8f88"
])
and ([.agents[].legacy_key_sorted_rendered_prompt_sha256] == [
  "0efcce39e823feafb2ce5e75dbef2b81577ca3c163b5a8aab858cd5a98035a93",
  "10d1a42d9b12162d08646403f84e01172e972c119ba276df7fe3ac497d2508a5",
  "2ad6d19729e0411a02c6b938c80d9f66940fc9fb27109ce069a885525395071f",
  "6425cc442f9d6dc13d5e77d8bbc3ec233ad60d3ff9b1675bf4cc54543b6d6981"
])
and ([.agents[].legacy_key_sorted_prompt_token_ids_sha256] == [
  "5fea2d2d7172912a2a6c2e1ad3f58478be7896b7c9b12cc3402e6b3d5e1a706e",
  "7b845c25826502f3cecdd529e06a3f2dccb92608faabccb7b44f90a31b94b26b",
  "fa4b56a307d7bd9843d2e5d99894dcbea704e8e08db0d63134f3bf2470c00516",
  "3cc368e9d52d78bba8ba9f6c7f8ec2321eb940c97f701f7c089b870f08e611ed"
])
