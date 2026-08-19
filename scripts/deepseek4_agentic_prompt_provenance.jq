if ($contract | length) != 1 then
  error("DeepSeek prompt provenance requires exactly one contract")
else
  $contract[0] as $c
  | def expected_agent($agent):
      first($c.agents[] | select(.agent == $agent));
    . as $receipt
  | (.schema_version == 2)
    and (.status == "pass")
    and (.prompt_contract_sha256 == $prompt_contract_sha256)
    and (.model_sha256 == $model_sha256)
    and ((.agents | type) == "array")
    and ((.agents | length) == 4)
    and ([.agents[].agent] == [1, 2, 3, 4])
    and all(.agents[];
      . as $agent
      | expected_agent($agent.agent) as $expected
      | $expected != null
        and .schema_version == 2
        and .status == "pass"
        and .prompt_contract_sha256 == $prompt_contract_sha256
        and .serialization_policy == $c.serialization.policy
        and .token_id_digest_encoding == $c.serialization.token_id_digest_encoding
        and .request_bytes == $expected.request_bytes
        and .request_sha256 == $expected.request_sha256
        and .rendered_prompt_sha256 == $expected.rendered_prompt_sha256
        and ((.rendered_prompt_bytes | type) == "number")
        and .rendered_prompt_bytes > 0
        and .prompt_token_ids_sha256 == $expected.prompt_token_ids_sha256
        and .prompt_tokens == $c.serialization.expected_prompt_tokens
        and .alternate_recursive_lexicographic_rendered_prompt_sha256 == $expected.alternate_recursive_lexicographic_rendered_prompt_sha256
        and ((.alternate_recursive_lexicographic_rendered_prompt_bytes | type) == "number")
        and .alternate_recursive_lexicographic_rendered_prompt_bytes > 0
        and .alternate_recursive_lexicographic_prompt_token_ids_sha256 == $expected.alternate_recursive_lexicographic_prompt_token_ids_sha256
        and .alternate_recursive_lexicographic_prompt_tokens == $c.serialization.alternate_recursive_lexicographic_prompt_tokens
        and .serialization_delta_proven == true)
end
