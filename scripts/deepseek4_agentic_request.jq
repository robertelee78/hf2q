{
  model: $model,
  messages: [
    {
      role: "system",
      content: $system_prompt
    },
    {
      role: "user",
      content: (
        "Agentic acceptance run " + $run_id
        + ". Inspect this Rust repository context and read " + $expected_path
        + " before making any recommendation. The requested manifest is intentionally not embedded; use read_file with exactly that absolute path. After read_file succeeds, reply with exactly "
        + $sentinel
        + " and nothing else. Repository context follows:\n\n"
        + $repo
      )
    }
  ],
  tools: [{
    type: "function",
    function: {
      name: "read_file",
      description: "Read a UTF-8 text file from the local workspace",
      parameters: {
        type: "object",
        properties: {path: {type: "string", description: "Absolute file path"}},
        required: ["path"],
        additionalProperties: false
      }
    }
  }],
  tool_choice: "required",
  temperature: 0,
  max_tokens: $max_tokens,
  stream: false
}
