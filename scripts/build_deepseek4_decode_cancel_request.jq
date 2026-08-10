$base[0]
| .messages += [
    {role: "assistant", content: $prior[0].choices[0].message.content},
    {role: "user", content: "Write at least 600 tokens explaining cache-coherent cancellation recovery. Do not call tools. End with DECODE_CANCEL_DONE."}
  ]
| .max_tokens = 768
| .stream = true
| .stream_options = {include_usage: true}
