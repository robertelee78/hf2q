($seed[0].messages | length) as $seed_messages
| (.tools | type) == "array"
  and (.tools | length) >= 1
  and .tools == $seed[0].tools
  and .tool_choice == "auto"
  and .tool_choice == $seed[0].tool_choice
  and .messages[0:$seed_messages] == $seed[0].messages
  and (.messages | length) == ($seed_messages + 2)
