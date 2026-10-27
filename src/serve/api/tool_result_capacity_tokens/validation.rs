use super::*;

pub(super) fn selected(
    f: &str,
) -> Option<(&'static str, &'static str, &'static str, &'static str)> {
    match f {
        "qwen38" => Some((
            "Qwen3.8 27B",
            "qwen35",
            "90ff2bcbb15ed3e13f779a72000af82fde72842c3a14f04ac0da415071e6fa4c",
            "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
        )),
        "qwen36" => Some((
            "Qwen3.6 35B A3B",
            "qwen35moe",
            "90ff2bcbb15ed3e13f779a72000af82fde72842c3a14f04ac0da415071e6fa4c",
            "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
        )),
        "deepseek4" => Some((
            "DeepSeek V4 Flash 0731",
            "deepseek4",
            "0b71612f16ee3125bc4e7d560da38f37f89b6ee65cd31e9da329a3b8ff6b2e7e",
            "aec3d906bce80d43a9e6e89922c04e8e752c05f71f10040ba3f61b746ebacd6d",
        )),
        "gemma4" => Some((
            "Gemma 4 26B A4B It",
            "gemma4",
            "30395845c857a582ba54a960819dd5492b22311d5b0fc5607852f09be080d955",
            "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
        )),
        _ => None,
    }
}
pub(super) fn root(path: &Path) -> Result<(), &'static str> {
    if !path.is_absolute() || path.canonicalize().ok().as_deref() != Some(path) {
        return Err("canonical absolute output");
    }
    let m = std::fs::symlink_metadata(path).map_err(|_| "output metadata")?;
    if !m.is_dir() || m.file_type().is_symlink() || m.permissions().mode() & 0o077 != 0 {
        return Err("private output directory");
    }
    if path
        .ancestors()
        .any(|p| std::fs::symlink_metadata(p.join(".git")).is_ok())
    {
        return Err("output Git ancestor");
    }
    if std::fs::read_dir(path)
        .map_err(|_| "output listing")?
        .next()
        .is_some()
    {
        return Err("empty exclusive output");
    }
    Ok(())
}
pub(super) fn shape(v: &Value, family: &str, model: &str, agent: usize, stage: usize) -> bool {
    let keys = [
        "model",
        "messages",
        "tools",
        "tool_choice",
        "max_completion_tokens",
        "logprobs",
        "hf2q_overflow_policy",
        "stream",
        "stream_options",
        "temperature",
        "reasoning_effort",
    ];
    let stream = if stage == 0 {
        agent % 2 == 1
    } else {
        agent % 2 == 0
    };
    v.as_object()
        .is_some_and(|m| m.keys().all(|k| keys.contains(&k.as_str())))
        && v["model"] == model
        && v["max_completion_tokens"] == 8192
        && v["tool_choice"] == "auto"
        && v["logprobs"] == false
        && v["hf2q_overflow_policy"] == "reject"
        && v["stream"] == stream
        && (if stream {
            v["stream_options"] == json!({"include_usage":true})
        } else {
            v.get("stream_options").is_none()
        })
        && (if family == "deepseek4" {
            v["temperature"] == 0 && v["reasoning_effort"] == "max"
        } else {
            v.get("temperature").is_none() && v.get("reasoning_effort").is_none()
        })
}
pub(super) fn flow(
    a: &Value,
    b: &Value,
    assistant: &Value,
    payload: &str,
) -> Result<(), &'static str> {
    let before = a["messages"].as_array().ok_or("initial messages")?;
    let after = b["messages"].as_array().ok_or("continued messages")?;
    if before.len() != 2
        || before[0]["role"] != "system"
        || before[1]["role"] != "user"
        || after.len() != 4
        || after[..2] != before[..]
    {
        return Err("history changed");
    }
    if &after[2] != assistant || assistant["role"] != "assistant" {
        return Err("actual assistant/reasoning changed");
    }
    let calls = assistant["tool_calls"].as_array().ok_or("actual call")?;
    if calls.len() != 1
        || calls[0]["type"] != "function"
        || calls[0]["function"]["name"] != "read_file"
    {
        return Err("one read required");
    }
    let args: Value = serde_json::from_str(
        calls[0]["function"]["arguments"]
            .as_str()
            .ok_or("arguments")?,
    )
    .map_err(|_| "arguments JSON")?;
    if args != json!({"path":"src/ports.rs"}) || calls[0]["id"].as_str().is_none_or(str::is_empty) {
        return Err("actual read identity");
    }
    if payload.len() != FILE
        || after[3]
            != json!({"role":"tool","tool_call_id":calls[0]["id"],"name":"read_file","content":payload})
    {
        return Err("whole tool result/call ID");
    }
    if a["tools"] != b["tools"] {
        return Err("tool schema changed");
    }
    Ok(())
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CacheScope {
    PromptCheckpointP0Lcp,
    DeepseekLiveOrRecovery,
}
pub(super) fn cache_scope(family: &str) -> Option<CacheScope> {
    match family {
        "qwen38" | "qwen36" | "gemma4" => Some(CacheScope::PromptCheckpointP0Lcp),
        "deepseek4" => Some(CacheScope::DeepseekLiveOrRecovery),
        _ => None,
    }
}
pub(super) fn accept_tokens(
    p: &[u32],
    prior: Option<&[u32]>,
    u: &Value,
    slack: usize,
    scope: CacheScope,
) -> Result<usize, &'static str> {
    if p.is_empty()
        || p.len()
            .checked_add(8192)
            .and_then(|n| n.checked_add(slack))
            .is_none_or(|n| n > 262144)
    {
        return Err("context/output");
    }
    let n = p.len() as u64;
    let c = u["completion_tokens"].as_u64().ok_or("completion count")?;
    if u["prompt_tokens"].as_u64() != Some(n)
        || c == 0
        || c > 8192
        || u["total_tokens"].as_u64() != Some(n + c)
    {
        return Err("actual usage");
    }
    let cached = u["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .ok_or("cached usage")?;
    if let Some(old) = prior {
        let lcp = old.iter().zip(p).take_while(|(a, b)| a == b).count();
        let bound = match scope {
            CacheScope::PromptCheckpointP0Lcp => lcp,
            CacheScope::DeepseekLiveOrRecovery => p.len(),
        };
        if cached == 0 || cached > bound as u64 {
            return Err("cached outside selected family scope");
        }
    } else if cached != 0 {
        return Err("initial must be cold");
    }
    Ok(cached as usize)
}
// Same normalized SSE assistant construction as the frozen canonical Python client.
pub(super) fn response(bytes: &[u8], stream: bool, model: &str, stage: usize) -> (Value, Value) {
    let expected = if stage == 0 { "tool_calls" } else { "stop" };
    if !stream {
        let x: Value = serde_json::from_slice(bytes).unwrap();
        assert!(x.get("error").is_none());
        assert_eq!(x["model"], model);
        let choices = x["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0]["index"], 0);
        assert_eq!(choices[0]["finish_reason"], expected);
        return (choices[0]["message"].clone(), x["usage"].clone());
    }
    let text = std::str::from_utf8(bytes).unwrap();
    assert!(text.ends_with("\n\n") || text.ends_with("\r\n\r\n"));
    let mut message = json!({"role":"assistant","content":"","reasoning_content":""});
    let (mut usage, mut finish, mut identity, mut done) = (None, None, None, false);
    let (mut id, mut name, mut args) = (None, String::new(), String::new());
    let mut frames = 0;
    for line in text
        .lines()
        .filter(|s| !s.is_empty() && !s.starts_with(':'))
    {
        assert!(!done, "data after DONE");
        frames += 1;
        assert!(frames <= 65536);
        let data = line.strip_prefix("data: ").expect("SSE data framing");
        if data == "[DONE]" {
            assert!(finish.is_some() && usage.is_some());
            done = true;
            continue;
        }
        let x: Value = serde_json::from_str(data).unwrap();
        assert!(x.get("error").is_none());
        assert_eq!(x["model"], model);
        let got = x["id"].as_str().unwrap();
        if let Some(old) = &identity {
            assert_eq!(old, got);
        } else {
            identity = Some(got.to_owned());
        }
        let choices = x["choices"].as_array().unwrap();
        assert!(choices.len() <= 1);
        if let Some(c) = choices.first() {
            assert!(finish.is_none());
            assert_eq!(c["index"], 0);
            let delta = c["delta"].as_object().unwrap();
            assert!(delta
                .keys()
                .all(|k| ["role", "content", "reasoning_content", "tool_calls"]
                    .contains(&k.as_str())));
            if let Some(role) = delta.get("role") {
                assert_eq!(role, "assistant");
            }
            for key in ["content", "reasoning_content"] {
                if let Some(s) = delta.get(key).filter(|v| !v.is_null()) {
                    let joined = message[key].as_str().unwrap().to_owned() + s.as_str().unwrap();
                    message[key] = json!(joined);
                }
            }
            if let Some(parts) = delta.get("tool_calls").filter(|v| !v.is_null()) {
                for part in parts.as_array().unwrap() {
                    assert_eq!(part["index"], 0);
                    if let Some(i) = part.get("id") {
                        assert!(id.is_none());
                        id = Some(i.as_str().unwrap().to_owned());
                    }
                    if let Some(t) = part.get("type") {
                        assert_eq!(t, "function");
                    }
                    if let Some(f) = part.get("function") {
                        assert!(f
                            .as_object()
                            .unwrap()
                            .keys()
                            .all(|k| ["name", "arguments"].contains(&k.as_str())));
                        if let Some(v) = f.get("name") {
                            name.push_str(v.as_str().unwrap());
                        }
                        if let Some(v) = f.get("arguments") {
                            args.push_str(v.as_str().unwrap());
                        }
                    }
                }
            }
            if !c["finish_reason"].is_null() {
                assert_eq!(c["finish_reason"], expected);
                finish = Some(expected);
            }
        } else {
            assert!(finish.is_some());
        }
        if !x["usage"].is_null() {
            assert!(finish.is_some() && usage.is_none());
            usage = Some(x["usage"].clone());
        }
    }
    assert!(done && finish.is_some());
    if let Some(id) = id {
        message["tool_calls"] =
            json!([{"id":id,"type":"function","function":{"name":name,"arguments":args}}]);
    } else {
        assert!(name.is_empty() && args.is_empty());
    }
    (message, usage.unwrap())
}
