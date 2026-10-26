//! Partial failed-run analysis only; never an exact-prefix acceptance oracle.
use super::*;

fn observed_usage(bytes: &[u8]) -> (Value, &'static str, usize) {
    if let Ok(value) = serde_json::from_slice::<Value>(bytes) {
        assert!(value["usage"].is_object(), "bound JSON has no actual usage");
        return (value["usage"].clone(), "json", 0);
    }
    let text = std::str::from_utf8(bytes).unwrap();
    let mut usages = Vec::new();
    let mut done = 0;
    for line in text.lines().filter(|line| !line.is_empty()) {
        let data = line.strip_prefix("data: ").expect("non-data SSE record");
        if data == "[DONE]" {
            done += 1;
        } else {
            let value: Value = serde_json::from_str(data).unwrap();
            if value["usage"].is_object() {
                usages.push(value["usage"].clone());
            }
        }
    }
    assert_eq!(usages.len(), 1, "exactly one actual terminal usage required");
    (usages.remove(0), "sse", done)
}

#[test]
#[ignore = "post-unload two-wire FAILED Gemma metadata analysis; never prefix PASS"]
fn gemma4_two_wire_prefix_failure_analyzer() {
    let bytes = std::fs::read(
        std::env::var("HF2Q_GEMMA_PREFIX_FAILURE_ANALYZER_CONFIG").unwrap(),
    )
    .unwrap();
    // Same deny_unknown_fields config/bound inputs as the strict seven-wire
    // oracle. For this analyzer only, receipt may bind actual raw JSON/SSE
    // response bytes when the failing client never published a receipt.
    let c: Config = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(c.turns.len(), 2);
    assert!(matches!(c.server_slots, 1 | 4));
    let template_bytes = read(&c.template);
    let template = String::from_utf8(template_bytes.clone()).unwrap();
    gemma_template(&c.family, &c.template_source, &template).unwrap();
    let tokenizer_bytes = read(&c.tokenizer);
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
    assert!(tokenizer.get_truncation().is_none());
    let readiness_bytes = read(&c.readiness);
    let readiness: Value = serde_json::from_slice(&readiness_bytes).unwrap();
    let snap = &readiness["runtime"]["measurement_snapshot"];
    assert_eq!(snap["schema_version"], "hf2q.measurement-snapshot.v1");
    assert_eq!(snap["architecture"], "gemma4");
    assert_eq!(snap["tokenizer_sha256"], semantic_hash(
        serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap()
    ));
    assert_eq!(snap["template_sha256"], semantic_hash(
        json!({"source":c.template_source,"template":template})
    ));
    assert_eq!(readiness["health"]["context_length"], 262144);
    assert_eq!(readiness["health"]["model"], snap["model_id"]);
    assert_eq!(readiness["readyz"]["ready"], true);
    assert_eq!(readiness["runtime"]["serve"]["scheduler_mode"], "slot_aware");
    assert_eq!(readiness["runtime"]["serve"]["max_slots"], c.server_slots);
    let controller_bytes = read(&c.controller_receipt);
    let controller: Value = serde_json::from_slice(&controller_bytes).unwrap();
    assert_eq!(controller["server_pid"], snap["process_pid"]);
    assert_eq!(controller["cleanup_proved"], true);
    assert_eq!(controller["protocol_complete"], false);
    assert_eq!(controller["status"], "FailedOrHeld");
    assert!(controller["failure"].is_string(), "actual failed run required");
    let approval_bytes = read(&c.family_approval);
    let approval: Value = serde_json::from_slice(&approval_bytes).unwrap();
    assert_eq!(approval["family"], "gemma4");
    assert_eq!(approval["family_controls"]["cache_family"], "gemma4");
    assert_eq!(approval["family_controls"]["slots"], c.server_slots);
    assert_eq!(approval["family_controls"]["context_per_slot"], 262144);
    assert_eq!(approval["family_controls"]["hybrid_kv"], true);
    let build_bytes = read(&c.server_build_receipt);
    let build: Value = serde_json::from_slice(&build_bytes).unwrap();
    assert!(build_matches(&c.expected_server_source_commit,
        &c.expected_server_binary_sha256, &build));
    let reg = registry::find_for_architecture("gemma4").unwrap();
    let mut prior_stable: Option<Vec<u32>> = None;
    let mut controls_prior = None;
    let mut observations = Vec::new();
    for (i, turn) in c.turns.iter().enumerate() {
        let wire = read(&turn.request);
        let response = read(&turn.receipt);
        let (usage, framing, done) = observed_usage(&response);
        let v: Value = serde_json::from_slice(&wire).unwrap();
        let allowed = ["model", "messages", "tools", "tool_choice",
            "max_completion_tokens", "logprobs", "hf2q_overflow_policy",
            "stream", "stream_options"];
        assert!(v.as_object().unwrap().keys().all(|key| allowed.contains(&key.as_str())));
        assert_eq!(v["model"], snap["model_id"]);
        assert_eq!(v["max_completion_tokens"], 8192);
        assert_eq!(v["hf2q_overflow_policy"], "reject");
        assert_eq!(v["stream"], i == 1);
        let req: schema::ChatCompletionRequest = serde_json::from_value(v.clone()).unwrap();
        assert!(req.messages.iter().all(|m|
            m.content.as_ref().is_none_or(|content| content.image_urls().is_empty())));
        assert!(matches!(grammar::request::validate_tool_request(&req).unwrap(),
            schema::ToolChoiceValue::Auto));
        let tools = req.tools.as_deref().unwrap();
        assert_eq!(tools.len(), 3);
        let controls = reasoning_controls_from_request(&req).unwrap();
        let thinking = resolve_api_enable_thinking(controls.thinking_enabled_override(),
            crate::serve::template_supports_enable_thinking(&template));
        let kwargs = resolve_api_template_kwargs(Some(&reg), req.chat_template_kwargs.as_ref(),
            controls.deepseek_reasoning_effort_tier());
        assert!(kwargs.is_none(), "no alternate family/history kwargs");
        let current = json!({"tools":v["tools"],"thinking":thinking,"kwargs":kwargs});
        if let Some(previous) = &controls_prior { assert_eq!(previous, &current); }
        controls_prior = Some(current);
        let render = |cue| render_chat_prompt_or_400_with_generation_prompt(
            &template, &req.messages, Some(tools), thinking, kwargs.as_ref(), cue).unwrap();
        let full = render(true);
        let stable = render(false);
        let p = tokenize_http_prompt(&tokenizer, &full).unwrap();
        let s = tokenize_http_prompt(&tokenizer, &stable).unwrap();
        let observed_prompt = usage["prompt_tokens"].as_u64().unwrap();
        let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
        assert!(cached <= observed_prompt);
        let common = prior_stable.as_ref().map(|prior|
            prior.iter().zip(&p).take_while(|(a, b)| a == b).count());
        observations.push(json!({"turn":i,"prompt_tokens":p.len(),
            "stable_tokens":s.len(),"stable_equals_full":s==p,
            "stable_is_strict_prefix":!s.is_empty()&&s.len()<p.len()&&p.starts_with(&s),
            "prior_stable_tokens":prior_stable.as_ref().map(Vec::len),
            "lcp_prior_stable_current_prompt":common,
            "current_prompt_starts_prior_stable":prior_stable.as_ref().map(|prior|p.starts_with(prior)),
            "observed_usage_prompt_tokens":observed_prompt,
            "observed_usage_matches_rendered":observed_prompt==p.len() as u64,
            "observed_cached_tokens":cached,"raw_framing":framing,"raw_done_records":done,
            "request_sha256":hash(&wire),"raw_response_or_receipt_sha256":hash(&response),
            "rendered_sha256":hash(full.as_bytes()),
            "prompt_token_ids_sha256":hash(&serde_json::to_vec(&p).unwrap()),
            "stable_token_ids_sha256":hash(&serde_json::to_vec(&s).unwrap())}));
        prior_stable = Some(s);
    }
    let output = json!({"schema":"hf2q.gemma4-prefix-failure-analysis.v1",
        "status":"AnalyzedGemmaPrefixFailure","full_prefix_reuse":"NotQualified",
        "model_inference_performed":false,"model_qualified":false,"seven_wire_pass":false,
        "config_sha256":hash(&bytes),"readiness_sha256":hash(&readiness_bytes),
        "controller_sha256":hash(&controller_bytes),"family_approval_sha256":hash(&approval_bytes),
        "server_build_receipt_sha256":hash(&build_bytes),
        "expected_server_source_commit":c.expected_server_source_commit,
        "expected_server_binary_sha256":c.expected_server_binary_sha256,
        "analyzer_source_sha256":hash(include_bytes!("handlers_gemma_prefix_failure_tests.rs")),
        "renderer_source_sha256":hash(include_bytes!("engine.rs")),"observations":observations});
    assert!(c.output.is_absolute());
    let parent = c.output.parent().unwrap().canonicalize().unwrap();
    assert!(parent.ancestors().all(|path| !path.join(".git").exists()));
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    options.open(&c.output).unwrap()
        .write_all(serde_json::to_string_pretty(&output).unwrap().as_bytes()).unwrap();
}
