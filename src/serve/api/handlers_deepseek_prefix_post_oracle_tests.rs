//! Post-unload native DeepSeek replay. No Engine, device, tensor or network work.
use super::prefix_post_oracle_tests::{accept_prefix, hash, read, semantic_hash, BoundFile};
use super::*;
use crate::serve::api::schema;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Turn {
    request: BoundFile,
    receipt: BoundFile,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    family: String,
    server_slots: u32,
    tokenizer: BoundFile,
    template: BoundFile,
    template_source: String,
    readiness: BoundFile,
    controller_receipt: BoundFile,
    server_build_receipt: BoundFile,
    #[serde(default = "default_server_source_commit")]
    expected_server_source_commit: String,
    #[serde(default = "default_server_binary_sha256")]
    expected_server_binary_sha256: String,
    turns: Vec<Turn>,
    output: PathBuf,
}

fn default_server_source_commit() -> String {
    "68804c57b44d4139cfd264c4544bed072d0da13d".into()
}
fn default_server_binary_sha256() -> String {
    "d88efd5c8e145b59de4c9b88093c3bfe77e7af34e21255e872146f118bceec13".into()
}
fn build_identity_matches(source: &str, binary: &str, build: &Value) -> bool {
    let hex = |value: &str, len| {
        value.len() == len && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    hex(source, 40) && hex(binary, 64)
        && build["source_commit"] == source && build["binary_sha256"] == binary
        && build["exit"] == 0 && build["source_clean"] == true
}

#[test]
fn deepseek_prefix_build_identity_defaults_and_explicit_mismatches() {
    let source = default_server_source_commit();
    let binary = default_server_binary_sha256();
    let mut build = json!({"source_commit":source,"binary_sha256":binary,"exit":0,"source_clean":true});
    assert!(build_identity_matches(&source, &binary, &build));
    assert!(!build_identity_matches(&source.to_uppercase(), &binary, &build));
    assert!(!build_identity_matches(&source, "bad", &build));
    build["source_commit"] = json!("2d7599907be92d17736bc205806907589e845306");
    assert!(!build_identity_matches(&source, &binary, &build));
    assert!(build_identity_matches(build["source_commit"].as_str().unwrap(), &binary, &build));
    build["source_clean"] = json!(false);
    assert!(!build_identity_matches(build["source_commit"].as_str().unwrap(), &binary, &build));
}

fn native_template(family: &str, source: &str, template: &str) -> Result<(), &'static str> {
    if family != "deepseek4"
        || source != "NativeEncoding { name: \"DEEPSEEK_V4_FLASH_0731\" }"
        || template != crate::core::chat_templates::DEEPSEEK_V4_FLASH_0731
    {
        return Err("not the exact DeepSeek native renderer");
    }
    Ok(())
}

#[test]
fn deepseek_prefix_rejects_qwen_or_jinja_template_fallback() {
    let source = "NativeEncoding { name: \"DEEPSEEK_V4_FLASH_0731\" }";
    let template = crate::core::chat_templates::DEEPSEEK_V4_FLASH_0731;
    assert!(native_template("deepseek4", source, template).is_ok());
    assert!(native_template("qwen35", source, template).is_err());
    assert!(native_template("deepseek4", "GgufEmbedded", template).is_err());
    assert!(native_template("deepseek4", source, "<|im_start|>assistant").is_err());
}

fn anchor(tokens: &[u32]) -> Result<&[u32], &'static str> {
    let tail = super::super::engine_deepseek4::RECOVERY_TAIL_TOKENS;
    if tokens.len() <= tail {
        return Err("no nonempty native recovery anchor");
    }
    Ok(&tokens[..tokens.len() - tail])
}

#[test]
fn deepseek_prefix_native_anchor_reuses_actual_acceptance_negative_controls() {
    let tail = super::super::engine_deepseek4::RECOVERY_TAIL_TOKENS;
    assert_eq!(tail, 8, "reviewed native predicate changed");
    assert!(anchor(&vec![1; tail]).is_err());
    let prior: Vec<u32> = (0..20).collect();
    let a = anchor(&prior).unwrap();
    let mut current = prior.clone();
    current.extend([50, 51]);
    let usage = |cached| json!({"prompt_tokens":current.len(),"prompt_tokens_details":{"cached_tokens":cached}});
    assert_eq!(
        accept_prefix(
            &current,
            anchor(&current).unwrap(),
            Some(a),
            &usage(a.len())
        ),
        Ok(a.len())
    );
    assert!(accept_prefix(
        &current,
        anchor(&current).unwrap(),
        Some(a),
        &usage(a.len() - 1)
    )
    .is_err());
    let mut rewritten = current.clone();
    rewritten[0] = 99;
    assert!(accept_prefix(
        &rewritten,
        anchor(&rewritten).unwrap(),
        Some(a),
        &usage(a.len())
    )
    .is_err());
    let mut tail_only = current.clone();
    tail_only[a.len()] = 99;
    assert!(accept_prefix(
        &tail_only,
        anchor(&tail_only).unwrap(),
        Some(a),
        &usage(a.len())
    )
    .is_ok());
}

#[test]
fn deepseek_prefix_native_dispatch_is_not_qwen_generation_cue_rendering() {
    let messages: Vec<schema::ChatMessage> = serde_json::from_value(json!([
        {"role":"system","content":"Exact source"}, {"role":"user","content":"Read the file"}
    ]))
    .unwrap();
    let template = crate::core::chat_templates::DEEPSEEK_V4_FLASH_0731;
    let p = render_chat_prompt_or_400_with_generation_prompt(
        template, &messages, None, true, None, true,
    )
    .unwrap();
    let no_cue = render_chat_prompt_or_400_with_generation_prompt(
        template, &messages, None, true, None, false,
    )
    .unwrap();
    assert_eq!(
        p, no_cue,
        "native dispatcher deliberately ignores Jinja cue flag"
    );
    assert!(p.starts_with("<｜begin▁of▁sentence｜>"));
    assert!(!p.contains("<|im_start|>"));
}

#[test]
#[ignore = "post-unload seven-wire DeepSeek metadata replay only; no model/device"]
fn deepseek_exact_native_prefix_post_oracle() {
    let config_bytes =
        std::fs::read(std::env::var("HF2Q_DEEPSEEK_PREFIX_POST_ORACLE_CONFIG").unwrap()).unwrap();
    let config: Config = serde_json::from_slice(&config_bytes).unwrap();
    assert_eq!(config.family, "deepseek4");
    assert!(matches!(config.server_slots, 1 | 4));
    assert_eq!(config.turns.len(), 7);
    assert_eq!(super::super::engine_deepseek4::RECOVERY_TAIL_TOKENS, 8);
    let template = String::from_utf8(read(&config.template)).unwrap();
    native_template(&config.family, &config.template_source, &template).unwrap();
    assert_eq!(
        template,
        crate::core::chat_templates::DEEPSEEK_V4_FLASH_0731,
        "no Jinja/family fallback"
    );
    assert_eq!(
        config.template_source,
        "NativeEncoding { name: \"DEEPSEEK_V4_FLASH_0731\" }"
    );
    let tokenizer_bytes = read(&config.tokenizer);
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
    assert!(tokenizer.get_truncation().is_none());
    let readiness_bytes = read(&config.readiness);
    let readiness: Value = serde_json::from_slice(&readiness_bytes).unwrap();
    let snapshot = &readiness["runtime"]["measurement_snapshot"];
    assert_eq!(snapshot["schema_version"], "hf2q.measurement-snapshot.v1");
    assert_eq!(snapshot["architecture"], "deepseek4");
    assert_eq!(
        snapshot["tokenizer_sha256"],
        semantic_hash(serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap())
    );
    assert_eq!(
        snapshot["template_sha256"],
        semantic_hash(json!({"source":config.template_source,"template":template}))
    );
    assert_eq!(readiness["health"]["context_length"], 262144);
    assert_eq!(readiness["health"]["model"], snapshot["model_id"]);
    assert_eq!(readiness["readyz"]["ready"], true);
    assert_eq!(
        readiness["runtime"]["serve"]["scheduler_mode"],
        "slot_aware"
    );
    assert_eq!(
        readiness["runtime"]["serve"]["max_slots"],
        config.server_slots
    );
    let controller_bytes = read(&config.controller_receipt);
    let controller: Value = serde_json::from_slice(&controller_bytes).unwrap();
    assert_eq!(controller["server_pid"], snapshot["process_pid"]);
    assert_eq!(controller["protocol_complete"], true);
    assert!(controller.get("failure").is_none());
    let terminal = if config.server_slots == 1 {
        "ProtocolCompleteAwaitingIndependentPatchReview"
    } else {
        "FourAgentProtocolCompleteAwaitingIndependentReview"
    };
    assert_eq!(controller["status"], terminal);
    assert_eq!(controller["final_verified"], true);
    assert_eq!(controller["cleanup_proved"], true);
    let build_bytes = read(&config.server_build_receipt);
    let build: Value = serde_json::from_slice(&build_bytes).unwrap();
    assert!(build_identity_matches(
        &config.expected_server_source_commit,
        &config.expected_server_binary_sha256,
        &build,
    ), "expected clean server build identity mismatch");
    let registration = registry::find_for_architecture("deepseek4").unwrap();
    let mut prior: Option<Vec<u32>> = None;
    let mut observations = Vec::new();
    let mut prior_controls = None;
    for (index, turn) in config.turns.iter().enumerate() {
        let wire = read(&turn.request);
        let receipt_bytes = read(&turn.receipt);
        let receipt: Value = serde_json::from_slice(&receipt_bytes).unwrap();
        assert_eq!(receipt["turn"].as_u64(), Some(index as u64));
        assert_eq!(receipt["http_status"], 200);
        assert_eq!(receipt["request_sha256"], hash(&wire));
        assert_eq!(receipt["request_bytes"].as_u64(), Some(wire.len() as u64));
        let value: Value = serde_json::from_slice(&wire).unwrap();
        let allowed = [
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
        assert!(value
            .as_object()
            .unwrap()
            .keys()
            .all(|k| allowed.contains(&k.as_str())));
        assert_eq!(value["model"], snapshot["model_id"]);
        assert_eq!(value["max_completion_tokens"], 8192);
        assert_eq!(value["hf2q_overflow_policy"], "reject");
        assert_eq!(value["logprobs"], false);
        assert_eq!(value["temperature"], 0);
        assert_eq!(value["reasoning_effort"], "max");
        let req: schema::ChatCompletionRequest = serde_json::from_value(value.clone()).unwrap();
        assert!(!req.messages.is_empty());
        assert!(req
            .messages
            .iter()
            .all(|m| m.content.as_ref().is_none_or(|c| c.image_urls().is_empty())));
        assert!(matches!(
            grammar::request::validate_tool_request(&req).unwrap(),
            schema::ToolChoiceValue::Auto
        ));
        let tools = req.tools.as_deref().unwrap();
        assert_eq!(tools.len(), 3);
        let controls = reasoning_controls_from_request(&req).unwrap();
        let thinking = resolve_api_enable_thinking(
            controls.thinking_enabled_override(),
            crate::serve::template_supports_enable_thinking(&template),
        );
        let kwargs = resolve_api_template_kwargs(
            Some(&registration),
            req.chat_template_kwargs.as_ref(),
            controls.deepseek_reasoning_effort_tier(),
        );
        assert_eq!(
            kwargs.as_ref().unwrap().get("reasoning_effort"),
            Some(&json!("max"))
        );
        let current_controls = json!({"tools":value["tools"],"thinking":thinking,"kwargs":kwargs});
        if let Some(previous) = &prior_controls {
            assert_eq!(previous, &current_controls);
        }
        prior_controls = Some(current_controls);
        let rendered = render_chat_prompt_or_400_with_generation_prompt(
            &template,
            &req.messages,
            Some(tools),
            thinking,
            kwargs.as_ref(),
            true,
        )
        .unwrap();
        let p = tokenize_http_prompt(&tokenizer, &rendered).unwrap();
        assert!(p.len() <= 262144 - 8192);
        let a = anchor(&p).unwrap();
        let cached = accept_prefix(&p, a, prior.as_deref(), &receipt["usage"])
            .unwrap_or_else(|why| panic!("turn {index}: {why}"));
        observations.push(json!({"turn":index,"prompt_tokens":p.len(),"anchor_tokens":a.len(),"excluded_native_tail_tokens":8,"prior_anchor_tokens":prior.as_ref().map(Vec::len),"cached_tokens":cached,"request_sha256":hash(&wire),"receipt_sha256":hash(&receipt_bytes),"rendered_sha256":hash(rendered.as_bytes()),"prompt_token_ids_sha256":hash(&serde_json::to_vec(&p).unwrap()),"anchor_token_ids_sha256":hash(&serde_json::to_vec(a).unwrap())}));
        prior = Some(a.to_vec());
    }
    assert!(config.output.is_absolute());
    let parent = config.output.parent().unwrap().canonicalize().unwrap();
    assert!(parent.ancestors().all(|p| !p.join(".git").exists()));
    let output = json!({"schema":"hf2q.deepseek4-prefix-post-oracle.v1","family":"deepseek4","server_slots":config.server_slots,"full_prefix_reuse":"PassedExactNativeSevenWireReplay","model_inference_performed":false,"model_qualified":false,"config_sha256":hash(&config_bytes),"tokenizer_sha256":hash(&tokenizer_bytes),"readiness_sha256":hash(&readiness_bytes),"controller_sha256":hash(&controller_bytes),"server_build_receipt_sha256":hash(&build_bytes),"expected_server_source_commit":config.expected_server_source_commit,"expected_server_binary_sha256":config.expected_server_binary_sha256,"oracle_source_sha256":hash(include_bytes!("handlers_deepseek_prefix_post_oracle_tests.rs")),"renderer_source_sha256":hash(include_bytes!("engine.rs")),"cache_source_sha256":hash(include_bytes!("engine_deepseek4.rs")),"native_encoder_source_sha256":hash(include_bytes!("../../core/deepseek_v4_encoding/mod.rs")),"observations":observations});
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    options
        .open(&config.output)
        .unwrap()
        .write_all(serde_json::to_string_pretty(&output).unwrap().as_bytes())
        .unwrap();
}
