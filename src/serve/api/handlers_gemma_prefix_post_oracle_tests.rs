//! Post-unload Gemma metadata replay; never constructs an engine/device.
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
    family_approval: BoundFile,
    server_build_receipt: BoundFile,
    expected_server_source_commit: String,
    expected_server_binary_sha256: String,
    turns: Vec<Turn>,
    output: PathBuf,
}
fn build_matches(source: &str, binary: &str, build: &Value) -> bool {
    let hex = |s: &str, n| {
        s.len() == n
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    hex(source, 40)
        && hex(binary, 64)
        && build["source_commit"] == source
        && build["binary_sha256"] == binary
        && build["source_clean"] == true
        && build["exit"] == 0
}
fn gemma_template(family: &str, source: &str, template: &str) -> Result<(), String> {
    if family != "gemma4" || source != "GgufEmbedded" {
        return Err("exact Gemma embedded template required".into());
    }
    crate::core::chat_templates::validate_tool_chat_template("gemma4", template)
}
#[test]
fn gemma_prefix_controls_do_not_inherit_other_family_kwargs() {
    let reg = registry::find_for_architecture("gemma4").unwrap();
    assert!(resolve_api_template_kwargs(Some(&reg), None, None).is_none());
    assert!(gemma_template("deepseek4", "GgufEmbedded", "x").is_err());
    assert!(gemma_template("gemma4", "NativeEncoding", "x").is_err());
    assert!(gemma_template("gemma4", "GgufEmbedded", "<|im_start|>").is_err());
}
#[test]
fn gemma_prefix_rejects_unproved_cue_and_partial_or_rewritten_history() {
    let p = [1, 2, 3, 4];
    let s = [1, 2, 3];
    let usage =
        |cached| json!({"prompt_tokens":4,"prompt_tokens_details":{"cached_tokens":cached}});
    assert!(accept_prefix(&p, &p, None, &usage(0)).is_err());
    assert!(accept_prefix(&p, &[], None, &usage(0)).is_err());
    assert!(accept_prefix(&p, &s, None, &usage(1)).is_err());
    assert!(accept_prefix(&p, &s, Some(&[1, 2]), &usage(1)).is_err());
    assert!(accept_prefix(&p, &s, Some(&[1, 9]), &usage(2)).is_err());
    assert_eq!(accept_prefix(&p, &s, Some(&[1, 2]), &usage(2)), Ok(2));
    assert!(accept_prefix(&p, &s, Some(&[1, 2]), &usage(5)).is_err());
}
#[test]
fn gemma_prefix_build_identity_rejects_dirty_or_mixed_source() {
    let source = "a".repeat(40);
    let binary = "b".repeat(64);
    let mut build =
        json!({"source_commit":source,"binary_sha256":binary,"exit":0,"source_clean":true});
    assert!(build_matches(&source, &binary, &build));
    assert!(!build_matches(&"c".repeat(40), &binary, &build));
    assert!(!build_matches(&source, &"B".repeat(64), &build));
    build["source_clean"] = json!(false);
    assert!(!build_matches(&source, &binary, &build));
}
#[test]
#[ignore = "post-unload exact seven-wire Gemma metadata replay only"]
fn gemma4_exact_prefix_post_oracle() {
    let bytes =
        std::fs::read(std::env::var("HF2Q_GEMMA_PREFIX_POST_ORACLE_CONFIG").unwrap()).unwrap();
    let c: Config = serde_json::from_slice(&bytes).unwrap();
    assert!(matches!(c.server_slots, 1 | 4));
    assert_eq!(c.turns.len(), 7);
    let template = String::from_utf8(read(&c.template)).unwrap();
    gemma_template(&c.family, &c.template_source, &template).unwrap();
    let tokenizer_bytes = read(&c.tokenizer);
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
    assert!(tokenizer.get_truncation().is_none());
    let readiness_bytes = read(&c.readiness);
    let readiness: Value = serde_json::from_slice(&readiness_bytes).unwrap();
    let snap = &readiness["runtime"]["measurement_snapshot"];
    assert_eq!(snap["schema_version"], "hf2q.measurement-snapshot.v1");
    assert_eq!(snap["architecture"], "gemma4");
    assert_eq!(
        snap["tokenizer_sha256"],
        semantic_hash(serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap())
    );
    assert_eq!(
        snap["template_sha256"],
        semantic_hash(json!({"source":c.template_source,"template":template}))
    );
    assert_eq!(readiness["health"]["context_length"], 262144);
    assert_eq!(readiness["health"]["model"], snap["model_id"]);
    assert_eq!(readiness["readyz"]["ready"], true);
    assert_eq!(
        readiness["runtime"]["serve"]["scheduler_mode"],
        "slot_aware"
    );
    assert_eq!(readiness["runtime"]["serve"]["max_slots"], c.server_slots);
    let approval_bytes = read(&c.family_approval);
    let approval: Value = serde_json::from_slice(&approval_bytes).unwrap();
    assert_eq!(approval["family"], "gemma4");
    let fc = &approval["family_controls"];
    assert_eq!(fc["cache_family"], "gemma4");
    assert_eq!(fc["slots"], c.server_slots);
    assert_eq!(fc["context_per_slot"], 262144);
    assert_eq!(fc["hybrid_kv"], true);
    assert_eq!(fc["graft"], false);
    let controller_bytes = read(&c.controller_receipt);
    let controller: Value = serde_json::from_slice(&controller_bytes).unwrap();
    assert_eq!(controller["server_pid"], snap["process_pid"]);
    assert_eq!(controller["protocol_complete"], true);
    assert_eq!(controller["final_verified"], true);
    assert_eq!(controller["cleanup_proved"], true);
    assert!(controller.get("failure").is_none());
    assert_eq!(
        controller["status"],
        if c.server_slots == 1 {
            "ProtocolCompleteAwaitingIndependentPatchReview"
        } else {
            "FourAgentProtocolCompleteAwaitingIndependentReview"
        }
    );
    let build_bytes = read(&c.server_build_receipt);
    let build: Value = serde_json::from_slice(&build_bytes).unwrap();
    assert!(build_matches(
        &c.expected_server_source_commit,
        &c.expected_server_binary_sha256,
        &build
    ));
    let reg = registry::find_for_architecture("gemma4").unwrap();
    let mut prior: Option<Vec<u32>> = None;
    let mut controls_prior = None;
    let mut observations = Vec::new();
    for (i, t) in c.turns.iter().enumerate() {
        let wire = read(&t.request);
        let rb = read(&t.receipt);
        let receipt: Value = serde_json::from_slice(&rb).unwrap();
        assert_eq!(receipt["turn"].as_u64(), Some(i as u64));
        assert_eq!(receipt["http_status"], 200);
        assert_eq!(receipt["request_sha256"], hash(&wire));
        assert_eq!(receipt["request_bytes"].as_u64(), Some(wire.len() as u64));
        let v: Value = serde_json::from_slice(&wire).unwrap();
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
        ];
        assert!(v
            .as_object()
            .unwrap()
            .keys()
            .all(|k| allowed.contains(&k.as_str())));
        assert_eq!(v["model"], snap["model_id"]);
        assert_eq!(v["max_completion_tokens"], 8192);
        assert_eq!(v["hf2q_overflow_policy"], "reject");
        assert_eq!(v["logprobs"], false);
        let req: schema::ChatCompletionRequest = serde_json::from_value(v.clone()).unwrap();
        assert!(!req.messages.is_empty());
        assert!(req
            .messages
            .iter()
            .all(|m| m.content.as_ref().is_none_or(|x| x.image_urls().is_empty())));
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
            Some(&reg),
            req.chat_template_kwargs.as_ref(),
            controls.deepseek_reasoning_effort_tier(),
        );
        assert!(
            kwargs.is_none(),
            "no Gemma preserve_thinking or DS effort injection"
        );
        let current = json!({"tools":v["tools"],"thinking":thinking,"kwargs":kwargs});
        if let Some(previous) = &controls_prior {
            assert_eq!(previous, &current);
        }
        controls_prior = Some(current);
        let render = |cue| {
            render_chat_prompt_or_400_with_generation_prompt(
                &template,
                &req.messages,
                Some(tools),
                thinking,
                kwargs.as_ref(),
                cue,
            )
            .unwrap()
        };
        let full = render(true);
        let stable = render(false);
        let p = tokenize_http_prompt(&tokenizer, &full).unwrap();
        let s = tokenize_http_prompt(&tokenizer, &stable).unwrap();
        assert!(p.len() <= 262144 - 8192);
        let cached = accept_prefix(&p, &s, prior.as_deref(), &receipt["usage"])
            .unwrap_or_else(|why| panic!("turn {i}: {why}"));
        observations.push(json!({"turn":i,"prompt_tokens":p.len(),"stable_tokens":s.len(),"prior_stable_tokens":prior.as_ref().map(Vec::len),"cached_tokens":cached,"request_sha256":hash(&wire),"receipt_sha256":hash(&rb),"rendered_sha256":hash(full.as_bytes()),"prompt_token_ids_sha256":hash(&serde_json::to_vec(&p).unwrap()),"stable_token_ids_sha256":hash(&serde_json::to_vec(&s).unwrap())}));
        prior = Some(s);
    }
    assert!(c.output.is_absolute());
    let parent = c.output.parent().unwrap().canonicalize().unwrap();
    assert!(parent.ancestors().all(|p| !p.join(".git").exists()));
    let output = json!({"schema":"hf2q.gemma4-prefix-post-oracle.v1","family":"gemma4","server_slots":c.server_slots,"full_prefix_reuse":"PassedExactGemmaSevenWireReplay","model_inference_performed":false,"model_qualified":false,"config_sha256":hash(&bytes),"tokenizer_sha256":hash(&tokenizer_bytes),"readiness_sha256":hash(&readiness_bytes),"controller_sha256":hash(&controller_bytes),"family_approval_sha256":hash(&approval_bytes),"server_build_receipt_sha256":hash(&build_bytes),"expected_server_source_commit":c.expected_server_source_commit,"expected_server_binary_sha256":c.expected_server_binary_sha256,"oracle_source_sha256":hash(include_bytes!("handlers_gemma_prefix_post_oracle_tests.rs")),"renderer_source_sha256":hash(include_bytes!("engine.rs")),"observations":observations});
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    options
        .open(&c.output)
        .unwrap()
        .write_all(serde_json::to_string_pretty(&output).unwrap().as_bytes())
        .unwrap();
}

#[path = "handlers_gemma_prefix_failure_tests.rs"]
mod partial_failure;
