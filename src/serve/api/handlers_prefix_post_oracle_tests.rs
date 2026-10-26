//! Post-run, engine-free exact prefix oracle. No device or GGUF payload access.
//! Run only after unload; HF2Q_PREFIX_POST_ORACLE_CONFIG supplies frozen evidence.
use super::*;
use crate::serve::api::schema;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundFile {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Turn {
    request: BoundFile,
    receipt: BoundFile,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    tokenizer: BoundFile,
    template: BoundFile,
    readiness: BoundFile,
    // Debug spelling used by measurement.rs, frozen from loader provenance.
    template_source: String,
    #[serde(default = "default_server_slots")]
    server_slots: u32,
    turns: Vec<Turn>,
    output: PathBuf,
}
fn default_server_slots() -> u32 {
    1
}

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn read(bound: &BoundFile) -> Vec<u8> {
    assert!(
        bound.path.is_absolute(),
        "absolute frozen evidence path required"
    );
    assert!(
        bound.sha256.len() == 64
            && bound
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    );
    let bytes = std::fs::read(&bound.path).unwrap();
    assert_eq!(
        hash(&bytes),
        bound.sha256,
        "frozen evidence changed: {}",
        bound.path.display()
    );
    bytes
}
// Same JSON canonicalization/domain as measurement.rs; no template/tokenizer logic.
fn canonical(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let ordered: std::collections::BTreeMap<_, _> = map.into_iter().collect();
            Value::Object(
                ordered
                    .into_iter()
                    .map(|(k, v)| (k, canonical(v)))
                    .collect(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonical).collect()),
        other => other,
    }
}
fn semantic_hash(value: Value) -> String {
    hash(canonical(value).to_string().as_bytes())
}

/// Actual acceptance seam: both archived replay and pure controls call this.
fn accept_prefix(
    p: &[u32],
    s: &[u32],
    prior: Option<&[u32]>,
    usage: &Value,
) -> Result<usize, &'static str> {
    if s.is_empty() || s.len() >= p.len() || !p.starts_with(s) {
        return Err("unproved stable boundary");
    }
    if usage["prompt_tokens"].as_u64() != Some(p.len() as u64) {
        return Err("actual usage/render mismatch");
    }
    let raw_cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .ok_or("actual cached-token usage required")?;
    let cached = usize::try_from(raw_cached).map_err(|_| "cached count overflow")?;
    if cached > p.len() {
        return Err("cached count exceeds prompt");
    }
    if let Some(previous) = prior {
        if !p.starts_with(previous) {
            return Err("previous stable input is not exact current prefix");
        }
        if cached < previous.len() {
            return Err("previous stable input only partially reused");
        }
    } else if cached != 0 {
        return Err("initial turn must be cold");
    }
    Ok(cached)
}

#[test]
fn prefix_acceptance_rejects_partial_and_divergent_reuse() {
    let p = [10, 11, 12, 13, 14];
    let s = [10, 11, 12, 13];
    let prior = [10, 11, 12];
    let usage =
        |cached| json!({"prompt_tokens":5,"prompt_tokens_details":{"cached_tokens":cached}});
    // Exact prior length is sufficient; more reused tokens are also valid.
    assert_eq!(accept_prefix(&p, &s, Some(&prior), &usage(3)), Ok(3));
    assert_eq!(accept_prefix(&p, &s, Some(&prior), &usage(5)), Ok(5));
    // Positive reuse is insufficient when even one prior stable token is absent.
    assert_eq!(
        accept_prefix(&p, &s, Some(&prior), &usage(2)),
        Err("previous stable input only partially reused")
    );
    assert_eq!(
        accept_prefix(&p, &s, Some(&[10, 99, 12]), &usage(3)),
        Err("previous stable input is not exact current prefix")
    );
    assert_eq!(
        accept_prefix(
            &[10, 99, 12, 13, 14],
            &[10, 99, 12, 13],
            Some(&prior),
            &usage(3)
        ),
        Err("previous stable input is not exact current prefix")
    );
    assert_eq!(
        accept_prefix(&p, &s, Some(&[10, 11, 12, 13, 14, 15]), &usage(5)),
        Err("previous stable input is not exact current prefix")
    );
}

#[test]
fn prefix_acceptance_rejects_unstable_cue_and_bad_usage() {
    let p = [10, 11, 12];
    let s = [10, 11];
    let cold = json!({"prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":0}});
    assert_eq!(accept_prefix(&p, &s, None, &cold), Ok(0));
    for unstable in [vec![], vec![10, 11, 12], vec![10, 11, 12, 13], vec![10, 99]] {
        assert_eq!(
            accept_prefix(&p, &unstable, None, &cold),
            Err("unproved stable boundary")
        );
    }
    for bad in [
        json!({}),
        json!({"prompt_tokens":2}),
        json!({"prompt_tokens":"3"}),
    ] {
        assert_eq!(
            accept_prefix(&p, &s, None, &bad),
            Err("actual usage/render mismatch")
        );
    }
    for cached in [Value::Null, json!(-1), json!(true), json!(1.5), json!("0")] {
        let bad = json!({"prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":cached}});
        assert_eq!(
            accept_prefix(&p, &s, None, &bad),
            Err("actual cached-token usage required")
        );
    }
    assert_eq!(
        accept_prefix(&p, &s, None, &json!({"prompt_tokens":3})),
        Err("actual cached-token usage required")
    );
    assert_eq!(
        accept_prefix(
            &p,
            &s,
            None,
            &json!({"prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":1}})
        ),
        Err("initial turn must be cold")
    );
    assert_eq!(
        accept_prefix(
            &p,
            &s,
            None,
            &json!({"prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":4}})
        ),
        Err("cached count exceeds prompt")
    );
}

#[test]
#[ignore = "post-unload archived wire replay only; requires HF2Q_PREFIX_POST_ORACLE_CONFIG"]
fn qwen38_n1_exact_prefix_post_oracle() {
    let config_bytes =
        std::fs::read(std::env::var("HF2Q_PREFIX_POST_ORACLE_CONFIG").unwrap()).unwrap();
    let config: Config = serde_json::from_slice(&config_bytes).unwrap();
    assert_eq!(config.turns.len(), 7, "all seven archived wires required");
    assert!(matches!(config.server_slots, 1 | 4), "supported slot count");
    assert_eq!(
        config.template_source, "GgufEmbedded",
        "this oracle binds the N1 GGUF template arm only"
    );
    let tokenizer_bytes = read(&config.tokenizer);
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
    assert!(
        tokenizer.get_truncation().is_none(),
        "runtime tokenizer must not truncate"
    );
    let template = String::from_utf8(read(&config.template)).unwrap();
    let readiness_bytes = read(&config.readiness);
    let readiness: Value = serde_json::from_slice(&readiness_bytes).unwrap();
    let snapshot = &readiness["runtime"]["measurement_snapshot"];
    assert_eq!(snapshot["schema_version"], "hf2q.measurement-snapshot.v1");
    assert_eq!(snapshot["architecture"], "qwen35");
    assert_eq!(
        snapshot["tokenizer_sha256"],
        semantic_hash(serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap())
    );
    assert_eq!(
        snapshot["template_sha256"],
        semantic_hash(json!({"source":config.template_source,"template":template}))
    );
    assert_eq!(snapshot["active_controls"]["grammar"]["active"], false);
    // A loaded projector does not affect image-free request tokenization.
    assert_eq!(readiness["health"]["context_length"], 262144);
    assert_eq!(
        readiness["runtime"]["serve"]["scheduler_mode"],
        "slot_aware"
    );
    assert_eq!(
        readiness["runtime"]["serve"]["max_slots"],
        config.server_slots
    );
    assert_eq!(readiness["readyz"]["ready"], true);
    let registration = registry::find_for_architecture("qwen35").unwrap();
    let mut previous_stable: Option<Vec<u32>> = None;
    let mut observations = Vec::new();
    let mut prior_tools = None;
    let mut prior_kwargs = None;
    let mut prior_thinking = None;
    for (index, turn) in config.turns.iter().enumerate() {
        let wire = read(&turn.request);
        let receipt_bytes = read(&turn.receipt);
        let receipt: Value = serde_json::from_slice(&receipt_bytes).unwrap();
        assert_eq!(receipt["turn"].as_u64(), Some(index as u64));
        assert_eq!(receipt["http_status"], 200);
        assert_eq!(receipt["request_sha256"], hash(&wire));
        assert_eq!(receipt["request_bytes"].as_u64(), Some(wire.len() as u64));
        let request_value: Value = serde_json::from_slice(&wire).unwrap();
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
            "reasoning",
            "reasoning_effort",
            "thinking_token_budget",
            "chat_template_kwargs",
        ];
        assert!(
            request_value
                .as_object()
                .unwrap()
                .keys()
                .all(|k| allowed.contains(&k.as_str())),
            "unhandled render/admission controls"
        );
        assert_eq!(request_value["hf2q_overflow_policy"], "reject");
        assert_eq!(request_value["max_completion_tokens"], 8192);
        assert_eq!(request_value["model"], snapshot["model_id"]);
        assert_eq!(request_value["logprobs"], false);
        let req: schema::ChatCompletionRequest =
            serde_json::from_value(request_value.clone()).unwrap();
        assert!(!req.messages.is_empty());
        assert!(req
            .messages
            .iter()
            .all(|m| m.content.as_ref().is_none_or(|c| c.image_urls().is_empty())));
        let choice = grammar::request::validate_tool_request(&req).unwrap();
        assert!(matches!(choice, schema::ToolChoiceValue::Auto));
        let tools = req.tools.as_deref().expect("actual tool schemas required");
        assert_eq!(tools.len(), 3);
        registry::qwen35_tool_argument_wire_kinds(tools).unwrap();
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
            kwargs.as_ref().unwrap().get("preserve_thinking"),
            Some(&Value::Bool(true))
        );
        if index > 0 {
            assert_eq!(prior_tools.as_ref(), Some(&request_value["tools"]));
            assert_eq!(prior_kwargs.as_ref(), Some(&kwargs));
            assert_eq!(prior_thinking, Some(thinking));
        }
        prior_tools = Some(request_value["tools"].clone());
        prior_kwargs = Some(kwargs.clone());
        prior_thinking = Some(thinking);
        let p = render_chat_prompt_or_400_with_generation_prompt(
            &template,
            &req.messages,
            Some(tools),
            thinking,
            kwargs.as_ref(),
            true,
        )
        .unwrap();
        let s = render_chat_prompt_or_400_with_generation_prompt(
            &template,
            &req.messages,
            Some(tools),
            thinking,
            kwargs.as_ref(),
            false,
        )
        .unwrap();
        let p_tokens = tokenize_http_prompt(&tokenizer, &p).unwrap();
        let s_tokens = tokenize_http_prompt(&tokenizer, &s).unwrap();
        assert!(
            p_tokens.len() <= 262144 - 8192,
            "no overflow rewrite admissible"
        );
        let cached = accept_prefix(
            &p_tokens,
            &s_tokens,
            previous_stable.as_deref(),
            &receipt["usage"],
        )
        .unwrap_or_else(|reason| panic!("turn {index}: {reason}"));
        observations.push(json!({"turn":index,"request_sha256":hash(&wire),"receipt_sha256":hash(&receipt_bytes),"prompt_tokens":p_tokens.len(),"stable_tokens":s_tokens.len(),"excluded_generation_cue_tokens":p_tokens.len()-s_tokens.len(),"prior_stable_tokens":previous_stable.as_ref().map(Vec::len),"cached_tokens":cached,"prompt_token_ids_sha256":hash(&serde_json::to_vec(&p_tokens).unwrap()),"stable_token_ids_sha256":hash(&serde_json::to_vec(&s_tokens).unwrap()),"rendered_sha256":hash(p.as_bytes()),"enable_thinking":thinking,"resolved_kwargs":kwargs}));
        previous_stable = Some(s_tokens);
    }
    assert!(config.output.is_absolute());
    let parent = config.output.parent().unwrap().canonicalize().unwrap();
    assert!(
        parent.ancestors().all(|p| !p.join(".git").exists()),
        "output outside every checkout required"
    );
    let output = json!({"schema":"hf2q.qwen38-n1-prefix-post-oracle.v1","full_prefix_reuse":"PassedExactSevenWireReplay","model_inference_performed":false,"model_qualified":false,"server_slots":config.server_slots,"runtime_vision_projector_loaded":snapshot["active_controls"]["vision_projector"],"config_sha256":hash(&config_bytes),"tokenizer_file_sha256":hash(&tokenizer_bytes),"template_file_sha256":hash(template.as_bytes()),"readiness_sha256":hash(&readiness_bytes),"oracle_source_sha256":hash(include_bytes!("handlers_prefix_post_oracle_tests.rs")),"handler_source_sha256":hash(include_bytes!("handlers.rs")),"renderer_source_sha256":hash(include_bytes!("engine.rs")),"measurement_source_sha256":hash(include_bytes!("measurement.rs")),"observations":observations});
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options.open(&config.output).unwrap();
    file.write_all(serde_json::to_string_pretty(&output).unwrap().as_bytes())
        .unwrap();
}
