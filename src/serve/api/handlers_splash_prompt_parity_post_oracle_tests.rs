//! Post-run, engine-free Splash prompt-parity oracle (ADR-063 "Splash peer of record for the Qwen families";
//! docs/research/adr063-splash-peer-design.md section 6.4). For every archived Splash request it renders the
//! canonical hf2q form of that request (scripts/four_agent/native_peer_splash.canonical: same messages and tools,
//! thinking on, preserve_thinking, no effort) with hf2q's production request schema, reasoning controls, Qwen
//! template kwargs and renderer, using the artifact's GGUF-embedded template, and requires the bytes to equal
//! Splash's own `/apply-template` rendering archived at closing (`splash-apply-template.raw`). A mismatch makes
//! the run Invalid. No device, GGUF payload or HTTP access; HF2Q_SPLASH_PARITY_POST_ORACLE_CONFIG supplies the
//! frozen evidence (`scripts/four_agent/splash_sweep.py parity-config` writes it).
use super::prefix_post_oracle_tests::{hash, read, BoundFile};
use super::*;
use crate::serve::api::schema;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Turn {
    agent: String,
    turn: u64,
    canonical_request: BoundFile,
    splash_apply_template: BoundFile,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    architecture: String,
    template: BoundFile,
    turns: Vec<Turn>,
    output: PathBuf,
}

/// Byte offset of the first difference between two renderings, or None when they are equal.
fn first_difference(left: &str, right: &str) -> Option<usize> {
    left.bytes()
        .zip(right.bytes())
        .position(|(a, b)| a != b)
        .or_else(|| (left.len() != right.len()).then(|| left.len().min(right.len())))
}

/// hf2q's prompt for a canonical Qwen chat request: the production request schema, reasoning controls, template
/// kwargs and renderer, with the generation prompt (the bytes a request is tokenized from).
fn hf2q_prompt(template: &str, architecture: &str, request: &Value) -> String {
    let req: schema::ChatCompletionRequest = serde_json::from_value(request.clone()).unwrap();
    assert!(
        req.messages
            .iter()
            .all(|m| m.content.as_ref().is_none_or(|c| c.image_urls().is_empty())),
        "the Splash arm serves text only"
    );
    let registration = registry::find_for_architecture(architecture).unwrap();
    let controls = reasoning_controls_from_request(&req).unwrap();
    let thinking = resolve_api_enable_thinking(
        controls.thinking_enabled_override(),
        crate::serve::template_supports_enable_thinking(template),
    );
    let kwargs = resolve_api_template_kwargs(
        Some(&registration),
        req.chat_template_kwargs.as_ref(),
        controls.deepseek_reasoning_effort_tier(),
    );
    assert_eq!(
        kwargs.as_ref().unwrap().get("preserve_thinking"),
        Some(&Value::Bool(true)),
        "hf2q's Qwen rendering keeps preserve_thinking, as the Splash request sends it"
    );
    render_chat_prompt_or_400_with_generation_prompt(
        template,
        &req.messages,
        req.tools.as_deref(),
        thinking,
        kwargs.as_ref(),
        true,
    )
    .unwrap()
}

#[test]
fn splash_parity_first_difference_locates_the_byte() {
    assert_eq!(first_difference("abc", "abc"), None);
    assert_eq!(first_difference("abc", "abd"), Some(2));
    assert_eq!(first_difference("ab", "abc"), Some(2));
    assert_eq!(first_difference("", "a"), Some(0));
}

#[test]
#[ignore = "post-run archived Splash render replay only; requires HF2Q_SPLASH_PARITY_POST_ORACLE_CONFIG"]
fn splash_prompt_parity_post_oracle() {
    let config_bytes =
        std::fs::read(std::env::var("HF2Q_SPLASH_PARITY_POST_ORACLE_CONFIG").unwrap()).unwrap();
    let config: Config = serde_json::from_slice(&config_bytes).unwrap();
    assert!(
        matches!(config.architecture.as_str(), "qwen35" | "qwen35moe"),
        "the Splash arm serves the Qwen3.8 (qwen35) and Qwen3.6 (qwen35moe) families only"
    );
    assert!(!config.turns.is_empty(), "archived Splash turns required");
    let template = String::from_utf8(read(&config.template)).unwrap();
    let mut observations = Vec::new();
    for turn in &config.turns {
        let wire = read(&turn.canonical_request);
        let request: Value = serde_json::from_slice(&wire).unwrap();
        assert_eq!(
            request["hf2q_overflow_policy"], "reject",
            "the canonical hf2q request form"
        );
        let mine = hf2q_prompt(&template, &config.architecture, &request);
        let raw = read(&turn.splash_apply_template);
        let splash: Value = serde_json::from_slice(&raw).unwrap();
        let theirs = splash["prompt"]
            .as_str()
            .expect("Splash /apply-template {\"prompt\": ...}");
        if let Some(offset) = first_difference(&mine, theirs) {
            let start = offset.saturating_sub(80);
            panic!(
                "{} turn {}: hf2q and Splash prompts differ at byte {offset} (hf2q {} bytes, Splash {} bytes); \
                 hf2q {:?} vs Splash {:?}",
                turn.agent,
                turn.turn,
                mine.len(),
                theirs.len(),
                &mine.as_bytes()[start..(offset + 80).min(mine.len())],
                &theirs.as_bytes()[start..(offset + 80).min(theirs.len())]
            );
        }
        observations.push(json!({"agent":turn.agent,"turn":turn.turn,
            "canonical_request_sha256":hash(&wire),"splash_apply_template_sha256":hash(&raw),
            "prompt_sha256":hash(mine.as_bytes()),"prompt_bytes":mine.len(),"equal":true}));
    }
    assert!(config.output.is_absolute());
    let parent = config.output.parent().unwrap().canonicalize().unwrap();
    assert!(
        parent.ancestors().all(|p| !p.join(".git").exists()),
        "output outside every checkout required"
    );
    let output = json!({"schema":"hf2q.splash-prompt-parity-post-oracle.v1",
        "hf2q_prompt_parity":"PassedByteIdentical","model_inference_performed":false,
        "architecture":config.architecture,"config_sha256":hash(&config_bytes),
        "template_file_sha256":hash(template.as_bytes()),
        "oracle_source_sha256":hash(include_bytes!("handlers_splash_prompt_parity_post_oracle_tests.rs")),
        "handler_source_sha256":hash(include_bytes!("handlers.rs")),
        "renderer_source_sha256":hash(include_bytes!("engine.rs")),"observations":observations});
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
