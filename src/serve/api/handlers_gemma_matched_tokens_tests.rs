//! Neutral post-study P-vector exporter: no model, engine, GPU or prefix verdict.
use super::prefix_post_oracle_tests::{hash, semantic_hash};
use super::*;
use crate::serve::api::schema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::os::unix::fs::OpenOptionsExt;
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};

const SERVER: &str = "18162bb702806ac6c083bf01ec80f4675207fda4";
const BINARY: &str = "ff655efd5013a11ad608330f35b3622c2e448f019dcc4e94c8c483eb03cd09e1";
const RAW_LIMIT: usize = 256 * 1024;
const NEUTRAL_DECL: &str = "\n#[cfg(test)]\n#[path = \"handlers_gemma_matched_tokens_tests.rs\"]\nmod gemma_matched_tokens_tests;\n";
const FAMILY_DECL: &str = "\n#[cfg(test)]\n#[path = \"handlers_gemma_full_input_post_oracle_tests.rs\"]\nmod gemma_full_input_post_oracle_tests;\n";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundFile {
    path: PathBuf,
    sha256: String,
}
impl BoundFile {
    fn read(&self, limit: usize) -> Vec<u8> {
        let meta = std::fs::symlink_metadata(&self.path).unwrap();
        assert!(meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= limit as u64);
        let bytes = std::fs::read(&self.path).unwrap();
        assert!(bytes.len() <= limit);
        assert_eq!(hash(&bytes), self.sha256, "bound input changed");
        bytes
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Turn {
    request: BoundFile,
    receipt: BoundFile,
    response: BoundFile,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    arm: String,
    tokenizer: BoundFile,
    template: BoundFile,
    readiness: BoundFile,
    controller_receipt: BoundFile,
    family_approval: BoundFile,
    server_environment: BoundFile,
    server_build_receipt: BoundFile,
    turns: BTreeMap<String, Vec<Turn>>,
    output: PathBuf,
}
#[derive(Serialize)]
struct Export<'a> {
    #[serde(flatten)]
    metadata: Value,
    vectors: &'a BTreeMap<String, Vec<Vec<u32>>>,
    initial_anchors: &'a BTreeMap<String, Vec<u32>>,
}
fn token_hash(tokens: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    hex::encode(digest.finalize())
}
fn output_path(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err("absolute output required".into());
    }
    let name = path.file_name().ok_or("output filename required")?;
    let parent = path.parent().ok_or("output parent required")?;
    let canonical = parent.canonicalize().map_err(|e| e.to_string())?;
    if canonical != parent {
        return Err("canonical output parent required".into());
    }
    if canonical
        .ancestors()
        .any(|p| std::fs::symlink_metadata(p.join(".git")).is_ok())
    {
        return Err("output must be outside every Git ancestor".into());
    }
    Ok(canonical.join(name))
}
fn transport_matches(request: &Value, receipt: &Value) -> bool {
    request["stream"].is_boolean() && request["stream"] == receipt["stream"]
}
fn production_handlers(mut text: &str) -> Option<&str> {
    let (mut neutral, mut family) = (false, false);
    for _ in 0..2 {
        if let Some(prefix) = text.strip_suffix(NEUTRAL_DECL) {
            if neutral {
                return None;
            }
            neutral = true;
            text = prefix;
        } else if let Some(prefix) = text.strip_suffix(FAMILY_DECL) {
            if family {
                return None;
            }
            family = true;
            text = prefix;
        } else {
            return None;
        }
    }
    (neutral && family).then_some(text)
}
fn source_units() -> Value {
    json!({
        "engine":hash(include_bytes!("engine.rs")),
        "handlers":hash(production_handlers(include_str!("handlers.rs")).expect("both reviewed test declarations only").as_bytes()),
        "schema":hash(include_bytes!("schema.rs")),
        "registry":hash(include_bytes!("registry.rs")),
        "reasoning":hash(include_bytes!("reasoning_controls.rs")),
        "templates":hash(include_bytes!("../../core/chat_templates.rs")),
        "lock":hash(include_bytes!("../../../Cargo.lock"))
    })
}
fn expected_source_units() -> Value {
    json!({
        "engine":"f3033d1b0c173c3edbf2e9875d450baf2f162fc23a30702dc393bf5b95b68f32",
        "handlers":"7add1be9d5e562f173870e5a0cd9626cec716d43f040628f43aba44394212e81",
        "schema":"769470235c7e5799a39e4e01ec4dc0faf0e9974952462fce2fc9227312bdbefd",
        "registry":"4c557211a584f5d7373e5e4bc1cbb21f3f59dd3b8f7a7de3045929dd92898746",
        "reasoning":"8fbc94e337add533f0c1db39ebc3d0373fa7befd3cd3a82371ed0cd677253183",
        "templates":"9a54dcac15b0ab40a5351f5c58837d9df9928db32a95980bc30ec38f68bf0438",
        "lock":"e3abe5d3483c5f21e2a8b0a0e1e90897461aec5ac11a91a90657f6e99091d9ff"
    })
}
fn arm_matches(arm: &str, env: &Value, approval: &Value) -> bool {
    let flag = "HF2Q_GEMMA_FULL_INPUT_CHECKPOINT";
    let ack = "HF2Q_UNSAFE_EXPERIMENTS";
    approval["benchmark_profile"] == "gemma-matched-v1"
        && match arm {
            "OFF" => {
                env.get(flag).is_none()
                    && env.get(ack).is_none()
                    && approval["experimental_gemma_full_input_checkpoint"] == false
            }
            "ON" => {
                env[flag] == "1"
                    && env[ack] == "1"
                    && approval["experimental_gemma_full_input_checkpoint"] == true
            }
            _ => false,
        }
}
fn request_shape(v: &Value, turn: usize, model: &str) -> bool {
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
    v.as_object()
        .is_some_and(|o| o.keys().all(|k| allowed.contains(&k.as_str())))
        && v["model"] == model
        && v["max_completion_tokens"] == 8192
        && v["logprobs"] == false
        && v["hf2q_overflow_policy"] == "reject"
        && v["tool_choice"] == "auto"
        && v["stream"] == (turn % 2 == 1)
}
// Same completed-wire checks as the reviewed familyV2 helper; no prefix logic.
fn completed_usage(bytes: &[u8], stream: bool, model: &str) -> Value {
    if !stream {
        let v: Value = serde_json::from_slice(bytes).unwrap();
        assert_eq!(v["model"], model);
        assert!(v.get("error").is_none());
        assert_eq!(v["choices"].as_array().unwrap().len(), 1);
        assert_eq!(v["choices"][0]["index"], 0);
        assert!(matches!(
            v["choices"][0]["finish_reason"].as_str(),
            Some("stop" | "tool_calls" | "length")
        ));
        assert!(v["usage"].is_object());
        return v["usage"].clone();
    }
    let text = std::str::from_utf8(bytes).unwrap();
    assert!(text.ends_with("\n\n") || text.ends_with("\r\n\r\n"));
    let (mut usage, mut done, mut finished) = (None, false, 0);
    for line in text.lines().filter(|s| !s.is_empty()) {
        assert!(!done, "data after DONE");
        let data = line.strip_prefix("data: ").expect("SSE data only");
        if data == "[DONE]" {
            done = true;
            continue;
        }
        let v: Value = serde_json::from_str(data).unwrap();
        assert_eq!(v["model"], model);
        assert!(v.get("error").is_none());
        for choice in v["choices"].as_array().unwrap() {
            assert_eq!(choice["index"], 0);
            if !choice["finish_reason"].is_null() {
                assert!(matches!(
                    choice["finish_reason"].as_str(),
                    Some("stop" | "tool_calls" | "length")
                ));
                finished += 1;
            }
        }
        if v["usage"].is_object() {
            assert!(usage.is_none());
            usage = Some(v["usage"].clone());
        }
    }
    assert!(done && finished == 1);
    usage.expect("real usage")
}
#[test]
#[ignore = "after ALL20 cohorts/unload; engine/device-free neutral exporter"]
fn gemma_matched_export_twenty_eight_input_vectors() {
    let config_path = PathBuf::from(std::env::var("HF2Q_GEMMA_MATCHED_EXPORT_CONFIG").unwrap());
    let meta = std::fs::symlink_metadata(&config_path).unwrap();
    assert!(meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= 2 * 1024 * 1024);
    let config_bytes = std::fs::read(config_path).unwrap();
    let c: Config = serde_json::from_slice(&config_bytes).unwrap();
    assert_eq!(
        source_units(),
        expected_source_units(),
        "merge reviewed181 source before export"
    );
    let build_bytes = c.server_build_receipt.read(1024 * 1024);
    let build: Value = serde_json::from_slice(&build_bytes).unwrap();
    assert_eq!(build["source_commit"], SERVER);
    assert_eq!(build["binary_sha256"], BINARY);
    assert_eq!(build["source_clean"], true);
    assert_eq!(build["exit"], 0);
    let readiness_bytes = c.readiness.read(1024 * 1024);
    let readiness: Value = serde_json::from_slice(&readiness_bytes).unwrap();
    let snap = &readiness["runtime"]["measurement_snapshot"];
    assert_eq!(snap["architecture"], "gemma4");
    assert_eq!(snap["active_controls"]["vision_projector"], false);
    assert_eq!(readiness["health"]["context_length"], 262144);
    assert_eq!(readiness["runtime"]["serve"]["max_slots"], 4);
    let approval_bytes = c.family_approval.read(1024 * 1024);
    let approval: Value = serde_json::from_slice(&approval_bytes).unwrap();
    let env_bytes = c.server_environment.read(1024 * 1024);
    let env: Value = serde_json::from_slice(&env_bytes).unwrap();
    assert!(arm_matches(&c.arm, &env, &approval));
    let fc = &approval["family_controls"];
    assert_eq!(fc["cache_family"], "gemma4");
    assert_eq!(fc["context_per_slot"], 262144);
    assert_eq!(fc["slots"], 4);
    assert_eq!(fc["hybrid_kv"], true);
    assert_eq!(fc["codebook_bits"], 8);
    let controller_bytes = c.controller_receipt.read(1024 * 1024);
    let controller: Value = serde_json::from_slice(&controller_bytes).unwrap();
    assert_eq!(
        controller["status"],
        "FourAgentProtocolCompleteAwaitingIndependentReview"
    );
    for key in ["protocol_complete", "final_verified", "cleanup_proved"] {
        assert_eq!(controller[key], true);
    }
    assert!(controller.get("failure").is_none());
    assert_eq!(controller["server_pid"], snap["process_pid"]);
    let tokenizer_bytes = c.tokenizer.read(64 * 1024 * 1024);
    assert_eq!(
        hash(&tokenizer_bytes),
        "30395845c857a582ba54a960819dd5492b22311d5b0fc5607852f09be080d955"
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
    assert!(tokenizer.get_truncation().is_none());
    assert_eq!(
        snap["tokenizer_sha256"],
        semantic_hash(serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap())
    );
    let template_bytes = c.template.read(1024 * 1024);
    assert_eq!(
        hash(&template_bytes),
        "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
    );
    let template = std::str::from_utf8(&template_bytes).unwrap();
    crate::core::chat_templates::validate_tool_chat_template("gemma4", template).unwrap();
    assert_eq!(
        snap["template_sha256"],
        semantic_hash(json!({"source":"GgufEmbedded","template":template}))
    );
    assert_eq!(
        c.turns.keys().cloned().collect::<Vec<_>>(),
        ["agent-0", "agent-1", "agent-2", "agent-3"]
    );
    let reg = registry::find_for_architecture("gemma4").unwrap();
    let mut vectors = BTreeMap::new();
    let mut initial_anchors = BTreeMap::new();
    let mut observations = BTreeMap::new();
    for (agent, turns) in &c.turns {
        assert_eq!(turns.len(), 7);
        let mut ids = Vec::new();
        let mut records = Vec::new();
        let mut controls_prior = None;
        for (turn, t) in turns.iter().enumerate() {
            let wire = t.request.read(RAW_LIMIT);
            let v: Value = serde_json::from_slice(&wire).unwrap();
            assert!(request_shape(&v, turn, snap["model_id"].as_str().unwrap()));
            let req: schema::ChatCompletionRequest = serde_json::from_value(v.clone()).unwrap();
            assert!(
                !req.messages.is_empty()
                    && req
                        .messages
                        .iter()
                        .all(|m| m.content.as_ref().is_none_or(|x| x.image_urls().is_empty()))
            );
            assert!(matches!(
                grammar::request::validate_tool_request(&req).unwrap(),
                schema::ToolChoiceValue::Auto
            ));
            let tools = req.tools.as_deref().unwrap();
            assert_eq!(tools.len(), 3);
            let controls = reasoning_controls_from_request(&req).unwrap();
            let thinking = resolve_api_enable_thinking(
                controls.thinking_enabled_override(),
                crate::serve::template_supports_enable_thinking(template),
            );
            let kwargs = resolve_api_template_kwargs(
                Some(&reg),
                req.chat_template_kwargs.as_ref(),
                controls.deepseek_reasoning_effort_tier(),
            );
            assert!(kwargs.is_none());
            let current = json!({"tools":v["tools"],"thinking":thinking,"kwargs":kwargs});
            if let Some(prior) = &controls_prior {
                assert_eq!(prior, &current);
            }
            controls_prior = Some(current);
            let full = render_chat_prompt_or_400_with_generation_prompt(
                template,
                &req.messages,
                Some(tools),
                thinking,
                kwargs.as_ref(),
                true,
            )
            .unwrap();
            let p = tokenize_http_prompt(&tokenizer, &full).unwrap();
            assert!(!p.is_empty() && p.len() <= 262144 - 8192);
            if turn == 0 {
                let s = render_chat_prompt_or_400_with_generation_prompt(
                    template,
                    &req.messages,
                    Some(tools),
                    thinking,
                    kwargs.as_ref(),
                    false,
                )
                .unwrap();
                let s = tokenize_http_prompt(&tokenizer, &s).unwrap();
                assert!(!s.is_empty() && s.len() < p.len() && p.starts_with(&s));
                initial_anchors.insert(agent.clone(), s);
            }
            let receipt_bytes = t.receipt.read(1024 * 1024);
            let receipt: Value = serde_json::from_slice(&receipt_bytes).unwrap();
            assert!(
                transport_matches(&v, &receipt),
                "receipt/request transport drift"
            );
            assert_eq!(receipt["turn"], turn);
            assert_eq!(receipt["http_status"], 200);
            assert_eq!(receipt["request_sha256"], hash(&wire));
            assert_eq!(receipt["request_bytes"], wire.len());
            let response = t.response.read(RAW_LIMIT);
            assert_eq!(receipt["response_sha256"], hash(&response));
            assert_eq!(receipt["response_bytes"], response.len());
            let usage =
                completed_usage(&response, turn % 2 == 1, snap["model_id"].as_str().unwrap());
            assert_eq!(usage, receipt["usage"]);
            assert_eq!(usage["prompt_tokens"], p.len());
            assert!(usage["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .is_some_and(|n| n <= p.len() as u64));
            records.push(json!({"turn":turn,"request_sha256":hash(&wire),"response_sha256":hash(&response),"receipt_sha256":hash(&receipt_bytes),"rendered_sha256":hash(full.as_bytes()),"input_tokens_sha256":token_hash(&p),"prompt_tokens":p.len(),"usage":usage}));
            ids.push(p);
        }
        vectors.insert(agent.clone(), ids);
        observations.insert(agent.clone(), records);
    }
    let metadata = json!({"schema":"hf2q.gemma-matched-token-vectors.v1","arm":c.arm,"server_source":SERVER,"server_binary_sha256":BINARY,"source_units":source_units(),"exporter_source_sha256":hash(include_bytes!("handlers_gemma_matched_tokens_tests.rs")),"config_sha256":hash(&config_bytes),"observations":observations,"tokenizer_sha256":hash(&tokenizer_bytes),"template_sha256":hash(&template_bytes),"readiness_sha256":hash(&readiness_bytes),"controller_sha256":hash(&controller_bytes),"approval_sha256":hash(&approval_bytes),"environment_sha256":hash(&env_bytes),"build_sha256":hash(&build_bytes),"model_inference_performed":false,"prefix_qualified":false,"performance_qualified":false});
    let result = Export {
        metadata,
        vectors: &vectors,
        initial_anchors: &initial_anchors,
    };
    let output = output_path(&c.output).expect("safe private exporter output");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(output)
        .unwrap();
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer(&mut writer, &result).unwrap();
    writer.write_all(b"\n").unwrap();
    writer.flush().unwrap();
    println!("GEMMA_MATCHED_EXPORT_COMPLETE agents=4 turns=28 neutral=true prefix_qualified=false model_inference=false");
}
#[test]
fn gemma_matched_export_arm_is_neutral_but_binding_strict() {
    assert!(arm_matches(
        "OFF",
        &json!({}),
        &json!({"benchmark_profile":"gemma-matched-v1","experimental_gemma_full_input_checkpoint":false})
    ));
    let on = json!({"HF2Q_GEMMA_FULL_INPUT_CHECKPOINT":"1","HF2Q_UNSAFE_EXPERIMENTS":"1"});
    let approval = json!({"benchmark_profile":"gemma-matched-v1","experimental_gemma_full_input_checkpoint":true});
    assert!(arm_matches("ON", &on, &approval));
    assert!(!arm_matches("OFF", &on, &approval));
    assert!(!arm_matches("ON", &json!({}), &approval));
    assert!(!arm_matches("unknown", &on, &approval));
}
#[test]
fn gemma_matched_export_strips_only_two_exact_test_declarations() {
    let prefix = "production\n";
    assert_eq!(
        production_handlers(&format!("{prefix}{NEUTRAL_DECL}{FAMILY_DECL}")),
        Some(prefix)
    );
    assert_eq!(
        production_handlers(&format!("{prefix}{FAMILY_DECL}{NEUTRAL_DECL}")),
        Some(prefix)
    );
    assert!(production_handlers(&format!("{prefix}{NEUTRAL_DECL}{NEUTRAL_DECL}")).is_none());
    assert!(production_handlers(prefix).is_none());
}
#[test]
fn gemma_matched_export_refuses_changed_wire_controls() {
    let v = json!({"model":"Gemma","messages":[],"tools":[],"tool_choice":"auto","max_completion_tokens":8192,"logprobs":false,"hf2q_overflow_policy":"reject","stream":false});
    assert!(request_shape(&v, 0, "Gemma"));
    for (key, value) in [
        ("temperature", json!(0)),
        ("max_completion_tokens", json!(512)),
        ("stream", json!(true)),
        ("benchmark_id", json!("hidden")),
    ] {
        let mut bad = v.clone();
        bad[key] = value;
        assert!(!request_shape(&bad, 0, "Gemma"));
    }
}
#[test]
fn gemma_matched_export_completed_response_is_not_an_unverified_receipt() {
    let valid=json!({"model":"Gemma","choices":[{"index":0,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4}}).to_string();
    assert_eq!(
        completed_usage(valid.as_bytes(), false, "Gemma")["prompt_tokens"],
        4
    );
    let bad = valid.replace("tool_calls", "invalid");
    assert!(std::panic::catch_unwind(|| completed_usage(bad.as_bytes(), false, "Gemma")).is_err());
    let incomplete =
        "data: {\"model\":\"Gemma\",\"choices\":[],\"usage\":{\"prompt_tokens\":4}}\n\n";
    assert!(
        std::panic::catch_unwind(|| completed_usage(incomplete.as_bytes(), true, "Gemma")).is_err()
    );
}

#[test]
fn gemma_matched_export_output_rejects_relative_git_or_alias_parent() {
    assert!(output_path(Path::new("relative.json")).is_err());
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    assert_eq!(
        output_path(&root.join("new.json")).unwrap(),
        root.join("new.json")
    );
    let git = root.join("repo");
    std::fs::create_dir(&git).unwrap();
    std::fs::write(git.join(".git"), b"gitdir: owned-fixture").unwrap();
    let nested = git.join("nested");
    std::fs::create_dir(&nested).unwrap();
    assert!(output_path(&nested.join("new.json")).is_err());
    let alias = root.join("alias");
    std::os::unix::fs::symlink(&root, &alias).unwrap();
    assert!(output_path(&alias.join("new.json")).is_err());
}
#[test]
fn gemma_matched_export_transport_is_actual_boolean_not_omission() {
    assert!(transport_matches(
        &json!({"stream":true}),
        &json!({"stream":true})
    ));
    assert!(!transport_matches(
        &json!({"stream":true}),
        &json!({"stream":false})
    ));
    assert!(!transport_matches(&json!({}), &json!({})));
}
