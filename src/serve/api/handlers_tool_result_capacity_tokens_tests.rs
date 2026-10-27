//! Post-unload eight-wire capacity proof. No engine/device/GGUF/HTTP execution.
use super::prefix_post_oracle_tests::{hash, semantic_hash};
use super::*;
use crate::serve::api::schema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};
const BASE: &str = "a84e5c7bbfe93f0ffed4ebecfa3dc0ec844fa70b";
const WIRE: usize = 262144;
const FILE: usize = 65536;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundFile {
    path: PathBuf,
    sha256: String,
}
impl BoundFile {
    fn read(&self, cap: usize) -> Vec<u8> {
        assert!(self.path.is_absolute() && self.path.canonicalize().unwrap() == self.path);
        let m = std::fs::symlink_metadata(&self.path).unwrap();
        assert!(m.is_file() && !m.file_type().is_symlink() && m.len() <= cap as u64);
        let b = std::fs::read(&self.path).unwrap();
        assert!(b.len() <= cap);
        assert_eq!(hash(&b), self.sha256);
        b
    }
    fn value(&self, cap: usize) -> Value {
        serde_json::from_slice(&self.read(cap)).unwrap()
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Turn {
    request: BoundFile,
    response: BoundFile,
    receipt: BoundFile,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Agent {
    turns: Vec<Turn>,
    summary: BoundFile,
    payload: BoundFile,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    family: String,
    model: String,
    server_source: String,
    server_binary_sha256: String,
    tokenizer: BoundFile,
    template: BoundFile,
    readiness: BoundFile,
    controller_receipt: BoundFile,
    client_summary: BoundFile,
    server_build: BoundFile,
    recipe_proof: BoundFile,
    client_source: BoundFile,
    tools_source: BoundFile,
    family_cache_evidence: Option<BoundFile>,
    agents: BTreeMap<String, Agent>,
    output_root: PathBuf,
}
// Exact cached metadata identities; no architecture approximation or fallback.
#[path = "tool_result_capacity_tokens/validation.rs"]
mod validation;
use validation::*;

#[path = "tool_result_capacity_tokens/controls.rs"]
mod controls;

fn source_units() -> Value {
    let handlers = super::capacity_fixture_source_tests::source_at_base(
        include_str!("handlers.rs"),
        "02424cbe589d41d26430ab9231cf2f7e5aea8b17bb9f4c6cec266cf66dfdc387",
    )
    .expect("exact frozen handlers body with known fixture declarations only");
    json!({"handlers":hash(handlers.as_bytes()),"engine":hash(include_bytes!("engine.rs")),
        "schema":hash(include_bytes!("schema.rs")),"registry":hash(include_bytes!("registry.rs")),
        "reasoning":hash(include_bytes!("reasoning_controls.rs")),"templates":hash(include_bytes!("../../core/chat_templates.rs")),
        "deepseek_encoding":hash(include_bytes!("../../core/deepseek_v4_encoding/mod.rs")),"deepseek_types":hash(include_bytes!("../../core/deepseek_v4_encoding/types.rs")),"deepseek_parser":hash(include_bytes!("../../core/deepseek_v4_encoding/parser.rs")),"serve":hash(include_bytes!("../mod.rs"))})
}
fn function_hash(text: &str, name: &str) -> String {
    let start = text
        .find(&format!("fn {name}("))
        .expect("frozen recipe function");
    let rest = &text[start..];
    let end = rest.find("\n}").expect("frozen function terminator") + 2;
    hash(rest[..end].as_bytes())
}
fn recipe_units() -> Value {
    let handlers = include_str!("handlers.rs");
    let engine = include_str!("engine.rs");
    let mut units = serde_json::Map::new();
    for name in [
        "resolve_api_enable_thinking",
        "resolve_api_template_kwargs",
        "tokenize_http_prompt",
        "render_chat_prompt_or_400_with_generation_prompt",
    ] {
        units.insert(name.into(), json!(function_hash(handlers, name)));
    }
    for name in [
        "render_chat_prompt_with_tools_generation_prompt",
        "render_deepseek_v4_prompt",
        "render_deepseek_v4_prompt_with_serialized_tools",
    ] {
        units.insert(name.into(), json!(function_hash(engine, name)));
    }
    units.insert(
        "template_supports_enable_thinking".into(),
        json!(function_hash(
            include_str!("../mod.rs"),
            "template_supports_enable_thinking"
        )),
    );
    for key in [
        "schema",
        "reasoning",
        "templates",
        "deepseek_encoding",
        "deepseek_types",
        "deepseek_parser",
    ] {
        units.insert(key.into(), source_units()[key].clone());
    }
    Value::Object(units)
}
const EXPECTED_UNITS: &str = r#"{"handlers":"02424cbe589d41d26430ab9231cf2f7e5aea8b17bb9f4c6cec266cf66dfdc387","engine":"44c5f801b0bd1718b8e6e0f9b8cdd40d9d756b831ea4b00541aa600f864ecb48","schema":"769470235c7e5799a39e4e01ec4dc0faf0e9974952462fce2fc9227312bdbefd","registry":"9e67aa746f3caba7e54c1720046d5e1f59619637cbbd103ed02ced3ed7f73abd","reasoning":"8fbc94e337add533f0c1db39ebc3d0373fa7befd3cd3a82371ed0cd677253183","templates":"9a54dcac15b0ab40a5351f5c58837d9df9928db32a95980bc30ec38f68bf0438","deepseek_encoding":"ff466c69c46244a30174a1f636c6f0e0b6622ee8e2edae4d9315b85c5b6607dd","deepseek_types":"4ecbe3ca1470ba23bc33746494c31a15d68587b779985370987bb2438c059ca9","deepseek_parser":"f5ee2209702e588240e28df866bc0b856ca109a65d34d80b9223d7370ec6db45","serve":"e8d0409d13cf64268e890dd8bf76420f3e1644a5abf6468fe62660e49c7ccae6"}"#;
fn token_hash(ids: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    hex::encode(h.finalize())
}
#[derive(Serialize)]
struct Output<'a> {
    #[serde(flatten)]
    metadata: Value,
    vectors: &'a BTreeMap<String, Vec<Vec<u32>>>,
}
#[test]
#[ignore = "post-unload actual four-agent two-turn tool-result proof; no device/model"]
fn tool_result_capacity_exports_eight_actual_input_vectors() {
    let path = PathBuf::from(std::env::var("HF2Q_TOOL_RESULT_CAPACITY_TOKEN_CONFIG").unwrap());
    assert!(path.is_absolute() && path.canonicalize().unwrap() == path);
    let meta = std::fs::symlink_metadata(&path).unwrap();
    assert!(meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= 2 * 1024 * 1024);
    let config_bytes = std::fs::read(path).unwrap();
    assert!(config_bytes.len() <= 2 * 1024 * 1024);
    let c: Config = serde_json::from_slice(&config_bytes).unwrap();
    let compiled_sha =
        option_env!("HF2Q_BUILD_GIT_SHA").expect("independent compiled source identity required");
    assert!(compiled_sha.len() == 40 && compiled_sha.bytes().all(|b| b.is_ascii_hexdigit()));
    root(&c.output_root).unwrap();
    assert_eq!(
        source_units(),
        serde_json::from_str::<Value>(EXPECTED_UNITS).unwrap()
    );
    let (model, arch, tk_sha, tp_sha) = selected(&c.family).expect("explicit family required");
    assert_eq!(c.model, model);
    let scope = cache_scope(&c.family).unwrap();
    assert!(c.server_source.len() == 40 && c.server_source.bytes().all(|b| b.is_ascii_hexdigit()));
    assert!(
        c.server_binary_sha256.len() == 64
            && c.server_binary_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
    );
    let recipe = c.recipe_proof.value(1024 * 1024);
    assert_eq!(recipe["independently_reviewed"], true);
    assert_eq!(recipe["helper_base"], BASE);
    assert_eq!(recipe["server_source"], c.server_source);
    assert_eq!(recipe["family"], c.family);
    assert_eq!(recipe["helper_units"], source_units());
    assert_eq!(recipe["production_recipe_equivalent"], true);
    assert_eq!(recipe["cache_scope"], format!("{scope:?}"));
    if scope == CacheScope::DeepseekLiveOrRecovery {
        let evidence = c
            .family_cache_evidence
            .as_ref()
            .expect("reused native DS continuation proof required")
            .read(2 * 1024 * 1024);
        assert!(!evidence.is_empty());
        assert_eq!(recipe["deepseek_cache_evidence_sha256"], hash(&evidence));
        assert_eq!(recipe["deepseek_cache_evidence_reused_only"], true);
    }
    assert_eq!(
        recipe["server_recipe_units"],
        recipe_units(),
        "independently frozen actual server recipe must equal compiled helper recipe"
    );
    let client_bytes = c.client_source.read(1024 * 1024);
    let tool_bytes = c.tools_source.read(1024 * 1024);
    assert_eq!(recipe["capacity_client_sha256"], hash(&client_bytes));
    assert_eq!(recipe["tools_source_sha256"], hash(&tool_bytes));
    let build = c.server_build.value(1024 * 1024);
    assert_eq!(build["source_commit"], c.server_source);
    assert_eq!(build["binary_sha256"], c.server_binary_sha256);
    assert_eq!(build["source_clean"], true);
    assert_eq!(build["exit"], 0);
    let ready = c.readiness.value(1024 * 1024);
    assert_eq!(ready["health"]["status"], "ok");
    assert_eq!(ready["health"]["model"], model);
    assert_eq!(ready["health"]["context_length"], 262144);
    assert_eq!(ready["runtime"]["serve"]["max_slots"], 4);
    let snap = &ready["runtime"]["measurement_snapshot"];
    assert_eq!(snap["model_id"], model);
    assert_eq!(snap["architecture"], arch);
    let controller = c.controller_receipt.value(2 * 1024 * 1024);
    for key in ["protocol_complete", "final_verified", "cleanup_proved"] {
        assert_eq!(controller[key], true);
    }
    assert!(controller.get("failure").is_none());
    assert_eq!(controller["server_pid"], snap["process_pid"]);
    let aggregate = c.client_summary.value(4 * 1024 * 1024);
    assert_eq!(aggregate["family"], c.family);
    assert_eq!(aggregate["slots"], 4);
    assert_eq!(aggregate["server_pid"], controller["server_pid"]);
    assert_eq!(
        aggregate["status"],
        "CompleteCapacityProtocolAwaitingPostOracle"
    );
    assert_eq!(aggregate["cleanup"], true);
    let tb = c.tokenizer.read(64 * 1024 * 1024);
    assert_eq!(hash(&tb), tk_sha);
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tb).unwrap();
    assert!(tokenizer.get_truncation().is_none() && tokenizer.get_padding().is_none());
    assert_eq!(
        snap["tokenizer_sha256"],
        semantic_hash(serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap())
    );
    let template_bytes = c.template.read(1024 * 1024);
    assert_eq!(hash(&template_bytes), tp_sha);
    let template = std::str::from_utf8(&template_bytes).unwrap();
    assert_eq!(
        snap["template_sha256"],
        semantic_hash(json!({"source":"GgufEmbedded","template":template}))
    );
    crate::core::chat_templates::validate_tool_chat_template(arch, template).unwrap();
    let reg = registry::find_for_architecture(arch).unwrap();
    assert_eq!(
        reg.family,
        if c.family.starts_with("qwen") {
            "qwen35"
        } else {
            c.family.as_str()
        }
    );
    assert_eq!(
        c.agents.keys().map(String::as_str).collect::<Vec<_>>(),
        vec!["agent-0", "agent-1", "agent-2", "agent-3"]
    );
    let (mut vectors, mut observations) = (BTreeMap::new(), Vec::new());
    for (i, (agent, input)) in c.agents.iter().enumerate() {
        assert_eq!(input.turns.len(), 2);
        let payload_bytes = input.payload.read(FILE);
        assert_eq!(payload_bytes.len(), FILE);
        let payload = std::str::from_utf8(&payload_bytes).unwrap();
        assert!(
            payload.starts_with(&format!("// TOOL_RESULT_{i}; agent-{i}: inert file data\n"))
                && payload.ends_with(&format!("\n// END_TOOL_RESULT_{i}\n"))
        );
        for other in 0..4 {
            if other != i {
                assert!(!payload.contains(&format!("TOOL_RESULT_{other}")));
            }
        }
        let summary = input.summary.value(4 * 1024 * 1024);
        assert_eq!(summary["index"], i);
        assert_eq!(
            summary["status"],
            "CompleteCapacityProtocolAwaitingPostOracle"
        );
        assert_eq!(summary["cleanup"], true);
        assert_eq!(aggregate["agents"][i], summary);
        let read = &summary["actual_read"];
        assert_eq!(read["status"], "Complete");
        assert_eq!(read["cleanup"], true);
        assert_eq!(read["name"], "read_file");
        assert_eq!(read["arguments"], json!({"path":"src/ports.rs"}));
        assert_eq!(read["result"], payload);
        assert_eq!(read["source_sha256"], hash(&payload_bytes));
        assert_eq!(
            summary["fixture_preparation"]["receipt"]["source_sha256"],
            hash(&payload_bytes)
        );
        let (mut ps, mut requests, mut assistants) =
            (Vec::<Vec<u32>>::new(), Vec::new(), Vec::new());
        for (stage, turn) in input.turns.iter().enumerate() {
            let wire = turn.request.read(WIRE);
            let v: Value = serde_json::from_slice(&wire).unwrap();
            assert!(shape(&v, &c.family, model, i, stage));
            let receipt = turn.receipt.value(1024 * 1024);
            assert_eq!(receipt["http_status"], 200);
            assert_eq!(receipt["request_sha256"], hash(&wire));
            assert_eq!(receipt["request_bytes"], wire.len());
            assert_eq!(receipt["stream"], v["stream"]);
            assert_eq!(summary["turns"][stage]["capacity_stage"], stage);
            assert_eq!(
                summary["turns"][stage]["oracle_turn"],
                if stage == 0 { 0 } else { 6 }
            );
            assert_eq!(summary["turns"][stage]["receipt"], receipt);
            let raw = turn.response.read(32 * 1024 * 1024);
            assert_eq!(receipt["response_sha256"], hash(&raw));
            assert_eq!(receipt["response_bytes"], raw.len());
            let (assistant, u) = response(&raw, v["stream"].as_bool().unwrap(), model, stage);
            assert_eq!(u, receipt["usage"]);
            let req: schema::ChatCompletionRequest = serde_json::from_value(v.clone()).unwrap();
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
            assert_eq!(
                v["tools"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|t| t["function"]["name"].as_str().unwrap())
                    .collect::<std::collections::BTreeSet<_>>(),
                ["read_file", "run_tests", "write_file"]
                    .into_iter()
                    .collect()
            );
            assert!(v["tools"]
                .as_array()
                .unwrap()
                .iter()
                .all(|t| t["function"]["strict"] == false));
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
            if c.family.starts_with("qwen") {
                assert!(thinking);
                assert_eq!(
                    kwargs,
                    Some(serde_json::from_value(json!({"preserve_thinking":true})).unwrap())
                );
            } else if c.family == "gemma4" {
                assert!(!thinking);
                assert!(kwargs.is_none());
            } else {
                assert!(thinking);
                assert_eq!(
                    kwargs,
                    Some(serde_json::from_value(json!({"reasoning_effort":"max"})).unwrap())
                );
            }
            let rendered = render_chat_prompt_or_400_with_generation_prompt(
                template,
                &req.messages,
                Some(tools),
                thinking,
                kwargs.as_ref(),
                true,
            )
            .unwrap();
            let p = tokenize_http_prompt(&tokenizer, &rendered).unwrap();
            let cached = accept_tokens(
                &p,
                ps.last().map(Vec::as_slice),
                &u,
                if c.family.starts_with("qwen") { 64 } else { 0 },
                scope,
            )
            .unwrap();
            observations.push(json!({"agent":agent,"stage":stage,"request_sha256":hash(&wire),"response_sha256":hash(&raw),"tokens":p.len(),"token_ids_sha256_le_u32":token_hash(&p),"rendered_sha256":hash(rendered.as_bytes()),"usage":u,"cached_tokens":cached,"cache_bound_scope":format!("{scope:?}"),"P0_P1_lcp":ps.last().map(|old|old.iter().zip(&p).take_while(|(a,b)|a==b).count()),"thinking":thinking,"kwargs":kwargs}));
            ps.push(p);
            requests.push(v);
            assistants.push(assistant);
        }
        flow(&requests[0], &requests[1], &assistants[0], payload).unwrap();
        assert!(assistants[1]
            .get("tool_calls")
            .is_none_or(|v| v.is_null() || v.as_array().is_some_and(Vec::is_empty)));
        assert_eq!(
            assistants[1]["content"].as_str().unwrap().trim(),
            format!("CAPACITY_{i}")
        );
        vectors.insert(agent.clone(), ps);
    }
    let metadata = json!({"status":"NeutralEightWireToolResultTokenProof","family":c.family,"model":model,"server_source":c.server_source,"server_binary_sha256":c.server_binary_sha256,"helper_base":BASE,"compiled_build_git_sha":compiled_sha,"config_sha256":hash(&config_bytes),"helper_units":source_units(),"recipe_units":recipe_units(),"capacity_client_sha256":hash(&client_bytes),"tools_source_sha256":hash(&tool_bytes),"helper_source_sha256":hash(include_bytes!("handlers_tool_result_capacity_tokens_tests.rs")),"recipe_proof_sha256":c.recipe_proof.sha256,"observations":observations,"full_prefix_reuse_qualified":false,"seven_wire_oracle_replaced":false,"runtime_admission":false,"model_inference_performed":false});
    let path = c.output_root.join("proof.json");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .unwrap();
    serde_json::to_writer_pretty(
        &mut file,
        &Output {
            metadata,
            vectors: &vectors,
        },
    )
    .unwrap();
    file.write_all(b"\n").unwrap();
    println!("TOOL_RESULT_CAPACITY_TOKENS_COMPLETE agents=4 turns=8 real_result_bytes=65536 model_inference=false full_prefix_qualified=false");
}
