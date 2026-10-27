//! Metadata-only largest ADMITTED Qwen3.8 input fixture. No engine/device or HTTP.
//! Uses the production Qwen 64-row safety reserve; no model payload or engine.
//! Dense qwen35 architecture is explicit, not a MoE architecture alias.
use super::prefix_post_oracle_tests::{hash, semantic_hash};
use super::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::{io::Write, path::PathBuf};
const BASE: &str = "a84e5c7bbfe93f0ffed4ebecfa3dc0ec844fa70b";
const MODEL: &str = "Qwen3.8 27B";
const MODEL_SHA: &str = "cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7";
const MODEL_BYTES: u64 = 16810714976;
const CONTEXT: usize = 262144;
const OUTPUT: usize = 8192;
// Both source-hash-bound guards: engine::validate_qwen35_generation_request and
// engine_qwen35::Qwen35PrefillState::begin require P+graft+output+64 <= C.
const QWEN_SEQUENCE_SLACK: usize = 64;
const GRAFT: usize = 0;
const TARGET: usize = CONTEXT - OUTPUT - QWEN_SEQUENCE_SLACK - GRAFT;
fn admitted_input(prompt: usize) -> bool {
    prompt
        .checked_add(GRAFT)
        .and_then(|n| n.checked_add(OUTPUT))
        .and_then(|n| n.checked_add(QWEN_SEQUENCE_SLACK))
        .is_some_and(|n| n <= CONTEXT)
}
const BODY_LIMIT: usize = 8 * 1024 * 1024;
const UNIT: &str = " a";
const DECL:&str="\n#[cfg(test)]\n#[path = \"handlers_qwen38_largest_prompt_tests.rs\"]\nmod qwen38_largest_prompt_tests;\n";
const TOKENIZER: &str = "90ff2bcbb15ed3e13f779a72000af82fde72842c3a14f04ac0da415071e6fa4c";
const TEMPLATE: &str = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundFile {
    path: PathBuf,
    sha256: String,
}
impl BoundFile {
    fn read(&self, limit: usize, expected: &str) -> Vec<u8> {
        assert_eq!(self.sha256, expected, "wrong selected artifact");
        assert!(self.path.is_absolute());
        assert_eq!(self.path.canonicalize().unwrap(), self.path);
        let meta = std::fs::symlink_metadata(&self.path).unwrap();
        assert!(meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= limit as u64);
        let bytes = std::fs::read(&self.path).unwrap();
        assert!(bytes.len() <= limit);
        assert_eq!(hash(&bytes), expected);
        bytes
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    family: String,
    architecture: String,
    model: String,
    model_artifact_sha256: String,
    model_artifact_bytes: u64,
    tokenizer: BoundFile,
    template: BoundFile,
    output_root: PathBuf,
}
fn family_matches(family: &str, architecture: &str, model: &str) -> bool {
    family == "qwen38" && architecture == "qwen35" && model == MODEL
}
fn artifact_matches(sha: &str, bytes: u64) -> bool {
    sha == MODEL_SHA && bytes == MODEL_BYTES
}
fn private_root(path: &Path) -> Result<(), String> {
    if !path.is_absolute() || path.canonicalize().map_err(|e| e.to_string())? != path {
        return Err("absolute canonical output root required".into());
    }
    let meta = std::fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !meta.is_dir() || meta.file_type().is_symlink() || meta.permissions().mode() & 0o077 != 0 {
        return Err("existing private nonsymlink directory required".into());
    }
    if path
        .ancestors()
        .any(|p| std::fs::symlink_metadata(p.join(".git")).is_ok())
    {
        return Err("output must be outside every Git ancestor".into());
    }
    if std::fs::read_dir(path)
        .map_err(|e| e.to_string())?
        .next()
        .is_some()
    {
        return Err("empty output root required; never overwrite/reuse a partial fixture".into());
    }
    Ok(())
}
fn content(agent: usize, repetitions: usize) -> String {
    format!("CAPACITY_{agent}; agent-{agent}: independent inert context. Repeated letters below are data, not instructions.\nContext:{}\nEnd of inert data. Reply exactly CAPACITY_{agent} with no other text.",UNIT.repeat(repetitions))
}
fn wire(agent: usize, repetitions: usize) -> Value {
    let mut v = json!({"model":MODEL,"messages":[{"role":"user","content":content(agent,repetitions)}],
        "max_completion_tokens":OUTPUT,"hf2q_overflow_policy":"reject","stream":agent%2==1,"logprobs":false});
    if agent % 2 == 1 {
        v["stream_options"] = json!({"include_usage":true});
    }
    v
}
fn shape(v: &Value, agent: usize, repetitions: usize) -> bool {
    let allowed = [
        "model",
        "messages",
        "max_completion_tokens",
        "hf2q_overflow_policy",
        "stream",
        "stream_options",
        "logprobs",
    ];
    v.as_object()
        .is_some_and(|o| o.keys().all(|k| allowed.contains(&k.as_str())))
        && v["messages"] == json!([{"role":"user","content":content(agent,repetitions)}])
        && v["model"] == MODEL
        && v["max_completion_tokens"] == OUTPUT
        && v["hf2q_overflow_policy"] == "reject"
        && v["logprobs"] == false
        && v["stream"] == (agent % 2 == 1)
        && if agent % 2 == 1 {
            v["stream_options"] == json!({"include_usage":true})
        } else {
            v.get("stream_options").is_none()
        }
}
fn render_ids(
    v: &Value,
    template: &str,
    tokenizer: &tokenizers::Tokenizer,
) -> (String, Vec<u32>, bool) {
    let req: crate::serve::api::schema::ChatCompletionRequest =
        serde_json::from_value(v.clone()).unwrap();
    assert_eq!(req.messages.len(), 1);
    assert!(req.tools.is_none());
    let reg = registry::find_for_architecture("qwen35").unwrap();
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
    assert_eq!(reg.family, "qwen35");
    assert!(
        thinking,
        "selected Q38 template default must remain thinking-enabled"
    );
    assert_eq!(
        kwargs,
        Some(serde_json::from_value(json!({"preserve_thinking":true})).unwrap())
    );
    let rendered = render_chat_prompt_or_400_with_generation_prompt(
        template,
        &req.messages,
        None,
        thinking,
        kwargs.as_ref(),
        true,
    )
    .unwrap();
    let ids = tokenize_http_prompt(tokenizer, &rendered).unwrap();
    (rendered, ids, thinking)
}
/// At most 20 binary steps plus bounded local search; never assume a unit's
/// token cost. The caller uses the ACTUAL renderer/tokenizer for every count.
fn find_repetitions(target: usize, mut count: impl FnMut(usize) -> usize) -> Result<usize, String> {
    if target == 0 || target > TARGET {
        return Err("bounded target required".into());
    }
    let (mut lo, mut hi) = (0, target);
    let (mut low_len, mut high_len) = (count(lo), count(hi));
    if low_len == target {
        return Ok(lo);
    }
    if high_len == target {
        return Ok(hi);
    }
    if low_len > target || high_len < target {
        return Err("fixture does not bracket target".into());
    }
    for _ in 0..20 {
        if hi - lo <= 1 {
            break;
        }
        let mid = lo + (hi - lo) / 2;
        let n = count(mid);
        if n < low_len || n > high_len {
            return Err("nonmonotone fixture; no guessed tokenization".into());
        }
        if n == target {
            return Ok(mid);
        }
        if n < target {
            lo = mid;
            low_len = n;
        } else {
            hi = mid;
            high_len = n;
        }
    }
    if hi - lo > 1 {
        return Err("search budget exhausted".into());
    }
    // Strictly bounded fallback; final exact replay is mandatory regardless.
    for n in lo.saturating_sub(4)..=hi.saturating_add(4).min(target) {
        if count(n) == target {
            return Ok(n);
        }
    }
    Err("no exact target found by bounded search".into())
}
fn token_hash(ids: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    hex::encode(h.finalize())
}
fn write_new(path: &Path, bytes: &[u8]) {
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .unwrap();
    f.write_all(bytes).unwrap();
    f.flush().unwrap();
}
fn source_units() -> Value {
    let handlers = include_str!("handlers.rs")
        .strip_suffix(DECL)
        .expect("one exact appended cfg(test) declaration");
    json!({"engine":hash(include_bytes!("engine.rs")),"qwen_engine":hash(include_bytes!("engine_qwen35.rs")),"handlers":hash(handlers.as_bytes()),
        "schema":hash(include_bytes!("schema.rs")),"registry":hash(include_bytes!("registry.rs")),
        "reasoning":hash(include_bytes!("reasoning_controls.rs")),"templates":hash(include_bytes!("../../core/chat_templates.rs")),
        "lock":hash(include_bytes!("../../../Cargo.lock"))})
}
fn expected_units() -> Value {
    json!({"engine":"44c5f801b0bd1718b8e6e0f9b8cdd40d9d756b831ea4b00541aa600f864ecb48","qwen_engine":"39eef94e81b911bd7f8b00a1d50e45ca955bcc3bd06cbe2c7eaedd9dd8888a3a","handlers":"02424cbe589d41d26430ab9231cf2f7e5aea8b17bb9f4c6cec266cf66dfdc387","schema":"769470235c7e5799a39e4e01ec4dc0faf0e9974952462fce2fc9227312bdbefd","registry":"9e67aa746f3caba7e54c1720046d5e1f59619637cbbd103ed02ced3ed7f73abd","reasoning":"8fbc94e337add533f0c1db39ebc3d0373fa7befd3cd3a82371ed0cd677253183","templates":"9a54dcac15b0ab40a5351f5c58837d9df9928db32a95980bc30ec38f68bf0438","lock":"65730c8a4f0b013ff33d85ebcb85514ee29eb786d77673e629d8e9be306d67b1"})
}
#[test]
#[ignore = "metadata-only exact largest-prompt wire export; no model/device/server"]
fn qwen38_largest_prompt_exports_four_exact_production_inputs() {
    let compiled_sha = option_env!("HF2Q_BUILD_GIT_SHA")
        .expect("explicit frozen compiled source identity required");
    assert!(
        compiled_sha.len() == 40
            && compiled_sha
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    );
    let path = PathBuf::from(std::env::var("HF2Q_QWEN38_LARGEST_PROMPT_CONFIG").unwrap());
    assert!(path.is_absolute());
    assert_eq!(path.canonicalize().unwrap(), path);
    let meta = std::fs::symlink_metadata(&path).unwrap();
    assert!(meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= 16384);
    let config_bytes = std::fs::read(&path).unwrap();
    assert!(config_bytes.len() <= 16384);
    let c: Config = serde_json::from_slice(&config_bytes).unwrap();
    assert!(family_matches(&c.family, &c.architecture, &c.model));
    assert!(artifact_matches(
        &c.model_artifact_sha256,
        c.model_artifact_bytes
    ));
    private_root(&c.output_root).unwrap();
    assert_eq!(
        source_units(),
        expected_units(),
        "production units must match frozen a84 base"
    );
    let tb = c.tokenizer.read(64 * 1024 * 1024, TOKENIZER);
    let template_bytes = c.template.read(1024 * 1024, TEMPLATE);
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tb).unwrap();
    assert!(tokenizer.get_truncation().is_none() && tokenizer.get_padding().is_none());
    let template = std::str::from_utf8(&template_bytes).unwrap();
    crate::core::chat_templates::validate_tool_chat_template("qwen35", template).unwrap();
    let mut records = Vec::new();
    let mut client_wires = Vec::new();
    let mut vectors = Vec::<Vec<u32>>::new();
    for agent in 0..4 {
        let repetitions = find_repetitions(TARGET, |n| {
            render_ids(&wire(agent, n), template, &tokenizer).1.len()
        })
        .unwrap();
        let value = wire(agent, repetitions);
        assert!(shape(&value, agent, repetitions));
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(bytes.len() < BODY_LIMIT);
        let decoded: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(shape(&decoded, agent, repetitions));
        let (rendered, ids, thinking) = render_ids(&decoded, template, &tokenizer);
        assert_eq!(ids.len(), TARGET);
        assert!(admitted_input(ids.len()));
        assert_eq!(
            ids.len()
                .checked_add(OUTPUT)
                .and_then(|n| n.checked_add(QWEN_SEQUENCE_SLACK)),
            Some(CONTEXT)
        );
        assert_eq!(
            ids,
            render_ids(&decoded, template, &tokenizer).1,
            "actual re-encoding drift"
        );
        let request_name = format!("agent-{agent}-request.json");
        let tokens_name = format!("agent-{agent}-token-ids.json");
        write_new(&c.output_root.join(&request_name), &bytes);
        let token_bytes = serde_json::to_vec(&ids).unwrap();
        write_new(&c.output_root.join(&tokens_name), &token_bytes);
        client_wires.push(json!({"path":c.output_root.join(&request_name),"sha256":hash(&bytes),
            "agent_id":format!("agent-{agent}"),"prompt_tokens":ids.len(),"marker":format!("CAPACITY_{agent}")}));
        records.push(json!({"agent":agent,"request_file":request_name,"request_sha256":hash(&bytes),"request_bytes":bytes.len(),
            "token_file":tokens_name,"token_file_sha256":hash(&token_bytes),"token_ids_sha256_le_u32":token_hash(&ids),
            "prompt_tokens":ids.len(),"max_completion_tokens":OUTPUT,"context":CONTEXT,"qwen_sequence_slack":QWEN_SEQUENCE_SLACK,"graft_tokens":GRAFT,"repetitions":repetitions,
            "stream":agent%2==1,"tools":false,"thinking":thinking,"resolved_template_kwargs":{"preserve_thinking":true},"encode_add_special_tokens":false,
            "add_generation_prompt":true,"rendered_sha256":hash(rendered.as_bytes()),
            "render_prefix":rendered.chars().take(96).collect::<String>(),"bos_and_header_token_ids":&ids[..ids.len().min(24)]}));
        vectors.push(ids);
    }
    let mut lcps = Vec::new();
    for i in 0..4 {
        for j in i + 1..4 {
            let n = vectors[i]
                .iter()
                .zip(&vectors[j])
                .take_while(|(a, b)| a == b)
                .count();
            assert!(
                n < 128,
                "independent agent identity must break LCP near beginning"
            );
            lcps.push(json!({"agents":[i,j],"common_prefix_tokens":n}));
        }
    }
    let manifest = json!({"schema":"hf2q.qwen38-largest-prompt-wires.v1","model":MODEL,
        "context_tokens":CONTEXT,"qwen_sequence_slack":QWEN_SEQUENCE_SLACK,"prompt_tokens":TARGET,"max_completion_tokens":OUTPUT,"wires":client_wires});
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    write_new(&c.output_root.join("manifest.json"), &manifest_bytes);
    let metadata = json!({"schema":"hf2q.qwen38-largest-prompt-metadata.v1","family":"qwen38","architecture":"qwen35","model":MODEL,"model_artifact_sha256":MODEL_SHA,"model_artifact_bytes":MODEL_BYTES,
        "manifest_sha256":hash(&manifest_bytes),
        "production_source_base":BASE,"production_source_units":source_units(),"helper_source_sha256":hash(include_bytes!("handlers_qwen38_largest_prompt_tests.rs")),
        "compiled_build_git_sha":compiled_sha,"compiler_provenance":"requires independent frozen source/binary recipe; not self-attestation",
        "config_sha256":hash(&config_bytes),"tokenizer_raw_sha256":hash(&tb),"template_raw_sha256":hash(&template_bytes),
        "tokenizer_semantic_sha256":semantic_hash(serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap()),
        "template_semantic_sha256":semantic_hash(json!({"source":"GgufEmbedded","template":template})),
        "case":"corrected_largest_admitted_input","qwen_sequence_slack":QWEN_SEQUENCE_SLACK,"graft_tokens":GRAFT,"admission_basis":"production_qwen_guard_64_rows","context":CONTEXT,"prompt_tokens":TARGET,"max_completion_tokens":OUTPUT,"product_body_limit_bytes":BODY_LIMIT,
        "artifact_identity_basis":"declared cached F1 binding only; no GGUF payload opened by this metadata helper","model_inference_performed":false,"admission_qualified":false,"quality_qualified":false,"prefix_qualified":false,
        "cross_agent_lcps":lcps,"requests":records});
    write_new(
        &c.output_root.join("metadata.json"),
        &serde_json::to_vec_pretty(&metadata).unwrap(),
    );
    println!("QWEN38_LARGEST_PROMPT_METADATA_COMPLETE agents=4 prompt_tokens=253888 max_completion_tokens=8192 safety_slack=64 model_inference=false admission=false");
}
#[test]
fn qwen38_largest_prompt_config_is_exact_family_and_closed_schema() {
    let v = json!({"family":"qwen38","architecture":"qwen35","model":MODEL,"model_artifact_sha256":MODEL_SHA,"model_artifact_bytes":MODEL_BYTES,"tokenizer":{"path":"/unused/tokenizer","sha256":TOKENIZER},
        "template":{"path":"/unused/template","sha256":TEMPLATE},"output_root":"/unused/private"});
    let c: Config = serde_json::from_value(v.clone()).unwrap();
    assert!(family_matches(&c.family, &c.architecture, &c.model));
    assert!(artifact_matches(
        &c.model_artifact_sha256,
        c.model_artifact_bytes
    ));
    assert!(!artifact_matches("other", MODEL_BYTES));
    assert!(!artifact_matches(MODEL_SHA, MODEL_BYTES - 1));
    assert!(!family_matches("deepseek4", "deepseek4", MODEL));
    assert!(!family_matches("qwen36", "qwen35moe", MODEL));
    assert!(!family_matches("qwen38", "qwen35moe", MODEL));
    assert!(!family_matches("qwen38", "qwen35", "Qwen3.6 35B A3B"));
    assert!(!family_matches("qwen38", "qwen35", "other"));
    let mut bad = v.clone();
    bad["prompt_tokens"] = json!(1);
    assert!(serde_json::from_value::<Config>(bad).is_err());
    let mut bad = v;
    bad["tokenizer"]["fallback"] = json!(true);
    assert!(serde_json::from_value::<Config>(bad).is_err());
}
#[test]
fn qwen38_largest_prompt_request_preserves_exact_boundary_and_payload() {
    assert_eq!(TARGET, 253888);
    assert_eq!(
        TARGET
            .checked_add(OUTPUT)
            .and_then(|n| n.checked_add(QWEN_SEQUENCE_SLACK)),
        Some(CONTEXT)
    );
    assert!(admitted_input(TARGET));
    assert!(!admitted_input(TARGET + 1));
    assert!(!admitted_input(253952));
    assert!(!admitted_input(usize::MAX));
    for agent in 0..4 {
        let v = wire(agent, 2);
        assert!(shape(&v, agent, 2));
        assert!(v["messages"][0]["content"]
            .as_str()
            .unwrap()
            .ends_with(&format!(
                "Reply exactly CAPACITY_{agent} with no other text."
            )));
        assert!(v["messages"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with(&format!("CAPACITY_{agent}; agent-{agent}:")));
        for (key, val) in [
            ("model", json!("other")),
            ("max_completion_tokens", json!(1)),
            ("tools", json!([])),
            ("hf2q_overflow_policy", json!("summarize")),
            ("stream", json!(agent % 2 == 0)),
        ] {
            let mut bad = v.clone();
            bad[key] = val;
            assert!(!shape(&bad, agent, 2));
        }
        let mut bad = v;
        bad["messages"][0]["content"] = json!("different payload");
        assert!(!shape(&bad, agent, 2));
    }
}
#[test]
fn qwen38_largest_prompt_search_is_bounded_and_cannot_bless_nonexact_counts() {
    let mut calls = 0;
    let found = find_repetitions(100, |n| {
        calls += 1;
        17 + n
    })
    .unwrap();
    assert_eq!(17 + found, 100);
    assert!(calls <= 32);
    assert!(find_repetitions(100, |n| 17 + 2 * n).is_err());
    assert!(find_repetitions(100, |_| 101).is_err());
    assert!(find_repetitions(100, |n| if n == 50 { 1000 } else { n + 1 }).is_err());
    assert!(find_repetitions(TARGET + 1, |_| panic!("reject before evaluation")).is_err());
}
#[test]
fn qwen38_largest_prompt_private_output_has_no_overwrite_or_git_alias() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
    private_root(&root).unwrap();
    assert!(private_root(Path::new("relative")).is_err());
    let output = root.join("new.json");
    write_new(&output, b"original");
    assert_eq!(
        std::fs::metadata(&output).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(std::panic::catch_unwind(|| write_new(&output, b"replacement")).is_err());
    assert_eq!(std::fs::read(&output).unwrap(), b"original");
    assert!(private_root(&root).is_err());
    let git = root.join("repo");
    std::fs::create_dir(&git).unwrap();
    std::fs::write(git.join(".git"), b"gitdir: fixture").unwrap();
    let nested = git.join("private");
    std::fs::create_dir(&nested).unwrap();
    std::fs::set_permissions(&nested, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(private_root(&nested).is_err());
    let alias = root.join("alias");
    std::os::unix::fs::symlink(&root, &alias).unwrap();
    assert!(private_root(&alias).is_err());
}
