//! Family-explicit experimental full-input replay. Strict generic oracle unchanged.
//! Engine/device-free; independent actual launch/quality review remains mandatory.
use super::prefix_post_oracle_tests::{hash, read, semantic_hash, BoundFile};
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
    response: BoundFile,
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
    capture_log: BoundFile,
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
fn gemma_full_input_v2_controls_do_not_inherit_other_family_kwargs() {
    let reg = registry::find_for_architecture("gemma4").unwrap();
    assert!(resolve_api_template_kwargs(Some(&reg), None, None).is_none());
    assert!(gemma_template("deepseek4", "GgufEmbedded", "x").is_err());
    assert!(gemma_template("gemma4", "NativeEncoding", "x").is_err());
    assert!(gemma_template("gemma4", "GgufEmbedded", "<|im_start|>").is_err());
}
#[test]
fn gemma_full_input_v2_build_identity_rejects_dirty_or_mixed_source() {
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
fn gemma4_full_input_seven_wire_post_oracle() {
    let bytes =
        std::fs::read(std::env::var("HF2Q_GEMMA_FULL_INPUT_POST_ORACLE_CONFIG").unwrap()).unwrap();
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
    assert_eq!(fc["codebook_bits"], 8);
    assert_eq!(approval["experimental_gemma_full_input_checkpoint"], true);
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
    assert!(super::engine::gemma_full_input::request_tag(true, true, &[1], &[1])
        .unwrap().is_some(), "replay requires explicit experimental stage + unsafe ack");
    let capture_bytes = read(&c.capture_log);
    let capture_text = std::str::from_utf8(&capture_bytes).unwrap();
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
        let response_bytes = read(&t.response);
        assert_eq!(receipt["response_sha256"], hash(&response_bytes));
        assert_eq!(receipt["response_bytes"].as_u64(), Some(response_bytes.len() as u64));
        assert_eq!(receipt["stream"], v["stream"]);
        let actual_usage = completed_usage(&response_bytes, req.stream.unwrap_or(false),
            snap["model_id"].as_str().unwrap());
        assert_eq!(actual_usage, receipt["usage"]);
        let last_tool = req.messages.last().is_some_and(|m| m.role == "tool");
        let tag = super::engine::gemma_full_input::request_tag(true, last_tool, &p, &s).unwrap();
        if let Some(tag) = &tag { tag.validate(&p).unwrap(); }
        let cached = accept_growth(&p, &s, prior.as_deref(), &actual_usage, tag.is_some())
            .unwrap_or_else(|why| panic!("turn {i}: {why}"));
        let anchor = if i == 0 { s.clone() } else { p.clone() };
        let slot = if i > 0 {
            Some(capture_slot(capture_text, p.len(), &token_hash(&p), c.server_slots)
                .expect("actual production capture len/digest/slot missing or ambiguous"))
        } else { None };
        observations.push(json!({"turn":i,"prompt_tokens":p.len(),"cue_less_tokens":s.len(),
            "accepted_anchor_tokens":anchor.len(),"prior_anchor_tokens":prior.as_ref().map(Vec::len),
            "cached_tokens":cached,"capture_slot":slot,"input_tokens_sha256":token_hash(&p),
            "request_sha256":hash(&wire),"response_sha256":hash(&response_bytes),
            "receipt_sha256":hash(&rb),"rendered_sha256":hash(full.as_bytes())}));
        prior = Some(anchor);
    }

    assert!(c.output.is_absolute());
    let parent = c.output.parent().unwrap().canonicalize().unwrap();
    assert!(parent.ancestors().all(|p| !p.join(".git").exists()));
    let output = json!({"schema":"hf2q.gemma4-full-input-post-oracle.v2","family":"gemma4","server_slots":c.server_slots,"full_prefix_reuse":"GemmaFullInputCheckpointSevenWireQualified","model_inference_performed":false,"model_qualified":false,"config_sha256":hash(&bytes),"tokenizer_sha256":hash(&tokenizer_bytes),"readiness_sha256":hash(&readiness_bytes),"controller_sha256":hash(&controller_bytes),"family_approval_sha256":hash(&approval_bytes),"server_build_receipt_sha256":hash(&build_bytes),"expected_server_source_commit":c.expected_server_source_commit,"expected_server_binary_sha256":c.expected_server_binary_sha256,"oracle_source_sha256":hash(include_bytes!("handlers_gemma_full_input_post_oracle_tests.rs")),"renderer_source_sha256":hash(include_bytes!("engine.rs")),"checkpoint_source_sha256":hash(include_bytes!("engine_gemma_full_input.rs")),"capture_log_sha256":hash(&capture_bytes),"generic_strict_oracle_qualified":false,"observations":observations});
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

fn token_hash(tokens: &[u32]) -> String {
    hash(&tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>())
}
// This suite contains growing conversations only. Identical-input P-1 retry
// belongs to a different test and NEVER supplies slack to this predicate.
fn accept_growth(p: &[u32], s: &[u32], prior: Option<&[u32]>, usage: &Value,
    eligible_full_tag: bool) -> Result<u64, String> {
    let fail = |why: &str| Err(why.to_owned());
    let Some(cached) = usage["prompt_tokens_details"]["cached_tokens"].as_u64() else {
        return fail("missing/integer cached usage");
    };
    if usage["prompt_tokens"].as_u64() != Some(p.len() as u64) || cached > p.len() as u64 {
        return fail("actual rendered usage mismatch");
    }
    match prior {
        None if !eligible_full_tag && !s.is_empty() && s.len() < p.len()
            && p.starts_with(s) && cached == 0 => Ok(cached),
        Some(a) if eligible_full_tag && !p.is_empty() && p == s && !a.is_empty()
            && p.len() > a.len() && p.starts_with(a) && cached >= a.len() as u64 => Ok(cached),
        _ => fail("unproved initial strict anchor or growing full-input edge"),
    }
}
fn completed_usage(bytes: &[u8], stream: bool, model: &str) -> Value {
    if !stream {
        let v: Value = serde_json::from_slice(bytes).unwrap();
        assert_eq!(v["model"], model);
        assert!(v.get("error").is_none(), "unary error body");
        assert_eq!(v["choices"].as_array().unwrap().len(), 1);
        assert_eq!(v["choices"][0]["index"].as_u64(), Some(0));
        assert!(matches!(v["choices"][0]["finish_reason"].as_str(), Some("stop" | "tool_calls" | "length")));
        assert!(v["usage"].is_object());
        return v["usage"].clone();
    }
    let text = std::str::from_utf8(bytes).unwrap();
    assert!(text.ends_with("\n\n") || text.ends_with("\r\n\r\n"));
    let mut usage = None;
    let mut done = false;
    let mut finished = 0;
    for line in text.lines().filter(|line| !line.is_empty()) {
        assert!(!done, "data after DONE");
        let data = line.strip_prefix("data: ").expect("SSE data-only wire");
        if data == "[DONE]" { done = true; continue; }
        let v: Value = serde_json::from_str(data).unwrap();
        assert_eq!(v["model"], model);
        assert!(v.get("error").is_none());
        for choice in v["choices"].as_array().unwrap() {
            assert_eq!(choice["index"], 0);
            if !choice["finish_reason"].is_null() {
                assert!(matches!(choice["finish_reason"].as_str(), Some("stop" | "tool_calls" | "length"))); finished += 1;
            }
        }
        if v["usage"].is_object() {
            assert!(usage.is_none(), "duplicate usage"); usage = Some(v["usage"].clone());
        }
    }
    assert!(done && finished == 1, "complete SSE finish and DONE required");
    usage.expect("actual SSE usage required")
}
fn capture_slot(text: &str, len: usize, digest: &str, slots: u32) -> Option<u32> {
    let field = |line: &str, name: &str| -> Option<String> {
        let pattern = format!(r"(?:^|\s){}=([^\s]+)", regex::escape(name));
        let regex = regex::Regex::new(&pattern).ok()?;
        let mut fields = regex.captures_iter(line);
        let value = fields.next()?.get(1)?.as_str().trim_matches('"').to_owned();
        if fields.next().is_some() { return None; }
        Some(value)
    };
    let ansi = regex::Regex::new(r"\x1b\[[0-9;]*m").unwrap();
    let plain = ansi.replace_all(text, "");
    let mut found = Vec::new();
    for line in plain.lines().filter(|l| l.contains("Gemma full-input checkpoint captured; publication pending")) {
        let actual_len = field(line, "prompt_tokens")?.parse::<usize>().ok()?;
        let actual_digest = field(line, "input_tokens_sha256")?;
        let slot = field(line, "slot")?.parse::<u32>().ok()?;
        if slot >= slots { return None; }
        if actual_len == len && actual_digest == digest { found.push(slot); }
    }
    // Capture alone is pending. Callers also require completed bound HTTP wire
    // and clean production build, whose reviewed route commits before seed emit.
    (found.len() == 1).then(|| found[0])
}
#[test]
fn gemma_full_input_v2_actual_growth_predicate_negative_controls() {
    let a = [1, 2]; let p = [1, 2, 3, 4];
    let usage = |n| json!({"prompt_tokens":4,"prompt_tokens_details":{"cached_tokens":n}});
    assert!(accept_growth(&p, &p[..3], None, &usage(0), false).is_ok());
    assert!(accept_growth(&p, &p, None, &usage(0), true).is_err());
    assert!(accept_growth(&p, &p, Some(&a), &usage(2), true).is_ok());
    for n in [0, 1, 5] { assert!(accept_growth(&p, &p, Some(&a), &usage(n), true).is_err()); }
    assert!(accept_growth(&p, &p, Some(&[1, 9]), &usage(2), true).is_err());
    assert!(accept_growth(&p, &p, Some(&a), &usage(2), false).is_err());
    assert!(accept_growth(&p, &p, Some(&p), &usage(3), true).is_err());
    assert!(accept_growth(&p, &p[..3], Some(&a), &usage(2), true).is_err());
    assert!(accept_growth(&p, &p, Some(&a), &json!({"prompt_tokens":3}), true).is_err());
    assert!(accept_growth(&p, &p, Some(&a), &json!({"prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":2}}), true).is_err());
    let growing = [1,2,3,4,5];
    assert!(accept_growth(&growing,&growing,Some(&p),&json!({"prompt_tokens":5,"prompt_tokens_details":{"cached_tokens":3}}),true).is_err());
}
#[test]
fn gemma_full_input_v2_capture_binding_negative_controls() {
    let line = "INFO Gemma full-input checkpoint captured; publication pending slot=2 prompt_tokens=4 input_tokens_sha256=abcd cpu_payload_bytes=99\n";
    assert_eq!(capture_slot(line,4,"abcd",4),Some(2));
    assert_eq!(capture_slot(line,4,"other",4),None);
    assert_eq!(capture_slot(line,5,"abcd",4),None);
    assert_eq!(capture_slot(line,4,"abcd",1),None);
    assert_eq!(capture_slot(&(line.to_owned()+line),4,"abcd",4),None);
    assert_eq!(capture_slot("",4,"abcd",4),None);
    for duplicate in ["slot=3", "prompt_tokens=4", "input_tokens_sha256=abcd"] {
        let malformed = format!("{} {duplicate}\n", line.trim_end());
        assert_eq!(capture_slot(&malformed,4,"abcd",4),None);
        assert_eq!(capture_slot(&(line.to_owned()+&malformed),4,"abcd",4),None);
    }
}
#[test]
fn gemma_full_input_v2_response_requires_real_completion_and_unique_usage() {
    let usage = json!({"prompt_tokens":4,"prompt_tokens_details":{"cached_tokens":2}});
    let unary = json!({"model":"gemma","choices":[{"index":0,"finish_reason":"tool_calls"}],"usage":usage});
    assert_eq!(completed_usage(&serde_json::to_vec(&unary).unwrap(),false,"gemma"),usage);
    for bad in [
        json!({"model":"gemma","choices":[{"index":0,"finish_reason":"tool_calls"}],"usage":usage,"error":{"message":"failed"}}),
        json!({"model":"gemma","choices":[{"index":1,"finish_reason":"tool_calls"}],"usage":usage}),
        json!({"model":"gemma","choices":[{"finish_reason":"tool_calls"}],"usage":usage}),
    ] {
        let bytes = serde_json::to_vec(&bad).unwrap();
        assert!(std::panic::catch_unwind(|| completed_usage(&bytes,false,"gemma")).is_err());
    }
    let finish = json!({"model":"gemma","choices":[{"index":0,"finish_reason":"tool_calls"}]});
    let stats = json!({"model":"gemma","choices":[],"usage":usage});
    let wire = format!("data: {finish}\n\ndata: {stats}\n\ndata: [DONE]\n\n");
    assert_eq!(completed_usage(wire.as_bytes(),true,"gemma"),usage);
    for bad in [wire.replace("data: [DONE]\n\n", ""),
        wire.replace("data: [DONE]",&format!("data: {stats}\n\ndata: [DONE]")),
        wire.replace("tool_calls",""), // empty finish is refused below
        wire.replace("gemma","other")] {
        assert!(std::panic::catch_unwind(|| completed_usage(bad.as_bytes(),true,"gemma")).is_err());
    }
}
