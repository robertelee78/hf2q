//! C2-only archived consumer: rendered token prefixes, never engine/cache state.
use super::*;
use std::cell::RefCell;
use std::collections::BTreeMap;

const WORKLOAD: &str = "divergent-edit-v1";
const FIXTURE: &str = "b6a671e3c52adc8bb0bbdd62584973f3720fd97c7a92b2c2f4eaadbd20849991";
const MARKER: &str = "[... hf2q divergent-edit-v1: earlier tool result truncated after 16 lines ...]";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct C2Context {
    schema: String,
    case: BoundFile,
    approval: BoundFile,
    server_log: BoundFile,
    compiled_source: BoundFile,
    subject_images: BoundFile,
    expected_source_commit: String,
    expected_anchor_depth: u32,
    expected_workload_kind: String,
    #[serde(skip)]
    captures: RefCell<BTreeMap<usize, usize>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct C2Config {
    tokenizer: BoundFile,
    template: BoundFile,
    health: BoundFile,
    runtime: BoundFile,
    calibration: BoundFile,
    summary: BoundFile,
    template_source: String,
    agents: BTreeMap<String, Vec<Turn>>,
    output: PathBuf,
    c2: C2Context,
}

#[test]
#[ignore = "post-unload C2 archive replay; requires HF2Q_N4_C2_PREFIX_POST_ORACLE_CONFIG"]
fn qwen_n4_c2_exact_prefix_post_oracle() {
    let bytes = std::fs::read(std::env::var("HF2Q_N4_C2_PREFIX_POST_ORACLE_CONFIG").unwrap()).unwrap();
    let config: C2Config = serde_json::from_slice(&bytes).unwrap();
    let n4 = N4Config { tokenizer: config.tokenizer, template: config.template,
        health: config.health, runtime: config.runtime, calibration: config.calibration,
        summary: config.summary, template_source: config.template_source,
        agents: config.agents, output: config.output };
    run_n4(&bytes, n4, Some(config.c2));
}

fn closed(value: &Value, keys: &[&str]) {
    let object = value.as_object().expect("closed evidence object required");
    assert_eq!(object.len(), keys.len(), "unknown or missing evidence controls");
    assert!(keys.iter().all(|key| object.contains_key(*key)));
}

fn bound(value: &Value) -> BoundFile {
    serde_json::from_value(value.clone()).unwrap()
}

fn startup(raw: &str) -> Result<(u32, u64), &'static str> {
    let ansi = regex::Regex::new(r"\x1b\[[0-9;]*m").unwrap();
    let declaration = regex::Regex::new(concat!(
        r"^(?:\d{4}-\d\d-\d\dT\S+\s+)?INFO\s+hf2q::serve::api::engine:\s+",
        r"Qwen35 multi-depth anchor store configured anchor_depth=([14]) ",
        r"anchor_bytes_per_slot_max=Some\(([1-9][0-9]*)\)$")).unwrap();
    let mut observed = None;
    for line in raw.lines() {
        let clean = ansi.replace_all(line, "");
        let line = clean.trim();
        if !line.contains("Qwen35 multi-depth anchor store configured") { continue; }
        let row = declaration.captures(line).ok_or("malformed startup declaration")?;
        if observed.is_some() { return Err("duplicate startup declaration"); }
        observed = Some((row[1].parse().map_err(|_| "invalid K")?,
            row[2].parse().map_err(|_| "invalid quota")?));
    }
    observed.ok_or("missing actual startup declaration")
}

#[path = "handlers_prefix_post_oracle_c2_source_tests.rs"]
mod source;

pub(super) fn validate_context(context: &C2Context, config: &N4Config, case: &Value,
    approval: &Value, calibration: &Value, runtime: &Value, summary: &Value) -> Value {
    assert_eq!(context.schema, "hf2q.c2-natural-prefix-context.v2");
    assert_eq!(context.expected_workload_kind, WORKLOAD);
    assert!(matches!(context.expected_anchor_depth, 1 | 4));
    assert_eq!(context.case.path.to_str(), calibration["case"]["path"].as_str());
    assert_eq!(context.case.sha256, calibration["case"]["sha256"]);
    assert_eq!(serde_json::from_slice::<Value>(&read(&context.case)).unwrap(), *case);
    let argv = calibration["client_argv"].as_array().unwrap();
    assert_eq!(context.approval.path.to_str(), Some(recorded_arg(argv, "--approval")));
    assert_eq!(context.approval.sha256, recorded_arg(argv, "--approval-sha256"));
    assert_eq!(serde_json::from_slice::<Value>(&read(&context.approval)).unwrap(), *approval);
    assert_eq!(recorded_arg(argv, "--workload"), WORKLOAD);
    assert_eq!(recorded_arg(case["client"]["argv"].as_array().unwrap(), "--workload"), WORKLOAD);
    assert_eq!(approval["workload_variant"], WORKLOAD);
    assert_eq!(approval["workload_fixture_sha256"], FIXTURE);
    let grant = &case["client"]["approval"]["set"];
    assert_eq!(grant["workload_variant"], WORKLOAD);
    let sources = grant["sources"]["sha256_of"].as_array().unwrap();
    assert_eq!(sources.len(), approval["sources"].as_object().unwrap().len());
    for path in sources {
        let path = PathBuf::from(path.as_str().unwrap());
        let name = path.file_name().unwrap().to_str().unwrap();
        read(&BoundFile { path: path.clone(), sha256: approval["sources"][name].as_str().unwrap().to_owned() });
    }
    for path in grant["c2_observation_sources"]["sha256_of"].as_array().unwrap() {
        let path = PathBuf::from(path.as_str().unwrap());
        let name = path.file_name().unwrap().to_str().unwrap();
        read(&BoundFile { path: path.clone(), sha256: approval["c2_observation_sources"][name].as_str().unwrap().to_owned() });
    }
    let fixture_path = grant["workload_fixture_sha256"]["sha256_of"].as_str().unwrap();
    let fixture = read(&BoundFile { path: fixture_path.into(), sha256: FIXTURE.into() });
    assert_eq!(std::str::from_utf8(&fixture).unwrap().trim_end_matches('\n'), MARKER);
    assert_eq!(summary["workload_variant"], json!({"id":WORKLOAD,"fixture_sha256":FIXTURE,"edit_turns":[3,5],"keep_lines":16}));
    let counts: Vec<_> = config.agents.iter().map(|(id, turns)| (id.as_str(), turns.len())).collect();
    cohort_coverage(&counts, runtime["measurement_snapshot"]["architecture"].as_str().unwrap(), repair_grant(summary).unwrap()).unwrap();
    let control = &summary["c2_cache_control"];
    closed(control, &["schema", "server_pid", "run_id", "server_log", "log_dev", "log_ino",
        "anchor_depth", "anchor_bytes_per_slot_max", "startup_log"]);
    assert_eq!(control["schema"], "hf2q.adr063.c2-cache-control.v1");
    assert_eq!(control["server_pid"], runtime["measurement_snapshot"]["process_pid"]);
    assert_eq!(control["run_id"], summary["run_id"]);
    assert!(approval.get("c2_cache_control").is_none());
    assert!(approval["agent_template"].get("c2_cache_control").is_none());
    assert_eq!(context.approval.path, config.summary.path.parent().unwrap().parent().unwrap().join("client-approval.json"));
    assert_eq!(recorded_arg(argv, "--output"), config.summary.path.parent().unwrap().to_str().unwrap());
    assert_eq!(recorded_arg(argv, "--server-pid"), summary["server_pid"].as_u64().unwrap().to_string());
    for actual in summary["agents"].as_object().unwrap().values() {
        assert_eq!(&actual["c2_cache_control"], control);
        assert_eq!(actual["workload_variant"], summary["workload_variant"]);
    }
    assert_eq!(control["startup_log"]["path"].as_str(), config.summary.path.parent().unwrap().join("c2-startup.log").to_str());
    assert_eq!(control["server_log"].as_str(), context.server_log.path.to_str());
    assert_eq!(approval["c2_server_log"], control["server_log"]);
    assert_eq!(recorded_arg(argv, "--c2-server-log"), context.server_log.path.to_str().unwrap());
    assert_eq!(context.server_log.path, config.summary.path.parent().unwrap().parent().unwrap().join("server.stderr"));
    let raw_log = read(&context.server_log);
    let log = std::str::from_utf8(&raw_log).unwrap();
    let (depth, quota) = observed_control(log, context.expected_anchor_depth).unwrap();
    assert_eq!(control["anchor_depth"].as_u64(), Some(u64::from(depth)));
    assert_eq!(control["anchor_bytes_per_slot_max"].as_u64(), Some(quota));
    for workload in [&case["workload"], &calibration["workload"]] {
        assert_eq!(workload["anchors_per_slot"].as_u64(), Some(u64::from(depth)));
        assert_eq!(workload["anchor_bytes_per_slot_max"].as_u64(), Some(quota));
    }
    for env in [&case["serve"]["env"], &calibration["serve_env"]] {
        assert_eq!(env["HF2Q_QWEN_ANCHOR_DEPTH"], depth.to_string());
        assert_eq!(env["HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX"], quota.to_string());
    }
    let retained = read(&bound(&control["startup_log"]));
    assert!(retained.len() <= 65536 && !retained.is_empty());
    assert!(raw_log.starts_with(&retained), "startup evidence is not from the original server log");
    assert_eq!(startup(std::str::from_utf8(&retained).unwrap()).unwrap(), (depth, quota));
    #[cfg(unix)] {
        use std::os::unix::fs::MetadataExt;
        let metadata = std::fs::metadata(&context.server_log.path).unwrap();
        assert_eq!(control["log_dev"].as_u64(), Some(metadata.dev()));
        assert_eq!(control["log_ino"].as_u64(), Some(metadata.ino()));
    }
    let capture = regex::Regex::new(r"Qwen35 slot-local stable prompt boundary captured slot=[0-3] prompt_tokens=([1-9][0-9]*) checkpoint_bytes=[1-9][0-9]*\s*$").unwrap();
    let mut captures = context.captures.borrow_mut();
    for line in log.lines() {
        if let Some(row) = capture.captures(line) {
            *captures.entry(row[1].parse().unwrap()).or_default() += 1;
        }
    }
    assert_eq!(captures.values().sum::<usize>(), config.agents.values().map(Vec::len).sum::<usize>(), "every archived request must have one actual stable capture");
    let compiled = read(&context.compiled_source);
    let source: Value = serde_json::from_slice(&compiled).unwrap();
    let image_bytes = read(&context.subject_images);
    let images: Value = serde_json::from_slice(&image_bytes).unwrap();
    source::validate_subject_identity(&source, &images, &calibration, &context.expected_source_commit).unwrap();
    let closure = source::rendering_closure(&source);
    assert_eq!(case["variables"]["HF2Q_BIN"], calibration["binary"]["path"]);
    assert_eq!(case["variables"]["MODEL_SHA256"], calibration["artifact"]["sha256"]);
    assert_eq!(case["variables"]["MODEL_ID"], runtime["measurement_snapshot"]["model_id"]);
    assert_eq!(calibration["text_only_locator"]["same_file_as_artifact"], true);
    assert_eq!(hash(calibration["served_model"].as_str().unwrap().as_bytes()), runtime["measurement_snapshot"]["model_locator_sha256"]);
    let proof = read_bound_json(recorded_arg(argv, "--sandbox-proof").into(), recorded_arg(argv, "--sandbox-proof-sha256"));
    assert!(proof.is_object());
    assert_eq!(approval["agent_template"]["sandbox_proof_sha256"], recorded_arg(argv, "--sandbox-proof-sha256"));
    let parent = config.output.parent().unwrap().canonicalize().unwrap();
    assert!(!parent.starts_with(config.summary.path.parent().unwrap().parent().unwrap().canonicalize().unwrap()), "proof must be separate from frozen run");
    json!({"anchor_depth":depth,"anchor_bytes_per_slot_max":quota,"workload":WORKLOAD,
        "case_sha256":context.case.sha256,"approval_sha256":context.approval.sha256,
        "server_log_sha256":context.server_log.sha256,"compiled_source_sha256":hash(&compiled),
        "observed_cache_control":control,
        "subject_source_commit":source["source_commit"],"subject_production_image":images["images"]["production"],
        "subject_image_metadata_sha256":hash(&image_bytes),"published_native_version":images["registry_native_version"],
        "source_binding_oracle_sha256":hash(include_bytes!("handlers_prefix_post_oracle_c2_source_tests.rs")),
        "compiled_production_render_closure":closure,"algorithm":"bounded stable boundaries; deepest exact prefix; cursor pruning; depth-sorted K eviction",
        "executor_is_subject_inference_image":false,"claims_cache_or_hidden_state_equality":false})
}

fn observed_control(log: &str, expected: u32) -> Result<(u32, u64), &'static str> {
    let actual = startup(log)?;
    if actual.0 != expected { return Err("expected K differs from observed model startup"); }
    Ok(actual)
}

fn cohort_coverage(counts: &[(&str, usize)], architecture: &str, repairs: u64) -> Result<(), &'static str> {
    let expected = match (architecture, repairs) {
        ("qwen35", 0) | ("qwen35moe", 0) => [7,7,7,7],
        ("qwen35moe", 1) => [9,7,7,7],
        _ => return Err("unsupported canonical C2 cohort"),
    };
    if counts.len() != 4 { return Err("all four independent agents required"); }
    for (index, id) in ["agent-0","agent-1","agent-2","agent-3"].iter().enumerate() {
        if counts[index] != (*id, expected[index]) { return Err("incomplete original C2 wires"); }
    }
    Ok(())
}

struct PrefixState { keep: usize, anchors: Vec<Vec<u32>>, previous_extent: usize }
impl PrefixState {
    fn new(keep: u32) -> Self { assert!(matches!(keep, 1 | 4)); Self { keep: keep as usize, anchors: Vec::new(), previous_extent: 0 } }
    fn observe(&mut self, turn: usize, edited: bool, p: &[u32], s: &[u32], usage: &Value) -> Result<Value, &'static str> {
        if s.is_empty() || s.len() >= p.len() || !p.starts_with(s) { return Err("unproved stable boundary"); }
        if usage["prompt_tokens"].as_u64() != Some(p.len() as u64) { return Err("actual usage/render mismatch"); }
        let cached = usize::try_from(usage["prompt_tokens_details"]["cached_tokens"].as_u64().ok_or("missing cached usage")?).map_err(|_| "cached overflow")?;
        let expected = self.anchors.iter().filter(|a| p.starts_with(a)).map(Vec::len).max().unwrap_or(0);
        let upper = if turn == 0 || edited { expected } else { s.len().min(self.previous_extent) };
        let completion = usize::try_from(usage["completion_tokens"].as_u64().ok_or("missing actual completion extent")?).map_err(|_| "completion overflow")?;
        let next_extent = p.len().checked_add(completion).ok_or("retained extent overflow")?;
        if turn == 0 {
            if cached != 0 || expected != 0 { return Err("initial request must be cold"); }
        } else if expected == 0 {
            if self.keep != 1 || !edited || !matches!(turn, 3 | 5) || cached != 0 { return Err("unapproved zero/dead lineage"); }
        } else if cached < expected || cached > upper {
            return Err("deepest stable boundary partially reused or divergent excess");
        }
        if expected == 0 { self.anchors.clear(); }
        else { self.anchors.retain(|a| a.len() <= expected && p.starts_with(a)); }
        self.anchors.retain(|a| a.len() != s.len());
        self.anchors.push(s.to_vec());
        self.anchors.sort_by_key(Vec::len);
        if self.anchors.len() > self.keep { self.anchors.drain(..self.anchors.len() - self.keep); }
        self.previous_extent = next_extent;
        Ok(json!({"cached_tokens":cached,"expected_deepest_stable_tokens":expected,
            "cached_lower_bound":expected,"cached_upper_bound":upper,
            "genuine_pinned_edit":edited,"published_stable_depths":self.anchors.iter().map(Vec::len).collect::<Vec<_>>()}))
    }
}

fn python_lines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        if matches!(ch, '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}') {
            lines.push(&text[start..index]);
            start = index + ch.len_utf8();
            if ch == '\r' && chars.peek().is_some_and(|(_, next)| *next == '\n') { chars.next(); start += 1; }
        }
    }
    if start < text.len() { lines.push(&text[start..]); }
    lines
}

fn edit_history(previous: &[Value], current: &[Value], turn: usize) -> Result<Option<Value>, &'static str> {
    if current.len() != previous.len() + 2 { return Err("noncanonical appended history"); }
    let prefix = &current[..previous.len()];
    let edit = if matches!(turn, 3 | 5) {
        let target = turn - 2;
        let index = previous.iter().enumerate().filter(|(_, m)| m["role"] == "tool").nth(target).map(|(i, _)| i).ok_or("missing pinned tool target")?;
        let text = previous[index]["content"].as_str().ok_or("nontext tool result")?;
        let mut lines = python_lines(text);
        let original_lines = lines.len();
        lines.truncate(16);
        lines.push(MARKER);
        let edited = lines.join("\n");
        if edited == text { return Err("history edit did not change bytes"); }
        let mut expected = previous[index].clone();
        expected["content"] = json!(edited);
        if prefix[..index] != previous[..index] || prefix[index] != expected || prefix[index+1..] != previous[index+1..] {
            return Err("history differs from the genuine pinned edit");
        }
        Some(json!({"turn":turn,"target_turn":target,"message_index":index,
            "original_bytes":text.len(),"edited_bytes":edited.len(),"original_lines":original_lines,
            "kept_lines":original_lines.min(16),"original_sha256":hash(text.as_bytes()),"edited_sha256":hash(edited.as_bytes())}))
    } else {
        if prefix != previous { return Err("noncanonical unchanged history"); }
        None
    };
    if current[previous.len()]["role"] != "assistant" || current[previous.len()+1]["role"] != "tool" { return Err("noncanonical continuation"); }
    Ok(edit)
}

fn response_message(raw: &[u8], streamed: bool, model: &Value, usage: &Value) -> Value {
    if !streamed {
        let obj: Value = serde_json::from_slice(raw).unwrap();
        assert_eq!(&obj["model"], model); assert_eq!(&obj["usage"], usage);
        assert_eq!(obj["choices"].as_array().unwrap().len(), 1);
        return obj["choices"][0]["message"].clone();
    }
    let mut message = json!({"role":"assistant","content":"","reasoning_content":""});
    let mut call = json!({"id":null,"type":"function","function":{"name":"","arguments":""}});
    let mut have_call = false;
    let mut done = false;
    let mut stats = None;
    let mut finish = false;
    let text = std::str::from_utf8(raw).unwrap().replace("\r\n", "\n");
    assert!(text.ends_with("\n\n"));
    for frame in text.split("\n\n").filter(|f| !f.is_empty()) {
        if frame.starts_with(':') { continue; }
        assert!(!done, "data after DONE");
        let data = frame.strip_prefix("data: ").expect("actual SSE data frame");
        if data == "[DONE]" { assert!(finish && stats.is_some()); done = true; continue; }
        let obj: Value = serde_json::from_str(data).unwrap();
        assert_eq!(&obj["model"], model);
        if !obj["usage"].is_null() { assert!(stats.is_none()); stats = Some(obj["usage"].clone()); }
        let choices = obj["choices"].as_array().unwrap();
        if choices.is_empty() { assert!(finish); continue; }
        assert!(!finish); assert_eq!(choices.len(), 1); assert_eq!(choices[0]["index"], 0);
        let delta = &choices[0]["delta"];
        assert!(delta.as_object().unwrap().keys().all(|k| ["role","content","reasoning_content","tool_calls"].contains(&k.as_str())));
        for key in ["content", "reasoning_content"] {
            if let Some(part) = delta[key].as_str() {
                let Value::String(text) = &mut message[key] else { unreachable!() };
                text.push_str(part);
            }
        }
        if let Some(parts) = delta["tool_calls"].as_array() { for part in parts {
            assert_eq!(part["index"], 0); have_call = true;
            if !part["id"].is_null() { assert!(call["id"].is_null()); call["id"] = part["id"].clone(); }
            for key in ["name", "arguments"] { if let Some(value) = part["function"][key].as_str() {
                let Value::String(text) = &mut call["function"][key] else { unreachable!() };
                text.push_str(value);
            }}
        }}
        finish = !choices[0]["finish_reason"].is_null();
    }
    assert!(done); assert_eq!(stats.as_ref(), Some(usage));
    if have_call { message["tool_calls"] = json!([call]); }
    message
}

pub(super) fn replay(context: &C2Context, tokenizer: &tokenizers::Tokenizer, template: &str,
    model: &Value, registration: &registry::ModelRegistration, turns: &[Turn], summary: &[Value],
    agent: &Value, approval: &Value) -> Vec<Value> {
    assert_eq!(agent["c2_cache_control"], approval["c2_cache_control"]);
    assert_eq!(agent["workload_variant"]["id"], WORKLOAD);
    assert_eq!(approval["workload_variant"], WORKLOAD);
    assert_eq!(approval["workload_fixture_sha256"], FIXTURE);
    let agent_root = turns[0].request.path.parent().unwrap().parent().unwrap();
    let archive = agent_root.parent().unwrap();
    let id = agent["agent_id"].as_str().unwrap();
    let command: Value = serde_json::from_slice(&std::fs::read(archive.join(format!("{id}.command.json"))).unwrap()).unwrap();
    let argv = command["argv"].as_array().unwrap();
    assert_eq!(recorded_arg(argv, "--workload"), WORKLOAD);
    let approval_path = archive.join(format!("{id}.approval.json"));
    assert_eq!(recorded_arg(argv, "--approval"), approval_path.to_str().unwrap());
    assert_eq!(read_bound_json(approval_path, recorded_arg(argv, "--approval-sha256")), *approval);
    let copy: Value = serde_json::from_slice(&std::fs::read(agent_root.join("approval-copy.json")).unwrap()).unwrap();
    assert_eq!(copy, *approval);
    assert_eq!(agent["tools"].as_array().unwrap().len() + 1, turns.len());
    let mut state = PrefixState::new(context.expected_anchor_depth);
    let mut previous: Vec<Value> = Vec::new();
    let mut assistant = Value::Null;
    let mut edits = Vec::new();
    let mut observer = |index: usize, request: &Value, p: &[u32], s: &[u32], usage: &Value| {
        let messages = request["messages"].as_array().unwrap();
        assert_eq!(summary[index]["stream"].as_bool(), Some(index % 2 == 1));
        assert_eq!(request["stream"], summary[index]["stream"]);
        let edit = if index == 0 { None } else {
            let edit = edit_history(&previous, messages, index).unwrap();
            assert_eq!(messages[previous.len()], assistant, "actual archived response differs from next wire");
            let call = &assistant["tool_calls"][0];
            let tool = &messages[previous.len()+1];
            assert_eq!(tool["tool_call_id"], call["id"]);
            assert_eq!(tool["name"], call["function"]["name"]);
            let actual_tool = &agent["tools"][index-1];
            assert_eq!(actual_tool["status"], "Complete");
            assert_eq!(actual_tool["cleanup"], true);
            assert_eq!(tool["content"], actual_tool["result"]);
            assert_eq!(tool["name"], actual_tool["name"]);
            assert_eq!(serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(), actual_tool["arguments"]);
            edit
        };
        if let Some(edit) = &edit { edits.push(edit.clone()); }
        let directory = turns[index].receipt.path.parent().unwrap();
        let raw = read(&BoundFile { path: directory.join("response.bin"), sha256: summary[index]["response_sha256"].as_str().unwrap().to_owned() });
        assert_eq!(summary[index]["response_bytes"].as_u64(), Some(raw.len() as u64));
        assistant = response_message(&raw, request["stream"].as_bool().unwrap(), model, usage);
        let assembled: Value = serde_json::from_slice(&std::fs::read(directory.join("assembled-assistant.json")).unwrap()).unwrap();
        assert_eq!(assistant, assembled);
        let mut captures = context.captures.borrow_mut();
        let count = captures.get_mut(&s.len()).expect("rendered stable boundary absent from actual server captures");
        assert!(*count > 0, "reused capture evidence"); *count -= 1;
        let detail = state.observe(index, edit.is_some(), p, s, usage).unwrap();
        previous = messages.clone();
        detail
    };
    let observations = replay_qwen_prefix_turns_with(tokenizer, template, model, registration, turns, Some(summary), Some(&mut observer));
    assert_eq!(json!(edits), agent["workload_variant_edits"], "pinned edit audit differs from actual wire reconstruction");
    observations
}

#[test]
fn c2_deepest_older_prefix_and_limited_k() {
    let usage = |n, cached| json!({"prompt_tokens":n,"completion_tokens":0,"prompt_tokens_details":{"cached_tokens":cached}});
    for keep in [1, 4] {
        let mut state = PrefixState::new(keep);
        state.observe(0, false, &[1,2,9], &[1,2], &usage(3,0)).unwrap();
        state.observe(1, false, &[1,2,3,9], &[1,2,3], &usage(4,2)).unwrap();
        state.observe(2, false, &[1,2,3,4,9], &[1,2,3,4], &usage(5,3)).unwrap();
        let cached = if keep == 1 { 0 } else { 2 };
        if keep == 4 { assert!(state.observe(3, true, &[1,2,8,9], &[1,2,8], &usage(4,3)).is_err()); }
        let row = state.observe(3, true, &[1,2,8,9], &[1,2,8], &usage(4,cached)).unwrap();
        assert_eq!(row["expected_deepest_stable_tokens"], cached);
        assert!(state.anchors.iter().all(|a| !a.contains(&3)));
        assert!(state.observe(4, false, &[1,2,8,5,9], &[1,2,8,5], &usage(5,0)).is_err());
        state.observe(4, false, &[1,2,8,5,9], &[1,2,8,5], &usage(5,3)).unwrap();
    }
}

#[test]
fn c2_rejects_partial_zero_nonedit_and_unstable_boundary() {
    let mut state = PrefixState::new(4);
    let usage = |n, c| json!({"prompt_tokens":n,"completion_tokens":0,"prompt_tokens_details":{"cached_tokens":c}});
    state.observe(0, false, &[1,2,3,9], &[1,2,3], &usage(4,0)).unwrap();
    assert!(state.observe(1, false, &[1,2,3,4,9], &[1,2,3,4], &usage(5,2)).is_err());
    assert!(state.observe(1, false, &[8,9], &[8], &usage(2,0)).is_err());
    assert!(state.observe(3, true, &[8,9], &[8], &usage(2,0)).is_err());
    assert!(state.observe(1, false, &[1,2,3,4,9], &[1,2,4], &usage(5,3)).is_err());
    let mut k1 = PrefixState::new(1);
    k1.observe(0, false, &[1,2,9], &[1,2], &usage(3,0)).unwrap();
    assert!(k1.observe(3, false, &[8,9], &[8], &usage(2,0)).is_err());
    assert!(k1.observe(4, true, &[8,9], &[8], &usage(2,0)).is_err());
}

#[test]
fn c2_rejects_wrong_keep_startup_altered_wire_and_incomplete_coverage() {
    let line = " INFO hf2q::serve::api::engine: Qwen35 multi-depth anchor store configured anchor_depth=4 anchor_bytes_per_slot_max=Some(64)";
    assert_eq!(startup(line), Ok((4,64)));
    assert!(observed_control(line, 1).is_err());
    assert!(cohort_coverage(&[("agent-0",7),("agent-1",7),("agent-2",7)], "qwen35", 0).is_err());
    assert!(cohort_coverage(&[("agent-0",7),("agent-1",7),("agent-2",7),("agent-3",7)], "qwen35moe", 1).is_err());
    assert!(cohort_coverage(&[("agent-0",9),("agent-1",7),("agent-2",7),("agent-3",7)], "qwen35moe", 1).is_ok());
    assert!(startup(&line.replace("depth=4", "depth=2")).is_err());
    assert!(startup(&format!("{line}\n{line}")).is_err());
    let previous = vec![json!({"role":"system","content":"s"}), json!({"role":"user","content":"u"})];
    let mut current = previous.clone(); current.extend([json!({"role":"assistant"}), json!({"role":"tool"})]);
    assert!(edit_history(&previous, &current, 1).is_ok());
    current[0]["content"] = json!("changed");
    assert!(edit_history(&previous, &current, 1).is_err());
    assert!(accept_turn_coverage(7, 0, 0x7f ^ 8).is_err());
    assert!(accept_turn_coverage(9, 0, 0x1ff).is_err());
}

#[test]
fn c2_pinned_edit_reconstruction_rejects_wrong_truncation() {
    let text = (0..20).map(|n| n.to_string()).collect::<Vec<_>>().join("\n");
    let previous = vec![json!({"role":"system","content":"s"}),
        json!({"role":"tool","content":"first"}), json!({"role":"tool","content":text})];
    let mut current = previous.clone();
    current[2]["content"] = json!(format!("{}\n{MARKER}", (0..16).map(|n| n.to_string()).collect::<Vec<_>>().join("\n")));
    current.extend([json!({"role":"assistant"}), json!({"role":"tool"})]);
    let edit = edit_history(&previous, &current, 3).unwrap().unwrap();
    assert_eq!(edit["kept_lines"], 16);
    assert_ne!(edit["original_sha256"], edit["edited_sha256"]);
    assert!(edit_history(&previous, &current, 4).is_err());
    current[2]["content"] = json!(format!("{}\n{MARKER}", (0..17).map(|n| n.to_string()).collect::<Vec<_>>().join("\n")));
    assert!(edit_history(&previous, &current, 3).is_err());
    assert_eq!(python_lines("a\r\nb\u{85}c\n"), vec!["a","b","c"]);
}
