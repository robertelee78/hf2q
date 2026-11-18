//! Engine-free replay of every canonical N4 wire, including authorized repairs.
use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct N4Config {
    tokenizer: BoundFile,
    template: BoundFile,
    health: BoundFile,
    runtime: BoundFile,
    calibration: BoundFile,
    summary: BoundFile,
    template_source: String,
    agents: std::collections::BTreeMap<String, Vec<Turn>>,
    output: PathBuf,
}

fn repair_grant(record: &Value) -> Result<u64, &'static str> {
    match record
        .as_object()
        .ok_or("repair grant requires an object")?
        .get("repair_turns")
    {
        None => Ok(0),
        Some(value) => match value.as_u64() {
            Some(grant @ (0 | 1)) => Ok(grant),
            _ => Err("unsupported or malformed repair grant"),
        },
    }
}

fn recorded_arg<'a>(argv: &'a [Value], flag: &str) -> &'a str {
    let mut matches = argv
        .iter()
        .enumerate()
        .filter(|(_, arg)| arg.as_str() == Some(flag));
    let (index, _) = matches.next().expect("missing recorded argument");
    assert!(matches.next().is_none(), "duplicate recorded argument");
    argv.get(index + 1)
        .and_then(Value::as_str)
        .expect("missing argument value")
}

fn read_bound_json(path: PathBuf, sha256: &str) -> Value {
    serde_json::from_slice(&read(&BoundFile {
        path,
        sha256: sha256.to_owned(),
    }))
    .unwrap()
}

fn initial_history_fingerprints(request: &Value) -> (String, String) {
    // The client hashes role/content in fixture insertion order, not sorted
    // object keys. Preserve that captured contract without changing its hash.
    #[derive(serde::Serialize)]
    struct Message<'a> {
        role: &'a str,
        content: &'a str,
    }
    let messages = request["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    let message = |index: usize| {
        let value = &messages[index];
        assert_eq!(value.as_object().unwrap().len(), 2);
        Message {
            role: value["role"].as_str().unwrap(),
            content: value["content"].as_str().unwrap(),
        }
    };
    let messages = [message(0), message(1)];
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[1].role, "user");
    let identity = messages[0].content.split_once("\n\n").unwrap().0;
    (
        hash(identity.as_bytes()),
        hash(&serde_json::to_vec(&messages).unwrap()),
    )
}

#[test]
fn absent_repairs_are_zero_but_malformed_grants_are_rejected() {
    assert_eq!(repair_grant(&json!({})), Ok(0));
    assert_eq!(repair_grant(&json!({"repair_turns":0})), Ok(0));
    assert_eq!(repair_grant(&json!({"repair_turns":1})), Ok(1));
    for value in [Value::Null, json!("1"), json!(-1), json!(2), json!(1.0)] {
        assert!(repair_grant(&json!({"repair_turns":value})).is_err());
    }
}

fn accept_turn_coverage(turns: usize, repair_grant: u64, mask: u16) -> Result<(), &'static str> {
    let expected = match (turns, repair_grant) {
        (7, 0 | 1) => 0x7f,
        (9, 1) => 0x1ff,
        _ => return Err("unsupported workflow or ungranted repair"),
    };
    if mask != expected {
        return Err("actual archived turn directories are not completely covered");
    }
    Ok(())
}

#[test]
fn repair_turn_coverage_rejects_omissions_and_wrong_grants() {
    assert_eq!(accept_turn_coverage(7, 0, 0x7f), Ok(()));
    assert_eq!(accept_turn_coverage(9, 1, 0x1ff), Ok(()));
    assert!(accept_turn_coverage(7, 1, 0x1ff).is_err());
    assert!(accept_turn_coverage(9, 1, 0x7f).is_err());
    assert!(accept_turn_coverage(9, 1, 0x1ff ^ (1 << 4)).is_err());
    assert!(accept_turn_coverage(9, 0, 0x1ff).is_err());
    assert!(accept_turn_coverage(9, 2, 0x1ff).is_err());
}

fn validate_agent_archive(root: &std::path::Path, turns: &[Turn], repair_grant: u64) {
    let mut mask = 0_u16;
    for entry in std::fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name = name.to_str().expect("UTF-8 archive name required");
        let Some(index) = name.strip_prefix("turn-") else {
            continue;
        };
        assert!(
            entry.file_type().unwrap().is_dir(),
            "actual turn directory required"
        );
        let index: u32 = index.parse().expect("invalid archived turn index");
        let bit = 1_u16
            .checked_shl(index)
            .expect("unsupported archived turn index");
        assert_eq!(mask & bit, 0, "duplicate archived turn index");
        mask |= bit;
    }
    accept_turn_coverage(turns.len(), repair_grant, mask).unwrap();
    for (index, turn) in turns.iter().enumerate() {
        let directory = root.join(format!("turn-{index}"));
        assert_eq!(turn.request.path, directory.join("request.json"));
        assert_eq!(turn.receipt.path, directory.join("receipt.json"));
    }
}

#[test]
#[ignore = "post-unload complete archived N4 replay; requires HF2Q_N4_PREFIX_POST_ORACLE_CONFIG"]
fn qwen_n4_exact_prefix_post_oracle() {
    let config_bytes =
        std::fs::read(std::env::var("HF2Q_N4_PREFIX_POST_ORACLE_CONFIG").unwrap()).unwrap();
    let config: N4Config = serde_json::from_slice(&config_bytes).unwrap();
    assert_eq!(config.template_source, "GgufEmbedded");
    assert_eq!(config.agents.len(), 4, "all four agents required");

    let tokenizer_bytes = read(&config.tokenizer);
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
    assert!(tokenizer.get_truncation().is_none());
    let template_bytes = read(&config.template);
    let template = std::str::from_utf8(&template_bytes).unwrap();
    let health_bytes = read(&config.health);
    let runtime_bytes = read(&config.runtime);
    let calibration_bytes = read(&config.calibration);
    let summary_bytes = read(&config.summary);
    let health: Value = serde_json::from_slice(&health_bytes).unwrap();
    let runtime: Value = serde_json::from_slice(&runtime_bytes).unwrap();
    let calibration: Value = serde_json::from_slice(&calibration_bytes).unwrap();
    let summary: Value = serde_json::from_slice(&summary_bytes).unwrap();
    let snapshot = &runtime["measurement_snapshot"];
    assert_eq!(snapshot["schema_version"], "hf2q.measurement-snapshot.v1");
    let architecture = snapshot["architecture"].as_str().unwrap();
    let family = match architecture {
        "qwen35moe" => "qwen36",
        "qwen35" => "qwen38",
        _ => panic!("unsupported canonical Qwen N4 architecture: {architecture}"),
    };
    // Use the production registry's explicit architecture mapping, not a
    // display-name match or a dense-model loader/template fallback.
    let registration = registry::find_for_architecture(architecture).unwrap();
    assert_eq!(calibration["family"], family);
    assert_eq!(snapshot["model_id"], health["model"]);
    assert_eq!(health["context_length"], 262144);
    assert_eq!(health["backend"], "mlx-native");
    assert_eq!(runtime["backend"], "mlx-native");
    assert_eq!(runtime["serve"]["scheduler_mode"], "slot_aware");
    assert_eq!(runtime["serve"]["max_slots"], 4);
    assert_eq!(snapshot["active_controls"]["grammar"]["active"], false);
    assert_eq!(
        snapshot["tokenizer_sha256"],
        semantic_hash(serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap())
    );
    assert_eq!(
        snapshot["template_sha256"],
        semantic_hash(json!({"source":config.template_source,"template":template}))
    );
    assert_eq!(calibration["schema"], "hf2q.q5.calibration-receipt.v1");
    assert_eq!(calibration["verdict"]["verdict"], "CalibrationPass");
    assert_eq!(calibration["cleanup_proven"], true);
    assert_eq!(calibration["cleanup"]["proven"], true);
    assert_eq!(calibration["listener_after_cleanup"]["clear"], true);
    assert!(calibration["readiness"]["ready_ns"].as_u64().is_some());
    assert!(calibration["readiness"]["checks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|check| check["ok"] == true));
    assert_eq!(summary["schema"], "hf2q.qwen-n4-coordinator-result.v1");
    assert_eq!(summary["profile"], "adr063-qwen-n4-live-v1");
    assert_eq!(summary["family"], "qwen");
    assert_eq!(summary["engine_kind"], "hf2q");
    assert_eq!(summary["diagnostic_slots"], 4);
    assert_eq!(summary["server_pid"], snapshot["process_pid"]);
    assert_eq!(summary["agents"].as_object().unwrap().len(), 4);
    let repair_grant = repair_grant(&summary).unwrap();
    if repair_grant != 0 {
        assert_eq!(architecture, "qwen35moe", "repair grant is MoE-only");
    }

    let archive = config.summary.path.parent().unwrap();
    let run = archive.parent().unwrap();
    for input in [&config.health, &config.runtime, &config.calibration] {
        assert_eq!(input.path.parent(), Some(run), "readiness from another run");
    }
    assert_eq!(calibration["server"]["pid"], snapshot["process_pid"]);
    assert_eq!(
        calibration["client"]["summary_path"].as_str(),
        config.summary.path.to_str()
    );
    let case = read_bound_json(
        PathBuf::from(calibration["case"]["path"].as_str().unwrap()),
        calibration["case"]["sha256"].as_str().unwrap(),
    );
    assert_eq!(case["schema"], "hf2q.q5.calibration-case.v1");
    assert_eq!(case["family"], family);
    assert_eq!(case["workload"], calibration["workload"]);
    assert_eq!(
        self::repair_grant(&case["client"]["approval"]["set"]).unwrap(),
        repair_grant
    );
    let argv = calibration["client_argv"].as_array().unwrap();
    let approval = read_bound_json(
        PathBuf::from(recorded_arg(argv, "--approval")),
        recorded_arg(argv, "--approval-sha256"),
    );
    assert_eq!(approval["schema"], "hf2q.qwen-n4-coordinator-approval.v1");
    assert_eq!(approval["approved"], true);
    assert_eq!(approval["server_pid"], snapshot["process_pid"]);
    assert_eq!(approval["run_id"], summary["run_id"]);
    assert_eq!(approval["model"], snapshot["model_id"]);
    assert_eq!(approval["family"], "qwen");
    assert_eq!(approval["engine_kind"], "hf2q");
    assert_eq!(self::repair_grant(&approval).unwrap(), repair_grant);
    let base = approval["base_url"]
        .as_str()
        .unwrap()
        .strip_suffix("/v1")
        .unwrap();
    let documents = calibration["readiness"]["documents"].as_object().unwrap();
    assert_eq!(documents.len(), 2);
    for (endpoint, input) in [
        ("health", &config.health),
        ("hf2q/v1/runtime", &config.runtime),
    ] {
        let relative = documents[&format!("{base}/{endpoint}")].as_str().unwrap();
        assert_eq!(
            run.join(relative),
            input.path,
            "recorded readiness document differs"
        );
    }
    let mut initial_histories = std::collections::HashSet::with_capacity(4);
    let mut agents = serde_json::Map::new();
    let mut wire_count = 0;
    for id in ["agent-0", "agent-1", "agent-2", "agent-3"] {
        let turns = config.agents.get(id).expect("missing canonical agent");
        let actual = &summary["agents"][id];
        assert_eq!(actual["agent_id"], id);
        assert_eq!(actual["server_pid"], snapshot["process_pid"]);
        assert_eq!(self::repair_grant(actual).unwrap(), repair_grant);
        assert_eq!(actual["run_id"], summary["run_id"]);
        let ready = read_bound_json(
            archive.join("barrier").join(format!("{id}.ready.json")),
            actual["barrier"]["ready_sha256"].as_str().unwrap(),
        );
        assert_eq!(ready["schema"], "hf2q.n4-agent-ready.v1");
        assert_eq!(ready["agent_id"], id);
        assert_eq!(ready["run_id"], summary["run_id"]);
        assert_eq!(ready["server_pid"], snapshot["process_pid"]);
        assert_eq!(ready["barrier_token"], approval["barrier_token"]);
        assert_eq!(ready["output"].as_str(), archive.join(id).to_str());
        assert_eq!(ready["identity_sha256"], actual["initial_identity_sha256"]);
        assert_eq!(
            ready["initial_messages_sha256"],
            actual["initial_messages_sha256"]
        );
        let agent_approval = read_bound_json(
            archive.join(format!("{id}.approval.json")),
            ready["approval_sha256"].as_str().unwrap(),
        );
        assert_eq!(agent_approval["schema"], "hf2q.qwen-n4-agent-approval.v1");
        assert_eq!(agent_approval["approved"], true);
        assert_eq!(agent_approval["agent_id"], id);
        assert_eq!(
            agent_approval["agent_output"].as_str(),
            archive.join(id).to_str()
        );
        for field in [
            "run_id",
            "server_pid",
            "barrier_token",
            "model",
            "base_url",
            "family",
            "engine_kind",
            "sources",
        ] {
            assert_eq!(
                agent_approval[field], approval[field],
                "agent approval differs: {field}"
            );
        }
        assert_eq!(
            agent_approval["initial_identity_sha256"],
            ready["identity_sha256"]
        );
        assert_eq!(self::repair_grant(&agent_approval).unwrap(), repair_grant);
        let initial_request: Value = serde_json::from_slice(&read(&turns[0].request)).unwrap();
        let (identity, history) = initial_history_fingerprints(&initial_request);
        assert_eq!(ready["identity_sha256"], identity);
        assert_eq!(ready["initial_messages_sha256"], history);
        assert!(
            initial_histories.insert(history),
            "duplicate independent agent conversation"
        );
        let summary_turns = actual["turns"].as_array().unwrap();
        assert_eq!(turns.len(), summary_turns.len(), "omitted actual wire");
        validate_agent_archive(&archive.join(id), turns, repair_grant);
        let observations = replay_qwen_prefix_turns(
            &tokenizer,
            template,
            &snapshot["model_id"],
            &registration,
            turns,
            Some(summary_turns),
        );
        wire_count += observations.len();
        agents.insert(id.to_owned(), json!({"observations":observations}));
    }
    assert!(config.output.is_absolute());
    let parent = config.output.parent().unwrap().canonicalize().unwrap();
    assert!(parent.ancestors().all(|path| !path.join(".git").exists()));
    let output = json!({
        "schema":"hf2q.qwen-n4-prefix-post-oracle.v1",
        "full_prefix_reuse":"PassedExactAllArchivedWires",
        "model_inference_performed":false,
        "model_qualified":false,
        "architecture":architecture,
        "family":family,
        "model_id":snapshot["model_id"],
        "server_slots":4,
        "repair_turns":repair_grant,
        "wire_count":wire_count,
        "config_sha256":hash(&config_bytes),
        "tokenizer_file_sha256":hash(&tokenizer_bytes),
        "template_file_sha256":hash(&template_bytes),
        "health_sha256":hash(&health_bytes),
        "runtime_sha256":hash(&runtime_bytes),
        "calibration_sha256":hash(&calibration_bytes),
        "summary_sha256":hash(&summary_bytes),
        "case_sha256":calibration["case"]["sha256"],
        "approval_sha256":recorded_arg(argv, "--approval-sha256"),
        "oracle_source_sha256":hash(include_bytes!("handlers_prefix_post_oracle_n4_tests.rs")),
        "shared_oracle_source_sha256":hash(include_bytes!("handlers_prefix_post_oracle_tests.rs")),
        "handler_source_sha256":hash(include_bytes!("handlers.rs")),
        "renderer_source_sha256":hash(include_bytes!("engine.rs")),
        "measurement_source_sha256":hash(include_bytes!("measurement.rs")),
        "agents":agents
    });
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
