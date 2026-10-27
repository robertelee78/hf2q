use super::*;

#[test]
fn capacity_token_config_and_family_are_closed() {
    for (f, arch) in [
        ("qwen38", "qwen35"),
        ("qwen36", "qwen35moe"),
        ("deepseek4", "deepseek4"),
        ("gemma4", "gemma4"),
    ] {
        assert_eq!(selected(f).unwrap().1, arch);
        assert!(cache_scope(f).is_some());
    }
    assert!(selected("unknown").is_none());
    assert!(cache_scope("unknown").is_none());
    let b = json!({"path":"/unused","sha256":"a".repeat(64)});
    let mut c = json!({"family":"qwen36","model":"Qwen3.6 35B A3B","server_source":"a".repeat(40),"server_binary_sha256":"b".repeat(64),"tokenizer":b,"template":b,"readiness":b,"controller_receipt":b,"client_summary":b,"server_build":b,"recipe_proof":b,"client_source":b,"tools_source":b,"family_cache_evidence":null,"agents":{},"output_root":"/unused"});
    assert!(serde_json::from_value::<Config>(c.clone()).is_ok());
    c["fallback"] = json!(true);
    assert!(serde_json::from_value::<Config>(c).is_err());
}
#[test]
fn capacity_token_wire_controls_reject_drift() {
    for f in ["qwen38", "qwen36", "deepseek4", "gemma4"] {
        for stage in 0..2 {
            for i in 0..4 {
                let model = selected(f).unwrap().0;
                let stream = if stage == 0 { i % 2 == 1 } else { i % 2 == 0 };
                let mut v = json!({"model":model,"messages":[],"tools":[],"tool_choice":"auto","max_completion_tokens":8192,"logprobs":false,"hf2q_overflow_policy":"reject","stream":stream});
                if stream {
                    v["stream_options"] = json!({"include_usage":true});
                }
                if f == "deepseek4" {
                    v["temperature"] = json!(0);
                    v["reasoning_effort"] = json!("max");
                }
                assert!(shape(&v, f, model, i, stage));
                for (key, val) in [
                    ("model", json!("wrong")),
                    ("max_completion_tokens", json!(8191)),
                    ("stream", json!(!stream)),
                    ("chat_template_kwargs", json!({})),
                ] {
                    let mut bad = v.clone();
                    bad[key] = val;
                    assert!(!shape(&bad, f, model, i, stage));
                }
            }
        }
    }
}
#[test]
fn capacity_token_actual_flow_preserves_reasoning_id_and_whole_result() {
    let a = json!({"messages":[{"role":"system","content":"same"},{"role":"user","content":"read"}],"tools":[]});
    let assistant = json!({"role":"assistant","content":null,"reasoning_content":"actual reasoning","tool_calls":[{"id":"actual","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/ports.rs\"}"}}]});
    let payload = "x".repeat(FILE);
    let mut msgs = a["messages"].as_array().unwrap().clone();
    msgs.push(assistant.clone());
    msgs.push(json!({"role":"tool","tool_call_id":"actual","name":"read_file","content":payload}));
    let b = json!({"messages":msgs,"tools":[]});
    assert!(flow(&a, &b, &assistant, &payload).is_ok());
    for bad in ["x".repeat(FILE - 1), "x".repeat(FILE + 1)] {
        assert!(flow(&a, &b, &assistant, &bad).is_err());
    }
    let mut bad = b.clone();
    bad["messages"][2]["reasoning_content"] = json!("dropped");
    assert!(flow(&a, &bad, &assistant, &payload).is_err());
    let mut bad = b;
    bad["messages"][3]["tool_call_id"] = json!("foreign");
    assert!(flow(&a, &bad, &assistant, &payload).is_err());
}
#[test]
fn capacity_token_cache_scope_does_not_fabricate_ds_history() {
    let old: Vec<u32> = (0..757).collect();
    let mut current: Vec<u32> = (0..846).collect();
    current[749] = 9000;
    let usage = |cached| json!({"prompt_tokens":846,"completion_tokens":1,"total_tokens":847,"prompt_tokens_details":{"cached_tokens":cached}});
    assert_eq!(
        accept_tokens(
            &current,
            Some(&old),
            &usage(749),
            64,
            CacheScope::PromptCheckpointP0Lcp
        ),
        Ok(749)
    );
    assert!(accept_tokens(
        &current,
        Some(&old),
        &usage(814),
        64,
        CacheScope::PromptCheckpointP0Lcp
    )
    .is_err());
    assert_eq!(
        accept_tokens(
            &current,
            Some(&old),
            &usage(814),
            0,
            CacheScope::DeepseekLiveOrRecovery
        ),
        Ok(814)
    );
    for n in [0, 847] {
        assert!(accept_tokens(
            &current,
            Some(&old),
            &usage(n),
            0,
            CacheScope::DeepseekLiveOrRecovery
        )
        .is_err());
    }
    let mut bad = usage(1);
    bad["prompt_tokens"] = json!(845);
    assert!(accept_tokens(
        &current,
        Some(&old),
        &bad,
        0,
        CacheScope::DeepseekLiveOrRecovery
    )
    .is_err());
}
#[test]
fn capacity_token_raw_completion_has_one_terminal_and_exact_assistant() {
    let model = "fixture";
    let call = json!({"index":0,"id":"actual","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/ports.rs\"}"}});
    let usage = json!({"prompt_tokens":9,"completion_tokens":2,"total_tokens":11,"prompt_tokens_details":{"cached_tokens":0}});
    let frames = [
        json!({"id":"same","model":model,"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[call]},"finish_reason":"tool_calls"}]}),
        json!({"id":"same","model":model,"choices":[],"usage":usage}),
    ];
    let raw = format!(
        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        frames[0], frames[1]
    );
    let (assistant, u) = response(raw.as_bytes(), true, model, 0);
    assert_eq!(u, usage);
    assert_eq!(assistant["tool_calls"][0]["id"], "actual");
    assert_eq!(assistant["reasoning_content"], "");
    let no_done = raw.replace("data: [DONE]\n\n", "");
    assert!(std::panic::catch_unwind(|| response(no_done.as_bytes(), true, model, 0)).is_err());
    let duplicate = format!("{raw}data: [DONE]\n\n");
    assert!(std::panic::catch_unwind(|| response(duplicate.as_bytes(), true, model, 0)).is_err());
}
#[test]
fn capacity_token_output_private_empty_and_no_git_alias() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().canonicalize().unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(root(&p).is_ok());
    std::fs::write(p.join("existing"), b"retain").unwrap();
    assert!(root(&p).is_err());
    assert!(root(Path::new("relative")).is_err());
    let git = p.join("repo");
    std::fs::create_dir(&git).unwrap();
    std::fs::write(git.join(".git"), b"fixture").unwrap();
    let nested = git.join("private");
    std::fs::create_dir(&nested).unwrap();
    std::fs::set_permissions(&nested, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(root(&nested).is_err());
}
