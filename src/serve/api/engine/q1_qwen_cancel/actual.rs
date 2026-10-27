//! Original64-boundary helper body; parent filter is preserved unchanged.
use super::*;
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    execution_approved: bool,
    acquisition: crate::adr063_invariance::acquisition::Binding,
    model: PathBuf,
    output: PathBuf,
    identity: crate::adr063_invariance::receipt::Identity,
    workload: crate::adr063_invariance::protocol::Workload,
    configured_context: u32,
    expected_cache: Vec<crate::adr063_invariance::receipt::ExpectedTensor>,
    host_payload_limit: u64,
}
pub(super) fn run() {
    use crate::adr063_invariance::{
        compare, qwen_cache,
        receipt::*,
        store::{hash, Store},
    };
    use std::{
        io::{Read, Write},
        os::unix::fs::{OpenOptionsExt, PermissionsExt},
    };
    let path = std::env::var_os("HF2Q_Q1_QWEN_ENGINE_CANCEL_PACKET")
        .expect("explicit engine cancellation packet");
    let mut raw = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .take(8 * 1024 * 1024 + 1)
        .read_to_end(&mut raw)
        .unwrap();
    assert!(raw.len() <= 8 * 1024 * 1024);
    let packet: Packet = serde_json::from_slice(&raw).unwrap();
    assert!(packet.execution_approved);
    packet.identity.validate().unwrap();
    packet.workload.validate().unwrap();
    assert_eq!(packet.identity.family, packet.workload.family);
    assert!(matches!(
        packet.identity.family,
        Family::Qwen38 | Family::Qwen36
    ));
    assert_eq!(packet.configured_context, 262144);
    assert!(packet.host_payload_limit <= 1024 * 1024 * 1024);
    assert!(packet
        .expected_cache
        .iter()
        .all(|e| e.at.logical_step == 64));
    assert!(packet.output.is_absolute() && !packet.output.exists());
    assert_eq!(
        packet.output.parent().unwrap().canonicalize().unwrap(),
        packet.output.parent().unwrap()
    );
    assert!(!packet.output.ancestors().any(|p| p.join(".git").exists()));
    std::fs::create_dir(&packet.output).unwrap();
    std::fs::set_permissions(&packet.output, std::fs::Permissions::from_mode(0o700)).unwrap();
    let before_root = packet.output.join("before");
    let after_root = packet.output.join("recovered");
    let records = packet.expected_cache.len();
    let mut before = Store::new(
        &before_root,
        packet.identity.clone(),
        "T6Checkpoint64".into(),
        packet.expected_cache.clone(),
        packet.host_payload_limit,
        records,
    )
    .unwrap();
    let mut after = Store::new(
        &after_root,
        packet.identity.clone(),
        "T6ActualEngineRecovered64".into(),
        packet.expected_cache.clone(),
        packet.host_payload_limit,
        records,
    )
    .unwrap();
    let acquisition_scope = crate::adr063_invariance::acquisition::Scope::begin(
        &packet.acquisition,
        &packet.identity,
        &packet.workload,
        &packet.model,
        &packet.output,
        &raw,
    )
    .unwrap();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let gguf = crate::adr063_invariance::acquisition::open_model(&packet.model).unwrap();
    assert_eq!(
        gguf.metadata_string("general.architecture"),
        Some(match packet.identity.family {
            Family::Qwen38 => "qwen35",
            Family::Qwen36 => "qwen35moe",
            _ => unreachable!(),
        })
    );
    let mut header_cfg = crate::inference::models::qwen35::Qwen35Config::from_gguf(&gguf).unwrap();
    header_cfg.max_position_embeddings = packet.configured_context;
    acquisition_scope
        .check_config(
            &packet.identity,
            &crate::adr063_invariance::acquisition::ConfigRecipe {
                family: packet.identity.family,
                capacity: packet.configured_context as u64,
                slots: 1,
                output_allowance: 8192,
                cache_mode: "QwenTQ8",
                route: "owning-engine-cancel64",
                actual_model_config_sha256: hash(format!("{:?}", &header_cfg).as_bytes()),
            },
        )
        .unwrap();
    drop(gguf);
    let opts = LoadOptions {
        model_path: packet.model,
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
        glp_path: None,
        glp_alpha: None,
        kv_graft_path: None,
    };
    let loaded = LoadedModel::load_with_context(&opts, Some(packet.configured_context)).unwrap();
    let LoadedModel::Qwen35(mut model) = loaded else {
        panic!("Qwen family fallback refused")
    };
    acquisition_scope
        .check_config(
            &packet.identity,
            &crate::adr063_invariance::acquisition::ConfigRecipe {
                family: packet.identity.family,
                capacity: packet.configured_context as u64,
                slots: 1,
                output_allowance: 8192,
                cache_mode: "QwenTQ8",
                route: "owning-engine-cancel64",
                actual_model_config_sha256: hash(format!("{:?}", &model.model.cfg).as_bytes()),
            },
        )
        .unwrap();
    let mut canonical_cache = Vec::new();
    for lane in 0..4 {
        canonical_cache.extend(
            qwen_cache::manifest(
                &model.model.cfg,
                packet.configured_context,
                &format!("agent-{lane}"),
                64,
                0,
            )
            .unwrap(),
        );
    }
    assert_eq!(
        packet.expected_cache, canonical_cache,
        "every active checkpoint cache role is mandatory"
    );
    model.model.ensure_gpu_cache_primed().unwrap();
    let observed = model
        .model
        .with_gpu_cache_mut(|device, _| {
            acquisition_scope.observed(device, &model as *const _ as usize as u64, 3)
        })
        .unwrap();
    let mut before_observed = observed.clone();
    before_observed.output_root = before_root.clone();
    let mut after_observed = observed;
    after_observed.output_root = after_root.clone();
    // Two checkpoints from ONE acquisition: exact recovery comparison is a
    // within-run state check, never cross-trajectory model qualification.
    before.bind_acquisition(before_observed).unwrap();
    after.bind_acquisition(after_observed).unwrap();

    model.provision_multi_seq_kv_for_slot_aware(1).unwrap();
    let layers = model.model.cfg.num_hidden_layers;
    let mut receipts = Vec::new();
    for lane in 0..4 {
        let result = run_loaded(
            &mut model,
            &packet.workload.lanes[lane][..128],
            SamplingParams {
                max_tokens: 8192,
                ..SamplingParams::default()
            },
            |phase, cache| {
                let store = match phase {
                    Phase::Checkpoint64 => &mut before,
                    Phase::Recovered64 => &mut after,
                };
                qwen_cache::after_checked_terminal(
                    store,
                    cache,
                    layers,
                    &format!("agent-{lane}"),
                    64,
                )
            },
        );
        match result {
            Ok(receipt) => receipts.push(receipt),
            Err(e) => {
                before.fail(format!("actual engine cancel: {e:#}"));
                after.fail(format!("actual engine cancel: {e:#}"));
                before.finish().unwrap();
                after.finish().unwrap();
                panic!("Q1 engine cancellation incomplete: {e:#}")
            }
        }
        // Each localizer lane owns the slot only until run_loaded returns;
        // all actual work is drained and its old anchor/work are dropped.
        model
            .persistent_kv_cache
            .as_mut()
            .unwrap()
            .reset_for_slot(SlotId(0))
            .unwrap();
    }
    acquisition_scope.check_unchanged().unwrap();
    let left = before.finish().unwrap();
    let right = after.finish().unwrap();
    let policies = left
        .coverage
        .expected
        .iter()
        .map(|e| (e.at.stage, NumericPolicy::Exact))
        .collect();
    let comparison = compare::compare_recovery(&before_root, &left, &after_root, &right, &policies);
    assert_eq!(
        comparison.disposition,
        Disposition::ExactEqual,
        "actual cancellation cache mismatch"
    );
    let result = serde_json::json!({"schema":"hf2q.adr063.q1.qwen-engine-cancel.v1","packet_sha256":hash(&raw),"lanes":receipts,
        "comparison":comparison,"full_144_token_trajectory":false,"whole_q1_qualified":false});
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(packet.output.join("execution.json"))
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(&result).unwrap())
        .unwrap();
    eprintln!("Q1_QWEN_ACTUAL_ENGINE_CANCEL_CACHE_COMPLETE lanes=4 recovered=64 full_144_token_trajectory=false whole_q1_qualified=false");
}
