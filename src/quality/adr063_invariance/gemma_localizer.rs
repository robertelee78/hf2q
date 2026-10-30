//! Actual Gemma Hybrid slot path, not a Qwen/DS approximation. Root must bind
//! an independently reviewed environment/resource packet before real execution.
use super::{
    family,
    receipt::*,
    store::{hash, Store},
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::PathBuf,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Target {
    source_step: u64,
    source_layer: u32,
    source_stage: Stage,
    target: CaptureTarget,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    execution_approved: bool,
    acquisition: super::acquisition::Binding,
    identity: Identity,
    model: PathBuf,
    output: PathBuf,
    tokens: Vec<u32>,
    workload: super::protocol::Workload,
    agent: usize,
    configured_context: u32,
    configured_output_allowance: usize,
    expected: Vec<ExpectedTensor>,
    targets: Vec<Target>,
    host_payload_limit: u64,
    rerun_id: String,
}
#[test]
#[ignore = "root-controlled real Gemma stage localizer, private resource packet required"]
fn q1_gemma_actual_stage_localizer() {
    let path =
        std::env::var_os("HF2Q_Q1_GEMMA_LOCALIZER_PACKET").expect("explicit Gemma Q1 packet");
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
    assert!(
        packet.agent < 4
            && packet.tokens.len() <= 144
            && packet.tokens == packet.workload.lanes[packet.agent][..packet.tokens.len()]
    );
    assert!(
        packet
            .expected
            .iter()
            .all(|e| e.at.agent == format!("agent-{}", packet.agent)),
        "localizer agent must match actual workload vector"
    );
    assert_eq!(packet.identity.family, Family::Gemma4);
    assert_eq!(
        packet.tokens.len(),
        144,
        "128 prompt +16 forced continuation"
    );
    assert_eq!(packet.configured_context, 262144);
    assert_eq!(packet.configured_output_allowance, 8192);
    assert!(packet.host_payload_limit <= 1024 * 1024 * 1024);
    assert!(!packet.rerun_id.trim().is_empty());
    assert!(!packet.targets.is_empty() && packet.targets.len() <= 4096);
    let mut targets = BTreeMap::new();
    for entry in packet.targets {
        assert_eq!(entry.source_stage, entry.target.at.stage);
        assert!(targets
            .insert(
                (entry.source_step, entry.source_layer, entry.source_stage),
                entry.target
            )
            .is_none());
    }
    let records = packet.expected.len();
    let mut store = Store::new(
        &packet.output,
        packet.identity.clone(),
        packet.rerun_id.clone(),
        packet.expected,
        packet.host_payload_limit,
        records,
    )
    .unwrap();
    if targets.keys().any(|(_, _, stage)| {
        matches!(
            stage,
            Stage::RouterLogits | Stage::RouterIds | Stage::RouterWeights
        )
    }) {
        store.missing_capability("Gemma actual native post-softmax/renormalized selection kth+1 GPU score is not a captured public output; IDs/weights/raw logits are distinct evidence");
    }
    let acquisition_scope = super::acquisition::Scope::begin(
        &packet.acquisition,
        &packet.identity,
        &packet.workload,
        &packet.model,
        &packet.output,
        &raw,
    )
    .unwrap();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    use crate::serve::api::engine::{LoadOptions, LoadedModel};
    let header = super::acquisition::open_model(&packet.model).unwrap();
    assert_eq!(
        header.metadata_string("general.architecture"),
        Some("gemma4"),
        "explicit Gemma4 architecture required"
    );
    let mut header_cfg = crate::serve::config::Gemma4Config::from_gguf(&header).unwrap();
    header_cfg.max_position_embeddings = packet.configured_context as usize;
    acquisition_scope
        .check_config(
            &packet.identity,
            &super::acquisition::ConfigRecipe {
                family: packet.identity.family,
                capacity: packet.configured_context as u64,
                slots: 4,
                output_allowance: 8192,
                cache_mode: "GemmaHybrid",
                route: "stage-localizer",
                actual_model_config_sha256: hash(format!("{:?}", &header_cfg).as_bytes()),
            },
        )
        .unwrap();
    drop(header);
    let options = LoadOptions {
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
    let loaded = LoadedModel::load_with_context(&options, Some(packet.configured_context))
        .expect("actual production family loader");
    let LoadedModel::Gemma(mut model) = loaded else {
        panic!("explicit Gemma artifact required, no family fallback")
    };
    assert!(packet
        .tokens
        .iter()
        .all(|&t| (t as usize) < model.weights.vocab_size));
    acquisition_scope
        .check_config(
            &packet.identity,
            &super::acquisition::ConfigRecipe {
                family: packet.identity.family,
                capacity: packet.configured_context as u64,
                slots: 4,
                output_allowance: 8192,
                cache_mode: "GemmaHybrid",
                route: "stage-localizer",
                actual_model_config_sha256: hash(format!("{:?}", &model.config).as_bytes()),
            },
        )
        .unwrap();
    store
        .bind_acquisition(
            acquisition_scope
                .observed(
                    model.ctx.executor.device(),
                    &model as *const _ as usize as u64,
                    3,
                )
                .unwrap(),
        )
        .unwrap();
    model
        .provision_multi_seq_kv_for_slot_aware(4)
        .expect("actual production four-slot scaffold");
    assert!(
        model.multi_seq_kv_hybrid.is_some(),
        "selected production Hybrid scaffold required"
    );
    let _capture_scope = family::install(Family::Gemma4, store, targets).unwrap();
    use crate::serve::multi_seq_kv::SlotId;
    let result = (|| -> anyhow::Result<()> {
        model.weights.forward_prefill_with_soft_tokens_slot_aware(
            &packet.tokens[..128],
            &[],
            packet.configured_output_allowance,
            &mut model.ctx,
            SlotId(0),
            model
                .multi_seq_kv
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("HB scaffold absent"))?,
            model.multi_seq_kv_hybrid.as_mut(),
            model.multi_seq_kv_dense.as_mut(),
            model.multi_seq_kv_mlx.as_mut(),
        )?;
        for position in 128..144 {
            model.weights.forward_decode_slot_aware(
                packet.tokens[position],
                position,
                &mut model.ctx,
                &mut None,
                SlotId(0),
                model
                    .multi_seq_kv
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("HB scaffold absent"))?,
                model.multi_seq_kv_hybrid.as_mut(),
                model.multi_seq_kv_dense.as_mut(),
                model.multi_seq_kv_mlx.as_mut(),
            )?;
        }
        Ok(())
    })();
    acquisition_scope.check_unchanged().unwrap();
    let trace = family::finish().unwrap();
    result.expect("actual Gemma stage transaction");
    assert!(
        trace.coverage.errors.is_empty(),
        "actual stage coverage incomplete: {:?}",
        trace.coverage.errors
    );
    let evidence = serde_json::json!({"schema":"hf2q.adr063.q1.stage-localizer.v1","packet_sha256":hash(&raw),"rerun_id":packet.rerun_id,
        "route":"GemmaActualHybridSlotAwareWithCheckedDiagnosticLayerSplits","logical_lane":"slot0","declared_slots":4,
        "tokens":144,"records":trace.coverage.captured.len(),"whole_q1_qualified":false,
        "full_prefill_logits":"NotEstablishedByLastRowProductionHead","other_three_lane_execution":"NotClaimedBySingleLaneLocalizer"});
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(packet.output.join("execution.json"))
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(&evidence).unwrap())
        .unwrap();
    eprintln!(
        "Q1_GEMMA_ACTUAL_STAGE_LOCALIZER_COMPLETE records={} whole_q1_qualified=false",
        trace.coverage.captured.len()
    );
}
