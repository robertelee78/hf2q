//! Actual Gemma Hybrid slot path, not a Qwen/DS approximation. Root must bind
//! an independently reviewed environment/resource packet before real execution.
//!
//! `prefill_chunks` selects the prompt route: `[128]` (All128, the T0 route)
//! or `[64,64]` (the T3 route: the second chunk resumes through the same
//! `forward_prefill_with_soft_tokens_slot_aware_resume` call `gemma_actual`
//! uses). Each chunk and decode commits the slot cursor exactly as
//! `gemma_actual` does. With `normalize_rows` targets keyed at every chunk
//! start, both routes emit the same per-row coordinates, so `exact` on the two
//! stores localizes the first differing (layer, stage). There is no packed
//! T2 (cohort) route here: T2 stage localization is recorded as Missing.
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
    /// Prompt chunk lengths; must sum to 128. Default All128.
    #[serde(default = "all128")]
    prefill_chunks: Vec<usize>,
    /// Warm live-resume kernel the loaded model must resolve; absent = VecHybrid.
    #[serde(default)]
    warm_live_kernel: super::gemma_actual::WarmLiveKernel,
}
fn all128() -> Vec<usize> {
    vec![128]
}
fn validate_prefill_chunks(chunks: &[usize]) -> Result<(), String> {
    if chunks.is_empty() || chunks.contains(&0) || chunks.iter().sum::<usize>() != 128 {
        return Err(format!("prefill_chunks {chunks:?} must be non-empty positive lengths summing to 128"));
    }
    Ok(())
}
#[test]
fn q1_gemma_localizer_prefill_chunks_must_cover_the_128_prompt() {
    assert!(validate_prefill_chunks(&all128()).is_ok());
    assert!(validate_prefill_chunks(&[64, 64]).is_ok());
    assert!(validate_prefill_chunks(&[64, 63]).is_err());
    assert!(validate_prefill_chunks(&[64, 65]).is_err());
    assert!(validate_prefill_chunks(&[0, 128]).is_err());
    assert!(validate_prefill_chunks(&[]).is_err());
    let packet: serde_json::Value = serde_json::json!({"prefill_chunks": [64, 64]});
    let chunks: Vec<usize> = serde_json::from_value(packet["prefill_chunks"].clone()).unwrap();
    assert_eq!(chunks, vec![64, 64]);
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
    validate_prefill_chunks(&packet.prefill_chunks).unwrap();
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
    let warm_live_kernel =
        super::gemma_actual::require_warm_kernel(packet.warm_live_kernel, model.weights.warm_live_kernel)
            .unwrap();
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
    let slot = SlotId(0);
    let result = (|| -> anyhow::Result<()> {
        let mut cached = 0usize;
        for &chunk in &packet.prefill_chunks {
            let end = cached + chunk;
            let hb = model
                .multi_seq_kv
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("HB scaffold absent"))?;
            if cached == 0 {
                model.weights.forward_prefill_with_soft_tokens_slot_aware(
                    &packet.tokens[..end],
                    &[],
                    packet.configured_output_allowance,
                    &mut model.ctx,
                    slot,
                    hb,
                    model.multi_seq_kv_hybrid.as_mut(),
                    model.multi_seq_kv_dense.as_mut(),
                    model.multi_seq_kv_mlx.as_mut(),
                )?;
            } else {
                model.weights.forward_prefill_with_soft_tokens_slot_aware_resume(
                    &packet.tokens[..end],
                    &[],
                    packet.configured_output_allowance,
                    &mut model.ctx,
                    slot,
                    hb,
                    model.multi_seq_kv_hybrid.as_mut(),
                    model.multi_seq_kv_dense.as_mut(),
                    model.multi_seq_kv_mlx.as_mut(),
                    cached,
                )?;
            }
            crate::serve::api::engine::q1_gemma_cohort::commit_after_checked_return(
                &mut model, slot, cached, end,
            )?;
            cached = end;
        }
        for position in 128..144 {
            model.weights.forward_decode_slot_aware(
                packet.tokens[position],
                position,
                &mut model.ctx,
                &mut None,
                slot,
                model
                    .multi_seq_kv
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("HB scaffold absent"))?,
                model.multi_seq_kv_hybrid.as_mut(),
                model.multi_seq_kv_dense.as_mut(),
                model.multi_seq_kv_mlx.as_mut(),
            )?;
            crate::serve::api::engine::q1_gemma_cohort::commit_after_checked_return(
                &mut model, slot, position, position + 1,
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
        "route":"GemmaActualHybridSlotAwareWithCheckedDiagnosticLayerSplits","prefill_chunks":packet.prefill_chunks,"warm_live_kernel":warm_live_kernel,
        "logical_lane":"slot0","declared_slots":4,
        "tokens":144,"records":trace.coverage.captured.len(),"whole_q1_qualified":false,
        "full_prefill_logits":"NotEstablishedByLastRowProductionHead","other_three_lane_execution":"NotClaimedBySingleLaneLocalizer",
        "packed_t2_route":"Missing: no packed-cohort stage localizer"});
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
