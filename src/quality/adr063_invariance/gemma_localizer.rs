//! Actual Gemma Hybrid slot path, not a Qwen/DS approximation. Root must bind
//! an independently reviewed environment/resource packet before real execution.
//!
//! `prefill_chunks` selects the prompt route: `[128]` (All128, the T0 route)
//! or `[64,64]` (the T3 route: the second chunk resumes through the same
//! `forward_prefill_with_soft_tokens_slot_aware_resume` call `gemma_actual`
//! uses). Each chunk and decode commits the slot cursor exactly as
//! `gemma_actual` does. With `normalize_rows` targets keyed at every chunk
//! start, both routes emit the same per-row coordinates, so `exact` on the two
//! stores localizes the first differing (layer, stage). This single-lane route
//! has no packed T2 (cohort) leg; that is `gemma_cohort_localizer`.
use super::{
    family,
    receipt::*,
    store::{hash, Store},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::PathBuf,
};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Target {
    pub(super) source_step: u64,
    pub(super) source_layer: u32,
    pub(super) source_stage: Stage,
    pub(super) target: CaptureTarget,
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
/// Localizer packets must keep `host_payload_limit` at or below this.
pub(super) const HOST_PAYLOAD_CAP: u64 = 1024 * 1024 * 1024;
/// The one Gemma stage-localizer config recipe. The localizer checks it on the
/// header config before load and on the loaded config after load; the metadata
/// prepare emits its digest. `cfg` has `max_position_embeddings` = `context`.
pub(super) fn localizer_recipe(
    cfg: &crate::serve::config::Gemma4Config,
    context: u32,
) -> super::acquisition::ConfigRecipe<'static> {
    super::acquisition::ConfigRecipe {
        family: Family::Gemma4,
        capacity: context as u64,
        slots: 4,
        output_allowance: 8192,
        cache_mode: "GemmaHybrid",
        route: "stage-localizer",
        actual_model_config_sha256: hash(format!("{cfg:?}").as_bytes()),
    }
}
/// The roles `gemma_prefill::after_layer` hands to the capture at the batched
/// prefill per-layer terminal (`forward_prefill_batched.rs`), in that order.
pub(super) const PREFILL_STAGES: [Stage; 14] = [
    Stage::Query,
    Stage::Key,
    Stage::Value,
    Stage::RopeQuery,
    Stage::RopeKey,
    Stage::Attention,
    Stage::Residual,
    Stage::FfnNorm,
    Stage::RouterLogits,
    Stage::RouterIds,
    Stage::RouterWeights,
    Stage::DenseOutput,
    Stage::MoeOutput,
    Stage::LayerOutput,
];
/// Scalar and per-row axes of one prefill role: the `vec![seq_len, ..]` shape
/// passed to `gemma_prefill::after_layer` without its leading row axis. The
/// batched prefill takes `nh`, `hd`, `nkv`, `top_k` from the same config.
pub(super) fn prefill_row(
    cfg: &crate::serve::config::Gemma4Config,
    layer: usize,
    stage: Stage,
) -> Option<(Scalar, Vec<usize>)> {
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim_for_layer(layer);
    let nkv = cfg.num_kv_heads_for_layer(layer);
    Some(match stage {
        Stage::Query | Stage::RopeQuery | Stage::Attention => (Scalar::F32, vec![nh, hd]),
        Stage::Key | Stage::Value | Stage::RopeKey => (Scalar::F32, vec![nkv, hd]),
        Stage::Residual
        | Stage::FfnNorm
        | Stage::DenseOutput
        | Stage::MoeOutput
        | Stage::LayerOutput => (Scalar::F32, vec![cfg.hidden_size]),
        Stage::RouterLogits => (Scalar::F32, vec![cfg.num_experts]),
        Stage::RouterIds => (Scalar::U32, vec![cfg.top_k_experts]),
        Stage::RouterWeights => (Scalar::F32, vec![cfg.top_k_experts]),
        _ => return None,
    })
}
/// `(targets, expected, planned payload bytes)` of one localizer route.
pub(super) type Manifest = (Vec<Target>, Vec<ExpectedTensor>, u64);
/// Targets keyed at every chunk start of `chunks` (the prefill hook's step is
/// the chunk's start position), each writing `logical_step` 0 and
/// `logical_row_start` = chunk start. With `normalize_rows` the `expected`
/// list is one row per prompt position in layer -> stage -> row order, so it
/// is identical for every route that covers the 128-row prompt.
pub(super) fn localizer_manifest(
    cfg: &crate::serve::config::Gemma4Config,
    agent: usize,
    chunks: &[usize],
    layers: &[u32],
    stages: &[Stage],
    normalize_rows: bool,
) -> Result<Manifest, String> {
    validate_prefill_chunks(chunks)?;
    if agent >= 4 {
        return Err("localizer agent must be one of the four workload lanes".into());
    }
    let mut layers = layers.to_vec();
    layers.sort_unstable();
    if layers.is_empty()
        || layers.windows(2).any(|w| w[0] == w[1])
        || layers.iter().any(|&l| l as usize >= cfg.num_hidden_layers)
    {
        return Err("localizer layers must be distinct actual Gemma layers".into());
    }
    let ordered: Vec<Stage> = PREFILL_STAGES
        .into_iter()
        .filter(|stage| stages.contains(stage))
        .collect();
    if stages.is_empty() || ordered.len() != stages.len() {
        return Err("localizer stages must be distinct Gemma batched-prefill roles".into());
    }
    let label = format!("agent-{agent}");
    let at = |layer: u32, stage: Stage, row: usize| Coordinate {
        logical_step: 0,
        layer,
        stage,
        agent: label.clone(),
        logical_row_start: row as u64,
        segment: 0,
    };
    let spans: Vec<(usize, usize)> = chunks
        .iter()
        .scan(0usize, |next, &len| {
            let start = *next;
            *next += len;
            Some((start, len))
        })
        .collect();
    let mut targets = Vec::new();
    let mut expected = Vec::new();
    for &layer in &layers {
        for &stage in &ordered {
            let (scalar, row) =
                prefill_row(cfg, layer as usize, stage).ok_or("unsupported Gemma prefill role")?;
            for &(start, _) in &spans {
                targets.push(Target {
                    source_step: start as u64,
                    source_layer: layer,
                    source_stage: stage,
                    target: CaptureTarget {
                        at: at(layer, stage, start),
                        normalize_rows,
                    },
                });
            }
            let extents: Vec<(usize, usize)> = if normalize_rows {
                (0..128).map(|r| (r, 1)).collect()
            } else {
                spans.clone()
            };
            for (start, rows) in extents {
                expected.push(ExpectedTensor {
                    at: at(layer, stage, start),
                    scalar,
                    shape: std::iter::once(rows).chain(row.iter().copied()).collect(),
                });
            }
        }
    }
    if targets.len() > 4096 || expected.len() > 65536 {
        return Err("localizer target/record quota exceeded".into());
    }
    let bytes = expected
        .iter()
        .try_fold(0u64, |sum, e| {
            e.shape
                .iter()
                .try_fold(e.scalar.bytes() as u64, |n, &d| n.checked_mul(d as u64))
                .and_then(|n| sum.checked_add(n))
        })
        .ok_or("localizer payload overflow")?;
    if bytes > HOST_PAYLOAD_CAP {
        return Err("localizer payload exceeds the 1 GiB host cap".into());
    }
    Ok((targets, expected, bytes))
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
    assert!(packet.host_payload_limit <= HOST_PAYLOAD_CAP);
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
            &localizer_recipe(&header_cfg, packet.configured_context),
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
            &localizer_recipe(&model.config, packet.configured_context),
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
        "packed_t2_route":"NotThisRoute: gemma_actual::q1_gemma_actual_cohort_stage_localizer"});
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
