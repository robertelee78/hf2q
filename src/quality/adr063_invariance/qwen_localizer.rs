//! Explicit Qwen operation localizer, using the production bisection site's
//! session-aware checked drain. Root supplies source-bound operation names;
//! a typo/missing hook is incomplete coverage, never a successful comparison.
//!
//! `prefill_chunks` selects the prompt route: absent = one forward over all
//! `tokens`; `[64,64]` on 128 tokens is the T3Chunk64 route (`qwen_runner`:
//! forward 0..64, then 64..128 on the same cache). Each forward is one
//! `dump_bisect` step, so targets key the chunk index as `source_step` and
//! write the chunk start as `logical_row_start`; with `normalize_rows` both
//! routes freeze the same `expected` and `exact` localizes between them.
use super::{
    qwen,
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
pub(crate) struct Target {
    pub(crate) source_step: u64,
    pub(crate) source_layer: Option<usize>,
    pub(crate) operation: String,
    pub(crate) target: CaptureTarget,
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
    configured_capacity: u32,
    expected: Vec<ExpectedTensor>,
    targets: Vec<Target>,
    host_payload_limit: u64,
    rerun_id: String,
    /// Prompt chunk lengths summing to `tokens.len()`; absent = one forward.
    #[serde(default)]
    prefill_chunks: Option<Vec<usize>>,
}
/// Localizer packets must keep `host_payload_limit` at or below this.
pub(crate) const HOST_PAYLOAD_CAP: u64 = 1024 * 1024 * 1024;
/// The one Qwen stage-localizer config recipe. The localizer checks it on the
/// un-overridden header config before load and on the loaded config after
/// load; the endpoint metadata prepare emits its digest.
pub(crate) fn localizer_recipe(
    family: Family,
    capacity: u32,
    cfg: &crate::inference::models::qwen35::Qwen35Config,
) -> super::acquisition::ConfigRecipe<'static> {
    super::acquisition::ConfigRecipe {
        family,
        capacity: capacity as u64,
        slots: 1,
        output_allowance: 0,
        cache_mode: "QwenTQ8",
        route: "stage-localizer",
        actual_model_config_sha256: hash(format!("{cfg:?}").as_bytes()),
    }
}
pub(crate) fn validate_prefill_chunks(chunks: &[usize], tokens: usize) -> Result<(), String> {
    if !matches!(tokens, 64 | 128)
        || chunks.is_empty()
        || chunks.contains(&0)
        || chunks.iter().sum::<usize>() != tokens
    {
        return Err(format!(
            "prefill_chunks {chunks:?} must be positive lengths summing to the 64/128 tokens"
        ));
    }
    Ok(())
}
/// Operations `forward_gpu_impl` hands to `dump_bisect::dump` on every
/// forward call, each a `[seq_len, hidden_size]` F32 role, with the stage
/// their coordinates carry. `embed` is layer-less (`None`, coordinate layer
/// `u32::MAX`); the other three are per layer, in execution order.
pub(crate) const PREFILL_OPERATIONS: [(&str, Stage); 4] = [
    ("embed", Stage::Input),
    ("attn_out", Stage::Attention),
    ("ffn_out", Stage::FfnWriter),
    ("layer_out", Stage::LayerOutput),
];
/// `(targets, expected, planned payload bytes)` of one localizer route.
pub(crate) type Manifest = (Vec<Target>, Vec<ExpectedTensor>, u64);
/// Targets keyed `(chunk index, layer, operation)`, each writing
/// `logical_step` 0 and `logical_row_start` = chunk start. With
/// `normalize_rows` the `expected` list is one row per prompt position in
/// embed -> layer -> operation -> row order, identical across routes.
pub(crate) fn localizer_manifest(
    cfg: &crate::inference::models::qwen35::Qwen35Config,
    agent: usize,
    chunks: &[usize],
    layers: &[usize],
    operations: &[String],
    normalize_rows: bool,
) -> Result<Manifest, String> {
    let tokens = chunks.iter().sum::<usize>();
    validate_prefill_chunks(chunks, tokens)?;
    if agent >= 4 {
        return Err("localizer agent must be one of the four workload lanes".into());
    }
    let ordered: Vec<(&str, Stage)> = PREFILL_OPERATIONS
        .into_iter()
        .filter(|(name, _)| operations.iter().any(|o| o.as_str() == *name))
        .collect();
    if operations.is_empty() || ordered.len() != operations.len() {
        return Err("localizer operations must be distinct Qwen prefill dump operations".into());
    }
    let mut layers = layers.to_vec();
    layers.sort_unstable();
    let per_layer = ordered.iter().any(|(name, _)| *name != "embed");
    if layers.windows(2).any(|w| w[0] == w[1])
        || layers.iter().any(|&l| l >= cfg.num_hidden_layers as usize)
        || layers.is_empty() == per_layer
    {
        return Err("localizer layers must be distinct Qwen layers for per-layer ops".into());
    }
    let mut scopes: Vec<(Option<usize>, &str, Stage)> = Vec::new();
    if let Some(&(name, stage)) = ordered.iter().find(|(name, _)| *name == "embed") {
        scopes.push((None, name, stage));
    }
    for &layer in &layers {
        for &(name, stage) in ordered.iter().filter(|(name, _)| *name != "embed") {
            scopes.push((Some(layer), name, stage));
        }
    }
    let label = format!("agent-{agent}");
    let at = |layer: Option<usize>, stage: Stage, row: usize| Coordinate {
        logical_step: 0,
        layer: layer.map_or(u32::MAX, |l| l as u32),
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
    let hidden = cfg.hidden_size as usize;
    let mut targets = Vec::new();
    let mut expected = Vec::new();
    for (layer, name, stage) in scopes {
        for (step, &(start, _)) in spans.iter().enumerate() {
            targets.push(Target {
                source_step: step as u64,
                source_layer: layer,
                operation: name.into(),
                target: CaptureTarget {
                    at: at(layer, stage, start),
                    normalize_rows,
                },
            });
        }
        let extents: Vec<(usize, usize)> = if normalize_rows {
            (0..tokens).map(|r| (r, 1)).collect()
        } else {
            spans.clone()
        };
        for (start, rows) in extents {
            expected.push(ExpectedTensor {
                at: at(layer, stage, start),
                scalar: Scalar::F32,
                shape: vec![rows, hidden],
            });
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
#[ignore = "root-reviewed real Q38/Q36 localizer; does not run in hosted tests"]
fn q1_qwen_actual_stage_localizer() {
    let path =
        std::env::var_os("HF2Q_Q1_QWEN_LOCALIZER_PACKET").expect("explicit localizer packet");
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
    assert!(matches!(
        packet.identity.family,
        Family::Qwen38 | Family::Qwen36
    ));
    assert!(matches!(packet.tokens.len(), 64 | 128));
    let tokens_len = packet.tokens.len();
    let prefill_chunks = packet
        .prefill_chunks
        .clone()
        .unwrap_or_else(|| vec![tokens_len]);
    validate_prefill_chunks(&prefill_chunks, tokens_len).unwrap();
    assert!((144..=262144).contains(&packet.configured_capacity));
    assert!(
        !packet.rerun_id.trim().is_empty()
            && packet.targets.len() <= 4096
            && !packet.targets.is_empty()
    );
    assert!(packet.host_payload_limit <= HOST_PAYLOAD_CAP);
    let mut targets = BTreeMap::new();
    for target in packet.targets {
        assert!((target.source_step as usize) < prefill_chunks.len());
        assert!(!target.operation.is_empty() && target.operation.len() <= 128);
        assert!(targets
            .insert(
                (target.source_step, target.source_layer, target.operation),
                target.target
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
    let gguf = super::acquisition::open_model(&packet.model).unwrap();
    assert_eq!(
        gguf.metadata_string("general.architecture"),
        Some(match packet.identity.family {
            Family::Qwen38 => "qwen35",
            Family::Qwen36 => "qwen35moe",
            _ => unreachable!(),
        }),
        "exact selected architecture before load"
    );
    let header_cfg = crate::inference::models::qwen35::Qwen35Config::from_gguf(&gguf).unwrap();
    acquisition_scope
        .check_config(
            &packet.identity,
            &localizer_recipe(packet.identity.family, packet.configured_capacity, &header_cfg),
        )
        .unwrap();
    let mut progress = crate::serve::header::LoadProgress::new(false, 1, 0);
    let model =
        crate::inference::models::qwen35::model::Qwen35Model::load_from_gguf(&gguf, &mut progress)
            .unwrap();
    use crate::inference::models::qwen35::{
        dump_bisect, kv_cache::HybridKvCache, spec_decode::positions_for_range, Qwen35Variant,
    };
    assert!(matches!(
        (packet.identity.family, model.cfg.variant),
        (Family::Qwen38, Qwen35Variant::Dense) | (Family::Qwen36, Qwen35Variant::Moe)
    ));
    assert!(packet.tokens.iter().all(|&t| t < model.cfg.vocab_size));
    acquisition_scope
        .check_config(
            &packet.identity,
            &localizer_recipe(packet.identity.family, packet.configured_capacity, &model.cfg),
        )
        .unwrap();
    model.ensure_gpu_cache_primed().unwrap();
    let mut cache = model
        .with_gpu_cache_mut(|device, _| {
            HybridKvCache::new_with_options(&model.cfg, device, packet.configured_capacity, 1, true)
        })
        .unwrap();
    if model.cfg.variant == Qwen35Variant::Moe
        && targets
            .keys()
            .any(|(_, _, op)| op.starts_with("moe_router_"))
    {
        store.missing_capability("Qwen native router kth+1 selection-domain score is not exposed as captured output; raw logits/selected IDs/weights are separate evidence");
    }
    let observed = model
        .with_gpu_cache_mut(|device, _| {
            acquisition_scope.observed(device, &model as *const _ as usize as u64, 1)
        })
        .unwrap();
    store.bind_acquisition(observed).unwrap();
    dump_bisect::reset_step();
    let _capture_scope = qwen::install(store, targets).unwrap();
    let result = (|| -> anyhow::Result<()> {
        use crate::serve::multi_seq_kv::SlotId;
        let mut cached = 0usize;
        for &chunk in &prefill_chunks {
            anyhow::ensure!(
                cache.sequence_len_for_slot(SlotId(0))? as usize == cached,
                "Qwen localizer chunk starts at the wrong logical position"
            );
            let end = cached + chunk;
            model.forward_gpu(
                &packet.tokens[cached..end],
                &positions_for_range(i32::try_from(cached)?, chunk),
                &mut cache,
                SlotId(0),
            )?;
            cached = end;
        }
        anyhow::ensure!(
            cache.sequence_len_for_slot(SlotId(0))? as usize == cached,
            "Qwen localizer terminal cursor"
        );
        Ok(())
    })();
    acquisition_scope.check_unchanged().unwrap();
    let trace = qwen::finish().unwrap();
    result.expect("actual Qwen localizer forward");
    assert!(
        trace.coverage.errors.is_empty(),
        "required complete operation coverage missing: {:?}",
        trace.coverage.errors
    );
    let evidence = serde_json::json!({"schema":"hf2q.adr063.q1.stage-localizer.v1","packet_sha256":hash(&raw),"rerun_id":packet.rerun_id,
        "route":"QwenActualSessionAwareBisectionDiagnostic","prefill_chunks":prefill_chunks,"tokens":packet.tokens.len(),"records":trace.coverage.captured.len(),
        "whole_q1_qualified":false,"serving_overlap_equivalence":"NotEstablishedByDiagnosticDrain"});
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(packet.output.join("execution.json"))
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(&evidence).unwrap())
        .unwrap();
    eprintln!(
        "Q1_QWEN_ACTUAL_STAGE_LOCALIZER_COMPLETE records={} whole_q1_qualified=false",
        trace.coverage.captured.len()
    );
}
