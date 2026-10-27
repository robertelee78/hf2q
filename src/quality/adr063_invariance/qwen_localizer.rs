//! Explicit Qwen operation localizer, using the production bisection site's
//! session-aware checked drain. Root supplies source-bound operation names;
//! a typo/missing hook is incomplete coverage, never a successful comparison.
use super::{
    qwen,
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
    source_layer: Option<usize>,
    operation: String,
    target: CaptureTarget,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    execution_approved: bool,
    identity: Identity,
    model: PathBuf,
    output: PathBuf,
    tokens: Vec<u32>,
    configured_capacity: u32,
    expected: Vec<ExpectedTensor>,
    targets: Vec<Target>,
    host_payload_limit: u64,
    rerun_id: String,
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
    assert!(matches!(
        packet.identity.family,
        Family::Qwen38 | Family::Qwen36
    ));
    assert!(matches!(packet.tokens.len(), 64 | 128));
    assert!((144..=262144).contains(&packet.configured_capacity));
    assert!(
        !packet.rerun_id.trim().is_empty()
            && packet.targets.len() <= 4096
            && !packet.targets.is_empty()
    );
    assert!(packet.host_payload_limit <= 1024 * 1024 * 1024);
    let mut targets = BTreeMap::new();
    for target in packet.targets {
        assert_eq!(target.source_step, 0);
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
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let gguf = mlx_native::gguf::GgufFile::open(&packet.model).unwrap();
    assert_eq!(
        gguf.metadata_str("general.architecture"),
        Some(match packet.identity.family {
            Family::Qwen38 => "qwen35",
            Family::Qwen36 => "qwen35moe",
            _ => unreachable!(),
        }),
        "exact selected architecture before load"
    );
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
    dump_bisect::reset_step();
    qwen::install(store, targets).unwrap();
    let result = model.forward_gpu(
        &packet.tokens,
        &positions_for_range(0, packet.tokens.len()),
        &mut cache,
        crate::serve::multi_seq_kv::SlotId(0),
    );
    let trace = qwen::finish().unwrap();
    result.expect("actual Qwen localizer forward");
    assert!(
        trace.coverage.errors.is_empty(),
        "required complete operation coverage missing: {:?}",
        trace.coverage.errors
    );
    let evidence = serde_json::json!({"schema":"hf2q.adr063.q1.stage-localizer.v1","packet_sha256":hash(&raw),"rerun_id":packet.rerun_id,
        "route":"QwenActualSessionAwareBisectionDiagnostic","tokens":packet.tokens.len(),"records":trace.coverage.captured.len(),
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
