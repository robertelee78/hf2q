//! Root-controlled bounded localizer rerun, separate from endpoint trajectories.
//! Uses actual DS stage roots; every CPU read follows the source-owned layer
//! terminal, and cancellation/failed execution never receives a PASS marker.
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
    identity: Identity,
    model: PathBuf,
    output: PathBuf,
    tokens: Vec<u32>,
    configured_capacity: usize,
    expected: Vec<ExpectedTensor>,
    targets: Vec<Target>,
    host_payload_limit: u64,
    rerun_id: String,
}
#[test]
#[ignore = "root-reviewed bounded real DS localizer rerun; no automatic model load"]
fn q1_deepseek_actual_stage_localizer() {
    let path = std::env::var_os("HF2Q_Q1_DS_LOCALIZER_PACKET").expect("explicit localizer packet");
    let mut raw = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .take(8 * 1024 * 1024 + 1)
        .read_to_end(&mut raw)
        .unwrap();
    assert!(raw.len() <= 8 * 1024 * 1024);
    let packet: Packet = serde_json::from_slice(&raw).unwrap();
    assert!(packet.execution_approved);
    assert_eq!(packet.identity.family, Family::Deepseek4);
    packet.identity.validate().unwrap();
    assert!(matches!(packet.tokens.len(), 64 | 128));
    assert!((144..=262144).contains(&packet.configured_capacity));
    assert!(!packet.rerun_id.trim().is_empty());
    assert!(packet.host_payload_limit <= 1024 * 1024 * 1024);
    assert!(!packet.targets.is_empty() && packet.targets.len() <= 4096);
    let mut targets = BTreeMap::new();
    for target in packet.targets {
        assert_eq!(target.source_step, 0, "fresh localizer source position");
        assert_eq!(target.target.at.stage, target.source_stage);
        assert!(targets
            .insert(
                (target.source_step, target.source_layer, target.source_stage),
                target.target
            )
            .is_none());
    }
    let records = packet.expected.len();
    let mut store = Store::new(
        &packet.output,
        packet.identity,
        packet.rerun_id.clone(),
        packet.expected,
        packet.host_payload_limit,
        records,
    )
    .unwrap();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let gguf = mlx_native::gguf::GgufFile::open(&packet.model).unwrap();
    let mut model =
        crate::inference::models::deepseek4::Deepseek4Model::load_from_gguf(&gguf).unwrap();
    assert!(packet.tokens.iter().all(|&t| t < model.cfg.vocab_size));
    if targets.keys().any(|(_, layer, stage)| {
        *layer >= model.cfg.hash_layer_count
            && matches!(
                stage,
                Stage::RouterLogits | Stage::RouterIds | Stage::RouterWeights
            )
    }) {
        store.missing_capability("DS score-routing transformed sqrtsoftplus+bias kth+1 GPU margin is not a published capture output; raw logits/bias/IDs/weights do not self-attest that margin");
    }
    let mut cache = model.allocate_cache(packet.configured_capacity).unwrap();
    family::install(Family::Deepseek4, store, targets).unwrap();
    let result = model.forward_verifier_prefill(&packet.tokens, &mut cache);
    let trace = family::finish().unwrap();
    result.expect("actual localizer transaction must complete");
    assert!(
        trace.coverage.errors.is_empty(),
        "required complete stage coverage missing: {:?}",
        trace.coverage.errors
    );
    let evidence = serde_json::json!({"schema":"hf2q.adr063.q1.stage-localizer.v1","packet_sha256":hash(&raw),
        "route":"ActualDSExistingSingleLayerDiagnosticRoute","rerun_id":packet.rerun_id,"tokens":packet.tokens.len(),
        "records":trace.coverage.captured.len(),"cursor":cache.position(),"whole_q1_qualified":false,
        "serving_grouped_route_equivalence":"NotEstablishedByDiagnosticSplit"});
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(packet.output.join("execution.json"))
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(&evidence).unwrap())
        .unwrap();
    eprintln!(
        "Q1_DS_ACTUAL_STAGE_LOCALIZER_COMPLETE records={} whole_q1_qualified=false",
        trace.coverage.captured.len()
    );
}
