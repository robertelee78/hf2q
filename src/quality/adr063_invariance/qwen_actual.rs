//! Real Q38/Q36 endpoint trace, root-controlled and ignored by default. This
//! captures actual native-download rows, not internal-stage equivalence.
use super::{
    protocol::{Trajectory, Workload},
    qwen_runner,
    receipt::*,
    store::{hash, Store},
};
use serde::Deserialize;
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::PathBuf,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    execution_approved: bool,
    model: PathBuf,
    output: PathBuf,
    identity: Identity,
    workload: Workload,
    trajectory: Trajectory,
    configured_capacity: u32,
    requested_payload_limit: u64,
    expected_cache: Vec<ExpectedTensor>,
}
#[test]
#[ignore = "root-controlled real Qwen model; immutable packet and independent source/resource review required"]
fn q1_qwen_actual_endpoint_trace() {
    assert!(
        cfg!(target_endian = "little"),
        "raw diagnostic encoding requires little-endian host"
    );
    let path = std::env::var_os("HF2Q_Q1_QWEN_PACKET").expect("explicit Q1 packet");
    let mut raw = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .take(1024 * 1024 + 1)
        .read_to_end(&mut raw)
        .unwrap();
    assert!(raw.len() <= 1024 * 1024);
    let packet: Packet = serde_json::from_slice(&raw).unwrap();
    assert!(packet.execution_approved);
    packet.identity.validate().unwrap();
    packet.workload.validate().unwrap();
    assert_eq!(packet.identity.family, packet.workload.family);
    assert!(matches!(
        packet.workload.family,
        Family::Qwen38 | Family::Qwen36
    ));
    assert!((144..=262144).contains(&packet.configured_capacity));
    assert!(packet.requested_payload_limit <= 1024 * 1024 * 1024);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let gguf = mlx_native::gguf::GgufFile::open(&packet.model).unwrap();
    assert_eq!(
        gguf.metadata_str("general.architecture"),
        Some(match packet.workload.family {
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
    assert_eq!(model.cfg.vocab_size, packet.workload.vocab_size);
    let layer = model.cfg.num_hidden_layers;
    let vocab = model.cfg.vocab_size as usize;
    let mut expected = Vec::new();
    for lane in 0..4 {
        for row in 0..144 {
            expected.push(ExpectedTensor {
                at: coordinate(lane, row, layer),
                scalar: Scalar::F32,
                shape: vec![1, vocab],
            });
        }
    }
    assert!(
        packet
            .expected_cache
            .iter()
            .all(|e| matches!(e.at.logical_step, 128 | 144)),
        "cache packet boundaries must be128/144"
    );
    let capture_cache = !packet.expected_cache.is_empty();
    expected.extend(packet.expected_cache);
    let record_limit = expected.len();
    let mut store = Store::new(
        &packet.output,
        packet.identity,
        packet.trajectory.label().into(),
        expected,
        packet.requested_payload_limit,
        record_limit,
    )
    .unwrap();
    let result = qwen_runner::run(
        &model,
        &packet.workload,
        packet.trajectory,
        packet.configured_capacity,
        |lane, end, values, cache| {
            anyhow::ensure!(values.len() % vocab == 0, "Qwen actual download row layout");
            let count = values.len() / vocab;
            let start = end
                .checked_sub(count)
                .ok_or_else(|| anyhow::anyhow!("Qwen logical row origin"))?;
            for (row, values) in values.chunks_exact(vocab).enumerate() {
                let bytes = bytemuck::cast_slice(values);
                store.capture(coordinate(lane,start+row,layer),Scalar::F32,vec![1,vocab],
                    values.as_ptr() as usize as u64,bytes.len() as u64,0,None,
                    "Qwen actual forward_gpu checked return/download; parent identity is host row, NOT Metal",bytes)
                    .map_err(anyhow::Error::msg)?;
            }
            if capture_cache && matches!(end, 128 | 144) {
                super::qwen_cache::after_checked_terminal(
                    &mut store,
                    cache,
                    layer,
                    &format!("agent-{lane}"),
                    end,
                )?;
            }
            Ok(())
        },
    );
    let execution = match result {
        Ok(e) => e,
        Err(e) => {
            store.fail(format!("actual Qwen execution: {e:#}"));
            store.finish().unwrap();
            panic!("Q1 incomplete actual Qwen adapter: {e:#}")
        }
    };
    let trace = store.finish().unwrap();
    assert!(trace.coverage.errors.is_empty());
    let evidence = serde_json::json!({"schema":"hf2q.adr063.q1.endpoint-acquisition.v1","packet_sha256":hash(&raw),"execution":execution,
        "trace_records":trace.coverage.captured.len(),"internal_stage_localization":"NotEstablishedByEndpointTrace",
        "whole_q1_qualified":false,"identity_attestation":"RequiresIndependentLaunchBuildArtifactComposition"});
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(packet.output.join("execution.json"))
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(&evidence).unwrap())
        .unwrap();
    eprintln!(
        "Q1_QWEN_ENDPOINT_ACQUISITION_COMPLETE records={} whole_q1_qualified=false",
        trace.coverage.captured.len()
    );
}
fn coordinate(lane: usize, row: usize, layer: u32) -> Coordinate {
    Coordinate {
        logical_step: 0,
        layer,
        stage: Stage::Logits,
        agent: format!("agent-{lane}"),
        logical_row_start: row as u64,
        segment: 0,
    }
}
