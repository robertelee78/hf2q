//! Root-run-only endpoint acquisition. A complete endpoint trace is NOT full
//! internal-stage localization, task quality, or whole-Q1 qualification.
use super::{
    capture, deepseek_runner,
    protocol::{Trajectory, Workload},
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
    acquisition: super::acquisition::Binding,
    model: PathBuf,
    output: PathBuf,
    identity: Identity,
    workload: Workload,
    trajectory: Trajectory,
    configured_capacity: usize,
    requested_payload_limit: u64,
}

#[test]
#[ignore = "root controlled real DeepSeek model; immutable packet and independent resource/identity review required"]
fn q1_deepseek_actual_endpoint_trace() {
    let config = std::env::var_os("HF2Q_Q1_DS_PACKET").expect("explicit Q1 packet path required");
    let mut raw = Vec::new();
    std::fs::File::open(config)
        .expect("open Q1 packet")
        .take(1024 * 1024 + 1)
        .read_to_end(&mut raw)
        .expect("read bounded Q1 packet");
    assert!(raw.len() <= 1024 * 1024, "bounded Q1 config");
    let packet: Packet = serde_json::from_slice(&raw).expect("closed Q1 packet");
    assert!(packet.execution_approved, "root approval missing");
    packet.identity.validate().unwrap();
    packet.workload.validate().unwrap();
    assert_eq!(packet.identity.family, Family::Deepseek4);
    assert_eq!(packet.workload.family, Family::Deepseek4);
    assert!((144..=262144).contains(&packet.configured_capacity));
    assert!(
        packet.requested_payload_limit <= 1024 * 1024 * 1024,
        "endpoint capture at most 1GiB host payload"
    );
    assert!(!super::family::enabled(Family::Deepseek4));
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
    let gguf =
        super::acquisition::open_model(&packet.model).expect("open explicitly bound artifact");
    let header_cfg =
        crate::inference::models::deepseek4::Deepseek4Config::from_gguf(&gguf).unwrap();
    acquisition_scope
        .check_config(
            &packet.identity,
            &super::acquisition::ConfigRecipe {
                family: packet.identity.family,
                capacity: packet.configured_capacity as u64,
                slots: 1,
                output_allowance: 0,
                cache_mode: "DeepseekNative",
                route: "endpoint",
                actual_model_config_sha256: hash(format!("{:?}", &header_cfg).as_bytes()),
            },
        )
        .unwrap();
    let mut model = crate::inference::models::deepseek4::Deepseek4Model::load_from_gguf(&gguf)
        .expect("actual DS model");
    assert_eq!(packet.workload.vocab_size, model.cfg.vocab_size);
    acquisition_scope
        .check_config(
            &packet.identity,
            &super::acquisition::ConfigRecipe {
                family: packet.identity.family,
                capacity: packet.configured_capacity as u64,
                slots: 1,
                output_allowance: 0,
                cache_mode: "DeepseekNative",
                route: "endpoint",
                actual_model_config_sha256: hash(format!("{:?}", &model.cfg).as_bytes()),
            },
        )
        .unwrap();
    let hc = model.cfg.hyper_connection_count as usize;
    let hidden = model.cfg.hidden_size as usize;
    let vocab = model.cfg.vocab_size as usize;
    let layer_count = model.cfg.num_hidden_layers;
    let mut expected = Vec::new();
    for position in 0..144 {
        for lane in 0..4 {
            for (stage, shape) in [
                (Stage::LayerOutput, vec![1, hc, hidden]),
                (Stage::Logits, vec![1, vocab]),
            ] {
                expected.push(ExpectedTensor {
                    at: coordinate(lane, position, stage, model.cfg.num_hidden_layers),
                    scalar: Scalar::F32,
                    shape,
                });
            }
        }
    }
    expected.sort_by(|a, b| {
        (a.at.logical_row_start, a.at.layer, a.at.stage, &a.at.agent).cmp(&(
            b.at.logical_row_start,
            b.at.layer,
            b.at.stage,
            &b.at.agent,
        ))
    });
    let mut store = Store::new(
        &packet.output,
        packet.identity.clone(),
        packet.trajectory.label().into(),
        expected,
        packet.requested_payload_limit,
        4 * 144 * 2,
    )
    .unwrap();
    store
        .bind_acquisition(
            acquisition_scope
                .observed(
                    model.ctx.executor.device(),
                    &model as *const _ as usize as u64,
                    1,
                )
                .unwrap(),
        )
        .unwrap();
    let result = deepseek_runner::run(
        &mut model,
        &packet.workload,
        packet.trajectory,
        packet.configured_capacity,
        |lane, end, state, logits, cache| {
            anyhow::ensure!(
                cache.position() == end,
                "actual cursor does not match callback"
            );
            let rows = state.shape()[0];
            let start = end
                .checked_sub(rows)
                .ok_or_else(|| anyhow::anyhow!("row origin underflow"))?;
            capture::rows_after_checked_drain(
                &mut store,
                coordinate(lane, start, Stage::LayerOutput, layer_count),
                state,
                None,
                "actual DS forward terminal + cursor publication",
            )
            .map_err(anyhow::Error::msg)?;
            capture::rows_after_checked_drain(
                &mut store,
                coordinate(lane, start, Stage::Logits, layer_count),
                logits,
                None,
                "actual DS full-row output head checked finish",
            )
            .map_err(anyhow::Error::msg)?;
            Ok(())
        },
    );
    let execution = match result {
        Ok(e) => e,
        Err(e) => {
            store.fail(format!("actual DS execution: {e:#}"));
            store.finish().unwrap();
            panic!("Q1 actual diagnostic incomplete: {e:#}")
        }
    };
    acquisition_scope.check_unchanged().unwrap();
    let trace = store.finish().unwrap();
    assert!(trace.coverage.errors.is_empty());
    let output = serde_json::json!({"schema":"hf2q.adr063.q1.endpoint-acquisition.v1","packet_sha256":hash(&raw),
        "execution":execution,"trace_records":trace.coverage.captured.len(),"internal_stage_localization":"NotEstablishedByEndpointTrace",
        "whole_q1_qualified":false,"identity_attestation":"RequiresIndependentLaunchBuildArtifactComposition"});
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(packet.output.join("execution.json"))
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(&output).unwrap())
        .unwrap();
    eprintln!(
        "Q1_DS_ENDPOINT_ACQUISITION_COMPLETE records={} whole_q1_qualified=false",
        trace.coverage.captured.len()
    );
}
fn coordinate(lane: usize, row: usize, stage: Stage, layer: u32) -> Coordinate {
    Coordinate {
        logical_step: 0,
        layer,
        stage,
        agent: format!("agent-{lane}"),
        logical_row_start: row as u64,
        segment: 0,
    }
}
