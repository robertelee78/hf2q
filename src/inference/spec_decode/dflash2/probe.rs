//! Per-dispatch attribution of one DFlash2 draft command at the served shape.
//!
//! The probe loads the real target GGUF and draft checkpoint, fills every
//! lane's ring, captures the draft command (input, layers, head, selection)
//! and the context commit exactly as the scheduler encodes them, and replays
//! every dispatch as its own command buffer. It reports the median GPU time
//! of each dispatch, the per-kernel totals, and the median GPU time of the
//! same dispatches fused into one command.

use std::path::PathBuf;

use super::{
    adapters::{dispatch_dflash2_input, target_vocabulary_head, QwenDFlash2Adapter},
    forward::{
        dispatch_dflash2_context_commit, dispatch_dflash2_forward, prepare_decode_positions,
    },
    runtime::DFlash2Runtime,
    schema::{DFlash2CheckpointDescriptor, QWEN36_DFLASH2, QWEN38_DFLASH2},
    selector::dispatch_dflash2_selection,
    staged::reads_prepared_head,
};
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::serve::header::LoadProgress;
use anyhow::{bail, Context, Result};
use mlx_native::{CapturedNode, KernelRegistry, MlxDevice};

const LANES: usize = 4;
const WARMUP_ROUNDS: usize = 3;

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        0.5 * (values[middle - 1] + values[middle])
    }
}

fn replay(device: &MlxDevice, nodes: &[CapturedNode], serialized: bool) -> Result<Vec<f64>> {
    let mut times = Vec::new();
    let mut encoder = device.command_encoder()?;
    for node in nodes {
        match node {
            CapturedNode::Barrier => {
                if !serialized {
                    encoder.memory_barrier();
                }
            }
            CapturedNode::Dispatch {
                pipeline,
                bindings,
                threads_per_grid,
                threads_per_threadgroup,
                threadgroup_memory,
                dispatch_kind,
                ..
            } => {
                encoder.replay_dispatch(
                    pipeline,
                    bindings,
                    threadgroup_memory,
                    *threads_per_grid,
                    *threads_per_threadgroup,
                    *dispatch_kind,
                );
                if serialized {
                    let (start, end) = encoder.commit_wait_with_gpu_time()?;
                    times.push((end - start) * 1.0e6);
                    encoder = device.command_encoder()?;
                }
            }
        }
    }
    if !serialized {
        let (start, end) = encoder.commit_wait_with_gpu_time()?;
        times.push((end - start) * 1.0e6);
    }
    Ok(times)
}

fn report(device: &MlxDevice, title: &str, nodes: &[CapturedNode], rounds: usize) -> Result<()> {
    let dispatches: Vec<(String, [u64; 3], [u64; 3])> = nodes
        .iter()
        .filter_map(|node| match node {
            CapturedNode::Dispatch {
                pipeline,
                threads_per_grid,
                threads_per_threadgroup,
                ..
            } => Some((
                pipeline.label().to_string(),
                [
                    threads_per_grid.width,
                    threads_per_grid.height,
                    threads_per_grid.depth,
                ],
                [
                    threads_per_threadgroup.width,
                    threads_per_threadgroup.height,
                    threads_per_threadgroup.depth,
                ],
            )),
            CapturedNode::Barrier => None,
        })
        .collect();
    let barriers = nodes.len() - dispatches.len();
    let mut samples = vec![Vec::with_capacity(rounds); dispatches.len()];
    let mut fused = Vec::with_capacity(rounds);
    for round in 0..WARMUP_ROUNDS + rounds {
        let serialized = replay(device, nodes, true)?;
        let whole = replay(device, nodes, false)?;
        if round >= WARMUP_ROUNDS {
            for (sample, value) in samples.iter_mut().zip(serialized) {
                sample.push(value);
            }
            fused.push(whole[0]);
        }
    }
    let medians: Vec<f64> = samples.iter_mut().map(|sample| median(sample)).collect();
    println!(
        "== {title}: {} dispatches, {barriers} barriers, median of {rounds} rounds ==",
        dispatches.len()
    );
    for (index, ((label, grid, group), micros)) in dispatches.iter().zip(&medians).enumerate() {
        println!(
            "{index:4} {label:54} grid {:>6}x{:<5}x{:<3} tg {:>4}x{:<3} {micros:9.2}",
            grid[0], grid[1], grid[2], group[0], group[1]
        );
    }
    let mut totals: Vec<(String, usize, f64)> = Vec::new();
    for ((label, _, _), micros) in dispatches.iter().zip(&medians) {
        match totals.iter_mut().find(|(name, _, _)| name == label) {
            Some(entry) => {
                entry.1 += 1;
                entry.2 += micros;
            }
            None => totals.push((label.clone(), 1, *micros)),
        }
    }
    totals.sort_by(|left, right| right.2.total_cmp(&left.2));
    let sum: f64 = medians.iter().sum();
    println!("-- {title}: per kernel --");
    for (label, count, micros) in &totals {
        println!(
            "{label:54} x{count:<4} {:9.3} ms {:5.1}%",
            micros / 1000.0,
            100.0 * micros / sum
        );
    }
    println!(
        "-- {title}: {:.3} ms as {} separate dispatches, {:.3} ms fused --",
        sum / 1000.0,
        dispatches.len(),
        median(&mut fused) / 1000.0
    );
    Ok(())
}

fn probe(
    model: &Qwen35Model,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    descriptor: &'static DFlash2CheckpointDescriptor,
    config_path: &PathBuf,
    weights_path: &PathBuf,
    context_rows: usize,
    rounds: usize,
) -> Result<()> {
    let config = descriptor.parse_config(&std::fs::read(config_path)?)?;
    let adapter = QwenDFlash2Adapter::new(model, &config)?;
    let mut runtime = DFlash2Runtime::load_verified(
        device,
        descriptor,
        config_path,
        weights_path,
        &adapter,
        LANES,
    )?;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for value in runtime
        .workspace
        .captured_target_hidden
        .as_mut_slice::<f32>()?
    {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *value = ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0;
    }
    let block = config.block_size;
    for _ in 0..context_rows / block {
        let positions: Vec<u32> = runtime
            .lane_lengths()
            .iter()
            .map(|length| *length as u32)
            .collect();
        prepare_decode_positions(&mut runtime.workspace, &config, &positions)?;
        runtime.commit_target_context(device, registry, &[block as u32; LANES])?;
    }
    let positions: Vec<u32> = runtime
        .lane_lengths()
        .iter()
        .map(|length| *length as u32)
        .collect();
    let anchors = [1_u32, 2, 3, 4];
    for _ in 0..2 {
        runtime.run_draft(
            device,
            registry,
            &adapter,
            &anchors,
            &positions,
            &[0.0; LANES],
            0,
            &[block as u32; LANES],
            &vec![0.5; LANES * config.proposal_tokens()],
        )?;
    }

    prepare_decode_positions(&mut runtime.workspace, &config, &positions)?;
    let mut encoder = device.command_encoder()?;
    encoder.start_capture();
    dispatch_dflash2_input(
        &runtime.tensors,
        &mut encoder,
        registry,
        device,
        &config,
        &adapter,
        &anchors,
        &mut runtime.workspace,
    )?;
    let vocabulary_head = target_vocabulary_head(
        &adapter,
        registry,
        device,
        reads_prepared_head(&runtime.workspace, &config),
    )?;
    dispatch_dflash2_forward(
        &mut encoder,
        registry,
        device,
        &config,
        &runtime.tensors,
        vocabulary_head,
        &mut runtime.workspace,
    )?;
    dispatch_dflash2_selection(
        &mut encoder,
        registry,
        device,
        &config,
        &runtime.tensors,
        &runtime.workspace,
        &anchors,
        &[0.0; LANES],
        0,
        &[block as u32; LANES],
        &[0, 1, 2, 3],
        None,
    )?;
    let draft = encoder.take_capture().context("draft capture")?;

    runtime
        .workspace
        .retained_counts
        .as_mut_slice::<u32>()?
        .fill(block as u32);
    let mut encoder = device.command_encoder()?;
    encoder.start_capture();
    dispatch_dflash2_context_commit(
        &mut encoder,
        registry,
        device,
        &config,
        &runtime.tensors,
        &runtime.workspace.retained_counts,
        &runtime.workspace,
    )?;
    let commit = encoder.take_capture().context("context commit capture")?;

    println!(
        "DFlash2 draft probe: checkpoint {} lanes {LANES} context rows {context_rows} projections {} staged {}",
        descriptor.id,
        runtime.tensors.layers[0].qkv_projection.format_label(),
        runtime.workspace.staged.is_some()
    );
    report(device, "draft command", &draft, rounds)?;
    report(device, "context commit", &commit, rounds)?;
    if runtime
        .proposals()?
        .iter()
        .any(|token| *token as usize >= config.vocab_size)
    {
        bail!("draft probe proposal exceeds the vocabulary");
    }
    Ok(())
}

#[test]
#[ignore = "requires HF2Q_DFLASH2_PROBE_GGUF, HF2Q_DFLASH2_PROBE_CONFIG, HF2Q_DFLASH2_PROBE_WEIGHTS and Apple Metal"]
fn dflash2_draft_command_dispatch_probe() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let path =
        |name: &str| PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name}")));
    let gguf_path = path("HF2Q_DFLASH2_PROBE_GGUF");
    let config_path = path("HF2Q_DFLASH2_PROBE_CONFIG");
    let weights_path = path("HF2Q_DFLASH2_PROBE_WEIGHTS");
    let number = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .map(|value| value.parse::<usize>().unwrap_or_else(|_| panic!("{name}")))
            .unwrap_or(default)
    };
    let context_rows = number("HF2Q_DFLASH2_PROBE_CONTEXT_ROWS", 512);
    let rounds = number("HF2Q_DFLASH2_PROBE_ROUNDS", 20);

    let gguf = mlx_native::gguf::GgufFile::open(&gguf_path).expect("open target GGUF");
    let mut progress = LoadProgress::new(false, 1, 0);
    let model = Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load target");
    model.ensure_gpu_cache_primed().expect("prime GPU cache");
    let descriptor: &'static DFlash2CheckpointDescriptor = if model.cfg.moe.is_some() {
        &QWEN36_DFLASH2
    } else {
        &QWEN38_DFLASH2
    };
    model
        .with_gpu_cache_mut(|device, registry| {
            probe(
                &model,
                device,
                registry,
                descriptor,
                &config_path,
                &weights_path,
                context_rows,
                rounds,
            )
        })
        .expect("probe DFlash2 draft command");
}
