//! Real Metal comparisons against the removed full CPU-transpose ingestion.

use super::*;
use super::context_commit_tests::{config, random, rings};
use crate::inference::spec_decode::dflash2::tensors::DFlash2Projection;
use mlx_native::DType;
use std::ops::Range;

fn runtime(device: &MlxDevice, lanes: usize, staged: bool) -> DFlash2Runtime {
    let config = DFlash2Config {
        num_target_layers: 2,
        target_layer_ids: vec![0, 1],
        ..config()
    };
    let tensors = DFlash2ModelTensors::synthetic_af4_g64(device, &config, 5);
    let workspace = DFlash2Workspace::new(device, &config, lanes, staged).unwrap();
    let window = config.sliding_window;
    DFlash2Runtime {
        descriptor: &crate::inference::spec_decode::dflash2::schema::QWEN38_DFLASH2,
        config,
        tensors,
        workspace,
        lane_lengths: vec![0; lanes],
        lane_bindings: (0..lanes).map(|_| DFlash2LaneBinding::new(window)).collect(),
        verify_budget: DFlash2VerifyBudget::default(),
        gpu_target_tokens: false,
        proposal_repetition_enabled: false,
        submitted_draft: std::sync::Mutex::new(None),
        prefetched: None,
        prefetch_counts: DraftPrefetchCounts::default(),
        budget_epoch: 0,
        context_positions: None,
    }
}

fn capture(device: &MlxDevice, runtime: &DFlash2Runtime, rows: usize, seed: &mut u64) -> DFlashCaptureSession {
    let mut capture = DFlashCaptureSession::new_shared(
        device, runtime.config.target_layer_ids.clone(), rows, runtime.config.hidden_size, false,
    ).unwrap();
    for value in capture.hidden_output.iter_mut() {
        *value = 2.0 * random(seed);
    }
    capture
}

/// Independent old construction: transpose all selected rows on the CPU,
/// preserve zero-padded full-pass geometry, then slice the one full allocation.
fn old_full_transpose(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    runtime: &mut DFlash2Runtime,
    capture: &DFlashCaptureSession,
    lane: usize,
    runs: &[Range<usize>],
) {
    let rows: Vec<usize> = runs.iter().flat_map(|run| run.clone()).collect();
    if rows.is_empty() {
        return;
    }
    let config = &runtime.config;
    let pass_rows = runtime.workspace.max_lanes * config.block_size;
    let passes = rows.len().div_ceil(pass_rows);
    let staged_rows = passes * pass_rows;
    let width = config.fc_input_dim();
    let mut full = device.alloc_buffer(staged_rows * width * 4, DType::F32, vec![staged_rows, width]).unwrap();
    let destination = full.as_mut_slice::<f32>().unwrap();
    destination.fill(0.0);
    for (index, row) in rows.iter().enumerate() {
        for layer in 0..capture.target_layer_ids.len() {
            for dim in 0..capture.hidden_size {
                destination[index * width + layer * capture.hidden_size + dim] =
                    capture.hidden_output[(layer * capture.seq_len + row) * capture.hidden_size + dim];
            }
        }
    }
    let start = runtime.lane_lengths[lane];
    let words = |values: Vec<u32>| {
        let mut buffer = device.alloc_buffer(values.len() * 4, DType::U32, vec![values.len()]).unwrap();
        buffer.as_mut_slice::<u32>().unwrap().copy_from_slice(&values);
        buffer
    };
    let positions = words((0..staged_rows).map(|row| (start + row) as u32).collect());
    let firsts = (0..passes * runtime.workspace.max_lanes).map(|group| group * config.block_size);
    let writes = words(firsts.clone().map(|first| ((start + first) % config.sliding_window) as u32).collect());
    let counts = words(firsts.map(|first| rows.len().saturating_sub(first).min(config.block_size) as u32).collect());
    let mut encoder = device.command_encoder().unwrap();
    for pass in 0..passes {
        dispatch_dflash2_lane_ingest(
            &mut encoder, registry, device, config, &runtime.tensors, &runtime.workspace, lane,
            &full.slice_view((pass * pass_rows * width * 4) as u64, pass_rows * width)
                .with_shape(vec![pass_rows, width]).unwrap(),
            &positions.slice_view((pass * pass_rows * 4) as u64, pass_rows),
            &writes.slice_view((pass * runtime.workspace.max_lanes * 4) as u64, runtime.workspace.max_lanes),
            &counts.slice_view((pass * runtime.workspace.max_lanes * 4) as u64, runtime.workspace.max_lanes),
        ).unwrap();
    }
    encoder.commit_and_wait().unwrap();
    runtime.lane_lengths[lane] += rows.len();
}

#[test]
fn shared_gather_matches_old_full_transpose_for_runs_tails_and_ring_wraps() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    for staged in [false, true] {
        for lanes in 1..=4 {
            let mut actual = runtime(&device, lanes, staged);
            let mut reference = runtime(&device, lanes, staged);
            let mut seed = 0xCA971E_u64;
            // Empty runs, overlaps, backwards run order, nonzero starts, tails,
            // and lengths greater than a ring; lanes are interleaved/resumed.
            for (rows, runs) in [
                (45, vec![3..8, 19..44]),
                (19, vec![0..0, 9..19, 2..7]),
                (301, vec![17..301, 0..13]),
                (9, vec![2..9]),
            ] {
                for lane in (0..lanes).rev() {
                    let capture = capture(&device, &actual, rows, &mut seed);
                    let source_bits: Vec<u32> = capture.hidden_output.iter().map(|value| value.to_bits()).collect();
                    let start = actual.lane_lengths[lane];
                    actual.ingest_target_capture_runs(&device, &mut registry, &capture, lane, start, &runs).unwrap();
                    old_full_transpose(&device, &mut registry, &mut reference, &capture, lane, &runs);
                    assert_eq!(rings(&actual), rings(&reference), "staged={staged}, lanes={lanes}, lane={lane}, runs={runs:?}");
                    assert_eq!(actual.lane_lengths, reference.lane_lengths);
                    assert_eq!(source_bits, capture.hidden_output.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
                    // The final caller-owned tile, including padding, must be
                    // the old reference's final slice, not merely equal rings.
                    let selected: Vec<usize> = runs.iter().flat_map(|run| run.clone()).collect();
                    let pass_rows = lanes * actual.config.block_size;
                    let first = (selected.len() - 1) / pass_rows * pass_rows;
                    let tile = actual.workspace.captured_target_hidden.as_slice::<f32>().unwrap();
                    for index in 0..pass_rows {
                        for layer in 0..2 {
                            for dim in 0..capture.hidden_size {
                                let expected = selected.get(first + index).map_or(0.0, |row| {
                                    capture.hidden_output[(layer * rows + row) * capture.hidden_size + dim]
                                });
                                assert_eq!(tile[index * actual.config.fc_input_dim() + layer * capture.hidden_size + dim].to_bits(), expected.to_bits());
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn rejected_capture_and_unsubmitted_encoding_leave_owners_available() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let mut runtime = runtime(&device, 2, false);
    let mut capture = capture(&device, &runtime, 41, &mut 19);
    let before = rings(&runtime);
    for (start, runs) in [(1, vec![0..4]), (0, vec![4..3]), (0, vec![0..42]), (usize::MAX, vec![0..1])] {
        assert!(runtime.ingest_target_capture_runs(&device, &mut registry, &capture, 0, start, &runs).is_err());
        assert_eq!(rings(&runtime), before);
        assert_eq!(runtime.lane_lengths, [0, 0]);
    }
    let snapshot = capture.clone();
    assert!(runtime.ingest_target_capture(&device, &mut registry, &snapshot, 0, 0).is_err());
    capture.seq_len = 1;
    assert!(runtime.ingest_target_capture(&device, &mut registry, &capture, 0, 0).is_err());
    capture.seq_len = 41;
    // Same element count but metadata no longer describes the backing shape.
    capture.seq_len = 82;
    capture.hidden_size /= 2;
    assert!(runtime.ingest_target_capture(&device, &mut registry, &capture, 0, 0).is_err());
    capture.seq_len = 41;
    capture.hidden_size *= 2;
    // The gathers are encoded first; an invalid FC weight must reject without
    // submitting them, and no borrowed source/tile may remain in flight.
    let good_projection = std::mem::replace(&mut runtime.tensors.context_projection,
        DFlash2Projection::Dense { weight: device.alloc_buffer(2, DType::BF16, vec![1, 1]).unwrap() });
    let tile_before = runtime.workspace.captured_target_hidden.as_slice::<f32>().unwrap()
        .iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    assert!(runtime.ingest_target_capture(&device, &mut registry, &capture, 0, 0).is_err());
    assert_eq!(rings(&runtime), before);
    assert_eq!(runtime.workspace.captured_target_hidden.as_slice::<f32>().unwrap()
        .iter().map(|value| value.to_bits()).collect::<Vec<_>>(), tile_before);
    assert_eq!(runtime.lane_lengths, [0, 0]);
    capture.hidden_output.fill(0.25);
    runtime.tensors.context_projection = good_projection;
    runtime.ingest_target_capture(&device, &mut registry, &capture, 0, 0).unwrap();
    assert_eq!(runtime.lane_lengths, [41, 0]);
}