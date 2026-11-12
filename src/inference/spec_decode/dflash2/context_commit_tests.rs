//! The context-first commit order (`HF2Q_DFLASH2_CONTEXT_FIRST=1`) against
//! the waited draft-context commit.
//!
//! The context-first order hands the context commit to the GPU, runs the
//! target's replay commit (another command, committed and waited for while
//! the context commit may still be executing) and then waits for the
//! context command itself. These tests call the function the scheduler's
//! context-first branch calls, [`DFlash2Runtime::settle_context_first`]:
//!
//! - success: the rings and lane lengths are the waited commit's bits at
//!   every step, also past a ring wrap and with partial and empty retained
//!   counts, on both draft routes;
//! - a replay that fails after the context was submitted: no lane length
//!   advances, the step's lanes are cleared, other lanes keep their bytes,
//!   and the replay's error is returned;
//! - a lane that cannot be cleared: the remaining lanes are still cleared
//!   and the failure is reported, not dropped.
//!
//! Needs Apple Metal (`hf2q_gpu_test_lock`):
//!
//! ```text
//! cargo test --release --bin hf2q context_commit_tests
//! ```

use anyhow::anyhow;
use mlx_native::ops::copy::dispatch_copy_f32;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::*;
use crate::inference::spec_decode::dflash2::config::DFlash2LayerType;
use crate::inference::spec_decode::dflash2::schema::QWEN38_DFLASH2;

const LANES: usize = 4;

/// The smallest shape both draft routes take: 16 query heads over 4 KV
/// heads of 128 and a 256-slot ring, so 40 eight-row commits wrap it.
fn config() -> DFlash2Config {
    DFlash2Config {
        draft_model_type: "qwen3".into(),
        hidden_size: 1_024,
        intermediate_size: 1_024,
        num_hidden_layers: 2,
        num_attention_heads: 16,
        num_key_value_heads: 4,
        head_dim: 128,
        vocab_size: 64,
        num_target_layers: 1,
        target_layer_ids: vec![0],
        layer_types: vec![DFlash2LayerType::SlidingAttention; 2],
        max_position_embeddings: 4_096,
        sliding_window: 256,
        rms_norm_eps: 1.0e-6,
        rope_theta: 1_000_000.0,
        block_size: 8,
        conv_group_size: 16,
        conv_kernel_size: 2,
        selector_rank: 256,
        selector_top_k: 16,
        mask_token_id: 0,
        final_logit_softcapping: None,
        output_multiplier: None,
    }
}

fn runtime(device: &MlxDevice, staged: bool) -> DFlash2Runtime {
    let config = config();
    let tensors = DFlash2ModelTensors::synthetic_af4_g64(device, &config, 5);
    let workspace = DFlash2Workspace::new(device, &config, LANES, staged).unwrap();
    let window = config.sliding_window;
    DFlash2Runtime {
        descriptor: &QWEN38_DFLASH2,
        config,
        tensors,
        workspace,
        lane_lengths: vec![0; LANES],
        lane_bindings: (0..LANES)
            .map(|_| DFlash2LaneBinding::new(window))
            .collect(),
        verify_budget: DFlash2VerifyBudget::default(),
        gpu_target_tokens: false,
    }
}

fn random(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 40) as f32 / (1_u64 << 24) as f32 - 0.5
}

/// Fill the step's captured target rows and position the rings' write
/// cursors at the lane lengths, as a draft cycle does before its commit.
fn stage_step(runtime: &mut DFlash2Runtime, state: &mut u64) {
    for value in runtime
        .workspace
        .captured_target_hidden
        .as_mut_slice::<f32>()
        .unwrap()
    {
        *value = 2.0 * random(state);
    }
    let positions = runtime
        .lane_lengths
        .iter()
        .map(|&length| length as u32)
        .collect::<Vec<_>>();
    prepare_decode_positions(&mut runtime.workspace, &runtime.config, &positions).unwrap();
}

fn rings(runtime: &DFlash2Runtime) -> Vec<Vec<u16>> {
    runtime
        .workspace
        .ring_keys
        .iter()
        .chain(&runtime.workspace.ring_values)
        .map(|ring| ring.as_slice::<u16>().unwrap().to_vec())
        .collect()
}

/// One lane's elements of every ring.
fn lane_rings(runtime: &DFlash2Runtime, lane: usize) -> Vec<Vec<u16>> {
    let per_lane = runtime.config.num_key_value_heads
        * runtime.config.sliding_window
        * runtime.config.head_dim;
    rings(runtime)
        .into_iter()
        .map(|ring| ring[lane * per_lane..(lane + 1) * per_lane].to_vec())
        .collect()
}

/// The stand-in for the target's replay commit: an unrelated command,
/// committed and waited for while the context commit may still run.
struct Replay {
    source: MlxBuffer,
    sink: MlxBuffer,
}

impl Replay {
    fn new(device: &MlxDevice) -> Self {
        let mut source = device
            .alloc_buffer(4_096 * 4, DType::F32, vec![4_096])
            .unwrap();
        source.as_mut_slice::<f32>().unwrap().fill(1.5);
        let sink = device
            .alloc_buffer(4_096 * 4, DType::F32, vec![4_096])
            .unwrap();
        Self { source, sink }
    }

    fn run(&self, device: &MlxDevice, registry: &mut KernelRegistry) -> anyhow::Result<()> {
        let mut encoder = device.command_encoder()?;
        dispatch_copy_f32(
            &mut encoder,
            registry,
            device.metal_device(),
            &self.source,
            &self.sink,
            0,
            0,
            4_096,
        )?;
        encoder.commit_and_wait()?;
        Ok(())
    }
}

#[test]
fn context_first_is_the_waited_commit_bitwise() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    for staged in [false, true] {
        let mut registry = KernelRegistry::new();
        let mut waited = runtime(&device, staged);
        let mut first = runtime(&device, staged);
        let replay = Replay::new(&device);
        let (mut waited_state, mut first_state) = (29_u64, 29_u64);
        for step in 0..40_u32 {
            let retained = [8, step % 9, 3, 0];
            stage_step(&mut waited, &mut waited_state);
            waited
                .commit_target_context(&device, &mut registry, &retained)
                .unwrap();

            stage_step(&mut first, &mut first_state);
            let before = first.lane_lengths.clone();
            let submitted = first.submit_target_context(&device, &mut registry, &retained);
            assert_eq!(
                first.lane_lengths, before,
                "step {step}: a submission moved the lane lengths"
            );
            let (committed, context) =
                first.settle_context_first(submitted, &retained, &[0, 1, 2, 3], || {
                    replay.run(&device, &mut registry)
                });
            committed.unwrap();
            context.unwrap();

            assert_eq!(
                first.lane_lengths, waited.lane_lengths,
                "staged={staged} step {step}: lane lengths"
            );
            assert!(
                rings(&first) == rings(&waited),
                "staged={staged} step {step}: the context-first order left other ring bytes"
            );
        }
        assert!(
            waited.lane_lengths[0] > waited.config.sliding_window,
            "the rings never wrapped"
        );
    }
}

/// The replay fails after the context commit was submitted.
#[test]
fn a_failed_replay_after_submission_advances_nothing_and_clears_the_step_lanes() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    for staged in [false, true] {
        let mut registry = KernelRegistry::new();
        let mut runtime = runtime(&device, staged);
        let mut state = 31_u64;
        // History in every lane, past a wrap in lanes 0 and 2.
        for _ in 0..36 {
            stage_step(&mut runtime, &mut state);
            runtime
                .commit_target_context(&device, &mut registry, &[8, 5, 8, 2])
                .unwrap();
        }
        let lengths = runtime.lane_lengths.clone();
        let untouched = [lane_rings(&runtime, 2), lane_rings(&runtime, 3)];
        assert!(lane_rings(&runtime, 0)
            .iter()
            .flatten()
            .any(|&bits| bits != 0));

        // The failing step holds lanes 0 and 1.
        let retained = [8, 3, 0, 0];
        stage_step(&mut runtime, &mut state);
        let submitted = runtime.submit_target_context(&device, &mut registry, &retained);
        let (committed, context) =
            runtime.settle_context_first(submitted, &retained, &[0, 1], || {
                Err(anyhow!("replay command failed"))
            });
        assert_eq!(
            committed.unwrap_err().to_string(),
            "replay command failed",
            "the replay's own error is what the caller sees"
        );
        context.expect("the context command was waited for and the lanes were cleared");
        assert_eq!(
            runtime.lane_lengths,
            [0, 0, lengths[2], lengths[3]],
            "staged={staged}: a failed step advanced a lane or left its lanes' lengths"
        );
        for lane in [0, 1] {
            assert!(
                lane_rings(&runtime, lane)
                    .iter()
                    .flatten()
                    .all(|&bits| bits == 0),
                "staged={staged}: lane {lane} of the failed step was not cleared"
            );
        }
        assert!(
            [lane_rings(&runtime, 2), lane_rings(&runtime, 3)] == untouched,
            "staged={staged}: a lane outside the failed step changed"
        );

        // The cleared lanes take a new history like fresh ones.
        stage_step(&mut runtime, &mut state);
        runtime
            .commit_target_context(&device, &mut registry, &[8, 8, 0, 0])
            .unwrap();
        assert_eq!(runtime.lane_lengths, [8, 8, lengths[2], lengths[3]]);
    }
}

/// A lane that cannot be cleared does not stop the cleanup of the others,
/// and its failure reaches the caller.
#[test]
fn a_failed_clear_is_reported_and_the_other_lanes_are_still_cleared() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut registry = KernelRegistry::new();
    let mut runtime = runtime(&device, false);
    let mut state = 37_u64;
    for _ in 0..4 {
        stage_step(&mut runtime, &mut state);
        runtime
            .commit_target_context(&device, &mut registry, &[8, 8, 8, 8])
            .unwrap();
    }
    let retained = [8, 8, 0, 0];
    stage_step(&mut runtime, &mut state);
    let submitted = runtime.submit_target_context(&device, &mut registry, &retained);
    // Lane 9 does not exist: its clear fails, first in the list.
    let (committed, context) =
        runtime.settle_context_first(submitted, &retained, &[9, 0, 1], || {
            Err(anyhow!("replay command failed"))
        });
    assert!(committed.is_err());
    let error = format!("{:#}", context.unwrap_err());
    assert!(
        error.contains("clear DFlash2 lane 9"),
        "the failed clear is not what was reported: {error}"
    );
    assert_eq!(runtime.lane_lengths, [0, 0, 32, 32]);
    for lane in [0, 1] {
        assert!(
            lane_rings(&runtime, lane)
                .iter()
                .flatten()
                .all(|&bits| bits == 0),
            "lane {lane} was not cleared after an earlier clear failed"
        );
    }

    // A submission that failed is reported after the replay ran; a replay
    // that succeeded is not turned into a failure of the step's lanes.
    let lengths = runtime.lane_lengths.clone();
    let mut replay_ran = false;
    let (committed, context) = runtime.settle_context_first(
        Err(anyhow!("context encode failed")),
        &retained,
        &[0, 1],
        || {
            replay_ran = true;
            Ok(())
        },
    );
    assert!(replay_ran && committed.is_ok());
    assert_eq!(context.unwrap_err().to_string(), "context encode failed");
    assert_eq!(runtime.lane_lengths, lengths);
}
