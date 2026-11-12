//! A submitted draft-context commit (`HF2Q_DFLASH2_CONTEXT_FIRST=1`) against
//! the waited one.
//!
//! The context-first order hands the context commit to the GPU, commits
//! another command behind it on the same queue (the DeltaNet replay) and
//! waits once. The rings and lane lengths it leaves must be the waited
//! commit's bits at every step, also past a ring wrap and with partial and
//! empty retained counts.
//!
//! Needs Apple Metal (`hf2q_gpu_test_lock`):
//!
//! ```text
//! cargo test --release --bin hf2q context_commit_tests
//! ```

use mlx_native::ops::copy::dispatch_copy_f32;
use mlx_native::{DType, KernelRegistry, MlxDevice};

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

#[test]
fn submitted_context_commit_is_the_waited_commit_bitwise() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    for staged in [false, true] {
        let mut registry = KernelRegistry::new();
        let mut waited = runtime(&device, staged);
        let mut submitted = runtime(&device, staged);
        // The command committed behind the submitted context: an unrelated
        // copy, standing in for the replay.
        let mut source = device
            .alloc_buffer(4_096 * 4, DType::F32, vec![4_096])
            .unwrap();
        source.as_mut_slice::<f32>().unwrap().fill(1.5);
        let sink = device
            .alloc_buffer(4_096 * 4, DType::F32, vec![4_096])
            .unwrap();
        let (mut waited_state, mut submitted_state) = (29_u64, 29_u64);
        for step in 0..40_u32 {
            let retained = [8, step % 9, 3, 0];
            stage_step(&mut waited, &mut waited_state);
            waited
                .commit_target_context(&device, &mut registry, &retained)
                .unwrap();

            stage_step(&mut submitted, &mut submitted_state);
            let pending = submitted
                .submit_target_context(&device, &mut registry, &retained)
                .unwrap();
            assert_eq!(
                submitted.lane_lengths,
                waited
                    .lane_lengths
                    .iter()
                    .zip(&retained)
                    .map(|(length, rows)| length - *rows as usize)
                    .collect::<Vec<_>>(),
                "step {step}: a submission moved the lane lengths"
            );
            let mut behind = device.command_encoder().unwrap();
            dispatch_copy_f32(
                &mut behind,
                &mut registry,
                device.metal_device(),
                &source,
                &sink,
                0,
                0,
                4_096,
            )
            .unwrap();
            behind.commit_and_wait().unwrap();
            submitted
                .finish_target_context(pending, &retained, true)
                .unwrap();

            assert_eq!(
                submitted.lane_lengths, waited.lane_lengths,
                "staged={staged} step {step}: lane lengths"
            );
            assert!(
                rings(&submitted) == rings(&waited),
                "staged={staged} step {step}: the submitted commit left other ring bytes"
            );
        }
        assert!(
            waited.lane_lengths[0] > waited.config.sliding_window,
            "the rings never wrapped"
        );

        // A failed step: the wait still happens, the lengths stay.
        let lengths = submitted.lane_lengths.clone();
        stage_step(&mut submitted, &mut submitted_state);
        let pending = submitted
            .submit_target_context(&device, &mut registry, &[8, 8, 8, 8])
            .unwrap();
        submitted
            .finish_target_context(pending, &[8, 8, 8, 8], false)
            .unwrap();
        assert_eq!(submitted.lane_lengths, lengths);
    }
}
