//! The draft prefetch (`HF2Q_DFLASH2_DRAFT_PREFETCH=1`) against the stock
//! order of a step that settles acceptance on the host.
//!
//! Stock: the context commit is committed and waited for, and the next step
//! issues its draft. Prefetch: the context commit is submitted reading its
//! own copy of the step's positions, the next step's draft is committed
//! behind it, a replay stand-in runs beside both, and the next step's draft
//! call adopts the command when its inputs are the prefetched ones. (The
//! scheduler commits its real replay between the two; the stand-in here is
//! committed and waited for behind the draft, which the bits do not see.)
//!
//! These tests drive the runtime calls the scheduler's prefetch branch
//! makes, on a synthetic checkpoint and both draft routes:
//!
//! - adopted prefetches: every draft output and the rings are the stock
//!   order's bits at every step, past a ring wrap, with full, partial and
//!   empty retained counts, through both draft entry points;
//! - a next step with other inputs (a lane left, another anchor, another
//!   row limit, the other entry point) and a ring change between the steps
//!   (a cleared lane, a prefill ingest) drop the prefetched command and
//!   draft again, with the stock bits;
//! - a step that fails after its submissions leaves no prefetched draft,
//!   clears its lanes and leaves the other lanes' bytes;
//! - the draft outputs are unreadable from a prefetch to the next draft.
//!
//! Needs Apple Metal (`hf2q_gpu_test_lock`):
//!
//! ```text
//! cargo test --release --bin hf2q draft_prefetch_tests
//! ```

use mlx_native::{DType, GgmlType, KernelRegistry, MlxDevice};

use super::context_commit_tests::{lane_rings, random, rings, runtime, Replay, LANES};
use super::*;
use crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession;
use crate::inference::spec_decode::dflash2::config::DFlash2TargetShape;
use crate::serve::forward_mlx_shared::MlxQWeight;
use crate::serve::gpu::QuantWeightInfo;

/// A dense F32 target of the synthetic checkpoint's shape.
struct Target {
    embedding: MlxQWeight,
    head: MlxQWeight,
}

impl Target {
    fn new(device: &MlxDevice, config: &DFlash2Config) -> Self {
        let dense = |seed: u64, scale: f32| {
            let (rows, cols) = (config.vocab_size, config.hidden_size);
            let mut buffer = device
                .alloc_buffer(rows * cols * 4, DType::F32, vec![rows, cols])
                .unwrap();
            let mut state = seed;
            for value in buffer.as_mut_slice::<f32>().unwrap() {
                *value = scale * random(&mut state);
            }
            MlxQWeight {
                buffer,
                info: QuantWeightInfo {
                    ggml_dtype: GgmlType::F32,
                    rows,
                    cols,
                },
                affine: None,
                f16_shadow: None,
                decode_record_q6k_m1: std::sync::OnceLock::new(),
            }
        };
        Self {
            embedding: dense(101, 1.0),
            head: dense(103, 0.25),
        }
    }
}

impl DFlash2TargetFamily for Target {
    fn family_name(&self) -> &'static str {
        "synthetic"
    }

    fn target_shape(&self) -> DFlash2TargetShape<'_> {
        DFlash2TargetShape {
            architecture: "synthetic",
            hidden_size: self.embedding.info.cols,
            vocab_size: self.embedding.info.rows,
            num_hidden_layers: 1,
        }
    }

    fn embedding(&self) -> Result<&MlxQWeight> {
        Ok(&self.embedding)
    }

    fn vocabulary_head(&self) -> Result<&MlxQWeight> {
        Ok(&self.head)
    }

    fn embedding_scale(&self) -> f32 {
        1.0
    }
}

/// One step's draft call.
#[derive(Clone, Debug, PartialEq)]
struct Draft {
    anchors: [u32; LANES],
    maximum_rows: [u32; LANES],
    /// `None`: [`DFlash2Runtime::run_draft`]; `Some(order)`:
    /// [`DFlash2Runtime::run_draft_for_gpu_target`].
    gpu_order: Option<Vec<u32>>,
}

const UNIFORMS: [f32; LANES * 7] = [0.5; LANES * 7];
const TEMPERATURES: [f32; LANES] = [0.0; LANES];

fn positions(runtime: &DFlash2Runtime) -> Vec<u32> {
    runtime
        .lane_lengths
        .iter()
        .map(|&length| length as u32)
        .collect()
}

/// Issue `draft` as the scheduler does and return every output a step reads
/// (waiting for an unwaited command first, as the target's verify would).
fn draft(
    runtime: &mut DFlash2Runtime,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    target: &Target,
    draft: &Draft,
) -> Vec<Vec<u32>> {
    let positions = positions(runtime);
    match &draft.gpu_order {
        None => runtime
            .run_draft(
                device,
                registry,
                target,
                &draft.anchors,
                &positions,
                &TEMPERATURES,
                0,
                &draft.maximum_rows,
                &UNIFORMS,
            )
            .unwrap(),
        Some(order) => {
            runtime
                .run_draft_for_gpu_target(
                    device,
                    registry,
                    target,
                    &draft.anchors,
                    &positions,
                    &TEMPERATURES,
                    0,
                    &draft.maximum_rows,
                    order,
                    &UNIFORMS,
                )
                .unwrap();
            // A command that reads a draft output is ordered behind the
            // draft, as the target's embedding command is in a served step.
            let sink = device.alloc_buffer(4, DType::F32, vec![1]).unwrap();
            let mut flush = device.command_encoder().unwrap();
            mlx_native::ops::copy::dispatch_copy_f32(
                &mut flush,
                registry,
                device.metal_device(),
                &runtime.workspace.confidence,
                &sink,
                0,
                0,
                1,
            )
            .unwrap();
            flush.commit_and_wait().unwrap();
        }
    }
    outputs(runtime)
}

fn outputs(runtime: &DFlash2Runtime) -> Vec<Vec<u32>> {
    let bits = |buffer: &mlx_native::MlxBuffer| {
        buffer
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    vec![
        runtime.proposals().unwrap().to_vec(),
        runtime.verify_rows().unwrap().to_vec(),
        runtime
            .workspace
            .target_tokens
            .as_slice::<u32>()
            .unwrap()
            .to_vec(),
        bits(&runtime.workspace.confidence),
        bits(&runtime.workspace.proposal_probabilities),
        bits(&runtime.workspace.logits),
    ]
}

/// The rows a target verify would capture for the step.
fn capture(runtime: &mut DFlash2Runtime, state: &mut u64) {
    for value in runtime
        .workspace
        .captured_target_hidden
        .as_mut_slice::<f32>()
        .unwrap()
    {
        *value = 2.0 * random(state);
    }
}

/// The scheduler's prefetch branch after a step's plan: context commit
/// submitted, `next` committed behind it, the replay stand-in beside both,
/// then the waits and the settle.
fn commit_and_prefetch(
    runtime: &mut DFlash2Runtime,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    target: &Target,
    replay: &Replay,
    retained: &[u32; LANES],
    next: Option<&Draft>,
    step_lanes: &[usize],
    replay_fails: bool,
) -> (bool, Result<()>) {
    let submitted = runtime.submit_target_context_detached(device, registry, retained);
    if let (Ok(_), Some(next)) = (&submitted, next) {
        let positions = positions(runtime);
        runtime
            .prefetch_draft(
                device,
                registry,
                target,
                &next.anchors,
                &positions,
                &TEMPERATURES,
                0,
                &next.maximum_rows,
                next.gpu_order.as_deref().unwrap_or(&[]),
                &UNIFORMS,
                next.gpu_order.is_none(),
            )
            .unwrap();
        assert!(runtime.prefetch_outstanding());
    }
    replay.run(device, registry).unwrap();
    let committed = !replay_fails;
    let context =
        submitted.and_then(|submitted| runtime.finish_target_context(submitted, retained, false));
    (
        committed,
        runtime.settle_submitted_step(committed, context, step_lanes),
    )
}

fn step_draft(step: u32, gpu: bool) -> Draft {
    Draft {
        anchors: [1 + step % 60, 2 + (step * 7) % 61, 3 + (step * 11) % 59, 5],
        maximum_rows: [8, 8, 8, 8],
        gpu_order: gpu.then(|| vec![0, 1, 2, 3]),
    }
}

#[test]
fn an_adopted_prefetch_is_the_stock_draft_bitwise() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    for staged in [false, true] {
        for gpu in [false, true] {
            let mut registry = KernelRegistry::new();
            let mut stock = runtime(&device, staged);
            let mut ahead = runtime(&device, staged);
            let target = Target::new(&device, &stock.config);
            let replay = Replay::new(&device);
            let (mut stock_state, mut ahead_state) = (41_u64, 41_u64);
            let mut varied = std::collections::HashSet::new();
            for step in 0..40_u32 {
                let this = step_draft(step, gpu);
                let expected = draft(&mut stock, &device, &mut registry, &target, &this);
                let adopted_before = ahead.prefetch_counts().adopted;
                let actual = draft(&mut ahead, &device, &mut registry, &target, &this);
                assert!(
                    actual == expected,
                    "staged={staged} gpu={gpu} step {step}: draft outputs differ"
                );
                assert_eq!(
                    ahead.prefetch_counts().adopted - adopted_before,
                    u64::from(step > 0),
                    "staged={staged} gpu={gpu} step {step}: the prefetched draft was not adopted"
                );
                varied.insert(expected[0].clone());

                let retained = [8, step % 9, 3, 0];
                capture(&mut stock, &mut stock_state);
                stock
                    .commit_target_context(&device, &mut registry, &retained)
                    .unwrap();

                capture(&mut ahead, &mut ahead_state);
                let next = step_draft(step + 1, gpu);
                let (committed, context) = commit_and_prefetch(
                    &mut ahead,
                    &device,
                    &mut registry,
                    &target,
                    &replay,
                    &retained,
                    Some(&next),
                    &[0, 1, 2, 3],
                    false,
                );
                assert!(committed);
                context.unwrap();
                assert_eq!(ahead.lane_lengths, stock.lane_lengths, "step {step}");
            }
            // Everything after the last prefetch: the rings need the queue idle.
            ahead.settle_prefetch().unwrap();
            assert!(
                rings(&ahead) == rings(&stock),
                "staged={staged} gpu={gpu}: the prefetch order left other ring bytes"
            );
            assert!(stock.lane_lengths[0] > stock.config.sliding_window);
            assert!(varied.len() > 8, "the draft outputs never changed");
            let counts = ahead.prefetch_counts();
            assert_eq!(
                (counts.prefetched, counts.adopted, counts.discarded),
                (40, 39, 1)
            );
        }
    }
}

#[test]
fn a_prefetch_that_does_not_match_the_next_step_is_dropped() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    for staged in [false, true] {
        let mut registry = KernelRegistry::new();
        let mut stock = runtime(&device, staged);
        let mut ahead = runtime(&device, staged);
        let target = Target::new(&device, &stock.config);
        let replay = Replay::new(&device);
        let (mut stock_state, mut ahead_state) = (43_u64, 43_u64);
        let guessed = step_draft(3, true);
        // What the next step really asks for, each differing in one input.
        let actual_next = [
            Draft {
                anchors: [9, 9, 9, 9],
                ..guessed.clone()
            },
            Draft {
                maximum_rows: [8, 3, 8, 8],
                ..guessed.clone()
            },
            Draft {
                gpu_order: Some(vec![0, 2, 3]),
                ..guessed.clone()
            },
            Draft {
                gpu_order: Some(vec![3, 2, 1, 0]),
                ..guessed.clone()
            },
            Draft {
                gpu_order: None,
                ..guessed.clone()
            },
        ];
        for (case, real) in actual_next.iter().enumerate() {
            let retained = [8, 5, 0, 2];
            capture(&mut stock, &mut stock_state);
            stock
                .commit_target_context(&device, &mut registry, &retained)
                .unwrap();
            capture(&mut ahead, &mut ahead_state);
            let (_, context) = commit_and_prefetch(
                &mut ahead,
                &device,
                &mut registry,
                &target,
                &replay,
                &retained,
                Some(&guessed),
                &[0, 1, 2, 3],
                false,
            );
            context.unwrap();
            // The step's outputs are the next draft's now.
            assert!(ahead.proposals().is_err() && ahead.verify_rows().is_err());
            assert!(ahead.select_verify_rows(&[0], &[8, 8, 8, 8]).is_err());

            let counts = ahead.prefetch_counts();
            let expected = draft(&mut stock, &device, &mut registry, &target, real);
            let outputs = draft(&mut ahead, &device, &mut registry, &target, real);
            assert!(
                outputs == expected,
                "staged={staged} case {case}: a dropped prefetch changed the draft"
            );
            let after = ahead.prefetch_counts();
            assert_eq!(
                (after.adopted, after.discarded),
                (counts.adopted, counts.discarded + 1),
                "staged={staged} case {case}"
            );
            assert!(
                rings(&ahead) == rings(&stock),
                "staged={staged} case {case}"
            );
        }

        // A ring change between the steps: a cleared lane, a prefill ingest.
        let width = stock.config.fc_input_dim();
        let ingest = DFlashCaptureSession {
            target_layer_ids: stock.config.target_layer_ids.clone(),
            hidden_output: {
                let mut state = 47_u64;
                (0..11 * width).map(|_| random(&mut state)).collect()
            },
            per_position_argmaxes: None,
            seq_len: 11,
            hidden_size: width / stock.config.target_layer_ids.len(),
        };
        for change in 0..2 {
            let retained = [3, 8, 8, 1];
            let next = step_draft(20 + change, true);
            capture(&mut stock, &mut stock_state);
            stock
                .commit_target_context(&device, &mut registry, &retained)
                .unwrap();
            capture(&mut ahead, &mut ahead_state);
            commit_and_prefetch(
                &mut ahead,
                &device,
                &mut registry,
                &target,
                &replay,
                &retained,
                Some(&next),
                &[0, 1, 2, 3],
                false,
            )
            .1
            .unwrap();
            let discarded = ahead.prefetch_counts().discarded;
            for runtime in [&mut stock, &mut ahead] {
                if change == 0 {
                    runtime.clear_lane(2).unwrap();
                } else {
                    let start = runtime.lane_lengths[1];
                    runtime
                        .ingest_target_capture(&device, &mut registry, &ingest, 1, start)
                        .unwrap();
                }
            }
            assert_eq!(ahead.prefetch_counts().discarded, discarded + 1);
            assert!(!ahead.prefetch_outstanding());
            let expected = draft(&mut stock, &device, &mut registry, &target, &next);
            let outputs = draft(&mut ahead, &device, &mut registry, &target, &next);
            assert!(
                outputs == expected,
                "staged={staged} change {change}: a draft over changed rings was adopted"
            );
            assert!(rings(&ahead) == rings(&stock));
            assert_eq!(ahead.lane_lengths, stock.lane_lengths);
        }
    }
}

#[test]
fn a_step_that_fails_after_its_submissions_leaves_no_prefetched_draft() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    for staged in [false, true] {
        let mut registry = KernelRegistry::new();
        let mut runtime = runtime(&device, staged);
        let target = Target::new(&device, &runtime.config);
        let replay = Replay::new(&device);
        let mut state = 53_u64;
        let mut step = 0;
        // History in every lane, past a wrap in lanes 0 and 2.
        for _ in 0..36 {
            draft(
                &mut runtime,
                &device,
                &mut registry,
                &target,
                &step_draft(step, true),
            );
            capture(&mut runtime, &mut state);
            runtime
                .commit_target_context(&device, &mut registry, &[8, 5, 8, 2])
                .unwrap();
            step += 1;
        }
        let lengths = runtime.lane_lengths.clone();
        let untouched = [lane_rings(&runtime, 2), lane_rings(&runtime, 3)];

        // The failing step holds lanes 0 and 1.
        draft(
            &mut runtime,
            &device,
            &mut registry,
            &target,
            &step_draft(step, true),
        );
        capture(&mut runtime, &mut state);
        let next = Draft {
            gpu_order: Some(vec![0, 1]),
            ..step_draft(step + 1, true)
        };
        let (committed, context) = commit_and_prefetch(
            &mut runtime,
            &device,
            &mut registry,
            &target,
            &replay,
            &[8, 3, 0, 0],
            Some(&next),
            &[0, 1],
            true,
        );
        assert!(!committed);
        context.expect("both commands were waited for and the lanes cleared");
        assert!(!runtime.prefetch_outstanding());
        assert_eq!(runtime.lane_lengths, [0, 0, lengths[2], lengths[3]]);
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
        // The cleared lanes draft and commit like fresh ones.
        let adopted = runtime.prefetch_counts().adopted;
        draft(
            &mut runtime,
            &device,
            &mut registry,
            &target,
            &Draft {
                gpu_order: Some(vec![0, 1]),
                ..step_draft(step + 1, true)
            },
        );
        assert_eq!(runtime.prefetch_counts().adopted, adopted);
        capture(&mut runtime, &mut state);
        runtime
            .commit_target_context(&device, &mut registry, &[8, 8, 0, 0])
            .unwrap();
        assert_eq!(runtime.lane_lengths, [8, 8, lengths[2], lengths[3]]);
    }
}
