//! A lane of a joint prompt prefill appends to its draft rings what its own
//! prefill appends (`HF2Q_QWEN_PREFILL_JOINT`).
//!
//! A joint prefill forward captures the target layers of every lane's rows
//! in one capture: the rows the sharing lanes have in common first, then
//! each lane's own rows. The engine appends a lane's chunk to that lane's
//! draft rings from the row runs of the capture that hold it
//! ([`DFlash2Runtime::ingest_target_capture_runs`] with
//! `JointPrefillOutput::forward_rows`), one call per segment of the chunk
//! (the content rows, then the cue rows of a folded chunk), and binds the
//! lane to the chunk's tokens.
//!
//! The lane's own prefill captures the same rows side by side (the joint
//! forward's rows of a lane are bitwise its own forward's:
//! `prefill_joint_tests`) and appends them with the same calls over a
//! contiguous row range. These tests take one capture of a joint forward,
//! cut each lane's own capture out of it, and require that both ways leave
//! every lane's rings, cursor and token binding identical, bit for bit, on
//! both draft routes: lanes that share a prefix and lanes that do not,
//! resumed lanes, folded cues, chunks shorter than a pass and chunks that
//! wrap the ring.
//!
//! Needs Apple Metal (`hf2q_gpu_test_lock`):
//!
//! ```text
//! cargo test --release --bin hf2q joint_capture_tests
//! ```

use mlx_native::{KernelRegistry, MlxDevice};

use super::context_commit_tests::{random, rings, runtime, LANES};
use super::*;
use crate::inference::models::qwen35::forward_gpu::JointPrefillOutput;
use crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession;

/// One lane of the joint forward.
#[derive(Clone, Copy, Debug)]
struct Lane {
    /// Draft lane (the slot).
    lane: usize,
    /// Rows the lane already holds (a resumed slot).
    history: usize,
    /// Whether the chunk starts with the forward's shared rows.
    shares: bool,
    /// Rows of the chunk, shared rows included.
    rows: usize,
    /// Content rows before a folded cue.
    fold: Option<usize>,
}

impl Lane {
    /// The segments the engine appends one after another.
    fn segments(&self) -> Vec<std::ops::Range<usize>> {
        match self.fold {
            Some(prefix) => vec![0..prefix, prefix..self.rows],
            None => vec![0..self.rows],
        }
    }

    fn tokens(&self) -> Vec<u32> {
        (0..self.rows)
            .map(|row| (self.lane * 1_000 + row) as u32)
            .collect()
    }
}

fn capture(rows: usize, width: usize, state: &mut u64) -> DFlashCaptureSession {
    DFlashCaptureSession {
        target_layer_ids: vec![0],
        hidden_output: (0..rows * width).map(|_| 2.0 * random(state)).collect(),
        per_position_argmaxes: None,
        seq_len: rows,
        hidden_size: width,
    }
}

/// The lanes' rings, cursors and bindings: `(rings, lengths, bound tokens,
/// binding digests)`.
fn state(runtime: &DFlash2Runtime) -> (Vec<Vec<u16>>, Vec<usize>, Vec<usize>, Vec<Vec<u8>>) {
    (
        rings(runtime),
        runtime.lane_lengths.clone(),
        runtime.lane_bindings.iter().map(|binding| binding.tokens).collect(),
        runtime
            .lane_bindings
            .iter()
            .map(|binding| binding.hasher.clone().finalize().to_vec())
            .collect(),
    )
}

fn joint_against_solo(staged: bool, shared: usize, lanes: &[Lane]) {
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let mut together = runtime(&device, staged);
    let mut alone = runtime(&device, staged);
    assert_eq!(together.workspace.staged.is_some(), staged);
    let width = together.config.fc_input_dim();
    assert!(lanes.iter().all(|lane| lane.lane < LANES));
    let mut seed = 0xC0FFEE_u64;

    // Each resumed lane's history, the same rows on both runtimes.
    for lane in lanes.iter().filter(|lane| lane.history > 0) {
        let history = capture(lane.history, width, &mut seed);
        let tokens: Vec<u32> = (0..lane.history).map(|row| (90_000 + row) as u32).collect();
        for runtime in [&mut together, &mut alone] {
            runtime
                .ingest_target_capture(&device, &mut registry, &history, lane.lane, 0)
                .unwrap();
            runtime.record_lane_tokens(lane.lane, &tokens).unwrap();
        }
    }

    // The joint forward's capture: the shared rows, then every lane's own.
    let mut placed = Vec::with_capacity(lanes.len());
    let mut total = shared;
    for lane in lanes {
        let lane_shared = if lane.shares { shared } else { 0 };
        assert!(lane.rows > lane_shared, "{lane:?} has no row of its own");
        placed.push(JointPrefillOutput::placed(total, lane_shared));
        total += lane.rows - lane_shared;
    }
    let forward = capture(total, width, &mut seed);

    for (lane, output) in lanes.iter().zip(&placed) {
        let tokens = lane.tokens();
        // The lane's own capture: its rows of the forward, side by side.
        let mut own = DFlashCaptureSession::new(vec![0], lane.rows, width, false);
        for (index, row) in output
            .forward_rows(0..lane.rows)
            .into_iter()
            .flatten()
            .enumerate()
        {
            own.hidden_output[index * width..(index + 1) * width]
                .copy_from_slice(&forward.hidden_output[row * width..(row + 1) * width]);
        }
        for segment in lane.segments() {
            let start = lane.history + segment.start;
            together
                .ingest_target_capture_runs(
                    &device,
                    &mut registry,
                    &forward,
                    lane.lane,
                    start,
                    &output.forward_rows(segment.clone()),
                )
                .unwrap();
            together
                .record_lane_tokens(lane.lane, &tokens[segment.clone()])
                .unwrap();
            alone
                .ingest_target_capture_rows(
                    &device,
                    &mut registry,
                    &own,
                    lane.lane,
                    start,
                    segment.clone(),
                )
                .unwrap();
            alone
                .record_lane_tokens(lane.lane, &tokens[segment])
                .unwrap();
        }
        assert_eq!(
            together.lane_lengths[lane.lane],
            lane.history + lane.rows,
            "{lane:?}"
        );
    }

    let (joint, solo) = (state(&together), state(&alone));
    assert!(
        joint.0.iter().flatten().any(|&bits| bits != 0),
        "nothing was appended"
    );
    let differing: usize = joint
        .0
        .iter()
        .zip(&solo.0)
        .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
        .sum();
    assert!(
        differing == 0,
        "staged {staged}, {shared} shared rows, {lanes:?}: {differing} ring values of the joint \
         lanes differ from their own prefill's"
    );
    assert_eq!(joint.1, solo.1, "staged {staged}: lane cursors");
    assert_eq!(joint.2, solo.2, "staged {staged}: bound token counts");
    assert!(joint.3 == solo.3, "staged {staged}: token bindings");
}

#[test]
fn a_joint_lane_appends_the_draft_rings_of_its_own_prefill() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let lane = |lane, history, shares, rows, fold| Lane {
        lane,
        history,
        shares,
        rows,
        fold,
    };
    for staged in [false, true] {
        // The four-agent first turn: four first prompts, 444 of 725 rows in
        // common, each folding its five cue rows (chunks longer than the
        // 256-slot ring).
        joint_against_solo(
            staged,
            444,
            &[
                lane(0, 0, true, 725, Some(720)),
                lane(1, 0, true, 725, Some(720)),
                lane(2, 0, true, 725, Some(720)),
                lane(3, 0, true, 725, Some(720)),
            ],
        );
        // A later turn: resumed lanes with nothing in common, in an order
        // that is not the lane order, a five-row cue lane among them.
        joint_against_solo(
            staged,
            0,
            &[
                lane(2, 150, false, 162, Some(157)),
                lane(0, 33, false, 5, None),
                lane(3, 200, false, 36, Some(31)),
                lane(1, 61, false, 271, Some(266)),
            ],
        );
        // Sharing and non-sharing lanes together: two first prompts with a
        // short common prefix beside a resumed lane and a first prompt that
        // shares nothing; one sharing lane is shorter than a pass.
        joint_against_solo(
            staged,
            40,
            &[
                lane(1, 0, true, 47, Some(42)),
                lane(3, 90, false, 80, None),
                lane(0, 0, true, 300, None),
                lane(2, 0, false, 64, Some(32)),
            ],
        );
    }
}
