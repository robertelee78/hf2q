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

/// Real-artifact gate, run explicitly with the served switches on one model:
/// `HF2Q_QWEN38_GGUF=… HF2Q_DFLASH2_CONFIG=… HF2Q_DFLASH2_WEIGHTS=… HF2Q_DFLASH2_DRAFT_AF4_G64=1
/// HF2Q_DFLASH2_DRAFT_STAGED=1 <test binary> a_joint_lane_holds_the_capture_and_draft_rings_of_its_solo_prefill_on_the_real_checkpoint
/// --ignored --nocapture --test-threads=1` (or `HF2Q_QWEN36_GGUF=…` with the served MoE switches).
///
/// The whole path of a joint prompt prefill on the real target and the real
/// draft checkpoint, against each lane's own prefill:
///
/// - the first turn of the four-agent workload (four first prompts of 720
///   content rows and a five-row cue, 444 tokens in common, shared in the
///   forward) and a later turn (resumed chunks of 31, 157, 200 and 266
///   content rows and their cues), each as one joint forward with the
///   DFlash2 target-layer capture, and each lane alone as its folded forward
///   on a cache whose other slots are empty;
/// - a lane's rows of the joint capture must be its own capture, bit for bit;
/// - a draft runtime fed from the joint capture through the engine's calls
///   (row runs, one call per segment) and one fed from the solo captures must
///   hold the same rings, lane cursors and token bindings after each turn,
///   and a draft cycle on each must give the same logits, proposals and
///   verify rows.
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF or HF2Q_QWEN36_GGUF, HF2Q_DFLASH2_CONFIG, HF2Q_DFLASH2_WEIGHTS and Apple Metal; run alone"]
fn a_joint_lane_holds_the_capture_and_draft_rings_of_its_solo_prefill_on_the_real_checkpoint() {
    use crate::inference::models::qwen35::forward_gpu::JointPrefillLane;
    use crate::inference::models::qwen35::kv_cache::HybridKvCache;
    use crate::inference::models::qwen35::model::Qwen35Model;
    use crate::inference::models::qwen35::prefill_prepared::PrefillScope;
    use crate::inference::models::qwen35::spec_decode::positions_for_range;
    use crate::inference::spec_decode::dflash2::adapters::QwenDFlash2Adapter;
    use crate::inference::spec_decode::dflash2::schema::{QWEN36_DFLASH2, QWEN38_DFLASH2};
    use crate::serve::header::LoadProgress;
    use crate::serve::multi_seq_kv::SlotId;
    use std::path::PathBuf;

    const PATTERN: [u32; 7] = [9707, 374, 279, 15, 1_782, 2_610, 13];
    const COMMON: usize = 444;
    const CONTENT: usize = 720;
    const CUE: usize = 5;
    const REPLY: usize = 150;
    const LATER: [usize; 4] = [31, 157, 200, 266];
    const SLOTS: u32 = 4;
    const CAPACITY: u32 = 4096;

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let (gguf_path, descriptor) = match (
        std::env::var_os("HF2Q_QWEN38_GGUF"),
        std::env::var_os("HF2Q_QWEN36_GGUF"),
    ) {
        (Some(path), None) => (PathBuf::from(path), &QWEN38_DFLASH2),
        (None, Some(path)) => (PathBuf::from(path), &QWEN36_DFLASH2),
        _ => panic!("set exactly one of HF2Q_QWEN38_GGUF and HF2Q_QWEN36_GGUF"),
    };
    let config_path =
        PathBuf::from(std::env::var_os("HF2Q_DFLASH2_CONFIG").expect("HF2Q_DFLASH2_CONFIG"));
    let weights_path =
        PathBuf::from(std::env::var_os("HF2Q_DFLASH2_WEIGHTS").expect("HF2Q_DFLASH2_WEIGHTS"));
    let policy =
        crate::inference::models::qwen35::routing_policy::resolve_qwen_ggml_routing_policy(
            &gguf_path,
        );
    let gguf = mlx_native::gguf::GgufFile::open(&gguf_path).expect("open target GGUF");
    let mut progress = LoadProgress::new(false, 1, 0);
    let mut model = Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load target");
    model.bind_ggml_routing_policy(policy);
    model.ensure_gpu_cache_primed().expect("prime GPU cache");
    let model = &model;
    let hidden = model.cfg.hidden_size as usize;

    let token = |slot: u32, position: usize| -> u32 {
        let seed = if position < COMMON { 0 } else { slot as usize + 1 };
        PATTERN[(position + seed) % PATTERN.len()]
    };
    let tokens_of = |slot: u32, start: usize, rows: usize| -> Vec<u32> {
        (start..start + rows).map(|position| token(slot, position)).collect()
    };
    let new_cache = || -> HybridKvCache {
        model
            .with_gpu_cache_mut(|device, _registry| {
                HybridKvCache::new_with_options(&model.cfg, device, CAPACITY, SLOTS, true)
            })
            .expect("allocate TQ cache")
    };
    let config_bytes = std::fs::read(&config_path).expect("read draft config");
    let config = descriptor.parse_config(&config_bytes).expect("parse draft config");
    let layer_ids = config.target_layer_ids.clone();
    let adapter = QwenDFlash2Adapter::new(model, &config).expect("draft adapter");
    let load_runtime = || -> DFlash2Runtime {
        model
            .with_gpu_cache_mut(|device, _registry| {
                DFlash2Runtime::load_verified(
                    device,
                    descriptor,
                    &config_path,
                    &weights_path,
                    &adapter,
                    SLOTS as usize,
                )
            })
            .expect("load the draft checkpoint")
    };
    let mut together = load_runtime();
    let mut alone = load_runtime();
    eprintln!(
        "[JOINT_RINGS] {} staged={} target layers {layer_ids:?}",
        descriptor.id,
        together.workspace.staged.is_some()
    );

    // One cache holds the four slots of the joint path; each solo lane runs
    // on a cache of its own, as its own prefill on empty peers.
    let mut joint_cache = new_cache();
    let mut solo_caches: Vec<HybridKvCache> = (0..SLOTS).map(|_| new_cache()).collect();
    // (name, shared rows, per slot: start, content rows)
    let first: Vec<(usize, usize)> = (0..SLOTS).map(|_| (0, CONTENT)).collect();
    let later: Vec<(usize, usize)> = LATER
        .iter()
        .map(|&content| (CONTENT + CUE + REPLY, content))
        .collect();
    for (turn, shared, shape) in [("first turn", COMMON, first), ("later turn", 0, later)] {
        if shared == 0 {
            // The reply rows between the turns, on every cache alike.
            for slot in 0..SLOTS {
                let reply = tokens_of(slot, CONTENT + CUE, REPLY);
                let positions = positions_for_range((CONTENT + CUE) as i32, REPLY);
                for cache in [&mut joint_cache, &mut solo_caches[slot as usize]] {
                    let _scope = PrefillScope::forced(true);
                    let mut session =
                        DFlashCaptureSession::new(layer_ids.clone(), REPLY, hidden, false);
                    model
                        .forward_gpu_last_logits_with_hidden_dflash(
                            &reply,
                            &positions,
                            cache,
                            Some(&mut session),
                            SlotId(slot),
                        )
                        .expect("reply rows");
                }
                // The draft lanes see the reply rows too (the same rows on
                // both runtimes: decode commits them in service).
                let mut seed = 0xD0_u64 + slot as u64;
                let reply_capture = capture(REPLY, config.fc_input_dim(), &mut seed);
                let reply_capture = DFlashCaptureSession {
                    target_layer_ids: layer_ids.clone(),
                    hidden_size: hidden,
                    ..reply_capture
                };
                model
                    .with_gpu_cache_mut(|device, registry| {
                        for runtime in [&mut together, &mut alone] {
                            runtime.ingest_target_capture(
                                device,
                                registry,
                                &reply_capture,
                                slot as usize,
                                CONTENT + CUE,
                            )?;
                            runtime.record_lane_tokens(slot as usize, &reply)?;
                        }
                        Ok(())
                    })
                    .expect("reply rows into the draft lanes");
            }
        }
        let chunks: Vec<(Vec<u32>, Vec<i32>)> = (0..SLOTS)
            .map(|slot| {
                let (start, content) = shape[slot as usize];
                (
                    tokens_of(slot, start, content + CUE),
                    positions_for_range(start as i32, content + CUE),
                )
            })
            .collect();
        let lanes: Vec<JointPrefillLane<'_>> = (0..SLOTS)
            .map(|slot| {
                let (start, content) = shape[slot as usize];
                JointPrefillLane {
                    slot_id: SlotId(slot),
                    tokens: &chunks[slot as usize].0,
                    positions_flat: &chunks[slot as usize].1,
                    shares_prefix: shared > 0,
                    fold: Some((content, start + content)),
                    image: None,
                }
            })
            .collect();
        let total = Qwen35Model::joint_prefill_rows(&lanes, shared);
        let mut joint_capture = DFlashCaptureSession::new(layer_ids.clone(), total, hidden, false);
        let outputs = {
            let _scope = PrefillScope::forced(true);
            model
                .forward_gpu_last_logits_joint(
                    &lanes,
                    shared,
                    &mut joint_cache,
                    Some(&mut joint_capture),
                )
                .unwrap_or_else(|error| panic!("{turn}: joint forward: {error:#}"))
        };
        for slot in 0..SLOTS {
            let (start, content) = shape[slot as usize];
            let rows = content + CUE;
            let (tokens, positions) = &chunks[slot as usize];
            let mut solo_capture = DFlashCaptureSession::new(layer_ids.clone(), rows, hidden, false);
            {
                let _scope = PrefillScope::forced(true);
                let image = solo_caches[slot as usize]
                    .boundary_anchor_image()
                    .expect("checkpoint storage");
                model
                    .forward_gpu_last_logits_folded(
                        tokens,
                        positions,
                        &mut solo_caches[slot as usize],
                        Some(&mut solo_capture),
                        SlotId(slot),
                        content,
                        start + content,
                        image,
                    )
                    .unwrap_or_else(|error| panic!("{turn}: solo forward of slot {slot}: {error:#}"));
            }
            // The lane's rows of the joint capture are its own capture.
            let output = &outputs[slot as usize];
            for layer in 0..layer_ids.len() {
                let joint_rows: Vec<u32> = output
                    .forward_rows(0..rows)
                    .into_iter()
                    .flat_map(|run| {
                        let at = layer * total + run.start;
                        joint_capture.hidden_output[at * hidden..(at + run.len()) * hidden]
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let solo_rows = solo_capture.hidden_output
                    [layer * rows * hidden..(layer + 1) * rows * hidden]
                    .iter()
                    .map(|value| value.to_bits());
                assert!(
                    joint_rows.iter().copied().eq(solo_rows),
                    "{turn}: slot {slot}, target layer {}: the joint capture differs from the solo capture",
                    layer_ids[layer]
                );
            }
            // The engine's ingest of each, segment by segment.
            model
                .with_gpu_cache_mut(|device, registry| {
                    for segment in [0..content, content..rows] {
                        together.ingest_target_capture_runs(
                            device,
                            registry,
                            &joint_capture,
                            slot as usize,
                            start + segment.start,
                            &output.forward_rows(segment.clone()),
                        )?;
                        together.record_lane_tokens(slot as usize, &tokens[segment.clone()])?;
                        alone.ingest_target_capture_rows(
                            device,
                            registry,
                            &solo_capture,
                            slot as usize,
                            start + segment.start,
                            segment.clone(),
                        )?;
                        alone.record_lane_tokens(slot as usize, &tokens[segment])?;
                    }
                    Ok(())
                })
                .unwrap_or_else(|error| panic!("{turn}: draft ingest of slot {slot}: {error:#}"));
        }
        let (joint, solo) = (state(&together), state(&alone));
        let differing: usize = joint
            .0
            .iter()
            .zip(&solo.0)
            .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
            .sum();
        assert!(
            joint.0.iter().flatten().any(|&bits| bits != 0),
            "{turn}: nothing was appended"
        );
        assert!(
            differing == 0,
            "{turn}: {differing} ring values of the joint lanes differ from their solo prefill's"
        );
        assert_eq!(joint.1, solo.1, "{turn}: lane cursors");
        assert_eq!(joint.2, solo.2, "{turn}: bound token counts");
        assert!(joint.3 == solo.3, "{turn}: token bindings");
        eprintln!(
            "[JOINT_RINGS] {} {turn}: {total} forward rows, {shared} shared: captures, rings, cursors and bindings identical (lane lengths {:?})",
            descriptor.id, joint.1
        );
    }

    // A draft cycle on each context.
    let lanes = SLOTS as usize;
    let proposals = together.config.proposal_tokens();
    let cycle = |runtime: &mut DFlash2Runtime| -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let positions: Vec<u32> = runtime
            .lane_lengths()
            .iter()
            .map(|length| *length as u32)
            .collect();
        model
            .with_gpu_cache_mut(|device, registry| {
                runtime.run_draft(
                    device,
                    registry,
                    &adapter,
                    &vec![1; lanes],
                    &positions,
                    &vec![0.0; lanes],
                    0,
                    &vec![8; lanes],
                    &vec![0.5; lanes * proposals],
                )?;
                Ok((
                    runtime.proposals()?.to_vec(),
                    runtime.verify_rows()?.to_vec(),
                    runtime.workspace.logits.as_slice::<u32>()?.to_vec(),
                ))
            })
            .expect("draft cycle")
    };
    let joint = cycle(&mut together);
    let solo = cycle(&mut alone);
    assert_eq!(joint.0, solo.0, "draft proposals");
    assert_eq!(joint.1, solo.1, "verify rows");
    assert!(joint.2 == solo.2, "draft logits");
    eprintln!(
        "[JOINT_RINGS] {} draft cycle after the joint and after the solo prefill: logits, proposals and verify rows identical",
        descriptor.id
    );
}
