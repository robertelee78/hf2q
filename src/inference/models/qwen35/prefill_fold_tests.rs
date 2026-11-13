//! Metal gates for the generation-cue fold (`HF2Q_QWEN_PREFILL_FOLD_CUE`,
//! [`super::prefill_fold`]).
//!
//! The fold is a Class A change: one forward over a content chunk and the
//! generation cue must leave exactly what the two forwards and the
//! checkpoint between them leave. These gates compare, bit for bit:
//! - the checkpoint at the stable boundary (cursors, DeltaNet image) and
//!   the logits of the last content row;
//! - the last-row logits;
//! - the captured target-layer rows of both segments;
//! - the slot afterwards: its DeltaNet state and, through every logits row
//!   of a follow-up chunk, its full-attention rows.
//!
//! They run the synthetic native K-quant model of the joint verify tests on
//! fresh and resumed slots, at chunk sizes on both sides of a 32-row tile
//! boundary, with and without the DFlash capture, in every slot of one- to
//! four-slot TQ and F16 caches, and on a variant whose FFN layers are
//! quantized MoE layers (the arm the fold builds once per segment). A
//! request that restores a folded checkpoint and diverges behind it reads
//! what it reads behind the two-forward checkpoint. The fold is refused, so
//! the scheduler keeps the two forwards, beside a bound GLP vector,
//! dump-bisect and the dense-Q split profile. The ignored real-artifact
//! gate in the middle of the file repeats the comparison on a served model
//! (`HF2Q_QWEN38_GGUF` or `HF2Q_QWEN36_GGUF`) and prints both walls. The
//! scheduler's side is in `serve::api::engine_qwen35`'s
//! `prefill_fold_scheduler_tests`.

use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use super::prefill_prepared::PrefillScope;
// The fixture rescaled so that every routed product of every layer reaches
// the logits: the joint-verify fixture's own scale saturates the residual
// stream after the first layer and would hide the attention the fold splits.
use super::prefill_prepared_tests::with_synthetic_model;
use super::spec_decode::positions_for_range;
use crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession;
use crate::serve::multi_seq_kv::SlotId;

const SLOTS: u32 = 2;
const CAPACITY: u32 = 256;
/// Target layers of the synthetic model the gates capture: DeltaNet layers
/// on and off a full-attention neighbour, and the last full-attention layer.
const CAPTURE_LAYERS: [usize; 4] = [1, 4, 6, 7];
/// Rows of the follow-up chunk that reads the slot back.
const FOLLOW_UP_ROWS: usize = 7;

fn token(slot: u32, position: usize) -> u32 {
    (slot * 131 + position as u32 * 37 + 11) % 512
}

fn tokens(slot: u32, start: usize, len: usize) -> Vec<u32> {
    (start..start + len)
        .map(|position| token(slot, position))
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

thread_local! {
    /// `(slots, TQ full-attention cache)` of the caches the gates allocate.
    static CACHE_SHAPE: std::cell::Cell<(u32, bool)> =
        const { std::cell::Cell::new((SLOTS, true)) };
}

/// Another cache shape for the gates on this thread, until dropped.
struct CacheShape;

impl CacheShape {
    fn set(slots: u32, tq: bool) -> Self {
        CACHE_SHAPE.with(|cell| cell.set((slots, tq)));
        Self
    }
}

impl Drop for CacheShape {
    fn drop(&mut self) {
        CACHE_SHAPE.with(|cell| cell.set((SLOTS, true)));
    }
}

fn cache(model: &Qwen35Model) -> HybridKvCache {
    let (slots, tq) = CACHE_SHAPE.with(std::cell::Cell::get);
    model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&model.cfg, device, CAPACITY, slots, tq)
        })
        .expect("allocate cache")
}

/// Everything a prefill over `[start, start + prefix + cue)` leaves behind.
#[derive(PartialEq)]
struct Outcome {
    boundary_logits: Vec<u32>,
    boundary: (usize, u64, u32, Vec<u32>, Option<u32>, Vec<u32>),
    logits: Vec<u32>,
    /// Captured rows per target layer, prefix rows then cue rows.
    captured: Vec<Vec<u32>>,
    /// The slot's DeltaNet state and cursors after the cue.
    after: (usize, u64, u32, Vec<u32>, Option<u32>, Vec<u32>),
    /// Every logits row of a chunk prefilled after the cue.
    follow_up: Vec<u32>,
}

impl Outcome {
    fn differences(&self, other: &Self) -> Vec<&'static str> {
        [
            (
                "boundary logits",
                self.boundary_logits == other.boundary_logits,
            ),
            ("boundary checkpoint", self.boundary == other.boundary),
            ("last-row logits", self.logits == other.logits),
            ("captured target layers", self.captured == other.captured),
            ("slot state after the cue", self.after == other.after),
            ("follow-up chunk", self.follow_up == other.follow_up),
        ]
        .into_iter()
        .filter_map(|(name, equal)| (!equal).then_some(name))
        .collect()
    }
}

/// How a prefill reaches the end of the cue.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Path {
    /// Content chunk, checkpoint, cue chunk.
    TwoForwards,
    /// One folded forward.
    Folded,
}

/// One forward of `len` rows at `start`, optionally capturing
/// [`CAPTURE_LAYERS`]. Returns the last-row logits and the captured rows per
/// layer.
fn forward(
    model: &Qwen35Model,
    cache: &mut HybridKvCache,
    slot: u32,
    start: usize,
    len: usize,
    capture: bool,
) -> (Vec<u32>, Vec<Vec<u32>>) {
    let hidden = model.cfg.hidden_size as usize;
    let chunk = tokens(slot, start, len);
    let positions = positions_for_range(start as i32, len);
    if !capture {
        let logits = model
            .forward_gpu_last_logits(&chunk, &positions, cache, SlotId(slot))
            .unwrap_or_else(|error| panic!("chunk start={start} n={len}: {error:#}"));
        return (bits(&logits), Vec::new());
    }
    let mut session = DFlashCaptureSession::new(CAPTURE_LAYERS.to_vec(), len, hidden, false);
    let (logits, _hidden) = model
        .forward_gpu_last_logits_with_hidden_dflash(
            &chunk,
            &positions,
            cache,
            Some(&mut session),
            SlotId(slot),
        )
        .unwrap_or_else(|error| panic!("captured chunk start={start} n={len}: {error:#}"));
    let rows = session
        .hidden_output
        .chunks(len * hidden)
        .map(bits)
        .collect();
    (bits(&logits), rows)
}

/// A folded forward as the scheduler runs it: the checkpoint storage is
/// allocated first, then the forward fills it.
#[allow(clippy::too_many_arguments)]
fn folded_forward(
    model: &Qwen35Model,
    tokens: &[u32],
    positions: &[i32],
    cache: &mut HybridKvCache,
    capture: Option<&mut DFlashCaptureSession>,
    slot: SlotId,
    prefix_rows: usize,
    boundary_len: usize,
) -> anyhow::Result<super::forward_gpu::FoldedPrefill> {
    let image = cache.boundary_anchor_image()?;
    model.forward_gpu_last_logits_folded(
        tokens,
        positions,
        cache,
        capture,
        slot,
        prefix_rows,
        boundary_len,
        image,
    )
}

/// Prefill `history` rows in `history_chunks` chunks, then the content and
/// cue rows along `path`, on `slot` of a cache whose other slot holds
/// `peer_rows` rows, on the prepared prefill route.
#[allow(clippy::too_many_arguments)]
fn prefill(
    model: &Qwen35Model,
    path: Path,
    slot: u32,
    history: &[usize],
    prefix: usize,
    cue: usize,
    capture: bool,
    peer_rows: usize,
) -> Outcome {
    prefill_on(
        model, true, path, slot, history, prefix, cue, capture, peer_rows,
    )
}

/// [`prefill`] on the prepared (`true`) or the stock prefill route.
#[allow(clippy::too_many_arguments)]
fn prefill_on(
    model: &Qwen35Model,
    prepared: bool,
    path: Path,
    slot: u32,
    history: &[usize],
    prefix: usize,
    cue: usize,
    capture: bool,
    peer_rows: usize,
) -> Outcome {
    let _scope = PrefillScope::forced(prepared);
    let mut cache = cache(model);
    if peer_rows > 0 {
        forward(model, &mut cache, 1 - slot, 0, peer_rows, false);
    }
    let mut start = 0;
    for &rows in history {
        forward(model, &mut cache, slot, start, rows, false);
        start += rows;
    }
    let boundary_len = start + prefix;
    let (boundary_logits, boundary, logits, captured) = match path {
        Path::TwoForwards => {
            let (boundary_logits, mut captured) =
                forward(model, &mut cache, slot, start, prefix, capture);
            let boundary = cache
                .snapshot_slot_anchor(SlotId(slot), boundary_len)
                .expect("checkpoint at the boundary")
                .test_words();
            let (logits, cue_rows) = forward(model, &mut cache, slot, boundary_len, cue, capture);
            for (layer, rows) in captured.iter_mut().zip(cue_rows) {
                layer.extend(rows);
            }
            (boundary_logits, boundary, logits, captured)
        }
        Path::Folded => {
            let hidden = model.cfg.hidden_size as usize;
            let len = prefix + cue;
            let mut session = capture
                .then(|| DFlashCaptureSession::new(CAPTURE_LAYERS.to_vec(), len, hidden, false));
            let folded = folded_forward(
                model,
                &tokens(slot, start, len),
                &positions_for_range(start as i32, len),
                &mut cache,
                session.as_mut(),
                SlotId(slot),
                prefix,
                boundary_len,
            )
            .unwrap_or_else(|error| {
                panic!("folded chunk start={start} n={prefix}+{cue}: {error:#}")
            });
            let captured = session.map_or_else(Vec::new, |session| {
                session
                    .hidden_output
                    .chunks(len * hidden)
                    .map(bits)
                    .collect()
            });
            (
                bits(&folded.boundary_logits),
                folded.boundary.expect("checkpoint").test_words(),
                bits(&folded.logits),
                captured,
            )
        }
    };
    let end = boundary_len + cue;
    let after = cache
        .snapshot_slot_anchor(SlotId(slot), end)
        .expect("slot state after the cue")
        .test_words();
    let follow_up = model
        .forward_gpu(
            &tokens(slot, end, FOLLOW_UP_ROWS),
            &positions_for_range(end as i32, FOLLOW_UP_ROWS),
            &mut cache,
            SlotId(slot),
        )
        .expect("follow-up chunk");
    Outcome {
        boundary_logits,
        boundary,
        logits,
        captured,
        after,
        follow_up: bits(&follow_up),
    }
}

/// `(history chunks, content rows, cue rows)`: fresh and resumed slots; a
/// content chunk whose last tile takes the cue, one whose cue starts a new
/// tile, one that ends on a tile boundary; the short-chunk direct TQ route
/// and the dequantized-history route for the content rows; the smallest
/// admitted fold and the longest admitted cue.
const SHAPES: [(&[usize], usize, usize); 8] = [
    (&[], 40, 5),
    (&[], 31, 5),
    (&[], 64, 5),
    (&[], 2, 2),
    (&[33], 20, 5),
    (&[64, 20], 40, 5),
    (&[17], 97, 8),
    (&[40], 37, 32),
];

#[test]
fn the_synthetic_model_folds_exactly_only_on_the_prepared_route() {
    with_synthetic_model(|model| {
        {
            let _scope = PrefillScope::forced(true);
            assert!(
                model.prefill_fold_exact().expect("exactness"),
                "every shared product of the synthetic model takes the prepared kernel"
            );
        }
        let _scope = PrefillScope::forced(false);
        assert!(
            !model.prefill_fold_exact().expect("exactness"),
            "the stock matmul picks its kernel by row count"
        );
    });
}

#[test]
fn a_folded_prefill_leaves_exactly_what_the_two_forwards_leave() {
    with_synthetic_model(|model| {
        for capture in [true, false] {
            for (history, prefix, cue) in SHAPES {
                let two = prefill(
                    model,
                    Path::TwoForwards,
                    0,
                    history,
                    prefix,
                    cue,
                    capture,
                    0,
                );
                let folded = prefill(model, Path::Folded, 0, history, prefix, cue, capture, 0);
                let differences = two.differences(&folded);
                assert!(
                    differences.is_empty(),
                    "history {history:?}, {prefix} + {cue} rows, capture {capture}: the fold \
                     differs from the two forwards in {differences:?}"
                );
                assert_eq!(
                    two.captured.len(),
                    if capture { CAPTURE_LAYERS.len() } else { 0 }
                );
            }
        }
    });
}

/// The comparison is not vacuous: on the stock route, whose matmul picks
/// its kernel by row count, the same fold changes the cue rows. That is why
/// the fold is taken only where [`Qwen35Model::prefill_fold_exact`] holds.
#[test]
fn on_the_stock_route_the_same_fold_changes_bits() {
    with_synthetic_model(|model| {
        let two = prefill_on(model, false, Path::TwoForwards, 0, &[], 40, 5, true, 0);
        let folded = prefill_on(model, false, Path::Folded, 0, &[], 40, 5, true, 0);
        let differences = two.differences(&folded);
        assert!(
            differences.contains(&"last-row logits"),
            "the stock route folded exactly: {differences:?}"
        );
    });
}

#[test]
fn a_folded_prefill_is_repeatable_and_does_not_depend_on_the_other_slot() {
    with_synthetic_model(|model| {
        let run = |peer_rows| prefill(model, Path::Folded, 1, &[33], 40, 5, true, peer_rows);
        let alone = run(0);
        // Other rows through the same pooled scratch and arenas.
        prefill(model, Path::Folded, 0, &[], 64, 5, false, 0);
        prefill(model, Path::TwoForwards, 0, &[17], 31, 5, true, 0);
        let again = run(0);
        assert!(
            alone.differences(&again).is_empty(),
            "a repeated folded prefill changed {:?}",
            alone.differences(&again)
        );
        let beside = run(70);
        assert!(
            alone.differences(&beside).is_empty(),
            "slot 1's folded prefill depends on what slot 0 holds: {:?}",
            alone.differences(&beside)
        );
        let two = prefill(model, Path::TwoForwards, 1, &[33], 40, 5, true, 70);
        assert!(
            two.differences(&beside).is_empty(),
            "slot 1 beside a busy slot 0: {:?}",
            two.differences(&beside)
        );
    });
}

#[test]
fn a_fold_outside_the_served_shape_is_refused_before_the_slot_changes() {
    with_synthetic_model(|model| {
        let _scope = PrefillScope::forced(true);
        let mut cache = cache(model);
        for (prefix, cue) in [(40, 1), (1, 5), (40, 33)] {
            let len = prefix + cue;
            let refused = folded_forward(
                model,
                &tokens(0, 0, len),
                &positions_for_range(0, len),
                &mut cache,
                None,
                SlotId(0),
                prefix,
                prefix,
            );
            assert!(refused.is_err(), "{prefix} + {cue} rows folded");
            assert_eq!(
                cache.sequence_len_for_slot(SlotId(0)).expect("cursor"),
                0,
                "a refused fold moved the slot"
            );
        }
    });
}

/// Real-artifact gate, run explicitly on one served model:
/// `HF2Q_QWEN38_GGUF=… <test binary> qwen_prefill_fold_real_model_gate --ignored --nocapture --test-threads=1`
/// (or `HF2Q_QWEN36_GGUF=…`; add `HF2Q_ENCODER_SESSION=1` for the served
/// encoder sessions).
///
/// On a TQ cache it prefills the chunk sizes of the four-agent workload, each
/// followed by its five-row cue, along both paths and checks that the fold
/// leaves the same bits as the two forwards and the checkpoint between them.
/// It prints each pair's wall time (second pass of each, so weights,
/// pipelines and prepared planes are warm).
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF or HF2Q_QWEN36_GGUF and Apple Metal; run alone"]
fn qwen_prefill_fold_real_model_gate() {
    use std::time::Instant;

    use crate::serve::header::LoadProgress;
    use mlx_native::gguf::GgufFile;

    const PATTERN: [u32; 7] = [9707, 374, 279, 15, 1_782, 2_610, 13];
    /// Content rows of consecutive turns; each is followed by the cue and by
    /// untimed rows standing in for the generated reply.
    const TURNS: [usize; 6] = [720, 31, 157, 200, 96, 266];
    const CUE: usize = 5;
    const REPLY: usize = 150;
    const GATE_CAPACITY: u32 = 4096;

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let (path, capture_layers) = match (
        std::env::var_os("HF2Q_QWEN38_GGUF"),
        std::env::var_os("HF2Q_QWEN36_GGUF"),
    ) {
        (Some(path), None) => (path, vec![5, 19, 33, 47, 61]),
        (None, Some(path)) => (path, vec![1, 6, 11, 16, 22, 27, 32, 37]),
        _ => panic!("set exactly one of HF2Q_QWEN38_GGUF and HF2Q_QWEN36_GGUF"),
    };
    let path = std::path::PathBuf::from(path);
    let policy = super::routing_policy::resolve_qwen_ggml_routing_policy(&path);
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let mut progress = LoadProgress::new(false, 1, 0);
    let mut model = Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load Qwen model");
    model.bind_ggml_routing_policy(policy);
    model.ensure_gpu_cache_primed().expect("prime GPU cache");
    let hidden = model.cfg.hidden_size as usize;
    let chunk_tokens = |start: usize, len: usize| -> Vec<u32> {
        (start..start + len)
            .map(|index| PATTERN[index % PATTERN.len()])
            .collect()
    };
    let new_cache = || {
        model
            .with_gpu_cache_mut(|device, _registry| {
                HybridKvCache::new_with_options(&model.cfg, device, GATE_CAPACITY, 1, true)
            })
            .expect("allocate TQ cache")
    };
    let _scope = PrefillScope::forced(true);
    assert!(
        model.prefill_fold_exact().expect("exactness"),
        "the fold is not exact for this artifact on the prepared route"
    );

    // One turn along `path`: the checkpoint, both logit rows and the
    // captured rows as bits, and the walls of its parts (content forward,
    // snapshot, cue forward; or the folded forward alone).
    let turn = |cache: &mut HybridKvCache, path: Path, start: usize, prefix: usize| {
        let ms = |since: Instant| since.elapsed().as_secs_f64() * 1e3;
        match path {
            Path::TwoForwards => {
                let mut content =
                    DFlashCaptureSession::new(capture_layers.clone(), prefix, hidden, false);
                let started = Instant::now();
                let (boundary_logits, _) = model
                    .forward_gpu_last_logits_with_hidden_dflash(
                        &chunk_tokens(start, prefix),
                        &positions_for_range(start as i32, prefix),
                        cache,
                        Some(&mut content),
                        SlotId(0),
                    )
                    .expect("content chunk");
                let content_ms = ms(started);
                let started = Instant::now();
                let boundary = cache
                    .snapshot_slot_anchor(SlotId(0), start + prefix)
                    .expect("checkpoint");
                let snapshot_ms = ms(started);
                let mut cue = DFlashCaptureSession::new(capture_layers.clone(), CUE, hidden, false);
                let started = Instant::now();
                let (logits, _) = model
                    .forward_gpu_last_logits_with_hidden_dflash(
                        &chunk_tokens(start + prefix, CUE),
                        &positions_for_range((start + prefix) as i32, CUE),
                        cache,
                        Some(&mut cue),
                        SlotId(0),
                    )
                    .expect("cue chunk");
                let cue_ms = ms(started);
                let captured: Vec<Vec<u32>> = content
                    .hidden_output
                    .chunks(prefix * hidden)
                    .zip(cue.hidden_output.chunks(CUE * hidden))
                    .map(|(content, cue)| {
                        let mut rows = bits(content);
                        rows.extend(bits(cue));
                        rows
                    })
                    .collect();
                (
                    [content_ms, snapshot_ms, cue_ms],
                    (boundary_logits, boundary.test_words(), logits, captured),
                )
            }
            Path::Folded => {
                let len = prefix + CUE;
                let mut session =
                    DFlashCaptureSession::new(capture_layers.clone(), len, hidden, false);
                let started = Instant::now();
                let folded = folded_forward(
                    &model,
                    &chunk_tokens(start, len),
                    &positions_for_range(start as i32, len),
                    cache,
                    Some(&mut session),
                    SlotId(0),
                    prefix,
                    start + prefix,
                )
                .expect("folded chunk");
                let folded_ms = ms(started);
                let captured = session
                    .hidden_output
                    .chunks(len * hidden)
                    .map(bits)
                    .collect();
                (
                    [folded_ms, 0.0, 0.0],
                    (
                        folded.boundary_logits,
                        folded.boundary.expect("checkpoint").test_words(),
                        folded.logits,
                        captured,
                    ),
                )
            }
        }
    };
    // Two passes per path: the first warms weights, pipelines and planes.
    let mut walls = Vec::new();
    let mut outcomes = Vec::new();
    for path in [Path::TwoForwards, Path::Folded] {
        for pass in 0..2 {
            let mut cache = new_cache();
            let mut start = 0;
            let mut pass_walls = Vec::new();
            let mut pass_outcomes = Vec::new();
            for prefix in TURNS {
                let (wall, (boundary_logits, boundary, logits, captured)) =
                    turn(&mut cache, path, start, prefix);
                assert!(logits.iter().all(|value| value.is_finite()));
                pass_walls.push(wall);
                pass_outcomes.push((bits(&boundary_logits), boundary, bits(&logits), captured));
                start += prefix + CUE;
                model
                    .forward_gpu_last_logits(
                        &chunk_tokens(start, REPLY),
                        &positions_for_range(start as i32, REPLY),
                        &mut cache,
                        SlotId(0),
                    )
                    .expect("reply rows");
                start += REPLY;
            }
            let after = cache
                .snapshot_slot_anchor(SlotId(0), start)
                .expect("final slot state")
                .test_words();
            if pass == 1 {
                walls.push(pass_walls);
                outcomes.push((pass_outcomes, after));
            }
        }
    }
    // The same rows as one ordinary forward (no boundary, no checkpoint):
    // what a chunk of that many rows costs, for the fold's own overhead.
    let mut plain = Vec::new();
    for _pass in 0..2 {
        let mut cache = new_cache();
        let mut start = 0;
        plain.clear();
        for prefix in TURNS {
            for (rows, timed) in [(prefix + CUE, true), (REPLY, false)] {
                let mut session =
                    DFlashCaptureSession::new(capture_layers.clone(), rows, hidden, false);
                let started = Instant::now();
                model
                    .forward_gpu_last_logits_with_hidden_dflash(
                        &chunk_tokens(start, rows),
                        &positions_for_range(start as i32, rows),
                        &mut cache,
                        Some(&mut session),
                        SlotId(0),
                    )
                    .expect("plain chunk");
                if timed {
                    plain.push(started.elapsed().as_secs_f64() * 1e3);
                }
                start += rows;
            }
        }
    }
    for (index, prefix) in TURNS.into_iter().enumerate() {
        let two: [f64; 3] = walls[0][index];
        let folded = walls[1][index][0];
        eprintln!(
            "[PREFILL_FOLD] rows={prefix}+{CUE} content_ms={:.1} snapshot_ms={:.1} cue_ms={:.1} \
             two_forwards_ms={:.1} folded_ms={folded:.1} saved_ms={:.1} one_plain_forward_ms={:.1}",
            two[0],
            two[1],
            two[2],
            two.iter().sum::<f64>(),
            two.iter().sum::<f64>() - folded,
            plain[index]
        );
    }
    let (two, folded) = (&outcomes[0], &outcomes[1]);
    for (index, prefix) in TURNS.into_iter().enumerate() {
        let (a, b) = (&two.0[index], &folded.0[index]);
        assert!(a.0 == b.0, "{prefix} rows: boundary logits differ");
        assert!(a.1 == b.1, "{prefix} rows: boundary checkpoint differs");
        assert!(a.2 == b.2, "{prefix} rows: last-row logits differ");
        assert!(a.3 == b.3, "{prefix} rows: captured target layers differ");
    }
    assert!(two.1 == folded.1, "final slot state differs");
}

#[test]
fn a_request_no_forward_took_is_an_error_and_does_not_linger() {
    use super::prefill_joint::{take_request, with_request, JointRequest, LaneBoundary, LaneRequest};
    with_synthetic_model(|model| {
        let image = cache(model)
            .boundary_anchor_image()
            .expect("checkpoint image");
        let outcome = with_request(
            JointRequest {
                shared_rows: 0,
                lanes: vec![LaneRequest {
                    slot: SlotId(0),
                    rows: 9,
                    shares_prefix: false,
                    boundary: Some(LaneBoundary {
                        prefix_rows: 4,
                        image,
                    }),
                }],
            },
            || Ok(()),
        );
        assert!(
            outcome.is_err(),
            "nothing took the fold or published logits"
        );
        assert!(take_request(9, SLOTS).expect("no request left").is_none());
    });
}

/// The fold is not exact, so the scheduler keeps the two forwards, for each
/// forward route that has no folded form.
fn no_fold_under(name: &str, routes: super::prefill_fold::UnfoldedRoutes) {
    with_synthetic_model(|model| {
        let _scope = PrefillScope::forced(true);
        assert!(
            model
                .prefill_fold_exact_under(super::prefill_fold::UnfoldedRoutes::default())
                .expect("exactness"),
            "the fixture folds exactly without {name}"
        );
        assert!(routes.any());
        assert!(
            !model.prefill_fold_exact_under(routes).expect("exactness"),
            "the fold stayed exact under {name}"
        );
    });
}

#[test]
fn the_fold_is_not_taken_with_a_glp_vector_bound() {
    use std::collections::BTreeMap;

    use crate::inference::glp::bind::BoundGlp;
    use crate::inference::glp::reader::{GlpHookPoint, GlpMode, GlpVector};

    no_fold_under(
        "a bound GLP vector",
        super::prefill_fold::UnfoldedRoutes {
            glp_bound: true,
            ..Default::default()
        },
    );
    // The model reports its own binding: what `hf2q serve --glp` leaves.
    super::prefill_prepared_tests::with_synthetic_model_mut(|model| {
        let _scope = PrefillScope::forced(true);
        assert!(model.prefill_fold_exact().expect("exactness"));
        let width = model.cfg.hidden_size as usize;
        model.glp = Some(BoundGlp {
            vector: GlpVector {
                mode: GlpMode::Add,
                hook_point: GlpHookPoint::ResidualStreamPostLayer,
                derived_at: None,
                alpha_default: 1.0,
                rank: 1,
                layers: BTreeMap::from([(3_u32, vec![0.25_f32; width])]),
                width,
                content_sha256: None,
                method: None,
                base_model_name: None,
            },
            alpha: 1.0,
            device_directions: BTreeMap::new(),
        });
        assert!(model.unfolded_routes().glp_bound);
        assert!(
            !model.prefill_fold_exact().expect("exactness"),
            "a model with a bound GLP vector still folds"
        );
    });
}

/// `HF2Q_MM_ID_ROUTING_THRESHOLD` moves the row count at which mlx-native
/// changes the MoE FFN's kernel family, and with it the bits of segments
/// between the two values: a MoE model then admits neither the fold nor a
/// joint forward (each request keeps its own forwards, whose bits are its
/// own by definition); a dense-FFN model has no such kernel and keeps both.
#[test]
fn a_moe_model_admits_no_fold_under_a_mm_id_threshold_override() {
    let overridden = super::prefill_fold::UnfoldedRoutes {
        moe_route_override: true,
        ..Default::default()
    };
    assert!(!overridden.any(), "the override concerns MoE layers only");
    with_synthetic_moe_model(|model| {
        let _scope = PrefillScope::forced(true);
        assert!(
            model
                .prefill_fold_exact_under(super::prefill_fold::UnfoldedRoutes::default())
                .expect("exactness"),
            "the synthetic MoE model does not fold at the stock threshold"
        );
        assert!(
            !model.prefill_fold_exact_under(overridden).expect("exactness"),
            "a MoE model still folds under a threshold override"
        );
    });
    with_synthetic_model(|model| {
        let _scope = PrefillScope::forced(true);
        assert!(
            model.prefill_fold_exact_under(overridden).expect("exactness"),
            "a dense-FFN model stopped folding under a MoE threshold override"
        );
    });
    // This process runs without the override: the production predicate
    // reads the environment, never a default.
    assert_eq!(
        super::prefill_fold::moe_route_override(),
        std::env::var_os("HF2Q_MM_ID_ROUTING_THRESHOLD").is_some()
    );
}

/// The same refusal through the process environment, which is what a
/// served worker reads: this test runs [`moe_fold_refusal_in_a_child`] in a
/// fresh process of this test binary with `HF2Q_MM_ID_ROUTING_THRESHOLD=64`
/// (the predicate and mlx-native both read the variable once per process).
#[test]
fn a_process_with_a_mm_id_threshold_override_admits_no_moe_fold() {
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "inference::models::qwen35::prefill_fold_tests::moe_fold_refusal_in_a_child",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("HF2Q_MM_ID_ROUTING_THRESHOLD", "64")
        .env("P834_OVERRIDE_CHILD", "1")
        .output()
        .expect("run the child test process");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.success() && text.contains("[OVERRIDE_CHILD] refused") && text.contains("1 passed"),
        "the child process under the override did not refuse the MoE fold:\n{text}"
    );
}

/// The child of [`a_process_with_a_mm_id_threshold_override_admits_no_moe_fold`];
/// a no-op in an ordinary test run.
#[test]
fn moe_fold_refusal_in_a_child() {
    if std::env::var_os("P834_OVERRIDE_CHILD").is_none() {
        return;
    }
    assert!(
        super::prefill_fold::moe_route_override(),
        "the override is not visible to the predicate"
    );
    with_synthetic_moe_model(|model| {
        let _scope = PrefillScope::forced(true);
        assert!(model.unfolded_routes().moe_route_override);
        assert!(
            !model.prefill_fold_exact().expect("exactness"),
            "a MoE model folds in a process with the threshold override"
        );
    });
    with_synthetic_model(|model| {
        let _scope = PrefillScope::forced(true);
        assert!(
            model.prefill_fold_exact().expect("exactness"),
            "a dense-FFN model stopped folding in a process with the MoE threshold override"
        );
    });
    eprintln!("[OVERRIDE_CHILD] refused");
}

#[test]
fn the_fold_is_not_taken_with_dump_bisect_enabled() {
    no_fold_under(
        "dump-bisect",
        super::prefill_fold::UnfoldedRoutes {
            dump_bisect: true,
            ..Default::default()
        },
    );
}

#[test]
fn the_fold_is_not_taken_under_the_dense_q_split_profile() {
    no_fold_under(
        "the dense-Q split profile",
        super::prefill_fold::UnfoldedRoutes {
            dense_q_split_profile: true,
            ..Default::default()
        },
    );
}

/// After the forward the slot holds both segments. A checkpoint that cannot
/// be assembled then is reported beside the logits, not instead of them: the
/// scheduler skips it as it skips a failed snapshot between two forwards.
#[test]
fn a_checkpoint_that_cannot_be_assembled_leaves_the_forward_and_the_slot_intact() {
    with_synthetic_model(|model| {
        let reference = prefill(model, Path::Folded, 0, &[], 40, 5, false, 0);
        let _scope = PrefillScope::forced(true);
        let mut cache = cache(model);
        // A boundary cursor at the chunk's end: no row lies past it.
        let folded = folded_forward(
            model,
            &tokens(0, 0, 45),
            &positions_for_range(0, 45),
            &mut cache,
            None,
            SlotId(0),
            40,
            45,
        )
        .expect("the forward itself");
        assert!(folded.boundary.is_err(), "a checkpoint behind the last row");
        assert!(bits(&folded.logits) == reference.logits);
        assert!(bits(&folded.boundary_logits) == reference.boundary_logits);
        let after = cache
            .snapshot_slot_anchor(SlotId(0), 45)
            .expect("slot state after the cue")
            .test_words();
        assert!(after == reference.after, "the slot differs after the cue");
    });
}

/// Shapes beside [`SHAPES`]: a cue that crosses a 32-row tile behind a
/// content chunk of more than 32 rows, a resumed chunk whose cue starts a
/// new tile, a content chunk that ends on a tile boundary behind a history,
/// and the short chunks the scheduler excludes (still exact at the model
/// level), fresh and resumed.
const SLOT_SHAPES: [(&[usize], usize, usize); 10] = [
    (&[], 40, 5),
    (&[], 60, 5),
    (&[50], 157, 5),
    (&[10], 96, 5),
    (&[], 30, 3),
    (&[100], 28, 5),
    (&[33], 20, 5),
    (&[64, 20], 40, 5),
    (&[17], 97, 8),
    (&[40], 37, 32),
];

/// The fold against the two forwards in the last slot of a one- to
/// four-slot cache, and in slot 1 beside a busy slot 0.
fn fold_at_one_to_four_slots(tq: bool) {
    with_synthetic_model(|model| {
        let mut failures = Vec::new();
        for slots in 1..=4_u32 {
            let _cache = CacheShape::set(slots, tq);
            let slot = slots - 1;
            for (history, prefix, cue) in SLOT_SHAPES {
                for capture in [true, false] {
                    if !capture && slots != 2 {
                        continue;
                    }
                    let two = prefill(
                        model,
                        Path::TwoForwards,
                        slot,
                        history,
                        prefix,
                        cue,
                        capture,
                        0,
                    );
                    let folded =
                        prefill(model, Path::Folded, slot, history, prefix, cue, capture, 0);
                    let differences = two.differences(&folded);
                    if !differences.is_empty() {
                        failures.push(format!(
                            "tq={tq} slots={slots} history={history:?} rows={prefix}+{cue} \
                             capture={capture}: {differences:?}"
                        ));
                    }
                }
            }
            if slots >= 2 {
                let two = prefill(model, Path::TwoForwards, 1, &[33], 40, 5, true, 70);
                let folded = prefill(model, Path::Folded, 1, &[33], 40, 5, true, 70);
                let differences = two.differences(&folded);
                if !differences.is_empty() {
                    failures.push(format!("tq={tq} slots={slots} busy peer: {differences:?}"));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    });
}

#[test]
fn a_folded_prefill_is_exact_at_one_to_four_slots_on_a_tq_cache() {
    fold_at_one_to_four_slots(true);
}

#[test]
fn a_folded_prefill_is_exact_at_one_to_four_slots_on_an_f16_cache() {
    fold_at_one_to_four_slots(false);
}

type AnchorWords = (usize, u64, u32, Vec<u32>, Option<u32>, Vec<u32>);

/// What a request that reuses the boundary checkpoint and diverges behind it
/// sees: every logits row of the diverging rows, the slot state after them,
/// and every logits row of a chunk behind them. `path = None` is the
/// control: nothing ever ran past the boundary on that slot.
fn restore_and_diverge(
    model: &Qwen35Model,
    path: Option<Path>,
    slot: u32,
    history: &[usize],
    prefix: usize,
    cue: usize,
) -> (Vec<u32>, AnchorWords, Vec<u32>) {
    const REPLY: usize = 6;
    const SUFFIX: usize = 11;
    let _scope = PrefillScope::forced(true);
    let mut cache = cache(model);
    let mut start = 0;
    for &rows in history {
        forward(model, &mut cache, slot, start, rows, false);
        start += rows;
    }
    let boundary_len = start + prefix;
    match path {
        None => {
            forward(model, &mut cache, slot, start, prefix, false);
        }
        Some(path) => {
            let anchor = match path {
                Path::TwoForwards => {
                    forward(model, &mut cache, slot, start, prefix, false);
                    let anchor = cache
                        .snapshot_slot_anchor(SlotId(slot), boundary_len)
                        .expect("checkpoint at the boundary");
                    forward(model, &mut cache, slot, boundary_len, cue, false);
                    anchor
                }
                Path::Folded => {
                    let len = prefix + cue;
                    folded_forward(
                        model,
                        &tokens(slot, start, len),
                        &positions_for_range(start as i32, len),
                        &mut cache,
                        None,
                        SlotId(slot),
                        prefix,
                        boundary_len,
                    )
                    .expect("folded chunk")
                    .boundary
                    .expect("checkpoint")
                }
            };
            // The rows a reply appends behind the cue.
            forward(model, &mut cache, slot, boundary_len + cue, REPLY, false);
            let plan = cache
                .plan_slot_anchor_restore(SlotId(slot), &anchor)
                .expect("restore plan");
            cache.apply_slot_anchor_restore(plan).expect("restore");
            assert_eq!(
                cache.sequence_len_for_slot(SlotId(slot)).expect("cursor") as usize,
                boundary_len,
                "the restore did not rewind to the boundary"
            );
        }
    }
    let diverged = model
        .forward_gpu(
            &tokens(slot + 7, boundary_len, SUFFIX),
            &positions_for_range(boundary_len as i32, SUFFIX),
            &mut cache,
            SlotId(slot),
        )
        .expect("diverging rows");
    let end = boundary_len + SUFFIX;
    let after = cache
        .snapshot_slot_anchor(SlotId(slot), end)
        .expect("slot state after the diverging rows")
        .test_words();
    let follow_up = model
        .forward_gpu(
            &tokens(slot, end, FOLLOW_UP_ROWS),
            &positions_for_range(end as i32, FOLLOW_UP_ROWS),
            &mut cache,
            SlotId(slot),
        )
        .expect("follow-up chunk");
    (bits(&diverged), after, bits(&follow_up))
}

/// A later request restores the checkpoint a folded forward took and
/// diverges behind it. It sees what it sees after the two-forward
/// checkpoint, and both see the logits, DeltaNet state and follow-up rows of
/// a slot that never ran past the boundary: no cue or reply row leaks into
/// the restored prefix.
#[test]
fn a_request_diverging_behind_a_folded_checkpoint_sees_the_two_forward_bytes() {
    const CASES: [(&[usize], usize, usize); 6] = [
        (&[], 40, 5),
        (&[33], 20, 5),
        (&[64, 20], 40, 5),
        (&[], 60, 5),
        (&[17], 97, 8),
        (&[50], 157, 5),
    ];
    with_synthetic_model(|model| {
        let mut failures = Vec::new();
        for (slots, tq) in [(1_u32, true), (3, true), (2, false)] {
            let _cache = CacheShape::set(slots, tq);
            let slot = slots - 1;
            for (history, prefix, cue) in CASES {
                let run = |path| restore_and_diverge(model, path, slot, history, prefix, cue);
                let two = run(Some(Path::TwoForwards));
                let folded = run(Some(Path::Folded));
                let control = run(None);
                // The state generation and cursors (`.0`-`.2`, `.4`) follow
                // the slot's own history; the rows and the DeltaNet words
                // are what a request reads.
                let reads_the_control = |run: &(Vec<u32>, AnchorWords, Vec<u32>)| {
                    run.0 == control.0
                        && run.1 .3 == control.1 .3
                        && run.1 .5 == control.1 .5
                        && run.2 == control.2
                };
                if two != folded || !reads_the_control(&two) || !reads_the_control(&folded) {
                    failures.push(format!(
                        "tq={tq} slots={slots} history={history:?} {prefix}+{cue}: \
                         folded equals two forwards: {}, two forwards read the control: {}, \
                         folded reads the control: {}",
                        two == folded,
                        reads_the_control(&two),
                        reads_the_control(&folded)
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    });
}

/// The synthetic model with every dense FFN replaced by a quantized MoE FFN
/// (four experts, two per row, a shared expert), so a prefill forward takes
/// the MoE arm the fold builds once per segment.
fn with_synthetic_moe_model(run: impl FnOnce(&Qwen35Model)) {
    use mlx_native::ops::quantized_matmul_ggml::GgmlType;
    use mlx_native::MlxDevice;

    use super::joint_decode_test_support::{random_f32, synthetic_blocks};
    use super::joint_verify_tests::{padded_policy, synthetic_model};
    use super::model::Qwen35FfnWeights;
    use super::prefill_prepared_tests::{rescale_routed_weights, set_scale, OBSERVABLE_SCALE};
    use super::weight_loader::MoeFfnWeightsQ;
    use super::{Qwen35MoeConfig, Qwen35Variant};

    const EXPERTS: usize = 4;
    const EXPERT_ROWS: usize = 256;
    const SHARED_ROWS: usize = 256;

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut model = synthetic_model(&device, padded_policy(), false);
    let hidden = model.cfg.hidden_size as usize;
    let mut seed = 0x51CE_0E57_u32;
    let stack = |ty: GgmlType, rows: usize, cols: usize, seed: &mut u32| {
        let blocks = synthetic_blocks(&device, ty, (EXPERTS * rows) as u32, cols as u32, seed);
        set_scale(&blocks, ty, OBSERVABLE_SCALE);
        blocks
    };
    let small = |seed: &mut u32, n: usize| -> Vec<f32> {
        random_f32(seed, n)
            .into_iter()
            .map(|value| value * 0.05)
            .collect()
    };
    for layer in &mut model.layers {
        *layer.ffn_mut() = Qwen35FfnWeights::MoeQ(MoeFfnWeightsQ {
            router: random_f32(&mut seed, EXPERTS * hidden),
            expert_gate_q: stack(GgmlType::Q4_K, EXPERT_ROWS, hidden, &mut seed),
            expert_up_q: stack(GgmlType::Q4_K, EXPERT_ROWS, hidden, &mut seed),
            expert_down_q: stack(GgmlType::Q6_K, hidden, EXPERT_ROWS, &mut seed),
            ggml_type_gate_up: GgmlType::Q4_K,
            ggml_type_down: GgmlType::Q6_K,
            shared_gate_logit: random_f32(&mut seed, hidden),
            shared_gate: small(&mut seed, SHARED_ROWS * hidden),
            shared_up: small(&mut seed, SHARED_ROWS * hidden),
            shared_down: small(&mut seed, hidden * SHARED_ROWS),
            native_dense: None,
            expert_gate_affine: None,
            expert_up_affine: None,
            expert_down_affine: None,
        });
    }
    model.cfg.variant = Qwen35Variant::Moe;
    model.cfg.moe = Some(Qwen35MoeConfig {
        moe_intermediate_size: EXPERT_ROWS as u32,
        num_experts: EXPERTS as u32,
        num_experts_per_tok: 2,
        shared_expert_intermediate_size: SHARED_ROWS as u32,
    });
    rescale_routed_weights(&model);
    model
        .ensure_gpu_cache_primed()
        .expect("prime the GPU cache for the MoE layers");
    run(&model);
}

/// The MoE FFN picks its router and shared-expert kernels by row count, so a
/// folded forward builds it once per segment. On a model whose FFN layers
/// are quantized MoE layers the fold leaves what the two forwards leave.
#[test]
fn a_folded_prefill_through_moe_layers_leaves_what_the_two_forwards_leave() {
    use mlx_native::ops::quantized_matmul_ggml::GgmlType;

    use super::model::Qwen35FfnWeights;
    use super::prefill_prepared_tests::xor_quants;

    with_synthetic_moe_model(|model| {
        {
            let _scope = PrefillScope::forced(true);
            assert!(
                model.prefill_fold_exact().expect("exactness"),
                "the MoE fixture does not fold exactly on the prepared route"
            );
        }
        // The expert stacks are live: the last layer's reaches the logits.
        let reference = prefill(model, Path::TwoForwards, 0, &[], 40, 5, false, 0);
        let Qwen35FfnWeights::MoeQ(last) = model.layers.last().expect("layers").ffn() else {
            panic!("the fixture's last layer has no MoE FFN");
        };
        xor_quants(&last.expert_down_q, GgmlType::Q6_K);
        let flipped = prefill(model, Path::TwoForwards, 0, &[], 40, 5, false, 0);
        xor_quants(&last.expert_down_q, GgmlType::Q6_K);
        assert!(
            flipped.logits != reference.logits,
            "the MoE expert weights do not reach the logits"
        );
        let restored = prefill(model, Path::TwoForwards, 0, &[], 40, 5, false, 0);
        assert!(restored.differences(&reference).is_empty());
        const MOE_SHAPES: [(&[usize], usize, usize); 5] = [
            (&[], 40, 5),
            (&[], 64, 5),
            (&[33], 20, 5),
            (&[17], 97, 8),
            (&[50], 157, 5),
        ];
        for (history, prefix, cue) in MOE_SHAPES {
            for capture in [true, false] {
                let two = prefill(
                    model,
                    Path::TwoForwards,
                    0,
                    history,
                    prefix,
                    cue,
                    capture,
                    0,
                );
                let folded = prefill(model, Path::Folded, 0, history, prefix, cue, capture, 0);
                assert!(
                    two.differences(&folded).is_empty(),
                    "MoE layers, history {history:?}, {prefix} + {cue} rows, capture {capture}: \
                     {:?} differ",
                    two.differences(&folded)
                );
            }
        }
    });
}
