//! Metal gates for the joint prompt prefill (`HF2Q_QWEN_PREFILL_JOINT`,
//! [`super::prefill_joint`]).
//!
//! A joint prefill is a Class A change: one forward over the pending chunks
//! of several slots must leave every slot exactly what that slot's own
//! prefill leaves. These gates compare, bit for bit and per lane:
//! - the last-row logits, and for a lane that folds its generation cue the
//!   stable-boundary checkpoint (cursors, DeltaNet image) and its logits;
//! - the captured target-layer rows;
//! - the slot afterwards: its cursors and DeltaNet state, the valid rows of
//!   every full-attention layer's K/V codes and norms, and every logits row
//!   of a follow-up chunk.
//!
//! The reference of a lane is its solo prefill on a cache whose other slots
//! are empty: an ordinary forward, or the folded forward of
//! [`super::prefill_fold`] (itself gated against the two forwards it
//! replaces). A lane that shares a prefix is compared with the same
//! unshared solo prefill, which proves that sharing changes no bit.
//!
//! They run the synthetic native K-quant model of the prepared-prefill gates
//! with one to four lanes of different row counts (a five-row and a 720-row
//! lane together), fresh and resumed slots, and folded cues. The ignored
//! real-artifact gate at the end repeats the comparison on a served model
//! (`HF2Q_QWEN38_GGUF` or `HF2Q_QWEN36_GGUF`, whose MoE FFN the synthetic
//! model does not have) and prints the walls.

use super::forward_gpu::JointPrefillLane;
use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use super::prefill_prepared::PrefillScope;
use super::prefill_prepared_tests::with_synthetic_model;
use super::spec_decode::positions_for_range;
use crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession;
use crate::serve::multi_seq_kv::SlotId;

const SLOTS: u32 = 4;
const CAPACITY: u32 = 1024;
/// Target layers of the synthetic model the gates capture: DeltaNet layers
/// on and off a full-attention neighbour, and the last full-attention layer.
const CAPTURE_LAYERS: [usize; 4] = [1, 4, 6, 7];
/// Rows of the follow-up chunk that reads the slot back.
const FOLLOW_UP_ROWS: usize = 7;
/// Token seed of a prompt prefix several lanes have in common.
const COMMON_SEED: u32 = 97;

type AnchorWords = (usize, u64, u32, Vec<u32>, Option<u32>, Vec<u32>);

/// One slot's pending chunk.
#[derive(Clone, Copy, Debug)]
struct Lane {
    slot: u32,
    /// Token seed of the lane's own prompt.
    seed: u32,
    /// Leading prompt tokens that come from the common prefix.
    common: usize,
    /// Chunks prefilled before the pending one.
    history: &'static [usize],
    /// Content rows of the pending chunk.
    content: usize,
    /// Generation-cue rows folded behind them (0 = no fold).
    cue: usize,
}

impl Lane {
    const fn new(slot: u32, history: &'static [usize], content: usize, cue: usize) -> Self {
        Self {
            slot,
            seed: slot + 1,
            common: 0,
            history,
            content,
            cue,
        }
    }

    /// A cold lane whose prompt starts with `common` tokens of the common
    /// prefix.
    const fn cold(slot: u32, common: usize, content: usize, cue: usize) -> Self {
        Self {
            slot,
            seed: slot + 1,
            common,
            history: &[],
            content,
            cue,
        }
    }

    fn start(&self) -> usize {
        self.history.iter().sum()
    }

    fn rows(&self) -> usize {
        self.content + self.cue
    }

    fn tokens(&self, start: usize, len: usize) -> Vec<u32> {
        (start..start + len)
            .map(|position| {
                let seed = if position < self.common {
                    COMMON_SEED
                } else {
                    self.seed
                };
                (seed * 131 + position as u32 * 37 + 11) % 512
            })
            .collect()
    }

    fn fold(&self) -> Option<(usize, usize)> {
        (self.cue > 0).then(|| (self.content, self.start() + self.content))
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn cache(model: &Qwen35Model) -> HybridKvCache {
    model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&model.cfg, device, CAPACITY, SLOTS, true)
        })
        .expect("allocate TQ cache")
}

/// The valid rows of every full-attention layer's K/V codes and norms of
/// `slot`: what the slot's attention reads.
fn kv_rows(model: &Qwen35Model, cache: &HybridKvCache, slot: u32) -> Vec<u8> {
    let rows = cache
        .sequence_len_for_slot(SlotId(slot))
        .expect("slot cursor") as usize;
    let heads = model.cfg.num_key_value_heads as usize;
    let mut bytes = Vec::new();
    for layer in &cache.full_attn {
        let tq = layer.tq.as_ref().expect("TQ full-attention slot");
        for buffer in [&tq.k_packed, &tq.k_norms, &tq.v_packed, &tq.v_norms] {
            let total = buffer.byte_len();
            let per_head = total / (SLOTS as usize * heads);
            let per_row = per_head / CAPACITY as usize;
            assert_eq!(
                per_row * CAPACITY as usize * heads * SLOTS as usize,
                total,
                "TQ buffer layout"
            );
            // SAFETY: the buffer's contents are `total` bytes; the GPU is idle.
            let all =
                unsafe { std::slice::from_raw_parts(buffer.contents_ptr().cast::<u8>(), total) };
            for head in 0..heads {
                let at = (slot as usize * heads + head) * per_head;
                bytes.extend_from_slice(&all[at..at + rows * per_row]);
            }
        }
    }
    bytes
}

/// Everything a lane's pending chunk leaves behind.
#[derive(PartialEq)]
struct Outcome {
    /// Logits of the last content row and the checkpoint there (folded cue).
    boundary: Option<(Vec<u32>, AnchorWords)>,
    logits: Vec<u32>,
    /// Captured rows per target layer, in the lane's row order.
    captured: Vec<Vec<u32>>,
    /// The slot's cursors and DeltaNet state after the chunk.
    after: AnchorWords,
    /// The slot's valid full-attention rows after the chunk.
    kv: Vec<u8>,
    /// Every logits row of a chunk prefilled after it.
    follow_up: Vec<u32>,
}

impl Outcome {
    fn differences(&self, other: &Self) -> Vec<&'static str> {
        [
            (
                "boundary checkpoint and logits",
                self.boundary == other.boundary,
            ),
            ("last-row logits", self.logits == other.logits),
            ("captured target layers", self.captured == other.captured),
            ("slot state after the chunk", self.after == other.after),
            ("full-attention rows", self.kv == other.kv),
            ("follow-up chunk", self.follow_up == other.follow_up),
        ]
        .into_iter()
        .filter_map(|(name, equal)| (!equal).then_some(name))
        .collect()
    }
}

/// Prefill the lane's history chunks, one ordinary forward each.
fn prefill_history(model: &Qwen35Model, cache: &mut HybridKvCache, lane: &Lane) {
    let mut start = 0;
    for &rows in lane.history {
        model
            .forward_gpu_last_logits(
                &lane.tokens(start, rows),
                &positions_for_range(start as i32, rows),
                cache,
                SlotId(lane.slot),
            )
            .unwrap_or_else(|error| panic!("history chunk start={start} n={rows}: {error:#}"));
        start += rows;
    }
}

/// The slot after its pending chunk: state, full-attention rows and a
/// follow-up chunk's logits.
fn read_back(
    model: &Qwen35Model,
    cache: &mut HybridKvCache,
    lane: &Lane,
) -> (AnchorWords, Vec<u8>, Vec<u32>) {
    let end = lane.start() + lane.rows();
    let after = cache
        .snapshot_slot_anchor(SlotId(lane.slot), end)
        .expect("slot state after the chunk")
        .test_words();
    let kv = kv_rows(model, cache, lane.slot);
    let follow_up = model
        .forward_gpu(
            &lane.tokens(end, FOLLOW_UP_ROWS),
            &positions_for_range(end as i32, FOLLOW_UP_ROWS),
            cache,
            SlotId(lane.slot),
        )
        .expect("follow-up chunk");
    (after, kv, bits(&follow_up))
}

/// The lane's solo prefill on a cache whose other slots are empty: the
/// forward(s) the server runs for it alone.
fn solo(model: &Qwen35Model, lane: &Lane, capture: bool) -> Outcome {
    let _scope = PrefillScope::forced(true);
    let hidden = model.cfg.hidden_size as usize;
    let mut cache = cache(model);
    prefill_history(model, &mut cache, lane);
    let (start, rows) = (lane.start(), lane.rows());
    let tokens = lane.tokens(start, rows);
    let positions = positions_for_range(start as i32, rows);
    let mut session =
        capture.then(|| DFlashCaptureSession::new(CAPTURE_LAYERS.to_vec(), rows, hidden, false));
    let (boundary, logits) = match lane.fold() {
        Some((prefix_rows, boundary_len)) => {
            let image = cache.boundary_anchor_image().expect("checkpoint storage");
            let folded = model
                .forward_gpu_last_logits_folded(
                    &tokens,
                    &positions,
                    &mut cache,
                    session.as_mut(),
                    SlotId(lane.slot),
                    prefix_rows,
                    boundary_len,
                    image,
                )
                .unwrap_or_else(|error| panic!("solo folded chunk of {lane:?}: {error:#}"));
            (
                Some((
                    bits(&folded.boundary_logits),
                    folded.boundary.expect("solo checkpoint").test_words(),
                )),
                bits(&folded.logits),
            )
        }
        None => {
            let logits = match session.as_mut() {
                Some(session) => model
                    .forward_gpu_last_logits_with_hidden_dflash(
                        &tokens,
                        &positions,
                        &mut cache,
                        Some(session),
                        SlotId(lane.slot),
                    )
                    .map(|(logits, _hidden)| logits),
                None => model.forward_gpu_last_logits(
                    &tokens,
                    &positions,
                    &mut cache,
                    SlotId(lane.slot),
                ),
            }
            .unwrap_or_else(|error| panic!("solo chunk of {lane:?}: {error:#}"));
            (None, bits(&logits))
        }
    };
    let captured = session.map_or_else(Vec::new, |session| {
        session
            .hidden_output
            .chunks(rows * hidden)
            .map(bits)
            .collect()
    });
    let (after, kv, follow_up) = read_back(model, &mut cache, lane);
    Outcome {
        boundary,
        logits,
        captured,
        after,
        kv,
        follow_up,
    }
}

/// One joint forward over the pending chunks of `lanes`, the first
/// `shared_rows` rows of the lanes with a common prefix shared, into
/// `cache`. Returns per lane the boundary, the logits and the captured rows.
fn joint_forward(
    model: &Qwen35Model,
    cache: &mut HybridKvCache,
    lanes: &[Lane],
    shared_rows: usize,
    capture: bool,
) -> anyhow::Result<Vec<(Option<(Vec<u32>, AnchorWords)>, Vec<u32>, Vec<Vec<u32>>)>> {
    let _scope = PrefillScope::forced(true);
    let hidden = model.cfg.hidden_size as usize;
    let chunks: Vec<(Vec<u32>, Vec<i32>)> = lanes
        .iter()
        .map(|lane| {
            (
                lane.tokens(lane.start(), lane.rows()),
                positions_for_range(lane.start() as i32, lane.rows()),
            )
        })
        .collect();
    let joint_lanes: Vec<JointPrefillLane<'_>> = lanes
        .iter()
        .zip(&chunks)
        .map(|(lane, (tokens, positions))| JointPrefillLane {
            slot_id: SlotId(lane.slot),
            tokens,
            positions_flat: positions,
            shares_prefix: shared_rows > 0 && lane.common >= shared_rows,
            fold: lane.fold(),
            image: None,
        })
        .collect();
    let total = Qwen35Model::joint_prefill_rows(&joint_lanes, shared_rows);
    let mut session =
        capture.then(|| DFlashCaptureSession::new(CAPTURE_LAYERS.to_vec(), total, hidden, false));
    let outputs =
        model.forward_gpu_last_logits_joint(&joint_lanes, shared_rows, cache, session.as_mut())?;
    Ok(lanes
        .iter()
        .zip(outputs)
        .map(|(lane, output)| {
            let captured = session.as_ref().map_or_else(Vec::new, |session| {
                (0..CAPTURE_LAYERS.len())
                    .map(|layer| {
                        output
                            .forward_rows(0..lane.rows())
                            .into_iter()
                            .flat_map(|rows| {
                                let at = layer * total + rows.start;
                                bits(
                                    &session.hidden_output[at * hidden..(at + rows.len()) * hidden],
                                )
                            })
                            .collect()
                    })
                    .collect()
            });
            (
                output
                    .boundary
                    .map(|(logits, anchor, _wall)| {
                        (bits(&logits), anchor.expect("lane checkpoint").test_words())
                    }),
                bits(&output.logits),
                captured,
            )
        })
        .collect())
}

/// The lanes' pending chunks as one joint forward on one cache, each lane's
/// history prefilled before it.
fn joint(model: &Qwen35Model, lanes: &[Lane], shared_rows: usize, capture: bool) -> Vec<Outcome> {
    let mut cache = cache(model);
    for lane in lanes {
        let _scope = PrefillScope::forced(true);
        prefill_history(model, &mut cache, lane);
    }
    let forward = joint_forward(model, &mut cache, lanes, shared_rows, capture)
        .unwrap_or_else(|error| panic!("joint forward of {lanes:?}: {error:#}"));
    let _scope = PrefillScope::forced(true);
    lanes
        .iter()
        .zip(forward)
        .map(|(lane, (boundary, logits, captured))| {
            let (after, kv, follow_up) = read_back(model, &mut cache, lane);
            Outcome {
                boundary,
                logits,
                captured,
                after,
                kv,
                follow_up,
            }
        })
        .collect()
}

/// Assert that every lane of the joint forward holds its solo prefill's bits.
fn assert_lanes_match_solo(model: &Qwen35Model, lanes: &[Lane], shared_rows: usize, capture: bool) {
    let together = joint(model, lanes, shared_rows, capture);
    assert_eq!(together.len(), lanes.len());
    for (lane, together) in lanes.iter().zip(&together) {
        let alone = solo(model, lane, capture);
        let differences = alone.differences(together);
        assert!(
            differences.is_empty(),
            "{lane:?} in a joint prefill of {} lanes sharing {shared_rows} rows (capture \
             {capture}) differs from its solo prefill in {differences:?}",
            lanes.len()
        );
        assert_eq!(
            alone.captured.len(),
            if capture { CAPTURE_LAYERS.len() } else { 0 }
        );
        assert_eq!(alone.boundary.is_some(), lane.cue > 0);
    }
}

#[test]
fn a_joint_prefill_leaves_each_lane_what_its_own_prefill_leaves() {
    with_synthetic_model(|model| {
        let cohorts: [&[Lane]; 5] = [
            // One lane: a joint forward is that lane's own forward.
            &[Lane::new(2, &[], 40, 5)],
            // The largest and the smallest served chunk together: a fresh
            // 720-row prompt with its folded cue, a resumed five-row chunk.
            &[Lane::new(0, &[], 720, 5), Lane::new(3, &[33], 5, 0)],
            // Resumed and fresh lanes, with and without a fold, in an order
            // that is not the slot order.
            &[
                Lane::new(1, &[64, 20], 40, 5),
                Lane::new(0, &[], 31, 0),
                Lane::new(2, &[17], 97, 8),
            ],
            // Four lanes: a tile-aligned chunk, the short-chunk direct TQ
            // route, the dequantized-history route, the longest cue.
            &[
                Lane::new(0, &[], 720, 5),
                Lane::new(1, &[33], 5, 0),
                Lane::new(2, &[40], 157, 0),
                Lane::new(3, &[], 33, 32),
            ],
            // Two-row segments everywhere.
            &[Lane::new(3, &[], 2, 2), Lane::new(0, &[5], 2, 0)],
        ];
        for lanes in cohorts {
            assert_lanes_match_solo(model, lanes, 0, true);
        }
        // The plain target route (no DFlash capture).
        assert_lanes_match_solo(model, cohorts[2], 0, false);
    });
}

#[test]
fn lanes_that_share_a_prefix_hold_the_bits_of_their_unshared_prefill() {
    with_synthetic_model(|model| {
        // The four-agent first turn: four cold prompts, 444 of 725 tokens in
        // common, each folding its cue.
        let agents: Vec<Lane> = (0..4).map(|slot| Lane::cold(slot, 444, 720, 5)).collect();
        assert_lanes_match_solo(model, &agents, 444, true);
        // Prompts of different lengths, with and without a fold; a resumed
        // lane and a cold lane with other tokens share nothing; the prefix
        // ends inside a 32-row tile and two rows before the shortest
        // sharing lane's content ends.
        let mixed = [
            Lane::cold(2, 70, 100, 5),
            Lane::new(1, &[33], 20, 5),
            Lane::cold(0, 70, 72, 0),
            Lane::new(3, &[], 50, 5),
        ];
        assert_lanes_match_solo(model, &mixed, 70, true);
        assert_lanes_match_solo(model, &mixed, 70, false);
        // The same lanes without sharing: sharing is a choice of the caller
        // that changes no bit.
        assert_lanes_match_solo(model, &mixed, 0, true);
        // A shorter shared prefix than the lanes have in common.
        assert_lanes_match_solo(model, &mixed, 33, true);
    });
}

#[test]
fn a_joint_prefill_is_repeatable() {
    with_synthetic_model(|model| {
        let lanes = [
            Lane::cold(1, 64, 130, 5),
            Lane::new(0, &[40], 37, 0),
            Lane::cold(3, 64, 97, 8),
        ];
        let first = joint(model, &lanes, 64, true);
        // Other rows through the same pooled scratch and arenas.
        joint(
            model,
            &[Lane::new(2, &[], 65, 5), Lane::new(0, &[], 9, 0)],
            0,
            false,
        );
        let again = joint(model, &lanes, 64, true);
        for ((lane, first), again) in lanes.iter().zip(&first).zip(&again) {
            let differences = first.differences(again);
            assert!(
                differences.is_empty(),
                "{lane:?}: a repeated joint prefill changed {differences:?}"
            );
        }
    });
}

/// After a shared prefill nothing of one lane is reachable from another:
/// resetting a lane, prefilling another conversation into its slot and
/// resetting it again leaves every byte the other lanes hold, and the lanes
/// still read back as their solo prefill does.
#[test]
fn a_lane_reset_after_a_shared_prefill_leaves_the_other_lanes_untouched() {
    with_synthetic_model(|model| {
        let lanes = [
            Lane::cold(0, 96, 150, 5),
            Lane::cold(1, 96, 120, 5),
            Lane::cold(2, 96, 110, 0),
        ];
        let mut cache = cache(model);
        let forward =
            joint_forward(model, &mut cache, &lanes, 96, true).expect("shared joint forward");
        let _scope = PrefillScope::forced(true);
        let held = |cache: &mut HybridKvCache, lane: &Lane| {
            (
                cache
                    .snapshot_slot_anchor(SlotId(lane.slot), lane.rows())
                    .expect("lane state")
                    .test_words(),
                kv_rows(model, cache, lane.slot),
            )
        };
        let before: Vec<_> = lanes[1..]
            .iter()
            .map(|lane| held(&mut cache, lane))
            .collect();

        // Lane 0's conversation ends; another one takes its slot, runs past
        // the shared prefix's length, and is evicted too.
        let evicted = lanes[0];
        cache.reset_for_slot(SlotId(evicted.slot)).expect("reset");
        let other = Lane {
            seed: 41,
            common: 0,
            ..Lane::new(evicted.slot, &[], 300, 5)
        };
        let other_solo = solo(model, &other, true);
        let other_here = joint_forward(model, &mut cache, &[other], 0, true)
            .expect("the other conversation in the reused slot")
            .remove(0);
        // The reset advanced the slot's state generation, which a checkpoint
        // records; everything else of the checkpoint is a fresh slot's.
        let same_generation = |boundary: Option<(Vec<u32>, AnchorWords)>| {
            boundary.map(|(logits, mut words)| {
                words.1 = 0;
                (logits, words)
            })
        };
        assert!(
            same_generation(other_solo.boundary) == same_generation(other_here.0)
                && other_solo.logits == other_here.1
                && other_solo.captured == other_here.2,
            "a slot that shared a prefix does not serve its next conversation as a fresh slot"
        );
        cache.reset_for_slot(SlotId(evicted.slot)).expect("reset");

        let after: Vec<_> = lanes[1..]
            .iter()
            .map(|lane| held(&mut cache, lane))
            .collect();
        assert!(
            before == after,
            "resetting and reusing one lane's slot changed another lane's bytes"
        );
        // The checkpoint a lane took inside the shared forward is its own.
        for ((lane, (boundary, logits, captured)), index) in
            lanes.iter().zip(forward).zip(0..).skip(1)
        {
            let alone = solo(model, lane, true);
            let (after, kv, follow_up) = read_back(model, &mut cache, lane);
            let here = Outcome {
                boundary,
                logits,
                captured,
                after,
                kv,
                follow_up,
            };
            let differences = alone.differences(&here);
            assert!(
                differences.is_empty(),
                "lane {index} after its neighbour's reset differs from its solo prefill in \
                 {differences:?}"
            );
        }
    });
}

#[test]
fn an_inadmissible_joint_prefill_is_refused_before_a_slot_changes() {
    with_synthetic_model(|model| {
        let mut cache = cache(model);
        let cursors = |cache: &HybridKvCache| -> Vec<u32> {
            (0..SLOTS)
                .map(|slot| cache.sequence_len_for_slot(SlotId(slot)).expect("cursor") as u32)
                .collect()
        };
        let refused: [(&[Lane], usize, &str); 5] = [
            (
                &[Lane::new(0, &[], 40, 5), Lane::new(0, &[], 20, 0)],
                0,
                "one slot twice",
            ),
            (
                &[Lane::new(0, &[], 40, 5), Lane::new(1, &[], 1, 0)],
                0,
                "a one-row lane",
            ),
            (
                &[Lane::new(0, &[], 40, 1), Lane::new(1, &[], 9, 0)],
                0,
                "a one-row cue",
            ),
            (
                &[Lane::cold(0, 40, 41, 5), Lane::cold(1, 40, 60, 0)],
                40,
                "a shared prefix that leaves a lane one content row",
            ),
            (
                &[
                    Lane::cold(0, 40, 60, 5),
                    Lane::cold(1, 40, 60, 0),
                    Lane::new(2, &[], 9, 0),
                ],
                0,
                "accepted",
            ),
        ];
        for (lanes, shared_rows, what) in refused {
            let result = joint_forward(model, &mut cache, lanes, shared_rows, true);
            if what == "accepted" {
                assert!(result.is_ok(), "the control cohort was refused");
                assert_eq!(cursors(&cache), vec![65, 60, 9, 0]);
            } else {
                assert!(result.is_err(), "{what} was accepted");
                assert_eq!(cursors(&cache), vec![0; 4], "{what} moved a slot");
            }
        }
        // Lanes marked as sharing whose tokens differ inside the prefix.
        let mut cache = self::cache(model);
        let _scope = PrefillScope::forced(true);
        let (first, second) = (Lane::cold(0, 40, 60, 0), Lane::cold(1, 39, 60, 0));
        let chunks: Vec<(Vec<u32>, Vec<i32>)> = [first, second]
            .iter()
            .map(|lane| (lane.tokens(0, 60), positions_for_range(0, 60)))
            .collect();
        let lanes: Vec<JointPrefillLane<'_>> = [first, second]
            .iter()
            .zip(&chunks)
            .map(|(lane, (tokens, positions))| JointPrefillLane {
                slot_id: SlotId(lane.slot),
                tokens,
                positions_flat: positions,
                shares_prefix: true,
                fold: None,
                image: None,
            })
            .collect();
        assert!(
            model
                .forward_gpu_last_logits_joint(&lanes, 40, &mut cache, None)
                .is_err(),
            "lanes that differ inside the shared prefix were accepted"
        );
        assert_eq!(cursors(&cache), vec![0; 4]);
    });
}

/// Real-artifact gate, run explicitly on one served model:
/// `HF2Q_QWEN38_GGUF=… <test binary> qwen_prefill_joint_real_model_gate --ignored --nocapture --test-threads=1`
/// (or `HF2Q_QWEN36_GGUF=…`; add `HF2Q_ENCODER_SESSION=1` for the served
/// encoder sessions and the served MoE switches for Qwen3.6).
///
/// On a four-slot TQ cache it prefills the first turn of the four-agent
/// workload (four cold prompts of 720 content rows and a five-row cue that
/// have their first 444 tokens in common) and a later turn (resumed chunks
/// of 31, 157, 200 and 266 content rows and their cues) three ways: each
/// lane alone, all lanes in one forward, and all lanes in one forward with
/// the common prefix shared. Every lane's bits must be those of its solo
/// prefill. It prints the walls (second pass of each, so weights, pipelines
/// and prepared planes are warm).
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF or HF2Q_QWEN36_GGUF and Apple Metal; run alone"]
fn qwen_prefill_joint_real_model_gate() {
    use std::time::Instant;

    use crate::serve::header::LoadProgress;
    use mlx_native::gguf::GgufFile;

    const PATTERN: [u32; 7] = [9707, 374, 279, 15, 1_782, 2_610, 13];
    const COMMON: usize = 444;
    const CONTENT: usize = 720;
    const CUE: usize = 5;
    const REPLY: usize = 150;
    const LATER: [usize; 4] = [31, 157, 200, 266];
    const GATE_CAPACITY: u32 = 4096;
    const GATE_SLOTS: u32 = 4;

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
    let vocab = model.cfg.vocab_size;
    // Lane `slot`'s prompt: the common prefix, then its own tokens.
    let lane_tokens = |slot: u32, start: usize, len: usize| -> Vec<u32> {
        (start..start + len)
            .map(|index| {
                let own = if index < COMMON { 0 } else { slot as usize + 1 };
                (PATTERN[(index + own) % PATTERN.len()] + (own * 17) as u32) % vocab
            })
            .collect()
    };
    let new_cache = || {
        model
            .with_gpu_cache_mut(|device, _registry| {
                HybridKvCache::new_with_options(&model.cfg, device, GATE_CAPACITY, GATE_SLOTS, true)
            })
            .expect("allocate TQ cache")
    };
    let _scope = PrefillScope::forced(true);
    assert!(
        model.prefill_fold_exact().expect("exactness"),
        "a joint prefill is not exact for this artifact on the prepared route"
    );
    let heads = model.cfg.num_key_value_heads as usize;
    let kv = |cache: &HybridKvCache, slot: u32| -> Vec<u8> {
        let rows = cache.sequence_len_for_slot(SlotId(slot)).expect("cursor") as usize;
        let mut bytes = Vec::new();
        for layer in &cache.full_attn {
            let tq = layer.tq.as_ref().expect("TQ full-attention slot");
            for buffer in [&tq.k_packed, &tq.k_norms, &tq.v_packed, &tq.v_norms] {
                let total = buffer.byte_len();
                let per_head = total / (GATE_SLOTS as usize * heads);
                let per_row = per_head / GATE_CAPACITY as usize;
                // SAFETY: the buffer's contents are `total` bytes; the GPU is idle.
                let all = unsafe {
                    std::slice::from_raw_parts(buffer.contents_ptr().cast::<u8>(), total)
                };
                for head in 0..heads {
                    let at = (slot as usize * heads + head) * per_head;
                    bytes.extend_from_slice(&all[at..at + rows * per_row]);
                }
            }
        }
        bytes
    };
    type LaneBits = (
        Vec<u32>,
        AnchorWords,
        Vec<u32>,
        Vec<Vec<u32>>,
        AnchorWords,
        Vec<u8>,
    );
    // One turn of all four lanes: `starts[slot]` rows are in the slot,
    // `contents[slot]` content rows and the cue are pending. `mode` 0: each
    // lane alone; 1: one joint forward; 2: one joint forward sharing
    // `shared` rows. Returns the wall and every lane's bits.
    let turn = |cache: &mut HybridKvCache,
                mode: u8,
                starts: [usize; 4],
                contents: [usize; 4],
                shared: usize|
     -> (f64, Vec<LaneBits>) {
        let chunks: Vec<(Vec<u32>, Vec<i32>)> = (0..4)
            .map(|slot| {
                let len = contents[slot] + CUE;
                (
                    lane_tokens(slot as u32, starts[slot], len),
                    positions_for_range(starts[slot] as i32, len),
                )
            })
            .collect();
        let read = |cache: &mut HybridKvCache, slot: usize| {
            (
                cache
                    .snapshot_slot_anchor(SlotId(slot as u32), starts[slot] + contents[slot] + CUE)
                    .expect("lane state")
                    .test_words(),
                kv(cache, slot as u32),
            )
        };
        let started = Instant::now();
        let mut lanes: Vec<LaneBits> = Vec::with_capacity(4);
        if mode == 0 {
            let mut forwards = Vec::with_capacity(4);
            for slot in 0..4 {
                let len = contents[slot] + CUE;
                let mut session =
                    DFlashCaptureSession::new(capture_layers.clone(), len, hidden, false);
                let image = cache.boundary_anchor_image().expect("checkpoint storage");
                let folded = model
                    .forward_gpu_last_logits_folded(
                        &chunks[slot].0,
                        &chunks[slot].1,
                        cache,
                        Some(&mut session),
                        SlotId(slot as u32),
                        contents[slot],
                        starts[slot] + contents[slot],
                        image,
                    )
                    .expect("solo folded chunk");
                forwards.push((folded, session));
            }
            let wall = started.elapsed().as_secs_f64() * 1e3;
            for (slot, (folded, session)) in forwards.into_iter().enumerate() {
                let len = contents[slot] + CUE;
                let (after, kv) = read(cache, slot);
                lanes.push((
                    bits(&folded.boundary_logits),
                    folded.boundary.expect("solo checkpoint").test_words(),
                    bits(&folded.logits),
                    session
                        .hidden_output
                        .chunks(len * hidden)
                        .map(bits)
                        .collect(),
                    after,
                    kv,
                ));
            }
            return (wall, lanes);
        }
        let shared = if mode == 2 { shared } else { 0 };
        let joint_lanes: Vec<JointPrefillLane<'_>> = (0..4)
            .map(|slot| JointPrefillLane {
                slot_id: SlotId(slot as u32),
                tokens: &chunks[slot].0,
                positions_flat: &chunks[slot].1,
                shares_prefix: shared > 0,
                fold: Some((contents[slot], starts[slot] + contents[slot])),
                image: None,
            })
            .collect();
        let total = Qwen35Model::joint_prefill_rows(&joint_lanes, shared);
        let mut session = DFlashCaptureSession::new(capture_layers.clone(), total, hidden, false);
        let outputs = model
            .forward_gpu_last_logits_joint(&joint_lanes, shared, cache, Some(&mut session))
            .expect("joint forward");
        let wall = started.elapsed().as_secs_f64() * 1e3;
        for (slot, output) in outputs.into_iter().enumerate() {
            let len = contents[slot] + CUE;
            let captured = (0..capture_layers.len())
                .map(|layer| {
                    output
                        .forward_rows(0..len)
                        .into_iter()
                        .flat_map(|rows| {
                            let at = layer * total + rows.start;
                            bits(&session.hidden_output[at * hidden..(at + rows.len()) * hidden])
                        })
                        .collect()
                })
                .collect();
            let (boundary_logits, boundary, _wall) = output.boundary.expect("lane checkpoint");
            let boundary = boundary.expect("lane checkpoint state");
            let (after, kv) = read(cache, slot);
            lanes.push((
                bits(&boundary_logits),
                boundary.test_words(),
                bits(&output.logits),
                captured,
                after,
                kv,
            ));
        }
        (wall, lanes)
    };
    // The reply rows between the turns, one ordinary forward per lane.
    let reply = |cache: &mut HybridKvCache, starts: [usize; 4]| {
        for slot in 0..4u32 {
            model
                .forward_gpu_last_logits(
                    &lane_tokens(slot, starts[slot as usize], REPLY),
                    &positions_for_range(starts[slot as usize] as i32, REPLY),
                    cache,
                    SlotId(slot),
                )
                .expect("reply rows");
        }
    };
    let names = ["solo", "joint", "joint+shared"];
    let mut reference: Option<Vec<Vec<LaneBits>>> = None;
    for pass in 0..2 {
        for mode in 0..3u8 {
            let mut cache = new_cache();
            let (first_ms, first) = turn(&mut cache, mode, [0; 4], [CONTENT; 4], COMMON);
            let after_first = [CONTENT + CUE; 4];
            reply(&mut cache, after_first);
            let later_starts = [CONTENT + CUE + REPLY; 4];
            let (later_ms, later) = turn(&mut cache, mode, later_starts, LATER, 0);
            let rows_first = if mode == 2 {
                COMMON + 4 * (CONTENT + CUE - COMMON)
            } else {
                4 * (CONTENT + CUE)
            };
            eprintln!(
                "[PREFILL_JOINT_GATE] pass={pass} mode={} first_turn_ms={first_ms:.1} \
                 first_turn_forward_rows={rows_first} later_turn_ms={later_ms:.1} \
                 later_turn_rows={}",
                names[mode as usize],
                LATER.iter().sum::<usize>() + 4 * CUE
            );
            let bits = vec![first, later];
            match reference.as_ref() {
                None => reference = Some(bits),
                Some(reference) => {
                    for (turn_index, (expected, actual)) in reference.iter().zip(&bits).enumerate()
                    {
                        for (slot, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                            let parts = [
                                ("boundary logits", expected.0 == actual.0),
                                ("boundary checkpoint", expected.1 == actual.1),
                                ("last-row logits", expected.2 == actual.2),
                                ("captured target layers", expected.3 == actual.3),
                                ("slot state", expected.4 == actual.4),
                                ("full-attention rows", expected.5 == actual.5),
                            ];
                            let differences: Vec<_> = parts
                                .iter()
                                .filter_map(|(name, equal)| (!equal).then_some(*name))
                                .collect();
                            assert!(
                                differences.is_empty(),
                                "pass {pass} mode {} turn {turn_index} slot {slot}: differs from \
                                 the solo prefill in {differences:?}",
                                names[mode as usize]
                            );
                        }
                    }
                }
            }
        }
    }
}
