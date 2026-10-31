//! ADR-063 turn 0: slot-mounted cold prefill without the SerialFifo dense
//! handoff and LCP snapshots. Real-artifact bitwise gate.
//!
//! Ignored. `HF2Q_GEMMA4_GGUF` names the official Gemma 4 26B-A4B GGUF, and
//! `HF2Q_KV_LCP_RESUME_CAPACITY=8g` (the four-agent launcher's value) lets
//! the legacy arm actually build its ~5.4 GB snapshots. Run alone with
//! `--exact --ignored --nocapture --test-threads=1`.
//!
//! Each arm loads a fresh model and runs the same schedule of cold
//! admissions at the production `max_tokens` (8192). Conversations reuse
//! slots that held other conversations, so stale rows in the persistent
//! scaffold are exercised. Per admission, the lean arm (production) must
//! equal the legacy arm (`FORCE_LEGACY_DENSE_HANDOFF`, the pre-change
//! route) in first token, final logits bits, every layer's K / TQ-HB V
//! codes / V scales bytes over the valid rows, the whole per-slot ring
//! regions the stable-boundary checkpoint copies, the checkpoint's size,
//! and the slot cursors. Within the lean arm, every admission of the same
//! conversation must be identical whichever slot and predecessor it got,
//! and no LCP snapshot may be left on the model.
//!
//! `gemma_cold_multi_seq_long_lane_ring_deterministic` (same environment):
//! a cold multi-sequence cohort with lanes longer than the 1,024-row ring.
//! Ten repeats must agree bit for bit (first tokens, logits, KV rows,
//! whole rings, cursors), a non-cohort slot must be untouched, and each
//! lane's layer-0 ring K / V codes (row-wise functions of its own tokens)
//! must equal a per-slot cold prefill of the same prompt, which pins that
//! the ring holds the last 1,024 positions.
use super::gemma_warm_batch_tests::{
    assert_lane_equal, lane_image, load_options, slot_cursors, slot_kv_hashes, synthetic_prompt,
    LaneImage,
};
use super::*;
use crate::inference::models::gemma4::kv_cache::snapshot_gemma_hybrid_slot_anchor;
use crate::serve::forward_prefill_batched::FORCE_LEGACY_DENSE_HANDOFF;
use std::sync::atomic::Ordering;

/// Prompt lengths per conversation: the four-agent stable boundary, one
/// that wraps the 1,024-row sliding ring, and one in between.
const CONVERSATIONS: [usize; 3] = [458, 1_300, 700];
/// `(slot, conversation)` in admission order.
const SCHEDULE: [(usize, usize); 7] = [(0, 0), (1, 1), (2, 2), (3, 0), (0, 1), (1, 0), (2, 0)];
const MAX_TOKENS: usize = 8_192;

#[derive(Debug, PartialEq, Eq)]
struct Admission {
    lane: LaneImage,
    /// Every ring row of every sliding layer (what the checkpoint copies),
    /// plus the valid global rows.
    ring: Vec<[u8; 32]>,
    /// Checkpoint `(prompt_len, total_bytes)`.
    anchor: (usize, usize),
}

struct Arm {
    admissions: Vec<Admission>,
    snapshot_left_on_model: Vec<bool>,
    prefill_ms: Vec<f64>,
}

fn run_arm(legacy: bool) -> Arm {
    FORCE_LEGACY_DENSE_HANDOFF.store(legacy, Ordering::Relaxed);
    let mut loaded = LoadedModel::load(&load_options()).expect("load official Gemma GGUF");
    let LoadedModel::Gemma(g) = &mut loaded else {
        panic!("HF2Q_GEMMA4_GGUF is not a Gemma 4 artifact")
    };
    g.provision_multi_seq_kv_for_slot_aware(4).unwrap();
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    assert_eq!(guard.model.weights.kv_graft_len(), 0, "no graft bound");
    let vocab = guard.model.weights.vocab_size;
    let prompts: Vec<Vec<u32>> = CONVERSATIONS
        .iter()
        .enumerate()
        .map(|(conversation, &len)| synthetic_prompt(conversation, len, vocab))
        .collect();
    let mut arm = Arm {
        admissions: Vec::new(),
        snapshot_left_on_model: Vec::new(),
        prefill_ms: Vec::new(),
    };
    for &(slot, conversation) in &SCHEDULE {
        let prompt = &prompts[conversation];
        let slot_id = SlotId(slot as u32);
        // `prefill_seed`'s cold entry: reset the slot's cursors, not bytes.
        for buf in guard.kv.iter_mut() {
            buf.reset_for_slot(slot_id).unwrap();
        }
        for buf in guard.hybrid.as_mut().unwrap().iter_mut() {
            buf.reset_for_slot(slot_id).unwrap();
        }
        let started = Instant::now();
        let first = guard
            .model
            .weights
            .forward_prefill_with_soft_tokens_slot_aware(
                prompt,
                &[],
                MAX_TOKENS,
                &mut guard.model.ctx,
                slot_id,
                &mut guard.kv,
                guard.hybrid.as_mut(),
                guard.dense.as_mut(),
                guard.mlx.as_mut(),
            )
            .unwrap();
        arm.prefill_ms
            .push(started.elapsed().as_secs_f64() * 1_000.0);
        clear_gemma4_self_mounts(guard.model);
        arm.snapshot_left_on_model.push(
            guard.model.weights.dense_kvs_snapshot_for_lcp.is_some()
                || guard.model.weights.hybrid_kv_snapshot_for_lcp.is_some(),
        );
        let logits = guard.model.weights.logits_view().unwrap().to_vec();
        assert!(logits.iter().all(|value| value.is_finite()));
        let anchor = snapshot_gemma_hybrid_slot_anchor(
            guard.hybrid.as_ref().unwrap(),
            slot_id,
            prompt.len(),
        )
        .unwrap();
        install_gemma4_slot_cursor(&mut guard, slot_id, prompt.len()).unwrap();
        arm.admissions.push(Admission {
            lane: lane_image(&guard, slot, prompt.len(), first, &logits),
            ring: slot_kv_hashes(&guard, slot, prompt.len(), prompt.len(), true),
            anchor: (anchor.prompt_len(), anchor.total_bytes()),
        });
    }
    FORCE_LEGACY_DENSE_HANDOFF.store(false, Ordering::Relaxed);
    drop(guard);
    drop(loaded);
    arm
}

#[test]
#[ignore = "official Gemma 4 GGUF: cold slot prefill without the dense handoff equals the legacy route bit for bit"]
fn gemma_cold_prefill_handoff_skip_matches_legacy_bitwise() {
    // This test thread plays the SlotAware worker that owns the scaffold and
    // the CPU checkpoint quota.
    crate::debug::investigation_env::gemma_full_input::set_worker_active(true);
    let legacy = run_arm(true);
    let lean = run_arm(false);

    for (index, &(slot, conversation)) in SCHEDULE.iter().enumerate() {
        let label = format!(
            "admission {index} slot {slot} conversation {conversation} ({} tokens)",
            CONVERSATIONS[conversation]
        );
        assert_lane_equal(
            &label,
            &lean.admissions[index].lane,
            &legacy.admissions[index].lane,
        );
        assert!(
            lean.admissions[index].ring == legacy.admissions[index].ring,
            "{label}: ring/checkpoint bytes differ"
        );
        assert_eq!(
            lean.admissions[index].anchor, legacy.admissions[index].anchor,
            "{label}: checkpoint size"
        );
        assert!(
            !lean.snapshot_left_on_model[index],
            "{label}: production left an LCP snapshot on the model"
        );
        if CONVERSATIONS[conversation] <= 1_024 {
            assert!(
                legacy.snapshot_left_on_model[index],
                "{label}: the legacy arm built no snapshot; set HF2Q_KV_LCP_RESUME_CAPACITY=8g"
            );
        }
    }
    // No stale-data leak: a conversation's image is independent of the slot
    // and of whatever conversation that slot held before.
    for conversation in 0..CONVERSATIONS.len() {
        let mut same = SCHEDULE
            .iter()
            .enumerate()
            .filter(|(_, &(_, c))| c == conversation)
            .map(|(index, _)| index);
        let first = same.next().unwrap();
        for index in same {
            assert_lane_equal(
                &format!("conversation {conversation}: admission {index} vs {first}"),
                &lean.admissions[index].lane,
                &lean.admissions[first].lane,
            );
        }
    }
    println!(
        "GEMMA_COLD_PREFILL_HANDOFF_TIMING legacy_ms={:?} lean_ms={:?}",
        legacy.prefill_ms, lean.prefill_ms
    );
    println!(
        "GEMMA_COLD_PREFILL_HANDOFF_BITWISE_COMPLETE admissions={} logits_bits=true kv_bytes=true rings=true anchors=true cursors=true cross_slot=true",
        SCHEDULE.len()
    );
}

/// `(slot, prompt length)` of the cohort: two lanes wrap the ring; 3,858 rows
/// in total, inside the 4,096-row cold transaction limit.
const RING_LANES: [(usize, usize); 3] = [(0, 2_100), (1, 1_300), (2, 458)];
const ISOLATED_SLOT: usize = 3;
const RING_REPEATS: usize = 10;

fn reset_slot(guard: &mut Gemma4KvGuard<'_>, slot: usize) {
    let slot_id = SlotId(slot as u32);
    for buf in guard.kv.iter_mut() {
        buf.reset_for_slot(slot_id).unwrap();
    }
    for buf in guard.hybrid.as_mut().unwrap().iter_mut() {
        buf.reset_for_slot(slot_id).unwrap();
    }
}

/// Per-slot cold prefill of `prompt`, as the cold gate above runs it.
fn per_slot_cold(guard: &mut Gemma4KvGuard<'_>, slot: usize, prompt: &[u32]) -> u32 {
    reset_slot(guard, slot);
    let first = guard
        .model
        .weights
        .forward_prefill_with_soft_tokens_slot_aware(
            prompt,
            &[],
            MAX_TOKENS,
            &mut guard.model.ctx,
            SlotId(slot as u32),
            &mut guard.kv,
            guard.hybrid.as_mut(),
            guard.dense.as_mut(),
            guard.mlx.as_mut(),
        )
        .unwrap();
    clear_gemma4_self_mounts(guard.model);
    install_gemma4_slot_cursor(guard, SlotId(slot as u32), prompt.len()).unwrap();
    first
}

#[test]
#[ignore = "official Gemma 4 GGUF: cold multi-sequence lanes longer than the ring are deterministic and keep the last window"]
fn gemma_cold_multi_seq_long_lane_ring_deterministic() {
    crate::debug::investigation_env::gemma_full_input::set_worker_active(true);
    let mut loaded = LoadedModel::load(&load_options()).expect("load official Gemma GGUF");
    let LoadedModel::Gemma(g) = &mut loaded else {
        panic!("HF2Q_GEMMA4_GGUF is not a Gemma 4 artifact")
    };
    g.provision_multi_seq_kv_for_slot_aware(4).unwrap();
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    assert_eq!(guard.model.weights.kv_graft_len(), 0, "no graft bound");
    let vocab = guard.model.weights.vocab_size;
    let prompts: Vec<Vec<u32>> = RING_LANES
        .iter()
        .map(|&(slot, len)| synthetic_prompt(slot, len, vocab))
        .collect();
    let longest = RING_LANES.iter().map(|&(_, len)| len).max().unwrap();
    assert!(gemma4_batch_rows_within_transaction_limit(
        RING_LANES.iter().map(|&(_, len)| len)
    ));

    // A neighbour the cohort must not touch.
    let isolated_prompt = synthetic_prompt(ISOLATED_SLOT, 700, vocab);
    per_slot_cold(&mut guard, ISOLATED_SLOT, &isolated_prompt);
    let isolation = |guard: &Gemma4KvGuard<'_>| {
        (
            slot_kv_hashes(guard, ISOLATED_SLOT, longest, longest, true),
            slot_cursors(guard, ISOLATED_SLOT),
        )
    };
    let isolated_before = isolation(&guard);

    let seqs: Vec<(Vec<u32>, SlotId)> = RING_LANES
        .iter()
        .zip(&prompts)
        .map(|(&(slot, _), prompt)| (prompt.clone(), SlotId(slot as u32)))
        .collect();
    // Per lane: (image, whole-ring hashes, live-row hashes).
    type LaneRun = (LaneImage, Vec<[u8; 32]>, Vec<[u8; 32]>);
    let mut repeats: Vec<Vec<LaneRun>> = Vec::with_capacity(RING_REPEATS);
    for _ in 0..RING_REPEATS {
        for &(slot, _) in &RING_LANES {
            reset_slot(&mut guard, slot);
        }
        let output = guard
            .model
            .weights
            .forward_prefill_batched_multi_seq(
                &seqs,
                guard.hybrid.as_deref().unwrap(),
                MAX_TOKENS,
                &mut guard.model.ctx,
            )
            .expect("cold multi-sequence prefill");
        clear_gemma4_self_mounts(guard.model);
        let mut lanes = Vec::with_capacity(RING_LANES.len());
        for (lane, &(slot, len)) in RING_LANES.iter().enumerate() {
            install_gemma4_slot_cursor(&mut guard, SlotId(slot as u32), len).unwrap();
            assert!(output.logits[lane].iter().all(|value| value.is_finite()));
            lanes.push((
                lane_image(
                    &guard,
                    slot,
                    len,
                    output.first_tokens[lane],
                    &output.logits[lane],
                ),
                slot_kv_hashes(&guard, slot, len, len, true),
                slot_kv_hashes(&guard, slot, len, len, false),
            ));
        }
        repeats.push(lanes);
    }
    for (repeat, lanes) in repeats.iter().enumerate().skip(1) {
        for (lane, &(slot, len)) in RING_LANES.iter().enumerate() {
            let label = format!("repeat {repeat} lane {lane} (slot {slot}, {len} rows)");
            assert_lane_equal(&label, &lanes[lane].0, &repeats[0][lane].0);
            assert!(
                lanes[lane].1 == repeats[0][lane].1,
                "{label}: whole-ring bytes differ from repeat 0"
            );
        }
    }
    assert!(
        isolation(&guard) == isolated_before,
        "the cohort changed non-cohort slot {ISOLATED_SLOT}"
    );

    // Layer 0 is a sliding layer whose K and V rows depend only on each
    // row's own token and position, so the live ring rows must equal a
    // per-slot cold prefill's: the ring holds positions [len - 1024, len).
    for (lane, (&(slot, len), prompt)) in RING_LANES.iter().zip(&prompts).enumerate() {
        let multi = &repeats[0][lane].2;
        per_slot_cold(&mut guard, slot, prompt);
        let single = slot_kv_hashes(&guard, slot, len, len, false);
        assert!(
            guard.hybrid.as_ref().unwrap()[0].is_sliding,
            "layer 0 is a sliding layer"
        );
        assert_eq!(
            [multi[0], multi[1]],
            [single[0], single[1]],
            "lane {lane} (slot {slot}, {len} rows): layer-0 ring K / V codes differ from per-slot cold prefill"
        );
    }
    println!(
        "GEMMA_COLD_MULTI_SEQ_RING_COMPLETE lanes={:?} repeats={RING_REPEATS} logits_bits=true kv_bytes=true rings=true cursors=true isolation=true layer0_vs_per_slot=true",
        RING_LANES
    );
    drop(guard);
    drop(loaded);
}
