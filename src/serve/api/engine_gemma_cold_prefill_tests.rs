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
use super::gemma_warm_batch_tests::{
    assert_lane_equal, lane_image, load_options, slot_kv_hashes, synthetic_prompt, LaneImage,
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
