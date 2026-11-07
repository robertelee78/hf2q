//! ADR-063 Q4-QWEN36 forced-close anchor (design option B), engine side.
//!
//! - G2: candidate selection. The forced-close anchor competes with the
//!   stable anchor and the live prefix by `starts_with`, longest wins; ties,
//!   a wrong vision fingerprint, a partial match and unrestorable cursors
//!   fall back to the stable anchor.
//! - G3: lifetime. Publication keeps only a coherent anchor, every admission
//!   takes it out of the slot (so a divergent request leaves no stale anchor
//!   behind), and slot spills never carry it. Every reset, cancellation and
//!   graft failure path drops the slot's prompt anchor, which owns it.

use mlx_native::MlxDevice;

use super::qwen35_slot_spill::{
    reclaim_qwen35_slot, spill_reclaimed_qwen35_slot, Qwen35SlotSpillStore,
};
use super::slot_spill::SlotSpillStore;
use super::{
    publish_qwen35_forced_close_anchor, qwen35_effective_anchor_tokens, qwen35_slot_affinity,
    take_qwen35_forced_close_for_admission, Qwen35PromptAnchor, Qwen35RetainedPrefix,
    Qwen35SlotAffinity, Qwen35SlotPreference, SlotId,
};
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::{default_layer_types, Qwen35Config, Qwen35Variant};
use crate::serve::api::engine_qwen35::Qwen35ForcedCloseAnchor;

const IMAGE: Option<[u8; 32]> = Some([7; 32]);

fn tiny_cfg() -> Qwen35Config {
    Qwen35Config {
        variant: Qwen35Variant::Dense,
        hidden_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 32,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 8,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 2,
        layer_types: default_layer_types(4, 2),
        partial_rotary_factor: 0.25,
        rope_theta: 1e7,
        rotary_dim: 8,
        mrope_section: [2, 2, 2, 2],
        mrope_interleaved: true,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 4096,
        vocab_size: 256,
        attn_output_gate: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: true,
        intermediate_size: Some(128),
        moe: None,
    }
}

fn cache(n_seqs: u32) -> HybridKvCache {
    let device = MlxDevice::new().expect("Metal device");
    HybridKvCache::new_with_options(&tiny_cfg(), &device, 64, n_seqs, true).expect("cache")
}

fn set_cursor(cache: &mut HybridKvCache, slot: SlotId, cursor: u32) {
    for block in &mut cache.full_attn {
        block.current_len[slot.0 as usize] = cursor;
    }
}

/// Previous prompt (12 tokens), its stable anchor at prompt − 5, the
/// generated span, and the forced-close anchor at prompt + 5.
struct Conversation {
    prompt: Vec<u32>,
    generated: Vec<u32>,
}

impl Conversation {
    fn new(seed: u32) -> Self {
        Self {
            prompt: (seed..seed + 12).collect(),
            generated: (seed + 100..seed + 120).collect(),
        }
    }

    fn stable(&self) -> &[u32] {
        &self.prompt[..7]
    }

    fn retained(&self) -> Vec<u32> {
        self.prompt.iter().chain(&self.generated).copied().collect()
    }

    fn forced(&self) -> Vec<u32> {
        self.retained()[..17].to_vec()
    }

    /// The follow-up the client renders: the forced-close prefix, then a
    /// re-rendered reasoning tail that differs from the generated one.
    fn follow_up(&self) -> Vec<u32> {
        let mut prompt = self.forced();
        prompt.extend([900, 901, 902, 903]);
        prompt
    }

    /// Seat the conversation on `slot` as decode completion leaves it, with
    /// the forced-close anchor published.
    fn seat(
        &self,
        cache: &mut HybridKvCache,
        slot: SlotId,
        vision: Option<[u8; 32]>,
    ) -> (Qwen35RetainedPrefix, Option<Qwen35PromptAnchor>) {
        set_cursor(cache, slot, self.stable().len() as u32);
        let stable_kv = cache
            .snapshot_slot_anchor(slot, self.stable().len())
            .expect("stable");
        set_cursor(cache, slot, self.forced().len() as u32);
        let forced_kv = cache
            .snapshot_slot_anchor(slot, self.forced().len())
            .expect("forced");
        let retained = self.retained();
        set_cursor(cache, slot, retained.len() as u32);
        let mut anchor = Some(Qwen35PromptAnchor {
            prompt_tokens: self.stable().to_vec(),
            kv: stable_kv,
            prefill_logits: vec![0.5; 4],
            vision_fingerprint: vision,
            spec: None,
            forced_close: None,
        });
        publish_qwen35_forced_close_anchor(
            &mut anchor,
            &retained,
            Some(Qwen35ForcedCloseAnchor {
                prompt_tokens: self.forced(),
                kv: forced_kv,
                vision_fingerprint: vision,
            }),
            cache,
            slot,
        );
        (
            Qwen35RetainedPrefix {
                tokens: retained,
                spec: None,
            },
            anchor,
        )
    }
}

fn forced_tokens(anchor: &Option<Qwen35PromptAnchor>) -> Option<&[u32]> {
    anchor
        .as_ref()
        .and_then(|anchor| anchor.forced_close.as_ref())
        .map(|forced| forced.prompt_tokens.as_slice())
}

/// The admission candidate one idle slot offers, as `admit_qwen35_slot`
/// builds it (exact vision fingerprint for this fixture).
fn candidate<'a>(
    anchor: &'a Option<Qwen35PromptAnchor>,
    prompt: &[u32],
    vision: Option<[u8; 32]>,
    cache: &HybridKvCache,
    slot: SlotId,
) -> Option<&'a [u32]> {
    let stable = anchor
        .as_ref()
        .filter(|anchor| anchor.vision_fingerprint == vision)
        .map(|anchor| anchor.prompt_tokens.as_slice());
    let forced = anchor
        .as_ref()
        .and_then(|anchor| anchor.forced_close_tokens_for(prompt, vision, cache, slot));
    qwen35_effective_anchor_tokens(stable, forced, prompt)
}

#[test]
fn g2_longest_matching_anchor_wins_and_ties_keep_the_stable_anchor() {
    let stable: Vec<u32> = (0..7).collect();
    let forced: Vec<u32> = (0..17).collect();
    let mut prompt: Vec<u32> = (0..17).collect();
    prompt.extend([99, 98]);
    let pick = |stable: Option<&[u32]>, forced: Option<&[u32]>, prompt: &[u32]| {
        qwen35_effective_anchor_tokens(stable, forced, prompt).map(<[u32]>::len)
    };
    assert_eq!(pick(Some(&stable), Some(&forced), &prompt), Some(17));
    assert_eq!(pick(None, Some(&forced), &prompt), Some(17));
    assert_eq!(pick(Some(&stable), None, &prompt), Some(7));
    // Partial match: the prompt diverges inside the forced span.
    let mut divergent = prompt.clone();
    divergent[12] = 500;
    assert_eq!(pick(Some(&stable), Some(&forced), &divergent), Some(7));
    // A prompt equal to the forced anchor is not served (no prefill logits).
    assert_eq!(pick(Some(&stable), Some(&forced), &forced), Some(7));
    // Tie: the stable anchor (it carries prefill logits).
    assert_eq!(
        pick(Some(&forced[..10]), Some(&forced[..10]), &prompt),
        Some(10)
    );
    // A stable anchor the prompt does not extend loses to a matching forced one.
    let other_stable = vec![42u32; 7];
    assert_eq!(pick(Some(&other_stable), Some(&forced), &prompt), Some(17));
}

#[test]
fn g2_affinity_prefers_the_forced_close_slot_and_restores_without_logits() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut kv = cache(3);
    let a = Conversation::new(1);
    let b = Conversation::new(1000);
    let (retained_a, anchor_a) = a.seat(&mut kv, SlotId(0), None);
    let (retained_b, anchor_b) = b.seat(&mut kv, SlotId(1), None);
    assert_eq!(forced_tokens(&anchor_a), Some(a.forced().as_slice()));
    let prompt = a.follow_up();
    // The live prefix cannot serve the re-render.
    assert!(!prompt.starts_with(&retained_a));

    let candidates = [
        (
            0usize,
            retained_a.as_slice(),
            candidate(&anchor_a, &prompt, None, &kv, SlotId(0)),
            None,
            false,
        ),
        (
            1usize,
            retained_b.as_slice(),
            candidate(&anchor_b, &prompt, None, &kv, SlotId(1)),
            None,
            false,
        ),
    ];
    assert_eq!(
        qwen35_slot_affinity(candidates, &prompt),
        Some(Qwen35SlotAffinity::Idle(Qwen35SlotPreference {
            slot: SlotId(0),
            cached_tokens: a.forced().len(),
            restore_prompt_anchor: true,
            use_prefill_logits: false,
        }))
    );

    // Equal forced-close matches in an active and an idle slot: the idle
    // slot wins, exactly as for any other equal-length candidates.
    let tie = [
        (0usize, &[][..], Some(&prompt[..17]), None, true),
        (1usize, &[][..], Some(&prompt[..17]), None, false),
    ];
    assert_eq!(
        qwen35_slot_affinity(tie, &prompt).map(|affinity| affinity.preference().slot),
        Some(SlotId(1))
    );
}

#[test]
fn g2_wrong_fingerprint_or_unrestorable_cursor_falls_back_to_the_stable_anchor() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut kv = cache(2);
    let a = Conversation::new(1);
    let (_, anchor) = a.seat(&mut kv, SlotId(0), IMAGE);
    let prompt = a.follow_up();
    assert_eq!(
        candidate(&anchor, &prompt, IMAGE, &kv, SlotId(0)).map(<[u32]>::len),
        Some(17)
    );
    // Another image: neither anchor (the stable filter is the caller's).
    assert_eq!(candidate(&anchor, &prompt, None, &kv, SlotId(0)), None);
    // The forced anchor alone checks its own fingerprint.
    let slot_anchor = anchor.as_ref().expect("anchor");
    assert!(slot_anchor
        .forced_close_tokens_for(&prompt, Some([8; 32]), &kv, SlotId(0))
        .is_none());
    // Live cursor behind the forced boundary: offered only the stable anchor.
    set_cursor(&mut kv, SlotId(0), 10);
    assert_eq!(
        candidate(&anchor, &prompt, IMAGE, &kv, SlotId(0)).map(<[u32]>::len),
        Some(7)
    );
}

#[test]
fn g3_publication_keeps_only_a_coherent_anchor() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut kv = cache(2);
    let a = Conversation::new(1);
    let slot = SlotId(0);
    let (retained, published) = a.seat(&mut kv, slot, None);
    assert!(
        forced_tokens(&published).is_some(),
        "coherent anchor is published"
    );
    let forced = published.as_ref().unwrap().forced_close.clone().unwrap();
    let base = || {
        let mut anchor = published.clone();
        anchor.as_mut().unwrap().forced_close = None;
        anchor
    };
    let publish = |anchor: &mut Option<Qwen35PromptAnchor>,
                   retained: &[u32],
                   forced: Qwen35ForcedCloseAnchor,
                   kv: &HybridKvCache| {
        publish_qwen35_forced_close_anchor(anchor, retained, Some(forced), kv, slot);
    };

    // No slot anchor: nothing to attach to.
    let mut none = None;
    publish(&mut none, &retained, forced.clone(), &kv);
    assert!(none.is_none());
    // Not inside the published live prefix.
    let mut anchor = base();
    publish(&mut anchor, &retained[..16], forced.clone(), &kv);
    assert!(forced_tokens(&anchor).is_none());
    // Does not extend the stable anchor.
    let mut anchor = base();
    let mut other = forced.clone();
    other.prompt_tokens[0] = 777;
    publish(&mut anchor, &retained, other, &kv);
    assert!(forced_tokens(&anchor).is_none());
    // Different vision fingerprint.
    let mut anchor = base();
    let mut other = forced.clone();
    other.vision_fingerprint = IMAGE;
    publish(&mut anchor, &retained, other, &kv);
    assert!(forced_tokens(&anchor).is_none());
    // Cursor no longer covers it.
    let mut anchor = base();
    set_cursor(&mut kv, slot, 12);
    publish(&mut anchor, &retained, forced.clone(), &kv);
    assert!(forced_tokens(&anchor).is_none());
    // A natural close publishes None and clears a previous one.
    let mut anchor = published.clone();
    publish_qwen35_forced_close_anchor(&mut anchor, &retained, None, &kv, slot);
    assert!(forced_tokens(&anchor).is_none());
}

#[test]
fn g3_every_admission_takes_the_anchor_out_of_the_slot() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut kv = cache(2);
    let a = Conversation::new(1);
    let slot = SlotId(0);
    let (_, published) = a.seat(&mut kv, slot, None);
    let forced_len = a.forced().len();
    let stable_len = a.stable().len();
    let follow_up = a.follow_up();
    let mut divergent = a.prompt.clone();
    divergent.extend([1, 2, 3, 4, 5, 6, 7, 8]);

    // (anchor_hit, cached_tokens, prompt) -> expect the anchor back
    let cases: [(bool, usize, &[u32], bool); 6] = [
        (true, forced_len, &follow_up, true),
        // Stable-anchor hit, live reuse, a cold miss, a divergent request.
        (true, stable_len, &follow_up, false),
        (false, forced_len, &follow_up, false),
        (false, 0, &follow_up, false),
        (true, stable_len, &divergent, false),
        // Right length but the prompt diverges inside the anchor.
        (true, forced_len, &divergent, false),
    ];
    for (anchor_hit, cached, prompt, expect) in cases {
        let mut slot_anchor = published.clone();
        let taken = take_qwen35_forced_close_for_admission(
            &mut slot_anchor,
            anchor_hit,
            cached,
            prompt,
            None,
        );
        assert_eq!(taken.is_some(), expect, "hit={anchor_hit} cached={cached}");
        assert!(
            forced_tokens(&slot_anchor).is_none(),
            "admission must never leave the anchor in the slot"
        );
        assert!(slot_anchor.is_some(), "the stable anchor stays");
    }
    // Wrong vision fingerprint at a matching length.
    let mut slot_anchor = published.clone();
    assert!(take_qwen35_forced_close_for_admission(
        &mut slot_anchor,
        true,
        forced_len,
        &follow_up,
        IMAGE
    )
    .is_none());
    assert!(forced_tokens(&slot_anchor).is_none());
    // No slot anchor (reset, cancellation without a checkpoint, graft or
    // restore failure all set it to None): nothing to take.
    let mut cleared: Option<Qwen35PromptAnchor> = None;
    assert!(take_qwen35_forced_close_for_admission(
        &mut cleared,
        true,
        forced_len,
        &follow_up,
        None
    )
    .is_none());
}

#[test]
fn g3_a_divergent_turn_leaves_no_stale_anchor_for_a_later_match() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut kv = cache(2);
    let a = Conversation::new(1);
    let slot = SlotId(0);
    let (_, mut slot_anchor) = a.seat(&mut kv, slot, None);

    // R2 diverges after the stable anchor: admission restores the stable
    // anchor and discards the forced-close one; R2 then overwrites the rows
    // the forced-close anchor relied on and grows the cursor past it.
    let mut r2 = a.stable().to_vec();
    r2.extend(300..330);
    assert!(take_qwen35_forced_close_for_admission(
        &mut slot_anchor,
        true,
        a.stable().len(),
        &r2,
        None
    )
    .is_none());
    set_cursor(&mut kv, slot, r2.len() as u32);

    // R3 matches the old forced-close tokens and the cursor covers them, but
    // the slot offers only the stable anchor.
    let r3 = a.follow_up();
    assert!(kv.sequence_len_for_slot(slot).unwrap() as usize >= a.forced().len());
    assert_eq!(
        candidate(&slot_anchor, &r3, None, &kv, slot).map(<[u32]>::len),
        Some(7)
    );
}

#[test]
fn g3_slot_spills_never_carry_the_anchor() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut kv = cache(2);
    let a = Conversation::new(1);
    let (mut retained, mut anchor) = a.seat(&mut kv, SlotId(0), None);
    assert!(forced_tokens(&anchor).is_some());
    let mut store: Qwen35SlotSpillStore = SlotSpillStore::new(1 << 30, 8);
    spill_reclaimed_qwen35_slot(&kv, &mut store, SlotId(0), &mut retained, &mut anchor);
    assert!(
        anchor.is_none() && retained.is_empty(),
        "the ledger moved to the store"
    );

    // The conversation returns to another slot through the spill.
    let mut retained_dest = Qwen35RetainedPrefix::default();
    let mut anchor_dest = None;
    let prompt = a.follow_up();
    let preference = reclaim_qwen35_slot(
        &mut kv,
        &mut store,
        SlotId(1),
        &mut retained_dest,
        &mut anchor_dest,
        None,
        &prompt,
        None,
        None,
    )
    .expect("spill restore")
    .expect("the spilled stable anchor matches");
    assert_eq!(preference.cached_tokens, a.stable().len());
    assert!(anchor_dest.is_some());
    assert!(
        forced_tokens(&anchor_dest).is_none(),
        "a spill restored the one-shot anchor"
    );
}
