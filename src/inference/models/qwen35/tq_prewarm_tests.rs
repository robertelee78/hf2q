//! Metal gates for [`HybridKvCache::prewarm_tq_storage`]: it must leave
//! every cursor at zero and change no served bit.
//!
//! The model is the synthetic native K-quant model of the joint verify
//! tests. Two caches, one prewarmed, run the same prefill chunks and greedy
//! decode steps on two slots; every logit must be bitwise equal.

use mlx_native::MlxDevice;

use super::joint_verify_tests::{padded_policy, synthetic_model};
use super::kv_cache::HybridKvCache;
use super::model::Qwen35Model;
use super::real_artifact_joint_decode_tests::argmax;
use super::spec_decode::positions_for_range;
use crate::serve::multi_seq_kv::SlotId;

const SLOTS: u32 = 2;
const CAPACITY: u32 = 128;
const DECODE_STEPS: usize = 8;

fn cache(model: &Qwen35Model, prewarm: bool) -> HybridKvCache {
    let mut cache = model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&model.cfg, device, CAPACITY, SLOTS, true)
        })
        .expect("allocate TQ cache");
    if prewarm {
        model
            .with_gpu_cache_mut(|device, registry| cache.prewarm_tq_storage(device, registry))
            .expect("prewarm TQ storage");
    }
    cache
}

fn tokens(slot: u32, start: usize, len: usize) -> Vec<u32> {
    (start..start + len)
        .map(|position| (slot * 131 + position as u32 * 37 + 11) % 512)
        .collect()
}

/// Prefill and greedy-decode both slots; every logit of every step, as bits.
fn served_bits(model: &Qwen35Model, cache: &mut HybridKvCache) -> Vec<u32> {
    let mut bits = Vec::new();
    for (slot, chunks) in [(0u32, [33usize, 20]), (1, [17, 5])] {
        let mut cursor = 0usize;
        let mut logits = Vec::new();
        for len in chunks {
            logits = model
                .forward_gpu_last_logits(
                    &tokens(slot, cursor, len),
                    &positions_for_range(cursor as i32, len),
                    cache,
                    SlotId(slot),
                )
                .unwrap_or_else(|error| panic!("slot {slot} chunk at {cursor}: {error:#}"));
            bits.extend(logits.iter().map(|value| value.to_bits()));
            cursor += len;
        }
        for step in 0..DECODE_STEPS {
            let token = argmax(&logits);
            logits = model
                .forward_gpu_last_logits(&[token], &[cursor as i32; 4], cache, SlotId(slot))
                .unwrap_or_else(|error| panic!("slot {slot} decode step {step}: {error:#}"));
            bits.push(token);
            bits.extend(logits.iter().map(|value| value.to_bits()));
            cursor += 1;
        }
    }
    bits
}

fn with_synthetic_model(run: impl FnOnce(&Qwen35Model)) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let model = synthetic_model(&device, padded_policy(), false);
    // `synthetic_model` primed the thread's GPU cache for the model's
    // address before the move out of it.
    model
        .ensure_gpu_cache_primed()
        .expect("prime the GPU cache at the model's final address");
    run(&model);
}

#[test]
fn tq_prewarm_leaves_cursors_at_zero_and_changes_no_served_bit() {
    with_synthetic_model(|model| {
        let mut prewarmed = cache(model, true);
        for slot in 0..SLOTS {
            assert_eq!(
                prewarmed.sequence_len_for_slot(SlotId(slot)).ok(),
                Some(0),
                "slot {slot} cursor moved"
            );
        }
        for (rank, layer) in prewarmed.full_attn.iter().enumerate() {
            assert!(
                layer.current_len.iter().all(|&len| len == 0),
                "full-attention layer {rank} cursor moved"
            );
        }
        let plain = served_bits(model, &mut cache(model, false));
        let warmed = served_bits(model, &mut prewarmed);
        assert!(plain == warmed, "the prewarmed cache served different bits");
    });
}

#[test]
fn tq_prewarm_refuses_a_cache_that_holds_rows() {
    with_synthetic_model(|model| {
        let mut used = cache(model, false);
        model
            .forward_gpu_last_logits(
                &tokens(0, 0, 17),
                &positions_for_range(0, 17),
                &mut used,
                SlotId(0),
            )
            .expect("prefill");
        let refused =
            model.with_gpu_cache_mut(|device, registry| used.prewarm_tq_storage(device, registry));
        assert!(refused.is_err(), "prewarm overwrote row 0 of a used cache");
    });
}
