//! Official-artifact gate for the Qwen3.6 prefill MoE schedule reuse.
//!
//! Run with:
//!
//! ```text
//! HF2Q_QWEN36_GGUF=/path/qwen36.gguf cargo test --release --bin hf2q \
//!   qwen36_prefill_shared_expert_schedule_matches_per_call_bitwise \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```

use super::gpu_ffn::{
    take_shared_expert_schedule_admissions, with_forced_per_call_expert_schedule,
};
use super::kv_cache::HybridKvCache;
use super::real_artifact_joint_decode_tests::{
    argmax, assert_caches_equal, bits, load, GateRouting,
};
use super::spec_decode::positions_for_range;
use crate::serve::multi_seq_kv::SlotId;

const TOKEN_PATTERN: [u32; 5] = [9707, 374, 279, 15, 1_782];

fn new_cache(model: &super::model::Qwen35Model, capacity: u32) -> HybridKvCache {
    model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&model.cfg, device, capacity, 1, true)
        })
        .expect("allocate one-slot TQ cache")
}

fn prompt(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| TOKEN_PATTERN[index % TOKEN_PATTERN.len()])
        .collect()
}

/// The shared gate/up/down routing schedule must preserve every prefill result
/// and cache byte relative to the pre-P1 per-call schedule. The cases straddle
/// the 32-row route floor and cover the repair-turn suffix sizes used by the
/// Qwen3.6 quality workload. Four greedy continuations prove that the equal
/// prefill state remains equal when consumed by decode.
#[test]
#[ignore = "requires HF2Q_QWEN36_GGUF (official Qwen3.6 artifact) and Apple Metal"]
fn qwen36_prefill_shared_expert_schedule_matches_per_call_bitwise() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model = load("HF2Q_QWEN36_GGUF", GateRouting::Exact);
    model
        .ensure_gpu_cache_primed()
        .expect("prime GPU cache at the gate's final model address");

    for prefill_len in [31usize, 33, 157, 682, 2048] {
        let capacity = (prefill_len + 8) as u32;
        let tokens = prompt(prefill_len);
        let positions = positions_for_range(0, prefill_len);
        let mut per_call_cache = new_cache(&model, capacity);
        let mut shared_cache = new_cache(&model, capacity);
        take_shared_expert_schedule_admissions();

        let mut per_call_logits = with_forced_per_call_expert_schedule(true, || {
            model.forward_gpu_last_logits(&tokens, &positions, &mut per_call_cache, SlotId(0))
        })
        .unwrap_or_else(|error| panic!("prefill {prefill_len} per-call route: {error:#}"));
        assert_eq!(
            take_shared_expert_schedule_admissions(),
            0,
            "prefill {prefill_len}: forced per-call arm admitted shared schedule"
        );
        let mut shared_logits = with_forced_per_call_expert_schedule(false, || {
            model.forward_gpu_last_logits(&tokens, &positions, &mut shared_cache, SlotId(0))
        })
        .unwrap_or_else(|error| panic!("prefill {prefill_len} shared route: {error:#}"));
        let shared_admissions = take_shared_expert_schedule_admissions();
        if prefill_len > 32 {
            assert!(
                shared_admissions > 0,
                "prefill {prefill_len}: official model did not exercise shared schedule"
            );
        } else {
            assert_eq!(
                shared_admissions, 0,
                "prefill {prefill_len}: below-floor chunk admitted shared schedule"
            );
        }

        assert_eq!(
            bits(&per_call_logits),
            bits(&shared_logits),
            "prefill {prefill_len}: last-row logits"
        );
        assert_caches_equal(
            &per_call_cache,
            &shared_cache,
            &format!("prefill {prefill_len}"),
        );

        let mut greedy_tokens = Vec::with_capacity(4);
        for step in 0..4usize {
            let token = argmax(&per_call_logits);
            assert_eq!(
                token,
                argmax(&shared_logits),
                "prefill {prefill_len} step {step}"
            );
            greedy_tokens.push(token);
            let decode_positions = positions_for_range((prefill_len + step) as i32, 1);
            per_call_logits = with_forced_per_call_expert_schedule(true, || {
                model.forward_gpu_last_logits(
                    &[token],
                    &decode_positions,
                    &mut per_call_cache,
                    SlotId(0),
                )
            })
            .unwrap_or_else(|error| {
                panic!("prefill {prefill_len} decode {step} per-call route: {error:#}")
            });
            shared_logits = with_forced_per_call_expert_schedule(false, || {
                model.forward_gpu_last_logits(
                    &[token],
                    &decode_positions,
                    &mut shared_cache,
                    SlotId(0),
                )
            })
            .unwrap_or_else(|error| {
                panic!("prefill {prefill_len} decode {step} shared route: {error:#}")
            });
            assert_eq!(
                bits(&per_call_logits),
                bits(&shared_logits),
                "prefill {prefill_len} decode {step}: logits"
            );
            assert_caches_equal(
                &per_call_cache,
                &shared_cache,
                &format!("prefill {prefill_len} decode {step}"),
            );
        }
        eprintln!(
            "QWEN36_PREFILL_SCHEDULE_GATE prefill_len={prefill_len} greedy_tokens={greedy_tokens:?} bitwise_equal=true"
        );
    }
}
