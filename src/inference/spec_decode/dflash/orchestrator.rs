//! DFlash spec-decode orchestrator (ADR-030 Phase 4).
//!
//! Wires together:
//! - Drafter model forward ([`super::forward::dispatch_dflash_model_forward`])
//! - Target forward verify (target's `forward_decode_verify_batched`,
//!   to be plumbed in subsequent iter with hidden-state capture)
//! - Accept-prefix logic (existing [`crate::inference::spec_decode::verifier::accept_prefix_argmax`])
//! - KV cache rollback (drafter cache + target cache)
//!
//! ## Phase 4 loop structure
//!
//! Per spec-decode step:
//! 1. **Propose**: drafter generates K candidate tokens. For K=0, skip
//!    drafting entirely (degrades to single-token target decode).
//! 2. **Verify**: target forward on `[last_committed, d_1, …, d_K]` (K+1
//!    positions). Captures hidden states at `target_layer_ids` for the
//!    next round's drafter input. Returns per-position argmaxes.
//! 3. **Accept**: `accept_prefix_argmax(drafts, target_argmaxes)`
//!    returns `(accept_count, model_token)`. The accepted tokens are
//!    `drafts[..accept_count]`; the final token is `model_token`
//!    (target's free continuation at the first mismatch).
//! 4. **Commit**: append `accepted_tokens + [model_token]` to the
//!    output sequence; advance target KV by `accept_count + 1`
//!    positions; advance drafter KV similarly.
//! 5. **Rollback**: target KV by `K - accept_count` positions (the
//!    rejected proposals' KV writes); drafter cache by the same.
//!
//! ## Greedy byte-identity invariant (the coherence guarantee)
//!
//! At temperature=0, this orchestrator must produce **byte-identical**
//! output to single-token target decode. The mathematical guarantee:
//! `accept_prefix_argmax` only accepts a draft token `d_i` if it equals
//! `argmax(target_logits_at_i)`. The "free" token at position
//! `accept_count` is `argmax(target_logits_at_accept_count)` — exactly
//! what greedy single-token decode would emit. KV rollback after the
//! K-positional verify undoes only the rejected portion.
//!
//! Phase 4 first piece (this commit): orchestrator state struct + the
//! pure-math step round (propose → verify result → accept → next-state).
//! Target forward integration + KV rollback wiring land in subsequent
//! commits.

use crate::inference::spec_decode::verifier::accept_prefix_argmax;

/// Output of one spec-decode round.
#[derive(Debug, Clone, PartialEq)]
pub struct RoundResult {
    /// Tokens to commit to the output sequence for this round.
    /// Always at least one element (the model's "free" token at the
    /// first-mismatch position).
    pub committed_tokens: Vec<u32>,
    /// How many of the proposed drafts were accepted.
    /// `committed_tokens.len() == accept_count + 1`.
    pub accept_count: usize,
    /// True if the model's free token is an EOS marker; orchestrator
    /// should stop generating.
    pub hit_eos: bool,
}

/// EOS-aware [`accept_prefix_argmax`] wrapper.
///
/// Applies the standard greedy accept rule, then:
/// - If any token in `drafts[..accept_count]` is in `eos_token_ids`, truncate
///   acceptance at the first EOS and treat it as the model's free
///   continuation (no further generation).
/// - If `model_token` is in `eos_token_ids`, mark `hit_eos = true`.
pub fn step_round_from_argmaxes(
    drafts: &[u32],
    target_argmaxes: &[u32],
    eos_token_ids: &[u32],
) -> RoundResult {
    let (accept_count, model_token) = accept_prefix_argmax(drafts, target_argmaxes);

    // Check for EOS inside the accepted prefix — if a draft emitted EOS
    // and target agreed, generation stops at that draft.
    let mut effective_accept = accept_count;
    let mut effective_final: u32 = model_token;
    let mut hit_eos = false;
    for (i, &t) in drafts.iter().take(accept_count).enumerate() {
        if eos_token_ids.contains(&t) {
            effective_accept = i;
            effective_final = t;
            hit_eos = true;
            break;
        }
    }
    if !hit_eos && eos_token_ids.contains(&model_token) {
        hit_eos = true;
    }

    let mut committed = drafts[..effective_accept].to_vec();
    committed.push(effective_final);
    RoundResult {
        committed_tokens: committed,
        accept_count: effective_accept,
        hit_eos,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::dflash::{
        config::DFlashConfig,
        forward::dispatch_dflash_model_forward,
        kv_cache::DFlashKvCache,
        tensors::DFlashModelTensors,
        weights::{DFlashWeights, DFlashWeightsFile},
    };
    use mlx_native::{DType, KernelRegistry, MlxDevice};

    #[test]
    fn k0_empty_drafts_degrades_to_single_token() {
        // K=0: no drafts proposed. The target argmaxes have just 1
        // entry (the next-token argmax of target at the last position).
        // Round emits exactly that 1 token.
        let drafts: Vec<u32> = vec![];
        let target_argmaxes: Vec<u32> = vec![42];
        let result = step_round_from_argmaxes(&drafts, &target_argmaxes, &[]);
        assert_eq!(result.committed_tokens, vec![42]);
        assert_eq!(result.accept_count, 0);
        assert!(!result.hit_eos);
    }

    #[test]
    fn full_accept_returns_all_drafts_plus_model_token() {
        // K=3 drafts, all match target → accept all 3, plus 1 free.
        let drafts = vec![10, 20, 30];
        // Target argmaxes for K+1=4 positions: first 3 match drafts,
        // last is the model's continuation.
        let target = vec![10, 20, 30, 99];
        let result = step_round_from_argmaxes(&drafts, &target, &[]);
        assert_eq!(result.committed_tokens, vec![10, 20, 30, 99]);
        assert_eq!(result.accept_count, 3);
        assert!(!result.hit_eos);
    }

    #[test]
    fn partial_accept_truncates_at_first_mismatch() {
        // K=4 drafts; first 2 match, 3rd differs. Accept 2, model
        // continues with whatever target predicted at position 2.
        let drafts = vec![10, 20, 30, 40];
        let target = vec![10, 20, 88, 99]; // mismatch at index 2
        let result = step_round_from_argmaxes(&drafts, &target, &[]);
        assert_eq!(result.committed_tokens, vec![10, 20, 88]);
        assert_eq!(result.accept_count, 2);
        assert!(!result.hit_eos);
    }

    #[test]
    fn eos_in_accepted_prefix_stops_generation() {
        // K=3 drafts; first 2 match target, 2nd is EOS. The 3rd draft
        // (even if it matched) doesn't matter — generation stops.
        let drafts = vec![10, 7, 30];
        let target = vec![10, 7, 30];
        let result = step_round_from_argmaxes(&drafts, &target, &[7]);
        assert_eq!(result.committed_tokens, vec![10, 7]);
        assert_eq!(result.accept_count, 1); // first EOS at index 1
        assert!(result.hit_eos);
    }

    #[test]
    fn eos_as_model_free_token_sets_hit_eos() {
        // K=2 drafts, partial-accept at index 1 (first matches), model's
        // continuation at position 1 IS an EOS.
        let drafts = vec![10, 20];
        let target = vec![10, 1, 30];
        let result = step_round_from_argmaxes(&drafts, &target, &[1]);
        assert_eq!(result.committed_tokens, vec![10, 1]);
        assert_eq!(result.accept_count, 1);
        assert!(result.hit_eos);
    }

    #[test]
    fn eos_check_handles_full_accept_with_eos_continuation() {
        // K=2 drafts both accepted, model's free token is EOS.
        let drafts = vec![10, 20];
        let target = vec![10, 20, 1];
        let result = step_round_from_argmaxes(&drafts, &target, &[1]);
        assert_eq!(result.committed_tokens, vec![10, 20, 1]);
        assert_eq!(result.accept_count, 2);
        assert!(result.hit_eos);
    }

    /// Integration smoke test for orchestrator + drafter: drafter runs
    /// end-to-end producing h_final; then we SIMULATE what the target
    /// verify would produce (synthetic argmaxes) and feed both into
    /// step_round_from_argmaxes. Validates the orchestrator+drafter
    /// API surface from the caller's perspective.
    ///
    /// This is the seam where Phase 4 target integration will plug in:
    /// the synthetic argmaxes here are placeholders for what
    /// `forward_decode_verify_batched` (modified per ADR-030 §3.5
    /// Phase 4) will return.
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_orchestrator_drafter_loop_with_simulated_target() {
        let cfg = DFlashConfig::from_json_str(
            crate::inference::spec_decode::dflash::config::tests::GEMMA4_26B_A4B_DFLASH_CONFIG,
        )
        .expect("config parse");
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");
        let mut cache = DFlashKvCache::new(&device, &cfg, 128).expect("cache");

        let block_size = 8u32;
        let ctx_chunk = 4u32;
        let hidden = cfg.hidden_size as u32;
        let fc_in = cfg.fc_input_dim() as u32;

        // Build synthetic h + target_hidden_concat (caller would supply
        // these from target's embed + hidden capture).
        let h_elem = (block_size as usize) * (hidden as usize);
        let mut h = device
            .alloc_buffer(h_elem * 4, DType::F32, vec![block_size as usize, hidden as usize])
            .expect("alloc h");
        {
            let s = h.as_mut_slice::<f32>().expect("h slice");
            for v in s.iter_mut() { *v = 1.0; }
        }
        let thc_elem = (ctx_chunk as usize) * (fc_in as usize);
        let mut target_hidden = device
            .alloc_buffer(thc_elem * 4, DType::F32, vec![ctx_chunk as usize, fc_in as usize])
            .expect("alloc target_hidden");
        {
            let s = target_hidden.as_mut_slice::<f32>().expect("target_hidden slice");
            for (i, v) in s.iter_mut().enumerate() {
                *v = 0.1 + ((i % 17) as f32) / 170.0;
            }
        }

        // Run drafter forward → h_final [L, hidden]
        let h_final = dispatch_dflash_model_forward(
            &mut registry, &device, &h, &target_hidden,
            &tensors, &mut cache, &cfg, block_size, ctx_chunk,
        )
        .expect("drafter forward");
        assert_eq!(h_final.element_count(), (block_size as usize) * (hidden as usize));

        // Caller would now apply target's lm_head + softcap on h_final
        // to get K draft tokens. For this test, we SIMULATE:
        // drafts = [1, 2, 3, 4, 5, 6, 7] (K=7 placeholder tokens)
        let drafts: Vec<u32> = (1..=7).collect();
        // Simulate target verify returning argmaxes — 4 match, 1 mismatch, 3 free
        let target_argmaxes = vec![1, 2, 3, 4, 99, 100, 101, 102];

        // Apply round math
        let round = step_round_from_argmaxes(&drafts, &target_argmaxes, &[]);
        assert_eq!(round.accept_count, 4);
        assert_eq!(round.committed_tokens, vec![1, 2, 3, 4, 99]);
        assert!(!round.hit_eos);

        // Cache should have advanced by ctx_chunk on all layers (drafter
        // forward side-effect).
        for (i, l) in cache.layers.iter().enumerate() {
            assert_eq!(
                l.seq_len, ctx_chunk,
                "layer {i} cache should advance by ctx_chunk"
            );
        }
    }
}
