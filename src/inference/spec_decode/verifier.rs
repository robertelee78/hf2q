//! Verifier API contract for ADR-029 Phase 2.
//!
//! This module defines the *contract* the multi-token verify forward
//! must satisfy. Phase 2 (forward_decode_verify implementation) lands
//! the GPU-backed `Verifier` impl that runs the model on K+1 tokens
//! in one batched forward and returns per-position logits.
//!
//! Phase 1 (this iter, iter-114) ships only the trait + the
//! `accept_prefix` decision logic + a mock-verifier integration test.
//! No production wire-up.
//!
//! # Spec-decode loop (target shape, lands in Phase 3)
//!
//! ```text
//! loop {
//!     drafts = ngram_proposer::propose(generated, &cfg);
//!     if drafts.is_empty() {
//!         // No proposal — fall back to single-token decode.
//!         t = forward_decode(generated.last(), seq_pos);
//!         generated.push(t);
//!         seq_pos += 1;
//!     } else {
//!         // Multi-token verify: model runs on
//!         // [generated.last(), draft_1..draft_K] = K+1 inputs.
//!         logits_per_pos = verifier.verify(&[last] ++ drafts)?;
//!         (accept_count, model_token) = accept_prefix(&drafts, &logits_per_pos);
//!         generated.extend_from_slice(&drafts[..accept_count]);
//!         generated.push(model_token);
//!         seq_pos += accept_count + 1;
//!         verifier.rollback_kv_to(seq_pos)?;  // truncate rejected
//!     }
//! }
//! ```
//!
//! # Falsifier gate (ADR-029 Phase 2 acceptance)
//!
//! At `cfg.k == 0` the proposer always returns empty, so the loop
//! degrades to default `forward_decode` — output MUST be byte-identical
//! to non-spec-decode generation. Phase 2's `forward_decode_verify`
//! impl must clear this gate via `scripts/sourdough_gate.sh` before
//! Phase 3 production wire-up.

/// Errors a verifier can produce.
#[derive(Debug, thiserror::Error)]
pub enum VerifierError {
    #[error("verifier: empty input — at least 1 token required")]
    EmptyInput,
    #[error("verifier: too many tokens for verify pass (got {got}, max {max})")]
    TooManyTokens { got: usize, max: usize },
    #[error("verifier: model error: {0}")]
    Model(#[from] anyhow::Error),
}

/// Per-position logits returned by a verify pass.
///
/// Shape: `[k+1][vocab_size]`. Position 0 corresponds to the model's
/// continuation of the existing context (i.e. argmax replaces the
/// drafter's output for position 0 if it disagrees). Positions 1..=K
/// correspond to the model's predictions GIVEN the K speculative
/// tokens — argmax at position `i` tells us what the model would emit
/// if the previous `i` drafts were all accepted.
pub type VerifyLogits = Vec<Vec<f32>>;

/// Multi-token verifier contract.
///
/// Implementations:
/// - **Phase 2** (pending): GPU-backed `MlxVerifier` that calls a
///   batched `forward_decode_verify` returning per-position logits in
///   one forward pass. KV cache snapshot + rollback live here.
/// - **Tests**: `MockVerifier` (this module) returns operator-supplied
///   logits, used to exercise `accept_prefix` + the proposer→verify
///   loop without a model load.
pub trait Verifier {
    /// Verify K+1 input tokens and return per-position logits.
    ///
    /// `tokens[0]` is the current verified position; `tokens[1..]` are
    /// the K speculative draft tokens. Returns `[K+1][vocab]` logits.
    fn verify(&mut self, tokens: &[u32]) -> Result<VerifyLogits, VerifierError>;

    /// Roll back the KV cache so the next call starts from `seq_pos`.
    ///
    /// Called after the spec-decode loop computes `accept_count`. The
    /// implementation must truncate any KV state past `seq_pos` so the
    /// subsequent verify pass sees the same KV state as if only
    /// `accept_count + 1` of the K+1 tokens had been processed.
    fn rollback_kv_to(&mut self, seq_pos: usize) -> Result<(), VerifierError>;
}

/// Decide how many of the proposed `drafts` to accept based on the
/// verifier's per-position logits.
///
/// Returns `(accept_count, model_token)` where:
/// - `accept_count` is the number of leading drafts whose argmax
///   matches the model's prediction (0..=drafts.len()).
/// - `model_token` is the model's argmax at position `accept_count` —
///   this is the "free" extra token gained from the verify pass.
///
/// # Greedy decode contract
///
/// At temperature=0 (greedy), spec-decode is byte-identical to default
/// decode: the accepted prefix is exactly what default decode would
/// have produced, and `model_token` is the next token default decode
/// would emit. Stochastic sampling (temperature>0) requires a more
/// involved acceptance distribution per Leviathan et al. (2023); this
/// function is the greedy variant and matches vLLM's
/// `RejectionSampler.greedy_match` semantics.
pub fn accept_prefix(drafts: &[u32], logits_per_pos: &VerifyLogits) -> (usize, u32) {
    if logits_per_pos.is_empty() {
        return (0, 0);
    }

    // Position 0 corresponds to the FIRST draft slot's prediction.
    // Iterate K positions checking draft[i] == argmax(logits[i]).
    let mut accept_count = 0;
    for i in 0..drafts.len() {
        if i >= logits_per_pos.len() { break; }
        let argmax = argmax_u32(&logits_per_pos[i]);
        if argmax == drafts[i] {
            accept_count += 1;
        } else {
            // First mismatch — model_token is the model's argmax here.
            return (accept_count, argmax);
        }
    }

    // All drafts accepted — model_token comes from the K+1th logits row
    // (one past the last accepted draft).
    if accept_count < logits_per_pos.len() {
        let model_token = argmax_u32(&logits_per_pos[accept_count]);
        (accept_count, model_token)
    } else {
        // Edge case: logits_per_pos.len() == drafts.len() (no extra
        // position for model_token). Fall back to last argmax.
        let model_token = argmax_u32(logits_per_pos.last().unwrap());
        (accept_count, model_token)
    }
}

fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::MIN;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a logits vector where `target` has the highest score.
    fn one_hot(vocab: usize, target: u32) -> Vec<f32> {
        let mut v = vec![0.0_f32; vocab];
        v[target as usize] = 1.0;
        v
    }

    #[test]
    fn accept_prefix_full_accept() {
        // drafts: [10, 20, 30]; verify says model would emit [10, 20, 30, 40]
        let drafts = vec![10u32, 20, 30];
        let logits = vec![
            one_hot(100, 10),
            one_hot(100, 20),
            one_hot(100, 30),
            one_hot(100, 40), // K+1th position
        ];
        let (accept, tok) = accept_prefix(&drafts, &logits);
        assert_eq!(accept, 3, "all 3 drafts should be accepted");
        assert_eq!(tok, 40, "model_token should be the K+1th argmax");
    }

    #[test]
    fn accept_prefix_partial_accept() {
        // drafts: [10, 20, 30]; model says [10, 20, 99 (mismatch), ...]
        let drafts = vec![10u32, 20, 30];
        let logits = vec![
            one_hot(100, 10),
            one_hot(100, 20),
            one_hot(100, 99),
            one_hot(100, 40),
        ];
        let (accept, tok) = accept_prefix(&drafts, &logits);
        assert_eq!(accept, 2, "first 2 accepted, 3rd mismatches");
        assert_eq!(tok, 99, "model_token = model's argmax at mismatch position");
    }

    #[test]
    fn accept_prefix_zero_accept() {
        // drafts: [10, 20, 30]; model disagrees on token 0
        let drafts = vec![10u32, 20, 30];
        let logits = vec![
            one_hot(100, 99),
            one_hot(100, 20),
            one_hot(100, 30),
            one_hot(100, 40),
        ];
        let (accept, tok) = accept_prefix(&drafts, &logits);
        assert_eq!(accept, 0);
        assert_eq!(tok, 99);
    }

    #[test]
    fn accept_prefix_empty_drafts_uses_first_logits() {
        // drafts: []; logits has 1 row (= the model's continuation).
        let drafts: Vec<u32> = Vec::new();
        let logits = vec![one_hot(100, 42)];
        let (accept, tok) = accept_prefix(&drafts, &logits);
        assert_eq!(accept, 0);
        assert_eq!(tok, 42, "model_token = argmax at position 0");
    }

    #[test]
    fn accept_prefix_empty_logits_returns_zero_zero() {
        let drafts = vec![10u32, 20];
        let logits: VerifyLogits = Vec::new();
        let (accept, tok) = accept_prefix(&drafts, &logits);
        assert_eq!(accept, 0);
        assert_eq!(tok, 0);
    }

    /// Mock verifier — returns operator-supplied logits, simulating
    /// the model's per-position predictions for an integration test.
    struct MockVerifier {
        scripted: VerifyLogits,
        rollbacks: Vec<usize>,
    }

    impl Verifier for MockVerifier {
        fn verify(&mut self, _tokens: &[u32]) -> Result<VerifyLogits, VerifierError> {
            Ok(self.scripted.clone())
        }
        fn rollback_kv_to(&mut self, seq_pos: usize) -> Result<(), VerifierError> {
            self.rollbacks.push(seq_pos);
            Ok(())
        }
    }

    #[test]
    fn proposer_to_verifier_loop_byte_identity_at_k_zero() {
        // ADR-029 Phase 2 falsifier gate: at K=0 the loop degrades to
        // default decode. The proposer returns empty drafts; the verify
        // pass produces 1 logits row (the model's next token).
        let mut mock = MockVerifier { scripted: vec![one_hot(100, 7)], rollbacks: Vec::new() };
        let drafts: Vec<u32> = Vec::new();
        let logits = mock.verify(&[5u32]).unwrap();
        let (accept, tok) = accept_prefix(&drafts, &logits);
        assert_eq!(accept, 0);
        assert_eq!(tok, 7, "K=0 reduces to single-token decode");
    }

    #[test]
    fn proposer_to_verifier_loop_full_accept_advances_seq_pos_by_k_plus_1() {
        // Simulate one spec-decode cycle: drafts=[10,20,30], all match,
        // model emits 40. seq_pos advances by 4 (K+1).
        let drafts = vec![10u32, 20, 30];
        let scripted = vec![
            one_hot(100, 10),
            one_hot(100, 20),
            one_hot(100, 30),
            one_hot(100, 40),
        ];
        let mut mock = MockVerifier { scripted: scripted.clone(), rollbacks: Vec::new() };

        let seq_pos_before: usize = 7;
        let logits = mock.verify(&[5u32, 10, 20, 30]).unwrap();
        let (accept, tok) = accept_prefix(&drafts, &logits);
        let seq_pos_after = seq_pos_before + accept + 1;

        assert_eq!(accept, 3);
        assert_eq!(tok, 40);
        assert_eq!(seq_pos_after, 11, "K=3 full-accept advances seq_pos by K+1=4");

        // Verifier should be told to roll back to the new seq_pos —
        // since accept_count == drafts.len() the rollback is a no-op
        // (no rejected positions), but the integration calls it anyway
        // to keep the contract simple.
        mock.rollback_kv_to(seq_pos_after).unwrap();
        assert_eq!(mock.rollbacks, vec![11]);
    }

    #[test]
    fn proposer_to_verifier_loop_partial_accept_rolls_back_rejected() {
        // drafts=[10,20,30], 2 accepted + model emits 99 instead of 30.
        // seq_pos advances by 3 (accept_count + 1 = 2 + 1 = 3).
        // Rollback truncates KV past the 3rd new position.
        let drafts = vec![10u32, 20, 30];
        let scripted = vec![
            one_hot(100, 10),
            one_hot(100, 20),
            one_hot(100, 99),
            one_hot(100, 40),
        ];
        let mut mock = MockVerifier { scripted, rollbacks: Vec::new() };

        let seq_pos_before: usize = 7;
        let logits = mock.verify(&[5u32, 10, 20, 30]).unwrap();
        let (accept, tok) = accept_prefix(&drafts, &logits);
        let seq_pos_after = seq_pos_before + accept + 1;

        assert_eq!(accept, 2);
        assert_eq!(tok, 99);
        assert_eq!(seq_pos_after, 10);

        mock.rollback_kv_to(seq_pos_after).unwrap();
        assert_eq!(mock.rollbacks, vec![10],
                  "rollback to seq_pos_after = before + accept + 1");
    }
}
