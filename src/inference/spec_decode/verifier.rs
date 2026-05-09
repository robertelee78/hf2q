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

/// Greedy variant of `accept_prefix` that takes per-position argmaxes
/// (Vec<u32>) instead of full logits. ADR-028 iter-123 Shape S.
///
/// Used by `forward_decode_verify_serial`: forward_decode already
/// returns argmax (no need to materialize full logits and re-argmax).
///
/// Returns `(accept_count, model_token)` matching `accept_prefix` semantics:
/// - `accept_count` = leading `drafts[i] == model_argmaxes[i]` matches.
/// - `model_token` = `model_argmaxes[accept_count]` (the "free" extra token).
///
/// Contract identical to `accept_prefix(drafts, &one_hot_at_each(argmaxes))`
/// but skips the O(K × vocab) one-hot allocation.
pub fn accept_prefix_argmax(drafts: &[u32], model_argmaxes: &[u32]) -> (usize, u32) {
    if model_argmaxes.is_empty() {
        return (0, 0);
    }
    let mut accept_count = 0;
    for i in 0..drafts.len() {
        if i >= model_argmaxes.len() { break; }
        if model_argmaxes[i] == drafts[i] {
            accept_count += 1;
        } else {
            return (accept_count, model_argmaxes[i]);
        }
    }
    if accept_count < model_argmaxes.len() {
        (accept_count, model_argmaxes[accept_count])
    } else {
        (accept_count, *model_argmaxes.last().unwrap())
    }
}

/// Pure KV-rollback math. ADR-028 iter-123 Shape S contract.
///
/// Given a per-layer KV-cache cursor `(write_pos, seq_len)` with
/// `capacity` slots and `is_sliding` mode, compute the new cursor
/// after rolling back `trim` writes.
///
/// - **Full attention** (`is_sliding=false`): `write_pos` is monotonic
///   from 0; rollback subtracts. `seq_len` decreases by the same amount.
/// - **Sliding window** (`is_sliding=true`): `write_pos` is modulo
///   `capacity`; rollback steps back with wrap-around. `seq_len` still
///   decreases monotonically (caller invariant: `seq_len ≤ capacity`).
///
/// `trim` is clamped to `seq_len` — over-rollback is a no-op past 0.
pub fn rollback_kv_state(
    write_pos: usize,
    seq_len: usize,
    capacity: usize,
    is_sliding: bool,
    trim: usize,
) -> (usize, usize) {
    let trim = trim.min(seq_len);
    let new_seq_len = seq_len - trim;
    let new_write_pos = if is_sliding {
        if capacity == 0 {
            0
        } else {
            // Wrap with: new = (old - trim mod cap + cap) mod cap.
            (write_pos + capacity - (trim % capacity)) % capacity
        }
    } else {
        write_pos.saturating_sub(trim)
    };
    (new_write_pos, new_seq_len)
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

    /// Mock verifier — drives the spec-decode loop without a model.
    /// Validates every `verify` call's input shape (`tokens.len() ==
    /// expected_input_len`) so a buggy caller is caught immediately,
    /// and records both the inputs it was called with and the
    /// rollback positions it received. Multi-cycle tests can supply
    /// a different `scripted` vector per cycle via `set_scripted`.
    struct MockVerifier {
        scripted: VerifyLogits,
        /// If `Some(n)`, asserts that every verify call gets exactly
        /// `n` input tokens. Catches "K+1 vs K" off-by-ones in callers.
        expected_input_len: Option<usize>,
        verify_inputs: Vec<Vec<u32>>,
        rollbacks: Vec<usize>,
    }

    impl MockVerifier {
        fn new(scripted: VerifyLogits) -> Self {
            Self {
                scripted,
                expected_input_len: None,
                verify_inputs: Vec::new(),
                rollbacks: Vec::new(),
            }
        }
        fn with_expected_input_len(mut self, n: usize) -> Self {
            self.expected_input_len = Some(n);
            self
        }
    }

    impl Verifier for MockVerifier {
        fn verify(&mut self, tokens: &[u32]) -> Result<VerifyLogits, VerifierError> {
            if tokens.is_empty() { return Err(VerifierError::EmptyInput); }
            if let Some(n) = self.expected_input_len {
                assert_eq!(tokens.len(), n,
                    "MockVerifier: expected {n} input tokens, got {}", tokens.len());
            }
            self.verify_inputs.push(tokens.to_vec());
            Ok(self.scripted.clone())
        }
        fn rollback_kv_to(&mut self, seq_pos: usize) -> Result<(), VerifierError> {
            self.rollbacks.push(seq_pos);
            Ok(())
        }
    }

    /// "Ground-truth" model: deterministic argmax-of-(token_seq) →
    /// next_token, used to simulate what default decode WOULD produce
    /// so spec-decode's output can be compared byte-for-byte.
    /// Algorithm: next = (sum(seq) * 31 + seq.last()) % vocab — a
    /// deterministic non-trivial function of the prefix.
    fn ground_truth_next(seq: &[u32], vocab: u32) -> u32 {
        let s: u64 = seq.iter().map(|&t| t as u64).sum();
        let last = *seq.last().unwrap_or(&0) as u64;
        ((s.wrapping_mul(31).wrapping_add(last)) % vocab as u64) as u32
    }

    /// Verifier that *consults the ground-truth model* for each
    /// position. Simulates a perfect verify pass: at each position i
    /// of `[T_t, draft_1..draft_K]`, returns one_hot(ground_truth_next(
    /// prefix_up_to_position_i)). Used to prove byte-identity between
    /// spec-decode and default decode.
    struct GroundTruthVerifier {
        vocab: u32,
        prefix: Vec<u32>,
        rollbacks: Vec<usize>,
    }

    impl GroundTruthVerifier {
        fn new(vocab: u32, initial_prefix: Vec<u32>) -> Self {
            Self { vocab, prefix: initial_prefix, rollbacks: Vec::new() }
        }
    }

    impl Verifier for GroundTruthVerifier {
        fn verify(&mut self, tokens: &[u32]) -> Result<VerifyLogits, VerifierError> {
            // Simulate the model receiving `tokens` past `prefix` and
            // emitting per-position predictions. Position i predicts
            // the token AFTER processing tokens[0..=i].
            let mut logits = Vec::with_capacity(tokens.len());
            for i in 0..tokens.len() {
                let mut seq = self.prefix.clone();
                // tokens[0] is the current verified position (already
                // in prefix conceptually for the first call); for i=0
                // we predict what follows it. For i>0 we extend with
                // the speculative drafts.
                if i > 0 { seq.extend_from_slice(&tokens[1..=i]); }
                let next = ground_truth_next(&seq, self.vocab);
                logits.push(one_hot(self.vocab as usize, next));
            }
            Ok(logits)
        }
        fn rollback_kv_to(&mut self, seq_pos: usize) -> Result<(), VerifierError> {
            // Truncate prefix to seq_pos.
            self.prefix.truncate(seq_pos);
            self.rollbacks.push(seq_pos);
            Ok(())
        }
    }

    /// Run the default (non-spec) decode loop using `ground_truth_next`
    /// as the oracle. Returns the generated sequence after `n_tokens`
    /// new tokens (excluding the prompt).
    fn default_decode(prompt: &[u32], vocab: u32, n_tokens: usize) -> Vec<u32> {
        let mut gen = prompt.to_vec();
        for _ in 0..n_tokens {
            let next = ground_truth_next(&gen, vocab);
            gen.push(next);
        }
        gen
    }

    /// Run a simulated spec-decode loop using `GroundTruthVerifier`
    /// and the n-gram proposer. Returns the generated sequence after
    /// at least `n_tokens` new tokens.
    fn spec_decode_loop(
        prompt: &[u32],
        vocab: u32,
        n_tokens: usize,
        cfg: &super::super::ngram_proposer::NgramConfig,
    ) -> (Vec<u32>, GroundTruthVerifier) {
        let mut gen = prompt.to_vec();
        let mut verifier = GroundTruthVerifier::new(vocab, prompt.to_vec());
        let target_len = prompt.len() + n_tokens;

        while gen.len() < target_len {
            let drafts = super::super::ngram_proposer::propose(&gen, cfg);
            // Build verify input: [last verified token] ++ drafts.
            let last = *gen.last().unwrap();
            let mut input = vec![last];
            input.extend_from_slice(&drafts);
            let logits = verifier.verify(&input).unwrap();
            let (accept, model_tok) = accept_prefix(&drafts, &logits);
            gen.extend_from_slice(&drafts[..accept]);
            gen.push(model_tok);
            verifier.prefix = gen.clone();
            verifier.rollback_kv_to(gen.len()).unwrap();
            if gen.len() >= target_len + cfg.k {
                break; // safety bound
            }
        }
        (gen, verifier)
    }

    #[test]
    fn spec_decode_byte_identity_vs_default_decode() {
        // ADR-029 PHASE 2 ACCEPTANCE GATE: the spec-decode loop driven
        // by the n-gram proposer + a ground-truth verifier MUST produce
        // a byte-identical generated sequence to default decode. This
        // is the fundamental correctness invariant of greedy spec
        // decode (Leviathan et al. 2023): accepted prefix is exactly
        // what default decode would produce; the K+1th model token is
        // the next token default decode would emit.
        let prompt = vec![1u32, 2, 3, 1, 2, 3, 4]; // has [1,2,3] repetition for proposer
        let vocab = 256u32;
        let n_tokens = 30;
        let cfg = super::super::ngram_proposer::NgramConfig {
            min_ngram: 1, max_ngram: 3, k: 3, max_model_len: 4096,
        };

        let default_out = default_decode(&prompt, vocab, n_tokens);
        let (spec_out, _v) = spec_decode_loop(&prompt, vocab, n_tokens, &cfg);

        // Spec output may overshoot by up to K, so compare prefixes.
        let cmp_len = prompt.len() + n_tokens;
        assert!(spec_out.len() >= cmp_len);
        assert_eq!(
            &spec_out[..cmp_len], &default_out[..cmp_len],
            "spec-decode must be byte-identical to default decode under greedy"
        );
    }

    #[test]
    fn spec_decode_at_k_zero_calls_verifier_with_single_token() {
        // K=0 forces drafts = []; verify input = [last] (length 1).
        // Loop runs once per output token, exactly like default decode.
        let prompt = vec![5u32, 6, 7];
        let vocab = 100u32;
        let cfg = super::super::ngram_proposer::NgramConfig {
            min_ngram: 1, max_ngram: 3, k: 0, max_model_len: 4096,
        };

        let mut verifier = GroundTruthVerifier::new(vocab, prompt.clone());
        let mut gen = prompt.clone();

        for _ in 0..5 {
            let drafts = super::super::ngram_proposer::propose(&gen, &cfg);
            assert!(drafts.is_empty(), "K=0 must always return empty drafts");
            let last = *gen.last().unwrap();
            let logits = verifier.verify(&[last]).unwrap();
            assert_eq!(logits.len(), 1, "K=0 verify produces 1 logits row");
            let (accept, tok) = accept_prefix(&drafts, &logits);
            assert_eq!(accept, 0);
            gen.push(tok);
            verifier.prefix = gen.clone();
            verifier.rollback_kv_to(gen.len()).unwrap();
        }

        // Compare to default decode for byte-identity.
        let default_out = default_decode(&prompt, vocab, 5);
        assert_eq!(gen, default_out);
    }

    #[test]
    fn accept_prefix_invariants_under_random_inputs() {
        // Property test: for random drafts + logits, the returned
        // (accept_count, model_token) MUST satisfy:
        //   1. accept_count <= drafts.len()
        //   2. accept_count <= logits.len()
        //   3. for all i < accept_count: drafts[i] == argmax(logits[i])
        //   4. if accept_count < drafts.len(): drafts[accept_count] !=
        //      argmax(logits[accept_count])
        //   5. model_token is some valid token id (< vocab)
        let vocab = 50usize;
        let mut state: u64 = 0xCAFE_BEEF;
        let next_rand = |s: &mut u64| -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s >> 33
        };

        for _ in 0..500 {
            let n_drafts = (next_rand(&mut state) % 6) as usize;
            let n_logits = (next_rand(&mut state) % 8) as usize;
            let drafts: Vec<u32> = (0..n_drafts)
                .map(|_| (next_rand(&mut state) % vocab as u64) as u32).collect();
            let logits: VerifyLogits = (0..n_logits)
                .map(|_| {
                    let target = (next_rand(&mut state) % vocab as u64) as u32;
                    one_hot(vocab, target)
                }).collect();

            let (accept, tok) = accept_prefix(&drafts, &logits);

            // Invariant 1, 2.
            assert!(accept <= drafts.len(), "accept_count > drafts.len()");
            assert!(accept <= logits.len() || logits.is_empty(),
                "accept_count > logits.len() (got {accept} vs {})", logits.len());

            // Invariant 3.
            for i in 0..accept {
                assert_eq!(drafts[i], argmax_u32(&logits[i]),
                    "accepted draft[{i}] = {} doesn't match argmax {} (logits len {}, drafts len {})",
                    drafts[i], argmax_u32(&logits[i]), logits.len(), drafts.len());
            }

            // Invariant 4.
            if accept < drafts.len() && accept < logits.len() {
                assert_ne!(drafts[accept], argmax_u32(&logits[accept]),
                    "first rejected draft equals model argmax — should have been accepted");
            }

            // Invariant 5.
            assert!((tok as usize) < vocab,
                "model_token {tok} out of vocab range {vocab}");
        }
    }

    #[test]
    fn spec_decode_loop_full_accept_advances_seq_pos_by_k_plus_1() {
        // Simulate one spec-decode cycle: drafts=[10,20,30], all match,
        // model emits 40. seq_pos advances by K+1=4.
        let drafts = vec![10u32, 20, 30];
        let scripted = vec![
            one_hot(100, 10),
            one_hot(100, 20),
            one_hot(100, 30),
            one_hot(100, 40),
        ];
        let mut mock = MockVerifier::new(scripted)
            .with_expected_input_len(4); // [last] ++ drafts = 1 + 3 = 4

        let seq_pos_before: usize = 7;
        let last = 5u32;
        let mut input = vec![last];
        input.extend_from_slice(&drafts);
        let logits = mock.verify(&input).unwrap();
        let (accept, tok) = accept_prefix(&drafts, &logits);
        let seq_pos_after = seq_pos_before + accept + 1;

        assert_eq!(accept, 3);
        assert_eq!(tok, 40);
        assert_eq!(seq_pos_after, 11);
        assert_eq!(mock.verify_inputs, vec![vec![5u32, 10, 20, 30]],
            "verifier saw [last] ++ drafts in correct order");
        mock.rollback_kv_to(seq_pos_after).unwrap();
        assert_eq!(mock.rollbacks, vec![11]);
    }

    #[test]
    fn spec_decode_loop_partial_accept_rolls_back_rejected() {
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
        let mut mock = MockVerifier::new(scripted)
            .with_expected_input_len(4); // [last] ++ K=3 drafts

        let seq_pos_before: usize = 7;
        let logits = mock.verify(&[5u32, 10, 20, 30]).unwrap();
        let (accept, tok) = accept_prefix(&drafts, &logits);
        let seq_pos_after = seq_pos_before + accept + 1;

        assert_eq!(accept, 2);
        assert_eq!(tok, 99);
        assert_eq!(seq_pos_after, 10);
        assert_eq!(mock.verify_inputs, vec![vec![5u32, 10, 20, 30]]);

        mock.rollback_kv_to(seq_pos_after).unwrap();
        assert_eq!(mock.rollbacks, vec![10],
                  "rollback to seq_pos_after = before + accept + 1");
    }

    #[test]
    fn mock_verifier_rejects_empty_input_per_contract() {
        // Verifier's contract requires at least 1 input token. Mock
        // mirrors the real impl's constraint so callers are caught.
        let mut mock = MockVerifier::new(vec![one_hot(100, 0)]);
        let result = mock.verify(&[]);
        assert!(matches!(result, Err(VerifierError::EmptyInput)));
    }

    #[test]
    fn mock_verifier_input_len_validation_catches_caller_bug() {
        // Set expected_input_len=4 (K=3 + last). Then call with wrong
        // number of tokens — should panic, catching the caller bug.
        let mut mock = MockVerifier::new(vec![one_hot(100, 0)])
            .with_expected_input_len(4);
        // Calling with 3 tokens (K=2 + last) violates the expectation.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = mock.verify(&[1u32, 2, 3]);
        }));
        assert!(result.is_err(),
            "MockVerifier with expected_input_len=4 should panic on 3 tokens");
    }

    #[test]
    fn ground_truth_decode_is_deterministic() {
        // Sanity check: ground_truth_next gives same answer for same
        // input across multiple calls (no internal state).
        let seq = vec![1u32, 2, 3, 4, 5];
        let a = ground_truth_next(&seq, 256);
        let b = ground_truth_next(&seq, 256);
        let c = ground_truth_next(&seq, 256);
        assert_eq!(a, b);
        assert_eq!(b, c);
        // And different prefixes give different outputs (catches bugs
        // where ground_truth_next ignores the input).
        let d = ground_truth_next(&[1u32, 2, 3, 4, 6], 256);
        assert_ne!(a, d, "ground_truth_next must be input-sensitive");
    }

    // ===== ADR-028 iter-123 Shape S — accept_prefix_argmax + rollback_kv_state =====

    #[test]
    fn accept_prefix_argmax_full_accept() {
        let drafts = vec![10u32, 20, 30];
        let argmaxes = vec![10u32, 20, 30, 40];
        let (accept, tok) = accept_prefix_argmax(&drafts, &argmaxes);
        assert_eq!(accept, 3);
        assert_eq!(tok, 40);
    }

    #[test]
    fn accept_prefix_argmax_partial_accept() {
        let drafts = vec![10u32, 20, 30];
        let argmaxes = vec![10u32, 20, 99, 40];
        let (accept, tok) = accept_prefix_argmax(&drafts, &argmaxes);
        assert_eq!(accept, 2);
        assert_eq!(tok, 99);
    }

    #[test]
    fn accept_prefix_argmax_zero_accept() {
        let drafts = vec![10u32, 20, 30];
        let argmaxes = vec![99u32, 20, 30, 40];
        let (accept, tok) = accept_prefix_argmax(&drafts, &argmaxes);
        assert_eq!(accept, 0);
        assert_eq!(tok, 99);
    }

    #[test]
    fn accept_prefix_argmax_matches_logits_variant() {
        // The argmax variant must agree with the logits variant on
        // the same problem.
        let drafts = vec![10u32, 20, 30];
        let argmaxes = vec![10u32, 20, 99, 40];
        let logits: VerifyLogits = argmaxes.iter().map(|&t| one_hot(100, t)).collect();
        let (a1, t1) = accept_prefix(&drafts, &logits);
        let (a2, t2) = accept_prefix_argmax(&drafts, &argmaxes);
        assert_eq!((a1, t1), (a2, t2));
    }

    #[test]
    fn accept_prefix_argmax_empty() {
        let (a, t) = accept_prefix_argmax(&[], &[]);
        assert_eq!((a, t), (0, 0));
    }

    #[test]
    fn rollback_full_attention_subtracts() {
        // capacity=4096, write_pos=100, seq_len=100, trim 3.
        let (wp, sl) = rollback_kv_state(100, 100, 4096, false, 3);
        assert_eq!((wp, sl), (97, 97));
    }

    #[test]
    fn rollback_full_attention_zero_trim() {
        let (wp, sl) = rollback_kv_state(100, 100, 4096, false, 0);
        assert_eq!((wp, sl), (100, 100));
    }

    #[test]
    fn rollback_full_attention_clamps_at_zero() {
        // Rollback past 0 → clamped (saturating_sub). seq_len caps trim.
        let (wp, sl) = rollback_kv_state(100, 100, 4096, false, 200);
        assert_eq!((wp, sl), (0, 0));
    }

    #[test]
    fn rollback_sliding_wraps_no_wrap() {
        // wp=10 trim=3, no wrap needed. cap=100.
        let (wp, sl) = rollback_kv_state(10, 50, 100, true, 3);
        assert_eq!((wp, sl), (7, 47));
    }

    #[test]
    fn rollback_sliding_wraps_through_zero() {
        // wp=2, trim=3, cap=10. (2 + 10 - 3) % 10 = 9.
        let (wp, sl) = rollback_kv_state(2, 10, 10, true, 3);
        assert_eq!((wp, sl), (9, 7));
    }

    #[test]
    fn rollback_sliding_wraps_full_circle() {
        // trim equal to capacity → write_pos unchanged (full wrap),
        // but seq_len capped to 0.
        let (wp, sl) = rollback_kv_state(2, 10, 10, true, 10);
        // (2 + 10 - 0) % 10 = 2  (trim%cap = 0).
        assert_eq!((wp, sl), (2, 0));
    }

    #[test]
    fn rollback_sliding_zero_capacity_safe() {
        // Defensive: capacity=0 should not divide-by-zero.
        let (wp, sl) = rollback_kv_state(0, 0, 0, true, 5);
        assert_eq!((wp, sl), (0, 0));
    }

    #[test]
    fn rollback_invariant_seq_len_le_capacity() {
        // Property: post-rollback seq_len never exceeds capacity.
        for cap in [1usize, 8, 64, 256, 1024] {
            for sl in 0..=cap {
                for wp in 0..=cap.saturating_sub(1).max(0) {
                    for is_sliding in [false, true] {
                        for trim in [0usize, 1, sl, sl/2, sl+1] {
                            let (_nwp, nsl) = rollback_kv_state(wp, sl, cap, is_sliding, trim);
                            assert!(nsl <= cap, "seq_len > cap: cap={cap} wp={wp} sl={sl} sliding={is_sliding} trim={trim} → nsl={nsl}");
                            assert!(nsl <= sl, "seq_len grew: cap={cap} wp={wp} sl={sl} trim={trim} → nsl={nsl}");
                        }
                    }
                }
            }
        }
    }
}
