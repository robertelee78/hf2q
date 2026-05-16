//! ADR-030 Phase 6 — Leviathan 2023 rejection sampling for temp > 0.
//!
//! Implements distribution-preserving speculative-sampling acceptance
//! per Leviathan et al. 2023 "Fast Inference from Transformers via
//! Speculative Decoding" (arxiv 2211.17192) §2.3.
//!
//! ## Rule
//!
//! For each drafted token `draft_i` at position `i`:
//! 1. Let `p_i = target_prob[i][draft_i]` (target softmax probability)
//! 2. Let `q_i = drafter_prob[i][draft_i]` (drafter softmax probability)
//! 3. Sample `u ~ Uniform[0, 1)`
//! 4. If `u < min(1, p_i / q_i)`: accept `draft_i`, continue to next i
//! 5. Else: REJECT — emit the model's continuation by sampling from
//!    the RESIDUAL distribution `max(0, p_v - q_v) / Z` where
//!    `Z = sum_v max(0, p_v - q_v)`. Stop drafting; emit this single
//!    replacement token as the round's free continuation.
//!
//! ## Why this preserves target's distribution exactly
//!
//! Per Leviathan §2.3 proof: at each position, the probability of
//! emitting token `t` after the accept-or-resample step equals
//! `p_t` (target's softmax), regardless of `q`. This is the property
//! greedy (argmax-vs-argmax) accept does NOT have at temp > 0 — naive
//! sampled-compare biases the output toward target's mode.
//!
//! ## Greedy (temp = 0) degeneration
//!
//! When the temperature feeding the softmax is 0, both `p` and `q`
//! become one-hot at the argmax. Then:
//! - If `argmax(p) == argmax(q) == draft_i`: p/q = 1, always accept.
//! - If `argmax(p) != argmax(q)`: but `draft_i = argmax(q)`, so
//!   `q_i = 1` and `p_i = 0`, so p/q = 0; always reject. Replacement
//!   is sampled from residual which equals one-hot at `argmax(p)`.
//! - Result: byte-identical to `accept_prefix_argmax`.
//!
//! So this function is the GENERALIZATION of greedy `accept_prefix_argmax`
//! that handles temp > 0 correctly.

use rand::Rng;

/// One Leviathan acceptance step for a single drafted position.
pub enum SampleStep {
    /// Draft accepted; continue to next position.
    Accept,
    /// Draft rejected; emit `replacement_token` and stop drafting.
    Reject { replacement_token: u32 },
}

/// Compute target probabilities from logprobs at a given temperature.
///
/// Returns a Vec<f32> of length `logprobs.len()`. NOT in-place; the
/// caller can reuse the buffer if performance matters.
///
/// Softmax with temperature:
/// ```text
///   p_v = exp(logprobs[v] / temp) / sum_v exp(logprobs[v] / temp)
/// ```
///
/// For numerical stability: subtract max before exp.
pub fn softmax_with_temp(logprobs: &[f32], temp: f32) -> Vec<f32> {
    assert!(temp > 0.0, "softmax_with_temp: temp must be > 0");
    let scale = 1.0 / temp;
    let max_scaled = logprobs
        .iter()
        .map(|&x| x * scale)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logprobs
        .iter()
        .map(|&x| (x * scale - max_scaled).exp())
        .collect();
    let z: f32 = probs.iter().sum();
    let inv_z = 1.0 / z;
    for p in probs.iter_mut() {
        *p *= inv_z;
    }
    probs
}

/// Leviathan 2023 §2.3 acceptance step for one drafted position.
///
/// # Arguments
///
/// - `draft_token`: the token the drafter proposed at this position
/// - `target_probs`: target's softmax distribution at this position
/// - `drafter_probs`: drafter's softmax distribution at this position
/// - `rng`: source of randomness for uniform sample + residual sampling
///
/// # Returns
///
/// `SampleStep::Accept` if the draft is accepted (continue drafting),
/// or `SampleStep::Reject{replacement_token}` if rejected (emit this
/// token as the round's free continuation; stop drafting).
pub fn leviathan_step(
    draft_token: u32,
    target_probs: &[f32],
    drafter_probs: &[f32],
    rng: &mut impl Rng,
) -> SampleStep {
    let vocab = target_probs.len();
    assert_eq!(
        drafter_probs.len(),
        vocab,
        "leviathan_step: target/drafter probs must have same vocab"
    );
    let v = draft_token as usize;
    assert!(v < vocab, "leviathan_step: draft_token {} out of vocab {}", v, vocab);

    let p = target_probs[v];
    let q = drafter_probs[v];

    // Acceptance probability: min(1, p/q). If q is 0, p/q is undefined;
    // in that pathological case treat as reject (drafter would never
    // have proposed this token at q=0, so we're already in residual
    // territory).
    let accept_prob = if q <= 0.0 { 0.0 } else { (p / q).min(1.0) };
    let u: f32 = rng.gen();
    if u < accept_prob {
        return SampleStep::Accept;
    }

    // Reject — sample replacement from residual `max(0, p_v - q_v)`.
    let mut residual: Vec<f32> = target_probs
        .iter()
        .zip(drafter_probs.iter())
        .map(|(&pv, &qv)| (pv - qv).max(0.0))
        .collect();
    let z: f32 = residual.iter().sum();
    if z <= 0.0 {
        // Pathological: residual is all-zero (target ⊆ drafter mass).
        // Per Leviathan, fall back to argmax(target).
        let argmax_t = target_probs
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(i_max, v_max), (i, &v)| {
                if v > v_max { (i, v) } else { (i_max, v_max) }
            })
            .0;
        return SampleStep::Reject { replacement_token: argmax_t as u32 };
    }

    // Normalize + sample
    let inv_z = 1.0 / z;
    for r in residual.iter_mut() {
        *r *= inv_z;
    }
    let u_resample: f32 = rng.gen();
    let mut acc = 0.0f32;
    for (i, &r) in residual.iter().enumerate() {
        acc += r;
        if u_resample < acc {
            return SampleStep::Reject { replacement_token: i as u32 };
        }
    }
    // Numerical edge: floats sum slightly < 1.0 due to rounding.
    // Fall through to last index.
    let last = residual.len() - 1;
    SampleStep::Reject { replacement_token: last as u32 }
}

/// Full multi-position Leviathan accept-prefix loop.
///
/// Iterates over `drafts`, calling `leviathan_step` at each position.
/// On accept: continue. On reject: emit replacement token + stop.
///
/// Returns `(accept_count, replacement_or_continuation)`:
/// - If any reject occurred: accept_count = position of reject;
///   replacement_or_continuation = the sampled residual token
/// - If all drafts accepted: accept_count = drafts.len();
///   replacement_or_continuation = a fresh sample from target_probs at
///   position `drafts.len()` (the "free" continuation)
///
/// # Arguments
///
/// - `drafts`: K drafted tokens
/// - `target_probs_per_pos`: target's softmax at K+1 positions
///   (positions 0..=K; position K is the post-last-draft prediction)
/// - `drafter_probs_per_pos`: drafter's softmax at K positions
///   (positions 0..K)
/// - `rng`: source of randomness
pub fn leviathan_accept_prefix(
    drafts: &[u32],
    target_probs_per_pos: &[Vec<f32>],
    drafter_probs_per_pos: &[Vec<f32>],
    rng: &mut impl Rng,
) -> (usize, u32) {
    assert_eq!(
        drafter_probs_per_pos.len(),
        drafts.len(),
        "leviathan_accept_prefix: drafter_probs must have len = drafts.len()"
    );
    assert_eq!(
        target_probs_per_pos.len(),
        drafts.len() + 1,
        "leviathan_accept_prefix: target_probs must have len = drafts.len() + 1"
    );

    for (i, &draft) in drafts.iter().enumerate() {
        let step = leviathan_step(
            draft,
            &target_probs_per_pos[i],
            &drafter_probs_per_pos[i],
            rng,
        );
        if let SampleStep::Reject { replacement_token } = step {
            return (i, replacement_token);
        }
    }
    // All drafts accepted — sample free continuation from target at
    // position drafts.len() (the post-last-draft prediction).
    let target_last = &target_probs_per_pos[drafts.len()];
    let u: f32 = rng.gen();
    let mut acc = 0.0f32;
    for (i, &p) in target_last.iter().enumerate() {
        acc += p;
        if u < acc {
            return (drafts.len(), i as u32);
        }
    }
    let last = target_last.len() - 1;
    (drafts.len(), last as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn softmax_with_temp_normalizes() {
        let logprobs = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax_with_temp(&logprobs, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1; got {sum}");
        // High logprob → higher probability
        assert!(probs[3] > probs[0]);
    }

    #[test]
    fn softmax_temp_zero_panics_via_assert() {
        // temp must be > 0 (we panic on temp == 0 since the limit is
        // argmax). Test that this is enforced.
        let result = std::panic::catch_unwind(|| {
            softmax_with_temp(&[1.0, 2.0, 3.0], 0.0)
        });
        assert!(result.is_err(), "temp=0 should panic via assert");
    }

    #[test]
    fn leviathan_step_accepts_when_target_dominates() {
        // p[draft] = 0.9, q[draft] = 0.1 → p/q = 9, always accept
        let target_probs = vec![0.05, 0.9, 0.05];
        let drafter_probs = vec![0.45, 0.1, 0.45];
        let mut rng = StdRng::seed_from_u64(42);
        // 100 trials, all should accept
        for _ in 0..100 {
            match leviathan_step(1, &target_probs, &drafter_probs, &mut rng) {
                SampleStep::Accept => {}
                SampleStep::Reject { .. } => panic!("should always accept when p >> q"),
            }
        }
    }

    #[test]
    fn leviathan_step_rejects_when_drafter_dominates() {
        // p[draft] = 0.05, q[draft] = 0.95 → p/q ≈ 0.053, rejects 95% of the time
        let target_probs = vec![0.475, 0.05, 0.475];
        let drafter_probs = vec![0.025, 0.95, 0.025];
        let mut rng = StdRng::seed_from_u64(42);
        let mut rejects = 0;
        for _ in 0..1000 {
            match leviathan_step(1, &target_probs, &drafter_probs, &mut rng) {
                SampleStep::Accept => {}
                SampleStep::Reject { .. } => rejects += 1,
            }
        }
        // Expect ~95% rejects; allow [80%, 99%] for statistical wiggle
        assert!(
            (800..=990).contains(&rejects),
            "expected ~950 rejects, got {rejects}"
        );
    }

    #[test]
    fn leviathan_step_residual_replacement_is_correct_token() {
        // p = [0.4, 0.1, 0.5], q = [0.1, 0.8, 0.1]
        // residual = [max(0, 0.4-0.1), max(0, 0.1-0.8), max(0, 0.5-0.1)]
        //          = [0.3, 0, 0.4]  (token 1 zeroed)
        // Normalized: [3/7, 0, 4/7]
        // Drafter proposed token 1 (q = 0.8 vs p = 0.1, will reject often).
        // Replacement is sampled from residual; cannot be token 1.
        let target_probs = vec![0.4, 0.1, 0.5];
        let drafter_probs = vec![0.1, 0.8, 0.1];
        let mut rng = StdRng::seed_from_u64(42);
        let mut sample_counts = [0u32; 3];
        let mut rejects = 0;
        for _ in 0..2000 {
            match leviathan_step(1, &target_probs, &drafter_probs, &mut rng) {
                SampleStep::Accept => {}
                SampleStep::Reject { replacement_token } => {
                    rejects += 1;
                    sample_counts[replacement_token as usize] += 1;
                }
            }
        }
        // Token 1 in residual is 0, so should NEVER be the replacement.
        assert_eq!(sample_counts[1], 0, "residual at token 1 was 0");
        // Expect roughly 3:4 ratio of token 0 vs token 2 among rejects
        // Probability ratio: 3/7 ≈ 0.429 vs 4/7 ≈ 0.571
        let total_resampled = sample_counts[0] + sample_counts[2];
        assert!(total_resampled > 100, "should have many rejects: {rejects}");
        let p0 = sample_counts[0] as f32 / total_resampled as f32;
        assert!(
            (0.35..=0.50).contains(&p0),
            "expected token 0 fraction ≈ 0.43, got {p0}"
        );
    }

    #[test]
    fn leviathan_accept_prefix_full_accept_returns_continuation() {
        // 2 drafts, both p >> q → always accept, then sample free continuation
        let drafts = vec![1u32, 2u32];
        let target_probs = vec![
            vec![0.1, 0.8, 0.1],
            vec![0.1, 0.1, 0.8],
            vec![0.5, 0.3, 0.2],
        ];
        let drafter_probs = vec![
            vec![0.45, 0.1, 0.45],
            vec![0.45, 0.45, 0.1],
        ];
        let mut rng = StdRng::seed_from_u64(42);
        let (accept_count, continuation) = leviathan_accept_prefix(
            &drafts, &target_probs, &drafter_probs, &mut rng,
        );
        assert_eq!(accept_count, 2, "all drafts accepted");
        // Continuation is sampled from target_probs[2] = [0.5, 0.3, 0.2]
        assert!(continuation < 3, "continuation in vocab");
    }

    #[test]
    fn leviathan_accept_prefix_partial_reject_truncates() {
        // 2 drafts, first accepts, second rejects → accept_count = 1
        let drafts = vec![1u32, 1u32];
        // pos 0: drafter dominates draft (q=0.8, p=0.8) → likely accept
        // pos 1: drafter dominates draft heavily, target prefers token 0
        //         (q[1]=0.95, p[1]=0.05) → likely reject
        let target_probs = vec![
            vec![0.1, 0.8, 0.1],
            vec![0.475, 0.05, 0.475],
            vec![0.5, 0.3, 0.2],
        ];
        let drafter_probs = vec![
            vec![0.1, 0.8, 0.1],
            vec![0.025, 0.95, 0.025],
        ];
        // Trial multiple seeds to verify partial-reject is achievable
        let mut saw_partial = false;
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (accept_count, _) = leviathan_accept_prefix(
                &drafts, &target_probs, &drafter_probs, &mut rng,
            );
            if accept_count == 1 {
                saw_partial = true;
                break;
            }
        }
        assert!(saw_partial, "should have seen partial-accept across seeds");
    }
}
