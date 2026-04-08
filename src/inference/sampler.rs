//! Token sampling with configurable strategies.
//!
//! Implements the standard sampling pipeline:
//! 1. Repetition penalty (penalize already-generated tokens)
//! 2. Temperature scaling (control randomness)
//! 3. Top-k filtering (keep only k highest logit tokens)
//! 4. Top-p nucleus sampling (keep smallest set with cumulative prob >= p)
//! 5. Categorical sample from the resulting distribution
//!
//! When temperature is 0 (or very close), greedy argmax is used instead.

use rand::Rng;

/// Configuration for the sampling pipeline.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature for softmax scaling. 0.0 = greedy argmax.
    pub temperature: f32,
    /// Top-p (nucleus) threshold. 1.0 = disabled.
    pub top_p: f32,
    /// Top-k count. 0 = disabled.
    pub top_k: usize,
    /// Repetition penalty factor. 1.0 = disabled.
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
        }
    }
}

/// Token sampler that applies the full sampling pipeline.
pub struct Sampler {
    config: SamplerConfig,
    rng: rand::rngs::ThreadRng,
}

impl Sampler {
    /// Create a new sampler with the given configuration.
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            config,
            rng: rand::thread_rng(),
        }
    }

    /// Sample the next token from logits, given the tokens generated so far.
    ///
    /// Applies the full pipeline: repetition penalty -> temperature -> top-k -> top-p -> sample.
    pub fn sample_next(&mut self, logits: &[f32], generated_ids: &[u32]) -> u32 {
        let mut working_logits = logits.to_vec();

        // Step 1: Repetition penalty
        if self.config.repetition_penalty != 1.0 {
            apply_repetition_penalty(
                &mut working_logits,
                generated_ids,
                self.config.repetition_penalty,
            );
        }

        // Step 2: Temperature scaling
        // If temperature is ~0, just do greedy argmax
        if self.config.temperature < 1e-7 {
            return argmax(&working_logits);
        }
        if self.config.temperature != 1.0 {
            apply_temperature(&mut working_logits, self.config.temperature);
        }

        // Step 3: Top-k filtering
        if self.config.top_k > 0 {
            apply_top_k(&mut working_logits, self.config.top_k);
        }

        // Step 4: Top-p nucleus filtering
        if self.config.top_p < 1.0 {
            apply_top_p(&mut working_logits, self.config.top_p);
        }

        // Step 5: Compute softmax ONCE, then sample categorically from probabilities
        let probs = softmax(&working_logits);
        sample_categorical_from_probs(&mut self.rng, &probs, &working_logits)
    }
}

/// Apply repetition penalty to logits.
///
/// For each token that appears in `generated_ids`, divide its logit by `penalty`
/// if the logit is positive, or multiply by `penalty` if negative. This ensures
/// the penalty always reduces the probability of repeated tokens.
fn apply_repetition_penalty(logits: &mut [f32], generated_ids: &[u32], penalty: f32) {
    for &id in generated_ids {
        let idx = id as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Apply temperature scaling to logits (divide all by temperature).
fn apply_temperature(logits: &mut [f32], temperature: f32) {
    for v in logits.iter_mut() {
        *v /= temperature;
    }
}

/// Apply top-k filtering: set all logits outside the top k to -infinity.
///
/// Uses `select_nth_unstable_by` for O(V) average-case partitioning instead of
/// O(V log V) full sort. For V=262,144 this saves roughly 4x comparisons.
fn apply_top_k(logits: &mut [f32], k: usize) {
    if k >= logits.len() {
        return;
    }

    // Build an indexed copy so we can find the k-th largest without sorting fully.
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();

    // O(V) partial sort: after this call, indexed[k-1] holds the element that
    // would be at position k-1 in descending order, and indexed[0..k] contains
    // (unordered) the top-k elements.
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let threshold = indexed[k - 1].1;

    // Mask everything below the threshold.
    // In case of ties at the boundary, keep all tokens with logit >= threshold.
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Apply top-p (nucleus) filtering.
///
/// Sort tokens by descending probability, accumulate until cumulative
/// probability exceeds p, then mask the rest to -infinity.
///
/// Computes softmax internally; the caller in `sample_next` will recompute
/// softmax once afterwards on the filtered logits so probabilities remain
/// consistent with whatever tokens survived.
fn apply_top_p(logits: &mut [f32], p: f32) {
    // Compute softmax probabilities for sorting decisions only.
    // A second softmax is computed later on the *filtered* logits to yield
    // the final sampling distribution — this ensures masked tokens have
    // exactly probability 0 regardless of floating-point rounding here.
    let probs = softmax(logits);

    // Sort by descending probability, tracking original indices
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Accumulate and find cutoff
    let mut cumulative = 0.0f32;
    let mut keep_set = vec![false; logits.len()];

    for &(idx, prob) in &indexed {
        keep_set[idx] = true;
        cumulative += prob;
        if cumulative >= p {
            break;
        }
    }

    // Mask tokens not in the keep set
    for (i, v) in logits.iter_mut().enumerate() {
        if !keep_set[i] {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Compute softmax of logits, returning a probability distribution.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    // Numerical stability: subtract max
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_sum: f32 = logits
        .iter()
        .map(|&v| {
            if v == f32::NEG_INFINITY {
                0.0
            } else {
                (v - max_logit).exp()
            }
        })
        .sum();

    if exp_sum == 0.0 {
        // All logits are -inf; return uniform over valid entries as fallback
        let n = logits.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
        if n == 0 {
            return vec![0.0; logits.len()];
        }
        let p = 1.0 / n as f32;
        return logits
            .iter()
            .map(|&v| if v == f32::NEG_INFINITY { 0.0 } else { p })
            .collect();
    }

    logits
        .iter()
        .map(|&v| {
            if v == f32::NEG_INFINITY {
                0.0
            } else {
                (v - max_logit).exp() / exp_sum
            }
        })
        .collect()
}

/// Sample from a categorical distribution given pre-computed probabilities.
///
/// `probs` must already be a softmax output (non-negative, sums to ~1).
/// `logits` is used only as a fallback for argmax when all probs are ~0.
///
/// This is the hot path called by `sample_next`; softmax is computed exactly
/// once upstream and passed in here to avoid redundant work.
fn sample_categorical_from_probs(rng: &mut impl Rng, probs: &[f32], logits: &[f32]) -> u32 {
    // Build cumulative distribution
    let mut cumulative = Vec::with_capacity(probs.len());
    let mut sum = 0.0f32;
    for &p in probs {
        sum += p;
        cumulative.push(sum);
    }

    // Handle edge case: if total probability is ~0, fall back to argmax of original logits
    if sum < 1e-10 {
        return argmax(logits);
    }

    // Normalize the cumulative sum in case of floating-point drift
    if sum != 1.0 {
        for v in cumulative.iter_mut() {
            *v /= sum;
        }
    }

    let u: f32 = rng.gen();
    for (i, &c) in cumulative.iter().enumerate() {
        if u < c {
            return i as u32;
        }
    }

    // Fallback: return the last token
    (probs.len() - 1) as u32
}

/// Sample from a categorical distribution defined by logits (not probabilities).
///
/// Converts logits to probabilities via softmax, then samples proportionally.
/// Kept for direct use in tests.
fn sample_categorical(rng: &mut impl Rng, logits: &[f32]) -> u32 {
    let probs = softmax(logits);
    sample_categorical_from_probs(rng, &probs, logits)
}

/// Return the index of the maximum value (greedy argmax).
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling_temperature_zero() {
        let config = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);

        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let token = sampler.sample_next(&logits, &[]);
        assert_eq!(token, 1); // index of highest logit

        // Should be deterministic
        let token2 = sampler.sample_next(&logits, &[]);
        assert_eq!(token2, 1);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[0.0, 0.0, 1.0]), 2);
    }

    #[test]
    fn test_repetition_penalty_positive_logits() {
        let mut logits = vec![2.0, 4.0, 1.0, 3.0];
        apply_repetition_penalty(&mut logits, &[1, 3], 2.0);
        // Token 1: 4.0 / 2.0 = 2.0
        // Token 3: 3.0 / 2.0 = 1.5
        assert!((logits[0] - 2.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 1.0).abs() < 1e-6);
        assert!((logits[3] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let mut logits = vec![-2.0, -4.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 1.5);
        // Negative logits get multiplied (making them more negative)
        assert!((logits[0] - (-3.0)).abs() < 1e-6);
        assert!((logits[1] - (-6.0)).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_disabled() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_temperature_scaling() {
        let mut logits = vec![2.0, 4.0, 6.0];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_filtering() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        apply_top_k(&mut logits, 3);
        // Top 3 by value: 5.0 (idx 1), 4.0 (idx 4), 3.0 (idx 2)
        // Rest should be -inf
        assert_eq!(logits[0], f32::NEG_INFINITY); // 1.0 masked
        assert!((logits[1] - 5.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
        assert_eq!(logits[3], f32::NEG_INFINITY); // 2.0 masked
        assert!((logits[4] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_larger_than_vocab() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_top_k(&mut logits, 10);
        assert_eq!(logits, original); // No masking when k > vocab
    }

    #[test]
    fn test_top_p_filtering() {
        // Create logits where the top token dominates
        let mut logits = vec![10.0, 1.0, 0.0, -1.0, -10.0];
        apply_top_p(&mut logits, 0.9);
        // The first token (logit 10.0) should have near-1.0 probability,
        // so it alone should exceed p=0.9. Everything else gets masked.
        assert!(logits[0].is_finite());
        // Some tokens should be masked
        let masked_count = logits.iter().filter(|&&v| v == f32::NEG_INFINITY).count();
        assert!(masked_count > 0, "Top-p should mask some tokens");
    }

    #[test]
    fn test_top_p_disabled_at_1() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_top_p(&mut logits, 1.0);
        // At p=1.0, no filtering happens (we accumulate to 1.0, keeping everything)
        assert_eq!(logits, original);
    }

    #[test]
    fn test_softmax_basic() {
        let logits = vec![0.0, 0.0, 0.0];
        let probs = softmax(&logits);
        // Uniform distribution
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_with_neg_inf() {
        let logits = vec![1.0, f32::NEG_INFINITY, 2.0];
        let probs = softmax(&logits);
        assert!((probs[1] - 0.0).abs() < 1e-6);
        assert!(probs[0] > 0.0);
        assert!(probs[2] > 0.0);
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large logits shouldn't cause overflow
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
        // Largest logit should have highest prob
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_sample_categorical_deterministic_with_one_option() {
        let logits = vec![f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY];
        let mut rng = rand::thread_rng();
        // Only token 1 is valid, so it must always be sampled
        for _ in 0..20 {
            let token = sample_categorical(&mut rng, &logits);
            assert_eq!(token, 1);
        }
    }

    #[test]
    fn test_full_pipeline_greedy() {
        let config = SamplerConfig {
            temperature: 0.0,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
        };
        let mut sampler = Sampler::new(config);

        let logits = vec![0.1, 0.2, 10.0, 0.3];
        let token = sampler.sample_next(&logits, &[]);
        assert_eq!(token, 2); // Greedy picks highest
    }

    #[test]
    fn test_full_pipeline_with_repetition_penalty() {
        let config = SamplerConfig {
            temperature: 0.0, // greedy to make test deterministic
            repetition_penalty: 100.0, // extreme penalty
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);

        // Token 2 has highest logit but is heavily penalized
        let logits = vec![0.1, 5.0, 5.1, 0.3];
        let token = sampler.sample_next(&logits, &[2]); // token 2 is penalized
        assert_eq!(token, 1); // token 1 should win after penalty
    }
}
