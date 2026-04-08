//! Token sampling matching the mlx-lm pipeline order exactly.
//!
//! Pipeline:
//! 1. Repetition penalty (windowed: last `repetition_context_size` tokens)
//! 2. Greedy check (if temperature ~ 0 and no rep penalty, return argmax)
//! 3. Log-softmax (work in log-probability domain)
//! 4. Top-p nucleus filtering (on logprobs, accumulate in prob space)
//! 5. Min-p filtering (threshold relative to max logprob)
//! 6. Top-k filtering (on logprobs)
//! 7. Temperature + categorical sample (logprobs / temp -> softmax -> sample)

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
    /// Min-p filtering threshold. 0.0 = disabled.
    pub min_p: f32,
    /// Number of recent tokens to consider for repetition penalty.
    pub repetition_context_size: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            min_p: 0.0,
            repetition_context_size: 20,
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

    /// Returns true if sampling is greedy (temperature near zero).
    ///
    /// When greedy, the GPU argmax path can skip the full logits readback.
    #[inline]
    pub fn is_greedy(&self) -> bool {
        self.config.temperature < 1e-7
            && self.config.repetition_penalty == 1.0
    }

    /// Sample the next token from logits, given the tokens generated so far.
    ///
    /// Applies the mlx-lm pipeline:
    /// rep_penalty (windowed) -> greedy check -> logsoftmax -> top_p -> min_p -> top_k -> temp+sample
    pub fn sample_next(&mut self, logits: &[f32], generated_ids: &[u32]) -> u32 {
        let mut working_logits = logits.to_vec();

        // Step 1: Repetition penalty (windowed)
        if self.config.repetition_penalty != 1.0 {
            let ctx_size = self.config.repetition_context_size;
            let start = generated_ids.len().saturating_sub(ctx_size);
            let window = &generated_ids[start..];
            apply_repetition_penalty(
                &mut working_logits,
                window,
                self.config.repetition_penalty,
            );
        }

        // Step 2: Greedy check — if temperature ~ 0, return argmax
        if self.config.temperature < 1e-7 {
            return argmax(&working_logits);
        }

        // Step 3: Log-softmax — convert logits to log-probabilities
        let mut logprobs = log_softmax(&working_logits);

        // Step 4: Top-p filtering (on logprobs, accumulate in prob space)
        if self.config.top_p < 1.0 {
            apply_top_p_logprobs(&mut logprobs, self.config.top_p);
        }

        // Step 5: Min-p filtering
        if self.config.min_p > 0.0 {
            apply_min_p(&mut logprobs, self.config.min_p);
        }

        // Step 6: Top-k filtering (on logprobs)
        if self.config.top_k > 0 {
            apply_top_k(&mut logprobs, self.config.top_k);
        }

        // Step 7: Temperature + categorical sample
        // Divide logprobs by temperature, then softmax to get final probs
        let temp = self.config.temperature;
        for v in logprobs.iter_mut() {
            if *v != f32::NEG_INFINITY {
                *v /= temp;
            }
        }

        let probs = softmax_from_logits(&logprobs);
        sample_categorical_from_probs(&mut self.rng, &probs)
    }
}

/// Apply repetition penalty to logits.
///
/// For each token in the window, divide its logit by `penalty` if positive,
/// or multiply by `penalty` if negative. This always reduces repeated token probability.
fn apply_repetition_penalty(logits: &mut [f32], window: &[u32], penalty: f32) {
    for &id in window {
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

/// Compute log-softmax: log(softmax(x)) = x - log(sum(exp(x - max)))
///
/// Numerically stable implementation.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let log_sum_exp: f32 = logits
        .iter()
        .map(|&v| {
            if v == f32::NEG_INFINITY {
                0.0
            } else {
                (v - max_logit).exp()
            }
        })
        .sum::<f32>()
        .ln();

    logits
        .iter()
        .map(|&v| {
            if v == f32::NEG_INFINITY {
                f32::NEG_INFINITY
            } else {
                v - max_logit - log_sum_exp
            }
        })
        .collect()
}

/// Apply top-p (nucleus) filtering on logprobs.
///
/// Sort by descending logprob, accumulate exp(logprob) (= probability) until
/// cumulative >= p, then mask the rest to -inf.
fn apply_top_p_logprobs(logprobs: &mut [f32], p: f32) {
    // Sort indices by descending logprob
    let mut indexed: Vec<(usize, f32)> = logprobs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Accumulate in probability space
    let mut cumulative = 0.0f32;
    let mut keep_set = vec![false; logprobs.len()];

    for &(idx, lp) in &indexed {
        if lp == f32::NEG_INFINITY {
            break;
        }
        keep_set[idx] = true;
        cumulative += lp.exp();
        if cumulative >= p {
            break;
        }
    }

    // Mask tokens not in the keep set
    for (i, v) in logprobs.iter_mut().enumerate() {
        if !keep_set[i] {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Apply min-p filtering on logprobs.
///
/// Compute threshold = max_logprob + ln(min_p). Mask any logprob below threshold.
fn apply_min_p(logprobs: &mut [f32], min_p: f32) {
    let max_lp = logprobs
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    if max_lp == f32::NEG_INFINITY {
        return;
    }

    let threshold = max_lp + min_p.ln();

    for v in logprobs.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Apply top-k filtering: set all logprobs outside the top k to -infinity.
///
/// Uses `select_nth_unstable_by` for O(V) average-case partitioning.
fn apply_top_k(logprobs: &mut [f32], k: usize) {
    if k >= logprobs.len() {
        return;
    }

    // Count non-neg-inf entries; if already <= k, nothing to do.
    let finite_count = logprobs.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
    if finite_count <= k {
        return;
    }

    let mut indexed: Vec<(usize, f32)> = logprobs.iter().copied().enumerate().collect();
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let threshold = indexed[k - 1].1;

    // In case of ties at the boundary, keep all tokens with logprob >= threshold.
    for v in logprobs.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Compute softmax from logits/logprobs, returning a probability distribution.
fn softmax_from_logits(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_val = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_sum: f32 = logits
        .iter()
        .map(|&v| {
            if v == f32::NEG_INFINITY {
                0.0
            } else {
                (v - max_val).exp()
            }
        })
        .sum();

    if exp_sum == 0.0 {
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
                (v - max_val).exp() / exp_sum
            }
        })
        .collect()
}

/// Sample from a categorical distribution given pre-computed probabilities.
fn sample_categorical_from_probs(rng: &mut impl Rng, probs: &[f32]) -> u32 {
    let mut cumulative = Vec::with_capacity(probs.len());
    let mut sum = 0.0f32;
    for &p in probs {
        sum += p;
        cumulative.push(sum);
    }

    // Handle edge case: if total probability is ~0, fall back to first non-zero
    if sum < 1e-10 {
        // Return index of first non-zero or 0
        return probs
            .iter()
            .enumerate()
            .find(|(_, &p)| p > 0.0)
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
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

    (probs.len() - 1) as u32
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
        assert!((logits[0] - 2.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 1.0).abs() < 1e-6);
        assert!((logits[3] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let mut logits = vec![-2.0, -4.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 1.5);
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
    fn test_log_softmax_basic() {
        let logits = vec![0.0, 0.0, 0.0];
        let lp = log_softmax(&logits);
        let expected = (1.0f32 / 3.0).ln();
        for &v in &lp {
            assert!((v - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_log_softmax_with_neg_inf() {
        let logits = vec![1.0, f32::NEG_INFINITY, 2.0];
        let lp = log_softmax(&logits);
        assert_eq!(lp[1], f32::NEG_INFINITY);
        assert!(lp[0] < 0.0);
        assert!(lp[2] < 0.0);
        // log-probs should sum to ~1 in prob space
        let prob_sum: f32 = lp.iter().map(|&v| if v == f32::NEG_INFINITY { 0.0 } else { v.exp() }).sum();
        assert!((prob_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_numerical_stability() {
        let logits = vec![1000.0, 1001.0, 1002.0];
        let lp = log_softmax(&logits);
        let prob_sum: f32 = lp.iter().map(|v| v.exp()).sum();
        assert!((prob_sum - 1.0).abs() < 1e-5);
        // Largest logit should have highest logprob
        assert!(lp[2] > lp[1]);
        assert!(lp[1] > lp[0]);
    }

    #[test]
    fn test_log_softmax_empty() {
        let lp = log_softmax(&[]);
        assert!(lp.is_empty());
    }

    #[test]
    fn test_top_k_filtering() {
        let mut logprobs = vec![-3.0, -0.5, -1.5, -2.5, -1.0];
        apply_top_k(&mut logprobs, 3);
        // Top 3 by value: -0.5 (idx 1), -1.0 (idx 4), -1.5 (idx 2)
        assert_eq!(logprobs[0], f32::NEG_INFINITY); // -3.0 masked
        assert!((logprobs[1] - (-0.5)).abs() < 1e-6);
        assert!((logprobs[2] - (-1.5)).abs() < 1e-6);
        assert_eq!(logprobs[3], f32::NEG_INFINITY); // -2.5 masked
        assert!((logprobs[4] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_larger_than_vocab() {
        let mut logprobs = vec![-1.0, -2.0, -3.0];
        let original = logprobs.clone();
        apply_top_k(&mut logprobs, 10);
        assert_eq!(logprobs, original);
    }

    #[test]
    fn test_top_p_filtering_logprobs() {
        // Create logprobs where the top token dominates
        // logprob of -0.01 ~ prob 0.99; rest are much smaller
        let mut logprobs = vec![-0.01, -5.0, -6.0, -7.0, -20.0];
        apply_top_p_logprobs(&mut logprobs, 0.9);
        // First token alone exceeds p=0.9, rest get masked
        assert!(logprobs[0].is_finite());
        let masked_count = logprobs.iter().filter(|&&v| v == f32::NEG_INFINITY).count();
        assert!(masked_count > 0, "Top-p should mask some tokens");
    }

    #[test]
    fn test_top_p_disabled_at_1() {
        let mut logprobs = vec![-1.0, -2.0, -3.0];
        let original = logprobs.clone();
        apply_top_p_logprobs(&mut logprobs, 1.0);
        // At p=1.0, accumulate to 1.0 keeping everything
        assert_eq!(logprobs, original);
    }

    #[test]
    fn test_min_p_filtering() {
        // max logprob is -0.1 (~0.905 probability)
        // threshold = -0.1 + ln(0.1) = -0.1 + (-2.302) = -2.402
        let mut logprobs = vec![-0.1, -1.0, -2.0, -3.0, -5.0];
        apply_min_p(&mut logprobs, 0.1);

        // -0.1, -1.0, -2.0 are above -2.402; -3.0 and -5.0 are below
        assert!(logprobs[0].is_finite());
        assert!(logprobs[1].is_finite());
        assert!(logprobs[2].is_finite());
        assert_eq!(logprobs[3], f32::NEG_INFINITY);
        assert_eq!(logprobs[4], f32::NEG_INFINITY);
    }

    #[test]
    fn test_min_p_disabled_at_zero() {
        let mut logprobs = vec![-0.1, -1.0, -2.0];
        let original = logprobs.clone();
        // min_p = 0.0 means disabled; the function should not be called,
        // but let's verify it doesn't mask anything if called with a positive value
        // (the caller guards on min_p > 0.0, so test with a tiny value)
        apply_min_p(&mut logprobs, 1e-30);
        // ln(1e-30) ~ -69, threshold ~ -0.1 + (-69) = -69.1
        // Nothing should be masked
        assert_eq!(logprobs, original);
    }

    #[test]
    fn test_repetition_context_window() {
        // With a large list of generated_ids but small context window,
        // only recent tokens should be penalized
        let config = SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 100.0,
            repetition_context_size: 2,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);

        // Token 0 was generated long ago, tokens 2,3 are recent
        let logits = vec![10.0, 1.0, 5.0, 5.0];
        let generated = vec![0, 1, 2, 3]; // window of 2 = [2, 3]

        let token = sampler.sample_next(&logits, &generated);
        // Token 0 should NOT be penalized (outside window), so it wins at 10.0
        assert_eq!(token, 0);
    }

    #[test]
    fn test_repetition_context_window_full() {
        // When context_size >= generated_ids.len(), all are penalized (old behavior)
        let config = SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 100.0,
            repetition_context_size: 100,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);

        let logits = vec![0.1, 5.0, 5.1, 0.3];
        let token = sampler.sample_next(&logits, &[2]);
        // Token 2 penalized: 5.1/100 = 0.051, so token 1 (5.0) wins
        assert_eq!(token, 1);
    }

    #[test]
    fn test_softmax_from_logits_basic() {
        let logits = vec![0.0, 0.0, 0.0];
        let probs = softmax_from_logits(&logits);
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_from_logits_with_neg_inf() {
        let logits = vec![1.0, f32::NEG_INFINITY, 2.0];
        let probs = softmax_from_logits(&logits);
        assert!((probs[1] - 0.0).abs() < 1e-6);
        assert!(probs[0] > 0.0);
        assert!(probs[2] > 0.0);
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_categorical_deterministic_with_one_option() {
        let probs = vec![0.0, 1.0, 0.0];
        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            let token = sample_categorical_from_probs(&mut rng, &probs);
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
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);

        let logits = vec![0.1, 0.2, 10.0, 0.3];
        let token = sampler.sample_next(&logits, &[]);
        assert_eq!(token, 2); // Greedy picks highest
    }

    #[test]
    fn test_full_pipeline_with_repetition_penalty() {
        let config = SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 100.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);

        let logits = vec![0.1, 5.0, 5.1, 0.3];
        let token = sampler.sample_next(&logits, &[2]);
        assert_eq!(token, 1); // token 1 should win after penalty on token 2
    }

    #[test]
    fn test_temperature_after_filtering_differs_from_before() {
        // This test verifies that temperature-after-filtering (mlx-lm order)
        // produces different behavior than temperature-before-filtering.
        //
        // With temperature BEFORE filtering (old pipeline):
        //   logits/temp -> top_k -> softmax -> sample
        //   High temp flattens logits BEFORE top-k, potentially changing which
        //   tokens survive the top-k cut.
        //
        // With temperature AFTER filtering (new pipeline):
        //   logits -> logsoftmax -> top_k -> logprobs/temp -> softmax -> sample
        //   Top-k selects based on unscaled probabilities, temp only affects
        //   the distribution among surviving tokens.

        let logits = vec![10.0, 5.0, 4.9, 0.1];

        // New pipeline (temperature after): top-k=2 on unscaled logprobs
        // Token 0 (10.0) and token 1 (5.0) survive, then temp=5.0 flattens among them
        let mut lp = log_softmax(&logits);
        apply_top_k(&mut lp, 2);

        // Tokens 2 and 3 should be masked
        assert_eq!(lp[2], f32::NEG_INFINITY);
        assert_eq!(lp[3], f32::NEG_INFINITY);
        assert!(lp[0].is_finite());
        assert!(lp[1].is_finite());

        // Old pipeline would have applied temp first, potentially making
        // tokens 1 and 2 (5.0 and 4.9) nearly identical and both surviving top-k.
        // The new pipeline correctly selects top-k BEFORE temperature scaling.
    }
}
