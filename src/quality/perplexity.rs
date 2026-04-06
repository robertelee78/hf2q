//! Perplexity delta measurement between pre-quant and post-quant models.
//!
//! Perplexity measures how well a model predicts a sample text.
//! Lower perplexity = better predictions.
//!
//! Perplexity delta = post_quant_perplexity - pre_quant_perplexity
//! A delta close to 0 means quantization preserved model quality.

use thiserror::Error;

/// Errors from perplexity computation.
#[derive(Error, Debug)]
pub enum PerplexityError {
    #[error("Empty logits sequence provided")]
    EmptySequence,

    #[error("Token count mismatch: expected {expected} logit sets, got {actual}")]
    TokenCountMismatch { expected: usize, actual: usize },

    #[error("Invalid logits: contains NaN or infinite values")]
    InvalidLogits,

    #[error("Vocabulary size mismatch: logits have {logits_vocab} entries but target token {token_id} is out of range")]
    VocabMismatch { logits_vocab: usize, token_id: u32 },
}

/// Result of perplexity measurement.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerplexityResult {
    /// Pre-quantization perplexity
    pub pre_quant: f64,
    /// Post-quantization perplexity
    pub post_quant: f64,
    /// Delta: post_quant - pre_quant (positive means quality degraded)
    pub delta: f64,
}

/// Compute perplexity from a sequence of logit vectors and corresponding target tokens.
///
/// For each position i, `logits[i]` is the model's logit distribution (length = vocab_size)
/// and `targets[i]` is the actual next token ID.
///
/// Perplexity = exp(-1/N * sum(log P(target_i | context)))
///
/// where P(target_i | context) is obtained by applying softmax to logits[i]
/// and taking the probability at index targets[i].
pub fn compute_perplexity(
    logits_sequence: &[Vec<f32>],
    targets: &[u32],
) -> Result<f64, PerplexityError> {
    if logits_sequence.is_empty() || targets.is_empty() {
        return Err(PerplexityError::EmptySequence);
    }

    if logits_sequence.len() != targets.len() {
        return Err(PerplexityError::TokenCountMismatch {
            expected: targets.len(),
            actual: logits_sequence.len(),
        });
    }

    let n = logits_sequence.len() as f64;
    let mut total_log_prob = 0.0_f64;

    for (logits, &target) in logits_sequence.iter().zip(targets.iter()) {
        if logits.is_empty() {
            return Err(PerplexityError::EmptySequence);
        }

        if target as usize >= logits.len() {
            return Err(PerplexityError::VocabMismatch {
                logits_vocab: logits.len(),
                token_id: target,
            });
        }

        if logits.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err(PerplexityError::InvalidLogits);
        }

        // Compute log-softmax at the target position using log-sum-exp trick
        let log_prob = log_softmax_at(logits, target as usize);
        total_log_prob += log_prob;
    }

    // Perplexity = exp(-1/N * sum(log P))
    let avg_neg_log_prob = -total_log_prob / n;
    Ok(avg_neg_log_prob.exp())
}

/// Compute perplexity delta between original and quantized model outputs.
///
/// Both models should be evaluated on the same input sequence.
/// `original_logits` and `quantized_logits` are sequences of logit vectors,
/// and `targets` are the actual next-token IDs.
pub fn perplexity_delta(
    original_logits: &[Vec<f32>],
    quantized_logits: &[Vec<f32>],
    targets: &[u32],
) -> Result<PerplexityResult, PerplexityError> {
    let pre_quant = compute_perplexity(original_logits, targets)?;
    let post_quant = compute_perplexity(quantized_logits, targets)?;

    Ok(PerplexityResult {
        pre_quant,
        post_quant,
        delta: post_quant - pre_quant,
    })
}

/// Compute log P(target) using log-softmax at a specific position.
///
/// log_softmax(x_i) = x_i - log(sum(exp(x_j)))
///
/// Uses the log-sum-exp trick: log(sum(exp(x_j))) = max(x) + log(sum(exp(x_j - max(x))))
fn log_softmax_at(logits: &[f32], target_idx: usize) -> f64 {
    let max_val = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;

    let log_sum_exp: f64 = logits
        .iter()
        .map(|&x| ((x as f64) - max_val).exp())
        .sum::<f64>()
        .ln()
        + max_val;

    (logits[target_idx] as f64) - log_sum_exp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perplexity_perfect_prediction() {
        // If the model assigns all probability to the correct token,
        // perplexity should be close to 1
        let logits = vec![
            vec![-100.0, 100.0, -100.0], // Strongly predicts token 1
            vec![100.0, -100.0, -100.0], // Strongly predicts token 0
        ];
        let targets = vec![1, 0];

        let ppl = compute_perplexity(&logits, &targets).unwrap();
        assert!(ppl < 1.01, "Perfect prediction should give perplexity ~1, got {}", ppl);
    }

    #[test]
    fn test_perplexity_uniform_prediction() {
        // Uniform distribution over 3 tokens should give perplexity = 3
        let logits = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        let targets = vec![0, 1];

        let ppl = compute_perplexity(&logits, &targets).unwrap();
        assert!(
            (ppl - 3.0).abs() < 0.01,
            "Uniform prediction over 3 tokens should give perplexity ~3, got {}",
            ppl
        );
    }

    #[test]
    fn test_perplexity_delta_identical_models() {
        let logits = vec![
            vec![1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0],
        ];
        let targets = vec![2, 0];

        let result = perplexity_delta(&logits, &logits, &targets).unwrap();
        assert!((result.delta).abs() < 1e-10, "Identical models should have zero delta");
        assert!((result.pre_quant - result.post_quant).abs() < 1e-10);
    }

    #[test]
    fn test_perplexity_delta_degraded_model() {
        // Original model predicts well
        let original = vec![
            vec![-100.0, 100.0, -100.0], // Correct: token 1
        ];
        // Quantized model predicts poorly (uniform)
        let quantized = vec![
            vec![0.0, 0.0, 0.0],
        ];
        let targets = vec![1];

        let result = perplexity_delta(&original, &quantized, &targets).unwrap();
        assert!(result.delta > 0.0, "Degraded model should have positive delta");
        assert!(result.post_quant > result.pre_quant);
    }

    #[test]
    fn test_empty_sequence_error() {
        let result = compute_perplexity(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_token_count_mismatch() {
        let logits = vec![vec![1.0, 2.0, 3.0]];
        let targets = vec![0, 1]; // more targets than logits
        let result = compute_perplexity(&logits, &targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_out_of_range_token() {
        let logits = vec![vec![1.0, 2.0, 3.0]];
        let targets = vec![5]; // token 5 > vocab size 3
        let result = compute_perplexity(&logits, &targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_nan_logits() {
        let logits = vec![vec![1.0, f32::NAN, 3.0]];
        let targets = vec![0];
        let result = compute_perplexity(&logits, &targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_log_softmax_numerical_stability() {
        // Large logit values shouldn't cause overflow
        let logits = vec![1000.0, 1001.0, 1002.0];
        let log_prob = log_softmax_at(&logits, 2);
        // log_softmax(1002) = 1002 - log(exp(1000) + exp(1001) + exp(1002))
        // = 1002 - (1002 + log(exp(-2) + exp(-1) + 1))
        // = -log(exp(-2) + exp(-1) + 1) ≈ -log(1.503) ≈ -0.407
        assert!(log_prob < 0.0, "Log probability should be negative");
        assert!(log_prob > -5.0, "Log probability should not be extremely negative");
    }

    #[test]
    fn test_perplexity_always_positive() {
        let logits = vec![
            vec![0.5, 1.0, 0.3],
            vec![1.0, 0.2, 0.8],
        ];
        let targets = vec![1, 2];
        let ppl = compute_perplexity(&logits, &targets).unwrap();
        assert!(ppl > 0.0, "Perplexity should always be positive");
    }
}
