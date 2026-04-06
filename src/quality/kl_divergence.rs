//! KL divergence measurement between original and quantized logit distributions.
//!
//! Computes both per-layer and overall KL divergence D_KL(P || Q) where:
//! - P is the original (pre-quantization) logit distribution
//! - Q is the quantized (post-quantization) logit distribution
//!
//! KL divergence measures information loss from quantization.
//! A KL of 0 means no information lost; higher values mean more degradation.

use thiserror::Error;

/// Errors from KL divergence computation.
#[derive(Error, Debug)]
pub enum KlDivergenceError {
    #[error("Distribution length mismatch: original has {original} elements, quantized has {quantized}")]
    LengthMismatch { original: usize, quantized: usize },

    #[error("Empty distribution provided")]
    EmptyDistribution,

    #[error("Invalid distribution: contains NaN or infinite values")]
    InvalidDistribution,
}

/// Result of KL divergence measurement for the entire model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KlDivergenceResult {
    /// Overall KL divergence (average across all layers)
    pub overall: f64,
    /// Per-layer KL divergence values
    pub per_layer: Vec<f64>,
}

/// Compute KL divergence D_KL(P || Q) between two probability distributions.
///
/// Both `original` and `quantized` should be raw logits (not yet softmaxed).
/// This function applies softmax internally to convert to probability distributions.
///
/// Returns the KL divergence in nats (natural log base).
pub fn kl_divergence(original: &[f32], quantized: &[f32]) -> Result<f64, KlDivergenceError> {
    if original.is_empty() || quantized.is_empty() {
        return Err(KlDivergenceError::EmptyDistribution);
    }

    if original.len() != quantized.len() {
        return Err(KlDivergenceError::LengthMismatch {
            original: original.len(),
            quantized: quantized.len(),
        });
    }

    // Check for NaN/Inf
    if original.iter().any(|x| x.is_nan() || x.is_infinite())
        || quantized.iter().any(|x| x.is_nan() || x.is_infinite())
    {
        return Err(KlDivergenceError::InvalidDistribution);
    }

    // Apply softmax to get probability distributions
    let p = softmax(original);
    let q = softmax(quantized);

    // Compute KL divergence: D_KL(P || Q) = sum(P * log(P / Q))
    // Use log-sum form for numerical stability: P * (log P - log Q)
    let epsilon = 1e-10_f64; // Avoid log(0)

    let kl: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let pi = pi.max(epsilon);
            let qi = qi.max(epsilon);
            pi * (pi.ln() - qi.ln())
        })
        .sum();

    // Clamp to non-negative (numerical errors can cause tiny negative values)
    Ok(kl.max(0.0))
}

/// Compute softmax of raw logits, producing a probability distribution.
///
/// Uses the log-sum-exp trick for numerical stability:
/// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
fn softmax(logits: &[f32]) -> Vec<f64> {
    let max_val = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;

    let exp_vals: Vec<f64> = logits
        .iter()
        .map(|&x| ((x as f64) - max_val).exp())
        .collect();

    let sum: f64 = exp_vals.iter().sum();

    if sum == 0.0 || sum.is_nan() || sum.is_infinite() {
        // Fallback: uniform distribution
        let n = logits.len() as f64;
        return vec![1.0 / n; logits.len()];
    }

    exp_vals.iter().map(|&x| x / sum).collect()
}

/// Compute per-layer KL divergence between original and quantized layer activations.
///
/// Each layer's activations are treated as logit-like values and compared
/// using KL divergence after softmax normalization.
///
/// `original_layers` and `quantized_layers` should have the same number of entries,
/// with each entry being the flattened activations for that layer.
pub fn per_layer_kl_divergence(
    original_layers: &[Vec<f32>],
    quantized_layers: &[Vec<f32>],
) -> Result<KlDivergenceResult, KlDivergenceError> {
    if original_layers.is_empty() {
        return Err(KlDivergenceError::EmptyDistribution);
    }

    let num_layers = original_layers.len().min(quantized_layers.len());
    let mut per_layer = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let orig = &original_layers[i];
        let quant = &quantized_layers[i];

        // If dimensions don't match, take the minimum
        let len = orig.len().min(quant.len());
        if len == 0 {
            per_layer.push(0.0);
            continue;
        }

        let kl = kl_divergence(&orig[..len], &quant[..len])?;
        per_layer.push(kl);
    }

    let overall = if per_layer.is_empty() {
        0.0
    } else {
        per_layer.iter().sum::<f64>() / per_layer.len() as f64
    };

    Ok(KlDivergenceResult { overall, per_layer })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_distributions_zero_kl() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kl = kl_divergence(&logits, &logits).unwrap();
        assert!(kl < 1e-10, "KL divergence of identical distributions should be ~0, got {}", kl);
    }

    #[test]
    fn test_different_distributions_positive_kl() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quantized = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let kl = kl_divergence(&original, &quantized).unwrap();
        assert!(kl > 0.0, "KL divergence of different distributions should be positive");
    }

    #[test]
    fn test_softmax_produces_valid_distribution() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Softmax should sum to 1.0");
        assert!(probs.iter().all(|&p| p >= 0.0), "All probabilities should be non-negative");
        // Higher logit should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without the log-sum-exp trick
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Softmax should still sum to 1.0 for large values");
    }

    #[test]
    fn test_length_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let result = kl_divergence(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_distribution() {
        let result = kl_divergence(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_nan_detection() {
        let a = vec![1.0, f32::NAN, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = kl_divergence(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_kl_asymmetry() {
        // KL divergence is asymmetric: D_KL(P||Q) != D_KL(Q||P) in general
        let p = vec![1.0, 5.0, 1.0];
        let q = vec![5.0, 1.0, 5.0];
        let kl_pq = kl_divergence(&p, &q).unwrap();
        let kl_qp = kl_divergence(&q, &p).unwrap();
        // They should generally be different (unless the distributions are symmetric)
        // Just verify both are non-negative
        assert!(kl_pq >= 0.0);
        assert!(kl_qp >= 0.0);
    }

    #[test]
    fn test_per_layer_kl_divergence() {
        let original = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let quantized = vec![
            vec![1.0, 2.0, 3.0],  // identical to original
            vec![6.0, 5.0, 4.0],  // different from original
        ];

        let result = per_layer_kl_divergence(&original, &quantized).unwrap();
        assert_eq!(result.per_layer.len(), 2);
        assert!(result.per_layer[0] < 1e-10, "Identical layers should have ~0 KL");
        assert!(result.per_layer[1] > 0.0, "Different layers should have positive KL");
        assert!(result.overall > 0.0, "Overall KL should be positive when some layers differ");
    }

    #[test]
    fn test_per_layer_empty_layers() {
        let result = per_layer_kl_divergence(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_kl_non_negative() {
        // KL divergence should always be non-negative
        for _ in 0..10 {
            let a = vec![0.1, 0.3, 0.2, 0.4, 0.5];
            let b = vec![0.5, 0.4, 0.3, 0.2, 0.1];
            let kl = kl_divergence(&a, &b).unwrap();
            assert!(kl >= 0.0, "KL divergence should be non-negative, got {}", kl);
        }
    }

    #[test]
    fn test_small_perturbation_small_kl() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quantized = vec![1.01, 2.01, 3.01, 4.01, 5.01]; // Small perturbation
        let kl = kl_divergence(&original, &quantized).unwrap();
        // KL should be very small for small perturbations
        assert!(kl < 0.01, "Small perturbation should give small KL, got {}", kl);
    }
}
