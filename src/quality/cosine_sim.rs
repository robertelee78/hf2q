//! Cosine similarity of layer activations between original and quantized models.
//!
//! Cosine similarity measures the angle between two vectors:
//! cos_sim(A, B) = (A . B) / (|A| * |B|)
//!
//! A value of 1.0 means identical direction (perfect preservation).
//! Values closer to 0 indicate significant deviation from the original.

use thiserror::Error;

/// Errors from cosine similarity computation.
#[derive(Error, Debug)]
pub enum CosineSimilarityError {
    #[error("Vector length mismatch: {a_len} vs {b_len}")]
    LengthMismatch { a_len: usize, b_len: usize },

    #[error("Empty vector provided")]
    EmptyVector,

    #[error("Zero-norm vector: cannot compute cosine similarity")]
    ZeroNorm,

    #[error("Invalid values: contains NaN or infinite")]
    InvalidValues,
}

/// Result of per-layer cosine similarity measurement.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CosineSimilarityResult {
    /// Per-layer cosine similarity values (one per layer)
    pub per_layer: Vec<f64>,
    /// Average cosine similarity across all layers
    pub average: f64,
    /// Minimum cosine similarity (worst layer)
    pub min: f64,
    /// Index of the layer with minimum similarity
    pub min_layer_idx: usize,
}

/// Compute cosine similarity between two f32 vectors.
///
/// Returns a value in [-1.0, 1.0] where:
/// - 1.0 means identical direction
/// - 0.0 means orthogonal
/// - -1.0 means opposite direction
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f64, CosineSimilarityError> {
    if a.is_empty() || b.is_empty() {
        return Err(CosineSimilarityError::EmptyVector);
    }

    if a.len() != b.len() {
        return Err(CosineSimilarityError::LengthMismatch {
            a_len: a.len(),
            b_len: b.len(),
        });
    }

    // Check for NaN/Inf
    if a.iter().any(|x| x.is_nan() || x.is_infinite())
        || b.iter().any(|x| x.is_nan() || x.is_infinite())
    {
        return Err(CosineSimilarityError::InvalidValues);
    }

    // Compute dot product and norms in f64 for precision
    let mut dot_product = 0.0_f64;
    let mut norm_a_sq = 0.0_f64;
    let mut norm_b_sq = 0.0_f64;

    for (ai, bi) in a.iter().zip(b.iter()) {
        let ai = *ai as f64;
        let bi = *bi as f64;
        dot_product += ai * bi;
        norm_a_sq += ai * ai;
        norm_b_sq += bi * bi;
    }

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    if norm_a < 1e-12 || norm_b < 1e-12 {
        return Err(CosineSimilarityError::ZeroNorm);
    }

    let similarity = dot_product / (norm_a * norm_b);

    // Clamp to [-1, 1] for numerical stability
    Ok(similarity.clamp(-1.0, 1.0))
}

/// Compute per-layer cosine similarity between original and quantized activations.
///
/// Each entry in `original_layers` and `quantized_layers` is the flattened
/// activation vector for that layer.
///
/// If layer dimensions don't match, the shorter vector is used as the comparison length.
pub fn per_layer_cosine_similarity(
    original_layers: &[Vec<f32>],
    quantized_layers: &[Vec<f32>],
) -> Result<CosineSimilarityResult, CosineSimilarityError> {
    if original_layers.is_empty() {
        return Err(CosineSimilarityError::EmptyVector);
    }

    let num_layers = original_layers.len().min(quantized_layers.len());
    let mut per_layer = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let orig = &original_layers[i];
        let quant = &quantized_layers[i];

        let len = orig.len().min(quant.len());
        if len == 0 {
            per_layer.push(1.0); // Empty layers are considered identical
            continue;
        }

        let sim = cosine_similarity(&orig[..len], &quant[..len])?;
        per_layer.push(sim);
    }

    let average = if per_layer.is_empty() {
        1.0
    } else {
        per_layer.iter().sum::<f64>() / per_layer.len() as f64
    };

    let (min_layer_idx, min) = per_layer
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, &val)| (idx, val))
        .unwrap_or((0, 1.0));

    Ok(CosineSimilarityResult {
        per_layer,
        average,
        min,
        min_layer_idx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors_similarity_one() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = cosine_similarity(&a, &a).unwrap();
        assert!((sim - 1.0).abs() < 1e-10, "Identical vectors should have similarity 1.0, got {}", sim);
    }

    #[test]
    fn test_opposite_vectors_similarity_negative_one() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - (-1.0)).abs() < 1e-10, "Opposite vectors should have similarity -1.0, got {}", sim);
    }

    #[test]
    fn test_orthogonal_vectors_similarity_zero() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-10, "Orthogonal vectors should have similarity ~0, got {}", sim);
    }

    #[test]
    fn test_similar_vectors_high_similarity() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.01, 2.01, 3.01, 4.01, 5.01]; // Small perturbation
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim > 0.999, "Similar vectors should have high similarity, got {}", sim);
    }

    #[test]
    fn test_length_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let result = cosine_similarity(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_vector() {
        let result = cosine_similarity(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = cosine_similarity(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_nan_detection() {
        let a = vec![1.0, f32::NAN, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = cosine_similarity(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_per_layer_cosine_similarity() {
        let original = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let quantized = vec![
            vec![1.0, 2.0, 3.0],     // identical
            vec![4.0, 5.0, 6.01],    // very similar
            vec![-7.0, -8.0, -9.0],  // opposite
        ];

        let result = per_layer_cosine_similarity(&original, &quantized).unwrap();
        assert_eq!(result.per_layer.len(), 3);
        assert!((result.per_layer[0] - 1.0).abs() < 1e-10);
        assert!(result.per_layer[1] > 0.99);
        assert!((result.per_layer[2] - (-1.0)).abs() < 1e-10);
        assert_eq!(result.min_layer_idx, 2);
    }

    #[test]
    fn test_per_layer_empty() {
        let result = per_layer_cosine_similarity(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cosine_similarity_range() {
        // Result should always be in [-1, 1]
        let a = vec![0.1, -0.5, 0.3, 0.8, -0.2];
        let b = vec![-0.3, 0.7, -0.1, 0.4, 0.6];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim >= -1.0 && sim <= 1.0, "Cosine similarity should be in [-1, 1], got {}", sim);
    }

    #[test]
    fn test_scaling_invariance() {
        // Cosine similarity should be scale-invariant
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0]; // 2x scaled
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-10, "Scaled vectors should have similarity 1.0, got {}", sim);
    }
}
