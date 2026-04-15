//! Activation-based sensitivity scoring for DWQ calibration.
//!
//! Computes per-layer sensitivity from captured hidden-state activations.
//! Higher sensitivity = more information loss under quantization = needs more bits.

use anyhow::Result;
use tracing::debug;

/// Per-layer sensitivity statistics.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LayerSensitivity {
    /// Layer index (0-based).
    pub layer_index: usize,
    /// Activation variance across the hidden dimension.
    pub variance: f64,
    /// Maximum absolute activation magnitude.
    pub max_magnitude: f64,
    /// Combined sensitivity score: sqrt(variance) * log2(1 + max_magnitude).
    pub score: f64,
}

/// Compute per-layer sensitivity scores from activation f32 slices.
///
/// Each activation slice is the flattened hidden-state values for one layer.
/// Sensitivity is computed as:
///   score = sqrt(variance) * log2(1 + max_magnitude)
///
/// Higher score means the layer is more sensitive to quantization error.
#[allow(dead_code)]
pub fn compute_layer_sensitivity(activations: &[Vec<f32>]) -> Result<Vec<LayerSensitivity>> {
    let mut sensitivities = Vec::with_capacity(activations.len());

    for (i, act) in activations.iter().enumerate() {
        let n = act.len() as f64;

        if n < 1.0 {
            sensitivities.push(LayerSensitivity {
                layer_index: i,
                variance: 0.0,
                max_magnitude: 0.0,
                score: 0.0,
            });
            continue;
        }

        // Compute variance: var = E[x^2] - E[x]^2
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut max_abs = 0.0_f64;

        for &v in act {
            let v = v as f64;
            sum += v;
            sum_sq += v * v;
            let abs_v = v.abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
        }

        let mean = sum / n;
        let mean_sq = sum_sq / n;
        let variance = (mean_sq - mean * mean).max(0.0);
        let max_magnitude = max_abs;

        // Combined score
        let score = variance.sqrt() * (1.0 + max_magnitude).log2();

        debug!(
            layer = i,
            variance = format!("{:.6}", variance),
            max_mag = format!("{:.4}", max_magnitude),
            score = format!("{:.6}", score),
            "Layer sensitivity"
        );

        sensitivities.push(LayerSensitivity {
            layer_index: i,
            variance,
            max_magnitude,
            score,
        });
    }

    Ok(sensitivities)
}

/// Allocate bits per layer based on sensitivity scores.
///
/// Uses a smooth gradient from `base_bits` to `sensitive_bits` based on
/// each layer's relative sensitivity within the distribution.
///
/// Layers with higher sensitivity get more bits. The allocation is:
///   normalized = (score - min_score) / (max_score - min_score)
///   bits = round(base_bits + normalized * (sensitive_bits - base_bits))
#[allow(dead_code)]
pub fn allocate_bits_by_sensitivity(
    sensitivity: &[LayerSensitivity],
    base_bits: u8,
    sensitive_bits: u8,
) -> Vec<u8> {
    if sensitivity.is_empty() {
        return Vec::new();
    }

    let scores: Vec<f64> = sensitivity.iter().map(|s| s.score).collect();

    let min_score = scores
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_score = scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let range = max_score - min_score;
    let bit_range = (sensitive_bits as f64) - (base_bits as f64);

    scores
        .iter()
        .map(|&score| {
            if range < 1e-12 {
                // All layers have the same sensitivity, use midpoint
                let mid = (base_bits as f64 + sensitive_bits as f64) / 2.0;
                mid.round() as u8
            } else {
                let normalized = (score - min_score) / range;
                let bits = base_bits as f64 + normalized * bit_range;
                bits.round().clamp(base_bits as f64, sensitive_bits as f64) as u8
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_layer_sensitivity_basic() {
        // Layer with low variance, small magnitude
        let low = vec![0.1f32, 0.2, 0.1, 0.2, 0.1, 0.2];

        // Layer with high variance, large magnitude
        let high = vec![-5.0f32, 10.0, -3.0, 8.0, -7.0, 12.0];

        let activations = vec![low, high];
        let result = compute_layer_sensitivity(&activations).unwrap();

        assert_eq!(result.len(), 2);
        // The high-variance layer should have a higher score
        assert!(
            result[1].score > result[0].score,
            "High-variance layer score ({}) should exceed low-variance layer score ({})",
            result[1].score,
            result[0].score,
        );
        assert!(result[0].variance > 0.0);
        assert!(result[1].variance > result[0].variance);
        assert!(result[1].max_magnitude > result[0].max_magnitude);
    }

    #[test]
    fn test_compute_layer_sensitivity_empty() {
        let activations: Vec<Vec<f32>> = vec![];
        let result = compute_layer_sensitivity(&activations).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_allocate_bits_uniform_sensitivity() {
        // All layers have the same sensitivity => midpoint bits
        let sensitivity = vec![
            LayerSensitivity { layer_index: 0, variance: 1.0, max_magnitude: 1.0, score: 1.0 },
            LayerSensitivity { layer_index: 1, variance: 1.0, max_magnitude: 1.0, score: 1.0 },
            LayerSensitivity { layer_index: 2, variance: 1.0, max_magnitude: 1.0, score: 1.0 },
        ];
        let bits = allocate_bits_by_sensitivity(&sensitivity, 4, 6);
        assert_eq!(bits, vec![5, 5, 5]); // midpoint of 4 and 6
    }

    #[test]
    fn test_allocate_bits_gradient() {
        let sensitivity = vec![
            LayerSensitivity { layer_index: 0, variance: 0.1, max_magnitude: 0.5, score: 0.0 },
            LayerSensitivity { layer_index: 1, variance: 0.5, max_magnitude: 2.0, score: 0.5 },
            LayerSensitivity { layer_index: 2, variance: 1.0, max_magnitude: 5.0, score: 1.0 },
        ];
        let bits = allocate_bits_by_sensitivity(&sensitivity, 4, 6);

        // Least sensitive gets base_bits, most sensitive gets sensitive_bits
        assert_eq!(bits[0], 4);
        assert_eq!(bits[1], 5);
        assert_eq!(bits[2], 6);
    }

    #[test]
    fn test_allocate_bits_empty() {
        let sensitivity: Vec<LayerSensitivity> = vec![];
        let bits = allocate_bits_by_sensitivity(&sensitivity, 4, 6);
        assert!(bits.is_empty());
    }

    #[test]
    fn test_allocate_bits_single_layer() {
        let sensitivity = vec![
            LayerSensitivity { layer_index: 0, variance: 1.0, max_magnitude: 5.0, score: 3.0 },
        ];
        let bits = allocate_bits_by_sensitivity(&sensitivity, 4, 6);
        assert_eq!(bits, vec![5]); // midpoint for single layer
    }

    #[test]
    fn test_allocate_bits_many_layers() {
        // 10 layers with linearly increasing sensitivity
        let sensitivity: Vec<LayerSensitivity> = (0..10)
            .map(|i| LayerSensitivity {
                layer_index: i,
                variance: i as f64,
                max_magnitude: i as f64,
                score: i as f64,
            })
            .collect();
        let bits = allocate_bits_by_sensitivity(&sensitivity, 2, 6);

        // First should be 2, last should be 6
        assert_eq!(bits[0], 2);
        assert_eq!(bits[9], 6);
        // Should be monotonically non-decreasing
        for w in bits.windows(2) {
            assert!(w[1] >= w[0], "Bits should be non-decreasing: {:?}", bits);
        }
    }
}
