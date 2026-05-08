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

/// ADR-020 iter-12b-3 — aggregate per-Linear FD sensitivity scores
/// into per-LAYER `LayerSensitivity` values.
///
/// Bridges `fd_sensitivity::compute_fd_sensitivity_per_linear`'s output
/// `BTreeMap<String, f32>` (keyed by GGUF tensor name like
/// `blk.{i}.attn_q.weight`) to the existing per-layer scoring shape
/// consumed by [`allocate_bits_by_sensitivity`].  Lets the calibrator
/// keep its bit-allocation pipeline unchanged while swapping the
/// sensitivity-source backend (variance heuristic →
/// per-Linear FD).
///
/// # Aggregation
///
/// Per-layer score = **sum** of |fd_score| across all Linears whose
/// name matches `blk.{i}.*`.  Sum (not max) because:
/// - DWQ bit allocation cares about cumulative quantization-induced
///   information loss in a layer; max would only see the worst
///   single Linear and ignore co-quantization synergy across
///   Q/K/V/O/gate/up/down.
/// - mlx-lm's `dynamic_quant.estimate_sensitivities` aggregates
///   per-tensor scores via tree_flatten then sorts at the tensor
///   level — equivalent to summation when projected to layer
///   granularity.
/// - Absolute value: sensitivity sign reflects which bit-pair member
///   is "more accurate" (per the formula `(grad · (W_low - W_high))
///   / params_M`); for ranking PURPOSES, sign cancels in the
///   per-layer aggregate (a layer with one Linear strongly
///   preferring high-bits and another strongly preferring low-bits
///   IS still sensitive).  Caller-side bit-allocation chooses the
///   higher-bits side via `allocate_bits_by_sensitivity`.
///
/// Layers with NO matching Linears in `scores` (e.g. partial
/// coverage) get `score = 0.0` and are flagged by `variance = -1.0`
/// as a sentinel for downstream code to detect missing-layer cases
/// (sentinel because the heuristic path's `variance` is always ≥ 0).
///
/// # Errors
///
/// `Err` if any name in `scores` doesn't match the `blk.{i}.*` pattern
/// (silently dropping unparsed names would lose sensitivity data).
pub fn aggregate_fd_scores_per_layer(
    scores: &std::collections::BTreeMap<String, f32>,
    num_layers: usize,
) -> Result<Vec<LayerSensitivity>> {
    let mut layer_acc = vec![0.0f64; num_layers];
    let mut layer_seen = vec![false; num_layers];

    for (name, score) in scores {
        let layer_idx = parse_layer_index(name)?;
        if layer_idx >= num_layers {
            return Err(anyhow::anyhow!(
                "aggregate_fd_scores_per_layer: name {name:?} has layer_idx {layer_idx} \
                 >= num_layers {num_layers}"
            ));
        }
        layer_acc[layer_idx] += (*score as f64).abs();
        layer_seen[layer_idx] = true;
    }

    let out = (0..num_layers)
        .map(|i| LayerSensitivity {
            layer_index: i,
            variance: if layer_seen[i] { 0.0 } else { -1.0 },
            max_magnitude: 0.0,
            score: layer_acc[i],
        })
        .collect();
    Ok(out)
}

/// Parse `blk.{i}.{tensor_path}.weight` → `i`.  Used by
/// [`aggregate_fd_scores_per_layer`].
fn parse_layer_index(name: &str) -> Result<usize> {
    let after_blk = name
        .strip_prefix("blk.")
        .ok_or_else(|| anyhow::anyhow!("name {name:?} doesn't start with 'blk.'"))?;
    let dot = after_blk
        .find('.')
        .ok_or_else(|| anyhow::anyhow!("name {name:?} has no dot after 'blk.<idx>'"))?;
    let idx_str = &after_blk[..dot];
    idx_str
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("name {name:?} has non-integer layer index {idx_str:?}"))
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

    /// ADR-020 iter-12b-3 — `aggregate_fd_scores_per_layer` sums
    /// per-Linear FD scores into per-layer LayerSensitivity values.
    #[test]
    fn aggregate_fd_scores_per_layer_sums_within_layer() {
        use std::collections::BTreeMap;

        let mut scores = BTreeMap::new();
        // Layer 0: 7 Linears with high scores (sum = 14.0).
        scores.insert("blk.0.attn_q.weight".into(), 2.0f32);
        scores.insert("blk.0.attn_k.weight".into(), 2.0);
        scores.insert("blk.0.attn_v.weight".into(), 2.0);
        scores.insert("blk.0.attn_output.weight".into(), 2.0);
        scores.insert("blk.0.ffn_gate.weight".into(), 2.0);
        scores.insert("blk.0.ffn_up.weight".into(), 2.0);
        scores.insert("blk.0.ffn_down.weight".into(), 2.0);
        // Layer 1: 7 Linears with low scores (sum = 0.7).
        scores.insert("blk.1.attn_q.weight".into(), 0.1);
        scores.insert("blk.1.attn_k.weight".into(), 0.1);
        scores.insert("blk.1.attn_v.weight".into(), 0.1);
        scores.insert("blk.1.attn_output.weight".into(), 0.1);
        scores.insert("blk.1.ffn_gate.weight".into(), 0.1);
        scores.insert("blk.1.ffn_up.weight".into(), 0.1);
        scores.insert("blk.1.ffn_down.weight".into(), 0.1);

        let aggregated = aggregate_fd_scores_per_layer(&scores, 2).unwrap();
        assert_eq!(aggregated.len(), 2);
        assert_eq!(aggregated[0].layer_index, 0);
        assert_eq!(aggregated[1].layer_index, 1);
        // Sum-aggregation: 7 × 2.0 = 14.0; 7 × 0.1 = 0.7.
        assert!((aggregated[0].score - 14.0).abs() < 1e-4);
        assert!((aggregated[1].score - 0.7).abs() < 1e-4);
        // Layer with sensitivity data → variance sentinel = 0 (not -1).
        assert_eq!(aggregated[0].variance, 0.0);
        assert_eq!(aggregated[1].variance, 0.0);

        // Bit allocation must promote the high-score layer to sensitive.
        let bits = allocate_bits_by_sensitivity(&aggregated, 4, 8);
        assert_eq!(bits.len(), 2);
        assert!(bits[0] >= bits[1], "high-score layer should get >= bits");
        assert_eq!(bits[0], 8, "high-score layer should land at sensitive_bits");
        assert_eq!(bits[1], 4, "low-score layer should land at base_bits");
    }

    /// ADR-020 iter-12b-3 — absolute value: a layer where some
    /// Linears' scores are negative still aggregates as positive
    /// magnitude sum.  Sensitivity is "how much does this layer
    /// matter", not "which direction does it lean".
    #[test]
    fn aggregate_fd_scores_per_layer_uses_absolute_value() {
        use std::collections::BTreeMap;

        let mut scores = BTreeMap::new();
        scores.insert("blk.0.attn_q.weight".into(), 1.0f32);
        scores.insert("blk.0.attn_k.weight".into(), -1.5);
        scores.insert("blk.0.attn_v.weight".into(), 0.5);

        let aggregated = aggregate_fd_scores_per_layer(&scores, 1).unwrap();
        // |1.0| + |-1.5| + |0.5| = 3.0
        assert!((aggregated[0].score - 3.0).abs() < 1e-4);
    }

    /// ADR-020 iter-12b-3 — partial coverage: layers without any
    /// scoring entries get sentinel `variance = -1.0` so downstream
    /// can detect missing data without misinterpreting `score = 0`
    /// as "this layer is insensitive".
    #[test]
    fn aggregate_fd_scores_per_layer_flags_missing_layers() {
        use std::collections::BTreeMap;

        let mut scores = BTreeMap::new();
        scores.insert("blk.0.attn_q.weight".into(), 1.0f32);
        // Layer 1 has NO scoring entries.

        let aggregated = aggregate_fd_scores_per_layer(&scores, 2).unwrap();
        assert_eq!(aggregated.len(), 2);
        assert_eq!(aggregated[0].variance, 0.0, "layer 0 has data");
        assert_eq!(aggregated[1].variance, -1.0, "layer 1 missing → sentinel");
        assert_eq!(aggregated[1].score, 0.0, "missing layer score = 0");
    }

    /// ADR-020 iter-12b-3 — input parsing: malformed names Err loudly,
    /// out-of-range layer indices Err loudly.  Falsifier for
    /// "silently drop unparsed names" anti-pattern.
    #[test]
    fn aggregate_fd_scores_per_layer_rejects_malformed_names() {
        use std::collections::BTreeMap;

        let mut bad = BTreeMap::new();
        bad.insert("not_blk.0.weight".into(), 1.0f32);
        let r = aggregate_fd_scores_per_layer(&bad, 2);
        assert!(r.is_err());
        assert!(format!("{}", r.err().unwrap()).contains("blk."));

        let mut bad2 = BTreeMap::new();
        bad2.insert("blk.99.attn_q.weight".into(), 1.0);
        let r2 = aggregate_fd_scores_per_layer(&bad2, 2);
        assert!(r2.is_err());
        assert!(
            format!("{}", r2.err().unwrap()).contains("num_layers"),
            "out-of-range index must mention num_layers"
        );
    }
}
