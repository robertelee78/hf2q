//! Sensitivity-ranking comparison harness — ADR-020 iter-11g.
//!
//! Compares two sensitivity rankings:
//! - **Gradient-Taylor** (mlx-lm `dynamic_quant.py` formula, this ADR's
//!   port): produces a per-Linear scalar
//!   `(grad · (W_low − W_high)).sum() / (numel/1e6)` via the GpuTape
//!   autograd machinery (iter 10 + 11).  Aggregated per-layer by max
//!   or sum across the 7 quantizable Linears in each transformer
//!   layer.
//! - **Variance-magnitude** (hf2q's existing DWQ-46 scorer at
//!   `src/calibrate/sensitivity.rs`): produces a per-layer scalar
//!   `sqrt(variance) · log2(1 + max_abs)` from captured hidden-state
//!   activations.
//!
//! Comparison metrics (this module):
//! - **Spearman rank correlation**: how well do the two rankings
//!   agree on the relative ORDERING of layers?  Range [-1, 1]; 1 =
//!   perfect agreement, 0 = random, -1 = perfect inversion.
//! - **Top-K overlap**: of the K most-sensitive layers per ranking,
//!   how many appear in both top-K sets?
//!
//! The "spot-check on Qwen3-0.6B-base + verify lm_head/output_norm
//! rank in top half" step from ADR-020 §8.2 iter-11g requires the
//! actual model file (not present at iter-11g time).  This module
//! ships the comparison plumbing + correctness tests; the
//! Qwen3-0.6B-base spot-check is queued as a follow-up gated on
//! model availability.

use std::collections::BTreeMap;

use anyhow::{anyhow, Result};

/// How to aggregate per-Linear gradient-Taylor scores into per-layer
/// scalars for comparison against the per-layer variance-magnitude
/// ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerAggregator {
    /// Maximum |sensitivity| across the 7 Linears in the layer.
    /// Captures "the worst-case bit-budget pressure for this layer".
    Max,
    /// Sum of |sensitivity| across the 7 Linears.  Captures the
    /// total information-loss volume per layer.
    Sum,
}

/// Aggregate per-Linear gradient-Taylor sensitivity scalars
/// (returned by `estimate_sensitivities`) into per-layer scalars
/// suitable for comparison against the per-layer variance-magnitude
/// ranker.
///
/// Looks up each layer's 7 Linears using the path-prefix
/// `format!("layer{i}.{label}")` for label ∈ `["W_q", "W_k", "W_v",
/// "W_o", "W_gate", "W_up", "W_down"]`.  Skips Linears not matching
/// this prefix (e.g. a non-layer `lm_head` entry is ignored — it has
/// no per-layer slot).
///
/// Returns `Vec<f64>` of length `n_layers`.
pub fn aggregate_per_layer_scores(
    per_linear: &BTreeMap<String, f64>,
    n_layers: usize,
    aggregator: LayerAggregator,
) -> Result<Vec<f64>> {
    if n_layers == 0 {
        return Err(anyhow!("aggregate_per_layer_scores: n_layers must be > 0"));
    }
    let labels = ["W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"];
    let mut per_layer: Vec<f64> = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let mut found = 0usize;
        let mut acc: f64 = match aggregator {
            LayerAggregator::Max => 0.0,
            LayerAggregator::Sum => 0.0,
        };
        for label in &labels {
            let key = format!("layer{i}.{label}");
            if let Some(&v) = per_linear.get(&key) {
                let abs_v = v.abs();
                match aggregator {
                    LayerAggregator::Max => {
                        if abs_v > acc {
                            acc = abs_v;
                        }
                    }
                    LayerAggregator::Sum => {
                        acc += abs_v;
                    }
                }
                found += 1;
            }
        }
        if found == 0 {
            return Err(anyhow!(
                "aggregate_per_layer_scores: layer {i} has no quantizable Linears in per_linear map"
            ));
        }
        per_layer.push(acc);
    }
    Ok(per_layer)
}

/// Spearman rank correlation coefficient between two equal-length
/// score vectors.
///
/// Computes `1 - 6·Σd_i² / (n·(n²−1))` where `d_i` is the difference
/// between the rank of `a[i]` and the rank of `b[i]` (both in
/// ascending order; tied values get the average rank).
///
/// Range `[-1, 1]`.  Returns `Err` if lengths differ or are < 2.
pub fn spearman_rank_correlation(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(anyhow!(
            "spearman: length mismatch a={} b={}",
            a.len(),
            b.len()
        ));
    }
    if a.len() < 2 {
        return Err(anyhow!("spearman: need at least 2 elements; got {}", a.len()));
    }
    let n = a.len();
    let ranks_a = average_ranks(a);
    let ranks_b = average_ranks(b);
    // Use the Pearson formula on the rank vectors — handles ties
    // correctly via average ranking.
    let mean = |v: &[f64]| -> f64 { v.iter().sum::<f64>() / (v.len() as f64) };
    let ma = mean(&ranks_a);
    let mb = mean(&ranks_b);
    let mut num = 0.0_f64;
    let mut da_sq = 0.0_f64;
    let mut db_sq = 0.0_f64;
    for i in 0..n {
        let da = ranks_a[i] - ma;
        let db = ranks_b[i] - mb;
        num += da * db;
        da_sq += da * da;
        db_sq += db * db;
    }
    let denom = (da_sq * db_sq).sqrt();
    if denom == 0.0 {
        // All values tied in at least one input — correlation undefined.
        return Err(anyhow!(
            "spearman: degenerate input — one or both rankings are entirely tied"
        ));
    }
    Ok(num / denom)
}

/// Compute average ranks (1-indexed) of `v` in ascending order.
/// Ties get the average of their tied positions.  Output preserves
/// input order.
fn average_ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (v[idx[j]] - v[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        // Positions [i, j) are tied; assign average rank.
        let avg_rank = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            ranks[idx[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Number of items in the top-K of both rankings.
///
/// "Top-K" = K highest-scoring (largest absolute) entries.  Returns
/// the size of the intersection of the two top-K index sets.
///
/// `k` must be ≤ a.len(), and a + b same length.
pub fn top_k_overlap(a: &[f64], b: &[f64], k: usize) -> Result<usize> {
    if a.len() != b.len() {
        return Err(anyhow!("top_k_overlap: length mismatch"));
    }
    if k == 0 || k > a.len() {
        return Err(anyhow!(
            "top_k_overlap: k={k} out of range [1, {}]",
            a.len()
        ));
    }
    let top_indices = |v: &[f64]| -> std::collections::BTreeSet<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        // Sort descending by absolute value.
        idx.sort_by(|&i, &j| {
            v[j].abs()
                .partial_cmp(&v[i].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx[..k].iter().copied().collect()
    };
    let set_a = top_indices(a);
    let set_b = top_indices(b);
    Ok(set_a.intersection(&set_b).count())
}

/// Position (0-indexed, ascending) of `index` within the ranking
/// `scores` sorted by descending magnitude.  Layer 0 is the
/// most-sensitive layer.  Returns `Err` if `index` is out of range.
pub fn rank_position(scores: &[f64], index: usize) -> Result<usize> {
    if index >= scores.len() {
        return Err(anyhow!(
            "rank_position: index {index} ≥ scores.len() = {}",
            scores.len()
        ));
    }
    let target = scores[index].abs();
    // Count how many entries have STRICTLY larger magnitude.
    let strict_above = scores
        .iter()
        .filter(|&&s| s.abs() > target + 1e-12)
        .count();
    Ok(strict_above)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_per_linear(per_layer: &[[f64; 7]], extra: &[(&str, f64)]) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        let labels = ["W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"];
        for (i, scores) in per_layer.iter().enumerate() {
            for (label, score) in labels.iter().zip(scores.iter()) {
                m.insert(format!("layer{i}.{label}"), *score);
            }
        }
        for (k, v) in extra {
            m.insert((*k).to_string(), *v);
        }
        m
    }

    #[test]
    fn aggregate_max_picks_largest_abs_per_layer() {
        let per_linear = build_per_linear(
            &[
                [0.1, -0.5, 0.3, 0.2, -0.7, 0.4, 0.05], // layer 0 max abs = 0.7
                [0.9, 0.1, -0.05, 0.2, 0.3, 0.4, 0.5],  // layer 1 max abs = 0.9
            ],
            &[("lm_head", 1.5)], // out-of-layer; ignored by aggregator
        );
        let agg =
            aggregate_per_layer_scores(&per_linear, 2, LayerAggregator::Max).unwrap();
        assert_eq!(agg.len(), 2);
        assert!((agg[0] - 0.7).abs() < 1e-9, "layer 0 max: {}", agg[0]);
        assert!((agg[1] - 0.9).abs() < 1e-9, "layer 1 max: {}", agg[1]);
    }

    #[test]
    fn aggregate_sum_sums_abs_per_layer() {
        let per_linear = build_per_linear(
            &[[0.1, -0.5, 0.3, 0.2, -0.7, 0.4, 0.05]],
            &[],
        );
        let agg =
            aggregate_per_layer_scores(&per_linear, 1, LayerAggregator::Sum).unwrap();
        let expected = 0.1 + 0.5 + 0.3 + 0.2 + 0.7 + 0.4 + 0.05;
        assert!((agg[0] - expected).abs() < 1e-9);
    }

    #[test]
    fn aggregate_missing_layer_errors() {
        let per_linear = build_per_linear(&[[0.1; 7]], &[]);
        // Asking for 3 layers but only 1 in map.
        let err =
            aggregate_per_layer_scores(&per_linear, 3, LayerAggregator::Max).unwrap_err();
        assert!(format!("{err}").contains("layer 1"));
    }

    #[test]
    fn aggregate_zero_layers_errors() {
        let per_linear = build_per_linear(&[[0.1; 7]], &[]);
        let err =
            aggregate_per_layer_scores(&per_linear, 0, LayerAggregator::Max).unwrap_err();
        assert!(format!("{err}").contains("n_layers must be > 0"));
    }

    #[test]
    fn spearman_perfect_agreement() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let rho = spearman_rank_correlation(&a, &b).unwrap();
        assert!((rho - 1.0).abs() < 1e-9, "rho={rho}");
    }

    #[test]
    fn spearman_perfect_inversion() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![50.0, 40.0, 30.0, 20.0, 10.0];
        let rho = spearman_rank_correlation(&a, &b).unwrap();
        assert!((rho + 1.0).abs() < 1e-9, "rho={rho}");
    }

    #[test]
    fn spearman_zero_for_uncorrelated_pattern() {
        // a strictly increasing; b: 3, 1, 5, 2, 4 — middle ranks
        // shuffled → low correlation.  Exact formula: ranks_a =
        // [1,2,3,4,5], ranks_b = [3,1,5,2,4].  Σd_i² = 4+1+4+4+1 = 14.
        // ρ = 1 - 6·14 / (5·24) = 1 - 84/120 = 0.3.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![30.0, 10.0, 50.0, 20.0, 40.0];
        let rho = spearman_rank_correlation(&a, &b).unwrap();
        assert!((rho - 0.3).abs() < 1e-9, "rho={rho}");
    }

    #[test]
    fn spearman_handles_ties() {
        // Tied middle pair — average ranks must keep correlation correct.
        let a = vec![1.0, 2.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 20.0, 30.0];
        let rho = spearman_rank_correlation(&a, &b).unwrap();
        assert!((rho - 1.0).abs() < 1e-9, "rho={rho}");
    }

    #[test]
    fn spearman_length_mismatch_errors() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        let err = spearman_rank_correlation(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("length mismatch"));
    }

    #[test]
    fn spearman_too_short_errors() {
        let a = vec![1.0];
        let b = vec![1.0];
        let err = spearman_rank_correlation(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("at least 2"));
    }

    #[test]
    fn spearman_all_tied_errors_with_clear_message() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let err = spearman_rank_correlation(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("entirely tied"));
    }

    #[test]
    fn top_k_overlap_perfect_agreement() {
        let a = vec![5.0, 1.0, 4.0, 2.0, 3.0];
        let b = vec![10.0, 1.0, 8.0, 2.0, 4.0];
        // Top-3 indices in both = {0, 2, 4}.
        let overlap = top_k_overlap(&a, &b, 3).unwrap();
        assert_eq!(overlap, 3);
    }

    #[test]
    fn top_k_overlap_disjoint() {
        let a = vec![5.0, 4.0, 1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 5.0, 4.0, 3.0];
        // Top-2 a = {0, 1}; top-2 b = {2, 3}.
        let overlap = top_k_overlap(&a, &b, 2).unwrap();
        assert_eq!(overlap, 0);
    }

    #[test]
    fn top_k_overlap_uses_absolute_value() {
        let a = vec![5.0, -10.0, 3.0];
        let b = vec![-100.0, 1.0, 2.0];
        // top-1 a = idx 1 (|−10|=10), top-1 b = idx 0 (|−100|=100).
        // Disjoint → 0 overlap.
        let overlap = top_k_overlap(&a, &b, 1).unwrap();
        assert_eq!(overlap, 0);
    }

    #[test]
    fn top_k_overlap_k_out_of_range_errors() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0];
        assert!(top_k_overlap(&a, &b, 0).is_err());
        assert!(top_k_overlap(&a, &b, 3).is_err());
    }

    #[test]
    fn rank_position_zero_for_largest() {
        let scores = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        // Max abs = 5.0 at index 0 → rank 0.
        assert_eq!(rank_position(&scores, 0).unwrap(), 0);
    }

    #[test]
    fn rank_position_works_with_negatives() {
        let scores = vec![-5.0, 1.0, 3.0, -10.0, 4.0];
        // |scores|: 5, 1, 3, 10, 4.  Sorted desc: 10, 5, 4, 3, 1.
        // index 3 (|−10|) → rank 0.
        assert_eq!(rank_position(&scores, 3).unwrap(), 0);
        // index 0 (|−5|) → rank 1 (only |−10|=10 is strictly larger).
        assert_eq!(rank_position(&scores, 0).unwrap(), 1);
    }

    #[test]
    fn rank_position_oob_errors() {
        let scores = vec![1.0, 2.0];
        assert!(rank_position(&scores, 2).is_err());
    }

    /// End-to-end plumbing test: aggregate per-Linear gradient-Taylor
    /// scores produced by our pipeline, verify they pair correctly
    /// against synthetic per-layer variance-magnitude scores, and
    /// compute a meaningful Spearman correlation.
    #[test]
    fn end_to_end_two_ranker_comparison() {
        // 4 layers; gradient-Taylor pipeline produced these per-Linear
        // scores (synthetic — in real runs, comes from
        // estimate_sensitivities over a model).
        let per_linear = build_per_linear(
            &[
                [0.10, 0.05, 0.20, 0.15, 0.08, 0.04, 0.06], // layer 0 — least sens
                [0.50, 0.30, 0.40, 0.20, 0.35, 0.45, 0.55], // layer 1 — middle
                [0.90, 0.85, 0.95, 0.70, 0.80, 0.75, 0.65], // layer 2 — most sens
                [0.60, 0.55, 0.50, 0.65, 0.45, 0.40, 0.35], // layer 3 — middle
            ],
            &[],
        );
        let grad_taylor =
            aggregate_per_layer_scores(&per_linear, 4, LayerAggregator::Max).unwrap();
        // Variance-magnitude (synthetic): roughly correlated with grad_taylor.
        let variance_mag = vec![0.5, 1.5, 3.0, 2.0];
        let rho =
            spearman_rank_correlation(&grad_taylor, &variance_mag).unwrap();
        // Both rankings agree: layer 2 most, layer 0 least, layers 1 + 3 middle.
        // Perfect agreement → ρ = 1.
        assert!(
            (rho - 1.0).abs() < 1e-9,
            "expected perfect rank agreement; got ρ={rho}"
        );

        // Top-2: both rankings have layers {2, 3} as top.
        let overlap = top_k_overlap(&grad_taylor, &variance_mag, 2).unwrap();
        assert_eq!(overlap, 2);

        // rank_position: layer 2 is rank 0 in both rankings.
        assert_eq!(rank_position(&grad_taylor, 2).unwrap(), 0);
        assert_eq!(rank_position(&variance_mag, 2).unwrap(), 0);
    }
}
