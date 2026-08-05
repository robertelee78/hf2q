//! Exact CPU reference for DeepSeek-V4 MoE routing and clamped SwiGLU.
//!
//! The production Metal kernels are checked against these small, deterministic
//! routines. Selection bias affects only expert choice; mixture weights always
//! come from the unbiased scores, matching the official 0731 implementation.

use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoreFunction {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}

#[derive(Clone, Copy, Debug)]
pub struct RoutingConfig {
    pub expert_count: usize,
    pub top_k: usize,
    pub route_scale: f32,
    pub score_function: ScoreFunction,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum RoutingError {
    #[error("expected {expected} router logits, got {actual}")]
    LogitCount { expected: usize, actual: usize },
    #[error("expected {expected} router bias values, got {actual}")]
    BiasCount { expected: usize, actual: usize },
    #[error("top_k must be in 1..={expert_count}, got {top_k}")]
    InvalidTopK { top_k: usize, expert_count: usize },
    #[error("router value at index {index} is not finite")]
    NonFinite { index: usize },
    #[error("hash route must select exactly {expected} experts, got {actual}")]
    HashWidth { expected: usize, actual: usize },
    #[error("hash route expert {expert} is outside 0..{expert_count}")]
    HashExpert { expert: usize, expert_count: usize },
    #[error("router score normalization produced a zero or non-finite sum")]
    InvalidWeightSum,
}

fn validate(config: RoutingConfig, logits: &[f32]) -> Result<(), RoutingError> {
    if config.top_k == 0 || config.top_k > config.expert_count {
        return Err(RoutingError::InvalidTopK {
            top_k: config.top_k,
            expert_count: config.expert_count,
        });
    }
    if logits.len() != config.expert_count {
        return Err(RoutingError::LogitCount {
            expected: config.expert_count,
            actual: logits.len(),
        });
    }
    if let Some((index, _)) = logits
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(RoutingError::NonFinite { index });
    }
    Ok(())
}

fn stable_softplus(value: f32) -> f32 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

fn scored_logits(logits: &[f32], score_function: ScoreFunction) -> Vec<f32> {
    match score_function {
        ScoreFunction::Softmax => {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut scores: Vec<f32> = logits.iter().map(|value| (*value - max).exp()).collect();
            let sum: f32 = scores.iter().sum();
            for score in &mut scores {
                *score /= sum;
            }
            scores
        }
        ScoreFunction::Sigmoid => logits
            .iter()
            .map(|value| 1.0 / (1.0 + (-*value).exp()))
            .collect(),
        ScoreFunction::SqrtSoftplus => logits
            .iter()
            .map(|value| stable_softplus(*value).sqrt())
            .collect(),
    }
}

fn weights_for_indices(
    config: RoutingConfig,
    original_scores: &[f32],
    indices: Vec<usize>,
) -> Result<Route, RoutingError> {
    let mut weights: Vec<f32> = indices
        .iter()
        .map(|index| original_scores[*index])
        .collect();
    if config.score_function != ScoreFunction::Softmax {
        let sum: f32 = weights.iter().sum();
        if !sum.is_finite() || sum == 0.0 {
            return Err(RoutingError::InvalidWeightSum);
        }
        for weight in &mut weights {
            *weight /= sum;
        }
    }
    for weight in &mut weights {
        *weight *= config.route_scale;
    }
    Ok(Route { indices, weights })
}

/// Score-based routing used after the hash-routed prefix layers.
pub fn route_by_score(
    config: RoutingConfig,
    logits: &[f32],
    selection_bias: Option<&[f32]>,
) -> Result<Route, RoutingError> {
    validate(config, logits)?;
    if let Some(bias) = selection_bias {
        if bias.len() != config.expert_count {
            return Err(RoutingError::BiasCount {
                expected: config.expert_count,
                actual: bias.len(),
            });
        }
        if let Some((index, _)) = bias
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(RoutingError::NonFinite { index });
        }
    }

    let original_scores = scored_logits(logits, config.score_function);
    let mut indices: Vec<usize> = (0..config.expert_count).collect();
    indices.sort_unstable_by(|left, right| {
        let left_score = original_scores[*left] + selection_bias.map_or(0.0, |bias| bias[*left]);
        let right_score = original_scores[*right] + selection_bias.map_or(0.0, |bias| bias[*right]);
        right_score
            .total_cmp(&left_score)
            .then_with(|| left.cmp(right))
    });
    indices.truncate(config.top_k);
    weights_for_indices(config, &original_scores, indices)
}

/// Hash-based routing used by the first `n_hash_layers` layers. The supplied
/// indices are the row selected from the checkpoint's integer `tid2eid` table.
pub fn route_by_hash(
    config: RoutingConfig,
    logits: &[f32],
    hash_indices: &[usize],
) -> Result<Route, RoutingError> {
    validate(config, logits)?;
    if hash_indices.len() != config.top_k {
        return Err(RoutingError::HashWidth {
            expected: config.top_k,
            actual: hash_indices.len(),
        });
    }
    for &expert in hash_indices {
        if expert >= config.expert_count {
            return Err(RoutingError::HashExpert {
                expert,
                expert_count: config.expert_count,
            });
        }
    }
    let original_scores = scored_logits(logits, config.score_function);
    weights_for_indices(config, &original_scores, hash_indices.to_vec())
}

/// DeepSeek-V4's bounded SwiGLU activation. The gate is upper-clamped only;
/// the up projection is clamped symmetrically before multiplication.
pub fn clamped_swiglu(gate: f32, up: f32, limit: f32) -> f32 {
    let (gate, up) = if limit > 0.0 {
        (gate.min(limit), up.clamp(-limit, limit))
    } else {
        (gate, up)
    };
    (gate / (1.0 + (-gate).exp())) * up
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dsv4_config() -> RoutingConfig {
        RoutingConfig {
            expert_count: 8,
            top_k: 3,
            route_scale: 1.5,
            score_function: ScoreFunction::SqrtSoftplus,
        }
    }

    #[test]
    fn bias_changes_selection_but_not_selected_weight_values() {
        let logits = [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
        let unbiased = route_by_score(dsv4_config(), &logits, None).unwrap();
        assert_eq!(unbiased.indices, vec![7, 6, 5]);

        let bias = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let biased = route_by_score(dsv4_config(), &logits, Some(&bias)).unwrap();
        assert_eq!(biased.indices, vec![0, 7, 6]);

        let hash = route_by_hash(dsv4_config(), &logits, &[0, 7, 6]).unwrap();
        for (actual, expected) in biased.weights.iter().zip(hash.weights.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!((biased.weights.iter().sum::<f32>() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn hash_route_uses_checkpoint_order_and_normalizes_sqrtsoftplus() {
        let route = route_by_hash(dsv4_config(), &[0.0; 8], &[6, 2, 5]).unwrap();
        assert_eq!(route.indices, vec![6, 2, 5]);
        for weight in route.weights {
            assert!((weight - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_keeps_global_probability_mass_before_route_scaling() {
        let config = RoutingConfig {
            expert_count: 4,
            top_k: 2,
            route_scale: 2.0,
            score_function: ScoreFunction::Softmax,
        };
        let route = route_by_score(config, &[0.0, 1.0, 2.0, 3.0], None).unwrap();
        assert_eq!(route.indices, vec![3, 2]);
        let selected_mass = route.weights.iter().sum::<f32>();
        assert!(selected_mass < config.route_scale);
        assert!(selected_mass > 1.0);
    }

    #[test]
    fn clamped_swiglu_matches_asymmetric_official_rule() {
        let expected = (10.0 / (1.0 + (-10.0_f32).exp())) * -10.0;
        assert!((clamped_swiglu(20.0, -20.0, 10.0) - expected).abs() < 1e-5);
        assert!(clamped_swiglu(-20.0, 20.0, 10.0).abs() < 1e-6);
    }

    #[test]
    fn malformed_routes_fail_closed() {
        let error = route_by_hash(dsv4_config(), &[0.0; 8], &[0, 1]).unwrap_err();
        assert_eq!(
            error,
            RoutingError::HashWidth {
                expected: 3,
                actual: 2
            }
        );
        let error = route_by_score(dsv4_config(), &[0.0; 7], None).unwrap_err();
        assert_eq!(
            error,
            RoutingError::LogitCount {
                expected: 8,
                actual: 7
            }
        );
    }
}
