//! ADR-037 Phase E4a — Drafter abstraction for dynamic tree expansion.
//!
//! Defines the `Drafter` trait that `dynamic_tree::expand_dynamic_tree`
//! consumes. The trait is intentionally minimal:
//!
//! ```text
//! given a parent tree-node-idx and a top-K budget,
//! return Vec<(token, log_prob)> for the K most-likely next tokens.
//! ```
//!
//! The trait makes the tree-expansion algorithm independently testable
//! without a GPU drafter forward (Phase E4b). The real GPU drafter
//! will implement this trait once Phase E4b ships.
//!
//! ## Why parent_node_idx and not a token path
//!
//! EAGLE-3 drafters maintain their OWN KV cache (separate from the
//! target's). The cache is keyed by tree-node-idx so the drafter can
//! efficiently reuse the cached state along the parent chain rather
//! than re-running the full path on every call. The trait API
//! reflects this — the drafter implementation knows how to look up
//! the cached state for `parent_node_idx`.
//!
//! ## Numerical contract
//!
//! - Returned `log_prob` values MUST be finite (no NaN, no +/-inf).
//!   Drafter implementations should clamp at the boundary (e.g.
//!   replace -inf with `LOG_PROB_FLOOR`). The expansion algorithm
//!   uses these as keys in a priority queue; NaN would corrupt the
//!   heap invariant.
//! - Returned tokens MUST be unique within a single `predict_topk` call.
//!   Duplicates within the top-K would create incoherent tree branches.

use anyhow::{ensure, Result};

/// One candidate next-token prediction from the drafter.
///
/// `log_prob` is the natural log of the drafter's posterior P(token |
/// path), NOT the raw logit. Must be finite (asserted at use sites).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftCandidate {
    pub token: u32,
    pub log_prob: f32,
}

/// Minimum log-prob floor — any drafter implementation that would emit
/// `log_prob == -inf` (e.g. from `log(0)`) must clamp at this value.
/// `f32::MIN / 2` is finite, deeply negative, and survives downstream
/// `f64` accumulation without overflow.
pub const LOG_PROB_FLOOR: f32 = f32::MIN / 2.0;

/// Drafter abstraction consumed by dynamic-tree expansion.
///
/// `&mut self` so implementations can maintain a per-call KV cache.
pub trait Drafter {
    /// Predict the top-K next tokens given the parent tree-node-idx.
    ///
    /// For the ROOT call (`parent_node_idx == 0`, depth 0), the
    /// drafter conditions on the target model's hidden state at the
    /// most-recently-generated token.
    ///
    /// For subsequent calls, the drafter conditions on the parent
    /// node's KV state plus the parent's token id.
    ///
    /// Returns up to `top_k` candidates ordered by `log_prob`
    /// descending. May return fewer than `top_k` if the drafter's
    /// effective vocabulary is smaller (rare in practice).
    fn predict_topk(
        &mut self,
        parent_node_idx: usize,
        top_k: usize,
    ) -> Result<Vec<DraftCandidate>>;
}

/// Validate a `predict_topk` result. Used by the expansion algorithm
/// at every drafter call site so a buggy drafter implementation
/// fails fast with a clear diagnostic rather than corrupting the
/// priority queue.
pub fn validate_candidates(candidates: &[DraftCandidate], top_k: usize) -> Result<()> {
    ensure!(
        candidates.len() <= top_k,
        "drafter returned {} candidates, exceeds top_k {}",
        candidates.len(),
        top_k
    );
    // Finite-log-prob contract.
    for (i, c) in candidates.iter().enumerate() {
        ensure!(
            c.log_prob.is_finite(),
            "candidate[{}].log_prob = {} is not finite",
            i,
            c.log_prob
        );
    }
    // Unique tokens.
    let mut seen = std::collections::HashSet::with_capacity(candidates.len());
    for (i, c) in candidates.iter().enumerate() {
        ensure!(
            seen.insert(c.token),
            "candidate[{}].token = {} duplicated in top-K",
            i,
            c.token
        );
    }
    // Sorted descending by log_prob (drafter contract — required so
    // best-first expansion can short-circuit on budget exhaustion).
    for w in candidates.windows(2) {
        ensure!(
            w[0].log_prob >= w[1].log_prob,
            "candidates must be sorted descending by log_prob, got {} then {}",
            w[0].log_prob,
            w[1].log_prob
        );
    }
    Ok(())
}

/// Deterministic mock drafter for tree-expansion algorithm tests.
///
/// Behavior: at node `i`, returns top-K candidates with tokens
/// `((i * 1000 + j) % vocab_size, base_log_prob - j * 0.5)` for j in
/// 0..top_k. The slope `-0.5` makes deeper paths exponentially less
/// likely, exercising the priority queue's ordering. The token
/// formula gives unique tokens per node.
///
/// To create biased trees (where some paths cumulatively score higher
/// than others), use [`BiasedMockDrafter`] instead.
#[derive(Debug, Clone)]
pub struct MockDrafter {
    pub vocab_size: u32,
    pub base_log_prob: f32,
    pub log_prob_slope: f32,
}

impl Default for MockDrafter {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            base_log_prob: -0.5,
            log_prob_slope: -0.5,
        }
    }
}

impl Drafter for MockDrafter {
    fn predict_topk(
        &mut self,
        parent_node_idx: usize,
        top_k: usize,
    ) -> Result<Vec<DraftCandidate>> {
        let mut out = Vec::with_capacity(top_k);
        for j in 0..top_k {
            let token = ((parent_node_idx * 1000 + j) as u32) % self.vocab_size;
            let log_prob = self.base_log_prob + (j as f32) * self.log_prob_slope;
            out.push(DraftCandidate { token, log_prob });
        }
        Ok(out)
    }
}

/// Biased mock drafter — produces non-uniform `log_prob` so the
/// priority queue's ordering is empirically exercised.
///
/// At node `i`, candidate `j` gets:
///   `log_prob = if i in bias_nodes then -0.1 else (base_log_prob + j * slope)`
///
/// So `bias_nodes` get a SHALLOW slope and quickly accumulate higher
/// cumulative log-prob, while non-biased nodes follow the normal
/// steep slope. This makes the dynamic tree EXPAND deeper down the
/// biased subtree (the EAGLE-2 intuition).
#[derive(Debug, Clone)]
pub struct BiasedMockDrafter {
    pub vocab_size: u32,
    pub base_log_prob: f32,
    pub log_prob_slope: f32,
    pub bias_nodes: std::collections::HashSet<usize>,
}

impl Drafter for BiasedMockDrafter {
    fn predict_topk(
        &mut self,
        parent_node_idx: usize,
        top_k: usize,
    ) -> Result<Vec<DraftCandidate>> {
        let mut out = Vec::with_capacity(top_k);
        let is_biased = self.bias_nodes.contains(&parent_node_idx);
        for j in 0..top_k {
            let token = ((parent_node_idx * 1000 + j + 7) as u32) % self.vocab_size;
            let log_prob = if is_biased {
                -0.1 + (j as f32) * (-0.05)
            } else {
                self.base_log_prob + (j as f32) * self.log_prob_slope
            };
            out.push(DraftCandidate { token, log_prob });
        }
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn adr_037_e4a_validate_rejects_nan_log_prob_2026_05_22() {
        let bad = vec![DraftCandidate {
            token: 10,
            log_prob: f32::NAN,
        }];
        let err = validate_candidates(&bad, 1).unwrap_err().to_string();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn adr_037_e4a_validate_rejects_inf_log_prob_2026_05_22() {
        let bad = vec![DraftCandidate {
            token: 10,
            log_prob: f32::NEG_INFINITY,
        }];
        let err = validate_candidates(&bad, 1).unwrap_err().to_string();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn adr_037_e4a_validate_rejects_duplicate_tokens_2026_05_22() {
        let bad = vec![
            DraftCandidate { token: 10, log_prob: -0.5 },
            DraftCandidate { token: 10, log_prob: -1.0 },
        ];
        let err = validate_candidates(&bad, 2).unwrap_err().to_string();
        assert!(err.contains("duplicated in top-K"), "got: {err}");
    }

    #[test]
    fn adr_037_e4a_validate_rejects_unsorted_2026_05_22() {
        let bad = vec![
            DraftCandidate { token: 10, log_prob: -1.0 },
            DraftCandidate { token: 11, log_prob: -0.5 }, // higher prob after lower
        ];
        let err = validate_candidates(&bad, 2).unwrap_err().to_string();
        assert!(err.contains("sorted descending"), "got: {err}");
    }

    #[test]
    fn adr_037_e4a_validate_rejects_too_many_candidates_2026_05_22() {
        let bad = vec![
            DraftCandidate { token: 10, log_prob: -0.5 },
            DraftCandidate { token: 11, log_prob: -1.0 },
            DraftCandidate { token: 12, log_prob: -1.5 },
        ];
        let err = validate_candidates(&bad, 2).unwrap_err().to_string();
        assert!(err.contains("exceeds top_k"), "got: {err}");
    }

    #[test]
    fn adr_037_e4a_mock_drafter_produces_unique_descending_candidates_2026_05_22() {
        let mut d = MockDrafter::default();
        let cands = d.predict_topk(0, 4).unwrap();
        validate_candidates(&cands, 4).expect("mock must be valid");
        assert_eq!(cands.len(), 4);
    }

    #[test]
    fn adr_037_e4a_mock_drafter_uses_parent_idx_in_token_2026_05_22() {
        let mut d = MockDrafter::default();
        let c0 = d.predict_topk(0, 1).unwrap();
        let c5 = d.predict_topk(5, 1).unwrap();
        assert_ne!(c0[0].token, c5[0].token, "different parent → different token");
    }

    #[test]
    fn adr_037_e4a_biased_drafter_shallower_slope_at_bias_nodes_2026_05_22() {
        let mut bias_nodes = std::collections::HashSet::new();
        bias_nodes.insert(3);
        let mut d = BiasedMockDrafter {
            vocab_size: 1000,
            base_log_prob: -0.5,
            log_prob_slope: -0.5,
            bias_nodes,
        };
        let unbiased = d.predict_topk(0, 2).unwrap();
        let biased = d.predict_topk(3, 2).unwrap();
        // Biased path's top-1 log_prob (−0.1) > unbiased top-1 (−0.5).
        assert!(biased[0].log_prob > unbiased[0].log_prob);
        validate_candidates(&biased, 2).unwrap();
    }

    #[test]
    fn adr_037_e4a_log_prob_floor_is_finite_2026_05_22() {
        // Floor must be finite so it survives validate_candidates.
        assert!(LOG_PROB_FLOOR.is_finite());
        // And deeply negative so it never wins the priority queue.
        assert!(LOG_PROB_FLOOR < -1e30);
    }
}
