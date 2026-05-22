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

/// Borrowed view of the tree-so-far, passed to the drafter.
///
/// Codex /cfa E4a Major (2026-05-22): the drafter needs to know the
/// PATH from root to the node it's expanding (so it can compute the
/// correct hidden state via its own KV cache). Passing parent_node_idx
/// alone is insufficient — the drafter would need an out-of-band
/// way to know which token was assigned to which tree-node-idx. This
/// view bundles the info the drafter needs to walk the parent chain.
#[derive(Debug, Clone, Copy)]
pub struct TreeContextView<'a> {
    /// Token id at each committed tree-node-idx, in commit order.
    /// `tokens[0]` is the root token.
    pub tokens: &'a [u32],
    /// Parent tree-node-idx for each committed node; `None` for root.
    pub parents: &'a [Option<usize>],
}

impl<'a> TreeContextView<'a> {
    /// Walk from `node_idx` up to root, returning tokens along the
    /// path (root first, then walk down to `node_idx`).
    pub fn path_tokens(&self, node_idx: usize) -> Vec<u32> {
        let mut rev: Vec<u32> = Vec::new();
        let mut cur = Some(node_idx);
        while let Some(i) = cur {
            rev.push(self.tokens[i]);
            cur = self.parents[i];
        }
        rev.reverse();
        rev
    }
}

/// Drafter abstraction consumed by dynamic-tree expansion.
///
/// `&mut self` so implementations can maintain a per-call KV cache.
pub trait Drafter {
    /// Predict the top-K next tokens that would follow `node_to_expand`
    /// in the current tree.
    ///
    /// `tree` provides the committed tree-so-far (tokens + parents) so
    /// the drafter can walk the parent chain to determine the
    /// conditioning context. `node_to_expand` is the node whose
    /// CHILDREN we want to predict.
    ///
    /// For the ROOT call (`node_to_expand == 0`, depth 0), the drafter
    /// conditions on the target model's hidden state at the
    /// most-recently-generated token (tokens[0]).
    ///
    /// Returns up to `top_k` candidates ordered by `log_prob`
    /// descending. May return fewer than `top_k` if the drafter's
    /// effective vocabulary is smaller (rare in practice).
    fn predict_topk(
        &mut self,
        tree: TreeContextView<'_>,
        node_to_expand: usize,
        top_k: usize,
    ) -> Result<Vec<DraftCandidate>>;
}

/// ADR-037 Phase E4b.10a (2026-05-22) — CPU-side top-K extraction
/// from F32 logits. Used by GPU drafter implementations after
/// downloading the lm_head output to convert logits → top-K
/// DraftCandidates conforming to the Phase E4a contract.
///
/// Algorithm:
/// 1. Compute log_softmax for numerical-stable log probabilities:
///        max_logit = max(logits)
///        log_sumexp = max_logit + log(sum(exp(logits - max_logit)))
///        log_prob[i] = logits[i] - log_sumexp
/// 2. Partial sort by log_prob descending via BinaryHeap of size K.
/// 3. Clamp any extreme log_probs at `LOG_PROB_FLOOR` so the
///    expansion algorithm's finite-check (per `validate_candidates`)
///    passes even at deep-tail tokens with log_prob ≈ -inf.
///
/// Complexity: O(V log K) where V = vocab_size, K = top_k.
///
/// # Errors
/// Returns `Err` if:
/// - `row_logits` is empty
/// - `top_k` is 0
/// - any logit is non-finite (NaN/inf — drafter contract requires
///   finite inputs; the lm_head GEMM should not produce these on
///   sane weights)
///
/// # Returns
/// Up to `top_k` candidates ordered by log_prob descending,
/// passing `validate_candidates`. May return fewer than `top_k`
/// if `row_logits.len() < top_k`.
pub fn extract_top_k_from_row_logits(
    row_logits: &[f32],
    top_k: usize,
) -> Result<Vec<DraftCandidate>> {
    ensure!(
        !row_logits.is_empty(),
        "extract_top_k: row_logits is empty"
    );
    ensure!(top_k > 0, "extract_top_k: top_k must be > 0");

    // Validate finite logits (drafter contract).
    for (i, &v) in row_logits.iter().enumerate() {
        ensure!(
            v.is_finite(),
            "extract_top_k: row_logits[{}] = {} is not finite",
            i,
            v
        );
    }

    // Reject vocab sizes that don't fit in u32 (DraftCandidate.token is u32).
    ensure!(
        row_logits.len() <= (u32::MAX as usize),
        "extract_top_k: row_logits.len() ({}) exceeds u32::MAX",
        row_logits.len()
    );

    let effective_k = top_k.min(row_logits.len());

    // Compute log_sumexp in f64 for stability.
    let max_logit: f32 = row_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp: f64 = 0.0;
    for &v in row_logits.iter() {
        sum_exp += ((v - max_logit) as f64).exp();
    }
    // sum_exp >= 1 (since e^0 = 1 contributed by max_logit).
    let log_sumexp = (max_logit as f64) + sum_exp.ln();

    // Partial sort via min-heap of size K: pop smallest each iteration
    // when heap is full and new item is larger. `BinaryHeap` is a
    // max-heap by default; wrap in `Reverse` to make it a min-heap.
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    /// Local comparable wrapper so f32 can be ordered via total_cmp
    /// (handles NaN already rejected; this is for the deterministic
    /// tiebreaker by token id when log_probs are equal).
    #[derive(Debug, Clone, Copy)]
    struct LogProbToken {
        log_prob: f32,
        token: u32,
    }
    impl PartialEq for LogProbToken {
        fn eq(&self, other: &Self) -> bool {
            self.log_prob == other.log_prob && self.token == other.token
        }
    }
    impl Eq for LogProbToken {}
    impl Ord for LogProbToken {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            // Order by log_prob ASC for min-heap semantics. Tiebreak
            // by SMALLER token first (deterministic, matches typical
            // argmax behavior).
            self.log_prob
                .partial_cmp(&other.log_prob)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| other.token.cmp(&self.token))
        }
    }
    impl PartialOrd for LogProbToken {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut heap: BinaryHeap<Reverse<LogProbToken>> = BinaryHeap::with_capacity(effective_k + 1);
    for (i, &v) in row_logits.iter().enumerate() {
        let log_prob_f64 = (v as f64) - log_sumexp;
        let log_prob = if log_prob_f64.is_finite() {
            (log_prob_f64 as f32).max(LOG_PROB_FLOOR)
        } else {
            LOG_PROB_FLOOR
        };
        let candidate = LogProbToken {
            log_prob,
            token: i as u32,
        };
        if heap.len() < effective_k {
            heap.push(Reverse(candidate));
        } else if let Some(Reverse(min)) = heap.peek() {
            // Larger log_prob ALWAYS replaces; equal log_prob with
            // SMALLER token replaces (matches LogProbToken's Ord).
            if candidate > *min {
                heap.pop();
                heap.push(Reverse(candidate));
            }
        }
    }

    // Drain heap → sort descending by log_prob.
    let mut out: Vec<DraftCandidate> = heap
        .into_iter()
        .map(|Reverse(lpt)| DraftCandidate {
            token: lpt.token,
            log_prob: lpt.log_prob,
        })
        .collect();
    out.sort_by(|a, b| {
        b.log_prob
            .partial_cmp(&a.log_prob)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.token.cmp(&b.token))
    });
    Ok(out)
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
        _tree: TreeContextView<'_>,
        node_to_expand: usize,
        top_k: usize,
    ) -> Result<Vec<DraftCandidate>> {
        let mut out = Vec::with_capacity(top_k);
        for j in 0..top_k {
            let token = ((node_to_expand * 1000 + j) as u32) % self.vocab_size;
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
        _tree: TreeContextView<'_>,
        node_to_expand: usize,
        top_k: usize,
    ) -> Result<Vec<DraftCandidate>> {
        let mut out = Vec::with_capacity(top_k);
        let is_biased = self.bias_nodes.contains(&node_to_expand);
        for j in 0..top_k {
            let token = ((node_to_expand * 1000 + j + 7) as u32) % self.vocab_size;
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

    fn empty_tree_view() -> TreeContextView<'static> {
        static EMPTY_TOKENS: &[u32] = &[];
        static EMPTY_PARENTS: &[Option<usize>] = &[];
        TreeContextView {
            tokens: EMPTY_TOKENS,
            parents: EMPTY_PARENTS,
        }
    }

    #[test]
    fn adr_037_e4a_mock_drafter_produces_unique_descending_candidates_2026_05_22() {
        let mut d = MockDrafter::default();
        let cands = d.predict_topk(empty_tree_view(), 0, 4).unwrap();
        validate_candidates(&cands, 4).expect("mock must be valid");
        assert_eq!(cands.len(), 4);
    }

    #[test]
    fn adr_037_e4a_mock_drafter_uses_parent_idx_in_token_2026_05_22() {
        let mut d = MockDrafter::default();
        let c0 = d.predict_topk(empty_tree_view(), 0, 1).unwrap();
        let c5 = d.predict_topk(empty_tree_view(), 5, 1).unwrap();
        assert_ne!(c0[0].token, c5[0].token, "different node → different token");
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
        let unbiased = d.predict_topk(empty_tree_view(), 0, 2).unwrap();
        let biased = d.predict_topk(empty_tree_view(), 3, 2).unwrap();
        // Biased path's top-1 log_prob (−0.1) > unbiased top-1 (−0.5).
        assert!(biased[0].log_prob > unbiased[0].log_prob);
        validate_candidates(&biased, 2).unwrap();
    }

    #[test]
    fn adr_037_e4a_tree_context_view_path_tokens_walks_to_root_2026_05_22() {
        // Tree: 10 ─ 20 ─ 30.
        let tokens = [10, 20, 30];
        let parents: [Option<usize>; 3] = [None, Some(0), Some(1)];
        let view = TreeContextView {
            tokens: &tokens,
            parents: &parents,
        };
        assert_eq!(view.path_tokens(0), vec![10]);
        assert_eq!(view.path_tokens(1), vec![10, 20]);
        assert_eq!(view.path_tokens(2), vec![10, 20, 30]);
    }

    // ----------------------------------------------------------------
    // Phase E4b.10a tests — top-K extraction
    // ----------------------------------------------------------------

    #[test]
    fn adr_037_e4b10a_top_k_basic_descending_2026_05_22() {
        // Logits [3, 1, 2, 0]; top-3 should be (0, 3.0), (2, 2.0), (1, 1.0).
        // log_softmax base = log(e^3 + e^1 + e^2 + e^0) ≈ 3.44
        // log_probs: [3-3.44, 1-3.44, 2-3.44, 0-3.44] = [-0.44, -2.44, -1.44, -3.44]
        let logits = vec![3.0f32, 1.0, 2.0, 0.0];
        let out = extract_top_k_from_row_logits(&logits, 3).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].token, 0);
        assert_eq!(out[1].token, 2);
        assert_eq!(out[2].token, 1);
        // Validate via Phase E4a contract.
        validate_candidates(&out, 3).expect("must pass validate_candidates");
    }

    #[test]
    fn adr_037_e4b10a_top_k_log_probs_sum_via_softmax_2026_05_22() {
        // Sanity: log_softmax of [a, a, a] gives log_prob = -ln(N) for each.
        // Pick logits [0, 0, 0, 0] → log_prob = -ln(4) ≈ -1.386 for all.
        let logits = vec![0.0f32; 4];
        let out = extract_top_k_from_row_logits(&logits, 4).unwrap();
        assert_eq!(out.len(), 4);
        let expected = -(4.0f32).ln();
        for c in &out {
            assert!(
                (c.log_prob - expected).abs() < 1e-5,
                "log_prob {} != {expected}",
                c.log_prob
            );
        }
    }

    #[test]
    fn adr_037_e4b10a_top_k_returns_at_most_vocab_size_2026_05_22() {
        // top_k > vocab_size → returns vocab_size candidates.
        let logits = vec![1.0f32, 2.0, 3.0];
        let out = extract_top_k_from_row_logits(&logits, 10).unwrap();
        assert_eq!(out.len(), 3, "vocab=3 caps the output count");
        // Descending order.
        assert_eq!(out[0].token, 2);
        assert_eq!(out[1].token, 1);
        assert_eq!(out[2].token, 0);
    }

    #[test]
    fn adr_037_e4b10a_top_k_rejects_empty_logits_2026_05_22() {
        let err = extract_top_k_from_row_logits(&[], 1).unwrap_err();
        assert!(err.to_string().contains("empty"), "got: {err}");
    }

    #[test]
    fn adr_037_e4b10a_top_k_rejects_top_k_zero_2026_05_22() {
        let err = extract_top_k_from_row_logits(&[1.0, 2.0], 0).unwrap_err();
        assert!(err.to_string().contains("top_k must be > 0"), "got: {err}");
    }

    #[test]
    fn adr_037_e4b10a_top_k_rejects_nan_logit_2026_05_22() {
        let logits = vec![1.0f32, f32::NAN, 2.0];
        let err = extract_top_k_from_row_logits(&logits, 2).unwrap_err();
        assert!(err.to_string().contains("not finite"), "got: {err}");
    }

    #[test]
    fn adr_037_e4b10a_top_k_rejects_inf_logit_2026_05_22() {
        let logits = vec![1.0f32, f32::INFINITY, 2.0];
        let err = extract_top_k_from_row_logits(&logits, 2).unwrap_err();
        assert!(err.to_string().contains("not finite"), "got: {err}");
    }

    #[test]
    fn adr_037_e4b10a_top_k_deterministic_tie_break_by_smaller_token_2026_05_22() {
        // All logits equal → top-2 = (token 0, token 1) by deterministic
        // tiebreaker (smaller token wins on log_prob ties).
        let logits = vec![5.0f32; 10];
        let out = extract_top_k_from_row_logits(&logits, 2).unwrap();
        assert_eq!(out.len(), 2);
        // Per LogProbToken's Ord, smaller token "wins" → top-2 are
        // tokens 0 and 1 (smallest).
        let mut tokens: Vec<u32> = out.iter().map(|c| c.token).collect();
        tokens.sort();
        assert_eq!(tokens, vec![0, 1]);
        // Both log_probs equal -ln(10).
        let expected = -(10.0f32).ln();
        for c in &out {
            assert!((c.log_prob - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn adr_037_e4b10a_top_k_passes_validate_candidates_2026_05_22() {
        // Sanity-check the Phase E4a contract: returned candidates
        // must pass validate_candidates(top_k).
        let logits: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let out = extract_top_k_from_row_logits(&logits, 10).unwrap();
        validate_candidates(&out, 10).expect("Phase E4a contract");
    }

    #[test]
    fn adr_037_e4b10a_top_k_clamps_extreme_log_probs_at_floor_2026_05_22() {
        // Very large logit differences → extreme log_probs that
        // might underflow to -inf without clamping. Verify the
        // floor protects against this.
        // logit[0] = 100, others = 0 → log_softmax for token 0 ≈ 0;
        // for others ≈ -100. The -100 is finite so no clamp fires
        // unless log_sumexp computation collides.
        // For a true test of LOG_PROB_FLOOR, we'd need values that
        // overflow to -inf; here we just check finiteness.
        let logits = vec![100.0f32, 0.0, 0.0, 0.0];
        let out = extract_top_k_from_row_logits(&logits, 4).unwrap();
        for c in &out {
            assert!(c.log_prob.is_finite(), "log_prob must be finite");
            assert!(
                c.log_prob >= LOG_PROB_FLOOR,
                "log_prob {} below floor {}",
                c.log_prob,
                LOG_PROB_FLOOR
            );
        }
    }

    #[test]
    fn adr_037_e4a_log_prob_floor_is_finite_2026_05_22() {
        // Floor must be finite so it survives validate_candidates.
        assert!(LOG_PROB_FLOOR.is_finite());
        // And deeply negative so it never wins the priority queue.
        assert!(LOG_PROB_FLOOR < -1e30);
    }
}
