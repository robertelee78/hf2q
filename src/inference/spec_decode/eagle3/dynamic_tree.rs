//! ADR-037 Phase E4a — Dynamic tree expansion (EAGLE-2 lineage).
//!
//! Best-first tree expansion: maintain a max-heap of tree nodes
//! ordered by cumulative log-probability from the root; repeatedly
//! pop the highest-scoring node, ask the drafter for its top-K
//! children, push them onto the heap. Stop when the total tree size
//! reaches `budget` or every reachable node has been explored.
//!
//! ## Algorithm rationale
//!
//! Per the EAGLE-2 paper (Li et al. 2024b), a FIXED-shape tree wastes
//! verification budget on low-confidence branches. The dynamic
//! variant reallocates budget from low-confidence subtrees to
//! high-confidence ones. At long context — where confidence
//! distribution is skewed — this recovers accept rate that static
//! Medusa-style trees lose.
//!
//! Best-first expansion (this implementation) is one of two common
//! variants; the other is "expand-all-then-prune". Both produce the
//! same tree shape for non-pathological drafters. Best-first is
//! cheaper because it never builds nodes that will be pruned.
//!
//! ## Output contract (matches Phase E1 tree_attention kernel)
//!
//! `ExpandedTree`:
//! - `tokens[i]` = u32 token id at tree node i. Root (i=0) is the
//!   most-recently-generated target token.
//! - `parents[i]` = `Some(parent_idx)` for non-root, `None` for root.
//! - `depths[i]` = depth from root (root = 0).
//! - `cum_log_probs[i]` = sum of edge log-probs from root to node i.
//!   Root's cum_log_prob is 0.0.
//!
//! `build_tree_mask(prefix_len)` produces the [q_seq_len, mask_stride]
//! buffer the Phase E1 tree_attention kernel consumes (matches the
//! `build_tree_mask_from_parents` test helper at
//! `mlx-native/tests/test_tree_attention_e1_1_parity.rs`).

use super::drafter::{validate_candidates, Drafter};
use anyhow::{anyhow, ensure, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Configuration for [`expand_dynamic_tree`].
#[derive(Debug, Clone, Copy)]
pub struct DynamicTreeConfig {
    /// Maximum total tree nodes (including root). Default 64 per
    /// ADR-037 §4 "default 64 nodes". Must be >= 1.
    pub budget: usize,
    /// Maximum tree depth. The root is depth 0. A node at
    /// `depth == max_depth` is not expanded further. Default 8.
    pub max_depth: usize,
    /// Number of children proposed by the drafter per expansion.
    /// Default 10. Real EAGLE-3 drafters use 10-12.
    pub top_k: usize,
}

impl Default for DynamicTreeConfig {
    fn default() -> Self {
        Self {
            budget: 64,
            max_depth: 8,
            top_k: 10,
        }
    }
}

impl DynamicTreeConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.budget >= 1, "budget must be >= 1");
        ensure!(self.max_depth >= 1, "max_depth must be >= 1");
        ensure!(self.top_k >= 1, "top_k must be >= 1");
        ensure!(
            self.budget <= 8192,
            "budget {} exceeds sane upper bound 8192",
            self.budget
        );
        Ok(())
    }
}

/// Output of [`expand_dynamic_tree`].
#[derive(Debug, Clone)]
pub struct ExpandedTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<Option<usize>>,
    pub depths: Vec<usize>,
    pub cum_log_probs: Vec<f32>,
}

impl ExpandedTree {
    /// Number of nodes in the tree (>= 1; root always present).
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Build the tree-attention mask buffer.
    ///
    /// Layout matches `tree_attention.metal` contract:
    ///   `mask[(iq1 * mask_stride) + k_pos]` ∈
    ///   `{TREE_MASK_ATTENDED (0.0), TREE_MASK_MASKED (-65504.0)}`.
    ///
    /// For each tree node `i`:
    /// - Cells `[0, prefix_len)` are ATTENDED (prefix tokens are
    ///   always visible).
    /// - Cell `prefix_len + j` is ATTENDED iff `j` is the index of
    ///   node `i` or any ancestor in the tree.
    ///
    /// `mask_stride = prefix_len + tree.len()`. The buffer is sized
    /// `tree.len() * mask_stride` floats.
    pub fn build_tree_mask(&self, prefix_len: usize) -> Vec<f32> {
        const ATTENDED: f32 = 0.0;
        const MASKED: f32 = -65504.0;
        let q = self.len();
        let mask_stride = prefix_len + q;
        let mut mask = vec![MASKED; q * mask_stride];
        for iq1 in 0..q {
            let row_base = iq1 * mask_stride;
            // Prefix always attended.
            for k in 0..prefix_len {
                mask[row_base + k] = ATTENDED;
            }
            // Self + ancestors. Walk parent chain.
            let mut cur = Some(iq1);
            while let Some(node) = cur {
                mask[row_base + prefix_len + node] = ATTENDED;
                cur = self.parents[node];
            }
        }
        mask
    }

    /// Verify internal consistency (debug/test aid). Checks:
    /// - len() consistent across vecs
    /// - exactly one root (parents[i] == None for exactly one i)
    /// - root is at index 0
    /// - parent[i] < i for i > 0 (topological order)
    /// - depth[i] == depth[parent[i]] + 1
    /// - cum_log_probs are finite
    pub fn validate(&self) -> Result<()> {
        let n = self.len();
        ensure!(n >= 1, "tree is empty");
        ensure!(
            self.parents.len() == n
                && self.depths.len() == n
                && self.cum_log_probs.len() == n,
            "tree vec lengths inconsistent"
        );
        ensure!(self.parents[0].is_none(), "root (index 0) must have no parent");
        ensure!(self.depths[0] == 0, "root depth must be 0");
        ensure!(
            self.cum_log_probs[0] == 0.0,
            "root cum_log_prob must be exactly 0.0"
        );
        for i in 1..n {
            let p = self.parents[i].ok_or_else(|| {
                anyhow!("non-root node {} must have a parent", i)
            })?;
            ensure!(
                p < i,
                "parents[{}] = {} violates topological order (parent must precede child)",
                i,
                p
            );
            ensure!(
                self.depths[i] == self.depths[p] + 1,
                "depths[{}] = {} but depths[parent={}] + 1 = {}",
                i,
                self.depths[i],
                p,
                self.depths[p] + 1
            );
            ensure!(
                self.cum_log_probs[i].is_finite(),
                "cum_log_probs[{}] is not finite",
                i
            );
        }
        Ok(())
    }
}

/// Heap entry: priority is cum_log_prob (max-heap), tie-broken by
/// smaller node_idx (earlier-inserted = closer to root). The
/// tiebreaker is deterministic so test fixtures are reproducible.
#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    cum_log_prob: f64,
    node_idx: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cum_log_prob == other.cum_log_prob && self.node_idx == other.node_idx
    }
}
impl Eq for HeapEntry {}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is max-heap. We want: highest cum_log_prob first.
        // Tie-break by SMALLER node_idx first → flip the natural
        // ordering on node_idx so a smaller idx is "greater" for the
        // heap.
        self.cum_log_prob
            .partial_cmp(&other.cum_log_prob)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node_idx.cmp(&self.node_idx))
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Expand a dynamic tree starting from `root_token`.
///
/// Algorithm: best-first expansion with a max-heap by
/// `cumulative_log_prob`. Returns when `tree.len() == cfg.budget` or
/// the queue is empty (every reachable node was expanded — happens
/// when `top_k * (cfg.max_depth - 1) + 1 < cfg.budget`).
///
/// Drafter contract: see `drafter.rs`. The drafter is called with
/// `parent_node_idx` (0 for the root expansion); each call must
/// return up to `top_k` candidates with strictly-decreasing
/// finite log_probs over distinct tokens.
pub fn expand_dynamic_tree<D: Drafter>(
    root_token: u32,
    drafter: &mut D,
    cfg: &DynamicTreeConfig,
) -> Result<ExpandedTree> {
    cfg.validate()?;

    let mut tokens: Vec<u32> = Vec::with_capacity(cfg.budget);
    let mut parents: Vec<Option<usize>> = Vec::with_capacity(cfg.budget);
    let mut depths: Vec<usize> = Vec::with_capacity(cfg.budget);
    let mut cum: Vec<f32> = Vec::with_capacity(cfg.budget);

    // Root.
    tokens.push(root_token);
    parents.push(None);
    depths.push(0);
    cum.push(0.0);

    let mut heap = BinaryHeap::<HeapEntry>::with_capacity(cfg.budget);
    heap.push(HeapEntry {
        cum_log_prob: 0.0,
        node_idx: 0,
    });

    while tokens.len() < cfg.budget {
        let Some(entry) = heap.pop() else {
            break; // queue exhausted
        };
        let parent_idx = entry.node_idx;
        if depths[parent_idx] >= cfg.max_depth {
            continue; // max depth reached for this subtree
        }
        let candidates = drafter.predict_topk(parent_idx, cfg.top_k)?;
        validate_candidates(&candidates, cfg.top_k)?;

        for cand in candidates {
            if tokens.len() >= cfg.budget {
                break;
            }
            let child_idx = tokens.len();
            let child_cum = (cum[parent_idx] as f64) + (cand.log_prob as f64);
            // Defensive: cum_log_prob accumulation in f64 could in
            // principle hit NEG_INFINITY at extreme depths × extreme
            // log_probs. Validate finiteness.
            ensure!(
                child_cum.is_finite(),
                "cumulative log_prob overflowed at depth {} (parent_cum={}, edge={})",
                depths[parent_idx] + 1,
                cum[parent_idx],
                cand.log_prob
            );
            tokens.push(cand.token);
            parents.push(Some(parent_idx));
            depths.push(depths[parent_idx] + 1);
            cum.push(child_cum as f32);
            heap.push(HeapEntry {
                cum_log_prob: child_cum,
                node_idx: child_idx,
            });
        }
    }

    let out = ExpandedTree {
        tokens,
        parents,
        depths,
        cum_log_probs: cum,
    };
    out.validate()
        .map_err(|e| anyhow!("expand_dynamic_tree produced invalid tree: {}", e))?;
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::super::drafter::{BiasedMockDrafter, DraftCandidate, MockDrafter};
    use super::*;
    use std::collections::HashSet;

    /// A drafter that returns a hard-coded sequence of candidates,
    /// indexed by call order. Useful for asserting exact tree shapes.
    struct ScriptedDrafter {
        scripts: Vec<Vec<DraftCandidate>>,
        call_count: usize,
    }
    impl Drafter for ScriptedDrafter {
        fn predict_topk(
            &mut self,
            _parent_node_idx: usize,
            _top_k: usize,
        ) -> Result<Vec<DraftCandidate>> {
            let script = self.scripts[self.call_count].clone();
            self.call_count += 1;
            Ok(script)
        }
    }

    #[test]
    fn adr_037_e4a_budget_1_returns_only_root_2026_05_22() {
        let cfg = DynamicTreeConfig {
            budget: 1,
            max_depth: 4,
            top_k: 4,
        };
        let mut d = MockDrafter::default();
        let tree = expand_dynamic_tree(12345, &mut d, &cfg).unwrap();
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.tokens[0], 12345);
        assert_eq!(tree.parents[0], None);
        assert_eq!(tree.depths[0], 0);
        assert_eq!(tree.cum_log_probs[0], 0.0);
    }

    #[test]
    fn adr_037_e4a_linear_chain_via_top_k_1_2026_05_22() {
        // top_k=1 forces linear chain. budget=5 → 5 nodes deep.
        let cfg = DynamicTreeConfig {
            budget: 5,
            max_depth: 10,
            top_k: 1,
        };
        let mut d = MockDrafter::default();
        let tree = expand_dynamic_tree(100, &mut d, &cfg).unwrap();
        assert_eq!(tree.len(), 5);
        // Verify linear chain: parents[i] == i-1 for i > 0.
        for i in 1..tree.len() {
            assert_eq!(tree.parents[i], Some(i - 1));
            assert_eq!(tree.depths[i], i);
        }
    }

    #[test]
    fn adr_037_e4a_fixed_square_via_uniform_drafter_2026_05_22() {
        // top_k=4, max_depth=2, budget=large → root + 4 children.
        // Children all have same log_prob (mock returns slope) so the
        // root expansion exhausts its 4 candidates, then each child
        // gets popped. Since cum_log_prob is strictly DECREASING with
        // depth (slope = -0.5), depth-2 expansions also occur until
        // budget reached.
        let cfg = DynamicTreeConfig {
            budget: 5, // root + 4 children only
            max_depth: 2,
            top_k: 4,
        };
        let mut d = MockDrafter::default();
        let tree = expand_dynamic_tree(0, &mut d, &cfg).unwrap();
        assert_eq!(tree.len(), 5);
        // All 4 children parent on root.
        for i in 1..5 {
            assert_eq!(tree.parents[i], Some(0));
            assert_eq!(tree.depths[i], 1);
        }
    }

    #[test]
    fn adr_037_e4a_dynamic_asymmetric_expands_biased_subtree_2026_05_22() {
        // Bias the drafter to favor node 1 (first child of root).
        // With budget=10 and top_k=2, we expect the biased subtree
        // to get DEEPER than the unbiased subtree because its
        // cumulative log_prob stays higher.
        let mut bias = HashSet::new();
        bias.insert(1); // first child
        bias.insert(3); // grandchild via biased path
        bias.insert(5);
        bias.insert(7);
        let mut d = BiasedMockDrafter {
            vocab_size: 32_000,
            base_log_prob: -1.0,
            log_prob_slope: -1.0,
            bias_nodes: bias,
        };
        let cfg = DynamicTreeConfig {
            budget: 10,
            max_depth: 6,
            top_k: 2,
        };
        let tree = expand_dynamic_tree(0, &mut d, &cfg).unwrap();
        assert_eq!(tree.len(), 10);
        // Sanity: at least one node has depth >= 3 (biased subtree expanded).
        let max_depth = *tree.depths.iter().max().unwrap();
        assert!(
            max_depth >= 3,
            "expected biased subtree to expand to depth >= 3, got max_depth = {max_depth}"
        );
    }

    #[test]
    fn adr_037_e4a_max_depth_caps_subtree_growth_2026_05_22() {
        // max_depth=2 → no node can have depth > 2.
        let cfg = DynamicTreeConfig {
            budget: 100,
            max_depth: 2,
            top_k: 3,
        };
        let mut d = MockDrafter::default();
        let tree = expand_dynamic_tree(0, &mut d, &cfg).unwrap();
        // Total nodes: 1 (root) + 3 (depth 1) + 3 * 3 (depth 2) = 13.
        // budget=100 so we're not budget-limited; we're depth-limited.
        assert_eq!(tree.len(), 13);
        assert_eq!(*tree.depths.iter().max().unwrap(), 2);
    }

    #[test]
    fn adr_037_e4a_budget_exhaustion_stops_expansion_2026_05_22() {
        let cfg = DynamicTreeConfig {
            budget: 7,
            max_depth: 10,
            top_k: 3,
        };
        let mut d = MockDrafter::default();
        let tree = expand_dynamic_tree(0, &mut d, &cfg).unwrap();
        assert_eq!(tree.len(), 7);
    }

    #[test]
    fn adr_037_e4a_scripted_tree_shape_matches_priority_2026_05_22() {
        // Drafter scripts: root returns 2 cands, child[0] returns 2
        // cands, child[1] returns 2 cands. budget=5.
        //
        // Root cum=0.0 → pop root → push A (cum=-0.5) + B (cum=-1.0).
        // Pop A (highest) → push A1 (-0.5-0.3=-0.8) + A2 (-0.5-0.6=-1.1).
        // Pop B (cum=-1.0) → push B1 (-1.0-0.3=-1.3) + B2 (-1.0-0.6=-1.6).
        //   But budget=5 means only A1 fits (tree.len becomes 5 after A1, so B's
        //   second child B2 is dropped... actually let me retrace.
        //
        // After root expand: tree=[R, A, B], len=3.
        // After A expand: tree=[R, A, B, A1, A2], len=5 → budget reached.
        // So B never expands.
        //
        // Verify: 5 nodes, parents = [None, 0, 0, 1, 1].
        let d = ScriptedDrafter {
            scripts: vec![
                vec![
                    DraftCandidate { token: 100, log_prob: -0.5 },
                    DraftCandidate { token: 101, log_prob: -1.0 },
                ],
                vec![
                    DraftCandidate { token: 200, log_prob: -0.3 },
                    DraftCandidate { token: 201, log_prob: -0.6 },
                ],
                vec![
                    DraftCandidate { token: 300, log_prob: -0.3 },
                    DraftCandidate { token: 301, log_prob: -0.6 },
                ],
            ],
            call_count: 0,
        };
        let cfg = DynamicTreeConfig {
            budget: 5,
            max_depth: 4,
            top_k: 2,
        };
        let mut d = d;
        let tree = expand_dynamic_tree(1, &mut d, &cfg).unwrap();
        assert_eq!(tree.len(), 5);
        assert_eq!(tree.tokens, vec![1, 100, 101, 200, 201]);
        assert_eq!(
            tree.parents,
            vec![None, Some(0), Some(0), Some(1), Some(1)]
        );
        assert_eq!(tree.depths, vec![0, 1, 1, 2, 2]);
        // Cum log_probs: [0, -0.5, -1.0, -0.8, -1.1].
        let expected = [0.0, -0.5, -1.0, -0.8, -1.1];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (tree.cum_log_probs[i] - exp).abs() < 1e-5,
                "cum[{}] = {} != {}",
                i,
                tree.cum_log_probs[i],
                exp
            );
        }
    }

    #[test]
    fn adr_037_e4a_build_tree_mask_matches_phase_e1_contract_2026_05_22() {
        // Build a small tree and verify the mask buffer matches the
        // contract used by mlx-native's tree_attention_e1_1_parity tests
        // (build_tree_mask_from_parents test helper).
        // Tree: root, child0 (parent root), child1 (parent root),
        // grandchild (parent child0).
        let d = ScriptedDrafter {
            scripts: vec![
                vec![
                    DraftCandidate { token: 10, log_prob: -0.2 },
                    DraftCandidate { token: 20, log_prob: -0.5 },
                ],
                vec![
                    DraftCandidate { token: 11, log_prob: -0.3 },
                ],
            ],
            call_count: 0,
        };
        let cfg = DynamicTreeConfig {
            budget: 4,
            max_depth: 3,
            top_k: 2,
        };
        let mut d = d;
        let tree = expand_dynamic_tree(1, &mut d, &cfg).unwrap();
        assert_eq!(tree.parents, vec![None, Some(0), Some(0), Some(1)]);

        let prefix_len = 5;
        let mask = tree.build_tree_mask(prefix_len);
        let q = tree.len();
        let mask_stride = prefix_len + q;
        assert_eq!(mask.len(), q * mask_stride);

        // Row 0 (root): attends prefix [0,5) + self at 5.
        // Row 1 (child0): attends prefix + root at 5 + self at 6.
        // Row 2 (child1): attends prefix + root at 5 + self at 7.
        // Row 3 (grandchild): attends prefix + root + child0 + self.
        const ATTENDED: f32 = 0.0;
        const MASKED: f32 = -65504.0;

        // Helper to check a cell.
        let check = |row: usize, col: usize, exp: f32, label: &str| {
            assert_eq!(
                mask[row * mask_stride + col],
                exp,
                "row {row} col {col} ({label})"
            );
        };

        // Row 0: prefix [0,5) attended, [5,9) — only self (col 5) attended.
        for k in 0..prefix_len {
            check(0, k, ATTENDED, "prefix");
        }
        check(0, 5, ATTENDED, "self");
        check(0, 6, MASKED, "child0 sibling");
        check(0, 7, MASKED, "child1 sibling");
        check(0, 8, MASKED, "grandchild");

        // Row 3 (grandchild): prefix + root (5) + child0 (6) + self (8).
        for k in 0..prefix_len {
            check(3, k, ATTENDED, "prefix");
        }
        check(3, 5, ATTENDED, "root ancestor");
        check(3, 6, ATTENDED, "child0 parent");
        check(3, 7, MASKED, "child1 sibling — not ancestor");
        check(3, 8, ATTENDED, "self");
    }

    #[test]
    fn adr_037_e4a_validate_invalid_config_rejected_2026_05_22() {
        let mut d = MockDrafter::default();
        // budget = 0
        let mut cfg = DynamicTreeConfig::default();
        cfg.budget = 0;
        assert!(expand_dynamic_tree(0, &mut d, &cfg).is_err());
        // top_k = 0
        let mut cfg = DynamicTreeConfig::default();
        cfg.top_k = 0;
        assert!(expand_dynamic_tree(0, &mut d, &cfg).is_err());
        // max_depth = 0
        let mut cfg = DynamicTreeConfig::default();
        cfg.max_depth = 0;
        assert!(expand_dynamic_tree(0, &mut d, &cfg).is_err());
    }

    #[test]
    fn adr_037_e4a_drafter_returning_unsorted_is_rejected_2026_05_22() {
        // A buggy drafter that returns unsorted candidates should
        // fail validation, not corrupt the heap.
        struct BadDrafter;
        impl Drafter for BadDrafter {
            fn predict_topk(
                &mut self,
                _parent: usize,
                _top_k: usize,
            ) -> Result<Vec<DraftCandidate>> {
                Ok(vec![
                    DraftCandidate { token: 10, log_prob: -1.0 },
                    DraftCandidate { token: 11, log_prob: -0.5 },
                ])
            }
        }
        let cfg = DynamicTreeConfig::default();
        let mut d = BadDrafter;
        let err = expand_dynamic_tree(0, &mut d, &cfg).unwrap_err().to_string();
        assert!(err.contains("sorted descending"), "got: {err}");
    }

    #[test]
    fn adr_037_e4a_expanded_tree_validate_catches_corruption_2026_05_22() {
        // Hand-construct an invalid tree to verify validate() catches it.
        let bad = ExpandedTree {
            tokens: vec![1, 2, 3],
            parents: vec![None, Some(2), Some(0)], // parent[1] = 2 violates topological order
            depths: vec![0, 1, 1],
            cum_log_probs: vec![0.0, -0.5, -0.5],
        };
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("topological"), "got: {err}");
    }
}
