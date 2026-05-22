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

use super::drafter::{validate_candidates, Drafter, TreeContextView};
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
///
/// Codex /cfa E4a Critical (2026-05-22): `cum_log_probs` is `Vec<f64>`,
/// not `Vec<f32>`. f32 accumulation at extreme depths × extreme
/// log_probs (even with `LOG_PROB_FLOOR`) can yield `f64` -> `f32`
/// cast underflow to `-inf`, which would later fail downstream
/// finite-checks despite valid input. f64 internally + at the public
/// surface is the simpler invariant to maintain.
#[derive(Debug, Clone)]
pub struct ExpandedTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<Option<usize>>,
    pub depths: Vec<usize>,
    pub cum_log_probs: Vec<f64>,
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
    ///
    /// Codex /cfa E4a Major (2026-05-22): returns `Result<Vec<f32>>`
    /// with checked arithmetic. Large prefix_len could overflow the
    /// total size on extreme shapes; previously this would panic on
    /// debug or wrap on release. Now an explicit error is returned.
    pub fn build_tree_mask(&self, prefix_len: usize) -> Result<Vec<f32>> {
        const ATTENDED: f32 = 0.0;
        const MASKED: f32 = -65504.0;
        let q = self.len();
        let mask_stride = prefix_len.checked_add(q).ok_or_else(|| {
            anyhow!(
                "build_tree_mask: prefix_len ({}) + tree.len ({}) overflows usize",
                prefix_len,
                q
            )
        })?;
        let total = q.checked_mul(mask_stride).ok_or_else(|| {
            anyhow!(
                "build_tree_mask: tree.len ({}) * mask_stride ({}) overflows usize",
                q,
                mask_stride
            )
        })?;
        let mut mask = vec![MASKED; total];
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
        Ok(mask)
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
            self.cum_log_probs[0] == 0.0_f64,
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

/// Pending-admission heap entry. Codex /cfa E4a Critical 1 fix
/// (2026-05-22): the algorithm uses a "pending candidate" heap, NOT a
/// "committed node" heap. Each heap entry is a CANDIDATE we MIGHT
/// admit — the heap orders candidates by their `would-be` cumulative
/// log-prob from root, so admission is GLOBALLY best-first (not just
/// locally best-first within each parent's batch).
///
/// Without this, batch-committing a parent's top-K children before
/// considering grandchildren of a higher-prob sibling-of-the-parent
/// would admit lower-prob nodes ahead of higher-prob ones — violating
/// EAGLE-2's published "best-first" admission guarantee.
#[derive(Debug, Clone, Copy)]
struct PendingCandidate {
    parent_idx: usize,
    token: u32,
    cum_log_prob: f64,
    /// Insertion sequence number for deterministic tiebreaks. Earlier
    /// insertions win ties so test fixtures are reproducible.
    seq: usize,
}

impl PartialEq for PendingCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cum_log_prob == other.cum_log_prob && self.seq == other.seq
    }
}
impl Eq for PendingCandidate {}
impl Ord for PendingCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is max-heap. Highest cum_log_prob wins. Tiebreak:
        // SMALLER seq wins (insert-order, deterministic).
        self.cum_log_prob
            .partial_cmp(&other.cum_log_prob)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}
impl PartialOrd for PendingCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Expand a dynamic tree starting from `root_token`.
///
/// Algorithm: GLOBALLY best-first expansion via a pending-candidate
/// heap ordered by cumulative log-prob from root.
///
/// 1. Push root, expand it once to seed the heap with its top-K children.
/// 2. Loop until budget reached or heap empty:
///    a. Pop the globally-best pending candidate.
///    b. ADMIT it as a new tree node (with proper parent / depth /
///       cum_log_prob).
///    c. If the new node is below max_depth, query the drafter for ITS
///       top-K children and push them onto the heap as new pending
///       candidates.
///
/// This guarantees that every admitted non-root node has the highest
/// cumulative log-prob of any node that COULD be admitted next — the
/// global best-first invariant from the EAGLE-2 paper.
pub fn expand_dynamic_tree<D: Drafter>(
    root_token: u32,
    drafter: &mut D,
    cfg: &DynamicTreeConfig,
) -> Result<ExpandedTree> {
    cfg.validate()?;

    let mut tokens: Vec<u32> = Vec::with_capacity(cfg.budget);
    let mut parents: Vec<Option<usize>> = Vec::with_capacity(cfg.budget);
    let mut depths: Vec<usize> = Vec::with_capacity(cfg.budget);
    let mut cum: Vec<f64> = Vec::with_capacity(cfg.budget);

    // Root.
    tokens.push(root_token);
    parents.push(None);
    depths.push(0);
    cum.push(0.0);

    let mut heap = BinaryHeap::<PendingCandidate>::new();
    let mut seq_counter: usize = 0;

    // Seed the heap by expanding the root once (if budget > 1 AND max_depth >= 1).
    if cfg.budget > 1 && cfg.max_depth >= 1 {
        let view = TreeContextView {
            tokens: &tokens,
            parents: &parents,
        };
        let candidates = drafter.predict_topk(view, 0, cfg.top_k)?;
        validate_candidates(&candidates, cfg.top_k)?;
        for cand in candidates {
            let child_cum = cand.log_prob as f64;
            ensure!(
                child_cum.is_finite(),
                "seed cum_log_prob not finite: {}",
                child_cum
            );
            heap.push(PendingCandidate {
                parent_idx: 0,
                token: cand.token,
                cum_log_prob: child_cum,
                seq: seq_counter,
            });
            seq_counter += 1;
        }
    }

    // Globally best-first admission loop.
    while tokens.len() < cfg.budget {
        let Some(pending) = heap.pop() else {
            break; // queue exhausted
        };
        // Admit this candidate as a new tree node.
        let parent_idx = pending.parent_idx;
        let child_idx = tokens.len();
        let new_depth = depths[parent_idx] + 1;
        tokens.push(pending.token);
        parents.push(Some(parent_idx));
        depths.push(new_depth);
        cum.push(pending.cum_log_prob);

        // Only expand if we'll potentially admit more children. When
        // tokens.len() == cfg.budget after the admit above, the next
        // loop iter exits anyway; calling the drafter to produce
        // candidates we'll never admit is wasted work AND lets buggy
        // tests with under-specified scripts crash unnecessarily.
        if tokens.len() < cfg.budget && new_depth < cfg.max_depth {
            let view = TreeContextView {
                tokens: &tokens,
                parents: &parents,
            };
            let candidates = drafter.predict_topk(view, child_idx, cfg.top_k)?;
            validate_candidates(&candidates, cfg.top_k)?;
            let parent_cum = cum[child_idx];
            for cand in candidates {
                let child_cum = parent_cum + (cand.log_prob as f64);
                ensure!(
                    child_cum.is_finite(),
                    "cumulative log_prob overflowed at depth {} (parent_cum={}, edge={})",
                    new_depth + 1,
                    parent_cum,
                    cand.log_prob
                );
                heap.push(PendingCandidate {
                    parent_idx: child_idx,
                    token: cand.token,
                    cum_log_prob: child_cum,
                    seq: seq_counter,
                });
                seq_counter += 1;
            }
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
            _tree: super::super::drafter::TreeContextView<'_>,
            _node_to_expand: usize,
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
    fn adr_037_e4a_fixed_square_via_max_depth_1_2026_05_22() {
        // Codex /cfa E4a fix (2026-05-22): with the new global
        // best-first algorithm, max_depth must equal 1 to get a
        // "fixed-square 4-leaf" shape — otherwise grandchildren of
        // the best root-child would interleave with root's siblings
        // (correct EAGLE-2 behavior, but not "fixed square" semantics).
        let cfg = DynamicTreeConfig {
            budget: 5, // root + 4 children
            max_depth: 1, // no grandchildren — every child stays at depth 1
            top_k: 4,
        };
        let mut d = MockDrafter::default();
        let tree = expand_dynamic_tree(0, &mut d, &cfg).unwrap();
        assert_eq!(tree.len(), 5);
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
    fn adr_037_e4a_global_best_first_admits_grandchild_before_sibling_2026_05_22() {
        // Codex /cfa E4a Critical 1 (2026-05-22): proves the algorithm
        // admits nodes in GLOBAL best-first order, not just local
        // best-first within each parent's batch.
        //
        // Scripts:
        //   script[0] (root → 2 children): A(-0.1), B(-2.0).  BIG gap.
        //   script[1] (A → 2 children): A1(-0.5), A2(-1.0).
        //                cums: A1 = -0.1 + -0.5 = -0.6,
        //                      A2 = -0.1 + -1.0 = -1.1.
        //   script[2] (A1 → 1 child): A1_1(-0.5).
        //                cum: A1_1 = -0.6 + -0.5 = -1.1.
        //
        // budget = 4. Expected admission ORDER:
        //   1. Seed heap with A(-0.1), B(-2.0).
        //   2. Pop A(-0.1), admit. Expand. Heap: B(-2.0), A1(-0.6), A2(-1.1).
        //   3. Pop A1(-0.6), admit. Expand A1 → A1_1(-1.1). Heap: B(-2.0), A2(-1.1), A1_1(-1.1).
        //   4. Pop A2(-1.1) (tied with A1_1 but A2 has smaller seq → wins).
        //      Admit A2. tokens.len() = 4 = budget. STOP.
        //
        // Final tree: tokens=[R, A, A1, A2], parents=[None, 0, 1, 1],
        // depths=[0, 1, 2, 2]. The proof: A2 (depth 2) and A1 (depth 2)
        // are admitted BEFORE B (depth 1, sibling of A) — because A's
        // subtree's cumulative log-prob is globally higher than B's.
        let d = ScriptedDrafter {
            scripts: vec![
                vec![
                    DraftCandidate { token: 100, log_prob: -0.1 },  // A
                    DraftCandidate { token: 200, log_prob: -2.0 },  // B
                ],
                vec![
                    DraftCandidate { token: 110, log_prob: -0.5 },  // A1
                    DraftCandidate { token: 120, log_prob: -1.0 },  // A2
                ],
                vec![
                    DraftCandidate { token: 111, log_prob: -0.5 },  // A1_1
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
        assert_eq!(tree.len(), 4);
        // CRITICAL ASSERTION: B (depth 1) is NOT in the tree, even
        // though it's a direct child of root. Instead, A's subtree
        // expanded deeper because globally better cum_log_prob.
        assert_eq!(tree.tokens, vec![1, 100, 110, 120]);
        assert_eq!(tree.parents, vec![None, Some(0), Some(1), Some(1)]);
        assert_eq!(tree.depths, vec![0, 1, 2, 2]);
        // Cum log_probs (f64): [0, -0.1, -0.6, -1.1].
        let expected = [0.0, -0.1, -0.6, -1.1];
        // Tolerance accommodates f32 → f64 cast (drafter log_probs are
        // f32, accumulated in f64). -0.1 as f32 has rounding error
        // ~1.5e-9 when promoted back to f64.
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (tree.cum_log_probs[i] - exp).abs() < 1e-6,
                "cum[{}] = {} != {}",
                i,
                tree.cum_log_probs[i],
                exp
            );
        }
        // Verify NO node with token 200 (B) exists.
        assert!(!tree.tokens.contains(&200), "B (low-prob sibling) should NOT have been admitted ahead of A's grandchildren");
    }

    #[test]
    fn adr_037_e4a_build_tree_mask_matches_phase_e1_contract_2026_05_22() {
        // Hand-construct a deterministic ExpandedTree (independent of
        // algorithm) and verify the mask buffer matches the contract
        // used by mlx-native's tree_attention_e1_1_parity tests
        // (build_tree_mask_from_parents test helper).
        //
        // Tree shape:
        //   0 (root)
        //   ├── 1 (child0)
        //   │   └── 3 (grandchild)
        //   └── 2 (child1)
        let tree = ExpandedTree {
            tokens: vec![1, 10, 20, 11],
            parents: vec![None, Some(0), Some(0), Some(1)],
            depths: vec![0, 1, 1, 2],
            cum_log_probs: vec![0.0, -0.2, -0.5, -0.5],
        };
        tree.validate().expect("hand-built tree must validate");

        let prefix_len = 5;
        let mask = tree.build_tree_mask(prefix_len).expect("build_tree_mask ok");
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
                _tree: super::super::drafter::TreeContextView<'_>,
                _node_to_expand: usize,
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
            cum_log_probs: vec![0.0_f64, -0.5, -0.5],
        };
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("topological"), "got: {err}");
    }
}
