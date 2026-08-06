//! ADR-037 Phase E5a — Tree-walk-accept algorithm.
//!
//! Given an `ExpandedTree` (from Phase E4a `expand_dynamic_tree`) and
//! the verifier's per-tree-node argmaxes (greedy next-token from each
//! node's context), find the LONGEST matching path from the root.
//!
//! ## Algorithm
//!
//! ```text
//! accepted = [0]                  // root always accepted
//! current = 0
//! loop:
//!     target = verifier_argmax[current]
//!     // find unique child c of current where tree.tokens[c] == target
//!     match tree.children_of(current).find(|c| tokens[c] == target):
//!         None: break             // verifier rejected — walk ends
//!         Some(c):
//!             accepted.push(c)
//!             current = c
//! return accepted
//! ```
//!
//! ## Semantics
//!
//! - `verifier_argmax[i]` is the token the verifier picks GREEDILY
//!   as the next token after position `i`. For tree decoding, the
//!   verifier runs once over all tree-node positions and emits one
//!   argmax per node.
//! - `tree.tokens[i]` for non-root `i` is the token the DRAFTER
//!   proposed at this tree node (as a candidate next-token from
//!   parent[i]).
//! - Acceptance: a draft node `i` is "accepted" iff
//!   `tree.tokens[i] == verifier_argmax[parent[i]]` — i.e. the
//!   drafter's proposal matches what the verifier would have
//!   greedy-decoded next from the parent.
//! - The walk takes the longest matching chain starting from the
//!   root. Branches that don't match are skipped (no fall-back to
//!   siblings).
//!
//! ## Edge cases
//!
//! - **Empty accept**: verifier rejects root's first-step prediction
//!   → walk returns `[0]` (just root).
//! - **Full chain**: every step matches → walk reaches a leaf.
//! - **Multiple children with same token**: per Phase E4a contract,
//!   `expand_dynamic_tree` produces unique tokens per `predict_topk`
//!   call (each top-K must have distinct tokens). At each parent
//!   the children's tokens ARE unique. The `find` is therefore
//!   unambiguous.

use super::dynamic_tree::ExpandedTree;
use anyhow::{ensure, Result};

/// Walk the tree and return the accepted node indices in order from root.
///
/// # Arguments
/// * `tree`: from `expand_dynamic_tree` (Phase E4a).
/// * `verifier_argmax`: greedy argmax tokens per tree-node position.
///   Length must equal `tree.len()`.
///
/// # Returns
/// `Vec<usize>` of accepted tree-node indices in walk order. Always
/// starts with `[0]` (root). May contain only `[0]` (empty accept
/// past root) or up to `tree.len()` elements (full accept).
///
/// # Errors
/// - `verifier_argmax.len() != tree.len()`: shape mismatch.
/// - `tree.validate()` fails: corrupt input.
pub fn walk_tree_accept(tree: &ExpandedTree, verifier_argmax: &[u32]) -> Result<Vec<usize>> {
    tree.validate()?;
    ensure!(
        verifier_argmax.len() == tree.len(),
        "walk_tree_accept: verifier_argmax len {} != tree.len() {}",
        verifier_argmax.len(),
        tree.len()
    );
    let mut accepted: Vec<usize> = Vec::with_capacity(tree.len());
    accepted.push(0); // root
    let mut current: usize = 0;
    loop {
        let target = verifier_argmax[current];
        // Find unique child of `current` whose drafted token matches.
        // Per Phase E4a contract, tokens are unique per top-K, so
        // children of any given parent have distinct tokens.
        let next: Option<usize> = (current + 1..tree.len())
            .find(|&c| tree.parents[c] == Some(current) && tree.tokens[c] == target);
        match next {
            None => break,
            Some(c) => {
                accepted.push(c);
                current = c;
            }
        }
    }
    Ok(accepted)
}

/// Result-like summary of an accept-walk. Wraps the indices with
/// convenience accessors.
///
/// Fields are private to prevent constructing invalid summaries
/// (codex /cfa E5a Minor 2026-05-22: public fields let callers
/// build `AcceptWalk { accepted: vec![999], tree: &t }` where 999
/// > t.len() — accessors would then panic on out-of-range indices).
/// Use [`walk_and_summarize`] to construct.
#[derive(Debug, Clone)]
pub struct AcceptWalk<'tree> {
    accepted: Vec<usize>,
    tree: &'tree ExpandedTree,
}

impl<'tree> AcceptWalk<'tree> {
    /// Borrow the accepted indices.
    pub fn accepted(&self) -> &[usize] {
        &self.accepted
    }
    /// Borrow the backing tree.
    pub fn tree(&self) -> &ExpandedTree {
        self.tree
    }
    /// Number of accepted nodes including root. Always >= 1 since
    /// `walk_tree_accept` always includes root.
    pub fn len(&self) -> usize {
        self.accepted.len()
    }
    /// Always `false` for any `AcceptWalk` constructed via
    /// `walk_and_summarize` (root is always present). Retained for
    /// API completeness.
    pub fn is_empty(&self) -> bool {
        self.accepted.is_empty()
    }
    /// Number of DRAFTED tokens accepted (excludes root, which is the
    /// target's last-generated token, not a draft).
    pub fn drafted_accepted(&self) -> usize {
        self.accepted.len().saturating_sub(1)
    }
    /// Token sequence along the accept path (root first).
    pub fn tokens(&self) -> Vec<u32> {
        self.accepted.iter().map(|&i| self.tree.tokens[i]).collect()
    }
    /// Depth of the deepest accepted node (root depth = 0).
    pub fn max_depth(&self) -> usize {
        self.accepted
            .iter()
            .map(|&i| self.tree.depths[i])
            .max()
            .unwrap_or(0)
    }
}

/// Convenience: walk + wrap in `AcceptWalk`.
pub fn walk_and_summarize<'a>(
    tree: &'a ExpandedTree,
    verifier_argmax: &[u32],
) -> Result<AcceptWalk<'a>> {
    let accepted = walk_tree_accept(tree, verifier_argmax)?;
    Ok(AcceptWalk { accepted, tree })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    /// Helper: build a small `ExpandedTree` from explicit arrays.
    fn make_tree(
        tokens: Vec<u32>,
        parents: Vec<Option<usize>>,
        depths: Vec<usize>,
    ) -> ExpandedTree {
        let n = tokens.len();
        let cum_log_probs = vec![0.0f64; n];
        ExpandedTree {
            tokens,
            parents,
            depths,
            cum_log_probs,
        }
    }

    #[test]
    fn adr_037_e5a_walk_empty_accept_when_root_rejected_2026_05_22() {
        // Tree: root=100, child=200.
        // Verifier argmax at root=999 (not 200).
        // Walk: [0] only.
        let tree = make_tree(vec![100, 200], vec![None, Some(0)], vec![0, 1]);
        let argmax = vec![999_u32, 0]; // verifier says next-after-root is 999
        let accepted = walk_tree_accept(&tree, &argmax).expect("walk");
        assert_eq!(accepted, vec![0]);
    }

    #[test]
    fn adr_037_e5a_walk_full_chain_accept_2026_05_22() {
        // Linear chain: 0 -> 1 -> 2 -> 3. All match.
        let tree = make_tree(
            vec![10, 20, 30, 40],
            vec![None, Some(0), Some(1), Some(2)],
            vec![0, 1, 2, 3],
        );
        // Verifier argmax: root predicts 20, node1 predicts 30, node2
        // predicts 40, node3 (leaf) predicts anything.
        let argmax = vec![20_u32, 30, 40, 99];
        let accepted = walk_tree_accept(&tree, &argmax).expect("walk");
        assert_eq!(accepted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn adr_037_e5a_walk_branches_takes_matching_child_2026_05_22() {
        // Tree: 0 -> [1, 2] (siblings). Verifier picks token at node 2.
        let tree = make_tree(
            vec![100, 200, 300],
            vec![None, Some(0), Some(0)],
            vec![0, 1, 1],
        );
        let argmax = vec![300_u32, 0, 0]; // verifier picks 300 → child idx 2
        let accepted = walk_tree_accept(&tree, &argmax).expect("walk");
        assert_eq!(accepted, vec![0, 2]);
    }

    #[test]
    fn adr_037_e5a_walk_partial_chain_then_no_match_2026_05_22() {
        // Tree: 0 -> 1 -> [2, 3].
        // Verifier: root→token 20 (match 1), node1→token 999 (no match).
        let tree = make_tree(
            vec![10, 20, 30, 40],
            vec![None, Some(0), Some(1), Some(1)],
            vec![0, 1, 2, 2],
        );
        let argmax = vec![20_u32, 999, 0, 0];
        let accepted = walk_tree_accept(&tree, &argmax).expect("walk");
        assert_eq!(accepted, vec![0, 1]);
    }

    #[test]
    fn adr_037_e5a_walk_rejects_size_mismatch_2026_05_22() {
        let tree = make_tree(vec![10, 20], vec![None, Some(0)], vec![0, 1]);
        let argmax = vec![20_u32]; // wrong size
        let err = walk_tree_accept(&tree, &argmax).unwrap_err();
        assert!(
            err.to_string().contains("verifier_argmax len"),
            "got: {err}"
        );
    }

    #[test]
    fn adr_037_e5a_walk_asymmetric_tree_picks_deepest_matching_path_2026_05_22() {
        // Tree:
        //   0 (root, token=1)
        //   ├── 1 (token=10)
        //   │   └── 3 (token=100)
        //   └── 2 (token=20)
        //
        // Verifier: root→10, node1→100, node3→(any).
        // Walk: 0 → 1 → 3. Path length 3.
        let tree = make_tree(
            vec![1, 10, 20, 100],
            vec![None, Some(0), Some(0), Some(1)],
            vec![0, 1, 1, 2],
        );
        let argmax = vec![10_u32, 100, 0, 0];
        let accepted = walk_tree_accept(&tree, &argmax).expect("walk");
        assert_eq!(accepted, vec![0, 1, 3]);
    }

    #[test]
    fn adr_037_e5a_walk_returns_only_root_for_single_node_tree_2026_05_22() {
        let tree = make_tree(vec![42], vec![None], vec![0]);
        let argmax = vec![999_u32];
        let accepted = walk_tree_accept(&tree, &argmax).expect("walk");
        assert_eq!(accepted, vec![0]);
    }

    #[test]
    fn adr_037_e5a_summary_accessor_helpers_2026_05_22() {
        // Tree: 0 → 1 → 2 → 3.
        let tree = make_tree(
            vec![10, 20, 30, 40],
            vec![None, Some(0), Some(1), Some(2)],
            vec![0, 1, 2, 3],
        );
        let argmax = vec![20_u32, 30, 40, 99];
        let summary = walk_and_summarize(&tree, &argmax).expect("walk");
        assert_eq!(summary.len(), 4);
        assert!(!summary.is_empty());
        assert_eq!(summary.drafted_accepted(), 3);
        assert_eq!(summary.tokens(), vec![10, 20, 30, 40]);
        assert_eq!(summary.max_depth(), 3);
    }

    #[test]
    fn adr_037_e5a_summary_root_only_2026_05_22() {
        let tree = make_tree(vec![100, 200], vec![None, Some(0)], vec![0, 1]);
        let argmax = vec![999_u32, 0];
        let summary = walk_and_summarize(&tree, &argmax).expect("walk");
        assert_eq!(summary.len(), 1);
        assert_eq!(summary.drafted_accepted(), 0); // root not counted as drafted
        assert_eq!(summary.tokens(), vec![100]);
        assert_eq!(summary.max_depth(), 0);
    }

    #[test]
    fn adr_037_e5a_walk_rejects_corrupt_tree_2026_05_22() {
        // Tree with parents[1] = 5 (out of bounds — Tree::validate
        // catches non-topological order).
        let tree = ExpandedTree {
            tokens: vec![1, 2],
            parents: vec![None, Some(5)],
            depths: vec![0, 1],
            cum_log_probs: vec![0.0, 0.0],
        };
        let argmax = vec![2_u32, 0];
        // walk_tree_accept calls tree.validate() which catches the
        // out-of-topological-order parent before walking.
        assert!(walk_tree_accept(&tree, &argmax).is_err());
    }

    #[test]
    fn adr_037_e5a_walk_skips_non_root_descendants_in_search_2026_05_22() {
        // Tree: 0 → 1 → 2 (chain). 0 → 3 (sibling of 1).
        //
        // Verifier picks token 30 at root (matches token of node 2,
        // but node 2 is GRANDCHILD of root, not a child). Walk
        // should NOT skip the parent chain — it must descend
        // only via direct children. Expected: walk returns [0]
        // since none of root's DIRECT children have token 30.
        let tree = make_tree(
            vec![1, 10, 30, 20],
            vec![None, Some(0), Some(1), Some(0)],
            vec![0, 1, 2, 1],
        );
        let argmax = vec![30_u32, 0, 0, 0];
        let accepted = walk_tree_accept(&tree, &argmax).expect("walk");
        // Root's direct children are 1 (token 10) and 3 (token 20).
        // Neither matches verifier's 30. Walk stops at root.
        assert_eq!(accepted, vec![0]);
    }

    #[test]
    fn adr_037_e5a_walk_integration_with_phase_e4a_expand_dynamic_tree_2026_05_22() {
        // Build a tree via Phase E4a, then walk through it with a
        // synthetic verifier that matches the drafter's top-1 child
        // at each level. Should accept the full top-1 chain.
        use crate::inference::spec_decode::eagle3::drafter::{
            DraftCandidate, Drafter, TreeContextView,
        };
        use crate::inference::spec_decode::eagle3::dynamic_tree::{
            expand_dynamic_tree, DynamicTreeConfig,
        };
        struct ScriptedDrafter;
        impl Drafter for ScriptedDrafter {
            fn predict_topk(
                &mut self,
                _tree: TreeContextView<'_>,
                node: usize,
                _top_k: usize,
            ) -> Result<Vec<DraftCandidate>> {
                // Deterministic: node N → top-2 candidates [N*10+1, N*10+2].
                Ok(vec![
                    DraftCandidate {
                        token: (node * 10 + 1) as u32,
                        log_prob: -0.1,
                    },
                    DraftCandidate {
                        token: (node * 10 + 2) as u32,
                        log_prob: -0.5,
                    },
                ])
            }
        }
        let cfg = DynamicTreeConfig {
            budget: 6,
            max_depth: 3,
            top_k: 2,
        };
        let mut d = ScriptedDrafter;
        let tree = expand_dynamic_tree(1000_u32, &mut d, &cfg).expect("expand");
        // Synthetic verifier that always picks the FIRST child's token
        // (parent node N → next = N*10+1). For root (idx 0), that's 1.
        // For node 1 (admitted from root, token = 0*10+1 = 1), the
        // verifier should pick token "node1_idx_in_tree * 10 + 1" — but
        // the verifier sees the TOKEN of the node, not the index.
        // We just want the verifier to match each chain step. Since
        // ScriptedDrafter's top-1 child of node N is `(N*10+1)`, the
        // verifier needs to predict `(child_node_in_tree * 10 + 1)` —
        // but verifier predicts BASED ON THE PARENT'S NODE INDEX,
        // not the parent's token.
        //
        // The simplest verifier: argmax[i] = tokens[i+1] if i+1 < len AND
        // parents[i+1] == i, else 0. That picks the next-admitted-child.
        let mut verifier_argmax = vec![0_u32; tree.len()];
        for i in 0..tree.len() {
            // Find any direct child of node i; verifier picks its token.
            if let Some(child) = (i + 1..tree.len()).find(|&c| tree.parents[c] == Some(i)) {
                verifier_argmax[i] = tree.tokens[child];
            }
        }
        let accepted = walk_tree_accept(&tree, &verifier_argmax).expect("walk");
        // Codex /cfa E5a Minor (2026-05-22): tighten the integration
        // assertion. The verifier picks the next-admitted-direct-child
        // at each level, so the walk should follow the chain of
        // "first child of each ancestor" all the way down.
        //
        // First node accepted is always root.
        assert_eq!(accepted[0], 0);
        // Walk should accept at least root + 1.
        assert!(accepted.len() >= 2, "walk should accept beyond root");
        // Every step (i, i+1) must follow the verifier picks: the
        // next admitted child of accepted[i] should be accepted[i+1].
        for w in accepted.windows(2) {
            let parent = w[0];
            let child = w[1];
            assert_eq!(
                tree.parents[child],
                Some(parent),
                "accept walk should follow direct-child edges"
            );
            assert_eq!(
                tree.tokens[child], verifier_argmax[parent],
                "accept walk should match verifier_argmax at each parent"
            );
        }
        // Walk terminates when verifier_argmax[last_accepted] != any
        // child's token. Check that condition holds at the leaf of
        // the accept walk.
        let last = *accepted.last().unwrap();
        let no_matching_child = (last + 1..tree.len())
            .find(|&c| tree.parents[c] == Some(last) && tree.tokens[c] == verifier_argmax[last])
            .is_none();
        assert!(
            no_matching_child,
            "walk should have stopped because no direct child of {} matches argmax {}",
            last, verifier_argmax[last]
        );
    }
}
