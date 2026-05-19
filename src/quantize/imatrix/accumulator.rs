//! Per-tensor activation-importance accumulator.
//!
//! Mirrors the `Stats` struct and `IMatrixCollector::collect_imatrix`
//! algorithm at `/opt/llama.cpp/tools/imatrix/imatrix.cpp:41-409` (pinned
//! SHA `data/llama_cpp_pin.txt`).
//!
//! ## Algorithm (per llama-imatrix)
//!
//! For each linear-layer activation tensor `src1` of shape
//! `[n_per_row, n_rows, ...]` (where `n_per_row` is the input dim of the
//! linear being calibrated):
//!
//! ```text
//! e.values[j] += src1[row, j]² for each token row, for each col j
//! e.counts[mat_id] += n_rows / n_mat
//! ```
//!
//! `n_mat` is `1` for ordinary dense linears (`GGML_OP_MUL_MAT`) and
//! `n_experts` for MoE indirect matmuls (`GGML_OP_MUL_MAT_ID`). For MoE,
//! each `e.counts[expert_idx]` records how many tokens routed to that
//! expert during the corpus run; the per-expert `e.values[exp_start..]`
//! slice records the per-column importance for that expert's input
//! activations only.
//!
//! ## Phase A scope
//!
//! Phase A v1 implements the accumulator + a dense-linear `absorb_dense`
//! entry point (for non-MoE linears). MoE per-expert `absorb_moe` is
//! present but stubbed for Phase B's forward-pass driver to wire — the
//! algorithm is identical, only the `expert_id` parameter changes.
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: shape mismatches at
//! [`Accumulator::absorb_dense`] surface as a typed error, not a silent
//! clip/skip.

use std::collections::BTreeMap;

use super::error::ImatrixError;

/// Per-tensor accumulator. Mirrors `Stats` at imatrix.cpp:41.
#[derive(Debug, Clone)]
pub struct Accumulator {
    /// Tensor name (e.g. `"blk.0.attn_q.weight"`). Matches the GGUF
    /// tensor name; emitted at write time as `"<name>.in_sum2"` /
    /// `"<name>.counts"`.
    pub name: String,
    /// `n_per_row` = the input dimension of the linear being calibrated.
    /// Equivalent to `src0->ne[0]` at imatrix.cpp:294. Fixed at
    /// `register` time.
    pub n_per_row: usize,
    /// `n_mat` = 1 for dense tensors, `n_experts` for MoE expert tensors
    /// (`GGML_OP_MUL_MAT_ID`). Fixed at `register` time.
    pub n_mat: usize,
    /// Sum-of-squared-activations, length `n_per_row * n_mat`. Laid out
    /// contiguously per llama-imatrix's convention:
    /// `values[mat_id * n_per_row + j]` is the importance for input
    /// column `j` of matrix `mat_id`.
    pub values: Vec<f32>,
    /// Per-matrix activation count. `counts[mat_id]` is the number of
    /// tokens that routed to matrix `mat_id`. For dense tensors length is
    /// 1; for MoE length is `n_experts`.
    pub counts: Vec<i64>,
}

impl Accumulator {
    /// Construct an empty accumulator for a tensor.
    pub fn new(name: String, n_per_row: usize, n_mat: usize) -> Self {
        Self {
            name,
            n_per_row,
            n_mat,
            values: vec![0.0; n_per_row * n_mat],
            counts: vec![0; n_mat],
        }
    }

    /// Absorb one token's activation row into the dense-tensor
    /// accumulator (i.e. `n_mat == 1`).
    ///
    /// `row` length must equal `n_per_row`. Per
    /// [[feedback-no-loop-suppression-2026-05-17]] a mismatch is a
    /// hard error, not a silent skip.
    ///
    /// Algorithm (imatrix.cpp:380-393):
    /// ```text
    /// e.values[j] += row[j]² for j in 0..n_per_row
    /// e.counts[0] += 1
    /// ```
    pub fn absorb_dense(&mut self, row: &[f32]) -> Result<(), ImatrixError> {
        if self.n_mat != 1 {
            return Err(ImatrixError::CorpusRead {
                path: format!("<accumulator:{}>", self.name),
                detail: format!(
                    "absorb_dense called on MoE-shaped accumulator (n_mat={})",
                    self.n_mat
                ),
            });
        }
        if row.len() != self.n_per_row {
            return Err(ImatrixError::CorpusRead {
                path: format!("<accumulator:{}>", self.name),
                detail: format!(
                    "row length mismatch: got {}, expected {}",
                    row.len(),
                    self.n_per_row
                ),
            });
        }
        // imatrix.cpp:381-382: e.values[j] += x[j] * x[j]
        for (j, &x) in row.iter().enumerate() {
            self.values[j] += x * x;
        }
        // imatrix.cpp:393: e.counts[i] += 1 per absorbed token row
        self.counts[0] += 1;
        Ok(())
    }

    /// Absorb one token's activation row into the MoE-tensor
    /// accumulator at the given expert index.
    ///
    /// `row` length must equal `n_per_row`. `expert_id` must be in
    /// `0..n_mat`.
    ///
    /// Algorithm (imatrix.cpp:310-330):
    /// ```text
    /// e.values[expert_id * n_per_row + j] += row[j]²
    /// e.counts[expert_id] += 1
    /// ```
    pub fn absorb_moe(&mut self, expert_id: usize, row: &[f32]) -> Result<(), ImatrixError> {
        if expert_id >= self.n_mat {
            return Err(ImatrixError::CorpusRead {
                path: format!("<accumulator:{}>", self.name),
                detail: format!(
                    "expert_id={} out of range (n_mat={})",
                    expert_id, self.n_mat
                ),
            });
        }
        if row.len() != self.n_per_row {
            return Err(ImatrixError::CorpusRead {
                path: format!("<accumulator:{}>", self.name),
                detail: format!(
                    "row length mismatch: got {}, expected {}",
                    row.len(),
                    self.n_per_row
                ),
            });
        }
        let e_start = expert_id * self.n_per_row;
        for (j, &x) in row.iter().enumerate() {
            self.values[e_start + j] += x * x;
        }
        self.counts[expert_id] += 1;
        Ok(())
    }

    /// True if at least one token has been absorbed into any matrix.
    pub fn has_data(&self) -> bool {
        self.counts.iter().any(|&c| c > 0)
    }
}

/// Registry of per-tensor accumulators.
///
/// Backed by a `BTreeMap` so the iteration order is deterministic by
/// tensor name (matches `save_imatrix`'s `std::sort` at
/// imatrix.cpp:568).
#[derive(Debug, Default)]
pub struct AccumulatorRegistry {
    inner: BTreeMap<String, Accumulator>,
}

impl AccumulatorRegistry {
    /// New empty registry.
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    /// Register a new accumulator. If `name` is already present and the
    /// shape (`n_per_row`, `n_mat`) doesn't match, returns the typed
    /// inconsistent-size error (mirrors imatrix.cpp:367-370).
    pub fn register(
        &mut self,
        name: &str,
        n_per_row: usize,
        n_mat: usize,
    ) -> Result<&mut Accumulator, ImatrixError> {
        // Two-phase borrow to satisfy the borrow checker without an
        // unsafe / `polonius_the_crab` workaround:
        //   1. Check shape compatibility under a short immutable borrow.
        //   2. Insert if absent, then `get_mut` once at the end.
        if let Some(existing) = self.inner.get(name) {
            if existing.n_per_row != n_per_row || existing.n_mat != n_mat {
                return Err(ImatrixError::CorpusRead {
                    path: format!("<accumulator:{name}>"),
                    detail: format!(
                        "inconsistent shape on re-register: existing ({},{}) vs new ({},{})",
                        existing.n_per_row, existing.n_mat, n_per_row, n_mat
                    ),
                });
            }
        } else {
            self.inner.insert(
                name.to_string(),
                Accumulator::new(name.to_string(), n_per_row, n_mat),
            );
        }
        Ok(self
            .inner
            .get_mut(name)
            .expect("entry was just registered or shape-checked"))
    }

    /// Lookup an accumulator by name. Returns `None` if not registered.
    pub fn get(&self, name: &str) -> Option<&Accumulator> {
        self.inner.get(name)
    }

    /// Mutable lookup.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Accumulator> {
        self.inner.get_mut(name)
    }

    /// Iterate accumulators in deterministic (sorted-by-name) order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Accumulator)> {
        self.inner.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Count of registered tensors.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True if no tensors registered.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `absorb_dense` produces sum-of-squares per llama-imatrix's
    /// canonical formula. Absorbing `[1.0, 2.0, 3.0]` then `[4.0, 5.0,
    /// 6.0]` yields `values = [17.0, 29.0, 45.0]` and `counts = [2]`.
    #[test]
    fn absorb_dense_canonical_formula() {
        let mut acc = Accumulator::new("test.weight".to_string(), 3, 1);
        acc.absorb_dense(&[1.0, 2.0, 3.0]).unwrap();
        acc.absorb_dense(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(acc.values, vec![1.0 + 16.0, 4.0 + 25.0, 9.0 + 36.0]);
        assert_eq!(acc.counts, vec![2]);
    }

    /// `absorb_moe` lays out per-expert per-column importance in the
    /// imatrix.cpp:294 convention: `values[expert_id*n_per_row + j]`.
    #[test]
    fn absorb_moe_layout() {
        let mut acc = Accumulator::new("blk.0.ffn_gate_exps.weight".to_string(), 3, 4);
        acc.absorb_moe(2, &[1.0, 2.0, 3.0]).unwrap();
        acc.absorb_moe(2, &[1.0, 1.0, 1.0]).unwrap();
        acc.absorb_moe(0, &[0.5, 0.5, 0.5]).unwrap();
        assert_eq!(acc.counts, vec![1, 0, 2, 0]);
        // expert 0 row 0..3
        assert_eq!(acc.values[0..3], [0.25, 0.25, 0.25]);
        // expert 1 untouched
        assert_eq!(acc.values[3..6], [0.0, 0.0, 0.0]);
        // expert 2 (1+1, 4+1, 9+1) = (2, 5, 10)
        assert_eq!(acc.values[6..9], [2.0, 5.0, 10.0]);
        // expert 3 untouched
        assert_eq!(acc.values[9..12], [0.0, 0.0, 0.0]);
    }

    /// Row-length mismatch is a typed error, not a silent clip.
    #[test]
    fn row_length_mismatch_errors() {
        let mut acc = Accumulator::new("t".to_string(), 3, 1);
        let err = acc.absorb_dense(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, ImatrixError::CorpusRead { .. }));
    }

    /// Calling `absorb_dense` on an MoE accumulator is a hard error.
    #[test]
    fn dense_on_moe_errors() {
        let mut acc = Accumulator::new("t".to_string(), 3, 4);
        let err = acc.absorb_dense(&[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, ImatrixError::CorpusRead { .. }));
    }

    /// `absorb_moe` rejects `expert_id >= n_mat`.
    #[test]
    fn moe_expert_out_of_range_errors() {
        let mut acc = Accumulator::new("t".to_string(), 3, 4);
        let err = acc.absorb_moe(5, &[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, ImatrixError::CorpusRead { .. }));
    }

    /// Registry registers + retrieves accumulators by name.
    #[test]
    fn registry_register_get() {
        let mut reg = AccumulatorRegistry::new();
        let acc = reg.register("a.weight", 4, 1).unwrap();
        acc.absorb_dense(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let _ = reg.register("b.weight", 8, 2).unwrap();
        assert_eq!(reg.len(), 2);
        assert!(reg.get("a.weight").is_some());
        assert!(reg.get("b.weight").is_some());
        assert!(reg.get("nonexistent").is_none());
        assert_eq!(reg.get("a.weight").unwrap().counts, vec![1]);
    }

    /// Re-registering with a different shape is a typed error.
    #[test]
    fn registry_inconsistent_shape_errors() {
        let mut reg = AccumulatorRegistry::new();
        let _ = reg.register("a.weight", 4, 1).unwrap();
        let err = reg.register("a.weight", 8, 1).unwrap_err();
        assert!(matches!(err, ImatrixError::CorpusRead { .. }));
    }

    /// Iteration order is sorted by name (matches imatrix.cpp:568's
    /// `std::sort`). Critical for byte-cmp against llama-imatrix.
    #[test]
    fn registry_iter_sorted() {
        let mut reg = AccumulatorRegistry::new();
        reg.register("c.weight", 4, 1).unwrap();
        reg.register("a.weight", 4, 1).unwrap();
        reg.register("b.weight", 4, 1).unwrap();
        let names: Vec<&str> = reg.iter().map(|(n, _)| n).collect();
        assert_eq!(names, vec!["a.weight", "b.weight", "c.weight"]);
    }
}
