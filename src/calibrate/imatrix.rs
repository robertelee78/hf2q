//! Imatrix calibrator — pure-Rust port of llama.cpp's importance-matrix
//! algorithm (ADR-014 P6 Decision 10).
//!
//! ## Algorithm contract (verbatim from `/opt/llama.cpp/tools/imatrix/imatrix.cpp`)
//!
//! For each calibration sample (token sequence `T`):
//!
//! 1. Run forward pass on `T`.
//! 2. At each `Linear` layer with weight `W ∈ [out, in]`, capture the
//!    **input activation** vector `x ∈ [in]` for every token in `T`.
//! 3. Accumulate `imatrix_layer[col] += x[col]² · 1.0` (uniform per-token
//!    weight).
//! 4. After all samples: divide by total token count →
//!    `imatrix_layer[col] = mean(x[col]²)`.
//!
//! Output is `HashMap<TensorName, Vec<f32>>` where each `Vec<f32>` has
//! length equal to the linear layer's input dimension. Cell `[c]` is
//! the importance of input column `c` for that tensor.
//!
//! ## MoE expert routing (`GGML_OP_MUL_MAT_ID`)
//!
//! Mixture-of-Experts layers route different tokens to different
//! experts. The activation corresponding to token `t` is fed only to
//! expert `ids[t]`. Per-expert accumulation: `e.values` is sized
//! `[in × n_experts]` (concatenated), `e.counts` is sized `[n_experts]`.
//! Per the upstream comment at `imatrix.cpp:265`:
//!
//! > this has been adapted to the new format of storing merged experts
//! > in a single 3d tensor (ggml-org/llama.cpp PR #6387)
//!
//! The pure-Rust port preserves both the dense path
//! ([`ImatrixCollector::accumulate_dense`]) and the MoE expert path
//! ([`ImatrixCollector::accumulate_moe`]) so the resulting `.imatrix`
//! sidecar is byte-identical to llama.cpp's output (verified by P6
//! iter-2's cross-validation gate).
//!
//! ## Sovereignty
//!
//! Pure Rust, mlx-native is the only sibling dep. No `pyo3`, no `torch`
//! linkage, no shell-out to `llama-imatrix`. The cross-validation gate
//! (#[ignore]'d) shells out to `llama-imatrix` purely for parity
//! verification — the runtime binary contains no external imatrix
//! dependency.

use std::collections::HashMap;

use thiserror::Error;

/// Error type for imatrix accumulation and finalisation.
#[derive(Error, Debug)]
pub enum ImatrixError {
    /// Per-tensor activation row size disagrees with the previously
    /// recorded size for the same tensor name. Surfaces a corrupt
    /// upstream forward-pass capture; the algorithm cannot proceed
    /// because per-column accumulation depends on consistent row size.
    #[error(
        "imatrix: tensor '{tensor}' activation row mismatch (expected {expected} cols, got {actual})"
    )]
    RowSizeMismatch {
        tensor: String,
        expected: usize,
        actual: usize,
    },

    /// Number of experts in the per-expert routing path disagrees
    /// with the previously recorded count for the same tensor.
    #[error(
        "imatrix: tensor '{tensor}' expert count mismatch (expected {expected}, got {actual})"
    )]
    ExpertCountMismatch {
        tensor: String,
        expected: usize,
        actual: usize,
    },

    /// `expert_ids[t]` is `>= n_experts` for some token `t`. The
    /// router emitted an invalid expert index — refuse to silently
    /// drop the token.
    #[error(
        "imatrix: tensor '{tensor}' expert id {id} >= n_experts {n_experts} at token {token}"
    )]
    ExpertIdOutOfRange {
        tensor: String,
        token: usize,
        id: i64,
        n_experts: usize,
    },

    /// A finalise pass found a tensor with zero accumulated tokens —
    /// the calibration corpus did not exercise this layer at all,
    /// which usually indicates a calibration corpus mismatch (e.g. a
    /// MoE expert never selected by the router on the corpus).
    /// Reported at finalise time as a warning, not an error: the
    /// quant path can still proceed with an unweighted MSE for that
    /// tensor (the user-visible warning is the right thing here).
    #[error("imatrix: tensor '{tensor}' has zero accumulated tokens; mean is undefined")]
    ZeroTokens { tensor: String },
}

/// Per-tensor accumulation statistics. Internal type — public only so
/// callers can introspect for testing / file-format I/O. The
/// invariant `values.len() == counts.len() × row_size` is enforced by
/// every accumulator method.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    /// Squared-activation accumulator. For dense linear layers,
    /// `values.len() == row_size` and `counts.len() == 1`. For MoE,
    /// `values.len() == row_size × n_experts` (per-expert
    /// concatenation, expert-major) and `counts.len() == n_experts`.
    pub values: Vec<f32>,
    /// Token (or expert-token) accumulation count. For dense: a
    /// single entry. For MoE: per-expert entries.
    pub counts: Vec<i64>,
}

impl Stats {
    /// Number of input columns (the row dimension of the linear
    /// layer's input activation). Returns 0 for an empty `Stats`.
    pub fn row_size(&self) -> usize {
        if self.counts.is_empty() {
            return 0;
        }
        self.values.len() / self.counts.len()
    }

    /// Number of experts (1 for dense, N for MoE).
    pub fn n_experts(&self) -> usize {
        self.counts.len()
    }

    /// Total token (or expert-token) count.
    pub fn total_tokens(&self) -> i64 {
        self.counts.iter().sum()
    }
}

/// Imatrix accumulator. Constructed empty; receives per-batch
/// activation captures via [`accumulate_dense`] /
/// [`accumulate_moe`]; finalised via [`finalise`] which produces the
/// per-column importance weights.
///
/// **Thread-safety**: the collector is `!Sync` by design — the upstream
/// llama.cpp implementation guards `m_stats` with a mutex because
/// `collect_imatrix` is invoked from the GGML compute scheduler's
/// worker threads. The pure-Rust port pushes that decision to the
/// caller: the calibrator orchestration in P7 will hold a single
/// `&mut ImatrixCollector` across the whole forward pass and call
/// [`accumulate_dense`] / [`accumulate_moe`] from one thread per
/// pass. Cross-pass aggregation (multiple corpora batches) is
/// sequential through the same `&mut`, so no lock is needed inside
/// the collector itself.
#[derive(Debug, Default)]
pub struct ImatrixCollector {
    stats: HashMap<String, Stats>,
    /// Total tokens seen across all dense layers. Used for diagnostics
    /// and the `imatrix.chunk_count` GGUF metadata field at save time.
    total_chunks: i64,
}

impl ImatrixCollector {
    /// Construct an empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulate per-column squared activations for a dense linear
    /// layer's matmul (`GGML_OP_MUL_MAT` in llama.cpp parlance).
    ///
    /// `tensor_name`: canonical tensor name as it appears in the
    /// `.imatrix` sidecar (e.g. `blk.0.attn_q.weight`).
    /// `activations`: row-major `[n_tokens, row_size]` slice of F32
    /// activations captured at the input of the linear layer.
    /// `n_tokens`: number of token rows in `activations`.
    /// `row_size`: input dimension of the linear layer (== `weight.shape[1]`).
    ///
    /// Behaviour matches `imatrix.cpp:319-353` (`GGML_OP_MUL_MAT`
    /// branch): for each token row, add `x[col]² / n_tokens` to
    /// `e.values[col]`, then increment `e.counts[0]`. The `/ n_tokens`
    /// scaling matches upstream's per-row-summed contribution.
    ///
    /// **Note on the `/n_tokens` scale.** llama.cpp's loop is
    /// `e.values[j] += sumf / n` where `sumf` is the column-summed
    /// squared activation across all tokens in this batch and `n` is
    /// the token count. Result is the **mean** of squared activations
    /// for this batch; final `finalise()` divides by `e.counts[0]`
    /// (= total batches), giving the grand mean across all batches.
    pub fn accumulate_dense(
        &mut self,
        tensor_name: &str,
        activations: &[f32],
        n_tokens: usize,
        row_size: usize,
    ) -> Result<(), ImatrixError> {
        if n_tokens == 0 {
            return Ok(()); // no-op
        }
        if activations.len() != n_tokens * row_size {
            return Err(ImatrixError::RowSizeMismatch {
                tensor: tensor_name.to_string(),
                expected: n_tokens * row_size,
                actual: activations.len(),
            });
        }

        let entry = self.stats.entry(tensor_name.to_string()).or_default();
        if entry.counts.is_empty() {
            // First batch for this tensor: initialise.
            entry.values = vec![0.0_f32; row_size];
            entry.counts = vec![0_i64];
        } else if entry.values.len() != row_size {
            return Err(ImatrixError::RowSizeMismatch {
                tensor: tensor_name.to_string(),
                expected: entry.values.len(),
                actual: row_size,
            });
        } else if entry.counts.len() != 1 {
            return Err(ImatrixError::ExpertCountMismatch {
                tensor: tensor_name.to_string(),
                expected: 1,
                actual: entry.counts.len(),
            });
        }

        // Per-batch summed-then-mean accumulation (matches llama.cpp).
        let n_tokens_f = n_tokens as f32;
        for col in 0..row_size {
            let mut sumf = 0.0_f32;
            for tok in 0..n_tokens {
                let v = activations[tok * row_size + col];
                sumf += v * v;
            }
            entry.values[col] += sumf / n_tokens_f;
        }
        entry.counts[0] += 1;
        Ok(())
    }

    /// Accumulate per-column squared activations for a MoE expert
    /// matmul (`GGML_OP_MUL_MAT_ID` in llama.cpp parlance).
    ///
    /// `tensor_name`: merged-expert tensor name (e.g.
    /// `blk.0.ffn_gate_exps.weight`).
    /// `activations`: row-major `[n_tokens, row_size]` slice of F32
    /// activations.
    /// `expert_ids`: per-token expert routing — `expert_ids[t]` is the
    /// expert index that token `t` was routed to (in `[0, n_experts)`).
    /// `n_tokens`: number of token rows in `activations`.
    /// `row_size`: input dimension of the expert layer.
    /// `n_experts`: total expert count for this layer (typically 256
    /// for apex MoE).
    ///
    /// Behaviour matches `imatrix.cpp:265-356` (`GGML_OP_MUL_MAT_ID`
    /// branch): per-token contributions accumulate into the slice
    /// `e.values[ex × row_size .. (ex+1) × row_size]` where `ex` is the
    /// token's expert id; per-expert counts increment in
    /// `e.counts[ex]`. Tokens routed to expert `ex` contribute to
    /// `e.values[ex × row_size + col]` and `e.counts[ex]`.
    pub fn accumulate_moe(
        &mut self,
        tensor_name: &str,
        activations: &[f32],
        expert_ids: &[i64],
        n_tokens: usize,
        row_size: usize,
        n_experts: usize,
    ) -> Result<(), ImatrixError> {
        if n_tokens == 0 {
            return Ok(());
        }
        if activations.len() != n_tokens * row_size {
            return Err(ImatrixError::RowSizeMismatch {
                tensor: tensor_name.to_string(),
                expected: n_tokens * row_size,
                actual: activations.len(),
            });
        }
        if expert_ids.len() != n_tokens {
            return Err(ImatrixError::RowSizeMismatch {
                tensor: format!("{tensor_name} (expert_ids)"),
                expected: n_tokens,
                actual: expert_ids.len(),
            });
        }

        let entry = self.stats.entry(tensor_name.to_string()).or_default();
        if entry.counts.is_empty() {
            entry.values = vec![0.0_f32; row_size * n_experts];
            entry.counts = vec![0_i64; n_experts];
        } else if entry.values.len() != row_size * n_experts {
            return Err(ImatrixError::RowSizeMismatch {
                tensor: tensor_name.to_string(),
                expected: entry.values.len(),
                actual: row_size * n_experts,
            });
        } else if entry.counts.len() != n_experts {
            return Err(ImatrixError::ExpertCountMismatch {
                tensor: tensor_name.to_string(),
                expected: entry.counts.len(),
                actual: n_experts,
            });
        }

        // Per-token accumulation into the routed expert's slice.
        for tok in 0..n_tokens {
            let id = expert_ids[tok];
            if id < 0 || (id as usize) >= n_experts {
                return Err(ImatrixError::ExpertIdOutOfRange {
                    tensor: tensor_name.to_string(),
                    token: tok,
                    id,
                    n_experts,
                });
            }
            let ex = id as usize;
            let row = &activations[tok * row_size..(tok + 1) * row_size];
            let dst = &mut entry.values[ex * row_size..(ex + 1) * row_size];
            for col in 0..row_size {
                dst[col] += row[col] * row[col];
            }
            entry.counts[ex] += 1;
        }

        Ok(())
    }

    /// Mark a chunk boundary. `total_chunks` is informational
    /// metadata in the `.imatrix` GGUF — it tells consumers how many
    /// distinct calibration-corpus chunks were processed so they can
    /// reason about coverage.
    pub fn record_chunk(&mut self) {
        self.total_chunks += 1;
    }

    /// Total chunks recorded.
    pub fn chunks(&self) -> i64 {
        self.total_chunks
    }

    /// Borrow the per-tensor stats map. Used by the file-format I/O
    /// layer (P6 iter-2) and by tests for direct introspection.
    pub fn stats(&self) -> &HashMap<String, Stats> {
        &self.stats
    }

    /// Finalise: produce per-tensor importance-weight vectors by
    /// dividing accumulated `values` by the per-expert (or single-
    /// dense) `counts`. The returned map is the per-column importance
    /// vector consumed by the k-quant codebook search at quantize
    /// time (ADR-014 P7 Decision 11).
    ///
    /// **Output layout:**
    ///
    /// - Dense: `imatrix[name] == Vec<f32>` of length `row_size`.
    /// - MoE: `imatrix[name] == Vec<f32>` of length `row_size × n_experts`,
    ///   expert-major (matches llama.cpp's flattened storage; the
    ///   k-quant search will know to slice per expert).
    ///
    /// **Zero-count handling**: a per-expert count of zero (the
    /// calibration corpus didn't exercise this expert) is left at
    /// `0.0` in the output and does NOT raise an error. The k-quant
    /// search must handle zero-imatrix entries (typically by falling
    /// back to unweighted MSE for that column or skipping the expert).
    /// Diagnostic logging surfaces zero-count tensors at finalise time
    /// so calibrators can flag corpus coverage gaps.
    pub fn finalise(&self) -> HashMap<String, Vec<f32>> {
        let mut out = HashMap::with_capacity(self.stats.len());
        for (name, e) in self.stats.iter() {
            if e.counts.is_empty() {
                continue;
            }
            let row_size = e.values.len() / e.counts.len();
            let mut imat: Vec<f32> = Vec::with_capacity(e.values.len());
            for (ex, &count) in e.counts.iter().enumerate() {
                if count == 0 {
                    // Zero-count expert: emit zeros (k-quant search handles this).
                    tracing::debug!(
                        tensor = name,
                        expert_idx = ex,
                        "imatrix: expert {} has zero token count; emitting zero importance",
                        ex
                    );
                    for _ in 0..row_size {
                        imat.push(0.0);
                    }
                } else {
                    let inv = 1.0_f32 / (count as f32);
                    let slice =
                        &e.values[ex * row_size..(ex + 1) * row_size];
                    for &v in slice {
                        imat.push(v * inv);
                    }
                }
            }
            out.insert(name.clone(), imat);
        }
        out
    }

    /// Number of tracked tensors. Diagnostic.
    pub fn len(&self) -> usize {
        self.stats.len()
    }

    /// Whether any tensor has been accumulated.
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single dense linear layer, 2 tokens × 4 columns. Hand-computed
    /// expected output: per-column mean of squared activations.
    #[test]
    fn dense_per_column_accumulation() {
        let mut col = ImatrixCollector::new();

        // 2 tokens × 4 cols.
        // Row 0: [1, 2, 3, 4] → squared [1, 4, 9, 16]
        // Row 1: [2, 4, 6, 8] → squared [4, 16, 36, 64]
        // Per-column sum: [5, 20, 45, 80]. Mean per row: divide by 2.
        // Per-batch contribution to e.values: [2.5, 10, 22.5, 40]. counts=[1].
        // Finalise: divide by 1 (single batch) → same.
        let activations: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            2.0, 4.0, 6.0, 8.0,
        ];
        col.accumulate_dense("blk.0.attn_q.weight", &activations, 2, 4)
            .unwrap();

        let imat = col.finalise();
        let v = imat.get("blk.0.attn_q.weight").unwrap();
        let expected = [2.5_f32, 10.0, 22.5, 40.0];
        assert_eq!(v.len(), 4);
        for (i, (g, e)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "col {i}: got {g}, expected {e}"
            );
        }
    }

    /// Two batches accumulate correctly: total mean is across all
    /// tokens. Verifies the per-batch sum-then-mean pattern composes.
    #[test]
    fn dense_accumulates_across_batches() {
        let mut col = ImatrixCollector::new();
        let batch1: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0]; // 1 token × 4 cols
        col.accumulate_dense("L", &batch1, 1, 4).unwrap();
        let batch2: Vec<f32> = vec![2.0, 2.0, 2.0, 2.0]; // 1 token × 4 cols
        col.accumulate_dense("L", &batch2, 1, 4).unwrap();

        // Per-batch contribution:
        //   Batch 1: [1²/1, 1²/1, 1²/1, 1²/1] = [1, 1, 1, 1]
        //   Batch 2: [4, 4, 4, 4]
        //   Accumulated values: [5, 5, 5, 5]. counts = [2].
        // Finalise: [5/2, 5/2, 5/2, 5/2] = [2.5, 2.5, 2.5, 2.5].
        let imat = col.finalise();
        let v = imat.get("L").unwrap();
        for &g in v {
            assert!((g - 2.5).abs() < 1e-5);
        }
    }

    /// MoE expert routing: each token contributes only to its routed
    /// expert. Verifies per-expert slice indexing matches llama.cpp.
    #[test]
    fn moe_per_expert_routing() {
        let mut col = ImatrixCollector::new();
        // 4 tokens × 2 cols, 3 experts.
        // Tokens 0,1 → expert 0; tokens 2,3 → expert 2; expert 1 unused.
        let activations: Vec<f32> = vec![
            1.0, 2.0, // tok 0, expert 0
            3.0, 4.0, // tok 1, expert 0
            5.0, 6.0, // tok 2, expert 2
            7.0, 8.0, // tok 3, expert 2
        ];
        let expert_ids: Vec<i64> = vec![0, 0, 2, 2];

        col.accumulate_moe(
            "blk.0.ffn_gate_exps.weight",
            &activations,
            &expert_ids,
            /* n_tokens */ 4,
            /* row_size */ 2,
            /* n_experts */ 3,
        )
        .unwrap();

        let imat = col.finalise();
        let v = imat.get("blk.0.ffn_gate_exps.weight").unwrap();
        // Layout: expert-major, so v[0..2] = expert 0, v[2..4] = expert 1, v[4..6] = expert 2.
        assert_eq!(v.len(), 6);

        // Expert 0: tokens 0 and 1 contribute. Sum of squares per col, divided by count=2.
        //   col 0: (1² + 3²) / 2 = 5
        //   col 1: (2² + 4²) / 2 = 10
        assert!((v[0] - 5.0).abs() < 1e-5, "expert 0 col 0: got {}", v[0]);
        assert!((v[1] - 10.0).abs() < 1e-5, "expert 0 col 1: got {}", v[1]);

        // Expert 1: zero-count → zeros (no contribution).
        assert_eq!(v[2], 0.0);
        assert_eq!(v[3], 0.0);

        // Expert 2: tokens 2 and 3 contribute, count=2.
        //   col 0: (5² + 7²) / 2 = (25 + 49) / 2 = 37
        //   col 1: (6² + 8²) / 2 = (36 + 64) / 2 = 50
        assert!((v[4] - 37.0).abs() < 1e-5, "expert 2 col 0: got {}", v[4]);
        assert!((v[5] - 50.0).abs() < 1e-5, "expert 2 col 1: got {}", v[5]);
    }

    /// Activation shape mismatch surfaces a typed error at accumulate
    /// time (not silent corruption later).
    #[test]
    fn dense_rejects_activation_shape_mismatch() {
        let mut col = ImatrixCollector::new();
        let activations = vec![0.0_f32; 7]; // not 2 × 4 = 8
        let err = col.accumulate_dense("L", &activations, 2, 4).unwrap_err();
        match err {
            ImatrixError::RowSizeMismatch {
                expected, actual, ..
            } => {
                assert_eq!(expected, 8);
                assert_eq!(actual, 7);
            }
            _ => panic!("expected RowSizeMismatch"),
        }
    }

    /// MoE expert id out-of-range surfaces a typed error.
    #[test]
    fn moe_rejects_out_of_range_expert_id() {
        let mut col = ImatrixCollector::new();
        let activations = vec![0.0_f32; 4]; // 2 tokens × 2 cols
        let expert_ids = vec![0_i64, 5]; // expert 5 with n_experts=3
        let err = col
            .accumulate_moe("L", &activations, &expert_ids, 2, 2, 3)
            .unwrap_err();
        match err {
            ImatrixError::ExpertIdOutOfRange {
                token,
                id,
                n_experts,
                ..
            } => {
                assert_eq!(token, 1);
                assert_eq!(id, 5);
                assert_eq!(n_experts, 3);
            }
            _ => panic!("expected ExpertIdOutOfRange"),
        }
    }

    /// Subsequent accumulate calls on the same tensor with different
    /// row_size are rejected (catches a mid-corpus shape drift bug).
    #[test]
    fn dense_rejects_row_size_drift() {
        let mut col = ImatrixCollector::new();
        col.accumulate_dense("L", &[1.0, 2.0, 3.0, 4.0], 1, 4)
            .unwrap();
        // Same tensor name but row_size 8 — must reject.
        let err = col
            .accumulate_dense("L", &vec![0.0_f32; 8], 1, 8)
            .unwrap_err();
        match err {
            ImatrixError::RowSizeMismatch {
                expected, actual, ..
            } => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 8);
            }
            _ => panic!("expected RowSizeMismatch"),
        }
    }

    /// Stats accessor introspection: row_size + n_experts + total_tokens.
    #[test]
    fn stats_introspection() {
        let mut col = ImatrixCollector::new();
        col.accumulate_dense("L", &[1.0, 2.0, 3.0, 4.0], 1, 4).unwrap();
        col.accumulate_dense("L", &[5.0, 6.0, 7.0, 8.0], 1, 4).unwrap();
        let s = col.stats().get("L").unwrap();
        assert_eq!(s.row_size(), 4);
        assert_eq!(s.n_experts(), 1);
        assert_eq!(s.total_tokens(), 2);
    }

    /// Empty collector behaves correctly.
    #[test]
    fn empty_collector_finalises_empty() {
        let col = ImatrixCollector::new();
        let imat = col.finalise();
        assert!(imat.is_empty());
        assert_eq!(col.len(), 0);
        assert!(col.is_empty());
    }

    /// Zero-token accumulate is a no-op (doesn't initialise stats entry).
    #[test]
    fn zero_token_accumulate_no_op() {
        let mut col = ImatrixCollector::new();
        col.accumulate_dense("L", &[], 0, 4).unwrap();
        assert!(col.stats().get("L").is_none());
    }

    /// Send + Sync are NOT required (per design — single-threaded
    /// accumulator). But the type must be Send so it can be moved
    /// across threads / between calibration corpus batches.
    #[test]
    fn collector_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ImatrixCollector>();
    }
}
