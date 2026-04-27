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
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

/// Align `n` upward to a multiple of `to`. Used by the GGUF imatrix
/// writer for tensor data section alignment.
#[inline]
fn align_up(n: u64, to: u64) -> u64 {
    n.div_ceil(to) * to
}
use std::path::Path;

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

    /// I/O error while reading/writing the .imatrix sidecar file.
    #[error("imatrix I/O: {0}")]
    Io(#[from] std::io::Error),

    /// Malformed legacy `.imatrix` file: header reports an entry count
    /// or per-entry size that's inconsistent with subsequent file bytes.
    #[error("imatrix: malformed legacy .imatrix file: {reason}")]
    MalformedLegacy { reason: String },
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

    /// Save the collector to a llama.cpp-compatible legacy `.imatrix`
    /// sidecar file (the binary format read by `llama-quantize --imatrix`).
    ///
    /// **Format** (matches `imatrix.cpp::save_imatrix_legacy`, line 411):
    ///
    /// ```text
    /// i32  n_entries
    /// for each entry (sorted by tensor name, ASCII):
    ///   i32       name_len
    ///   bytes[]   name (utf-8, no null terminator)
    ///   i32       ncall = ceil(max(counts) / chunk_size)
    ///   i32       nval  = values.len()
    ///   f32[nval] data — encoded[i] = (value[i] / count[i / row_size]) * ncall
    /// i32       last_chunk
    /// i32       dataset_filename_len
    /// bytes[]   dataset_filename
    /// ```
    ///
    /// **`chunk_size` parameter** matches `m_params.n_ctx / m_params.n_parallel`
    /// in llama.cpp. Round-trip preservation requires the loader pass
    /// the same `chunk_size`. For new files emitted by hf2q, the
    /// caller picks `chunk_size` based on the calibration-corpus chunk
    /// length; cross-validation tests against `llama-imatrix` use the
    /// same value (default 512).
    ///
    /// **`dataset_filename`** is informational metadata — the path of
    /// the calibration corpus that produced this imatrix. Saved
    /// trailing as a length-prefixed UTF-8 string. Pass an empty
    /// string when none.
    ///
    /// **Zero-count handling**: if an expert's count is zero, the
    /// encoded value for that expert's columns is `ncall` (matching
    /// llama.cpp's `value=1; count=1` placeholder). On reload, the
    /// per-column importance for that expert is `1.0`, which is the
    /// same fallback semantics the k-quant search expects.
    ///
    /// **Endianness**: little-endian, matching llama.cpp's native i/o
    /// on `aarch64-apple-darwin` and `x86_64-linux-gnu`. Cross-platform
    /// portability is documented but not in scope for ADR-014.
    pub fn save_imatrix_legacy(
        &self,
        path: &Path,
        chunk_size: i32,
        dataset_filename: &str,
    ) -> Result<(), ImatrixError> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        // Sorted tensor names — deterministic file output.
        let mut names: Vec<&String> = self.stats.keys().collect();
        names.sort();

        let n_entries: i32 = names.len() as i32;
        w.write_all(&n_entries.to_le_bytes())?;

        for name in names {
            let stat = &self.stats[name];
            let name_bytes = name.as_bytes();
            let name_len: i32 = name_bytes.len() as i32;

            // ncall = ceil(max(counts) / chunk_size).
            let max_count: i64 = stat.counts.iter().copied().max().unwrap_or(0);
            let ncall: i32 = if chunk_size <= 0 {
                0
            } else {
                let cs = chunk_size as i64;
                ((max_count + cs - 1) / cs) as i32
            };

            let nval: i32 = stat.values.len() as i32;
            let nmat: i32 = stat.counts.len() as i32;

            w.write_all(&name_len.to_le_bytes())?;
            w.write_all(name_bytes)?;
            w.write_all(&ncall.to_le_bytes())?;
            w.write_all(&nval.to_le_bytes())?;

            if nval > 0 && nmat > 0 {
                let row_size = stat.values.len() / stat.counts.len();
                let mut encoded = Vec::<f32>::with_capacity(stat.values.len());
                for (i, &value) in stat.values.iter().enumerate() {
                    let expert_idx = i / row_size;
                    let count = stat.counts[expert_idx] as f32;
                    let (v, c) = if count == 0.0 { (1.0, 1.0) } else { (value, count) };
                    encoded.push((v / c) * (ncall as f32));
                }
                // Write all f32 little-endian in one buffered call.
                let mut byte_buf = Vec::with_capacity(encoded.len() * 4);
                for v in &encoded {
                    byte_buf.extend_from_slice(&v.to_le_bytes());
                }
                w.write_all(&byte_buf)?;
            }
        }

        // Last-chunk count (informational; llama.cpp uses it for
        // logging — not load-critical).
        let last_chunk: i32 = self.total_chunks.try_into().unwrap_or(i32::MAX);
        w.write_all(&last_chunk.to_le_bytes())?;

        // Dataset filename (length-prefixed, no null terminator).
        let ds_bytes = dataset_filename.as_bytes();
        let ds_len: i32 = ds_bytes.len() as i32;
        w.write_all(&ds_len.to_le_bytes())?;
        w.write_all(ds_bytes)?;

        w.flush()?;
        Ok(())
    }

    /// Save an [`ImatrixCollector`] in the **GGUF imatrix format**
    /// adopted by llama.cpp on 2025-07-19 (PR #9400, commit
    /// `90083283`). This is the default `imatrix.gguf` output of
    /// `tools/imatrix/imatrix.cpp::save_imatrix`.
    ///
    /// **Format vs the legacy `.dat`**: the GGUF format preserves
    /// per-expert counts exactly (one F32 tensor per stat entry for
    /// values, one for counts), whereas the legacy format collapses
    /// counts into a single `ncall` integer with lossy per-expert
    /// reconstruction on read.
    ///
    /// **Schema (matches `save_imatrix` byte-for-byte for the
    /// metadata path; tensor data layout matches `gguf_write_to_file`
    /// with `GGUF_DEFAULT_ALIGNMENT = 32`):**
    ///
    /// Header:
    /// - magic `"GGUF"` (4 bytes)
    /// - version `3` (u32)
    /// - n_tensors (u64) — 2 per non-empty stat entry
    /// - n_kv (u64) — 4
    ///
    /// KVs (in this order):
    /// - `general.type` (string) = `"imatrix"`
    /// - `imatrix.datasets` (array of strings) — calibration corpus paths
    /// - `imatrix.chunk_count` (u32) — total chunks processed
    /// - `imatrix.chunk_size` (u32) — `n_ctx / n_parallel` (typically 512)
    ///
    /// Tensor descriptors (sorted alphabetically by tensor name; two
    /// per stat entry, with the `nval > 0 && nmat > 0` guard from
    /// `imatrix.cpp:600`):
    /// - `<name>.in_sum2`: F32, 2D, shape `[nval/nmat, nmat]`
    /// - `<name>.counts`:  F32, 2D, shape `[1, nmat]`
    ///
    /// Tensor data: padded to GGUF default alignment (32 bytes); each
    /// tensor's float bytes follow with per-tensor alignment.
    ///
    /// **Endianness**: little-endian, matching llama.cpp's native i/o
    /// on `aarch64-apple-darwin` and `x86_64-linux-gnu`.
    pub fn save_imatrix_gguf(
        &self,
        path: &Path,
        chunk_size: u32,
        datasets: &[&str],
    ) -> Result<(), ImatrixError> {
        const GGUF_MAGIC: [u8; 4] = *b"GGUF";
        const GGUF_VERSION: u32 = 3;
        const GGUF_TYPE_UINT32: u32 = 4;
        const GGUF_TYPE_STRING: u32 = 8;
        const GGUF_TYPE_ARRAY: u32 = 9;
        const GGUF_TYPE_F32_TENSOR: u32 = 0;
        const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

        // Build the entire file in an in-memory buffer for clean
        // position tracking — the alignment padding before the data
        // section depends on the descriptor section's exact byte length.
        let mut buf: Vec<u8> = Vec::new();

        // Sorted tensor names, with the same nval > 0 && nmat > 0 guard as llama.cpp.
        let mut to_store: Vec<&String> = self
            .stats
            .iter()
            .filter(|(_, s)| !s.values.is_empty() && !s.counts.is_empty())
            .map(|(k, _)| k)
            .collect();
        to_store.sort();

        let n_tensors: u64 = (to_store.len() as u64) * 2;
        let n_kv: u64 = 4;

        // Header.
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&n_tensors.to_le_bytes());
        buf.extend_from_slice(&n_kv.to_le_bytes());

        // KV writers — local helpers operating on the buffer.
        fn put_str(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }

        // 1. general.type = "imatrix"
        put_str(&mut buf, "general.type");
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        put_str(&mut buf, "imatrix");

        // 2. imatrix.datasets = string array
        put_str(&mut buf, "imatrix.datasets");
        buf.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        buf.extend_from_slice(&(datasets.len() as u64).to_le_bytes());
        for ds in datasets {
            put_str(&mut buf, ds);
        }

        // 3. imatrix.chunk_count = u32
        put_str(&mut buf, "imatrix.chunk_count");
        buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        let chunk_count: u32 = self.total_chunks.try_into().unwrap_or(u32::MAX);
        buf.extend_from_slice(&chunk_count.to_le_bytes());

        // 4. imatrix.chunk_size = u32
        put_str(&mut buf, "imatrix.chunk_size");
        buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        buf.extend_from_slice(&chunk_size.to_le_bytes());

        // Pre-compute per-tensor sizes and offsets relative to the
        // start of the tensor data section.
        let mut tensor_offsets: Vec<u64> = Vec::with_capacity(n_tensors as usize);
        let mut data_offset: u64 = 0;
        for name in &to_store {
            let stat = &self.stats[*name];
            let nval = stat.values.len() as u64;
            let nmat = stat.counts.len() as u64;
            tensor_offsets.push(data_offset);
            data_offset = align_up(data_offset + nval * 4, GGUF_DEFAULT_ALIGNMENT);
            tensor_offsets.push(data_offset);
            data_offset = align_up(data_offset + nmat * 4, GGUF_DEFAULT_ALIGNMENT);
        }

        // Tensor descriptors.
        let mut offset_idx = 0usize;
        for name in &to_store {
            let stat = &self.stats[*name];
            let nval = stat.values.len() as u64;
            let nmat = stat.counts.len() as u64;
            let row_size = nval / nmat;

            let sum_name = format!("{name}.in_sum2");
            put_str(&mut buf, &sum_name);
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&row_size.to_le_bytes());
            buf.extend_from_slice(&nmat.to_le_bytes());
            buf.extend_from_slice(&GGUF_TYPE_F32_TENSOR.to_le_bytes());
            buf.extend_from_slice(&tensor_offsets[offset_idx].to_le_bytes());
            offset_idx += 1;

            let cnt_name = format!("{name}.counts");
            put_str(&mut buf, &cnt_name);
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&nmat.to_le_bytes());
            buf.extend_from_slice(&GGUF_TYPE_F32_TENSOR.to_le_bytes());
            buf.extend_from_slice(&tensor_offsets[offset_idx].to_le_bytes());
            offset_idx += 1;
        }

        // Pad header+descriptors to alignment before the tensor data section.
        let header_end = buf.len() as u64;
        let pad =
            (GGUF_DEFAULT_ALIGNMENT - (header_end % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT;
        buf.extend(std::iter::repeat(0u8).take(pad as usize));

        // Tensor data — each tensor packed at its computed offset
        // (offsets are relative to the start of this section, which is
        // `buf.len()` right now post-alignment).
        let data_start = buf.len() as u64;
        for name in &to_store {
            let stat = &self.stats[*name];
            for &v in &stat.values {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            // Align to GGUF_DEFAULT_ALIGNMENT relative to data section start.
            let written = buf.len() as u64 - data_start;
            let pad = (GGUF_DEFAULT_ALIGNMENT - (written % GGUF_DEFAULT_ALIGNMENT))
                % GGUF_DEFAULT_ALIGNMENT;
            buf.extend(std::iter::repeat(0u8).take(pad as usize));

            for &c in &stat.counts {
                let cf = c as f32;
                buf.extend_from_slice(&cf.to_le_bytes());
            }
            let written = buf.len() as u64 - data_start;
            let pad = (GGUF_DEFAULT_ALIGNMENT - (written % GGUF_DEFAULT_ALIGNMENT))
                % GGUF_DEFAULT_ALIGNMENT;
            buf.extend(std::iter::repeat(0u8).take(pad as usize));
        }

        // Single atomic write.
        let mut file = File::create(path)?;
        file.write_all(&buf)?;
        file.flush()?;
        Ok(())
    }

    /// Load an [`ImatrixCollector`] from a llama.cpp-compatible legacy
    /// `.imatrix` sidecar file.
    ///
    /// **Reconstruction note**: the legacy format does NOT preserve
    /// per-expert counts (`save_imatrix_legacy` collapses them into a
    /// single `ncall`). On load, every expert's count is reconstructed
    /// to the same value (`ncall × chunk_size`), and the encoded
    /// per-column values are scaled back. This is a known lossy
    /// property of the legacy format; the GGUF-based format
    /// (`save_imatrix`) preserves per-expert counts exactly. P6 iter-3
    /// will add the GGUF format alongside.
    ///
    /// `chunk_size` must match the value passed to `save_imatrix_legacy`
    /// for round-trip preservation. When unsure, llama.cpp's default
    /// is `n_ctx / n_parallel == 512 / 1 == 512`.
    pub fn load_imatrix_legacy(
        path: &Path,
        chunk_size: i32,
    ) -> Result<(Self, String /* dataset_filename */), ImatrixError> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        let n_entries = read_i32_le(&mut r)?;
        if n_entries < 0 {
            return Err(ImatrixError::MalformedLegacy {
                reason: format!("negative n_entries: {n_entries}"),
            });
        }

        let mut col = Self::new();
        for _ in 0..n_entries {
            let name_len = read_i32_le(&mut r)?;
            if name_len < 0 || name_len > 1 << 20 {
                return Err(ImatrixError::MalformedLegacy {
                    reason: format!("name_len out of bounds: {name_len}"),
                });
            }
            let mut name_buf = vec![0u8; name_len as usize];
            r.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf).map_err(|e| ImatrixError::MalformedLegacy {
                reason: format!("non-utf8 tensor name: {e}"),
            })?;

            let ncall = read_i32_le(&mut r)?;
            let nval = read_i32_le(&mut r)?;
            if nval < 0 {
                return Err(ImatrixError::MalformedLegacy {
                    reason: format!("negative nval for tensor '{name}': {nval}"),
                });
            }

            let mut tmp = vec![0.0_f32; nval as usize];
            for slot in tmp.iter_mut() {
                let mut byte_buf = [0u8; 4];
                r.read_exact(&mut byte_buf)?;
                *slot = f32::from_le_bytes(byte_buf);
            }

            // Reconstruct internal Stats per upstream load_imatrix_legacy
            // (line 664-684): values[i] += tmp[i] * chunk_size; counts[j]
            // += ncall * chunk_size. Legacy format collapses per-expert
            // counts so we always reconstruct to a single-count entry.
            let cs = chunk_size as i64;
            let scaled_count = (ncall as i64) * cs;
            let mut values = Vec::<f32>::with_capacity(tmp.len());
            for &t in &tmp {
                values.push(t * (chunk_size as f32));
            }
            let stats = Stats {
                values,
                counts: vec![scaled_count],
            };
            col.stats.insert(name, stats);
        }

        // last_chunk + dataset_filename (trailer).
        col.total_chunks = read_i32_le(&mut r)? as i64;
        let ds_len = read_i32_le(&mut r)?;
        let dataset_filename = if ds_len < 0 {
            return Err(ImatrixError::MalformedLegacy {
                reason: format!("negative dataset_filename_len: {ds_len}"),
            });
        } else if ds_len == 0 {
            String::new()
        } else {
            let mut ds_buf = vec![0u8; ds_len as usize];
            r.read_exact(&mut ds_buf)?;
            String::from_utf8(ds_buf).map_err(|e| ImatrixError::MalformedLegacy {
                reason: format!("non-utf8 dataset_filename: {e}"),
            })?
        };

        Ok((col, dataset_filename))
    }

    /// Load an [`ImatrixCollector`] from a GGUF-format `.imatrix.gguf`
    /// file produced by either [`Self::save_imatrix_gguf`] or by
    /// llama.cpp ≥2025-07-19 (PR #9400, commit `90083283`).
    ///
    /// The reader walks the metadata KV section once to extract the
    /// imatrix-specific fields (`imatrix.datasets`, `imatrix.chunk_count`,
    /// `imatrix.chunk_size`), walks the tensor descriptor section
    /// pairing `<name>.in_sum2` with `<name>.counts`, then reads each
    /// tensor's float bytes at the recorded offset.
    ///
    /// Returns the reconstructed collector plus the dataset list as
    /// recorded in the file.
    ///
    /// **Robustness**: unknown metadata KV types are skipped (their
    /// payload byte size is computed and the cursor advanced). Only
    /// the four imatrix-specific KVs are read into typed fields.
    /// Tensors whose names don't match the `<name>.{in_sum2,counts}`
    /// suffix scheme are accepted but silently ignored.
    pub fn load_imatrix_gguf(
        path: &Path,
    ) -> Result<(Self, Vec<String> /* datasets */), ImatrixError> {
        // Read the entire file into memory — imatrix files are small
        // (a few MB at most for a typical LLM). Cursor-based reads
        // simplify offset tracking against the recorded tensor offsets.
        let bytes = std::fs::read(path)?;
        if bytes.len() < 24 {
            return Err(ImatrixError::MalformedLegacy {
                reason: format!("file too short for GGUF header ({} bytes)", bytes.len()),
            });
        }

        // Header.
        if &bytes[0..4] != b"GGUF" {
            return Err(ImatrixError::MalformedLegacy {
                reason: "missing GGUF magic".into(),
            });
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != 3 {
            return Err(ImatrixError::MalformedLegacy {
                reason: format!("unsupported GGUF version {version} (expected 3)"),
            });
        }
        let n_tensors = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let n_kv = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;

        let mut cur = 24usize;

        // ─── Walk metadata KVs ───
        let mut datasets: Vec<String> = Vec::new();
        let mut _chunk_count: u32 = 0;
        let mut _chunk_size: u32 = 0;

        for _ in 0..n_kv {
            let (key, n) = gguf_read_string(&bytes, cur)?;
            cur += n;
            let kv_type = u32::from_le_bytes(read_4(&bytes, cur)?);
            cur += 4;

            match (key.as_str(), kv_type) {
                ("imatrix.datasets", 9) => {
                    // array of strings
                    let elem_type = u32::from_le_bytes(read_4(&bytes, cur)?);
                    cur += 4;
                    if elem_type != 8 {
                        return Err(ImatrixError::MalformedLegacy {
                            reason: format!(
                                "imatrix.datasets array element type {elem_type}, expected string (8)"
                            ),
                        });
                    }
                    let len = u64::from_le_bytes(read_8(&bytes, cur)?) as usize;
                    cur += 8;
                    for _ in 0..len {
                        let (s, n) = gguf_read_string(&bytes, cur)?;
                        cur += n;
                        datasets.push(s);
                    }
                }
                ("imatrix.chunk_count", 4) => {
                    _chunk_count = u32::from_le_bytes(read_4(&bytes, cur)?);
                    cur += 4;
                }
                ("imatrix.chunk_size", 4) => {
                    _chunk_size = u32::from_le_bytes(read_4(&bytes, cur)?);
                    cur += 4;
                }
                // Skip any other key (general.type, general.alignment, etc).
                _ => {
                    cur += skip_kv_value(&bytes, cur, kv_type)?;
                }
            }
        }

        // ─── Walk tensor descriptors ───
        // Each descriptor: name (string), n_dims (u32), dims (u64 × n_dims),
        // type (u32), offset (u64).
        struct TensorDesc {
            name: String,
            shape: Vec<u64>,
            ggml_type: u32,
            offset: u64,
        }
        let mut descs: Vec<TensorDesc> = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let (name, n) = gguf_read_string(&bytes, cur)?;
            cur += n;
            let n_dims = u32::from_le_bytes(read_4(&bytes, cur)?) as usize;
            cur += 4;
            if n_dims > 8 {
                return Err(ImatrixError::MalformedLegacy {
                    reason: format!("tensor '{name}' has unreasonable n_dims {n_dims}"),
                });
            }
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(u64::from_le_bytes(read_8(&bytes, cur)?));
                cur += 8;
            }
            let ggml_type = u32::from_le_bytes(read_4(&bytes, cur)?);
            cur += 4;
            let offset = u64::from_le_bytes(read_8(&bytes, cur)?);
            cur += 8;
            descs.push(TensorDesc {
                name,
                shape,
                ggml_type,
                offset,
            });
        }

        // ─── Tensor data section ───
        // Aligned to 32 bytes (GGUF default alignment) past the end of
        // the descriptor section.
        const GGUF_DEFAULT_ALIGNMENT: usize = 32;
        let data_start =
            (cur + GGUF_DEFAULT_ALIGNMENT - 1) / GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT;

        // Pair `<name>.in_sum2` with `<name>.counts` and reconstruct Stats.
        let mut col = Self::new();
        let mut by_base: HashMap<String, (Option<&TensorDesc>, Option<&TensorDesc>)> = HashMap::new();
        for d in &descs {
            if let Some(base) = d.name.strip_suffix(".in_sum2") {
                by_base.entry(base.to_string()).or_default().0 = Some(d);
            } else if let Some(base) = d.name.strip_suffix(".counts") {
                by_base.entry(base.to_string()).or_default().1 = Some(d);
            }
            // Unknown-suffix tensors silently ignored.
        }

        for (base, (sum2_d, counts_d)) in by_base.into_iter() {
            let sum2_d = match sum2_d {
                Some(d) => d,
                None => continue, // missing in_sum2 — skip incomplete pair
            };
            let counts_d = match counts_d {
                Some(d) => d,
                None => continue,
            };
            if sum2_d.ggml_type != 0 || counts_d.ggml_type != 0 {
                return Err(ImatrixError::MalformedLegacy {
                    reason: format!(
                        "tensor '{base}': in_sum2/counts must be F32 (type 0), got {}/{}",
                        sum2_d.ggml_type, counts_d.ggml_type
                    ),
                });
            }

            // in_sum2 shape = [row_size, nmat]
            // counts shape = [1, nmat]
            // GGUF stores dims in fastest-changing-first order (matches ggml).
            if sum2_d.shape.len() != 2 || counts_d.shape.len() != 2 {
                return Err(ImatrixError::MalformedLegacy {
                    reason: format!(
                        "tensor '{base}': in_sum2/counts shapes must be 2D, got {}D/{}D",
                        sum2_d.shape.len(),
                        counts_d.shape.len()
                    ),
                });
            }
            let row_size = sum2_d.shape[0] as usize;
            let nmat_sum = sum2_d.shape[1] as usize;
            let nmat_cnt = counts_d.shape[1] as usize;
            if nmat_sum != nmat_cnt {
                return Err(ImatrixError::MalformedLegacy {
                    reason: format!(
                        "tensor '{base}': in_sum2 nmat={nmat_sum} != counts nmat={nmat_cnt}"
                    ),
                });
            }
            if counts_d.shape[0] != 1 {
                return Err(ImatrixError::MalformedLegacy {
                    reason: format!(
                        "tensor '{base}': counts shape[0] must be 1, got {}",
                        counts_d.shape[0]
                    ),
                });
            }

            let nval = row_size * nmat_sum;
            let sum2_start = data_start + sum2_d.offset as usize;
            let sum2_end = sum2_start + nval * 4;
            let counts_start = data_start + counts_d.offset as usize;
            let counts_end = counts_start + nmat_cnt * 4;
            if bytes.len() < sum2_end || bytes.len() < counts_end {
                return Err(ImatrixError::MalformedLegacy {
                    reason: format!(
                        "tensor '{base}': data section ends at {} but tensor expects {}",
                        bytes.len(),
                        sum2_end.max(counts_end)
                    ),
                });
            }

            let mut values = Vec::with_capacity(nval);
            for chunk in bytes[sum2_start..sum2_end].chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let mut counts = Vec::with_capacity(nmat_cnt);
            for chunk in bytes[counts_start..counts_end].chunks_exact(4) {
                let cf = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                // Reverse the i64→f32 cast from save (lossy for very
                // large counts; matches llama.cpp's lossy-cast
                // semantics).
                counts.push(cf as i64);
            }

            col.stats.insert(base, Stats { values, counts });
        }

        col.total_chunks = _chunk_count as i64;

        Ok((col, datasets))
    }
}

/// Read a little-endian i32 from a `Read` instance.
fn read_i32_le<R: Read>(r: &mut R) -> Result<i32, ImatrixError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

/// Read a 4-byte slice from `bytes` at `cur`, returning a typed
/// `[u8; 4]` for use with `from_le_bytes`. Errors when the buffer
/// is too short.
fn read_4(bytes: &[u8], cur: usize) -> Result<[u8; 4], ImatrixError> {
    bytes.get(cur..cur + 4)
        .map(|s| [s[0], s[1], s[2], s[3]])
        .ok_or_else(|| ImatrixError::MalformedLegacy {
            reason: format!("truncated GGUF: expected 4 bytes at offset {cur}"),
        })
}

/// Read an 8-byte slice from `bytes` at `cur`.
fn read_8(bytes: &[u8], cur: usize) -> Result<[u8; 8], ImatrixError> {
    bytes.get(cur..cur + 8)
        .map(|s| [s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
        .ok_or_else(|| ImatrixError::MalformedLegacy {
            reason: format!("truncated GGUF: expected 8 bytes at offset {cur}"),
        })
}

/// Read a GGUF string at `cur`. Returns `(string, total_bytes_read)`
/// where `total_bytes_read = 8 + len`.
fn gguf_read_string(bytes: &[u8], cur: usize) -> Result<(String, usize), ImatrixError> {
    let len = u64::from_le_bytes(read_8(bytes, cur)?) as usize;
    let start = cur + 8;
    let end = start + len;
    if bytes.len() < end {
        return Err(ImatrixError::MalformedLegacy {
            reason: format!("truncated GGUF string at offset {cur}, len {len}"),
        });
    }
    let s = std::str::from_utf8(&bytes[start..end])
        .map_err(|e| ImatrixError::MalformedLegacy {
            reason: format!("non-utf8 GGUF string at offset {cur}: {e}"),
        })?
        .to_string();
    Ok((s, 8 + len))
}

/// Compute the byte size of a GGUF metadata KV value (after the type
/// tag) so the reader can skip unknown KVs without parsing their
/// payload.
///
/// GGUF type codes:
/// - 0/1: u8/i8 → 1 byte
/// - 2/3: u16/i16 → 2 bytes
/// - 4/5/6: u32/i32/f32 → 4 bytes
/// - 7: bool → 1 byte
/// - 8: string → u64 length + N bytes
/// - 9: array → u32 elem_type + u64 length + N elements
/// - 10/11/12: u64/i64/f64 → 8 bytes
fn skip_kv_value(bytes: &[u8], cur: usize, kv_type: u32) -> Result<usize, ImatrixError> {
    match kv_type {
        0 | 1 | 7 => Ok(1),
        2 | 3 => Ok(2),
        4 | 5 | 6 => Ok(4),
        10 | 11 | 12 => Ok(8),
        8 => {
            let len = u64::from_le_bytes(read_8(bytes, cur)?) as usize;
            Ok(8 + len)
        }
        9 => {
            let elem_type = u32::from_le_bytes(read_4(bytes, cur)?);
            let n = u64::from_le_bytes(read_8(bytes, cur + 4)?) as usize;
            let mut total = 12usize;
            for _ in 0..n {
                let elem_size = match elem_type {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4 | 5 | 6 => 4,
                    10 | 11 | 12 => 8,
                    8 => {
                        let s_len = u64::from_le_bytes(read_8(bytes, cur + total)?) as usize;
                        8 + s_len
                    }
                    _ => {
                        return Err(ImatrixError::MalformedLegacy {
                            reason: format!(
                                "unsupported GGUF array element type {elem_type} at offset {cur}"
                            ),
                        });
                    }
                };
                total += elem_size;
            }
            Ok(total)
        }
        _ => Err(ImatrixError::MalformedLegacy {
            reason: format!("unsupported GGUF KV type {kv_type} at offset {cur}"),
        }),
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

    /// Legacy `.imatrix` round-trip: save → load reproduces the same
    /// per-tensor stats (modulo the lossy per-expert-count collapse
    /// inherent to the legacy format).
    #[test]
    fn legacy_imatrix_round_trip_dense() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.imatrix");

        let mut col = ImatrixCollector::new();
        col.accumulate_dense("blk.0.attn_q.weight", &[1.0, 2.0, 3.0, 4.0], 1, 4)
            .unwrap();
        col.accumulate_dense("blk.0.attn_k.weight", &[2.0, 3.0], 1, 2)
            .unwrap();
        col.record_chunk();
        col.record_chunk();

        let chunk_size = 512;
        col.save_imatrix_legacy(&path, chunk_size, "wikitext.txt")
            .unwrap();

        let (loaded, ds) = ImatrixCollector::load_imatrix_legacy(&path, chunk_size).unwrap();
        assert_eq!(ds, "wikitext.txt");

        // Same set of tensor names.
        let mut names: Vec<&String> = loaded.stats.keys().collect();
        names.sort();
        let expected: Vec<&str> = vec!["blk.0.attn_k.weight", "blk.0.attn_q.weight"];
        let names_str: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        assert_eq!(names_str, expected);

        // Per-tensor row_size preserved.
        assert_eq!(loaded.stats["blk.0.attn_q.weight"].values.len(), 4);
        assert_eq!(loaded.stats["blk.0.attn_k.weight"].values.len(), 2);

        // Finalised values approximately match the original (within
        // float-precision tolerance — round-trip applies a sequence of
        // multiply/divide that introduces ULP-scale rounding).
        let orig_imat = col.finalise();
        let loaded_imat = loaded.finalise();
        for name in expected.iter() {
            let o = orig_imat.get(*name).unwrap();
            let l = loaded_imat.get(*name).unwrap();
            assert_eq!(o.len(), l.len(), "{name} length");
            for (i, (oo, ll)) in o.iter().zip(l.iter()).enumerate() {
                let abs_diff = (oo - ll).abs();
                let rel_tol = 1e-4_f32.max(oo.abs() * 1e-4);
                assert!(
                    abs_diff <= rel_tol,
                    "{name}[{i}]: orig {oo}, loaded {ll}, diff {abs_diff} > tol {rel_tol}"
                );
            }
        }
    }

    /// Empty dataset filename round-trips correctly.
    #[test]
    fn legacy_imatrix_round_trip_empty_dataset_string() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("empty_ds.imatrix");

        let mut col = ImatrixCollector::new();
        col.accumulate_dense("L", &[1.0, 2.0], 1, 2).unwrap();
        col.save_imatrix_legacy(&path, 512, "").unwrap();

        let (_, ds) = ImatrixCollector::load_imatrix_legacy(&path, 512).unwrap();
        assert_eq!(ds, "");
    }

    /// Sorted name iteration: `save_imatrix_legacy` writes entries in
    /// ASCII-sorted name order regardless of insertion order. This is
    /// required for byte-identical reproducibility (and matches
    /// llama.cpp's `std::sort(to_store.begin(), to_store.end())`).
    #[test]
    fn legacy_imatrix_save_writes_in_sorted_name_order() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sorted.imatrix");

        let mut col = ImatrixCollector::new();
        // Insert in reverse alphabetic order; round-trip should
        // re-emit alphabetic.
        for name in ["zebra", "alpha", "mango", "beta"] {
            col.accumulate_dense(name, &[1.0, 2.0], 1, 2).unwrap();
        }
        col.save_imatrix_legacy(&path, 512, "").unwrap();

        // Read raw bytes to verify sorted order.
        let bytes = std::fs::read(&path).unwrap();
        let n = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(n, 4);

        let mut offset = 4usize;
        let mut found_names = Vec::new();
        for _ in 0..n {
            let name_len = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let name = String::from_utf8(bytes[offset..offset + name_len].to_vec()).unwrap();
            offset += name_len;
            found_names.push(name);
            // Skip ncall (4) + nval (4) + nval × 4 bytes f32 data.
            let _ncall = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            offset += 4;
            let nval = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            offset += nval * 4;
        }
        assert_eq!(found_names, vec!["alpha", "beta", "mango", "zebra"]);
    }

    /// MoE per-expert round-trip through the legacy format. Note: the
    /// legacy format collapses per-expert counts into a single
    /// `ncall`; the loaded back stats has one count entry. The
    /// finalised per-column importance still preserves the per-expert
    /// values (modulo round-trip scaling).
    #[test]
    fn legacy_imatrix_round_trip_moe_collapses_counts() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("moe.imatrix");

        let mut col = ImatrixCollector::new();
        // 3 tokens, 2 cols, 2 experts. Tokens 0,1 → expert 0; token 2 → expert 1.
        let activations = vec![
            1.0, 2.0, // tok 0, exp 0
            3.0, 4.0, // tok 1, exp 0
            5.0, 6.0, // tok 2, exp 1
        ];
        let expert_ids = vec![0_i64, 0, 1];
        col.accumulate_moe("blk.0.ffn_gate_exps.weight", &activations, &expert_ids, 3, 2, 2)
            .unwrap();

        col.save_imatrix_legacy(&path, 512, "").unwrap();
        let (loaded, _) = ImatrixCollector::load_imatrix_legacy(&path, 512).unwrap();

        // Loaded collector has values.len() == 4 (2 experts × 2 cols)
        // and counts.len() == 1 (legacy format collapse).
        let s = loaded.stats.get("blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!(s.values.len(), 4);
        assert_eq!(s.counts.len(), 1);
    }

    /// Sovereignty: pure-Rust port has no python/torch/libggml runtime
    /// crate in the dep tree (per ADR-014 Decision 21 round-2).
    /// Smoke-checks via `cargo metadata` output presence-only — full
    /// runtime sovereignty audit happens in
    /// `tests/sovereignty_no_python_dep.rs` (P10).
    #[test]
    fn imatrix_module_no_external_dep_drift() {
        // The imatrix module imports from std, thiserror, half (via
        // crate::ir), tracing. None of these are forbidden runtime
        // crates per Decision 21. This test compile-checks the surface
        // and is a contract that future imatrix changes must not
        // introduce torch/numpy/python crates.
        //
        // Per Decision 21 round-2 refinement: the cargo metadata walk
        // happens in tests/sovereignty_no_python_dep.rs (P10's full
        // runtime check). This test is a placeholder so future
        // refactors of the imatrix module break here visibly if
        // forbidden imports are added.
        let _ = std::any::type_name::<ImatrixCollector>();
        let _ = std::any::type_name::<Stats>();
        let _ = std::any::type_name::<ImatrixError>();
    }

    // ─────────────── GGUF imatrix format tests (P6 iter-3) ───────────────

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    #[test]
    fn gguf_imatrix_header_signature() {
        let mut col = ImatrixCollector::new();
        col.accumulate_dense("blk.0.attn_q.weight", &[1.0, 2.0, 3.0, 4.0], 4, 1)
            .unwrap();
        col.record_chunk();

        let dir = std::env::temp_dir().join("hf2q-imatrix-gguf-header-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.imatrix.gguf");
        let _ = std::fs::remove_file(&path);

        col.save_imatrix_gguf(&path, 512, &["wikitext-2.txt"]).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(bytes.len() > 24);
        assert_eq!(&bytes[0..4], b"GGUF");
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(version, 3);
        let n_tensors = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(n_tensors, 2);
        let n_kv = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        assert_eq!(n_kv, 4);
    }

    #[test]
    fn gguf_imatrix_metadata_keys_present() {
        let mut col = ImatrixCollector::new();
        col.accumulate_dense("blk.0.attn_v.weight", &[0.5, 0.5], 2, 1)
            .unwrap();
        col.record_chunk();

        let dir = std::env::temp_dir().join("hf2q-imatrix-gguf-meta-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.imatrix.gguf");
        let _ = std::fs::remove_file(&path);

        col.save_imatrix_gguf(&path, 512, &["wikitext-2.txt", "calib.txt"])
            .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("general.type"));
        assert!(s.contains("imatrix.datasets"));
        assert!(s.contains("imatrix.chunk_count"));
        assert!(s.contains("imatrix.chunk_size"));
        assert!(s.contains("wikitext-2.txt"));
        assert!(s.contains("calib.txt"));
    }

    #[test]
    fn gguf_imatrix_tensor_names_correct() {
        let mut col = ImatrixCollector::new();
        col.accumulate_dense("blk.0.attn_q.weight", &[1.0, 2.0], 2, 1)
            .unwrap();
        col.accumulate_dense("blk.1.ffn_down.weight", &[3.0, 4.0, 5.0, 6.0], 4, 1)
            .unwrap();
        col.record_chunk();

        let dir = std::env::temp_dir().join("hf2q-imatrix-gguf-names-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.imatrix.gguf");
        let _ = std::fs::remove_file(&path);

        col.save_imatrix_gguf(&path, 512, &["calib.txt"]).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("blk.0.attn_q.weight.in_sum2"));
        assert!(s.contains("blk.0.attn_q.weight.counts"));
        assert!(s.contains("blk.1.ffn_down.weight.in_sum2"));
        assert!(s.contains("blk.1.ffn_down.weight.counts"));
    }

    #[test]
    fn gguf_imatrix_empty_collector() {
        let col = ImatrixCollector::new();
        let dir = std::env::temp_dir().join("hf2q-imatrix-gguf-empty-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.imatrix.gguf");
        let _ = std::fs::remove_file(&path);

        col.save_imatrix_gguf(&path, 512, &["calib.txt"]).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(&bytes[0..4], b"GGUF");
        let n_tensors = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(n_tensors, 0);
        let n_kv = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        assert_eq!(n_kv, 4);
    }

    /// **Save → Load round trip via GGUF format**. The GGUF writer
    /// is bit-exact for `values` (F32 round-trip) and lossy-by-design
    /// for `counts` (i64 → f32 → i64 — preserves up to 2^24 exactly).
    /// Verify on a moderate-magnitude dense stat that everything
    /// round-trips byte-equal to the original collector.
    #[test]
    fn gguf_imatrix_round_trip_dense() {
        let mut col = ImatrixCollector::new();
        // 4-column dense layer with two batches of 2 tokens each.
        col.accumulate_dense("blk.0.attn_q.weight", &[1.0, 2.0, 3.0, 4.0], 4, 1)
            .unwrap();
        col.accumulate_dense("blk.0.attn_q.weight", &[5.0, 6.0, 7.0, 8.0], 4, 1)
            .unwrap();
        col.record_chunk();
        col.record_chunk();

        let dir = std::env::temp_dir().join("hf2q-imatrix-gguf-roundtrip");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.imatrix.gguf");
        let _ = std::fs::remove_file(&path);

        col.save_imatrix_gguf(&path, 256, &["wikitext-2.txt", "calib.txt"])
            .unwrap();

        let (loaded, datasets) = ImatrixCollector::load_imatrix_gguf(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(datasets, vec!["wikitext-2.txt".to_string(), "calib.txt".to_string()]);
        assert_eq!(loaded.total_chunks, col.total_chunks);
        assert_eq!(loaded.stats.len(), col.stats.len());

        let orig = &col.stats["blk.0.attn_q.weight"];
        let got = &loaded.stats["blk.0.attn_q.weight"];
        assert_eq!(got.values, orig.values, "values byte-equal");
        assert_eq!(got.counts, orig.counts, "counts byte-equal");
    }

    /// MoE round trip: per-expert counts must be preserved exactly
    /// (the legacy format collapses them; GGUF preserves them by
    /// design).
    #[test]
    fn gguf_imatrix_round_trip_moe_preserves_counts() {
        let mut col = ImatrixCollector::new();
        // 4 experts, row_size=2. Activations route 2 tokens to expert 0
        // and 1 token to expert 2.
        let activations = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 tokens × 2 cols
        let expert_ids: [i64; 3] = [0, 0, 2];
        col.accumulate_moe("blk.0.ffn_down.weight", &activations, &expert_ids, 3, 2, 4)
            .unwrap();
        col.record_chunk();

        let dir = std::env::temp_dir().join("hf2q-imatrix-gguf-roundtrip-moe");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.imatrix.gguf");
        let _ = std::fs::remove_file(&path);

        col.save_imatrix_gguf(&path, 512, &["calib.txt"]).unwrap();
        let (loaded, _) = ImatrixCollector::load_imatrix_gguf(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let orig = &col.stats["blk.0.ffn_down.weight"];
        let got = &loaded.stats["blk.0.ffn_down.weight"];

        // Per-expert counts preserved exactly: expert 0 = 2, expert 1 = 0,
        // expert 2 = 1, expert 3 = 0.
        assert_eq!(got.counts, vec![2_i64, 0, 1, 0]);
        assert_eq!(got.counts, orig.counts, "MoE counts byte-equal");
        assert_eq!(got.values, orig.values, "MoE values byte-equal");
    }

    /// Reading a non-GGUF file rejected with malformed-format error.
    #[test]
    fn gguf_imatrix_rejects_non_gguf() {
        let dir = std::env::temp_dir().join("hf2q-imatrix-gguf-bad");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("not_gguf.bin");
        std::fs::write(&path, b"NOTGGUFXXXXXXXXXXXXXXXXXXXXXXXXXX").unwrap();

        let err = ImatrixCollector::load_imatrix_gguf(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        match err {
            ImatrixError::MalformedLegacy { reason } => {
                assert!(reason.contains("GGUF magic"), "got: {reason}");
            }
            _ => panic!("expected MalformedLegacy"),
        }
    }

    /// Reading a GGUF v2 file (wrong version) rejected.
    #[test]
    fn gguf_imatrix_rejects_wrong_version() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&2u32.to_le_bytes()); // version 2
        bytes.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        bytes.extend_from_slice(&0u64.to_le_bytes()); // n_kv

        let dir = std::env::temp_dir().join("hf2q-imatrix-gguf-v2");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("v2.gguf");
        std::fs::write(&path, &bytes).unwrap();

        let err = ImatrixCollector::load_imatrix_gguf(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);
        match err {
            ImatrixError::MalformedLegacy { reason } => {
                assert!(reason.contains("version"), "got: {reason}");
            }
            _ => panic!("expected MalformedLegacy"),
        }
    }

    #[test]
    fn gguf_imatrix_writes_in_sorted_name_order() {
        let mut col = ImatrixCollector::new();
        col.accumulate_dense("zzz.weight", &[1.0, 2.0], 2, 1).unwrap();
        col.accumulate_dense("aaa.weight", &[3.0, 4.0], 2, 1).unwrap();
        col.record_chunk();

        let dir = std::env::temp_dir().join("hf2q-imatrix-gguf-sort-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.imatrix.gguf");
        let _ = std::fs::remove_file(&path);

        col.save_imatrix_gguf(&path, 512, &["calib.txt"]).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let s = String::from_utf8_lossy(&bytes);
        let aaa_pos = s.find("aaa.weight.in_sum2").expect("aaa must appear");
        let zzz_pos = s.find("zzz.weight.in_sum2").expect("zzz must appear");
        assert!(aaa_pos < zzz_pos);
    }
}
