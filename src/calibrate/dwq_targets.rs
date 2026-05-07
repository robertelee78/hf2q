//! ADR-020 iter-14 — `compute_dwq_targets` stream-to-disk port.
//!
//! Direct port of `mlx-lm/mlx_lm/quant/dwq.py:29-66`'s
//! `compute_dwq_targets`: drives the teacher model over a calibration
//! corpus, computes top-K logits per (batch, seq) position, and
//! streams results to disk as one safetensors file per batch under
//! `<save_dir>/{train,valid}/{i:010d}.safetensors`.  Each file
//! contains:
//!
//!   - `"logits"`:  shape `[batch, seq_len-1, top_k]`, FP32
//!   - `"indices"`: shape `[batch, seq_len-1, top_k]`, U32 vocab indices
//!
//! Per `dwq.py:53`, the seq dim is dropped by 1 (last token has no
//! target).  Per `dwq.py:59-60`, mlx-lm uses
//! `argpartition(logits, kth=-K, axis=-1)[..., -K:]` (top-K unsorted)
//! followed by `take_along_axis(logits, idx, axis=-1)` for the
//! corresponding logit values.  We match this exact semantics
//! host-side for iter-14 (see `top_k_per_row` below): the GPU
//! `top_k_f32` kernel caps at K=128 (mlx-lm uses K=1024), and a
//! one-shot teacher precompute is dominated by the teacher forward
//! anyway — host-side BinaryHeap-based partial sort is the right
//! tool here.
//!
//! At consumer time, `dwq.py:113` calls
//! `mx.take_along_axis(logits, ids, axis=-1)` to gather student
//! logits at the same K indices, then computes
//! `kl_div_loss(scale * student, scale * teacher)` per
//! `dwq.py:114`.  Our [`load_dwq_target`] returns the same
//! `(logits, indices)` pair for the consumer.
//!
//! The teacher is supplied via the [`TeacherLogitsProvider`] trait —
//! abstract enough that a synthetic test fixture, a real GGUF model
//! loaded via the existing inference path, or a remote endpoint can
//! all plug in.  Iter-14b will land the GGUF-backed
//! `GgufTeacherProvider` once the model-loading prerequisites are
//! complete.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;

/// Source of teacher logits.  Implementations produce
/// `[batch_size, seq_len, vocab]` FP32 logits for a flat
/// `[batch_size, seq_len]` u32 token-id batch.
///
/// `seq_len` is the FULL sequence length the consumer presents — this
/// trait does not perform the `[:, :-1]` last-token trim that
/// mlx-lm's `compute_dwq_targets` does at the loop level.  The trim
/// is handled in [`compute_dwq_targets`] before the teacher is
/// invoked, so providers can be unaware of next-token-prediction
/// semantics.
pub trait TeacherLogitsProvider {
    /// Run the teacher's forward pass on `tokens` (shape
    /// `[batch_size, seq_len]`, row-major) and return a flat
    /// `[batch_size, seq_len, vocab]` FP32 logit buffer.
    ///
    /// Caller passes `vocab` for sanity-checking the returned
    /// element count: `tokens.len() * vocab` must equal the returned
    /// vec's length.
    fn forward_logits(
        &mut self,
        tokens: &[u32],
        batch_size: usize,
        seq_len: usize,
        vocab: usize,
    ) -> Result<Vec<f32>>;
}

/// Configuration for [`compute_dwq_targets`].
#[derive(Debug, Clone)]
pub struct ComputeTargetsConfig {
    /// Per-batch top-K to retain (mlx-lm default = 1024 per
    /// `dwq.py:59`).
    pub top_k: usize,
    /// Output root directory.  `train` and `valid` split subdirs are
    /// created under this path.
    pub save_dir: PathBuf,
    /// Vocab size of the teacher's logits.
    pub vocab: usize,
}

/// One calibration split (train or valid).  Each batch is a
/// `[batch_size, seq_len]` u32 token-id slab, row-major.  All batches
/// in a split must agree on `batch_size` and `seq_len`.
#[derive(Debug, Clone)]
pub struct CalibrationSplit<'a> {
    pub name: &'static str, // "train" | "valid"
    pub batches: &'a [Vec<u32>],
    pub batch_size: usize,
    pub seq_len: usize,
}

/// Drive the teacher model over one or more calibration splits and
/// stream top-K targets to disk under
/// `<cfg.save_dir>/<split>/<i:010d>.safetensors`.  Returns the number
/// of batches written per split.
pub fn compute_dwq_targets<T: TeacherLogitsProvider>(
    teacher: &mut T,
    splits: &[CalibrationSplit<'_>],
    cfg: &ComputeTargetsConfig,
) -> Result<Vec<(String, usize)>> {
    if cfg.top_k == 0 {
        return Err(anyhow!("compute_dwq_targets: top_k must be > 0"));
    }
    if cfg.top_k > cfg.vocab {
        return Err(anyhow!(
            "compute_dwq_targets: top_k ({}) > vocab ({})",
            cfg.top_k,
            cfg.vocab
        ));
    }
    if cfg.vocab == 0 {
        return Err(anyhow!("compute_dwq_targets: vocab must be > 0"));
    }
    std::fs::create_dir_all(&cfg.save_dir)
        .with_context(|| format!("create save_dir {}", cfg.save_dir.display()))?;

    let mut summary: Vec<(String, usize)> = Vec::with_capacity(splits.len());
    for split in splits {
        let split_dir = cfg.save_dir.join(split.name);
        std::fs::create_dir_all(&split_dir)
            .with_context(|| format!("create split dir {}", split_dir.display()))?;
        let n = write_split(teacher, split, cfg, &split_dir)?;
        summary.push((split.name.to_string(), n));
    }
    Ok(summary)
}

fn write_split<T: TeacherLogitsProvider>(
    teacher: &mut T,
    split: &CalibrationSplit<'_>,
    cfg: &ComputeTargetsConfig,
    split_dir: &Path,
) -> Result<usize> {
    if split.seq_len < 2 {
        return Err(anyhow!(
            "compute_dwq_targets: split '{}' seq_len ({}) must be >= 2 (next-token target needs at least one input + one target)",
            split.name,
            split.seq_len
        ));
    }
    let target_seq = split.seq_len - 1;
    let expected_tokens_per_batch = split.batch_size * split.seq_len;

    for (i, batch) in split.batches.iter().enumerate() {
        if batch.len() != expected_tokens_per_batch {
            return Err(anyhow!(
                "compute_dwq_targets: split '{}' batch[{}] has {} tokens; expected batch_size * seq_len = {}",
                split.name,
                i,
                batch.len(),
                expected_tokens_per_batch
            ));
        }
        // Per dwq.py:53 — drop the last seq position (no next-token
        // target for it).  Slice the input batch to [:, :-1] in flat
        // u32 memory by dropping the trailing seq position of each
        // row.
        let trimmed = trim_last_seq(batch, split.batch_size, split.seq_len);

        let logits =
            teacher
                .forward_logits(&trimmed, split.batch_size, target_seq, cfg.vocab)
                .with_context(|| {
                    format!(
                        "teacher forward on split '{}' batch[{}]",
                        split.name, i
                    )
                })?;
        let expected_logits = split.batch_size * target_seq * cfg.vocab;
        if logits.len() != expected_logits {
            return Err(anyhow!(
                "compute_dwq_targets: teacher returned {} logits; expected batch_size * (seq_len-1) * vocab = {}",
                logits.len(),
                expected_logits
            ));
        }

        // Top-K extraction per row (one row = one (batch, seq)
        // position).  Output shape: [B, S-1, K] for both
        // values and indices.
        let n_rows = split.batch_size * target_seq;
        let mut top_logits: Vec<f32> = Vec::with_capacity(n_rows * cfg.top_k);
        let mut top_indices: Vec<u32> = Vec::with_capacity(n_rows * cfg.top_k);
        for row in 0..n_rows {
            let row_start = row * cfg.vocab;
            let row_logits = &logits[row_start..row_start + cfg.vocab];
            let (row_top_logits, row_top_idx) = top_k_per_row(row_logits, cfg.top_k);
            top_logits.extend_from_slice(&row_top_logits);
            top_indices.extend_from_slice(&row_top_idx);
        }

        let path = split_dir.join(format!("{:010}.safetensors", i));
        write_target_safetensors(
            &path,
            &top_logits,
            &top_indices,
            split.batch_size,
            target_seq,
            cfg.top_k,
        )
        .with_context(|| format!("write safetensors {}", path.display()))?;
    }
    Ok(split.batches.len())
}

/// Slice `[:, :-1]` of a `[batch_size, seq_len]` flat u32 row-major
/// buffer.  Returns a fresh `[batch_size, seq_len-1]` flat buffer.
fn trim_last_seq(tokens: &[u32], batch_size: usize, seq_len: usize) -> Vec<u32> {
    debug_assert_eq!(tokens.len(), batch_size * seq_len);
    let target_seq = seq_len - 1;
    let mut out = Vec::with_capacity(batch_size * target_seq);
    for r in 0..batch_size {
        let row_start = r * seq_len;
        out.extend_from_slice(&tokens[row_start..row_start + target_seq]);
    }
    out
}

/// Host-side top-K extraction matching mlx-lm's
/// `argpartition(... kth=-K)[..., -K:]` semantics: returns the K
/// LARGEST values + their original indices, in DESCENDING order of
/// value.  Ties broken by lower index (deterministic).
///
/// O(V log K) via a min-heap of size K — keeps the smallest of the
/// "currently top-K" at the heap's root and replaces if a new
/// value beats it.  This matches mlx-lm's
/// `argpartition(... kth=-K)[..., -K:]` semantics: it returns the K
/// LARGEST values; the order WITHIN the top-K is implementation-
/// defined for argpartition, but we sort descending here for
/// determinism + downstream consumer simplicity.
fn top_k_per_row(values: &[f32], k: usize) -> (Vec<f32>, Vec<u32>) {
    debug_assert!(k > 0 && k <= values.len());

    // Min-heap over (Reverse(NaN-safe-value), Reverse(idx)) so the
    // smallest top-K candidate is at the root.  We push k items
    // first, then for each remaining value: if v > root, pop+push.
    #[derive(PartialEq)]
    struct Entry {
        // Use the bit pattern to side-step f32 NaN-Ord issues:
        // values arriving here must be finite (asserted by the
        // caller's contract on logit dtype = f32 dense).
        value: f32,
        idx: u32,
    }
    impl Eq for Entry {}
    impl Ord for Entry {
        fn cmp(&self, other: &Self) -> Ordering {
            // NOT total_cmp on the bit pattern (signed/unsigned
            // mismatch); use partial_cmp + a deterministic tie
            // break.  All inputs assumed finite.
            self.value
                .partial_cmp(&other.value)
                .unwrap_or(Ordering::Equal)
                .then_with(|| other.idx.cmp(&self.idx))
        }
    }
    impl PartialOrd for Entry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    // BinaryHeap is a MAX-heap; we want a MIN-heap, so reverse via
    // std::cmp::Reverse.
    let mut heap: BinaryHeap<std::cmp::Reverse<Entry>> = BinaryHeap::with_capacity(k + 1);
    for (idx, &v) in values.iter().enumerate().take(k) {
        heap.push(std::cmp::Reverse(Entry {
            value: v,
            idx: idx as u32,
        }));
    }
    for (idx, &v) in values.iter().enumerate().skip(k) {
        let entry = Entry {
            value: v,
            idx: idx as u32,
        };
        // SAFETY: heap has exactly k > 0 elements at this point.
        let root = &heap.peek().unwrap().0;
        if entry.cmp(root) == Ordering::Greater {
            heap.pop();
            heap.push(std::cmp::Reverse(entry));
        }
    }
    let mut topk: Vec<Entry> = heap.into_iter().map(|r| r.0).collect();
    // Sort descending by value (with deterministic tie-break by
    // lower idx) for downstream consumer simplicity.
    topk.sort_by(|a, b| b.cmp(a));
    let mut out_logits = Vec::with_capacity(k);
    let mut out_idx = Vec::with_capacity(k);
    for e in topk {
        out_logits.push(e.value);
        out_idx.push(e.idx);
    }
    (out_logits, out_idx)
}

fn write_target_safetensors(
    path: &Path,
    logits: &[f32],
    indices: &[u32],
    batch: usize,
    seq: usize,
    top_k: usize,
) -> Result<()> {
    let logits_bytes: Vec<u8> = {
        let mut v = Vec::with_capacity(logits.len() * 4);
        for f in logits {
            v.extend_from_slice(&f.to_le_bytes());
        }
        v
    };
    let indices_bytes: Vec<u8> = {
        let mut v = Vec::with_capacity(indices.len() * 4);
        for u in indices {
            v.extend_from_slice(&u.to_le_bytes());
        }
        v
    };
    let logits_view = TensorView::new(
        Dtype::F32,
        vec![batch, seq, top_k],
        &logits_bytes,
    )
    .map_err(|e| anyhow!("logits TensorView: {e:?}"))?;
    let indices_view = TensorView::new(
        Dtype::U32,
        vec![batch, seq, top_k],
        &indices_bytes,
    )
    .map_err(|e| anyhow!("indices TensorView: {e:?}"))?;

    let bytes = safetensors::tensor::serialize(
        [
            ("logits".to_string(), &logits_view),
            ("indices".to_string(), &indices_view),
        ],
        None,
    )
    .map_err(|e| anyhow!("safetensors serialize: {e:?}"))?;
    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(&bytes)
        .with_context(|| format!("write {}", path.display()))?;
    f.sync_all()
        .with_context(|| format!("sync {}", path.display()))?;
    Ok(())
}

/// Read back a single batch's top-K targets.  Returns
/// `(logits, indices, batch, seq, top_k)` — the metadata is
/// re-derived from the safetensors header so the caller doesn't need
/// to remember `cfg`.
pub fn load_dwq_target(
    save_dir: &Path,
    split: &str,
    idx: usize,
) -> Result<(Vec<f32>, Vec<u32>, usize, usize, usize)> {
    let path = save_dir
        .join(split)
        .join(format!("{:010}.safetensors", idx));
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)
        .map_err(|e| anyhow!("safetensors deserialize {}: {e:?}", path.display()))?;

    let logits_t = st
        .tensor("logits")
        .map_err(|e| anyhow!("missing 'logits' tensor: {e:?}"))?;
    let indices_t = st
        .tensor("indices")
        .map_err(|e| anyhow!("missing 'indices' tensor: {e:?}"))?;

    if logits_t.dtype() != Dtype::F32 {
        return Err(anyhow!(
            "logits dtype {:?} != F32",
            logits_t.dtype()
        ));
    }
    if indices_t.dtype() != Dtype::U32 {
        return Err(anyhow!(
            "indices dtype {:?} != U32",
            indices_t.dtype()
        ));
    }
    if logits_t.shape() != indices_t.shape() {
        return Err(anyhow!(
            "logits shape {:?} != indices shape {:?}",
            logits_t.shape(),
            indices_t.shape()
        ));
    }
    if logits_t.shape().len() != 3 {
        return Err(anyhow!(
            "expected 3-D [batch, seq, top_k]; got shape {:?}",
            logits_t.shape()
        ));
    }
    let batch = logits_t.shape()[0];
    let seq = logits_t.shape()[1];
    let top_k = logits_t.shape()[2];

    let logits_data = logits_t.data();
    let indices_data = indices_t.data();
    if logits_data.len() != batch * seq * top_k * 4 {
        return Err(anyhow!(
            "logits data byte len {} != expected {}",
            logits_data.len(),
            batch * seq * top_k * 4
        ));
    }

    let mut logits = Vec::with_capacity(batch * seq * top_k);
    for chunk in logits_data.chunks_exact(4) {
        logits.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    let mut indices = Vec::with_capacity(batch * seq * top_k);
    for chunk in indices_data.chunks_exact(4) {
        indices.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }

    Ok((logits, indices, batch, seq, top_k))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic teacher: `logits[r, t, v] = (sin(0.01*v + 0.1*r) +
    /// 0.7*cos(0.05*v + 0.3*t)) * 5.0` — deterministic, distinct
    /// per (row, timestep, vocab), distributions skewed enough that
    /// top-K selection is meaningful.
    struct SyntheticTeacher;
    impl TeacherLogitsProvider for SyntheticTeacher {
        fn forward_logits(
            &mut self,
            _tokens: &[u32],
            batch_size: usize,
            seq_len: usize,
            vocab: usize,
        ) -> Result<Vec<f32>> {
            let mut out = Vec::with_capacity(batch_size * seq_len * vocab);
            for r in 0..batch_size {
                for t in 0..seq_len {
                    for v in 0..vocab {
                        let val = ((0.01 * v as f32) + 0.1 * r as f32).sin()
                            + 0.7 * ((0.05 * v as f32) + 0.3 * t as f32).cos();
                        out.push(val * 5.0);
                    }
                }
            }
            Ok(out)
        }
    }

    /// Naive top-K oracle for cross-validation: full sort, take last K.
    fn top_k_naive(values: &[f32], k: usize) -> (Vec<f32>, Vec<u32>) {
        let mut indexed: Vec<(f32, u32)> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect();
        indexed.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        indexed.truncate(k);
        let logits = indexed.iter().map(|(v, _)| *v).collect();
        let idx = indexed.iter().map(|(_, i)| *i).collect();
        (logits, idx)
    }

    #[test]
    fn top_k_per_row_matches_naive_oracle() {
        // Use an irregular, NOT-monotone fixture to catch off-by-one
        // and tie-handling bugs.
        let v: Vec<f32> = (0..256)
            .map(|i| ((i as f32) * 0.137 + 1.5).sin() * 7.0 - 0.3)
            .collect();
        for k in [1, 2, 5, 32, 128, 256].iter().copied() {
            let (a_l, a_i) = top_k_per_row(&v, k);
            let (b_l, b_i) = top_k_naive(&v, k);
            assert_eq!(a_l.len(), k);
            assert_eq!(a_i.len(), k);
            for j in 0..k {
                assert!(
                    (a_l[j] - b_l[j]).abs() < 1e-6,
                    "top_k mismatch at k={k} j={j}: heap={} naive={}",
                    a_l[j],
                    b_l[j]
                );
                assert_eq!(a_i[j], b_i[j], "top_k idx mismatch at k={k} j={j}");
            }
        }
    }

    #[test]
    fn top_k_per_row_handles_ties_deterministically() {
        // All-equal slice — every position is a "top".  Oracle ties
        // sorted by ascending index.
        let v = vec![3.14f32; 16];
        let (l, idx) = top_k_per_row(&v, 5);
        assert_eq!(l, vec![3.14; 5]);
        assert_eq!(idx, vec![0u32, 1, 2, 3, 4]);
    }

    #[test]
    fn compute_dwq_targets_round_trip_synthetic() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg = ComputeTargetsConfig {
            top_k: 8,
            save_dir: tmp.path().to_path_buf(),
            vocab: 64,
        };
        let batch_size = 2usize;
        let seq_len = 5usize;
        let make_batch = |seed: u32| -> Vec<u32> {
            (0..(batch_size * seq_len))
                .map(|i| ((i as u32 * 13 + seed) % cfg.vocab as u32))
                .collect()
        };
        let train_batches = vec![make_batch(1), make_batch(2), make_batch(3)];
        let valid_batches = vec![make_batch(101), make_batch(102)];
        let splits = vec![
            CalibrationSplit {
                name: "train",
                batches: &train_batches,
                batch_size,
                seq_len,
            },
            CalibrationSplit {
                name: "valid",
                batches: &valid_batches,
                batch_size,
                seq_len,
            },
        ];
        let mut teacher = SyntheticTeacher;
        let summary = compute_dwq_targets(&mut teacher, &splits, &cfg).expect("compute");
        assert_eq!(summary, vec![("train".into(), 3), ("valid".into(), 2)]);

        // Train split: i=1 — verify file exists at the right path.
        let path = tmp
            .path()
            .join("train")
            .join(format!("{:010}.safetensors", 1));
        assert!(path.exists(), "expected {} to exist", path.display());

        // Round-trip every batch and cross-validate top-K against
        // the synthetic teacher's known output.
        for (split_idx, split) in splits.iter().enumerate() {
            for (i, _) in split.batches.iter().enumerate() {
                let (logits, indices, b, s, k) =
                    load_dwq_target(tmp.path(), split.name, i).expect("load");
                assert_eq!(b, batch_size);
                assert_eq!(s, seq_len - 1);
                assert_eq!(k, cfg.top_k);

                // Re-compute teacher logits for THIS batch's trimmed
                // tokens and verify the saved top-K matches the naive
                // oracle position-by-position.
                let mut teacher_check = SyntheticTeacher;
                let trimmed = trim_last_seq(
                    &split.batches[i],
                    split.batch_size,
                    split.seq_len,
                );
                let full_logits = teacher_check
                    .forward_logits(
                        &trimmed,
                        split.batch_size,
                        split.seq_len - 1,
                        cfg.vocab,
                    )
                    .unwrap();
                let n_rows = split.batch_size * (split.seq_len - 1);
                for row in 0..n_rows {
                    let row_start = row * cfg.vocab;
                    let row_logits = &full_logits[row_start..row_start + cfg.vocab];
                    let (oracle_l, oracle_i) = top_k_naive(row_logits, cfg.top_k);
                    let off = row * cfg.top_k;
                    let saved_l = &logits[off..off + cfg.top_k];
                    let saved_i = &indices[off..off + cfg.top_k];
                    for j in 0..cfg.top_k {
                        assert!(
                            (saved_l[j] - oracle_l[j]).abs() < 1e-6,
                            "split {} batch[{}] row[{}] top[{}]: saved={} oracle={}",
                            split.name,
                            i,
                            row,
                            j,
                            saved_l[j],
                            oracle_l[j]
                        );
                        assert_eq!(
                            saved_i[j], oracle_i[j],
                            "split {} batch[{}] row[{}] top[{}] idx",
                            split.name, i, row, j
                        );
                    }
                }
                let _ = split_idx;
            }
        }
    }

    #[test]
    fn compute_dwq_targets_rejects_seq_len_lt_2() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = ComputeTargetsConfig {
            top_k: 2,
            save_dir: tmp.path().to_path_buf(),
            vocab: 8,
        };
        let batches = vec![vec![0u32; 4]];
        let splits = vec![CalibrationSplit {
            name: "train",
            batches: &batches,
            batch_size: 4,
            seq_len: 1,
        }];
        let mut teacher = SyntheticTeacher;
        let res = compute_dwq_targets(&mut teacher, &splits, &cfg);
        assert!(res.is_err());
        let msg = format!("{:?}", res.err().unwrap());
        assert!(msg.contains("seq_len"), "unexpected error: {msg}");
    }

    #[test]
    fn compute_dwq_targets_rejects_topk_gt_vocab() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = ComputeTargetsConfig {
            top_k: 100,
            save_dir: tmp.path().to_path_buf(),
            vocab: 8,
        };
        let batches: Vec<Vec<u32>> = vec![];
        let splits = vec![CalibrationSplit {
            name: "train",
            batches: &batches,
            batch_size: 1,
            seq_len: 4,
        }];
        let mut teacher = SyntheticTeacher;
        let res = compute_dwq_targets(&mut teacher, &splits, &cfg);
        assert!(res.is_err());
        let msg = format!("{:?}", res.err().unwrap());
        assert!(msg.contains("top_k"), "unexpected error: {msg}");
    }

    #[test]
    fn compute_dwq_targets_rejects_batch_shape_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = ComputeTargetsConfig {
            top_k: 2,
            save_dir: tmp.path().to_path_buf(),
            vocab: 8,
        };
        // Declares batch_size=2 seq_len=3 (=6 tokens) but provides 5.
        let batches = vec![vec![0u32; 5]];
        let splits = vec![CalibrationSplit {
            name: "train",
            batches: &batches,
            batch_size: 2,
            seq_len: 3,
        }];
        let mut teacher = SyntheticTeacher;
        let res = compute_dwq_targets(&mut teacher, &splits, &cfg);
        assert!(res.is_err());
    }

    #[test]
    fn safetensors_round_trip_preserves_byte_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("foo.safetensors");
        let logits: Vec<f32> = (0..(2 * 3 * 4))
            .map(|i| (i as f32) * 0.137 - 0.5)
            .collect();
        let indices: Vec<u32> = (0..(2 * 3 * 4)).map(|i| (i * 3 + 1) as u32).collect();
        write_target_safetensors(&path, &logits, &indices, 2, 3, 4).unwrap();

        std::fs::create_dir_all(tmp.path().join("split")).unwrap();
        std::fs::rename(&path, tmp.path().join("split").join("0000000000.safetensors"))
            .unwrap();

        let (l, i, b, s, k) = load_dwq_target(tmp.path(), "split", 0).unwrap();
        assert_eq!((b, s, k), (2, 3, 4));
        assert_eq!(l, logits);
        assert_eq!(i, indices);
    }
}
