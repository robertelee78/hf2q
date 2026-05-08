//! ADR-020 iter-14b — GGUF-backed `TeacherLogitsProvider`.
//!
//! [`GgufTeacherProvider`] wraps hf2q's existing GGUF loader
//! (`Qwen35Model::load_from_gguf`) and GPU forward path
//! (`Qwen35Model::forward_gpu` returning `[seq_len, vocab]` logits)
//! into the [`TeacherLogitsProvider`] trait that
//! [`super::dwq_targets::compute_dwq_targets`] consumes.
//!
//! # Memory profile
//!
//! Holds one [`Qwen35Model`] in RAM (~14 GB Q4_0 27B-class, ~70 GB
//! BF16 35B-class) plus one [`MlxDevice`].  Per mlx-lm
//! `dwq.py:386-387` the canonical pattern is to drop the teacher
//! before the student starts training so the student's GPU memory
//! pressure can absorb the released weights — callers should
//! `drop(provider)` (or let it go out of scope) between
//! `compute_dwq_targets` and `dwq_quantize`.
//!
//! # Batching
//!
//! `forward_logits` runs one row at a time through `forward_gpu`
//! with a fresh [`HybridKvCache`].  Multi-row batching via a single
//! prefill is a future optimisation — single-row keeps memory bounded
//! at one `[seq_len, vocab]` slab in flight.
//!
//! # Why text-only positions
//!
//! Qwen 3.5 / 3.6 use 4-axis MROPE.  For purely-textual calibration
//! corpora (the only kind `compute_dwq_targets` handles today) the
//! canonical layout is `flat[axis * seq + t] = t` for all four axes
//! — see `activation_capture_real.rs:122-128` for the production
//! parallel.  Vision-aware calibration is out of scope for ADR-020;
//! the structured 4-int positions slot is preserved for that path
//! but not exposed in this iter.

use std::path::Path;

use anyhow::{anyhow, Context, Result};

use mlx_native::{GgufFile, MlxDevice};

use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::serve::header::LoadProgress;

use super::dwq_targets::TeacherLogitsProvider;

/// `TeacherLogitsProvider` backed by hf2q's existing GGUF model
/// loader + GPU forward path.
pub struct GgufTeacherProvider {
    model: Qwen35Model,
    device: MlxDevice,
    vocab: usize,
    /// Pre-allocated `[4 * seq_len]` axis-major positions buffer,
    /// resized on demand in `forward_logits`.  Reused across rows of
    /// the same batch so we only pay the allocation once.
    positions_buf: Vec<i32>,
}

impl GgufTeacherProvider {
    /// Load a GGUF teacher from `path`.  All progress reporting is
    /// suppressed (silent load) — call sites that want progress
    /// chrome should use `Qwen35Model::load_from_gguf` directly.
    pub fn from_gguf_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let gguf = GgufFile::open(path)
            .map_err(|e| anyhow!("GgufFile::open {}: {e}", path.display()))?;
        let cfg = Qwen35Model::load_config_only(&gguf)
            .with_context(|| format!("load_config_only {}", path.display()))?;
        let n_layers = cfg.num_hidden_layers as usize;
        let mut progress = LoadProgress::new(false, 1, n_layers);
        let model = Qwen35Model::load_from_gguf(&gguf, &mut progress)
            .with_context(|| format!("Qwen35Model::load_from_gguf {}", path.display()))?;
        let device = MlxDevice::new()
            .map_err(|e| anyhow!("GgufTeacherProvider: MlxDevice::new: {e}"))?;
        let vocab = model.cfg.vocab_size as usize;
        Ok(Self {
            model,
            device,
            vocab,
            positions_buf: Vec::new(),
        })
    }

    /// Vocab size as observed in the loaded model (may differ from
    /// the GGUF metadata `tokenizer.ggml.tokens` length when the
    /// embedding table is pad-aligned — see `model.rs:193-214`).
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Number of transformer layers in the loaded model.
    pub fn num_layers(&self) -> usize {
        self.model.cfg.num_hidden_layers as usize
    }
}

/// Pure validation helper, factored out so it's unit-testable
/// without loading a model.  Returns `Ok(())` if every invariant
/// expected by [`GgufTeacherProvider::forward_logits`] holds.
pub(crate) fn validate_forward_logits_args(
    caller_vocab: usize,
    model_vocab: usize,
    tokens_len: usize,
    batch_size: usize,
    seq_len: usize,
) -> Result<()> {
    if caller_vocab != model_vocab {
        return Err(anyhow!(
            "GgufTeacherProvider::forward_logits: vocab mismatch: caller={} model={}",
            caller_vocab,
            model_vocab
        ));
    }
    if batch_size == 0 || seq_len == 0 {
        return Err(anyhow!(
            "GgufTeacherProvider::forward_logits: batch_size and seq_len must be > 0 (got {}, {})",
            batch_size,
            seq_len
        ));
    }
    let expected = batch_size
        .checked_mul(seq_len)
        .ok_or_else(|| anyhow!("batch_size * seq_len overflows usize"))?;
    if tokens_len != expected {
        return Err(anyhow!(
            "GgufTeacherProvider::forward_logits: token-shape mismatch: expected batch_size*seq_len={} got {}",
            expected,
            tokens_len
        ));
    }
    Ok(())
}

/// Build the canonical text-only 4-axis MROPE positions buffer of
/// length `4 * seq_len` into `dst` (resized as needed).
pub(crate) fn fill_text_positions(dst: &mut Vec<i32>, seq_len: usize) {
    dst.clear();
    dst.reserve(4 * seq_len);
    for _axis in 0..4 {
        for t in 0..seq_len {
            dst.push(t as i32);
        }
    }
    debug_assert_eq!(dst.len(), 4 * seq_len);
}

impl TeacherLogitsProvider for GgufTeacherProvider {
    fn forward_logits(
        &mut self,
        tokens: &[u32],
        batch_size: usize,
        seq_len: usize,
        vocab: usize,
    ) -> Result<Vec<f32>> {
        validate_forward_logits_args(vocab, self.vocab, tokens.len(), batch_size, seq_len)?;
        fill_text_positions(&mut self.positions_buf, seq_len);

        let row_logits_len = seq_len * vocab;
        let mut out = Vec::with_capacity(batch_size * row_logits_len);
        for row in 0..batch_size {
            let row_tokens = &tokens[row * seq_len..(row + 1) * seq_len];

            // Fresh KV cache per row so different rows don't
            // contaminate each other's attention.  `seq_len.max(1)`
            // matches the production-side `activation_capture_real.rs`
            // pattern.
            let mut kv = HybridKvCache::new(
                &self.model.cfg,
                &self.device,
                seq_len.max(1) as u32,
                1,
            )
            .map_err(|e| anyhow!("HybridKvCache::new (row {}): {e}", row))?;

            let row_logits = self
                .model
                .forward_gpu(row_tokens, &self.positions_buf, &mut kv)
                .with_context(|| format!("forward_gpu row {}", row))?;
            if row_logits.len() != row_logits_len {
                return Err(anyhow!(
                    "forward_gpu row {} returned {} logits, expected seq_len*vocab={}",
                    row,
                    row_logits.len(),
                    row_logits_len
                ));
            }
            out.extend_from_slice(&row_logits);
        }
        debug_assert_eq!(out.len(), batch_size * row_logits_len);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ────────────────────────────────────────────────────────────────
    //  Pure unit tests (no model load, no GPU) — exercise the
    //  trait-validation helper that gates the heavy forward_logits
    //  path.
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn validate_rejects_vocab_mismatch() {
        let err = validate_forward_logits_args(
            /* caller_vocab */ 1024,
            /* model_vocab  */ 2048,
            /* tokens_len   */ 16,
            /* batch_size   */ 2,
            /* seq_len      */ 8,
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("vocab mismatch"), "unexpected error: {msg}");
        assert!(msg.contains("1024"));
        assert!(msg.contains("2048"));
    }

    #[test]
    fn validate_rejects_zero_batch_or_seq() {
        for (b, s) in [(0usize, 8usize), (4, 0), (0, 0)] {
            let err =
                validate_forward_logits_args(1024, 1024, /* tokens_len */ 0, b, s).unwrap_err();
            assert!(format!("{err:#}").contains("must be > 0"));
        }
    }

    #[test]
    fn validate_rejects_token_shape_mismatch() {
        let err = validate_forward_logits_args(1024, 1024, 17, 2, 8).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("token-shape mismatch"), "unexpected: {msg}");
        assert!(msg.contains("16"));
        assert!(msg.contains("17"));
    }

    #[test]
    fn validate_accepts_well_formed_args() {
        validate_forward_logits_args(1024, 1024, 16, 2, 8).expect("valid args");
        validate_forward_logits_args(1, 1, 1, 1, 1).expect("min-shape");
    }

    #[test]
    fn fill_text_positions_layout_is_axis_major_replicated() {
        let mut buf = Vec::new();
        fill_text_positions(&mut buf, 5);
        assert_eq!(buf.len(), 20);
        // axis-major: [a0_t0..t4, a1_t0..t4, a2_t0..t4, a3_t0..t4]
        for axis in 0..4 {
            for t in 0..5 {
                assert_eq!(buf[axis * 5 + t], t as i32);
            }
        }
    }

    #[test]
    fn fill_text_positions_clears_prior_contents() {
        let mut buf = vec![999_i32; 1000];
        fill_text_positions(&mut buf, 3);
        assert_eq!(buf, vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
    }

    // ────────────────────────────────────────────────────────────────
    //  Real-model smoke tests — `#[ignore]`d because they load a
    //  multi-GB GGUF off disk + spin up the GPU.  Run manually:
    //    cargo test --release gguf_teacher -- --ignored --nocapture
    //  Override path via HF2Q_TEST_GGUF=… (matches the convention
    //  used by `dwq_loop.rs:700-710`).
    // ────────────────────────────────────────────────────────────────

    fn test_gguf_path() -> Option<std::path::PathBuf> {
        let p = std::env::var("HF2Q_TEST_GGUF").unwrap_or_else(|_| {
            "/opt/hf2q/models/qwen3.6-27b-mtp-q4_0/qwen3.6-27b-mtp-q4_0.gguf".to_string()
        });
        let pb = std::path::PathBuf::from(p);
        if pb.exists() {
            Some(pb)
        } else {
            None
        }
    }

    #[test]
    #[ignore = "loads multi-GB GGUF; run with --ignored when a Qwen35 GGUF is present"]
    fn gguf_teacher_smoke_returns_finite_logits_with_distinct_top1() {
        let path = match test_gguf_path() {
            Some(p) => p,
            None => {
                eprintln!("[gguf_teacher_smoke] SKIP: no GGUF at HF2Q_TEST_GGUF or default path");
                return;
            }
        };
        let mut teacher = GgufTeacherProvider::from_gguf_path(&path)
            .expect("GgufTeacherProvider::from_gguf_path");
        let vocab = teacher.vocab();
        assert!(vocab > 0, "vocab must be positive");

        let seq_len = 8usize;
        let batch_size = 1usize;
        // Deterministic well-distributed token ids — no special
        // tokens, no out-of-vocab.
        let tokens: Vec<u32> = (0..batch_size * seq_len)
            .map(|i| ((i as u64 * 7919 + 13) % vocab as u64) as u32)
            .collect();

        let logits = teacher
            .forward_logits(&tokens, batch_size, seq_len, vocab)
            .expect("forward_logits");

        assert_eq!(logits.len(), batch_size * seq_len * vocab);
        let n_finite = logits.iter().filter(|x| x.is_finite()).count();
        assert_eq!(
            n_finite,
            logits.len(),
            "expected all logits finite; got {} non-finite",
            logits.len() - n_finite
        );

        // Top-1 should differ across positions on a well-formed
        // model — a degenerate teacher (e.g. all-zero embeddings or
        // RMSNorm bug) collapses to the same argmax everywhere.
        let mut top1_per_pos: Vec<u32> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let row = &logits[t * vocab..(t + 1) * vocab];
            let (idx, _) = row
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                });
            top1_per_pos.push(idx as u32);
        }
        let n_distinct: std::collections::BTreeSet<u32> = top1_per_pos.iter().copied().collect();
        assert!(
            n_distinct.len() >= 2,
            "top-1 collapsed to {:?} across {} positions — likely degenerate forward",
            n_distinct,
            seq_len
        );
    }

    #[test]
    #[ignore = "loads multi-GB GGUF; integration with compute_dwq_targets"]
    fn gguf_teacher_drives_compute_dwq_targets_end_to_end() {
        use super::super::dwq_targets::{
            compute_dwq_targets, load_dwq_target, CalibrationSplit, ComputeTargetsConfig,
        };

        let path = match test_gguf_path() {
            Some(p) => p,
            None => {
                eprintln!("[gguf_teacher_e2e] SKIP: no GGUF");
                return;
            }
        };
        let mut teacher = GgufTeacherProvider::from_gguf_path(&path).expect("load");
        let vocab = teacher.vocab();

        let seq_len = 6usize;
        let batch_size = 1usize;
        let make_batch = |seed: u64| -> Vec<u32> {
            (0..batch_size * seq_len)
                .map(|i| ((i as u64 * 7919 + seed) % vocab as u64) as u32)
                .collect()
        };
        let train_batches = vec![make_batch(1), make_batch(2)];
        let splits = vec![CalibrationSplit {
            name: "train",
            batches: &train_batches,
            batch_size,
            seq_len,
        }];

        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg = ComputeTargetsConfig {
            top_k: 16,
            save_dir: tmp.path().to_path_buf(),
            vocab,
        };
        let summary =
            compute_dwq_targets(&mut teacher, &splits, &cfg).expect("compute_dwq_targets");
        assert_eq!(summary, vec![("train".into(), 2)]);

        // Round-trip a target file and assert structure.
        let (logits, indices, b, s, k) =
            load_dwq_target(tmp.path(), "train", 0).expect("load_dwq_target");
        assert_eq!(b, batch_size);
        assert_eq!(s, seq_len - 1);
        assert_eq!(k, 16);
        assert_eq!(logits.len(), b * s * k);
        assert_eq!(indices.len(), b * s * k);
        assert!(logits.iter().all(|x| x.is_finite()));
        assert!(indices.iter().all(|&i| (i as usize) < vocab));
    }
}
