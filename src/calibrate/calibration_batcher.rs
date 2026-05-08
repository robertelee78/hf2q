//! Calibration data pipeline — pre-tokenized → padded batches.
//!
//! ADR-020 iter-11f.  The streaming sensitivity estimator (iter-10d)
//! consumes one `Vec<u32>` of length `seq_len` per batch.  This
//! module turns a heterogeneous-length `Vec<Vec<u32>>` of calibration
//! sequences (typically produced by tokenizing sample text) into
//! `n_batches` such fixed-length buffers, padding short sequences
//! with `pad_id` and truncating long ones at `seq_len`.
//!
//! Tokenization itself is the caller's responsibility — this module
//! is tokenizer-agnostic.  For iter-11g we'll wire it to a real
//! Qwen3 tokenizer; for iter-11f tests we use a deterministic
//! whitespace+hash stub (see [`whitespace_hash_tokenize`]).

use anyhow::{anyhow, Result};

/// Iterates pre-tokenized sequences as `[seq_len]` u32 buffers
/// (one per batch).  Each output buffer feeds directly into
/// [`crate::calibrate::qwen35_model::forward`] with `cfg.batch =
/// seq_len`.
///
/// In our model convention `batch` is the sequence dim (attention
/// spans `batch` positions, no causal mask), so one calibration
/// sequence = one batch.
pub struct CalibrationBatcher {
    sequences: Vec<Vec<u32>>,
    seq_len: usize,
    pad_id: u32,
}

impl CalibrationBatcher {
    /// Build a batcher from a list of variable-length token-ID sequences.
    ///
    /// Empty input is rejected.  `seq_len` and `pad_id` are validated
    /// (seq_len > 0).  Sequences are stored verbatim; padding +
    /// truncation happen at `batch()` time.
    pub fn new(sequences: Vec<Vec<u32>>, seq_len: usize, pad_id: u32) -> Result<Self> {
        if sequences.is_empty() {
            return Err(anyhow!("CalibrationBatcher: sequences must not be empty"));
        }
        if seq_len == 0 {
            return Err(anyhow!("CalibrationBatcher: seq_len must be > 0"));
        }
        Ok(Self {
            sequences,
            seq_len,
            pad_id,
        })
    }

    /// Number of batches available — exactly one per sequence
    /// (each sequence becomes one batch of length `seq_len`).
    pub fn n_batches(&self) -> usize {
        self.sequences.len()
    }

    /// Return the `idx`-th batch as a `Vec<u32>` of length `seq_len`.
    /// Sequences shorter than `seq_len` are right-padded with `pad_id`;
    /// sequences longer than `seq_len` are truncated to the first
    /// `seq_len` tokens.
    pub fn batch(&self, idx: usize) -> Result<Vec<u32>> {
        if idx >= self.sequences.len() {
            return Err(anyhow!(
                "CalibrationBatcher::batch({idx}) out of range (n_batches={})",
                self.sequences.len()
            ));
        }
        let seq = &self.sequences[idx];
        let mut out = vec![self.pad_id; self.seq_len];
        let take = std::cmp::min(seq.len(), self.seq_len);
        out[..take].copy_from_slice(&seq[..take]);
        Ok(out)
    }

    /// Iterate all batches in order.
    pub fn iter(&self) -> impl Iterator<Item = Vec<u32>> + '_ {
        (0..self.n_batches()).map(move |i| {
            // Safe by construction — i < n_batches.
            self.batch(i).expect("batch index in range")
        })
    }

    /// Constant-time access to seq_len.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Constant-time access to pad_id.
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }
}

/// Deterministic whitespace + hash tokenizer for tests + prototyping.
///
/// Splits `text` on whitespace, hashes each word with FNV-1a, mods by
/// `vocab` (then masks the BOS/pad slot at id=0 by adding 1 mod
/// (vocab-1) + 1 — keeps ID 0 reserved as a pad token).  Deterministic;
/// safe to use in unit tests where reproducibility matters.
///
/// **NOT** semantically equivalent to any real BPE/SentencePiece
/// tokenizer.  Replace with a real tokenizer in iter-11g when wiring
/// against an actual Qwen3 model.
pub fn whitespace_hash_tokenize(text: &str, vocab: u32) -> Vec<u32> {
    if vocab < 2 {
        return Vec::new();
    }
    text.split_whitespace()
        .map(|word| {
            // FNV-1a 32-bit.
            let mut h: u32 = 0x811c_9dc5;
            for b in word.bytes() {
                h ^= b as u32;
                h = h.wrapping_mul(0x0100_0193);
            }
            // Reserve id=0 for pad — map into [1, vocab).
            (h % (vocab - 1)) + 1
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batcher_pads_short_sequences() {
        let seqs = vec![vec![1, 2, 3], vec![4, 5]];
        let b = CalibrationBatcher::new(seqs, 5, 0).unwrap();
        assert_eq!(b.n_batches(), 2);
        assert_eq!(b.batch(0).unwrap(), vec![1, 2, 3, 0, 0]);
        assert_eq!(b.batch(1).unwrap(), vec![4, 5, 0, 0, 0]);
    }

    #[test]
    fn batcher_truncates_long_sequences() {
        let seqs = vec![vec![1, 2, 3, 4, 5, 6, 7]];
        let b = CalibrationBatcher::new(seqs, 4, 99).unwrap();
        assert_eq!(b.batch(0).unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn batcher_exact_length_passes_through() {
        let seqs = vec![vec![1, 2, 3, 4]];
        let b = CalibrationBatcher::new(seqs, 4, 99).unwrap();
        assert_eq!(b.batch(0).unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn batcher_iter_yields_all_in_order() {
        let seqs = vec![vec![10, 11], vec![20, 21], vec![30, 31]];
        let b = CalibrationBatcher::new(seqs, 3, 0).unwrap();
        let collected: Vec<Vec<u32>> = b.iter().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], vec![10, 11, 0]);
        assert_eq!(collected[1], vec![20, 21, 0]);
        assert_eq!(collected[2], vec![30, 31, 0]);
    }

    #[test]
    fn batcher_rejects_empty_sequences() {
        match CalibrationBatcher::new(Vec::new(), 4, 0) {
            Err(e) => assert!(format!("{e}").contains("must not be empty")),
            Ok(_) => panic!("expected empty error"),
        }
    }

    #[test]
    fn batcher_rejects_zero_seq_len() {
        match CalibrationBatcher::new(vec![vec![1, 2]], 0, 0) {
            Err(e) => assert!(format!("{e}").contains("seq_len must be > 0")),
            Ok(_) => panic!("expected seq_len error"),
        }
    }

    #[test]
    fn batcher_out_of_range_errors() {
        let b = CalibrationBatcher::new(vec![vec![1]], 4, 0).unwrap();
        match b.batch(5) {
            Err(e) => assert!(format!("{e}").contains("out of range")),
            Ok(_) => panic!("expected out-of-range error"),
        }
    }

    #[test]
    fn whitespace_hash_tokenize_deterministic() {
        let a = whitespace_hash_tokenize("hello world", 1024);
        let b = whitespace_hash_tokenize("hello world", 1024);
        assert_eq!(a, b);
    }

    #[test]
    fn whitespace_hash_tokenize_respects_vocab() {
        // All ids must be in [1, vocab).
        let ids = whitespace_hash_tokenize(
            "the quick brown fox jumps over the lazy dog",
            32,
        );
        assert!(!ids.is_empty());
        for id in ids {
            assert!(id >= 1, "id={id} should be ≥ 1 (pad reserved)");
            assert!(id < 32, "id={id} should be < vocab=32");
        }
    }

    #[test]
    fn whitespace_hash_tokenize_different_words_likely_different_ids() {
        // FNV-1a should produce distinct hashes for distinct short words
        // most of the time.  We just check non-trivial collision rate
        // on a sample.
        let words = ["one", "two", "three", "four", "five", "six", "seven"];
        let ids: Vec<u32> = words
            .iter()
            .map(|w| whitespace_hash_tokenize(w, 4096)[0])
            .collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert!(
            sorted.len() >= ids.len() - 1,
            "expected at most 1 collision in 7 words at vocab=4096; got {} unique",
            sorted.len()
        );
    }

    #[test]
    fn whitespace_hash_tokenize_handles_empty() {
        let ids = whitespace_hash_tokenize("", 1024);
        assert!(ids.is_empty());
    }

    #[test]
    fn whitespace_hash_tokenize_invalid_vocab_returns_empty() {
        // vocab < 2 means no room for non-pad ids.
        let ids = whitespace_hash_tokenize("hello world", 1);
        assert!(ids.is_empty());
    }

    /// End-to-end: tokenize sample text, build batcher, feed first
    /// batch through the model.  Output must be `[seq_len, vocab]`
    /// finite logits.
    #[test]
    fn calibration_pipeline_end_to_end_through_model() {
        use crate::calibrate::autograd_gpu_tape::GpuTape;
        use crate::calibrate::qwen35_gguf_adapter::weights_from_gguf_tensors;
        use crate::calibrate::qwen35_model::{forward, ModelConfig, ModelLeaves};
        use mlx_native::MlxDevice;
        use std::collections::BTreeMap;

        let cfg = ModelConfig {
            vocab: 64,
            hidden: 64,
            n_layers: 2,
            n_heads: 2,
            head_dim: 32,
            intermediate: 128,
            eps: 1e-6,
        };
        let seq_len = 32;

        // Tokenize 4 calibration prompts.
        let prompts = [
            "the quick brown fox jumps over the lazy dog",
            "to be or not to be that is the question",
            "all that glitters is not gold",
            "ask not what your country can do for you",
        ];
        let sequences: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| whitespace_hash_tokenize(p, cfg.vocab as u32))
            .collect();
        let batcher = CalibrationBatcher::new(sequences, seq_len, 0).unwrap();
        assert_eq!(batcher.n_batches(), 4);

        // Build a synthetic GGUF map + ModelWeights.
        let mut gguf_map: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let mut state = 12345_u64;
        let mut next = || {
            state ^= state >> 33;
            state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
            state ^= state >> 33;
            ((state as i64) as f32) / (i64::MAX as f32)
        };
        gguf_map.insert(
            "token_embd.weight".into(),
            (0..cfg.vocab * cfg.hidden).map(|_| next() * 0.2).collect(),
        );
        gguf_map.insert(
            "output.weight".into(),
            (0..cfg.vocab * cfg.hidden).map(|_| next() * 0.2).collect(),
        );
        gguf_map.insert(
            "output_norm.weight".into(),
            (0..cfg.hidden).map(|_| 1.0 + next() * 0.05).collect(),
        );
        for i in 0..cfg.n_layers {
            gguf_map.insert(
                format!("blk.{i}.attn_norm.weight"),
                (0..cfg.hidden).map(|_| 1.0 + next() * 0.05).collect(),
            );
            gguf_map.insert(
                format!("blk.{i}.post_attention_norm.weight"),
                (0..cfg.hidden).map(|_| 1.0 + next() * 0.05).collect(),
            );
            for name in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                gguf_map.insert(
                    format!("blk.{i}.{name}.weight"),
                    (0..cfg.hidden * cfg.hidden).map(|_| next() * 0.2).collect(),
                );
            }
            gguf_map.insert(
                format!("blk.{i}.ffn_gate.weight"),
                (0..cfg.intermediate * cfg.hidden).map(|_| next() * 0.2).collect(),
            );
            gguf_map.insert(
                format!("blk.{i}.ffn_up.weight"),
                (0..cfg.intermediate * cfg.hidden).map(|_| next() * 0.2).collect(),
            );
            gguf_map.insert(
                format!("blk.{i}.ffn_down.weight"),
                (0..cfg.hidden * cfg.intermediate).map(|_| next() * 0.2).collect(),
            );
        }
        let weights = weights_from_gguf_tensors(&cfg, &gguf_map).unwrap();

        let device = MlxDevice::new().expect("device");
        let tape = GpuTape::new(device);
        let leaves = ModelLeaves::from_weights(&tape, &cfg, &weights, seq_len).unwrap();

        // Run forward on each batch — verify shape + finiteness.
        for ids in batcher.iter() {
            assert_eq!(ids.len(), seq_len);
            let logits = forward(&cfg, &ids, &leaves).unwrap();
            assert_eq!(logits.shape(), [seq_len, cfg.vocab]);
            let l_vec: Vec<f32> = logits.to_vec().unwrap();
            for (i, v) in l_vec.iter().enumerate() {
                assert!(v.is_finite(), "logits[{i}] = {v} not finite");
            }
            tape.reset();
            // Re-create leaves on fresh tape for next iteration.
            // (In production the streaming pattern handles this — see
            // estimate_attention_block_sensitivities_streaming for the
            // shared-device pattern.)
            // For this smoke test we just confirm forward succeeds
            // once per batch; the streaming integration belongs to
            // iter-11g.
            break;
        }
    }
}
