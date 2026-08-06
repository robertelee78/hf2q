//! ADR-037 Phase E3a — Multi-layer hidden state collection for EAGLE-3
//! drafter input.
//!
//! Per vLLM `model_executor/models/llama_eagle3.py:174-215`, the EAGLE-3
//! drafter consumes hidden states from N selected layers of the target
//! model. The N captured `[seq_len, hidden_size]` slabs are concatenated
//! along the last axis to form a `[seq_len, num_aux * hidden_size]`
//! tensor, which the drafter's `fc` layer projects back to
//! `[seq_len, drafter_hidden_size]` before feeding into the 1-layer
//! drafter transformer alongside the input embeddings.
//!
//! ## Layout contract
//!
//! The internal buffer is row-major `[seq_len, num_aux, hidden_size]`
//! flat, indexed as
//!
//! ```text
//!   hidden[(token_pos * num_aux + capture_idx) * hidden_size + dim]
//! ```
//!
//! This is the **transpose** of the DFlash capture layout
//! (`[capture_layer, token, dim]`) so that `concatenated_hidden()` can
//! return the buffer directly without a permute step — the buffer
//! *is* the EAGLE-3 `[seq_len, num_aux * hidden_size]` tensor in
//! C-order. Per-layer writes via `write_layer_slab` perform the
//! transpose at write time (cheap — once per capture, vs.
//! per-step at consume time).
//!
//! `capture_idx` is the position within `target_layer_ids`, NOT the
//! original target-layer index. Order of `target_layer_ids` is the
//! order of concatenation in the output tensor — EAGLE-3 paper +
//! published checkpoints fix this order at training time, so it must
//! be preserved at inference.
//!
//! ## Why a separate type from `DFlashCaptureSession`
//!
//! DFlash uses `[capture_layer, token, dim]` (layer-outer) because its
//! orchestrator permutes lazily. EAGLE-3 prefers `[token, capture, dim]`
//! (token-outer) because the drafter FC consumes per-token concatenated
//! rows. Putting these in the same struct would force one consumer to
//! permute on every forward, which dominates the small capture cost.
//!
//! ## Peer reference
//!
//! `/opt/vllm/vllm/model_executor/models/llama_eagle3.py:174-215` —
//! `num_aux_hidden_states`, `target_hidden_size`, `fc_input_size =
//! target_hidden_size * num_aux_hidden_states`, FC layer construction.

use anyhow::{anyhow, ensure, Result};

/// Multi-layer hidden state collector for EAGLE-3 drafter input.
///
/// Stores N captured `[seq_len, hidden_size]` layer outputs in the
/// EAGLE-3 concatenation layout directly, ready to feed into the
/// drafter FC layer.
#[derive(Debug, Clone)]
pub struct Eagle3HiddenCollector {
    /// Target-model layer indices to capture, in concatenation order.
    /// Order is significant (EAGLE-3 checkpoints fix it at training).
    target_layer_ids: Vec<usize>,
    /// Sequence length (number of token positions).
    seq_len: usize,
    /// Per-layer hidden_size (target-model hidden size — `fc` projects
    /// `num_aux * hidden_size` → drafter hidden_size).
    hidden_size: usize,
    /// Flat buffer `[seq_len, num_aux, hidden_size]` row-major.
    /// Length = seq_len * target_layer_ids.len() * hidden_size.
    buffer: Vec<f32>,
    /// Tracks which `capture_idx`es have been written (bit set).
    /// `write_layer_slab` sets the bit; `concatenated_hidden` requires
    /// all bits set. Prevents the drafter from consuming uninitialized
    /// zeros if a capture-loop hook skips a target layer.
    written_mask: u64,
}

impl Eagle3HiddenCollector {
    /// Allocate a collector for `seq_len` × `target_layer_ids.len()`
    /// hidden states. Validates the layer-id list (non-empty, no
    /// duplicates, ordered set of `usize`).
    ///
    /// Per vLLM EAGLE-3 default: 3 aux layers (`num_aux_hidden_states`
    /// = 3). Other counts are supported up to 64 (the `written_mask`
    /// is a `u64`).
    pub fn new(target_layer_ids: Vec<usize>, seq_len: usize, hidden_size: usize) -> Result<Self> {
        ensure!(
            !target_layer_ids.is_empty(),
            "Eagle3HiddenCollector: target_layer_ids must be non-empty"
        );
        ensure!(
            target_layer_ids.len() <= 64,
            "Eagle3HiddenCollector: at most 64 aux layers supported (written_mask is u64); got {}",
            target_layer_ids.len()
        );
        // Detect duplicate layer IDs — EAGLE-3 paper assumes a SET of
        // layer indices; duplicates would silently waste capture
        // bandwidth and shape the FC input in a way no published
        // checkpoint trains for.
        let mut sorted = target_layer_ids.clone();
        sorted.sort_unstable();
        for w in sorted.windows(2) {
            ensure!(
                w[0] != w[1],
                "Eagle3HiddenCollector: target_layer_ids has duplicate entry {}",
                w[0]
            );
        }
        ensure!(seq_len > 0, "Eagle3HiddenCollector: seq_len must be > 0");
        ensure!(
            hidden_size > 0,
            "Eagle3HiddenCollector: hidden_size must be > 0"
        );
        // Defensive overflow check: seq_len * num_aux * hidden_size
        // must fit in usize so the Vec allocation doesn't silently
        // wrap. At realistic shapes (seq=8K, num_aux=3, hidden=5120)
        // this is ~492 MB — comfortably within usize on 64-bit.
        let total = seq_len
            .checked_mul(target_layer_ids.len())
            .and_then(|v| v.checked_mul(hidden_size))
            .ok_or_else(|| {
                anyhow!(
                    "Eagle3HiddenCollector: seq_len({}) * num_aux({}) * hidden_size({}) overflows usize",
                    seq_len,
                    target_layer_ids.len(),
                    hidden_size
                )
            })?;
        Ok(Self {
            target_layer_ids,
            seq_len,
            hidden_size,
            buffer: vec![0.0f32; total],
            written_mask: 0,
        })
    }

    /// Number of captured layers (== `target_layer_ids.len()`).
    #[inline]
    pub fn num_aux(&self) -> usize {
        self.target_layer_ids.len()
    }

    /// Sequence length.
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Per-layer hidden size.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// `fc_input_size` per vLLM EAGLE-3 — the width the drafter FC
    /// layer expects on input.
    #[inline]
    pub fn fc_input_size(&self) -> usize {
        self.hidden_size * self.num_aux()
    }

    /// Borrow the target layer IDs in concatenation order.
    #[inline]
    pub fn target_layer_ids(&self) -> &[usize] {
        &self.target_layer_ids
    }

    /// Find the capture index for a given target-layer index, or
    /// `None` if this layer is not captured. Used by capture-loop
    /// hooks that walk all target layers and need to know whether to
    /// emit a slab.
    pub fn capture_index_for(&self, target_layer_idx: usize) -> Option<usize> {
        self.target_layer_ids
            .iter()
            .position(|&i| i == target_layer_idx)
    }

    /// Write one captured layer's `[seq_len, hidden_size]` row-major
    /// slab into the collector at the given `capture_idx`. Transposes
    /// the input from `[token, dim]` (layer-local) into the collector's
    /// `[token, capture_idx, dim]` layout.
    ///
    /// `capture_idx` must be in `[0, num_aux)` and must not have been
    /// written before in this collector's lifetime.
    pub fn write_layer_slab(&mut self, capture_idx: usize, slab: &[f32]) -> Result<()> {
        ensure!(
            capture_idx < self.num_aux(),
            "Eagle3HiddenCollector::write_layer_slab: capture_idx {} >= num_aux {}",
            capture_idx,
            self.num_aux()
        );
        let expected = self.seq_len * self.hidden_size;
        ensure!(
            slab.len() == expected,
            "Eagle3HiddenCollector::write_layer_slab: slab len {} != seq_len({}) * hidden_size({}) = {}",
            slab.len(),
            self.seq_len,
            self.hidden_size,
            expected
        );
        let bit = 1u64 << capture_idx;
        ensure!(
            (self.written_mask & bit) == 0,
            "Eagle3HiddenCollector::write_layer_slab: capture_idx {} already written",
            capture_idx
        );

        let num_aux = self.num_aux();
        let hs = self.hidden_size;
        // Transpose slab[token, dim] → buffer[token, capture_idx, dim].
        for token_pos in 0..self.seq_len {
            let src = token_pos * hs;
            let dst = (token_pos * num_aux + capture_idx) * hs;
            self.buffer[dst..dst + hs].copy_from_slice(&slab[src..src + hs]);
        }
        self.written_mask |= bit;
        Ok(())
    }

    /// Returns `true` iff every `capture_idx` has been written via
    /// `write_layer_slab`.
    pub fn is_complete(&self) -> bool {
        let full_mask = if self.num_aux() == 64 {
            !0u64
        } else {
            (1u64 << self.num_aux()) - 1
        };
        self.written_mask == full_mask
    }

    /// Borrow the flat `[seq_len, num_aux * hidden_size]` concatenated
    /// hidden buffer ready to feed into the EAGLE-3 drafter FC.
    /// Errors if any `capture_idx` has not yet been written
    /// (would otherwise return uninitialized zeros).
    pub fn concatenated_hidden(&self) -> Result<&[f32]> {
        ensure!(
            self.is_complete(),
            "Eagle3HiddenCollector::concatenated_hidden: incomplete capture — written_mask={:#066b}, expected {} bits set",
            self.written_mask,
            self.num_aux()
        );
        Ok(&self.buffer)
    }

    /// Reset the collector for reuse across spec-decode rounds. Keeps
    /// the same shape (saves the allocation) but clears the
    /// written-bit mask. The buffer contents are NOT zeroed — that
    /// would waste cycles. A new `write_layer_slab` for each
    /// `capture_idx` overwrites every byte; `concatenated_hidden`
    /// gates on `is_complete` to prevent reading stale data.
    pub fn reset(&mut self) {
        self.written_mask = 0;
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn make_slab(seed: u64, seq_len: usize, hidden_size: usize) -> Vec<f32> {
        (0..seq_len * hidden_size)
            .map(|i| ((seed.wrapping_add(i as u64)) % 1000) as f32 * 0.001)
            .collect()
    }

    #[test]
    fn adr_037_e3a_constructor_rejects_empty_layer_ids_2026_05_22() {
        let err = Eagle3HiddenCollector::new(vec![], 8, 128).unwrap_err();
        assert!(err.to_string().contains("non-empty"), "got: {err}");
    }

    #[test]
    fn adr_037_e3a_constructor_rejects_duplicate_layer_ids_2026_05_22() {
        let err = Eagle3HiddenCollector::new(vec![4, 16, 4], 8, 128).unwrap_err();
        assert!(err.to_string().contains("duplicate"), "got: {err}");
    }

    #[test]
    fn adr_037_e3a_constructor_rejects_excessive_layer_count_2026_05_22() {
        let layer_ids: Vec<usize> = (0..65).collect();
        let err = Eagle3HiddenCollector::new(layer_ids, 8, 128).unwrap_err();
        assert!(err.to_string().contains("at most 64"), "got: {err}");
    }

    #[test]
    fn adr_037_e3a_constructor_rejects_zero_seq_len_2026_05_22() {
        let err = Eagle3HiddenCollector::new(vec![4], 0, 128).unwrap_err();
        assert!(err.to_string().contains("seq_len"), "got: {err}");
    }

    #[test]
    fn adr_037_e3a_constructor_rejects_zero_hidden_size_2026_05_22() {
        let err = Eagle3HiddenCollector::new(vec![4], 8, 0).unwrap_err();
        assert!(err.to_string().contains("hidden_size"), "got: {err}");
    }

    #[test]
    fn adr_037_e3a_fc_input_size_matches_vllm_contract_2026_05_22() {
        // vLLM: fc_input_size = target_hidden_size * num_aux_hidden_states.
        // For default 3 aux × Qwen 3.6 27B hidden 5120 = 15360.
        let c = Eagle3HiddenCollector::new(vec![4, 16, 31], 1, 5120).unwrap();
        assert_eq!(c.num_aux(), 3);
        assert_eq!(c.hidden_size(), 5120);
        assert_eq!(c.fc_input_size(), 15360);
    }

    #[test]
    fn adr_037_e3a_capture_index_for_returns_position_or_none_2026_05_22() {
        let c = Eagle3HiddenCollector::new(vec![4, 16, 31], 1, 128).unwrap();
        assert_eq!(c.capture_index_for(4), Some(0));
        assert_eq!(c.capture_index_for(16), Some(1));
        assert_eq!(c.capture_index_for(31), Some(2));
        assert_eq!(c.capture_index_for(5), None);
        assert_eq!(c.capture_index_for(63), None);
    }

    #[test]
    fn adr_037_e3a_write_layer_slab_rejects_wrong_size_2026_05_22() {
        let mut c = Eagle3HiddenCollector::new(vec![4, 16, 31], 8, 128).unwrap();
        let bad = vec![0.0f32; 100]; // wrong length
        let err = c.write_layer_slab(0, &bad).unwrap_err();
        assert!(err.to_string().contains("slab len"), "got: {err}");
    }

    #[test]
    fn adr_037_e3a_write_layer_slab_rejects_out_of_range_2026_05_22() {
        let mut c = Eagle3HiddenCollector::new(vec![4, 16, 31], 8, 128).unwrap();
        let slab = make_slab(0, 8, 128);
        let err = c.write_layer_slab(3, &slab).unwrap_err();
        assert!(err.to_string().contains("capture_idx 3"), "got: {err}");
    }

    #[test]
    fn adr_037_e3a_write_layer_slab_rejects_double_write_2026_05_22() {
        let mut c = Eagle3HiddenCollector::new(vec![4, 16, 31], 4, 16).unwrap();
        let slab = make_slab(0, 4, 16);
        c.write_layer_slab(1, &slab).unwrap();
        let err = c.write_layer_slab(1, &slab).unwrap_err();
        assert!(err.to_string().contains("already written"), "got: {err}");
    }

    #[test]
    fn adr_037_e3a_concatenated_hidden_rejects_incomplete_capture_2026_05_22() {
        let mut c = Eagle3HiddenCollector::new(vec![4, 16, 31], 4, 16).unwrap();
        let slab = make_slab(0, 4, 16);
        c.write_layer_slab(0, &slab).unwrap();
        c.write_layer_slab(1, &slab).unwrap();
        // capture_idx 2 NOT written
        let err = c.concatenated_hidden().unwrap_err();
        assert!(err.to_string().contains("incomplete"), "got: {err}");
    }

    #[test]
    fn adr_037_e3a_concatenated_hidden_layout_matches_vllm_2026_05_22() {
        // Layout contract: hidden[(token_pos * num_aux + capture_idx) * hidden_size + dim]
        let seq_len = 3;
        let hidden_size = 4;
        let mut c = Eagle3HiddenCollector::new(vec![10, 20], seq_len, hidden_size).unwrap();

        // Layer 0 (capture_idx=0): all 1.0s at position 0, all 2.0s at pos 1, all 3.0s at pos 2.
        let mut layer0 = Vec::with_capacity(seq_len * hidden_size);
        for token in 0..seq_len {
            for _ in 0..hidden_size {
                layer0.push((token + 1) as f32);
            }
        }
        // Layer 1 (capture_idx=1): all 10.0s at position 0, 20.0s at pos 1, 30.0s at pos 2.
        let mut layer1 = Vec::with_capacity(seq_len * hidden_size);
        for token in 0..seq_len {
            for _ in 0..hidden_size {
                layer1.push(((token + 1) * 10) as f32);
            }
        }

        c.write_layer_slab(0, &layer0).unwrap();
        c.write_layer_slab(1, &layer1).unwrap();
        let cat = c.concatenated_hidden().unwrap();

        // For each token, the concatenated row is layer0 first then layer1.
        // token 0: [1,1,1,1, 10,10,10,10]
        // token 1: [2,2,2,2, 20,20,20,20]
        // token 2: [3,3,3,3, 30,30,30,30]
        assert_eq!(cat.len(), seq_len * 2 * hidden_size);
        for token in 0..seq_len {
            let base = token * 2 * hidden_size;
            for d in 0..hidden_size {
                assert_eq!(
                    cat[base + d],
                    (token + 1) as f32,
                    "token {token} layer0 dim {d}: got {}",
                    cat[base + d]
                );
            }
            for d in 0..hidden_size {
                assert_eq!(
                    cat[base + hidden_size + d],
                    ((token + 1) * 10) as f32,
                    "token {token} layer1 dim {d}: got {}",
                    cat[base + hidden_size + d]
                );
            }
        }
    }

    #[test]
    fn adr_037_e3a_layer_id_order_preserved_in_concat_2026_05_22() {
        // EAGLE-3 checkpoints fix layer-id order at training; assert
        // the capture_idx → layer mapping IS the user-supplied order.
        let layer_ids = vec![31, 4, 16]; // intentionally NOT sorted
        let c = Eagle3HiddenCollector::new(layer_ids, 4, 16).unwrap();
        assert_eq!(c.target_layer_ids(), &[31, 4, 16]);
        assert_eq!(c.capture_index_for(31), Some(0));
        assert_eq!(c.capture_index_for(4), Some(1));
        assert_eq!(c.capture_index_for(16), Some(2));
    }

    #[test]
    fn adr_037_e3a_reset_clears_written_mask_2026_05_22() {
        let mut c = Eagle3HiddenCollector::new(vec![4, 16], 4, 16).unwrap();
        let slab = make_slab(0, 4, 16);
        c.write_layer_slab(0, &slab).unwrap();
        c.write_layer_slab(1, &slab).unwrap();
        assert!(c.is_complete());
        c.reset();
        assert!(!c.is_complete());
        let err = c.concatenated_hidden().unwrap_err();
        assert!(err.to_string().contains("incomplete"), "got: {err}");
        // After reset, can write again (no "already written" error).
        c.write_layer_slab(0, &slab).unwrap();
        c.write_layer_slab(1, &slab).unwrap();
        assert!(c.is_complete());
    }

    #[test]
    fn adr_037_e3a_realistic_qwen35_shape_2026_05_22() {
        // Qwen 3.6 27B-like shape: 64-layer model, sample layers
        // [8, 16, 32, 48] (mid-late spread), seq_len 200 (typical
        // long-context prefill chunk), hidden_size 5120.
        let layer_ids = vec![8, 16, 32, 48];
        let seq_len = 200;
        let hidden_size = 5120;
        let mut c = Eagle3HiddenCollector::new(layer_ids.clone(), seq_len, hidden_size).unwrap();
        assert_eq!(c.num_aux(), 4);
        assert_eq!(c.fc_input_size(), 4 * 5120);
        // ~16 MB buffer per capture — well within prefill memory budget.
        assert_eq!(c.buffer.len(), seq_len * 4 * hidden_size);

        // Synthesize 4 distinct slabs.
        for (capture_idx, &layer_id) in layer_ids.iter().enumerate() {
            let slab = make_slab(layer_id as u64 * 1000, seq_len, hidden_size);
            c.write_layer_slab(capture_idx, &slab).unwrap();
        }
        let cat = c.concatenated_hidden().unwrap();
        assert_eq!(cat.len(), seq_len * 4 * hidden_size);
    }
}
