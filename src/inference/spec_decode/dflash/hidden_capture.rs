//! DFlash target hidden-state capture buffer (ADR-030 Phase 4).
//!
//! Defines the buffer layout + helpers used by the spec-decode
//! orchestrator to consume per-layer hidden states from the TARGET
//! model's forward pass at `cfg.target_layer_ids` positions.
//!
//! ## Capture buffer layout
//!
//! The capture buffer is a single flat `[f32]` slice of length
//! `num_capture_layers * seq_len * hidden_size`. Layout is row-major
//! by (capture_layer_idx, token_pos, dim):
//!
//! ```text
//!   captured[(capture_layer_idx * seq_len + token_pos) * hidden_size + dim]
//! ```
//!
//! `capture_layer_idx` indexes into `target_layer_ids` (0..N), NOT
//! into the target's full layer count (0..30 for gemma-4).
//!
//! ## Concat semantics for drafter input
//!
//! `dispatch_dflash_fc` expects `target_hidden_concat` shape
//! `[ctx_seq_len, num_capture_layers * hidden_size]` — i.e., for each
//! token position, the N captured hidden vectors are concatenated
//! along the last axis. The capture buffer layout above gives the
//! TRANSPOSE of this. The orchestrator must permute
//! `[capture_layer, token, dim]` → `[token, capture_layer, dim]`
//! before feeding to `dispatch_dflash_fc`.
//!
//! The permute is a one-time per-step cost; for our shapes
//! (num_capture_layers=6, seq_len ≤ block_size + ctx_chunk ≈ 16,
//! hidden_size=2816) the buffer is small (~270KB).
//!
//! ## Why a flat slice (not a Vec<Vec<f32>>)
//!
//! - Single contiguous allocation = one upload to GPU
//! - Caller controls allocation; the capture can write into a
//!   pre-allocated pool buffer to avoid per-call alloc
//! - Easy to pass to forward_prefill_batched via &mut [f32]

use super::config::DFlashConfig;
use anyhow::{anyhow, Result};

/// Strict shape descriptor + flat buffer for target hidden capture.
///
/// Lifetime parameter: the inner buffers borrow from caller-supplied
/// storage; `PrefillCapture` itself is a typed view.
pub struct PrefillCapture<'a> {
    /// Layer indices (in target's 0..num_target_layers numbering) at
    /// which to capture pf_hidden.
    pub target_layer_ids: &'a [usize],
    /// Captured hidden states: flat F32, length
    /// `target_layer_ids.len() * seq_len * hidden_size`.
    /// Layout (row-major): `[(layer_idx_in_capture, token_pos, dim)]`.
    pub hidden_output: &'a mut [f32],
    /// Per-position argmaxes from the target's LM head. Length =
    /// seq_len. Pre-allocated by caller. When None, the legacy
    /// single-row argmax return is preserved.
    pub per_position_argmaxes: Option<&'a mut [u32]>,
}

impl<'a> PrefillCapture<'a> {
    /// Validate that the buffer sizes match the declared seq_len +
    /// hidden_size + num_capture_layers. Cheap check at call boundary
    /// to avoid silent buffer overruns deep in the layer loop.
    pub fn validate(&self, seq_len: usize, hidden_size: usize) -> Result<()> {
        let expected = self.target_layer_ids.len() * seq_len * hidden_size;
        if self.hidden_output.len() != expected {
            return Err(anyhow!(
                "PrefillCapture: hidden_output len {} != target_layer_ids({}) * seq_len({}) * hidden_size({}) = {}",
                self.hidden_output.len(),
                self.target_layer_ids.len(),
                seq_len,
                hidden_size,
                expected
            ));
        }
        if let Some(ref pa) = self.per_position_argmaxes {
            if pa.len() != seq_len {
                return Err(anyhow!(
                    "PrefillCapture: per_position_argmaxes len {} != seq_len {}",
                    pa.len(),
                    seq_len
                ));
            }
        }
        // Validate target_layer_ids: strictly increasing, all in bounds
        // (caller's responsibility — we can't know num_target_layers here).
        for w in self.target_layer_ids.windows(2) {
            if w[0] >= w[1] {
                return Err(anyhow!(
                    "PrefillCapture: target_layer_ids must be strictly increasing; got {:?}",
                    self.target_layer_ids
                ));
            }
        }
        Ok(())
    }

    /// Compute the byte offset (in `hidden_output`) where the captured
    /// row for `(capture_layer_idx, token_pos)` begins. `hidden_size`
    /// elements follow contiguously.
    pub fn offset_for(
        capture_layer_idx: usize,
        token_pos: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> usize {
        (capture_layer_idx * seq_len + token_pos) * hidden_size
    }

    /// Write one (capture_layer_idx, all token positions) slab into
    /// `hidden_output`. Used by the target's layer-loop end hook:
    /// pf_hidden contains the layer's `[seq_len, hidden_size]` output;
    /// this copies all rows in one memcpy if `pf_hidden_data` is
    /// row-major contiguous.
    pub fn write_layer_slab(
        &mut self,
        capture_layer_idx: usize,
        pf_hidden_data: &[f32],
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<()> {
        let slab_len = seq_len * hidden_size;
        if pf_hidden_data.len() != slab_len {
            return Err(anyhow!(
                "PrefillCapture::write_layer_slab: pf_hidden_data len {} != seq_len({}) * hidden_size({}) = {}",
                pf_hidden_data.len(), seq_len, hidden_size, slab_len
            ));
        }
        let start = Self::offset_for(capture_layer_idx, 0, seq_len, hidden_size);
        let end = start + slab_len;
        if end > self.hidden_output.len() {
            return Err(anyhow!(
                "PrefillCapture::write_layer_slab: write out of bounds (start {}, end {}, buf {})",
                start, end, self.hidden_output.len()
            ));
        }
        self.hidden_output[start..end].copy_from_slice(pf_hidden_data);
        Ok(())
    }

    /// Permute `[capture_layer, token, dim]` → `[token, capture_layer, dim]`
    /// into a freshly-allocated Vec<f32>. Used by the orchestrator to
    /// build the `target_hidden_concat` argument for the drafter's
    /// `dispatch_dflash_fc` (which expects `[seq_len, num_layers * hidden]`
    /// flat = `[seq_len, num_layers, hidden]` with last 2 dims merged).
    pub fn permute_to_concat(&self, seq_len: usize, hidden_size: usize) -> Vec<f32> {
        let n_layers = self.target_layer_ids.len();
        let mut out = vec![0.0f32; seq_len * n_layers * hidden_size];
        for layer_idx in 0..n_layers {
            for t in 0..seq_len {
                let src = (layer_idx * seq_len + t) * hidden_size;
                let dst = (t * n_layers + layer_idx) * hidden_size;
                out[dst..dst + hidden_size]
                    .copy_from_slice(&self.hidden_output[src..src + hidden_size]);
            }
        }
        out
    }
}

/// Compute total flat buffer length for the hidden capture.
pub fn capture_buffer_len(cfg: &DFlashConfig, seq_len: usize) -> usize {
    cfg.target_layer_ids.len() * seq_len * cfg.hidden_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::dflash::config::DFlashConfig;

    fn gemma4_cfg() -> DFlashConfig {
        DFlashConfig::from_json_str(
            super::super::config::tests::GEMMA4_26B_A4B_DFLASH_CONFIG,
        )
        .expect("config parse")
    }

    #[test]
    fn capture_buffer_len_matches_drafter_fc_input() {
        let cfg = gemma4_cfg();
        // Per drafter config: 6 target_layer_ids × hidden_size 2816.
        // For seq_len=4: 6 × 4 × 2816 = 67584 floats.
        assert_eq!(capture_buffer_len(&cfg, 4), 6 * 4 * 2816);
    }

    #[test]
    fn validate_catches_wrong_buffer_size() {
        let target_layer_ids = vec![1, 6, 11, 17, 22, 27];
        let mut hidden = vec![0.0f32; 100]; // way too small
        let cap = PrefillCapture {
            target_layer_ids: &target_layer_ids,
            hidden_output: &mut hidden,
            per_position_argmaxes: None,
        };
        let err = cap.validate(4, 2816).unwrap_err();
        assert!(format!("{err}").contains("hidden_output len"));
    }

    #[test]
    fn validate_catches_non_monotonic_layer_ids() {
        let target_layer_ids = vec![1, 11, 6, 17, 22, 27]; // 11 > 6
        let mut hidden = vec![0.0f32; 6 * 4 * 2816];
        let cap = PrefillCapture {
            target_layer_ids: &target_layer_ids,
            hidden_output: &mut hidden,
            per_position_argmaxes: None,
        };
        let err = cap.validate(4, 2816).unwrap_err();
        assert!(format!("{err}").contains("strictly increasing"));
    }

    #[test]
    fn validate_catches_argmax_size_mismatch() {
        let target_layer_ids = vec![1, 6, 11, 17, 22, 27];
        let mut hidden = vec![0.0f32; 6 * 4 * 2816];
        let mut argmaxes = vec![0u32; 99]; // wrong: should be seq_len=4
        let cap = PrefillCapture {
            target_layer_ids: &target_layer_ids,
            hidden_output: &mut hidden,
            per_position_argmaxes: Some(&mut argmaxes),
        };
        let err = cap.validate(4, 2816).unwrap_err();
        assert!(format!("{err}").contains("per_position_argmaxes"));
    }

    #[test]
    fn offset_for_matches_row_major_layout() {
        // (layer=0, token=0): offset 0
        assert_eq!(PrefillCapture::offset_for(0, 0, 4, 2816), 0);
        // (layer=0, token=1): one row in
        assert_eq!(PrefillCapture::offset_for(0, 1, 4, 2816), 2816);
        // (layer=1, token=0): one (seq_len * hidden) slab in
        assert_eq!(PrefillCapture::offset_for(1, 0, 4, 2816), 4 * 2816);
        // (layer=5, token=3): last position in 6×4×2816 buffer
        assert_eq!(
            PrefillCapture::offset_for(5, 3, 4, 2816),
            (5 * 4 + 3) * 2816
        );
    }

    #[test]
    fn write_layer_slab_places_data_correctly() {
        let target_layer_ids = vec![1, 6, 11, 17, 22, 27];
        let mut hidden = vec![0.0f32; 6 * 4 * 2816];
        let mut cap = PrefillCapture {
            target_layer_ids: &target_layer_ids,
            hidden_output: &mut hidden,
            per_position_argmaxes: None,
        };
        // Write layer 2's slab as ones
        let slab = vec![1.0f32; 4 * 2816];
        cap.write_layer_slab(2, &slab, 4, 2816).expect("write_layer_slab");
        // Verify: layer 2's slab is ones; layer 0/1/3/4/5 still zero
        let layer2_start = 2 * 4 * 2816;
        let layer2_end = 3 * 4 * 2816;
        for i in 0..hidden.len() {
            let expected = if i >= layer2_start && i < layer2_end { 1.0 } else { 0.0 };
            assert_eq!(hidden[i], expected, "at index {i}");
        }
    }

    #[test]
    fn permute_to_concat_round_trips_layout() {
        let target_layer_ids = vec![1, 6];
        let n_layers = target_layer_ids.len();
        let seq_len = 3;
        let hidden_size = 4;
        // Build a recognizable input: value = layer_idx * 10 + token * 1 + dim * 0.1
        let mut hidden = vec![0.0f32; n_layers * seq_len * hidden_size];
        for layer_idx in 0..n_layers {
            for t in 0..seq_len {
                for d in 0..hidden_size {
                    let src = (layer_idx * seq_len + t) * hidden_size + d;
                    hidden[src] = (layer_idx * 10) as f32 + t as f32 + (d as f32) * 0.1;
                }
            }
        }
        let cap = PrefillCapture {
            target_layer_ids: &target_layer_ids,
            hidden_output: &mut hidden,
            per_position_argmaxes: None,
        };
        let concat = cap.permute_to_concat(seq_len, hidden_size);
        assert_eq!(concat.len(), seq_len * n_layers * hidden_size);
        // Verify: concat[(t * n_layers + layer_idx) * hidden_size + d]
        //       == hidden[(layer_idx * seq_len + t) * hidden_size + d]
        for layer_idx in 0..n_layers {
            for t in 0..seq_len {
                for d in 0..hidden_size {
                    let dst = (t * n_layers + layer_idx) * hidden_size + d;
                    let expected = (layer_idx * 10) as f32 + t as f32 + (d as f32) * 0.1;
                    assert_eq!(concat[dst], expected,
                        "concat mismatch t={t} layer={layer_idx} d={d}");
                }
            }
        }
    }
}
