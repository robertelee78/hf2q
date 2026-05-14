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

/// Owned variant of `PrefillCapture` for installation as an
/// `MlxModelWeights` field. Avoids the lifetime parameter (Vec-backed)
/// so the capture session can live across the forward call without
/// borrow-checker gymnastics.
///
/// Set `seq_len` and `hidden_size` at install time so the layer-loop
/// hook can compute offsets without re-deriving them. The
/// `target_layer_ids` are checked once at install via `validate()`.
#[derive(Debug, Default, Clone)]
pub struct DFlashCaptureSession {
    pub target_layer_ids: Vec<usize>,
    pub hidden_output: Vec<f32>,
    pub per_position_argmaxes: Option<Vec<u32>>,
    /// Cached so the hook doesn't need to re-derive.
    pub seq_len: usize,
    pub hidden_size: usize,
}

impl DFlashCaptureSession {
    /// Allocate a capture session with the given layout. Buffers are
    /// pre-sized to `target_layer_ids.len() * seq_len * hidden_size`
    /// (and `seq_len` for argmaxes if requested).
    pub fn new(
        target_layer_ids: Vec<usize>,
        seq_len: usize,
        hidden_size: usize,
        with_argmaxes: bool,
    ) -> Self {
        let hidden_output = vec![0.0f32; target_layer_ids.len() * seq_len * hidden_size];
        let per_position_argmaxes = if with_argmaxes {
            Some(vec![0u32; seq_len])
        } else {
            None
        };
        Self {
            target_layer_ids,
            hidden_output,
            per_position_argmaxes,
            seq_len,
            hidden_size,
        }
    }

    /// Find the capture index (0..num_capture_layers) for a given
    /// target layer index, or None if this layer is not captured.
    pub fn capture_index_for(&self, target_layer_idx: usize) -> Option<usize> {
        self.target_layer_ids
            .iter()
            .position(|&i| i == target_layer_idx)
    }

    /// Write one layer's full `[seq_len, hidden_size]` row-major slab
    /// from a target hidden buffer into the capture. Used by the
    /// layer-loop hook in `forward_prefill_batched`.
    pub fn write_layer_slab(
        &mut self,
        capture_layer_idx: usize,
        pf_hidden_data: &[f32],
    ) -> Result<()> {
        let slab_len = self.seq_len * self.hidden_size;
        if pf_hidden_data.len() != slab_len {
            return Err(anyhow!(
                "DFlashCaptureSession::write_layer_slab: pf_hidden_data len {} != seq_len({}) * hidden_size({}) = {}",
                pf_hidden_data.len(), self.seq_len, self.hidden_size, slab_len
            ));
        }
        let start = PrefillCapture::offset_for(
            capture_layer_idx,
            0,
            self.seq_len,
            self.hidden_size,
        );
        let end = start + slab_len;
        if end > self.hidden_output.len() {
            return Err(anyhow!(
                "DFlashCaptureSession::write_layer_slab: write out of bounds (start {}, end {}, buf {})",
                start, end, self.hidden_output.len()
            ));
        }
        self.hidden_output[start..end].copy_from_slice(pf_hidden_data);
        Ok(())
    }

    /// Convenience: borrow as a `PrefillCapture` view for consumers
    /// that need the borrowed-buffer API.
    pub fn as_view(&mut self) -> PrefillCapture<'_> {
        PrefillCapture {
            target_layer_ids: &self.target_layer_ids,
            hidden_output: &mut self.hidden_output,
            per_position_argmaxes: self.per_position_argmaxes.as_deref_mut(),
        }
    }
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

    /// DFlashCaptureSession: new() pre-allocates correct buffer sizes.
    #[test]
    fn session_new_allocates_correct_sizes() {
        let sess = DFlashCaptureSession::new(
            vec![1, 6, 11, 17, 22, 27],
            4,
            2816,
            true,
        );
        assert_eq!(sess.hidden_output.len(), 6 * 4 * 2816);
        assert_eq!(sess.per_position_argmaxes.as_ref().map(|v| v.len()), Some(4));
        assert_eq!(sess.seq_len, 4);
        assert_eq!(sess.hidden_size, 2816);

        // No-argmaxes path
        let sess2 = DFlashCaptureSession::new(vec![1, 6], 8, 128, false);
        assert_eq!(sess2.hidden_output.len(), 2 * 8 * 128);
        assert!(sess2.per_position_argmaxes.is_none());
    }

    #[test]
    fn session_capture_index_for_finds_target_layers() {
        let sess = DFlashCaptureSession::new(vec![1, 6, 11, 17, 22, 27], 4, 2816, false);
        assert_eq!(sess.capture_index_for(1), Some(0));
        assert_eq!(sess.capture_index_for(11), Some(2));
        assert_eq!(sess.capture_index_for(27), Some(5));
        assert_eq!(sess.capture_index_for(0), None);
        assert_eq!(sess.capture_index_for(5), None);
        assert_eq!(sess.capture_index_for(28), None);
    }

    #[test]
    fn session_write_layer_slab_places_data_at_correct_offset() {
        let mut sess = DFlashCaptureSession::new(vec![1, 6, 11, 17, 22, 27], 3, 4, false);
        // Layer-2 slab: capture_layer_idx=2; offset = 2 * 3 * 4 = 24
        let slab = vec![5.0f32; 3 * 4];
        sess.write_layer_slab(2, &slab).expect("write");
        for i in 0..sess.hidden_output.len() {
            let expected = if (24..36).contains(&i) { 5.0 } else { 0.0 };
            assert_eq!(sess.hidden_output[i], expected, "i={i}");
        }
    }

    #[test]
    fn session_as_view_borrows_buffers_consistently() {
        let mut sess = DFlashCaptureSession::new(vec![1, 6], 2, 4, true);
        // Write something via session
        let slab = vec![3.0f32; 2 * 4];
        sess.write_layer_slab(0, &slab).expect("write");
        // Borrow as view — should see the same data
        let view = sess.as_view();
        for i in 0..8 {
            assert_eq!(view.hidden_output[i], 3.0);
        }
        assert!(view.per_position_argmaxes.is_some());
    }

    /// End-to-end integration: simulates the post-target-verify state
    /// (capture buffer populated by hypothetical layer-loop hook) and
    /// drives the drafter forward on the permuted concat. Validates
    /// the Phase 4 data path from capture → permute → drafter forward
    /// → finite output.
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_capture_to_drafter_forward_pipeline() {
        use crate::inference::spec_decode::dflash::{
            forward::dispatch_dflash_model_forward,
            kv_cache::DFlashKvCache,
            tensors::DFlashModelTensors,
            weights::{DFlashWeights, DFlashWeightsFile},
        };
        use mlx_native::{DType, KernelRegistry, MlxDevice};

        let cfg = gemma4_cfg();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        // Load drafter
        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");
        let mut cache = DFlashKvCache::new(&device, &cfg, 128).expect("cache");

        // Allocate capture buffer + populate as if target forward had
        // run and hooks had been written.
        let ctx_chunk = 4usize;
        let hidden = cfg.hidden_size;
        let block_size = 8u32;

        let mut hidden_buf = vec![0.0f32; capture_buffer_len(&cfg, ctx_chunk)];
        let target_layer_ids: Vec<usize> = cfg.target_layer_ids.clone();

        // Populate with synthetic values per (capture_layer_idx, token,
        // dim) so we can verify permute placement
        for cli in 0..target_layer_ids.len() {
            for t in 0..ctx_chunk {
                for d in 0..hidden {
                    let off = (cli * ctx_chunk + t) * hidden + d;
                    hidden_buf[off] = 0.1 + ((off % 31) as f32) / 310.0;
                }
            }
        }

        // Build PrefillCapture view + validate.
        // per_position_argmaxes is for the VERIFY forward pass, not the
        // ctx capture; we leave it None here since this test only
        // exercises the ctx-capture → drafter-forward path.
        let cap = PrefillCapture {
            target_layer_ids: &target_layer_ids,
            hidden_output: &mut hidden_buf,
            per_position_argmaxes: None,
        };
        cap.validate(ctx_chunk, hidden).expect("validate");

        // Permute capture buffer to drafter's expected concat layout
        // [seq_len, num_capture_layers * hidden_size]
        let concat = cap.permute_to_concat(ctx_chunk, hidden);
        assert_eq!(concat.len(), ctx_chunk * target_layer_ids.len() * hidden);

        // Upload concat to GPU as target_hidden_concat for the drafter
        let mut target_hidden = device
            .alloc_buffer(
                concat.len() * 4,
                DType::F32,
                vec![ctx_chunk, target_layer_ids.len() * hidden],
            )
            .expect("alloc target_hidden");
        target_hidden
            .as_mut_slice::<f32>()
            .expect("target_hidden slice")
            .copy_from_slice(&concat);

        // Allocate input h (simulated embed_tokens output)
        let h_elem = (block_size as usize) * hidden;
        let mut h = device
            .alloc_buffer(h_elem * 4, DType::F32, vec![block_size as usize, hidden])
            .expect("alloc h");
        {
            let s = h.as_mut_slice::<f32>().expect("h slice");
            for v in s.iter_mut() { *v = 1.0; }
        }

        // Run drafter forward end-to-end on the permuted captured hidden
        let h_final = dispatch_dflash_model_forward(
            &mut registry, &device, &h, &target_hidden,
            &tensors, &mut cache, &cfg,
            block_size, ctx_chunk as u32,
        )
        .expect("drafter forward on permuted capture");

        // Validate output [L, hidden] all finite
        assert_eq!(h_final.element_count(), (block_size as usize) * hidden);
        let host: &[f32] = h_final.as_slice::<f32>().expect("h_final slice");
        let n_finite = host.iter().filter(|v| v.is_finite()).count();
        assert_eq!(
            n_finite, host.len(),
            "drafter output on permuted capture must be all finite"
        );
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
