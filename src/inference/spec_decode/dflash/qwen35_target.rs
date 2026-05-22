//! ADR-034 task #78 Step 3b (2026-05-21) — Qwen35DFlashTarget wrapper.
//!
//! `MlxModelWeights` owns its kv_caches as a field, so its [`DFlashTarget`]
//! impl can mutate the cache directly via `&mut self`. `Qwen35Model` does
//! NOT own its `HybridKvCache` — the cache is caller-managed (passed into
//! every forward call). This wrapper bundles a mutable borrow of both so
//! the trait's `rollback_kv` and `forward_decode_verify_batched` can
//! access the cache without changing `Qwen35Model`'s ownership model.
//!
//! Use this wrapper at the orchestrator boundary:
//!
//! ```ignore
//! let mut target = Qwen35DFlashTarget { model: &mut qwen35_model, kv_cache: &mut hybrid_kv };
//! dispatch_dflash_spec_decode_round_target_side(&mut target, ...)?;
//! ```
//!
//! Math: `forward_decode_verify_batched` calls
//! [`super::super::super::models::qwen35::model::Qwen35Model::forward_gpu_with_hidden`]
//! and extracts per-position argmaxes from the returned logits. Identical
//! semantics to `MlxModelWeights::forward_decode_verify_batched`.

use anyhow::Result;

use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::serve::gpu::GpuContext;

use super::hidden_capture::DFlashCaptureSession;
use super::target::DFlashTarget;

/// Mutable wrapper bundling `Qwen35Model` + its `HybridKvCache` so the
/// [`DFlashTarget`] trait methods can mutate both via `&mut self`.
///
/// Lifetime `'a` is the caller's borrow of both inner references — the
/// wrapper is short-lived (typically constructed at the call site of
/// `dispatch_dflash_spec_decode_round_target_side` and dropped after the
/// round returns).
pub struct Qwen35DFlashTarget<'a> {
    pub model: &'a mut Qwen35Model,
    pub kv_cache: &'a mut HybridKvCache,
}

impl<'a> Qwen35DFlashTarget<'a> {
    /// Construct from mutable refs to model + cache.
    pub fn new(model: &'a mut Qwen35Model, kv_cache: &'a mut HybridKvCache) -> Self {
        Self { model, kv_cache }
    }
}

impl<'a> DFlashTarget for Qwen35DFlashTarget<'a> {
    fn install_dflash_capture(&mut self, session: DFlashCaptureSession) {
        self.model.dflash_capture = Some(session);
    }

    fn take_dflash_capture(&mut self) -> Option<DFlashCaptureSession> {
        self.model.dflash_capture.take()
    }

    fn has_dflash_capture(&self) -> bool {
        self.model.dflash_capture.is_some()
    }

    /// Roll back the cache by `trim` positions across full-attn + mtp slots.
    ///
    /// LA-state rollback (recurrent + conv) is NOT applied here. The
    /// HybridKvCache's `rollback_la_to(accepted_idx)` requires a prior
    /// `ensure_la_capture()` call to populate the per-position capture
    /// buffers (see task #90 Step 4c). The DFlash orchestrator's
    /// integration with Qwen35Model (Step 3c) will arrange that capture
    /// before the verify forward. Until then, callers should only invoke
    /// this rollback at points where LA state correctness is irrelevant
    /// (e.g., end-of-generation cleanup, prefix re-init).
    fn rollback_kv(&mut self, trim: usize) {
        if trim == 0 {
            return;
        }
        // current_len is a `[u32; n_seqs]` per slot. We use seq 0 as the
        // canonical cursor (DFlash spec-decode runs single-sequence).
        let trim_u32 = trim as u32;
        for slot in self.kv_cache.full_attn.iter_mut() {
            for c in slot.current_len.iter_mut() {
                *c = c.saturating_sub(trim_u32);
            }
        }
        if let Some(mtp) = self.kv_cache.mtp_slot.as_mut() {
            for c in mtp.current_len.iter_mut() {
                *c = c.saturating_sub(trim_u32);
            }
        }
        // LA recurrent + conv state rollback is a no-op here — covered in
        // Step 3c which will set up the capture buffer pre-verify.
    }

    /// Run a batched verify forward over `tokens` starting at
    /// `start_seq_pos`. Returns per-position argmax (Vec length =
    /// tokens.len()).
    ///
    /// Internals: calls `Qwen35Model::forward_gpu_with_hidden` with
    /// positions broadcast from `start_seq_pos`, then extracts argmax
    /// of each per-token logits row. Identical math to
    /// `MlxModelWeights::forward_decode_verify_batched`.
    fn forward_decode_verify_batched(
        &mut self,
        tokens: &[u32],
        start_seq_pos: usize,
        _gpu: &mut GpuContext,
    ) -> Result<Vec<u32>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        // Build positions: Qwen35 RoPE expects an AXIS-MAJOR layout
        // `[axis0_t0, axis0_t1, ..., axis0_tN-1, axis1_t0, ..., axis3_tN-1]`
        // — 4 contiguous per-axis spans of length `seq_len` each. Use the
        // canonical helper `positions_for_range` from
        // `crate::inference::models::qwen35::spec_decode` to guarantee
        // the right layout (cont. — codex /cfa caught a previous
        // token-major layout bug in this wrapper).
        let positions_flat =
            crate::inference::models::qwen35::spec_decode::positions_for_range(
                start_seq_pos as i32,
                tokens.len(),
            );
        let seq_len = tokens.len();

        // Forward call. `forward_gpu_with_hidden` returns
        // (logits: Vec<f32>[seq_len * vocab_size], hidden: MlxBuffer).
        let (logits, _hidden) = self
            .model
            .forward_gpu_with_hidden(tokens, &positions_flat, self.kv_cache)?;

        let vocab = self.model.cfg.vocab_size as usize;
        if logits.len() != seq_len * vocab {
            anyhow::bail!(
                "forward_decode_verify_batched: expected logits len {} (seq_len={} × vocab={}), got {}",
                seq_len * vocab,
                seq_len,
                vocab,
                logits.len()
            );
        }

        // Per-position argmax. Single-pass linear scan.
        let mut argmaxes = Vec::with_capacity(seq_len);
        for row in logits.chunks_exact(vocab) {
            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }
            argmaxes.push(best_idx);
        }
        Ok(argmaxes)
    }
}
