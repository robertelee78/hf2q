//! ADR-034 task #78 (2026-05-21) — DFlash target abstraction.
//!
//! The DFlash orchestrator (`dispatch_dflash_spec_decode_round_target_side`
//! and friends) was originally written against `&mut MlxModelWeights`
//! directly. That works for the legacy non-Qwen35 archs (Llama 3,
//! Gemma 4, etc.) which all live under `MlxModelWeights`, but blocks
//! DFlash on Qwen 3.5/3.6 because `Qwen35Model` uses a separate forward
//! stack (`HybridKvCache` + per-layer hybrid attention) and CANNOT
//! be expressed as `MlxModelWeights`.
//!
//! This module introduces the [`DFlashTarget`] trait — the minimal
//! interface contract the orchestrator needs. Both `MlxModelWeights`
//! (current) and `Qwen35Model` (future) implement it.
//!
//! # Migration plan
//!
//! 1. Define [`DFlashTarget`] here + implement for `MlxModelWeights`
//!    via thin delegation to existing inherent methods. **THIS COMMIT.**
//! 2. Refactor `dispatch_dflash_spec_decode_round_target_side` and
//!    callers to take `&mut impl DFlashTarget`. Verify byte-identity vs
//!    current behavior on Llama 3 / Gemma 4 / Qwen 3.6 27B test models.
//! 3. Implement [`DFlashTarget`] for `Qwen35Model`:
//!    - `install_capture` / `take_capture` / `has_capture` — new fields
//!      on `Qwen35Model` (or its inner state holder).
//!    - `rollback_kv(trim)` — wire into `HybridKvCache.truncate_full_attn_to`
//!      / `truncate_mtp_to` / `rollback_la_to` (the LA slot machinery
//!      already exists per task #90 Step 4c).
//!    - `forward_decode_verify_batched(tokens, start_pos, gpu)` — call
//!      `forward_gpu_with_hidden` + per-position argmax extraction.
//! 4. Enable `HF2Q_SPEC_DFLASH=1` codepath in `serve/mod.rs` for the
//!    Qwen35 family.
//!
//! Per ADR-034 §1.2 Cell B: estimated 500-1500 LOC total across all
//! 4 steps. This commit is Step 1 (~80 LOC, foundational).

use anyhow::Result;

use crate::serve::gpu::GpuContext;

use super::hidden_capture::DFlashCaptureSession;

/// Minimal interface contract the DFlash orchestrator needs from a target
/// model. Implementing this trait makes a model eligible for DFlash spec
/// decoding via `HF2Q_SPEC_DFLASH=1`.
///
/// All methods are intentionally `&mut self` because the legacy
/// `MlxModelWeights` implementations mutate internal state
/// (KV cache pointers, capture session, etc.). Qwen35Model's eventual
/// impl will also mutate `HybridKvCache` internals.
///
/// # Byte-identity contract
///
/// `forward_decode_verify_batched` MUST be a pure dispatcher-equivalent
/// of K+1 sequential single-token decodes (no semantic change vs the
/// non-DFlash decode path). This is what enables the orchestrator's
/// greedy byte-identity invariant: at temperature=0, DFlash's committed
/// tokens are byte-identical to what single-token decode would emit.
pub trait DFlashTarget {
    /// Install a DFlash hidden-capture session so the next
    /// `forward_decode_verify_batched` populates it with per-position
    /// hidden states. Re-install overwrites.
    ///
    /// Method names match the existing inherent methods on
    /// `MlxModelWeights` so call sites don't need rewriting.
    fn install_dflash_capture(&mut self, session: DFlashCaptureSession);

    /// Take back the previously-installed capture session (consuming it).
    fn take_dflash_capture(&mut self) -> Option<DFlashCaptureSession>;

    /// True if a capture session is currently installed.
    fn has_dflash_capture(&self) -> bool;

    /// Roll the KV cache back by `trim` positions.
    fn rollback_kv(&mut self, trim: usize);

    /// Run a batched verify forward over `tokens` starting at
    /// `start_seq_pos`. Returns per-position argmax (Vec length == tokens.len()).
    fn forward_decode_verify_batched(
        &mut self,
        tokens: &[u32],
        start_seq_pos: usize,
        gpu: &mut GpuContext,
    ) -> Result<Vec<u32>>;
}

/// Blanket delegation impl for [`crate::serve::forward_mlx::MlxModelWeights`].
///
/// Trait methods share names with inherent methods. Within each `fn` body
/// we call the inherent method via `MlxModelWeights::method(self, ...)`
/// universal function call syntax to disambiguate from the trait
/// (otherwise `self.method(...)` would resolve to the trait itself and
/// recurse infinitely).
impl DFlashTarget for crate::serve::forward_mlx::MlxModelWeights {
    fn install_dflash_capture(&mut self, session: DFlashCaptureSession) {
        Self::install_dflash_capture(self, session)
    }

    fn take_dflash_capture(&mut self) -> Option<DFlashCaptureSession> {
        Self::take_dflash_capture(self)
    }

    fn has_dflash_capture(&self) -> bool {
        Self::has_dflash_capture(self)
    }

    fn rollback_kv(&mut self, trim: usize) {
        Self::rollback_kv(self, trim)
    }

    fn forward_decode_verify_batched(
        &mut self,
        tokens: &[u32],
        start_seq_pos: usize,
        gpu: &mut GpuContext,
    ) -> Result<Vec<u32>> {
        Self::forward_decode_verify_batched(self, tokens, start_seq_pos, gpu)
    }
}
