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
    /// hidden states. Returns no value — failure to install is silently
    /// idempotent (re-install overwrites).
    fn install_capture(&mut self, session: DFlashCaptureSession);

    /// Take back the previously-installed capture session (consuming it).
    /// Returns `None` if no session is installed. After this call,
    /// subsequent forwards revert to legacy non-capturing behavior.
    fn take_capture(&mut self) -> Option<DFlashCaptureSession>;

    /// True if a capture session is currently installed.
    fn has_capture(&self) -> bool;

    /// Roll the KV cache back by `trim` positions. Used by the
    /// orchestrator after a partial-reject to discard the K/V written
    /// for the rejected suffix. Idempotent: `trim=0` is a no-op.
    fn rollback_kv(&mut self, trim: usize);

    /// Run a batched verify forward over `tokens` starting at
    /// `start_seq_pos`. Returns a `Vec<u32>` of length `tokens.len()`
    /// containing the per-position argmax (top-1 token) at each
    /// position's logits.
    ///
    /// This is the per-position equivalent of the orchestrator's
    /// `accept_prefix_argmax` reference: position `i`'s argmax is the
    /// token the target would emit if asked to decode after
    /// `tokens[0..=i]` autoregressively.
    fn forward_decode_verify_batched(
        &mut self,
        tokens: &[u32],
        start_seq_pos: usize,
        gpu: &mut GpuContext,
    ) -> Result<Vec<u32>>;
}

/// Blanket delegation impl for [`crate::serve::forward_mlx::MlxModelWeights`].
///
/// All methods delegate to the existing inherent methods on
/// `MlxModelWeights` — this commit is non-behavioral, just exposes the
/// existing methods through the trait so future iterations can refactor
/// the orchestrator to be generic over `T: DFlashTarget`.
impl DFlashTarget for crate::serve::forward_mlx::MlxModelWeights {
    fn install_capture(&mut self, session: DFlashCaptureSession) {
        self.install_dflash_capture(session)
    }

    fn take_capture(&mut self) -> Option<DFlashCaptureSession> {
        self.take_dflash_capture()
    }

    fn has_capture(&self) -> bool {
        self.has_dflash_capture()
    }

    fn rollback_kv(&mut self, trim: usize) {
        // Delegate to the inherent rollback_kv (NOT recursive — `self`
        // here is `MlxModelWeights`, the inherent method exists there).
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
