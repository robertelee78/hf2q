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
use crate::serve::multi_seq_kv::SlotId;

use super::hidden_capture::DFlashCaptureSession;
use super::target::DFlashTarget;

/// Mutable wrapper bundling `Qwen35Model` + its `HybridKvCache` so the
/// [`DFlashTarget`] trait methods can mutate both via `&mut self`.
///
/// Lifetime `'a` is the caller's borrow of both inner references — the
/// wrapper is short-lived (typically constructed at the call site of
/// `dispatch_dflash_spec_decode_round_target_side` and dropped after the
/// round returns).
///
/// ADR-040 Phase B4d (2026-05-30) — the wrapper now carries a
/// `slot_id: SlotId` field that names the active multi-seq slot for
/// the per-round verify + rollback.  The shared
/// [`super::target::DFlashTarget`] trait is NOT lifted (would touch
/// ~30 Gemma 4 callsites + would break sibling discipline per the
/// brief constraint "Gemma 4 + Qwen3VL UNCHANGED").  Instead the
/// `slot_id` rides as wrapper state — set via [`Self::new_with_slot`]
/// or [`Self::with_slot_id`] and consumed inside
/// [`Self::forward_decode_verify_batched`] +
/// [`Self::rollback_kv`].
pub struct Qwen35DFlashTarget<'a> {
    pub model: &'a mut Qwen35Model,
    pub kv_cache: &'a mut HybridKvCache,
    /// ADR-034 task #78 Step 3c.A (2026-05-21 cont. 37) — capture session
    /// lives on the wrapper, NOT on Qwen35Model.
    ///
    /// Rationale: Qwen35Model's forward_gpu_impl takes `&self` (immutable)
    /// — putting capture on the model and threading it through &self would
    /// require either interior mutability (RefCell, ugly) or refactoring
    /// 8+ &self callers in spec_decode.rs to &mut (high churn). Instead,
    /// the orchestrator stack-frame owns the capture and passes it as
    /// `Option<&mut DFlashCaptureSession>` into a new
    /// `Qwen35Model::forward_gpu_with_hidden_dflash` variant (Step 3c.A
    /// follow-up).
    pub dflash_capture: Option<DFlashCaptureSession>,
    /// ADR-040 Phase B4d (2026-05-30) — active multi-seq slot for this
    /// dflash spec-decode round.  `SlotId(0)` (default via
    /// [`Self::new`]) preserves pre-B4d single-seq byte-identical
    /// behaviour; `SlotId(N>0)` rebases per-layer K/V writes through
    /// the B4a-cont `slice_view` discipline at
    /// `gpu_full_attn.rs::slot_k_v_region_for_full_attn`.  Consumed
    /// inside [`Self::forward_decode_verify_batched`] (routes to
    /// `forward_gpu_with_hidden_dflash(.., slot_id)`) +
    /// [`Self::rollback_kv`] (per-slot truncate via the
    /// `HybridKvCache::truncate_*_for_slot` siblings).
    pub slot_id: SlotId,
}

impl<'a> Qwen35DFlashTarget<'a> {
    /// Construct from mutable refs to model + cache, defaulting to
    /// `SlotId(0)` (pre-B4d single-seq path; byte-identical to
    /// pre-Phase-A2c spec-decode).  Capture starts as `None`; install
    /// via the [`DFlashTarget::install_dflash_capture`] trait method.
    ///
    /// For multi-seq DFlash spec-decode use [`Self::new_with_slot`].
    pub fn new(model: &'a mut Qwen35Model, kv_cache: &'a mut HybridKvCache) -> Self {
        Self::new_with_slot(model, kv_cache, SlotId(0))
    }

    /// ADR-040 Phase B4d (2026-05-30) — construct with an explicit
    /// `slot_id` (the active multi-seq slot for this DFlash round).
    /// `slot_id.0 < kv_cache.n_seqs` is bounds-checked at the
    /// [`Self::forward_decode_verify_batched`] +
    /// [`Self::rollback_kv`] entry sites (forwarded into
    /// `forward_gpu_impl` + `truncate_*_for_slot`).
    pub fn new_with_slot(
        model: &'a mut Qwen35Model,
        kv_cache: &'a mut HybridKvCache,
        slot_id: SlotId,
    ) -> Self {
        Self {
            model,
            kv_cache,
            dflash_capture: None,
            slot_id,
        }
    }

    /// ADR-040 Phase B4d (2026-05-30) — builder-style override of
    /// `slot_id` after construction.  Useful when the orchestrator
    /// receives a pre-constructed target via the
    /// [`super::target::DFlashTarget`] generic surface and wants to
    /// promote it to a specific slot before the verify forward.
    pub fn with_slot_id(mut self, slot_id: SlotId) -> Self {
        self.slot_id = slot_id;
        self
    }
}

impl<'a> DFlashTarget for Qwen35DFlashTarget<'a> {
    fn install_dflash_capture(&mut self, session: DFlashCaptureSession) {
        self.dflash_capture = Some(session);
    }

    fn take_dflash_capture(&mut self) -> Option<DFlashCaptureSession> {
        self.dflash_capture.take()
    }

    fn has_dflash_capture(&self) -> bool {
        self.dflash_capture.is_some()
    }

    /// Roll back the cache by `trim` positions across full-attn + mtp +
    /// linear-attn slots.
    ///
    /// LA-state rollback uses the per-position capture buffer (task #90
    /// Step 4c machinery). If the capture buffer is set up (caller
    /// arranged `ensure_la_capture` before the verify forward), this
    /// method copies capture[accepted_idx] back into the active
    /// `recurrent` + `conv_state` buffers, restoring the LA state to
    /// what it was after the last accepted token.
    ///
    /// If the capture buffer is NOT set up (no prior `ensure_la_capture`),
    /// LA rollback is silently skipped. Full-attn + MTP cursors still
    /// get decremented (those have no capture-buffer dependency). This
    /// "best-effort" behavior matches what the Step 3c orchestrator
    /// will eventually arrange — pre-verify capture install.
    ///
    /// `accepted_idx` math: capture buffer covers `n_tokens_max = K+1`
    /// positions (the batched verify size). `trim` positions to discard
    /// means `accept_count = K + 1 - trim` tokens are kept, and
    /// `accepted_idx = accept_count - 1 = K - trim = n_tokens_max - 1 - trim`.
    fn rollback_kv(&mut self, trim: usize) {
        if trim == 0 {
            return;
        }
        let trim_u32 = trim as u32;
        let slot = self.slot_id;
        // ADR-040 Phase B4d (2026-05-30) — per-slot rollback.  The
        // previous body iterated every `current_len[]` entry across
        // every slot, which broke sibling-slot isolation under
        // SlotAware spec-decode at SlotId(N>0).  The new body
        // mutates ONLY `current_len[slot.0]` via the per-slot
        // `*_for_slot` helpers added to `HybridKvCache`.  `SlotId(0)`
        // at `n_seqs == 1` is byte-identical to the pre-B4d path
        // (the old `iter_mut()` over `current_len` had exactly one
        // element at n_seqs=1 — H168 pins this).
        //
        // Full-attn cursor rollback.  Saturating-sub semantics preserved.
        let cur_full = self
            .kv_cache
            .full_attn
            .first()
            .and_then(|s| s.current_len.get(slot.0 as usize).copied())
            .unwrap_or(0);
        let new_len_full = cur_full.saturating_sub(trim_u32);
        if let Err(e) = self
            .kv_cache
            .truncate_full_attn_to_for_slot(slot, new_len_full)
        {
            eprintln!(
                "[Qwen35DFlashTarget] truncate_full_attn_to_for_slot({:?}, {}) \
                 failed: {} — full-attn cursor may be stale by {} positions",
                slot, new_len_full, e, trim
            );
        }
        // MTP slot cursor rollback (if model has MTP).
        let cur_mtp = self
            .kv_cache
            .mtp_slot
            .as_ref()
            .and_then(|s| s.current_len.get(slot.0 as usize).copied())
            .unwrap_or(0);
        let new_len_mtp = cur_mtp.saturating_sub(trim_u32);
        if let Err(e) = self
            .kv_cache
            .truncate_mtp_to_for_slot(slot, new_len_mtp)
        {
            eprintln!(
                "[Qwen35DFlashTarget] truncate_mtp_to_for_slot({:?}, {}) \
                 failed: {} — MTP cursor may be stale by {} positions",
                slot, new_len_mtp, e, trim
            );
        }
        // LA-state rollback — only if capture buffer was set up pre-verify.
        if !self.kv_cache.linear_attn.is_empty() {
            let first = &self.kv_cache.linear_attn[0];
            if let Some(capture) = first.capture_states.as_ref() {
                let recurrent_elems = first.recurrent.element_count();
                if recurrent_elems > 0 {
                    let capture_elems = capture.element_count();
                    let n_tokens_max = capture_elems / recurrent_elems;
                    // Skip if trim exceeds the captured window — caller bug
                    // (would have to discard more than was just verified).
                    if (trim_u32 as usize) < n_tokens_max && n_tokens_max > 0 {
                        let accepted_idx =
                            (n_tokens_max as u32) - 1 - trim_u32;
                        // Best-effort: log + skip on error rather than
                        // propagating (the trait method is infallible by
                        // signature; orchestrator already preserved
                        // forward-pass correctness even without LA rollback).
                        // ADR-040 Phase B4d (2026-05-30) — route the
                        // LA rollback through the wrapper's `slot_id`
                        // (was hard-coded SlotId(0) per B4d deferral).
                        // SlotId(0) is byte-identical to pre-B4d via
                        // A2b's per-slot rollback path that already
                        // routes through slot 0 at n_seqs==1.
                        if let Err(e) = self.kv_cache.rollback_la_to(
                            slot,
                            accepted_idx,
                        ) {
                            eprintln!(
                                "[Qwen35DFlashTarget] rollback_la_to({:?}, {}) failed: {} \
                                 — LA state may be stale by {} positions",
                                slot, accepted_idx, e, trim
                            );
                        }
                    }
                }
            }
        }
    }

    /// Run a batched verify forward over `tokens` starting at
    /// `start_seq_pos`. Returns per-position argmax (Vec length =
    /// tokens.len()).
    ///
    /// Internals: calls `Qwen35Model::forward_gpu_with_hidden_dflash`
    /// (which delegates to `forward_gpu_with_hidden` when no capture is
    /// installed) with positions broadcast from `start_seq_pos`, then
    /// extracts argmax of each per-token logits row. Identical math to
    /// `MlxModelWeights::forward_decode_verify_batched`.
    ///
    /// **Step 3c.A wiring (cont. 39)**: when `self.dflash_capture` is
    /// `Some`, the call routes through `forward_gpu_with_hidden_dflash`
    /// which post-processes the verifier's per-layer hidden states into
    /// the session via the LayerActivations capture path. The session is
    /// then consumed by the orchestrator's drafter input
    /// (`extract_drafter_concat`) on the next round.
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

        // Forward call. `forward_gpu_with_hidden_dflash` returns
        // (logits: Vec<f32>[seq_len * vocab_size], hidden: MlxBuffer)
        // and additionally populates `self.dflash_capture` if set.
        // When no capture is installed it delegates to the plain
        // `forward_gpu_with_hidden` path with no extra overhead.
        //
        // ADR-040 Phase B4d (2026-05-30) — route the verify forward
        // through the wrapper's `slot_id` (was hard-coded SlotId(0)
        // per the B4a / B4d deferral on
        // `forward_gpu_with_hidden_dflash`'s signature).  SlotId(0)
        // at n_seqs==1 is byte-identical to pre-B4d.
        let (logits, _hidden) = self.model.forward_gpu_with_hidden_dflash(
            tokens,
            &positions_flat,
            self.kv_cache,
            self.dflash_capture.as_mut(),
            self.slot_id,
        )?;

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
        // ADR-040 §6.1.55 (iter-A4-cont-acceptance-telemetry, 2026-05-30) —
        // DFlash verify-step emission seam.  The verifier-side
        // dataflow here returns per-position argmaxes; the
        // orchestrator above (rejection-sampler etc.) decides which
        // are accepted.  We emit a structural shape record naming
        // `slot_id` + `seq_len` as the drafted-token budget.
        // `accepted_tokens` is reported as `seq_len` (the verify-side
        // upper bound) — the orchestrator-side emission seam at
        // EAGLE-3's `run_iteration` carries the precise
        // walk-tree-accept count.  Both seams are structurally
        // grep-able per H233d.  Production wiring lands at
        // iter-A4-cont-acceptance-telemetry-prod (gated on
        // `/metrics` schema extension per dossier §6 + §7).
        crate::inference::spec_decode::emit_acceptance_metric(
            crate::inference::spec_decode::SpecDecodeAcceptanceMetric::new(
                self.slot_id,
                seq_len as u32,
                seq_len as u32,
                0,
            ),
        );
        Ok(argmaxes)
    }
}
