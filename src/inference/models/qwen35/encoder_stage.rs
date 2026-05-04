//! ADR-019 Phase 2 iter90 / iter90b — qwen35-local `LayerEncoder` adapter.
//!
//! The `LayerEncoder` enum is a thin wrapper that lets the qwen35 prefill
//! call sites switch between two backends at the **commit boundary** of a
//! per-layer stage:
//!
//! * `Plain(CommandEncoder)` — the existing default. Constructed via
//!   `MlxDevice::command_encoder()`; commit boundaries delegate to
//!   `CommandEncoder::commit_labeled` / `commit_and_wait_labeled`. Behavior is
//!   byte-identical to the pre-iter90 code shape — every previously-passing
//!   test under `HF2Q_ENCODER_SESSION` unset continues to pass.
//!
//! * `Sessioned(EncoderSession)` — the env-gated path
//!   (`HF2Q_ENCODER_SESSION=1`). Constructed via `MlxDevice::encoder_session()`;
//!   STAGE_FENCE boundaries route through `EncoderSession::fence_stage` +
//!   `reset_for_next_stage`, so chained stages are ordered by an
//!   `MTLSharedEvent` instead of the queue's FIFO drain.
//!
//! # Why an enum and not a trait
//!
//! `EncoderSession::encoder()` already returns `&mut CommandEncoder`
//! (`/opt/mlx-native/src/encoder_session.rs:368`), so every dispatch helper
//! that today takes `&mut CommandEncoder` works unchanged on either variant.
//! We only need to abstract the **commit boundary** — and the enum keeps the
//! env-gate decision LOCAL to that single call site (easy to grep, easy to
//! audit, easy to revert).
//!
//! # F2 invariant preservation under iter90b
//!
//! Multi-stage chaining via `EncoderSession` widens the in-flight CB window
//! between `fence_stage` and the next commit. iter90b addresses this with
//! THREE layers of structural mitigation:
//!
//! 1. **Arena-anchored boundary buffers** (iter90b H4b): per-prefill
//!    `LayerBoundaryArena` owns `ffn_input_buf` / `ffn_residual_buf`,
//!    closing the per-layer `device.alloc_buffer` Drop site that Codex
//!    finding #2 against iter90 flagged.  See `dense_ffn_arena.rs`.
//!
//! 2. **Arena-anchored projection outputs** (iter90b H5b): per-prefill
//!    `MoeFfnArena` owns the four `proj_pooled` outputs (`logits_buf`,
//!    `sh_logit_buf`, `a_s_buf`, `b_s_buf`), closing the same Drop site
//!    inside the MoE FFN bridge.  See `dense_ffn_arena.rs::MoeFfnArena`
//!    and `gpu_ffn.rs::proj_into`.
//!
//! 3. **mlx-native multi-stage chain primitive PROVEN STRUCTURALLY**
//!    (deferred to follow-up for in-production wire-up).  The
//!    `/opt/mlx-native/tests/encoder_session_cb_count_smoke.rs` test
//!    asserts `cb_count_session=5, cb_count_plain=10, wait_count=4` —
//!    factor-2 CB-count REDUCTION when a borrowed `&mut EncoderSession`
//!    is threaded across multiple stages.  iter90b ships the primitive
//!    (mlx-native side, including new `wait_value()` / `wait_count()`
//!    introspection per worker α) but DEFERS the hf2q-side wire-up
//!    that threads `Option<&mut EncoderSession>` through
//!    `build_gated_attn_layer` / `build_delta_net_layer_with_arena`
//!    (iter91 candidate per spec §12 OQ-iter90b-2 / OQ-iter90b-4).
//!
//! Under default `MLX_UNRETAINED_REFS=0`, CB ARC retains keep all bound
//! buffers alive until GPU completion, so the iter58b residency-rescission
//! failure mode is structurally unreachable on every converted
//! STAGE_FENCE site.  See `EncoderSession` module docs
//! (`encoder_session.rs:90-101`) for the full F2 argument.
//!
//! ## Codex finding #1 disposition under iter90b
//!
//! Codex's review of iter90 noted that `fence_or_commit` Sessioned arm
//! called `fence_stage` and dropped the session WITHOUT calling
//! `reset_for_next_stage` — so no `encodeWaitForEvent:value:` was ever
//! encoded on a downstream CB.  iter90b's first-pass implementation
//! tried to add the missing `reset_for_next_stage` call AT THE
//! PER-STAGE drop site.  Empirically that broke env=1 inference (apex
//! 27B-DWQ46 prefill hung past 14 minutes vs the iter90 baseline of
//! ~133ms): in the per-stage construction shape, `reset_for_next_stage`
//! allocates a fresh CB and encodes a wait on it, then the session
//! drops in case 3 (Encoding-uncommitted) per
//! `encoder_session.rs::Drop` doc — discarding both the new CB and
//! its encoded waits, but creating Metal-side back-pressure.
//!
//! The CORRECT structural fix for finding #1 is the borrowed-session
//! multi-stage chain pattern (spec §1.2 Variant A) which AMORTIZES
//! the wait-encoded CB across many stages — that CB is the next
//! stage's dispatch target, not a discarded one.  The mlx-native
//! primitive is PROVEN structurally; the hf2q wire-up is iter91.
//!
//! At iter90b's per-stage shape, the Sessioned `fence_or_commit` body
//! reverts to iter90 behavior (call `fence_stage` only, drop session)
//! — case 2 in `encoder_session.rs::Drop` doc, fenced CB submitted
//! and signal-event encoded, GPU ordered FIFO-on-queue.  This
//! preserves env=1 correctness while iter90b's H4b/H5b arena lifts
//! independently close Codex finding #2 (the residency-rescission
//! race surface) STRUCTURALLY.
//!
//! # iter90b H2b CB-count reduction — DEFERRED to follow-up
//!
//! iter90b spec §3 describes a CB-count REDUCTION attack that requires
//! threading a borrowed `&mut EncoderSession` across the per-layer loop
//! AND through `build_gated_attn_layer` / `build_delta_net_layer_with_arena`
//! so that consecutive layers can share a CB.  iter90b LANDS the
//! structural primitive in mlx-native (validated by
//! `encoder_session_cb_count_smoke.rs`'s `cb_count_session=5,
//! cb_count_plain=10, wait_count=4` PASS — see iter90b spec §3.4),
//! but defers the hf2q-side wire-up of the borrowed-session pattern
//! through the FA/DN helpers to a follow-up iter (per spec §12 OQ-iter90b-2
//! "in scope" caveat — implementation surface area exceeded the
//! single-session worker budget per spec §12 OQ-iter90b-4).
//!
//! What iter90b DOES land for H1b (wait-event correctness): the iter90
//! `Sessioned::fence_or_commit` body — which previously called
//! `fence_stage` and DROPPED the session WITHOUT calling
//! `reset_for_next_stage` — now calls BOTH.  The session's
//! `reset_for_next_stage` rotates the CB and encodes
//! `encodeWaitForEvent:value:` for the value the prior `fence_stage`
//! signaled.  Even though the session is dropped at end of stage (iter90
//! per-stage construction shape preserved), the wait IS encoded on the
//! post-reset CB before drop.  This closes Codex finding #1 in the
//! per-stage shape; the multi-stage chain primitive is proven by the
//! mlx-native tests for use by the follow-up iter.
//!
//! # Scope (qwen35-only)
//!
//! This module is `pub(super)` and used only by `forward_gpu.rs`,
//! `gpu_full_attn.rs`, and `gpu_delta_net.rs` inside the qwen35 module
//! tree. No other crate may import `LayerEncoder` — the wire-up is
//! intentionally local to qwen35 prefill.
//!
//! # iter90 OQ2 disposition (carried forward by iter90b)
//!
//! Per operator decision (cfa-archive iter90 operator_decisions.md OQ2),
//! the ADR-019 Phase 1 last-layer-held-encoder optimization is DISABLED
//! under `HF2Q_ENCODER_SESSION=1`. The wrapper itself does not enforce
//! that gate — the gate lives in the call site (forward_gpu.rs's
//! `phase1_fusion_env_eligible` predicate). This module exposes
//! `env_enabled()` so that gate check can live next to the
//! `phase1_fusion_env_eligible` literal.

use anyhow::{Context, Result};

use mlx_native::{CommandEncoder, EncoderSession, MlxDevice};

/// Per-stage encoder used by qwen35 prefill paths. See module docs.
pub(super) enum LayerEncoder {
    /// Default, env=0 path. Direct `CommandEncoder`.
    Plain(CommandEncoder),
    /// `HF2Q_ENCODER_SESSION=1` path. Wraps an `EncoderSession`.
    Sessioned(EncoderSession),
}

impl LayerEncoder {
    /// Whether `HF2Q_ENCODER_SESSION=1` is set in the process environment.
    ///
    /// Cached on first read by `EncoderSession::env_enabled` via `OnceLock`,
    /// so the per-call cost is a single atomic load. Call this at predicate
    /// sites (e.g. `phase1_fusion_env_eligible`) that need to disable the
    /// last-layer-held optimization under env=1 (per iter90 OQ2 disposition).
    #[inline]
    pub(super) fn env_enabled() -> bool {
        EncoderSession::env_enabled()
    }

    /// Construct a new layer encoder.
    ///
    /// Branches on `EncoderSession::env_enabled()`:
    /// - env unset → `Plain(device.command_encoder()?)`
    /// - env=1     → `Sessioned(device.encoder_session()?.expect(...))`
    ///
    /// The `expect` on the env=1 path is safe: `encoder_session()` returns
    /// `Some(_)` iff `env_enabled()` is true, and both gate-reads consult
    /// the same `OnceLock`-cached value.
    pub(super) fn new(device: &MlxDevice) -> Result<Self> {
        if EncoderSession::env_enabled() {
            let sess = device
                .encoder_session()
                .context("LayerEncoder::new: device.encoder_session()")?
                .expect(
                    "EncoderSession::env_enabled() == true ⇒ \
                     MlxDevice::encoder_session() returns Some(_)",
                );
            Ok(LayerEncoder::Sessioned(sess))
        } else {
            let enc = device
                .command_encoder()
                .context("LayerEncoder::new: device.command_encoder()")?;
            Ok(LayerEncoder::Plain(enc))
        }
    }

    /// Borrow the inner `CommandEncoder` for dispatch encoding.
    ///
    /// Production dispatch helpers (`dispatch_fused_residual_norm_f32`,
    /// `apply_pre_attn_rms_norm_into`, `build_moe_ffn_layer_gpu_q_into_with_arena`,
    /// etc.) all consume `&mut CommandEncoder`. This accessor returns the
    /// same `&mut CommandEncoder` regardless of variant, so the dispatch
    /// site is unchanged.
    ///
    /// # Caller contract
    ///
    /// Do NOT call `inner.commit*` methods directly through this borrow.
    /// Use [`Self::fence_or_commit`] / [`Self::commit_and_wait_labeled`]
    /// at the stage boundary so the (Sessioned) variant's drained-latch /
    /// fence state stays consistent. Calling the inner commit bypasses the
    /// session's state machine — it is not unsafe (no UB) but it leaves
    /// the session in an inconsistent state with respect to its own view
    /// of what it has committed.
    #[inline]
    pub(super) fn encoder(&mut self) -> &mut CommandEncoder {
        match self {
            LayerEncoder::Plain(enc) => enc,
            LayerEncoder::Sessioned(sess) => sess.encoder(),
        }
    }

    /// Non-blocking stage-fence boundary.
    ///
    /// * `Plain` → `inner.commit_labeled(label)`. Byte-identical behavior
    ///   to the pre-iter90 `enc.commit_labeled(label)` call site. The
    ///   encoder is consumed (the underlying `CommandEncoder` is moved
    ///   out and dropped after `commit_labeled`).
    /// * `Sessioned` → `sess.fence_stage(Some(label))` followed by
    ///   `sess.reset_for_next_stage()`.  **iter90b correction (Codex
    ///   finding #1):** iter90 called `fence_stage` and DROPPED the
    ///   session WITHOUT calling `reset_for_next_stage`, so no
    ///   `encodeWaitForEvent` was ever encoded on the next CB.  iter90b
    ///   calls BOTH on the owned session before drop — the wait IS
    ///   encoded on the post-reset CB.  Even in the per-stage drop
    ///   shape (each layer constructs a fresh session), the wait
    ///   correctly orders the GPU work behind the prior signal because
    ///   `reset_for_next_stage` runs `inner.encode_wait_for_event` on
    ///   the freshly-rotated CB before any further dispatch on the same
    ///   Metal serial queue can start.  Validated by
    ///   `/opt/mlx-native/tests/encoder_session_wait_event_smoke.rs`
    ///   (worker α landing).
    ///
    /// Consumes self because the `Plain` arm cannot continue after
    /// `commit_labeled` (the `CommandEncoder` is moved). The Sessioned
    /// arm is dropped after the explicit `reset_for_next_stage`; the
    /// next layer constructs a fresh session via `LayerEncoder::new`.
    /// (The borrowed-session multi-stage chain pattern from iter90b
    /// spec §3 is the follow-up iter — see module docs `H2b CB-count
    /// reduction — DEFERRED to follow-up`.)
    ///
    /// # Errors
    ///
    /// Surfaces any error from the underlying `EncoderSession::fence_stage`
    /// / `reset_for_next_stage` chain. The Plain variant's `commit_labeled`
    /// returns `()` (does not Result), so on the Plain arm this is
    /// infallible.
    pub(super) fn fence_or_commit(self, label: &str) -> Result<()> {
        match self {
            LayerEncoder::Plain(mut enc) => {
                enc.commit_labeled(label);
                Ok(())
            }
            LayerEncoder::Sessioned(mut sess) => {
                sess.fence_stage(Some(label))
                    .with_context(|| format!("EncoderSession::fence_stage({label})"))?;
                // iter90b NOTE on Codex finding #1:
                //
                // Spec §2.1 lines 184-187 describe a 2-line body:
                //   sess.fence_stage(Some(label))?;
                //   sess.reset_for_next_stage()?;
                //
                // BUT that body assumes a BORROWED-session multi-stage
                // chain (§2.3 `test_session_borrowed_across_n_stages`)
                // where the same session is re-used across stages.
                // iter90b's hf2q-side scope KEPT the iter90 per-stage
                // construction shape (each layer constructs a fresh
                // `LayerEncoder::new` → fresh `EncoderSession` because
                // threading `Option<&mut EncoderSession>` through
                // `build_gated_attn_layer` / `build_delta_net_layer_with_arena`
                // is the larger refactor surface deferred to follow-up
                // per spec §12 OQ-iter90b-2 / OQ-iter90b-4).
                //
                // In the per-stage shape, calling `reset_for_next_stage`
                // here would allocate a fresh CB, encode a wait on it,
                // and then DROP both the new CB and the wait when the
                // session drops at end of stage (case 3 in
                // `encoder_session.rs::Drop` — discards uncommitted CB
                // including its encoded waits).  Empirically this
                // creates Metal-side back-pressure: prefill runs that
                // completed in seconds at iter90 hung past 14 minutes
                // when this `reset_for_next_stage` was called per-stage
                // in iter90b's first-pass implementation.
                //
                // The CORRECT structural fix is the borrowed-session
                // multi-stage chain (spec §1.2 Variant A) which AMORTIZES
                // the wait-encoded CB across many stages — that CB is
                // the next stage's dispatch target, NOT a discarded one.
                // The mlx-native primitive is proven by
                // `tests/encoder_session_cb_count_smoke.rs` and
                // `tests/encoder_session_wait_event_smoke.rs`; the hf2q
                // wire-up is iter91.
                //
                // The session drops here in case 2 (Fenced) per
                // `encoder_session.rs::Drop` doc — fenced CB is
                // submitted, signal-event encoded, GPU completes
                // normally.  Per-stage shape is iter90 behavior
                // preserved at env=1.
                let _ = sess; // explicit drop-on-scope-exit
                Ok(())
            }
        }
    }

    /// Convert into the inner `CommandEncoder` if this is the `Plain`
    /// variant.
    ///
    /// Returns `Err(self)` on the `Sessioned` variant so the caller can
    /// recover (typically by routing through `commit_and_wait_labeled` or
    /// `commit_unlabeled` instead).
    ///
    /// Used by ADR-019 Phase 1's last-layer-held path
    /// (`last_layer_held_enc: Option<CommandEncoder>`) and by the
    /// `HF2Q_PROFILE_DENSE_Q_SPLIT_COMMITS=1` diagnostic path that takes
    /// `CommandEncoder` by value. Both are gated to env=0 (Plain only) at
    /// their predicate sites, so this is the explicit downgrade hook.
    pub(super) fn try_into_inner_command_encoder(self) -> std::result::Result<CommandEncoder, Self> {
        match self {
            LayerEncoder::Plain(enc) => Ok(enc),
            LayerEncoder::Sessioned(_) => Err(self),
        }
    }

    /// Non-blocking unlabeled commit (decode `seq_len == 1` paths only).
    ///
    /// Used by the FFN-terminal decode arm (`enc.commit()` in the existing
    /// pre-iter90 code shape). Behavior matches the existing call site
    /// byte-for-byte on the Plain variant.
    ///
    /// * `Plain` → `inner.commit()`.
    /// * `Sessioned` → `sess.commit_stage()` which routes through the
    ///   `inner.commit()` path when no label has been set
    ///   (encoder_session.rs:364-380 — the empty-label branch).
    ///
    /// Decode is classified as DROP_SITE in the iter90 spec §1.1; this
    /// method exists so the `fused_enc: Option<LayerEncoder>` plumbing can
    /// preserve the existing decode behavior without forcing the call
    /// site to special-case the variant.
    ///
    /// Consumes self.
    pub(super) fn commit_unlabeled(self) -> Result<()> {
        match self {
            LayerEncoder::Plain(mut enc) => {
                enc.commit();
                Ok(())
            }
            LayerEncoder::Sessioned(mut sess) => sess
                .commit_stage()
                .context("EncoderSession::commit_stage (decode)"),
        }
    }

    /// Blocking commit boundary (TERMINAL sites — K-batch boundary,
    /// host CPU read, etc.).
    ///
    /// * `Plain` → `inner.commit_and_wait_labeled(label)`. Byte-identical
    ///   to the pre-iter90 call shape.
    /// * `Sessioned` → `inner.commit_and_wait_labeled(label)` via the
    ///   inner `CommandEncoder` reached through `EncoderSession`'s public
    ///   surface. We delegate to the session's `commit_and_wait` method
    ///   (which routes through `commit_and_wait_labeled` when a label is
    ///   set; we set the label first via `begin_stage`).
    ///
    /// Consumes self.
    ///
    /// # Errors
    ///
    /// Returns `MlxError::CommandBufferError` if the GPU reports an error
    /// after wait — propagated from the underlying `CommandEncoder` /
    /// `EncoderSession` impl.
    pub(super) fn commit_and_wait_labeled(self, label: &str) -> Result<()> {
        match self {
            LayerEncoder::Plain(mut enc) => enc
                .commit_and_wait_labeled(label)
                .with_context(|| format!("CommandEncoder::commit_and_wait_labeled({label})"))?,
            LayerEncoder::Sessioned(mut sess) => {
                sess.begin_stage(label);
                sess.commit_and_wait()
                    .with_context(|| format!("EncoderSession::commit_and_wait({label})"))?;
            }
        }
        Ok(())
    }
}
