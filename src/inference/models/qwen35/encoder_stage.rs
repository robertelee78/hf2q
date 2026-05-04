//! ADR-019 Phase 2 iter90 — qwen35-local `LayerEncoder` adapter.
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
//! (`/opt/mlx-native/src/encoder_session.rs:337`), so every dispatch helper
//! that today takes `&mut CommandEncoder` works unchanged on either variant.
//! We only need to abstract the **commit boundary** — and the enum keeps the
//! env-gate decision LOCAL to that single call site (easy to grep, easy to
//! audit, easy to revert).
//!
//! # F2 invariant preservation
//!
//! Multi-stage chaining via `EncoderSession` widens the in-flight CB window
//! between `fence_stage` and the next commit. Under default
//! `MLX_UNRETAINED_REFS=0`, CB ARC retains keep all bound buffers alive
//! until GPU completion, so the iter58b residency-rescission failure mode
//! is structurally unreachable on every converted STAGE_FENCE site
//! (DenseQ/MoeQ intra-K, FA `stage_a`, DN `stage_a` — all operating on
//! arena-anchored buffers that outlive the entire prefill). The
//! `tests/encoder_session_multistage.rs::test_session_arena_lifetime_under_fence_no_rescission`
//! adversarial test in mlx-native validates this contract.
//!
//! See `EncoderSession` module docs (encoder_session.rs:90-101) for the
//! full F2 argument.
//!
//! # Scope (qwen35-only)
//!
//! This module is `pub(super)` and used only by `forward_gpu.rs`,
//! `gpu_full_attn.rs`, and `gpu_delta_net.rs` inside the qwen35 module
//! tree. No other crate may import `LayerEncoder` — the wire-up is
//! intentionally local to qwen35 prefill.
//!
//! # iter90 OQ2 disposition
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
    ///   `sess.reset_for_next_stage()`. The session enters the
    ///   `Encoding` state on a fresh CB with a wait-event encoded; the
    ///   next `encoder()` borrow lazy-opens a new persistent compute
    ///   encoder on that fresh CB.
    ///
    /// Consumes self because the `Plain` arm cannot continue after
    /// `commit_labeled` (the `CommandEncoder` is moved). For per-layer
    /// chaining (intended use of the `Sessioned` variant), the caller
    /// constructs a fresh `LayerEncoder::new` per stage — same shape as
    /// the existing per-layer `device.command_encoder()` flow.
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
                // Per iter90 spec §2.3: post-condition is "ready for next
                // dispatch" on both variants. On Sessioned, that means
                // rotating to a fresh CB with the wait-event encoded.
                // The session is dropped at end of scope; reset would only
                // matter if the same session were reused. Since iter90
                // ships per-layer LayerEncoder::new (not multi-stage
                // chaining within one session — that's iter90b territory
                // per OQ1), we drop here and the next layer constructs a
                // fresh session. Drop is safe per encoder_session.rs:683-748
                // case 2 ("Fenced"): the signal-event was encoded onto the
                // prior CB and the CB has been submitted non-blocking.
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
