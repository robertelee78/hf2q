//! ADR-019 Phase 2 iter91 — qwen35-local borrowed-`&mut EncoderSession` adapter.
//!
//! `LayerEncoder<'sess>` is a thin wrapper that lets the qwen35 prefill call
//! sites switch between two backends at the **commit boundary** of a per-layer
//! stage:
//!
//! * `Plain(CommandEncoder)` — the env=0 default. Constructed via
//!   `MlxDevice::command_encoder()`; commit boundaries delegate to
//!   `CommandEncoder::commit_labeled` / `commit_and_wait_labeled`. Behavior is
//!   byte-identical to the pre-iter90 code shape — every previously-passing
//!   test under `HF2Q_ENCODER_SESSION` unset continues to pass.
//!
//! * `Sessioned(&'sess mut EncoderSession)` — the env-gated path
//!   (`HF2Q_ENCODER_SESSION=1`).  Borrows a session that lives for the entire
//!   `forward_gpu_impl` call (one allocation, one drop at function exit),
//!   so STAGE_FENCE boundaries route through `EncoderSession::fence_stage` +
//!   `reset_for_next_stage` AND consecutive layers can share the same CB
//!   (intra-CB carry via `memory_barrier`) — chained stages are ordered by
//!   an `MTLSharedEvent` instead of the queue's FIFO drain.
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
//! # F2 invariant preservation under iter91 borrowed-session
//!
//! Multi-stage chaining via `EncoderSession` widens the in-flight CB window
//! between `fence_stage` and the next commit. iter90b shipped TWO layers of
//! arena-anchored mitigation (`LayerBoundaryArena.ffn_input_buf` /
//! `ffn_residual_buf` + `MoeFfnArena.{logits,sh_logit,a_s,b_s}_buf` — see
//! `dense_ffn_arena.rs`). iter91 ADDS the borrowed-session retain mechanism
//! on top: as long as the session is alive, its persistent compute encoder
//! holds an implicit retain on every buffer bound into ANY active stage.
//! The `down_out` / `sum_buf` cross-layer ownership in
//! `build_dense_ffn_layer_gpu_q_into_with_arena` (gpu_ffn.rs:1254-1362) is
//! retained by the session's CB chain even under `MLX_UNRETAINED_REFS=1`.
//!
//! # Codex finding #1 disposition CLOSED by iter91 borrowed-session
//!
//! iter90 noted that `fence_or_commit` Sessioned arm called `fence_stage`
//! and dropped the session WITHOUT calling `reset_for_next_stage` — so no
//! `encodeWaitForEvent:value:` was ever encoded on a downstream CB.
//! iter90b's first-pass fix added `reset_for_next_stage` AT THE PER-STAGE
//! drop site; empirically that broke env=1 inference (apex 27B-DWQ46
//! prefill hung past 14 minutes vs the iter90 baseline of ~133ms): in the
//! per-stage construction shape, `reset_for_next_stage` allocates a fresh
//! CB and encodes a wait on it, then the session drops in case 3
//! (Encoding-uncommitted) per `encoder_session.rs::Drop` — discarding both
//! the new CB and its encoded waits, but creating Metal-side back-pressure.
//!
//! iter91 implements the CORRECT structural fix: the borrowed-session
//! multi-stage chain.  The session is constructed ONCE per `forward_gpu_impl`
//! (one allocation, one drop).  The `reset_for_next_stage` call inside the
//! Sessioned arm of `fence_or_commit` rotates to a fresh CB that is THE
//! NEXT STAGE'S DISPATCH TARGET, not a discarded one.  The encoded wait
//! orders that next CB behind the prior fence's signal-event.  Each
//! `reset_for_next_stage` is matched 1:1 by a subsequent fence or commit on
//! the same session.  The terminal output-head `commit_and_wait` drains the
//! GPU and clears `fence_pending`; the session drops at end of
//! `forward_gpu_impl` in the Drained state — `Drop` Case 1 — clean release.
//!
//! # iter91 H2 CB-count reduction LANDED
//!
//! `carry_into_next_stage(label)` is the iter91 verb that REPLACES intra-K
//! FFN `fence_or_commit` calls with `enc.encoder().memory_barrier()` only
//! on the Sessioned arm — keeping the CB OPEN so the next layer's first
//! dispatch encodes into the SAME persistent compute encoder.  This is the
//! pattern that `/opt/mlx-native/tests/encoder_session_cb_count_smoke.rs`
//! proves achieves a factor-2x CB-count reduction (`cb_count_session=5,
//! cb_count_plain=10`).  The hf2q production-level reduction is empirically
//! measured at the `scripts/iter91-cb-count-probe.sh` site (Worker B
//! deliverable for AC-2 evidence).
//!
//! # Scope (qwen35-only)
//!
//! This module is `pub(super)` and used only by `forward_gpu.rs`,
//! `gpu_full_attn.rs`, and `gpu_delta_net.rs` inside the qwen35 module
//! tree. No other crate may import `LayerEncoder` — the wire-up is
//! intentionally local to qwen35 prefill.
//!
//! # iter90 OQ2 disposition (carried forward by iter91)
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

/// Per-stage encoder used by qwen35 prefill paths.  The lifetime parameter
/// `'sess` ties the `Sessioned` arm to a borrow of an outer-scope
/// `EncoderSession` (constructed once per `forward_gpu_impl` call).  The
/// `Plain` arm carries no borrow; the `'sess` parameter is unused on that
/// variant (Rust permits this — the compiler infers `'static` for unused
/// lifetime carriers when the enum is constructed via `Plain(...)`).
pub(super) enum LayerEncoder<'sess> {
    /// Default, env=0 path. Direct owned `CommandEncoder` — committed at the
    /// stage boundary like the pre-iter90 code shape.
    Plain(CommandEncoder),
    /// `HF2Q_ENCODER_SESSION=1` path. Borrows the per-`forward_gpu_impl`
    /// `EncoderSession` so consecutive stages can share the same CB
    /// (`carry_into_next_stage` Sessioned arm) and stage-fence boundaries
    /// route through `EncoderSession::fence_stage` + `reset_for_next_stage`.
    Sessioned(&'sess mut EncoderSession),
}

impl<'sess> LayerEncoder<'sess> {
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

    /// Construct a new layer encoder, branching on the optional borrowed
    /// session.
    ///
    /// * `session = None` → `Plain(device.command_encoder()?)`.  This is the
    ///   env=0 path and is byte-identical to the pre-iter91 `LayerEncoder::new`
    ///   Plain arm.  Callers under env=0 always pass `None`.
    /// * `session = Some(sess)` → `Sessioned(sess)`.  The borrow is held for
    ///   the lifetime of the returned `LayerEncoder`; when the encoder is
    ///   consumed by `fence_or_commit` / `carry_into_next_stage` /
    ///   `commit_unlabeled` / `commit_and_wait_labeled`, the borrow is
    ///   released back to the caller's `Option<EncoderSession>` and the
    ///   next stage can re-borrow.
    ///
    /// # Caller contract for env=1
    ///
    /// Under `HF2Q_ENCODER_SESSION=1`, the caller (`forward_gpu_impl`) is
    /// expected to allocate a single `Option<EncoderSession>` between the
    /// arena setup and the per-layer loop, and to thread `as_deref_mut()`
    /// (or `as_mut()`) through helper calls down to every per-stage
    /// `LayerEncoder::from_session_or_plain` site.  The session MUST NOT
    /// be dropped before the terminal output-head `commit_and_wait_labeled`
    /// drains the GPU — dropping a Fenced session before the matching wait
    /// is the iter90b "14-minute Metal back-pressure hang" antipattern.
    ///
    /// # Errors
    ///
    /// Returns `Err` only on the Plain arm, propagating
    /// `MlxDevice::command_encoder` failures.  The Sessioned arm is
    /// infallible (the borrow is already validated by the caller's outer
    /// allocation site).
    pub(super) fn from_session_or_plain(
        device: &MlxDevice,
        session: Option<&'sess mut EncoderSession>,
    ) -> Result<Self> {
        match session {
            Some(sess) => Ok(LayerEncoder::Sessioned(sess)),
            None => {
                let enc = device
                    .command_encoder()
                    .context("LayerEncoder::from_session_or_plain: device.command_encoder()")?;
                Ok(LayerEncoder::Plain(enc))
            }
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
    /// Use [`Self::fence_or_commit`] / [`Self::carry_into_next_stage`] /
    /// [`Self::commit_and_wait_labeled`] at the stage boundary so the
    /// (Sessioned) variant's drained-latch / fence state stays consistent.
    /// Calling the inner commit bypasses the session's state machine — it
    /// is not unsafe (no UB) but it leaves the session in an inconsistent
    /// state with respect to its own view of what it has committed.
    #[inline]
    pub(super) fn encoder(&mut self) -> &mut CommandEncoder {
        match self {
            LayerEncoder::Plain(enc) => enc,
            LayerEncoder::Sessioned(sess) => sess.encoder(),
        }
    }

    /// Non-blocking stage-fence boundary (full CB termination + fence).
    ///
    /// * `Plain` → `inner.commit_labeled(label)`. Byte-identical behavior
    ///   to the pre-iter90 `enc.commit_labeled(label)` call site. The
    ///   encoder is consumed (the underlying `CommandEncoder` is moved
    ///   out and dropped after `commit_labeled`).
    /// * `Sessioned` → `sess.fence_stage(Some(label))?` followed by
    ///   `sess.reset_for_next_stage()?`.  Submits the current CB with a
    ///   signal-event encoded, then opens a fresh CB and encodes the
    ///   matching `encodeWaitForEvent:value:` on it so the next dispatch
    ///   blocks behind the prior signal.  In iter91's borrowed-session
    ///   shape, the freshly-rotated CB is the NEXT stage's dispatch
    ///   target — the wait orders that next stage's GPU work behind the
    ///   prior fence's signal.  No CB is discarded; no Metal back-pressure.
    ///
    /// Consumes self because the `Plain` arm cannot continue after
    /// `commit_labeled` (the `CommandEncoder` is moved).  The `Sessioned`
    /// arm releases the session borrow back to the caller's `Option`.
    ///
    /// # Use at attention-stage fences (KEEP in iter91)
    ///
    /// The two FA `layer.full_attn.stage_a` / `layer.full_attn.ops6-7`
    /// sites in `gpu_full_attn.rs` and the `layer.gdn.stage_a` site in
    /// `gpu_delta_net.rs` continue to use `fence_or_commit` per iter91
    /// spec §1.6 — they are the natural stage boundaries between attention
    /// and FFN within a single layer, where the FFN reads attention's
    /// output and the next CB starts under the wait-event.
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
            LayerEncoder::Sessioned(sess) => {
                sess.fence_stage(Some(label))
                    .with_context(|| format!("EncoderSession::fence_stage({label})"))?;
                // iter91 borrowed-session: the session lives until the outer
                // forward_gpu_impl drops the Option<EncoderSession> AFTER the
                // output-head terminal commit_and_wait drains the GPU.  The
                // freshly-rotated CB here is the NEXT stage's dispatch target,
                // and the encoded wait orders its GPU work behind this fence's
                // signal.  This is exactly the iter90b spec §2 H1b shape that
                // the wait_event + cb_count smoke tests prove correct in
                // mlx-native — see /opt/mlx-native/tests/encoder_session_*.rs.
                sess.reset_for_next_stage()
                    .with_context(|| format!("EncoderSession::reset_for_next_stage({label})"))?;
                Ok(())
            }
        }
    }

    /// Intra-CB carry — single-CB sequencing of the next stage.
    ///
    /// This is iter91's H2 CB-count reduction primitive.  The Sessioned arm
    /// keeps the persistent compute encoder OPEN so the next stage's first
    /// dispatch encodes into the SAME CB; the Plain arm preserves
    /// pre-iter91 semantics (per-stage CB termination via
    /// `commit_labeled`).
    ///
    /// * `Plain` → `inner.commit_labeled(label)` — byte-identical to
    ///   `fence_or_commit` Plain arm.  At env=0 there is no session to
    ///   carry, so each stage owns its own CB just like before.
    /// * `Sessioned` → `sess.encoder().memory_barrier()` ONLY.  No fence,
    ///   no reset, no commit.  The next stage's first dispatch encodes
    ///   into the SAME persistent compute encoder; the memory_barrier
    ///   enforces the producer→consumer RAW edge across the stage
    ///   boundary (Metal's `MTLDispatchTypeConcurrent` would otherwise
    ///   reorder the two stages within the same CB).
    ///
    /// # When to use
    ///
    /// Use `carry_into_next_stage` at intra-K FFN-end boundaries where
    /// the next layer's pre-attention norm reads the FFN's output:
    ///   - `forward_gpu.rs::forward_gpu_impl` DenseQ intra-K terminal
    ///     (`layer.dense_ffn`).
    ///   - `forward_gpu.rs::forward_gpu_impl` MoeQ intra-K terminal
    ///     (`layer.moe_ffn`).
    ///
    /// Do NOT use at K-boundary terminals (use `commit_and_wait_labeled`)
    /// or at attention-stage fences within a single layer (use
    /// `fence_or_commit` — per iter91 spec §1.6 those sites stay).
    ///
    /// # Errors
    ///
    /// Infallible on both arms.  The `Result` is preserved for symmetry
    /// with `fence_or_commit` and for future-proofing.
    pub(super) fn carry_into_next_stage(self, label: &str) -> Result<()> {
        match self {
            LayerEncoder::Plain(mut enc) => {
                // env=0 has no session to carry; preserve pre-iter91
                // per-stage CB termination behavior byte-for-byte.
                enc.commit_labeled(label);
                Ok(())
            }
            LayerEncoder::Sessioned(sess) => {
                // env=1 H2 reduction: keep the CB open; emit only the
                // RAW barrier the next stage's first dispatch needs.
                // The label is intentionally not propagated to the
                // active CB (it would only matter at a CB termination
                // boundary, which this is not).  xctrace MST attribution
                // for the carried section falls under the next
                // fence/commit's label.
                let _ = label;
                sess.encoder().memory_barrier();
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
    ///
    /// Under iter91 the `Err` arm carries the lifetime parameter `'sess`
    /// back to the caller; the caller drops it (releasing the session
    /// borrow) before continuing on the Sessioned-error fallback path.
    pub(super) fn try_into_inner_command_encoder(
        self,
    ) -> std::result::Result<CommandEncoder, Self> {
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
    ///   (encoder_session.rs:395-411 — the empty-label branch).
    ///
    /// Decode is classified as DROP_SITE in the iter90 spec §1.1; this
    /// method exists so the `fused_enc: Option<LayerEncoder<'sess>>`
    /// plumbing can preserve the existing decode behavior without forcing
    /// the call site to special-case the variant.
    ///
    /// Consumes self.  Note: in the iter91 borrowed-session shape, decode
    /// (`seq_len == 1`) does NOT allocate a session in `forward_gpu_impl`,
    /// so the Sessioned arm here is structurally unreachable on the
    /// decode path.  The arm is preserved for completeness — callers may
    /// reach it on greedy/argmax paths if they thread a session in the
    /// future.
    pub(super) fn commit_unlabeled(self) -> Result<()> {
        match self {
            LayerEncoder::Plain(mut enc) => {
                enc.commit();
                Ok(())
            }
            LayerEncoder::Sessioned(sess) => sess
                .commit_stage()
                .context("EncoderSession::commit_stage (decode)"),
        }
    }

    /// Blocking commit boundary (TERMINAL sites — K-batch boundary,
    /// host CPU read, etc.).
    ///
    /// * `Plain` → `inner.commit_and_wait_labeled(label)`. Byte-identical
    ///   to the pre-iter90 call shape.
    /// * `Sessioned` → `sess.begin_stage(label)` then
    ///   `sess.commit_and_wait()`.  The session enters the `Drained` state
    ///   with no fence pending — blocking commit fully drains the GPU, so
    ///   the next stage (after `reset_for_next_stage` on a subsequent
    ///   re-borrow) needs no wait-event.  On the K-boundary path the next
    ///   per-layer `LayerEncoder::from_session_or_plain` re-borrows the
    ///   same session; its first dispatch reuses the now-Drained state
    ///   and `reset_for_next_stage` opens a fresh CB without a wait.
    ///
    /// Consumes self; releases the session borrow.
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
            LayerEncoder::Sessioned(sess) => {
                sess.begin_stage(label);
                sess.commit_and_wait()
                    .with_context(|| format!("EncoderSession::commit_and_wait({label})"))?;
                // iter91 borrowed-session: after commit_and_wait the inner
                // CommandEncoder still holds the COMMITTED CB.  The session
                // is reused across layers (the next layer's FA / DN helper
                // re-borrows `layer_session.as_mut()` and dispatches into
                // `sess.encoder()` which lazily opens a compute encoder on
                // `sess.inner.cmd_buf` via `get_or_create_encoder`) — so we
                // MUST allocate a fresh CB before the borrow returns.
                // `reset_for_next_stage` is safe here: drained == true after
                // commit_and_wait, and fence_pending == false, so no wait
                // event is encoded on the new CB (none is needed — the wait
                // already happened on the host).  This is the same shape
                // K-boundary callers needed pre-iter91 from the Plain arm
                // (`device.command_encoder()` opens a fresh CB for the
                // next layer).
                sess.reset_for_next_stage()
                    .with_context(|| {
                        format!(
                            "EncoderSession::reset_for_next_stage after commit_and_wait({label})"
                        )
                    })?;
            }
        }
        Ok(())
    }
}
