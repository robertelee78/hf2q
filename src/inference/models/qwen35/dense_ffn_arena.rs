//! Caller-owned scratch arenas for FFN scratches across all FFN layers in
//! a single prefill (ADR-015 iter72).
//!
//! Two arena types are provided:
//! - [`DenseFfnArena`] for the dense-Q path
//!   ([`super::gpu_ffn::build_dense_ffn_layer_gpu_q_into_with_arena`])
//! - [`MoeFfnArena`] for the MoE-Q path
//!   ([`super::gpu_ffn::build_moe_ffn_layer_gpu_q_into_with_arena`])
//!
//! # Why
//!
//! `wave5b8_profile` measurements at pp4096 27B q4_0-flat showed
//! `ffn.alloc_scratch` accounting for ~365 ms / ~14 % of prefill wall — the
//! sum of 40 layers' worth of `pooled_alloc_buffer` calls inside
//! `build_dense_ffn_layer_gpu_q_into_pooled`. p50 = 1.3 ms but
//! max = 41.892 ms: a small number of cold first-allocation outliers (Metal
//! `new_buffer` + zero-init memset + residency `add_allocation` +
//! `set.commit()`) dominate the bucket. Subsequent layers re-pop from the
//! pool's free list, but the K-batch reset (default K=8) plus the per-call
//! `set.commit()` keeps the path on the slow path more often than necessary.
//!
//! Lifting the four transient scratches (gate, up, hidden, silu_params) to
//! caller scope eliminates the per-layer alloc churn entirely. Allocated
//! ONCE at the top of `forward_gpu_impl`, reused across all 40 dense layers,
//! dropped at the end of `forward_gpu_impl` AFTER the final
//! `commit_and_wait_labeled` at the output head.
//!
//! # Lifetime contract (mirrors FaPrefillArena)
//!
//! The arena is allocated ONCE per prefill (`seq_len > 1`) when the model
//! has at least one Dense FFN layer. It is reused across all dense layers in
//! the loop, then dropped at the end of `forward_gpu_impl` AFTER the final
//! encoder `commit_and_wait_labeled` at the output-head.
//!
//! **Why this prevents the iter58b residency-rescission failure mode:**
//! The arena buffers are owned by `forward_gpu_impl` for the entire prefill.
//! They do NOT drop at wrapper return. Therefore:
//!
//! 1. The wrapper's `enc.commit*` returns immediately; the wrapper returns
//!    immediately.
//! 2. Nothing inside the wrapper drops a `device.alloc_buffer` `MlxBuffer`
//!    that is still referenced by an in-flight command buffer.
//! 3. No deferred `removeAllocation:` is staged on the residency set when
//!    the wrapper returns.
//! 4. The next encoder's `commit*` does NOT flush a stale
//!    residency-rescission for buffers still referenced by the wrapper's
//!    CB.
//!
//! # Lifetime: WHY only the four transient scratches
//!
//! `gate_buf`, `up_buf`, `hidden_buf`, `silu_params_buf` are written + read
//! within ONE FFN call's encoder and never read by the next layer:
//! they are pure intra-layer scratch.
//!
//! The FINAL OUTPUT (`down_out` when no residual; `sum_buf` when residual is
//! folded) is intentionally **NOT** included in the arena — its ARC clone
//! leaves the function via `Ok(result)` and becomes the next layer's
//! `hidden`, crossing layer boundaries. A pooled output here would alias the
//! next layer's `gate_buf` when both happen to land in the same arena slot,
//! corrupting the residual stream silently. The current
//! `build_dense_ffn_layer_gpu_q_into_pooled` device-allocates the output
//! buffer at prefill (line 974-978 of gpu_ffn.rs); we keep that path
//! verbatim.
//!
//! # ADR-013 P21 Stage 1 precedent
//!
//! Identical pattern to `FaPrefillArena`, validated as the structural fix
//! for the iter58b commit/commit_and_wait race. See
//! `qwen35::fa_prefill_arena` doc-module for the long form.

use anyhow::{anyhow, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

/// Caller-owned scratch arena for the dense-Q FFN bridge across all dense
/// layers in a single prefill.
///
/// Contains the four F32 transient scratches that
/// `build_dense_ffn_layer_gpu_q_into_pooled` currently allocates per-layer
/// via `decode_pool::pooled_alloc_buffer`. Lifting them to caller scope
/// keeps them alive for the full prefill, eliminating the per-layer alloc
/// churn captured by the W-5b.8 `ffn.alloc_scratch` bucket.
///
/// All four scratches are sized for the actual prefill `(seq_len, h, m)`
/// shape; per-layer `validate_fits` guards against accidental shape drift
/// on a future model that mixes dense layers of different intermediate
/// sizes (Qwen3.6 27B q4_0-flat is uniform — h=5120, m=17408 for all
/// dense layers).
pub struct DenseFfnArena {
    /// `[seq_len, m]` F32 — gate projection output.
    pub gate_buf: MlxBuffer,
    /// `[seq_len, m]` F32 — up projection output.
    pub up_buf: MlxBuffer,
    /// `[seq_len, m]` F32 — silu(gate)*up intermediate.
    pub hidden_buf: MlxBuffer,
    /// `[1]` U32 — silu_mul element count parameter buffer.
    pub silu_params_buf: MlxBuffer,

    // ── Capacity bookkeeping ─────────────────────────────────────────────
    /// The `seq_capacity` value the arena was allocated for.
    pub seq_capacity: u32,
    /// The `hidden_size` the arena was allocated for.
    pub hidden_size: u32,
    /// The `intermediate_size` the arena was allocated for.
    pub intermediate_size: u32,
}

impl DenseFfnArena {
    /// Allocate all four F32 scratches sized for a single prefill pass with
    /// the given `(seq_capacity, hidden_size, intermediate_size)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Any dimension is zero.
    /// - Any `device.alloc_buffer` call fails (out of GPU memory).
    pub fn new(
        device: &MlxDevice,
        seq_capacity: u32,
        hidden_size: u32,
        intermediate_size: u32,
    ) -> Result<Self> {
        if seq_capacity == 0 || hidden_size == 0 || intermediate_size == 0 {
            return Err(anyhow!(
                "DenseFfnArena::new: zero dim \
                 seq_capacity={} hidden_size={} intermediate_size={}",
                seq_capacity,
                hidden_size,
                intermediate_size
            ));
        }

        let seq = seq_capacity as usize;
        let m = intermediate_size as usize;
        let n_h_bytes = seq * m * 4;

        let gate_buf = device
            .alloc_buffer(n_h_bytes, DType::F32, vec![seq, m])
            .map_err(|e| anyhow!("DenseFfnArena alloc gate_buf: {e}"))?;
        let up_buf = device
            .alloc_buffer(n_h_bytes, DType::F32, vec![seq, m])
            .map_err(|e| anyhow!("DenseFfnArena alloc up_buf: {e}"))?;
        let hidden_buf = device
            .alloc_buffer(n_h_bytes, DType::F32, vec![seq, m])
            .map_err(|e| anyhow!("DenseFfnArena alloc hidden_buf: {e}"))?;
        let silu_params_buf = device
            .alloc_buffer(4, DType::U32, vec![1])
            .map_err(|e| anyhow!("DenseFfnArena alloc silu_params_buf: {e}"))?;

        Ok(Self {
            gate_buf,
            up_buf,
            hidden_buf,
            silu_params_buf,
            seq_capacity,
            hidden_size,
            intermediate_size,
        })
    }

    /// Validate that a per-layer call's shape fits inside the arena's
    /// capacity. The arena is sized for the actual prefill `seq_len`, so
    /// equality is the common case.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - `seq_len > self.seq_capacity` (would overrun the allocated buffer).
    /// - `hidden_size` or `intermediate_size` differ from the recorded
    ///   values (buffers were sized for a different shape).
    pub fn validate_fits(
        &self,
        seq_len: u32,
        hidden_size: u32,
        intermediate_size: u32,
    ) -> Result<()> {
        if seq_len > self.seq_capacity {
            return Err(anyhow!(
                "DenseFfnArena::validate_fits: seq_len {} exceeds capacity {}",
                seq_len,
                self.seq_capacity
            ));
        }
        if hidden_size != self.hidden_size || intermediate_size != self.intermediate_size {
            return Err(anyhow!(
                "DenseFfnArena::validate_fits: shape mismatch — \
                 arena (hidden_size={}, intermediate_size={}) vs \
                 call (hidden_size={}, intermediate_size={})",
                self.hidden_size,
                self.intermediate_size,
                hidden_size,
                intermediate_size,
            ));
        }
        Ok(())
    }
}

/// Caller-owned scratch arena for the MoE-Q FFN bridge across all MoE layers
/// in a single prefill.
///
/// Contains the eight transient scratches that
/// `build_moe_ffn_layer_gpu_q_into` currently allocates per-layer via
/// `decode_pool::pooled_alloc_buffer`. Lifting them to caller scope keeps
/// them alive for the full prefill, eliminating the per-layer alloc churn
/// captured by the W-5b.8 `ffn.alloc_scratch` bucket.
///
/// **NOT included** in the arena (kept device-allocated per layer):
/// - `out_buf` — function return value, becomes next layer's `hidden`.
///   Lifetime crosses layer boundary, so the same prefill exclusion
///   applied by the per-layer pool reset (gpu_ffn.rs:2189-2196,
///   "iter40 fix" comment) applies here.
///
/// **Memory footprint at 35B-A3B q4_0-flat pp4096** (h=5120, n_experts=256,
/// num_experts_per_tok=8, moe_intermediate_size=512, shared_intermediate=512):
///   - total_rows = seq * topk = 4096 * 8 = 32768
///   - gate_all + up_all + h_all = 3 × (32768 × 512 × 4) = 192 MB
///   - y_all = 32768 × 5120 × 4 = 670 MB
///   - h_s = seq × m_sh × 4 = 4096 × 512 × 4 = 8 MB
///   - ids + weights = ~256 KB
///   - silu_params + silu_sh_params = 8 bytes
///   Total: ~870 MB. Live for the entire prefill duration; M5 Max 128 GB
///   unified memory keeps this well within budget.
pub struct MoeFfnArena {
    /// `[total_rows]` U32 — top-k expert ids, total_rows = seq × topk.
    pub ids_buf: MlxBuffer,
    /// `[total_rows]` F32 — top-k expert weights.
    pub weights_buf: MlxBuffer,
    /// `[total_rows, m_moe]` F32 — concatenated expert gate projection.
    pub gate_all_buf: MlxBuffer,
    /// `[total_rows, m_moe]` F32 — concatenated expert up projection.
    pub up_all_buf: MlxBuffer,
    /// `[total_rows, m_moe]` F32 — silu(gate)*up intermediate.
    pub h_all_buf: MlxBuffer,
    /// `[total_rows, h]` F32 — concatenated expert down output.
    pub y_all_buf: MlxBuffer,
    /// `[seq, m_sh]` F32 — shared expert silu_mul intermediate.
    pub h_s_buf: MlxBuffer,
    /// `[1]` U32 — silu_mul element count for expert path.
    pub silu_params_buf: MlxBuffer,
    /// `[1]` U32 — silu_mul element count for shared expert path.
    pub silu_sh_params_buf: MlxBuffer,
    /// `[1]` F32 — placeholder when add_residual=None (kernel requires a
    /// valid buffer reference even when the add_residual flag is 0).
    pub dummy_residual_buf: MlxBuffer,

    // ── ADR-019 Phase 2 iter90b H5b — Phase A projection outputs ────────
    //
    // The four `proj_pooled` outputs at `gpu_ffn.rs:2651-2691` (router
    // + shared gate + shared up projections) bind helper-local
    // `MlxBuffer`s into the FFN encoder.  Codex finding #2 flagged these
    // as crossing the non-blocking commit boundary under
    // `MLX_UNRETAINED_REFS=1`.  Lifting them to arena-anchored buffers
    // (which outlive the entire prefill) is the structural mitigation.
    //
    // All four shapes are `seq * out_features * 4` bytes — same shape
    // category as the existing arena fields above.  Memory cost on apex
    // 35B-A3B (h=5120, ne=128, m_sh=512, pp4096):
    //   logits_buf:   4096 × 128 × 4 = 2.0 MB
    //   sh_logit_buf: 4096 × 1   × 4 = 16 KB
    //   a_s_buf:      4096 × 512 × 4 = 8.0 MB
    //   b_s_buf:      4096 × 512 × 4 = 8.0 MB
    // Total: ~18 MB — negligible vs the existing ~870 MB MoeFfnArena.

    /// `[seq, num_experts]` F32 — router logits.  Replaces the
    /// helper-local `proj_pooled` allocation at `gpu_ffn.rs:2651`.
    pub logits_buf: MlxBuffer,
    /// `[seq, 1]` F32 — shared expert gate input logit.  Replaces
    /// `gpu_ffn.rs:2661`.
    pub sh_logit_buf: MlxBuffer,
    /// `[seq, m_sh]` F32 — shared expert gate output.  Replaces
    /// `gpu_ffn.rs:2671`.
    pub a_s_buf: MlxBuffer,
    /// `[seq, m_sh]` F32 — shared expert up output.  Replaces
    /// `gpu_ffn.rs:2681`.
    pub b_s_buf: MlxBuffer,

    // ── Capacity bookkeeping ─────────────────────────────────────────────
    /// The `seq_capacity` value the arena was allocated for.
    pub seq_capacity: u32,
    /// The `hidden_size` the arena was allocated for.
    pub hidden_size: u32,
    /// The `num_experts_per_tok` (topk) the arena was allocated for.
    pub num_experts_per_tok: u32,
    /// The `moe_intermediate_size` the arena was allocated for.
    pub moe_intermediate_size: u32,
    /// The `shared_intermediate_size` the arena was allocated for.
    pub shared_intermediate_size: u32,
    /// The `num_experts` (router output dim) the arena was allocated for.
    /// Iter90b H5b — sizes `logits_buf`.
    pub num_experts: u32,
}

impl MoeFfnArena {
    /// Allocate all eight transient scratches sized for a single prefill
    /// pass with the given shape parameters.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any dimension is zero or any `device.alloc_buffer`
    /// call fails.
    pub fn new(
        device: &MlxDevice,
        seq_capacity: u32,
        hidden_size: u32,
        num_experts_per_tok: u32,
        moe_intermediate_size: u32,
        shared_intermediate_size: u32,
        num_experts: u32,
    ) -> Result<Self> {
        if seq_capacity == 0
            || hidden_size == 0
            || num_experts_per_tok == 0
            || moe_intermediate_size == 0
            || shared_intermediate_size == 0
            || num_experts == 0
        {
            return Err(anyhow!(
                "MoeFfnArena::new: zero dim \
                 seq_capacity={} hidden_size={} num_experts_per_tok={} \
                 moe_intermediate_size={} shared_intermediate_size={} \
                 num_experts={}",
                seq_capacity,
                hidden_size,
                num_experts_per_tok,
                moe_intermediate_size,
                shared_intermediate_size,
                num_experts,
            ));
        }

        let seq = seq_capacity as usize;
        let h = hidden_size as usize;
        let topk = num_experts_per_tok as usize;
        let m_moe = moe_intermediate_size as usize;
        let m_sh = shared_intermediate_size as usize;
        let ne = num_experts as usize;
        let total_rows = seq * topk;

        let ids_buf = device
            .alloc_buffer(total_rows * 4, DType::U32, vec![total_rows])
            .map_err(|e| anyhow!("MoeFfnArena alloc ids_buf: {e}"))?;
        let weights_buf = device
            .alloc_buffer(total_rows * 4, DType::F32, vec![total_rows])
            .map_err(|e| anyhow!("MoeFfnArena alloc weights_buf: {e}"))?;
        let gate_all_bytes = total_rows * m_moe * 4;
        let gate_all_buf = device
            .alloc_buffer(gate_all_bytes, DType::F32, vec![total_rows, m_moe])
            .map_err(|e| anyhow!("MoeFfnArena alloc gate_all_buf: {e}"))?;
        let up_all_buf = device
            .alloc_buffer(gate_all_bytes, DType::F32, vec![total_rows, m_moe])
            .map_err(|e| anyhow!("MoeFfnArena alloc up_all_buf: {e}"))?;
        let h_all_buf = device
            .alloc_buffer(gate_all_bytes, DType::F32, vec![total_rows, m_moe])
            .map_err(|e| anyhow!("MoeFfnArena alloc h_all_buf: {e}"))?;
        let y_all_bytes = total_rows * h * 4;
        let y_all_buf = device
            .alloc_buffer(y_all_bytes, DType::F32, vec![total_rows, h])
            .map_err(|e| anyhow!("MoeFfnArena alloc y_all_buf: {e}"))?;
        let h_s_bytes = seq * m_sh * 4;
        let h_s_buf = device
            .alloc_buffer(h_s_bytes, DType::F32, vec![seq, m_sh])
            .map_err(|e| anyhow!("MoeFfnArena alloc h_s_buf: {e}"))?;
        let silu_params_buf = device
            .alloc_buffer(4, DType::U32, vec![1])
            .map_err(|e| anyhow!("MoeFfnArena alloc silu_params_buf: {e}"))?;
        let silu_sh_params_buf = device
            .alloc_buffer(4, DType::U32, vec![1])
            .map_err(|e| anyhow!("MoeFfnArena alloc silu_sh_params_buf: {e}"))?;
        let dummy_residual_buf = device
            .alloc_buffer(4, DType::F32, vec![1])
            .map_err(|e| anyhow!("MoeFfnArena alloc dummy_residual_buf: {e}"))?;

        // ── ADR-019 Phase 2 iter90b H5b — Phase A projection arena slots ──
        let logits_bytes = seq * ne * 4;
        let logits_buf = device
            .alloc_buffer(logits_bytes, DType::F32, vec![seq, ne])
            .map_err(|e| anyhow!("MoeFfnArena alloc logits_buf: {e}"))?;
        let sh_logit_bytes = seq * 4;
        let sh_logit_buf = device
            .alloc_buffer(sh_logit_bytes, DType::F32, vec![seq, 1])
            .map_err(|e| anyhow!("MoeFfnArena alloc sh_logit_buf: {e}"))?;
        let a_s_bytes = seq * m_sh * 4;
        let a_s_buf = device
            .alloc_buffer(a_s_bytes, DType::F32, vec![seq, m_sh])
            .map_err(|e| anyhow!("MoeFfnArena alloc a_s_buf: {e}"))?;
        let b_s_buf = device
            .alloc_buffer(a_s_bytes, DType::F32, vec![seq, m_sh])
            .map_err(|e| anyhow!("MoeFfnArena alloc b_s_buf: {e}"))?;

        Ok(Self {
            ids_buf,
            weights_buf,
            gate_all_buf,
            up_all_buf,
            h_all_buf,
            y_all_buf,
            h_s_buf,
            silu_params_buf,
            silu_sh_params_buf,
            dummy_residual_buf,
            logits_buf,
            sh_logit_buf,
            a_s_buf,
            b_s_buf,
            seq_capacity,
            hidden_size,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_intermediate_size,
            num_experts,
        })
    }

    /// Validate that a per-layer call's shape fits inside the arena's
    /// capacity. The arena is sized for the actual prefill `seq_len`, so
    /// equality is the common case.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `seq_len > self.seq_capacity` or any shape field
    /// differs from the recorded values.
    pub fn validate_fits(
        &self,
        seq_len: u32,
        hidden_size: u32,
        num_experts_per_tok: u32,
        moe_intermediate_size: u32,
        shared_intermediate_size: u32,
        num_experts: u32,
    ) -> Result<()> {
        if seq_len > self.seq_capacity {
            return Err(anyhow!(
                "MoeFfnArena::validate_fits: seq_len {} exceeds capacity {}",
                seq_len,
                self.seq_capacity
            ));
        }
        if hidden_size != self.hidden_size
            || num_experts_per_tok != self.num_experts_per_tok
            || moe_intermediate_size != self.moe_intermediate_size
            || shared_intermediate_size != self.shared_intermediate_size
            || num_experts != self.num_experts
        {
            return Err(anyhow!(
                "MoeFfnArena::validate_fits: shape mismatch — \
                 arena (h={}, topk={}, m_moe={}, m_sh={}, ne={}) vs \
                 call (h={}, topk={}, m_moe={}, m_sh={}, ne={})",
                self.hidden_size,
                self.num_experts_per_tok,
                self.moe_intermediate_size,
                self.shared_intermediate_size,
                self.num_experts,
                hidden_size,
                num_experts_per_tok,
                moe_intermediate_size,
                shared_intermediate_size,
                num_experts,
            ));
        }
        Ok(())
    }
}

/// ADR-019 Phase 2 iter90b H4b — caller-owned arena for the FFN-boundary
/// `ffn_input` and `ffn_residual` buffers, lifted from the per-layer
/// `device.alloc_buffer` calls at `forward_gpu.rs:2776-2790`.
///
/// # Why
///
/// Codex finding #2 against iter90 flagged `ffn_input_buf` and
/// `ffn_residual_buf` as helper-local `MlxBuffer`s bound into the FFN
/// encoder.  Under `MLX_UNRETAINED_REFS=1` (where CB ARC retains are
/// SKIPPED), these buffers can drop after the non-blocking
/// `fence_or_commit` on the FFN CB while the GPU is still pipelining.
/// That is the iter58b residency-rescission failure mode.
///
/// Hoisting both buffers to a per-prefill arena owned by `forward_gpu_impl`
/// — same lifetime pattern as `MoeFfnArena` / `DenseFfnArena` /
/// `DnPrefillArena` — eliminates the failure mode structurally: the
/// arena outlives every per-layer encoder commit, so no `MlxBuffer` drop
/// stages a deferred residency-removal during the in-flight CB window.
///
/// # Lifetime contract
///
/// Allocated ONCE per prefill (`seq_len > 1`) just before the per-layer
/// loop in `forward_gpu_impl`.  The two buffers are reused across all N
/// layers — content is overwritten by each layer's
/// `dispatch_fused_residual_norm_f32` call (which writes both
/// `ffn_input` = rms_norm(hidden + attn_out) and
/// `ffn_residual` = hidden + attn_out).  The arena is dropped at the
/// end of `forward_gpu_impl` AFTER the output-head terminal
/// `commit_and_wait_labeled`, which drains the GPU and frees the
/// buffers safely.
///
/// **Decode (seq_len == 1) policy:** decode is DROP_SITE per iter90b
/// spec §1.1 / §5.2.  The decode arm continues to use per-call
/// `device.alloc_buffer` (one alloc per call, dropped at end of
/// `forward_gpu_greedy`).  This arena is `Option<>` and gated on
/// `seq_len > 1`.
///
/// # Memory cost
///
/// Two `[seq_capacity, hidden_size]` F32 buffers.  At pp4096 × h=5120:
///   2 × 4096 × 5120 × 4 = 167 MB.  Negligible vs the existing ~870 MB
/// MoeFfnArena footprint on apex.
///
/// # Risk register
///
/// F2 invariant preservation: the lift mirrors `MoeFfnArena` /
/// `DenseFfnArena` exactly (caller-owned, prefill-lifetime, F32 layout).
/// No new fence-class risk; the iter58b argument applies verbatim.
pub struct LayerBoundaryArena {
    /// `[seq_capacity, hidden_size]` F32 — FFN input
    /// (= `rms_norm(hidden + attn_out)`).  Reused across all N layers.
    pub ffn_input_buf: MlxBuffer,
    /// `[seq_capacity, hidden_size]` F32 — pre-FFN residual
    /// (= `hidden + attn_out`).  Reused across all N layers.
    pub ffn_residual_buf: MlxBuffer,

    // ── Capacity bookkeeping ─────────────────────────────────────────────
    pub seq_capacity: u32,
    pub hidden_size: u32,
}

impl LayerBoundaryArena {
    /// Allocate both F32 boundary buffers for a single prefill.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any dimension is zero or any underlying
    /// `device.alloc_buffer` fails.
    pub fn new(device: &MlxDevice, seq_capacity: u32, hidden_size: u32) -> Result<Self> {
        if seq_capacity == 0 || hidden_size == 0 {
            return Err(anyhow!(
                "LayerBoundaryArena::new: zero dim seq_capacity={} hidden_size={}",
                seq_capacity,
                hidden_size,
            ));
        }
        let bytes = (seq_capacity as usize) * (hidden_size as usize) * 4;
        let shape = vec![seq_capacity as usize, hidden_size as usize];
        let ffn_input_buf = device
            .alloc_buffer(bytes, DType::F32, shape.clone())
            .map_err(|e| anyhow!("LayerBoundaryArena alloc ffn_input_buf: {e}"))?;
        let ffn_residual_buf = device
            .alloc_buffer(bytes, DType::F32, shape)
            .map_err(|e| anyhow!("LayerBoundaryArena alloc ffn_residual_buf: {e}"))?;
        Ok(Self {
            ffn_input_buf,
            ffn_residual_buf,
            seq_capacity,
            hidden_size,
        })
    }

    /// Validate that a per-layer call's shape fits inside the arena's capacity.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `seq_len > self.seq_capacity` or
    /// `hidden_size != self.hidden_size`.
    pub fn validate_fits(&self, seq_len: u32, hidden_size: u32) -> Result<()> {
        if seq_len > self.seq_capacity {
            return Err(anyhow!(
                "LayerBoundaryArena::validate_fits: seq_len {} exceeds capacity {}",
                seq_len,
                self.seq_capacity
            ));
        }
        if hidden_size != self.hidden_size {
            return Err(anyhow!(
                "LayerBoundaryArena::validate_fits: hidden_size {} != arena {}",
                hidden_size,
                self.hidden_size
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device_or_skip() -> Option<MlxDevice> {
        MlxDevice::new().ok()
    }

    /// Qwen3.6 27B canonical shape at pp=4096: seq=4096, h=5120, m=17408.
    /// Verifies all four fields' byte_len() match the formula and capacity
    /// is recorded correctly.
    #[test]
    fn test_arena_new_qwen36_27b_pp4096() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_arena_new_qwen36_27b_pp4096: skipping — no Metal device");
                return;
            }
        };
        let (seq, h, m) = (4096u32, 5120u32, 17408u32);
        let arena = DenseFfnArena::new(&device, seq, h, m).expect("arena new pp4096");

        assert_eq!(arena.seq_capacity, seq);
        assert_eq!(arena.hidden_size, h);
        assert_eq!(arena.intermediate_size, m);

        let n_h_bytes = (seq as usize) * (m as usize) * 4;
        assert_eq!(arena.gate_buf.byte_len(), n_h_bytes, "gate_buf byte_len");
        assert_eq!(arena.up_buf.byte_len(), n_h_bytes, "up_buf byte_len");
        assert_eq!(arena.hidden_buf.byte_len(), n_h_bytes, "hidden_buf byte_len");
        assert_eq!(arena.silu_params_buf.byte_len(), 4, "silu_params byte_len");
    }

    /// Smaller shape sanity-check.
    #[test]
    fn test_arena_new_small_shape() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_arena_new_small_shape: skipping — no Metal device");
                return;
            }
        };
        let arena = DenseFfnArena::new(&device, 64, 128, 256).expect("arena new small");
        assert_eq!(arena.seq_capacity, 64);
    }

    /// Zero-dim rejection.
    #[test]
    fn test_arena_new_zero_dim_rejected() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_arena_new_zero_dim_rejected: skipping — no Metal device");
                return;
            }
        };
        assert!(DenseFfnArena::new(&device, 0, 128, 256).is_err());
        assert!(DenseFfnArena::new(&device, 64, 0, 256).is_err());
        assert!(DenseFfnArena::new(&device, 64, 128, 0).is_err());
    }

    /// validate_fits exact match returns Ok.
    #[test]
    fn test_validate_fits_exact_match() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_validate_fits_exact_match: skipping — no Metal device");
                return;
            }
        };
        let arena = DenseFfnArena::new(&device, 128, 256, 512).expect("arena new");
        assert!(arena.validate_fits(128, 256, 512).is_ok());
        // seq_len < capacity also Ok.
        assert!(arena.validate_fits(64, 256, 512).is_ok());
    }

    /// validate_fits seq overrun returns Err.
    #[test]
    fn test_validate_fits_seq_overrun() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_validate_fits_seq_overrun: skipping — no Metal device");
                return;
            }
        };
        let arena = DenseFfnArena::new(&device, 128, 256, 512).expect("arena new");
        assert!(arena.validate_fits(256, 256, 512).is_err());
    }

    /// validate_fits shape mismatch returns Err.
    #[test]
    fn test_validate_fits_shape_mismatch() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_validate_fits_shape_mismatch: skipping — no Metal device");
                return;
            }
        };
        let arena = DenseFfnArena::new(&device, 128, 256, 512).expect("arena new");
        assert!(arena.validate_fits(128, 128, 512).is_err());
        assert!(arena.validate_fits(128, 256, 256).is_err());
    }

    /// Verifies that device.alloc_buffer zero-initializes all buffers.
    /// Mirrors fa_prefill_arena's identical test and the ADR-015 iter61a
    /// zero-init guarantee documented in mlx_native/src/device.rs.
    #[test]
    fn test_arena_buffers_zero_initialized() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_arena_buffers_zero_initialized: skipping — no Metal device");
                return;
            }
        };
        let arena = DenseFfnArena::new(&device, 64, 128, 256).expect("arena new");

        let bufs: [(&MlxBuffer, &str); 3] = [
            (&arena.gate_buf, "gate_buf"),
            (&arena.up_buf, "up_buf"),
            (&arena.hidden_buf, "hidden_buf"),
        ];
        for (buf, name) in &bufs {
            let slice = buf
                .as_slice::<f32>()
                .unwrap_or_else(|e| panic!("{name} as_slice::<f32> failed: {e}"));
            let check_len = 16.min(slice.len());
            for (i, &v) in slice[..check_len].iter().enumerate() {
                assert_eq!(
                    v, 0.0f32,
                    "{name}[{i}] = {v} (expected zero from device.alloc_buffer)"
                );
            }
        }

        // silu_params is U32; verify zero-init too.
        let slice = arena
            .silu_params_buf
            .as_slice::<u32>()
            .expect("silu_params as_slice::<u32>");
        assert_eq!(slice[0], 0u32, "silu_params[0] should be zero from device.alloc_buffer");
    }

    // ── MoeFfnArena tests ───────────────────────────────────────────────

    /// Qwen3.6 35B-A3B canonical shape at pp=4096: seq=4096, h=5120, topk=8,
    /// m_moe=512, m_sh=512.
    #[test]
    fn test_moe_arena_new_qwen36_35b_pp4096() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_moe_arena_new_qwen36_35b_pp4096: skipping — no Metal device");
                return;
            }
        };
        let (seq, h, topk, m_moe, m_sh, ne) = (4096u32, 5120u32, 8u32, 512u32, 512u32, 128u32);
        let arena = MoeFfnArena::new(&device, seq, h, topk, m_moe, m_sh, ne).expect("moe arena new");

        assert_eq!(arena.seq_capacity, seq);
        assert_eq!(arena.hidden_size, h);
        assert_eq!(arena.num_experts_per_tok, topk);
        assert_eq!(arena.moe_intermediate_size, m_moe);
        assert_eq!(arena.shared_intermediate_size, m_sh);
        assert_eq!(arena.num_experts, ne);

        let total_rows = (seq as usize) * (topk as usize);
        let gate_all_bytes = total_rows * (m_moe as usize) * 4;
        let y_all_bytes = total_rows * (h as usize) * 4;
        let h_s_bytes = (seq as usize) * (m_sh as usize) * 4;
        assert_eq!(arena.gate_all_buf.byte_len(), gate_all_bytes, "gate_all_buf");
        assert_eq!(arena.up_all_buf.byte_len(), gate_all_bytes, "up_all_buf");
        assert_eq!(arena.h_all_buf.byte_len(), gate_all_bytes, "h_all_buf");
        assert_eq!(arena.y_all_buf.byte_len(), y_all_bytes, "y_all_buf");
        assert_eq!(arena.h_s_buf.byte_len(), h_s_bytes, "h_s_buf");

        // ── iter90b H5b — Phase A projection arena slot sizes ──
        let logits_bytes = (seq as usize) * (ne as usize) * 4;
        let sh_logit_bytes = (seq as usize) * 4;
        let a_s_bytes = (seq as usize) * (m_sh as usize) * 4;
        assert_eq!(arena.logits_buf.byte_len(), logits_bytes, "logits_buf");
        assert_eq!(arena.sh_logit_buf.byte_len(), sh_logit_bytes, "sh_logit_buf");
        assert_eq!(arena.a_s_buf.byte_len(), a_s_bytes, "a_s_buf");
        assert_eq!(arena.b_s_buf.byte_len(), a_s_bytes, "b_s_buf");
    }

    /// Smaller MoE shape sanity-check.
    #[test]
    fn test_moe_arena_new_small_shape() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_moe_arena_new_small_shape: skipping — no Metal device");
                return;
            }
        };
        let arena = MoeFfnArena::new(&device, 64, 128, 4, 256, 128, 8).expect("moe arena new");
        assert_eq!(arena.seq_capacity, 64);
        assert_eq!(arena.num_experts, 8);
    }

    /// MoE zero-dim rejection.
    #[test]
    fn test_moe_arena_new_zero_dim_rejected() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_moe_arena_new_zero_dim_rejected: skipping — no Metal device");
                return;
            }
        };
        assert!(MoeFfnArena::new(&device, 0, 128, 4, 256, 128, 8).is_err());
        assert!(MoeFfnArena::new(&device, 64, 0, 4, 256, 128, 8).is_err());
        assert!(MoeFfnArena::new(&device, 64, 128, 0, 256, 128, 8).is_err());
        assert!(MoeFfnArena::new(&device, 64, 128, 4, 0, 128, 8).is_err());
        assert!(MoeFfnArena::new(&device, 64, 128, 4, 256, 0, 8).is_err());
        assert!(MoeFfnArena::new(&device, 64, 128, 4, 256, 128, 0).is_err());
    }

    /// MoE validate_fits exact match returns Ok.
    #[test]
    fn test_moe_validate_fits_exact_match() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_moe_validate_fits_exact_match: skipping — no Metal device");
                return;
            }
        };
        let arena = MoeFfnArena::new(&device, 128, 256, 4, 512, 256, 16).expect("moe arena new");
        assert!(arena.validate_fits(128, 256, 4, 512, 256, 16).is_ok());
        assert!(arena.validate_fits(64, 256, 4, 512, 256, 16).is_ok());
    }

    /// MoE validate_fits seq overrun returns Err.
    #[test]
    fn test_moe_validate_fits_seq_overrun() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_moe_validate_fits_seq_overrun: skipping — no Metal device");
                return;
            }
        };
        let arena = MoeFfnArena::new(&device, 128, 256, 4, 512, 256, 16).expect("moe arena new");
        assert!(arena.validate_fits(256, 256, 4, 512, 256, 16).is_err());
    }

    /// MoE validate_fits shape mismatch returns Err.
    #[test]
    fn test_moe_validate_fits_shape_mismatch() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_moe_validate_fits_shape_mismatch: skipping — no Metal device");
                return;
            }
        };
        let arena = MoeFfnArena::new(&device, 128, 256, 4, 512, 256, 16).expect("moe arena new");
        assert!(arena.validate_fits(128, 128, 4, 512, 256, 16).is_err());
        assert!(arena.validate_fits(128, 256, 8, 512, 256, 16).is_err());
        assert!(arena.validate_fits(128, 256, 4, 256, 256, 16).is_err());
        assert!(arena.validate_fits(128, 256, 4, 512, 128, 16).is_err());
        assert!(arena.validate_fits(128, 256, 4, 512, 256, 32).is_err());
    }

    // ── ADR-019 Phase 2 iter90b H4b — LayerBoundaryArena tests ──

    /// Apex shape: pp4096 × h=5120 (Qwen3.6 27B/35B prefill).
    #[test]
    fn test_layer_boundary_arena_new_apex_shape() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_layer_boundary_arena_new_apex_shape: skipping — no Metal device"
                );
                return;
            }
        };
        let (seq, h) = (4096u32, 5120u32);
        let arena = LayerBoundaryArena::new(&device, seq, h).expect("new");
        assert_eq!(arena.seq_capacity, seq);
        assert_eq!(arena.hidden_size, h);
        let bytes = (seq as usize) * (h as usize) * 4;
        assert_eq!(arena.ffn_input_buf.byte_len(), bytes, "ffn_input_buf");
        assert_eq!(arena.ffn_residual_buf.byte_len(), bytes, "ffn_residual_buf");
    }

    /// Zero-dim rejection.
    #[test]
    fn test_layer_boundary_arena_zero_dim_rejected() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_layer_boundary_arena_zero_dim_rejected: skipping — no Metal device"
                );
                return;
            }
        };
        assert!(LayerBoundaryArena::new(&device, 0, 128).is_err());
        assert!(LayerBoundaryArena::new(&device, 128, 0).is_err());
    }

    /// validate_fits exact-match returns Ok; smaller seq_len OK; overrun Err;
    /// hidden_size mismatch Err.
    #[test]
    fn test_layer_boundary_arena_validate_fits() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_layer_boundary_arena_validate_fits: skipping — no Metal device"
                );
                return;
            }
        };
        let arena = LayerBoundaryArena::new(&device, 128, 256).expect("new");
        assert!(arena.validate_fits(128, 256).is_ok());
        assert!(arena.validate_fits(64, 256).is_ok());
        assert!(arena.validate_fits(256, 256).is_err()); // overrun
        assert!(arena.validate_fits(128, 128).is_err()); // shape mismatch
    }

    /// `MlxBuffer::clone()` preserves the underlying allocation
    /// (`contents_ptr` is identical) — the per-prefill arena lift
    /// relies on cheap Arc-clones of the owned buffers being bound
    /// into per-layer dispatches.  This is the structural property
    /// iter90b H4b depends on.
    #[test]
    fn test_layer_boundary_arena_clone_preserves_pointer() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_layer_boundary_arena_clone_preserves_pointer: skipping — no Metal device"
                );
                return;
            }
        };
        let arena = LayerBoundaryArena::new(&device, 64, 128).expect("new");
        let original_ptr = arena.ffn_input_buf.contents_ptr();
        let cloned = arena.ffn_input_buf.clone();
        assert_eq!(
            cloned.contents_ptr(),
            original_ptr,
            "MlxBuffer::clone must preserve the underlying Metal allocation pointer \
             (Arc-based)"
        );
        // Drop the clone; arena still holds the original.
        drop(cloned);
        assert_eq!(
            arena.ffn_input_buf.contents_ptr(),
            original_ptr,
            "arena buffer pointer unchanged after clone drop"
        );
    }
}
