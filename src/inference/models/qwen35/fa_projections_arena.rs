//! Caller-owned scratch arena for the FA-prefill projection + helper-output
//! buffers across all FullAttn layers in a single prefill (ADR-015 iter86).
//!
//! Mirrors [`super::DenseFfnArena`] / [`super::MoeFfnArena`] (iter72),
//! [`super::DnPrefillArena`] (iter74), [`super::ChunkAllocsArena`] (iter78),
//! and [`super::FaPrefillArena`] (ADR-013 P21 stage-1) — but for the
//! per-FA-layer projection / per-head-norm / IMROPE / sigmoid-gate scratch
//! buffers in [`super::gpu_full_attn::build_gated_attn_layer`]'s prefill body.
//!
//! # Why
//!
//! `wave5b8_profile` measurements at apex pp4127 (qwen3.6-35b-a3b q4_0-flat,
//! 30 DN layers + 10 FA layers, default-axis = autoreg path with
//! 4127 % 64 != 0) showed `fa.ops1_4` accounting for ~96 ms / 2.8 % of wall
//! at p95/p50 skew = 3.1× — the moderate-skew + small-mass profile that
//! arena lifts target. Within this bucket, the 4 large projection outputs
//! (Q/K/V/Gate) plus the 5 small helper scratches (x_norm, q_normed,
//! k_normed, q_rope, k_rope) are written + read across at most 3 encoders
//! per FA layer (ops1-4 builder, SDPA dispatcher, ops6-7 builder), then
//! never read by the next layer:
//!
//! - `q_flat`/`k_flat`/`v_flat`/`gate_flat`: per-FA-layer 4 device.alloc_buffer
//!   F32 outputs. At pp4127 these are 32+8+8+32 = 80 MiB / FA layer × 10
//!   layers = ~800 MiB churn per prefill. Currently allocated by
//!   [`super::gpu_full_attn::apply_linear_projection_f32`] (NOT pooled —
//!   prefill K/V need exact byte_len for downstream consumers; the existing
//!   bucket-rounded pool path is decode-only by design).
//!
//! - `x_norm`, `q_normed`, `k_normed`, `q_rope`, `k_rope`: 5 helper outputs
//!   from `apply_pre_attn_rms_norm` / `apply_q_or_k_per_head_rms_norm` /
//!   `apply_imrope`. These currently take `decode_pool::pooled_alloc_buffer`
//!   which is bucket-rounded (safe for these scratches because consumers
//!   don't touch byte_len()) but still incurs per-call thread-local pool
//!   bookkeeping at prefill granularity.
//!
//! - `gated`: ops6-7 sigmoid-gate multiply output, `[seq * q_total]` F32.
//!
//! Per Chesterton's-fence analysis (iter85 CANDIDATES.md §A): the
//! "decode-pool bucket-rounding inflates byte_len()" fence in
//! [`super::gpu_full_attn::apply_linear_projection_f32_pooled`] is
//! decode-pool-specific, NOT arena-general. Caller-owned arenas with
//! EXACT-SIZE allocations via `device.alloc_buffer` avoid byte_len
//! inflation entirely — `download_f32 → as_slice` reads the correct size.
//!
//! # Lifetime contract (mirrors FaPrefillArena)
//!
//! Allocated ONCE per prefill (`seq_len > 1`) when the model has at least
//! one `FullAttn` layer. Reused across all FA layers in the loop, then
//! dropped at the end of `forward_gpu_impl` AFTER the final
//! `commit_and_wait_labeled` at the output-head.
//!
//! **Why this prevents the iter58b residency-rescission failure mode:**
//! The arena buffers are owned by `forward_gpu_impl` for the entire prefill.
//! They do NOT drop at wrapper return. Therefore:
//!
//! 1. The wrapper's `enc.commit_labeled` returns immediately; the wrapper
//!    returns immediately.
//! 2. Nothing inside the wrapper drops a `device.alloc_buffer` `MlxBuffer`
//!    that is still referenced by an in-flight command buffer.
//! 3. No deferred `removeAllocation:` is staged on the residency set when
//!    the wrapper returns.
//! 4. The next encoder's `commit*` does NOT flush a stale residency
//!    rescission for buffers still referenced by the wrapper's CB.
//!
//! # NOT included in the arena
//!
//! The **O-proj output** (the F32 buffer returned by
//! [`super::gpu_full_attn::build_gated_attn_layer`] as the layer's
//! attention contribution) is intentionally **NOT** in this arena —
//! its ARC clone leaves the function via `Ok(out)` and is consumed by the
//! caller's `dispatch_fused_residual_norm_f32` in a subsequent encoder.
//! At the next FA layer's start, overwriting an arena slot for O-proj
//! could alias the previous layer's attention contribution while
//! `dispatch_fused_residual_norm_f32`'s CB is still in flight on the
//! Metal serial queue. Same exclusion pattern as
//! [`super::DenseFfnArena`]'s `down_out` and
//! [`super::ChunkAllocsArena`]'s `output_buf`.
//!
//! # NOT shared with FaPrefillArena
//!
//! [`super::FaPrefillArena`] holds **BF16** SDPA-internal scratches
//! (`apply_flash_attn_prefill_seq_major`'s 7 buffers). This arena holds
//! **F32** projection / per-head-norm / IMROPE / sigmoid-gate outputs.
//! Different dtype, different scope (FaPrefillArena's lifetime is the
//! SDPA call; this arena's lifetime spans the entire FA layer). Keeping
//! the two arenas separate matches the iter72/74/78 single-concern pattern.
//!
//! # Memory footprint (35B-A3B q4_0-flat pp4127)
//!
//! With `seq=4127, h=2048, n_head=16, n_kv=2, head_dim=256, q_total=4096,
//! kv_total=512`:
//!
//! | Buffer | DType | Bytes |
//! |---|---|---:|
//! | `x_norm_buf` | F32 | seq*h*4 = 33.8 MB |
//! | `q_proj_buf` | F32 | seq*q_total*4 = 67.6 MB |
//! | `k_proj_buf` | F32 | seq*kv_total*4 = 8.4 MB |
//! | `v_proj_buf` | F32 | seq*kv_total*4 = 8.4 MB |
//! | `gate_proj_buf` | F32 | seq*q_total*4 = 67.6 MB |
//! | `q_normed_buf` | F32 | seq*q_total*4 = 67.6 MB |
//! | `k_normed_buf` | F32 | seq*kv_total*4 = 8.4 MB |
//! | `q_rope_buf` | F32 | seq*q_total*4 = 67.6 MB |
//! | `k_rope_buf` | F32 | seq*kv_total*4 = 8.4 MB |
//! | `gated_buf` | F32 | seq*q_total*4 = 67.6 MB |
//! | `pre_norm_params_buf` | F32 | 8 |
//! | `qk_rms_params_buf` | F32 | 8 |
//! | `sigmoid_params_buf` | U32 | 4 |
//!
//! Total ≈ 405 MB. Live for the entire prefill duration; M5 Max 128 GB
//! unified memory keeps this well within budget.

use anyhow::{anyhow, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

/// Caller-owned scratch arena for the per-FA-layer projection + helper-output
/// buffers across all FA layers in a single prefill.
///
/// Contains the 4 large F32 projection outputs (Q/K/V/Gate), the 5 F32
/// helper outputs (x_norm, q_normed, k_normed, q_rope, k_rope), the F32
/// gated buffer, and the 3 small param buffers
/// (pre_norm_params, qk_rms_params, sigmoid_params) currently allocated
/// per-call by [`super::gpu_full_attn::build_gated_attn_layer`]'s prefill body.
///
/// The O-proj output is **NOT** in this arena — see module-level doc.
///
/// All scratches are sized for the actual prefill `(seq_len, …)` shape;
/// per-layer `validate_fits` guards against accidental shape drift on a
/// future model that mixes FA layers of different shapes (Qwen3.5/3.6 is
/// uniform — every FA layer has the same hidden_size, n_head, n_kv,
/// head_dim).
pub struct FaProjectionsArena {
    // ── Pre-attn RMSNorm output + params ──────────────────────────────────
    /// `[seq, hidden_size]` F32 — pre-attention RMSNorm output (op1).
    pub x_norm_buf: MlxBuffer,
    /// `[2]` F32 — pre_attn_rms_norm params `[eps, hidden_size_as_f32]`.
    /// Written ONCE at arena construction; constant for the whole prefill.
    pub pre_norm_params_buf: MlxBuffer,

    // ── Q/K/V/Gate projections (op2) ──────────────────────────────────────
    /// `[seq, n_head * head_dim]` F32 — Q projection output.
    pub q_proj_buf: MlxBuffer,
    /// `[seq, n_kv * head_dim]` F32 — K projection output.
    pub k_proj_buf: MlxBuffer,
    /// `[seq, n_kv * head_dim]` F32 — V projection output.
    pub v_proj_buf: MlxBuffer,
    /// `[seq, n_head * head_dim]` F32 — Gate projection output.
    pub gate_proj_buf: MlxBuffer,

    // ── Per-head RMSNorm outputs (op3) + shared params ────────────────────
    /// `[seq * n_head, head_dim]` F32 — Q after per-head RMSNorm.
    pub q_normed_buf: MlxBuffer,
    /// `[seq * n_kv, head_dim]` F32 — K after per-head RMSNorm.
    pub k_normed_buf: MlxBuffer,
    /// `[2]` F32 — per-head RMSNorm params `[eps, head_dim_as_f32]`. Same
    /// values for both Q and K (both use head_dim along the norm axis), so
    /// shared between calls. Written ONCE at arena construction.
    pub qk_rms_params_buf: MlxBuffer,

    // ── IMROPE outputs (op4) ──────────────────────────────────────────────
    /// `[seq, n_head, head_dim]` F32 — Q after IMROPE.
    pub q_rope_buf: MlxBuffer,
    /// `[seq, n_kv, head_dim]` F32 — K after IMROPE.
    pub k_rope_buf: MlxBuffer,

    // ── Sigmoid-gate multiply output + params (op6) ───────────────────────
    /// `[seq * n_head * head_dim]` F32 — sigmoid(gate) * attn_out.
    pub gated_buf: MlxBuffer,
    /// `[1]` U32 — sigmoid_mul element count = `seq * n_head * head_dim`.
    /// Written ONCE at arena construction; constant for the whole prefill.
    pub sigmoid_params_buf: MlxBuffer,

    // ── Capacity bookkeeping ──────────────────────────────────────────────
    /// The `seq_capacity` value the arena was allocated for.
    pub seq_capacity: u32,
    /// The `hidden_size` value the arena was allocated for.
    pub hidden_size: u32,
    /// The `n_head` value the arena was allocated for.
    pub n_head: u32,
    /// The `n_kv` value the arena was allocated for.
    pub n_kv: u32,
    /// The `head_dim` value the arena was allocated for.
    pub head_dim: u32,
}

impl FaProjectionsArena {
    /// Allocate all arena scratches sized for a single prefill pass with
    /// the given shape parameters.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Any dimension is zero.
    /// - `n_head % n_kv != 0` (GQA divisibility violated).
    /// - Any `device.alloc_buffer` call fails (out of GPU memory).
    /// - The param buffers cannot be initialized via `as_mut_slice`.
    pub fn new(
        device: &MlxDevice,
        seq_capacity: u32,
        hidden_size: u32,
        n_head: u32,
        n_kv: u32,
        head_dim: u32,
        rms_norm_eps: f32,
    ) -> Result<Self> {
        if seq_capacity == 0
            || hidden_size == 0
            || n_head == 0
            || n_kv == 0
            || head_dim == 0
        {
            return Err(anyhow!(
                "FaProjectionsArena::new: zero dim \
                 seq_capacity={} hidden_size={} n_head={} n_kv={} head_dim={}",
                seq_capacity,
                hidden_size,
                n_head,
                n_kv,
                head_dim,
            ));
        }
        if n_head % n_kv != 0 {
            return Err(anyhow!(
                "FaProjectionsArena::new: n_head ({}) must be divisible by n_kv ({})",
                n_head,
                n_kv,
            ));
        }

        let seq = seq_capacity as usize;
        let h = hidden_size as usize;
        let nh = n_head as usize;
        let nkv = n_kv as usize;
        let d = head_dim as usize;
        let q_total = nh * d;
        let kv_total = nkv * d;

        // ── Large F32 projection / helper outputs ──
        let x_norm_buf = device
            .alloc_buffer(seq * h * 4, DType::F32, vec![seq, h])
            .map_err(|e| anyhow!("FaProjectionsArena alloc x_norm_buf: {e}"))?;

        let q_proj_buf = device
            .alloc_buffer(seq * q_total * 4, DType::F32, vec![seq, q_total])
            .map_err(|e| anyhow!("FaProjectionsArena alloc q_proj_buf: {e}"))?;
        let k_proj_buf = device
            .alloc_buffer(seq * kv_total * 4, DType::F32, vec![seq, kv_total])
            .map_err(|e| anyhow!("FaProjectionsArena alloc k_proj_buf: {e}"))?;
        let v_proj_buf = device
            .alloc_buffer(seq * kv_total * 4, DType::F32, vec![seq, kv_total])
            .map_err(|e| anyhow!("FaProjectionsArena alloc v_proj_buf: {e}"))?;
        let gate_proj_buf = device
            .alloc_buffer(seq * q_total * 4, DType::F32, vec![seq, q_total])
            .map_err(|e| anyhow!("FaProjectionsArena alloc gate_proj_buf: {e}"))?;

        // Per-head RMSNorm outputs are sized [seq*n_head, head_dim] for Q
        // and [seq*n_kv, head_dim] for K — element count exactly matches
        // q_total*seq and kv_total*seq respectively.
        let q_normed_buf = device
            .alloc_buffer(seq * q_total * 4, DType::F32, vec![seq * nh, d])
            .map_err(|e| anyhow!("FaProjectionsArena alloc q_normed_buf: {e}"))?;
        let k_normed_buf = device
            .alloc_buffer(seq * kv_total * 4, DType::F32, vec![seq * nkv, d])
            .map_err(|e| anyhow!("FaProjectionsArena alloc k_normed_buf: {e}"))?;

        // IMROPE outputs share the per-head shape `[seq, n_heads, head_dim]`.
        let q_rope_buf = device
            .alloc_buffer(seq * q_total * 4, DType::F32, vec![seq, nh, d])
            .map_err(|e| anyhow!("FaProjectionsArena alloc q_rope_buf: {e}"))?;
        let k_rope_buf = device
            .alloc_buffer(seq * kv_total * 4, DType::F32, vec![seq, nkv, d])
            .map_err(|e| anyhow!("FaProjectionsArena alloc k_rope_buf: {e}"))?;

        // Sigmoid-gate multiply output is flat `[seq * q_total]` F32.
        let gated_buf = device
            .alloc_buffer(seq * q_total * 4, DType::F32, vec![seq * q_total])
            .map_err(|e| anyhow!("FaProjectionsArena alloc gated_buf: {e}"))?;

        // ── Small param buffers — written ONCE at arena construction ──
        // Pre-attn RMSNorm params: [eps, hidden_size as f32].
        let mut pre_norm_params_buf = device
            .alloc_buffer(8, DType::F32, vec![2])
            .map_err(|e| anyhow!("FaProjectionsArena alloc pre_norm_params_buf: {e}"))?;
        {
            let s = pre_norm_params_buf
                .as_mut_slice::<f32>()
                .map_err(|e| anyhow!("pre_norm_params_buf as_mut_slice: {e}"))?;
            s[0] = rms_norm_eps;
            s[1] = hidden_size as f32;
        }

        // Per-head RMSNorm params: [eps, head_dim as f32]. Shared between
        // Q and K because both norm along head_dim.
        let mut qk_rms_params_buf = device
            .alloc_buffer(8, DType::F32, vec![2])
            .map_err(|e| anyhow!("FaProjectionsArena alloc qk_rms_params_buf: {e}"))?;
        {
            let s = qk_rms_params_buf
                .as_mut_slice::<f32>()
                .map_err(|e| anyhow!("qk_rms_params_buf as_mut_slice: {e}"))?;
            s[0] = rms_norm_eps;
            s[1] = head_dim as f32;
        }

        // Sigmoid-mul element count: seq * q_total.
        let mut sigmoid_params_buf = device
            .alloc_buffer(4, DType::U32, vec![1])
            .map_err(|e| anyhow!("FaProjectionsArena alloc sigmoid_params_buf: {e}"))?;
        {
            let s = sigmoid_params_buf
                .as_mut_slice::<u32>()
                .map_err(|e| anyhow!("sigmoid_params_buf as_mut_slice: {e}"))?;
            s[0] = (seq_capacity * (n_head * head_dim)) as u32;
        }

        Ok(Self {
            x_norm_buf,
            pre_norm_params_buf,
            q_proj_buf,
            k_proj_buf,
            v_proj_buf,
            gate_proj_buf,
            q_normed_buf,
            k_normed_buf,
            qk_rms_params_buf,
            q_rope_buf,
            k_rope_buf,
            gated_buf,
            sigmoid_params_buf,
            seq_capacity,
            hidden_size,
            n_head,
            n_kv,
            head_dim,
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
        n_head: u32,
        n_kv: u32,
        head_dim: u32,
    ) -> Result<()> {
        if seq_len > self.seq_capacity {
            return Err(anyhow!(
                "FaProjectionsArena::validate_fits: seq_len {} exceeds capacity {}",
                seq_len,
                self.seq_capacity
            ));
        }
        if hidden_size != self.hidden_size
            || n_head != self.n_head
            || n_kv != self.n_kv
            || head_dim != self.head_dim
        {
            return Err(anyhow!(
                "FaProjectionsArena::validate_fits: shape mismatch — \
                 arena (h={}, n_head={}, n_kv={}, head_dim={}) vs \
                 call (h={}, n_head={}, n_kv={}, head_dim={})",
                self.hidden_size,
                self.n_head,
                self.n_kv,
                self.head_dim,
                hidden_size,
                n_head,
                n_kv,
                head_dim,
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

    /// Qwen3.6-35B-A3B canonical FA shape at pp=4127:
    /// h=2048, n_head=16, n_kv=2, head_dim=256, eps=1e-6.
    /// Verifies all fields' byte_len() match the formula and capacity is
    /// recorded correctly. Param buffers are populated with the expected
    /// constants.
    #[test]
    fn test_fa_proj_arena_new_qwen36_pp4127() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_fa_proj_arena_new_qwen36_pp4127: skipping — no Metal device");
                return;
            }
        };
        let (seq, h, nh, nkv, d, eps) = (4127u32, 2048u32, 16u32, 2u32, 256u32, 1e-6f32);
        let arena = FaProjectionsArena::new(&device, seq, h, nh, nkv, d, eps)
            .expect("fa proj arena new pp4127");

        assert_eq!(arena.seq_capacity, seq);
        assert_eq!(arena.hidden_size, h);
        assert_eq!(arena.n_head, nh);
        assert_eq!(arena.n_kv, nkv);
        assert_eq!(arena.head_dim, d);

        let q_bytes = (seq as usize) * (nh as usize) * (d as usize) * 4;
        let kv_bytes = (seq as usize) * (nkv as usize) * (d as usize) * 4;
        let h_bytes = (seq as usize) * (h as usize) * 4;

        assert_eq!(arena.x_norm_buf.byte_len(), h_bytes, "x_norm_buf");
        assert_eq!(arena.q_proj_buf.byte_len(), q_bytes, "q_proj_buf");
        assert_eq!(arena.k_proj_buf.byte_len(), kv_bytes, "k_proj_buf");
        assert_eq!(arena.v_proj_buf.byte_len(), kv_bytes, "v_proj_buf");
        assert_eq!(arena.gate_proj_buf.byte_len(), q_bytes, "gate_proj_buf");
        assert_eq!(arena.q_normed_buf.byte_len(), q_bytes, "q_normed_buf");
        assert_eq!(arena.k_normed_buf.byte_len(), kv_bytes, "k_normed_buf");
        assert_eq!(arena.q_rope_buf.byte_len(), q_bytes, "q_rope_buf");
        assert_eq!(arena.k_rope_buf.byte_len(), kv_bytes, "k_rope_buf");
        assert_eq!(arena.gated_buf.byte_len(), q_bytes, "gated_buf");

        assert_eq!(arena.pre_norm_params_buf.byte_len(), 8, "pre_norm_params");
        assert_eq!(arena.qk_rms_params_buf.byte_len(), 8, "qk_rms_params");
        assert_eq!(arena.sigmoid_params_buf.byte_len(), 4, "sigmoid_params");

        // Verify pre-populated param values.
        let pn = arena
            .pre_norm_params_buf
            .as_slice::<f32>()
            .expect("pre_norm_params_buf as_slice");
        assert_eq!(pn[0], eps, "pre_norm_params[0] = eps");
        assert_eq!(pn[1], h as f32, "pre_norm_params[1] = hidden_size");

        let qk = arena
            .qk_rms_params_buf
            .as_slice::<f32>()
            .expect("qk_rms_params_buf as_slice");
        assert_eq!(qk[0], eps, "qk_rms_params[0] = eps");
        assert_eq!(qk[1], d as f32, "qk_rms_params[1] = head_dim");

        let sg = arena
            .sigmoid_params_buf
            .as_slice::<u32>()
            .expect("sigmoid_params_buf as_slice");
        assert_eq!(sg[0], seq * nh * d, "sigmoid_params[0] = seq*n_head*head_dim");
    }

    /// Smaller shape sanity-check.
    #[test]
    fn test_fa_proj_arena_new_small_shape() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_fa_proj_arena_new_small_shape: skipping — no Metal device");
                return;
            }
        };
        let arena = FaProjectionsArena::new(&device, 64, 128, 4, 2, 32, 1e-5)
            .expect("fa proj arena small");
        assert_eq!(arena.seq_capacity, 64);
    }

    /// Zero-dim rejection.
    #[test]
    fn test_fa_proj_arena_new_zero_dim_rejected() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_fa_proj_arena_new_zero_dim_rejected: skipping — no Metal device");
                return;
            }
        };
        assert!(FaProjectionsArena::new(&device, 0, 128, 4, 2, 32, 1e-5).is_err());
        assert!(FaProjectionsArena::new(&device, 64, 0, 4, 2, 32, 1e-5).is_err());
        assert!(FaProjectionsArena::new(&device, 64, 128, 0, 2, 32, 1e-5).is_err());
        assert!(FaProjectionsArena::new(&device, 64, 128, 4, 0, 32, 1e-5).is_err());
        assert!(FaProjectionsArena::new(&device, 64, 128, 4, 2, 0, 1e-5).is_err());
    }

    /// GQA divisibility: n_head=15 with n_kv=2 must be rejected.
    #[test]
    fn test_fa_proj_arena_new_gqa_divisibility_rejected() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_fa_proj_arena_new_gqa_divisibility_rejected: skipping — no Metal device"
                );
                return;
            }
        };
        assert!(FaProjectionsArena::new(&device, 64, 128, 15, 2, 32, 1e-5).is_err());
    }

    /// validate_fits exact match returns Ok.
    #[test]
    fn test_fa_proj_validate_fits_exact_match() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_fa_proj_validate_fits_exact_match: skipping — no Metal device");
                return;
            }
        };
        let arena =
            FaProjectionsArena::new(&device, 128, 256, 4, 2, 32, 1e-5).expect("arena");
        assert!(arena.validate_fits(128, 256, 4, 2, 32).is_ok());
        assert!(arena.validate_fits(64, 256, 4, 2, 32).is_ok());
    }

    /// validate_fits seq overrun returns Err.
    #[test]
    fn test_fa_proj_validate_fits_seq_overrun() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_fa_proj_validate_fits_seq_overrun: skipping — no Metal device");
                return;
            }
        };
        let arena =
            FaProjectionsArena::new(&device, 128, 256, 4, 2, 32, 1e-5).expect("arena");
        assert!(arena.validate_fits(256, 256, 4, 2, 32).is_err());
    }

    /// validate_fits shape mismatch returns Err for each axis.
    #[test]
    fn test_fa_proj_validate_fits_shape_mismatch() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_fa_proj_validate_fits_shape_mismatch: skipping — no Metal device"
                );
                return;
            }
        };
        let arena =
            FaProjectionsArena::new(&device, 128, 256, 4, 2, 32, 1e-5).expect("arena");
        assert!(arena.validate_fits(128, 128, 4, 2, 32).is_err()); // h
        assert!(arena.validate_fits(128, 256, 8, 2, 32).is_err()); // nh
        assert!(arena.validate_fits(128, 256, 4, 1, 32).is_err()); // nkv
        assert!(arena.validate_fits(128, 256, 4, 2, 16).is_err()); // d
    }

    /// device.alloc_buffer zero-initializes all F32 buffers (iter61a guarantee).
    #[test]
    fn test_fa_proj_arena_buffers_zero_initialized() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!(
                    "test_fa_proj_arena_buffers_zero_initialized: skipping — no Metal device"
                );
                return;
            }
        };
        let arena =
            FaProjectionsArena::new(&device, 64, 128, 4, 2, 32, 1e-5).expect("arena");

        let f32_outs: [(&MlxBuffer, &str); 10] = [
            (&arena.x_norm_buf, "x_norm_buf"),
            (&arena.q_proj_buf, "q_proj_buf"),
            (&arena.k_proj_buf, "k_proj_buf"),
            (&arena.v_proj_buf, "v_proj_buf"),
            (&arena.gate_proj_buf, "gate_proj_buf"),
            (&arena.q_normed_buf, "q_normed_buf"),
            (&arena.k_normed_buf, "k_normed_buf"),
            (&arena.q_rope_buf, "q_rope_buf"),
            (&arena.k_rope_buf, "k_rope_buf"),
            (&arena.gated_buf, "gated_buf"),
        ];
        for (buf, name) in &f32_outs {
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
    }
}
