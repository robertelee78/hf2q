//! Caller-owned scratch arena for the Gated DeltaNet (DN) prefill path
//! across all DN layers in a single prefill (ADR-015 iter74).
//!
//! Mirrors [`super::DenseFfnArena`] / [`super::MoeFfnArena`] (iter72) and
//! [`super::FaPrefillArena`] (ADR-013 P21 stage-1) for the DeltaNet wrapper
//! body in [`super::gpu_delta_net::build_delta_net_layer`].
//!
//! # Why
//!
//! `wave5b8_profile` measurements at apex pp4096 (qwen3.6-35b-a3b q4_0-flat,
//! 30 DN layers + 10 FA layers) showed three DN-side encoder Sections with
//! ffn.alloc_scratch-shaped quantile skew:
//!
//! | Bucket | n | sum_ms | mean | p50 | max | mean/p50 | max/p50 |
//! |---|---:|---:|---:|---:|---:|---:|---:|
//! | layer.ops1_3 | 30 | 112.7 | 3.76 | 0.022 | 19.6 | 170× | 890× |
//! | layer.qkv_deinterleave | 30 | 62.5 | 2.08 | 0.011 | 10.9 | 189× | 994× |
//! | layer.autoreg_ops5_9 | 30 | 48.3 | 1.61 | 0.019 | 8.6 | 85× | 453× |
//! | **Combined** | | **223.5** | | | | | |
//!
//! Total ≈ 13 % of post-iter72 prefill wall (1700 ms). p50 ≈ 0.020 ms (flat,
//! sub-µs encoder build) but mean ≈ 1.6–3.8 ms (single big cold path per
//! layer). Bit-for-bit the same `metal::new_buffer + zero-init + addAllocation
//! + set.commit()` cold-path mechanism that iter72 fixed for
//! `ffn.alloc_scratch`.
//!
//! # Mechanism mirror to iter72
//!
//! The DN per-layer body allocates ~22 transient buffers from the thread-local
//! `decode_pool`:
//!
//! - 9 outer-scope slots in [`super::gpu_delta_net::build_delta_net_layer`]
//!   between the entry and the first sub-section: `qkv_conv`, `ssm_params_buf`,
//!   `q_scaled`, `g_buf`, `beta_buf`, `g_params_buf`, `gated_buf`,
//!   `op8_params`, `attn_out_buf`, `gdn_params_buf` (10 total — the verdict's
//!   "9" undercounts by one).
//! - 3 in the `qkv_split` block (`q_gpu`, `k_gpu`, `v_gpu`).
//! - ~10 in the helpers `apply_pre_norm`, `apply_proj` (called 4× from the
//!   prefill autoregressive path for `qkv_raw`, `z`, `alpha_logit`,
//!   `beta_logit`), and `apply_l2_norm_per_head` (called 2× for `q_l2`,
//!   `k_normed`).
//!
//! All 23 transient buffers are written + read within ONE DN layer's encoders
//! and never read by the next layer (the only inter-layer values are the
//! conv_state ping-pong and the recurrent state ping-pong slots, both
//! caller-owned in `kv_cache.linear_attn[]`, NOT arena).
//!
//! Lifting them to caller scope keeps them alive for the full prefill,
//! eliminating the per-layer alloc churn. Allocated ONCE at the top of
//! `forward_gpu_impl`, reused across all DN layers, dropped at the end of
//! `forward_gpu_impl` AFTER the final `commit_and_wait_labeled` at the
//! output head — same iter58b residency-rescission protection contract as
//! [`super::FaPrefillArena`] / [`super::DenseFfnArena`] / [`super::MoeFfnArena`].
//!
//! # NOT included in the arena
//!
//! The **final out_proj output** produced by the last `apply_proj` call
//! inside `build_delta_net_layer` is intentionally **NOT** in the arena —
//! its ARC clone leaves the function via `Ok(output)` and becomes the next
//! layer's `hidden`, crossing layer boundaries. A pooled output here would
//! alias the next layer's `qkv_raw_buf` when both happen to land in the same
//! arena slot, corrupting the residual stream silently. The current
//! `build_delta_net_layer` allocates this through `apply_proj`'s internal
//! `pooled_alloc_buffer`; we keep that path verbatim.
//!
//! # Memory footprint (35B-A3B q4_0-flat pp4096)
//!
//! Real shape (resolved at runtime from the GGUF config): typical
//! `seq=4123`, `hidden=2048`, `n_k_heads=16`, `n_v_heads=32`, `d_k=128`,
//! `d_v=128`, so `qkv_channels = 2*16*128 + 32*128 = 8192`,
//! `z_channels = n_v_heads * d_v = 4096`. (Indicative; the arena sizes
//! itself from runtime values.)
//!
//! Ballpark: ≈ 0.3–0.5 GB. Live for the entire prefill duration; M5 Max
//! 128 GB unified memory keeps this well within budget.

use anyhow::{anyhow, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

/// Caller-owned scratch arena for the Gated DeltaNet prefill path across all
/// DN layers in a single prefill.
///
/// Contains the 23 transient F32/U32 scratches that
/// [`super::gpu_delta_net::build_delta_net_layer`]'s prefill branch
/// (`seq > 1`) currently allocates per-layer via
/// [`super::decode_pool::pooled_alloc_buffer`]. Lifting them to caller scope
/// keeps them alive for the full prefill, eliminating the per-layer alloc
/// churn captured by the W-5b.8 `layer.ops1_3` / `layer.qkv_deinterleave` /
/// `layer.autoreg_ops5_9` skew buckets.
///
/// All scratches are sized for the actual prefill `(seq_len, …)` shape;
/// per-layer `validate_fits` guards against accidental shape drift on a
/// future model that mixes DN layers of different shapes (Qwen3.6 35B-A3B is
/// uniform — every DN layer has the same h, n_k_heads, n_v_heads, d_k, d_v).
pub struct DnPrefillArena {
    // ── Outer-scope `build_delta_net_layer` body slots ─────────────────────
    /// `[seq, qkv_channels]` F32 — ssm_conv output (de-interleaved later).
    pub qkv_conv_buf: MlxBuffer,
    /// `[4]` U32 — ssm_conv params.
    pub ssm_params_buf: MlxBuffer,
    /// `[seq * n_k_heads * d_k]` F32 — scaled Q for GDN kernel.
    pub q_scaled_buf: MlxBuffer,
    /// `[seq * n_v_heads]` F32 — softplus(alpha_logit + dt_bias) * (-ssm_a).
    pub g_buf: MlxBuffer,
    /// `[seq * n_v_heads]` F32 — sigmoid(beta_logit).
    pub beta_buf: MlxBuffer,
    /// `[2]` U32 — compute_g_beta params (n_v_heads, seq_len).
    pub g_params_buf: MlxBuffer,
    /// `[seq * n_v_heads * d_v]` F32 — ssm_norm_gate output.
    pub gated_buf: MlxBuffer,
    /// `[2]` F32 — ssm_norm_gate params (rms_eps, d_v as f32).
    pub op8_params_buf: MlxBuffer,
    /// `[n_v_heads * seq * d_v]` F32 — GDN attention output.
    pub attn_out_buf: MlxBuffer,
    /// `[8]` U32 — gated_delta_net params.
    pub gdn_params_buf: MlxBuffer,

    // ── qkv_split outputs ──────────────────────────────────────────────────
    /// `[seq, q_sp]` F32 — Q after qkv_split.
    pub q_split_buf: MlxBuffer,
    /// `[seq, k_sp]` F32 — K after qkv_split.
    pub k_split_buf: MlxBuffer,
    /// `[seq, v_sp]` F32 — V after qkv_split (`v_sp = n_v_heads * d_v`).
    pub v_split_buf: MlxBuffer,

    // ── apply_pre_norm slots ───────────────────────────────────────────────
    /// `[seq, hidden]` F32 — output of the pre-attention RMSNorm.
    pub x_norm_buf: MlxBuffer,
    /// `[2]` F32 — pre_norm params (eps, hidden_size as f32).
    pub pre_norm_params_buf: MlxBuffer,

    // ── apply_proj slots (qkv_raw, z, alpha_logit, beta_logit) ─────────────
    /// `[seq, qkv_channels]` F32 — qkv_proj output.
    pub qkv_raw_buf: MlxBuffer,
    /// `[seq, z_channels]` F32 — z_proj output (`z_channels = n_v_heads * d_v`).
    pub z_buf: MlxBuffer,
    /// `[seq, n_v_heads]` F32 — alpha_proj output.
    pub alpha_logit_buf: MlxBuffer,
    /// `[seq, n_v_heads]` F32 — beta_proj output.
    pub beta_logit_buf: MlxBuffer,

    // ── apply_l2_norm_per_head slots ───────────────────────────────────────
    /// `[seq * n_k_heads, d_k]` F32 — L2-normalised Q (pre q_scale).
    pub q_l2_buf: MlxBuffer,
    /// `[seq * n_k_heads, d_k]` F32 — L2-normalised K.
    pub k_normed_buf: MlxBuffer,
    /// `[2]` F32 — l2_norm params for Q (eps, d_k as f32).
    pub l2_params_q_buf: MlxBuffer,
    /// `[2]` F32 — l2_norm params for K (eps, d_k as f32).
    pub l2_params_k_buf: MlxBuffer,

    // ── Capacity bookkeeping ───────────────────────────────────────────────
    /// The `seq_capacity` value the arena was allocated for.
    pub seq_capacity: u32,
    /// The `hidden_size` the arena was allocated for.
    pub hidden_size: u32,
    /// The `n_k_heads` the arena was allocated for.
    pub n_k_heads: u32,
    /// The `n_v_heads` the arena was allocated for.
    pub n_v_heads: u32,
    /// The `d_k` the arena was allocated for.
    pub d_k: u32,
    /// The `d_v` the arena was allocated for.
    pub d_v: u32,
}

impl DnPrefillArena {
    /// Allocate all transient scratches sized for a single prefill pass with
    /// the given shape parameters.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any dimension is zero or any
    /// [`MlxDevice::alloc_buffer`] call fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &MlxDevice,
        seq_capacity: u32,
        hidden_size: u32,
        n_k_heads: u32,
        n_v_heads: u32,
        d_k: u32,
        d_v: u32,
    ) -> Result<Self> {
        if seq_capacity == 0
            || hidden_size == 0
            || n_k_heads == 0
            || n_v_heads == 0
            || d_k == 0
            || d_v == 0
        {
            return Err(anyhow!(
                "DnPrefillArena::new: zero dim \
                 seq_capacity={} hidden_size={} n_k_heads={} n_v_heads={} d_k={} d_v={}",
                seq_capacity,
                hidden_size,
                n_k_heads,
                n_v_heads,
                d_k,
                d_v,
            ));
        }

        let seq = seq_capacity as usize;
        let h = hidden_size as usize;
        let nk = n_k_heads as usize;
        let nv = n_v_heads as usize;
        let dk = d_k as usize;
        let dv = d_v as usize;
        let qkv_channels = 2 * nk * dk + nv * dv;
        let z_channels = nv * dv;
        let q_sp = nk * dk;
        let k_sp = nk * dk;
        let v_sp = nv * dv;
        let n_q_elems = seq * nk * dk;
        let g_n = seq * nv;
        let gated_elems = seq * nv * dv;
        let attn_out_elems = nv * seq * dv;

        // Outer-scope slots
        let qkv_conv_buf = device
            .alloc_buffer(seq * qkv_channels * 4, DType::F32, vec![seq, qkv_channels])
            .map_err(|e| anyhow!("DnPrefillArena alloc qkv_conv_buf: {e}"))?;
        let ssm_params_buf = device
            .alloc_buffer(4 * 4, DType::U32, vec![4])
            .map_err(|e| anyhow!("DnPrefillArena alloc ssm_params_buf: {e}"))?;
        let q_scaled_buf = device
            .alloc_buffer(n_q_elems * 4, DType::F32, vec![n_q_elems])
            .map_err(|e| anyhow!("DnPrefillArena alloc q_scaled_buf: {e}"))?;
        let g_buf = device
            .alloc_buffer(g_n * 4, DType::F32, vec![g_n])
            .map_err(|e| anyhow!("DnPrefillArena alloc g_buf: {e}"))?;
        let beta_buf = device
            .alloc_buffer(g_n * 4, DType::F32, vec![g_n])
            .map_err(|e| anyhow!("DnPrefillArena alloc beta_buf: {e}"))?;
        let g_params_buf = device
            .alloc_buffer(8, DType::U32, vec![2])
            .map_err(|e| anyhow!("DnPrefillArena alloc g_params_buf: {e}"))?;
        let gated_buf = device
            .alloc_buffer(gated_elems * 4, DType::F32, vec![gated_elems])
            .map_err(|e| anyhow!("DnPrefillArena alloc gated_buf: {e}"))?;
        let op8_params_buf = device
            .alloc_buffer(8, DType::F32, vec![2])
            .map_err(|e| anyhow!("DnPrefillArena alloc op8_params_buf: {e}"))?;
        let attn_out_buf = device
            .alloc_buffer(attn_out_elems * 4, DType::F32, vec![attn_out_elems])
            .map_err(|e| anyhow!("DnPrefillArena alloc attn_out_buf: {e}"))?;
        let gdn_params_buf = device
            .alloc_buffer(8 * 4, DType::U32, vec![8])
            .map_err(|e| anyhow!("DnPrefillArena alloc gdn_params_buf: {e}"))?;

        // qkv_split slots
        let q_split_buf = device
            .alloc_buffer(seq * q_sp * 4, DType::F32, vec![seq, q_sp])
            .map_err(|e| anyhow!("DnPrefillArena alloc q_split_buf: {e}"))?;
        let k_split_buf = device
            .alloc_buffer(seq * k_sp * 4, DType::F32, vec![seq, k_sp])
            .map_err(|e| anyhow!("DnPrefillArena alloc k_split_buf: {e}"))?;
        let v_split_buf = device
            .alloc_buffer(seq * v_sp * 4, DType::F32, vec![seq, v_sp])
            .map_err(|e| anyhow!("DnPrefillArena alloc v_split_buf: {e}"))?;

        // apply_pre_norm slots
        let x_norm_buf = device
            .alloc_buffer(seq * h * 4, DType::F32, vec![seq, h])
            .map_err(|e| anyhow!("DnPrefillArena alloc x_norm_buf: {e}"))?;
        let pre_norm_params_buf = device
            .alloc_buffer(8, DType::F32, vec![2])
            .map_err(|e| anyhow!("DnPrefillArena alloc pre_norm_params_buf: {e}"))?;

        // apply_proj slots (qkv_raw, z, alpha_logit, beta_logit)
        let qkv_raw_buf = device
            .alloc_buffer(seq * qkv_channels * 4, DType::F32, vec![seq, qkv_channels])
            .map_err(|e| anyhow!("DnPrefillArena alloc qkv_raw_buf: {e}"))?;
        let z_buf = device
            .alloc_buffer(seq * z_channels * 4, DType::F32, vec![seq, z_channels])
            .map_err(|e| anyhow!("DnPrefillArena alloc z_buf: {e}"))?;
        let alpha_logit_buf = device
            .alloc_buffer(seq * nv * 4, DType::F32, vec![seq, nv])
            .map_err(|e| anyhow!("DnPrefillArena alloc alpha_logit_buf: {e}"))?;
        let beta_logit_buf = device
            .alloc_buffer(seq * nv * 4, DType::F32, vec![seq, nv])
            .map_err(|e| anyhow!("DnPrefillArena alloc beta_logit_buf: {e}"))?;

        // apply_l2_norm_per_head slots
        let q_l2_rows = seq * nk;
        let q_l2_buf = device
            .alloc_buffer(q_l2_rows * dk * 4, DType::F32, vec![q_l2_rows, dk])
            .map_err(|e| anyhow!("DnPrefillArena alloc q_l2_buf: {e}"))?;
        let k_normed_buf = device
            .alloc_buffer(q_l2_rows * dk * 4, DType::F32, vec![q_l2_rows, dk])
            .map_err(|e| anyhow!("DnPrefillArena alloc k_normed_buf: {e}"))?;
        let l2_params_q_buf = device
            .alloc_buffer(8, DType::F32, vec![2])
            .map_err(|e| anyhow!("DnPrefillArena alloc l2_params_q_buf: {e}"))?;
        let l2_params_k_buf = device
            .alloc_buffer(8, DType::F32, vec![2])
            .map_err(|e| anyhow!("DnPrefillArena alloc l2_params_k_buf: {e}"))?;

        Ok(Self {
            qkv_conv_buf,
            ssm_params_buf,
            q_scaled_buf,
            g_buf,
            beta_buf,
            g_params_buf,
            gated_buf,
            op8_params_buf,
            attn_out_buf,
            gdn_params_buf,
            q_split_buf,
            k_split_buf,
            v_split_buf,
            x_norm_buf,
            pre_norm_params_buf,
            qkv_raw_buf,
            z_buf,
            alpha_logit_buf,
            beta_logit_buf,
            q_l2_buf,
            k_normed_buf,
            l2_params_q_buf,
            l2_params_k_buf,
            seq_capacity,
            hidden_size,
            n_k_heads,
            n_v_heads,
            d_k,
            d_v,
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
        n_k_heads: u32,
        n_v_heads: u32,
        d_k: u32,
        d_v: u32,
    ) -> Result<()> {
        if seq_len > self.seq_capacity {
            return Err(anyhow!(
                "DnPrefillArena::validate_fits: seq_len {} exceeds capacity {}",
                seq_len,
                self.seq_capacity
            ));
        }
        if hidden_size != self.hidden_size
            || n_k_heads != self.n_k_heads
            || n_v_heads != self.n_v_heads
            || d_k != self.d_k
            || d_v != self.d_v
        {
            return Err(anyhow!(
                "DnPrefillArena::validate_fits: shape mismatch — \
                 arena (h={}, nk={}, nv={}, dk={}, dv={}) vs \
                 call (h={}, nk={}, nv={}, dk={}, dv={})",
                self.hidden_size,
                self.n_k_heads,
                self.n_v_heads,
                self.d_k,
                self.d_v,
                hidden_size,
                n_k_heads,
                n_v_heads,
                d_k,
                d_v,
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

    /// Qwen3.6 35B-A3B canonical DN shape at pp=4096:
    /// h=2048, n_k_heads=16, n_v_heads=32, d_k=128, d_v=128.
    /// (Shape values consistent with the apex DWQ q4_0-flat fixture.)
    #[test]
    fn test_dn_arena_new_qwen36_35b_pp4096() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_dn_arena_new_qwen36_35b_pp4096: skipping — no Metal device");
                return;
            }
        };
        let (seq, h, nk, nv, dk, dv) = (4096u32, 2048u32, 16u32, 32u32, 128u32, 128u32);
        let arena = DnPrefillArena::new(&device, seq, h, nk, nv, dk, dv).expect("dn arena pp4096");

        assert_eq!(arena.seq_capacity, seq);
        assert_eq!(arena.hidden_size, h);
        assert_eq!(arena.n_k_heads, nk);
        assert_eq!(arena.n_v_heads, nv);
        assert_eq!(arena.d_k, dk);
        assert_eq!(arena.d_v, dv);

        let qkv_channels = (2 * nk * dk + nv * dv) as usize;
        let z_channels = (nv * dv) as usize;

        // Outer-scope expected sizes
        assert_eq!(
            arena.qkv_conv_buf.byte_len(),
            (seq as usize) * qkv_channels * 4,
            "qkv_conv_buf"
        );
        assert_eq!(arena.ssm_params_buf.byte_len(), 16, "ssm_params_buf");
        assert_eq!(
            arena.q_scaled_buf.byte_len(),
            (seq as usize) * (nk as usize) * (dk as usize) * 4,
            "q_scaled_buf"
        );
        assert_eq!(
            arena.g_buf.byte_len(),
            (seq as usize) * (nv as usize) * 4,
            "g_buf"
        );
        assert_eq!(
            arena.beta_buf.byte_len(),
            (seq as usize) * (nv as usize) * 4,
            "beta_buf"
        );
        assert_eq!(arena.g_params_buf.byte_len(), 8, "g_params_buf");
        assert_eq!(
            arena.gated_buf.byte_len(),
            (seq as usize) * (nv as usize) * (dv as usize) * 4,
            "gated_buf"
        );
        assert_eq!(arena.op8_params_buf.byte_len(), 8, "op8_params_buf");
        assert_eq!(
            arena.attn_out_buf.byte_len(),
            (nv as usize) * (seq as usize) * (dv as usize) * 4,
            "attn_out_buf"
        );
        assert_eq!(arena.gdn_params_buf.byte_len(), 32, "gdn_params_buf");

        // qkv_split expected sizes
        let q_sp = (nk * dk) as usize;
        let k_sp = (nk * dk) as usize;
        let v_sp = (nv * dv) as usize;
        assert_eq!(
            arena.q_split_buf.byte_len(),
            (seq as usize) * q_sp * 4,
            "q_split_buf"
        );
        assert_eq!(
            arena.k_split_buf.byte_len(),
            (seq as usize) * k_sp * 4,
            "k_split_buf"
        );
        assert_eq!(
            arena.v_split_buf.byte_len(),
            (seq as usize) * v_sp * 4,
            "v_split_buf"
        );

        // apply_pre_norm expected sizes
        assert_eq!(
            arena.x_norm_buf.byte_len(),
            (seq as usize) * (h as usize) * 4,
            "x_norm_buf"
        );
        assert_eq!(arena.pre_norm_params_buf.byte_len(), 8, "pre_norm_params_buf");

        // apply_proj expected sizes
        assert_eq!(
            arena.qkv_raw_buf.byte_len(),
            (seq as usize) * qkv_channels * 4,
            "qkv_raw_buf"
        );
        assert_eq!(
            arena.z_buf.byte_len(),
            (seq as usize) * z_channels * 4,
            "z_buf"
        );
        assert_eq!(
            arena.alpha_logit_buf.byte_len(),
            (seq as usize) * (nv as usize) * 4,
            "alpha_logit_buf"
        );
        assert_eq!(
            arena.beta_logit_buf.byte_len(),
            (seq as usize) * (nv as usize) * 4,
            "beta_logit_buf"
        );

        // apply_l2_norm_per_head expected sizes
        let q_l2_rows = (seq as usize) * (nk as usize);
        assert_eq!(
            arena.q_l2_buf.byte_len(),
            q_l2_rows * (dk as usize) * 4,
            "q_l2_buf"
        );
        assert_eq!(
            arena.k_normed_buf.byte_len(),
            q_l2_rows * (dk as usize) * 4,
            "k_normed_buf"
        );
        assert_eq!(arena.l2_params_q_buf.byte_len(), 8, "l2_params_q_buf");
        assert_eq!(arena.l2_params_k_buf.byte_len(), 8, "l2_params_k_buf");
    }

    /// Smaller shape sanity-check.
    #[test]
    fn test_dn_arena_new_small_shape() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_dn_arena_new_small_shape: skipping — no Metal device");
                return;
            }
        };
        let arena = DnPrefillArena::new(&device, 64, 128, 4, 8, 16, 16).expect("dn arena small");
        assert_eq!(arena.seq_capacity, 64);
        assert_eq!(arena.hidden_size, 128);
        assert_eq!(arena.n_k_heads, 4);
        assert_eq!(arena.n_v_heads, 8);
        assert_eq!(arena.d_k, 16);
        assert_eq!(arena.d_v, 16);
    }

    /// Zero-dim rejection (each parameter individually).
    #[test]
    fn test_dn_arena_new_zero_dim_rejected() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_dn_arena_new_zero_dim_rejected: skipping — no Metal device");
                return;
            }
        };
        assert!(DnPrefillArena::new(&device, 0, 128, 4, 8, 16, 16).is_err());
        assert!(DnPrefillArena::new(&device, 64, 0, 4, 8, 16, 16).is_err());
        assert!(DnPrefillArena::new(&device, 64, 128, 0, 8, 16, 16).is_err());
        assert!(DnPrefillArena::new(&device, 64, 128, 4, 0, 16, 16).is_err());
        assert!(DnPrefillArena::new(&device, 64, 128, 4, 8, 0, 16).is_err());
        assert!(DnPrefillArena::new(&device, 64, 128, 4, 8, 16, 0).is_err());
    }

    /// validate_fits exact match returns Ok.
    #[test]
    fn test_dn_validate_fits_exact_match() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_dn_validate_fits_exact_match: skipping — no Metal device");
                return;
            }
        };
        let arena = DnPrefillArena::new(&device, 128, 256, 4, 8, 32, 32).expect("dn arena new");
        assert!(arena.validate_fits(128, 256, 4, 8, 32, 32).is_ok());
        // seq_len < capacity also Ok.
        assert!(arena.validate_fits(64, 256, 4, 8, 32, 32).is_ok());
    }

    /// validate_fits seq overrun returns Err.
    #[test]
    fn test_dn_validate_fits_seq_overrun() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_dn_validate_fits_seq_overrun: skipping — no Metal device");
                return;
            }
        };
        let arena = DnPrefillArena::new(&device, 128, 256, 4, 8, 32, 32).expect("dn arena new");
        assert!(arena.validate_fits(256, 256, 4, 8, 32, 32).is_err());
    }

    /// validate_fits shape mismatch returns Err for each axis individually.
    #[test]
    fn test_dn_validate_fits_shape_mismatch() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_dn_validate_fits_shape_mismatch: skipping — no Metal device");
                return;
            }
        };
        let arena = DnPrefillArena::new(&device, 128, 256, 4, 8, 32, 32).expect("dn arena new");
        assert!(arena.validate_fits(128, 128, 4, 8, 32, 32).is_err()); // h
        assert!(arena.validate_fits(128, 256, 8, 8, 32, 32).is_err()); // nk
        assert!(arena.validate_fits(128, 256, 4, 4, 32, 32).is_err()); // nv
        assert!(arena.validate_fits(128, 256, 4, 8, 16, 32).is_err()); // dk
        assert!(arena.validate_fits(128, 256, 4, 8, 32, 16).is_err()); // dv
    }

    /// Verifies that device.alloc_buffer zero-initialises all F32/U32
    /// buffers — same iter61a guarantee FaPrefillArena/DenseFfnArena rely on.
    #[test]
    fn test_dn_arena_buffers_zero_initialised() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_dn_arena_buffers_zero_initialised: skipping — no Metal device");
                return;
            }
        };
        let arena = DnPrefillArena::new(&device, 64, 128, 4, 8, 16, 16).expect("dn arena new");

        // F32 buffers
        let f32_bufs: [(&MlxBuffer, &str); 14] = [
            (&arena.qkv_conv_buf, "qkv_conv_buf"),
            (&arena.q_scaled_buf, "q_scaled_buf"),
            (&arena.g_buf, "g_buf"),
            (&arena.beta_buf, "beta_buf"),
            (&arena.gated_buf, "gated_buf"),
            (&arena.attn_out_buf, "attn_out_buf"),
            (&arena.q_split_buf, "q_split_buf"),
            (&arena.k_split_buf, "k_split_buf"),
            (&arena.v_split_buf, "v_split_buf"),
            (&arena.x_norm_buf, "x_norm_buf"),
            (&arena.qkv_raw_buf, "qkv_raw_buf"),
            (&arena.z_buf, "z_buf"),
            (&arena.alpha_logit_buf, "alpha_logit_buf"),
            (&arena.beta_logit_buf, "beta_logit_buf"),
        ];
        for (buf, name) in &f32_bufs {
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

        let f32_params: [(&MlxBuffer, &str); 4] = [
            (&arena.op8_params_buf, "op8_params_buf"),
            (&arena.pre_norm_params_buf, "pre_norm_params_buf"),
            (&arena.l2_params_q_buf, "l2_params_q_buf"),
            (&arena.l2_params_k_buf, "l2_params_k_buf"),
        ];
        for (buf, name) in &f32_params {
            let slice = buf
                .as_slice::<f32>()
                .unwrap_or_else(|e| panic!("{name} as_slice::<f32> failed: {e}"));
            assert_eq!(slice[0], 0.0f32, "{name}[0] should be zero");
        }

        // U32 params
        let u32_params: [(&MlxBuffer, &str); 3] = [
            (&arena.ssm_params_buf, "ssm_params_buf"),
            (&arena.g_params_buf, "g_params_buf"),
            (&arena.gdn_params_buf, "gdn_params_buf"),
        ];
        for (buf, name) in &u32_params {
            let slice = buf
                .as_slice::<u32>()
                .unwrap_or_else(|e| panic!("{name} as_slice::<u32> failed: {e}"));
            assert_eq!(slice[0], 0u32, "{name}[0] should be zero");
        }
    }

    /// Q/K splits use the same row count and head_dim — make sure we keep
    /// shape parity with the qkv_split kernel's contract (q_sp = k_sp).
    #[test]
    fn test_dn_arena_q_k_split_buf_same_size() {
        let device = match device_or_skip() {
            Some(d) => d,
            None => {
                eprintln!("test_dn_arena_q_k_split_buf_same_size: skipping — no Metal device");
                return;
            }
        };
        let arena = DnPrefillArena::new(&device, 128, 256, 4, 8, 32, 32).expect("dn arena new");
        assert_eq!(
            arena.q_split_buf.byte_len(),
            arena.k_split_buf.byte_len(),
            "q_split and k_split must be the same size (q_sp == k_sp)"
        );
    }
}
