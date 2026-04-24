//! Qwen3.5 Gated DeltaNet linear-attention layer — scalar CPU reference
//! (ADR-013 Decision 8).
//!
//! Orchestrates the linear-attention layer forward pass:
//!
//! ```text
//!  1. x_norm = RMSNorm(x, attn_norm_w, eps)
//!  2. QKV = attn_qkv @ x_norm         // concatenated [n_k*D_k + n_k*D_k + n_v*D_v]
//!  3. Z   = attn_gate @ x_norm        // output gate  [n_v*D_v]
//!  4. QKV = ssm_conv1d(QKV, conv_state, SiLU)  // per-channel causal conv width 4
//!  5. Split QKV -> Q[seq,n_k,D_k], K[seq,n_k,D_k], V[seq,n_v,D_v]
//!  6. Q = L2Norm(Q over D_k)                     (ADR Decision 3)
//!     K = L2Norm(K over D_k)
//!  7. alpha_logit = ssm_alpha @ x_norm + ssm_dt_bias     // [seq, n_v]
//!     g = softplus(alpha_logit) * exp(ssm_a)             // ssm_a is log-decay base
//!     (alpha = exp(-g) is applied inside GATED_DELTA_NET)
//!     beta_logit  = ssm_beta  @ x_norm                    // [seq, n_v]
//!     beta = sigmoid(beta_logit)
//!  8. (output, state') = GATED_DELTA_NET(Q, K, V, g, beta, state)  (Decision 6)
//!  9. output = ssm_norm(output over D_v) * sigmoid(Z)   // output gate
//! 10. residual_contribution = ssm_out @ output          // [seq, hidden]
//! ```
//!
//! The actual DeltaNet recurrence step (8) reuses mlx-native's
//! `gated_delta_net::cpu_reference_f32` rather than re-implementing —
//! the layer reference owns only the orchestration, not the math.
//!
//! This module is the authoritative correctness spec for ADR-013's
//! linear-attention inference path. Never runs in production. The GPU
//! builder (next iter) matches its output to ≤1e-3.

use mlx_native::ops::gated_delta_net::{
    cpu_reference_f32 as gdn_cpu_ref, GatedDeltaNetParams,
};

use crate::inference::models::qwen35::Qwen35Config;

// ================================================================
// Layer weights
// ================================================================

/// Weights for a single Qwen3.5 Gated DeltaNet linear-attention layer.
///
/// All tensors stored as flat f32 row-major vectors; GGUF-native layout is
/// row-major so the loader can copy directly. Shape comments use the exact
/// dimensions the apex GGUF emits (2026-04-23 dump).
#[derive(Debug, Clone)]
pub struct DeltaNetLayerWeights {
    /// Pre-attention RMSNorm: `[hidden_size]`.
    pub attn_norm: Vec<f32>,
    /// Post-attention RMSNorm applied between the attention residual and the FFN:
    /// `[hidden_size]`.  Stored as `blk.{i}.post_attention_norm.weight` in GGUF.
    /// Applied in the forward pass as: `hidden = RMSNorm(hidden, post_attn_norm)`
    /// before the FFN projection.  Omitting this causes hidden-state blow-up
    /// across 40 layers → uniform logits → constant output token.
    pub post_attn_norm: Vec<f32>,
    /// QKV concatenated projection: `[qkv_total, hidden_size]` where
    /// `qkv_total = 2 * n_k_heads * D_k + n_v_heads * D_v`.
    pub attn_qkv: Vec<f32>,
    /// Z-gate (output gate) projection: `[n_v_heads * D_v, hidden_size]`.
    pub attn_gate: Vec<f32>,
    /// Depthwise causal 1D conv kernel: `[K=4, qkv_total]`.
    pub ssm_conv1d: Vec<f32>,
    /// α gate projection: `[n_v_heads, hidden_size]`.
    pub ssm_alpha: Vec<f32>,
    /// α gate time-step bias: `[n_v_heads]`.
    pub ssm_dt_bias: Vec<f32>,
    /// β gate projection: `[n_v_heads, hidden_size]`.
    pub ssm_beta: Vec<f32>,
    /// Per-head log-decay base: `[n_v_heads]`. Note ADR-012 Gotcha #2
    /// (A_log negation): this is stored as log(|A|) with sign handled by
    /// `g = softplus(logit) * exp(ssm_a)` producing a positive decay rate.
    pub ssm_a: Vec<f32>,
    /// Output per-head RMSNorm: `[n_v_heads * D_v]`.
    pub ssm_norm: Vec<f32>,
    /// Output projection: `[hidden_size, n_v_heads * D_v]`.
    pub ssm_out: Vec<f32>,
}

/// Shape parameters for a DeltaNet layer.
#[derive(Debug, Clone, Copy)]
pub struct DeltaNetLayerShape {
    pub hidden_size: u32,
    pub n_k_heads: u32,
    pub n_v_heads: u32,
    pub d_k: u32,
    pub d_v: u32,
    pub conv_kernel: u32, // K (= 4 for Qwen3.5)
    pub rms_norm_eps: f32,
}

impl DeltaNetLayerShape {
    pub fn from_config(cfg: &Qwen35Config) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            n_k_heads: cfg.linear_num_key_heads,
            n_v_heads: cfg.linear_num_value_heads,
            d_k: cfg.linear_key_head_dim,
            d_v: cfg.linear_value_head_dim,
            conv_kernel: cfg.linear_conv_kernel_dim,
            rms_norm_eps: cfg.rms_norm_eps,
        }
    }

    /// Total QKV channel count. Used for both the projection output width
    /// and the conv1d input channel dimension.
    pub fn qkv_channels(&self) -> u32 {
        2 * self.n_k_heads * self.d_k + self.n_v_heads * self.d_v
    }
}

// ================================================================
// Helpers
// ================================================================

fn rms_norm_row(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv = (sum_sq / n + eps).sqrt().recip();
    x.iter()
        .zip(weight.iter())
        .map(|(xi, wi)| xi * inv * wi)
        .collect()
}

fn l2_norm_row(x: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv = (sum_sq + eps).sqrt().recip();
    x.iter().map(|v| v * inv).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    // Numerically stable softplus.
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// `out[i, j] = sum_k lhs[i, k] * rhs[j, k]` (GGUF row-major rhs).
fn matmul_a_by_bt(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += lhs[i * k + kk] * rhs[j * k + kk];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Scalar SSM causal conv1d + SiLU. Stateless variant: takes `conv_state`
/// of `[K-1, channels]` and input `[seq, channels]` (row-major per-token),
/// produces output `[seq, channels]`. Does not update state — caller manages.
///
/// Matches mlx-native's ssm_conv math exactly.
fn ssm_conv_scalar(
    x: &[f32],           // [seq, channels] row-major
    kernel: &[f32],      // [K, channels] row-major (k inner)
    conv_state: &[f32],  // [K-1, channels]
    seq: usize,
    channels: usize,
    k_width: usize,
) -> Vec<f32> {
    let km1 = k_width - 1;
    let mut out = vec![0.0f32; seq * channels];
    for t in 0..seq {
        for c in 0..channels {
            let mut acc = 0.0f32;
            for kk in 0..k_width {
                let t_ext = t + kk;
                let val = if t_ext < km1 {
                    conv_state[t_ext * channels + c]
                } else {
                    x[(t_ext - km1) * channels + c]
                };
                acc += kernel[kk * channels + c] * val;
            }
            out[t * channels + c] = silu(acc);
        }
    }
    out
}

// ================================================================
// Public: scalar CPU reference for a full DeltaNet layer forward
// ================================================================

/// Pure-Rust f32 reference for the Qwen3.5 Gated DeltaNet linear-attention
/// layer forward pass (prefill regime — consumes a full sequence).
///
/// # Inputs
///
/// - `x`: residual stream, `[seq_len, hidden_size]` row-major.
/// - `weights`: layer weights.
/// - `shape`: shape parameters.
/// - `state_in`: recurrent state at start, `[D_k * D_v * n_v_heads]` for a
///   single sequence (n_seqs=1 in the reference). Matches mlx-native's
///   gated_delta_net layout with d_k innermost.
/// - `conv_state`: conv1d ring buffer at start, `[K-1, qkv_channels]`.
///
/// # Returns
///
/// Tuple `(residual_contribution, new_state, new_conv_state)`:
///
/// - `residual_contribution`: `[seq_len, hidden_size]` — caller adds to x.
/// - `new_state`: same layout as `state_in`.
/// - `new_conv_state`: last K-1 tokens of conv-extended input.
pub fn delta_net_layer_cpu_ref(
    x: &[f32],
    weights: &DeltaNetLayerWeights,
    shape: DeltaNetLayerShape,
    state_in: &[f32],
    conv_state: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let h = shape.hidden_size as usize;
    let nk = shape.n_k_heads as usize;
    let nv = shape.n_v_heads as usize;
    let dk = shape.d_k as usize;
    let dv = shape.d_v as usize;
    let k_width = shape.conv_kernel as usize;
    let km1 = k_width - 1;
    let qkv_channels = shape.qkv_channels() as usize;
    let z_channels = nv * dv;
    let seq = x.len() / h;

    assert_eq!(x.len(), seq * h);
    assert_eq!(weights.attn_norm.len(), h);
    assert_eq!(weights.attn_qkv.len(), qkv_channels * h);
    assert_eq!(weights.attn_gate.len(), z_channels * h);
    assert_eq!(weights.ssm_conv1d.len(), k_width * qkv_channels);
    assert_eq!(weights.ssm_alpha.len(), nv * h);
    assert_eq!(weights.ssm_dt_bias.len(), nv);
    assert_eq!(weights.ssm_beta.len(), nv * h);
    assert_eq!(weights.ssm_a.len(), nv);
    // ssm_norm has shape [D_v] (per-head dim only), broadcast across n_v_heads.
    // This matches the apex GGUF's blk.{i}.ssm_norm.weight shape = [D_v].
    assert_eq!(weights.ssm_norm.len(), dv);
    assert_eq!(weights.ssm_out.len(), h * z_channels);
    assert_eq!(state_in.len(), dk * dv * nv);
    assert_eq!(conv_state.len(), km1 * qkv_channels);

    // 1. Pre-norm on x.
    let mut x_norm = vec![0.0f32; seq * h];
    for t in 0..seq {
        let row = &x[t * h..(t + 1) * h];
        let normed = rms_norm_row(row, &weights.attn_norm, shape.rms_norm_eps);
        x_norm[t * h..(t + 1) * h].copy_from_slice(&normed);
    }

    // 2. QKV concatenated projection: [seq, qkv_channels].
    let qkv = matmul_a_by_bt(&x_norm, &weights.attn_qkv, seq, h, qkv_channels);

    // 3. Z-gate projection: [seq, z_channels].
    let z = matmul_a_by_bt(&x_norm, &weights.attn_gate, seq, h, z_channels);

    // 4. Conv1d + SiLU. Output same shape as input [seq, qkv_channels].
    //    Kernel stored [K, qkv_channels] with K fastest.
    //    Our helper expects kernel[kk*channels + c] so flat row-major matches.
    let qkv_conv = ssm_conv_scalar(
        &qkv,
        &weights.ssm_conv1d,
        conv_state,
        seq,
        qkv_channels,
        k_width,
    );

    // 5. Split qkv_conv into Q, K, V.
    //    Per-token layout: [Q: nk*dk, K: nk*dk, V: nv*dv]
    let q_span = nk * dk;
    let k_span = nk * dk;
    let v_span = nv * dv;
    let mut q_buf = vec![0.0f32; seq * q_span];
    let mut k_buf = vec![0.0f32; seq * k_span];
    let mut v_buf = vec![0.0f32; seq * v_span];
    for t in 0..seq {
        let base = t * qkv_channels;
        q_buf[t * q_span..(t + 1) * q_span]
            .copy_from_slice(&qkv_conv[base..base + q_span]);
        k_buf[t * k_span..(t + 1) * k_span]
            .copy_from_slice(&qkv_conv[base + q_span..base + q_span + k_span]);
        v_buf[t * v_span..(t + 1) * v_span]
            .copy_from_slice(&qkv_conv[base + q_span + k_span..base + qkv_channels]);
    }

    // 6. L2 norm Q and K per-head (over D_k).
    for t in 0..seq {
        for h_idx in 0..nk {
            let off = (t * nk + h_idx) * dk;
            let row = &q_buf[off..off + dk];
            let normed = l2_norm_row(row, shape.rms_norm_eps);
            q_buf[off..off + dk].copy_from_slice(&normed);
        }
        for h_idx in 0..nk {
            let off = (t * nk + h_idx) * dk;
            let row = &k_buf[off..off + dk];
            let normed = l2_norm_row(row, shape.rms_norm_eps);
            k_buf[off..off + dk].copy_from_slice(&normed);
        }
    }

    // 7. α and β gate projections.
    //    alpha_logit[t, h] = ssm_alpha @ x_norm[t] + ssm_dt_bias[h]
    let alpha_logits = matmul_a_by_bt(&x_norm, &weights.ssm_alpha, seq, h, nv);
    let beta_logits = matmul_a_by_bt(&x_norm, &weights.ssm_beta, seq, h, nv);
    let mut g = vec![0.0f32; seq * nv];
    let mut beta = vec![0.0f32; seq * nv];
    for t in 0..seq {
        for h_idx in 0..nv {
            let a_logit = alpha_logits[t * nv + h_idx] + weights.ssm_dt_bias[h_idx];
            // g = softplus(a_logit) * exp(ssm_a[h]). ssm_a is log-decay base.
            g[t * nv + h_idx] = softplus(a_logit) * weights.ssm_a[h_idx].exp();
            beta[t * nv + h_idx] = sigmoid(beta_logits[t * nv + h_idx]);
        }
    }

    // 8. GATED_DELTA_NET recurrence via mlx-native's authoritative CPU ref.
    //
    //    mlx-native layout convention:
    //      q, k: [D_k, n_k_heads, n_tokens, n_seqs]  (d_k innermost)
    //      v:    [D_v, n_v_heads, n_tokens, n_seqs]
    //      g, beta: [n_v_heads, n_tokens, n_seqs]
    //      state: [D_k, D_v, n_v_heads, n_seqs]
    //      output: [D_v, n_v_heads, n_tokens, n_seqs]
    //
    //    Our buffers q_buf/k_buf are currently [seq, n_k_heads, D_k]
    //    (token-major / d_k innermost-per-head). Transpose to mlx-native's
    //    [D_k, n_k_heads, n_tokens] (n_seqs=1 implicit).
    let q_trans = transpose_for_gdn(&q_buf, seq, nk, dk);
    let k_trans = transpose_for_gdn(&k_buf, seq, nk, dk);
    let v_trans = transpose_for_gdn(&v_buf, seq, nv, dv);
    // g and beta are already [seq, n_v_heads] which matches [n_v_heads, n_tokens]
    // when reshape happens implicitly (seq treated as n_tokens). But mlx-native
    // reads [n_v_heads, n_tokens, n_seqs] with n_v_heads innermost. So we need
    // to transpose [seq, nv] -> [nv, seq]... actually looking at the mlx-native
    // spec: g[scalar_base] = g[seq * scalar_seq_stride + t * n_v_heads + vh].
    // So for n_seqs=1, offset = t * n_v_heads + vh — which is [seq, n_v_heads]
    // row-major (n_v_heads innermost-per-token). That's what we already have.

    let params = GatedDeltaNetParams {
        d_k: dk as u32,
        d_v: dv as u32,
        n_k_heads: nk as u32,
        n_v_heads: nv as u32,
        n_tokens: seq as u32,
        n_seqs: 1,
    };
    let (gdn_out_mlx, new_state) =
        gdn_cpu_ref(&q_trans, &k_trans, &v_trans, &g, &beta, state_in, params);

    // gdn_out_mlx has layout [D_v, n_v_heads, n_tokens]. Transpose back to
    // [n_tokens, n_v_heads, D_v] (token-major) for the output gating step.
    let mut attn_out = vec![0.0f32; seq * nv * dv];
    for t in 0..seq {
        for vh in 0..nv {
            for d in 0..dv {
                // mlx-native offset: t * n_v_heads * D_v + vh * D_v + d
                let src = t * nv * dv + vh * dv + d;
                // token-major: t * nv * dv + vh * dv + d (same) — coincides
                // because mlx-native v_token_stride = n_v_heads * D_v and
                // our reshape target matches. Keep the explicit write for
                // clarity of intent.
                attn_out[t * nv * dv + vh * dv + d] = gdn_out_mlx[src];
            }
        }
    }

    // 9. Output RMSNorm per-head then SiLU-gate by z.
    //    ssm_norm has shape [D_v] and is BROADCAST across all n_v_heads per token.
    //    For each token and each value head, RMSNorm is applied independently over
    //    the D_v elements of that head using the same ssm_norm weights.
    //    Gate uses SiLU (x * sigmoid(x)) matching llama.cpp's build_norm_gated:
    //      ggml_silu(ctx0, gate) — SiLU, not plain sigmoid.
    //    See llama.cpp qwen35moe.cpp::build_layer_attn_linear.
    assert_eq!(
        weights.ssm_norm.len(),
        dv,
        "ssm_norm shape mismatch: expected [D_v={}] got {}",
        dv,
        weights.ssm_norm.len()
    );
    let mut gated = vec![0.0f32; seq * z_channels];
    for t in 0..seq {
        for vh in 0..nv {
            let head_off = t * z_channels + vh * dv;
            let head_row = &attn_out[head_off..head_off + dv];
            let normed = rms_norm_row(head_row, &weights.ssm_norm, shape.rms_norm_eps);
            for d in 0..dv {
                let z_val = z[head_off + d];
                // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                // Matches llama.cpp build_norm_gated: ggml_silu(ctx0, gate)
                let z_silu = z_val / (1.0 + (-z_val).exp());
                gated[head_off + d] = normed[d] * z_silu;
            }
        }
    }

    // 10. Output projection.
    let output = matmul_a_by_bt(&gated, &weights.ssm_out, seq, z_channels, h);

    // New conv state: last K-1 tokens of extended input.
    let mut new_conv_state = vec![0.0f32; km1 * qkv_channels];
    for i in 0..km1 {
        let t_ext = seq + i; // virtual index
        if t_ext < km1 {
            new_conv_state[i * qkv_channels..(i + 1) * qkv_channels]
                .copy_from_slice(&conv_state[t_ext * qkv_channels..(t_ext + 1) * qkv_channels]);
        } else {
            let t_in = t_ext - km1;
            new_conv_state[i * qkv_channels..(i + 1) * qkv_channels]
                .copy_from_slice(&qkv[t_in * qkv_channels..(t_in + 1) * qkv_channels]);
        }
    }

    (output, new_state, new_conv_state)
}

/// Transpose a `[seq, n_heads, D]` row-major buffer into mlx-native's
/// `[D, n_heads, seq, 1]` layout (d innermost). Only for the 1-seq case;
/// multi-seq extension is trivial.
fn transpose_for_gdn(src: &[f32], seq: usize, n_heads: usize, d: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; seq * n_heads * d];
    // mlx-native offset: s * (n_tokens * n_heads * D) + t * (n_heads * D) + kh * D + i
    // With n_seqs=1, s=0. So dst[t * n_heads * D + kh * D + i] = val.
    // Our src layout: [seq, n_heads, d] means src[t * n_heads * d + h * d + i].
    // These coincide — d innermost in both. So it's a direct copy.
    dst.copy_from_slice(src);
    dst
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_rand(seed: &mut u32, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((*seed as i32 as f32) / (i32::MAX as f32)) * scale
            })
            .collect()
    }

    fn small_shape() -> DeltaNetLayerShape {
        DeltaNetLayerShape {
            hidden_size: 8,
            n_k_heads: 2,
            n_v_heads: 4, // group_ratio = 2
            d_k: 4,
            d_v: 4,
            conv_kernel: 4,
            rms_norm_eps: 1e-6,
        }
    }

    fn synthetic_weights(shape: DeltaNetLayerShape, seed_init: u32) -> DeltaNetLayerWeights {
        let h = shape.hidden_size as usize;
        let nk = shape.n_k_heads as usize;
        let nv = shape.n_v_heads as usize;
        let dk = shape.d_k as usize;
        let dv = shape.d_v as usize;
        let k_width = shape.conv_kernel as usize;
        let qkv_channels = shape.qkv_channels() as usize;
        let z_channels = nv * dv;

        let mut seed = seed_init;
        DeltaNetLayerWeights {
            attn_norm: {
                let mut v = vec![1.0f32; h];
                for (i, x) in v.iter_mut().enumerate() {
                    *x += 0.01 * (i as f32);
                }
                v
            },
            post_attn_norm: vec![1.0f32; h],
            attn_qkv: mk_rand(&mut seed, qkv_channels * h, 0.1),
            attn_gate: mk_rand(&mut seed, z_channels * h, 0.1),
            ssm_conv1d: mk_rand(&mut seed, k_width * qkv_channels, 0.1),
            ssm_alpha: mk_rand(&mut seed, nv * h, 0.1),
            ssm_dt_bias: mk_rand(&mut seed, nv, 0.05),
            ssm_beta: mk_rand(&mut seed, nv * h, 0.1),
            ssm_a: mk_rand(&mut seed, nv, 0.1),
            // ssm_norm shape is [D_v] (broadcast across heads), not [z_channels].
            ssm_norm: {
                let mut v = vec![1.0f32; dv];
                for (i, x) in v.iter_mut().enumerate() {
                    *x += 0.01 * (i as f32);
                }
                v
            },
            ssm_out: mk_rand(&mut seed, h * z_channels, 0.1),
        }
    }

    #[test]
    fn shape_qkv_channels() {
        let s = small_shape();
        // 2 * nk*dk + nv*dv = 2*2*4 + 4*4 = 16 + 16 = 32
        assert_eq!(s.qkv_channels(), 32);
    }

    /// Layer runs end-to-end and produces finite, non-trivial output of the
    /// expected shape.
    #[test]
    fn delta_net_layer_produces_expected_shape() {
        let shape = small_shape();
        let weights = synthetic_weights(shape, 0xABCD);
        let seq_len = 3;
        let h = shape.hidden_size as usize;
        let km1 = (shape.conv_kernel - 1) as usize;
        let qkv_channels = shape.qkv_channels() as usize;

        let x: Vec<f32> = (0..seq_len * h).map(|i| 0.01 * i as f32).collect();
        let state_in = vec![0.0f32; (shape.d_k * shape.d_v * shape.n_v_heads) as usize];
        let conv_state = vec![0.0f32; km1 * qkv_channels];

        let (out, new_state, new_conv) =
            delta_net_layer_cpu_ref(&x, &weights, shape, &state_in, &conv_state);

        assert_eq!(out.len(), seq_len * h);
        assert_eq!(new_state.len(), state_in.len());
        assert_eq!(new_conv.len(), conv_state.len());
        assert!(out.iter().all(|v| v.is_finite()));
        let sum_abs: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(sum_abs > 1e-6, "output is nearly zero — something broken");
    }

    /// Determinism: same input yields bit-identical output across reruns.
    #[test]
    fn delta_net_layer_deterministic() {
        let shape = small_shape();
        let weights = synthetic_weights(shape, 0x1234);
        let seq_len = 2;
        let h = shape.hidden_size as usize;
        let km1 = (shape.conv_kernel - 1) as usize;
        let qkv_channels = shape.qkv_channels() as usize;
        let x: Vec<f32> = (0..seq_len * h).map(|i| 0.02 * i as f32).collect();
        let state_in = vec![0.0f32; (shape.d_k * shape.d_v * shape.n_v_heads) as usize];
        let conv_state = vec![0.0f32; km1 * qkv_channels];

        let (o1, s1, c1) = delta_net_layer_cpu_ref(&x, &weights, shape, &state_in, &conv_state);
        let (o2, s2, c2) = delta_net_layer_cpu_ref(&x, &weights, shape, &state_in, &conv_state);

        for i in 0..o1.len() {
            assert_eq!(o1[i].to_bits(), o2[i].to_bits());
        }
        for i in 0..s1.len() {
            assert_eq!(s1[i].to_bits(), s2[i].to_bits());
        }
        for i in 0..c1.len() {
            assert_eq!(c1[i].to_bits(), c2[i].to_bits());
        }
    }

    /// Initial state propagation: non-zero `state_in` should affect the
    /// output. Verify by running with two distinct initial states.
    #[test]
    fn delta_net_layer_state_in_affects_output() {
        let shape = small_shape();
        let weights = synthetic_weights(shape, 0x5678);
        let seq_len = 2;
        let h = shape.hidden_size as usize;
        let km1 = (shape.conv_kernel - 1) as usize;
        let qkv_channels = shape.qkv_channels() as usize;
        let x: Vec<f32> = (0..seq_len * h).map(|i| 0.03 * i as f32).collect();
        let conv_state = vec![0.0f32; km1 * qkv_channels];

        let state_zeros = vec![0.0f32; (shape.d_k * shape.d_v * shape.n_v_heads) as usize];
        let mut state_nonzero = state_zeros.clone();
        for (i, v) in state_nonzero.iter_mut().enumerate() {
            *v = 0.1 * ((i % 13) as f32);
        }

        let (o_zero, _, _) =
            delta_net_layer_cpu_ref(&x, &weights, shape, &state_zeros, &conv_state);
        let (o_nonzero, _, _) =
            delta_net_layer_cpu_ref(&x, &weights, shape, &state_nonzero, &conv_state);

        let mut any_diff = false;
        for i in 0..o_zero.len() {
            if (o_zero[i] - o_nonzero[i]).abs() > 1e-5 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "initial state had no effect on output");
    }

    /// Incremental vs monolithic: running [token 0] then [token 1] through
    /// two chunked calls (using intermediate state) should produce the same
    /// output as one monolithic call on [token 0, token 1]. This is the
    /// cross-cutting correctness invariant for DeltaNet state propagation.
    #[test]
    fn delta_net_layer_chunked_equals_monolithic() {
        let shape = small_shape();
        let weights = synthetic_weights(shape, 0xFACE);
        let h = shape.hidden_size as usize;
        let km1 = (shape.conv_kernel - 1) as usize;
        let qkv_channels = shape.qkv_channels() as usize;
        let state_zeros = vec![0.0f32; (shape.d_k * shape.d_v * shape.n_v_heads) as usize];
        let conv_zeros = vec![0.0f32; km1 * qkv_channels];

        let x_full: Vec<f32> = (0..2 * h).map(|i| 0.1 + 0.01 * i as f32).collect();

        // Monolithic run.
        let (out_mono, _, _) = delta_net_layer_cpu_ref(
            &x_full, &weights, shape, &state_zeros, &conv_zeros,
        );

        // Chunked: token 0, then token 1 using intermediate state.
        let x_t0 = x_full[0..h].to_vec();
        let x_t1 = x_full[h..2 * h].to_vec();
        let (out_t0, state_after_t0, conv_after_t0) =
            delta_net_layer_cpu_ref(&x_t0, &weights, shape, &state_zeros, &conv_zeros);
        let (out_t1, _, _) = delta_net_layer_cpu_ref(
            &x_t1, &weights, shape, &state_after_t0, &conv_after_t0,
        );

        // Chunked outputs concatenated should match monolithic.
        for i in 0..h {
            let d = (out_mono[i] - out_t0[i]).abs();
            assert!(
                d < 1e-5,
                "chunked-vs-mono t0 mismatch at {}: mono={}, chunk={}",
                i, out_mono[i], out_t0[i]
            );
        }
        for i in 0..h {
            let d = (out_mono[h + i] - out_t1[i]).abs();
            assert!(
                d < 1e-5,
                "chunked-vs-mono t1 mismatch at {}: mono={}, chunk={}",
                i, out_mono[h + i], out_t1[i]
            );
        }
    }

    #[test]
    fn delta_net_layer_rejects_wrong_state_shape() {
        let shape = small_shape();
        let weights = synthetic_weights(shape, 0xDEAD);
        let seq_len = 1;
        let h = shape.hidden_size as usize;
        let km1 = (shape.conv_kernel - 1) as usize;
        let qkv_channels = shape.qkv_channels() as usize;
        let x: Vec<f32> = (0..seq_len * h).map(|i| 0.01 * i as f32).collect();
        let wrong_state = vec![0.0f32; 123];
        let conv_state = vec![0.0f32; km1 * qkv_channels];

        // Panic is the assert behavior; catch_unwind would add noise. Just
        // ensure the test is visible by using a should_panic attribute if we
        // had one; we'll do the equivalent by wrapping in a closure.
        let res = std::panic::catch_unwind(|| {
            delta_net_layer_cpu_ref(&x, &weights, shape, &wrong_state, &conv_state);
        });
        assert!(res.is_err(), "wrong state shape should panic");
    }
}
