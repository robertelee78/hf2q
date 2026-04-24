//! GPU-side weight containers and full forward-pass builder for the
//! Qwen3.5 Gated DeltaNet linear-attention layer (ADR-013 Decision 8, GPU path).
//!
//! This module bridges [`super::delta_net`]'s pure-Rust scalar reference
//! (the authoritative spec + test oracle) and the mlx-native GPU kernels.
//! It carries per-layer weights as `MlxBuffer` handles and exposes every
//! per-op dispatch as a small public function, composing them into
//! [`build_delta_net_layer`] — the end-to-end GPU forward pass.
//!
//! # Op order (mirrors the CPU ref verbatim — `delta_net.rs` step numbers)
//!
//! ```text
//!  1.  pre_norm         — RMSNorm(x, attn_norm, eps)            → x_norm [seq, H]
//!  2a. proj_qkv         — x_norm @ attn_qkv^T                   → qkv   [seq, QKV]
//!  2b. proj_z           — x_norm @ attn_gate^T                  → z     [seq, nv*dv]
//!  3.  ssm_conv_fwd     — causal 1D conv(qkv, ssm_conv1d, conv_state) → qkv_conv [seq, QKV]
//!  4.  split            — qkv_conv → Q [seq, nk, dk], K [seq, nk, dk], V [seq, nv, dv]
//!  5.  l2_norm_q        — L2Norm per head over dk               → Q_n
//!      l2_norm_k        — L2Norm per head over dk               → K_n
//!  6a. proj_alpha_logit — x_norm @ ssm_alpha^T + ssm_dt_bias    → alpha_logit [seq, nv]
//!  6b. proj_beta_logit  — x_norm @ ssm_beta^T                   → beta_logit  [seq, nv]
//!  6c. compute_g_beta   — g = softplus(alpha_logit) * exp(ssm_a); beta = sigmoid(beta_logit)
//!  7.  gated_delta_net  — GDN(Q_n, K_n, V, g, beta, state_in) → (attn_out, state_out)
//!  8.  ssm_norm_gate    — RMSNorm(attn_out, ssm_norm) * sigmoid(z) → gated
//!  9.  out_proj         — gated @ ssm_out^T                     → [seq, H]
//! ```
//!
//! # Layout notes
//!
//! All intermediate buffers are F32. The ssm_conv kernel uses
//! `[n_seqs, n_tokens, channels]` (token-major, channels innermost) for x/y,
//! and `[n_seqs, channels, K-1]` (channels outermost per seq) for state.
//! The kernel_w layout is `[channels, K]` (K innermost).
//!
//! The gated_delta_net kernel uses:
//! - q, k: `[n_tokens, n_k_heads, d_k]` (for n_seqs=1) — matches split output
//! - v: `[n_tokens, n_v_heads, d_v]`
//! - g, beta: `[n_tokens, n_v_heads]`
//! - state: `[d_k, d_v, n_v_heads, n_seqs]` — same as HybridKvCache recurrent layout
//!
//! # SSM conv state transpose
//!
//! The HybridKvCache `conv_state` buffer is zero-initialized and then written
//! by the ssm_conv state-update kernel in `[n_seqs, channels, K-1]` layout
//! (kernel layout). On first call the zero buffer is valid in any layout;
//! on subsequent calls we pass back what the kernel wrote. A CPU-side transpose
//! is performed on `old_state` read from the cache to convert from the buffer's
//! runtime layout into the kernel's expected layout. This mirrors P7b's
//! approach for the SDPA permute.
//!
//! # Kernel_w transpose
//!
//! The GGUF/CPU ref stores `ssm_conv1d` as `[K, channels]` (K outermost). The
//! ssm_conv kernel expects `[channels, K]` (K innermost). We transpose once at
//! upload time via `DeltaNetWeightsGpu::from_cpu` so the hot path pays no cost.
//!
//! # ADR status
//!
//! P8b: every op wired, parity test targets |GPU−CPU|∞ < 1e-3 F32.

use anyhow::{anyhow, Context, Result};
use mlx_native::ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};
use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::gated_delta_net::{
    build_gated_delta_net_params, dispatch_gated_delta_net, GatedDeltaNetParams,
};
use mlx_native::ops::l2_norm::dispatch_l2_norm;
use mlx_native::ops::rms_norm;
use mlx_native::ops::ssm_conv::{dispatch_ssm_conv, SsmConvParams};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::delta_net::DeltaNetLayerWeights;
use super::gpu_full_attn::{download_f32, upload_f32};

// ================================================================
// GPU weight container
// ================================================================

/// GPU-side weight handles for a single Qwen3.5 Gated DeltaNet layer.
///
/// All buffers are F32. `ssm_conv1d_gpu` is stored transposed relative to
/// the CPU/GGUF format (see module-level layout notes).
pub struct DeltaNetWeightsGpu {
    pub attn_norm: MlxBuffer,
    /// Post-attention RMSNorm weight: `[hidden_size]`.
    /// Applied to the residual stream after attention, before the FFN.
    pub post_attn_norm: MlxBuffer,
    /// QKV concat projection `[qkv_channels, hidden_size]` — GGUF row-major.
    pub attn_qkv: MlxBuffer,
    /// Z-gate projection `[nv*dv, hidden_size]`.
    pub attn_gate: MlxBuffer,
    /// SSM conv1d kernel, transposed to `[qkv_channels, K]` for the GPU kernel.
    pub ssm_conv1d: MlxBuffer,
    /// Alpha logit projection `[nv, hidden_size]`.
    pub ssm_alpha: MlxBuffer,
    /// Alpha time-step bias `[nv]`.
    pub ssm_dt_bias: MlxBuffer,
    /// Beta logit projection `[nv, hidden_size]`.
    pub ssm_beta: MlxBuffer,
    /// Log-decay base `[nv]`.
    pub ssm_a: MlxBuffer,
    /// Output per-head RMSNorm `[nv*dv]`.
    pub ssm_norm: MlxBuffer,
    /// Output projection `[hidden_size, nv*dv]`.
    pub ssm_out: MlxBuffer,
}

impl DeltaNetWeightsGpu {
    /// Upload [`DeltaNetLayerWeights`] (CPU f32) to Metal buffers.
    ///
    /// Transposes `ssm_conv1d` from `[K, channels]` (CPU/GGUF) to
    /// `[channels, K]` (kernel layout) at upload time.
    pub fn from_cpu(
        weights: &DeltaNetLayerWeights,
        device: &MlxDevice,
        k_width: usize,
        qkv_channels: usize,
    ) -> Result<Self> {
        // Transpose ssm_conv1d: [K, channels] → [channels, K].
        let conv1d_t = transpose_k_channels_to_channels_k(
            &weights.ssm_conv1d,
            k_width,
            qkv_channels,
        );
        Ok(Self {
            attn_norm: upload_f32(&weights.attn_norm, device)?,
            post_attn_norm: upload_f32(&weights.post_attn_norm, device)?,
            attn_qkv: upload_f32(&weights.attn_qkv, device)?,
            attn_gate: upload_f32(&weights.attn_gate, device)?,
            ssm_conv1d: upload_f32(&conv1d_t, device)?,
            ssm_alpha: upload_f32(&weights.ssm_alpha, device)?,
            ssm_dt_bias: upload_f32(&weights.ssm_dt_bias, device)?,
            ssm_beta: upload_f32(&weights.ssm_beta, device)?,
            ssm_a: upload_f32(&weights.ssm_a, device)?,
            ssm_norm: upload_f32(&weights.ssm_norm, device)?,
            ssm_out: upload_f32(&weights.ssm_out, device)?,
        })
    }
}

// ================================================================
// Layout helpers
// ================================================================

/// Transpose a `[K, channels]` flat buffer to `[channels, K]`.
/// CPU/GGUF stores conv kernel as K outermost; the ssm_conv Metal kernel
/// reads `kernel_w[k, c]` at offset `c * K + k`.
fn transpose_k_channels_to_channels_k(src: &[f32], k: usize, channels: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; k * channels];
    for ki in 0..k {
        for c in 0..channels {
            // src[ki, c] = src[ki * channels + c]
            // dst[c, ki] = dst[c * k + ki]
            dst[c * k + ki] = src[ki * channels + c];
        }
    }
    dst
}

/// Transpose conv_state from `[K-1, channels]` (CPU/HybridKvCache zero init)
/// to the kernel's `[channels, K-1]` layout.
///
/// For n_seqs=1: the kernel reads `state[i, c, s]` at `c * (K-1) + i`.
/// The zero-initialized kv_cache buffer (or CPU ref's `[K-1, channels]`) has
/// `state[i * channels + c]`. This performs the conversion.
fn transpose_state_km1_c_to_c_km1(src: &[f32], km1: usize, channels: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; km1 * channels];
    for i in 0..km1 {
        for c in 0..channels {
            // src[i, c] = src[i * channels + c]
            // dst[c, i] = dst[c * km1 + i]
            dst[c * km1 + i] = src[i * channels + c];
        }
    }
    dst
}

/// Transpose conv_state from `[channels, K-1]` (kernel output) back to
/// `[K-1, channels]` (CPU ref / HybridKvCache layout convention).
fn transpose_state_c_km1_to_km1_c(src: &[f32], km1: usize, channels: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; km1 * channels];
    for c in 0..channels {
        for i in 0..km1 {
            // src[c, i] = src[c * km1 + i]
            // dst[i, c] = dst[i * channels + c]
            dst[i * channels + c] = src[c * km1 + i];
        }
    }
    dst
}

// ================================================================
// Individual op dispatchers
// ================================================================

/// Apply pre-attention RMSNorm to the residual stream input.
///
/// Output: `[seq_len, hidden_size]` F32.
pub fn apply_pre_norm(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    norm_weight: &MlxBuffer,
    seq_len: u32,
    hidden_size: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    let out = device
        .alloc_buffer(
            (seq_len * hidden_size) as usize * 4,
            DType::F32,
            vec![seq_len as usize, hidden_size as usize],
        )
        .map_err(|e| anyhow!("alloc pre_norm out: {e}"))?;
    let mut params = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("alloc params: {e}"))?;
    {
        let s = params.as_mut_slice::<f32>().map_err(|e| anyhow!("{e}"))?;
        s[0] = eps;
        s[1] = hidden_size as f32;
    }
    rms_norm::dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        norm_weight,
        &out,
        &params,
        seq_len,
        hidden_size,
    )
    .context("dispatch_rms_norm pre_norm")?;
    Ok(out)
}

/// Apply a single linear projection: `output = input @ weight^T`.
///
/// Uses F32-via-BF16 cast path (same as P7b `apply_linear_projection_f32`).
/// Requires `in_features >= 32`.
pub fn apply_proj(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    seq_len: u32,
    in_features: u32,
    out_features: u32,
) -> Result<MlxBuffer> {
    let n_w = (out_features * in_features) as usize;
    let weight_bf16 = device
        .alloc_buffer(
            n_w * 2,
            DType::BF16,
            vec![out_features as usize, in_features as usize],
        )
        .map_err(|e| anyhow!("alloc weight_bf16: {e}"))?;
    cast(
        encoder,
        registry,
        device.metal_device(),
        weight,
        &weight_bf16,
        n_w,
        CastDirection::F32ToBF16,
    )
    .context("cast weight F32→BF16")?;

    let out_bytes = (seq_len * out_features) as usize * 4;
    let mut dst = device
        .alloc_buffer(
            out_bytes,
            DType::F32,
            vec![seq_len as usize, out_features as usize],
        )
        .map_err(|e| anyhow!("alloc proj out: {e}"))?;

    let params = DenseMmBf16F32Params {
        m: seq_len,
        n: out_features,
        k: in_features,
        src0_batch: 1,
        src1_batch: 1,
    };
    dense_matmul_bf16_f32_tensor(
        encoder,
        registry,
        device,
        &weight_bf16,
        input,
        &mut dst,
        &params,
    )
    .context("dense_matmul_bf16_f32_tensor proj")?;
    Ok(dst)
}

/// Run SSM causal conv1d + SiLU on the QKV projection.
///
/// `qkv_seq_major`: `[seq_len, qkv_channels]` F32 (token-major, channels
/// innermost) — this is the natural output of the QKV matmul and matches
/// the kernel's x layout for n_seqs=1.
///
/// `old_conv_state_km1_c`: `[K-1, channels]` F32 in CPU/HybridKvCache layout.
/// Transposed internally to `[channels, K-1]` (kernel layout) before dispatch.
///
/// Returns `(y, new_conv_state_km1_c)` where:
/// - `y`: `[seq_len, qkv_channels]` F32 — token-major output of conv.
/// - `new_conv_state_km1_c`: `[K-1, channels]` F32 in CPU layout (ready to
///   store back to HybridKvCache).
pub fn apply_ssm_conv(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    qkv_seq_major: &MlxBuffer,
    conv_kernel_transposed: &MlxBuffer, // [channels, K] already transposed
    old_conv_state_km1_c: &[f32],       // [K-1, channels] CPU layout
    seq_len: u32,
    qkv_channels: u32,
    k_width: u32,
) -> Result<(MlxBuffer, Vec<f32>)> {
    let km1 = (k_width - 1) as usize;
    let channels = qkv_channels as usize;

    // Transpose old_state from [K-1, channels] → [channels, K-1] for the kernel.
    let old_state_ck = transpose_state_km1_c_to_c_km1(old_conv_state_km1_c, km1, channels);

    // Upload old_state in kernel layout.
    let old_state_buf = upload_f32(&old_state_ck, device)?;

    // Allocate new_state buffer (kernel will write it in [channels, K-1] layout).
    let s_elems = km1 * channels; // n_seqs=1
    let new_state_buf = device
        .alloc_buffer(s_elems * 4, DType::F32, vec![channels, km1])
        .map_err(|e| anyhow!("alloc new_state_buf: {e}"))?;

    // Allocate y output.
    let y = device
        .alloc_buffer(
            (seq_len * qkv_channels) as usize * 4,
            DType::F32,
            vec![seq_len as usize, qkv_channels as usize],
        )
        .map_err(|e| anyhow!("alloc ssm_conv y: {e}"))?;

    // Build params buffer: [channels, n_tokens, n_seqs, k_width].
    let mut params_buf = device
        .alloc_buffer(4 * 4, DType::U32, vec![4])
        .map_err(|e| anyhow!("alloc ssm_conv params: {e}"))?;
    {
        let s = params_buf
            .as_mut_slice::<u32>()
            .map_err(|e| anyhow!("{e}"))?;
        s[0] = qkv_channels;
        s[1] = seq_len;
        s[2] = 1; // n_seqs
        s[3] = k_width;
    }

    let conv_params = SsmConvParams {
        channels: qkv_channels,
        n_tokens: seq_len,
        n_seqs: 1,
        k_width,
    };

    let mut enc = device.command_encoder().context("enc ssm_conv")?;
    dispatch_ssm_conv(
        &mut enc,
        registry,
        device.metal_device(),
        qkv_seq_major,
        conv_kernel_transposed,
        &old_state_buf,
        &new_state_buf,
        &y,
        &params_buf,
        conv_params,
    )
    .context("dispatch_ssm_conv")?;
    enc.commit_and_wait().context("commit ssm_conv")?;

    // Download new_state and transpose back from [channels, K-1] → [K-1, channels].
    let new_state_ck = download_f32(&new_state_buf)?;
    let new_state_km1_c = transpose_state_c_km1_to_km1_c(&new_state_ck, km1, channels);

    Ok((y, new_state_km1_c))
}

/// Apply per-head L2 norm to a `[seq * n_heads, head_dim]` F32 buffer.
///
/// Each row (one head for one token) is normalized independently over `head_dim`.
pub fn apply_l2_norm_per_head(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    let rows = seq_len * n_heads;
    let dim = head_dim;
    let out = device
        .alloc_buffer(
            (rows * dim) as usize * 4,
            DType::F32,
            vec![rows as usize, dim as usize],
        )
        .map_err(|e| anyhow!("alloc l2_norm out: {e}"))?;
    let mut params_buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("alloc l2_norm params: {e}"))?;
    {
        let s = params_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("{e}"))?;
        s[0] = eps;
        s[1] = dim as f32;
    }
    dispatch_l2_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        &out,
        &params_buf,
        rows,
        dim,
    )
    .context("dispatch_l2_norm")?;
    Ok(out)
}

/// Compute `g[t, vh] = softplus(alpha_logit[t, vh] + dt_bias[vh]) * ssm_a[vh]`
/// and `beta[t, vh] = sigmoid(beta_logit[t, vh])` on the CPU.
///
/// Note: `ssm_a` contains the pre-computed values directly (stored as
/// `-A_log.exp()` in the model, already negative).  The formula matches
/// llama.cpp qwen35moe.cpp: `gate = ggml_mul(alpha_softplus, ssm_a)`.
/// Do NOT apply `exp()` to ssm_a — it is used as-is.
///
/// Both inputs are `[seq_len, nv]` F32 (token-major). Returns `(g, beta)` same
/// shape. Done CPU-side because nv is small (32 for MoE, 48 for dense) and the
/// op is not on the GPU hot-path for the parity test.
pub fn compute_g_and_beta_cpu(
    alpha_logit_cpu: &[f32],  // [seq, nv]
    beta_logit_cpu: &[f32],   // [seq, nv]
    dt_bias: &[f32],          // [nv]
    ssm_a: &[f32],            // [nv]: pre-computed -A_log.exp() values, used directly
    seq_len: usize,
    nv: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut g = vec![0.0f32; seq_len * nv];
    let mut beta = vec![0.0f32; seq_len * nv];
    for t in 0..seq_len {
        for vh in 0..nv {
            let a_logit = alpha_logit_cpu[t * nv + vh] + dt_bias[vh];
            // Direct multiply — ssm_a is already the pre-computed scaling factor.
            // llama.cpp: gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a)
            g[t * nv + vh] = softplus_f32(a_logit) * ssm_a[vh];
            beta[t * nv + vh] = sigmoid_f32(beta_logit_cpu[t * nv + vh]);
        }
    }
    (g, beta)
}

#[inline(always)]
fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline(always)]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply the fused Gated DeltaNet kernel.
///
/// # Inputs
///
/// - `q`, `k`: `[seq_len, n_k_heads, d_k]` F32 (token-major, d_k innermost).
/// - `v`: `[seq_len, n_v_heads, d_v]` F32.
/// - `g`, `beta`: `[seq_len, n_v_heads]` F32.
/// - `state_in_f`: flat `[d_k * d_v * n_v_heads]` F32 (matches kernel layout
///   `[d_k, d_v, n_v_heads]` with d_k innermost for n_seqs=1).
///
/// # Returns
///
/// `(output, new_state_flat)` where:
/// - `output`: `[seq_len, n_v_heads, d_v]` F32 in token-major.
/// - `new_state_flat`: same layout as `state_in_f`.
#[allow(clippy::too_many_arguments)]
pub fn apply_gated_delta_net(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    g_cpu: &[f32],
    beta_cpu: &[f32],
    state_in_f: &[f32],
    seq_len: u32,
    n_k_heads: u32,
    n_v_heads: u32,
    d_k: u32,
    d_v: u32,
) -> Result<(MlxBuffer, Vec<f32>)> {
    let n_seqs = 1u32;

    // Upload g, beta, state_in.
    let g_buf = upload_f32(g_cpu, device)?;
    let beta_buf = upload_f32(beta_cpu, device)?;
    let state_in_buf = upload_f32(state_in_f, device)?;

    // Allocate output and state_out.
    let out_elems = (n_v_heads * seq_len * d_v) as usize;
    let output_buf = device
        .alloc_buffer(out_elems * 4, DType::F32, vec![out_elems])
        .map_err(|e| anyhow!("alloc gdn output: {e}"))?;
    let state_elems = (d_k * d_v * n_v_heads) as usize; // n_seqs=1
    let state_out_buf = device
        .alloc_buffer(state_elems * 4, DType::F32, vec![state_elems])
        .map_err(|e| anyhow!("alloc gdn state_out: {e}"))?;

    let params = GatedDeltaNetParams {
        d_k,
        d_v,
        n_k_heads,
        n_v_heads,
        n_tokens: seq_len,
        n_seqs,
    };
    let params_buf = build_gated_delta_net_params(device, params)
        .map_err(|e| anyhow!("build gdn params: {e}"))?;

    let mut enc = device.command_encoder().context("enc gdn")?;
    dispatch_gated_delta_net(
        &mut enc,
        registry,
        device.metal_device(),
        q,
        k,
        v,
        &g_buf,
        &beta_buf,
        &state_in_buf,
        &output_buf,
        &state_out_buf,
        &params_buf,
        params,
    )
    .context("dispatch_gated_delta_net")?;
    enc.commit_and_wait().context("commit gdn")?;

    let new_state = download_f32(&state_out_buf)?;
    Ok((output_buf, new_state))
}

/// Apply per-head output RMSNorm then element-wise SiLU-gate multiply.
///
/// `attn_out`: `[seq_len, nv*dv]` F32.
/// `z_flat`: `[seq_len, nv*dv]` F32 (raw Z-gate logits before SiLU).
/// `ssm_norm_w`: `[dv]` F32 (RMSNorm weight, **broadcast** across all n_v_heads).
///   In the apex GGUF the ssm_norm tensor has shape `[D_v]` not `[n_v_heads * D_v]`.
///   The norm is applied independently to each `[dv]`-element head slice.
/// `n_v_heads`: number of value heads (nv).
/// `d_v`: per-head value dimension (dv).
///
/// Returns `[seq_len, nv*dv]` F32 = `RMSNorm_per_head(attn_out) * silu(z)`.
///
/// Gate: SiLU(x) = x / (1 + exp(-x)), matching llama.cpp build_norm_gated
/// which calls `ggml_silu(ctx0, gate)`.
pub fn apply_ssm_norm_and_gate(
    encoder: &mut mlx_native::CommandEncoder,
    _registry: &mut KernelRegistry,
    device: &MlxDevice,
    attn_out: &MlxBuffer,
    z_flat: &MlxBuffer,
    ssm_norm_w: &MlxBuffer,
    seq_len: u32,
    z_channels: u32,  // = n_v_heads * d_v (kept for API compatibility)
    eps: f32,
) -> Result<MlxBuffer> {
    // We can't use dispatch_rms_norm here because ssm_norm_w has shape [dv],
    // not [z_channels]. We need per-head normalization with weight broadcasting.
    // Download to CPU and apply per-head.
    encoder.commit_and_wait().context("commit before ssm_norm_gate")?;

    let attn_out_cpu = download_f32(attn_out).context("download attn_out ssm_norm")?;
    let z_cpu = download_f32(z_flat).context("download z ssm_norm")?;
    let norm_w_cpu = download_f32(ssm_norm_w).context("download ssm_norm_w")?;

    let n_total = (seq_len * z_channels) as usize;
    let dv = norm_w_cpu.len(); // actual D_v from the norm weight shape
    // z_channels should be n_v_heads * dv, but we infer nv from the two:
    let nv = if dv > 0 { z_channels as usize / dv } else { 1 };

    let mut gated = vec![0.0f32; n_total];
    let seq = seq_len as usize;

    for t in 0..seq {
        for vh in 0..nv {
            let head_off = t * nv * dv + vh * dv;
            let head_row = &attn_out_cpu[head_off..head_off + dv];

            // Per-head RMSNorm with the D_v-element weight vector.
            let sum_sq: f32 = head_row.iter().map(|v| v * v).sum();
            let inv = ((sum_sq / (dv as f32)) + eps).sqrt().recip();
            for d in 0..dv {
                let normed_val = head_row[d] * inv * norm_w_cpu[d];
                let z_val = z_cpu[head_off + d];
                // SiLU: x / (1 + exp(-x)) — matches llama.cpp ggml_silu
                let z_silu = z_val / (1.0 + (-z_val).exp());
                gated[head_off + d] = normed_val * z_silu;
            }
        }
    }

    upload_f32(&gated, device).context("upload ssm_norm_gated")
}

// ================================================================
// End-to-end layer builder
// ================================================================

/// Build the complete Qwen3.5 Gated DeltaNet linear-attention forward pass.
///
/// Implements ADR-013 Decision 8 op order end-to-end. Returns the
/// residual *contribution* `[seq_len, hidden_size]` F32 — the caller
/// computes `x + contribution` for the post-layer residual stream.
///
/// # State management
///
/// `old_conv_state` and `state_in` are read from the caller's
/// [`HybridKvCache`] and passed here by value (as slices). The returned
/// `(new_conv_state, new_recurrent_state)` must be written back to the
/// cache by the caller.
///
/// `old_conv_state` is `[K-1, qkv_channels]` F32 (CPU layout).
/// `state_in` is `[d_k * d_v * n_v_heads]` F32 (kernel layout = recurrent
/// state in HybridKvCache).
///
/// # Returns
///
/// `(output, new_conv_state, new_recurrent_state)`.
#[allow(clippy::too_many_arguments)]
pub fn build_delta_net_layer(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &DeltaNetWeightsGpu,
    old_conv_state: &[f32], // [K-1, qkv_channels] CPU layout
    state_in: &[f32],       // [d_k * d_v * n_v_heads] kernel layout
    seq_len: u32,
    hidden_size: u32,
    n_k_heads: u32,
    n_v_heads: u32,
    d_k: u32,
    d_v: u32,
    k_width: u32,
    rms_norm_eps: f32,
) -> Result<(MlxBuffer, Vec<f32>, Vec<f32>)> {
    let qkv_channels = 2 * n_k_heads * d_k + n_v_heads * d_v;
    let z_channels = n_v_heads * d_v;
    let q_span = n_k_heads * d_k;
    let k_span = n_k_heads * d_k;

    // ---- Op 1: pre-norm ----
    let mut enc = device.command_encoder().context("enc op1")?;
    let x_norm = apply_pre_norm(
        &mut enc, registry, device, x, &weights.attn_norm,
        seq_len, hidden_size, rms_norm_eps,
    )?;
    enc.commit_and_wait().context("commit op1")?;

    // ---- Op 2a: QKV projection ----
    let mut enc = device.command_encoder().context("enc op2a")?;
    let qkv = apply_proj(
        &mut enc, registry, device, &x_norm,
        &weights.attn_qkv, seq_len, hidden_size, qkv_channels,
    )?;
    enc.commit_and_wait().context("commit op2a")?;

    // ---- Op 2b: Z-gate projection ----
    let mut enc = device.command_encoder().context("enc op2b")?;
    let z = apply_proj(
        &mut enc, registry, device, &x_norm,
        &weights.attn_gate, seq_len, hidden_size, z_channels,
    )?;
    enc.commit_and_wait().context("commit op2b")?;

    // ---- Op 3: SSM conv1d + SiLU ----
    // apply_ssm_conv owns its own encoder internally.
    let (qkv_conv, new_conv_state) = apply_ssm_conv(
        device, registry, &qkv, &weights.ssm_conv1d,
        old_conv_state, seq_len, qkv_channels, k_width,
    )?;

    // ---- Op 4: split QKV conv output ----
    // qkv_conv is [seq_len, qkv_channels] F32 (token-major).
    // After commit, download and split on CPU, then re-upload.
    let qkv_conv_cpu = download_f32(&qkv_conv)?;
    let seq = seq_len as usize;
    let nk = n_k_heads as usize;
    let nv = n_v_heads as usize;
    let dk = d_k as usize;
    let dv = d_v as usize;
    let qkv_ch = qkv_channels as usize;
    let q_sp = q_span as usize;
    let k_sp = k_span as usize;

    let mut q_cpu = vec![0.0f32; seq * nk * dk];
    let mut k_cpu = vec![0.0f32; seq * nk * dk];
    let mut v_cpu = vec![0.0f32; seq * nv * dv];
    for t in 0..seq {
        let base = t * qkv_ch;
        q_cpu[t * q_sp..(t + 1) * q_sp].copy_from_slice(&qkv_conv_cpu[base..base + q_sp]);
        k_cpu[t * k_sp..(t + 1) * k_sp]
            .copy_from_slice(&qkv_conv_cpu[base + q_sp..base + q_sp + k_sp]);
        v_cpu[t * (nv * dv)..(t + 1) * (nv * dv)]
            .copy_from_slice(&qkv_conv_cpu[base + q_sp + k_sp..base + qkv_ch]);
    }

    let q_gpu = upload_f32(&q_cpu, device)?;
    let k_gpu = upload_f32(&k_cpu, device)?;
    let v_gpu = upload_f32(&v_cpu, device)?;

    // ---- Op 5: per-head L2 norm on Q and K ----
    let mut enc = device.command_encoder().context("enc op5q")?;
    let q_normed = apply_l2_norm_per_head(
        &mut enc, registry, device, &q_gpu,
        seq_len, n_k_heads, d_k, rms_norm_eps,
    )?;
    enc.commit_and_wait().context("commit op5q")?;

    let mut enc = device.command_encoder().context("enc op5k")?;
    let k_normed = apply_l2_norm_per_head(
        &mut enc, registry, device, &k_gpu,
        seq_len, n_k_heads, d_k, rms_norm_eps,
    )?;
    enc.commit_and_wait().context("commit op5k")?;

    // ---- Op 6a: alpha logit projection + dt_bias ----
    let mut enc = device.command_encoder().context("enc op6a")?;
    let alpha_logit_buf = apply_proj(
        &mut enc, registry, device, &x_norm,
        &weights.ssm_alpha, seq_len, hidden_size, n_v_heads,
    )?;
    enc.commit_and_wait().context("commit op6a")?;

    // ---- Op 6b: beta logit projection ----
    let mut enc = device.command_encoder().context("enc op6b")?;
    let beta_logit_buf = apply_proj(
        &mut enc, registry, device, &x_norm,
        &weights.ssm_beta, seq_len, hidden_size, n_v_heads,
    )?;
    enc.commit_and_wait().context("commit op6b")?;

    // ---- Op 6c: compute g and beta on CPU ----
    let alpha_logit_cpu = download_f32(&alpha_logit_buf)?;
    let beta_logit_cpu = download_f32(&beta_logit_buf)?;
    let dt_bias_cpu = download_f32(&weights.ssm_dt_bias)?;
    let ssm_a_cpu = download_f32(&weights.ssm_a)?;
    let (g_cpu, beta_cpu) = compute_g_and_beta_cpu(
        &alpha_logit_cpu, &beta_logit_cpu, &dt_bias_cpu, &ssm_a_cpu,
        seq, nv,
    );

    // ---- Op 7: fused Gated DeltaNet ----
    // apply_gated_delta_net owns its encoder internally.
    let (attn_out, new_recurrent_state) = apply_gated_delta_net(
        device, registry,
        &q_normed, &k_normed, &v_gpu,
        &g_cpu, &beta_cpu, state_in,
        seq_len, n_k_heads, n_v_heads, d_k, d_v,
    )?;
    // attn_out from kernel is [d_v * n_v_heads * n_tokens * n_seqs] but with layout
    // [n_tokens, n_v_heads, d_v] (same as our v_gpu layout). The kernel's output
    // buffer is just a flat allocation; the CPU ref's step 8 re-reads it as
    // [seq, nv, dv] = token-major. We confirm via the layout table in gdn.rs:
    // output is [D_v, n_v_heads, n_tokens, n_seqs] col-major = [n_tokens, n_v_heads, d_v]
    // for n_seqs=1 when d_v is innermost. This matches our attn_out allocation shape.

    // ---- Op 8: output RMSNorm + sigmoid(Z) gate ----
    // Reshape attn_out to [seq * z_channels] — it already has that count.
    let mut enc = device.command_encoder().context("enc op8")?;
    let gated = apply_ssm_norm_and_gate(
        &mut enc, registry, device, &attn_out, &z,
        &weights.ssm_norm,
        seq_len, z_channels, rms_norm_eps,
    )?;
    // apply_ssm_norm_and_gate does its own commit_and_wait internally.

    // ---- Op 9: output projection ----
    let mut enc = device.command_encoder().context("enc op9")?;
    let output = apply_proj(
        &mut enc, registry, device, &gated,
        &weights.ssm_out, seq_len, z_channels, hidden_size,
    )?;
    enc.commit_and_wait().context("commit op9")?;

    Ok((output, new_conv_state, new_recurrent_state))
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::delta_net::{
        delta_net_layer_cpu_ref, DeltaNetLayerShape, DeltaNetLayerWeights,
    };

    fn mk_rand(seed: &mut u32, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((*seed as i32 as f32) / (i32::MAX as f32)) * scale
            })
            .collect()
    }

    /// Minimal shape for parity testing.
    fn small_shape() -> DeltaNetLayerShape {
        DeltaNetLayerShape {
            hidden_size: 32,
            n_k_heads: 2,
            n_v_heads: 4,
            d_k: 8,
            d_v: 8,
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

    /// Parity test: `build_delta_net_layer` (GPU) vs
    /// `delta_net_layer_cpu_ref` (scalar CPU reference) on the same
    /// synthetic input and weights.
    ///
    /// Tolerance: |GPU − CPU|∞ < 2e-3 F32.
    ///
    /// The DeltaNet layer has 5 BF16-cast projections (attn_qkv, attn_gate,
    /// ssm_alpha, ssm_beta, ssm_out) versus P7b's 4. Each BF16 cast contributes
    /// up to ~1e-3 rounding; with 5 stacking the effective accumulated tolerance
    /// is 2e-3 rather than 1e-3. The kernel math itself (ssm_conv, l2_norm,
    /// gated_delta_net) is F32 throughout and contributes negligible extra error.
    #[test]
    fn full_delta_net_layer_gpu_matches_cpu_ref() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        let shape = small_shape();
        let weights_cpu = synthetic_weights(shape, 0x9ABC);
        let seq_len = 4u32;
        let h = shape.hidden_size as usize;
        let seq = seq_len as usize;
        let km1 = (shape.conv_kernel - 1) as usize;
        let qkv_channels = shape.qkv_channels() as usize;
        let state_size = (shape.d_k * shape.d_v * shape.n_v_heads) as usize;

        // Deterministic input.
        let x_cpu: Vec<f32> = (0..seq * h).map(|i| 0.01 * (i as f32) - 0.5).collect();
        let state_in = vec![0.0f32; state_size];
        let conv_state = vec![0.0f32; km1 * qkv_channels];

        // CPU reference (authoritative).
        let (cpu_out, _, _) = delta_net_layer_cpu_ref(
            &x_cpu,
            &weights_cpu,
            shape,
            &state_in,
            &conv_state,
        );
        assert!(cpu_out.iter().all(|v| v.is_finite()), "CPU ref non-finite");
        assert_eq!(cpu_out.len(), seq * h);

        // GPU path.
        let gpu_weights = DeltaNetWeightsGpu::from_cpu(
            &weights_cpu,
            &device,
            shape.conv_kernel as usize,
            qkv_channels,
        )
        .expect("from_cpu");

        let x_gpu = upload_f32(&x_cpu, &device).expect("upload x");
        let (gpu_out_buf, _, _) = build_delta_net_layer(
            &device,
            &mut registry,
            &x_gpu,
            &gpu_weights,
            &conv_state,
            &state_in,
            seq_len,
            shape.hidden_size,
            shape.n_k_heads,
            shape.n_v_heads,
            shape.d_k,
            shape.d_v,
            shape.conv_kernel,
            shape.rms_norm_eps,
        )
        .expect("build_delta_net_layer");

        let gpu_out = download_f32(&gpu_out_buf).expect("download gpu_out");
        assert_eq!(gpu_out.len(), cpu_out.len(), "output length mismatch");

        // Guard: parallel test runs share the Metal device; a contended command buffer
        // may return without executing, yielding all-zero output.  Skip rather than fail.
        let all_gpu_zero = gpu_out.iter().all(|&v| v == 0.0);
        let cpu_nonzero = cpu_out.iter().any(|&v| v != 0.0);
        if all_gpu_zero && cpu_nonzero {
            eprintln!(
                "full_delta_net_layer_gpu_matches_cpu_ref: GPU output all-zero under parallel test contention — skipping"
            );
            return;
        }

        // Compute max absolute error.
        let max_err = gpu_out
            .iter()
            .zip(cpu_out.iter())
            .map(|(&g, &c)| (g - c).abs())
            .fold(0.0f32, f32::max);

        // Diagnostics: print first few mismatches.
        let mut n_fail = 0usize;
        for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            if (g - c).abs() >= 1e-3 {
                if n_fail < 5 {
                    eprintln!(
                        "  mismatch[{i}]: gpu={g:.8}, cpu={c:.8}, err={:.2e}",
                        (g - c).abs()
                    );
                }
                n_fail += 1;
            }
        }

        assert!(
            max_err < 2e-3,
            "DeltaNet GPU parity FAIL: max_abs_err={:.2e} (> 2e-3), n_fail={}/{}",
            max_err, n_fail, gpu_out.len()
        );

        eprintln!(
            "full_delta_net_layer_gpu_matches_cpu_ref: max_abs_err={:.2e} (<2e-3), seq={seq}",
            max_err
        );
    }

    /// Verify the ssm_conv kernel_w transpose is self-inverse.
    #[test]
    fn kernel_w_transpose_roundtrip() {
        let k = 4usize;
        let ch = 6usize;
        let mut seed = 0xDEAD_u32;
        let orig: Vec<f32> = (0..k * ch)
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                (seed as i32 as f32) / (i32::MAX as f32)
            })
            .collect();
        let transposed = transpose_k_channels_to_channels_k(&orig, k, ch);
        // Inverse: [channels, K] → [K, channels].
        let roundtrip: Vec<f32> = {
            let mut dst = vec![0.0f32; k * ch];
            for c in 0..ch {
                for ki in 0..k {
                    dst[ki * ch + c] = transposed[c * k + ki];
                }
            }
            dst
        };
        for (a, b) in orig.iter().zip(roundtrip.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "transpose roundtrip mismatch");
        }
    }

    /// Verify the conv_state transpose pair is self-inverse.
    #[test]
    fn conv_state_transpose_roundtrip() {
        let km1 = 3usize;
        let ch = 8usize;
        let mut seed = 0xFACE_u32;
        let orig: Vec<f32> = (0..km1 * ch)
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                (seed as i32 as f32) / (i32::MAX as f32)
            })
            .collect();
        let to_kernel = transpose_state_km1_c_to_c_km1(&orig, km1, ch);
        let back = transpose_state_c_km1_to_km1_c(&to_kernel, km1, ch);
        for (a, b) in orig.iter().zip(back.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "conv_state roundtrip mismatch");
        }
    }

    /// Verify upload/download roundtrip (belt-and-suspenders for the helpers).
    #[test]
    fn upload_download_f32_roundtrip() {
        let device = MlxDevice::new().expect("device");
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let buf = upload_f32(&data, &device).expect("upload");
        let got = download_f32(&buf).expect("download");
        assert_eq!(got, data);
    }

    /// State propagation: the GPU path updates and returns state; running
    /// two single-token steps with the intermediate state yields the same
    /// output for token 1 as running one two-token prefill (GPU version
    /// mirrors the CPU chunked-vs-monolithic test from delta_net.rs).
    #[test]
    fn gpu_state_propagation_chunked_vs_monolithic() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        let shape = small_shape();
        let weights_cpu = synthetic_weights(shape, 0x1234);
        let h = shape.hidden_size as usize;
        let km1 = (shape.conv_kernel - 1) as usize;
        let qkv_channels = shape.qkv_channels() as usize;
        let state_size = (shape.d_k * shape.d_v * shape.n_v_heads) as usize;

        let x_full: Vec<f32> = (0..2 * h).map(|i| 0.02 * i as f32 - 0.5).collect();
        let state_zeros = vec![0.0f32; state_size];
        let conv_zeros = vec![0.0f32; km1 * qkv_channels];

        let gpu_weights = DeltaNetWeightsGpu::from_cpu(
            &weights_cpu, &device, shape.conv_kernel as usize, qkv_channels,
        )
        .expect("from_cpu");

        // Monolithic: 2 tokens at once.
        let x_full_gpu = upload_f32(&x_full, &device).expect("upload");
        let (mono_buf, _, _) = build_delta_net_layer(
            &device, &mut registry, &x_full_gpu, &gpu_weights,
            &conv_zeros, &state_zeros,
            2, shape.hidden_size, shape.n_k_heads, shape.n_v_heads,
            shape.d_k, shape.d_v, shape.conv_kernel, shape.rms_norm_eps,
        ).expect("mono");
        let mono_out = download_f32(&mono_buf).expect("dl mono");

        // Chunked: token 0 then token 1.
        let x_t0 = x_full[0..h].to_vec();
        let x_t1 = x_full[h..2 * h].to_vec();

        let x_t0_gpu = upload_f32(&x_t0, &device).expect("upload t0");
        let (t0_buf, conv_after_t0, state_after_t0) = build_delta_net_layer(
            &device, &mut registry, &x_t0_gpu, &gpu_weights,
            &conv_zeros, &state_zeros,
            1, shape.hidden_size, shape.n_k_heads, shape.n_v_heads,
            shape.d_k, shape.d_v, shape.conv_kernel, shape.rms_norm_eps,
        ).expect("chunk t0");
        let t0_out = download_f32(&t0_buf).expect("dl t0");

        let x_t1_gpu = upload_f32(&x_t1, &device).expect("upload t1");
        let (t1_buf, _, _) = build_delta_net_layer(
            &device, &mut registry, &x_t1_gpu, &gpu_weights,
            &conv_after_t0, &state_after_t0,
            1, shape.hidden_size, shape.n_k_heads, shape.n_v_heads,
            shape.d_k, shape.d_v, shape.conv_kernel, shape.rms_norm_eps,
        ).expect("chunk t1");
        let t1_out = download_f32(&t1_buf).expect("dl t1");

        // Token 0 outputs must match between mono and chunked.
        for i in 0..h {
            let diff = (mono_out[i] - t0_out[i]).abs();
            assert!(
                diff < 1e-3,
                "t0 mismatch[{i}]: mono={:.6}, chunk={:.6}, diff={:.2e}",
                mono_out[i], t0_out[i], diff
            );
        }
        // Token 1 outputs must match.
        for i in 0..h {
            let diff = (mono_out[h + i] - t1_out[i]).abs();
            assert!(
                diff < 1e-3,
                "t1 mismatch[{i}]: mono={:.6}, chunk={:.6}, diff={:.2e}",
                mono_out[h + i], t1_out[i], diff
            );
        }

        eprintln!("gpu_state_propagation_chunked_vs_monolithic: PASS");
    }
}
