//! ViT forward-pass primitives (ADR-005 Phase 2c, Task #15 begins).
//!
//! The mmproj-hosted vision tower is a standard CLIP/SigLIP-style ViT:
//!
//!   patch_embd   ─ Conv2d(in=3, out=hidden, k=patch, s=patch, p=0)
//!   + pos_embd   ─ [N_patches, hidden]
//!   blk[0..L] ── LN → QKV → attn → proj → LN → MLP(gelu) → residual
//!   post_ln      ─ final LayerNorm
//!   projector    ─ MLP: hidden → text_hidden
//!
//! This module ports each stage as a pure-f32 CPU function first, testable
//! against synthetic weights (identity kernels, delta inputs). The GPU
//! port follows once CPU parity is locked against an mlx-lm Gemma 4
//! vision reference (live-model-gated).
//!
//! # Tensor-layout conventions (match llama.cpp's mmproj writer)
//!
//!   - `pixel_values`  — CHW, `[3, H, W]` f32, row-major within each channel,
//!     channels concatenated. Produced by `preprocess::preprocess_rgb_chw`.
//!   - `patch_embd_weight`  — `[hidden, 3, patch, patch]` in
//!     out-channel-major layout. Element index:
//!       `oc*(3*p*p) + ic*(p*p) + dy*p + dx`.
//!   - `patch_embd_bias`    — `[hidden]`, optional (None when absent).
//!   - `patch_embeddings`   — `[N_patches, hidden]` row-major, with patches
//!     enumerated row-by-row top-to-bottom (patch index =
//!     `py*num_patches_side + px`).
//!
//! # Not done in this iter
//!
//!   - GGUF weight loader (iter 28) — reads tensors out of the mmproj
//!     file and wraps them in typed slices. This module consumes `&[f32]`
//!     so it's decoupled from the loader.
//!   - Position embeddings, LN, attention, MLP — iter 29+.
//!   - GPU dispatch — deferred until CPU parity is locked.

#![allow(dead_code)]

use anyhow::{anyhow, Result};

/// Add a learned position-embedding tensor in-place to a patch-embedding
/// tensor. Both are `[N_patches, hidden]` row-major.
///
/// This is the trivial elementwise-add stage between `patch_embed_forward`
/// and the first transformer block.
///
/// # Errors
///
/// - shape mismatch between `patch_embeds` and `pos_embeds`.
pub fn position_embed_add(patch_embeds: &mut [f32], pos_embeds: &[f32]) -> Result<()> {
    if patch_embeds.len() != pos_embeds.len() {
        return Err(anyhow!(
            "position_embed_add: patch len {} != pos len {}",
            patch_embeds.len(),
            pos_embeds.len()
        ));
    }
    for (p, pe) in patch_embeds.iter_mut().zip(pos_embeds.iter()) {
        *p += *pe;
    }
    Ok(())
}

/// Forward-pass LayerNorm over the last dimension of a
/// `[..., hidden]`-shaped tensor.
///
/// For each `hidden`-element slice `x`:
///
/// ```text
/// μ    = mean(x)
/// σ²   = mean((x - μ)²)            # population variance (not sample)
/// y_i  = (x_i - μ) / sqrt(σ² + ε)
/// y_i  = y_i * γ_i + β_i           # affine scale/shift
/// ```
///
/// Matches PyTorch / HF `nn.LayerNorm` byte-for-byte when
/// `elementwise_affine=True`.
///
/// `input` is overwritten with the output. `gamma` and `beta` are each
/// `[hidden]`; pass both even when one is conceptually absent — zero bias
/// is `vec![0.0; hidden]`, unit gain is `vec![1.0; hidden]`.
///
/// # Errors
///
/// - `input.len() % hidden != 0`
/// - `gamma.len() != hidden`
/// - `beta.len() != hidden`
/// - `hidden == 0`
pub fn layer_norm_forward(
    input: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    hidden: usize,
    eps: f32,
) -> Result<()> {
    if hidden == 0 {
        return Err(anyhow!("layer_norm_forward: hidden must be > 0"));
    }
    if input.len() % hidden != 0 {
        return Err(anyhow!(
            "layer_norm_forward: input len {} not divisible by hidden {}",
            input.len(),
            hidden
        ));
    }
    if gamma.len() != hidden {
        return Err(anyhow!(
            "layer_norm_forward: gamma len {} != hidden {}",
            gamma.len(),
            hidden
        ));
    }
    if beta.len() != hidden {
        return Err(anyhow!(
            "layer_norm_forward: beta len {} != hidden {}",
            beta.len(),
            hidden
        ));
    }

    let n_rows = input.len() / hidden;
    let inv_hidden = 1.0f32 / (hidden as f32);
    for row in 0..n_rows {
        let off = row * hidden;
        let slice = &mut input[off..off + hidden];
        // Pass 1: mean.
        let mut sum = 0.0f32;
        for &v in slice.iter() {
            sum += v;
        }
        let mean = sum * inv_hidden;
        // Pass 2: population variance.
        let mut var_sum = 0.0f32;
        for &v in slice.iter() {
            let d = v - mean;
            var_sum += d * d;
        }
        let variance = var_sum * inv_hidden;
        let inv_std = 1.0f32 / (variance + eps).sqrt();
        // Pass 3: normalize + affine.
        for (i, v) in slice.iter_mut().enumerate() {
            *v = (*v - mean) * inv_std * gamma[i] + beta[i];
        }
    }
    Ok(())
}

/// Forward-pass RMSNorm over the last dimension of a `[..., hidden]`
/// tensor, single-parameter (gain only, no bias).
///
/// For each `hidden`-element slice `x`:
///
/// ```text
/// rms = sqrt(mean(x²) + ε)
/// y_i = x_i / rms * gamma_i
/// ```
///
/// Matches PyTorch / HF `nn.RMSNorm(normalized_shape, eps,
/// elementwise_affine=True)` byte-for-byte (the formulation introduced in
/// Llama's pre-norm + carried through SigLIP2, Gemma 4 vision, and the
/// Gemma / Llama / Mistral / Qwen text stacks).
///
/// `input` is overwritten with the output. `gamma` is `[hidden]`; there
/// is no `beta` (RMSNorm has no bias by design — one of its ergonomic
/// wins over LayerNorm).
///
/// # Why Gemma 4 uses RMSNorm despite the `ln1`/`ln2` tensor names
///
/// The llama.cpp mmproj writer inherits CLIP's naming even when the
/// underlying model family switched from LayerNorm to RMSNorm.
/// Gemma 4's vision tower ships `ln1.weight` + `ln2.weight` +
/// `ffn_norm.weight` + `post_ffw_norm.weight` but NO matching
/// `.bias` tensors — the single-parameter signature is the RMSNorm
/// tell. Forward-pass dispatch branches on `ArchProfile`: Gemma 4 routes
/// these to `rms_norm_forward`, CLIP-classic routes to `layer_norm_forward`.
///
/// # Errors
///
/// - `input.len() % hidden != 0`
/// - `gamma.len() != hidden`
/// - `hidden == 0`
pub fn rms_norm_forward(
    input: &mut [f32],
    gamma: &[f32],
    hidden: usize,
    eps: f32,
) -> Result<()> {
    if hidden == 0 {
        return Err(anyhow!("rms_norm_forward: hidden must be > 0"));
    }
    if input.len() % hidden != 0 {
        return Err(anyhow!(
            "rms_norm_forward: input len {} not divisible by hidden {}",
            input.len(),
            hidden
        ));
    }
    if gamma.len() != hidden {
        return Err(anyhow!(
            "rms_norm_forward: gamma len {} != hidden {}",
            gamma.len(),
            hidden
        ));
    }

    let n_rows = input.len() / hidden;
    let inv_hidden = 1.0f32 / (hidden as f32);
    for row in 0..n_rows {
        let off = row * hidden;
        let slice = &mut input[off..off + hidden];
        // Pass 1: sum of squares.
        let mut sq_sum = 0.0f32;
        for &v in slice.iter() {
            sq_sum += v * v;
        }
        let inv_rms = 1.0f32 / (sq_sum * inv_hidden + eps).sqrt();
        // Pass 2: normalize + gain.
        for (i, v) in slice.iter_mut().enumerate() {
            *v = *v * inv_rms * gamma[i];
        }
    }
    Ok(())
}

/// Apply RMSNorm per-head to a `[batch, num_heads, head_dim]` tensor
/// (equivalently `[batch, hidden]` with `hidden = num_heads × head_dim`).
///
/// Gemma 4's SigLIP vision tower applies RMSNorm on each head's 72-dim
/// Q and K slices after QKV projection, before the scaled-dot-product.
/// The gain vector is shared across heads — a single `[head_dim]`
/// tensor (`attn_q_norm.weight [72]` / `attn_k_norm.weight [72]`) is
/// broadcast across all `batch × num_heads` slices.
///
/// Implementation: since `[batch, num_heads, head_dim]` in row-major is
/// byte-identical to `[batch × num_heads, head_dim]`, this is just
/// `rms_norm_forward` with `hidden = head_dim`. The wrapper exists to
/// validate shape + document the semantic.
///
/// V is NOT normalized — only Q and K. Call twice with the respective
/// gain vectors.
///
/// `input` is overwritten in-place. Shape `[batch, num_heads, head_dim]`.
/// `gamma` is `[head_dim]`.
///
/// # Errors
///
/// - `input.len() != batch * num_heads * head_dim`
/// - `gamma.len() != head_dim`
/// - any dim is 0
pub fn per_head_rms_norm_forward(
    input: &mut [f32],
    gamma: &[f32],
    batch: usize,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || head_dim == 0 {
        return Err(anyhow!(
            "per_head_rms_norm_forward: batch ({}), num_heads ({}), head_dim ({}) must all be > 0",
            batch, num_heads, head_dim
        ));
    }
    let expected_input_len = batch * num_heads * head_dim;
    if input.len() != expected_input_len {
        return Err(anyhow!(
            "per_head_rms_norm_forward: input len {} != batch*num_heads*head_dim = {}*{}*{} = {}",
            input.len(),
            batch,
            num_heads,
            head_dim,
            expected_input_len
        ));
    }
    if gamma.len() != head_dim {
        return Err(anyhow!(
            "per_head_rms_norm_forward: gamma len {} != head_dim {}",
            gamma.len(),
            head_dim
        ));
    }
    // Row-major byte-identity means we can dispatch to the 2D rms_norm
    // with hidden=head_dim; each row is exactly one (batch, head) slice.
    rms_norm_forward(input, gamma, head_dim, eps)
}

/// Numerically-stable softmax over the last dimension of a `[..., hidden]`
/// tensor. Subtracts the per-row max before `exp` so that even rows with
/// large positive entries don't overflow f32 (max finite exp input ≈ 88.7).
///
/// Matches PyTorch `torch.softmax(x, dim=-1)` byte-for-byte.
///
/// `input` is overwritten with the output (in-place).
///
/// # Errors
///
/// - `hidden == 0`
/// - `input.len() % hidden != 0`
pub fn softmax_last_dim(input: &mut [f32], hidden: usize) -> Result<()> {
    if hidden == 0 {
        return Err(anyhow!("softmax_last_dim: hidden must be > 0"));
    }
    if input.len() % hidden != 0 {
        return Err(anyhow!(
            "softmax_last_dim: input len {} not divisible by hidden {}",
            input.len(),
            hidden
        ));
    }
    let n_rows = input.len() / hidden;
    for row in 0..n_rows {
        let off = row * hidden;
        let slice = &mut input[off..off + hidden];
        // Pass 1: max.
        let mut m = f32::NEG_INFINITY;
        for &v in slice.iter() {
            if v > m {
                m = v;
            }
        }
        // Pass 2: exp(x - max) + accumulate sum. `m` is finite-guaranteed
        // as long as the row has at least one finite element; an all-NaN
        // or all -inf row produces NaN softmax, which matches PyTorch.
        let mut sum = 0.0f32;
        for v in slice.iter_mut() {
            let e = (*v - m).exp();
            *v = e;
            sum += e;
        }
        // Pass 3: divide. If sum is 0 (all -inf input) we produce NaN,
        // matching PyTorch.
        let inv = 1.0f32 / sum;
        for v in slice.iter_mut() {
            *v *= inv;
        }
    }
    Ok(())
}

/// GELU activation, tanh-approximation form.
///
/// ```text
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// ```
///
/// Matches PyTorch's `approximate="tanh"` mode, HuggingFace's
/// `gelu_new` / `gelu_pytorch_tanh`, and the activation that BERT,
/// GPT-2, and the Gemma 4 vision tower all use. Exact GELU (via erf)
/// agrees to within ~2e-4; the tanh form is the one serialized weights
/// were trained with, so it's the correct reference.
///
/// In-place on `input`.
pub fn gelu_tanh_approx(input: &mut [f32]) {
    // sqrt(2 / π) pre-computed to f32 precision.
    const C: f32 = 0.7978845608028654_f32;
    const K: f32 = 0.044715_f32;
    for v in input.iter_mut() {
        let x = *v;
        let inner = C * (x + K * x * x * x);
        *v = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// Apply the patch-embedding Conv2d to a CHW pixel tensor.
///
/// Mathematically equivalent to:
///
/// ```text
/// for py in 0..num_patches_side:
///   for px in 0..num_patches_side:
///     for oc in 0..hidden:
///       acc = bias[oc] if bias is Some else 0
///       for ic in 0..3:
///         for dy in 0..patch_size:
///           for dx in 0..patch_size:
///             y = py*patch_size + dy
///             x = px*patch_size + dx
///             acc += pixels[ic*H*W + y*W + x] * weight[oc*3*p*p + ic*p*p + dy*p + dx]
///       out[(py*num_patches_side + px)*hidden + oc] = acc
/// ```
///
/// Returned `Vec<f32>` has length `num_patches² × hidden`.
///
/// # Errors
///
/// - `pixel_values.len() != 3 * image_size²`
/// - `patch_embd_weight.len() != hidden * 3 * patch_size²`
/// - `patch_embd_bias.map(|b| b.len() != hidden).unwrap_or(false)`
/// - `image_size % patch_size != 0`
///
/// Algorithm cost: O(N_patches × hidden × 3 × patch² ) — for Gemma 4
/// (64×64 patches × 1152 hidden × 3 × 196) ~= 2.7 GFLOP per image. This
/// is the CPU-correctness reference; the GPU port is where performance
/// lives.
///
/// # Storage-layout note (iter 33 real-data check)
///
/// Gemma 4's `v.patch_embd.weight` is stored as a 2D tensor
/// `[hidden, 3·p·p]` (shape `[1152, 768]` for Gemma 4). This is
/// byte-identical to a 4D `[hidden, 3, p, p]` tensor in row-major
/// order — the inner `3·p·p` dim iterates `ic*(p*p) + dy*p + dx`
/// exactly like the 4D layout's `[ic, dy, dx]` nested iteration. So
/// the formula above and `weight[oc*3*p*p + ic*p*p + dy*p + dx]`
/// produces correct output against either layout's raw bytes
/// without reshape.
pub fn patch_embed_forward(
    pixel_values: &[f32],
    patch_embd_weight: &[f32],
    patch_embd_bias: Option<&[f32]>,
    image_size: u32,
    patch_size: u32,
    hidden: u32,
) -> Result<Vec<f32>> {
    // --- Shape validation ---
    if patch_size == 0 {
        return Err(anyhow!("patch_size must be > 0"));
    }
    if image_size % patch_size != 0 {
        return Err(anyhow!(
            "image_size ({}) must be divisible by patch_size ({})",
            image_size,
            patch_size
        ));
    }
    let num_patches_side = image_size / patch_size;
    let p = patch_size as usize;
    let h = hidden as usize;
    let hw = (image_size as usize) * (image_size as usize);
    let expected_pixels = 3 * hw;
    if pixel_values.len() != expected_pixels {
        return Err(anyhow!(
            "pixel_values len {} != expected 3*{}*{} = {}",
            pixel_values.len(),
            image_size,
            image_size,
            expected_pixels
        ));
    }
    let expected_w = h * 3 * p * p;
    if patch_embd_weight.len() != expected_w {
        return Err(anyhow!(
            "patch_embd_weight len {} != expected {}*3*{}*{} = {}",
            patch_embd_weight.len(),
            hidden,
            patch_size,
            patch_size,
            expected_w
        ));
    }
    if let Some(b) = patch_embd_bias {
        if b.len() != h {
            return Err(anyhow!(
                "patch_embd_bias len {} != hidden {}",
                b.len(),
                hidden
            ));
        }
    }

    let w = image_size as usize;
    let nps = num_patches_side as usize;
    let num_patches = nps * nps;
    let mut out = vec![0f32; num_patches * h];

    // Stride constants for weight indexing.
    let ws_oc = 3 * p * p; // stride between out-channels
    let ws_ic = p * p; //     stride between in-channels within one out-channel
    let ws_y = p; //          stride between rows within one in-channel slice

    // Stride constants for pixel_values (CHW).
    let ps_c = hw; // stride between channels
    let ps_y = w; //  stride between rows within one channel

    for py in 0..nps {
        let y0 = py * p;
        for px in 0..nps {
            let x0 = px * p;
            let patch_idx = py * nps + px;
            let out_base = patch_idx * h;
            for oc in 0..h {
                let mut acc: f32 = match patch_embd_bias {
                    Some(b) => b[oc],
                    None => 0.0,
                };
                let w_base = oc * ws_oc;
                for ic in 0..3usize {
                    let w_ic_base = w_base + ic * ws_ic;
                    let p_ic_base = ic * ps_c;
                    for dy in 0..p {
                        let w_row_base = w_ic_base + dy * ws_y;
                        let p_row_base = p_ic_base + (y0 + dy) * ps_y + x0;
                        for dx in 0..p {
                            acc += pixel_values[p_row_base + dx] * patch_embd_weight[w_row_base + dx];
                        }
                    }
                }
                out[out_base + oc] = acc;
            }
        }
    }
    Ok(out)
}

/// Dense linear layer forward: `y = x @ W.T` (+ optional bias).
///
/// This is the workhorse primitive for every projection in the ViT
/// (Q/K/V, attn_output, FFN up/gate/down, and `mm.0` projector).
/// PyTorch's `nn.Linear(in, out)` stores `weight` as `[out, in]` so
/// `y[n, o] = Σᵢ x[n, i] * weight[o, i]`; this function consumes the
/// exact same memory layout, no transpose required at the call site.
///
/// `input` shape: `[batch, in_features]` row-major.
/// `weight` shape: `[out_features, in_features]` row-major.
/// `bias`: `Some([out_features])` or `None`.
/// Output shape: `[batch, out_features]` row-major, freshly allocated.
///
/// # Errors
///
/// - `input.len() != batch * in_features`
/// - `weight.len() != out_features * in_features`
/// - `bias.map(|b| b.len() != out_features).unwrap_or(false)`
/// - any dim is 0
///
/// Algorithm is the naive triple-nested-loop GEMM. CPU correctness
/// reference only; the GPU port substitutes `mlx_native::ops::mul_mm`
/// or BLAS. For Gemma 4's per-block Q projection (196 × 1152 × 1152)
/// this is ~450 MFLOP — CPU runs it in ~50ms unoptimized, acceptable
/// for the reference path.
pub fn linear_forward(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    if batch == 0 || in_features == 0 || out_features == 0 {
        return Err(anyhow!(
            "linear_forward: batch ({}), in_features ({}), out_features ({}) must all be > 0",
            batch,
            in_features,
            out_features
        ));
    }
    if input.len() != batch * in_features {
        return Err(anyhow!(
            "linear_forward: input len {} != batch*in_features = {}*{} = {}",
            input.len(),
            batch,
            in_features,
            batch * in_features
        ));
    }
    if weight.len() != out_features * in_features {
        return Err(anyhow!(
            "linear_forward: weight len {} != out_features*in_features = {}*{} = {}",
            weight.len(),
            out_features,
            in_features,
            out_features * in_features
        ));
    }
    if let Some(b) = bias {
        if b.len() != out_features {
            return Err(anyhow!(
                "linear_forward: bias len {} != out_features {}",
                b.len(),
                out_features
            ));
        }
    }

    let mut out = vec![0f32; batch * out_features];
    for n in 0..batch {
        let x_off = n * in_features;
        let y_off = n * out_features;
        for o in 0..out_features {
            let w_off = o * in_features;
            let mut acc: f32 = match bias {
                Some(b) => b[o],
                None => 0.0,
            };
            for i in 0..in_features {
                acc += input[x_off + i] * weight[w_off + i];
            }
            out[y_off + o] = acc;
        }
    }
    Ok(out)
}

/// Apply the three Q/K/V projections from a ViT attention block. Returns
/// `(q, k, v)` each of shape `[batch, hidden]` (for Gemma 4 at
/// `hidden=1152`; head reshape is a separate downstream op).
///
/// Bias is `None` across the board for Gemma 4's SigLIP vision tower.
/// CLIP-classic producers carry biases; when that arch lands, this
/// signature gains `Option<&[f32]>` per-projection (or the dispatch
/// wraps `linear_forward` directly at the call site).
///
/// # Errors
///
/// Propagated from `linear_forward` for any of the three calls.
pub fn qkv_projection_forward(
    input: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    v_weight: &[f32],
    batch: usize,
    hidden: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let q = linear_forward(input, q_weight, None, batch, hidden, hidden)
        .map_err(|e| anyhow!("qkv_projection_forward Q: {e}"))?;
    let k = linear_forward(input, k_weight, None, batch, hidden, hidden)
        .map_err(|e| anyhow!("qkv_projection_forward K: {e}"))?;
    let v = linear_forward(input, v_weight, None, batch, hidden, hidden)
        .map_err(|e| anyhow!("qkv_projection_forward V: {e}"))?;
    Ok((q, k, v))
}

/// Drive `patch_embed_forward` from a `LoadedMmprojWeights` + parsed
/// `MmprojConfig`, reading the `v.patch_embd.weight` buffer directly
/// off the GPU. This is the hook iter 34+ will call from the handler's
/// `process_multimodal_content` path once the full ViT forward is
/// wired; iter 33 only validates that the call-path returns sensible
/// patch embeddings on real Gemma 4 weights.
///
/// Note: reads the Metal buffer back to CPU via `MlxBuffer::as_slice`
/// — which is a zero-copy view into the unified-memory region on
/// Apple Silicon (no H→D copy). The CPU forward is the correctness
/// reference; the GPU dispatch port follows when the whole ViT's
/// CPU parity is locked.
pub fn patch_embed_from_mmproj_weights(
    pixel_values: &[f32],
    weights: &super::mmproj_weights::LoadedMmprojWeights,
    cfg: &super::mmproj::MmprojConfig,
) -> Result<Vec<f32>> {
    let patch_buf = weights
        .patch_embd_weight()
        .map_err(|e| anyhow!("patch_embd_forward: {e}"))?;
    let weight: &[f32] = patch_buf
        .as_slice::<f32>()
        .map_err(|e| anyhow!("patch_embd_forward as_slice f32: {e}"))?;
    patch_embed_forward(
        pixel_values,
        weight,
        None, // Gemma 4 vision tower has no patch_embd bias
        cfg.image_size,
        cfg.patch_size,
        cfg.hidden_size,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a patch_embd weight tensor where `weight[oc, ic, dy, dx]`
    /// equals the supplied closure result. Layout matches
    /// `patch_embed_forward`'s expected shape.
    fn make_weight<F: FnMut(usize, usize, usize, usize) -> f32>(
        hidden: usize,
        patch_size: usize,
        mut f: F,
    ) -> Vec<f32> {
        let mut w = vec![0f32; hidden * 3 * patch_size * patch_size];
        for oc in 0..hidden {
            for ic in 0..3 {
                for dy in 0..patch_size {
                    for dx in 0..patch_size {
                        let idx = oc * 3 * patch_size * patch_size
                            + ic * patch_size * patch_size
                            + dy * patch_size
                            + dx;
                        w[idx] = f(oc, ic, dy, dx);
                    }
                }
            }
        }
        w
    }

    /// Build a CHW pixel tensor from a closure `pixel(c, y, x) -> f32`.
    fn make_pixels<F: FnMut(usize, usize, usize) -> f32>(
        image_size: usize,
        mut f: F,
    ) -> Vec<f32> {
        let mut p = vec![0f32; 3 * image_size * image_size];
        for c in 0..3 {
            for y in 0..image_size {
                for x in 0..image_size {
                    let idx = c * image_size * image_size + y * image_size + x;
                    p[idx] = f(c, y, x);
                }
            }
        }
        p
    }

    #[test]
    fn delta_kernel_copies_top_left_pixel_per_patch() {
        // hidden=3. weight[oc, ic, dy, dx] = 1 iff (oc == ic && dy == 0 && dx == 0), else 0.
        // No bias. Image 8×8, patch 4 → 2×2 patches. Each patch output
        // should equal the top-left pixel of that patch in the matching
        // channel.
        let img = 8;
        let patch = 4;
        let hidden = 3;
        let weight = make_weight(hidden, patch, |oc, ic, dy, dx| {
            if oc == ic && dy == 0 && dx == 0 {
                1.0
            } else {
                0.0
            }
        });
        // Put a unique value at each pixel so we can assert exactly.
        let pixels = make_pixels(img, |c, y, x| (c * 100 + y * 10 + x) as f32);
        let out = patch_embed_forward(
            &pixels,
            &weight,
            None,
            img as u32,
            patch as u32,
            hidden as u32,
        )
        .unwrap();
        assert_eq!(out.len(), 2 * 2 * hidden);
        // Patch (0,0) — top-left of each channel at (0,0).
        assert_eq!(out[0 * hidden + 0], 0.0); // c=0 pixel (0,0) = 0
        assert_eq!(out[0 * hidden + 1], 100.0); // c=1 pixel (0,0) = 100
        assert_eq!(out[0 * hidden + 2], 200.0); // c=2 pixel (0,0) = 200
        // Patch (0,1) — top-left at (0,4).
        assert_eq!(out[1 * hidden + 0], 4.0); // c=0 pixel (0,4) = 4
        assert_eq!(out[1 * hidden + 1], 104.0);
        assert_eq!(out[1 * hidden + 2], 204.0);
        // Patch (1,0) — top-left at (4,0).
        assert_eq!(out[2 * hidden + 0], 40.0);
        assert_eq!(out[2 * hidden + 1], 140.0);
        assert_eq!(out[2 * hidden + 2], 240.0);
        // Patch (1,1) — top-left at (4,4).
        assert_eq!(out[3 * hidden + 0], 44.0);
        assert_eq!(out[3 * hidden + 1], 144.0);
        assert_eq!(out[3 * hidden + 2], 244.0);
    }

    #[test]
    fn all_ones_kernel_produces_patch_sum() {
        // hidden=1. All weights = 1 → output = sum of all pixels in patch
        // window across all 3 channels.
        let img: usize = 4;
        let patch: usize = 2;
        let hidden: usize = 1;
        let weight = vec![1f32; hidden * 3 * patch * patch];
        // Each channel is uniform: c=0 → 1, c=1 → 2, c=2 → 3. One 2×2
        // patch sums 4 pixels per channel × 3 channels = 4*(1+2+3) = 24.
        let pixels = make_pixels(img, |c, _y, _x| (c as f32) + 1.0);
        let out = patch_embed_forward(
            &pixels,
            &weight,
            None,
            img as u32,
            patch as u32,
            hidden as u32,
        )
        .unwrap();
        assert_eq!(out.len(), 4 * 1); // 2×2 patches × 1 hidden
        for (i, v) in out.iter().enumerate() {
            assert!(
                (*v - 24.0).abs() < 1e-5,
                "patch {}: expected 24, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn bias_is_added_once_per_output_element() {
        // hidden=2, all weights = 0, bias = [10, 20]. Output should be
        // bias repeated for every patch.
        let img: usize = 4;
        let patch: usize = 2;
        let hidden: usize = 2;
        let weight = vec![0f32; hidden * 3 * patch * patch];
        let bias = vec![10f32, 20.0];
        let pixels = vec![999f32; 3 * img * img]; // garbage input, weights zero them
        let out = patch_embed_forward(
            &pixels,
            &weight,
            Some(&bias),
            img as u32,
            patch as u32,
            hidden as u32,
        )
        .unwrap();
        assert_eq!(out.len(), 4 * 2);
        for i in 0..4 {
            assert!((out[i * 2 + 0] - 10.0).abs() < 1e-5);
            assert!((out[i * 2 + 1] - 20.0).abs() < 1e-5);
        }
    }

    #[test]
    fn single_patch_covers_whole_image() {
        // image_size == patch_size → 1 patch covering everything.
        let img: usize = 4;
        let patch: usize = 4;
        let hidden: usize = 1;
        // Weight: all ones on channel 0 top-left pixel only.
        let weight = make_weight(hidden, patch, |_oc, ic, dy, dx| {
            if ic == 0 && dy == 0 && dx == 0 {
                1.0
            } else {
                0.0
            }
        });
        let pixels = make_pixels(img, |c, _, _| (c + 1) as f32);
        let out = patch_embed_forward(
            &pixels,
            &weight,
            None,
            img as u32,
            patch as u32,
            hidden as u32,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 1.0); // c=0 value everywhere = 1
    }

    #[test]
    fn rejects_mismatched_pixel_len() {
        let weight = vec![0f32; 1 * 3 * 2 * 2];
        let pixels = vec![0f32; 10]; // wrong size
        let err = patch_embed_forward(&pixels, &weight, None, 4, 2, 1).unwrap_err();
        assert!(format!("{err}").contains("pixel_values len"));
    }

    #[test]
    fn rejects_mismatched_weight_len() {
        let pixels = vec![0f32; 3 * 4 * 4];
        let weight = vec![0f32; 5]; // wrong size
        let err = patch_embed_forward(&pixels, &weight, None, 4, 2, 1).unwrap_err();
        assert!(format!("{err}").contains("patch_embd_weight len"));
    }

    #[test]
    fn rejects_bias_length_mismatch() {
        let weight = vec![0f32; 2 * 3 * 2 * 2];
        let pixels = vec![0f32; 3 * 4 * 4];
        let bias = vec![0f32; 99];
        let err = patch_embed_forward(&pixels, &weight, Some(&bias), 4, 2, 2).unwrap_err();
        assert!(format!("{err}").contains("patch_embd_bias len"));
    }

    #[test]
    fn rejects_non_divisible_image_size() {
        let weight = vec![0f32; 1 * 3 * 3 * 3];
        let pixels = vec![0f32; 3 * 5 * 5];
        let err = patch_embed_forward(&pixels, &weight, None, 5, 3, 1).unwrap_err();
        assert!(format!("{err}").contains("divisible"));
    }

    #[test]
    fn rejects_zero_patch_size() {
        let weight = vec![0f32; 0];
        let pixels = vec![0f32; 3 * 4 * 4];
        let err = patch_embed_forward(&pixels, &weight, None, 4, 0, 1).unwrap_err();
        assert!(format!("{err}").contains("patch_size"));
    }

    // -----------------------------------------------------------------------
    // position_embed_add
    // -----------------------------------------------------------------------

    #[test]
    fn position_embed_add_is_elementwise() {
        let mut patch = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pos = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        position_embed_add(&mut patch, &pos).unwrap();
        let expected = [1.1f32, 2.2, 3.3, 4.4, 5.5, 6.6];
        for (got, want) in patch.iter().zip(expected.iter()) {
            assert!((*got - *want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn position_embed_add_rejects_shape_mismatch() {
        let mut patch = vec![0.0f32; 10];
        let pos = vec![0.0f32; 11];
        let err = position_embed_add(&mut patch, &pos).unwrap_err();
        assert!(format!("{err}").contains("!= pos len"));
    }

    #[test]
    fn position_embed_add_zero_pos_is_identity() {
        let mut patch = vec![0.5f32, -0.3, 1.7, -2.1];
        let snapshot = patch.clone();
        let pos = vec![0.0f32; 4];
        position_embed_add(&mut patch, &pos).unwrap();
        assert_eq!(patch, snapshot);
    }

    // -----------------------------------------------------------------------
    // layer_norm_forward
    // -----------------------------------------------------------------------

    #[test]
    fn layer_norm_constant_row_goes_to_zero_then_beta() {
        // Constant row has var=0 and mean=the constant; (x-mean)=0 so
        // normalized = 0; output = 0 * gamma + beta = beta.
        let mut x = vec![7.0f32; 4];
        let gamma = vec![1.0f32; 4];
        let beta = vec![0.1f32, 0.2, 0.3, 0.4];
        layer_norm_forward(&mut x, &gamma, &beta, 4, 1e-6).unwrap();
        for (got, want) in x.iter().zip(beta.iter()) {
            assert!((*got - *want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn layer_norm_pytorch_reference_values() {
        // x = [1, 2, 3, 4], mean=2.5, var = mean([1.5², 0.5², 0.5², 1.5²])
        //                             = mean([2.25, 0.25, 0.25, 2.25]) = 1.25
        // inv_std = 1/sqrt(1.25 + 1e-5) ≈ 0.8944270...
        // normalized = [-1.3416, -0.4472, 0.4472, 1.3416]
        // With γ=1, β=0 these are the outputs.
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        let beta = vec![0.0f32; 4];
        layer_norm_forward(&mut x, &gamma, &beta, 4, 1e-5).unwrap();
        let expected = [-1.3416408f32, -0.4472136, 0.4472136, 1.3416408];
        for (got, want) in x.iter().zip(expected.iter()) {
            assert!((*got - *want).abs() < 1e-3, "got {got}, want {want}");
        }
    }

    #[test]
    fn layer_norm_applies_affine_scale_and_shift() {
        // After normalize, multiply by γ=[2,2,2,2] then add β=[10,20,30,40].
        // Normalized [1,2,3,4] = [-1.3416, -0.4472, 0.4472, 1.3416].
        // * 2      = [-2.683, -0.894, 0.894, 2.683]
        // + β      = [7.317, 19.106, 30.894, 42.683]
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![2.0f32; 4];
        let beta = vec![10.0f32, 20.0, 30.0, 40.0];
        layer_norm_forward(&mut x, &gamma, &beta, 4, 1e-5).unwrap();
        let expected = [7.3167f32, 19.1056, 30.8944, 42.6833];
        for (got, want) in x.iter().zip(expected.iter()) {
            assert!((*got - *want).abs() < 1e-2, "got {got}, want {want}");
        }
    }

    #[test]
    fn layer_norm_normalizes_multiple_rows_independently() {
        // Two rows, completely different scales. Each row should
        // independently normalize to zero-mean-unit-variance.
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0];
        let gamma = vec![1.0f32; 4];
        let beta = vec![0.0f32; 4];
        layer_norm_forward(&mut x, &gamma, &beta, 4, 1e-5).unwrap();
        // Both rows should yield the exact same normalized shape because
        // the second is a 100× scaling of the first (mean and variance
        // scale together).
        for i in 0..4 {
            assert!(
                (x[i] - x[4 + i]).abs() < 1e-3,
                "row 0 element {} = {} != row 1 = {}",
                i,
                x[i],
                x[4 + i]
            );
        }
    }

    #[test]
    fn layer_norm_mean_after_is_approximately_zero() {
        let mut x = vec![0.5f32, 1.7, -2.3, 4.1, -0.8, 3.2];
        let gamma = vec![1.0f32; 6];
        let beta = vec![0.0f32; 6];
        layer_norm_forward(&mut x, &gamma, &beta, 6, 1e-5).unwrap();
        let mean: f32 = x.iter().sum::<f32>() / 6.0;
        assert!(mean.abs() < 1e-5, "post-LN mean = {mean}");
    }

    #[test]
    fn layer_norm_rejects_hidden_zero() {
        let mut x = vec![1.0f32; 4];
        let err = layer_norm_forward(&mut x, &[], &[], 0, 1e-5).unwrap_err();
        assert!(format!("{err}").contains("hidden must be > 0"));
    }

    #[test]
    fn layer_norm_rejects_non_divisible_input_len() {
        let mut x = vec![1.0f32; 7];
        let gamma = vec![1.0f32; 3];
        let beta = vec![0.0f32; 3];
        let err = layer_norm_forward(&mut x, &gamma, &beta, 3, 1e-5).unwrap_err();
        assert!(format!("{err}").contains("not divisible"));
    }

    #[test]
    fn layer_norm_rejects_wrong_gamma_len() {
        let mut x = vec![1.0f32; 4];
        let gamma = vec![1.0f32; 3];
        let beta = vec![0.0f32; 4];
        let err = layer_norm_forward(&mut x, &gamma, &beta, 4, 1e-5).unwrap_err();
        assert!(format!("{err}").contains("gamma len"));
    }

    #[test]
    fn layer_norm_rejects_wrong_beta_len() {
        let mut x = vec![1.0f32; 4];
        let gamma = vec![1.0f32; 4];
        let beta = vec![0.0f32; 3];
        let err = layer_norm_forward(&mut x, &gamma, &beta, 4, 1e-5).unwrap_err();
        assert!(format!("{err}").contains("beta len"));
    }

    #[test]
    fn layer_norm_does_not_divide_by_zero_when_variance_is_zero() {
        // All-ones row → variance = 0. With eps > 0 this normalizes
        // cleanly; without eps it would divide by zero and NaN.
        let mut x = vec![5.0f32; 8];
        let gamma = vec![1.0f32; 8];
        let beta = vec![0.0f32; 8];
        layer_norm_forward(&mut x, &gamma, &beta, 8, 1e-5).unwrap();
        for v in &x {
            assert!(v.is_finite(), "got non-finite {}", v);
            assert!(v.abs() < 1e-5, "constant row should normalize to 0, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // per_head_rms_norm_forward (iter 36 — Gemma 4 Q/K per-head norm)
    // -----------------------------------------------------------------------

    #[test]
    fn per_head_rms_norm_with_unit_gain_normalizes_each_head_slice_independently() {
        // batch=2, num_heads=3, head_dim=4. Per-(b, h) RMS-normalized,
        // each row ends at mean(x²)≈1 independent of the others.
        let batch = 2;
        let num_heads = 3;
        let head_dim = 4;
        // Different magnitude per head so we can see independence.
        let mut input = vec![0f32; batch * num_heads * head_dim];
        for b in 0..batch {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    input[b * (num_heads * head_dim) + h * head_dim + d] =
                        ((h + 1) * 100) as f32 + (d as f32);
                }
            }
        }
        let gamma = vec![1.0f32; head_dim];
        per_head_rms_norm_forward(&mut input, &gamma, batch, num_heads, head_dim, 1e-6)
            .unwrap();
        // Every head slice should have mean(x²) ≈ 1 since γ=1.
        for b in 0..batch {
            for h in 0..num_heads {
                let off = b * (num_heads * head_dim) + h * head_dim;
                let slice = &input[off..off + head_dim];
                let ms: f32 = slice.iter().map(|v| v * v).sum::<f32>() / (head_dim as f32);
                assert!(
                    (ms - 1.0).abs() < 1e-3,
                    "head ({},{}): ms = {}",
                    b, h, ms
                );
            }
        }
    }

    #[test]
    fn per_head_rms_norm_broadcasts_same_gamma_across_heads() {
        // All heads same constant input. Per-head RMSNorm → each
        // slice independently normalized → output = γ (uniform input
        // with RMSNorm produces [1,1,...] which γ scales).
        let batch = 1;
        let num_heads = 3;
        let head_dim = 4;
        let mut input = vec![5.0f32; batch * num_heads * head_dim];
        let gamma = vec![0.5f32, 1.0, 2.0, 4.0];
        per_head_rms_norm_forward(&mut input, &gamma, batch, num_heads, head_dim, 1e-6)
            .unwrap();
        for h in 0..num_heads {
            let off = h * head_dim;
            for (i, g) in gamma.iter().enumerate() {
                assert!(
                    (input[off + i] - g).abs() < 1e-4,
                    "head {}: got {}, want {}",
                    h,
                    input[off + i],
                    g
                );
            }
        }
    }

    #[test]
    fn per_head_rms_norm_rejects_zero_dims() {
        let err =
            per_head_rms_norm_forward(&mut [], &[], 0, 1, 1, 1e-5).unwrap_err();
        assert!(format!("{err}").contains("must all be > 0"));
    }

    #[test]
    fn per_head_rms_norm_rejects_mismatched_input_len() {
        let mut input = vec![0f32; 5];
        let gamma = vec![1f32; 2];
        let err = per_head_rms_norm_forward(&mut input, &gamma, 2, 2, 2, 1e-5)
            .unwrap_err();
        assert!(format!("{err}").contains("input len"));
    }

    #[test]
    fn per_head_rms_norm_rejects_wrong_gamma_len() {
        let mut input = vec![0f32; 8];
        let gamma = vec![1f32; 3]; // should be 2 (head_dim)
        let err = per_head_rms_norm_forward(&mut input, &gamma, 2, 2, 2, 1e-5)
            .unwrap_err();
        assert!(format!("{err}").contains("gamma len"));
    }

    #[test]
    fn per_head_rms_norm_end_to_end_real_gemma4_chain() {
        // The deepest real-data chain test to date. Runs:
        //   preprocess gradient → patch_embed → rms_norm(ln1) →
        //   qkv_projection → per-head RMSNorm on Q and K.
        // Asserts every stage yields finite, non-trivial output.
        use super::super::mmproj::MmprojConfig;
        use super::super::mmproj_weights::LoadedMmprojWeights;
        use mlx_native::gguf::GgufFile;
        let path = std::path::Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found at {}", GEMMA4_MMPROJ_PATH);
            return;
        }
        let gguf = GgufFile::open(path).expect("open");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let device = mlx_native::MlxDevice::new().expect("device");
        let weights = LoadedMmprojWeights::load(&gguf, &cfg, device).expect("load");

        // Gemma 4 vision tower: 16 heads × 72 head_dim = 1152.
        let hidden = cfg.hidden_size as usize;
        let num_heads = cfg.num_attention_heads as usize;
        let head_dim = hidden / num_heads;
        assert_eq!(num_heads, 16);
        assert_eq!(head_dim, 72);

        // Stage 1: synthetic 224×224 preprocessed gradient image →
        // [196, 1152] patch embeddings.
        let img = cfg.image_size as usize;
        let mut pixels = vec![0f32; 3 * img * img];
        for c in 0..3 {
            for y in 0..img {
                for x in 0..img {
                    pixels[c * img * img + y * img + x] =
                        ((c + 1) as f32) * 0.05 + (y as f32) * 0.001 + (x as f32) * 0.001;
                }
            }
        }
        let mut hidden_states =
            patch_embed_from_mmproj_weights(&pixels, &weights, &cfg).expect("patch_embed");
        let num_patches = 196usize;
        assert_eq!(hidden_states.len(), num_patches * hidden);

        // Stage 2: ln1 RMSNorm.
        let ln1 = weights.block_tensor(0, "ln1.weight").expect("ln1");
        let ln1_gamma: &[f32] = ln1.as_slice::<f32>().expect("ln1 as_slice");
        rms_norm_forward(&mut hidden_states, ln1_gamma, hidden, 1e-6).expect("rms ln1");

        // Stage 3: QKV projection.
        let q_buf = weights.block_tensor(0, "attn_q.weight").expect("attn_q");
        let k_buf = weights.block_tensor(0, "attn_k.weight").expect("attn_k");
        let v_buf = weights.block_tensor(0, "attn_v.weight").expect("attn_v");
        let q_w: &[f32] = q_buf.as_slice::<f32>().expect("q slice");
        let k_w: &[f32] = k_buf.as_slice::<f32>().expect("k slice");
        let v_w: &[f32] = v_buf.as_slice::<f32>().expect("v slice");
        let (mut q, mut k, v) = qkv_projection_forward(
            &hidden_states,
            q_w,
            k_w,
            v_w,
            num_patches,
            hidden,
        )
        .expect("qkv");

        // Stage 4: per-head RMSNorm on Q and K (Gemma 4 quirk).
        let q_norm = weights
            .block_tensor(0, "attn_q_norm.weight")
            .expect("attn_q_norm");
        let k_norm = weights
            .block_tensor(0, "attn_k_norm.weight")
            .expect("attn_k_norm");
        let q_norm_gamma: &[f32] = q_norm.as_slice::<f32>().expect("q_norm slice");
        let k_norm_gamma: &[f32] = k_norm.as_slice::<f32>().expect("k_norm slice");
        assert_eq!(q_norm_gamma.len(), head_dim);
        assert_eq!(k_norm_gamma.len(), head_dim);

        per_head_rms_norm_forward(&mut q, q_norm_gamma, num_patches, num_heads, head_dim, 1e-6)
            .expect("per-head Q");
        per_head_rms_norm_forward(&mut k, k_norm_gamma, num_patches, num_heads, head_dim, 1e-6)
            .expect("per-head K");

        // V stays un-normalized — sanity-check it's still finite from
        // stage 3 but deliberately NOT shaped by per-head norm.
        for t in [&q, &k, &v] {
            for val in t.iter() {
                assert!(val.is_finite(), "non-finite: {val}");
            }
        }
        // After per-head RMSNorm, every Q and K head slice has mean(x²)
        // scaled by γ². Catches silent-no-op (if the norm didn't fire,
        // Q would still have the post-projection distribution which has
        // far wider spread than post-norm).
        let q_slice0 = &q[0..head_dim];
        let q_ms: f32 = q_slice0.iter().map(|v| v * v).sum::<f32>() / (head_dim as f32);
        // γ values for pretrained Gemma 4 are O(1), so mean(x²) should
        // be in O(1) range after normalization.
        assert!(
            q_ms > 0.01 && q_ms < 100.0,
            "Q head-0 mean(x²) outside expected range: {q_ms}"
        );
    }

    // -----------------------------------------------------------------------
    // linear_forward + qkv_projection_forward (iter 35)
    // -----------------------------------------------------------------------

    #[test]
    fn linear_identity_weight_preserves_input() {
        // W = I_d (identity) → y = x. batch=2, d=4.
        let d = 4;
        let batch = 2;
        let mut weight = vec![0f32; d * d];
        for i in 0..d {
            weight[i * d + i] = 1.0;
        }
        let input: Vec<f32> = (0..(batch * d)).map(|i| i as f32).collect();
        let out = linear_forward(&input, &weight, None, batch, d, d).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn linear_all_ones_weight_produces_row_sum_per_output() {
        // W[o, i] = 1 for all (o, i) → y[n, o] = sum(x[n, :]) for every o.
        let batch = 3;
        let d_in = 4;
        let d_out = 2;
        let weight = vec![1f32; d_out * d_in];
        let input: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let out = linear_forward(&input, &weight, None, batch, d_in, d_out).unwrap();
        // Row 0 sum = 10, row 1 sum = 26, row 2 sum = 42.
        let expected = [10.0, 10.0, 26.0, 26.0, 42.0, 42.0];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!((*got - *want).abs() < 1e-5, "got {got} want {want}");
        }
    }

    #[test]
    fn linear_applies_bias_per_output_once() {
        // All-zero weight, nonzero bias → output = bias repeated per row.
        let batch = 3;
        let d_in = 2;
        let d_out = 3;
        let weight = vec![0f32; d_out * d_in];
        let bias = vec![0.5f32, 1.5, 2.5];
        let input = vec![99f32; batch * d_in]; // garbage, weights zero
        let out = linear_forward(&input, &weight, Some(&bias), batch, d_in, d_out).unwrap();
        assert_eq!(out.len(), batch * d_out);
        for n in 0..batch {
            for o in 0..d_out {
                assert!((out[n * d_out + o] - bias[o]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn linear_reference_dot_product() {
        // y[n, o] = sum_i x[n, i] * W[o, i].
        // batch=1, d_in=3, d_out=2.
        // x = [1, 2, 3]
        // W = [[0.1, 0.2, 0.3],   → y[0] = 1*0.1 + 2*0.2 + 3*0.3 = 1.4
        //      [0.4, 0.5, 0.6]]   → y[1] = 1*0.4 + 2*0.5 + 3*0.6 = 3.2
        let input = vec![1.0f32, 2.0, 3.0];
        let weight = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let out = linear_forward(&input, &weight, None, 1, 3, 2).unwrap();
        assert!((out[0] - 1.4).abs() < 1e-5);
        assert!((out[1] - 3.2).abs() < 1e-5);
    }

    #[test]
    fn linear_rejects_mismatched_input_len() {
        let weight = vec![0f32; 6];
        let input = vec![0f32; 5]; // should be 2*3 = 6 for batch=2, in=3
        let err = linear_forward(&input, &weight, None, 2, 3, 2).unwrap_err();
        assert!(format!("{err}").contains("input len"));
    }

    #[test]
    fn linear_rejects_mismatched_weight_len() {
        let weight = vec![0f32; 5]; // should be 2*3 = 6 for out=2, in=3
        let input = vec![0f32; 6];
        let err = linear_forward(&input, &weight, None, 2, 3, 2).unwrap_err();
        assert!(format!("{err}").contains("weight len"));
    }

    #[test]
    fn linear_rejects_wrong_bias_len() {
        let weight = vec![0f32; 6];
        let input = vec![0f32; 6];
        let bias = vec![0f32; 3]; // should be 2 (out_features)
        let err = linear_forward(&input, &weight, Some(&bias), 2, 3, 2).unwrap_err();
        assert!(format!("{err}").contains("bias len"));
    }

    #[test]
    fn linear_rejects_zero_dims() {
        let err = linear_forward(&[], &[], None, 0, 1, 1).unwrap_err();
        assert!(format!("{err}").contains("must all be > 0"));
    }

    #[test]
    fn qkv_projection_returns_three_tensors_of_expected_shape() {
        // Synthetic: all Q weights = 1.0, K = 2.0, V = 3.0. Uniform input
        // x = 1. For hidden=4, batch=2: each output element of Q should
        // be = 4 (sum of 4 ones), K = 8, V = 12.
        let batch = 2;
        let hidden = 4;
        let input = vec![1f32; batch * hidden];
        let q_w = vec![1f32; hidden * hidden];
        let k_w = vec![2f32; hidden * hidden];
        let v_w = vec![3f32; hidden * hidden];
        let (q, k, v) =
            qkv_projection_forward(&input, &q_w, &k_w, &v_w, batch, hidden).unwrap();
        assert_eq!(q.len(), batch * hidden);
        assert_eq!(k.len(), batch * hidden);
        assert_eq!(v.len(), batch * hidden);
        for val in &q { assert!((*val - 4.0).abs() < 1e-5); }
        for val in &k { assert!((*val - 8.0).abs() < 1e-5); }
        for val in &v { assert!((*val - 12.0).abs() < 1e-5); }
    }

    #[test]
    fn qkv_projection_propagates_shape_errors() {
        // Deliberately broken K-weight size to verify error reaches caller.
        let batch = 2;
        let hidden = 4;
        let input = vec![1f32; batch * hidden];
        let q_w = vec![1f32; hidden * hidden];
        let k_w = vec![2f32; hidden * hidden - 1]; // too short
        let v_w = vec![3f32; hidden * hidden];
        let err = qkv_projection_forward(&input, &q_w, &k_w, &v_w, batch, hidden).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("qkv_projection_forward K"), "got: {msg}");
    }

    #[test]
    fn qkv_projection_against_real_gemma4_block0_weights() {
        // End-to-end real-data chain test — the most valuable test in
        // this module so far. Load real Gemma 4 mmproj, reads block-0's
        // q/k/v weights, feeds a synthetic [196, 1152] input (the shape
        // that falls out of patch_embed → rms_norm), asserts q/k/v
        // output shapes + sanity distribution properties.
        use super::super::mmproj::MmprojConfig;
        use super::super::mmproj_weights::LoadedMmprojWeights;
        use mlx_native::gguf::GgufFile;
        let path = std::path::Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found at {}", GEMMA4_MMPROJ_PATH);
            return;
        }
        let gguf = GgufFile::open(path).expect("open");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let device = mlx_native::MlxDevice::new().expect("device");
        let weights = LoadedMmprojWeights::load(&gguf, &cfg, device).expect("load");

        let q_buf = weights.block_tensor(0, "attn_q.weight").expect("attn_q");
        let k_buf = weights.block_tensor(0, "attn_k.weight").expect("attn_k");
        let v_buf = weights.block_tensor(0, "attn_v.weight").expect("attn_v");
        let q_w: &[f32] = q_buf.as_slice::<f32>().expect("q as_slice");
        let k_w: &[f32] = k_buf.as_slice::<f32>().expect("k as_slice");
        let v_w: &[f32] = v_buf.as_slice::<f32>().expect("v as_slice");
        let hidden = cfg.hidden_size as usize;
        assert_eq!(q_w.len(), hidden * hidden);
        assert_eq!(k_w.len(), hidden * hidden);
        assert_eq!(v_w.len(), hidden * hidden);

        // Synthetic [196, 1152] input shaped like a post-rms-norm
        // patch embedding block. Small magnitudes so no overflow.
        let batch = 196;
        let mut input = vec![0f32; batch * hidden];
        for p in 0..batch {
            for h in 0..hidden {
                input[p * hidden + h] = 0.01 + (p as f32) * 0.0001 + (h as f32) * 1e-5;
            }
        }
        let (q, k, v) = qkv_projection_forward(&input, q_w, k_w, v_w, batch, hidden)
            .expect("qkv");
        assert_eq!(q.len(), batch * hidden);

        // Sanity: Q, K, V distributions should differ (they're separate
        // projections with independent weights). Compare variances;
        // identical variance would suggest a weight-alias bug.
        let var = |t: &[f32]| {
            let m = t.iter().sum::<f32>() / (t.len() as f32);
            t.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (t.len() as f32)
        };
        let q_var = var(&q);
        let k_var = var(&k);
        let v_var = var(&v);
        assert!(q_var > 1e-6, "Q variance too low: {q_var}");
        assert!(k_var > 1e-6, "K variance too low: {k_var}");
        assert!(v_var > 1e-6, "V variance too low: {v_var}");
        // Pretrained Q/K/V weights are trained to be distinct.
        assert!(
            (q_var - k_var).abs() > 1e-8 || (q_var - v_var).abs() > 1e-8,
            "Q/K/V variances all equal — weight aliasing bug?"
        );
        // Finite outputs (catches NaN from bad dequant).
        for t in [&q, &k, &v] {
            for val in t.iter() {
                assert!(val.is_finite(), "non-finite in projection output: {val}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // rms_norm_forward (iter 34 — Gemma 4 vision ln1/ln2/ffn_norm/post_ffw)
    // -----------------------------------------------------------------------

    #[test]
    fn rms_norm_unit_gamma_normalizes_to_unit_rms() {
        // After RMSNorm with γ=1, the row's mean-squared should be ~1.
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        rms_norm_forward(&mut x, &gamma, 4, 1e-6).unwrap();
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!(
            (ms - 1.0).abs() < 1e-4,
            "mean-squared should be ~1, got {ms}"
        );
    }

    #[test]
    fn rms_norm_pytorch_reference_values() {
        // x = [1, 2, 3, 4]. mean(x²) = (1+4+9+16)/4 = 7.5.
        // rms = sqrt(7.5 + 1e-5) ≈ 2.7386128.
        // y = x / rms = [0.36514837, 0.73029674, 1.09544511, 1.46059349]
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        rms_norm_forward(&mut x, &gamma, 4, 1e-5).unwrap();
        let expected = [0.36514837f32, 0.73029674, 1.09544511, 1.46059349];
        for (got, want) in x.iter().zip(expected.iter()) {
            assert!((*got - *want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn rms_norm_applies_gain_elementwise() {
        // γ scales each element post-normalization. For a uniform input,
        // pre-gain output is all-1; post-gain equals γ directly.
        let mut x = vec![5.0f32; 4];
        let gamma = vec![0.5f32, 1.0, 2.0, 3.0];
        rms_norm_forward(&mut x, &gamma, 4, 1e-6).unwrap();
        for (got, want) in x.iter().zip(gamma.iter()) {
            assert!((*got - *want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn rms_norm_normalizes_rows_independently() {
        // Two rows scaled by 100× should produce identical outputs since
        // RMSNorm is scale-invariant.
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0];
        let gamma = vec![1.0f32; 4];
        rms_norm_forward(&mut x, &gamma, 4, 1e-6).unwrap();
        for i in 0..4 {
            assert!(
                (x[i] - x[4 + i]).abs() < 1e-3,
                "row 0 elem {} = {} != row 1 = {}",
                i,
                x[i],
                x[4 + i]
            );
        }
    }

    #[test]
    fn rms_norm_does_not_divide_by_zero_when_input_is_zero() {
        // All-zeros row has RMS=0, would divide by zero without eps.
        // With eps > 0 the output is finite (and zero, since x is zero).
        let mut x = vec![0.0f32; 8];
        let gamma = vec![1.0f32; 8];
        rms_norm_forward(&mut x, &gamma, 8, 1e-6).unwrap();
        for v in &x {
            assert!(v.is_finite(), "got non-finite {}", v);
            assert!(v.abs() < 1e-5, "zero input should stay zero, got {v}");
        }
    }

    #[test]
    fn rms_norm_large_inputs_do_not_overflow() {
        // Large magnitudes: without stable normalization, sum of squares
        // could overflow. With f32 max ≈ 3.4e38, a hidden-size of 4 with
        // values of 1e18 squared = 1e36; sum is 4e36, still fits.
        let mut x = vec![1e18f32, 2e18, 3e18, 4e18];
        let gamma = vec![1.0f32; 4];
        rms_norm_forward(&mut x, &gamma, 4, 1e-6).unwrap();
        for v in &x {
            assert!(v.is_finite(), "got {v} — f32 overflow");
        }
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((ms - 1.0).abs() < 1e-3, "ms = {ms}");
    }

    #[test]
    fn rms_norm_rejects_hidden_zero() {
        let mut x = vec![1.0f32; 4];
        let err = rms_norm_forward(&mut x, &[], 0, 1e-5).unwrap_err();
        assert!(format!("{err}").contains("hidden must be > 0"));
    }

    #[test]
    fn rms_norm_rejects_non_divisible_input_len() {
        let mut x = vec![1.0f32; 7];
        let gamma = vec![1.0f32; 3];
        let err = rms_norm_forward(&mut x, &gamma, 3, 1e-5).unwrap_err();
        assert!(format!("{err}").contains("not divisible"));
    }

    #[test]
    fn rms_norm_rejects_wrong_gamma_len() {
        let mut x = vec![1.0f32; 4];
        let gamma = vec![1.0f32; 3];
        let err = rms_norm_forward(&mut x, &gamma, 4, 1e-5).unwrap_err();
        assert!(format!("{err}").contains("gamma len"));
    }

    #[test]
    fn rms_norm_runs_against_real_gemma4_ln1_weights() {
        // Use the real `v.blk.0.ln1.weight` gain vector (1152 f32) from
        // the loaded Gemma 4 mmproj. Apply to a synthetic [196, 1152]
        // patch-embedding-shaped input. Verifies:
        //   - gains are dequantized cleanly (no f16→f32 conversion bugs)
        //   - rms_norm_forward handles the production hidden size (1152)
        //   - output has finite, non-trivial values
        use super::super::mmproj::MmprojConfig;
        use super::super::mmproj_weights::LoadedMmprojWeights;
        use mlx_native::gguf::GgufFile;
        let path = std::path::Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found at {}", GEMMA4_MMPROJ_PATH);
            return;
        }
        let gguf = GgufFile::open(path).expect("open");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let device = mlx_native::MlxDevice::new().expect("device");
        let weights = LoadedMmprojWeights::load(&gguf, &cfg, device).expect("load");
        let ln1 = weights.block_tensor(0, "ln1.weight").expect("ln1");
        let gamma: &[f32] = ln1.as_slice::<f32>().expect("as_slice f32");
        assert_eq!(gamma.len(), cfg.hidden_size as usize);

        // Synthetic [196 patches × 1152 hidden] input with per-patch
        // unique gradient so all rows are distinguishable.
        let num_patches = 196usize;
        let hidden = cfg.hidden_size as usize;
        let mut input = vec![0f32; num_patches * hidden];
        for p in 0..num_patches {
            for h in 0..hidden {
                input[p * hidden + h] = 0.01 + (p as f32) * 0.001 + (h as f32) * 0.0001;
            }
        }
        rms_norm_forward(&mut input, gamma, hidden, 1e-6).expect("rms_norm");
        for v in &input {
            assert!(v.is_finite(), "non-finite post-RMS: {v}");
        }
        let mean_sq: f32 = input.iter().map(|v| v * v).sum::<f32>()
            / (input.len() as f32);
        // Per-row unit-RMS gets modulated by γ. Gemma's γ values are
        // O(1) so the overall mean-sq stays near γ²-ish ≈ 0.1..10 range.
        assert!(
            mean_sq > 1e-4 && mean_sq < 1e6,
            "mean_sq {} outside sane range",
            mean_sq
        );
    }

    #[test]
    fn rms_norm_differs_from_layer_norm_on_nonzero_mean_input() {
        // Key behavioral difference: LayerNorm subtracts mean first,
        // RMSNorm does not. A non-zero-mean input normalizes to
        // different values under the two.
        let mut x_rms = vec![2.0f32, 2.0, 2.0, 2.0];
        let mut x_ln = x_rms.clone();
        let gamma = vec![1.0f32; 4];
        let beta = vec![0.0f32; 4];
        rms_norm_forward(&mut x_rms, &gamma, 4, 1e-6).unwrap();
        layer_norm_forward(&mut x_ln, &gamma, &beta, 4, 1e-6).unwrap();
        // RMSNorm of [2,2,2,2]: rms=2, output=[1,1,1,1].
        // LayerNorm of [2,2,2,2]: variance=0 → (x-mean)*inv_std*γ = 0.
        for v in &x_rms {
            assert!((*v - 1.0).abs() < 1e-5, "RMS output not 1: {v}");
        }
        for v in &x_ln {
            assert!(v.abs() < 1e-5, "LN output not 0: {v}");
        }
    }

    // -----------------------------------------------------------------------
    // softmax_last_dim
    // -----------------------------------------------------------------------

    #[test]
    fn softmax_sums_to_one_per_row() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0];
        softmax_last_dim(&mut x, 4).unwrap();
        for row in 0..2 {
            let s: f32 = x[row * 4..row * 4 + 4].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {} sum = {}", row, s);
        }
    }

    #[test]
    fn softmax_uniform_input_yields_uniform_output() {
        let mut x = vec![5.0f32; 8];
        softmax_last_dim(&mut x, 8).unwrap();
        for v in &x {
            assert!(
                (*v - 0.125).abs() < 1e-6,
                "expected 1/8 = 0.125, got {}",
                v
            );
        }
    }

    #[test]
    fn softmax_reference_values_match_pytorch() {
        // torch.softmax(tensor([1, 2, 3]), dim=-1) =
        //   [0.09003057, 0.24472848, 0.66524094]
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax_last_dim(&mut x, 3).unwrap();
        assert!((x[0] - 0.09003057).abs() < 1e-6, "x[0] = {}", x[0]);
        assert!((x[1] - 0.24472848).abs() < 1e-6, "x[1] = {}", x[1]);
        assert!((x[2] - 0.66524094).abs() < 1e-6, "x[2] = {}", x[2]);
    }

    #[test]
    fn softmax_is_numerically_stable_for_large_inputs() {
        // Without max-subtraction, exp(1000) overflows to +inf and the
        // whole row becomes NaN. With the stable form, the row should
        // cleanly peak at the max element.
        let mut x = vec![1000.0f32, 999.0, 998.0];
        softmax_last_dim(&mut x, 3).unwrap();
        // Expected same as softmax([2, 1, 0]) = softmax([1000, 999, 998])
        // because we subtract max before exp.
        // softmax([2, 1, 0]) = [0.6652, 0.2447, 0.0900]
        assert!((x[0] - 0.6652).abs() < 1e-3, "x[0] = {}", x[0]);
        assert!((x[1] - 0.2447).abs() < 1e-3, "x[1] = {}", x[1]);
        assert!((x[2] - 0.0900).abs() < 1e-3, "x[2] = {}", x[2]);
    }

    #[test]
    fn softmax_concentrates_on_the_dominant_element() {
        // A single much-larger element → softmax ≈ one-hot.
        let mut x = vec![0.0f32, 20.0, 0.0, 0.0];
        softmax_last_dim(&mut x, 4).unwrap();
        assert!(x[1] > 0.999, "dominant element prob = {}", x[1]);
        for i in [0, 2, 3] {
            assert!(x[i] < 1e-8, "x[{}] = {}", i, x[i]);
        }
    }

    #[test]
    fn softmax_rejects_hidden_zero() {
        let mut x = vec![1.0f32; 4];
        let err = softmax_last_dim(&mut x, 0).unwrap_err();
        assert!(format!("{err}").contains("hidden must be > 0"));
    }

    #[test]
    fn softmax_rejects_non_divisible_input_len() {
        let mut x = vec![1.0f32; 7];
        let err = softmax_last_dim(&mut x, 3).unwrap_err();
        assert!(format!("{err}").contains("not divisible"));
    }

    // -----------------------------------------------------------------------
    // gelu_tanh_approx
    // -----------------------------------------------------------------------

    #[test]
    fn gelu_zero_is_zero() {
        let mut x = vec![0.0f32];
        gelu_tanh_approx(&mut x);
        assert!(x[0].abs() < 1e-7, "gelu(0) should be 0, got {}", x[0]);
    }

    #[test]
    fn gelu_reference_values_match_pytorch_tanh_approximate() {
        // PyTorch reference (approximate="tanh"):
        //   gelu(-3) ≈ -0.00363752
        //   gelu(-1) ≈ -0.15880802
        //   gelu(-0.5) ≈ -0.15428595
        //   gelu(0.5) ≈  0.34571406
        //   gelu(1)  ≈  0.84119198
        //   gelu(3)  ≈  2.99636247
        let inputs = [-3.0f32, -1.0, -0.5, 0.5, 1.0, 3.0];
        let expected = [
            -0.00363752_f32,
            -0.15880802,
            -0.15428595,
            0.34571406,
            0.84119198,
            2.99636247,
        ];
        let mut x = inputs.to_vec();
        gelu_tanh_approx(&mut x);
        for (i, (got, want)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*got - *want).abs() < 1e-4,
                "i={}: gelu({}) got {} want {}",
                i,
                inputs[i],
                got,
                want
            );
        }
    }

    #[test]
    fn gelu_is_monotonic_on_nonneg_inputs() {
        // GELU is NOT globally monotone — it has a local minimum near
        // x ≈ -0.7517 where derivative Φ(x) + x·φ(x) crosses zero. But
        // for x ≥ 0 the derivative is strictly positive (Φ(x) ≥ 0.5,
        // x·φ(x) ≥ 0), so monotone-increasing holds. This catches sign
        // errors in the inner polynomial without falsely flagging the
        // correct near-zero dip.
        let mut x: Vec<f32> = (0..80).map(|i| i as f32 * 0.1).collect();
        gelu_tanh_approx(&mut x);
        for i in 1..x.len() {
            assert!(
                x[i] > x[i - 1] - 1e-6,
                "gelu not monotone on x>=0 at i={}: {} -> {}",
                i,
                x[i - 1],
                x[i]
            );
        }
    }

    #[test]
    fn gelu_has_local_minimum_near_negative_point_seven_five() {
        // Sanity: confirm the non-monotone region exists where theory
        // predicts. The true local min is at x ≈ -0.7517. At x=-0.75
        // gelu ≈ -0.16998. Values at x=-0.5 and x=-1.0 should BOTH be
        // LESS NEGATIVE (greater) than at x=-0.75, confirming the dip.
        let mut probe = vec![-1.0f32, -0.75, -0.5];
        gelu_tanh_approx(&mut probe);
        assert!(
            probe[1] < probe[0] && probe[1] < probe[2],
            "expected local min at x=-0.75, got: \
             gelu(-1.0)={}, gelu(-0.75)={}, gelu(-0.5)={}",
            probe[0],
            probe[1],
            probe[2]
        );
    }

    #[test]
    fn gelu_large_positive_approaches_x() {
        // As x → +∞, GELU(x) → x. At x=10 the approximation is within 1e-5.
        let mut x = vec![10.0f32];
        gelu_tanh_approx(&mut x);
        assert!((x[0] - 10.0).abs() < 1e-4, "gelu(10) ≈ 10, got {}", x[0]);
    }

    #[test]
    fn gelu_large_negative_approaches_zero() {
        // As x → -∞, GELU(x) → 0. At x=-10 the value is near zero.
        let mut x = vec![-10.0f32];
        gelu_tanh_approx(&mut x);
        assert!(x[0].abs() < 1e-5, "gelu(-10) ≈ 0, got {}", x[0]);
    }

    // -----------------------------------------------------------------------
    // patch_embed_forward production-shape smoke
    // -----------------------------------------------------------------------

    #[test]
    fn gemma4_shape_does_not_panic() {
        // Smoke: exercise the function at the Gemma 4 shape to ensure
        // index arithmetic is sound at the production dims. 896 is too
        // slow for a unit test; scale the image + hidden to sqrt(Gemma)
        // while keeping the patch shape identical (14×14). Target: image
        // 56 = 4 patches per side = 16 patches total; hidden 32 (was 1152).
        let img: usize = 56;
        let patch: usize = 14;
        let hidden: usize = 32;
        let weight = vec![0.01f32; hidden * 3 * patch * patch];
        let bias = vec![0.1f32; hidden];
        let pixels = vec![0.5f32; 3 * img * img];
        let out = patch_embed_forward(
            &pixels,
            &weight,
            Some(&bias),
            img as u32,
            patch as u32,
            hidden as u32,
        )
        .unwrap();
        let num_patches = (img / patch) * (img / patch);
        assert_eq!(out.len(), num_patches * hidden);
        // Sanity: each output element = 0.5 * 0.01 * 3 * 14 * 14 + 0.1
        //                              = 0.5 * 0.01 * 588 + 0.1 = 2.94 + 0.1 = 3.04
        for v in &out {
            assert!((*v - 3.04).abs() < 1e-3, "expected 3.04, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // patch_embed_from_mmproj_weights — real-data integration (iter 33)
    // -----------------------------------------------------------------------

    const GEMMA4_MMPROJ_PATH: &str =
        "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq-mmproj.gguf";

    #[test]
    fn patch_embed_from_real_gemma4_weights_produces_sensible_embeddings() {
        // Real-data check: load the Gemma 4 mmproj, preprocess a solid-
        // gray synthetic PNG to [3, 224, 224] f32, run patch_embed →
        // assert shape = [196, 1152] and the output has non-trivial
        // magnitude + non-zero variance (catches silent zeroed-weights /
        // wrong-stride bugs).
        use super::super::mmproj::MmprojConfig;
        use super::super::mmproj_weights::LoadedMmprojWeights;
        use mlx_native::gguf::GgufFile;

        let path = std::path::Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found at {}", GEMMA4_MMPROJ_PATH);
            return;
        }
        let gguf = GgufFile::open(path).expect("open gemma4 mmproj");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        assert_eq!(cfg.image_size, 224);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.hidden_size, 1152);
        let num_patches_side = (cfg.image_size / cfg.patch_size) as usize;
        assert_eq!(num_patches_side, 14);
        let num_patches = num_patches_side * num_patches_side;
        assert_eq!(num_patches, 196);

        let device = mlx_native::MlxDevice::new().expect("device");
        let weights = LoadedMmprojWeights::load(&gguf, &cfg, device).expect("load");

        // Build a deterministic synthetic [3, 224, 224] pixel tensor with
        // a gentle gradient — non-constant to exercise the full kernel.
        let img = cfg.image_size as usize;
        let mut pixels = vec![0f32; 3 * img * img];
        for c in 0..3 {
            for y in 0..img {
                for x in 0..img {
                    let idx = c * img * img + y * img + x;
                    pixels[idx] = ((c + 1) as f32) * 0.1
                        + (y as f32) * 0.001
                        + (x as f32) * 0.001;
                }
            }
        }

        let out = patch_embed_from_mmproj_weights(&pixels, &weights, &cfg)
            .expect("patch_embed_from_mmproj_weights");
        assert_eq!(out.len(), num_patches * (cfg.hidden_size as usize));

        // Variance + magnitude sanity: pretrained weights against a
        // non-trivial input should produce a non-trivial output. A silent
        // all-zeros dequant or wrong-stride bug would yield out[..] == 0.
        let mean: f32 = out.iter().sum::<f32>() / (out.len() as f32);
        let var: f32 = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (out.len() as f32);
        assert!(var > 1e-4, "output variance unexpectedly low: {var}");
        let max_abs = out.iter().map(|v| v.abs()).fold(0f32, f32::max);
        assert!(max_abs > 1e-3, "output max_abs unexpectedly low: {max_abs}");

        // Non-trivial cross-patch variation: different spatial patches
        // of the gradient input should produce different embedding rows.
        // Compare patch 0 (top-left) vs patch 195 (bottom-right).
        let p0 = &out[0..cfg.hidden_size as usize];
        let p_last = &out[195 * (cfg.hidden_size as usize)..196 * (cfg.hidden_size as usize)];
        let l2_diff: f32 = p0
            .iter()
            .zip(p_last.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(l2_diff > 1e-3, "patch 0 and patch 195 are identical — stride bug");
    }
}
