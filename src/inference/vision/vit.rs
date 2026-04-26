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

/// Elementwise in-place residual add: `a += b`. Shape-validated thin
/// wrapper — the reason it exists as a named function (rather than an
/// inline `for` loop) is that the transformer block has multiple
/// residual additions (post-attn, post-FFN) and every one of them is a
/// good place to accidentally skip-or-duplicate. Naming the operation
/// makes the block-construction code self-documenting.
///
/// # Errors
///
/// - `a.len() != b.len()`
pub fn residual_add(a: &mut [f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!(
            "residual_add: a len {} != b len {}",
            a.len(),
            b.len()
        ));
    }
    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        *ai += *bi;
    }
    Ok(())
}

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

/// SiLU (Swish) activation, in-place: `y = x · σ(x) = x / (1 + exp(-x))`.
///
/// The gating activation for SwiGLU FFNs (used by Gemma / Llama / Mistral
/// / SigLIP2 vision towers, among others). Matches PyTorch
/// `nn.functional.silu(x)` byte-for-byte.
///
/// Numerical notes:
/// - For large positive `x`, `1/(1+exp(-x)) → 1`, so SiLU(x) → x.
/// - For large negative `x`, `1/(1+exp(-x)) → 0`, so SiLU(x) → 0.
/// - At `x = 0`, SiLU = 0 exactly.
/// - The implementation guards against exp overflow by clamping the
///   input to `exp(-x)` above a threshold — for `x < -40`, `exp(-x)`
///   would be huge but `x · σ(x) → 0` anyway; for `x > 40`, `exp(-x)`
///   is effectively 0 and we just return `x`. Rust's f32::exp handles
///   both ends without NaN.
pub fn silu_in_place(input: &mut [f32]) {
    for v in input.iter_mut() {
        let x = *v;
        *v = x / (1.0 + (-x).exp());
    }
}

/// Elementwise in-place multiply: `a[i] *= b[i]`. Shape-validated.
///
/// Used in SwiGLU gating (`silu(gate) * up`) and any other
/// elementwise-modulation path. Named wrapper mirrors `residual_add`:
/// keeps the block-construction call site self-documenting.
///
/// # Errors
///
/// - `a.len() != b.len()`
pub fn elementwise_mul_in_place(a: &mut [f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!(
            "elementwise_mul_in_place: a len {} != b len {}",
            a.len(),
            b.len()
        ));
    }
    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        *ai *= *bi;
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

/// Scaled-dot-product attention over `[batch, num_heads, head_dim]`
/// Q/K/V, returning `[batch, num_heads, head_dim]`.
///
/// ```text
/// scale = 1 / sqrt(head_dim)
/// for each head h:
///     scores[i, j] = <Q[i, h, :], K[j, h, :]> * scale        [batch × batch]
///     scores[i, :] = softmax(scores[i, :])                    # over j (last dim)
///     out[i, h, :] = Σ_j scores[i, j] * V[j, h, :]           [head_dim]
/// ```
///
/// **No mask.** ViT attention is bidirectional (unlike decoder-style
/// causal LMs). When a masked variant lands for a different model
/// family, it gets its own function — don't retrofit an `Option<&[bool]>`
/// parameter into this one and pay the branch tax.
///
/// # Complexity
///
/// `O(batch² × num_heads × head_dim)` — for Gemma 4 block 0 at
/// `[196, 16, 72]`: 196² × 16 × 72 ≈ 44 MFLOP for the two matmuls
/// combined, plus a negligible softmax pass. CPU runs it in ~30ms
/// unoptimized. GPU port dispatches `mlx_native::ops::flash_attn_*`.
///
/// # Errors
///
/// - Any of `q`/`k`/`v` length != `batch * num_heads * head_dim`
/// - Any dim is 0
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if batch == 0 || num_heads == 0 || head_dim == 0 {
        return Err(anyhow!(
            "scaled_dot_product_attention: batch ({}), num_heads ({}), head_dim ({}) must all be > 0",
            batch, num_heads, head_dim
        ));
    }
    let expected = batch * num_heads * head_dim;
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        if t.len() != expected {
            return Err(anyhow!(
                "scaled_dot_product_attention: {} len {} != batch*num_heads*head_dim = {}*{}*{} = {}",
                name,
                t.len(),
                batch,
                num_heads,
                head_dim,
                expected
            ));
        }
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let stride_batch = num_heads * head_dim;
    let stride_head = head_dim;

    let mut out = vec![0f32; expected];
    // One [batch × batch] score matrix reused across heads.
    let mut scores = vec![0f32; batch * batch];

    for h in 0..num_heads {
        // --- scores[i, j] = <Q[i, h, :], K[j, h, :]> * scale ---
        for i in 0..batch {
            let q_off = i * stride_batch + h * stride_head;
            for j in 0..batch {
                let k_off = j * stride_batch + h * stride_head;
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += q[q_off + d] * k[k_off + d];
                }
                scores[i * batch + j] = acc * scale;
            }
        }
        // --- softmax along j ---
        softmax_last_dim(&mut scores, batch)
            .map_err(|e| anyhow!("scaled_dot_product_attention softmax: {e}"))?;
        // --- out[i, h, :] = Σ_j scores[i, j] * V[j, h, :] ---
        for i in 0..batch {
            let out_off = i * stride_batch + h * stride_head;
            let sc_off = i * batch;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..batch {
                    let v_elem = v[j * stride_batch + h * stride_head + d];
                    acc += scores[sc_off + j] * v_elem;
                }
                out[out_off + d] = acc;
            }
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

/// In-place scalar multiply: `x *= c` across every element. Used by
/// Gemma 4V's post-blocks pipeline (`ggml_scale(cur, sqrtf(n_embd))`).
pub fn scale_in_place(x: &mut [f32], c: f32) {
    for v in x.iter_mut() {
        *v *= c;
    }
}

/// 2×2 spatial average-pool on a `[N_patches, hidden]` row-major tensor
/// laid out as an `N_side × N_side` patch grid. Returns
/// `[(N_side/2)², hidden]` where each output row is the mean of 4
/// adjacent input rows.
///
/// For Gemma 4V: `[196, 1152]` (14×14 patches) → `[49, 1152]` (7×7
/// patches). Matches llama.cpp's `ggml_pool_2d(..., AVG, 2, 2, 2, 2)`.
///
/// Output memory layout is row-major over the new 7×7 grid:
/// `out[y*7 + x]` = mean of input patches `(2y, 2x)`, `(2y, 2x+1)`,
/// `(2y+1, 2x)`, `(2y+1, 2x+1)`.
///
/// # Errors
///
/// - `input.len() != n_patches * hidden`
/// - `n_side * n_side != n_patches`
/// - `n_side % 2 != 0`
pub fn avg_pool_2x2_spatial(
    input: &[f32],
    n_side: usize,
    hidden: usize,
) -> Result<Vec<f32>> {
    let n_patches = n_side * n_side;
    if input.len() != n_patches * hidden {
        return Err(anyhow!(
            "avg_pool_2x2_spatial: input len {} != n_patches*hidden = {}*{} = {}",
            input.len(),
            n_patches,
            hidden,
            n_patches * hidden
        ));
    }
    if n_side == 0 || n_side % 2 != 0 {
        return Err(anyhow!(
            "avg_pool_2x2_spatial: n_side {} must be positive and even",
            n_side
        ));
    }

    let out_side = n_side / 2;
    let out_patches = out_side * out_side;
    let mut out = vec![0f32; out_patches * hidden];
    let inv4 = 0.25f32;
    for oy in 0..out_side {
        for ox in 0..out_side {
            let iy0 = oy * 2;
            let ix0 = ox * 2;
            let out_off = (oy * out_side + ox) * hidden;
            for d in 0..hidden {
                let a = input[(iy0 * n_side + ix0) * hidden + d];
                let b = input[(iy0 * n_side + (ix0 + 1)) * hidden + d];
                let c = input[((iy0 + 1) * n_side + ix0) * hidden + d];
                let d4 = input[((iy0 + 1) * n_side + (ix0 + 1)) * hidden + d];
                out[out_off + d] = (a + b + c + d4) * inv4;
            }
        }
    }
    Ok(out)
}

/// Elementwise per-channel bias/scale normalization, in-place across
/// `[batch, hidden]` rows: `x[b, i] = (x[b, i] - bias[i]) * scale[i]`.
///
/// Gemma 4 vision tower uses this between the post-blocks avg-pool and
/// the `mm.0` projector (`v.std_bias [1152]`, `v.std_scale [1152]`).
/// Not a standard stage in other ViTs — SigLIP2's Gemma 4 variant
/// introduced it as a pre-projector normalization.
///
/// # Errors
///
/// - `x.len() % hidden != 0`
/// - `bias.len() != hidden`
/// - `scale.len() != hidden`
pub fn std_bias_scale_in_place(
    x: &mut [f32],
    bias: &[f32],
    scale: &[f32],
    hidden: usize,
) -> Result<()> {
    if hidden == 0 {
        return Err(anyhow!("std_bias_scale_in_place: hidden must be > 0"));
    }
    if x.len() % hidden != 0 {
        return Err(anyhow!(
            "std_bias_scale_in_place: x len {} not divisible by hidden {}",
            x.len(),
            hidden
        ));
    }
    if bias.len() != hidden {
        return Err(anyhow!(
            "std_bias_scale_in_place: bias len {} != hidden {}",
            bias.len(),
            hidden
        ));
    }
    if scale.len() != hidden {
        return Err(anyhow!(
            "std_bias_scale_in_place: scale len {} != hidden {}",
            scale.len(),
            hidden
        ));
    }
    let n_rows = x.len() / hidden;
    for row in 0..n_rows {
        let off = row * hidden;
        for i in 0..hidden {
            x[off + i] = (x[off + i] - bias[i]) * scale[i];
        }
    }
    Ok(())
}

/// Execute one ViT transformer block end-to-end on a `[batch, hidden]`
/// residual-stream tensor. Composes the 11-stage pipeline:
///
/// ```text
/// cur = rms_norm(x, ln1)
/// (q, k, v) = qkv_projection(cur)
/// per_head_rms_norm(q, attn_q_norm)
/// per_head_rms_norm(k, attn_k_norm)
/// attn = scaled_dot_product_attention(q, k, v)     # TODO scale=1.0 for Gemma4V
/// attn = linear(attn, attn_output)
/// x = residual_add(x, attn)
/// cur = rms_norm(x, ln2)
/// gate = silu(linear(cur, ffn_gate))
/// up   = linear(cur, ffn_up)
/// cur  = gate * up
/// cur  = linear(cur, ffn_down)
/// cur  = rms_norm(cur, post_ffw_norm)
/// x    = residual_add(x, cur)
/// ```
///
/// `hidden_states` is consumed and the returned `Vec<f32>` replaces it
/// for the next block's input. Matches llama.cpp `build_vit` Gemma 4V
/// path structure; known block-parity TODOs (Gemma 4V `scale=1.0`,
/// V-RMSNorm, 2D RoPE) are unchanged from iter 40.
pub fn apply_vit_block_forward(
    hidden_states: Vec<f32>,
    weights: &super::mmproj_weights::LoadedMmprojWeights,
    cfg: &super::mmproj::MmprojConfig,
    block_idx: usize,
) -> Result<Vec<f32>> {
    let hidden = cfg.hidden_size as usize;
    let num_heads = cfg.num_attention_heads as usize;
    let head_dim = hidden / num_heads;
    let intermediate = cfg.intermediate_size as usize;
    let eps = cfg.layer_norm_eps;
    if hidden_states.len() % hidden != 0 {
        return Err(anyhow!(
            "apply_vit_block_forward: input len {} not divisible by hidden {}",
            hidden_states.len(),
            hidden
        ));
    }
    let batch = hidden_states.len() / hidden;

    // Helper: extract a block-local tensor as &[f32].
    let slice = |suffix: &str| -> Result<&[f32]> {
        let buf = weights.block_tensor(block_idx, suffix)?;
        buf.as_slice::<f32>()
            .map_err(|e| anyhow!("block {}: {} as_slice: {e}", block_idx, suffix))
    };

    // --- Attention half ---
    let mut residual = hidden_states;
    let mut cur = residual.clone();
    rms_norm_forward(&mut cur, slice("ln1.weight")?, hidden, eps)?;

    let (mut q, mut k, v) = qkv_projection_forward(
        &cur,
        slice("attn_q.weight")?,
        slice("attn_k.weight")?,
        slice("attn_v.weight")?,
        batch,
        hidden,
    )?;
    per_head_rms_norm_forward(
        &mut q,
        slice("attn_q_norm.weight")?,
        batch,
        num_heads,
        head_dim,
        eps,
    )?;
    per_head_rms_norm_forward(
        &mut k,
        slice("attn_k_norm.weight")?,
        batch,
        num_heads,
        head_dim,
        eps,
    )?;

    let attn = scaled_dot_product_attention(&q, &k, &v, batch, num_heads, head_dim)?;
    let attn_projected = linear_forward(
        &attn,
        slice("attn_output.weight")?,
        None,
        batch,
        hidden,
        hidden,
    )?;
    residual_add(&mut residual, &attn_projected)?;

    // --- FFN half ---
    let mut cur = residual.clone();
    rms_norm_forward(&mut cur, slice("ln2.weight")?, hidden, eps)?;

    let mut gate =
        linear_forward(&cur, slice("ffn_gate.weight")?, None, batch, hidden, intermediate)?;
    let up = linear_forward(&cur, slice("ffn_up.weight")?, None, batch, hidden, intermediate)?;
    silu_in_place(&mut gate);
    elementwise_mul_in_place(&mut gate, &up)?;

    let mut down = linear_forward(
        &gate,
        slice("ffn_down.weight")?,
        None,
        batch,
        intermediate,
        hidden,
    )?;
    rms_norm_forward(&mut down, slice("post_ffw_norm.weight")?, hidden, eps)?;
    residual_add(&mut residual, &down)?;

    Ok(residual)
}

/// Full CPU ViT forward: pixel tensor → projected multimodal embeddings.
///
/// Pipeline (from llama.cpp `clip_graph_gemma4v::build`):
///
/// ```text
///   1. hidden = patch_embed(pixels)                          [196, 1152]
///   2. for block in 0..n_layers:
///          hidden = apply_vit_block_forward(hidden, block)
///   3. hidden = avg_pool_2x2_spatial(hidden, 14, 1152)       [49, 1152]
///   4. scale_in_place(hidden, sqrt(n_embd))
///   5. std_bias_scale_in_place(hidden, v.std_bias, v.std_scale)
///   6. hidden = linear(hidden, mm.0.weight)                  [49, text_hidden]
///   7. rms_norm(hidden, gamma=ones, eps)    # no-gain RMSNorm
/// ```
///
/// Returns the final `[num_patches_side² / 4, text_hidden]` tensor
/// ready to be injected into the chat model's token stream.
///
/// # Cost
///
/// Per-block CPU compute ≈ 3 GFLOP; 27 blocks + post-blocks ≈ 81 GFLOP.
/// Naive-loop GEMM on a single CPU thread runs this in ~15–17 min —
/// which is why the full-forward integration test below is
/// `#[ignore]`'d. The GPU port dispatches `mlx_native::ops::mul_mm`
/// + `flash_attn_*` and runs in <1s for the same pipeline.
///
/// # Parity TODOs (from iter 40)
///
/// Not yet applied — mlx-lm reference needed for byte-identical
/// validation:
///   1. `scaled_dot_product_attention` scale=1.0 for Gemma 4V
///      (not 1/√d_head).
///   2. V RMSNorm (no gain) between QKV and attention.
///   3. 2D RoPE on Q, K against `v.position_embd.weight`.
///
/// # Errors
///
/// Propagated from any stage; see individual function docs.
pub fn apply_vit_full_forward(
    pixel_values: &[f32],
    weights: &super::mmproj_weights::LoadedMmprojWeights,
    cfg: &super::mmproj::MmprojConfig,
) -> Result<Vec<f32>> {
    // --- Stage 1: patch embedding ---
    let mut hidden_states = patch_embed_from_mmproj_weights(pixel_values, weights, cfg)?;

    // --- Stage 2: transformer blocks ---
    let n_layers = cfg.num_hidden_layers as usize;
    for block_idx in 0..n_layers {
        hidden_states = apply_vit_block_forward(hidden_states, weights, cfg, block_idx)?;
    }

    // --- Stage 3: 2×2 spatial avg-pool ---
    let n_side = cfg.num_patches_side as usize;
    let hidden = cfg.hidden_size as usize;
    let mut hidden_states = avg_pool_2x2_spatial(&hidden_states, n_side, hidden)?;

    // --- Stage 4: scale by sqrt(n_embd) ---
    scale_in_place(&mut hidden_states, (hidden as f32).sqrt());

    // --- Stage 5: std_bias / std_scale normalization ---
    let std_bias_buf = weights
        .get("v.std_bias")
        .ok_or_else(|| anyhow!("apply_vit_full_forward: missing v.std_bias"))?;
    let std_scale_buf = weights
        .get("v.std_scale")
        .ok_or_else(|| anyhow!("apply_vit_full_forward: missing v.std_scale"))?;
    let std_bias: &[f32] = std_bias_buf
        .as_slice::<f32>()
        .map_err(|e| anyhow!("std_bias as_slice: {e}"))?;
    let std_scale: &[f32] = std_scale_buf
        .as_slice::<f32>()
        .map_err(|e| anyhow!("std_scale as_slice: {e}"))?;
    std_bias_scale_in_place(&mut hidden_states, std_bias, std_scale, hidden)?;

    // --- Stage 6: mm.0 projector [n_patches_out, hidden] → [n_patches_out, text_hidden] ---
    let mm0_buf = weights
        .get("mm.0.weight")
        .ok_or_else(|| anyhow!("apply_vit_full_forward: missing mm.0.weight"))?;
    let mm0: &[f32] = mm0_buf
        .as_slice::<f32>()
        .map_err(|e| anyhow!("mm.0 as_slice: {e}"))?;
    // Shape: mm.0.weight is [text_hidden, hidden] per PyTorch nn.Linear
    // convention. Text hidden = mm0.len() / hidden.
    let text_hidden = mm0.len() / hidden;
    if mm0.len() != text_hidden * hidden {
        return Err(anyhow!(
            "mm.0.weight len {} not divisible by hidden {}",
            mm0.len(),
            hidden
        ));
    }
    let n_patches_out = hidden_states.len() / hidden;
    let mut projected = linear_forward(
        &hidden_states,
        mm0,
        None,
        n_patches_out,
        hidden,
        text_hidden,
    )?;

    // --- Stage 7: no-gain final RMSNorm ---
    // llama.cpp calls ggml_rms_norm without a weight param, so gamma is
    // effectively all-ones. Allocate once per call; cheap relative to
    // the ~81 GFLOP that preceded.
    let ones = vec![1.0f32; text_hidden];
    rms_norm_forward(&mut projected, &ones, text_hidden, cfg.layer_norm_eps)?;

    Ok(projected)
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
// Gemma4V (variable-resolution, 2D-RoPE) CPU primitives
// ---------------------------------------------------------------------------
//
// These mirror `patch_embed_forward` and `position_embed_add` (above) but
// follow the gemma4v graph from `/opt/llama.cpp/tools/mtmd/models/gemma4v.cpp`
// + the candle reference at `/opt/candle/.../gemma4/vision.rs:114-183`:
//
//   - PatchEmbedder is a Linear-no-bias `[hidden, p²·3]`, NOT a Conv2d.
//     The pre-flattened `[N_patches, p²·3]` patches buffer (produced by
//     `preprocess_gemma4v`) is fed straight in.
//   - Position embedding is a DUAL `[2, pos_size, hidden]` table. For
//     each patch at `(pos_x, pos_y)`, we add `pe[0][pos_x] + pe[1][pos_y]`
//     to the patch embedding.
//
// Both functions are CPU correctness references — the GPU port lives in
// `vit_gpu.rs` (`gemma4v_patch_embed_gpu` + `gemma4v_apply_position_embed_gpu`).
// They are byte-identical-on-tiny-shapes parity validators, never the
// production hot path on real `[256, 1152]` shapes.

/// Gemma 4.6 (gemma4v) patch embedding: flat Linear with NO bias.
///
/// Equivalent to `candle_nn::linear_no_bias(p² · 3, hidden)` in the
/// candle reference (`gemma4/vision.rs:127`). The expected weight layout
/// is `[hidden, p² · 3]` row-major (PyTorch's `nn.Linear.weight`
/// convention) so:
///
///   `out[n, o] = Σᵢ patches[n, i] * weight[o, i]`
///
/// Inputs:
///   - `patches`: `[N_patches, p² · 3]` flat buffer in row-major order
///     (typically the `Gemma4vPreprocessed::patches` field).
///   - `weight`: `[hidden, p² · 3]` flat buffer (the gemma4v
///     `v.patch_embd.weight` tensor as loaded by the mmproj loader).
///   - `n_patches`, `inner` (= `p² · 3`), `hidden`: shape parameters.
///
/// Output: `[N_patches, hidden]` row-major, freshly allocated.
///
/// # Errors
///
/// Shape-mismatch: `patches.len() != n_patches * inner`,
/// `weight.len() != hidden * inner`, or any zero dim.
///
/// # Why a sibling instead of reusing `linear_forward`
///
/// `linear_forward` already implements `y = x @ W.T` for `[batch,
/// in_features]` × `[out_features, in_features]`. We delegate to it
/// rather than re-rolling the GEMM — keeps the parity story honest
/// (any future linear-kernel fix touches one place).
pub fn gemma4v_patch_embed_forward(
    patches: &[f32],
    weight: &[f32],
    n_patches: u32,
    inner: u32,
    hidden: u32,
) -> Result<Vec<f32>> {
    if n_patches == 0 || inner == 0 || hidden == 0 {
        return Err(anyhow!(
            "gemma4v_patch_embed_forward: n_patches ({n_patches}), inner ({inner}), \
             hidden ({hidden}) must all be > 0"
        ));
    }
    let n_us = n_patches as usize;
    let in_us = inner as usize;
    let h_us = hidden as usize;
    if patches.len() != n_us * in_us {
        return Err(anyhow!(
            "gemma4v_patch_embed_forward: patches.len() ({}) != n_patches*inner ({})",
            patches.len(),
            n_us * in_us
        ));
    }
    if weight.len() != h_us * in_us {
        return Err(anyhow!(
            "gemma4v_patch_embed_forward: weight.len() ({}) != hidden*inner ({})",
            weight.len(),
            h_us * in_us
        ));
    }
    // Naive triple-nested-loop GEMM. The CPU reference is for tiny-shape
    // parity testing; production goes through `gemma4v_patch_embed_gpu`.
    let mut out = vec![0f32; n_us * h_us];
    for n in 0..n_us {
        let p_base = n * in_us;
        let o_base = n * h_us;
        for o in 0..h_us {
            let w_base = o * in_us;
            let mut acc: f32 = 0.0;
            for i in 0..in_us {
                acc += patches[p_base + i] * weight[w_base + i];
            }
            out[o_base + o] = acc;
        }
    }
    Ok(out)
}

/// Dual position-embed lookup for the gemma4v vision tower.
///
/// Implements the C++ reference at `/opt/llama.cpp/tools/mtmd/models/gemma4v.cpp:18-42`:
///
/// ```text
///   tbl_x = pe_table[0]                  // [pos_size, hidden]
///   tbl_y = pe_table[1]                  // [pos_size, hidden]
///   emb_x = ggml_get_rows(tbl_x, pos_x)  // [N_patches, hidden]
///   emb_y = ggml_get_rows(tbl_y, pos_y)
///   out[n, h] = emb_x[n, h] + emb_y[n, h]
/// ```
///
/// Equivalent candle reference: `vision.rs:163-179` — two
/// `index_select` calls + an add.
///
/// Inputs:
///   - `pos_x`, `pos_y`: per-patch indices, length `N_patches` each.
///     Out-of-range indices are clamped to `pos_size − 1` (matches
///     candle's `clamp(0, i64::MAX)` + the table's natural bound;
///     a panic-on-OOB would surface a real producer bug).
///   - `pe_table`: `[2, pos_size, hidden]` flat buffer (row-major,
///     X-axis table first then Y-axis table — matches the GGUF
///     writer at `src/backends/gguf.rs:1783`'s 3-D shape).
///   - `pos_size`, `hidden`: table shape parameters.
///
/// Output: `[N_patches, hidden]` row-major, freshly allocated.
///
/// # Errors
///
/// Shape-mismatch: `pos_x.len() != pos_y.len()`,
/// `pe_table.len() != 2 * pos_size * hidden`, any zero dim.
pub fn gemma4v_position_embed_lookup(
    pos_x: &[u32],
    pos_y: &[u32],
    pe_table: &[f32],
    pos_size: u32,
    hidden: u32,
) -> Result<Vec<f32>> {
    if pos_size == 0 || hidden == 0 {
        return Err(anyhow!(
            "gemma4v_position_embed_lookup: pos_size ({pos_size}) and hidden ({hidden}) must be > 0"
        ));
    }
    if pos_x.len() != pos_y.len() {
        return Err(anyhow!(
            "gemma4v_position_embed_lookup: pos_x.len() ({}) != pos_y.len() ({})",
            pos_x.len(),
            pos_y.len()
        ));
    }
    let n = pos_x.len();
    let h_us = hidden as usize;
    let ps_us = pos_size as usize;
    let expected = 2 * ps_us * h_us;
    if pe_table.len() != expected {
        return Err(anyhow!(
            "gemma4v_position_embed_lookup: pe_table.len() ({}) != 2*pos_size*hidden ({})",
            pe_table.len(),
            expected
        ));
    }
    let table_x_base = 0;
    let table_y_base = ps_us * h_us;
    let max_idx = (pos_size - 1) as u32;
    let mut out = vec![0f32; n * h_us];
    for k in 0..n {
        let x_idx = pos_x[k].min(max_idx) as usize;
        let y_idx = pos_y[k].min(max_idx) as usize;
        let row_x = table_x_base + x_idx * h_us;
        let row_y = table_y_base + y_idx * h_us;
        let out_base = k * h_us;
        for j in 0..h_us {
            out[out_base + j] = pe_table[row_x + j] + pe_table[row_y + j];
        }
    }
    Ok(out)
}

/// Add a dual position embedding into a patch-embedding tensor in place.
///
/// Convenience composition: `gemma4v_position_embed_lookup` produces
/// `pos_emb`, then we sum `patch_embeds += pos_emb` row-wise. Matches
/// candle's final line at `vision.rs:181` (`patches + pos_emb`).
///
/// `patch_embeds` is `[N_patches, hidden]` and is mutated in place.
pub fn gemma4v_position_embed_add(
    patch_embeds: &mut [f32],
    pos_x: &[u32],
    pos_y: &[u32],
    pe_table: &[f32],
    pos_size: u32,
    hidden: u32,
) -> Result<()> {
    let pos_emb = gemma4v_position_embed_lookup(pos_x, pos_y, pe_table, pos_size, hidden)?;
    if patch_embeds.len() != pos_emb.len() {
        return Err(anyhow!(
            "gemma4v_position_embed_add: patch_embeds.len() ({}) != pos_emb.len() ({})",
            patch_embeds.len(),
            pos_emb.len()
        ));
    }
    for (dst, src) in patch_embeds.iter_mut().zip(pos_emb.iter()) {
        *dst += *src;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Gemma4V per-block forward (4-RMSNorm + GQA + 2D RoPE + V-norm)
// ---------------------------------------------------------------------------
//
// Sibling to the SigLIP-49 `apply_vit_block_forward` (above). Differences
// from the SigLIP path:
//
//   1. Four RMSNorms per block (input_layernorm / post_attention_layernorm /
//      pre_feedforward_layernorm / post_feedforward_layernorm) vs SigLIP's
//      two (ln1 / post_ffw_norm).
//   2. RMSNorms use the Gemma "weight + 1" convention (centered-at-zero
//      learned gain). SigLIP uses raw `weight`.
//   3. Q-norm and K-norm also use the "weight + 1" convention.
//   4. V is RMS-normalized too — but with NO learned gain (pure
//      `x / rms(x)` per-head). No `attn_v_norm.weight` tensor.
//   5. GQA: `num_kv_heads` may be < `num_attention_heads`. K and V are
//      "repeat_kv"-expanded to match Q heads before attention.
//   6. Q/K rotated with 2D NeoX RoPE driven by `(pos_x, pos_y)` per
//      patch (not the standard 1-D RoPE).
//   7. Activation is GELU(pytorch_tanh) on the gate proj (not SiLU).
//   8. Attention scale is 1.0 (Q is RMS-normalized).
//
// All four refs:
//   - `/opt/llama.cpp/tools/mtmd/models/gemma4v.cpp:46-99`
//   - `/opt/candle/.../gemma4/vision.rs:202-365`
//   - `/opt/llama.cpp/tools/mtmd/clip.cpp:1334-1343` (gemma4v hparams)
//   - `/opt/llama.cpp/ggml/src/ggml-cpu/ops.cpp` (NeoX rope semantics)

/// Gemma-style RMSNorm: `y = x * rsqrt(mean(x²) + eps) * (weight + 1)`.
///
/// Same math as `rms_norm_forward` plus a `+1` on the gain — matches
/// `/opt/candle/.../gemma4/vision.rs:39` (`(&self.weight + 1.0)`). The
/// SigLIP path's `rms_norm_forward` does NOT add 1; this helper exists
/// so the gemma4v block forward stays a true peer to that path without
/// changing it.
pub fn gemma_rms_norm_forward(
    input: &mut [f32],
    weight: &[f32],
    hidden: usize,
    eps: f32,
) -> Result<()> {
    if hidden == 0 {
        return Err(anyhow!("gemma_rms_norm_forward: hidden must be > 0"));
    }
    if input.len() % hidden != 0 {
        return Err(anyhow!(
            "gemma_rms_norm_forward: input len {} not divisible by hidden {}",
            input.len(),
            hidden
        ));
    }
    if weight.len() != hidden {
        return Err(anyhow!(
            "gemma_rms_norm_forward: weight len {} != hidden {}",
            weight.len(),
            hidden
        ));
    }
    let inv_h = 1.0_f32 / hidden as f32;
    let n_rows = input.len() / hidden;
    for row in 0..n_rows {
        let off = row * hidden;
        let slice = &mut input[off..off + hidden];
        let mut sq = 0.0_f32;
        for &v in slice.iter() {
            sq += v * v;
        }
        let inv_rms = 1.0_f32 / (sq * inv_h + eps).sqrt();
        for (i, v) in slice.iter_mut().enumerate() {
            *v = (*v * inv_rms) * (weight[i] + 1.0);
        }
    }
    Ok(())
}

/// Per-head Gemma-style RMSNorm: `[batch, num_heads, head_dim]` →
/// `[batch * num_heads, head_dim]` with `(weight + 1)` gain shared across
/// heads. Used for `q_norm` and `k_norm` in the gemma4v ViT block.
pub fn gemma_per_head_rms_norm_forward(
    input: &mut [f32],
    weight: &[f32],
    batch: usize,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || head_dim == 0 {
        return Err(anyhow!(
            "gemma_per_head_rms_norm_forward: batch ({batch}), num_heads ({num_heads}), \
             head_dim ({head_dim}) must all be > 0"
        ));
    }
    let expected = batch * num_heads * head_dim;
    if input.len() != expected {
        return Err(anyhow!(
            "gemma_per_head_rms_norm_forward: input len {} != batch*num_heads*head_dim = {}",
            input.len(),
            expected
        ));
    }
    if weight.len() != head_dim {
        return Err(anyhow!(
            "gemma_per_head_rms_norm_forward: weight len {} != head_dim {}",
            weight.len(),
            head_dim
        ));
    }
    gemma_rms_norm_forward(input, weight, head_dim, eps)
}

/// Pure RMS normalization with NO learned gain: `y = x * rsqrt(mean(x²) + eps)`.
///
/// Used for V-norm in gemma4v (per `/opt/candle/.../gemma4/vision.rs:43-50`).
/// Internally f32 — even when called with an f32 input, this stays
/// numerically equivalent to the candle reference's `to_dtype(F32)` cast.
pub fn v_norm_no_scale_forward(
    input: &mut [f32],
    hidden: usize,
    eps: f32,
) -> Result<()> {
    if hidden == 0 {
        return Err(anyhow!("v_norm_no_scale_forward: hidden must be > 0"));
    }
    if input.len() % hidden != 0 {
        return Err(anyhow!(
            "v_norm_no_scale_forward: input len {} not divisible by hidden {}",
            input.len(),
            hidden
        ));
    }
    let inv_h = 1.0_f32 / hidden as f32;
    let n_rows = input.len() / hidden;
    for row in 0..n_rows {
        let off = row * hidden;
        let slice = &mut input[off..off + hidden];
        let mut sq = 0.0_f32;
        for &v in slice.iter() {
            sq += v * v;
        }
        let inv_rms = 1.0_f32 / (sq * inv_h + eps).sqrt();
        for v in slice.iter_mut() {
            *v *= inv_rms;
        }
    }
    Ok(())
}

/// Optional pair of `[input_min, input_max]` and `[output_min, output_max]`
/// scalar bounds for `gemma4v_clippable_linear_forward`. Each bound is
/// represented as an `Option<f32>`; `None` collapses to `f32::NEG_INFINITY`
/// (for min) or `f32::INFINITY` (for max), making that side a no-op.
///
/// Mirrors llama.cpp's `clamp_info` struct (`tools/mtmd/clip.cpp:1952-1957`)
/// which carries the four scalars as plain f32 with `+/- FLT_MAX` defaults.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gemma4ClippableLinearBounds {
    pub input_min: Option<f32>,
    pub input_max: Option<f32>,
    pub output_min: Option<f32>,
    pub output_max: Option<f32>,
}

impl Gemma4ClippableLinearBounds {
    /// `true` when at least one of the four scalars is present. A
    /// clippable linear with no bounds is byte-identical to a plain
    /// `linear_forward(_, weight, None, ...)` — callers can short-circuit.
    pub fn any(&self) -> bool {
        self.input_min.is_some()
            || self.input_max.is_some()
            || self.output_min.is_some()
            || self.output_max.is_some()
    }

    /// Resolve `(input_min, input_max)` with default sentinels. Use
    /// `f32::NEG_INFINITY` for missing min and `f32::INFINITY` for
    /// missing max so a kernel-level clamp degenerates to a no-op for
    /// any finite input.
    pub fn resolved_input(&self) -> (f32, f32) {
        (
            self.input_min.unwrap_or(f32::NEG_INFINITY),
            self.input_max.unwrap_or(f32::INFINITY),
        )
    }

    /// Resolve `(output_min, output_max)` with default sentinels.
    pub fn resolved_output(&self) -> (f32, f32) {
        (
            self.output_min.unwrap_or(f32::NEG_INFINITY),
            self.output_max.unwrap_or(f32::INFINITY),
        )
    }
}

/// CPU reference for the `Gemma4ClippableLinear` projector primitive
/// (`/opt/llama.cpp/tools/mtmd/models/gemma4v.cpp:138-151`).
///
/// Pipeline:
///   1. (optional) clamp `input` to `[input_min, input_max]`
///   2. `y = matmul(input, weight)` (single Linear, no bias — gemma4v
///      uses `linear_no_bias` per `/opt/candle/.../gemma4/multimodal_
///      embedding.rs:46`)
///   3. (optional) clamp `y` to `[output_min, output_max]`
///
/// `bounds.any() == false` is byte-equivalent to
/// `linear_forward(input, weight, None, batch, in_features, out_features)`
/// — we still go through the function so the call-site reads as the
/// gemma4v graph step. The GPU sibling `gemma4v_clippable_linear_gpu`
/// matches this exact ordering.
///
/// # Errors
///
/// - any zero shape arg
/// - input or weight length mismatch (delegated to `linear_forward`)
pub fn gemma4v_clippable_linear_forward(
    input: &[f32],
    weight: &[f32],
    bounds: &Gemma4ClippableLinearBounds,
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    if batch == 0 || in_features == 0 || out_features == 0 {
        return Err(anyhow!(
            "gemma4v_clippable_linear_forward: batch ({batch}), in_features \
             ({in_features}), out_features ({out_features}) must all be > 0"
        ));
    }
    // --- Stage 1: input clamp (only if bounds present) ---
    let clamped_input_owned: Option<Vec<f32>> =
        if bounds.input_min.is_some() || bounds.input_max.is_some() {
            let (mn, mx) = bounds.resolved_input();
            if mn > mx {
                return Err(anyhow!(
                    "gemma4v_clippable_linear_forward: input_min ({mn}) > input_max ({mx})"
                ));
            }
            let mut v = input.to_vec();
            for x in v.iter_mut() {
                *x = x.clamp(mn, mx);
            }
            Some(v)
        } else {
            None
        };
    let input_view: &[f32] = clamped_input_owned.as_deref().unwrap_or(input);

    // --- Stage 2: linear ---
    let mut y = linear_forward(input_view, weight, None, batch, in_features, out_features)?;

    // --- Stage 3: output clamp (only if bounds present) ---
    if bounds.output_min.is_some() || bounds.output_max.is_some() {
        let (mn, mx) = bounds.resolved_output();
        if mn > mx {
            return Err(anyhow!(
                "gemma4v_clippable_linear_forward: output_min ({mn}) > output_max ({mx})"
            ));
        }
        for v in y.iter_mut() {
            *v = v.clamp(mn, mx);
        }
    }
    Ok(y)
}

/// 2-D NeoX RoPE for ViT (CPU reference for `dispatch_vision_2d_rope`).
///
/// Layout: `[seq_len * n_heads, head_dim]` row-major.
/// First half rotates by `pos_x[seq]`, second half by `pos_y[seq]`,
/// each NeoX-style with pair `(d[i], d[i + d_quarter])` for
/// `i ∈ [0, d_quarter = head_dim / 4)` and theta denominator `d_half`.
///
/// `head_dim` MUST be divisible by 4 (clean dual NeoX split).
pub fn vision_2d_rope_forward_cpu(
    input: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    pos_x: &[u32],
    pos_y: &[u32],
    theta: f32,
) -> Result<Vec<f32>> {
    if head_dim == 0 || seq_len == 0 || n_heads == 0 {
        return Err(anyhow!(
            "vision_2d_rope_forward_cpu: head_dim ({head_dim}), seq_len ({seq_len}), \
             n_heads ({n_heads}) must all be > 0"
        ));
    }
    if head_dim % 4 != 0 {
        return Err(anyhow!(
            "vision_2d_rope_forward_cpu: head_dim ({head_dim}) must be divisible by 4"
        ));
    }
    if pos_x.len() != seq_len {
        return Err(anyhow!(
            "vision_2d_rope_forward_cpu: pos_x len {} != seq_len {seq_len}",
            pos_x.len()
        ));
    }
    if pos_y.len() != seq_len {
        return Err(anyhow!(
            "vision_2d_rope_forward_cpu: pos_y len {} != seq_len {seq_len}",
            pos_y.len()
        ));
    }
    let n_rows = seq_len * n_heads;
    let n_elem = n_rows * head_dim;
    if input.len() != n_elem {
        return Err(anyhow!(
            "vision_2d_rope_forward_cpu: input len {} != seq_len*n_heads*head_dim = {n_elem}",
            input.len()
        ));
    }
    let d_half = head_dim / 2;
    let d_quarter = d_half / 2;
    let mut out = vec![0_f32; n_elem];
    out.copy_from_slice(input);

    for row in 0..n_rows {
        let seq_idx = row / n_heads;
        let p_x = pos_x[seq_idx] as f32;
        let p_y = pos_y[seq_idx] as f32;
        let base = row * head_dim;
        for i in 0..d_quarter {
            let dim_ratio = (2 * i) as f32 / d_half as f32;
            let freq = 1.0_f32 / theta.powf(dim_ratio);
            let ax = p_x * freq;
            let ay = p_y * freq;
            let cx = ax.cos();
            let sx = ax.sin();
            let cy = ay.cos();
            let sy = ay.sin();
            // First half pair (i, i + d_quarter)
            let x0 = input[base + i];
            let x1 = input[base + i + d_quarter];
            out[base + i] = x0 * cx - x1 * sx;
            out[base + i + d_quarter] = x0 * sx + x1 * cx;
            // Second half pair (d_half + i, d_half + i + d_quarter)
            let y0 = input[base + d_half + i];
            let y1 = input[base + d_half + i + d_quarter];
            out[base + d_half + i] = y0 * cy - y1 * sy;
            out[base + d_half + i + d_quarter] = y0 * sy + y1 * cy;
        }
    }
    Ok(out)
}

/// GELU pytorch_tanh, in-place.
///
/// `y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
/// Matches both `torch.nn.functional.gelu(approximate="tanh")` and
/// mlx-native's `gelu_f32` kernel. Used for the gate-proj activation in
/// the gemma4v MLP.
pub fn gelu_pytorch_tanh_in_place(input: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56_f32;
    const COEFF: f32 = 0.044_715_f32;
    for v in input.iter_mut() {
        let x = *v;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        *v = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// CPU repeat-kv: replicate K (or V) heads to match Q's `num_heads`.
///
/// Input shape: `[batch, num_kv_heads, head_dim]` (== `[batch, num_kv_heads * head_dim]`
/// in row-major); each KV head is repeated `num_kv_groups = num_heads / num_kv_heads`
/// times, producing `[batch, num_heads, head_dim]`. The repetition order
/// matches PyTorch's `repeat_kv`: kv_head k → expanded heads
/// `[k * num_kv_groups, k * num_kv_groups + 1, ..., (k+1) * num_kv_groups - 1]`.
pub fn repeat_kv_cpu(
    input: &[f32],
    batch: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if batch == 0 || num_kv_heads == 0 || num_kv_groups == 0 || head_dim == 0 {
        return Err(anyhow!(
            "repeat_kv_cpu: batch, num_kv_heads, num_kv_groups, head_dim must all be > 0"
        ));
    }
    let expected_in = batch * num_kv_heads * head_dim;
    if input.len() != expected_in {
        return Err(anyhow!(
            "repeat_kv_cpu: input len {} != batch*num_kv_heads*head_dim = {expected_in}",
            input.len()
        ));
    }
    let num_heads = num_kv_heads * num_kv_groups;
    let mut out = vec![0_f32; batch * num_heads * head_dim];
    for b in 0..batch {
        for k in 0..num_kv_heads {
            let in_base = (b * num_kv_heads + k) * head_dim;
            for g in 0..num_kv_groups {
                let h_idx = k * num_kv_groups + g;
                let out_base = (b * num_heads + h_idx) * head_dim;
                out[out_base..out_base + head_dim]
                    .copy_from_slice(&input[in_base..in_base + head_dim]);
            }
        }
    }
    Ok(out)
}

/// Gemma 4 Vision per-block forward (CPU reference).
///
/// Mirrors `/opt/candle/.../gemma4/vision.rs:353-365` exactly:
///
/// ```text
///   x = x + post_attention_layernorm(self_attention(input_layernorm(x)))
///   x = x + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))
/// ```
///
/// Where `self_attention` does:
///
/// ```text
///   q = q_proj(in)              [batch, num_heads * head_dim]
///   k = k_proj(in)              [batch, num_kv_heads * head_dim]
///   v = v_proj(in)              [batch, num_kv_heads * head_dim]
///   q = gemma_rms(q, q_norm)
///   k = gemma_rms(k, k_norm)
///   v = v_norm(v)               # no learned gain
///   q = vision_2d_rope(q, pos_x, pos_y, theta)
///   k = vision_2d_rope(k, pos_x, pos_y, theta)
///   k = repeat_kv(k, num_kv_groups)
///   v = repeat_kv(v, num_kv_groups)
///   out = scaled_dot_product_attention(q, k, v, scale=1.0)
///   out = o_proj(out)
/// ```
///
/// And `mlp`:
///
/// ```text
///   gate = gate_proj(in); gate = gelu_pytorch_tanh(gate)
///   up   = up_proj(in)
///   down = down_proj(gate * up)
/// ```
///
/// # Arguments
///
/// * `hidden_states` — `[batch, hidden]` row-major (consumed; replaced
///   by next block's input).
/// * `weights[*]`    — block-local tensors, layout matches the GGUF
///   loader's `block_tensor(idx, suffix)` outputs.
/// * `pos_x`, `pos_y` — per-patch (batch == seq_len == num_patches)
///   positions for 2-D RoPE.
/// * shape parameters — see [`Gemma4VisionBlockShape`].
///
/// # Errors
///
/// Any shape mismatch in the supplied weights (validated row-by-row) or
/// any propagated error from the per-stage helpers.
#[allow(clippy::too_many_arguments)]
pub fn gemma4v_block_forward(
    hidden_states: Vec<f32>,
    block_weights: &Gemma4VisionBlockWeights<'_>,
    shape: &Gemma4VisionBlockShape,
    pos_x: &[u32],
    pos_y: &[u32],
) -> Result<Vec<f32>> {
    let hidden = shape.hidden as usize;
    let num_heads = shape.num_heads as usize;
    let num_kv_heads = shape.num_kv_heads as usize;
    let head_dim = shape.head_dim as usize;
    let intermediate = shape.intermediate as usize;
    let eps = shape.rms_norm_eps;
    let theta = shape.rope_theta;

    if hidden == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || intermediate == 0 {
        return Err(anyhow!(
            "gemma4v_block_forward: zero dim in shape: {shape:?}"
        ));
    }
    if num_heads % num_kv_heads != 0 {
        return Err(anyhow!(
            "gemma4v_block_forward: num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        ));
    }
    if hidden_states.len() % hidden != 0 {
        return Err(anyhow!(
            "gemma4v_block_forward: hidden_states len {} not divisible by hidden {hidden}",
            hidden_states.len()
        ));
    }
    let batch = hidden_states.len() / hidden;
    if pos_x.len() != batch || pos_y.len() != batch {
        return Err(anyhow!(
            "gemma4v_block_forward: pos_x ({}) / pos_y ({}) lengths must equal batch ({batch})",
            pos_x.len(),
            pos_y.len()
        ));
    }
    let num_kv_groups = num_heads / num_kv_heads;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    if q_dim != hidden {
        return Err(anyhow!(
            "gemma4v_block_forward: num_heads*head_dim ({q_dim}) must equal hidden ({hidden})"
        ));
    }

    // ----- Attention half -----
    let mut residual = hidden_states;

    // input_layernorm (gemma RMS, weight+1)
    let mut cur = residual.clone();
    gemma_rms_norm_forward(&mut cur, block_weights.input_layernorm, hidden, eps)?;

    // QKV projections
    let mut q = linear_forward(
        &cur,
        block_weights.q_proj,
        None,
        batch,
        hidden,
        q_dim,
    )?;
    let mut k = linear_forward(
        &cur,
        block_weights.k_proj,
        None,
        batch,
        hidden,
        kv_dim,
    )?;
    let mut v = linear_forward(
        &cur,
        block_weights.v_proj,
        None,
        batch,
        hidden,
        kv_dim,
    )?;

    // q_norm / k_norm (gemma per-head RMS, weight+1) and v_norm (no gain)
    gemma_per_head_rms_norm_forward(&mut q, block_weights.q_norm, batch, num_heads, head_dim, eps)?;
    gemma_per_head_rms_norm_forward(
        &mut k,
        block_weights.k_norm,
        batch,
        num_kv_heads,
        head_dim,
        eps,
    )?;
    v_norm_no_scale_forward(&mut v, head_dim, eps)?;

    // 2-D RoPE on Q and K (separate denominators for num_heads vs num_kv_heads)
    let q = vision_2d_rope_forward_cpu(&q, batch, num_heads, head_dim, pos_x, pos_y, theta)?;
    let k = vision_2d_rope_forward_cpu(&k, batch, num_kv_heads, head_dim, pos_x, pos_y, theta)?;

    // GQA: repeat K / V to match Q's head count.
    let k_full = repeat_kv_cpu(&k, batch, num_kv_heads, num_kv_groups, head_dim)?;
    let v_full = repeat_kv_cpu(&v, batch, num_kv_heads, num_kv_groups, head_dim)?;

    // Scaled-dot-product attention (gemma4v scale = 1.0 because Q is RMS-normalized).
    // Existing helper applies the standard 1/sqrt(head_dim) scale internally; for
    // gemma4v we want NO scale, so pre-multiply Q by sqrt(head_dim) to undo it.
    // The cleanest approach is to inline the math here so we don't smuggle the
    // arch-specific scale into the SigLIP helper.
    let attn = gemma4v_attention_unit_scale(&q, &k_full, &v_full, batch, num_heads, head_dim)?;

    // o_proj
    let attn_proj = linear_forward(
        &attn,
        block_weights.o_proj,
        None,
        batch,
        hidden,
        hidden,
    )?;

    // post_attention_layernorm (applied to the attention OUTPUT before the residual add).
    let mut attn_out = attn_proj;
    gemma_rms_norm_forward(
        &mut attn_out,
        block_weights.post_attention_layernorm,
        hidden,
        eps,
    )?;
    residual_add(&mut residual, &attn_out)?;

    // ----- MLP half -----
    let mut cur = residual.clone();
    gemma_rms_norm_forward(
        &mut cur,
        block_weights.pre_feedforward_layernorm,
        hidden,
        eps,
    )?;

    let mut gate = linear_forward(
        &cur,
        block_weights.gate_proj,
        None,
        batch,
        hidden,
        intermediate,
    )?;
    let up = linear_forward(
        &cur,
        block_weights.up_proj,
        None,
        batch,
        hidden,
        intermediate,
    )?;
    gelu_pytorch_tanh_in_place(&mut gate);
    elementwise_mul_in_place(&mut gate, &up)?;
    let mut down = linear_forward(
        &gate,
        block_weights.down_proj,
        None,
        batch,
        intermediate,
        hidden,
    )?;
    gemma_rms_norm_forward(
        &mut down,
        block_weights.post_feedforward_layernorm,
        hidden,
        eps,
    )?;
    residual_add(&mut residual, &down)?;

    Ok(residual)
}

/// Local helper: scaled-dot-product attention with `scale = 1.0` for gemma4v.
///
/// Same algorithm as `scaled_dot_product_attention` but with no implicit
/// `1/sqrt(head_dim)` scaling — gemma4v doesn't divide because Q has
/// already been RMS-normalized. Inputs are `[batch, num_heads, head_dim]`
/// (Q, K, V); output is `[batch, num_heads * head_dim]`.
fn gemma4v_attention_unit_scale(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if batch == 0 || num_heads == 0 || head_dim == 0 {
        return Err(anyhow!(
            "gemma4v_attention_unit_scale: zero dim batch={batch} heads={num_heads} d={head_dim}"
        ));
    }
    let expected = batch * num_heads * head_dim;
    if q.len() != expected || k.len() != expected || v.len() != expected {
        return Err(anyhow!(
            "gemma4v_attention_unit_scale: shape mismatch q={} k={} v={} expected {expected}",
            q.len(),
            k.len(),
            v.len()
        ));
    }
    // Compute scores[h, i, j] = sum_d Q[i, h, d] * K[j, h, d] for batch_q=batch_k=batch.
    // No sqrt scaling.
    let mut scores = vec![0_f32; num_heads * batch * batch];
    for h in 0..num_heads {
        for i in 0..batch {
            for j in 0..batch {
                let mut acc = 0_f32;
                let q_base = i * num_heads * head_dim + h * head_dim;
                let k_base = j * num_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    acc += q[q_base + d] * k[k_base + d];
                }
                scores[h * batch * batch + i * batch + j] = acc;
            }
        }
    }
    // Softmax along last (j) axis per (h, i).
    softmax_last_dim(&mut scores, batch)?;
    // attn[i, h, d] = sum_j scores[h, i, j] * V[j, h, d]
    let mut attn = vec![0_f32; batch * num_heads * head_dim];
    for h in 0..num_heads {
        for i in 0..batch {
            for d in 0..head_dim {
                let mut acc = 0_f32;
                for j in 0..batch {
                    let v_idx = j * num_heads * head_dim + h * head_dim + d;
                    let s_idx = h * batch * batch + i * batch + j;
                    acc += scores[s_idx] * v[v_idx];
                }
                attn[i * num_heads * head_dim + h * head_dim + d] = acc;
            }
        }
    }
    Ok(attn)
}

/// Shape parameters for `gemma4v_block_forward`. Built by the caller from
/// the model config + the loaded block tensors.
#[derive(Debug, Clone, Copy)]
pub struct Gemma4VisionBlockShape {
    pub hidden: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

/// Borrowed view of a single block's weights for the CPU forward.
/// Each field is `[out, in]` row-major (PyTorch nn.Linear convention)
/// for projection weights, and `[head_dim]` (or `[hidden]`) for norms.
#[derive(Debug)]
pub struct Gemma4VisionBlockWeights<'a> {
    pub input_layernorm: &'a [f32],            // ln1.weight,        [hidden]
    pub post_attention_layernorm: &'a [f32],    // ln2.weight,        [hidden]
    pub pre_feedforward_layernorm: &'a [f32],   // ffn_norm.weight,   [hidden]
    pub post_feedforward_layernorm: &'a [f32],  // post_ffw_norm,     [hidden]
    pub q_proj: &'a [f32],                      // attn_q.weight,     [num_heads*head_dim, hidden]
    pub k_proj: &'a [f32],                      // attn_k.weight,     [num_kv_heads*head_dim, hidden]
    pub v_proj: &'a [f32],                      // attn_v.weight,     [num_kv_heads*head_dim, hidden]
    pub o_proj: &'a [f32],                      // attn_output.weight,[hidden, num_heads*head_dim]
    pub q_norm: &'a [f32],                      // attn_q_norm.weight,[head_dim]
    pub k_norm: &'a [f32],                      // attn_k_norm.weight,[head_dim]
    pub gate_proj: &'a [f32],                   // ffn_gate.weight,   [intermediate, hidden]
    pub up_proj: &'a [f32],                     // ffn_up.weight,     [intermediate, hidden]
    pub down_proj: &'a [f32],                   // ffn_down.weight,   [hidden, intermediate]
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
    // std_bias_scale_in_place + apply_vit_full_forward (iter 42)
    // -----------------------------------------------------------------------

    #[test]
    fn std_bias_scale_subtracts_and_scales_per_channel() {
        // batch=2, hidden=3. bias=[1, 2, 3], scale=[10, 20, 30].
        // Row 0: [5, 10, 15] → (row - bias) = [4, 8, 12] → * scale = [40, 160, 360].
        // Row 1: [10, 20, 30] → (row - bias) = [9, 18, 27] → * scale = [90, 360, 810].
        let mut x = vec![5.0f32, 10.0, 15.0, 10.0, 20.0, 30.0];
        let bias = vec![1.0f32, 2.0, 3.0];
        let scale = vec![10.0f32, 20.0, 30.0];
        std_bias_scale_in_place(&mut x, &bias, &scale, 3).unwrap();
        assert_eq!(x, vec![40.0, 160.0, 360.0, 90.0, 360.0, 810.0]);
    }

    #[test]
    fn std_bias_scale_zero_bias_unit_scale_is_identity() {
        let mut x = vec![0.5f32, 1.7, -2.3, 4.1];
        let snap = x.clone();
        let bias = vec![0.0f32; 2];
        let scale = vec![1.0f32; 2];
        std_bias_scale_in_place(&mut x, &bias, &scale, 2).unwrap();
        assert_eq!(x, snap);
    }

    #[test]
    fn std_bias_scale_rejects_hidden_zero() {
        let mut x = vec![0f32; 4];
        let err = std_bias_scale_in_place(&mut x, &[], &[], 0).unwrap_err();
        assert!(format!("{err}").contains("hidden must be > 0"));
    }

    #[test]
    fn std_bias_scale_rejects_non_divisible_len() {
        let mut x = vec![0f32; 7];
        let bias = vec![0f32; 3];
        let scale = vec![1f32; 3];
        let err = std_bias_scale_in_place(&mut x, &bias, &scale, 3).unwrap_err();
        assert!(format!("{err}").contains("not divisible"));
    }

    #[test]
    fn std_bias_scale_rejects_wrong_bias_len() {
        let mut x = vec![0f32; 6];
        let bias = vec![0f32; 2]; // should be 3
        let scale = vec![1f32; 3];
        let err = std_bias_scale_in_place(&mut x, &bias, &scale, 3).unwrap_err();
        assert!(format!("{err}").contains("bias len"));
    }

    #[test]
    fn std_bias_scale_rejects_wrong_scale_len() {
        let mut x = vec![0f32; 6];
        let bias = vec![0f32; 3];
        let scale = vec![1f32; 2]; // should be 3
        let err = std_bias_scale_in_place(&mut x, &bias, &scale, 3).unwrap_err();
        assert!(format!("{err}").contains("scale len"));
    }

    // Retired 2026-04-24 per "CPU inference == poop" directive: the
    // 27-block CPU full-forward test (formerly ignored at ~15-17 min
    // CPU runtime) is deleted. Production test coverage lives in
    // `vit_gpu::tests` — those GPU dispatches run in <1s each. The
    // CPU `apply_vit_full_forward` function itself stays as a
    // tiny-input parity reference only, invoked by `vit_gpu::tests`
    // on 4×4 or 8×8 synthetic shapes to byte-compare each GPU op's
    // output. It is NEVER invoked on production `[196, 1152]` shapes.
    #[allow(dead_code)]
    fn _retired_cpu_full_forward_stub() {
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

        // Preprocess a deterministic gradient image to [3, 224, 224] f32.
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

        let t0 = std::time::Instant::now();
        let out = apply_vit_full_forward(&pixels, &weights, &cfg).expect("full forward");
        let elapsed = t0.elapsed();
        eprintln!("apply_vit_full_forward: {:?}", elapsed);

        // Expected shape: [49 patches, text_hidden=2816].
        let n_patches_out = 49;
        let mm0_buf = weights.get("mm.0.weight").expect("mm.0");
        let mm0_slice: &[f32] = mm0_buf.as_slice::<f32>().expect("slice");
        let text_hidden = mm0_slice.len() / (cfg.hidden_size as usize);
        assert_eq!(text_hidden, 2816, "Gemma 4 projector output width");
        assert_eq!(out.len(), n_patches_out * text_hidden);

        // Every element finite.
        for v in &out {
            assert!(v.is_finite(), "non-finite in ViT output: {v}");
        }
        // Post-final-RMSNorm mean(x²) per row ≈ 1 (no-gain norm).
        for p in 0..n_patches_out {
            let row = &out[p * text_hidden..(p + 1) * text_hidden];
            let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / (text_hidden as f32);
            assert!(
                (ms - 1.0).abs() < 1e-2,
                "patch {p} mean(x²) = {ms}, expected ≈ 1.0 after no-gain RMSNorm"
            );
        }
        // Cross-patch distinction preserved.
        let p0 = &out[0..text_hidden];
        let p_last = &out[(n_patches_out - 1) * text_hidden..n_patches_out * text_hidden];
        let l2: f32 = p0
            .iter()
            .zip(p_last.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(l2 > 1e-3, "token 0 and token 48 collapsed to identical output");
    }

    // -----------------------------------------------------------------------
    // scale_in_place + avg_pool_2x2_spatial + apply_vit_block_forward (iter 41)
    // -----------------------------------------------------------------------

    #[test]
    fn scale_in_place_multiplies_every_element() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        scale_in_place(&mut x, 2.5);
        assert_eq!(x, vec![2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn scale_in_place_by_one_is_identity() {
        let mut x = vec![0.1f32, -2.3, 7.7];
        let snap = x.clone();
        scale_in_place(&mut x, 1.0);
        assert_eq!(x, snap);
    }

    #[test]
    fn scale_in_place_by_zero_zeros_everything() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        scale_in_place(&mut x, 0.0);
        assert_eq!(x, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn avg_pool_2x2_averages_each_2x2_block() {
        // 4×4 grid × 2 hidden dims → 2×2 output.
        // Each output should be the mean of 4 input values per dim.
        // Layout: input[y*4 + x] has hidden=2 values.
        // Values: input[y*4+x][0] = y*4+x, input[...][1] = (y*4+x)*10.
        let n_side = 4;
        let hidden = 2;
        let mut input = vec![0f32; n_side * n_side * hidden];
        for y in 0..n_side {
            for x in 0..n_side {
                let patch = y * n_side + x;
                input[patch * hidden + 0] = patch as f32;
                input[patch * hidden + 1] = (patch as f32) * 10.0;
            }
        }
        let out = avg_pool_2x2_spatial(&input, n_side, hidden).unwrap();
        // Expected out[0,0] = avg(patches 0,1,4,5) = avg(0,1,4,5) = 2.5
        // Expected out[0,1] = avg(patches 2,3,6,7) = avg(2,3,6,7) = 4.5
        // Expected out[1,0] = avg(patches 8,9,12,13) = 10.5
        // Expected out[1,1] = avg(patches 10,11,14,15) = 12.5
        assert_eq!(out.len(), 2 * 2 * hidden);
        assert!((out[0 * hidden + 0] - 2.5).abs() < 1e-6);
        assert!((out[0 * hidden + 1] - 25.0).abs() < 1e-6);
        assert!((out[1 * hidden + 0] - 4.5).abs() < 1e-6);
        assert!((out[1 * hidden + 1] - 45.0).abs() < 1e-6);
        assert!((out[2 * hidden + 0] - 10.5).abs() < 1e-6);
        assert!((out[2 * hidden + 1] - 105.0).abs() < 1e-6);
        assert!((out[3 * hidden + 0] - 12.5).abs() < 1e-6);
        assert!((out[3 * hidden + 1] - 125.0).abs() < 1e-6);
    }

    #[test]
    fn avg_pool_gemma4_shape_14x14_to_7x7() {
        // Production Gemma 4 shape: 196 patches (14×14) × 1152 hidden
        // → 49 patches (7×7) × 1152 hidden.
        let n_side = 14;
        let hidden = 1152;
        let input = vec![1.0f32; n_side * n_side * hidden];
        let out = avg_pool_2x2_spatial(&input, n_side, hidden).unwrap();
        assert_eq!(out.len(), 7 * 7 * hidden);
        // Uniform input → uniform output at the same value.
        for v in &out {
            assert!((*v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn avg_pool_rejects_non_even_n_side() {
        let err = avg_pool_2x2_spatial(&[0f32; 27], 3, 3).unwrap_err();
        assert!(format!("{err}").contains("positive and even"));
    }

    #[test]
    fn avg_pool_rejects_mismatched_input_len() {
        let err = avg_pool_2x2_spatial(&[0f32; 15], 4, 2).unwrap_err();
        assert!(format!("{err}").contains("input len"));
    }

    #[test]
    fn apply_vit_block_forward_real_gemma4_block0_matches_inline_chain() {
        // Wraps iter 40's inline block-0 chain into one function call.
        // Asserts output shape + that the new API produces the same
        // block-out distribution as the iter 40 explicit pipeline.
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

        let hidden = cfg.hidden_size as usize;
        let num_patches = 196usize;
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
        let patch_embed =
            patch_embed_from_mmproj_weights(&pixels, &weights, &cfg).expect("patch_embed");

        // NEW API: single call replaces iter 40's inline 11-stage chain.
        let block_out =
            apply_vit_block_forward(patch_embed, &weights, &cfg, 0).expect("block_forward");

        assert_eq!(block_out.len(), num_patches * hidden);
        for v in &block_out {
            assert!(v.is_finite(), "non-finite: {v}");
        }
        // Cross-patch differentiation (no stride bug inside the wrapper).
        let p0 = &block_out[0..hidden];
        let p_last = &block_out[(num_patches - 1) * hidden..num_patches * hidden];
        let l2: f32 = p0
            .iter()
            .zip(p_last.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(l2 > 1e-3, "wrapper produced stride-bugged output");
    }

    #[test]
    fn apply_vit_block_forward_rejects_mismatched_hidden() {
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

        // Wrong-shape hidden state: len not divisible by hidden.
        let bad = vec![0f32; 196 * 1152 + 1];
        let err = apply_vit_block_forward(bad, &weights, &cfg, 0).unwrap_err();
        assert!(format!("{err}").contains("not divisible"));
    }

    // -----------------------------------------------------------------------
    // Block 0 CPU parity — full forward through one transformer block (iter 40)
    // -----------------------------------------------------------------------
    //
    // Structural closeout. Known Gemma4V block-parity deltas (documented
    // here; fixed in a dedicated "block parity iter" once an mlx-lm
    // reference is available for byte-identical comparison):
    //   1. `kq_scale = 1.0` for Gemma 4V (iter 37 uses 1/√d_head default).
    //   2. V gets its own RMSNorm (no gain) before attention (currently skipped).
    //   3. 2D RoPE on Q and K (not yet ported).
    // Ordering + tensor wiring below matches llama.cpp's clip.cpp
    // `build_vit` exactly for the Gemma 4V path.

    #[test]
    fn block_0_full_forward_on_real_gemma4() {
        // 11-stage end-to-end block 0 on real Gemma 4 pretrained weights:
        //   1. preprocess(pixel gradient) → patch_embed
        //   2. snapshot residual_stream
        //   3. rms_norm(residual_stream, ln1) → hidden
        //   4. QKV projection → Q, K, V
        //   5. per-head RMSNorm on Q, K (V skipped — block-parity TODO)
        //   6. scaled-dot-product attention (TODO: kq_scale=1.0 for Gemma 4V)
        //   7. linear(attn, attn_output) → attn_projected
        //   8. residual_add(residual_stream, attn_projected) → post_attn
        //   9. rms_norm(post_attn, ln2) → pre_ffn
        //  10. SwiGLU(pre_ffn) = silu(gate) * up → activated
        //  11. linear(activated, ffn_down) → down            [NEW]
        //      rms_norm(down, post_ffw_norm)                 [NEW]
        //      residual_add(post_attn, down) → block_out     [NEW]
        //
        // Output: [196, 1152] block_0 residual stream, input to block 1.
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

        let hidden = cfg.hidden_size as usize;
        let num_heads = cfg.num_attention_heads as usize;
        let head_dim = hidden / num_heads;
        let intermediate = cfg.intermediate_size as usize;
        let num_patches = 196usize;
        let img = cfg.image_size as usize;

        // === Stages 1-10 (condensed from iter 39's chain) ===
        let mut pixels = vec![0f32; 3 * img * img];
        for c in 0..3 {
            for y in 0..img {
                for x in 0..img {
                    pixels[c * img * img + y * img + x] =
                        ((c + 1) as f32) * 0.05 + (y as f32) * 0.001 + (x as f32) * 0.001;
                }
            }
        }
        let residual_stream =
            patch_embed_from_mmproj_weights(&pixels, &weights, &cfg).expect("patch_embed");
        let mut hidden_states = residual_stream.clone();
        let slice = |name: &str| -> &[f32] {
            // SAFETY: as_slice returns an &[f32] tied to the &MlxBuffer borrow;
            // each closure call gets a fresh borrow through the weights ref.
            weights
                .block_tensor(0, name)
                .expect(name)
                .as_slice::<f32>()
                .expect(name)
        };
        rms_norm_forward(&mut hidden_states, slice("ln1.weight"), hidden, 1e-6).unwrap();
        let (mut q, mut k, v) = qkv_projection_forward(
            &hidden_states,
            slice("attn_q.weight"),
            slice("attn_k.weight"),
            slice("attn_v.weight"),
            num_patches,
            hidden,
        )
        .unwrap();
        per_head_rms_norm_forward(
            &mut q,
            slice("attn_q_norm.weight"),
            num_patches,
            num_heads,
            head_dim,
            1e-6,
        )
        .unwrap();
        per_head_rms_norm_forward(
            &mut k,
            slice("attn_k_norm.weight"),
            num_patches,
            num_heads,
            head_dim,
            1e-6,
        )
        .unwrap();
        let attn =
            scaled_dot_product_attention(&q, &k, &v, num_patches, num_heads, head_dim).unwrap();
        let attn_projected = linear_forward(
            &attn,
            slice("attn_output.weight"),
            None,
            num_patches,
            hidden,
            hidden,
        )
        .unwrap();
        let mut post_attn = residual_stream.clone();
        residual_add(&mut post_attn, &attn_projected).unwrap();
        let mut pre_ffn = post_attn.clone();
        rms_norm_forward(&mut pre_ffn, slice("ln2.weight"), hidden, 1e-6).unwrap();
        let mut gate = linear_forward(
            &pre_ffn,
            slice("ffn_gate.weight"),
            None,
            num_patches,
            hidden,
            intermediate,
        )
        .unwrap();
        let up = linear_forward(
            &pre_ffn,
            slice("ffn_up.weight"),
            None,
            num_patches,
            hidden,
            intermediate,
        )
        .unwrap();
        silu_in_place(&mut gate);
        elementwise_mul_in_place(&mut gate, &up).unwrap();
        let activated = gate;

        // === Stage 11 (NEW iter 40): FFN down projection + post-norm + residual ===
        let mut down = linear_forward(
            &activated,
            slice("ffn_down.weight"),
            None,
            num_patches,
            intermediate,
            hidden,
        )
        .expect("ffn_down");
        assert_eq!(down.len(), num_patches * hidden);

        // post_ffw_norm applied to the FFN output BEFORE the residual add
        // (matches llama.cpp build_vit: `cur = build_norm(cur, ff_post_norm_w)`
        // before `cur = inpL + cur`).
        rms_norm_forward(&mut down, slice("post_ffw_norm.weight"), hidden, 1e-6)
            .expect("post_ffw_norm");

        let mut block_out = post_attn.clone();
        residual_add(&mut block_out, &down).expect("ffn residual");

        // --- Invariants ---
        assert_eq!(block_out.len(), num_patches * hidden);
        for val in &block_out {
            assert!(val.is_finite(), "block_out non-finite: {val}");
        }
        // block_out differs from post_attn (FFN half contributed).
        let delta_l2: f32 = post_attn
            .iter()
            .zip(block_out.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(delta_l2 > 1e-3, "FFN half was a silent no-op");
        // Cross-patch distinction preserved through the entire block.
        let p0 = &block_out[0..hidden];
        let p_last = &block_out[(num_patches - 1) * hidden..num_patches * hidden];
        let cross_l2: f32 = p0
            .iter()
            .zip(p_last.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            cross_l2 > 1e-3,
            "token 0 and token 195 identical post-block — probable stride bug"
        );
    }

    // -----------------------------------------------------------------------
    // silu_in_place + elementwise_mul_in_place + SwiGLU gated (iter 39)
    // -----------------------------------------------------------------------

    #[test]
    fn silu_zero_is_zero_exactly() {
        let mut x = vec![0.0f32];
        silu_in_place(&mut x);
        assert_eq!(x[0], 0.0);
    }

    #[test]
    fn silu_reference_values_match_pytorch() {
        // PyTorch F.silu reference values:
        //   silu(-3) ≈ -0.14227
        //   silu(-1) ≈ -0.26894
        //   silu(-0.5) ≈ -0.18877
        //   silu(0.5) ≈  0.31123
        //   silu(1)  ≈  0.73106
        //   silu(3)  ≈  2.85773
        let inputs = [-3.0f32, -1.0, -0.5, 0.5, 1.0, 3.0];
        let expected = [
            -0.14227762_f32,
            -0.26894143,
            -0.18877034,
            0.31122967,
            0.73105858,
            2.85772238,
        ];
        let mut x = inputs.to_vec();
        silu_in_place(&mut x);
        for (got, want) in x.iter().zip(expected.iter()) {
            assert!((*got - *want).abs() < 1e-5, "got {got} want {want}");
        }
    }

    #[test]
    fn silu_large_positive_approaches_x() {
        // x=20 → σ(x) ≈ 1 → silu(x) ≈ x.
        let mut x = vec![20.0f32];
        silu_in_place(&mut x);
        assert!((x[0] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn silu_large_negative_approaches_zero() {
        let mut x = vec![-20.0f32];
        silu_in_place(&mut x);
        assert!(x[0].abs() < 1e-4);
    }

    #[test]
    fn silu_has_local_minimum_near_negative_point_two_eight() {
        // SiLU has a local min at x ≈ -1.2785 where value ≈ -0.27846.
        // Probe x=-1, x=-1.28, x=-1.5; x=-1.28 should be the lowest.
        let mut x = vec![-1.0f32, -1.28, -1.5];
        silu_in_place(&mut x);
        assert!(x[1] < x[0] && x[1] < x[2], "got {:?}", x);
    }

    #[test]
    fn elementwise_mul_pairs_each_index() {
        let mut a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        elementwise_mul_in_place(&mut a, &b).unwrap();
        assert_eq!(a, vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn elementwise_mul_with_zero_zeros_everything() {
        let mut a = vec![1.0f32, 2.0, 3.0];
        let b = vec![0f32; 3];
        elementwise_mul_in_place(&mut a, &b).unwrap();
        assert_eq!(a, vec![0f32; 3]);
    }

    #[test]
    fn elementwise_mul_rejects_shape_mismatch() {
        let mut a = vec![0f32; 4];
        let b = vec![0f32; 3];
        let err = elementwise_mul_in_place(&mut a, &b).unwrap_err();
        assert!(format!("{err}").contains("len"));
    }

    #[test]
    fn swiglu_gated_activation_on_real_gemma4_ffn() {
        // Runs iter 38's full attention-half chain + iter 39's new
        // stages: gate = linear(pre_ffn, ffn_gate), up = linear(pre_ffn,
        // ffn_up), silu(gate), gate *= up. Output shape: [196, 4304].
        // This is the exact input that iter 40's ffn_down + residual
        // + post_ffw_norm will consume.
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

        let hidden = cfg.hidden_size as usize;
        let num_heads = cfg.num_attention_heads as usize;
        let head_dim = hidden / num_heads;
        let num_patches = 196usize;
        let img = cfg.image_size as usize;

        // Stages 1-7 (iter 38's chain, condensed).
        let mut pixels = vec![0f32; 3 * img * img];
        for c in 0..3 {
            for y in 0..img {
                for x in 0..img {
                    pixels[c * img * img + y * img + x] =
                        ((c + 1) as f32) * 0.05 + (y as f32) * 0.001 + (x as f32) * 0.001;
                }
            }
        }
        let residual_stream =
            patch_embed_from_mmproj_weights(&pixels, &weights, &cfg).expect("patch_embed");
        let mut hidden_states = residual_stream.clone();
        rms_norm_forward(
            &mut hidden_states,
            weights
                .block_tensor(0, "ln1.weight")
                .unwrap()
                .as_slice::<f32>()
                .unwrap(),
            hidden,
            1e-6,
        )
        .unwrap();
        let (mut q, mut k, v) = qkv_projection_forward(
            &hidden_states,
            weights.block_tensor(0, "attn_q.weight").unwrap().as_slice::<f32>().unwrap(),
            weights.block_tensor(0, "attn_k.weight").unwrap().as_slice::<f32>().unwrap(),
            weights.block_tensor(0, "attn_v.weight").unwrap().as_slice::<f32>().unwrap(),
            num_patches,
            hidden,
        )
        .unwrap();
        per_head_rms_norm_forward(
            &mut q,
            weights.block_tensor(0, "attn_q_norm.weight").unwrap().as_slice::<f32>().unwrap(),
            num_patches,
            num_heads,
            head_dim,
            1e-6,
        )
        .unwrap();
        per_head_rms_norm_forward(
            &mut k,
            weights.block_tensor(0, "attn_k_norm.weight").unwrap().as_slice::<f32>().unwrap(),
            num_patches,
            num_heads,
            head_dim,
            1e-6,
        )
        .unwrap();
        let attn =
            scaled_dot_product_attention(&q, &k, &v, num_patches, num_heads, head_dim).unwrap();
        let attn_projected = linear_forward(
            &attn,
            weights.block_tensor(0, "attn_output.weight").unwrap().as_slice::<f32>().unwrap(),
            None,
            num_patches,
            hidden,
            hidden,
        )
        .unwrap();
        let mut post_attn = residual_stream.clone();
        residual_add(&mut post_attn, &attn_projected).unwrap();
        let mut pre_ffn = post_attn.clone();
        rms_norm_forward(
            &mut pre_ffn,
            weights.block_tensor(0, "ln2.weight").unwrap().as_slice::<f32>().unwrap(),
            hidden,
            1e-6,
        )
        .unwrap();

        // Stage 8 (NEW): SwiGLU gated activation.
        // gate_w: [intermediate, hidden] = [4304, 1152]
        // up_w:   [intermediate, hidden] = [4304, 1152]
        let ffn_gate_buf = weights.block_tensor(0, "ffn_gate.weight").expect("ffn_gate");
        let ffn_up_buf = weights.block_tensor(0, "ffn_up.weight").expect("ffn_up");
        let intermediate = cfg.intermediate_size as usize;
        assert_eq!(intermediate, 4304);
        let gate_w: &[f32] = ffn_gate_buf.as_slice::<f32>().unwrap();
        let up_w: &[f32] = ffn_up_buf.as_slice::<f32>().unwrap();
        assert_eq!(gate_w.len(), intermediate * hidden);
        assert_eq!(up_w.len(), intermediate * hidden);

        // gate = pre_ffn @ gate_w.T → [num_patches, intermediate]
        let mut gate =
            linear_forward(&pre_ffn, gate_w, None, num_patches, hidden, intermediate)
                .expect("gate proj");
        let up = linear_forward(&pre_ffn, up_w, None, num_patches, hidden, intermediate)
            .expect("up proj");

        silu_in_place(&mut gate);
        elementwise_mul_in_place(&mut gate, &up).expect("mul");

        let activated = gate; // rename for clarity
        assert_eq!(activated.len(), num_patches * intermediate);
        for val in &activated {
            assert!(val.is_finite(), "non-finite: {val}");
        }
        // Sanity: activated has non-trivial variance. A silent silu-no-op
        // or wrong-stride bug would collapse the distribution.
        let mean: f32 = activated.iter().sum::<f32>() / (activated.len() as f32);
        let var: f32 = activated
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>()
            / (activated.len() as f32);
        assert!(var > 1e-6, "activated variance too low: {var}");
        // Different patches should produce different post-gate rows.
        let p0 = &activated[0..intermediate];
        let p_last =
            &activated[(num_patches - 1) * intermediate..num_patches * intermediate];
        let l2_diff: f32 = p0.iter()
            .zip(p_last.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(l2_diff > 1e-3, "all patches produce identical gated output");
    }

    // -----------------------------------------------------------------------
    // residual_add + full attention-half block 0 (iter 38)
    // -----------------------------------------------------------------------

    #[test]
    fn residual_add_is_elementwise() {
        let mut a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0];
        residual_add(&mut a, &b).unwrap();
        assert_eq!(a, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn residual_add_with_zero_is_identity() {
        let mut a = vec![0.5f32, -1.2, 3.7];
        let snap = a.clone();
        let b = vec![0.0f32; 3];
        residual_add(&mut a, &b).unwrap();
        assert_eq!(a, snap);
    }

    #[test]
    fn residual_add_rejects_shape_mismatch() {
        let mut a = vec![0f32; 4];
        let b = vec![0f32; 3];
        let err = residual_add(&mut a, &b).unwrap_err();
        assert!(format!("{err}").contains("len"));
    }

    #[test]
    fn residual_add_is_commutative_against_add_in_place_loop() {
        // Sanity check vs a hand-rolled loop — catches any silent
        // sign-flip or off-by-one in the iterator.
        let a: Vec<f32> = (0..10).map(|i| i as f32 * 0.3).collect();
        let b: Vec<f32> = (0..10).map(|i| i as f32 * 0.7).collect();
        let mut via_fn = a.clone();
        residual_add(&mut via_fn, &b).unwrap();
        let mut via_loop = a.clone();
        for (x, y) in via_loop.iter_mut().zip(b.iter()) {
            *x += *y;
        }
        assert_eq!(via_fn, via_loop);
    }

    #[test]
    fn attention_half_block_end_to_end_real_gemma4() {
        // Completes the attention half of ViT block 0 on real Gemma 4
        // weights: chain from iter 37 extended with attn_output
        // projection, residual add (pre-attn → post-attn), and ln2
        // RMSNorm. This is the exact input that the FFN half (iter 39+)
        // will consume.
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

        let hidden = cfg.hidden_size as usize;
        let num_heads = cfg.num_attention_heads as usize;
        let head_dim = hidden / num_heads;
        let num_patches = 196usize;
        let img = cfg.image_size as usize;

        // Stage 1: preprocess + patch_embed → residual_stream.
        let mut pixels = vec![0f32; 3 * img * img];
        for c in 0..3 {
            for y in 0..img {
                for x in 0..img {
                    pixels[c * img * img + y * img + x] =
                        ((c + 1) as f32) * 0.05 + (y as f32) * 0.001 + (x as f32) * 0.001;
                }
            }
        }
        let residual_stream =
            patch_embed_from_mmproj_weights(&pixels, &weights, &cfg).expect("patch_embed");

        // Stage 2: copy → ln1 → QKV → per-head norm → attention.
        let mut hidden_states = residual_stream.clone();
        let ln1 = weights.block_tensor(0, "ln1.weight").expect("ln1");
        rms_norm_forward(
            &mut hidden_states,
            ln1.as_slice::<f32>().expect("ln1 slice"),
            hidden,
            1e-6,
        )
        .expect("rms ln1");

        let q_buf = weights.block_tensor(0, "attn_q.weight").expect("q");
        let k_buf = weights.block_tensor(0, "attn_k.weight").expect("k");
        let v_buf = weights.block_tensor(0, "attn_v.weight").expect("v");
        let (mut q, mut k, v) = qkv_projection_forward(
            &hidden_states,
            q_buf.as_slice::<f32>().expect("q slice"),
            k_buf.as_slice::<f32>().expect("k slice"),
            v_buf.as_slice::<f32>().expect("v slice"),
            num_patches,
            hidden,
        )
        .expect("qkv");

        let q_norm = weights
            .block_tensor(0, "attn_q_norm.weight")
            .expect("attn_q_norm");
        let k_norm = weights
            .block_tensor(0, "attn_k_norm.weight")
            .expect("attn_k_norm");
        per_head_rms_norm_forward(
            &mut q,
            q_norm.as_slice::<f32>().expect("q_norm slice"),
            num_patches,
            num_heads,
            head_dim,
            1e-6,
        )
        .expect("per-head Q");
        per_head_rms_norm_forward(
            &mut k,
            k_norm.as_slice::<f32>().expect("k_norm slice"),
            num_patches,
            num_heads,
            head_dim,
            1e-6,
        )
        .expect("per-head K");

        let attn = scaled_dot_product_attention(&q, &k, &v, num_patches, num_heads, head_dim)
            .expect("attention");

        // Stage 3 (NEW): attn_output projection.
        let attn_output = weights
            .block_tensor(0, "attn_output.weight")
            .expect("attn_output");
        let attn_output_w: &[f32] =
            attn_output.as_slice::<f32>().expect("attn_output slice");
        assert_eq!(attn_output_w.len(), hidden * hidden);
        let attn_projected =
            linear_forward(&attn, attn_output_w, None, num_patches, hidden, hidden)
                .expect("attn_output proj");

        // Stage 4 (NEW): residual add → post-attention hidden states.
        let mut post_attn = residual_stream.clone();
        residual_add(&mut post_attn, &attn_projected).expect("residual");

        // Stage 5 (NEW): ln2 RMSNorm.
        let mut pre_ffn = post_attn.clone();
        let ln2 = weights.block_tensor(0, "ln2.weight").expect("ln2");
        rms_norm_forward(
            &mut pre_ffn,
            ln2.as_slice::<f32>().expect("ln2 slice"),
            hidden,
            1e-6,
        )
        .expect("rms ln2");

        // Sanity: every stage's output is finite, shape-correct, and
        // differs from the bare patch_embed (stages have changed the
        // signal, not silently no-oped).
        for t in [&attn_projected, &post_attn, &pre_ffn] {
            assert_eq!(t.len(), num_patches * hidden);
            for v in t.iter() {
                assert!(v.is_finite(), "non-finite: {v}");
            }
        }
        // post_attn = residual + attn_projected, so it should differ
        // from the raw residual unless attention produced all-zero
        // output (which would itself be a real problem).
        let delta_l2: f32 = residual_stream
            .iter()
            .zip(post_attn.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            delta_l2 > 1e-3,
            "post-attn residual is identical to pre-attn — attn_output all zero?"
        );
        // pre_ffn (post-LN) should differ from post_attn (pre-LN) — catches
        // LN being a silent no-op.
        let ln_delta: f32 = post_attn
            .iter()
            .zip(pre_ffn.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(ln_delta > 1e-3, "ln2 was a no-op");
    }

    // -----------------------------------------------------------------------
    // scaled_dot_product_attention (iter 37)
    // -----------------------------------------------------------------------

    #[test]
    fn attention_uniform_qk_averages_v_across_tokens() {
        // Q and K all equal → every softmax is uniform 1/batch → output
        // row = mean of V rows per head.
        let batch = 3;
        let num_heads = 2;
        let head_dim = 4;
        let n = batch * num_heads * head_dim;
        let q = vec![1f32; n];
        let k = vec![1f32; n];
        // V per head: head 0 rows = [1,2,3,4]×3, head 1 rows = [10,20,30,40]×3.
        // Actually give distinct rows so averaging is non-trivial:
        // token 0: head 0 = [1,2,3,4], head 1 = [10,20,30,40]
        // token 1: head 0 = [5,6,7,8], head 1 = [50,60,70,80]
        // token 2: head 0 = [9,10,11,12], head 1 = [90,100,110,120]
        let mut v = vec![0f32; n];
        for b in 0..batch {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    v[b * num_heads * head_dim + h * head_dim + d] = if h == 0 {
                        (b * 4 + d + 1) as f32
                    } else {
                        ((b * 4 + d + 1) * 10) as f32
                    };
                }
            }
        }
        let out = scaled_dot_product_attention(&q, &k, &v, batch, num_heads, head_dim).unwrap();
        // Mean of head-0 rows across 3 tokens:
        //   d=0: (1+5+9)/3 = 5
        //   d=1: (2+6+10)/3 = 6
        //   d=2: (3+7+11)/3 = 7
        //   d=3: (4+8+12)/3 = 8
        // Mean of head-1 rows: 10× those, so [50, 60, 70, 80].
        let expected_h0 = [5.0f32, 6.0, 7.0, 8.0];
        let expected_h1 = [50.0f32, 60.0, 70.0, 80.0];
        for b in 0..batch {
            for d in 0..head_dim {
                let got_h0 = out[b * num_heads * head_dim + 0 * head_dim + d];
                let got_h1 = out[b * num_heads * head_dim + 1 * head_dim + d];
                assert!(
                    (got_h0 - expected_h0[d]).abs() < 1e-4,
                    "h0 b{b} d{d}: {got_h0} vs {}",
                    expected_h0[d]
                );
                assert!(
                    (got_h1 - expected_h1[d]).abs() < 1e-4,
                    "h1 b{b} d{d}: {got_h1} vs {}",
                    expected_h1[d]
                );
            }
        }
    }

    #[test]
    fn attention_single_key_dominant_selects_its_value() {
        // Make token 1's K overwhelmingly similar to every Q, rest
        // orthogonal. Softmax picks token 1 → output row ≈ V[1].
        let batch = 3;
        let num_heads = 1;
        let head_dim = 4;
        let n = batch * num_heads * head_dim;
        // Q all = [1, 0, 0, 0] (points at dim 0)
        let mut q = vec![0f32; n];
        for b in 0..batch {
            q[b * head_dim] = 1.0;
        }
        // K: token 0 = [100,0,0,0] (huge along dim 0), others orthogonal.
        // That inverts the usual delta test — the point is ONE key
        // dominates softmax and we should read V[that-key].
        let mut k = vec![0f32; n];
        k[0 * head_dim] = 100.0; // token 0's key points huge along dim 0
        // Token 1: orthogonal
        k[1 * head_dim + 1] = 100.0;
        // Token 2: orthogonal
        k[2 * head_dim + 2] = 100.0;
        // V: each token has a unique signature
        let v = vec![
            7.0, 7.0, 7.0, 7.0, // token 0
            1.0, 2.0, 3.0, 4.0, // token 1
            99.0, 99.0, 99.0, 99.0, // token 2
        ];
        let out = scaled_dot_product_attention(&q, &k, &v, batch, num_heads, head_dim).unwrap();
        // Every query [1, 0, 0, 0] dots with K[0] = 100 (high score),
        // K[1] = 0, K[2] = 0. Softmax overwhelmingly picks token 0 → V[0] = [7,7,7,7].
        for b in 0..batch {
            for d in 0..head_dim {
                let got = out[b * head_dim + d];
                assert!(
                    (got - 7.0).abs() < 0.1,
                    "b{b} d{d}: {got} (expected ≈7.0)"
                );
            }
        }
    }

    #[test]
    fn attention_scale_factor_applied_to_logits() {
        // Large head_dim + moderate Q/K magnitudes → without scaling,
        // logits overflow exp. Check that the scale keeps outputs finite.
        let batch = 4;
        let num_heads = 1;
        let head_dim = 1024; // large; uses 1/sqrt(1024) = 0.03125
        let n = batch * num_heads * head_dim;
        let q = vec![3.0f32; n];
        let k = vec![3.0f32; n];
        let v = vec![1.0f32; n];
        let out = scaled_dot_product_attention(&q, &k, &v, batch, num_heads, head_dim).unwrap();
        for val in &out {
            assert!(val.is_finite(), "non-finite: {val}");
        }
        // Uniform Q/K → uniform softmax → output == mean(V) = 1.0.
        for val in &out {
            assert!((*val - 1.0).abs() < 1e-3, "got {val}");
        }
    }

    #[test]
    fn attention_rejects_zero_dims() {
        let err = scaled_dot_product_attention(&[], &[], &[], 0, 1, 1).unwrap_err();
        assert!(format!("{err}").contains("must all be > 0"));
    }

    #[test]
    fn attention_rejects_mismatched_q_len() {
        let batch = 2;
        let num_heads = 2;
        let head_dim = 2;
        let n = batch * num_heads * head_dim;
        let q = vec![0f32; n - 1];
        let k = vec![0f32; n];
        let v = vec![0f32; n];
        let err = scaled_dot_product_attention(&q, &k, &v, batch, num_heads, head_dim)
            .unwrap_err();
        assert!(format!("{err}").contains("q len"));
    }

    #[test]
    fn attention_rejects_mismatched_k_len() {
        let batch = 2;
        let num_heads = 2;
        let head_dim = 2;
        let n = batch * num_heads * head_dim;
        let q = vec![0f32; n];
        let k = vec![0f32; n + 1];
        let v = vec![0f32; n];
        let err = scaled_dot_product_attention(&q, &k, &v, batch, num_heads, head_dim)
            .unwrap_err();
        assert!(format!("{err}").contains("k len"));
    }

    #[test]
    fn attention_end_to_end_real_gemma4_full_self_attention() {
        // Extends iter 36's chain test by running the full self-attention
        // through scaled-dot-product. Drives:
        //   preprocess → patch_embed → rms_norm(ln1) → qkv_projection →
        //   per_head_rms_norm(Q, K) → scaled_dot_product_attention
        // All against real Gemma 4 pretrained block-0 weights.
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

        let hidden = cfg.hidden_size as usize;
        let num_heads = cfg.num_attention_heads as usize;
        let head_dim = hidden / num_heads;
        let num_patches = 196usize;
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
        let ln1 = weights.block_tensor(0, "ln1.weight").expect("ln1");
        let ln1_g: &[f32] = ln1.as_slice::<f32>().expect("ln1 slice");
        rms_norm_forward(&mut hidden_states, ln1_g, hidden, 1e-6).expect("rms ln1");

        let q_buf = weights.block_tensor(0, "attn_q.weight").expect("q");
        let k_buf = weights.block_tensor(0, "attn_k.weight").expect("k");
        let v_buf = weights.block_tensor(0, "attn_v.weight").expect("v");
        let (mut q, mut k, v) = qkv_projection_forward(
            &hidden_states,
            q_buf.as_slice::<f32>().expect("q slice"),
            k_buf.as_slice::<f32>().expect("k slice"),
            v_buf.as_slice::<f32>().expect("v slice"),
            num_patches,
            hidden,
        )
        .expect("qkv");

        let q_norm = weights
            .block_tensor(0, "attn_q_norm.weight")
            .expect("attn_q_norm");
        let k_norm = weights
            .block_tensor(0, "attn_k_norm.weight")
            .expect("attn_k_norm");
        per_head_rms_norm_forward(
            &mut q,
            q_norm.as_slice::<f32>().expect("q_norm slice"),
            num_patches,
            num_heads,
            head_dim,
            1e-6,
        )
        .expect("per-head Q");
        per_head_rms_norm_forward(
            &mut k,
            k_norm.as_slice::<f32>().expect("k_norm slice"),
            num_patches,
            num_heads,
            head_dim,
            1e-6,
        )
        .expect("per-head K");

        // THE NEW STAGE: scaled-dot-product attention.
        let attn_out = scaled_dot_product_attention(&q, &k, &v, num_patches, num_heads, head_dim)
            .expect("attention");
        assert_eq!(attn_out.len(), num_patches * hidden);
        for val in &attn_out {
            assert!(val.is_finite(), "non-finite attention output: {val}");
        }
        // Distribution sanity: different query positions should produce
        // different attention outputs (bidirectional but each query is
        // unique because the input gradient was per-pixel).
        let token_0 = &attn_out[0..hidden];
        let token_last = &attn_out[(num_patches - 1) * hidden..num_patches * hidden];
        let l2_diff: f32 = token_0
            .iter()
            .zip(token_last.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            l2_diff > 1e-3,
            "token 0 and token 195 attention outputs identical — stride bug?"
        );
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

    // -------------------------------------------------------------------
    // gemma4v primitives — CPU-side correctness + shape tests
    // -------------------------------------------------------------------

    #[test]
    fn gemma4v_patch_embed_shapes_and_math() {
        // 4 patches × inner=12 → hidden=8. Hand-rolled GEMM check on
        // a tiny shape: y[n, o] = Σᵢ patches[n, i] * weight[o, i].
        let n_patches = 4u32;
        let inner = 12u32;
        let hidden = 8u32;
        let patches: Vec<f32> = (0..(n_patches * inner)).map(|i| i as f32 * 0.01).collect();
        let weight: Vec<f32> = (0..(hidden * inner))
            .map(|i| ((i % 5) as f32) * 0.1 - 0.2)
            .collect();
        let out =
            gemma4v_patch_embed_forward(&patches, &weight, n_patches, inner, hidden).unwrap();
        assert_eq!(out.len(), (n_patches * hidden) as usize);

        // Spot-check one cell against the formula directly.
        let n = 2usize;
        let o = 3usize;
        let in_us = inner as usize;
        let mut expect: f32 = 0.0;
        for i in 0..in_us {
            expect += patches[n * in_us + i] * weight[o * in_us + i];
        }
        let got = out[n * (hidden as usize) + o];
        assert!(
            (got - expect).abs() < 1e-5,
            "got {got} expect {expect}"
        );
    }

    #[test]
    fn gemma4v_patch_embed_rejects_zero_dims() {
        let err = gemma4v_patch_embed_forward(&[1.0], &[1.0], 0, 1, 1).unwrap_err();
        assert!(format!("{err}").contains("must all be > 0"));
        let err2 = gemma4v_patch_embed_forward(&[1.0], &[1.0], 1, 0, 1).unwrap_err();
        assert!(format!("{err2}").contains("must all be > 0"));
    }

    #[test]
    fn gemma4v_patch_embed_rejects_shape_mismatch() {
        // Patches buf too small.
        let err = gemma4v_patch_embed_forward(&[1.0; 5], &[1.0; 24], 4, 12, 2).unwrap_err();
        assert!(format!("{err}").contains("patches.len()"));
        // Weight buf too small.
        let err2 = gemma4v_patch_embed_forward(&[1.0; 48], &[1.0; 5], 4, 12, 2).unwrap_err();
        assert!(format!("{err2}").contains("weight.len()"));
    }

    #[test]
    fn gemma4v_position_embed_lookup_basic() {
        // pos_size=3, hidden=4 → table is 2*3*4 = 24 floats.
        // Table[0] (X-axis): rows [10,11,12,13], [20,21,22,23], [30,31,32,33]
        // Table[1] (Y-axis): rows [40,41,42,43], [50,51,52,53], [60,61,62,63]
        let pe: Vec<f32> = vec![
            10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0, 33.0,
            40.0, 41.0, 42.0, 43.0, 50.0, 51.0, 52.0, 53.0, 60.0, 61.0, 62.0, 63.0,
        ];
        // 4 patches at (x, y) = (0,0), (1,0), (0,1), (2,2).
        let pos_x = vec![0u32, 1, 0, 2];
        let pos_y = vec![0u32, 0, 1, 2];
        let out = gemma4v_position_embed_lookup(&pos_x, &pos_y, &pe, 3, 4).unwrap();
        assert_eq!(out.len(), 16);
        // patch 0: pe_x[0] + pe_y[0] = [10+40, 11+41, 12+42, 13+43]
        assert_eq!(&out[0..4], &[50.0, 52.0, 54.0, 56.0]);
        // patch 1: pe_x[1] + pe_y[0]
        assert_eq!(&out[4..8], &[60.0, 62.0, 64.0, 66.0]);
        // patch 2: pe_x[0] + pe_y[1]
        assert_eq!(&out[8..12], &[60.0, 62.0, 64.0, 66.0]);
        // patch 3: pe_x[2] + pe_y[2]
        assert_eq!(&out[12..16], &[90.0, 92.0, 94.0, 96.0]);
    }

    #[test]
    fn gemma4v_position_embed_lookup_clamps_out_of_range() {
        // pos_size=2; index 99 should clamp to 1 (max_idx = pos_size − 1).
        let pe: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let pos_x = vec![99u32];
        let pos_y = vec![99u32];
        // pos_size=2, hidden=2 → table is 2*2*2 = 8.
        let out = gemma4v_position_embed_lookup(&pos_x, &pos_y, &pe, 2, 2).unwrap();
        // X-table[1] = [3, 4], Y-table[1] = [30, 40], sum = [33, 44].
        assert_eq!(out, vec![33.0, 44.0]);
    }

    #[test]
    fn gemma4v_position_embed_lookup_shape_errors() {
        let pe = vec![0f32; 12];
        // Mismatched pos arrays.
        let err =
            gemma4v_position_embed_lookup(&[0u32, 1], &[0u32], &pe, 3, 2).unwrap_err();
        assert!(format!("{err}").contains("pos_x.len()"));
        // Wrong table size.
        let err2 = gemma4v_position_embed_lookup(&[0u32], &[0u32], &pe, 7, 2).unwrap_err();
        assert!(format!("{err2}").contains("pe_table.len()"));
    }

    #[test]
    fn gemma4v_position_embed_add_in_place() {
        let mut patch_embeds = vec![1.0_f32; 8]; // [2, 4]
        // pos_size=2, hidden=4 → table is 16 floats.
        let pe: Vec<f32> = vec![
            // X-table:
            1.0, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
            // Y-table:
            9.0, 10.0, 11.0, 12.0, // row 0
            13.0, 14.0, 15.0, 16.0, // row 1
        ];
        let pos_x = vec![0u32, 1];
        let pos_y = vec![1u32, 0];
        gemma4v_position_embed_add(&mut patch_embeds, &pos_x, &pos_y, &pe, 2, 4).unwrap();
        // patch 0: 1 + (1+13, 2+14, 3+15, 4+16) = [15, 17, 19, 21]
        assert_eq!(&patch_embeds[0..4], &[15.0, 17.0, 19.0, 21.0]);
        // patch 1: 1 + (5+9, 6+10, 7+11, 8+12) = [15, 17, 19, 21]
        assert_eq!(&patch_embeds[4..8], &[15.0, 17.0, 19.0, 21.0]);
    }

    // -----------------------------------------------------------------------
    // gemma4v_clippable_linear_forward (iter 115)
    // -----------------------------------------------------------------------

    #[test]
    fn gemma4v_clippable_linear_forward_no_bounds_matches_plain_linear() {
        // bounds.any() == false → byte-equivalent to linear_forward(_, None).
        let batch = 3usize;
        let in_features = 4usize;
        let out_features = 2usize;
        let input: Vec<f32> = vec![
            -1.0, 2.0, 3.0, 4.0,
             5.0, -6.0, 7.0, 8.0,
            -9.0, 10.0, -11.0, 12.0,
        ];
        let weight: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
        ];
        let bounds = Gemma4ClippableLinearBounds::default();
        assert!(!bounds.any());

        let plain =
            linear_forward(&input, &weight, None, batch, in_features, out_features).unwrap();
        let clipped = gemma4v_clippable_linear_forward(
            &input, &weight, &bounds, batch, in_features, out_features,
        )
        .unwrap();
        assert_eq!(plain, clipped, "no-bounds clippable_linear must match plain linear");
    }

    #[test]
    fn gemma4v_clippable_linear_forward_input_clamp_only() {
        // Input element -100 should be clamped to -1.0 BEFORE the matmul.
        let batch = 1usize;
        let in_features = 4usize;
        let out_features = 1usize;
        let input: Vec<f32> = vec![-100.0, 1.0, 1.0, 1.0];
        let weight: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let bounds = Gemma4ClippableLinearBounds {
            input_min: Some(-1.0),
            input_max: Some(1.0),
            output_min: None,
            output_max: None,
        };
        let out = gemma4v_clippable_linear_forward(
            &input, &weight, &bounds, batch, in_features, out_features,
        )
        .unwrap();
        // Expected: clamp(-100, -1, 1) = -1, then dot([- 1, 1, 1, 1], [1, 1, 1, 1]) = 2.
        assert_eq!(out, vec![2.0]);
    }

    #[test]
    fn gemma4v_clippable_linear_forward_both_clamps() {
        // Input pre-clamps to [-2, 2]; output post-clamps to [-3, 3].
        let batch = 1usize;
        let in_features = 4usize;
        let out_features = 1usize;
        let input: Vec<f32> = vec![100.0, 100.0, 100.0, 100.0];
        let weight: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let bounds = Gemma4ClippableLinearBounds {
            input_min: Some(-2.0),
            input_max: Some(2.0),
            output_min: Some(-3.0),
            output_max: Some(3.0),
        };
        let out = gemma4v_clippable_linear_forward(
            &input, &weight, &bounds, batch, in_features, out_features,
        )
        .unwrap();
        // After input clamp: [2, 2, 2, 2]. dot = 8. Output clamped to 3.
        assert_eq!(out, vec![3.0]);
    }

    #[test]
    fn gemma4v_clippable_linear_bounds_resolve_default_to_neg_pos_inf() {
        let bounds = Gemma4ClippableLinearBounds::default();
        let (mn_in, mx_in) = bounds.resolved_input();
        let (mn_out, mx_out) = bounds.resolved_output();
        assert_eq!(mn_in, f32::NEG_INFINITY);
        assert_eq!(mx_in, f32::INFINITY);
        assert_eq!(mn_out, f32::NEG_INFINITY);
        assert_eq!(mx_out, f32::INFINITY);
        assert!(!bounds.any());
    }

    #[test]
    fn gemma4v_clippable_linear_forward_rejects_min_gt_max() {
        let input: Vec<f32> = vec![0.0; 4];
        let weight: Vec<f32> = vec![0.0; 4];
        let bounds = Gemma4ClippableLinearBounds {
            input_min: Some(5.0),
            input_max: Some(1.0),
            ..Default::default()
        };
        let err = gemma4v_clippable_linear_forward(&input, &weight, &bounds, 1, 4, 1).unwrap_err();
        assert!(format!("{err}").contains("input_min"));
    }
}
