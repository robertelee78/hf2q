//! DFlash drafter forward dispatch primitives (ADR-030 Phase 2).
//!
//! This module composes mlx-native dispatchers into the small forward
//! steps that make up `DFlashDraftModel.__call__`
//! (`/opt/dflash/dflash/model_mlx.py:181-198`) and `DFlashAttention.__call__`
//! (`model_mlx.py:82-116`).
//!
//! Builds up incrementally — each commit adds one composable piece with
//! its own smoke test against real drafter weights on M5 Max Metal.
//! Per mantra (no stubs, no fallback): every function COMPUTES SOMETHING
//! USEFUL and ships tested. The final `DFlashDraftModel::forward` is
//! composed from these pieces in a later iter.
//!
//! ## Current iter (iter-18): input_layernorm + Q projection
//!
//! ```text
//!   h: [L, hidden] F32
//!         │
//!         ▼  input_layernorm (RMSNorm, weight from DFlashLayerTensors)
//!   normed: [L, hidden] F32
//!         │
//!         ▼  q_proj (BF16 weight @ F32 input via apply_linear_projection_f32)
//!   q: [L, num_q_heads * head_dim] F32
//! ```
//!
//! Future iters add: K/V proj, q_norm/k_norm, RoPE, SDPA, concat, O proj
//! (attention complete); SwiGLU MLP; layer composition (residuals);
//! model composition (fc + per-layer + final norm + lm_head).

use super::config::DFlashConfig;
use super::tensors::DFlashLayerTensors;
use anyhow::{anyhow, Context, Result};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};
use mlx_native::ops::elementwise::elementwise_add;
use mlx_native::ops::rms_norm::dispatch_rms_norm;
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::ops::softcap::dispatch_softcap;
use crate::inference::models::qwen35::gpu_full_attn::{
    apply_imrope, apply_linear_projection_f32, apply_sdpa_causal_from_seq_major,
    permute_seq_head_dim_to_head_seq_dim_cpu, upload_f32,
};
use mlx_native::ops::sdpa::{sdpa, SdpaParams};

/// Logical-bounded F32 download for DFlash drafter CPU-read sites.
///
/// `MlxBuffer::as_slice` returns a view of the full storage, which can
/// be bucket-rounded by `decode_pool::pooled_alloc_buffer` (used inside
/// `apply_imrope`). Bucket-rounded storage is safe for GPU-only feeds
/// (the qwen35 hot path's contract), but the DFlash drafter must
/// CPU-read the K/V projections to populate the cursor-mode cache, and
/// the trailing bucket padding must NOT be appended.
///
/// This helper truncates to `buf.element_count()` (the product of the
/// buffer's logical shape) before copying to a `Vec<f32>`. When the
/// allocator returned a non-pooled buffer (e.g. `apply_linear_projection_f32`)
/// `element_count()` equals storage and the helper is a no-op vs
/// `download_f32`.
///
/// See `src/inference/models/qwen35/gpu_full_attn.rs:649` (apply_imrope
/// pool alloc + the `kv_cache_slot=Some` GPU-only safety note that we
/// explicitly do not satisfy in the DFlash drafter).
fn download_f32_logical(buf: &MlxBuffer) -> Result<Vec<f32>> {
    if buf.dtype() != DType::F32 {
        return Err(anyhow!(
            "download_f32_logical: buffer dtype {} != f32",
            buf.dtype()
        ));
    }
    let storage: &[f32] = buf.as_slice().map_err(|e| anyhow!("as_slice: {e}"))?;
    let logical = buf.element_count();
    if storage.len() < logical {
        return Err(anyhow!(
            "download_f32_logical: storage slice len {} < logical element_count {} \
             (shape {:?})",
            storage.len(),
            logical,
            buf.shape(),
        ));
    }
    Ok(storage[..logical].to_vec())
}

/// Build the `[eps, dim]` F32 params buffer required by [`dispatch_rms_norm`].
///
/// Per `mlx_native::ops::rms_norm` doc, the kernel reads two F32 values
/// from this buffer. We allocate a tiny dedicated buffer per call; the
/// cost is amortized over the rms_norm dispatch itself.
fn alloc_rms_norm_params(device: &MlxDevice, eps: f32, dim: u32) -> Result<MlxBuffer> {
    let mut params = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("alloc rms_norm params: {e}"))?;
    let slice = params
        .as_mut_slice::<f32>()
        .map_err(|e| anyhow!("rms_norm params slice: {e}"))?;
    slice[0] = eps;
    slice[1] = dim as f32;
    Ok(params)
}

/// Dispatch input_layernorm (`RMSNorm(h)`) into the encoder.
///
/// Mirrors `model_mlx.py:DFlashDecoderLayer.__call__` first stage:
/// `self.input_layernorm(x)`.
///
/// # Arguments
///
/// - `h_input`: `[L, hidden_size]` F32 input hidden state
/// - `layer`: per-layer drafter weights (uses `input_layernorm`)
/// - Returns: `[L, hidden_size]` F32 normalized output
///
/// The output buffer is allocated fresh; the encoder is NOT committed
/// by this function (caller composes multiple dispatches before commit).
pub fn dispatch_dflash_input_layernorm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    h_input: &MlxBuffer,
    layer: &DFlashLayerTensors,
    cfg: &DFlashConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    let hidden = cfg.hidden_size as u32;
    let element_count = (seq_len as usize) * (hidden as usize);
    if h_input.element_count() != element_count {
        return Err(anyhow!(
            "dflash input_layernorm: h_input element count {} != L({}) * hidden({})",
            h_input.element_count(),
            seq_len,
            hidden
        ));
    }
    let normed = device
        .alloc_buffer(element_count * 4, DType::F32, vec![seq_len as usize, hidden as usize])
        .map_err(|e| anyhow!("alloc input_layernorm output: {e}"))?;
    let params = alloc_rms_norm_params(device, cfg.rms_norm_eps, hidden)?;
    dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        h_input,
        &layer.input_layernorm,
        &normed,
        &params,
        seq_len,
        hidden,
    )
    .context("dispatch_rms_norm input_layernorm")?;
    Ok(normed)
}

/// Dispatch the Q projection: `q_proj @ input`.
///
/// Mirrors `model_mlx.py:DFlashAttention.__call__` line 92:
/// `queries = self.q_proj(x)`.
///
/// # Arguments
///
/// - `input`: `[L, hidden_size]` F32 input (normed `x` for queries)
/// - `layer`: per-layer drafter weights (uses `q_proj` BF16 weight
///   shape `[num_q_heads * head_dim, hidden_size]`)
/// - Returns: `[L, num_q_heads * head_dim]` F32 (unreshaped)
///
/// Reshape into `[L, num_q_heads, head_dim]` for q_norm + RoPE happens
/// in a later piece (it is metadata-only in flat row-major layout).
/// The encoder is NOT committed.
pub fn dispatch_dflash_q_proj(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    layer: &DFlashLayerTensors,
    cfg: &DFlashConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    let hidden = cfg.hidden_size as u32;
    let q_out_dim = (cfg.num_attention_heads * cfg.head_dim) as u32;
    apply_linear_projection_f32(
        encoder,
        registry,
        device,
        input,
        &layer.q_proj,
        seq_len,
        hidden,
        q_out_dim,
    )
    .context("dispatch_dflash_q_proj")
}

/// Dispatch the K projection: `k_proj @ input`.
///
/// Mirrors `model_mlx.py:DFlashAttention.__call__` line 93/95:
/// `ctx_keys = self.k_proj(x_ctx)` / `prop_keys = self.k_proj(x)`.
///
/// The K projection is called TWICE per attention forward — once on
/// `x_ctx` (target hidden states the cache hasn't seen yet) and once
/// on the normed block input itself (for the prop_keys path). This
/// helper handles either call; caller passes whichever `input` is
/// being projected.
///
/// # Arguments
///
/// - `input`: `[L_or_S, hidden_size]` F32 input
/// - `layer`: uses `k_proj` BF16 weight `[num_kv_heads * head_dim, hidden_size]`
/// - Returns: `[L_or_S, num_kv_heads * head_dim]` F32
pub fn dispatch_dflash_k_proj(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    layer: &DFlashLayerTensors,
    cfg: &DFlashConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    let hidden = cfg.hidden_size as u32;
    let kv_out_dim = (cfg.num_key_value_heads * cfg.head_dim) as u32;
    apply_linear_projection_f32(
        encoder,
        registry,
        device,
        input,
        &layer.k_proj,
        seq_len,
        hidden,
        kv_out_dim,
    )
    .context("dispatch_dflash_k_proj")
}

/// Dispatch the V projection: `v_proj @ input`.
///
/// Mirrors `model_mlx.py:DFlashAttention.__call__` line 94/96:
/// `ctx_values = self.v_proj(x_ctx)` / `prop_values = self.v_proj(x)`.
///
/// Same call sites as `dispatch_dflash_k_proj` — V is computed
/// alongside K for both `x_ctx` and `x`. Per qwen3 convention there is
/// no `v_norm` (only `q_norm` and `k_norm`).
///
/// # Arguments
///
/// - `input`: `[L_or_S, hidden_size]` F32 input
/// - `layer`: uses `v_proj` BF16 weight `[num_kv_heads * head_dim, hidden_size]`
/// - Returns: `[L_or_S, num_kv_heads * head_dim]` F32
/// Apply per-head RMSNorm to a `[L, num_heads, head_dim]` projection
/// output. Used for both `q_norm` (on Q) and `k_norm` (on K).
///
/// Mirrors `model_mlx.py:DFlashAttention.__call__` lines 97-100:
/// `q_norm(queries.reshape(B, L, n_heads, -1))`.
///
/// The reshape `[L, n_heads * head_dim] → [L, n_heads, head_dim]` is
/// metadata-only in our flat row-major layout. With `rows = L * n_heads`
/// and `dim = head_dim`, mlx-native's `dispatch_rms_norm` normalizes
/// each head independently — exactly what qwen3-style per-head norm
/// requires.
///
/// The transpose to `[n_heads, L, head_dim]` for SDPA is NOT done here
/// (deferred to a separate dispatcher / SDPA layout-conversion).
///
/// # Arguments
///
/// - `proj`: `[L, num_heads * head_dim]` F32 (output of Q or K proj)
/// - `norm_weight`: `[head_dim]` F32 (q_norm or k_norm)
/// - Returns: `[L, num_heads * head_dim]` F32 (same shape, normalized)
pub fn dispatch_dflash_head_norm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    proj: &MlxBuffer,
    norm_weight: &MlxBuffer,
    cfg: &DFlashConfig,
    seq_len: u32,
    num_heads: u32,
) -> Result<MlxBuffer> {
    let head_dim = cfg.head_dim as u32;
    let rows = seq_len * num_heads;
    let expected_elem = (rows as usize) * (head_dim as usize);
    if proj.element_count() != expected_elem {
        return Err(anyhow!(
            "dflash head_norm: proj element count {} != rows({}) * head_dim({})",
            proj.element_count(),
            rows,
            head_dim
        ));
    }
    let normed = device
        .alloc_buffer(expected_elem * 4, DType::F32, vec![seq_len as usize, num_heads as usize, head_dim as usize])
        .map_err(|e| anyhow!("alloc head_norm output: {e}"))?;
    let params = alloc_rms_norm_params(device, cfg.rms_norm_eps, head_dim)?;
    dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        proj,
        norm_weight,
        &normed,
        &params,
        rows,
        head_dim,
    )
    .context("dispatch_rms_norm head_norm")?;
    Ok(normed)
}

pub fn dispatch_dflash_v_proj(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    layer: &DFlashLayerTensors,
    cfg: &DFlashConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    let hidden = cfg.hidden_size as u32;
    let kv_out_dim = (cfg.num_key_value_heads * cfg.head_dim) as u32;
    apply_linear_projection_f32(
        encoder,
        registry,
        device,
        input,
        &layer.v_proj,
        seq_len,
        hidden,
        kv_out_dim,
    )
    .context("dispatch_dflash_v_proj")
}

/// Dispatch the SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
///
/// Mirrors `mlx_lm.models.qwen3.MLP.__call__` (the qwen3 MLP that
/// DFlashDecoderLayer uses per `model_mlx.py:123`):
///
/// ```text
///   gate = gate_proj @ x        [L, intermediate_size]
///   up   = up_proj   @ x        [L, intermediate_size]
///   h    = silu(gate) * up      [L, intermediate_size]  (fused via dispatch_silu_mul)
///   out  = down_proj @ h        [L, hidden_size]
/// ```
///
/// Three matmuls + one fused element-wise op. The encoder is NOT
/// committed; caller composes with residual add + next layer.
///
/// # Arguments
///
/// - `input`: `[L, hidden_size]` F32 (typically post-attention-norm output)
/// - `layer`: uses `mlp_gate`, `mlp_up`, `mlp_down` BF16 weights
/// - Returns: `[L, hidden_size]` F32
pub fn dispatch_dflash_mlp(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    layer: &DFlashLayerTensors,
    cfg: &DFlashConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    let hidden = cfg.hidden_size as u32;
    let inter = cfg.intermediate_size as u32;

    // 1. gate = mlp_gate @ input → [L, intermediate_size]
    let gate = apply_linear_projection_f32(
        encoder, registry, device, input, &layer.mlp_gate, seq_len, hidden, inter,
    )
    .context("dispatch_dflash_mlp: gate_proj")?;

    // 2. up = mlp_up @ input → [L, intermediate_size]
    let up = apply_linear_projection_f32(
        encoder, registry, device, input, &layer.mlp_up, seq_len, hidden, inter,
    )
    .context("dispatch_dflash_mlp: up_proj")?;

    // RAW barrier: silu_mul reads gate + up just written by the two matmuls.
    encoder.memory_barrier();

    // 3. silu(gate) * up → activated [L, intermediate_size]
    let n_h = seq_len * inter;
    let mut silu_params = device
        .alloc_buffer(4, DType::U32, vec![1])
        .map_err(|e| anyhow!("alloc mlp silu_params: {e}"))?;
    silu_params
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("silu_params slice: {e}"))?[0] = n_h;
    let activated = device
        .alloc_buffer(
            (n_h as usize) * 4,
            DType::F32,
            vec![seq_len as usize, inter as usize],
        )
        .map_err(|e| anyhow!("alloc mlp activated: {e}"))?;
    dispatch_silu_mul(
        encoder,
        registry,
        device.metal_device(),
        &gate,
        &up,
        &activated,
        &silu_params,
        n_h,
    )
    .context("dispatch_dflash_mlp: silu_mul")?;

    // RAW barrier: down_proj reads activated.
    encoder.memory_barrier();

    // 4. mlp_down @ activated → [L, hidden_size]
    apply_linear_projection_f32(
        encoder, registry, device, &activated, &layer.mlp_down, seq_len, inter, hidden,
    )
    .context("dispatch_dflash_mlp: down_proj")
}

/// Build the per-axis position buffer required by mlx-native's
/// IMROPE kernel.
///
/// Format: `int32[4 * seq_len]` — four axis blocks each of length
/// `seq_len`. For DFlash's plain NeoX RoPE (text-only) we replicate the
/// same `[offset, offset+1, …, offset+seq_len-1]` block across all four
/// axes; the sections=`[head_dim/2, 0, 0, 0]` config sends every pair
/// to axis 0 so axes 1-3 are unused, but the kernel still expects a
/// non-zero pos_buf of the right shape.
fn build_dflash_pos_buf(
    device: &MlxDevice,
    seq_len: u32,
    offset: u32,
) -> Result<MlxBuffer> {
    let n_pos = 4 * (seq_len as usize);
    let mut buf = device
        .alloc_buffer(n_pos * 4, DType::I32, vec![n_pos])
        .map_err(|e| anyhow!("alloc rope pos_buf: {e}"))?;
    let slice = buf
        .as_mut_slice::<i32>()
        .map_err(|e| anyhow!("rope pos_buf slice: {e}"))?;
    let l = seq_len as usize;
    let base = offset as i32;
    for axis in 0..4 {
        let dst = &mut slice[axis * l..(axis + 1) * l];
        for (i, v) in dst.iter_mut().enumerate() {
            *v = base + (i as i32);
        }
    }
    Ok(buf)
}

/// Apply NeoX-style RoPE to a per-head-normalized Q or K buffer.
///
/// Mirrors `model_mlx.py:DFlashAttention.__call__` lines 102-104:
/// `queries = rope(queries, offset=cache.offset + S)` and similar for
/// `ctx_keys` / `prop_keys` with different offsets.
///
/// Reuses qwen35's `apply_imrope` (gpu_full_attn.rs:620) since drafter
/// is qwen3-style. Drafter RoPE config: head_dim=128, rope_dim=128 (full
/// rotation, traditional=False), freq_base=1_000_000, plain NeoX = all
/// pairs in axis 0 via sections=`[head_dim/2, 0, 0, 0]`.
///
/// # Arguments
///
/// - `qk_in`: `[seq_len * num_heads, head_dim]` F32 (post per-head norm)
/// - `seq_len`: number of token positions in the input
/// - `num_heads`: number of heads (Q: num_q_heads; K: num_kv_heads)
/// - `offset`: starting position in the sequence (the Python `offset` arg)
/// - Returns: `[seq_len * num_heads, head_dim]` F32 rotated
pub fn dispatch_dflash_rope(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    qk_in: &MlxBuffer,
    cfg: &DFlashConfig,
    seq_len: u32,
    num_heads: u32,
    offset: u32,
) -> Result<MlxBuffer> {
    let head_dim = cfg.head_dim as u32;
    let rope_dim = head_dim; // DFlash drafter uses full rotation
    let freq_base = cfg.rope_theta;
    // Plain NeoX: all pairs go to axis 0. Sum must equal rope_dim / 2.
    let sections = [head_dim / 2, 0, 0, 0];
    let positions = build_dflash_pos_buf(device, seq_len, offset)?;
    apply_imrope(
        encoder,
        registry,
        device,
        qk_in,
        &positions,
        seq_len,
        num_heads,
        head_dim,
        rope_dim,
        freq_base,
        sections,
    )
    .context("dispatch_dflash_rope")
}

/// Dispatch the fc projection: `fc @ target_hidden_concat`.
///
/// Mirrors `model_mlx.py:DFlashDraftModel.__call__` line 189:
/// `h_ctx = self.hidden_norm(self.fc(target_hidden))`. This is the
/// fc(.) half — projects the concatenated multi-layer target hidden
/// state (one row per ctx position) into the drafter's hidden space.
///
/// # Arguments
///
/// - `target_hidden_concat`: `[ctx_seq_len, num_target_layers_used * hidden_size]`
///   F32 — concat along the last axis of target hidden states at
///   `target_layer_ids = [1, 6, 11, 17, 22, 27]` (6 × 2816 = 16896)
/// - `model`: drafter model weights (uses `fc` BF16 `[hidden, fc_input_dim]`)
/// - Returns: `[ctx_seq_len, hidden_size]` F32
pub fn dispatch_dflash_fc(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    target_hidden_concat: &MlxBuffer,
    model: &super::tensors::DFlashModelTensors,
    cfg: &DFlashConfig,
    ctx_seq_len: u32,
) -> Result<MlxBuffer> {
    let fc_in = cfg.fc_input_dim() as u32;
    let hidden = cfg.hidden_size as u32;
    apply_linear_projection_f32(
        encoder, registry, device,
        target_hidden_concat, &model.fc,
        ctx_seq_len, fc_in, hidden,
    )
    .context("dispatch_dflash_fc")
}

/// Apply `hidden_norm` RMSNorm to the fc output → h_ctx.
///
/// Mirrors `model_mlx.py:DFlashDraftModel.__call__` line 189's `hidden_norm(.)`.
/// Uses model-level (not per-layer) RMSNorm weight `[hidden_size]`.
pub fn dispatch_dflash_hidden_norm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    fc_out: &MlxBuffer,
    model: &super::tensors::DFlashModelTensors,
    cfg: &DFlashConfig,
    ctx_seq_len: u32,
) -> Result<MlxBuffer> {
    let hidden = cfg.hidden_size as u32;
    let element_count = (ctx_seq_len as usize) * (hidden as usize);
    if fc_out.element_count() != element_count {
        return Err(anyhow!(
            "dflash hidden_norm: fc_out element count {} != S({}) * hidden({})",
            fc_out.element_count(),
            ctx_seq_len,
            hidden
        ));
    }
    let normed = device
        .alloc_buffer(
            element_count * 4,
            DType::F32,
            vec![ctx_seq_len as usize, hidden as usize],
        )
        .map_err(|e| anyhow!("alloc hidden_norm output: {e}"))?;
    let params = alloc_rms_norm_params(device, cfg.rms_norm_eps, hidden)?;
    dispatch_rms_norm(
        encoder, registry, device.metal_device(),
        fc_out, &model.hidden_norm, &normed, &params,
        ctx_seq_len, hidden,
    )
    .context("dispatch_rms_norm hidden_norm")?;
    Ok(normed)
}

/// Apply `final_norm` (`norm.weight`) RMSNorm to the last layer's
/// output, just before lm_head.
///
/// Mirrors `model_mlx.py:DFlashDraftModel.__call__` line 194:
/// `logits = self.lm_head(self.norm(h))`. This is `self.norm(h)` —
/// the final RMSNorm with model-level weight `norm.weight`.
pub fn dispatch_dflash_final_norm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    h: &MlxBuffer,
    model: &super::tensors::DFlashModelTensors,
    cfg: &DFlashConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    let hidden = cfg.hidden_size as u32;
    let element_count = (seq_len as usize) * (hidden as usize);
    if h.element_count() != element_count {
        return Err(anyhow!(
            "dflash final_norm: h element count {} != L({}) * hidden({})",
            h.element_count(),
            seq_len,
            hidden
        ));
    }
    let normed = device
        .alloc_buffer(
            element_count * 4,
            DType::F32,
            vec![seq_len as usize, hidden as usize],
        )
        .map_err(|e| anyhow!("alloc final_norm output: {e}"))?;
    let params = alloc_rms_norm_params(device, cfg.rms_norm_eps, hidden)?;
    dispatch_rms_norm(
        encoder, registry, device.metal_device(),
        h, &model.final_norm, &normed, &params,
        seq_len, hidden,
    )
    .context("dispatch_rms_norm final_norm")?;
    Ok(normed)
}

/// Apply Gemma-style logit softcap: `out = tanh(logits / cap) * cap`.
///
/// Mirrors `model_mlx.py:DFlashDraftModel.__call__` lines 195-197:
/// ```text
///   if final_logit_softcapping is not None:
///       cap = final_logit_softcapping
///       logits = tanh(logits / cap) * cap
/// ```
/// For our drafter `cfg.final_logit_softcapping == Some(30.0)`. The
/// op is element-wise on F32 logits in place — output buffer is
/// allocated fresh here.
///
/// Returns the original buffer unchanged if `cfg.final_logit_softcapping`
/// is `None` (no softcap configured).
pub fn dispatch_dflash_softcap(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    logits: &MlxBuffer,
    cfg: &DFlashConfig,
) -> Result<Option<MlxBuffer>> {
    let cap = match cfg.final_logit_softcapping {
        Some(c) => c,
        None => return Ok(None),
    };
    let n = logits.element_count();
    let shape: Vec<usize> = logits.shape().to_vec();
    let mut out = device
        .alloc_buffer(n * 4, DType::F32, shape)
        .map_err(|e| anyhow!("alloc softcap output: {e}"))?;
    // params_buf: [cap, n_elements_as_f32_bits]
    let mut params = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| anyhow!("alloc softcap params: {e}"))?;
    {
        let s = params
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("softcap params slice: {e}"))?;
        s[0] = cap;
        s[1] = f32::from_bits(n as u32);
    }
    dispatch_softcap(
        encoder, registry, device.metal_device(),
        logits, &out, &params, cap,
    )
    .context("dispatch_softcap")?;
    let _ = registry; // silence
    Ok(Some({
        let _ = &mut out; // ensure out is mutable in the binding above
        out
    }))
}

/// Element-wise F32 residual add: `out = a + b`.
///
/// Mirrors the `+ attn(...)` / `+ mlp(...)` residual connections in
/// `model_mlx.py:DFlashDecoderLayer.__call__`:
/// ```text
///   h = x + self.self_attn(self.input_layernorm(x), …)
///   return h + self.mlp(self.post_attention_layernorm(h))
/// ```
/// Both buffers must have the same shape; output is freshly allocated.
pub fn dispatch_dflash_residual_add(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    a: &MlxBuffer,
    b: &MlxBuffer,
) -> Result<MlxBuffer> {
    let n = a.element_count();
    if b.element_count() != n {
        return Err(anyhow!(
            "dflash residual_add: length mismatch a={} b={}",
            n,
            b.element_count()
        ));
    }
    let shape: Vec<usize> = a.shape().to_vec();
    let out = device
        .alloc_buffer(n * 4, DType::F32, shape)
        .map_err(|e| anyhow!("alloc residual_add output: {e}"))?;
    elementwise_add(
        encoder,
        registry,
        device.metal_device(),
        a,
        b,
        &out,
        n,
        DType::F32,
    )
    .map_err(|e| anyhow!("elementwise_add: {e}"))?;
    Ok(out)
}

/// Apply SDPA to seq-major Q (post-RoPE) + K (post-RoPE) + V (no RoPE).
///
/// This is the **self-attention** form: Q seq_len == K/V seq_len. In
/// DFlash's full algorithm, K/V are the CONCAT of (ctx KV from cache +
/// prop KV from current block), so K/V seq_len > Q seq_len. That cross-
/// length form is handled by a separate dispatcher in Phase 3 (where
/// the KV cache is wired in). This wrapper exists so Phase 2 can
/// smoke-test the SDPA primitive on drafter shapes (head_dim=128).
///
/// Mirrors `model_mlx.py:DFlashAttention.__call__` line 115:
/// `output = mx.fast.scaled_dot_product_attention(queries, keys, values,
///                                                  scale=self.scale, mask=mask)`
/// — minus mask (causal-only here; sliding-window adds in Phase 3) and
/// minus the ctx/prop K/V concat.
///
/// Reuses qwen35's `apply_sdpa_causal_from_seq_major` (gpu_full_attn.rs:1270)
/// — same seq-major layout the rest of our pipeline produces.
///
/// # Arguments
///
/// - `q_roped`: `[seq_len * num_q_heads, head_dim]` F32 (post-norm + RoPE)
/// - `k_roped`: `[seq_len * num_kv_heads, head_dim]` F32 (post-norm + RoPE)
/// - `v`:       `[seq_len * num_kv_heads, head_dim]` F32 (no V norm/RoPE)
/// - Returns:   `[seq_len * num_q_heads, head_dim]` F32 (seq-major)
pub fn dispatch_dflash_sdpa_self_attn(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q_roped: &MlxBuffer,
    k_roped: &MlxBuffer,
    v: &MlxBuffer,
    cfg: &DFlashConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    let n_heads = cfg.num_attention_heads as u32;
    let n_kv_heads = cfg.num_key_value_heads as u32;
    let head_dim = cfg.head_dim as u32;
    apply_sdpa_causal_from_seq_major(
        encoder, registry, device,
        q_roped, k_roped, v,
        seq_len, n_heads, n_kv_heads, head_dim,
    )
    .context("dispatch_dflash_sdpa_self_attn")
}

/// Run the full DFlashAttention sub-block for one layer with KV cache.
///
/// Mirrors `model_mlx.py:DFlashAttention.__call__` (lines 82-116) in
/// full: input_layernorm + Q/K_ctx/V_ctx/K_prop/V_prop projections +
/// q_norm/k_norm + RoPE + cache update + cross-length SDPA + O proj.
///
/// Returns the attention output (NOT residual-summed); caller adds
/// `h + attn_out` for the residual.
///
/// # Manages encoders internally
///
/// This function commits multiple encoders during its run (Phase A
/// GPU dispatches → CPU cache writes → Phase B SDPA encoder → Phase C
/// o_proj encoder). Callers cannot pass an open encoder; they should
/// open a fresh one for any subsequent dispatches.
///
/// # Cache mutation
///
/// On return, `cache_layer.seq_len` has grown by `ctx_chunk_size`
/// (the new ctx K/V were appended). The block-size prop K/V live in
/// the slack region beyond `seq_len` but are NOT included in `seq_len`
/// — they get overwritten on the next call.
///
/// # Arguments
///
/// - `h`: `[L, hidden_size]` F32 — block input (post-residual from prior layer)
/// - `h_ctx`: `[S, hidden_size]` F32 — same across all layers (model-level fc + hidden_norm output)
/// - `layer_weights`: per-layer drafter weights
/// - `cache_layer`: per-layer KV cache state (mutated)
/// - `block_size`: L (number of block positions, typically 8)
/// - `ctx_chunk_size`: S (number of NEW ctx positions to append this call)
///
/// # Caveat
///
/// SDPA uses causal masking (mlx-native default). For DFlash full-attn
/// layer (1 of 5) Python uses mask=None; for sliding (4 of 5) Python
/// uses causal+window. Smoke tests are mask-invariant; Phase 4 parity
/// gates against Python will surface any mask-induced divergence.
pub fn dispatch_dflash_decoder_layer_attention(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    h: &MlxBuffer,
    h_ctx: &MlxBuffer,
    layer_weights: &DFlashLayerTensors,
    cache_layer: &mut super::kv_cache::DFlashLayerKvCache,
    cfg: &DFlashConfig,
    block_size: u32,
    ctx_chunk_size: u32,
) -> Result<MlxBuffer> {
    let prior_offset = cache_layer.seq_len;
    let n_q = cfg.num_attention_heads as u32;
    let n_kv = cfg.num_key_value_heads as u32;
    let head_dim = cfg.head_dim as u32;
    let l = block_size;
    let s = ctx_chunk_size;

    // -------- Phase A: GPU dispatches (single encoder) --------
    let mut enc = device
        .command_encoder()
        .context("decoder_layer_attention: open encoder A")?;

    let normed_h = dispatch_dflash_input_layernorm(
        &mut enc, registry, device, h, layer_weights, cfg, l,
    )
    .context("layer attn: input_layernorm")?;
    enc.memory_barrier();

    let q = dispatch_dflash_q_proj(&mut enc, registry, device, &normed_h, layer_weights, cfg, l)
        .context("layer attn: q_proj")?;
    let k_ctx = dispatch_dflash_k_proj(&mut enc, registry, device, h_ctx, layer_weights, cfg, s)
        .context("layer attn: k_ctx_proj")?;
    let v_ctx = dispatch_dflash_v_proj(&mut enc, registry, device, h_ctx, layer_weights, cfg, s)
        .context("layer attn: v_ctx_proj")?;
    let k_prop =
        dispatch_dflash_k_proj(&mut enc, registry, device, &normed_h, layer_weights, cfg, l)
            .context("layer attn: k_prop_proj")?;
    let v_prop =
        dispatch_dflash_v_proj(&mut enc, registry, device, &normed_h, layer_weights, cfg, l)
            .context("layer attn: v_prop_proj")?;
    enc.memory_barrier();

    let q_normed = dispatch_dflash_head_norm(
        &mut enc, registry, device, &q, &layer_weights.q_norm, cfg, l, n_q,
    )
    .context("layer attn: q_norm")?;
    let k_ctx_normed = dispatch_dflash_head_norm(
        &mut enc, registry, device, &k_ctx, &layer_weights.k_norm, cfg, s, n_kv,
    )
    .context("layer attn: k_ctx_norm")?;
    let k_prop_normed = dispatch_dflash_head_norm(
        &mut enc, registry, device, &k_prop, &layer_weights.k_norm, cfg, l, n_kv,
    )
    .context("layer attn: k_prop_norm")?;
    enc.memory_barrier();

    // RoPE offsets per Python (model_mlx.py:102-104):
    //   queries → offset = cache.offset + S        (= prior + S — the block positions sit after ctx)
    //   ctx_keys → offset = cache.offset            (= prior — appended positions)
    //   prop_keys → offset = cache.offset + S       (same as queries — the block positions)
    let q_roped = dispatch_dflash_rope(
        &mut enc, registry, device, &q_normed, cfg, l, n_q, prior_offset + s,
    )
    .context("layer attn: q rope")?;
    let k_ctx_roped = dispatch_dflash_rope(
        &mut enc, registry, device, &k_ctx_normed, cfg, s, n_kv, prior_offset,
    )
    .context("layer attn: k_ctx rope")?;
    let k_prop_roped = dispatch_dflash_rope(
        &mut enc, registry, device, &k_prop_normed, cfg, l, n_kv, prior_offset + s,
    )
    .context("layer attn: k_prop rope")?;

    // -------- Phase B: commit + CPU download + cache writes --------
    enc.commit_and_wait().context("layer attn: commit phase A")?;

    // download_f32_logical: K buffers come from apply_imrope which uses
    // `pooled_alloc_buffer` (qwen35/gpu_full_attn.rs:649) that can be
    // bucket-rounded.  We must truncate to the logical element count
    // before appending into the cursor-mode cache, otherwise stale
    // trailing pool bytes appear as extra "ctx" positions (iter-63 bug).
    let k_ctx_cpu = download_f32_logical(&k_ctx_roped).context("download k_ctx")?;
    let v_ctx_cpu = download_f32_logical(&v_ctx).context("download v_ctx")?;
    cache_layer
        .append_seq_major_kv(&k_ctx_cpu, &v_ctx_cpu, s, n_kv, head_dim)
        .context("layer attn: append ctx K/V to cache")?;
    debug_assert_eq!(cache_layer.seq_len, prior_offset + s);

    let k_prop_cpu = download_f32_logical(&k_prop_roped).context("download k_prop")?;
    let v_prop_cpu = download_f32_logical(&v_prop).context("download v_prop")?;
    cache_layer
        .write_slack_kv(&k_prop_cpu, &v_prop_cpu, l, n_kv, head_dim)
        .context("layer attn: write prop K/V to cache slack")?;
    debug_assert_eq!(cache_layer.seq_len, prior_offset + s, "slack must not advance seq_len");

    // -------- Phase C: cross-length SDPA --------
    // kv_seq_len = ctx (now in cache via append) + L (prop in slack)
    let kv_seq_len = cache_layer.seq_len + l;
    let mut sdpa_enc = device
        .command_encoder()
        .context("decoder_layer_attention: open encoder for sdpa")?;
    let attn_out = dispatch_dflash_sdpa_cross_length(
        &mut sdpa_enc, registry, device, &q_roped, cache_layer, cfg, l, kv_seq_len,
    )
    .context("layer attn: sdpa cross-length")?;
    // dispatch_dflash_sdpa_cross_length commits internally; sdpa_enc is dead.

    // -------- Phase D: O proj --------
    let mut o_enc = device
        .command_encoder()
        .context("decoder_layer_attention: open encoder for o_proj")?;
    let attn_proj = dispatch_dflash_o_proj(
        &mut o_enc, registry, device, &attn_out, layer_weights, cfg, l,
    )
    .context("layer attn: o_proj")?;
    o_enc.commit_and_wait().context("layer attn: commit o_proj")?;

    Ok(attn_proj)
}

/// Run the full DFlashDraftModel forward through all `num_hidden_layers`
/// decoder layers + model-level globals.
///
/// Mirrors `model_mlx.py:DFlashDraftModel.__call__` (lines 181-198):
/// ```text
///   h = embed_tokens(inputs) * embed_scale
///   h_ctx = hidden_norm(fc(target_hidden))
///   for layer, c in zip(layers, cache):
///       h = layer(h, h_ctx, rope, c)
///   h = norm(h)
///   logits = lm_head(h)            ← caller's job (target's lm_head)
///   if softcap: logits = tanh(logits / cap) * cap   ← caller's job
///   return logits
/// ```
///
/// Returns the post-final-norm hidden state `[L, hidden_size]`. The
/// caller orchestrator (Phase 4+) does:
/// 1. Embed tokens via target's embed_tokens (target.embed @ input_ids)
///    + embed_scale → pass as `h`
/// 2. Concat target hidden states at `cfg.target_layer_ids` (captured
///    during the target's verify forward) → pass as `target_hidden_concat`
/// 3. After this function returns h_final, run target's lm_head matmul
///    on it → logits
/// 4. Apply softcap via `dispatch_dflash_softcap` if cfg requires it
///
/// # Arguments
///
/// - `h`: `[L, hidden_size]` F32 — embedded block input (already scaled)
/// - `target_hidden_concat`: `[S, fc_input_dim]` F32 — concat of target
///   hidden states at the configured `target_layer_ids` (single tensor
///   layout, not per-layer)
/// - `model`: GPU drafter weights
/// - `cache`: full KV cache (5 per-layer states)
/// - `block_size`: L (block positions for the spec-decode round)
/// - `ctx_chunk_size`: S (new ctx positions to append this call)
///
/// Returns the final-normed `h` after all layers, ready for lm_head.
pub fn dispatch_dflash_model_forward(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    h: &MlxBuffer,
    target_hidden_concat: &MlxBuffer,
    model: &super::tensors::DFlashModelTensors,
    cache: &mut super::kv_cache::DFlashKvCache,
    cfg: &DFlashConfig,
    block_size: u32,
    ctx_chunk_size: u32,
) -> Result<MlxBuffer> {
    // -------- Prelude: fc + hidden_norm to build h_ctx --------
    let mut enc = device
        .command_encoder()
        .context("model_forward: open encoder for fc + hidden_norm")?;
    let fc_out = dispatch_dflash_fc(&mut enc, registry, device, target_hidden_concat, model, cfg, ctx_chunk_size)
        .context("model_forward: fc")?;
    enc.memory_barrier();
    let h_ctx = dispatch_dflash_hidden_norm(&mut enc, registry, device, &fc_out, model, cfg, ctx_chunk_size)
        .context("model_forward: hidden_norm")?;
    enc.commit_and_wait().context("model_forward: commit prelude")?;

    // -------- Loop over decoder layers --------
    let mut h_curr_owned: Option<MlxBuffer> = None;
    for layer_idx in 0..cfg.num_hidden_layers {
        let h_in: &MlxBuffer = match h_curr_owned.as_ref() {
            Some(b) => b,
            None => h,
        };
        let h_out = dispatch_dflash_decoder_layer(
            registry, device, h_in, &h_ctx,
            &model.layers[layer_idx], &mut cache.layers[layer_idx],
            cfg, block_size, ctx_chunk_size,
        )
        .with_context(|| format!("model_forward: layer {layer_idx}"))?;
        h_curr_owned = Some(h_out);
    }
    let h_after_layers = h_curr_owned.expect("at least 1 decoder layer in drafter");

    // -------- Epilogue: final_norm --------
    let mut enc = device
        .command_encoder()
        .context("model_forward: open encoder for final_norm")?;
    let h_final = dispatch_dflash_final_norm(&mut enc, registry, device, &h_after_layers, model, cfg, block_size)
        .context("model_forward: final_norm")?;
    enc.commit_and_wait().context("model_forward: commit epilogue")?;

    Ok(h_final)
}

/// Run the full DFlashDecoderLayer forward — attention sub-block +
/// residual + post-norm + MLP + residual.
///
/// Mirrors `model_mlx.py:DFlashDecoderLayer.__call__` (line 127):
/// ```text
///   h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
///   return h + self.mlp(self.post_attention_layernorm(h))
/// ```
///
/// Composes:
/// - `dispatch_dflash_decoder_layer_attention` (the big attention block)
/// - residual add: `h + attn_proj` → `h_after_attn`
/// - post_attention_layernorm
/// - MLP (gate + up + silu_mul + down)
/// - residual add: `h_after_attn + mlp_out` → `h_out`
///
/// Like the attention sub-block, manages encoders internally
/// (multi-commit needed for the attn function's CPU-cache writes).
pub fn dispatch_dflash_decoder_layer(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    h: &MlxBuffer,
    h_ctx: &MlxBuffer,
    layer_weights: &DFlashLayerTensors,
    cache_layer: &mut super::kv_cache::DFlashLayerKvCache,
    cfg: &DFlashConfig,
    block_size: u32,
    ctx_chunk_size: u32,
) -> Result<MlxBuffer> {
    // 1. Attention sub-block (internally commits + opens encoders).
    let attn_proj = dispatch_dflash_decoder_layer_attention(
        registry, device, h, h_ctx, layer_weights, cache_layer, cfg, block_size, ctx_chunk_size,
    )
    .context("decoder_layer: attention sub-block")?;

    // 2. Residual + post_norm + MLP + residual on a fresh encoder.
    let mut enc = device
        .command_encoder()
        .context("decoder_layer: open encoder for residual + MLP")?;

    let h_after_attn = dispatch_dflash_residual_add(&mut enc, registry, device, h, &attn_proj)
        .context("decoder_layer: residual 1")?;
    enc.memory_barrier();

    let post_normed = dispatch_dflash_post_attention_layernorm(
        &mut enc, registry, device, &h_after_attn, layer_weights, cfg, block_size,
    )
    .context("decoder_layer: post_attention_layernorm")?;
    enc.memory_barrier();

    let mlp_out = dispatch_dflash_mlp(&mut enc, registry, device, &post_normed, layer_weights, cfg, block_size)
        .context("decoder_layer: mlp")?;
    enc.memory_barrier();

    let h_out = dispatch_dflash_residual_add(&mut enc, registry, device, &h_after_attn, &mlp_out)
        .context("decoder_layer: residual 2")?;
    enc.commit_and_wait().context("decoder_layer: commit residual+MLP")?;

    Ok(h_out)
}

/// Apply SDPA where Q seq_len < K/V seq_len (DFlash's true form).
///
/// Mirrors `model_mlx.py:DFlashAttention.__call__` lines 105-115:
/// ```text
///   keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
///   ctx_len = keys.shape[2]
///   keys = mx.concatenate([keys, prop_keys], axis=2)
///   values = mx.concatenate([values, prop_values], axis=2)
///   output = scaled_dot_product_attention(queries, keys, values, scale, mask)
/// ```
///
/// In our design, `cache_layer.keys / .values` already hold the full
/// merged K/V (ctx + prop) — the caller appends both via
/// `append_seq_major_kv` before calling here. `kv_seq_len =
/// cache_layer.seq_len`, `kv_capacity = cache_layer.capacity`.
///
/// # Layout staging (Phase 3 first cut)
///
/// The SDPA kernel needs head-major Q `[1, n_heads, seq_len, head_dim]`
/// and reads K/V from cache.keys/values which are ALREADY head-major
/// `[n_kv_heads, capacity, head_dim]` (per `append_seq_major_kv`).
///
/// So only Q needs the permute (seq-major in our pipeline → head-major).
/// We download Q to CPU, permute, re-upload — mirrors qwen35's
/// `apply_sdpa_causal_from_seq_major` pattern. The drafter's per-call
/// Q size is small (block_size=8 × num_q_heads=32 × head_dim=128 = 32KB)
/// so the CPU round-trip cost is bounded.
///
/// # Arguments
///
/// - `q_seq_major`: `[q_seq_len * num_q_heads, head_dim]` F32
/// - `cache_layer`: per-layer KV cache (already populated with full
///   ctx + prop K/V for this step)
/// - Returns: `[q_seq_len * num_q_heads, head_dim]` F32 seq-major
///
/// # Caveat (Phase 3 partial)
///
/// The current mlx-native `sdpa()` kernel applies CAUSAL masking. For
/// DFlash full-attention layers (layer 4 in gemma-4-26B-A4B-it
/// drafter), Python passes `mask=None` (bidirectional). For sliding
/// layers (0-3), Python uses "causal" or windowed-causal. This first
/// cut uses causal everywhere; correctness gates in Phase 4 will
/// surface any mask-induced parity gap and we'll plumb through a mask
/// configuration then. For the smoke test we just verify shape +
/// finiteness, which is mask-invariant.
pub fn dispatch_dflash_sdpa_cross_length(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q_seq_major: &MlxBuffer,
    cache_layer: &super::kv_cache::DFlashLayerKvCache,
    cfg: &DFlashConfig,
    q_seq_len: u32,
    kv_seq_len: u32,
) -> Result<MlxBuffer> {
    let n_heads = cfg.num_attention_heads as u32;
    let n_kv_heads = cfg.num_key_value_heads as u32;
    let head_dim = cfg.head_dim as u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    if kv_seq_len > cache_layer.capacity {
        return Err(anyhow!(
            "sdpa cross-length: kv_seq_len {} > cache.capacity {}",
            kv_seq_len, cache_layer.capacity
        ));
    }

    // Finalize Q (commit pending dispatches).
    encoder.commit_and_wait().context("commit before sdpa cross-length permute")?;

    // CPU permute Q seq-major [L, H_q, D] → head-major [1, H_q, L, D].
    // q_seq_major can come from apply_imrope (pool-allocated, bucket-rounded
    // storage); use logical element count to avoid permuting trailing junk.
    let q_cpu = download_f32_logical(q_seq_major)?;
    let q_hm_cpu = permute_seq_head_dim_to_head_seq_dim_cpu(
        &q_cpu,
        q_seq_len as usize,
        n_heads as usize,
        head_dim as usize,
    );
    let q_hm = upload_f32(&q_hm_cpu, device)?;

    // Allocate output buffer head-major [1, H_q, L, D].
    let out_elem = (n_heads as usize) * (q_seq_len as usize) * (head_dim as usize);
    let out_hm = device
        .alloc_buffer(
            out_elem * 4,
            DType::F32,
            vec![1, n_heads as usize, q_seq_len as usize, head_dim as usize],
        )
        .map_err(|e| anyhow!("alloc sdpa cross-length output: {e}"))?;

    // SDPA params: separate Q seq_len + KV seq_len + capacity.
    // kv_seq_len is now an explicit parameter (was cache_layer.seq_len) so
    // callers can include slack-written prop K/V in the kv range.
    let params = SdpaParams {
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len: q_seq_len,
        kv_seq_len,
        scale,
        kv_capacity: cache_layer.capacity,
    };

    // Fresh encoder for SDPA dispatch.
    let mut enc2 = device.command_encoder().context("enc sdpa cross-length")?;
    sdpa(
        &mut enc2, registry, device,
        &q_hm, &cache_layer.keys, &cache_layer.values,
        &out_hm, &params, 1,
    )
    .map_err(|e| anyhow!("sdpa cross-length: {e}"))?;
    enc2.commit_and_wait().context("commit sdpa cross-length")?;

    // CPU permute output head-major → seq-major. `out_hm` was created by
    // `device.alloc_buffer` (not pooled), so its element_count already
    // equals storage — `download_f32_logical` is defensive parity with
    // the other dflash download sites.
    let out_hm_cpu = download_f32_logical(&out_hm)?;
    let mut out_sm_cpu = vec![0.0f32; out_elem];
    let nh = n_heads as usize;
    let l = q_seq_len as usize;
    let d = head_dim as usize;
    for h in 0..nh {
        for t in 0..l {
            let src = (h * l + t) * d;
            let dst = (t * nh + h) * d;
            out_sm_cpu[dst..dst + d].copy_from_slice(&out_hm_cpu[src..src + d]);
        }
    }
    upload_f32(&out_sm_cpu, device).map_err(|e| anyhow!("upload sdpa cross-length output: {e}"))
}

/// Dispatch the O projection: `o_proj @ attn_out`.
///
/// Mirrors `model_mlx.py:DFlashAttention.__call__` line 116:
/// `self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))`.
///
/// In our flat row-major layout, the transpose-reshape from the SDPA
/// output is metadata-only IF the SDPA output is already in
/// `[L, n_heads, head_dim]` flat order. If SDPA emits `[n_heads, L,
/// head_dim]` order, an explicit transpose is required upstream (handled
/// in the SDPA dispatcher, not here).
///
/// # Arguments
///
/// - `attn_out`: `[L, num_q_heads * head_dim]` F32 (SDPA output post-transpose)
/// - `layer`: uses `o_proj` BF16 weight `[hidden_size, num_q_heads * head_dim]`
/// - Returns: `[L, hidden_size]` F32
pub fn dispatch_dflash_o_proj(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    attn_out: &MlxBuffer,
    layer: &DFlashLayerTensors,
    cfg: &DFlashConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    let hidden = cfg.hidden_size as u32;
    let q_in_dim = (cfg.num_attention_heads * cfg.head_dim) as u32;
    apply_linear_projection_f32(
        encoder,
        registry,
        device,
        attn_out,
        &layer.o_proj,
        seq_len,
        q_in_dim,
        hidden,
    )
    .context("dispatch_dflash_o_proj")
}

/// Dispatch post_attention_layernorm (`RMSNorm(h)`) — same shape as
/// input_layernorm but applied between the attention residual and MLP.
///
/// Mirrors `model_mlx.py:DFlashDecoderLayer.__call__`:
/// `h + self.mlp(self.post_attention_layernorm(h))` (the inner RMSNorm).
pub fn dispatch_dflash_post_attention_layernorm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    h: &MlxBuffer,
    layer: &DFlashLayerTensors,
    cfg: &DFlashConfig,
    seq_len: u32,
) -> Result<MlxBuffer> {
    let hidden = cfg.hidden_size as u32;
    let element_count = (seq_len as usize) * (hidden as usize);
    if h.element_count() != element_count {
        return Err(anyhow!(
            "dflash post_attention_layernorm: h element count {} != L({}) * hidden({})",
            h.element_count(),
            seq_len,
            hidden
        ));
    }
    let normed = device
        .alloc_buffer(
            element_count * 4,
            DType::F32,
            vec![seq_len as usize, hidden as usize],
        )
        .map_err(|e| anyhow!("alloc post_attention_layernorm output: {e}"))?;
    let params = alloc_rms_norm_params(device, cfg.rms_norm_eps, hidden)?;
    dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        h,
        &layer.post_attention_layernorm,
        &normed,
        &params,
        seq_len,
        hidden,
    )
    .context("dispatch_rms_norm post_attention_layernorm")?;
    Ok(normed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::dflash::{
        config::DFlashConfig,
        tensors::DFlashModelTensors,
        weights::{DFlashWeights, DFlashWeightsFile},
    };

    fn gemma4_26b_a4b_dflash_config() -> DFlashConfig {
        DFlashConfig::from_json_str(
            crate::inference::spec_decode::dflash::config::tests::GEMMA4_26B_A4B_DFLASH_CONFIG,
        )
        .expect("test fixture must parse")
    }

    /// Smoke test: load real drafter weights to GPU, dispatch
    /// input_layernorm + Q + K + V projections on a deterministic F32
    /// input, verify the Q/K/V output shapes + that all values are
    /// finite + non-trivially-non-zero.
    ///
    /// At ones input through RMSNorm:
    ///   sum_sq = hidden * 1 = hidden
    ///   inv_rms = sqrt(hidden/hidden + eps).recip() ≈ 1
    ///   normed[i] = 1 * 1 * weight[i] = weight[i]
    /// Then each projection is a regular matmul with finite BF16
    /// weights — all outputs are finite iff dispatch plumbing works.
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_input_norm_and_q_proj() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        // Load + upload weights.
        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");

        // Allocate a [L=block_size, hidden] F32 input filled with 1.0.
        let block_size = 8u32; // K=7 per Phase 1.5 optimal
        let hidden = cfg.hidden_size as u32;
        let elem = (block_size as usize) * (hidden as usize);
        let mut h_input = device
            .alloc_buffer(elem * 4, DType::F32, vec![block_size as usize, hidden as usize])
            .expect("alloc h_input");
        {
            let slice = h_input.as_mut_slice::<f32>().expect("h_input slice");
            for v in slice.iter_mut() {
                *v = 1.0;
            }
        }

        // Dispatch into a single encoder.
        let mut encoder = device.command_encoder().expect("encoder");
        let layer = &tensors.layers[0];
        let normed = dispatch_dflash_input_layernorm(
            &mut encoder, &mut registry, &device, &h_input, layer, &cfg, block_size,
        )
        .expect("input_layernorm dispatch");
        encoder.memory_barrier(); // RAW: Q/K/V read `normed`
        let q = dispatch_dflash_q_proj(
            &mut encoder, &mut registry, &device, &normed, layer, &cfg, block_size,
        )
        .expect("q_proj dispatch");
        let k = dispatch_dflash_k_proj(
            &mut encoder, &mut registry, &device, &normed, layer, &cfg, block_size,
        )
        .expect("k_proj dispatch");
        let v = dispatch_dflash_v_proj(
            &mut encoder, &mut registry, &device, &normed, layer, &cfg, block_size,
        )
        .expect("v_proj dispatch");
        encoder.memory_barrier(); // RAW: head_norm reads Q/K just written

        // Per-head RMSNorm on Q and K (no v_norm by qwen3 convention)
        let q_normed = dispatch_dflash_head_norm(
            &mut encoder, &mut registry, &device, &q, &layer.q_norm, &cfg,
            block_size, cfg.num_attention_heads as u32,
        )
        .expect("q_norm dispatch");
        let k_normed = dispatch_dflash_head_norm(
            &mut encoder, &mut registry, &device, &k, &layer.k_norm, &cfg,
            block_size, cfg.num_key_value_heads as u32,
        )
        .expect("k_norm dispatch");
        encoder.memory_barrier(); // RAW: RoPE reads q_normed / k_normed
        // RoPE applied to Q and K (V passes through unrotated)
        let q_roped = dispatch_dflash_rope(
            &mut encoder, &mut registry, &device, &q_normed, &cfg,
            block_size, cfg.num_attention_heads as u32, 0,
        )
        .expect("q rope dispatch");
        let k_roped = dispatch_dflash_rope(
            &mut encoder, &mut registry, &device, &k_normed, &cfg,
            block_size, cfg.num_key_value_heads as u32, 0,
        )
        .expect("k rope dispatch");
        encoder.memory_barrier(); // RAW: SDPA reads Q_roped/K_roped/V
        // SDPA: self-attention form (kv_seq_len = q_seq_len = block_size).
        // The cross-length form (kv_seq_len > q_seq_len) for DFlash's
        // ctx+prop concat is Phase 3 territory.
        let attn_out = dispatch_dflash_sdpa_self_attn(
            &mut encoder, &mut registry, &device,
            &q_roped, &k_roped, &v, &cfg, block_size,
        )
        .expect("sdpa dispatch");
        // apply_sdpa_causal_from_seq_major internally commits, so the
        // next encoder is a fresh one — final O proj happens on it.
        let mut encoder = device.command_encoder().expect("encoder2");
        let h_out = dispatch_dflash_o_proj(
            &mut encoder, &mut registry, &device, &attn_out, layer, &cfg, block_size,
        )
        .expect("o_proj dispatch");
        encoder.commit_and_wait().expect("commit2");

        // Validate shapes
        let q_dim = (cfg.num_attention_heads * cfg.head_dim) as usize;
        let kv_dim = (cfg.num_key_value_heads * cfg.head_dim) as usize;
        let l = block_size as usize;
        assert_eq!(q.element_count(), l * q_dim);
        assert_eq!(k.element_count(), l * kv_dim);
        assert_eq!(v.element_count(), l * kv_dim);
        assert_eq!(q_normed.element_count(), l * q_dim);
        assert_eq!(k_normed.element_count(), l * kv_dim);

        // Q_roped / K_roped shapes match their pre-RoPE shapes
        assert_eq!(q_roped.element_count(), l * q_dim);
        assert_eq!(k_roped.element_count(), l * kv_dim);
        // SDPA output: [L, n_q_heads * head_dim]
        assert_eq!(attn_out.element_count(), l * q_dim);
        // O proj output: [L, hidden_size]
        let h_dim = cfg.hidden_size;
        assert_eq!(h_out.element_count(), l * h_dim);

        // Validate all nine outputs are finite + non-trivially non-zero.
        for (name, buf, dim) in [
            ("Q", &q, q_dim),
            ("K", &k, kv_dim),
            ("V", &v, kv_dim),
            ("Q_normed", &q_normed, q_dim),
            ("K_normed", &k_normed, kv_dim),
            ("Q_roped", &q_roped, q_dim),
            ("K_roped", &k_roped, kv_dim),
            ("Attn_out", &attn_out, q_dim),
            ("H_out", &h_out, h_dim),
        ] {
            let host: &[f32] = buf.as_slice::<f32>().expect("host slice");
            assert_eq!(host.len(), l * dim, "{name} length");
            let n_finite = host.iter().filter(|v| v.is_finite()).count();
            let n_zero = host.iter().filter(|v| **v == 0.0).count();
            assert_eq!(
                n_finite,
                host.len(),
                "{name}: all values must be finite (got {n_finite}/{}; n_zero={n_zero})",
                host.len()
            );
            assert!(
                n_zero < host.len() / 2,
                "{name} suspiciously sparse (n_zero={n_zero}/{})",
                host.len()
            );
        }
    }

    /// Smoke test for the FULL DFlashDraftModel forward — all 5
    /// decoder layers + fc + hidden_norm + final_norm.
    ///
    /// This is the Phase 3 integration milestone: the entire drafter
    /// model forward end-to-end on M5 Max with real weights + KV cache.
    /// The only missing pieces for a complete spec-decode loop are:
    /// (1) input embedding via target's embed_tokens (orchestrator job)
    /// (2) lm_head matmul via target's lm_head (orchestrator job)
    /// (3) softcap on logits (dispatch_dflash_softcap)
    /// — all of which are Phase 4+ orchestrator concerns.
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_model_forward_with_cache() {
        use super::super::kv_cache::DFlashKvCache;

        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");
        // Full-attn layer caches: need capacity >= ctx_chunk + block_size.
        // Sliding layers use sliding_window-1 = 2047 (plenty for 4+8=12).
        let mut cache = DFlashKvCache::new(&device, &cfg, 128).expect("cache");

        let block_size = 8u32;
        let ctx_chunk = 4u32;
        let hidden = cfg.hidden_size as u32;
        let fc_in = cfg.fc_input_dim() as u32;

        // Input h [L, hidden] F32-ones (stand-in for embed_tokens output)
        let h_elem = (block_size as usize) * (hidden as usize);
        let mut h = device
            .alloc_buffer(h_elem * 4, DType::F32, vec![block_size as usize, hidden as usize])
            .expect("alloc h");
        {
            let s = h.as_mut_slice::<f32>().expect("h slice");
            for v in s.iter_mut() { *v = 1.0; }
        }

        // target_hidden_concat [S, fc_input_dim] F32 (synthetic)
        let thc_elem = (ctx_chunk as usize) * (fc_in as usize);
        let mut target_hidden = device
            .alloc_buffer(thc_elem * 4, DType::F32, vec![ctx_chunk as usize, fc_in as usize])
            .expect("alloc target_hidden");
        {
            let s = target_hidden.as_mut_slice::<f32>().expect("target_hidden slice");
            for (i, v) in s.iter_mut().enumerate() {
                *v = 0.1 + ((i % 17) as f32) / 170.0;
            }
        }

        let initial_seq_lens: Vec<u32> = cache.layers.iter().map(|l| l.seq_len).collect();

        let h_final = dispatch_dflash_model_forward(
            &mut registry, &device, &h, &target_hidden,
            &tensors, &mut cache, &cfg, block_size, ctx_chunk,
        )
        .expect("model forward");

        // Shape: [L, hidden]
        assert_eq!(h_final.element_count(), (block_size as usize) * (hidden as usize));
        let host: &[f32] = h_final.as_slice::<f32>().expect("h_final slice");
        let n_finite = host.iter().filter(|v| v.is_finite()).count();
        let n_zero = host.iter().filter(|v| **v == 0.0).count();
        assert_eq!(
            n_finite, host.len(),
            "model forward output must be all finite (got {n_finite}/{}; n_zero={n_zero})",
            host.len()
        );
        assert!(
            n_zero < host.len() / 2,
            "model forward output suspiciously sparse (n_zero={n_zero}/{})",
            host.len()
        );

        // All 5 layer caches should have advanced by ctx_chunk.
        for (i, l) in cache.layers.iter().enumerate() {
            assert_eq!(
                l.seq_len,
                initial_seq_lens[i] + ctx_chunk,
                "layer {i} cache should advance by ctx_chunk_size"
            );
        }
    }

    /// Smoke test for the FULL DFlashDecoderLayer forward (attn + 2
    /// residuals + post_norm + MLP).
    ///
    /// This is the integration check that one complete drafter layer
    /// runs end-to-end on M5 Max with real weights + KV cache, mirroring
    /// what the multi-layer model forward will do per layer.
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_decoder_layer_full_with_cache() {
        use super::super::kv_cache::DFlashKvCache;

        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");
        let mut cache = DFlashKvCache::new(&device, &cfg, 64).expect("cache");

        let block_size = 8u32;
        let ctx_chunk = 4u32;
        let hidden = cfg.hidden_size as u32;

        let h_elem = (block_size as usize) * (hidden as usize);
        let mut h = device
            .alloc_buffer(h_elem * 4, DType::F32, vec![block_size as usize, hidden as usize])
            .expect("alloc h");
        {
            let s = h.as_mut_slice::<f32>().expect("h slice");
            for v in s.iter_mut() { *v = 1.0; }
        }
        let hctx_elem = (ctx_chunk as usize) * (hidden as usize);
        let mut h_ctx = device
            .alloc_buffer(hctx_elem * 4, DType::F32, vec![ctx_chunk as usize, hidden as usize])
            .expect("alloc h_ctx");
        {
            let s = h_ctx.as_mut_slice::<f32>().expect("h_ctx slice");
            for (i, v) in s.iter_mut().enumerate() {
                *v = 0.5 + ((i % 31) as f32) / 31.0;
            }
        }

        let layer_idx = 4usize; // full-attention layer
        let initial_seq_len = cache.layers[layer_idx].seq_len;

        let h_out = dispatch_dflash_decoder_layer(
            &mut registry, &device, &h, &h_ctx,
            &tensors.layers[layer_idx], &mut cache.layers[layer_idx],
            &cfg, block_size, ctx_chunk,
        )
        .expect("decoder layer forward");

        assert_eq!(h_out.element_count(), (block_size as usize) * (hidden as usize));
        let host: &[f32] = h_out.as_slice::<f32>().expect("h_out slice");
        let n_finite = host.iter().filter(|v| v.is_finite()).count();
        let n_zero = host.iter().filter(|v| **v == 0.0).count();
        assert_eq!(
            n_finite, host.len(),
            "decoder layer forward output must be all finite (got {n_finite}/{}; n_zero={n_zero})",
            host.len()
        );
        assert!(
            n_zero < host.len() / 2,
            "decoder layer forward output suspiciously sparse (n_zero={n_zero}/{})",
            host.len()
        );
        assert_eq!(
            cache.layers[layer_idx].seq_len,
            initial_seq_len + ctx_chunk,
            "cache should advance by ctx_chunk only after full layer forward"
        );
    }

    /// Smoke test for the full DFlashAttention sub-block with KV cache.
    ///
    /// Composes ALL attention dispatchers + cache append + slack write
    /// + cross-length SDPA + O proj into the function the model forward
    /// will call once per layer. Validates:
    /// - return shape [L, hidden]
    /// - all values finite + non-trivial
    /// - cache_layer.seq_len incremented by ctx_chunk_size
    /// - prop K/V wrote to slack (cache.seq_len unchanged for prop)
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_decoder_layer_attention_with_cache() {
        use super::super::kv_cache::DFlashKvCache;

        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");
        let mut cache = DFlashKvCache::new(&device, &cfg, 64).expect("cache");

        let block_size = 8u32;
        let ctx_chunk = 4u32;
        let hidden = cfg.hidden_size as u32;

        // Build synthetic h [L, hidden] F32-ones
        let h_elem = (block_size as usize) * (hidden as usize);
        let mut h = device
            .alloc_buffer(h_elem * 4, DType::F32, vec![block_size as usize, hidden as usize])
            .expect("alloc h");
        {
            let s = h.as_mut_slice::<f32>().expect("h slice");
            for v in s.iter_mut() { *v = 1.0; }
        }
        // Build synthetic h_ctx [S, hidden] F32 with varied values
        let hctx_elem = (ctx_chunk as usize) * (hidden as usize);
        let mut h_ctx = device
            .alloc_buffer(hctx_elem * 4, DType::F32, vec![ctx_chunk as usize, hidden as usize])
            .expect("alloc h_ctx");
        {
            let s = h_ctx.as_mut_slice::<f32>().expect("h_ctx slice");
            for (i, v) in s.iter_mut().enumerate() {
                *v = 0.5 + ((i % 31) as f32) / 31.0;
            }
        }

        // Use the FULL-attention layer (index 4) for first test — clean kv_seq_len.
        let layer_idx = 4usize;
        let initial_seq_len = cache.layers[layer_idx].seq_len;

        let attn_out = dispatch_dflash_decoder_layer_attention(
            &mut registry, &device, &h, &h_ctx,
            &tensors.layers[layer_idx], &mut cache.layers[layer_idx],
            &cfg, block_size, ctx_chunk,
        )
        .expect("decoder layer attention");

        // Validate output [L, hidden] F32
        assert_eq!(attn_out.element_count(), (block_size as usize) * (hidden as usize));

        let host: &[f32] = attn_out.as_slice::<f32>().expect("attn_out slice");
        let n_finite = host.iter().filter(|v| v.is_finite()).count();
        let n_zero = host.iter().filter(|v| **v == 0.0).count();
        assert_eq!(
            n_finite, host.len(),
            "decoder layer attention output must be finite (got {n_finite}/{}; n_zero={n_zero})",
            host.len()
        );
        assert!(
            n_zero < host.len() / 2,
            "decoder layer attention output suspiciously sparse (n_zero={n_zero}/{})",
            host.len()
        );

        // Cache state: ctx appended (seq_len += ctx_chunk), prop is in slack (NOT in seq_len).
        assert_eq!(
            cache.layers[layer_idx].seq_len,
            initial_seq_len + ctx_chunk,
            "cache should advance by ctx_chunk_size only; prop lives in slack"
        );
    }

    /// Cross-length SDPA smoke test (Phase 3 form).
    ///
    /// Builds a populated cache layer (synthetic head-major K/V via
    /// append_seq_major_kv with random-ish ones inputs), runs Q of
    /// seq_len=8 against cache with seq_len=12 (kv_seq_len > q_seq_len),
    /// validates output shape + finiteness. This validates the layout
    /// permute + cross-length params end-to-end.
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_sdpa_cross_length() {
        use super::super::kv_cache::DFlashKvCache;

        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        // Allocate cache + populate with 12 positions of synthetic K/V.
        let mut cache = DFlashKvCache::new(&device, &cfg, 64).expect("cache");
        let layer_idx = 4usize; // full-attention layer
        let n_kv = cfg.num_key_value_heads as u32;
        let d = cfg.head_dim as u32;
        let ctx_len = 12u32;
        let total = (ctx_len as usize) * (n_kv as usize) * (d as usize);
        let mut k_input = vec![0.0f32; total];
        let mut v_input = vec![0.0f32; total];
        for (i, v) in k_input.iter_mut().enumerate() {
            *v = (i % 17) as f32 / 17.0; // varied finite values
        }
        for (i, v) in v_input.iter_mut().enumerate() {
            *v = (i % 23) as f32 / 23.0;
        }
        cache.layers[layer_idx]
            .append_seq_major_kv(&k_input, &v_input, ctx_len, n_kv, d)
            .expect("populate cache");
        assert_eq!(cache.layers[layer_idx].seq_len, ctx_len);

        // Build Q seq-major [L=8, n_q_heads, head_dim] F32 ones-ish.
        let q_seq_len = 8u32;
        let n_q = cfg.num_attention_heads as u32;
        let q_elem = (q_seq_len as usize) * (n_q as usize) * (d as usize);
        let mut q = device
            .alloc_buffer(
                q_elem * 4,
                DType::F32,
                vec![q_seq_len as usize, n_q as usize, d as usize],
            )
            .expect("alloc q");
        {
            let s = q.as_mut_slice::<f32>().expect("q slice");
            for (i, v) in s.iter_mut().enumerate() {
                *v = ((i % 11) as f32) * 0.01;
            }
        }

        let mut encoder = device.command_encoder().expect("encoder");
        let out = dispatch_dflash_sdpa_cross_length(
            &mut encoder, &mut registry, &device,
            &q, &cache.layers[layer_idx], &cfg, q_seq_len, ctx_len,
        )
        .expect("sdpa cross-length");

        // Shape: [L * n_q_heads * head_dim] F32 seq-major.
        let expected_elem = (q_seq_len as usize) * (n_q as usize) * (d as usize);
        assert_eq!(out.element_count(), expected_elem);
        let host: &[f32] = out.as_slice::<f32>().expect("out slice");
        let n_finite = host.iter().filter(|v| v.is_finite()).count();
        assert_eq!(
            n_finite, host.len(),
            "cross-length SDPA output must be all finite (got {n_finite}/{})",
            host.len()
        );
    }

    /// Smoke test for the model-level globals: fc + hidden_norm +
    /// final_norm + softcap.
    ///
    /// Validates that the four "outside-the-layer" dispatchers work
    /// with the real drafter weights. Inputs are synthetic F32-ones:
    /// - target_hidden_concat [S=4, fc_input_dim=16896] for fc/hidden_norm
    /// - h [L=8, hidden=2816] for final_norm
    /// - logits [L=8, vocab=262144] for softcap
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_model_level_globals() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");

        // ===== fc + hidden_norm path =====
        let ctx_seq_len = 4u32;
        let fc_in = cfg.fc_input_dim() as u32;
        let target_hidden_elem = (ctx_seq_len as usize) * (fc_in as usize);
        let mut target_hidden = device
            .alloc_buffer(
                target_hidden_elem * 4,
                DType::F32,
                vec![ctx_seq_len as usize, fc_in as usize],
            )
            .expect("alloc target_hidden");
        {
            let s = target_hidden
                .as_mut_slice::<f32>()
                .expect("target_hidden slice");
            for v in s.iter_mut() { *v = 1.0; }
        }

        let mut encoder = device.command_encoder().expect("encoder1");
        let fc_out = dispatch_dflash_fc(
            &mut encoder, &mut registry, &device, &target_hidden, &tensors, &cfg, ctx_seq_len,
        )
        .expect("fc dispatch");
        encoder.memory_barrier();
        let h_ctx = dispatch_dflash_hidden_norm(
            &mut encoder, &mut registry, &device, &fc_out, &tensors, &cfg, ctx_seq_len,
        )
        .expect("hidden_norm dispatch");

        // ===== final_norm path =====
        let block_size = 8u32;
        let hidden = cfg.hidden_size as u32;
        let h_elem = (block_size as usize) * (hidden as usize);
        let mut h = device
            .alloc_buffer(h_elem * 4, DType::F32, vec![block_size as usize, hidden as usize])
            .expect("alloc h");
        {
            let s = h.as_mut_slice::<f32>().expect("h slice");
            for v in s.iter_mut() { *v = 1.0; }
        }
        let h_final = dispatch_dflash_final_norm(
            &mut encoder, &mut registry, &device, &h, &tensors, &cfg, block_size,
        )
        .expect("final_norm dispatch");

        // ===== softcap path (synthetic logits) =====
        // Logits would normally come from lm_head; synthesize a small
        // logit buffer to exercise softcap kernel.
        let logits_n = (block_size as usize) * 256; // tiny vocab proxy
        let mut logits = device
            .alloc_buffer(logits_n * 4, DType::F32, vec![block_size as usize, 256])
            .expect("alloc logits");
        {
            let s = logits.as_mut_slice::<f32>().expect("logits slice");
            // Mix of values: most around 5.0, some over the cap (30.0)
            for (i, v) in s.iter_mut().enumerate() {
                *v = if i % 17 == 0 { 100.0 } else { 5.0 };
            }
        }
        let capped = dispatch_dflash_softcap(&mut encoder, &mut registry, &device, &logits, &cfg)
            .expect("softcap dispatch")
            .expect("softcap should be Some (cfg has final_logit_softcapping=30.0)");

        encoder.commit_and_wait().expect("commit");

        // ===== validations =====
        let h_dim = cfg.hidden_size;
        assert_eq!(fc_out.element_count(), (ctx_seq_len as usize) * h_dim);
        assert_eq!(h_ctx.element_count(), (ctx_seq_len as usize) * h_dim);
        assert_eq!(h_final.element_count(), (block_size as usize) * h_dim);
        assert_eq!(capped.element_count(), logits_n);

        for (name, buf) in [
            ("fc_out", &fc_out),
            ("h_ctx", &h_ctx),
            ("h_final", &h_final),
            ("softcap_out", &capped),
        ] {
            let host: &[f32] = buf.as_slice::<f32>().expect("host slice");
            let n_finite = host.iter().filter(|v| v.is_finite()).count();
            assert_eq!(
                n_finite, host.len(),
                "{name}: all values must be finite (got {n_finite}/{})",
                host.len()
            );
        }

        // Softcap-specific check: all values must satisfy |v| < cap.
        let cap = cfg.final_logit_softcapping.unwrap();
        let s_host: &[f32] = capped.as_slice::<f32>().expect("capped slice");
        let max_abs = s_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < cap,
            "softcap output |v|.max = {max_abs} >= cap {cap}; softcap kernel broken"
        );
        // And: the 100.0 inputs must be capped significantly down.
        let near_cap = s_host.iter().filter(|v| **v > cap - 0.5).count();
        assert!(near_cap > 0, "expected some outputs near cap; got max_abs={max_abs}");
    }

    /// Independent layer-forward smoke test (self-attn form).
    ///
    /// Composes ALL Phase 2 dispatchers into a single decoder layer
    /// forward, mirroring `model_mlx.py:DFlashDecoderLayer.__call__`
    /// (line 127):
    /// ```text
    ///   h_attn = h + self.self_attn(self.input_layernorm(h), …)
    ///   h_out  = h_attn + self.mlp(self.post_attention_layernorm(h_attn))
    /// ```
    /// The SDPA call uses self-attention semantics (Q seq_len == K/V
    /// seq_len) — Phase 3 will replace this with the cross-length form
    /// once KV cache state + ctx/prop concat are wired.
    ///
    /// This test is the integration milestone for Phase 2: every
    /// dispatcher composes correctly, output is finite + non-trivial,
    /// and the layer pipeline runs end-to-end on M5 Max in one test.
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_decoder_layer_self_attn() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");

        // F32-ones input
        let block_size = 8u32;
        let hidden = cfg.hidden_size as u32;
        let elem = (block_size as usize) * (hidden as usize);
        let mut h0 = device
            .alloc_buffer(elem * 4, DType::F32, vec![block_size as usize, hidden as usize])
            .expect("alloc h0");
        {
            let s = h0.as_mut_slice::<f32>().expect("h0 slice");
            for v in s.iter_mut() { *v = 1.0; }
        }

        let layer = &tensors.layers[0];
        let l = block_size as usize;
        let q_dim = (cfg.num_attention_heads * cfg.head_dim) as usize;

        // ====== Attention sub-block ======
        let mut encoder = device.command_encoder().expect("encoder1");
        let h0_normed = dispatch_dflash_input_layernorm(
            &mut encoder, &mut registry, &device, &h0, layer, &cfg, block_size,
        ).expect("input_norm");
        encoder.memory_barrier();
        let q = dispatch_dflash_q_proj(
            &mut encoder, &mut registry, &device, &h0_normed, layer, &cfg, block_size,
        ).expect("q_proj");
        let k = dispatch_dflash_k_proj(
            &mut encoder, &mut registry, &device, &h0_normed, layer, &cfg, block_size,
        ).expect("k_proj");
        let v = dispatch_dflash_v_proj(
            &mut encoder, &mut registry, &device, &h0_normed, layer, &cfg, block_size,
        ).expect("v_proj");
        encoder.memory_barrier();
        let q_normed = dispatch_dflash_head_norm(
            &mut encoder, &mut registry, &device, &q, &layer.q_norm, &cfg,
            block_size, cfg.num_attention_heads as u32,
        ).expect("q_norm");
        let k_normed = dispatch_dflash_head_norm(
            &mut encoder, &mut registry, &device, &k, &layer.k_norm, &cfg,
            block_size, cfg.num_key_value_heads as u32,
        ).expect("k_norm");
        encoder.memory_barrier();
        let q_roped = dispatch_dflash_rope(
            &mut encoder, &mut registry, &device, &q_normed, &cfg,
            block_size, cfg.num_attention_heads as u32, 0,
        ).expect("q rope");
        let k_roped = dispatch_dflash_rope(
            &mut encoder, &mut registry, &device, &k_normed, &cfg,
            block_size, cfg.num_key_value_heads as u32, 0,
        ).expect("k rope");
        encoder.memory_barrier();
        let attn_out = dispatch_dflash_sdpa_self_attn(
            &mut encoder, &mut registry, &device, &q_roped, &k_roped, &v, &cfg, block_size,
        ).expect("sdpa");
        // sdpa internally commits + returns; fresh encoder for o_proj + residual
        let mut encoder = device.command_encoder().expect("encoder2");
        let h_after_attn_proj = dispatch_dflash_o_proj(
            &mut encoder, &mut registry, &device, &attn_out, layer, &cfg, block_size,
        ).expect("o_proj");
        encoder.memory_barrier();
        let h_after_attn = dispatch_dflash_residual_add(
            &mut encoder, &mut registry, &device, &h0, &h_after_attn_proj,
        ).expect("residual1");

        // ====== MLP sub-block ======
        encoder.memory_barrier();
        let post_normed = dispatch_dflash_post_attention_layernorm(
            &mut encoder, &mut registry, &device, &h_after_attn, layer, &cfg, block_size,
        ).expect("post_norm");
        encoder.memory_barrier();
        let mlp_out = dispatch_dflash_mlp(
            &mut encoder, &mut registry, &device, &post_normed, layer, &cfg, block_size,
        ).expect("mlp");
        encoder.memory_barrier();
        let h_out = dispatch_dflash_residual_add(
            &mut encoder, &mut registry, &device, &h_after_attn, &mlp_out,
        ).expect("residual2");
        encoder.commit_and_wait().expect("final commit");

        // Validate final layer output shape + finite/non-trivial.
        let h_dim = cfg.hidden_size;
        assert_eq!(h_out.element_count(), l * h_dim);
        assert_eq!(h_after_attn.element_count(), l * h_dim);

        let host_final: &[f32] = h_out.as_slice::<f32>().expect("h_out slice");
        let n_finite = host_final.iter().filter(|v| v.is_finite()).count();
        let n_zero = host_final.iter().filter(|v| **v == 0.0).count();
        assert_eq!(
            n_finite, host_final.len(),
            "decoder layer output must be all finite (got {n_finite}/{}; n_zero={n_zero})",
            host_final.len()
        );
        assert!(
            n_zero < host_final.len() / 2,
            "decoder layer output suspiciously sparse (n_zero={n_zero}/{})",
            host_final.len()
        );
        // Sanity: h_out != h0 (the layer did SOMETHING non-trivial).
        let _ = q_dim; // silence unused
    }

    /// Independent O projection smoke test: validates that o_proj
    /// dispatcher takes a `[L, num_q_heads * head_dim]` F32 input and
    /// produces a `[L, hidden_size]` F32 output. Input is F32-ones (a
    /// stand-in for SDPA output — the dispatcher cares about shape +
    /// arithmetic, not the semantic meaning of the input).
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_o_proj() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");

        let block_size = 8u32;
        let q_dim = (cfg.num_attention_heads * cfg.head_dim) as u32; // 32 * 128 = 4096
        let elem = (block_size as usize) * (q_dim as usize);
        let mut attn_out = device
            .alloc_buffer(elem * 4, DType::F32, vec![block_size as usize, q_dim as usize])
            .expect("alloc attn_out");
        {
            let s = attn_out.as_mut_slice::<f32>().expect("attn_out slice");
            for v in s.iter_mut() { *v = 1.0; }
        }

        let mut encoder = device.command_encoder().expect("encoder");
        let layer = &tensors.layers[0];
        let h_out = dispatch_dflash_o_proj(
            &mut encoder, &mut registry, &device, &attn_out, layer, &cfg, block_size,
        )
        .expect("o_proj dispatch");
        encoder.commit_and_wait().expect("commit");

        // [L, hidden_size]
        let h_dim = cfg.hidden_size;
        assert_eq!(h_out.element_count(), (block_size as usize) * h_dim);
        let host: &[f32] = h_out.as_slice::<f32>().expect("h_out slice");
        let n_finite = host.iter().filter(|v| v.is_finite()).count();
        let n_zero = host.iter().filter(|v| **v == 0.0).count();
        assert_eq!(n_finite, host.len(),
            "O projection output must be all finite (got {n_finite}/{}; n_zero={n_zero})", host.len());
        assert!(n_zero < host.len() / 2,
            "O projection output suspiciously sparse (n_zero={n_zero}/{})", host.len());
    }

    /// Independent MLP smoke test: load drafter, apply
    /// post_attention_layernorm + SwiGLU MLP to F32-ones input,
    /// verify output shape and finite values.
    ///
    /// At ones input → post_norm → ~weight (per-row), → 3 matmuls + silu_mul,
    /// output is [L, hidden_size] F32 with non-trivial values.
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn smoke_post_norm_and_mlp() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let mut registry = KernelRegistry::new();

        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");

        // F32-ones input [L=8, hidden=2816]
        let block_size = 8u32;
        let hidden = cfg.hidden_size as u32;
        let elem = (block_size as usize) * (hidden as usize);
        let mut h = device
            .alloc_buffer(elem * 4, DType::F32, vec![block_size as usize, hidden as usize])
            .expect("alloc h");
        {
            let slice = h.as_mut_slice::<f32>().expect("h slice");
            for v in slice.iter_mut() { *v = 1.0; }
        }

        // Encoder: post_norm → MLP
        let mut encoder = device.command_encoder().expect("encoder");
        let layer = &tensors.layers[0];
        let post_normed = dispatch_dflash_post_attention_layernorm(
            &mut encoder, &mut registry, &device, &h, layer, &cfg, block_size,
        ).expect("post_norm");
        encoder.memory_barrier();
        let mlp_out = dispatch_dflash_mlp(
            &mut encoder, &mut registry, &device, &post_normed, layer, &cfg, block_size,
        ).expect("mlp");
        encoder.commit_and_wait().expect("commit");

        // Validate shape: [block_size, hidden_size]
        let h_dim = cfg.hidden_size;
        assert_eq!(mlp_out.element_count(), (block_size as usize) * h_dim);

        let host: &[f32] = mlp_out.as_slice::<f32>().expect("mlp_out host slice");
        let n_finite = host.iter().filter(|v| v.is_finite()).count();
        let n_zero = host.iter().filter(|v| **v == 0.0).count();
        assert_eq!(
            n_finite, host.len(),
            "MLP output must be all finite (got {n_finite}/{}; n_zero={n_zero})",
            host.len()
        );
        assert!(
            n_zero < host.len() / 2,
            "MLP output suspiciously sparse (n_zero={n_zero}/{})",
            host.len()
        );
    }
}
