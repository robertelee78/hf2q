//! nomic-bert encoder forward pass on Metal GPU (ADR-005 Phase 2b, Task #16).
//!
//! Composes the per-block topology that matches llama.cpp's `llm_build_bert`
//! with `arch == LLM_ARCH_NOMIC_BERT` (per `/opt/llama.cpp/src/models/bert.cpp`):
//!
//! ```text
//!   inpL  = LayerNorm(token_embd[ids] + maybe token_types[type_ids])
//!   for il in 0..N:
//!       cur = inpL
//!       qkv  = ffn_up_style_linear(cur, attn_qkv.weight)   // [seq, 3*hidden]
//!       q, k, v = split_qkv(qkv)                             // each [seq, hidden]
//!       q = rope_neox(q, freq_base, n_rot=head_dim)
//!       k = rope_neox(k, freq_base, n_rot=head_dim)
//!       y = bidirectional_attn_with_mask(q, k, v)            // [seq, hidden]
//!       y = linear(y, attn_output.weight, attn_output.bias?)  // [seq, hidden]
//!       cur = LayerNorm(y + cur, attn_output_norm)            // post-attn LN
//!       up    = linear(cur, ffn_up.weight)                    // [seq, n_ff]
//!       gate  = linear(cur, ffn_gate.weight)                  // [seq, n_ff]
//!       gated = silu(gate) * up                               // SwiGLU
//!       down  = linear(gated, ffn_down.weight)                // [seq, hidden]
//!       inpL  = LayerNorm(down + cur, layer_output_norm)      // post-FFN LN
//!   pooled = pool(inpL, pooling_type)                          // [hidden]
//!   return l2_normalize(pooled)                                // unit-norm
//! ```
//!
//! Three structural deltas vs. the BERT lane (per `bert.cpp:60-68, 131-138`):
//!
//! 1. **Position encoding.** RoPE-NeoX on Q and K (V unrotated), no
//!    `position_embd` lookup at the embed stage.
//! 2. **MLP.** `silu(gate(x)) * up(x) → down`. Three linears per block
//!    (`ffn_up`, `ffn_gate`, `ffn_down`). BERT uses two with GeLU.
//! 3. **Tensor manifest.** Fused `attn_qkv.weight [hidden, 3*hidden]`
//!    instead of separate `attn_q/k/v.weight`. Split via `MlxBuffer::slice_view`
//!    + three independent `bert_linear_gpu` calls — each cast-to-bf16 +
//!    matmul produces its own logical Q / K / V projection.
//!
//! GPU primitives are reused from `bert::bert_gpu` (linear, layer_norm,
//! attention with mask, residual_add, embed_gather, pool, l2_normalize).
//! mlx-native ops are imported directly: `dispatch_rope_neox_f32` for the
//! RoPE step, `dispatch_silu_mul` for the SwiGLU step, `cast` is internal
//! to `bert_linear_gpu`.
//!
//! # Concurrent-dispatch barriers
//!
//! mlx-native's command encoder uses `MTLDispatchType::Concurrent`, so
//! every read-after-write between dispatches needs an explicit
//! `encoder.memory_barrier()`. Without it the next dispatch can read
//! pre-write garbage. This composer inserts barriers at every RAW point;
//! callers MUST preserve them when refactoring.

#![allow(dead_code)] // handler wiring lands in iter 77+

use anyhow::{anyhow, Context, Result};

use mlx_native::ops::rope::dispatch_rope_neox_f32;
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::super::bert::bert_gpu::{
    alloc_bert_attention_mask, bert_attention_with_mask_gpu, bert_embed_gather_gpu,
    bert_l2_normalize_gpu, bert_layer_norm_gpu, bert_linear_gpu, bert_pool_gpu, bert_residual_add_gpu,
    register_bert_custom_shaders, BertPoolKind,
};
use super::super::bert::config::PoolingType;
use super::config::NomicBertConfig;
use super::weights::LoadedNomicBertWeights;

// ---------------------------------------------------------------------------
// Kernel registration
// ---------------------------------------------------------------------------

/// Register every kernel `apply_nomic_bert_*` dispatches. Idempotent —
/// safe to call multiple times on the same registry. Callers MUST run
/// this once before the first encode in any session.
///
/// Includes the BERT custom shaders (LayerNorm, mask-add, mean-pool,
/// bias-add) plus the mlx-native RoPE-NeoX-F32 and SiLU-mul-F32
/// shaders. Shaders that the BERT lane registers under different names
/// (`bert_*_f32`) are NOT duplicated here — `register_bert_custom_shaders`
/// is the single source.
pub fn register_nomic_bert_kernels(registry: &mut KernelRegistry) {
    register_bert_custom_shaders(registry);
    mlx_native::ops::rope::register(registry);
    mlx_native::ops::silu_mul::register(registry);
}

// ---------------------------------------------------------------------------
// Embedding stage (no position embedding)
// ---------------------------------------------------------------------------

/// nomic-bert embedding stage:
/// `out = LayerNorm(token_embd[input_ids] + maybe token_types[type_ids])`.
///
/// Differs from `bert_embeddings_gpu` in that there is NO `position_embd`
/// gather — RoPE inside each attention block carries the position
/// information instead.
///
/// Inputs (F32 on device unless noted):
/// - `input_ids`           U32 `[seq_len]`
/// - `type_ids_opt`        U32 `[seq_len]` — optional
/// - `token_embd`          F32 `[vocab, hidden]`
/// - `token_types_opt`     F32 `[type_vocab, hidden]` — must accompany
///                          `type_ids_opt` (one None implies the other None)
/// - `embed_norm_gamma`,
///   `embed_norm_beta`     F32 `[hidden]`
///
/// Returns F32 `[seq_len, hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn nomic_bert_embeddings_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input_ids: &MlxBuffer,
    type_ids_opt: Option<&MlxBuffer>,
    token_embd: &MlxBuffer,
    token_types_opt: Option<&MlxBuffer>,
    embed_norm_gamma: &MlxBuffer,
    embed_norm_beta: &MlxBuffer,
    eps: f32,
    seq_len: u32,
    hidden: u32,
    vocab: u32,
    type_vocab: u32,
) -> Result<MlxBuffer> {
    if seq_len == 0 || hidden == 0 {
        return Err(anyhow!(
            "nomic_bert_embeddings_gpu: seq_len ({}) and hidden ({}) must be > 0",
            seq_len,
            hidden
        ));
    }
    match (type_ids_opt.is_some(), token_types_opt.is_some()) {
        (true, true) | (false, false) => {}
        (a, b) => {
            return Err(anyhow!(
                "nomic_bert_embeddings_gpu: type_ids and token_types must both be Some or both None (got {} / {})",
                a, b
            ));
        }
    }

    let n_hidden = (seq_len as usize) * (hidden as usize);

    // 1. Token embedding gather.
    let tok =
        bert_embed_gather_gpu(encoder, registry, device, token_embd, input_ids, vocab, hidden, seq_len)
            .context("nomic embeddings: token gather")?;
    encoder.memory_barrier();

    // 2. Optional segment-type embedding gather + add. nomic-embed-text-v1.5
    //    has type_vocab_size=2 and the loader synthesizes an all-zero
    //    type_ids buffer when caller passes None (mirrors BERT's behavior
    //    via apply_nomic_bert_full_forward_gpu).
    let summed = if let (Some(type_ids), Some(token_types)) = (type_ids_opt, token_types_opt) {
        let typ = bert_embed_gather_gpu(
            encoder, registry, device, token_types, type_ids, type_vocab, hidden, seq_len,
        )
        .context("nomic embeddings: type gather")?;
        encoder.memory_barrier();
        let s =
            bert_residual_add_gpu(encoder, registry, device, &tok, &typ, n_hidden as u32)
                .context("nomic embeddings: token + type add")?;
        encoder.memory_barrier();
        s
    } else {
        tok
    };

    // 3. LayerNorm finalize.
    bert_layer_norm_gpu(
        encoder,
        registry,
        device,
        &summed,
        embed_norm_gamma,
        embed_norm_beta,
        eps,
        seq_len,
        hidden,
    )
    .context("nomic embeddings: post-sum LayerNorm")
}

// ---------------------------------------------------------------------------
// Position-id + RoPE-params helpers
// ---------------------------------------------------------------------------

/// Build a U32 buffer holding `[0, 1, 2, ..., seq_len - 1]`. Used as the
/// RoPE positions buffer inside each attention block. Allocated on the
/// device's unified memory; populated via the CPU pointer view (Apple
/// Silicon: same bytes the GPU sees).
fn alloc_rope_positions(device: &MlxDevice, seq_len: u32) -> Result<MlxBuffer> {
    let n = seq_len as usize;
    let buf = device
        .alloc_buffer(n * 4, DType::U32, vec![n])
        .map_err(|e| anyhow!("alloc rope positions: {e}"))?;
    // SAFETY: just-allocated u32 buffer; exclusive access.
    let slice: &mut [u32] =
        unsafe { std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut u32, n) };
    for (i, slot) in slice.iter_mut().enumerate() {
        *slot = i as u32;
    }
    Ok(buf)
}

/// Build a U32 buffer holding `[0, 0, ..., 0]` of length `seq_len` —
/// used when the model has a `token_types.weight` table but the caller
/// didn't supply explicit type_ids (the BERT-family default for
/// single-segment input is segment-id 0 for every token).
fn alloc_zero_type_ids(device: &MlxDevice, seq_len: u32) -> Result<MlxBuffer> {
    let n = seq_len as usize;
    let buf = device
        .alloc_buffer(n * 4, DType::U32, vec![n])
        .map_err(|e| anyhow!("alloc zero type_ids: {e}"))?;
    // SAFETY: just-allocated u32 buffer; exclusive access. Zero-init
    // explicit (alloc_buffer doesn't guarantee zeroed contents).
    let slice: &mut [u32] =
        unsafe { std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut u32, n) };
    for slot in slice.iter_mut() {
        *slot = 0;
    }
    Ok(buf)
}

/// Build the F32 RoPE params buffer expected by `dispatch_rope_neox_f32`
/// at buffer-binding(2): `[theta, head_dim, rope_dim, 0]` (16 bytes).
fn alloc_rope_params(
    device: &MlxDevice,
    theta: f32,
    head_dim: u32,
    rope_dim: u32,
) -> Result<MlxBuffer> {
    let buf = device
        .alloc_buffer(16, DType::F32, vec![4])
        .map_err(|e| anyhow!("alloc rope params: {e}"))?;
    // SAFETY: just-allocated 4-element f32 buffer; exclusive access.
    let slice: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut f32, 4) };
    slice[0] = theta;
    slice[1] = head_dim as f32;
    slice[2] = rope_dim as f32;
    slice[3] = 0.0;
    Ok(buf)
}

/// Build the U32 params buffer expected by `dispatch_silu_mul`: a single
/// `n` (4 bytes). Caller MUST keep this alive until the encoder commits.
fn alloc_silu_mul_params(device: &MlxDevice, n: u32) -> Result<MlxBuffer> {
    let buf = device
        .alloc_buffer(4, DType::U32, vec![1])
        .map_err(|e| anyhow!("alloc silu_mul params: {e}"))?;
    // SAFETY: just-allocated 1-element u32 buffer; exclusive access.
    let slice: &mut [u32] =
        unsafe { std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut u32, 1) };
    slice[0] = n;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Per-block tensor bundle
// ---------------------------------------------------------------------------

/// Per-layer tensor refs that drive `apply_nomic_bert_encoder_block_gpu`.
/// Caller pulls each tensor from `LoadedNomicBertWeights`. Biases are
/// optional (nomic-embed-text-v1.5 ships zero biases on its linears).
pub struct NomicBertEncoderBlockTensors<'a> {
    /// Fused QKV projection: `[hidden, 3*hidden]` weight + optional
    /// `[3*hidden]` bias. Splitting happens at use site via `slice_view`.
    pub qkv_w: &'a MlxBuffer,
    pub qkv_b: Option<&'a MlxBuffer>,
    /// Attention output projection: `[hidden, hidden]` + optional `[hidden]` bias.
    pub o_w: &'a MlxBuffer,
    pub o_b: Option<&'a MlxBuffer>,
    /// Post-attention LayerNorm γ, β. Both `[hidden]`.
    pub attn_norm_gamma: &'a MlxBuffer,
    pub attn_norm_beta: &'a MlxBuffer,
    /// FFN up projection: `[intermediate, hidden]` + optional `[intermediate]` bias.
    pub up_w: &'a MlxBuffer,
    pub up_b: Option<&'a MlxBuffer>,
    /// FFN gate projection: `[intermediate, hidden]` + optional `[intermediate]` bias.
    pub gate_w: &'a MlxBuffer,
    pub gate_b: Option<&'a MlxBuffer>,
    /// FFN down projection: `[hidden, intermediate]` + optional `[hidden]` bias.
    pub down_w: &'a MlxBuffer,
    pub down_b: Option<&'a MlxBuffer>,
    /// Post-FFN LayerNorm γ, β. Both `[hidden]`.
    pub ffn_norm_gamma: &'a MlxBuffer,
    pub ffn_norm_beta: &'a MlxBuffer,
}

// ---------------------------------------------------------------------------
// One encoder block forward pass
// ---------------------------------------------------------------------------

/// One nomic-bert encoder block forward pass on GPU.
///
/// Topology (post-norm, RoPE-on-Q/K, SwiGLU MLP):
///
/// ```text
///   q, k, v = split_qkv(linear(x, qkv_w))      // 3 contiguous weight slices
///   q       = rope_neox(q)                      // [seq, n_heads, head_dim]
///   k       = rope_neox(k)
///   y       = bidirectional_attn_with_mask(q, k, v)
///   y       = linear(y, o_w, o_b?)
///   x'      = LayerNorm(x + y, attn_out_norm)
///   up      = linear(x', up_w, up_b?)
///   gate    = linear(x', gate_w, gate_b?)
///   gated   = silu(gate) * up                  // mlx-native silu_mul
///   down    = linear(gated, down_w, down_b?)
///   x''     = LayerNorm(x' + down, layer_out_norm)
///   return x''
/// ```
///
/// Inputs:
/// - `input`   F32 `[seq_len, hidden]`
/// - `tensors` per-layer weight bundle
/// - `mask`    F32 `[seq_len, seq_len]` attention mask (always-on per
///             iter-67 BERT empirical finding — bert_attention_with_mask_gpu
///             requires it)
/// - `rope_positions` U32 `[seq_len]` — `[0, 1, ..., seq_len-1]`
/// - `rope_params`    F32 `[4]` — `[theta, head_dim, rope_dim, 0]`
///
/// Returns F32 `[seq_len, hidden]`.
///
/// `seq_len ≥ 32` floor inherited from `bert_attention_with_mask_gpu`
/// (post-softmax matmul has K = seq_len).
/// `head_dim ≥ 32` floor inherited from the same.
#[allow(clippy::too_many_arguments)]
pub fn apply_nomic_bert_encoder_block_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    tensors: &NomicBertEncoderBlockTensors<'_>,
    mask: &MlxBuffer,
    rope_positions: &MlxBuffer,
    rope_params: &MlxBuffer,
    seq_len: u32,
    hidden: u32,
    num_heads: u32,
    intermediate: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    if hidden % num_heads != 0 {
        return Err(anyhow!(
            "apply_nomic_bert_encoder_block_gpu: hidden ({}) not divisible by num_heads ({})",
            hidden,
            num_heads
        ));
    }
    let head_dim = hidden / num_heads;
    let n_hidden_elems = (seq_len as usize) * (hidden as usize);

    // ---- 1. Fused QKV projection: split via slice_view + 3 matmuls ----
    //
    // GGUF layout for fused QKV: weight shape `[3*hidden, hidden]` with
    // out_dim contiguous in memory (out_dim 0..hidden = Q, hidden..2*hidden
    // = K, 2*hidden..3*hidden = V). Per-block sub-slice:
    //   Q weight = bytes [0 .. hidden*hidden*4)
    //   K weight = bytes [hidden*hidden*4 .. 2*hidden*hidden*4)
    //   V weight = bytes [2*hidden*hidden*4 .. 3*hidden*hidden*4)
    //
    // `MlxBuffer::slice_view` returns a sub-buffer that, when bound to a
    // kernel, passes the byte offset via `setBuffer:offset:atIndex:`.
    // The matmul kernel reads `out_features × in_features` from the
    // bound buffer and stays within the slice region.
    let weight_bytes_per_block = (hidden as usize) * (hidden as usize) * 4;
    let weight_elems_per_block = (hidden as usize) * (hidden as usize);
    let q_w = tensors
        .qkv_w
        .slice_view(0, weight_elems_per_block);
    let k_w = tensors
        .qkv_w
        .slice_view(weight_bytes_per_block as u64, weight_elems_per_block);
    let v_w = tensors.qkv_w.slice_view(
        (2 * weight_bytes_per_block) as u64,
        weight_elems_per_block,
    );

    // Optional bias also slices three ways (each is `[hidden]` of the
    // `[3*hidden]` fused bias). nomic-embed-text-v1.5 has none, but a
    // future variant might.
    let (q_b, k_b, v_b) = match tensors.qkv_b {
        None => (None, None, None),
        Some(qkvb) => {
            let q = qkvb.slice_view(0, hidden as usize);
            let k = qkvb.slice_view((hidden as usize * 4) as u64, hidden as usize);
            let v = qkvb.slice_view((2 * hidden as usize * 4) as u64, hidden as usize);
            (Some(q), Some(k), Some(v))
        }
    };
    let q_b_ref = q_b.as_ref();
    let k_b_ref = k_b.as_ref();
    let v_b_ref = v_b.as_ref();

    let q_proj = bert_linear_gpu(
        encoder, registry, device, input, &q_w, q_b_ref, seq_len, hidden, hidden,
    )
    .context("nomic block: q_proj linear")?;
    encoder.memory_barrier();
    let k_proj = bert_linear_gpu(
        encoder, registry, device, input, &k_w, k_b_ref, seq_len, hidden, hidden,
    )
    .context("nomic block: k_proj linear")?;
    encoder.memory_barrier();
    let v_proj = bert_linear_gpu(
        encoder, registry, device, input, &v_w, v_b_ref, seq_len, hidden, hidden,
    )
    .context("nomic block: v_proj linear")?;
    encoder.memory_barrier();

    // ---- 2. RoPE-NeoX on Q and K (V unrotated) ----
    //
    // `dispatch_rope_neox_f32` reads from `input` and writes to `output`;
    // both are `[seq_len * n_heads, head_dim]` in element layout. The
    // `[seq_len, hidden]` row-major layout above is byte-identical
    // because `hidden = n_heads * head_dim` and per-row the heads are
    // contiguous.
    //
    // `rope_dim = head_dim` for nomic-bert (full rotary, every dimension
    // rotated). Per `bert.cpp:63-67` llama.cpp passes `n_rot` which is
    // `n_embd_head` (the head_dim).
    let q_rotated = device
        .alloc_buffer(
            n_hidden_elems * 4,
            DType::F32,
            vec![seq_len as usize, hidden as usize],
        )
        .map_err(|e| anyhow!("alloc q_rotated: {e}"))?;
    let k_rotated = device
        .alloc_buffer(
            n_hidden_elems * 4,
            DType::F32,
            vec![seq_len as usize, hidden as usize],
        )
        .map_err(|e| anyhow!("alloc k_rotated: {e}"))?;

    dispatch_rope_neox_f32(
        encoder,
        registry,
        device.metal_device(),
        &q_proj,
        &q_rotated,
        rope_params,
        rope_positions,
        None, // freq_factors (no Gemma-style mixing for nomic-bert)
        seq_len,
        num_heads,
        head_dim,
        head_dim, // rope_dim = head_dim (full rotary)
    )
    .map_err(|e| anyhow!("nomic block: rope on Q: {e}"))?;
    dispatch_rope_neox_f32(
        encoder,
        registry,
        device.metal_device(),
        &k_proj,
        &k_rotated,
        rope_params,
        rope_positions,
        None,
        seq_len,
        num_heads,
        head_dim,
        head_dim,
    )
    .map_err(|e| anyhow!("nomic block: rope on K: {e}"))?;
    encoder.memory_barrier();

    // ---- 3. Attention with padding mask ----
    //
    // `bert_attention_with_mask_gpu` expects Q/K/V in seq-major
    // `[seq_len, num_heads, head_dim]` layout — same memory as
    // `[seq_len, hidden]`. Same mask + scale convention as the BERT lane.
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let attn_out = bert_attention_with_mask_gpu(
        encoder, registry, device, &q_rotated, &k_rotated, &v_proj, mask, seq_len, num_heads,
        head_dim, scale,
    )
    .context("nomic block: bidirectional attention")?;
    encoder.memory_barrier();

    // ---- 4. Output projection ----
    let attn_proj = bert_linear_gpu(
        encoder,
        registry,
        device,
        &attn_out,
        tensors.o_w,
        tensors.o_b,
        seq_len,
        hidden,
        hidden,
    )
    .context("nomic block: attention output projection")?;
    encoder.memory_barrier();

    // ---- 5. Residual + post-attention LayerNorm ----
    let after_attn_resid = bert_residual_add_gpu(
        encoder,
        registry,
        device,
        &attn_proj,
        input,
        n_hidden_elems as u32,
    )
    .context("nomic block: post-attention residual")?;
    encoder.memory_barrier();

    let after_attn_norm = bert_layer_norm_gpu(
        encoder,
        registry,
        device,
        &after_attn_resid,
        tensors.attn_norm_gamma,
        tensors.attn_norm_beta,
        eps,
        seq_len,
        hidden,
    )
    .context("nomic block: post-attention LayerNorm")?;
    encoder.memory_barrier();

    // ---- 6. SwiGLU FFN ----
    //
    // Per llama.cpp `build_ffn(LLM_FFN_SILU, LLM_FFN_PAR)` semantics
    // (graph.cpp:1141-1280):
    //   tmp  = up(cur)            // ffn_up linear
    //   cur  = gate(cur)          // ffn_gate linear (PAR: from input, not tmp)
    //   cur  = swiglu_split(cur, tmp) = silu(cur) * tmp
    //   cur  = down(cur)          // ffn_down linear
    //
    // mlx-native's `dispatch_silu_mul(gate, up, output)` computes
    // `output[i] = silu(gate[i]) * up[i]` exactly per the doc-comment at
    // `ops/silu_mul.rs:25-26`.
    let n_ffn = (seq_len as usize) * (intermediate as usize);

    let up_proj = bert_linear_gpu(
        encoder,
        registry,
        device,
        &after_attn_norm,
        tensors.up_w,
        tensors.up_b,
        seq_len,
        hidden,
        intermediate,
    )
    .context("nomic block: ffn_up linear")?;
    encoder.memory_barrier();

    let gate_proj = bert_linear_gpu(
        encoder,
        registry,
        device,
        &after_attn_norm,
        tensors.gate_w,
        tensors.gate_b,
        seq_len,
        hidden,
        intermediate,
    )
    .context("nomic block: ffn_gate linear")?;
    encoder.memory_barrier();

    let silu_gated = device
        .alloc_buffer(
            n_ffn * 4,
            DType::F32,
            vec![seq_len as usize, intermediate as usize],
        )
        .map_err(|e| anyhow!("alloc silu_gated: {e}"))?;
    // The silu_mul params buffer must outlive the encoder commit.
    // `_silu_params` keeps it on the local stack frame for the
    // remainder of this function — long enough because the caller's
    // `encoder.commit_and_wait()` must run before this function's
    // returned buffer can be used.
    let _silu_params = alloc_silu_mul_params(device, n_ffn as u32)?;
    dispatch_silu_mul(
        encoder,
        registry,
        device.metal_device(),
        &gate_proj,
        &up_proj,
        &silu_gated,
        &_silu_params,
        n_ffn as u32,
    )
    .map_err(|e| anyhow!("nomic block: silu_mul: {e}"))?;
    encoder.memory_barrier();

    let down_proj = bert_linear_gpu(
        encoder,
        registry,
        device,
        &silu_gated,
        tensors.down_w,
        tensors.down_b,
        seq_len,
        intermediate,
        hidden,
    )
    .context("nomic block: ffn_down linear")?;
    encoder.memory_barrier();

    // ---- 7. Residual + post-FFN LayerNorm ----
    let after_ffn_resid = bert_residual_add_gpu(
        encoder,
        registry,
        device,
        &down_proj,
        &after_attn_norm,
        n_hidden_elems as u32,
    )
    .context("nomic block: post-FFN residual")?;
    encoder.memory_barrier();

    let block_out = bert_layer_norm_gpu(
        encoder,
        registry,
        device,
        &after_ffn_resid,
        tensors.ffn_norm_gamma,
        tensors.ffn_norm_beta,
        eps,
        seq_len,
        hidden,
    )
    .context("nomic block: post-FFN LayerNorm")?;
    encoder.memory_barrier();

    // The silu_mul params buffer is dropped at function exit, AFTER
    // the encoder records all dispatches. Caller's `commit_and_wait`
    // must come before any use of the returned `block_out` — the params
    // buffer lifetime concern is bounded by THIS function call, not by
    // the buffer's eventual consumer.
    drop(_silu_params);

    Ok(block_out)
}

// ---------------------------------------------------------------------------
// Full forward pass
// ---------------------------------------------------------------------------

/// nomic-bert full-encoder forward pass on GPU. Embeds tokens, runs N
/// encoder blocks with RoPE+SwiGLU, pools, and L2-normalizes.
///
/// `valid_token_count` controls the attention mask: positions
/// `[valid_token_count, seq_len)` are masked out (mask value −1e30 →
/// post-softmax weight ≈ 0). For unpadded inputs pass
/// `valid_token_count = seq_len`. The mask is built unconditionally
/// because the empirical finding from BERT iter 67 is that the no-mask
/// path (`vit_attention_gpu` delegate) drops cosine to ~0.75-0.92 vs
/// llama-embedding while the masked path stays ≥ 0.99999.
///
/// Returns F32 `[hidden]` unit-norm vector.
pub fn apply_nomic_bert_full_forward_gpu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input_ids: &MlxBuffer,
    type_ids_opt: Option<&MlxBuffer>,
    weights: &LoadedNomicBertWeights,
    cfg: &NomicBertConfig,
    seq_len: u32,
    valid_token_count: u32,
) -> Result<MlxBuffer> {
    let pool_kind = match cfg.pooling_type {
        PoolingType::Mean => BertPoolKind::Mean,
        PoolingType::Cls => BertPoolKind::Cls,
        PoolingType::Last => BertPoolKind::Last,
        PoolingType::None => {
            return Err(anyhow!(
                "apply_nomic_bert_full_forward_gpu: pooling_type=None is not a single-vector embedding"
            ));
        }
        PoolingType::Rank => {
            return Err(anyhow!(
                "apply_nomic_bert_full_forward_gpu: pooling_type=Rank is reranker-only (out of scope for /v1/embeddings)"
            ));
        }
    };

    let hidden = cfg.hidden_size as u32;
    let num_heads = cfg.num_attention_heads as u32;
    let intermediate = cfg.intermediate_size as u32;
    let vocab = cfg.vocab_size as u32;
    let max_pos = cfg.max_position_embeddings as u32;
    let type_vocab = cfg.type_vocab_size as u32;
    let head_dim = (hidden / num_heads) as u32;
    let eps = cfg.layer_norm_eps;

    if seq_len < 32 {
        return Err(anyhow!(
            "apply_nomic_bert_full_forward_gpu: seq_len ({}) must be >= 32 (post-softmax matmul K floor)",
            seq_len
        ));
    }
    if head_dim < 32 {
        return Err(anyhow!(
            "apply_nomic_bert_full_forward_gpu: head_dim ({}) must be >= 32",
            head_dim
        ));
    }
    if seq_len > max_pos {
        return Err(anyhow!(
            "apply_nomic_bert_full_forward_gpu: seq_len ({}) > max_position_embeddings ({})",
            seq_len,
            max_pos
        ));
    }

    // ---- Embeddings ----
    //
    // BERT-family GGUFs that ship `token_types.weight` need a synthetic
    // all-zero type_ids buffer when caller passes None — single-segment
    // is the universal default and llama-embedding's behavior locked.
    let synthesized_type_ids: Option<MlxBuffer> =
        match (type_ids_opt, weights.token_types_weight()) {
            (None, Some(_)) => Some(alloc_zero_type_ids(device, seq_len)?),
            _ => None,
        };
    let effective_type_ids: Option<&MlxBuffer> =
        match (type_ids_opt, synthesized_type_ids.as_ref()) {
            (Some(b), _) => Some(b),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
    let token_types_for_call = if effective_type_ids.is_some() {
        weights.token_types_weight()
    } else {
        None
    };

    let mut hidden_states = nomic_bert_embeddings_gpu(
        encoder,
        registry,
        device,
        input_ids,
        effective_type_ids,
        weights.token_embd_weight()?,
        token_types_for_call,
        weights.embed_norm_weight()?,
        weights.embed_norm_bias()?,
        eps,
        seq_len,
        hidden,
        vocab,
        type_vocab,
    )
    .context("nomic full-forward: embeddings")?;
    encoder.memory_barrier();

    // ---- Padding mask + RoPE positions + RoPE params (built once) ----
    let mask = alloc_bert_attention_mask(device, seq_len, valid_token_count)?;
    let rope_positions = alloc_rope_positions(device, seq_len)?;
    let rope_params = alloc_rope_params(device, cfg.rope_freq_base, head_dim, head_dim)?;

    // ---- N encoder blocks ----
    for layer_idx in 0..cfg.num_hidden_layers {
        let tensors = NomicBertEncoderBlockTensors {
            qkv_w: weights.block_required(layer_idx, "attn_qkv.weight")?,
            qkv_b: weights.block_optional(layer_idx, "attn_qkv.bias"),
            o_w: weights.block_required(layer_idx, "attn_output.weight")?,
            o_b: weights.block_optional(layer_idx, "attn_output.bias"),
            attn_norm_gamma: weights.block_required(layer_idx, "attn_output_norm.weight")?,
            attn_norm_beta: weights.block_required(layer_idx, "attn_output_norm.bias")?,
            up_w: weights.block_required(layer_idx, "ffn_up.weight")?,
            up_b: weights.block_optional(layer_idx, "ffn_up.bias"),
            gate_w: weights.block_required(layer_idx, "ffn_gate.weight")?,
            gate_b: weights.block_optional(layer_idx, "ffn_gate.bias"),
            down_w: weights.block_required(layer_idx, "ffn_down.weight")?,
            down_b: weights.block_optional(layer_idx, "ffn_down.bias"),
            ffn_norm_gamma: weights.block_required(layer_idx, "layer_output_norm.weight")?,
            ffn_norm_beta: weights.block_required(layer_idx, "layer_output_norm.bias")?,
        };
        hidden_states = apply_nomic_bert_encoder_block_gpu(
            encoder,
            registry,
            device,
            &hidden_states,
            &tensors,
            &mask,
            &rope_positions,
            &rope_params,
            seq_len,
            hidden,
            num_heads,
            intermediate,
            eps,
        )
        .with_context(|| format!("nomic full-forward: block {}", layer_idx))?;
        encoder.memory_barrier();
    }

    // ---- Pool ----
    let pooled =
        bert_pool_gpu(encoder, registry, device, &hidden_states, pool_kind, seq_len, hidden)
            .context("nomic full-forward: pool")?;
    encoder.memory_barrier();

    // ---- L2 normalize ----
    //
    // After mean / cls pooling the buffer is `[1, hidden]` (one row per
    // request). `bert_l2_normalize_gpu` takes (rows, dim, eps); pass
    // rows=1 for a single embedding output. Eps mirrors the BERT lane's
    // l2 normalize eps to keep the post-norm scale consistent across
    // embedding models.
    bert_l2_normalize_gpu(encoder, registry, device, &pooled, 1e-12, 1, hidden)
        .context("nomic full-forward: l2 normalize")
}

// ---------------------------------------------------------------------------
// Tests — synthetic-shape encoder & full-forward
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    /// Minimum-shape config that satisfies every kernel floor:
    ///   seq_len >= 32, head_dim >= 32, even rope_dim.
    fn synthetic_min_cfg(num_layers: usize) -> NomicBertConfig {
        NomicBertConfig {
            hidden_size: 64,            // 2 heads * 32 head_dim
            num_attention_heads: 2,
            num_hidden_layers: num_layers,
            intermediate_size: 128,
            max_position_embeddings: 128,
            vocab_size: 100,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            pooling_type: PoolingType::Mean,
            rope_freq_base: 1000.0,
            causal_attention: false,
        }
    }

    /// Build a HashMap<String, MlxBuffer> with every required tensor for
    /// `num_layers` blocks at `synthetic_min_cfg`. All weights are tiny
    /// random F32 (deterministic seed for repeatability).
    fn synthetic_weights(
        device: &MlxDevice,
        cfg: &NomicBertConfig,
    ) -> Result<HashMap<String, MlxBuffer>> {
        // Tiny pseudo-random — deterministic linear hash of (key, idx).
        // Range chosen small enough that LayerNorm doesn't blow up
        // (post-LN scale/shift uses gamma=1, beta=0 for stability).
        let make_buf = |device: &MlxDevice, n: usize, key: &str| -> Result<MlxBuffer> {
            let buf = device
                .alloc_buffer(n * 4, DType::F32, vec![n])
                .map_err(|e| anyhow!("alloc {key}: {e}"))?;
            let slice: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut f32, n) };
            let key_hash: u32 = key.bytes().fold(2166136261u32, |acc, b| {
                acc.wrapping_mul(16777619).wrapping_add(b as u32)
            });
            for (i, slot) in slice.iter_mut().enumerate() {
                let h = key_hash
                    .wrapping_mul(2654435761)
                    .wrapping_add((i as u32).wrapping_mul(2246822519));
                // map u32 → f32 in roughly [-0.05, 0.05]
                *slot = ((h as i32) as f32 / i32::MAX as f32) * 0.05;
            }
            Ok(buf)
        };
        let make_ones = |device: &MlxDevice, n: usize, key: &str| -> Result<MlxBuffer> {
            let buf = device
                .alloc_buffer(n * 4, DType::F32, vec![n])
                .map_err(|e| anyhow!("alloc {key}: {e}"))?;
            let slice: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut f32, n) };
            slice.fill(1.0);
            Ok(buf)
        };
        let make_zeros = |device: &MlxDevice, n: usize, key: &str| -> Result<MlxBuffer> {
            let buf = device
                .alloc_buffer(n * 4, DType::F32, vec![n])
                .map_err(|e| anyhow!("alloc {key}: {e}"))?;
            let slice: &mut [f32] =
                unsafe { std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut f32, n) };
            slice.fill(0.0);
            Ok(buf)
        };

        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let type_vocab = cfg.type_vocab_size;

        let mut tensors: HashMap<String, MlxBuffer> = HashMap::new();
        // Stem.
        tensors.insert(
            "token_embd.weight".into(),
            make_buf(device, vocab * hidden, "token_embd")?,
        );
        tensors.insert(
            "token_types.weight".into(),
            make_buf(device, type_vocab * hidden, "token_types")?,
        );
        tensors.insert(
            "token_embd_norm.weight".into(),
            make_ones(device, hidden, "embd_norm_w")?,
        );
        tensors.insert(
            "token_embd_norm.bias".into(),
            make_zeros(device, hidden, "embd_norm_b")?,
        );
        // Per-layer.
        for il in 0..cfg.num_hidden_layers {
            // Fused QKV: [3*hidden, hidden].
            tensors.insert(
                format!("blk.{}.attn_qkv.weight", il),
                make_buf(device, 3 * hidden * hidden, &format!("qkv_w_{il}"))?,
            );
            tensors.insert(
                format!("blk.{}.attn_output.weight", il),
                make_buf(device, hidden * hidden, &format!("o_w_{il}"))?,
            );
            tensors.insert(
                format!("blk.{}.attn_output_norm.weight", il),
                make_ones(device, hidden, &format!("attn_norm_w_{il}"))?,
            );
            tensors.insert(
                format!("blk.{}.attn_output_norm.bias", il),
                make_zeros(device, hidden, &format!("attn_norm_b_{il}"))?,
            );
            tensors.insert(
                format!("blk.{}.ffn_up.weight", il),
                make_buf(device, intermediate * hidden, &format!("up_w_{il}"))?,
            );
            tensors.insert(
                format!("blk.{}.ffn_gate.weight", il),
                make_buf(device, intermediate * hidden, &format!("gate_w_{il}"))?,
            );
            tensors.insert(
                format!("blk.{}.ffn_down.weight", il),
                make_buf(device, hidden * intermediate, &format!("down_w_{il}"))?,
            );
            tensors.insert(
                format!("blk.{}.layer_output_norm.weight", il),
                make_ones(device, hidden, &format!("ffn_norm_w_{il}"))?,
            );
            tensors.insert(
                format!("blk.{}.layer_output_norm.bias", il),
                make_zeros(device, hidden, &format!("ffn_norm_b_{il}"))?,
            );
        }
        Ok(tensors)
    }

    #[test]
    fn full_forward_at_synthetic_min_config_produces_unit_norm_output() {
        let cfg = synthetic_min_cfg(2);
        let device = MlxDevice::new().expect("create device");
        let mut registry = KernelRegistry::new();
        register_nomic_bert_kernels(&mut registry);

        let tensors = synthetic_weights(&device, &cfg).expect("build synthetic weights");
        // MlxDevice doesn't impl Clone; create a separate handle. On Apple
        // Silicon both `MlxDevice::new()` instances bind to the same shared
        // Metal device — buffers allocated via `device` work in `weights`
        // (loaded-weights only holds the device for RAII).
        let weights_device = MlxDevice::new().expect("create weights device");
        let weights = LoadedNomicBertWeights::from_tensors_for_test(tensors, weights_device);

        // 32 input ids in vocab range [0, 100).
        let seq_len: u32 = 32;
        let input_ids = device
            .alloc_buffer((seq_len as usize) * 4, DType::U32, vec![seq_len as usize])
            .expect("alloc input_ids");
        {
            let slice: &mut [u32] = unsafe {
                std::slice::from_raw_parts_mut(input_ids.contents_ptr() as *mut u32, seq_len as usize)
            };
            for (i, slot) in slice.iter_mut().enumerate() {
                *slot = (i as u32 * 7 + 3) % cfg.vocab_size as u32;
            }
        }

        let mut encoder = device.command_encoder().expect("command_encoder");
        let pooled = apply_nomic_bert_full_forward_gpu(
            &mut encoder,
            &mut registry,
            &device,
            &input_ids,
            None,
            &weights,
            &cfg,
            seq_len,
            seq_len,
        )
        .expect("nomic full forward");
        encoder.commit_and_wait().expect("commit_and_wait");

        // Assert: output shape correct + unit-norm.
        assert_eq!(pooled.element_count(), cfg.hidden_size);
        let view: &[f32] = pooled.as_slice::<f32>().expect("read pooled f32");
        let norm: f32 = view.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "expected ||y||₂ ≈ 1.0 (post-l2-normalize), got {norm}"
        );
        // Sanity: not all zeros, not NaN/Inf.
        let n_nonzero = view.iter().filter(|v| v.abs() > 1e-12).count();
        assert!(
            n_nonzero >= cfg.hidden_size / 2,
            "expected most components non-zero, got {n_nonzero} of {}",
            cfg.hidden_size
        );
        for &v in view {
            assert!(v.is_finite(), "non-finite output element: {v}");
        }
    }

    #[test]
    fn full_forward_rejects_seq_len_below_floor() {
        let cfg = synthetic_min_cfg(1);
        let device = MlxDevice::new().expect("create device");
        let mut registry = KernelRegistry::new();
        register_nomic_bert_kernels(&mut registry);

        let tensors = synthetic_weights(&device, &cfg).expect("build synthetic weights");
        // MlxDevice doesn't impl Clone; create a separate handle. On Apple
        // Silicon both `MlxDevice::new()` instances bind to the same shared
        // Metal device — buffers allocated via `device` work in `weights`
        // (loaded-weights only holds the device for RAII).
        let weights_device = MlxDevice::new().expect("create weights device");
        let weights = LoadedNomicBertWeights::from_tensors_for_test(tensors, weights_device);

        let seq_len: u32 = 16; // below 32 floor
        let input_ids = device
            .alloc_buffer((seq_len as usize) * 4, DType::U32, vec![seq_len as usize])
            .expect("alloc input_ids");

        let mut encoder = device.command_encoder().expect("command_encoder");
        let err = apply_nomic_bert_full_forward_gpu(
            &mut encoder,
            &mut registry,
            &device,
            &input_ids,
            None,
            &weights,
            &cfg,
            seq_len,
            seq_len,
        )
        .expect_err("seq_len < 32 must reject");
        let msg = format!("{err}");
        assert!(msg.contains("seq_len") && msg.contains("32"), "error: {msg}");
    }

    /// End-to-end smoke against the on-disk `nomic-embed-text-v1.5-f16.gguf`.
    /// Loads the real config + weights + tokenizer, encodes "hello world",
    /// pads to seq_len=32, runs the full forward, asserts the pooled
    /// 768-dim output is unit-norm and finite. Validates that the
    /// production-shape forward composes correctly at scale (hidden=768,
    /// n_heads=12, head_dim=64, n_ff=3072, num_layers=12) with REAL
    /// weights — strictly stronger than the synthetic-min-shape gate.
    ///
    /// Skips cleanly when the model isn't on disk so CI / fresh checkouts
    /// don't false-fail.
    ///
    /// The cosine-≥0.999 parity-vs-llama-embedding test lands in iter 78
    /// — once the ground-truth vector is generated and burned in as a
    /// constant array. This test is the prerequisite gate ("can the
    /// production-scale forward actually run?").
    #[test]
    fn full_forward_at_production_scale_on_real_nomic_gguf_produces_unit_norm_output() {
        use mlx_native::gguf::GgufFile;
        use std::path::Path;

        use super::super::tokenizer::build_nomic_wordpiece_tokenizer;

        let model_path =
            Path::new("/opt/hf2q/models/bert-test/nomic-embed-text-v1.5-f16.gguf");
        if !model_path.exists() {
            eprintln!(
                "skipping: nomic GGUF fixture not at {}",
                model_path.display()
            );
            return;
        }

        // Open + parse config.
        let gguf = GgufFile::open(model_path).expect("open nomic GGUF");
        let cfg = NomicBertConfig::from_gguf(&gguf).expect("parse nomic config");
        // Validate the production-shape parameters we expect for
        // nomic-embed-text-v1.5 — protects against a different model
        // accidentally being placed at this path.
        assert_eq!(cfg.hidden_size, 768, "expected nomic-embed-text-v1.5 hidden=768");
        assert_eq!(cfg.num_hidden_layers, 12, "expected 12 blocks");
        assert_eq!(cfg.num_attention_heads, 12, "expected 12 heads");
        assert_eq!(cfg.intermediate_size, 3072, "expected n_ff=3072");

        // Tokenize "hello world" with [CLS]/[SEP] brackets.
        let tok = build_nomic_wordpiece_tokenizer(model_path).expect("build tokenizer");
        let real_ids = tok.encode("hello world", true);
        assert_eq!(real_ids.len(), 4, "expected [CLS] hello world [SEP], got {real_ids:?}");

        // Pad right with [PAD] up to seq_len = 32 (the kernel-floor minimum).
        let seq_len: u32 = 32;
        let pad_id = tok.specials().pad;
        let mut padded_ids: Vec<u32> = real_ids.clone();
        while padded_ids.len() < seq_len as usize {
            padded_ids.push(pad_id);
        }
        let valid_token_count: u32 = real_ids.len() as u32;

        // Build the device + registry.
        let device = MlxDevice::new().expect("create device");
        let mut registry = KernelRegistry::new();
        register_nomic_bert_kernels(&mut registry);

        // Load real weights.
        let weights = LoadedNomicBertWeights::load_from_path(model_path, &cfg)
            .expect("load real nomic weights");
        assert!(weights.len() > 0, "loader returned zero tensors");
        // 12 blocks × 9 required suffixes + 3 stem (token_embd, embd_norm.{w,b})
        // + 1 token_types = 112 tensors. Lock that to detect manifest drift.
        assert_eq!(
            weights.len(),
            112,
            "expected 112 tensors loaded from nomic GGUF, got {}",
            weights.len()
        );

        // Build input_ids buffer.
        let input_ids = device
            .alloc_buffer(
                (seq_len as usize) * 4,
                DType::U32,
                vec![seq_len as usize],
            )
            .expect("alloc input_ids");
        {
            let slice: &mut [u32] = unsafe {
                std::slice::from_raw_parts_mut(
                    input_ids.contents_ptr() as *mut u32,
                    seq_len as usize,
                )
            };
            slice.copy_from_slice(&padded_ids);
        }

        // Run full forward.
        let mut encoder = device.command_encoder().expect("command_encoder");
        let pooled = apply_nomic_bert_full_forward_gpu(
            &mut encoder,
            &mut registry,
            &device,
            &input_ids,
            None, // single-segment input → loader synthesizes zero type_ids
            &weights,
            &cfg,
            seq_len,
            valid_token_count,
        )
        .expect("nomic full forward at production scale");
        encoder.commit_and_wait().expect("commit_and_wait");

        // ---- Asserts ----
        assert_eq!(
            pooled.element_count(),
            cfg.hidden_size,
            "expected output dim = hidden_size = 768"
        );
        let view: &[f32] = pooled.as_slice::<f32>().expect("read pooled f32");
        let norm: f32 = view.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "expected ||y||₂ ≈ 1.0 (post-l2-normalize), got {norm}"
        );
        // Most components should be non-trivial (real weights => meaningful
        // output). Threshold 1e-6 is generous; a healthy output has
        // magnitudes around 1/sqrt(hidden) ≈ 0.036 average per component.
        let n_nontrivial = view.iter().filter(|v| v.abs() > 1e-6).count();
        assert!(
            n_nontrivial >= 700,
            "expected most of 768 components non-trivial, got {n_nontrivial}"
        );
        for &v in view {
            assert!(v.is_finite(), "non-finite output element: {v}");
        }
        // Per-element magnitude sanity: max abs should be small (post-L2
        // ||y||=1 on 768 components → max element bounded by 1.0; real
        // embeddings rarely exceed 0.5).
        let max_abs = view.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
        assert!(
            max_abs < 1.0,
            "max |y_i| = {max_abs} unexpectedly large (post-l2 should be ≤ 1.0)"
        );
        eprintln!(
            "[nomic real-gguf smoke] hidden={}, ||y||₂={:.6}, max|y|={:.4}, first4={:?}",
            view.len(),
            norm,
            max_abs,
            &view[..4]
        );
    }
}
