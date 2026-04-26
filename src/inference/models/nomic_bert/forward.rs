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
    bert_l2_normalize_gpu, bert_layer_norm_gpu, bert_linear_bf16_gpu, bert_linear_gpu,
    bert_pool_gpu, bert_residual_add_gpu, bert_residual_layer_norm_gpu,
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
    /// Fused QKV projection: `[3*hidden, hidden]` F32 weight + optional
    /// `[3*hidden]` bias. Splitting happens at use site via `slice_view`.
    pub qkv_w: &'a MlxBuffer,
    pub qkv_b: Option<&'a MlxBuffer>,
    /// Pre-cast BF16 fused QKV weight (`[3*hidden, hidden]`, 2 bytes/elem).
    /// When `Some`, the composer slices this BF16 buffer for Q/K/V and
    /// dispatches `bert_linear_bf16_gpu` (no per-call cast). When `None`,
    /// falls back to the F32 path via `bert_linear_gpu`. Iter-83 perf
    /// optimization — populated by `LoadedNomicBertWeights::block_weight_bf16`.
    pub qkv_w_bf16: Option<&'a MlxBuffer>,
    /// Attention output projection: `[hidden, hidden]` F32 + optional bias.
    pub o_w: &'a MlxBuffer,
    pub o_b: Option<&'a MlxBuffer>,
    /// Pre-cast BF16 attention output weight (`[hidden, hidden]`).
    pub o_w_bf16: Option<&'a MlxBuffer>,
    /// Post-attention LayerNorm γ, β. Both `[hidden]` F32.
    pub attn_norm_gamma: &'a MlxBuffer,
    pub attn_norm_beta: &'a MlxBuffer,
    /// FFN up projection: `[intermediate, hidden]` F32 + optional bias.
    pub up_w: &'a MlxBuffer,
    pub up_b: Option<&'a MlxBuffer>,
    pub up_w_bf16: Option<&'a MlxBuffer>,
    /// FFN gate projection: `[intermediate, hidden]` F32 + optional bias.
    pub gate_w: &'a MlxBuffer,
    pub gate_b: Option<&'a MlxBuffer>,
    pub gate_w_bf16: Option<&'a MlxBuffer>,
    /// FFN down projection: `[hidden, intermediate]` F32 + optional bias.
    pub down_w: &'a MlxBuffer,
    pub down_b: Option<&'a MlxBuffer>,
    pub down_w_bf16: Option<&'a MlxBuffer>,
    /// Post-FFN LayerNorm γ, β. Both `[hidden]` F32.
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
    // **Iter 83 perf fast path: BF16 pre-cast.** When `qkv_w_bf16` is
    // present (production lane via `LoadedNomicBertWeights::load`), slice
    // it directly into Q/K/V BF16 sub-views and dispatch
    // `bert_linear_bf16_gpu` (no per-call cast, no per-call BF16 alloc).
    // Eliminates 3 × 12 = 36 cast dispatches per request for the QKV
    // step alone. Falls back to F32 path when BF16 is absent (test
    // scaffolding only). Slice math: BF16 is 2 bytes/elem so per-block
    // byte offsets are half the F32 ones, while element counts stay
    // the same.
    //
    // **Iter 80 enabling fix (mlx-native v0.4.3+):** `MlxBuffer::slice_view`
    // now propagates `byte_offset` through `KernelArg::Buffer` (was
    // hardcoded 0 pre-fix), making the documented slice contract honored
    // end-to-end through the matmul kernel.
    let weight_elems_per_block = (hidden as usize) * (hidden as usize);
    let weight_bytes_per_block_f32 = weight_elems_per_block * 4;
    let weight_bytes_per_block_bf16 = weight_elems_per_block * 2;

    let q_w = tensors.qkv_w.slice_view(0, weight_elems_per_block);
    let k_w = tensors
        .qkv_w
        .slice_view(weight_bytes_per_block_f32 as u64, weight_elems_per_block);
    let v_w = tensors.qkv_w.slice_view(
        (2 * weight_bytes_per_block_f32) as u64,
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

    // Dispatch QKV via BF16 fast path when pre-cast is available; else
    // F32 (the cast happens inside bert_linear_gpu per-call).
    let (q_proj, k_proj, v_proj) = if let Some(qkv_bf16) = tensors.qkv_w_bf16 {
        let q_w_bf16 = qkv_bf16.slice_view(0, weight_elems_per_block);
        let k_w_bf16 =
            qkv_bf16.slice_view(weight_bytes_per_block_bf16 as u64, weight_elems_per_block);
        let v_w_bf16 = qkv_bf16.slice_view(
            (2 * weight_bytes_per_block_bf16) as u64,
            weight_elems_per_block,
        );
        // Iter-87 perf: Q/K/V matmuls all read `input` and write to
        // disjoint output buffers (q_proj, k_proj, v_proj). No RAW
        // hazard between them — Metal's MTLDispatchType::Concurrent
        // can overlap their execution. The single barrier AFTER V is
        // sufficient: it gates the RoPE/attention reads of all three
        // outputs that follow. Removing the inter-QKV barriers lets
        // the GPU schedule the three matmuls concurrently when
        // hardware resources allow.
        let q = bert_linear_bf16_gpu(
            encoder, registry, device, input, &q_w_bf16, q_b_ref, seq_len, hidden, hidden,
        )
        .context("nomic block: q_proj bf16 linear")?;
        let k = bert_linear_bf16_gpu(
            encoder, registry, device, input, &k_w_bf16, k_b_ref, seq_len, hidden, hidden,
        )
        .context("nomic block: k_proj bf16 linear")?;
        let v = bert_linear_bf16_gpu(
            encoder, registry, device, input, &v_w_bf16, v_b_ref, seq_len, hidden, hidden,
        )
        .context("nomic block: v_proj bf16 linear")?;
        encoder.memory_barrier();
        (q, k, v)
    } else {
        let q = bert_linear_gpu(
            encoder, registry, device, input, &q_w, q_b_ref, seq_len, hidden, hidden,
        )
        .context("nomic block: q_proj linear (f32 fallback)")?;
        let k = bert_linear_gpu(
            encoder, registry, device, input, &k_w, k_b_ref, seq_len, hidden, hidden,
        )
        .context("nomic block: k_proj linear (f32 fallback)")?;
        let v = bert_linear_gpu(
            encoder, registry, device, input, &v_w, v_b_ref, seq_len, hidden, hidden,
        )
        .context("nomic block: v_proj linear (f32 fallback)")?;
        encoder.memory_barrier();
        (q, k, v)
    };

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
        None,
        seq_len,
        num_heads,
        head_dim,
        head_dim,
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
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let attn_out = bert_attention_with_mask_gpu(
        encoder, registry, device, &q_rotated, &k_rotated, &v_proj, mask, seq_len, num_heads,
        head_dim, scale,
    )
    .context("nomic block: bidirectional attention")?;
    encoder.memory_barrier();

    // ---- 4. Output projection ----
    let attn_proj = if let Some(o_w_bf16) = tensors.o_w_bf16 {
        bert_linear_bf16_gpu(
            encoder, registry, device, &attn_out, o_w_bf16, tensors.o_b, seq_len, hidden, hidden,
        )
        .context("nomic block: attention output bf16 projection")?
    } else {
        bert_linear_gpu(
            encoder, registry, device, &attn_out, tensors.o_w, tensors.o_b, seq_len, hidden, hidden,
        )
        .context("nomic block: attention output projection (f32 fallback)")?
    };
    encoder.memory_barrier();

    // ---- 5. Fused residual + post-attention LayerNorm ----
    // Iter-86 perf optimization: replaces the bert_residual_add_gpu →
    // bert_layer_norm_gpu pair with one fused dispatch. Saves the
    // intermediate `after_attn_resid` writethrough + one Metal
    // dispatch + one barrier.
    let _ = n_hidden_elems; // shape arithmetic preserved for the embed-stage caller
    let after_attn_norm = bert_residual_layer_norm_gpu(
        encoder,
        registry,
        device,
        &attn_proj,
        input,
        tensors.attn_norm_gamma,
        tensors.attn_norm_beta,
        eps,
        seq_len,
        hidden,
    )
    .context("nomic block: post-attention residual+LayerNorm (fused)")?;
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

    // Iter-87 perf: ffn_up and ffn_gate matmuls both read
    // after_attn_norm and write to disjoint outputs (up_proj,
    // gate_proj). No RAW between them — Metal can overlap the two
    // matmuls. Single barrier AFTER both gates the silu_mul read of
    // both outputs that follows.
    let up_proj = if let Some(up_w_bf16) = tensors.up_w_bf16 {
        bert_linear_bf16_gpu(
            encoder, registry, device, &after_attn_norm, up_w_bf16, tensors.up_b, seq_len, hidden,
            intermediate,
        )
        .context("nomic block: ffn_up bf16 linear")?
    } else {
        bert_linear_gpu(
            encoder, registry, device, &after_attn_norm, tensors.up_w, tensors.up_b, seq_len,
            hidden, intermediate,
        )
        .context("nomic block: ffn_up linear (f32 fallback)")?
    };

    let gate_proj = if let Some(gate_w_bf16) = tensors.gate_w_bf16 {
        bert_linear_bf16_gpu(
            encoder, registry, device, &after_attn_norm, gate_w_bf16, tensors.gate_b, seq_len,
            hidden, intermediate,
        )
        .context("nomic block: ffn_gate bf16 linear")?
    } else {
        bert_linear_gpu(
            encoder, registry, device, &after_attn_norm, tensors.gate_w, tensors.gate_b, seq_len,
            hidden, intermediate,
        )
        .context("nomic block: ffn_gate linear (f32 fallback)")?
    };
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
    // Operand order verified iter 79: `silu_mul(gate, up, out)` =
    // `silu(gate) * up`. Swapping to (up, gate) made cosine WORSE
    // (0.098 → 0.039), confirming `silu(gate)*up` is correct.
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

    let down_proj = if let Some(down_w_bf16) = tensors.down_w_bf16 {
        bert_linear_bf16_gpu(
            encoder, registry, device, &silu_gated, down_w_bf16, tensors.down_b, seq_len,
            intermediate, hidden,
        )
        .context("nomic block: ffn_down bf16 linear")?
    } else {
        bert_linear_gpu(
            encoder, registry, device, &silu_gated, tensors.down_w, tensors.down_b, seq_len,
            intermediate, hidden,
        )
        .context("nomic block: ffn_down linear (f32 fallback)")?
    };
    encoder.memory_barrier();

    // ---- 7. Fused residual + post-FFN LayerNorm ----
    // Same fusion as step 5 — saves the intermediate writethrough +
    // one dispatch + one barrier.
    let block_out = bert_residual_layer_norm_gpu(
        encoder,
        registry,
        device,
        &down_proj,
        &after_attn_norm,
        tensors.ffn_norm_gamma,
        tensors.ffn_norm_beta,
        eps,
        seq_len,
        hidden,
    )
    .context("nomic block: post-FFN residual+LayerNorm (fused)")?;
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
            qkv_w_bf16: weights.block_weight_bf16(layer_idx, "attn_qkv.weight"),
            o_w: weights.block_required(layer_idx, "attn_output.weight")?,
            o_b: weights.block_optional(layer_idx, "attn_output.bias"),
            o_w_bf16: weights.block_weight_bf16(layer_idx, "attn_output.weight"),
            attn_norm_gamma: weights.block_required(layer_idx, "attn_output_norm.weight")?,
            attn_norm_beta: weights.block_required(layer_idx, "attn_output_norm.bias")?,
            up_w: weights.block_required(layer_idx, "ffn_up.weight")?,
            up_b: weights.block_optional(layer_idx, "ffn_up.bias"),
            up_w_bf16: weights.block_weight_bf16(layer_idx, "ffn_up.weight"),
            gate_w: weights.block_required(layer_idx, "ffn_gate.weight")?,
            gate_b: weights.block_optional(layer_idx, "ffn_gate.bias"),
            gate_w_bf16: weights.block_weight_bf16(layer_idx, "ffn_gate.weight"),
            down_w: weights.block_required(layer_idx, "ffn_down.weight")?,
            down_b: weights.block_optional(layer_idx, "ffn_down.bias"),
            down_w_bf16: weights.block_weight_bf16(layer_idx, "ffn_down.weight"),
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
    //
    // Pass `valid_token_count` (not `seq_len`) so:
    //   - Mean averages over real positions only (sum 0..valid / valid).
    //     The existing `bert_pool_mean_f32` kernel iterates `[0, seq_len)`
    //     of the kernel-binding param and divides by that value — passing
    //     `valid_token_count` yields the correct masked mean over real
    //     positions, with no kernel change needed.
    //   - Last reads row `valid_token_count - 1` (the actual last token,
    //     not a padded position).
    //   - Cls reads row 0 unconditionally — unaffected by either argument.
    //
    // Without this, mean-pooled output diverges from llama-embedding on
    // any input shorter than `seq_len` (which is essentially every input
    // due to the K=32 floor). nomic-embed-text-v1.5 uses Mean pool per
    // its GGUF metadata `nomic-bert.pooling_type = 1`, so this fix is
    // load-bearing for cosine parity.
    let pooled = bert_pool_gpu(
        encoder,
        registry,
        device,
        &hidden_states,
        pool_kind,
        valid_token_count,
        hidden,
    )
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

    /// llama-embedding ground-truth vector for "hello world" tokenized
    /// against `nomic-embed-text-v1.5-f16.gguf` with `--pooling mean`.
    /// Generated 2026-04-26 via:
    ///   `llama-embedding -m nomic-embed-text-v1.5-f16.gguf -p "hello world" --pooling mean --embd-output-format json`
    /// (binary: `/opt/homebrew/Cellar/llama.cpp/8680/bin/llama-embedding`,
    /// release b8680). The output is l2-normalized (||y||₂ ≈ 1.000000)
    /// per llama-embedding's default normalization.
    #[rustfmt::skip]
    const LLAMA_EMBEDDING_GROUND_TRUTH_HELLO_WORLD: [f32; 768] = [
        -6.6696000e-03f32, -1.3524000e-03f32, -1.7149610e-01f32, 8.4113000e-03f32,
        5.8636000e-03f32, 6.9821200e-02f32, -2.0240000e-04f32, -4.3022800e-02f32,
        -1.4626900e-02f32, -5.4056500e-02f32, 5.4160000e-04f32, 3.9272200e-02f32,
        2.7769300e-02f32, 8.0812800e-02f32, 4.5334100e-02f32, -6.2951900e-02f32,
        1.0281800e-02f32, -2.9656100e-02f32, -4.2753000e-02f32, 2.9597000e-02f32,
        -3.7053000e-03f32, -9.4301000e-02f32, -7.5451000e-03f32, 3.8064000e-02f32,
        9.2231700e-02f32, -1.4276000e-02f32, -1.4984500e-02f32, 6.1637500e-02f32,
        6.4217000e-03f32, -2.1997000e-02f32, -1.1787000e-03f32, -1.0889600e-02f32,
        -2.0770000e-04f32, 1.5721300e-02f32, 3.9444200e-02f32, 2.7844000e-03f32,
        3.2542000e-02f32, 1.7387900e-02f32, 1.6315700e-02f32, 5.8692000e-03f32,
        -4.7176000e-03f32, -1.4858700e-02f32, 1.1955900e-02f32, 1.0195000e-02f32,
        6.5921400e-02f32, -1.5323000e-03f32, -4.1892000e-03f32, 2.5850000e-04f32,
        8.6810500e-02f32, -6.0505200e-02f32, -1.8267700e-02f32, 5.3402000e-03f32,
        -9.7460000e-04f32, 6.0159100e-02f32, 6.7261100e-02f32, 3.5314900e-02f32,
        4.9696500e-02f32, -6.1601000e-02f32, 2.4186600e-02f32, 3.4579300e-02f32,
        2.1759600e-02f32, 4.3670700e-02f32, 3.2912600e-02f32, 6.5303800e-02f32,
        -1.7461700e-02f32, -3.3584400e-02f32, -2.5229100e-02f32, 3.5515100e-02f32,
        -2.7050000e-03f32, 1.8090300e-02f32, 7.3137600e-02f32, 4.2705000e-03f32,
        1.0861100e-02f32, 1.4041100e-02f32, 2.4605300e-02f32, 2.8004000e-02f32,
        1.8594200e-02f32, 7.9048000e-03f32, -9.1900000e-04f32, -1.1905300e-02f32,
        3.5421100e-02f32, -4.1623100e-02f32, 5.5484100e-02f32, -4.4268500e-02f32,
        -3.2420500e-02f32, -7.3050200e-02f32, -4.0210600e-02f32, 1.5107500e-02f32,
        -7.4528400e-02f32, -2.3277800e-02f32, 7.2323200e-02f32, 2.4769200e-02f32,
        -5.1810000e-04f32, -1.8537900e-02f32, -4.0699700e-02f32, 2.0469300e-02f32,
        -4.6032500e-02f32, -1.4164000e-03f32, -2.0057200e-02f32, -1.0966100e-02f32,
        1.5139200e-02f32, -2.6177000e-02f32, -2.1159000e-03f32, 3.9117600e-02f32,
        7.4003400e-02f32, 3.3914800e-02f32, -4.2793600e-02f32, -1.1042500e-02f32,
        -5.4241700e-02f32, -3.8041700e-02f32, -3.3099100e-02f32, 1.0952800e-02f32,
        -7.7578000e-03f32, 1.6444700e-02f32, 1.1090300e-02f32, -1.8598600e-02f32,
        3.7661700e-02f32, -8.0060300e-02f32, 1.3430200e-02f32, 3.8800400e-02f32,
        -3.6834600e-02f32, -3.0368000e-03f32, -3.9599900e-02f32, 1.6384100e-02f32,
        5.1860500e-02f32, 5.1746800e-02f32, -7.9243300e-02f32, -1.9026100e-02f32,
        2.3724100e-02f32, 5.8361000e-03f32, 1.3606100e-02f32, -3.2963600e-02f32,
        -2.5896500e-02f32, -2.4711400e-02f32, -2.5973700e-02f32, 1.4827800e-02f32,
        -1.5112600e-02f32, -1.5942600e-02f32, 5.6933500e-02f32, 3.3677600e-02f32,
        8.9373000e-03f32, 1.3058600e-02f32, 2.4422200e-02f32, -4.9721500e-02f32,
        -4.0742700e-02f32, -3.7903000e-02f32, 6.7944700e-02f32, -5.4161300e-02f32,
        2.5137400e-02f32, -2.9522100e-02f32, 1.1179000e-03f32, 4.3306200e-02f32,
        4.2004700e-02f32, 3.6765400e-02f32, 1.7319600e-02f32, -9.2276000e-03f32,
        -1.9733600e-02f32, 2.4374200e-02f32, 3.7715700e-02f32, -4.6261000e-02f32,
        1.3235600e-02f32, -8.5385000e-03f32, -7.7219000e-03f32, 1.0578100e-02f32,
        3.6785100e-02f32, -5.5686800e-02f32, 3.2216600e-02f32, 6.3634400e-02f32,
        4.7028000e-03f32, 3.2951100e-02f32, -3.6591600e-02f32, -3.8097500e-02f32,
        3.5200100e-02f32, -3.2147600e-02f32, -2.0294400e-02f32, -8.4025000e-03f32,
        -8.9155000e-03f32, -2.8797300e-02f32, 2.7853000e-02f32, 6.2297000e-03f32,
        1.7037400e-02f32, -4.1399900e-02f32, 5.3571000e-03f32, 2.3880700e-02f32,
        -1.3950300e-02f32, -2.4504100e-02f32, -2.2816600e-02f32, 2.1248000e-03f32,
        2.1516600e-02f32, -3.5409700e-02f32, -8.3450000e-03f32, 1.7045600e-02f32,
        -6.2701600e-02f32, -3.6372200e-02f32, 2.0275800e-02f32, -6.0924000e-03f32,
        1.9449000e-02f32, 1.3768700e-02f32, 2.3274300e-02f32, -8.7041200e-02f32,
        -4.0821800e-02f32, -1.5951000e-03f32, -2.7441100e-02f32, -2.3361100e-02f32,
        -1.2320000e-04f32, 8.0487700e-02f32, 4.9427200e-02f32, 3.5657900e-02f32,
        3.8404300e-02f32, 1.3656900e-02f32, 6.7240500e-02f32, -6.1013000e-02f32,
        -5.1268500e-02f32, -1.9204600e-02f32, -1.8456100e-02f32, -9.4280000e-03f32,
        -1.3013600e-02f32, -3.9915000e-02f32, -6.8722000e-03f32, -5.2381000e-03f32,
        4.1953500e-02f32, -3.7336400e-02f32, 4.6914900e-02f32, 1.5287100e-02f32,
        6.0664000e-02f32, -4.3220000e-04f32, -7.6455200e-02f32, 3.9420000e-04f32,
        -6.3508300e-02f32, 2.0475900e-02f32, -3.1551200e-02f32, -6.4586800e-02f32,
        1.8433200e-02f32, 2.4937000e-02f32, -1.7175900e-02f32, 3.8528800e-02f32,
        4.5732900e-02f32, 6.8505000e-02f32, -1.3920300e-02f32, -2.9826000e-02f32,
        -2.9718900e-02f32, 3.1809800e-02f32, 1.2525400e-02f32, -3.7864700e-02f32,
        -4.5741600e-02f32, 2.0945800e-02f32, 1.3378600e-02f32, 3.9765000e-03f32,
        -1.6016100e-02f32, 7.9120000e-04f32, -5.0325400e-02f32, -2.6335300e-02f32,
        -2.0792100e-02f32, 1.3855000e-03f32, 2.7455600e-02f32, -1.1552200e-02f32,
        -1.8247500e-02f32, 1.2089200e-02f32, 1.1794200e-02f32, -1.8306400e-02f32,
        2.4806100e-02f32, -1.1313790e-01f32, 3.8121800e-02f32, -2.6435400e-02f32,
        -4.8520900e-02f32, 3.4552000e-02f32, -4.8187700e-02f32, 3.4003100e-02f32,
        3.5869800e-02f32, -4.3102900e-02f32, 1.2138600e-02f32, 9.6684000e-03f32,
        9.5008000e-03f32, 2.7620800e-02f32, -4.3571100e-02f32, -6.1012000e-03f32,
        2.4550600e-02f32, 1.4137200e-02f32, -2.1751800e-02f32, 1.8641100e-02f32,
        -2.5030500e-02f32, -3.0357800e-02f32, -1.2105300e-02f32, -3.4376800e-02f32,
        8.0324000e-03f32, 1.1767000e-02f32, -9.7419000e-03f32, 1.2958400e-02f32,
        -3.3330700e-02f32, -1.3954200e-02f32, 1.3599500e-02f32, 4.6106700e-02f32,
        3.0477600e-02f32, 6.9333100e-02f32, 1.2361600e-02f32, 2.9699600e-02f32,
        2.6872200e-02f32, 2.8872900e-02f32, -1.1028400e-02f32, -9.7210000e-03f32,
        -1.2504600e-02f32, -3.0737000e-03f32, 5.5915000e-02f32, -5.5938000e-03f32,
        -1.8363800e-02f32, 2.6282000e-03f32, 7.4948300e-02f32, -1.8678300e-02f32,
        4.8992800e-02f32, -4.4942000e-03f32, -4.6219300e-02f32, 7.8223400e-02f32,
        -9.1623600e-02f32, -2.8647000e-03f32, -5.4467100e-02f32, 4.3308000e-02f32,
        -2.0742300e-02f32, 3.2399100e-02f32, 6.9515000e-02f32, 1.8378400e-02f32,
        -3.8602500e-02f32, -5.0964700e-02f32, 2.5752500e-02f32, -3.2207600e-02f32,
        -2.0015000e-03f32, 2.2829800e-02f32, 3.4840500e-02f32, 2.6006800e-02f32,
        2.3608000e-02f32, -4.8204200e-02f32, -1.4688700e-02f32, 1.2953900e-02f32,
        -4.0061300e-02f32, -4.3547100e-02f32, -1.5404500e-02f32, 4.3021400e-02f32,
        1.4217300e-02f32, -1.9293600e-02f32, -4.3386300e-02f32, 5.0931100e-02f32,
        -5.2732000e-03f32, 2.3002600e-02f32, 4.0056100e-02f32, -1.2371900e-02f32,
        -2.2102400e-02f32, -1.9255200e-02f32, -2.2667500e-02f32, 1.6477400e-02f32,
        -1.5024500e-02f32, 2.3886200e-02f32, 6.4518000e-03f32, 2.5857200e-02f32,
        -3.7417800e-02f32, 2.5491300e-02f32, 6.3017000e-03f32, 2.4324100e-02f32,
        1.1121200e-02f32, 3.2617300e-02f32, 9.7421000e-03f32, -6.4688500e-02f32,
        -2.6930400e-02f32, -1.0810000e-03f32, 1.4429800e-02f32, -1.4482400e-02f32,
        -2.9594800e-02f32, 3.6383500e-02f32, 2.7113400e-02f32, 6.3338000e-03f32,
        2.5931000e-02f32, -1.8233900e-02f32, -9.2634000e-03f32, -2.1566000e-02f32,
        1.7372000e-03f32, 3.1390600e-02f32, 2.2766800e-02f32, 6.7679000e-03f32,
        -4.8544900e-02f32, -3.4086800e-02f32, 8.0462000e-03f32, 4.0696200e-02f32,
        -1.6917700e-02f32, -2.5898600e-02f32, 3.7125400e-02f32, -2.8145100e-02f32,
        1.1707400e-02f32, 3.4267900e-02f32, 1.5698300e-02f32, -2.7624200e-02f32,
        6.9315000e-03f32, 3.6654900e-02f32, 1.4375900e-02f32, -2.4399900e-02f32,
        3.5763000e-03f32, 2.1591000e-03f32, -4.3111000e-03f32, -2.9810300e-02f32,
        6.8243000e-03f32, -2.2369500e-02f32, -2.1174000e-02f32, 6.8368000e-03f32,
        -6.7607000e-03f32, 1.2870100e-02f32, 2.8253000e-02f32, -7.1764500e-02f32,
        6.3303000e-03f32, -9.4630000e-04f32, -5.1895500e-02f32, -2.5373100e-02f32,
        8.5210000e-03f32, -1.5810700e-02f32, 6.4544500e-02f32, 6.3795000e-02f32,
        4.5600200e-02f32, -5.5528200e-02f32, -4.1763200e-02f32, 1.1410400e-02f32,
        3.6577900e-02f32, -6.8033600e-02f32, -1.2944800e-02f32, -1.1250000e-03f32,
        1.7747800e-02f32, 7.7825100e-02f32, 9.6088000e-03f32, -1.3749000e-02f32,
        -3.6817300e-02f32, 6.7867900e-02f32, 2.8122900e-02f32, 2.5646100e-02f32,
        1.0362000e-03f32, -3.9197900e-02f32, -9.8872000e-03f32, 1.4315400e-02f32,
        1.8575000e-02f32, 1.2935500e-02f32, -2.2592300e-02f32, -2.4799100e-02f32,
        4.4715800e-02f32, -1.6318900e-02f32, 3.3302100e-02f32, 3.5868800e-02f32,
        6.6783100e-02f32, -3.4930700e-02f32, -5.7269400e-02f32, -5.9972000e-03f32,
        -7.8350000e-03f32, 1.2211830e-01f32, 8.8283900e-02f32, -9.9352000e-03f32,
        -5.0083900e-02f32, -5.3087000e-03f32, -2.0628500e-02f32, 1.7979200e-02f32,
        4.4053000e-02f32, 9.6263000e-03f32, 8.6672300e-02f32, -5.0582700e-02f32,
        4.5380800e-02f32, -1.9539800e-02f32, -4.5610000e-04f32, 4.2559600e-02f32,
        -1.9513300e-02f32, 6.2631000e-03f32, -3.1664400e-02f32, 8.3836000e-03f32,
        1.1540100e-02f32, -5.4867100e-02f32, 5.8530000e-04f32, 1.3838000e-03f32,
        -7.2131000e-03f32, 9.4528000e-03f32, -4.6331800e-02f32, -5.2411900e-02f32,
        -1.9209000e-02f32, -1.3257400e-02f32, -6.0218300e-02f32, 2.2961200e-02f32,
        -1.6933200e-02f32, -1.3100100e-02f32, -1.4583500e-02f32, 2.6643300e-02f32,
        2.7661400e-02f32, 2.8252300e-02f32, -6.9593600e-02f32, 1.6248700e-02f32,
        5.0440000e-02f32, 6.9895800e-02f32, -1.1571000e-03f32, 1.3644400e-02f32,
        -1.7439800e-02f32, 7.1650000e-03f32, -2.8896000e-03f32, -2.4393600e-02f32,
        3.6425400e-02f32, -2.9890000e-04f32, -2.7375400e-02f32, -4.7094000e-03f32,
        -4.5289300e-02f32, 3.5725500e-02f32, -4.5007200e-02f32, 1.1070000e-03f32,
        3.8081600e-02f32, -1.1230100e-02f32, 7.1999000e-03f32, 3.3610400e-02f32,
        6.8639000e-03f32, 2.3139700e-02f32, 2.6155600e-02f32, -5.7708000e-02f32,
        9.9852000e-03f32, 2.1354700e-02f32, 3.1218800e-02f32, -1.1297100e-02f32,
        -2.3065700e-02f32, 4.2179300e-02f32, 9.3753200e-02f32, -4.2773900e-02f32,
        2.5180000e-02f32, -1.1069100e-02f32, -3.0348900e-02f32, 5.0103700e-02f32,
        1.1354600e-02f32, -1.8240100e-02f32, -3.2781300e-02f32, 1.2266100e-02f32,
        -7.6380600e-02f32, 6.8787400e-02f32, -1.2997500e-02f32, -5.4016200e-02f32,
        2.7228400e-02f32, 2.6439900e-02f32, 1.4635100e-02f32, 1.0713200e-02f32,
        -2.3172800e-02f32, -4.4703300e-02f32, 2.8427700e-02f32, -6.5769000e-03f32,
        4.3078000e-03f32, 1.5217600e-02f32, 2.5405900e-02f32, 1.5774300e-02f32,
        -2.9051000e-02f32, -4.0942000e-03f32, -8.7075000e-03f32, -3.5142900e-02f32,
        4.2014600e-02f32, 3.4278600e-02f32, -5.1490600e-02f32, 1.6976500e-02f32,
        2.2717700e-02f32, 3.1094700e-02f32, -6.9700300e-02f32, -4.6632600e-02f32,
        -2.8557600e-02f32, -1.0152700e-02f32, -4.7277100e-02f32, -5.7679200e-02f32,
        -5.0320000e-04f32, 2.1446700e-02f32, -3.2161400e-02f32, -7.4619700e-02f32,
        4.6693000e-02f32, 4.7707500e-02f32, -2.4216800e-02f32, 1.5537500e-02f32,
        3.1047200e-02f32, -5.2569000e-03f32, -1.5880500e-02f32, -1.2518600e-02f32,
        1.3172300e-02f32, -1.7127400e-02f32, -2.9639000e-02f32, -4.1154700e-02f32,
        2.2904300e-02f32, -2.9324900e-02f32, 1.6750600e-02f32, -4.9756000e-03f32,
        4.0822200e-02f32, -4.2819000e-03f32, -6.4213000e-02f32, -1.8390500e-02f32,
        -7.2800000e-05f32, -5.5501300e-02f32, -5.8984000e-03f32, 5.1533200e-02f32,
        -1.3947800e-02f32, 1.2336100e-02f32, 5.5780000e-04f32, -7.4789400e-02f32,
        3.9419300e-02f32, -3.4653400e-02f32, -2.4352800e-02f32, 2.6578200e-02f32,
        5.3825000e-02f32, -2.4245500e-02f32, -3.2574900e-02f32, 4.9105500e-02f32,
        -3.9906800e-02f32, -4.3803200e-02f32, -1.7663800e-02f32, -4.4167900e-02f32,
        -2.9276000e-02f32, 6.4075000e-03f32, 6.0689900e-02f32, -6.9809000e-02f32,
        4.9774200e-02f32, 7.8172000e-02f32, 8.2345000e-03f32, 4.1580200e-02f32,
        1.8442300e-02f32, 1.5560500e-02f32, 7.5401900e-02f32, 2.9353300e-02f32,
        -2.2047500e-02f32, 8.5528000e-03f32, 2.7844900e-02f32, -1.5099400e-02f32,
        4.2347800e-02f32, -2.0616000e-03f32, -1.7948600e-02f32, -6.9906400e-02f32,
        -3.4608900e-02f32, -1.5580200e-02f32, 4.9552700e-02f32, 2.4922100e-02f32,
        2.7784400e-02f32, -6.3345000e-03f32, -4.4251600e-02f32, -5.0236200e-02f32,
        -5.7502200e-02f32, 6.2764400e-02f32, 4.0139600e-02f32, -6.8978000e-03f32,
        -6.4271800e-02f32, 2.3647000e-03f32, 1.6232200e-02f32, 2.9681700e-02f32,
        1.9716900e-02f32, -2.7960000e-03f32, -3.1999600e-02f32, 1.7260500e-02f32,
        6.1012600e-02f32, 1.3232500e-02f32, 1.8163400e-02f32, 1.7620000e-04f32,
        1.4968500e-02f32, -4.0804300e-02f32, 4.3764700e-02f32, 2.4680400e-02f32,
        5.5778100e-02f32, 4.4632200e-02f32, 7.5896700e-02f32, 6.1313200e-02f32,
        4.8259900e-02f32, -1.3964600e-02f32, -2.7013200e-02f32, -1.1387000e-02f32,
        1.2016400e-02f32, -2.7300600e-02f32, -8.4480900e-02f32, 2.0433600e-02f32,
        -1.0788300e-02f32, 2.6292000e-03f32, -6.6455100e-02f32, -2.4444200e-02f32,
        3.3388000e-02f32, -2.1442300e-02f32, -3.2666300e-02f32, 1.9507800e-02f32,
        -9.2234800e-02f32, 1.3595900e-02f32, -1.5368600e-02f32, -2.0472100e-02f32,
        -2.8691200e-02f32, -4.4806300e-02f32, -2.7665200e-02f32, 3.8195300e-02f32,
        2.7114000e-02f32, 2.2422500e-02f32, 2.9953900e-02f32, 2.4472000e-03f32,
        1.1154500e-02f32, -1.4125200e-02f32, -4.3632000e-02f32, 3.4539100e-02f32,
        4.5745500e-02f32, -4.3739300e-02f32, 6.4070700e-02f32, -1.9190600e-02f32,
        -7.7880300e-02f32, -6.0991400e-02f32, -1.2944600e-02f32, -1.5316700e-02f32,
        -5.9819000e-03f32, -3.1322900e-02f32, -2.6103200e-02f32, 2.9772100e-02f32,
        -1.2600200e-02f32, 1.2044100e-02f32, -3.9712600e-02f32, 3.5522000e-02f32,
        -4.1178100e-02f32, -1.2571100e-02f32, 2.2523000e-02f32, -7.8828000e-03f32,
        4.6103000e-03f32, -3.9207100e-02f32, 1.3137100e-02f32, 4.1068200e-02f32,
        -9.2080000e-03f32, -1.5108000e-03f32, -1.3505600e-02f32, 6.4108600e-02f32,
        1.5352400e-02f32, 3.4981400e-02f32, -9.6561000e-03f32, 4.0101400e-02f32,
        -2.7272300e-02f32, -1.0268500e-02f32, -6.3159000e-03f32, 5.9788300e-02f32,
        7.2369200e-02f32, 4.2342700e-02f32, -4.1509400e-02f32, -2.3098600e-02f32,
        -2.6804800e-02f32, 2.0771000e-03f32, 1.8563900e-02f32, -3.3814200e-02f32,
        1.5673800e-02f32, -3.7488400e-02f32, -2.7946300e-02f32, -3.7747300e-02f32,
        -3.2442600e-02f32, 2.8004300e-02f32, -2.6214700e-02f32, 2.7615200e-02f32,
        -7.3416000e-03f32, -5.4686100e-02f32, 5.0802000e-03f32, -3.3259000e-02f32,
        -2.3903400e-02f32, -7.0778800e-02f32, 1.7292100e-02f32, 6.2792200e-02f32,
        -4.9236000e-03f32, -2.3950700e-02f32, 3.4221000e-02f32, 7.2967300e-02f32,
        -9.6511000e-03f32, -2.0971600e-02f32, 2.4748800e-02f32, 5.5330000e-04f32,
        -1.0364100e-02f32, -7.1326500e-02f32, -4.3360000e-04f32, 3.6105500e-02f32,
        9.9634000e-03f32, 2.2385100e-02f32, 6.2977100e-02f32, -4.1682900e-02f32,
        4.3001200e-02f32, -1.4988600e-02f32, -2.2700000e-04f32, 9.6763000e-03f32,
        2.5719400e-02f32, -2.6735100e-02f32, -5.0922400e-02f32, -4.4518000e-03f32,
    ];

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

    /// Iter-83 perf isolation test: time 10 sequential full-forward
    /// dispatches on the same loaded weights, single process, no HTTP /
    /// spawn_blocking overhead. Identifies the floor cost of the
    /// forward pass alone. Comparison target: llama-embedding's
    /// internal `prompt_eval=4.54 ms` for the same 4-token input.
    ///
    /// `#[ignore]` because it depends on the real GGUF and the timing
    /// is environment-sensitive; run with
    /// `cargo test --release --bin hf2q -- forward_timing_10x_warm --ignored --nocapture`.
    #[test]
    #[ignore = "perf timing test; run with --ignored --nocapture"]
    fn forward_timing_10x_warm() {
        use mlx_native::gguf::GgufFile;
        use std::path::Path;
        use std::time::Instant;

        use super::super::tokenizer::build_nomic_wordpiece_tokenizer;

        let model_path =
            Path::new("/opt/hf2q/models/bert-test/nomic-embed-text-v1.5-f16.gguf");
        if !model_path.exists() {
            eprintln!("skipping: nomic GGUF fixture not at {}", model_path.display());
            return;
        }

        let gguf = GgufFile::open(model_path).expect("open nomic GGUF");
        let cfg = NomicBertConfig::from_gguf(&gguf).expect("parse cfg");
        let tok = build_nomic_wordpiece_tokenizer(model_path).expect("tok");

        let real_ids = tok.encode("hello world", true);
        let valid_token_count = real_ids.len() as u32;
        let seq_len: u32 = 32;
        let pad_id = tok.specials().pad;
        let mut padded_ids: Vec<u32> = real_ids.clone();
        while padded_ids.len() < seq_len as usize {
            padded_ids.push(pad_id);
        }

        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        register_nomic_bert_kernels(&mut registry);
        let weights = LoadedNomicBertWeights::load_from_path(model_path, &cfg).expect("load");

        let input_ids = device
            .alloc_buffer((seq_len as usize) * 4, DType::U32, vec![seq_len as usize])
            .expect("alloc input_ids");
        unsafe {
            let s: &mut [u32] = std::slice::from_raw_parts_mut(
                input_ids.contents_ptr() as *mut u32,
                seq_len as usize,
            );
            s.copy_from_slice(&padded_ids);
        }

        // 10 sequential forwards. Each gets its own encoder + commit.
        eprintln!("--- 10 sequential forwards (single process, no HTTP) ---");
        let mut timings: Vec<f64> = Vec::with_capacity(10);
        for i in 0..10 {
            let t0 = Instant::now();
            let mut encoder = device.command_encoder().expect("encoder");
            let pooled = apply_nomic_bert_full_forward_gpu(
                &mut encoder,
                &mut registry,
                &device,
                &input_ids,
                None,
                &weights,
                &cfg,
                seq_len,
                valid_token_count,
            )
            .expect("forward");
            encoder.commit_and_wait().expect("commit");
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            timings.push(elapsed_ms);
            // Force a memory read so the GPU is forced to finish.
            let _ = pooled.as_slice::<f32>().expect("read")[0];
            eprintln!("  forward {}: {:.2} ms", i + 1, elapsed_ms);
        }
        let mean = timings.iter().sum::<f64>() / timings.len() as f64;
        let min = timings.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!(
            "  --> mean {:.2} ms, min {:.2} ms (llama-embedding reference: ~4.54 ms)",
            mean, min
        );
    }

    /// Cosine-parity gate: hf2q's full-forward output for "hello world"
    /// against `nomic-embed-text-v1.5-f16.gguf` must match
    /// `llama-embedding`'s output to cosine ≥ 0.999. **This is the
    /// correctness gate that closes Task #16.**
    ///
    /// **STATUS (iter 78): FAILING. Cosine = 0.098589 — structural
    /// divergence (vectors near-orthogonal despite both being unit-norm
    /// with similar per-element magnitudes). Marked `#[ignore]` until
    /// bisection identifies the root cause. Suspect list, in priority
    /// order:**
    /// 1. **Fused-QKV slice convention** — verify Q at bytes [0, K·N·4),
    ///    K at [K·N·4, 2·K·N·4), V at [2·K·N·4, 3·K·N·4) matches the
    ///    output-dim ordering llama.cpp uses in `create_tensor_qkv`.
    ///    The diagnostic A/B is to extract Q/K/V into separate buffers
    ///    at load-time and re-run; if cosine improves to ≥ 0.999, the
    ///    slice ordering is wrong.
    /// 2. **RoPE convention** — verify NeoX pair convention matches
    ///    llama.cpp's `LLAMA_ROPE_TYPE_NORM` for nomic-bert (per
    ///    llama-arch.cpp:9266). Try interleaved (`dispatch_rope`) as A/B.
    /// 3. **SwiGLU operand order** — `dispatch_silu_mul(gate, up, out)`
    ///    computes `silu(gate) * up`. llama.cpp's `swiglu_split(cur=gate,
    ///    tmp=up)` per graph.cpp:1220. Should match. Verify by swapping
    ///    args.
    /// 4. **Pre-existing BERT-lane parity gap** — bge uses CLS pool, so
    ///    even if the per-token output diverges, CLS taking position 0
    ///    masks the mismatch. The "cosine ≥ 0.999" claim in the ADR for
    ///    bge was never automated. Run a BERT-lane parity test as the
    ///    first bisection step; if it also fails, the issue is in the
    ///    shared primitives, not nomic-specific.
    ///
    /// Diagnostic plan in iter 79:
    /// - Add a bge cosine-parity test using llama-embedding ground truth
    ///   on "hello world" with mxbai or bge as the corroborating arch.
    /// - If bge passes: focus bisection on nomic-specific code (fused
    ///   QKV split, RoPE, SwiGLU).
    /// - If bge fails: shared primitive bug (most likely matmul layout
    ///   convention or mean-pool divisor); fix before nomic.
    ///
    /// Both engines see identical tokenized input (`[CLS] hello world
    /// [SEP]` → 4 tokens). hf2q pads to seq_len=32 with `[PAD]` and
    /// passes `valid_token_count=4` to the forward, which routes through
    /// `bert_pool_gpu(Mean, valid_token_count=4)` — sums 4 real-position
    /// rows and divides by 4, identical semantically to llama-embedding's
    /// 4-token mean pool.
    ///
    /// Both outputs are l2-normalized, so cosine = dot product.
    ///
    /// Skips when the model isn't on disk.
    ///
    /// **Iter 79 (closed): GREEN at cosine 0.999962** after bisecting the
    /// 0.098 failure to a `slice_view` + `KernelArg::Buffer` interaction
    /// in mlx-native v0.4.2 (encoder.rs line 166 ignores
    /// MlxBuffer::byte_offset). Workaround: explicit byte-copy of Q/K/V
    /// weight regions into fresh buffers in `apply_nomic_bert_encoder_block_gpu`.
    /// Upstream mlx-native fix queued for iter 80.
    #[test]
    fn full_forward_matches_llama_embedding_on_hello_world() {
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

        // ---- Load model + tokenize ----
        let gguf = GgufFile::open(model_path).expect("open nomic GGUF");
        let cfg = NomicBertConfig::from_gguf(&gguf).expect("parse nomic config");
        let tok = build_nomic_wordpiece_tokenizer(model_path).expect("build tokenizer");

        let real_ids = tok.encode("hello world", true);
        let valid_token_count: u32 = real_ids.len() as u32;

        // ---- Pad right to seq_len=32 with [PAD] ----
        let seq_len: u32 = 32;
        let pad_id = tok.specials().pad;
        let mut padded_ids: Vec<u32> = real_ids.clone();
        while padded_ids.len() < seq_len as usize {
            padded_ids.push(pad_id);
        }

        // ---- Build device + load weights ----
        let device = MlxDevice::new().expect("create device");
        let mut registry = KernelRegistry::new();
        register_nomic_bert_kernels(&mut registry);

        let weights = LoadedNomicBertWeights::load_from_path(model_path, &cfg)
            .expect("load real nomic weights");

        // ---- Build input_ids buffer ----
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

        // ---- Run hf2q full forward ----
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
            valid_token_count,
        )
        .expect("nomic full forward");
        encoder.commit_and_wait().expect("commit_and_wait");

        // ---- Compute cosine similarity vs ground truth ----
        let hf2q_view: &[f32] = pooled.as_slice::<f32>().expect("read hf2q pooled f32");
        assert_eq!(hf2q_view.len(), 768);
        let truth: &[f32] = &LLAMA_EMBEDDING_GROUND_TRUTH_HELLO_WORLD;

        // Both vectors are l2-normalized so cosine = dot product. We
        // still divide by ||a|| * ||b|| to be robust to any tiny
        // post-pool drift in either pipeline (and to surface a
        // non-unit-norm regression).
        let dot: f32 = hf2q_view.iter().zip(truth.iter()).map(|(a, b)| a * b).sum();
        let na: f32 = hf2q_view.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = truth.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cosine = dot / (na * nb);

        let max_abs_diff = hf2q_view
            .iter()
            .zip(truth.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        eprintln!(
            "[nomic parity] cosine={:.6}, ||hf2q||₂={:.6}, ||truth||₂={:.6}, max_abs_diff={:.4e}",
            cosine, na, nb, max_abs_diff
        );
        eprintln!(
            "  hf2q  first4 = {:?}\n  truth first4 = {:?}",
            &hf2q_view[..4],
            &truth[..4]
        );

        // Cosine gate. The Phase 2b accuracy contract is ≥ 0.999 (the
        // bge / mxbai accuracy bar). Anything below indicates a real
        // numerical divergence that must be diagnosed, not papered over.
        assert!(
            cosine >= 0.999,
            "cosine {cosine:.6} below 0.999 gate; hf2q diverges from llama-embedding",
        );
    }
}
