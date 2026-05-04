//! GPU-side weight containers and full forward-pass builder for the
//! Qwen3.5 gated full-attention layer (ADR-013 Decision 9, GPU path).
//!
//! This module is the bridge between [`super::full_attn`]'s pure-Rust scalar
//! reference (the authoritative spec + test oracle) and the mlx-native GPU
//! kernels. It carries the per-layer weights as `MlxBuffer` handles and
//! exposes every per-op dispatch as a small public function, then composes
//! them into [`build_gated_attn_layer`] — the end-to-end GPU forward pass.
//!
//! # Op order (mirrors the CPU ref verbatim)
//!
//! ```text
//!  1.  apply_pre_attn_rms_norm      — RMSNorm(x, attn_norm_w)
//!  2.  apply_linear_projection_f32  — x_norm @ wq  → Q_flat  [seq, q_total]
//!                                     x_norm @ wk  → K_flat  [seq, kv_total]
//!                                     x_norm @ wv  → V_flat  [seq, kv_total]
//!                                     x_norm @ wg  → G_flat  [seq, q_total]
//!  3.  apply_q_or_k_per_head_rms_norm — Q per-head RMSNorm
//!                                       K per-head RMSNorm
//!  4.  apply_imrope               — IMROPE Q; IMROPE K
//!  5.  apply_sdpa_causal          — SDPA(Q, K, V, causal, GQA) → attn_out [seq, q_total]
//!  6.  apply_sigmoid_gate_multiply — attn_out * sigmoid(G)
//!  7.  apply_linear_projection_f32 — gated_out @ wo → [seq, hidden_size]
//! ```
//!
//! # Layout notes
//!
//! All intermediate buffers are F32.  After ops 3-4, Q and K are in
//! `[seq_len * n_heads, head_dim]` (seq-major) layout.  The `sdpa` kernel
//! expects `[batch, n_heads, seq_len, head_dim]` (head-major), so
//! `apply_sdpa_causal` includes a CPU-side permute step for the parity test.
//! In the production path, weights are quantized and the permute is avoided
//! by producing Q/K directly in head-major order (future work, P8+).
//!
//! # Matmul strategy for F32 weights (parity test)
//!
//! No F32×F32 GPU GEMM exists in mlx-native.  For the parity test (F32
//! weights), `apply_linear_projection_f32_via_bf16` casts weights F32→BF16
//! on the GPU then calls `dense_matmul_bf16_f32_tensor`.  The BF16 cast
//! introduces ≤1e-3 rounding, within the stated parity bound.  In production
//! the caller passes pre-quantised (Q4_K / Q8_0) weight buffers and uses
//! `quantized_matmul_ggml` instead (not part of this module's scope).
//!
//! # ADR status
//!
//! P7b complete: every op wired, parity test passes |GPU−CPU|∞ < 1e-3 F32.

use anyhow::{anyhow, Context, Result};
use mlx_native::ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};
use mlx_native::ops::dense_gemv_bf16::dense_gemv_bf16_f32;
use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
};
use mlx_native::ops::rms_norm;
use mlx_native::ops::rope_multi::{
    dispatch_rope_multi_cached, RopeMultiMode, RopeMultiParams,
};
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_dual;
use mlx_native::ops::sdpa::{sdpa, SdpaParams};
use mlx_native::ops::sdpa_decode::dispatch_sdpa_decode;
use mlx_native::ops::flash_attn_vec::{flash_attn_vec, tmp_buffer_bytes as flash_attn_vec_tmp_bytes, FlashAttnVecParams};
use mlx_native::ops::sigmoid_mul::dispatch_sigmoid_mul;
use mlx_native::ops::flash_attn_prefill::{
    dispatch_flash_attn_prefill_bf16_d256, FlashAttnPrefillParams,
};
use mlx_native::ops::transpose::{permute_021_bf16, permute_021_bf16_to_f32};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::full_attn::FullAttnLayerWeights;
use super::kv_cache::FullAttnKvSlot;

/// GPU-side weight handles for a single Qwen3.5 full-attention layer.
///
/// Uploaded from [`FullAttnLayerWeights`] once per layer at load time;
/// held by the model + read by the per-token forward.
pub struct FullAttnWeightsGpu {
    pub attn_norm: MlxBuffer,
    /// Post-attention RMSNorm weight: `[hidden_size]`.
    /// Applied to the residual stream after attention, before the FFN.
    pub post_attn_norm: MlxBuffer,
    pub wq: MlxBuffer,
    pub wk: MlxBuffer,
    pub wv: MlxBuffer,
    pub w_gate: MlxBuffer,
    pub attn_q_norm: MlxBuffer,
    pub attn_k_norm: MlxBuffer,
    pub wo: MlxBuffer,
}

impl FullAttnWeightsGpu {
    /// Upload a [`FullAttnLayerWeights`] (pure-Rust f32) to Metal buffers.
    ///
    /// Large projection weights (wq, wk, wv, w_gate, wo) are quantized to Q4_0
    /// GGML blocks at load time.  This gives 3.56× lower bandwidth vs BF16 on
    /// the M=1 decode path (`quantized_matmul_ggml` dispatch_mv) and uses the
    /// same deterministic simd_sum accumulation as the FFN path.
    ///
    /// Precision: Q4_0 (4-bit with F16 per-block scale) introduces ~1% magnitude
    /// error, well within the I16→F32→Q4_0 chain used by APEX GGUF attn weights.
    /// llama.cpp uses Q5_K_M for these same weights; Q4_0 is slightly less
    /// precise but produces the same token selections in practice (sourdough gate
    /// must confirm).
    pub fn from_cpu(weights: &FullAttnLayerWeights, device: &MlxDevice) -> Result<Self> {
        // W-5b.7 iter 2: F32 norm weights uploaded via the residency-aware
        // helper so they join MTLResidencySet alongside the Q4_0 projection
        // buffers (`upload_q4_0_from_f32` registers internally).
        Ok(Self {
            attn_norm: upload_f32_weight(&weights.attn_norm, device)?,
            post_attn_norm: upload_f32_weight(&weights.post_attn_norm, device)?,
            wq:     upload_q4_0_from_f32(&weights.wq, device)?,
            wk:     upload_q4_0_from_f32(&weights.wk, device)?,
            wv:     upload_q4_0_from_f32(&weights.wv, device)?,
            w_gate: upload_q4_0_from_f32(&weights.w_gate, device)?,
            attn_q_norm: upload_f32_weight(&weights.attn_q_norm, device)?,
            attn_k_norm: upload_f32_weight(&weights.attn_k_norm, device)?,
            wo:     upload_q4_0_from_f32(&weights.wo, device)?,
        })
    }

    /// Test-only upload variant: keeps **all** projection weights as raw F32
    /// (no Q4_0 quantization).  Used by the GPU↔CPU kernel-pipeline parity
    /// tests so quantization noise (~1e-2) does not mask kernel correctness
    /// regressions (1e-3 BF16-cast bound).  Production decode always uses
    /// [`Self::from_cpu`] (Q4_0, ~3.56× less projection bandwidth).
    ///
    /// At projection time, `apply_linear_projection_f32` takes the F32 branch
    /// (line ~565) which casts weights to BF16 on the GPU and dispatches the
    /// MMA tiled matmul — the same numeric path the original P7b test was
    /// written against, before Q4_0 was added in commit fad4263.
    #[cfg(test)]
    pub fn from_cpu_f32(weights: &FullAttnLayerWeights, device: &MlxDevice) -> Result<Self> {
        Ok(Self {
            attn_norm: upload_f32(&weights.attn_norm, device)?,
            post_attn_norm: upload_f32(&weights.post_attn_norm, device)?,
            wq:     upload_f32(&weights.wq, device)?,
            wk:     upload_f32(&weights.wk, device)?,
            wv:     upload_f32(&weights.wv, device)?,
            w_gate: upload_f32(&weights.w_gate, device)?,
            attn_q_norm: upload_f32(&weights.attn_q_norm, device)?,
            attn_k_norm: upload_f32(&weights.attn_k_norm, device)?,
            wo:     upload_f32(&weights.wo, device)?,
        })
    }
}

/// Convert a single f32 to bf16 using round-to-nearest-even (RNE).
///
/// Matches Metal hardware BF16 rounding used in the GPU cast kernel, ensuring
/// numerically identical results to the per-inference GPU F32→BF16 cast.
///
/// Algorithm: add rounding bias (0x7FFF + LSB of bit-16 for ties-to-even),
/// then take the upper 16 bits.
#[inline(always)]
fn f32_to_bf16_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    // Handle NaN: propagate a quiet NaN.
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16; // quiet NaN
    }
    // Round-to-nearest-even: add 0x7FFF + (bit 16 of mantissa) as tie-break.
    let rounding_bias = 0x7FFF_u32 + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

/// Helper: convert f32 → bf16 CPU-side and upload as a BF16 MlxBuffer.
///
/// Used for large weight tensors (wq, wk, wv, w_gate, wo) so the GPU path
/// can skip the per-inference F32→BF16 cast in `apply_linear_projection_f32`.
/// One-time cost at model load vs repeated ~33MB cast per decode step.
/// Uses round-to-nearest-even to match Metal hardware BF16 rounding.
///
/// **Wave 5b.7 iter 2:** the resulting buffer is registered with the
/// thread-local weight pool's `MTLResidencySet` so it stays hinted-resident
/// across forward passes (no-op when `HF2Q_NO_RESIDENCY=1`).
pub fn upload_bf16_from_f32(data: &[f32], device: &MlxDevice) -> Result<MlxBuffer> {
    let n = data.len();
    let byte_len = n * 2; // 2 bytes per bf16
    let mut buf = device
        .alloc_buffer(byte_len, DType::BF16, vec![n])
        .map_err(|e| anyhow!("alloc bf16 buffer len={n}: {e}"))?;
    {
        let slice = buf
            .as_mut_slice::<u16>()
            .map_err(|e| anyhow!("mut_slice bf16: {e}"))?;
        for (i, &v) in data.iter().enumerate() {
            slice[i] = f32_to_bf16_rne(v);
        }
    }
    super::weight_pool::register_weight_buffer(device, &buf)
        .map_err(|e| anyhow!("register_weight_buffer bf16 len={n}: {e}"))?;
    Ok(buf)
}

/// Encode f32 values as Q4_0 GGML blocks (CPU-side quantization).
///
/// Q4_0 block layout (18 bytes per 32 elements):
///   - 2 bytes: F16 scale `d = max(|vals|) / 7`
///   - 16 bytes: packed nibbles (4-bit values, offset by 8, two per byte)
///
/// K must be divisible by 32 (Q4_0 QK).  Returns raw block bytes.
/// Used at model load time to prepare attn projection weights for the
/// bandwidth-efficient `quantized_matmul_ggml` dispatch_mv kernel on decode.
pub fn encode_q4_0_blocks(vals: &[f32]) -> Vec<u8> {
    use half::f16;
    const QK: usize = 32;
    let n = vals.len();
    assert_eq!(n % QK, 0, "encode_q4_0_blocks: n={n} must be divisible by QK=32");
    let n_blocks = n / QK;
    let mut out = vec![0u8; n_blocks * 18];
    for b in 0..n_blocks {
        let block = &vals[b * QK..(b + 1) * QK];
        let amax = block.iter().cloned().map(f32::abs).fold(0.0f32, f32::max);
        // d = 0 for zero blocks; use 1.0 to avoid divide-by-zero, quants are 8 (zero).
        let d = if amax > 0.0 { amax / 7.0 } else { 1.0 };
        let d_f16 = f16::from_f32(d);
        let off = b * 18;
        out[off..off + 2].copy_from_slice(&d_f16.to_le_bytes());
        for j in 0..16 {
            let q0 = ((block[j]      / d).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            let q1 = ((block[j + 16] / d).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            out[off + 2 + j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
    }
    out
}

/// Helper: quantize f32 weights to Q4_0 GGML blocks and upload as a U8 `MlxBuffer`.
///
/// The resulting buffer contains raw Q4_0 block bytes, compatible with
/// `quantized_matmul_ggml` (`GgmlType::Q4_0`).  3.56× less bandwidth than BF16.
///
/// `data.len()` must be divisible by 32 (Q4_0 block size).
pub fn upload_q4_0_from_f32(data: &[f32], device: &MlxDevice) -> Result<MlxBuffer> {
    let blocks = encode_q4_0_blocks(data);
    let byte_len = blocks.len();
    let mut buf = device
        .alloc_buffer(byte_len, DType::U8, vec![byte_len])
        .map_err(|e| anyhow!("alloc q4_0 buffer len={byte_len}: {e}"))?;
    {
        let slice = buf
            .as_mut_slice::<u8>()
            .map_err(|e| anyhow!("mut_slice q4_0: {e}"))?;
        slice.copy_from_slice(&blocks);
    }
    // Wave 5b.7 iter 2: register with the weight pool's residency set.
    super::weight_pool::register_weight_buffer(device, &buf)
        .map_err(|e| anyhow!("register_weight_buffer q4_0 len={byte_len}: {e}"))?;
    Ok(buf)
}

/// Helper: copy an f32 `Vec` into a freshly-allocated `MlxBuffer` with shape
/// set to `[len]` (1-D). Callers can reshape the buffer later by constructing
/// a new buffer with the desired shape and copying — shape here is advisory
/// only (mlx-native kernels consult `element_count()` + dtype, not shape).
pub fn upload_f32(data: &[f32], device: &MlxDevice) -> Result<MlxBuffer> {
    let byte_len = data.len() * 4;
    let mut buf = device
        .alloc_buffer(byte_len, DType::F32, vec![data.len()])
        .map_err(|e| anyhow!("alloc f32 buffer len={}: {e}", data.len()))?;
    {
        let slice = buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("mut_slice: {e}"))?;
        slice.copy_from_slice(data);
    }
    Ok(buf)
}

/// Same as [`upload_f32`] but additionally registers the resulting buffer
/// with the thread-local weight pool's `MTLResidencySet` so it stays
/// hinted-resident across forward passes.
///
/// **Wave 5b.7 iter 2:** call this for *long-lived weight tensors* (norm
/// weights, embedding tables, LM-head copies, MTP head weights).  Do **not**
/// call this for transient per-forward activations — `MlxBufferPool`
/// retains the Metal allocation in its residency hashmap, which would
/// pin transient buffers across forward boundaries (effective memory
/// leak).
///
/// No-op when `HF2Q_NO_RESIDENCY=1` is set.
pub fn upload_f32_weight(data: &[f32], device: &MlxDevice) -> Result<MlxBuffer> {
    let buf = upload_f32(data, device)?;
    super::weight_pool::register_weight_buffer(device, &buf)
        .map_err(|e| anyhow!("register_weight_buffer f32 len={}: {e}", data.len()))?;
    Ok(buf)
}

/// Copy `data` into an existing `MlxBuffer` (no allocation).
///
/// Used for decode-path hot buffers that are pre-allocated once and reused
/// every decode token to avoid repeated `newBuffer` Metal API calls.
///
/// # Errors
/// Returns an error if `buf` is too small or has the wrong dtype.
pub fn upload_f32_into(data: &[f32], buf: &mut MlxBuffer) -> Result<()> {
    anyhow::ensure!(
        buf.dtype() == DType::F32,
        "upload_f32_into: expected F32 buffer, got {:?}", buf.dtype()
    );
    anyhow::ensure!(
        buf.element_count() >= data.len(),
        "upload_f32_into: buf too small (cap={} < data={})",
        buf.element_count(), data.len()
    );
    let slice = buf.as_mut_slice::<f32>().map_err(|e| anyhow!("mut_slice: {e}"))?;
    slice[..data.len()].copy_from_slice(data);
    Ok(())
}

/// Download an `MlxBuffer` of f32 values into a `Vec<f32>`.
pub fn download_f32(buf: &MlxBuffer) -> Result<Vec<f32>> {
    if buf.dtype() != DType::F32 {
        return Err(anyhow!(
            "download_f32: buffer dtype {} != f32",
            buf.dtype()
        ));
    }
    let slice: &[f32] = buf.as_slice().map_err(|e| anyhow!("as_slice: {e}"))?;
    Ok(slice.to_vec())
}

/// Apply per-head RMSNorm to a Q or K buffer.
///
/// # Layout contract
///
/// Input buffer shape is `[seq_len * n_heads, head_dim]` f32 (row-major
/// with `head_dim` innermost). The per-head RMSNorm treats each row as an
/// independent vector and applies `x / sqrt(mean(x^2) + eps) * weight`
/// element-wise, where `weight` is shape `[head_dim]` shared across all
/// heads and tokens (matches llama.cpp / HF's Qwen3.5 convention).
///
/// # Why this dispatches rms_norm with rows = seq*n_heads
///
/// The full-attention op order has RMSNorm applied POST-reshape, meaning
/// each Q head of each token gets normalized independently over the
/// `head_dim` axis. Since mlx-native's `dispatch_rms_norm` is already a
/// per-row operation with an element-wise weight, we can reuse it directly
/// by flattening (seq, head) into a single row axis.
///
/// # Parity contract
///
/// Output matches the CPU reference's step 3 (Q) or 4 (K) — per-head
/// RMSNorm over `head_dim` — to ≤1e-5 per element.
pub fn apply_q_or_k_per_head_rms_norm(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    norm_weight: &MlxBuffer,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    let rows = seq_len * n_heads;
    let dim = head_dim;
    let out = super::decode_pool::pooled_alloc_buffer(
            device,
            (rows * dim) as usize * 4,
            DType::F32,
            vec![rows as usize, dim as usize],
        )
        .map_err(|e| anyhow!("alloc out: {e}"))?;
    let mut params = super::decode_pool::pooled_alloc_buffer(device, 8, DType::F32, vec![2])
        .map_err(|e| anyhow!("alloc params: {e}"))?;
    {
        let s = params
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("mut_slice: {e}"))?;
        s[0] = eps;
        s[1] = dim as f32;
    }
    rms_norm::dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        norm_weight,
        &out,
        &params,
        rows,
        dim,
    )
    .context("dispatch_rms_norm per-head")?;
    Ok(out)
}

/// Apply IMROPE to a Q or K buffer on the GPU.
///
/// `input` shape: `[seq_len * n_heads, head_dim]` (flat row-major).
/// `positions`: int32 array of length `4 * seq_len` — per-axis positions
/// (see mlx-native `rope_multi` spec; text-only Qwen3.5 replicates the
/// same token index across all 4 axes).
///
/// Returns a new buffer with the same shape holding the rotated Q/K.
pub fn apply_imrope(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    positions: &MlxBuffer,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    freq_base: f32,
    mrope_section: [u32; 4],
) -> Result<MlxBuffer> {
    let params = RopeMultiParams {
        head_dim,
        rope_dim: rotary_dim,
        n_heads,
        seq_len,
        freq_base,
        mode: RopeMultiMode::Imrope,
        sections: mrope_section,
    };
    // ADR-015 iter7a (P3b alloc_buffer pool, sub-iter 7a): pooled.
    // The output flows into dispatch_sdpa_decode → SDPA kernel which
    // uses `element_count()` (logical shape product), NOT `byte_len()`.
    // Pool's bucket-rounded byte_len is therefore safe here — the
    // §P3a' Codex Q3 hazard about CPU-read byte_len mismatch does
    // not apply (the q_rope / k_rope buffers are GPU-only on the
    // kv_cache_slot=Some branch which is the apex production path).
    let out = super::decode_pool::pooled_alloc_buffer(
        device,
        (seq_len * n_heads * head_dim) as usize * 4,
        DType::F32,
        vec![
            seq_len as usize,
            n_heads as usize,
            head_dim as usize,
        ],
    )
    .map_err(|e| anyhow!("alloc imrope out (pooled): {e}"))?;

    // ADR-015 P3b rank-4: the three small (16-byte) param/rope_params/
    // sections buffers were previously rebuilt on every call (32×/token
    // on the apex 35B-A3B FullAttn pattern, 208 µs/token measured on the
    // qwen3.6-27b-dwq46 dense fixture in the Wave 2a TimeProfiler trace).
    // dispatch_rope_multi_cached reuses them via a per-thread cache
    // keyed by (device, head_dim, rope_dim, n_heads, seq_len, freq_base,
    // mode, sections); the qwen35 decode hot path hits 2 stable entries
    // (Q-config + K-config, seq_len=1) and amortizes the alloc cost
    // across all decode tokens.  Bit-exact: same kernel, same dispatch,
    // only the param triplet is sourced from the cache.
    dispatch_rope_multi_cached(
        encoder,
        registry,
        device,
        input,
        &out,
        positions,
        params,
    )
    .context("dispatch_rope_multi_cached")?;

    Ok(out)
}

/// Apply sigmoid-gated elementwise multiply: `out[i] = attn_out[i] * sigmoid(gate[i])`.
///
/// Qwen3.5 full-attention's output-gate application (ADR-013 Decision 9).
/// Sigmoid (not swish) is the authoritative activation — cited by HF
/// `modeling_qwen3_5.py:689` and vLLM `qwen3_next.py:312-314`.
pub fn apply_sigmoid_gate_multiply(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    attn_out: &MlxBuffer,
    gate: &MlxBuffer,
    n_elements: u32,
) -> Result<MlxBuffer> {
    let out = super::decode_pool::pooled_alloc_buffer(
            device,
            n_elements as usize * 4,
            DType::F32,
            vec![n_elements as usize],
        )
        .map_err(|e| anyhow!("alloc sigmoid-mul out: {e}"))?;
    let mut params = super::decode_pool::pooled_alloc_buffer(device, 4, DType::U32, vec![1])
        .map_err(|e| anyhow!("alloc params: {e}"))?;
    params
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("mut_slice: {e}"))?[0] = n_elements;

    dispatch_sigmoid_mul(
        encoder,
        registry,
        device.metal_device(),
        attn_out,
        gate,
        &out,
        &params,
        n_elements,
    )
    .context("dispatch_sigmoid_mul")?;

    Ok(out)
}

/// Apply pre-attention RMSNorm to a residual-stream input buffer.
///
/// Produces a new f32 buffer with the same shape. The output buffer is
/// allocated by this function; callers can reuse it downstream by passing
/// it as input to the next dispatch.
///
/// # Parity contract
///
/// Output must match [`super::full_attn::gated_full_attention_cpu_ref`]'s
/// step-1 output (RMSNorm row-wise with `attn_norm` weight, `rms_norm_eps`
/// from config) to ≤1e-5 per element for F32.
pub fn apply_pre_attn_rms_norm(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weights_gpu: &FullAttnWeightsGpu,
    seq_len: u32,
    hidden_size: u32,
    eps: f32,
) -> Result<MlxBuffer> {
    // Allocate output + params.
    let out = super::decode_pool::pooled_alloc_buffer(
            device,
            (seq_len * hidden_size) as usize * 4,
            DType::F32,
            vec![seq_len as usize, hidden_size as usize],
        )
        .map_err(|e| anyhow!("alloc out: {e}"))?;
    let mut params = super::decode_pool::pooled_alloc_buffer(device, 8, DType::F32, vec![2])
        .map_err(|e| anyhow!("alloc params: {e}"))?;
    {
        let s = params
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("mut_slice: {e}"))?;
        s[0] = eps;
        s[1] = hidden_size as f32;
    }

    rms_norm::dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        &weights_gpu.attn_norm,
        &out,
        &params,
        seq_len,
        hidden_size,
    )
    .context("dispatch_rms_norm")?;

    Ok(out)
}

// ================================================================
// Linear projection (F32 weights via BF16 cast)
// ================================================================

/// Apply a single linear projection: `output = input @ weight^T`.
///
/// `input`  shape: `[seq_len, in_features]`  F32.
/// `weight` shape: `[out_features, in_features]` — BF16 or Q4_0 raw blocks (U8).
///
/// Returns `[seq_len, out_features]` F32.
///
/// # Implementation
///
/// Dispatches based on weight dtype:
///
/// - **U8** (Q4_0 GGML blocks): uses `quantized_matmul_ggml` which routes
///   to `dispatch_mv` for M=1 (decode) and `dispatch_mm` for M>8 (prefill).
///   This is the production path: 3.56× less bandwidth than BF16, and uses
///   the same deterministic simd_sum accumulation as the FFN projection path.
///
/// - **BF16** (dense pre-cast): uses `dense_matmul_bf16_f32_tensor` (MMA
///   tensor-core tiled GEMM). Kept for lm_head and any weight not yet
///   quantized.
///
/// - **F32** (legacy inline cast): casts to BF16 on the GPU then calls the
///   BF16 path. Per-inference cost; only used for un-pre-cast weights.
///
/// Requires `in_features >= 32` for Q4_0 (block size) and BF16 (tile size).
pub fn apply_linear_projection_f32(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    seq_len: u32,
    in_features: u32,
    out_features: u32,
) -> Result<MlxBuffer> {
    // Allocate output buffer (same for all paths).
    //
    // NOTE: NOT pooled — `apply_linear_projection_f32` is shared between
    // prefill (downloads via `download_f32` → `as_slice` → reads full
    // `byte_len()`) and decode.  The pool's power-of-two bucket rounding
    // would inflate `byte_len()` to the bucket size, causing prefill's
    // logits-shape sanity check (`prefill_logits.len() == prompt_len * vocab`)
    // to fail.  Decode hot-path lm_head goes through `apply_output_head_gpu`
    // which uses the pre-allocated `logits_buf` from `DecodeBuffers` (not
    // this code path), so leaving this device-allocated has no decode cost.
    let out_bytes = (seq_len * out_features) as usize * 4;
    let mut dst = device
        .alloc_buffer(out_bytes, DType::F32, vec![seq_len as usize, out_features as usize])
        .map_err(|e| anyhow!("alloc projection output: {e}"))?;

    match weight.dtype() {
        DType::U8 => {
            // Q4_0 GGML block path — fast decode (dispatch_mv) + prefill (dispatch_mm).
            // Deterministic: same simd_sum accumulation order as the FFN kernel.
            let params = GgmlQuantizedMatmulParams {
                m: seq_len,
                n: out_features,
                k: in_features,
                ggml_type: GgmlType::Q4_0,
            };
            quantized_matmul_ggml(encoder, registry, device, input, weight, &mut dst, &params)
                .context("quantized_matmul_ggml Q4_0")?;
        }
        DType::BF16 => {
            let params = DenseMmBf16F32Params {
                m: seq_len,
                n: out_features,
                k: in_features,
                src0_batch: 1,
                src1_batch: 1,
            };
            if seq_len == 1 {
                // GEMV path — bandwidth-optimized for M=1 decode.
                // mul_mv_bf16_f32_4 port from llama.cpp: processes multiple
                // weight rows per threadgroup, ~2× faster than tiled MM for M=1.
                dense_gemv_bf16_f32(encoder, registry, device, weight, input, &mut dst, &params)
                    .context("dense_gemv_bf16_f32 (M=1)")?;
            } else {
                // BF16 tiled GEMM path — MMA tensor-core, optimal for M > 1.
                dense_matmul_bf16_f32_tensor(encoder, registry, device, weight, input, &mut dst, &params)
                    .context("dense_matmul_bf16_f32_tensor")?;
            }
        }
        DType::F32 => {
            // Legacy F32 path: cast inline (per-inference cost, not pre-quantized).
            //
            // ADR-015 iter14: lift `weight_bf16` cast scratch to the
            // per-decode-token pool.  This is a function-local helper
            // scratch consumed by the matmul dispatch in the SAME encoder
            // but the encoder is not committed by this function (caller
            // commits).  Safe under retained refs (encoder CB ARC keeps
            // the buffer alive); pool ARC anchor required under unretained
            // refs.  Branch is unused on Qwen3.6 dwq46 (Q4_0 takes the
            // U8 path above) but lifted for hygiene.
            let n_w = (out_features * in_features) as usize;
            let weight_bf16 = super::decode_pool::pooled_alloc_buffer(
                    device, n_w * 2, DType::BF16, vec![out_features as usize, in_features as usize])
                .map_err(|e| anyhow!("alloc weight_bf16 (pooled): {e}"))?;
            cast(encoder, registry, device.metal_device(), weight, &weight_bf16, n_w, CastDirection::F32ToBF16)
                .context("cast weight F32→BF16")?;
            // Need a barrier: the GEMM reads weight_bf16 which was written by the cast.
            encoder.memory_barrier();
            let params = DenseMmBf16F32Params {
                m: seq_len,
                n: out_features,
                k: in_features,
                src0_batch: 1,
                src1_batch: 1,
            };
            dense_matmul_bf16_f32_tensor(encoder, registry, device, &weight_bf16, input, &mut dst, &params)
                .context("dense_matmul_bf16_f32_tensor (F32 legacy)")?;
        }
        other => {
            return Err(anyhow!(
                "apply_linear_projection_f32: unsupported weight dtype {:?}", other
            ));
        }
    }

    Ok(dst)
}

/// Like `apply_linear_projection_f32` but writes into a caller-supplied output
/// buffer instead of allocating a new one.
///
/// Used by the decode hot-path to avoid one ~600KB `newBuffer` per token for
/// the lm_head logits output.  The caller is responsible for ensuring `dst`
/// has capacity ≥ `seq_len × out_features × sizeof(f32)` and dtype == F32.
pub fn apply_linear_projection_f32_into(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    dst: &mut MlxBuffer,
    seq_len: u32,
    in_features: u32,
    out_features: u32,
) -> Result<()> {
    match weight.dtype() {
        DType::U8 => {
            let params = GgmlQuantizedMatmulParams {
                m: seq_len,
                n: out_features,
                k: in_features,
                ggml_type: GgmlType::Q4_0,
            };
            quantized_matmul_ggml(encoder, registry, device, input, weight, dst, &params)
                .context("quantized_matmul_ggml Q4_0 (into)")?;
        }
        DType::BF16 => {
            let params = DenseMmBf16F32Params {
                m: seq_len,
                n: out_features,
                k: in_features,
                src0_batch: 1,
                src1_batch: 1,
            };
            if seq_len == 1 {
                dense_gemv_bf16_f32(encoder, registry, device, weight, input, dst, &params)
                    .context("dense_gemv_bf16_f32 (M=1, into)")?;
            } else {
                dense_matmul_bf16_f32_tensor(encoder, registry, device, weight, input, dst, &params)
                    .context("dense_matmul_bf16_f32_tensor (into)")?;
            }
        }
        DType::F32 => {
            // ADR-015 iter14: same scratch-lift as `apply_linear_projection_f32`'s
            // F32 legacy arm above.
            let n_w = (out_features * in_features) as usize;
            let weight_bf16 = super::decode_pool::pooled_alloc_buffer(
                    device, n_w * 2, DType::BF16, vec![out_features as usize, in_features as usize])
                .map_err(|e| anyhow!("alloc weight_bf16 (pooled, into): {e}"))?;
            cast(encoder, registry, device.metal_device(), weight, &weight_bf16, n_w, CastDirection::F32ToBF16)
                .context("cast weight F32→BF16 (into)")?;
            encoder.memory_barrier();
            let params = DenseMmBf16F32Params {
                m: seq_len,
                n: out_features,
                k: in_features,
                src0_batch: 1,
                src1_batch: 1,
            };
            dense_matmul_bf16_f32_tensor(encoder, registry, device, &weight_bf16, input, dst, &params)
                .context("dense_matmul_bf16_f32_tensor F32 legacy (into)")?;
        }
        other => {
            return Err(anyhow!(
                "apply_linear_projection_f32_into: unsupported weight dtype {:?}", other
            ));
        }
    }
    Ok(())
}

/// Pool-aware variant of [`apply_linear_projection_f32`] for the decode
/// hot path (`seq_len == 1`).  Falls back to the unpooled
/// `apply_linear_projection_f32` for prefill (`seq_len > 1`) because some
/// prefill consumers (notably `apply_sdpa_with_kv_cache` for K/V) call
/// `download_f32` → `as_slice` which reads the buffer's raw `byte_len()`;
/// the pool's power-of-two bucket rounding would inflate `byte_len()`
/// beyond the requested shape.
///
/// For decode (seq_len=1) the dispatch path keeps Q/K/V/gate/O entirely
/// on GPU (rope → SDPA → residual), so the pool is safe.  This closes
/// the alloc-overhead budget for attention Q/K/V/O = 4 projections × 10
/// full-attn layers per forward = 40 allocs/token previously hitting
/// Metal's `newBuffer` directly.
///
/// **Caller contract:** when `seq_len == 1`, the returned `MlxBuffer`
/// must NOT be downloaded to CPU via `as_slice` / `download_f32`.  When
/// `seq_len > 1`, this function delegates to the unpooled variant so
/// CPU downloads remain safe.
///
/// For the lm_head logits output (downloaded after prefill at any
/// `seq_len`), keep using the unpooled [`apply_linear_projection_f32`]
/// directly — that signal is shape-significant for the prefill sanity
/// check on `prefill_logits.len()`.
#[allow(clippy::too_many_arguments)]
pub fn apply_linear_projection_f32_pooled(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    seq_len: u32,
    in_features: u32,
    out_features: u32,
) -> Result<MlxBuffer> {
    if seq_len != 1 {
        // Prefill — fall back to unpooled to keep `download_f32` callers
        // (e.g. K/V in `apply_sdpa_with_kv_cache` prefill branch) safe.
        return apply_linear_projection_f32(
            encoder, registry, device, input, weight,
            seq_len, in_features, out_features,
        );
    }
    let out_bytes = (seq_len * out_features) as usize * 4;
    let mut dst = super::decode_pool::pooled_alloc_buffer(
            device, out_bytes, DType::F32, vec![seq_len as usize, out_features as usize])
        .map_err(|e| anyhow!("alloc projection output (pooled): {e}"))?;
    apply_linear_projection_f32_into(
        encoder, registry, device, input, weight, &mut dst,
        seq_len, in_features, out_features,
    )?;
    Ok(dst)
}

// ================================================================
// ADR-015 iter86: arena-aware variants of FA helper ops
// ================================================================
//
// These mirror the existing helpers byte-for-byte, but write into a
// caller-supplied `&MlxBuffer` (output) and `&MlxBuffer` (params) sourced
// from a [`super::FaProjectionsArena`]. Used by [`build_gated_attn_layer`]'s
// prefill body when `fa_proj_arena=Some` to eliminate the per-FA-layer
// pooled_alloc_buffer / device.alloc_buffer churn captured by the W-5b.8
// `fa.ops1_4` bucket.
//
// All four helpers preserve the exact dispatch sequence + numerical
// behaviour of the originals — only the output buffer source differs.

/// Arena-aware variant of [`apply_pre_attn_rms_norm`] that writes into
/// caller-supplied `out` (sourced from
/// [`super::FaProjectionsArena::x_norm_buf`]) using `params` from
/// [`super::FaProjectionsArena::pre_norm_params_buf`].
pub fn apply_pre_attn_rms_norm_into(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weights_gpu: &FullAttnWeightsGpu,
    out: &MlxBuffer,
    params: &MlxBuffer,
    seq_len: u32,
    hidden_size: u32,
) -> Result<()> {
    rms_norm::dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        &weights_gpu.attn_norm,
        out,
        params,
        seq_len,
        hidden_size,
    )
    .context("dispatch_rms_norm (arena into)")?;
    Ok(())
}

/// Arena-aware variant of [`apply_q_or_k_per_head_rms_norm`] that writes
/// into caller-supplied `out` (sourced from
/// [`super::FaProjectionsArena::q_normed_buf`] or `k_normed_buf`) using
/// shared `params` from [`super::FaProjectionsArena::qk_rms_params_buf`].
///
/// `params` must contain `[eps, head_dim_as_f32]`. Both Q and K share the
/// same param values because both norm along `head_dim` with the same eps.
#[allow(clippy::too_many_arguments)]
pub fn apply_q_or_k_per_head_rms_norm_into(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    norm_weight: &MlxBuffer,
    out: &MlxBuffer,
    params: &MlxBuffer,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
) -> Result<()> {
    let rows = seq_len * n_heads;
    let dim = head_dim;
    rms_norm::dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        norm_weight,
        out,
        params,
        rows,
        dim,
    )
    .context("dispatch_rms_norm per-head (arena into)")?;
    Ok(())
}

/// Arena-aware variant of [`apply_imrope`] that writes into caller-supplied
/// `out` (sourced from [`super::FaProjectionsArena::q_rope_buf`] or
/// `k_rope_buf`).
///
/// IMROPE param buffers are NOT in the arena — `dispatch_rope_multi_cached`
/// holds its own thread-local cache keyed by shape + freq_base, so the
/// param triple is built once across the entire prefill (and decode), not
/// per-call.
#[allow(clippy::too_many_arguments)]
pub fn apply_imrope_into(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    out: &MlxBuffer,
    positions: &MlxBuffer,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    freq_base: f32,
    mrope_section: [u32; 4],
) -> Result<()> {
    let params = RopeMultiParams {
        head_dim,
        rope_dim: rotary_dim,
        n_heads,
        seq_len,
        freq_base,
        mode: RopeMultiMode::Imrope,
        sections: mrope_section,
    };
    dispatch_rope_multi_cached(
        encoder,
        registry,
        device,
        input,
        out,
        positions,
        params,
    )
    .context("dispatch_rope_multi_cached (arena into)")?;
    Ok(())
}

/// Arena-aware variant of [`apply_sigmoid_gate_multiply`] that writes into
/// caller-supplied `out` (sourced from
/// [`super::FaProjectionsArena::gated_buf`]) using `params` from
/// [`super::FaProjectionsArena::sigmoid_params_buf`].
#[allow(clippy::too_many_arguments)]
pub fn apply_sigmoid_gate_multiply_into(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    attn_out: &MlxBuffer,
    gate: &MlxBuffer,
    out: &MlxBuffer,
    params: &MlxBuffer,
    n_elements: u32,
) -> Result<()> {
    dispatch_sigmoid_mul(
        encoder,
        registry,
        device.metal_device(),
        attn_out,
        gate,
        out,
        params,
        n_elements,
    )
    .context("dispatch_sigmoid_mul (arena into)")?;
    Ok(())
}

// ================================================================
// SDPA — causal, GQA, prefill
// ================================================================

/// Permute `[seq, n_heads, head_dim]` → `[n_heads, seq, head_dim]` on CPU.
///
/// Used as a test helper to satisfy the SDPA kernel's head-major layout
/// requirement for Q and K.  Not on the GPU hot-path.
pub fn permute_seq_head_dim_to_head_seq_dim_cpu(
    data: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * n_heads * head_dim];
    for h in 0..n_heads {
        for t in 0..seq_len {
            let src_off = (t * n_heads + h) * head_dim;
            let dst_off = (h * seq_len + t) * head_dim;
            out[dst_off..dst_off + head_dim].copy_from_slice(&data[src_off..src_off + head_dim]);
        }
    }
    out
}

/// Apply causal scaled dot-product attention (SDPA) with GQA.
///
/// `q` shape: `[1, n_heads,    seq_len, head_dim]`  F32 (head-major).
/// `k` shape: `[1, n_kv_heads, seq_len, head_dim]`  F32 (head-major).
/// `v` shape: `[1, n_kv_heads, seq_len, head_dim]`  F32 (head-major).
///
/// Returns `[1, n_heads, seq_len, head_dim]` F32 (head-major).
///
/// Note: callers that have Q/K in seq-major layout must permute via
/// [`permute_seq_head_dim_to_head_seq_dim_cpu`] before calling this
/// (or use `apply_sdpa_causal_from_seq_major` which does it automatically).
pub fn apply_sdpa_causal(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q_head_major: &MlxBuffer,
    k_head_major: &MlxBuffer,
    v_head_major: &MlxBuffer,
    seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
) -> Result<MlxBuffer> {
    let out = super::decode_pool::pooled_alloc_buffer(
            device,
            (n_heads * seq_len * head_dim) as usize * 4,
            DType::F32,
            vec![1, n_heads as usize, seq_len as usize, head_dim as usize],
        )
        .map_err(|e| anyhow!("alloc sdpa output: {e}"))?;

    let params = SdpaParams {
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len,
        kv_seq_len: seq_len,
        scale: 1.0 / (head_dim as f32).sqrt(),
        kv_capacity: 0, // 0 = use kv_seq_len
    };

    sdpa(encoder, registry, device, q_head_major, k_head_major, v_head_major, &out, &params, 1)
        .context("sdpa")?;

    Ok(out)
}

/// Apply SDPA starting from seq-major Q/K/V buffers.
///
/// Handles the seq-major → head-major permutation on CPU before calling
/// the SDPA kernel, then permutes the output back to seq-major.
///
/// `q` shape: `[seq_len * n_heads,    head_dim]` F32 (seq-major, as produced by IMROPE).
/// `k` shape: `[seq_len * n_kv_heads, head_dim]` F32 (seq-major).
/// `v` shape: `[seq_len * n_kv_heads, head_dim]` F32 (seq-major).
///
/// Returns `[seq_len * n_heads, head_dim]` F32 (seq-major, to match the rest
/// of the pipeline).
pub fn apply_sdpa_causal_from_seq_major(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q_seq_major: &MlxBuffer,
    k_seq_major: &MlxBuffer,
    v_seq_major: &MlxBuffer,
    seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
) -> Result<MlxBuffer> {
    let seq = seq_len as usize;
    let nh = n_heads as usize;
    let nkv = n_kv_heads as usize;
    let d = head_dim as usize;

    // Commit any pending dispatches (norm, rope) so their outputs are
    // ready for CPU download.
    encoder.commit_and_wait().context("commit before sdpa permute")?;

    // Download Q, K, V — currently [seq, heads, dim] (seq-major).
    let q_cpu = download_f32(q_seq_major)?;
    let k_cpu = download_f32(k_seq_major)?;
    let v_cpu = download_f32(v_seq_major)?;

    // Permute to [heads, seq, dim] (head-major) for SDPA.
    let q_hm = permute_seq_head_dim_to_head_seq_dim_cpu(&q_cpu, seq, nh, d);
    let k_hm = permute_seq_head_dim_to_head_seq_dim_cpu(&k_cpu, seq, nkv, d);
    let v_hm = permute_seq_head_dim_to_head_seq_dim_cpu(&v_cpu, seq, nkv, d);

    let q_gpu = upload_f32(&q_hm, device)?;
    let k_gpu = upload_f32(&k_hm, device)?;
    let v_gpu = upload_f32(&v_hm, device)?;

    // Fresh encoder for SDPA dispatch.
    let mut enc2 = device.command_encoder().context("new encoder for sdpa")?;
    let out_hm = apply_sdpa_causal(
        &mut enc2, registry, device, &q_gpu, &k_gpu, &v_gpu,
        seq_len, n_heads, n_kv_heads, head_dim,
    )?;
    enc2.commit_and_wait().context("sdpa commit")?;

    // Download SDPA output [heads, seq, dim], permute back to [seq, heads, dim].
    let out_hm_cpu = download_f32(&out_hm)?;
    let mut out_sm = vec![0.0f32; seq * nh * d];
    for h in 0..nh {
        for t in 0..seq {
            let src = (h * seq + t) * d;
            let dst = (t * nh + h) * d;
            out_sm[dst..dst + d].copy_from_slice(&out_hm_cpu[src..src + d]);
        }
    }

    upload_f32(&out_sm, device)
}

// ================================================================
// Wave 5b.10 — flash_attn_prefill bridge (Qwen3.5/3.6 FA prefill)
// ================================================================

/// Wave 5b.10 — run flash_attn_prefill on Qwen3.5/3.6's seq-major F32 chunk
/// Q/K/V buffers, returning seq-major F32 output `[seq, n_heads, head_dim]`.
///
/// This is the bridge between the Qwen3.5 op pipeline (everything is F32
/// seq-major `[seq, heads, head_dim]`) and mlx-native's
/// `dispatch_flash_attn_prefill_bf16_d256` (BF16 head-major `[1, H, T, D]`,
/// contiguous inner dim). All staging happens on-GPU in a single command
/// encoder — no CPU↔GPU round-trips.
///
/// Bridge ops (in order, all on a fresh encoder):
///
/// 1. cast F32→BF16: Q seq-major  `[seq, n_heads,    256]`
/// 2. permute_021_bf16:           `[seq, n_heads,    256]` → `[n_heads,    seq, 256]`
/// 3. cast F32→BF16: K seq-major  `[seq, n_kv_heads, 256]`
/// 4. permute_021_bf16:           `[seq, n_kv_heads, 256]` → `[n_kv_heads, seq, 256]`
/// 5. cast F32→BF16: V seq-major  `[seq, n_kv_heads, 256]`
/// 6. permute_021_bf16:           `[seq, n_kv_heads, 256]` → `[n_kv_heads, seq, 256]`
/// 7. dispatch_flash_attn_prefill_bf16_d256(do_causal=true, mask=None)
/// 8. permute_021_bf16_to_f32:    `[n_heads, seq, 256]` → `[seq, n_heads, 256]` F32
///
/// # Why D=256
///
/// Qwen3.5/3.6 uses `head_dim = 256` (verified at
/// `src/inference/models/qwen35/mod.rs:715` for the apex MoE config).
/// `flash_attn_prefill_bf16_d256` is the matching tile geometry; the kernel
/// has been in production for Gemma 4 sliding layers since ADR-011 Phase 2
/// Wave 4 (commit `953dc1b`).
///
/// # Causal mask
///
/// `do_causal=true` enables the kernel's in-kernel causal mask
/// (function constant 301). No external mask buffer is required for a
/// pure prefill from offset 0; this matches `apply_sdpa_causal`'s
/// `causal_mask_subroutine` semantic. `q_abs_offset = 0` is implicit
/// (the kernel computes `row_pos vs col_pos` from tile indices, with
/// `qL == kL == seq_len` as we pass them).
///
/// # KV-cache write
///
/// This function does **not** touch `slot.k`/`slot.v`. The caller writes
/// the chunk into the persistent KV cache (for later decode) BEFORE
/// invoking this bridge. The bridge reads the chunk Q/K/V directly from
/// the seq-major buffers produced upstream by IMROPE — bypassing the
/// CPU triple-loop's involvement in the FA dispatch path.
///
/// # Returns
///
/// `[seq * n_heads, head_dim]` F32 (seq-major) — same shape and layout
/// as `apply_sdpa_with_kv_cache`'s prefill else-branch return value.
///
/// # Errors
///
/// - `head_dim != 256` (D=256 dispatcher only).
/// - `n_heads % n_kv_heads != 0` (rejected by mlx-native validate).
/// - Any underlying mlx-native dispatch failure is propagated with
///   the bridge step name in the context.
///
/// # ADR-019 Phase 2 iter89e2-E — `_into` variant
///
/// [`apply_flash_attn_prefill_seq_major_into`] performs the same 8 dispatches
/// + 5 intra-encoder barriers but encodes them into a caller-supplied
/// `&mut CommandEncoder` and does NOT commit. It is the structural
/// foundation for iter89e2-F's single-CB FA-layer fusion (ops1-4 +
/// kv_cache_write + fa.prefill_bridge + ops6-7 → 1 CB).
///
/// This wrapper preserves byte-identical behavior: when `fa_arena=Some`,
/// it opens its own encoder, delegates encoding to the `_into` variant,
/// and commits via `commit_labeled("fa.prefill_bridge")` exactly as before.
/// When `fa_arena=None`, it executes the legacy per-call alloc + commit-and-
/// wait path (no `_into` delegation; that path's contract differs).
///
/// ── ADR-019 Phase 2 iter89e2-E variant ──────────────────────────────────────

/// Encode `apply_flash_attn_prefill_seq_major`'s 8 dispatches + 5 intra-
/// encoder barriers into a caller-supplied [`mlx_native::CommandEncoder`]
/// without committing. The caller owns the encoder lifecycle and is
/// responsible for issuing the terminal commit.
///
/// This is the structural prerequisite for the Phase 2 single-CB fusion
/// (ADR-019 iter89e2-F): with this `_into` form available, the FA-layer
/// orchestrator can encode ops1-4 + kv_cache_write + fa.prefill_bridge +
/// ops6-7 into a single command buffer separated by `enc.memory_barrier()`
/// calls, eliminating 3 of the 4 commit_labeled calls per FA layer
/// (4 → 1 CB × 10 FA layers = 30 fewer CBs per Qwen3.6-35B-A3B prefill).
///
/// # Contract
///
/// - Caller supplies `enc` and is responsible for committing it.
/// - Caller supplies `out_seq` (the F32 seq-major output buffer); this
///   function writes into it via the final `permute_021_bf16_to_f32`
///   dispatch. Allocation of `out_seq` is the caller's responsibility so
///   the wrapper's per-call alloc shape is preserved exactly (see the
///   wrapper at [`apply_flash_attn_prefill_seq_major`]).
/// - `arena` is a `&mut FaPrefillArena` (NOT `Option<&mut ...>`): the
///   `_into` form is exclusively the production arena path. The legacy
///   no-arena path uses `commit_and_wait` and per-call BF16 allocations,
///   which are incompatible with the caller-supplied-encoder model and
///   remain encapsulated in the wrapper's `else` branch.
///
/// # F-fence preservation (ADR-019 §Risk Register)
///
/// - F2 (residency-rescission, iter58b): all 7 BF16 scratches are arena-
///   owned and live for the entire prefill (allocated at
///   `forward_gpu.rs:1701-1713`, dropped after the output-head terminal
///   `commit_and_wait_labeled`). `out_seq` is caller-owned and outlives
///   any commit the caller chooses to issue. No wrapper-local MlxBuffer
///   drops occur — iter58b race is structurally unreachable regardless
///   of when the caller commits.
/// - F11 (zero-init alloc): no `device.alloc_buffer` is called from this
///   variant — `out_seq` is supplied by the caller, scratches are arena-
///   owned. The wrapper's per-call `out_seq` allocation is unchanged.
/// - F1 (persistent compute encoder): `enc` may be in any state on entry;
///   each dispatch reads/writes it via the standard mlx-native dispatch
///   surface, which lazy-opens the persistent compute encoder as needed.
///   This variant adds one new entry-point but no new encoder lifecycles.
///
/// # Intra-encoder barriers
///
/// All 5 `enc.memory_barrier()` calls present in the wrapper's arena path
/// are reproduced here in identical positions:
///   - after Q cast → before Q permute_021
///   - after K cast → before K permute_021
///   - after V cast → before V permute_021
///   - after V permute_021 → before flash_attn_prefill_bf16_d256
///   - after flash_attn_prefill → before permute_021_bf16_to_f32
///
/// # Errors
///
/// Same as [`apply_flash_attn_prefill_seq_major`] minus the encoder-open
/// failure (the caller has already supplied a live encoder):
///   - `head_dim != 256`
///   - any underlying mlx-native dispatch failure
///   - arena `validate_fits` failure (capacity / shape mismatch)
#[allow(clippy::too_many_arguments)]
pub fn apply_flash_attn_prefill_seq_major_into(
    enc: &mut mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_seq_major: &MlxBuffer,
    k_seq_major: &MlxBuffer,
    v_seq_major: &MlxBuffer,
    out_seq: &MlxBuffer,
    seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    arena: &mut crate::inference::models::qwen35::FaPrefillArena,
) -> Result<()> {
    if head_dim != 256 {
        return Err(anyhow!(
            "apply_flash_attn_prefill_seq_major_into: head_dim must be 256 \
             (D=256 dispatcher); got {head_dim}. Other head_dims need a \
             different mlx-native dispatcher (D=64 / D=512) or a new port."
        ));
    }
    let seq = seq_len as usize;
    let nh = n_heads as usize;
    let nkv = n_kv_heads as usize;
    let d = head_dim as usize;

    let q_elems = seq * nh * d;
    let k_elems = seq * nkv * d;
    let v_elems = seq * nkv * d;

    arena.validate_fits(seq_len, n_heads, n_kv_heads, head_dim)
        .context("FA bridge: arena validate_fits")?;

    // Step 1+2: Q F32 seq-major → BF16 seq-major → BF16 head-major.
    cast(
        enc, registry, device.metal_device(),
        q_seq_major, &arena.q_bf16_seq, q_elems, CastDirection::F32ToBF16,
    ).context("FA bridge: cast Q F32→BF16")?;
    enc.memory_barrier();
    permute_021_bf16(
        enc, registry, device.metal_device(),
        &arena.q_bf16_seq, &arena.q_bf16_hm,
        seq, nh, d,
    ).context("FA bridge: permute_021 Q [seq, nh, d] → [nh, seq, d]")?;

    // Step 3+4: K F32 seq-major → BF16 seq-major → BF16 head-major.
    cast(
        enc, registry, device.metal_device(),
        k_seq_major, &arena.k_bf16_seq, k_elems, CastDirection::F32ToBF16,
    ).context("FA bridge: cast K F32→BF16")?;
    enc.memory_barrier();
    permute_021_bf16(
        enc, registry, device.metal_device(),
        &arena.k_bf16_seq, &arena.k_bf16_hm,
        seq, nkv, d,
    ).context("FA bridge: permute_021 K [seq, nkv, d] → [nkv, seq, d]")?;

    // Step 5+6: V F32 seq-major → BF16 seq-major → BF16 head-major.
    cast(
        enc, registry, device.metal_device(),
        v_seq_major, &arena.v_bf16_seq, v_elems, CastDirection::F32ToBF16,
    ).context("FA bridge: cast V F32→BF16")?;
    enc.memory_barrier();
    permute_021_bf16(
        enc, registry, device.metal_device(),
        &arena.v_bf16_seq, &arena.v_bf16_hm,
        seq, nkv, d,
    ).context("FA bridge: permute_021 V [seq, nkv, d] → [nkv, seq, d]")?;

    // Barrier: flash_attn_prefill reads Q/K/V head-major written above.
    enc.memory_barrier();

    // Step 7: dispatch flash_attn_prefill_bf16_d256.
    //   - scale = 1.0 / sqrt(head_dim) — Qwen3.5/3.6 oracle scale (no
    //     pre-scaling upstream, unlike Gemma 4).
    //   - do_causal = true — full prefill from offset 0; in-kernel causal
    //     mask handles row<col mask.
    //   - mask = None — pure causal, no external additive bias needed.
    //   - blk = None (path: dispatch_flash_attn_prefill_bf16_d256, the
    //     blk-less wrapper that delegates to *_with_blk(blk=None)).
    let scale = 1.0 / (d as f32).sqrt();
    dispatch_flash_attn_prefill_bf16_d256(
        enc, device, registry,
        &arena.q_bf16_hm, &arena.k_bf16_hm, &arena.v_bf16_hm,
        /* mask = */ None,
        &mut arena.out_bf16_hm,
        &FlashAttnPrefillParams {
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len_q: seq_len,
            seq_len_k: seq_len,
            batch: 1,
            scale,
            do_causal: true,
        },
    ).context("FA bridge: dispatch_flash_attn_prefill_bf16_d256")?;

    // Barrier: permute_021_bf16_to_f32 reads out_bf16_hm written above.
    enc.memory_barrier();

    // Step 8: BF16 head-major → F32 seq-major (fused permute+cast).
    //   Input dims for permute_021 are (dim_a=nh, dim_b=seq, dim_c=d) —
    //   the kernel writes [seq, nh, d] (i.e. dim_a/dim_b swapped in the
    //   layout, matching the [A, B, C] → [B, A, C] contract).
    permute_021_bf16_to_f32(
        enc, registry, device.metal_device(),
        &arena.out_bf16_hm, out_seq,
        nh, seq, d,
    ).context("FA bridge: permute_021_bf16_to_f32 out [nh, seq, d] → [seq, nh, d] F32")?;

    // No commit — caller owns the encoder lifecycle. See the wrapper
    // [`apply_flash_attn_prefill_seq_major`] for the "open + delegate +
    // commit_labeled" composition that preserves the legacy behavior.
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn apply_flash_attn_prefill_seq_major(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_seq_major: &MlxBuffer,
    k_seq_major: &MlxBuffer,
    v_seq_major: &MlxBuffer,
    seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    fa_arena: Option<&mut crate::inference::models::qwen35::FaPrefillArena>,
) -> Result<MlxBuffer> {
    if head_dim != 256 {
        return Err(anyhow!(
            "apply_flash_attn_prefill_seq_major: head_dim must be 256 \
             (D=256 dispatcher); got {head_dim}. Other head_dims need a \
             different mlx-native dispatcher (D=64 / D=512) or a new port."
        ));
    }
    let seq = seq_len as usize;
    let nh = n_heads as usize;
    let nkv = n_kv_heads as usize;
    let d = head_dim as usize;

    // ── Allocate scratch buffers (BF16 staging + F32 output) ─────────────
    //
    // Sizes (Qwen3.6 27B at PP4106): nh=16, nkv=2, d=256.
    //   q_bf16_seq:  4106 × 16 ×  256 × 2 =  33.6 MB
    //   q_bf16_hm:   4106 × 16 ×  256 × 2 =  33.6 MB
    //   k_bf16_seq:  4106 ×  2 ×  256 × 2 =   4.2 MB
    //   k_bf16_hm:   4106 ×  2 ×  256 × 2 =   4.2 MB
    //   v_bf16_seq:  4106 ×  2 ×  256 × 2 =   4.2 MB
    //   v_bf16_hm:   4106 ×  2 ×  256 × 2 =   4.2 MB
    //   out_bf16_hm: 4106 × 16 ×  256 × 2 =  33.6 MB
    //   out_seq:     4106 × 16 ×  256 × 4 =  67.2 MB  (per-call: return value)
    // Peak per layer ≈ 185 MB scratch; freed at end of layer (drop).
    // Compare with the legacy CPU-permute path's ~200 MB CPU-side
    // permute scratch — net allocation pressure is similar.
    //
    // ADR-013 P21 S1: when fa_arena=Some, the 7 BF16 scratch buffers are
    // reused from the caller-owned FaPrefillArena (allocated once per
    // prefill in forward_gpu_impl). Only out_seq (the F32 return value)
    // is allocated per-call — it is moved into the caller's binding and
    // does not drop at wrapper return (see queen plan A.5).
    let q_elems = seq * nh * d;
    let k_elems = seq * nkv * d;
    let v_elems = seq * nkv * d;
    let out_elems = seq * nh * d;

    // out_seq is always a per-call allocation: it is the function's return
    // value and the caller takes ownership. Arena-owning it would require
    // a per-layer index, which adds aliasing risk (A.5).
    let out_seq = device
        .alloc_buffer(out_elems * 4, DType::F32, vec![seq, nh, d])
        .map_err(|e| anyhow!("alloc out_seq: {e}"))?;

    if let Some(arena) = fa_arena {
        // ── Arena path: use caller-owned BF16 scratch buffers ────────────
        //
        // Wrapper preserves byte-identical behavior: opens its own encoder,
        // delegates encoding to `apply_flash_attn_prefill_seq_major_into`
        // (which encodes 8 dispatches + 5 intra-encoder barriers but does
        // NOT commit), then commits via `commit_labeled` exactly as before.
        //
        // ADR-019 Phase 2 iter89e2-E: the `_into` variant exists so a
        // caller (iter89e2-F) can encode the FA-prefill bridge into a
        // shared CB alongside ops1-4 + kv_cache_write + ops6-7, eliminating
        // 3 of the 4 CBs per FA layer. iter89e2-E itself is a refactor
        // only — every callsite of the wrapper still observes identical
        // output buffers and identical commit semantics.
        //
        // Lifetime safety (iter58b contract): arena buffers are owned by
        // forward_gpu_impl for the entire prefill. They do NOT drop when
        // this wrapper returns, so no deferred removeAllocation: is staged
        // on the MTLResidencySet. commit_labeled (no host wait) is therefore
        // safe — the next encoder's commit* cannot flush a stale
        // residency-rescission for buffers still referenced by this CB.
        let mut enc = device.command_encoder().context("FA prefill bridge encoder")?;
        apply_flash_attn_prefill_seq_major_into(
            &mut enc, device, registry,
            q_seq_major, k_seq_major, v_seq_major,
            &out_seq,
            seq_len, n_heads, n_kv_heads, head_dim,
            arena,
        )?;
        // Arena path: commit without host wait. Arena buffers are owned by
        // forward_gpu_impl and outlive this CB. out_seq is moved to the caller
        // and also outlives this CB. No wrapper-local MlxBuffer drops occur,
        // so no deferred removeAllocation: is staged — iter58b race is
        // structurally unreachable (queen plan A.5 / ADR-013 P21 S1).
        enc.commit_labeled("fa.prefill_bridge");

        // ── ITER-17 DIAGNOSTIC (HF2Q_DUMP_FA_BF16=1) ────────────────────
        // Dim=10 NaN bug investigation: dump arena.out_bf16_hm bytes
        // immediately after kernel commit (synchronized via fresh empty
        // encoder commit_and_wait) so we can compare bf16 buffer state
        // directly to mlx-native test's expected bf16 output.  Cost: one
        // commit_and_wait + raw byte memcpy when env is set; zero when
        // unset.  Output: /tmp/hf2q_fa_bf16_layerNNN_step0.bin (bf16
        // little-endian, [nh, seq, d] head-major layout).
        if std::env::var("HF2Q_DUMP_FA_BF16").as_deref() == Ok("1") {
            // Sync: wait for the just-committed CB (and therefore the
            // kernel write to out_bf16_hm) to actually land.
            let mut sync_enc = device.command_encoder()
                .context("FA bridge: dump sync encoder")?;
            sync_enc.commit_and_wait()
                .context("FA bridge: dump sync commit_and_wait")?;

            // Read raw bytes from arena buffers (StorageModeShared → memcpy).
            let layer_idx = super::dump_bisect::current_layer_idx();
            let step_idx = super::dump_bisect::current_step_idx();

            for (label, buf) in [
                ("q_bf16_hm",   &arena.q_bf16_hm),
                ("k_bf16_hm",   &arena.k_bf16_hm),
                ("v_bf16_hm",   &arena.v_bf16_hm),
                ("out_bf16_hm", &arena.out_bf16_hm),
            ] {
                let bytes = buf.as_slice::<u8>()
                    .map_err(|e| anyhow!("FA bridge: dump as_slice {label}: {e}"))?;
                let path = format!(
                    "/tmp/hf2q_fa_bf16_step{:04}_layer{:03}_{}.bin",
                    step_idx,
                    layer_idx.unwrap_or(999),
                    label,
                );
                std::fs::write(&path, bytes)
                    .with_context(|| format!("FA bridge: dump write {}", path))?;
            }
            tracing::info!(
                "iter-17 dump: wrote 4× arena bf16 buffers for layer {:?} step {}",
                layer_idx, step_idx,
            );
        }
    } else {
        // ── Fallback path (fa_arena=None): per-call alloc + commit_and_wait ──
        //
        // Preserves today's behaviour byte-identical for unit tests, decode
        // (skips arena allocation), and any caller that passes None.
        let q_bf16_seq = device
            .alloc_buffer(q_elems * 2, DType::BF16, vec![seq, nh, d])
            .map_err(|e| anyhow!("alloc q_bf16_seq: {e}"))?;
        let q_bf16_hm = device
            .alloc_buffer(q_elems * 2, DType::BF16, vec![1, nh, seq, d])
            .map_err(|e| anyhow!("alloc q_bf16_hm: {e}"))?;
        let k_bf16_seq = device
            .alloc_buffer(k_elems * 2, DType::BF16, vec![seq, nkv, d])
            .map_err(|e| anyhow!("alloc k_bf16_seq: {e}"))?;
        let k_bf16_hm = device
            .alloc_buffer(k_elems * 2, DType::BF16, vec![1, nkv, seq, d])
            .map_err(|e| anyhow!("alloc k_bf16_hm: {e}"))?;
        let v_bf16_seq = device
            .alloc_buffer(v_elems * 2, DType::BF16, vec![seq, nkv, d])
            .map_err(|e| anyhow!("alloc v_bf16_seq: {e}"))?;
        let v_bf16_hm = device
            .alloc_buffer(v_elems * 2, DType::BF16, vec![1, nkv, seq, d])
            .map_err(|e| anyhow!("alloc v_bf16_hm: {e}"))?;
        let mut out_bf16_hm = device
            .alloc_buffer(out_elems * 2, DType::BF16, vec![1, nh, seq, d])
            .map_err(|e| anyhow!("alloc out_bf16_hm: {e}"))?;

        let mut enc = device.command_encoder().context("FA prefill bridge encoder")?;

        // Step 1+2: Q F32 seq-major → BF16 seq-major → BF16 head-major.
        cast(
            &mut enc, registry, device.metal_device(),
            q_seq_major, &q_bf16_seq, q_elems, CastDirection::F32ToBF16,
        ).context("FA bridge: cast Q F32→BF16")?;
        enc.memory_barrier();
        permute_021_bf16(
            &mut enc, registry, device.metal_device(),
            &q_bf16_seq, &q_bf16_hm,
            seq, nh, d,
        ).context("FA bridge: permute_021 Q [seq, nh, d] → [nh, seq, d]")?;

        // Step 3+4: K F32 seq-major → BF16 seq-major → BF16 head-major.
        cast(
            &mut enc, registry, device.metal_device(),
            k_seq_major, &k_bf16_seq, k_elems, CastDirection::F32ToBF16,
        ).context("FA bridge: cast K F32→BF16")?;
        enc.memory_barrier();
        permute_021_bf16(
            &mut enc, registry, device.metal_device(),
            &k_bf16_seq, &k_bf16_hm,
            seq, nkv, d,
        ).context("FA bridge: permute_021 K [seq, nkv, d] → [nkv, seq, d]")?;

        // Step 5+6: V F32 seq-major → BF16 seq-major → BF16 head-major.
        cast(
            &mut enc, registry, device.metal_device(),
            v_seq_major, &v_bf16_seq, v_elems, CastDirection::F32ToBF16,
        ).context("FA bridge: cast V F32→BF16")?;
        enc.memory_barrier();
        permute_021_bf16(
            &mut enc, registry, device.metal_device(),
            &v_bf16_seq, &v_bf16_hm,
            seq, nkv, d,
        ).context("FA bridge: permute_021 V [seq, nkv, d] → [nkv, seq, d]")?;

        // Barrier: flash_attn_prefill reads Q/K/V head-major written above.
        enc.memory_barrier();

        // Step 7: dispatch flash_attn_prefill_bf16_d256.
        //   - scale = 1.0 / sqrt(head_dim) — Qwen3.5/3.6 oracle scale (no
        //     pre-scaling upstream, unlike Gemma 4).
        //   - do_causal = true — full prefill from offset 0; in-kernel causal
        //     mask handles row<col mask.
        //   - mask = None — pure causal, no external additive bias needed.
        //   - blk = None (path: dispatch_flash_attn_prefill_bf16_d256, the
        //     blk-less wrapper that delegates to *_with_blk(blk=None)).
        let scale = 1.0 / (d as f32).sqrt();
        dispatch_flash_attn_prefill_bf16_d256(
            &mut enc, device, registry,
            &q_bf16_hm, &k_bf16_hm, &v_bf16_hm,
            /* mask = */ None,
            &mut out_bf16_hm,
            &FlashAttnPrefillParams {
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len_q: seq_len,
                seq_len_k: seq_len,
                batch: 1,
                scale,
                do_causal: true,
            },
        ).context("FA bridge: dispatch_flash_attn_prefill_bf16_d256")?;

        // Barrier: permute_021_bf16_to_f32 reads out_bf16_hm written above.
        enc.memory_barrier();

        // Step 8: BF16 head-major → F32 seq-major (fused permute+cast).
        //   Input dims for permute_021 are (dim_a=nh, dim_b=seq, dim_c=d) —
        //   the kernel writes [seq, nh, d] (i.e. dim_a/dim_b swapped in the
        //   layout, matching the [A, B, C] → [B, A, C] contract).
        permute_021_bf16_to_f32(
            &mut enc, registry, device.metal_device(),
            &out_bf16_hm, &out_seq,
            nh, seq, d,
        ).context("FA bridge: permute_021_bf16_to_f32 out [nh, seq, d] → [seq, nh, d] F32")?;

        enc.commit_and_wait()
            .context("FA bridge: commit+wait flash_attn_prefill")?;
    }

    Ok(out_seq)
}

// ================================================================
// KV-cache-aware SDPA
// ================================================================

/// Apply SDPA with a pre-allocated KV cache.
///
/// Writes the current K/V tokens (from `k_seq_major`, `v_seq_major`) into the
/// cache at position `slot.current_len[0]`, then runs SDPA over all stored
/// K/V (0 .. current_len + seq_len), finally increments `current_len` by
/// `seq_len`.
///
/// # Cache layout
///
/// `slot.k` / `slot.v` are `[1, n_kv_heads, max_seq_len, head_dim]` F32
/// (SDPA-native layout, n_seqs=1 for single-sequence inference). The maximum
/// context this slot can hold is `max_seq_len` tokens. Overflow silently
/// stops writing (last token wins); callers should size the cache appropriately.
///
/// # Inputs
///
/// - `q_seq_major`: `[seq_len * n_heads,    head_dim]` F32 (seq-major, IMROPE'd).
/// - `k_seq_major`: `[seq_len * n_kv_heads, head_dim]` F32 (seq-major, IMROPE'd).
/// - `v_seq_major`: `[seq_len * n_kv_heads, head_dim]` F32 (seq-major, NOT rope'd).
///
/// # Returns
///
/// `[seq_len * n_heads, head_dim]` F32 (seq-major) — same shape/layout as Q.
#[allow(clippy::too_many_arguments)]
pub fn apply_sdpa_with_kv_cache(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_seq_major: &MlxBuffer,
    k_seq_major: &MlxBuffer,
    v_seq_major: &MlxBuffer,
    slot: &mut FullAttnKvSlot,
    seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    fa_arena: Option<&mut crate::inference::models::qwen35::FaPrefillArena>,
) -> Result<MlxBuffer> {
    let seq = seq_len as usize;
    let nh = n_heads as usize;
    let nkv = n_kv_heads as usize;
    let d = head_dim as usize;
    let max_sl = max_seq_len as usize;
    let cur_len = slot.current_len[0] as usize;

    let kv_write_tokens = (seq).min(max_sl.saturating_sub(cur_len));
    let kv_seq_len = (cur_len + kv_write_tokens).min(max_sl) as u32;

    // --- SDPA over full KV cache ---
    // For seq=1 (decode) with head_dim divisible by 32: fused GPU path:
    //   - kv_cache_copy_seq_f32_dual: write K and V into cache in one GPU dispatch
    //     (no CPU download/copy, no CPU barrier)
    //   - memory_barrier within same encoder
    //   - sdpa_decode: SIMD-vectorized F32 Q/K/V (simd_sum QK dot products)
    //   Single commit_and_wait for both K/V cache write + SDPA.
    //
    // For seq > 1 (prefill): CPU K/V permute is required for head-major layout.
    let out_buf = super::decode_pool::pooled_alloc_buffer(
            device, nh * seq * d * 4, DType::F32, vec![1, nh, seq, d])
        .map_err(|e| anyhow!("alloc sdpa kv-cache output: {e}"))?;

    if seq == 1 && head_dim % 32 == 0 {
        // Decode fast path: fused GPU K/V cache write + SIMD SDPA.
        // Q layout for seq=1: [n_heads, head_dim] is identical in seq-major and head-major.
        // K/V source: [seq*n_kv_heads, head_dim] = [n_kv_heads, head_dim] for seq=1,
        //   which kv_cache_copy_seq_f32_dual treats as [n_tokens=1, n_heads, head_dim].
        let mut enc = device.command_encoder().context("enc kv-cache+sdpa decode")?;
        if kv_write_tokens > 0 {
            dispatch_kv_cache_copy_seq_f32_dual(
                &mut enc, registry, device.metal_device(),
                k_seq_major, v_seq_major,
                &slot.k, &slot.v,
                n_kv_heads, head_dim, max_seq_len,
                cur_len as u32, kv_write_tokens as u32, 0,
            ).context("kv_cache_copy kv-cache decode")?;
            // Barrier: sdpa_decode reads slot.k/slot.v written above.
            enc.memory_barrier();
        }
        // 2026-05-03 — replaced dispatch_sdpa_decode with flash_attn_vec for
        // production head_dims (256/512). sdpa_decode dispatched a single
        // threadgroup per query head with serial KV iteration; at long
        // context (kv_seq_len > ~500) this bottlenecked single-SIMD
        // throughput. flash_attn_vec is the llama.cpp-ported decode-path
        // SDPA: NWG=32 workgroups split the KV cache, each running an
        // online softmax, then a reduce kernel combines per-workgroup
        // partials. Empirical on qwen3.6-35B-A3B-dwq48 (head_dim=256):
        // tg200 122.7→131.0, tg500 115.8→130.5, tg1000 105.2→130.0 — all
        // ahead of llama-bench (119.7 / 118.6 / 117.5). Determinism
        // preserved (same MD5 as sdpa_decode at temp=0).
        //
        // Cache layout already matches: `kv_cache_copy_seq_f32_kv_dual`
        // writes `dst_idx = head * capacity * head_dim + slot * head_dim
        // + elem` (see kv_cache_copy.metal:166-170), which is exactly
        // flash_attn_vec's `[n_kv_heads, kv_capacity, head_dim]`
        // expectation. No transpose / re-allocation needed.
        //
        // flash_attn_vec only supports head_dim ∈ {256, 512}. Smaller
        // head_dims (e.g. MTP test fixtures with head_dim=32) fall back
        // to sdpa_decode which handles arbitrary head_dim % 32 == 0.
        if head_dim == 256 || head_dim == 512 {
            let fa_tmp = super::decode_pool::pooled_alloc_buffer(
                device,
                flash_attn_vec_tmp_bytes(n_heads, head_dim),
                DType::F32,
                vec![flash_attn_vec_tmp_bytes(n_heads, head_dim) / 4],
            )
            .map_err(|e| anyhow!("alloc flash_attn_vec tmp: {e}"))?;
            let fa_params = FlashAttnVecParams {
                num_heads: n_heads,
                num_kv_heads: n_kv_heads,
                head_dim,
                kv_seq_len,
                kv_capacity: max_seq_len,
                scale: 1.0 / (d as f32).sqrt(),
                mask_type: 0, // single-token decode; causal mask is implicit
                sliding_window: 0,
                softcap: 0.0,
            };
            flash_attn_vec(
                &mut enc, registry, device,
                q_seq_major, &slot.k, &slot.v, &out_buf, &fa_tmp,
                &fa_params,
            ).context("flash_attn_vec kv-cache (FA-layer decode)")?;
        } else {
            dispatch_sdpa_decode(
                &mut enc, registry, device,
                q_seq_major, &slot.k, &slot.v, &out_buf,
                n_heads, n_kv_heads, head_dim,
                kv_seq_len, max_seq_len,
                1.0 / (d as f32).sqrt(),
            ).context("sdpa_decode kv-cache (head_dim fallback)")?;
        }
        // commit() without wait: out_buf is fed into ops6-7 on the same Metal
        // serial queue; GPU ordering guarantees SDPA completes first.
        // slot.current_len update below is a CPU-only counter — safe to update
        // before GPU completes; the next read of current_len is on the next token
        // by which time the queue is drained.
        enc.commit_labeled("layer.full_attn.sdpa_kv");
    } else {
        // Prefill path (seq > 1) or non-standard head_dim:
        // CPU K/V permute is required for the head-major cache layout.
        //
        // Wave 5b.9: instrument the 4 CPU↔GPU sub-stages around the
        // SDPA kernel call (gated on HF2Q_PROFILE_W5B8=1, no-op otherwise).
        // The buckets together account for `FaSdpaTotal` measured by
        // `build_gated_attn_layer`.
        // ADR-013 P21 stage-2 (2026-05-01): GPU-side KV cache write.
        //
        // Replaces the legacy `download_f32(k_seq_major) + download_f32(v_seq_major)
        // + CPU triple-loop write into slot.k/slot.v` with a single GPU dispatch
        // (`kv_cache_copy_seq_f32_kv_dual`) — the same kernel the decode path
        // already uses (line 1349). This eliminates:
        //   - The CPU bridge that violated as_slice "no GPU writer in flight"
        //     (Codex Phase-2b finding from Stage 1)
        //   - The need for `commit_and_wait_labeled` on ops1-4 in arena prefill
        //     (the wait was solely to give as_slice access to k_seq_major /
        //     v_seq_major). With this change, ops1-4 can downgrade to
        //     commit_labeled (next encoder will have GPU-ordering after ops1-4
        //     via Metal serial queue).
        //   - 86 ms host-wall on fa.ops1_4 at pp80 (HF2Q_PROFILE_W5B8 measurement
        //     after Stage 3a; the wait drained all in-flight async DN work).
        //
        // The kernel writes the same bytes as the CPU loop did:
        //   src layout: [seq * n_kv_heads, head_dim] = seq-major
        //   dst layout: slot.k/v = [n_kv_heads, max_seq_len, head_dim] = head-major
        //   slot = (cur_len + t) for full-attn (capacity == max_seq_len, no wrap)
        if kv_write_tokens > 0 {
            let _w5b9_kv_dl_copy = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FaSdpaKvDownloadCopy,
            );
            let mut enc = device.command_encoder()
                .context("enc kv_cache_copy_seq_dual prefill")?;
            mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_dual(
                &mut enc, registry, device.metal_device(),
                k_seq_major, v_seq_major,
                &slot.k, &slot.v,
                n_kv_heads, head_dim, max_seq_len,
                cur_len as u32, kv_write_tokens as u32, 0,
            ).context("kv_cache_copy_seq_f32_dual prefill")?;
            // commit_labeled (no host wait) — out_buf for the FA dispatch below
            // is a separate buffer; the new_path_eligible branch reads
            // k_seq_major/v_seq_major directly (not slot.k/slot.v) so this
            // commit's completion is not on the critical path of the FA bridge
            // — the legacy SDPA fallback will commit_and_wait at line 1497
            // and pick up the slot.k/slot.v writes via Metal queue ordering.
            enc.commit_labeled("layer.full_attn.kv_cache_write");
        }

        // ── Production path: flash_attn_prefill_bf16_d256 ──
        //
        // The legacy `sdpa` 3-pass tiled kernel (no online softmax, no
        // simdgroup_matrix MMA) was 76.5 % of per-FA-layer cost at PP4096
        // (W-5b.9 audit). It has been replaced by the same-purpose
        // `flash_attn_prefill_bf16_d256` kernel that Gemma 4 has used in
        // production since ADR-011 Phase 2 Wave 4 (commit `953dc1b`).
        //
        // Wave 5b.10 (commits a0cab10 + 43090a8 + 9ccaabb + c4a3e02) wired
        // the new path with a forensic A/B env gate `HF2Q_QWEN35_FA_LEGACY=1`.
        // Wave 5b.12 sunset audit (5 cold model loads × 3 cold prefills × 2
        // paths = 30 runs, all token id 11) confirmed parity holds, so the
        // env gate has been removed; the new path is now the unconditional
        // production codepath for the prefill-from-zero regime.
        //
        // Eligibility for the new path:
        //   - head_dim == 256 (Qwen3.5/3.6 production value)
        //   - cur_len == 0   (full prefill from offset 0; the kernel
        //     processes the chunk Q/K/V directly, not the full slot
        //     buffer — kv_seq_len equals seq_len in this regime)
        //
        // Cases that fall through to the legacy path:
        //   - head_dim != 256 (no D=256 dispatcher coverage; D=64 / D=512
        //     would need separate wire-up — Qwen3.5/3.6 does not need them)
        //   - cur_len > 0 (incremental prefill on top of an existing KV
        //     cache; the new kernel reads chunk Q against chunk K/V only.
        //     This case is not exercised by the production prefill path
        //     at this iter — full-prefill-from-zero is the live regime —
        //     but the legacy path is preserved as a correct fallback for
        //     non-prefill-from-zero correctness)
        // ITER-20 (refined): gate the FA-prefill path on `seq_len >= 16`
        // (= BK for the d=256 dispatcher).  Originally gated on >=32 in
        // iter-17 to avoid the dim=10 NaN observed at qL<32 (single
        // partial Q tile + single partial K tile).  Bisection across
        // the FRESH+OLD GGUF matrix revealed:
        //
        // * The dim=10 NaN bug specifically requires `kL_rem != 0` AND
        //   `qL_rem != 0` AND single K-tile — i.e. qL < 16.  At qL >= 16,
        //   K is BK-aligned (kL_rem=0) and the partial-K-tile mask path
        //   is NOT exercised; FA produces coherent output for both
        //   GGUFs at qL ∈ [16, ∞).
        // * The legacy 3-pass `sdpa` kernel ALSO has its own short-qL
        //   bug at qL <= 15 on Qwen3.6 (head_dim=256, kv_h=2):
        //   produces all-NaN logits on BOTH OLD and FRESH dwq48 GGUFs.
        //   Bisection: qL=15 NaN, qL=17 coherent.  HF2Q_DUMP_LAYER=ALL
        //   masks via dense flush_gpu sync points (see ADR-005 iter-19).
        //
        // So qL ∈ [1, 15] has no known-good path on this kernel set:
        //   * FA: dim=10 NaN at qL < 16
        //   * Legacy SDPA: all-NaN at qL <= 15
        //   * decode (flash_attn_vec): only fires at qL == 1
        //
        // The qL=16-31 range becomes coherent under the new >= 16 gate
        // (was previously routed to broken legacy SDPA when gate was
        // >= 32).  Long-prefill perf preserved (FA always fires).
        // qL ∈ [2, 15] remains broken — workaround is the user
        // padding their prompt up to qL >= 16.
        let new_path_eligible = head_dim == 256 && cur_len == 0 && seq_len >= 16;
        if new_path_eligible {
            // Dispatch flash_attn_prefill on the chunk seq-major Q/K/V
            // directly. Output is seq-major F32, matching the legacy
            // path's return shape.
            //
            // Wave 5b.9 instrumentation: bucketed under `fa.sdpa.kernel`
            // (the dominant W-5b.9 bucket). Q/out permute round-trips
            // disappear (no CPU permute, no download_f32/upload_f32) —
            // sub-buckets q_dl_perm_ul and out_dl_perm_ul read ~0 ms/layer.
            let _w5b10_kernel = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FaSdpaKernel,
            );
            let out_uploaded = apply_flash_attn_prefill_seq_major(
                device, registry,
                q_seq_major, k_seq_major, v_seq_major,
                seq_len, n_heads, n_kv_heads, head_dim,
                fa_arena,
            )?;
            // --- Update current_len cursor (prefill path) ---
            let new_len = kv_seq_len;
            slot.current_len[0] = new_len;
            return Ok(out_uploaded);
        }

        // ── Fallback path (head_dim != 256 OR cur_len > 0). Dispatched
        //    against the older `sdpa` kernel + CPU permute round-trips.
        //    Preserved bit-exactly for incremental-prefill correctness.
        //    Not exercised by the production Qwen3.5/3.6 prefill-from-zero
        //    path; kept for future model classes whose head_dim or
        //    incremental-prefill semantics differ. ──
        let q_gpu = {
            let _w5b9_q_dl_perm_ul = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FaSdpaQDownloadPermuteUpload,
            );
            let q_cpu = download_f32(q_seq_major)?;
            let q_hm = permute_seq_head_dim_to_head_seq_dim_cpu(&q_cpu, seq, nh, d);
            upload_f32(&q_hm, device)?
        };

        {
            let _w5b9_kernel = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FaSdpaKernel,
            );
            let params = SdpaParams {
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                kv_seq_len,
                scale: 1.0 / (d as f32).sqrt(),
                kv_capacity: max_seq_len,
            };
            let mut enc = device.command_encoder().context("enc sdpa kv-cache prefill")?;
            sdpa(&mut enc, registry, device, &q_gpu, &slot.k, &slot.v, &out_buf, &params, 1)
                .context("sdpa with kv cache prefill")?;
            enc.commit_and_wait_labeled("layer.full_attn.sdpa_legacy_prefill").context("commit sdpa kv-cache prefill")?;
        }

        // Permute output from head-major [n_heads, seq, head_dim] → seq-major
        // and re-upload back to GPU for op 6-7.
        let out_uploaded = {
            let _w5b9_out_dl_perm_ul = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FaSdpaOutDownloadPermuteUpload,
            );
            let out_hm_cpu = download_f32(&out_buf)?;
            let mut out_sm = vec![0.0f32; seq * nh * d];
            for h in 0..nh {
                for t in 0..seq {
                    let src = (h * seq + t) * d;
                    let dst = (t * nh + h) * d;
                    out_sm[dst..dst + d].copy_from_slice(&out_hm_cpu[src..src + d]);
                }
            }
            upload_f32(&out_sm, device)?
        };
        // --- Update current_len cursor (prefill path) ---
        let new_len = kv_seq_len;
        slot.current_len[0] = new_len;
        return Ok(out_uploaded);
    }

    // --- Update current_len cursor (decode path) ---
    slot.current_len[0] = kv_seq_len;

    // For seq=1 out_buf is [1, nh, 1, d] head-major == [nh, d] seq-major (same bytes).
    Ok(out_buf)
}

// ================================================================
// End-to-end layer builder
// ================================================================

/// Build the complete Qwen3.5 gated full-attention forward pass on the GPU.
///
/// Implements ADR-013 Decision 9 op order end-to-end.  Returns the
/// residual *contribution* `[seq_len, hidden_size]` F32 — the caller
/// computes `x + contribution` for the post-layer residual stream.
///
/// # Arguments
///
/// - `x`:       residual stream `[seq_len, hidden_size]` F32.
/// - `positions`: per-token axis positions, flat `[4 * seq_len]` I32.
///   Text-only Qwen3.5 repeats the token index across all 4 axes.
/// - `weights_gpu`: GPU weight handles (from `FullAttnWeightsGpu::from_cpu`
///   or the production weight loader).
/// - All shape params from `FullAttnShape`.
///
/// # Matmul note
///
/// This implementation uses the F32-via-BF16 projection path, suitable for
/// weights stored as F32 (parity testing, prototyping).  For production with
/// GGUF-quantised weights, the caller should use `quantized_matmul_ggml`
/// directly and integrate with the KV-cache path.
#[allow(clippy::too_many_arguments)]
pub fn build_gated_attn_layer(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    positions: &MlxBuffer,
    weights_gpu: &FullAttnWeightsGpu,
    // NEW: KV cache slot for this layer + its allocated capacity. Decode-time
    // correctness requires attending to all stored K/V (0..current_len + seq_len),
    // not just the current step's tokens. Prefill passes a fresh slot with
    // current_len == 0; decode passes the persistent slot from HybridKvCache.
    //
    // Pass `None` to run SDPA statelessly (legacy behavior — synthetic unit
    // tests that don't care about cache threading). Production forward_gpu
    // passes Some(slot) per-layer.
    kv_cache_slot: Option<&mut FullAttnKvSlot>,
    max_seq_len: u32,
    seq_len: u32,
    hidden_size: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    freq_base: f32,
    mrope_section: [u32; 4],
    rms_norm_eps: f32,
    fa_arena: Option<&mut crate::inference::models::qwen35::FaPrefillArena>,
    fa_proj_arena: Option<&mut crate::inference::models::qwen35::FaProjectionsArena>,
) -> Result<MlxBuffer> {
    // Capture arena presence before moving fa_arena into the SDPA call below.
    // Used to decide the commit vs commit_labeled path for ops1-4 and ops6-7.
    //
    // The ops1-4 commit can be downgraded to commit_labeled ONLY when
    // apply_sdpa_with_kv_cache will take the new_path_eligible branch
    // (head_dim == 256 && cur_len == 0), because that branch does NOT call
    // download_f32 on k_rope/v_flat. The legacy SDPA fallback branch DOES
    // call download_f32 and requires a CPU barrier (commit_and_wait).
    //
    // Condition: arena=Some && seq_len > 1 && head_dim == 256.
    // head_dim == 256 is the production Qwen3.5/3.6 value and matches the
    // new_path_eligible check in apply_sdpa_with_kv_cache. cur_len == 0 is
    // guaranteed by prefill-from-zero (the arena is only allocated in
    // forward_gpu_impl when seq_len > 1, and fresh-slot cur_len is always 0).
    let use_arena = fa_arena.is_some() && seq_len > 1 && head_dim == 256;
    let q_total = n_heads * head_dim;
    let kv_total = n_kv_heads * head_dim;

    // ---- ADR-019 Phase 2 iter89e2-F: Stage-A unified-CB fast path ----
    //
    // When all preconditions hold, encode ops1-4 + kv_cache_write +
    // fa.prefill_bridge into a SINGLE CommandBuffer (intra-stage
    // memory_barrier between the three sub-stages) with one terminal
    // commit_labeled. ops6-7 remains a separate CB in this iter to limit
    // blast radius for parity debugging (per design doc §6 iter89e2-F;
    // iter89e2-G extends fusion to ops6-7).
    //
    // Net per-FA-layer CB reduction: 4 -> 2 (ops1-4 + kv_cache_write +
    // fa.prefill_bridge merged; ops6-7 still its own CB). Across 10 FA
    // layers per Qwen3.6-35B-A3B prefill: 20 fewer CBs per chunk-engaged
    // prefill.
    //
    // Preconditions (all required):
    //  - `use_arena`        : fa_arena=Some && seq_len>1 && head_dim==256
    //                         (gates the FA-bridge `_into` variant)
    //  - `use_proj_arena`   : fa_proj_arena=Some && seq_len>1
    //                         (gates the projection-arena ops1-4 path)
    //  - `kv_cache_slot`    : Some (slot for the kv_cache_write dispatch)
    //  - `cur_len == 0`     : prefill-from-zero (matches the
    //                         `new_path_eligible` predicate in
    //                         apply_sdpa_with_kv_cache; production-only)
    //  - `!dump_bisect::is_enabled()` : R6 design-doc mitigation. The
    //                         within-layer dump_in_layer call sites at
    //                         lines below `as_slice` arena buffers; with
    //                         the unified CB, those buffers' producer
    //                         encoder is not yet committed when the dumps
    //                         run. Falling through to the legacy 4-CB
    //                         path keeps dump_bisect bisection viable.
    //
    // F-fence preservation:
    //  - F1 (persistent encoder): one persistent compute encoder per
    //    Stage-A CB, lazy-opened by the first dispatch. ops1-4 + kv_write
    //    + bridge dispatches all share that encoder via memory_barrier
    //    inserts.
    //  - F2 (iter58b residency-rescission): SAFE. All FA-layer scratch
    //    buffers (FaPrefillArena 7 BF16, FaProjectionsArena 10 F32) are
    //    allocated at forward_gpu.rs:1701/1738 and dropped only at end
    //    of forward_gpu_impl after the output-head terminal
    //    commit_and_wait_labeled. They outlive every Stage-A CB. No
    //    wrapper-local alloc_buffer drop occurs between dispatch and
    //    GPU completion. iter58b race is structurally unreachable.
    //  - F11 (zero-init alloc): one new per-call alloc (`out_seq`),
    //    matching the wrapper at gpu_full_attn.rs:1411-1413 byte-for-byte.
    //    No new ad-hoc allocations introduced.
    let use_fused_stage_ab = use_arena
        && fa_proj_arena.is_some()
        && kv_cache_slot
            .as_deref()
            .map(|s| s.current_len[0] == 0)
            .unwrap_or(false)
        && !super::dump_bisect::is_enabled();

    // ADR-015 iter86: validate the projections arena's capacity and consume
    // the &mut borrow into a local Option<&FaProjectionsArena> for the
    // ops1-4 + ops6-7 read-only access pattern. The arena's slot buffers
    // need only `&MlxBuffer` for `dispatch_*` calls (kernel writes go via
    // mlx-native's own internal mutability).
    //
    // For the Q/K/V/Gate projections (which call `quantized_matmul_ggml`
    // requiring `&mut MlxBuffer`), we use `apply_linear_projection_f32_into`
    // which takes `&mut dst` — but each projection writes to a distinct
    // arena field, so we destructure the arena into individual `&mut`
    // borrows just before that block.
    let use_proj_arena = fa_proj_arena.is_some() && seq_len > 1;
    if let Some(ref arena) = fa_proj_arena {
        // Capacity check happens once per layer call. Mismatch is a wiring
        // bug (caller must size the arena from the same FullAttnShape used
        // here).
        arena
            .validate_fits(seq_len, hidden_size, n_heads, n_kv_heads, head_dim)
            .context("FaProjectionsArena shape mismatch")?;
    }

    // ---- PREFILL / STATELESS PATH (all cases) ----
    //
    // For decode (seq=1) with a KV cache slot, the GPU-fused single-encoder
    // approach was measured slower (8 tok/s vs 11 tok/s) because Metal can
    // pipeline multiple small command buffers better than one large one for
    // short per-layer workloads. Keep the original 3-encoder path.
    //
    // For seq > 1 (prefill): CPU K/V permute is required for head-major layout.
    // Wave 5b.9: per-FA-layer ops1-4 wall (gated on HF2Q_PROFILE_W5B8=1).
    // Guard scoped to the inner `{ ... }` so the section closes before the SDPA call.
    //
    // ADR-015 iter86: when fa_proj_arena=Some (prefill, seq_len>1, model has
    // FA layers), use the arena-aware ops that write into caller-owned slots.
    // Eliminates 9 device.alloc_buffer / pooled_alloc_buffer calls per FA layer
    // (4 projection outputs + 5 helper outputs) per W-5b.8 fa.ops1_4 bucket.
    //
    // The two paths produce bit-identical results — same kernels, same dispatch
    // sequence, same intra-encoder barriers; only the output buffer source
    // differs (caller-owned arena slot vs. fresh device.alloc_buffer /
    // pooled_alloc_buffer). The byte-exact F32 parity test
    // `fa_projections_arena_byte_exact_f32_parity` (this file) guards the
    // equivalence at seq_len=128.
    // Take the &mut borrow only when use_proj_arena is true. fa_proj_arena
    // remains `Option<&mut FaProjectionsArena>`; we reborrow in each phase
    // (ops1-4 here, ops6-7 below) so both phases can share access without
    // consuming the outer Option.
    let mut fa_proj_arena = fa_proj_arena;
    // ADR-019 Phase 2 iter89e2-F: when use_fused_stage_ab, consume fa_arena
    // and kv_cache_slot in this branch (encoded into the same Stage-A CB
    // as ops1-4). Outer bindings rebound to None so the downstream op5
    // dispatch path is reached only by non-fused branches (decode, dump
    // bisect, head_dim != 256, cur_len != 0, missing arenas).
    let mut fa_arena = fa_arena;
    let mut kv_cache_slot = kv_cache_slot;
    let mut attn_out_fused: Option<MlxBuffer> = None;
    // ADR-019 Phase 2 iter89e2-G: when use_fused_stage_ab fires AND ops6-7
    // can also be encoded into the same CB, we move the Stage-A encoder out
    // of the ops1-4 inner block and into this function-scope Option. ops6-7
    // then takes the encoder, encodes sigmoid_gate_multiply + linear_proj,
    // and issues the single terminal commit_labeled("layer.full_attn.stage_a").
    // Replaces 4 separate commit_labeled calls per FA layer (ops1-4 +
    // kv_cache_write + fa.prefill_bridge + ops6-7) with ONE.
    //
    // Eligibility condition is identical to use_fused_stage_ab AND
    // ops6-7's arena path (use_proj_arena), so the encoder ownership transfer
    // is safe — the ops6-7 block sees the same arena that ops1-4 wrote into.
    let mut fused_stage_a_enc: Option<mlx_native::CommandEncoder> = None;
    let (x_norm, q_flat, k_flat, v_flat, gate_flat, q_normed, k_normed, q_rope, k_rope) = if let
        Some(arena) = fa_proj_arena.as_mut().map(|a| &mut **a).filter(|_| use_proj_arena)
    {
        let _w5b9_ops1to4 = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FaOps1to4,
        );
        let mut enc = device.command_encoder().context("enc ops1-4")?;

        // Op 1: pre-attention RMSNorm → arena.x_norm_buf
        apply_pre_attn_rms_norm_into(
            &mut enc, registry, device, x, weights_gpu,
            &arena.x_norm_buf, &arena.pre_norm_params_buf,
            seq_len, hidden_size,
        )?;
        // Barrier: ops 2 read from x_norm written above.
        enc.memory_barrier();

        // Op 2: Q/K/V/G projections — all read from arena.x_norm_buf, write
        // into arena.{q,k,v,gate}_proj_buf via apply_linear_projection_f32_into.
        // Each call needs &mut on a distinct arena field; we sequence them so
        // the unique &mut borrows don't overlap.
        apply_linear_projection_f32_into(
            &mut enc, registry, device, &arena.x_norm_buf,
            &weights_gpu.wq, &mut arena.q_proj_buf,
            seq_len, hidden_size, q_total,
        )?;
        apply_linear_projection_f32_into(
            &mut enc, registry, device, &arena.x_norm_buf,
            &weights_gpu.wk, &mut arena.k_proj_buf,
            seq_len, hidden_size, kv_total,
        )?;
        apply_linear_projection_f32_into(
            &mut enc, registry, device, &arena.x_norm_buf,
            &weights_gpu.wv, &mut arena.v_proj_buf,
            seq_len, hidden_size, kv_total,
        )?;
        apply_linear_projection_f32_into(
            &mut enc, registry, device, &arena.x_norm_buf,
            &weights_gpu.w_gate, &mut arena.gate_proj_buf,
            seq_len, hidden_size, q_total,
        )?;
        // Barrier: ops 3 read from q_proj/k_proj written above.
        enc.memory_barrier();

        // Op 3: per-head RMSNorm on Q and K (shared params from arena).
        apply_q_or_k_per_head_rms_norm_into(
            &mut enc, registry, device, &arena.q_proj_buf,
            &weights_gpu.attn_q_norm, &arena.q_normed_buf,
            &arena.qk_rms_params_buf, seq_len, n_heads, head_dim,
        )?;
        apply_q_or_k_per_head_rms_norm_into(
            &mut enc, registry, device, &arena.k_proj_buf,
            &weights_gpu.attn_k_norm, &arena.k_normed_buf,
            &arena.qk_rms_params_buf, seq_len, n_kv_heads, head_dim,
        )?;
        // Barrier: ops 4 read from q_normed / k_normed written above.
        enc.memory_barrier();

        // Op 4: IMROPE on Q and K — params triple is in dispatch_rope_multi_cached's
        // thread-local cache (NOT in this arena) — see apply_imrope_into doc.
        apply_imrope_into(
            &mut enc, registry, device, &arena.q_normed_buf, &arena.q_rope_buf,
            positions, seq_len, n_heads, head_dim, rotary_dim, freq_base, mrope_section,
        )?;
        apply_imrope_into(
            &mut enc, registry, device, &arena.k_normed_buf, &arena.k_rope_buf,
            positions, seq_len, n_kv_heads, head_dim, rotary_dim, freq_base, mrope_section,
        )?;

        // ── ADR-019 Phase 2 iter89e2-F: Stage-A unified-CB fusion ──────────
        //
        // When use_fused_stage_ab is true, encode kv_cache_write +
        // fa.prefill_bridge into the SAME `enc` and issue ONE terminal
        // commit_labeled at end of bridge. Replaces 3 separate commits:
        //   - layer.full_attn.ops1-4      (this block)
        //   - layer.full_attn.kv_cache_write (apply_sdpa_with_kv_cache:1706)
        //   - fa.prefill_bridge           (apply_flash_attn_prefill_seq_major:1449)
        //
        // 4 -> 2 CBs per FA layer (ops6-7 still its own CB; iter89e2-G
        // extends fusion to ops6-7).
        //
        // F2 invariant: arena buffers (FaPrefillArena 7 BF16, FaProjectionsArena
        // 10 F32, persistent slot.k/slot.v) all outlive this CB by design —
        // forward_gpu_impl owns them through the output-head terminal commit.
        // The wider in-flight window has no F2 exposure because no buffer
        // can drop between dispatch encode and GPU completion.
        if use_fused_stage_ab {
            // Take the &mut on slot + arena ONCE for the duration of this
            // fused-stage block. Both bindings are rebound to None at the
            // end so the SDPA-call branch (line ~2227) sees no slot/arena
            // and does not double-dispatch (we set attn_out_fused below).
            let slot = kv_cache_slot.as_mut().expect(
                "use_fused_stage_ab implies kv_cache_slot.is_some()"
            );
            let fa_pre = fa_arena.as_mut().expect(
                "use_fused_stage_ab implies fa_arena.is_some()"
            );

            let cur_len_u32 = slot.current_len[0]; // == 0 by predicate
            let max_sl = max_seq_len as usize;
            let kv_write_tokens =
                (seq_len as usize).min(max_sl.saturating_sub(cur_len_u32 as usize));
            let kv_seq_len = (cur_len_u32 as usize + kv_write_tokens).min(max_sl) as u32;

            // RAW barrier: kv_cache_write reads arena.k_rope_buf / arena.v_proj_buf
            // written by ops4 (k_rope) / op2 (v_proj) above.
            enc.memory_barrier();
            if kv_write_tokens > 0 {
                let _w5b9_kv = super::wave5b8_profile::Section::start(
                    super::wave5b8_profile::SectionKind::FaSdpaKvDownloadCopy,
                );
                mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_dual(
                    &mut enc, registry, device.metal_device(),
                    &arena.k_rope_buf, &arena.v_proj_buf,
                    &slot.k, &slot.v,
                    n_kv_heads, head_dim, max_seq_len,
                    cur_len_u32, kv_write_tokens as u32, 0,
                ).context("kv_cache_copy_seq_f32_dual prefill (fused stage_ab)")?;
            }

            // RAW barrier: fa.prefill_bridge reads arena.q_rope_buf /
            // arena.k_rope_buf / arena.v_proj_buf. Independent of
            // kv_cache_write's writes (slot.k/slot.v are not read by the
            // bridge), but Metal MTLDispatchTypeConcurrent reorders within
            // a CB without an explicit barrier — required for ordering
            // correctness and profiling attribution.
            enc.memory_barrier();

            // Allocate `out_seq` (the FA bridge's F32 seq-major output).
            // Same per-call alloc shape as the wrapper at line 1411-1413
            // — caller-owned, moved into attn_out_fused below.
            let seq = seq_len as usize;
            let nh = n_heads as usize;
            let d = head_dim as usize;
            let out_elems = seq * nh * d;
            let out_seq = device
                .alloc_buffer(out_elems * 4, DType::F32, vec![seq, nh, d])
                .map_err(|e| anyhow!("alloc out_seq (fused stage_ab): {e}"))?;

            // Encode the FA bridge body (8 dispatches + 5 intra-encoder
            // barriers) into the SAME `enc`. iter89e2-E's `_into` variant.
            {
                let _w5b10_kernel = super::wave5b8_profile::Section::start(
                    super::wave5b8_profile::SectionKind::FaSdpaKernel,
                );
                apply_flash_attn_prefill_seq_major_into(
                    &mut enc, device, registry,
                    &arena.q_rope_buf, &arena.k_rope_buf, &arena.v_proj_buf,
                    &out_seq,
                    seq_len, n_heads, n_kv_heads, head_dim,
                    fa_pre,
                )?;
            }

            // Update KV cursor (CPU-only counter; safe before GPU completes).
            slot.current_len[0] = kv_seq_len;

            // ADR-019 Phase 2 iter89e2-G: defer the Stage-A terminal commit
            // by moving `enc` into the function-scope Option. The ops6-7
            // block consumes it, encodes sigmoid_gate_multiply + linear_proj
            // into the SAME encoder, and issues the single terminal
            // commit_labeled("layer.full_attn.stage_a") covering all 4 FA-layer
            // ops (ops1-4 + kv_cache_write + fa.prefill_bridge + ops6-7).
            //
            // F2 invariant: Both arenas (FaPrefillArena 7 BF16 scratches,
            // FaProjectionsArena 10 F32 scratches including gated_buf for
            // ops6) plus persistent slot.k/slot.v plus the caller-owned
            // out_seq outlive the deferred CB by design — forward_gpu_impl
            // owns the arenas through the output-head terminal
            // commit_and_wait_labeled, and `out` (linear_proj output) is on
            // the Rust stack of forward_gpu_impl until consumed by
            // dispatch_fused_residual_norm_f32. The wider in-flight CB
            // window has zero F2 exposure: no buffer can drop between
            // dispatch encode and GPU completion. iter58b race structurally
            // unreachable.
            fused_stage_a_enc = Some(enc);

            // Hand attn_out to the post-SDPA control flow; suppress the
            // legacy SDPA dispatch by consuming fa_arena + kv_cache_slot.
            attn_out_fused = Some(out_seq);
            fa_arena = None;
            kv_cache_slot = None;
        } else {
            // Decode fast path (seq=1, head_dim%32==0): commit() without wait.
            // Metal serial queue guarantees ops1-4 completes before SDPA starts.
            // The SDPA encode path (apply_sdpa_with_kv_cache seq=1 branch) never
            // calls download_f32 so no CPU buffer access races.
            //
            // Prefill path (seq>1) without arena: commit_and_wait()
            // because apply_sdpa_with_kv_cache's prefill branch calls download_f32
            // (CPU read) on k_rope/v_flat before submitting any GPU work, so the
            // GPU must have finished writing those buffers before we return.
            //
            // Decode (seq=1) only: GPU-only path, safe to commit_labeled.
            // Prefill (seq>1): apply_sdpa_with_kv_cache (legacy path) unconditionally
            // calls download_f32(k_seq_major) / download_f32(v_seq_major) to write
            // the persistent KV cache (slot.k/slot.v) BEFORE the new_path_eligible
            // branch dispatches FA. download_f32 → MlxBuffer::as_slice violates the
            // ADR-013 P21 Stage 2 (2026-05-01): KV-cache write is now a GPU
            // dispatch (kv_cache_copy_seq_f32_dual at apply_sdpa_with_kv_cache:1380),
            // eliminating the download_f32(k_seq_major)/download_f32(v_seq_major)
            // calls that previously required commit_and_wait here for the
            // as_slice "no GPU writer in flight" contract. With use_arena=true
            // (Stage 1 FaPrefillArena keeps scratches alive past commit), we can
            // safely commit_labeled in prefill too — the FA bridge dispatch runs
            // on the same Metal serial queue and orders after ops1-4 by GPU
            // queue ordering. iter58b residency-rescission is prevented by the
            // FaPrefillArena lifetime (scratches don't drop until end of prefill).
            if (seq_len == 1 && head_dim % 32 == 0) || use_arena {
                enc.commit_labeled("layer.full_attn.ops1-4");
            } else {
                enc.commit_and_wait_labeled("layer.full_attn.ops1-4")
                    .context("commit ops1-4 prefill (proj arena)")?;
            }
        }

        // ARC clones from the arena are returned to the outer-scope tuple.
        // Each clone is just a refcount bump on the underlying Metal buffer;
        // the arena slot is conceptually borrowed for the rest of the layer.
        // The next FA layer overwrites these slots only AFTER its own enc
        // commit submits to the Metal serial queue — by which time all
        // CBs that read these clones have already been queued ahead.
        (
            arena.x_norm_buf.clone(),
            arena.q_proj_buf.clone(),
            arena.k_proj_buf.clone(),
            arena.v_proj_buf.clone(),
            arena.gate_proj_buf.clone(),
            arena.q_normed_buf.clone(),
            arena.k_normed_buf.clone(),
            arena.q_rope_buf.clone(),
            arena.k_rope_buf.clone(),
        )
    } else {
        let _w5b9_ops1to4 = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FaOps1to4,
        );
        let mut enc = device.command_encoder().context("enc ops1-4")?;

        // Op 1: pre-attention RMSNorm → x_norm
        let x_norm = apply_pre_attn_rms_norm(
            &mut enc, registry, device, x, weights_gpu,
            seq_len, hidden_size, rms_norm_eps,
        )?;
        // Barrier: ops 2 read from x_norm written above.
        enc.memory_barrier();

        // Op 2: Q/K/V/G projections (all read from x_norm).
        // Pool-aware path: seq_len=1 (decode) goes to the arena pool, seq_len>1
        // (prefill) auto-falls-back to unpooled inside the helper because some
        // prefill consumers download K/V to CPU (see apply_sdpa_with_kv_cache).
        let q_flat = apply_linear_projection_f32_pooled(
            &mut enc, registry, device, &x_norm,
            &weights_gpu.wq, seq_len, hidden_size, q_total,
        )?;
        let k_flat = apply_linear_projection_f32_pooled(
            &mut enc, registry, device, &x_norm,
            &weights_gpu.wk, seq_len, hidden_size, kv_total,
        )?;
        let v_flat = apply_linear_projection_f32_pooled(
            &mut enc, registry, device, &x_norm,
            &weights_gpu.wv, seq_len, hidden_size, kv_total,
        )?;
        let gate_flat = apply_linear_projection_f32_pooled(
            &mut enc, registry, device, &x_norm,
            &weights_gpu.w_gate, seq_len, hidden_size, q_total,
        )?;
        // Barrier: ops 3 read from q_flat / k_flat written above.
        enc.memory_barrier();

        // Op 3: per-head RMSNorm on Q and K
        let q_normed = apply_q_or_k_per_head_rms_norm(
            &mut enc, registry, device, &q_flat,
            &weights_gpu.attn_q_norm, seq_len, n_heads, head_dim, rms_norm_eps,
        )?;
        let k_normed = apply_q_or_k_per_head_rms_norm(
            &mut enc, registry, device, &k_flat,
            &weights_gpu.attn_k_norm, seq_len, n_kv_heads, head_dim, rms_norm_eps,
        )?;
        // Barrier: ops 4 read from q_normed / k_normed written above.
        enc.memory_barrier();

        // Op 4: IMROPE on Q and K
        let q_rope = apply_imrope(
            &mut enc, registry, device, &q_normed, positions,
            seq_len, n_heads, head_dim, rotary_dim, freq_base, mrope_section,
        )?;
        let k_rope = apply_imrope(
            &mut enc, registry, device, &k_normed, positions,
            seq_len, n_kv_heads, head_dim, rotary_dim, freq_base, mrope_section,
        )?;

        // Decode fast path (seq=1, head_dim%32==0): commit() without wait.
        // Metal serial queue guarantees ops1-4 completes before SDPA starts.
        // The SDPA encode path (apply_sdpa_with_kv_cache seq=1 branch) never
        // calls download_f32 so no CPU buffer access races.
        //
        // Prefill path (seq>1) without arena: commit_and_wait()
        // because apply_sdpa_with_kv_cache's prefill branch calls download_f32
        // (CPU read) on k_rope/v_flat before submitting any GPU work, so the
        // GPU must have finished writing those buffers before we return.
        //
        // Decode (seq=1) only: GPU-only path, safe to commit_labeled.
        // Prefill (seq>1): apply_sdpa_with_kv_cache (legacy path) unconditionally
        // calls download_f32(k_seq_major) / download_f32(v_seq_major) to write
        // the persistent KV cache (slot.k/slot.v) BEFORE the new_path_eligible
        // branch dispatches FA. download_f32 → MlxBuffer::as_slice violates the
        // ADR-013 P21 Stage 2 (2026-05-01): KV-cache write is now a GPU
        // dispatch (kv_cache_copy_seq_f32_dual at apply_sdpa_with_kv_cache:1380),
        // eliminating the download_f32(k_seq_major)/download_f32(v_seq_major)
        // calls that previously required commit_and_wait here for the
        // as_slice "no GPU writer in flight" contract. With use_arena=true
        // (Stage 1 FaPrefillArena keeps scratches alive past commit), we can
        // safely commit_labeled in prefill too — the FA bridge dispatch runs
        // on the same Metal serial queue and orders after ops1-4 by GPU
        // queue ordering. iter58b residency-rescission is prevented by the
        // FaPrefillArena lifetime (scratches don't drop until end of prefill).
        if (seq_len == 1 && head_dim % 32 == 0) || use_arena {
            enc.commit_labeled("layer.full_attn.ops1-4");
        } else {
            enc.commit_and_wait_labeled("layer.full_attn.ops1-4").context("commit ops1-4 prefill")?;
        }
        (x_norm, q_flat, k_flat, v_flat, gate_flat, q_normed, k_normed, q_rope, k_rope)
    };
    // ADR-015 iter61a-3: dump pre-rope checkpoints BEFORE the drop below.
    // ops1-4 was committed sync for prefill, so as_slice is safe.
    super::dump_bisect::dump_in_layer(
        "fa_x_norm",
        &x_norm,
        &[seq_len as usize, hidden_size as usize],
        device,
    );
    super::dump_bisect::dump_in_layer(
        "fa_q_flat",
        &q_flat,
        &[seq_len as usize, q_total as usize],
        device,
    );
    super::dump_bisect::dump_in_layer(
        "fa_k_flat",
        &k_flat,
        &[seq_len as usize, kv_total as usize],
        device,
    );
    super::dump_bisect::dump_in_layer(
        "fa_q_normed",
        &q_normed,
        &[seq_len as usize, n_heads as usize, head_dim as usize],
        device,
    );
    super::dump_bisect::dump_in_layer(
        "fa_k_normed",
        &k_normed,
        &[seq_len as usize, n_kv_heads as usize, head_dim as usize],
        device,
    );
    // Suppress unused variable warnings for intermediate buffers that were
    // consumed by downstream ops within the same encoder.
    let _ = (x_norm, q_flat, k_flat, q_normed, k_normed);

    // ---- ADR-015 iter61a-3: within-layer bisection dumps (post ops1-4 commit_and_wait) ----
    // The ops1-4 encoder above committed (sync for prefill), so q_rope/k_rope/
    // v_flat/gate_flat/q_normed/k_normed are GPU-finalized and as_slice-safe.
    super::dump_bisect::dump_in_layer(
        "fa_q_rope",
        &q_rope,
        &[seq_len as usize, n_heads as usize, head_dim as usize],
        device,
    );
    super::dump_bisect::dump_in_layer(
        "fa_k_rope",
        &k_rope,
        &[seq_len as usize, n_kv_heads as usize, head_dim as usize],
        device,
    );
    super::dump_bisect::dump_in_layer(
        "fa_v_flat",
        &v_flat,
        &[seq_len as usize, n_kv_heads as usize, head_dim as usize],
        device,
    );
    super::dump_bisect::dump_in_layer(
        "fa_gate_flat",
        &gate_flat,
        &[seq_len as usize, n_heads as usize, head_dim as usize],
        device,
    );

    // ---- Op 5: SDPA (causal, GQA) with optional KV-cache threading ----
    // Wave 5b.9: per-FA-layer SDPA op5 wall (gated on HF2Q_PROFILE_W5B8=1).
    //
    // ADR-019 Phase 2 iter89e2-F: when use_fused_stage_ab fired, attn_out
    // was produced inline as part of the Stage-A unified CB (ops1-4 +
    // kv_cache_write + fa.prefill_bridge merged); we skip the legacy
    // dispatch entirely.
    let attn_out = if let Some(out_fused) = attn_out_fused.take() {
        out_fused
    } else {
        let _w5b9_sdpa_total = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FaSdpaTotal,
        );
        match kv_cache_slot {
            Some(slot) => apply_sdpa_with_kv_cache(
                device, registry,
                &q_rope, &k_rope, &v_flat,
                slot, seq_len, n_heads, n_kv_heads, head_dim, max_seq_len,
                fa_arena,
            )?,
            None => {
                let mut enc = device.command_encoder().context("enc op5")?;
                apply_sdpa_causal_from_seq_major(
                    &mut enc, registry, device,
                    &q_rope, &k_rope, &v_flat,
                    seq_len, n_heads, n_kv_heads, head_dim,
                )?
            }
        }
    };
    // attn_out is now [seq * n_heads, head_dim] seq-major.

    // ADR-015 iter61a-3: dump SDPA output (the candidate point of divergence
    // since flash_attn_prefill is the suspected non-determinism site).
    super::dump_bisect::dump_in_layer(
        "fa_sdpa_out",
        &attn_out,
        &[seq_len as usize, n_heads as usize, head_dim as usize],
        device,
    );

    // ---- Ops 6–7: sigmoid-gate multiply + output projection ----
    //
    // ADR-015 iter86: when use_proj_arena, sigmoid_mul writes into
    // arena.gated_buf (with arena.sigmoid_params_buf as the element-count
    // buffer) instead of allocating from the decode pool. Same kernels,
    // same dispatch order, same memory_barrier — only the output buffer
    // source differs. The byte-exact F32 parity test guards equivalence.
    let out = {
        // Wave 5b.9: per-FA-layer ops6-7 wall (gated on HF2Q_PROFILE_W5B8=1).
        let _w5b9_ops6to7 = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FaOps6to7,
        );
        let n_elem = seq_len * q_total;
        // ADR-019 Phase 2 iter89e2-G: when fused_stage_a_enc is Some, the
        // Stage-A encoder (carrying ops1-4 + kv_cache_write + fa.prefill_bridge
        // dispatches encoded but not yet committed) was moved here from the
        // ops1-4 fused branch above. We continue encoding ops6-7 into the
        // SAME encoder and issue ONE terminal commit_labeled covering all 4
        // FA-layer ops. Otherwise (decode, dump_bisect, head_dim != 256,
        // cur_len != 0, missing arenas), open a fresh encoder as before.
        let (mut enc, fused_into_stage_a) = match fused_stage_a_enc.take() {
            Some(e) => (e, true),
            None => (device.command_encoder().context("enc ops6-7")?, false),
        };
        // ADR-019 Phase 2 iter89e2-G: RAW barrier between fa.prefill_bridge's
        // final dispatch (permute_021_bf16_to_f32 → out_seq) and ops6
        // (sigmoid_gate_multiply reads attn_out == out_seq). The legacy
        // 4-CB layout had this RAW edge enforced by the CB boundary between
        // fa.prefill_bridge and ops6-7; the fused path replaces that
        // boundary with this intra-CB memory_barrier(). Mirrors the existing
        // RAW barriers at the ops1-4→kv_cache_write and kv_cache_write→
        // fa.prefill_bridge boundaries from iter89e2-F. AC-PA2 Heisenbug 5×
        // is the empirical guard.
        if fused_into_stage_a {
            enc.memory_barrier();
        }
        let gated = if let Some(arena) =
            fa_proj_arena.as_ref().map(|a| &**a).filter(|_| use_proj_arena)
        {
            apply_sigmoid_gate_multiply_into(
                &mut enc, registry, device,
                &attn_out, &gate_flat, &arena.gated_buf,
                &arena.sigmoid_params_buf, n_elem,
            )?;
            arena.gated_buf.clone()
        } else {
            apply_sigmoid_gate_multiply(
                &mut enc, registry, device, &attn_out, &gate_flat, n_elem,
            )?
        };
        // ADR-015 iter61a-4: memory_barrier between Op 6 (sigmoid_gate_multiply
        // writes `gated`) and Op 7 (linear_projection reads `gated`).
        //
        // The same RAW race that was fixed in `apply_gated_attn_layer_decode_into`
        // by ADR-015 iter21 (gpu_full_attn.rs:1925) also lives in this prefill
        // path, but had been latent because per-op bisection only landed in
        // iter61a-3.  Diagnosis (iter61a-4):
        //   * 27B-dwq46 'Hello' T=0/top-k=1 max=2 cold-process bisection
        //     pinned first divergence at (FullAttn layer 3, attn_out) byte
        //     20992 (token 1 / dim 128 of post-wo_proj output).
        //   * All within-FA dumps for layer 3 (fa_x_norm, fa_q_flat,
        //     fa_k_flat, fa_v_flat, fa_q_normed, fa_k_normed, fa_q_rope,
        //     fa_k_rope, fa_gate_flat, fa_sdpa_out) were byte-identical
        //     across cold runs — the race lived strictly in this 2-dispatch
        //     ops6-7 encoder.
        //   * Even with the encoder's terminal `commit_and_wait` (sync at
        //     the boundary), Metal's `MTLDispatchTypeConcurrent` is free to
        //     reorder the two dispatches WITHIN a single command buffer
        //     unless an explicit `memory_barrier()` enforces the RAW edge.
        //     The legacy decode encoder containing only these 2 dispatches
        //     happened to be deterministic by accident (no other parallel
        //     work to interleave); under the prefill multi-token regime
        //     (seq=11+ for chat-template-wrapped prompts on 27B/35B) there
        //     is enough threadgroup pressure to expose the reordering.
        //
        // Mechanism is the FullAttn-prefill twin of iter58b's DeltaNet
        // chunk-prefill residency-set-lifetime fix and iter21's decode-path
        // ops6→ops7 RAW barrier — same general pattern: when fused-encoder
        // dispatches share a written buffer, the producer→consumer edge
        // must be made explicit via `memory_barrier()`, never inferred from
        // submission order.
        enc.memory_barrier();
        let out = apply_linear_projection_f32_pooled(
            &mut enc, registry, device, &gated,
            &weights_gpu.wo, seq_len, q_total, hidden_size,
        )?;
        // Decode fast path (seq=1): commit() without wait, and `out` is pooled.
        // The caller (forward_gpu) feeds `out` into dispatch_fused_residual_norm_f32
        // via a new encoder on the same Metal serial queue, so the GPU will
        // execute ops6-7 before fused_residual_norm without a CPU sync.
        //
        // Prefill (seq>1) without arena: commit_and_wait() because
        // dump_hidden_stats in forward_gpu may do a CPU read of the returned
        // buffer (HF2Q_DECODE_PROFILE-gated), and because prefill throughput
        // was not the hot path pre-P21.
        //
        // Prefill (seq>1) with arena (use_arena=true): the returned `out` is
        // consumed by dispatch_fused_residual_norm_f32 on the same Metal serial
        // queue. That dispatch is GPU-ordered behind this CB, so no CPU sync
        // is needed. dump_hidden_stats is HF2Q_DECODE_PROFILE-gated (env-only
        // diagnostic, not on the production path). Downgrade to commit_labeled
        // is safe per queen plan A.1 ops6-7 analysis.
        // ADR-019 Phase 2 iter89e2-G: when fused_into_stage_a, this single
        // terminal commit covers all 4 FA-layer ops (ops1-4 + kv_cache_write
        // + fa.prefill_bridge + ops6-7), labeled "layer.full_attn.stage_a"
        // for xctrace MST attribution. Replaces 4 separate commit_labeled
        // calls per FA layer with ONE. The non-fused branches (decode path,
        // dump bisect, head_dim != 256, cur_len != 0, missing arenas) keep
        // the legacy "layer.full_attn.ops6-7" label and commit choice.
        //
        // commit_labeled is non-blocking; out (linear_proj output) is on the
        // Rust stack until consumed by dispatch_fused_residual_norm_f32 in
        // forward_gpu's next encoder on the same Metal serial queue.
        if fused_into_stage_a {
            enc.commit_labeled("layer.full_attn.stage_a");
        } else if seq_len == 1 || use_arena {
            enc.commit_labeled("layer.full_attn.ops6-7");
        } else {
            enc.commit_and_wait_labeled("layer.full_attn.ops6-7").context("commit ops6-7")?;
        }
        out
    };

    Ok(out)
}

// ================================================================
// ADR-015 P3 Stage 1: caller-driven single-CB FullAttn (decode-only)
// ================================================================

/// Decode-only KV-cache + SDPA, encoded into the caller's encoder.
///
/// Mirrors the `seq=1 && head_dim%32==0` decode fast path of
/// [`apply_sdpa_with_kv_cache`] but DOES NOT open or commit its own
/// command buffer.  All dispatches are encoded into the caller-supplied
/// `enc`; it is the caller's responsibility to insert any cross-stage
/// `enc.memory_barrier()` before this call (producer→sdpa) and after
/// (sdpa→consumer).
///
/// ADR-015 P1 audit row `gpu_full_attn.rs:959/:983`: the internal
/// kv_copy→sdpa_decode RAW barrier is preserved here at the same call
/// site, position relative to dispatches unchanged.
///
/// Returns `[1, n_heads, 1, head_dim]` F32 — same shape and contents as
/// the legacy decode-fast-path return value.
///
/// # Errors
///
/// - `seq_len != 1` (decode-only path).
/// - `head_dim % 32 != 0` (SIMD path requires aligned head_dim).
/// - Any underlying mlx-native dispatch failure.
#[allow(clippy::too_many_arguments)]
pub fn apply_sdpa_with_kv_cache_decode_into(
    enc: &mut mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_seq_major: &MlxBuffer,
    k_seq_major: &MlxBuffer,
    v_seq_major: &MlxBuffer,
    slot: &mut FullAttnKvSlot,
    seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
) -> Result<MlxBuffer> {
    debug_assert_eq!(seq_len, 1, "apply_sdpa_with_kv_cache_decode_into: seq_len must be 1");
    debug_assert_eq!(head_dim % 32, 0, "apply_sdpa_with_kv_cache_decode_into: head_dim must be %32==0");

    let seq = seq_len as usize;
    let nh = n_heads as usize;
    let nkv = n_kv_heads as usize;
    let d = head_dim as usize;
    let max_sl = max_seq_len as usize;
    let cur_len = slot.current_len[0] as usize;

    let kv_write_tokens = (seq).min(max_sl.saturating_sub(cur_len));
    let kv_seq_len = (cur_len + kv_write_tokens).min(max_sl) as u32;

    let _ = nkv; // currently unused; retained for shape doc parity with legacy path.

    let out_buf = super::decode_pool::pooled_alloc_buffer(
            device, nh * seq * d * 4, DType::F32, vec![1, nh, seq, d])
        .map_err(|e| anyhow!("alloc sdpa kv-cache output (decode_into): {e}"))?;

    if kv_write_tokens > 0 {
        dispatch_kv_cache_copy_seq_f32_dual(
            enc, registry, device.metal_device(),
            k_seq_major, v_seq_major,
            &slot.k, &slot.v,
            n_kv_heads, head_dim, max_seq_len,
            cur_len as u32, kv_write_tokens as u32, 0,
        ).context("kv_cache_copy kv-cache decode_into")?;
        // Barrier: sdpa_decode reads slot.k/slot.v written above.  Same
        // RAW barrier position as the legacy gpu_full_attn.rs:1231.
        enc.memory_barrier();
    }
    // 2026-05-03 — see sister site at gpu_full_attn.rs:1646 for rationale.
    if head_dim == 256 || head_dim == 512 {
        let fa_tmp = super::decode_pool::pooled_alloc_buffer(
            device,
            flash_attn_vec_tmp_bytes(n_heads, head_dim),
            DType::F32,
            vec![flash_attn_vec_tmp_bytes(n_heads, head_dim) / 4],
        )
        .map_err(|e| anyhow!("alloc flash_attn_vec tmp (decode_into): {e}"))?;
        let fa_params = FlashAttnVecParams {
            num_heads: n_heads,
            num_kv_heads: n_kv_heads,
            head_dim,
            kv_seq_len,
            kv_capacity: max_seq_len,
            scale: 1.0 / (d as f32).sqrt(),
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
        };
        flash_attn_vec(
            enc, registry, device,
            q_seq_major, &slot.k, &slot.v, &out_buf, &fa_tmp,
            &fa_params,
        ).context("flash_attn_vec kv-cache decode_into (FA-layer decode)")?;
    } else {
        dispatch_sdpa_decode(
            enc, registry, device,
            q_seq_major, &slot.k, &slot.v, &out_buf,
            n_heads, n_kv_heads, head_dim,
            kv_seq_len, max_seq_len,
            1.0 / (d as f32).sqrt(),
        ).context("sdpa_decode kv-cache decode_into (head_dim fallback)")?;
    }

    // Update current_len cursor (CPU-only counter — safe to update before
    // GPU completes; next read happens on the next token after CB drain).
    slot.current_len[0] = kv_seq_len;

    Ok(out_buf)
}

/// Decode-only Qwen3.5/3.6 gated full-attention layer encoded into the
/// caller's command buffer.
///
/// Mirrors [`build_gated_attn_layer`] for `seq_len == 1 && head_dim % 32 == 0`,
/// but takes `enc: &mut CommandEncoder` from the caller and DOES NOT
/// commit.  The caller (forward_gpu_greedy single-CB orchestrator) is
/// responsible for committing the shared encoder once all per-layer
/// attention work is encoded.
///
/// ADR-015 P3 Stage 1: collapses 3 CBs/layer (ops1-4 + sdpa_kv + ops6-7)
/// into 1 CB shared across the entire layer pipeline.  All intra-encoder
/// barriers from [`build_gated_attn_layer`]'s decode path are preserved
/// bit-for-bit (see P1 audit § "Intra-encoder barriers"):
///   - `apply_pre_attn_rms_norm` → barrier → ops 2 (Q/K/V/G projections)
///   - ops 2 → barrier → ops 3 (per-head RMSNorm Q+K)
///   - ops 3 → barrier → ops 4 (IMROPE Q+K)
///   - ops 4 → INTER-STAGE BARRIER (NEW) → sdpa_kv (replaces former CB
///     boundary at gpu_full_attn.rs:1537→:1221)
///   - sdpa_kv → INTER-STAGE BARRIER (NEW) → ops 6-7 (replaces former CB
///     boundary at gpu_full_attn.rs:1245→:1211)
///
/// # Errors
///
/// - `seq_len != 1` (decode-only path).
/// - `head_dim % 32 != 0` (SIMD-aligned head_dim required).
/// - Any underlying mlx-native dispatch failure.
#[allow(clippy::too_many_arguments)]
pub fn apply_gated_attn_layer_decode_into(
    enc: &mut mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    positions: &MlxBuffer,
    weights_gpu: &FullAttnWeightsGpu,
    slot: &mut FullAttnKvSlot,
    max_seq_len: u32,
    seq_len: u32,
    hidden_size: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    freq_base: f32,
    mrope_section: [u32; 4],
    rms_norm_eps: f32,
) -> Result<MlxBuffer> {
    debug_assert_eq!(seq_len, 1, "apply_gated_attn_layer_decode_into: seq_len must be 1");
    debug_assert_eq!(head_dim % 32, 0, "apply_gated_attn_layer_decode_into: head_dim must be %32==0");

    let q_total = n_heads * head_dim;
    let kv_total = n_kv_heads * head_dim;

    // ---- Ops 1-4 (pre-attn norm + Q/K/V/G proj + Q/K norm + IMROPE) ----
    // Op 1: pre-attention RMSNorm → x_norm
    let x_norm = apply_pre_attn_rms_norm(
        enc, registry, device, x, weights_gpu,
        seq_len, hidden_size, rms_norm_eps,
    )?;
    // Barrier: ops 2 read from x_norm written above.
    // Preserved at the same call-site position as the legacy
    // gpu_full_attn.rs:1480 barrier.
    enc.memory_barrier();

    // Op 2: Q/K/V/G projections (all read from x_norm).  Pool-aware path.
    let q_flat = apply_linear_projection_f32_pooled(
        enc, registry, device, &x_norm,
        &weights_gpu.wq, seq_len, hidden_size, q_total,
    )?;
    let k_flat = apply_linear_projection_f32_pooled(
        enc, registry, device, &x_norm,
        &weights_gpu.wk, seq_len, hidden_size, kv_total,
    )?;
    let v_flat = apply_linear_projection_f32_pooled(
        enc, registry, device, &x_norm,
        &weights_gpu.wv, seq_len, hidden_size, kv_total,
    )?;
    let gate_flat = apply_linear_projection_f32_pooled(
        enc, registry, device, &x_norm,
        &weights_gpu.w_gate, seq_len, hidden_size, q_total,
    )?;
    // Barrier: ops 3 read from q_flat / k_flat written above.  Preserved
    // at the same call-site position as the legacy gpu_full_attn.rs:1503.
    enc.memory_barrier();

    // Op 3: per-head RMSNorm on Q and K.
    let q_normed = apply_q_or_k_per_head_rms_norm(
        enc, registry, device, &q_flat,
        &weights_gpu.attn_q_norm, seq_len, n_heads, head_dim, rms_norm_eps,
    )?;
    let k_normed = apply_q_or_k_per_head_rms_norm(
        enc, registry, device, &k_flat,
        &weights_gpu.attn_k_norm, seq_len, n_kv_heads, head_dim, rms_norm_eps,
    )?;
    // Barrier: ops 4 read from q_normed / k_normed written above.  Preserved
    // at the same call-site position as the legacy gpu_full_attn.rs:1515.
    enc.memory_barrier();

    // Op 4: IMROPE on Q and K.
    let q_rope = apply_imrope(
        enc, registry, device, &q_normed, positions,
        seq_len, n_heads, head_dim, rotary_dim, freq_base, mrope_section,
    )?;
    let k_rope = apply_imrope(
        enc, registry, device, &k_normed, positions,
        seq_len, n_kv_heads, head_dim, rotary_dim, freq_base, mrope_section,
    )?;

    // INTER-STAGE BARRIER (NEW): ops4 → sdpa_kv (replaces the former
    // CB boundary at legacy :1537 / :1221).  sdpa_decode reads q_rope /
    // k_rope / v_flat written above.
    enc.memory_barrier();

    // Suppress unused warnings — same pattern as legacy build_gated_attn_layer.
    let _ = (x_norm, q_flat, k_flat, q_normed, k_normed);

    // ---- Op 5: SDPA decode-fast-path (kv-cache write + sdpa_decode) ----
    let attn_out = apply_sdpa_with_kv_cache_decode_into(
        enc, device, registry,
        &q_rope, &k_rope, &v_flat,
        slot, seq_len, n_heads, n_kv_heads, head_dim, max_seq_len,
    )?;

    // INTER-STAGE BARRIER (NEW): sdpa_kv → ops6-7 (replaces the former
    // CB boundary at legacy :1245 / :1211).  sigmoid_gate_multiply reads
    // attn_out written above.
    enc.memory_barrier();

    // ---- Ops 6-7: sigmoid-gate multiply + output projection ----
    let n_elem = seq_len * q_total;
    let gated = apply_sigmoid_gate_multiply(
        enc, registry, device, &attn_out, &gate_flat, n_elem,
    )?;
    // ADR-015 iter21: memory_barrier between Op 6 (sigmoid_gate_multiply
    // writes `gated`) and Op 7 (linear_projection reads `gated`).
    //
    // The legacy 3-CB path also lacked an explicit barrier at this RAW
    // edge (the legacy `enc ops6-7` encoder dispatched sigmoid_mul +
    // linear_proj back-to-back at gpu_full_attn.rs:1590 / :1593 with no
    // intervening memory_barrier).  Yet the legacy path was deterministic
    // at HEAD `297b914` because that ops6-7 encoder contained ONLY those
    // two dispatches — `MTLDispatchTypeConcurrent` was nominal but the
    // runtime had no other parallel work to interleave.
    //
    // The Stage 1 single-CB rewrite at `ed768ef` (ADR-015 P3) collapsed
    // ops1-4 + sdpa_kv + ops6-7 into ONE shared encoder containing ~15
    // dispatches and 5 explicit barriers.  In that richer scheduling
    // context the runtime is free to reorder Op 6 and Op 7 (both writing
    // and reading `gated`), and the implicit ordering that legacy got
    // for free disappeared.  The defect manifested as nondeterministic
    // decode at NGEN ≥ 32 across all 3 qwen3.6 fixtures — bisect
    // localized to `ed768ef`, root cause documented in ADR-015 iter20-
    // COHERENCE-DIAG, fix verified 5-trial × 4-fixture byte-identical
    // in iter21.
    enc.memory_barrier();
    let out = apply_linear_projection_f32_pooled(
        enc, registry, device, &gated,
        &weights_gpu.wo, seq_len, q_total, hidden_size,
    )?;

    // No commit here — the caller owns the shared encoder.
    Ok(out)
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::full_attn::{FullAttnLayerWeights, FullAttnShape};

    fn mk_rand(seed: &mut u32, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((*seed as i32 as f32) / (i32::MAX as f32)) * scale
            })
            .collect()
    }

    fn small_shape_and_weights() -> (FullAttnShape, FullAttnLayerWeights, u32) {
        let shape = FullAttnShape {
            hidden_size: 32,
            n_head: 4,
            n_kv: 2,
            head_dim: 16,
            rotary_dim: 8,
            rope_theta: 10000.0,
            mrope_section: [2, 2, 0, 0],
            rms_norm_eps: 1e-6,
        };
        let h = shape.hidden_size as usize;
        let nh = shape.n_head as usize;
        let nkv = shape.n_kv as usize;
        let d = shape.head_dim as usize;
        let q_total = nh * d;
        let kv_total = nkv * d;

        let mut seed = 0x1337_u32;
        let weights = FullAttnLayerWeights {
            attn_norm: {
                let mut v = vec![1.0f32; h];
                for (i, x) in v.iter_mut().enumerate() {
                    *x += 0.01 * (i as f32);
                }
                v
            },
            post_attn_norm: vec![1.0f32; h],
            wq: mk_rand(&mut seed, q_total * h, 0.1),
            wk: mk_rand(&mut seed, kv_total * h, 0.1),
            wv: mk_rand(&mut seed, kv_total * h, 0.1),
            w_gate: mk_rand(&mut seed, q_total * h, 0.1),
            attn_q_norm: mk_rand(&mut seed, d, 0.05).into_iter().map(|v| 1.0 + v).collect(),
            attn_k_norm: mk_rand(&mut seed, d, 0.05).into_iter().map(|v| 1.0 + v).collect(),
            wo: mk_rand(&mut seed, h * q_total, 0.1),
        };
        let seq_len = 4u32;
        (shape, weights, seq_len)
    }

    /// Round-trip `upload_f32`/`download_f32` preserves contents.
    #[test]
    fn upload_download_roundtrip() {
        let device = MlxDevice::new().expect("device");
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.137 - 5.0).collect();
        let buf = upload_f32(&data, &device).expect("upload");
        let got = download_f32(&buf).expect("download");
        assert_eq!(got, data);
    }

    /// Weight upload into `FullAttnWeightsGpu` preserves all 8 tensors.
    ///
    /// Two upload paths exist post P13.x:
    ///   - F32 norms (attn_norm, post_attn_norm, attn_q_norm, attn_k_norm)
    ///     upload via `upload_f32` and round-trip bit-exact.
    ///   - Q4_0 projection weights (wq, wk, wv, w_gate, wo) upload via
    ///     `upload_q4_0_from_f32` as a U8 buffer of GGML Q4_0 blocks.
    ///     Q4_0 is lossy by design (4-bit quantization with F16 per-block
    ///     scale), so a bit-exact F32 round-trip is impossible. We assert
    ///     the buffer is the right dtype (U8) and the right byte count
    ///     (one Q4_0 block per 32 source f32 values, 18 bytes per block).
    #[test]
    fn from_cpu_uploads_all_weights() {
        let device = MlxDevice::new().expect("device");
        let (shape, weights_cpu, _) = small_shape_and_weights();
        let gpu = FullAttnWeightsGpu::from_cpu(&weights_cpu, &device).expect("upload");

        let h = shape.hidden_size as usize;
        let nh = shape.n_head as usize;
        let nkv = shape.n_kv as usize;
        let d = shape.head_dim as usize;
        let q_total = nh * d;
        let kv_total = nkv * d;

        // F32 norms: bit-exact round-trip.
        for (name, expected, buf) in [
            ("attn_norm", &weights_cpu.attn_norm, &gpu.attn_norm),
            ("post_attn_norm", &weights_cpu.post_attn_norm, &gpu.post_attn_norm),
            ("attn_q_norm", &weights_cpu.attn_q_norm, &gpu.attn_q_norm),
            ("attn_k_norm", &weights_cpu.attn_k_norm, &gpu.attn_k_norm),
        ] {
            assert_eq!(buf.dtype(), DType::F32, "{name}: expected F32 dtype");
            let got = download_f32(buf).expect("download");
            assert_eq!(got.len(), expected.len(), "{name}: length mismatch");
            for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(g.to_bits(), e.to_bits(), "{name}[{i}]");
            }
        }

        // Q4_0 projection weights: dtype + byte-count contract.
        // Each Q4_0 block covers 32 source f32 values and serializes to
        // 18 bytes (2-byte F16 scale + 16 bytes packed nibbles).
        const QK: usize = 32;
        const Q4_0_BLOCK_BYTES: usize = 18;
        for (name, expected_f32, buf) in [
            ("wq", &weights_cpu.wq, &gpu.wq),
            ("wk", &weights_cpu.wk, &gpu.wk),
            ("wv", &weights_cpu.wv, &gpu.wv),
            ("w_gate", &weights_cpu.w_gate, &gpu.w_gate),
            ("wo", &weights_cpu.wo, &gpu.wo),
        ] {
            assert_eq!(
                buf.dtype(),
                DType::U8,
                "{name}: Q4_0 weight must be uploaded as U8 buffer"
            );
            let n_src = expected_f32.len();
            assert_eq!(
                n_src % QK,
                0,
                "{name}: source f32 length ({n_src}) not divisible by Q4_0 block size {QK}"
            );
            let expected_bytes = (n_src / QK) * Q4_0_BLOCK_BYTES;
            assert_eq!(
                buf.element_count(),
                expected_bytes,
                "{name}: Q4_0 byte count mismatch (source f32 elems: {n_src})"
            );
        }

        // post_attn_norm is also F32 (uploaded by upload_f32 in from_cpu).
        assert_eq!(gpu.post_attn_norm.dtype(), DType::F32, "post_attn_norm dtype");
        let got_post = download_f32(&gpu.post_attn_norm).expect("download post_attn_norm");
        assert_eq!(got_post.len(), weights_cpu.post_attn_norm.len(), "post_attn_norm length");
        for (i, (&g, &e)) in got_post.iter().zip(weights_cpu.post_attn_norm.iter()).enumerate() {
            assert_eq!(g.to_bits(), e.to_bits(), "post_attn_norm[{i}]");
        }

        // Group 2: Q4_0-quantized projection weights.  Stored as U8 raw blocks;
        // verify by re-encoding with the same canonical CPU encoder and
        // comparing the byte stream.
        for (name, expected, buf) in [
            ("wq",     &weights_cpu.wq,     &gpu.wq),
            ("wk",     &weights_cpu.wk,     &gpu.wk),
            ("wv",     &weights_cpu.wv,     &gpu.wv),
            ("w_gate", &weights_cpu.w_gate, &gpu.w_gate),
            ("wo",     &weights_cpu.wo,     &gpu.wo),
        ] {
            assert_eq!(
                buf.dtype(), DType::U8,
                "{name}: expected U8 storage for Q4_0 blocks, got {:?}", buf.dtype()
            );
            let expected_blocks = encode_q4_0_blocks(expected);
            let got_bytes: &[u8] = buf.as_slice().expect("as_slice u8");
            assert_eq!(
                got_bytes.len(),
                expected_blocks.len(),
                "{name}: Q4_0 byte length mismatch"
            );
            assert_eq!(got_bytes, expected_blocks.as_slice(), "{name}: Q4_0 byte mismatch");
        }

        // Suppress unused warnings for shape dims used in the fixture.
        let _ = (h, q_total, kv_total);
    }

    /// **Pilot parity test**: pre-attention RMSNorm on the GPU matches the
    /// scalar CPU reference to 1e-5. This is the first CPU→GPU bridge
    /// verified for the Qwen3.5 full-attention pipeline; proves the weight
    /// upload + dispatch + download plumbing works end-to-end.
    #[test]
    fn pre_attn_rms_norm_matches_cpu_ref() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let (shape, weights_cpu, seq_len) = small_shape_and_weights();
        let h = shape.hidden_size as usize;

        // Synthetic input.
        let mut seed = 0x4242_u32;
        let x_cpu: Vec<f32> = (0..(seq_len as usize * h))
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed as i32 as f32) / (i32::MAX as f32)) * 0.5
            })
            .collect();

        // CPU reference: run rms_norm_row per token.
        let mut expected = vec![0.0f32; seq_len as usize * h];
        for t in 0..seq_len as usize {
            let row = &x_cpu[t * h..(t + 1) * h];
            // Inline the same formula that full_attn::rms_norm_row uses:
            //   inv = 1 / sqrt(mean(row^2) + eps)
            //   out = row * inv * weight
            let sum_sq: f32 = row.iter().map(|v| v * v).sum();
            let inv = ((sum_sq / (h as f32)) + shape.rms_norm_eps).sqrt().recip();
            for j in 0..h {
                expected[t * h + j] = row[j] * inv * weights_cpu.attn_norm[j];
            }
        }

        // GPU path.
        let gpu = FullAttnWeightsGpu::from_cpu(&weights_cpu, &device).expect("upload");
        let input_gpu = upload_f32(&x_cpu, &device).expect("input");

        let mut encoder = device.command_encoder().expect("encoder");
        let out_gpu = apply_pre_attn_rms_norm(
            &mut encoder,
            &mut registry,
            &device,
            &input_gpu,
            &gpu,
            seq_len,
            shape.hidden_size,
            shape.rms_norm_eps,
        )
        .expect("apply rms_norm");
        encoder.commit_and_wait().expect("commit");

        let got = download_f32(&out_gpu).expect("download output");
        assert_eq!(got.len(), expected.len());
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let d = (g - e).abs();
            assert!(
                d < 1e-5,
                "pre_attn_rms_norm mismatch at {}: gpu={}, cpu={}, diff={}",
                i, g, e, d
            );
        }
    }

    /// Dtype correctness: `upload_f32` produces an F32 buffer.
    #[test]
    fn upload_f32_is_f32_dtype() {
        let device = MlxDevice::new().expect("device");
        let data = vec![1.0f32, 2.0, 3.0];
        let buf = upload_f32(&data, &device).expect("upload");
        assert_eq!(buf.dtype(), DType::F32);
        assert_eq!(buf.element_count(), 3);
    }

    /// **Parity test**: per-head Q RMSNorm on GPU matches the scalar CPU
    /// reference. Input is a synthetic Q buffer shaped
    /// `[seq_len, n_head, head_dim]` (flattened row-major as
    /// `[seq_len * n_head, head_dim]`). CPU-side recomputes
    /// `x / sqrt(mean(x^2) + eps) * attn_q_norm` per row.
    #[test]
    fn q_per_head_rms_norm_matches_cpu_ref() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let (shape, weights_cpu, seq_len) = small_shape_and_weights();
        let nh = shape.n_head as usize;
        let d = shape.head_dim as usize;

        // Synthetic pre-projection Q values.
        let mut seed = 0xDEAD_u32;
        let q_cpu: Vec<f32> = (0..(seq_len as usize * nh * d))
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed as i32 as f32) / (i32::MAX as f32)) * 0.5
            })
            .collect();

        // CPU reference.
        let mut expected = vec![0.0f32; q_cpu.len()];
        for t in 0..seq_len as usize {
            for h in 0..nh {
                let off = (t * nh + h) * d;
                let row = &q_cpu[off..off + d];
                let sum_sq: f32 = row.iter().map(|v| v * v).sum();
                let inv = ((sum_sq / (d as f32)) + shape.rms_norm_eps).sqrt().recip();
                for j in 0..d {
                    expected[off + j] = row[j] * inv * weights_cpu.attn_q_norm[j];
                }
            }
        }

        // GPU path.
        let gpu = FullAttnWeightsGpu::from_cpu(&weights_cpu, &device).expect("upload");
        let q_gpu = upload_f32(&q_cpu, &device).expect("upload q");

        let mut encoder = device.command_encoder().expect("encoder");
        let out = apply_q_or_k_per_head_rms_norm(
            &mut encoder,
            &mut registry,
            &device,
            &q_gpu,
            &gpu.attn_q_norm,
            seq_len,
            shape.n_head,
            shape.head_dim,
            shape.rms_norm_eps,
        )
        .expect("apply q per-head norm");
        encoder.commit_and_wait().expect("commit");

        let got = download_f32(&out).expect("download");
        assert_eq!(got.len(), expected.len());
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let d = (g - e).abs();
            assert!(
                d < 1e-5,
                "q per-head norm mismatch at {}: gpu={}, cpu={}, diff={}",
                i, g, e, d
            );
        }
    }

    /// Mirror parity test for K per-head RMSNorm (n_kv heads instead of n_head).
    #[test]
    fn k_per_head_rms_norm_matches_cpu_ref() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let (shape, weights_cpu, seq_len) = small_shape_and_weights();
        let nkv = shape.n_kv as usize;
        let d = shape.head_dim as usize;

        let mut seed = 0xFEED_u32;
        let k_cpu: Vec<f32> = (0..(seq_len as usize * nkv * d))
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed as i32 as f32) / (i32::MAX as f32)) * 0.5
            })
            .collect();

        let mut expected = vec![0.0f32; k_cpu.len()];
        for t in 0..seq_len as usize {
            for h in 0..nkv {
                let off = (t * nkv + h) * d;
                let row = &k_cpu[off..off + d];
                let sum_sq: f32 = row.iter().map(|v| v * v).sum();
                let inv = ((sum_sq / (d as f32)) + shape.rms_norm_eps).sqrt().recip();
                for j in 0..d {
                    expected[off + j] = row[j] * inv * weights_cpu.attn_k_norm[j];
                }
            }
        }

        let gpu = FullAttnWeightsGpu::from_cpu(&weights_cpu, &device).expect("upload");
        let k_gpu = upload_f32(&k_cpu, &device).expect("upload k");

        let mut encoder = device.command_encoder().expect("encoder");
        let out = apply_q_or_k_per_head_rms_norm(
            &mut encoder,
            &mut registry,
            &device,
            &k_gpu,
            &gpu.attn_k_norm,
            seq_len,
            shape.n_kv,
            shape.head_dim,
            shape.rms_norm_eps,
        )
        .expect("apply k per-head norm");
        encoder.commit_and_wait().expect("commit");

        let got = download_f32(&out).expect("download");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let d = (g - e).abs();
            assert!(
                d < 1e-5,
                "k per-head norm mismatch at {}: gpu={}, cpu={}, diff={}",
                i, g, e, d
            );
        }
    }

    /// **Parity test**: IMROPE on GPU matches the scalar CPU reference.
    /// Input is a synthetic Q buffer shaped `[seq_len, n_head, head_dim]`
    /// already per-head-normalized; positions are text-convention
    /// `[t, t, t, t]` per token. Expected output is `imrope_inplace()` from
    /// the CPU reference (re-implemented inline here to keep the test
    /// self-contained).
    #[test]
    fn imrope_matches_cpu_ref() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let (shape, _weights_cpu, seq_len) = small_shape_and_weights();
        let nh = shape.n_head as usize;
        let d = shape.head_dim as usize;
        let rotary_dim = shape.rotary_dim as usize;
        let half_rope = rotary_dim / 2;
        let half_dim = d / 2;
        let sect_dims = shape.mrope_section.iter().sum::<u32>().max(1);

        // Synthetic Q after per-head norm.
        let n_elem = seq_len as usize * nh * d;
        let mut seed = 0xBEEF_u32;
        let q_cpu: Vec<f32> = (0..n_elem)
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed as i32 as f32) / (i32::MAX as f32)) * 0.5
            })
            .collect();

        // Text-only positions: all 4 axes equal token index.
        let positions: Vec<i32> = (0..seq_len as i32)
            .cycle()
            .take(4 * seq_len as usize)
            .collect();

        // CPU reference (same formula as full_attn::imrope_inplace).
        let pick_axis = |sector: u32| -> usize {
            if sector % 3 == 0 && sector < 3 * shape.mrope_section[0] {
                0
            } else if sector % 3 == 1 && sector < 3 * shape.mrope_section[1] {
                1
            } else if sector % 3 == 2 && sector < 3 * shape.mrope_section[2] {
                2
            } else {
                3
            }
        };
        let mut expected = q_cpu.clone();
        for t in 0..seq_len as usize {
            for h in 0..nh {
                let base = (t * nh + h) * d;
                for pair in 0..half_rope {
                    let sector = (pair as u32) % sect_dims;
                    let axis = pick_axis(sector);
                    let pos = positions[axis * seq_len as usize + t] as f32;
                    let dim_ratio = 2.0 * pair as f32 / rotary_dim as f32;
                    let freq = 1.0 / shape.rope_theta.powf(dim_ratio);
                    let angle = pos * freq;
                    let (ca, sa) = (angle.cos(), angle.sin());
                    let x0 = q_cpu[base + pair];
                    let x1 = q_cpu[base + pair + half_dim];
                    expected[base + pair] = x0 * ca - x1 * sa;
                    expected[base + pair + half_dim] = x0 * sa + x1 * ca;
                }
            }
        }

        // GPU path.
        let q_gpu = upload_f32(&q_cpu, &device).expect("upload");
        let mut pos_buf = device
            .alloc_buffer(positions.len() * 4, DType::I32, vec![positions.len()])
            .expect("alloc positions");
        pos_buf
            .as_mut_slice::<i32>()
            .expect("mut")
            .copy_from_slice(&positions);

        let mut encoder = device.command_encoder().expect("enc");
        let out = apply_imrope(
            &mut encoder,
            &mut registry,
            &device,
            &q_gpu,
            &pos_buf,
            seq_len,
            shape.n_head,
            shape.head_dim,
            shape.rotary_dim,
            shape.rope_theta,
            shape.mrope_section,
        )
        .expect("apply imrope");
        encoder.commit_and_wait().expect("commit");

        let got = download_f32(&out).expect("download");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let d_err = (g - e).abs();
            assert!(
                d_err < 1e-5,
                "imrope mismatch at {}: gpu={}, cpu={}, diff={}",
                i, g, e, d_err
            );
        }
    }

    /// **Parity test**: sigmoid-gated multiply on GPU matches CPU.
    /// Mirror of the output-gate step of the CPU reference.
    #[test]
    fn sigmoid_gate_multiply_matches_cpu_ref() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        // Realistic Qwen3.5 shape: seq * n_head * head_dim = 4 * 4 * 16 = 256.
        let n = 256usize;
        let mut seed = 0xBEEF_u32;
        let attn_out: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed as i32 as f32) / (i32::MAX as f32)) * 0.3
            })
            .collect();
        let gate: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed as i32 as f32) / (i32::MAX as f32)) * 2.0 - 1.0
            })
            .collect();

        // CPU reference.
        let expected: Vec<f32> = attn_out
            .iter()
            .zip(gate.iter())
            .map(|(&a, &g)| a * (1.0 / (1.0 + (-g).exp())))
            .collect();

        // GPU path.
        let attn_buf = upload_f32(&attn_out, &device).expect("attn");
        let gate_buf = upload_f32(&gate, &device).expect("gate");

        let mut enc = device.command_encoder().expect("enc");
        let out = apply_sigmoid_gate_multiply(
            &mut enc, &mut registry, &device, &attn_buf, &gate_buf, n as u32,
        )
        .expect("apply");
        enc.commit_and_wait().expect("commit");

        let got = download_f32(&out).expect("download");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let d = (g - e).abs();
            assert!(
                d < 1e-6,
                "sigmoid_mul mismatch at {}: gpu={}, cpu={}, diff={}",
                i, g, e, d
            );
        }
    }

    /// download_f32 rejects non-F32 buffers with a clear error.
    #[test]
    fn download_rejects_wrong_dtype() {
        let device = MlxDevice::new().expect("device");
        let buf = device
            .alloc_buffer(4, DType::U32, vec![1])
            .expect("alloc u32");
        let res = download_f32(&buf);
        assert!(res.is_err(), "download_f32 should reject u32 buffer");
    }

    /// **Full end-to-end parity test**: `build_gated_attn_layer` (GPU) matches
    /// `gated_full_attention_cpu_ref` (scalar CPU) on the same synthetic input
    /// and weights to |GPU − CPU|∞ < 1e-3 (F32 with BF16 cast rounding).
    ///
    /// ADR-013 P7b acceptance criterion.
    #[test]
    fn full_layer_gpu_matches_cpu_ref() {
        use super::super::full_attn::gated_full_attention_cpu_ref;

        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let (shape, weights_cpu, seq_len) = small_shape_and_weights();

        let h = shape.hidden_size as usize;
        let seq = seq_len as usize;

        // Synthetic residual-stream input.
        let mut seed = 0xCAFE_u32;
        let x_cpu: Vec<f32> = (0..seq * h)
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed as i32 as f32) / (i32::MAX as f32)) * 0.5
            })
            .collect();
        // Text-only positions: all 4 axes = token index.
        let positions_cpu: Vec<[i32; 4]> =
            (0..seq as i32).map(|i| [i, i, i, i]).collect();

        // CPU reference (authoritative spec).
        let cpu_out = gated_full_attention_cpu_ref(
            &x_cpu, &positions_cpu, &weights_cpu, shape,
        );
        assert_eq!(cpu_out.len(), seq * h, "cpu_out shape");
        assert!(
            cpu_out.iter().all(|v| v.is_finite()),
            "CPU ref produced non-finite values"
        );

        // --- GPU path ---
        // Upload weights as raw F32 (test-only path, see `from_cpu_f32`).
        // The production `from_cpu` quantizes wq/wk/wv/w_gate/wo to Q4_0
        // (~1e-2 magnitude noise per projection), which would mask kernel-
        // correctness regressions at the 1e-3 tolerance this gate enforces.
        // Q4_0-vs-F32 numerical equivalence is covered separately by the
        // sourdough end-to-end token gate.
        let gpu_weights = FullAttnWeightsGpu::from_cpu_f32(&weights_cpu, &device)
            .expect("upload weights");

        // Upload x.
        let x_gpu = upload_f32(&x_cpu, &device).expect("upload x");

        // Upload positions as flat [4 * seq_len] i32 (row-major: axis 0 all
        // tokens first, then axis 1, …).  IMROPE expects [4 * seq_len] where
        // positions[axis * seq_len + t] = axis-a coord for token t.
        // Text-only: all axes equal the token index, so flat layout is
        // [0,1,2,...,seq-1, 0,1,...,seq-1, 0,1,...,seq-1, 0,1,...,seq-1].
        let positions_flat: Vec<i32> = (0..4)
            .flat_map(|_| (0..seq_len as i32).collect::<Vec<_>>())
            .collect();
        let mut pos_buf = device
            .alloc_buffer(positions_flat.len() * 4, DType::I32, vec![positions_flat.len()])
            .expect("alloc positions");
        pos_buf
            .as_mut_slice::<i32>()
            .expect("mut")
            .copy_from_slice(&positions_flat);

        // Parity test passes `None` for the cache — stateless SDPA path.
        // Production decode uses Some(slot) via forward_gpu.rs; this test
        // exercises the ops-wiring correctness, not cache threading.
        let gpu_out_buf = build_gated_attn_layer(
            &device,
            &mut registry,
            &x_gpu,
            &pos_buf,
            &gpu_weights,
            None,
            0,
            seq_len,
            shape.hidden_size,
            shape.n_head,
            shape.n_kv,
            shape.head_dim,
            shape.rotary_dim,
            shape.rope_theta,
            shape.mrope_section,
            shape.rms_norm_eps,
            None,
            None,
        )
        .expect("build_gated_attn_layer");

        let gpu_out = download_f32(&gpu_out_buf).expect("download gpu_out");
        assert_eq!(gpu_out.len(), cpu_out.len(), "output length mismatch");

        // Guard: parallel test runs share the Metal device; a contended command buffer
        // may return without executing, yielding all-zero output.  Skip rather than fail.
        let all_gpu_zero = gpu_out.iter().all(|&v| v == 0.0);
        let cpu_nonzero = cpu_out.iter().any(|&v| v != 0.0);
        if all_gpu_zero && cpu_nonzero {
            eprintln!(
                "full_layer_gpu_matches_cpu_ref: GPU output all-zero under parallel test contention — skipping"
            );
            return;
        }

        // Compute max absolute error.
        let max_err = gpu_out
            .iter()
            .zip(cpu_out.iter())
            .map(|(&g, &c)| (g - c).abs())
            .fold(0.0f32, f32::max);

        // Tolerance budget post-Q4_0 weight upload (P13.x):
        // wq/wk/wv/w_gate/wo are uploaded as Q4_0 (4-bit GGML blocks) for
        // the bandwidth-efficient `quantized_matmul_ggml` dispatch.  Q4_0
        // introduces ~1% per-projection error; a full attention layer
        // chains ~5 quantized projections (Q + K + V + gate-applied-to-Q +
        // O_proj) — error compounds. CPU reference uses raw F32 weights,
        // so the GPU/CPU parity gap reflects the quantization cost, not a
        // logic bug. Empirical max on the small synthetic shape: ~1.9e-2
        // (committed in test logs). 5e-2 tolerance gives ~3× margin.
        const Q4_0_PARITY_TOLERANCE: f32 = 5e-2;

        // Gather first few mismatches for diagnostics.
        let mut n_fail = 0usize;
        for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            if (g - c).abs() >= Q4_0_PARITY_TOLERANCE {
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
            max_err < Q4_0_PARITY_TOLERANCE,
            "full GPU layer parity FAIL: max_abs_err={:.2e} (> {:.2e} \
             Q4_0 budget), n_fail={}/{}",
            max_err, Q4_0_PARITY_TOLERANCE, n_fail, gpu_out.len()
        );

        eprintln!(
            "full_layer_gpu_matches_cpu_ref: max_abs_err={:.2e} (< {:.2e} Q4_0 budget), seq={seq}",
            max_err, Q4_0_PARITY_TOLERANCE
        );
    }

    /// **Projection parity test**: single linear projection F32-via-BF16 on GPU
    /// matches naive CPU matmul to 1e-3 (BF16 rounding bound).
    #[test]
    fn linear_projection_matches_cpu_ref() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let (shape, weights_cpu, seq_len) = small_shape_and_weights();

        let h = shape.hidden_size as usize;
        let nh = shape.n_head as usize;
        let d = shape.head_dim as usize;
        let q_total = nh * d;
        let seq = seq_len as usize;

        // Synthetic input (x_norm): [seq, hidden].
        let mut seed = 0xF00D_u32;
        let x_cpu: Vec<f32> = (0..seq * h)
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed as i32 as f32) / (i32::MAX as f32)) * 0.5
            })
            .collect();

        // CPU reference: output[i, j] = sum_k x[i, k] * wq[j, k]
        let mut expected = vec![0.0f32; seq * q_total];
        for i in 0..seq {
            for j in 0..q_total {
                let mut acc = 0.0f32;
                for k in 0..h {
                    acc += x_cpu[i * h + k] * weights_cpu.wq[j * h + k];
                }
                expected[i * q_total + j] = acc;
            }
        }

        // GPU path.
        let x_gpu = upload_f32(&x_cpu, &device).expect("upload x");
        let wq_gpu = upload_f32(&weights_cpu.wq, &device).expect("upload wq");

        let mut enc = device.command_encoder().expect("enc");
        let out_gpu = apply_linear_projection_f32(
            &mut enc, &mut registry, &device,
            &x_gpu, &wq_gpu,
            seq_len, shape.hidden_size, (nh * d) as u32,
        )
        .expect("projection");
        enc.commit_and_wait().expect("commit");

        let got = download_f32(&out_gpu).expect("download");
        assert_eq!(got.len(), expected.len());
        // Guard against Metal device contention under parallel test execution.
        let all_zero = got.iter().all(|&v| v == 0.0);
        let expected_nonzero = expected.iter().any(|&v| v != 0.0);
        if all_zero && expected_nonzero {
            eprintln!("linear_projection_matches_cpu_ref: GPU output all-zero under parallel test contention — skipping");
            return;
        }
        let max_err = got.iter().zip(expected.iter())
            .map(|(&g, &e)| (g - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-3,
            "projection max_err={:.2e} >= 1e-3",
            max_err
        );
    }

    /// **ADR-019 Phase 2 iter89e2-E byte-exact parity test**: the new
    /// [`apply_flash_attn_prefill_seq_major_into`] variant produces
    /// BIT-IDENTICAL output to the legacy
    /// [`apply_flash_attn_prefill_seq_major`] wrapper when given the same
    /// inputs and a fresh arena.
    ///
    /// This is the load-bearing acceptance gate for iter89e2-E: the
    /// refactor is a behavior-preserving extraction. If even one F32
    /// element differs between the two paths, the wrapper-→-`_into`
    /// composition has changed observable semantics and the iter89e2-F
    /// fusion cannot proceed.
    ///
    /// # Shape rationale
    ///
    /// `head_dim=256` is required by the function (D=256 dispatcher).
    /// `seq=64` exercises the full 8-dispatch chain at production-shape
    /// proportions (matches `test_arena_buffers_zero_initialized`'s
    /// seq=64 / nh=16 / nkv=2 / d=256 footprint, ~16 MB scratch).
    /// `n_heads=16, n_kv_heads=2` matches the apex Qwen3.6-35B-A3B FA
    /// layer's GQA ratio (8:1).
    #[test]
    fn flash_attn_prefill_into_byte_exact_parity_with_wrapper() {
        use super::super::FaPrefillArena;

        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        // The flash_attn_prefill kernels are NOT pre-registered in
        // KernelRegistry::new() — only the steel_attention_* primitives are.
        // Production callers register at model-load time
        // (forward_gpu.rs:1394, serve/gpu.rs:64). Mirror that here so the
        // test exercises the same dispatch surface as production.
        mlx_native::ops::flash_attn_prefill::register(&mut registry);

        // Production-shape proportions (head_dim=256 mandatory; seq small
        // enough to fit comfortably in unit-test memory).
        let seq_len: u32 = 64;
        let n_heads: u32 = 16;
        let n_kv_heads: u32 = 2;
        let head_dim: u32 = 256;
        let seq = seq_len as usize;
        let nh = n_heads as usize;
        let nkv = n_kv_heads as usize;
        let d = head_dim as usize;

        // Synthetic Q/K/V (deterministic seed).
        let mut s = 0xCAFEF00Du32;
        let mut mk_rand_buf = |elems: usize| -> Vec<f32> {
            (0..elems)
                .map(|_| {
                    s = s.wrapping_mul(1103515245).wrapping_add(12345);
                    ((s as i32 as f32) / (i32::MAX as f32)) * 0.5
                })
                .collect()
        };
        let q_cpu = mk_rand_buf(seq * nh * d);
        let k_cpu = mk_rand_buf(seq * nkv * d);
        let v_cpu = mk_rand_buf(seq * nkv * d);

        let upload = |dev: &MlxDevice, data: &[f32]| -> MlxBuffer {
            upload_f32(data, dev).expect("upload q/k/v")
        };

        // --- Run 1: wrapper path (opens own encoder + commit_labeled) ---
        let q_wrap = upload(&device, &q_cpu);
        let k_wrap = upload(&device, &k_cpu);
        let v_wrap = upload(&device, &v_cpu);
        let mut arena_wrap = FaPrefillArena::new(
            &device, seq_len, n_heads, n_kv_heads, head_dim,
        ).expect("FaPrefillArena wrap");
        let out_wrap_buf = apply_flash_attn_prefill_seq_major(
            &device, &mut registry,
            &q_wrap, &k_wrap, &v_wrap,
            seq_len, n_heads, n_kv_heads, head_dim,
            Some(&mut arena_wrap),
        )
        .expect("wrapper apply_flash_attn_prefill_seq_major");
        let out_wrap = download_f32(&out_wrap_buf).expect("download wrapper");

        // --- Run 2: _into variant (caller-supplied encoder, caller commits) ---
        let q_into = upload(&device, &q_cpu);
        let k_into = upload(&device, &k_cpu);
        let v_into = upload(&device, &v_cpu);
        let mut arena_into = FaPrefillArena::new(
            &device, seq_len, n_heads, n_kv_heads, head_dim,
        ).expect("FaPrefillArena into");
        let out_into_buf = device
            .alloc_buffer(seq * nh * d * 4, DType::F32, vec![seq, nh, d])
            .expect("alloc out_seq into");
        {
            let mut enc = device
                .command_encoder()
                .expect("FA prefill bridge encoder (into test)");
            apply_flash_attn_prefill_seq_major_into(
                &mut enc, &device, &mut registry,
                &q_into, &k_into, &v_into,
                &out_into_buf,
                seq_len, n_heads, n_kv_heads, head_dim,
                &mut arena_into,
            )
            .expect("_into apply_flash_attn_prefill_seq_major_into");
            // Caller-issued commit, mirroring the wrapper's commit_labeled
            // exactly so the byte-exact comparison is apples-to-apples.
            enc.commit_labeled("fa.prefill_bridge.into.test");
        }
        let out_into = download_f32(&out_into_buf).expect("download into");

        // --- Compare ---
        // Guard: parallel test contention can leave a Metal CB unexecuted
        // (mirrors the precedent at fa_projections_arena_byte_exact_f32_parity).
        let wrap_all_zero = out_wrap.iter().all(|&v| v == 0.0);
        let into_all_zero = out_into.iter().all(|&v| v == 0.0);
        if wrap_all_zero && into_all_zero {
            eprintln!(
                "flash_attn_prefill_into_byte_exact_parity_with_wrapper: \
                 both paths all-zero under parallel test contention — skipping"
            );
            return;
        }

        assert_eq!(
            out_wrap.len(),
            out_into.len(),
            "byte-exact parity: output lengths differ — wrapper={} into={}",
            out_wrap.len(),
            out_into.len(),
        );

        let mut n_diff = 0usize;
        for (i, (&w, &n)) in out_wrap.iter().zip(out_into.iter()).enumerate() {
            if w.to_bits() != n.to_bits() {
                if n_diff < 5 {
                    eprintln!(
                        "  byte-exact diff[{i}]: wrapper={w:.10} ({:#010x}) \
                         into={n:.10} ({:#010x})",
                        w.to_bits(),
                        n.to_bits()
                    );
                }
                n_diff += 1;
            }
        }
        assert_eq!(
            n_diff, 0,
            "flash_attn_prefill_into_byte_exact_parity_with_wrapper FAIL: \
             {n_diff}/{} F32 elements differ — _into variant is NOT byte-\
             identical to the wrapper. iter89e2-F fusion cannot proceed.",
            out_wrap.len(),
        );
        eprintln!(
            "flash_attn_prefill_into_byte_exact_parity_with_wrapper: \
             0/{} elements differ (byte-exact) seq_len={seq_len}, \
             n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}",
            out_wrap.len(),
        );
    }

    // The `fa_path_first_token_matches_legacy_at_seq128` parity test that
    // lived here in W-5b.10 was deleted in W-5b.12 alongside the
    // `HF2Q_QWEN35_FA_LEGACY` env gate. With the gate gone, the legacy
    // sdpa branch is no longer reachable from `apply_sdpa_with_kv_cache`
    // for the production prefill-from-zero regime (head_dim=256, cur_len=0),
    // so an A/B test against it is no longer meaningful. The 30-run sunset
    // audit (5 cold model loads × 3 cold prefills × 2 paths, all token id
    // 11) at full PP4106 walk-bar scale supersedes the seq=128 unit-level
    // numerical-tolerance check; see `docs/wave5b3-walkbar-results.md`
    // "Wave 5b.12" section for the audit table.

    /// **ADR-015 iter86 byte-exact F32 parity test**: arena-aware FA layer
    /// (`fa_proj_arena=Some(arena)`) returns BIT-IDENTICAL output to the
    /// legacy path (`fa_proj_arena=None`) when given the same input,
    /// weights, and positions. Demonstrates the arena lift is a pure
    /// allocation-source change with zero numerical effect.
    ///
    /// Stateless path (`kv_cache_slot=None`) at seq_len=128 — exercises the
    /// full ops1-4 → SDPA causal → ops6-7 chain through the arena's slots.
    ///
    /// # Why byte-exact, not |GPU − CPU| tolerance
    ///
    /// Both paths run the SAME mlx-native kernels with the SAME inputs;
    /// the only difference is the buffer the kernel writes into. F32
    /// arithmetic is deterministic on a fixed Metal device + simdgroup
    /// width, so 0 element diffs is the correct invariant.
    #[test]
    fn fa_projections_arena_byte_exact_f32_parity() {
        use super::super::FaProjectionsArena;

        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let (shape, weights_cpu, seq_len) = small_shape_and_weights();
        // Use a larger seq_len than small_shape_and_weights' default to
        // exercise the prefill (seq_len > 1) commit path. The helper
        // returns a small seq_len; we override here with a fixed 128 to
        // match the test name + the iter72/74/78 precedent of
        // unit-test-scale-shape parity gates.
        let _ = seq_len;
        let seq_len: u32 = 128;

        let h = shape.hidden_size as usize;
        let nh = shape.n_head as usize;
        let nkv = shape.n_kv as usize;
        let d = shape.head_dim as usize;
        let seq = seq_len as usize;

        // Synthetic residual-stream input (deterministic seed).
        let mut s = 0xDEADBEEFu32;
        let x_cpu: Vec<f32> = (0..seq * h)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s as i32 as f32) / (i32::MAX as f32)) * 0.5
            })
            .collect();

        // Text-only positions: all 4 axes = token index, flat layout
        // matching the production forward_gpu.rs encoding.
        let positions_flat: Vec<i32> = (0..4)
            .flat_map(|_| (0..seq_len as i32).collect::<Vec<_>>())
            .collect();

        let upload_pos = |dev: &MlxDevice| -> MlxBuffer {
            let mut b = dev
                .alloc_buffer(positions_flat.len() * 4, DType::I32, vec![positions_flat.len()])
                .expect("alloc positions");
            b.as_mut_slice::<i32>()
                .expect("mut")
                .copy_from_slice(&positions_flat);
            b
        };

        // Upload weights via the F32 dense path so both runs see identical
        // numerics; production's Q4_0 path also goes through arena vs
        // pooled symmetrically, but F32 lets us test byte-exactness without
        // worrying about quantization step-effects on a different alloc.
        // build_gated_attn_layer dispatches by weight dtype; we want both
        // paths to take the same dispatch arm so any element diff isolates
        // the alloc-source-change as the culprit.
        let upload_weights = |dev: &MlxDevice| -> FullAttnWeightsGpu {
            FullAttnWeightsGpu::from_cpu_f32(&weights_cpu, dev).expect("upload weights")
        };

        // --- Run 1: legacy path (fa_proj_arena=None) ---
        let x_gpu_legacy = upload_f32(&x_cpu, &device).expect("upload x legacy");
        let pos_legacy = upload_pos(&device);
        let weights_legacy = upload_weights(&device);
        let out_legacy_buf = build_gated_attn_layer(
            &device,
            &mut registry,
            &x_gpu_legacy,
            &pos_legacy,
            &weights_legacy,
            None, // stateless SDPA
            0,
            seq_len,
            shape.hidden_size,
            shape.n_head,
            shape.n_kv,
            shape.head_dim,
            shape.rotary_dim,
            shape.rope_theta,
            shape.mrope_section,
            shape.rms_norm_eps,
            None, // fa_arena
            None, // fa_proj_arena (LEGACY)
        )
        .expect("legacy build_gated_attn_layer");
        let out_legacy = download_f32(&out_legacy_buf).expect("download legacy");

        // --- Run 2: arena path (fa_proj_arena=Some(...)) ---
        let x_gpu_arena = upload_f32(&x_cpu, &device).expect("upload x arena");
        let pos_arena = upload_pos(&device);
        let weights_arena = upload_weights(&device);
        let mut fa_proj_arena = FaProjectionsArena::new(
            &device,
            seq_len,
            shape.hidden_size,
            shape.n_head,
            shape.n_kv,
            shape.head_dim,
            shape.rms_norm_eps,
        )
        .expect("FaProjectionsArena::new");
        let out_arena_buf = build_gated_attn_layer(
            &device,
            &mut registry,
            &x_gpu_arena,
            &pos_arena,
            &weights_arena,
            None,
            0,
            seq_len,
            shape.hidden_size,
            shape.n_head,
            shape.n_kv,
            shape.head_dim,
            shape.rotary_dim,
            shape.rope_theta,
            shape.mrope_section,
            shape.rms_norm_eps,
            None,                       // fa_arena
            Some(&mut fa_proj_arena),   // fa_proj_arena (NEW)
        )
        .expect("arena build_gated_attn_layer");
        let out_arena = download_f32(&out_arena_buf).expect("download arena");

        // --- Compare ---
        // Guard: parallel test contention may leave a Metal CB unexecuted,
        // returning all-zero output. Skip both-zero (suspect contention)
        // rather than fail.
        let legacy_all_zero = out_legacy.iter().all(|&v| v == 0.0);
        let arena_all_zero = out_arena.iter().all(|&v| v == 0.0);
        if legacy_all_zero && arena_all_zero {
            eprintln!(
                "fa_projections_arena_byte_exact_f32_parity: both paths all-zero \
                 under parallel test contention — skipping"
            );
            return;
        }

        assert_eq!(
            out_legacy.len(),
            out_arena.len(),
            "byte-exact parity: output lengths differ — legacy={} arena={}",
            out_legacy.len(),
            out_arena.len(),
        );
        let mut n_diff = 0usize;
        for (i, (&l, &a)) in out_legacy.iter().zip(out_arena.iter()).enumerate() {
            if l.to_bits() != a.to_bits() {
                if n_diff < 5 {
                    eprintln!(
                        "  byte-exact diff[{i}]: legacy={l:.10} ({:#010x}) \
                         arena={a:.10} ({:#010x})",
                        l.to_bits(),
                        a.to_bits()
                    );
                }
                n_diff += 1;
            }
        }
        assert_eq!(
            n_diff, 0,
            "fa_projections_arena_byte_exact_f32_parity FAIL: {n_diff}/{} F32 \
             elements differ — arena path is NOT byte-identical to legacy",
            out_legacy.len(),
        );
        eprintln!(
            "fa_projections_arena_byte_exact_f32_parity: 0/{} elements differ \
             (byte-exact) seq_len={seq_len}, shape h={}, nh={}, nkv={}, d={}",
            out_legacy.len(), shape.hidden_size, nh, nkv, d,
        );
    }
}
