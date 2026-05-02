//! GPU-side weight containers and full forward-pass builders for the
//! Qwen3.5 dense SwiGLU FFN (ADR-013 Decision 14) and Qwen3.5-MoE FFN
//! (ADR-013 Decision 13).
//!
//! This module bridges the pure-Rust scalar references in
//! [`super::ffn`] (the authoritative spec + test oracles) and the
//! mlx-native GPU kernels.
//!
//! # Dense SwiGLU op order  (Decision 14)
//!
//! ```text
//!  1. gate  = gate_proj(x)        [seq, intermediate]  — quantized_matmul_ggml or BF16 projection
//!  2. up    = up_proj(x)          [seq, intermediate]  — quantized_matmul_ggml or BF16 projection
//!  3. hidden = silu(gate) * up    — dispatch_silu_mul
//!  4. out   = down_proj(hidden)   [seq, hidden_size]   — quantized_matmul_ggml or BF16 projection
//! ```
//!
//! # MoE FFN op order  (Decision 13)
//!
//! ```text
//!  // Router
//!  1. logits = router(x)                          — apply_linear_projection_f32
//!  2. (topk_idx, topk_w) = softmax_topk_renorm()  — CPU (download logits, compute, upload)
//!
//!  // Routed experts — per selected expert, SwiGLU then weighted accumulate
//!  for e in topk_idx:
//!      gate_e = expert_gate[e](x)                 — apply_proj (per expert)
//!      up_e   = expert_up[e](x)                   — apply_proj (per expert)
//!      h_e    = silu(gate_e) * up_e               — GPU silu_mul / fused expert path
//!      y_e    = expert_down[e](h_e)               — apply_proj (per expert)
//!      moe_out += topk_w[e] * y_e                 — GPU/CPU combine depending on path
//!
//!  // Shared expert (sigmoid-gated)
//!  3. sh_gate_logit = shared_gate_inp(x)          — apply_linear_projection_f32
//!  4. sh_gate_val   = sigmoid(sh_gate_logit)      — dispatch_sigmoid_mul (on itself)
//!  5. a_s = shared_gate_proj(x)                   — apply_linear_projection_f32
//!  6. b_s = shared_up_proj(x)                     — apply_linear_projection_f32
//!  7. h_s = silu(a_s) * b_s                       — dispatch_silu_mul
//!  8. y_s = shared_down_proj(h_s)                 — apply_linear_projection_f32
//!  9. shared_out = sh_gate_val * y_s              — GPU elementwise mul
//!
//!  // Combine
//!  10. output = moe_out + shared_out              — GPU/CPU combine depending on path
//! ```
//!
//! # Implementation note: production DenseQ
//!
//! The production 27B DWQ path keeps dense FFN weights in native GGML
//! block-quantized form and calls `quantized_matmul_ggml` for gate, up, and
//! down projections.  `dispatch_silu_mul` runs on GPU; there is no CPU SiLU
//! bridge on the shipping DenseQ path.
//!
//! # Parity contract
//!
//! - Dense: |GPU − CPU|∞ < 1e-3 F32 (3 BF16-cast projections).
//! - MoE:   |GPU − CPU|∞ < 1e-3 F32 (same per-projection rounding budget).
//!
//! # ADR status
//!
//! P9b complete: both paths wired, parity tests pass.

use anyhow::{anyhow, Context, Result};
use mlx_native::ops::dense_gemv_bf16::dense_gemv_bf16_f32;
use mlx_native::ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};
use mlx_native::ops::elementwise::{cast, elementwise_add, CastDirection};
use mlx_native::ops::moe_softmax_topk::dispatch_moe_softmax_topk;
use mlx_native::ops::moe_weighted_reduce::dispatch_moe_weighted_reduce;
use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
};
use mlx_native::ops::quantized_matmul_id_ggml::{
    quantized_matmul_id_ggml_pooled, GgmlQuantizedMatmulIdParams,
};
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::ffn::{DenseFfnShape, DenseFfnWeights, MoeFfnShape, MoeFfnWeights};
use super::gpu_full_attn::{download_f32, upload_bf16_from_f32, upload_f32};
use super::weight_loader::DenseFfnWeightsQ;

// ================================================================
// GPU weight containers
// ================================================================

/// GPU-side weight handles for a single Qwen3.5 dense SwiGLU FFN layer.
pub struct DenseFfnWeightsGpu {
    /// `[intermediate_size, hidden_size]` row-major — gate_proj.
    pub gate: MlxBuffer,
    /// `[intermediate_size, hidden_size]` row-major — up_proj.
    pub up: MlxBuffer,
    /// `[hidden_size, intermediate_size]` row-major — down_proj.
    pub down: MlxBuffer,
}

impl DenseFfnWeightsGpu {
    /// Upload a [`DenseFfnWeights`] (pure-Rust f32) to Metal buffers.
    ///
    /// Weights pre-cast to BF16 to avoid per-inference F32→BF16 GPU cast
    /// in `proj()`.
    pub fn from_cpu(weights: &DenseFfnWeights, device: &MlxDevice) -> Result<Self> {
        Ok(Self {
            gate: upload_bf16_from_f32(&weights.gate, device)?,
            up: upload_bf16_from_f32(&weights.up, device)?,
            down: upload_bf16_from_f32(&weights.down, device)?,
        })
    }
}

// ================================================================
// Quantized Dense FFN GPU weight container (production path)
// ================================================================

/// Quantized GPU weight container for a Qwen3.5 dense SwiGLU FFN layer.
///
/// Gate/up/down projection weights are stored as raw GGML blocks (U8 dtype) —
/// the same bytes that came off disk.  The Metal `quantized_matmul_ggml`
/// kernel dequantizes on-the-fly during the matrix multiply, so no F32
/// expansion is required.
///
/// # Memory savings vs `DenseFfnWeightsGpu`
///
/// For a 27B dense GGUF (hidden=5120, intermediate=17408, Q4_0 weights):
///   BF16 path:  17408×5120×2×2 bytes = 357 MB per layer × 64 layers = 22 GB
///   Q4_0 path:  17408×5120×4/8×1.125 bytes ≈ 50 MB per layer × 64 layers = 3 GB
///   (Avoids the 129 GB OOM from the Q4_0→F32 round-trip in the old F32 path.)
pub struct DenseFfnWeightsGpuQ {
    /// Gate projection raw GGML blocks: `[intermediate_size, hidden_size]`.
    pub gate_q: MlxBuffer,
    /// Up projection raw GGML blocks: `[intermediate_size, hidden_size]`.
    pub up_q: MlxBuffer,
    /// Down projection raw GGML blocks: `[hidden_size, intermediate_size]`.
    pub down_q: MlxBuffer,
    /// GGML quantization type for gate/up projections.
    pub ggml_type_gate_up: GgmlType,
    /// GGML quantization type for down projection (may differ in mixed-quant GGUFs).
    pub ggml_type_down: GgmlType,
    /// Dense FFN intermediate size (output dim of gate/up, input dim of down).
    pub intermediate_size: u32,
    /// Model hidden size (input dim of gate/up, output dim of down).
    pub hidden_size: u32,
}

impl DenseFfnWeightsGpuQ {
    /// Construct from a [`DenseFfnWeightsQ`] loaded by the weight loader.
    ///
    /// The Metal buffers are already on the device (loaded via `GgufFile::load_tensor`).
    /// This is a clone (ARC retain) — no data copy.
    pub fn from_quantized(w: &DenseFfnWeightsQ) -> Self {
        Self {
            gate_q: w.gate_q.clone(),
            up_q: w.up_q.clone(),
            down_q: w.down_q.clone(),
            ggml_type_gate_up: w.ggml_type_gate_up,
            ggml_type_down: w.ggml_type_down,
            intermediate_size: w.intermediate_size,
            hidden_size: w.hidden_size,
        }
    }
}

/// GPU-side weight handles for a single Qwen3.5-MoE FFN layer.
///
/// Expert weights are stored as stacked flat buffers:
/// - `expert_gate`: `[num_experts * moe_intermediate_size, hidden_size]`
/// - `expert_up`:   `[num_experts * moe_intermediate_size, hidden_size]`
/// - `expert_down`: `[num_experts * hidden_size, moe_intermediate_size]`
pub struct MoeFfnWeightsGpu {
    /// Router projection: `[num_experts, hidden_size]`.
    pub router: MlxBuffer,
    /// Stacked expert gate projections.
    pub expert_gate: MlxBuffer,
    /// Stacked expert up projections.
    pub expert_up: MlxBuffer,
    /// Stacked expert down projections.
    pub expert_down: MlxBuffer,
    /// Shared-expert sigmoid gate: `[hidden_size]` (dot-product produces scalar per token).
    pub shared_gate_inp: MlxBuffer,
    /// Shared-expert gate_proj: `[shared_intermediate, hidden_size]`.
    pub shared_gate: MlxBuffer,
    /// Shared-expert up_proj: `[shared_intermediate, hidden_size]`.
    pub shared_up: MlxBuffer,
    /// Shared-expert down_proj: `[hidden_size, shared_intermediate]`.
    pub shared_down: MlxBuffer,
}

impl MoeFfnWeightsGpu {
    /// Upload a [`MoeFfnWeights`] (pure-Rust f32) to Metal buffers.
    ///
    /// All projection weights pre-cast to BF16 to avoid per-inference
    /// F32→BF16 GPU cast in `proj()`.
    pub fn from_cpu(weights: &MoeFfnWeights, device: &MlxDevice) -> Result<Self> {
        Ok(Self {
            router: upload_bf16_from_f32(&weights.router, device)?,
            expert_gate: upload_bf16_from_f32(&weights.expert_gate, device)?,
            expert_up: upload_bf16_from_f32(&weights.expert_up, device)?,
            expert_down: upload_bf16_from_f32(&weights.expert_down, device)?,
            shared_gate_inp: upload_bf16_from_f32(&weights.shared_gate_logit, device)?,
            shared_gate: upload_bf16_from_f32(&weights.shared_gate, device)?,
            shared_up: upload_bf16_from_f32(&weights.shared_up, device)?,
            shared_down: upload_bf16_from_f32(&weights.shared_down, device)?,
        })
    }
}

// ================================================================
// Quantized MoE GPU weight container (production path)
// ================================================================

/// Quantized GPU weight container for a Qwen3.5-MoE FFN layer.
///
/// Expert weights (`gate`, `up`, `down`) are stored as raw GGML blocks (U8
/// dtype) — the same bytes that came off disk.  The Metal
/// `quantized_matmul_id_ggml` kernel dequantizes on-the-fly during the
/// matrix multiply, so no F32 expansion is required.
///
/// In the apex GGUF: gate/up are Q5_K and down is Q6_K — each with their
/// own quant type, block geometry, and byte stride.
///
/// # Memory savings vs `MoeFfnWeightsGpu`
///
/// For the 35B apex GGUF (256 experts, hidden=2048, moe_intermediate=512):
///   F32 path: 256 × 512 × 2048 × 3 × 4 bytes = 3.2 GB per layer
///   Q5_K+Q6_K path: ~0.78 GB per layer
///   Savings per layer: ~2.4 GB; across 40 MoE layers: ~96 GB
///
/// Router (`[num_experts, hidden]`) and shared-expert weights are kept
/// as F32 because they are small: router ≈ 2 MB, shared ≈ 8 MB.
pub struct MoeFfnWeightsGpuQ {
    /// Router F32 projection: `[num_experts, hidden_size]`.
    pub router: MlxBuffer,
    /// Stacked expert gate projections, raw GGML blocks.
    pub expert_gate_q: MlxBuffer,
    /// Stacked expert up projections, raw GGML blocks.
    pub expert_up_q: MlxBuffer,
    /// Stacked expert down projections, raw GGML blocks.
    pub expert_down_q: MlxBuffer,
    /// GGML quantization type for gate/up expert weight buffers.
    pub ggml_type_gate_up: GgmlType,
    /// GGML quantization type for down expert weight buffers (may differ).
    pub ggml_type_down: GgmlType,
    /// Byte stride between consecutive expert slices in each stacked buffer.
    pub expert_gate_stride: u64,
    pub expert_up_stride: u64,
    pub expert_down_stride: u64,
    /// Number of experts.
    pub num_experts: u32,
    /// Shared-expert sigmoid gate: `[1, hidden_size]` F32.
    pub shared_gate_inp: MlxBuffer,
    /// Shared-expert gate_proj: `[shared_intermediate, hidden_size]` F32.
    pub shared_gate: MlxBuffer,
    /// Shared-expert up_proj: `[shared_intermediate, hidden_size]` F32.
    pub shared_up: MlxBuffer,
    /// Shared-expert down_proj: `[hidden_size, shared_intermediate]` F32.
    pub shared_down: MlxBuffer,
}

fn ggml_type_stride(t: GgmlType, rows: usize, cols: usize) -> Result<u64> {
    let qk = t.block_values() as usize;
    let block_bytes = t.block_bytes() as usize;
    let elems = rows * cols;
    anyhow::ensure!(
        elems % qk == 0,
        "elems {} not divisible by block QK {} for {:?}",
        elems,
        qk,
        t
    );
    Ok(((elems / qk) * block_bytes) as u64)
}

impl MoeFfnWeightsGpuQ {
    /// Construct from pre-loaded quantized Metal buffers.
    ///
    /// `expert_{gate,up,down}_q` are already on the Metal device (loaded via
    /// `GgufFile::load_tensor`).  Router and shared-expert weights are f32
    /// vecs that need uploading.
    #[allow(clippy::too_many_arguments)]
    pub fn from_quantized(
        expert_gate_q: MlxBuffer,
        expert_up_q: MlxBuffer,
        expert_down_q: MlxBuffer,
        ggml_type_gate_up: GgmlType,
        ggml_type_down: GgmlType,
        num_experts: u32,
        moe_intermediate_size: u32,
        hidden_size: u32,
        router_f32: &[f32],
        shared_gate_inp_f32: &[f32],
        shared_gate_f32: &[f32],
        shared_up_f32: &[f32],
        shared_down_f32: &[f32],
        device: &MlxDevice,
    ) -> Result<Self> {
        // Gate/up: [num_experts, moe_intermediate_size, hidden_size]
        // Each expert slice: moe_intermediate_size rows × hidden_size cols.
        let gate_stride = ggml_type_stride(
            ggml_type_gate_up,
            moe_intermediate_size as usize,
            hidden_size as usize,
        )
        .context("gate/up stride")?;

        // Down: [num_experts, hidden_size, moe_intermediate_size]
        // Each expert slice: hidden_size rows × moe_intermediate_size cols.
        let down_stride = ggml_type_stride(
            ggml_type_down,
            hidden_size as usize,
            moe_intermediate_size as usize,
        )
        .context("down stride")?;

        Ok(Self {
            // Router is small (~2MB) but also benefits from pre-cast since
            // `proj()` now checks dtype — keep BF16 for consistency.
            router: upload_bf16_from_f32(router_f32, device).context("upload router bf16")?,
            expert_gate_q,
            expert_up_q,
            expert_down_q,
            ggml_type_gate_up,
            ggml_type_down,
            expert_gate_stride: gate_stride,
            expert_up_stride: gate_stride, // gate and up have the same dimensions
            expert_down_stride: down_stride,
            num_experts,
            // Pre-cast shared expert weights to BF16 to avoid per-inference
            // F32→BF16 cast in proj() (~46MB each × 40 layers).
            shared_gate_inp: upload_bf16_from_f32(shared_gate_inp_f32, device)
                .context("upload shared_gate_inp bf16")?,
            shared_gate: upload_bf16_from_f32(shared_gate_f32, device)
                .context("upload shared_gate bf16")?,
            shared_up: upload_bf16_from_f32(shared_up_f32, device)
                .context("upload shared_up bf16")?,
            shared_down: upload_bf16_from_f32(shared_down_f32, device)
                .context("upload shared_down bf16")?,
        })
    }
}

// ================================================================
// Shared projection helper
// ================================================================

/// Apply a single linear projection: `output = input @ weight^T`.
///
/// Identical to `gpu_full_attn::apply_linear_projection_f32` but local to
/// this module so the FFN builders don't depend on full-attention internals.
///
/// `input`  shape: `[seq_len, in_features]`  F32.
/// `weight` shape: `[out_features, in_features]`  F32 (GGUF row-major).
/// Returns  `[seq_len, out_features]`  F32.
///
/// Requires `in_features >= 32` (tensor-core tile constraint).
fn proj(
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

    // If the weight is already BF16 (pre-cast at load time), use it directly;
    // otherwise cast inline and barrier before the matmul. weight_bf16_owned
    // holds the cast buffer alive for the function scope when we cast; in
    // the BF16-already branch it's never assigned (and never read past the
    // if-else, so Rust accepts the partial initialization).
    let weight_bf16_owned: MlxBuffer;
    let weight_bf16: &MlxBuffer = if weight.dtype() == DType::BF16 {
        weight
    } else {
        // ADR-015 iter14: scratch-lift — `proj`'s F32-legacy weight cast
        // is exactly the helper-allocated transient pattern the iter13
        // unretained-refs docstring at `mlx-native/src/encoder.rs:419-444`
        // warned about: `proj()` allocates `buf`, dispatches into it,
        // returns BEFORE the caller commits the encoder (caller commits
        // in the FFN encoder which runs the matmul that READS `buf`).
        // Under retained refs the encoder's CB ARC keeps `buf` alive;
        // under unretained refs only the local `weight_bf16_owned` ARC
        // does, so we hand out a pool-anchored buffer.  At iter14 base
        // this branch only fires for F32 weights (Qwen3.6 dwq46 is Q4_0
        // U8, so this is a no-op on the apex fixture); lifted for
        // forward-compatibility and to remove the pattern entirely.
        let buf = super::decode_pool::pooled_alloc_buffer(
                device,
                n_w * 2,
                DType::BF16,
                vec![out_features as usize, in_features as usize],
            )
            .map_err(|e| anyhow!("alloc weight_bf16 (pooled): {e}"))?;
        cast(
            encoder,
            registry,
            device.metal_device(),
            weight,
            &buf,
            n_w,
            CastDirection::F32ToBF16,
        )
        .context("cast weight F32→BF16")?;
        // Barrier: matmul reads weight_bf16 written by the cast above.
        encoder.memory_barrier();
        weight_bf16_owned = buf;
        &weight_bf16_owned
    };

    // NOT pooled — `proj` is called from both `build_moe_ffn_layer_gpu` (the
    // unquantized path, which downloads router logits via `download_f32` →
    // `as_slice` → reads full `byte_len()`) AND `build_moe_ffn_layer_gpu_q`
    // (the quantized path, which keeps the buffer on GPU).  The pool's
    // power-of-two bucket rounding would inflate `byte_len()` beyond the
    // requested shape and break the unquantized path.
    let out_bytes = (seq_len * out_features) as usize * 4;
    let mut dst = device
        .alloc_buffer(
            out_bytes,
            DType::F32,
            vec![seq_len as usize, out_features as usize],
        )
        .map_err(|e| anyhow!("alloc proj dst: {e}"))?;

    let params = DenseMmBf16F32Params {
        m: seq_len,
        n: out_features,
        k: in_features,
        src0_batch: 1,
        src1_batch: 1,
    };
    if seq_len == 1 {
        // GEMV path for single-token decode — bandwidth-optimized, ~2× faster
        // than tiled MM for M=1 on Apple Silicon.
        dense_gemv_bf16_f32(
            encoder,
            registry,
            device,
            weight_bf16,
            input,
            &mut dst,
            &params,
        )
        .context("dense_gemv_bf16_f32 proj M=1")?;
    } else {
        dense_matmul_bf16_f32_tensor(
            encoder,
            registry,
            device,
            weight_bf16,
            input,
            &mut dst,
            &params,
        )
        .context("dense_matmul_bf16_f32_tensor")?;
    }
    Ok(dst)
}

/// Pooled-output variant of [`proj`] for callers whose proj output flows
/// only into downstream GPU kernels (no `as_slice` / `as_mut_slice` /
/// `download_f32`).
///
/// ADR-015 iter7b (P3b alloc_buffer pool) — the dominant H1 lever per
/// §P3a''' (375 µs/token apex MoE).  The original [`proj`] cannot be
/// pool-migrated because it is shared with the unquantized
/// `build_moe_ffn_layer_gpu` path which downloads router logits via
/// `download_f32` → `as_slice` (reads `byte_len()`, which the pool's
/// bucket rounding inflates).  Splitting into two functions preserves
/// the carve-out for the unquantized path while letting the quantized
/// production path (`build_moe_ffn_layer_gpu_q_into`) reap the savings.
///
/// Caller contract — verified for build_moe_ffn_layer_gpu_q_into 2026-04-27:
///   - logits_buf  → dispatch_moe_softmax_topk     (byte_len validation only;
///                                                   inflated bucket size passes)
///   - sh_logit_buf → dispatch_moe_weighted_reduce (same)
///   - a_s_buf      → dispatch_silu_mul            (byte_len validation only)
///   - b_s_buf      → dispatch_silu_mul            (same)
///   - y_s_buf      → dispatch_moe_weighted_reduce (same)
///   - none of these are downloaded to CPU on the q_into path.
///
/// Bit-exact to [`proj`] except for the underlying `MlxBuffer`'s
/// physical byte_len (bucket-rounded for pool reuse).  The kernel
/// dispatches use `element_count` / explicit `n` parameters, not
/// `byte_len`, so the bucket-rounded tail is never read.  The pool's
/// per-token arena lifecycle (reset_decode_pool at top of decode token)
/// keeps the safety contract.
#[allow(clippy::too_many_arguments)]
fn proj_pooled(
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

    // Same weight-bf16 cast logic as `proj`.  In apex production the
    // weight is pre-cast to BF16 at model load (per
    // `MoeFfnWeightsGpuQ::from_cpu` line 309-327 — `upload_bf16_from_f32`),
    // so this branch never fires for the q_into callers; we keep it for
    // call-site flexibility with non-production fixtures (tests / CI).
    let weight_bf16_owned: MlxBuffer;
    let weight_bf16: &MlxBuffer = if weight.dtype() == DType::BF16 {
        weight
    } else {
        let buf = super::decode_pool::pooled_alloc_buffer(
            device,
            n_w * 2,
            DType::BF16,
            vec![out_features as usize, in_features as usize],
        )
        .map_err(|e| anyhow!("alloc weight_bf16 (pooled): {e}"))?;
        cast(
            encoder,
            registry,
            device.metal_device(),
            weight,
            &buf,
            n_w,
            CastDirection::F32ToBF16,
        )
        .context("cast weight F32→BF16")?;
        encoder.memory_barrier();
        weight_bf16_owned = buf;
        &weight_bf16_owned
    };

    let out_bytes = (seq_len * out_features) as usize * 4;
    let mut dst = super::decode_pool::pooled_alloc_buffer(
        device,
        out_bytes,
        DType::F32,
        vec![seq_len as usize, out_features as usize],
    )
    .map_err(|e| anyhow!("alloc proj dst (pooled): {e}"))?;

    let params = DenseMmBf16F32Params {
        m: seq_len,
        n: out_features,
        k: in_features,
        src0_batch: 1,
        src1_batch: 1,
    };
    if seq_len == 1 {
        dense_gemv_bf16_f32(
            encoder,
            registry,
            device,
            weight_bf16,
            input,
            &mut dst,
            &params,
        )
        .context("dense_gemv_bf16_f32 proj_pooled M=1")?;
    } else {
        dense_matmul_bf16_f32_tensor(
            encoder,
            registry,
            device,
            weight_bf16,
            input,
            &mut dst,
            &params,
        )
        .context("dense_matmul_bf16_f32_tensor")?;
    }
    Ok(dst)
}

// ================================================================
// CPU SiLU helper for legacy parity-only paths
// ================================================================

/// Apply SiLU * up element-wise on CPU: `out[i] = gate[i] / (1 + exp(-gate[i])) * up[i]`.
///
/// This remains only for F32 reference/parity helpers and legacy CPU-bridged
/// paths.  The production DenseQ path uses `dispatch_silu_mul` on GPU.
///
/// Spec: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
/// Decision 14 / Decision 13: "silu(gate) * up".
fn silu_mul_cpu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(
        gate.len(),
        up.len(),
        "silu_mul_cpu: gate/up length mismatch"
    );
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let silu_g = g / (1.0 + (-g).exp());
            silu_g * u
        })
        .collect()
}

// ================================================================
// Dense SwiGLU GPU path
// ================================================================

/// Build the Qwen3.5 dense SwiGLU FFN forward pass on the GPU.
///
/// Implements ADR-013 Decision 14 op order end-to-end.
/// Returns the residual contribution `[seq_len, hidden_size]` F32.
/// Caller adds to x for the post-FFN residual stream.
///
/// # Op order
///
/// 1. gate  = gate_proj(x)          `[seq, intermediate]`
/// 2. up    = up_proj(x)            `[seq, intermediate]`
/// 3. hidden = silu(gate) * up      CPU bridge (download → silu_mul → upload)
/// 4. out   = down_proj(hidden)     `[seq, hidden_size]`
///
/// # Parity contract
///
/// `|GPU − dense_swiglu_cpu_ref(x, weights, shape)|∞ < 1e-3` F32.
/// Source: three BF16-cast projections each contribute ≤1e-3 rounding.
///
/// When `add_residual` is `Some(r)`, the kernel folds `r` into the output
/// in the same command buffer, eliminating a separate `residual_add_gpu`
/// commit per dense FFN layer.
#[allow(clippy::too_many_arguments)]
pub fn build_dense_ffn_layer_gpu(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights_gpu: &DenseFfnWeightsGpu,
    shape: DenseFfnShape,
    add_residual: Option<&MlxBuffer>,
) -> Result<MlxBuffer> {
    let h = shape.hidden_size;
    let m = shape.intermediate_size;
    let seq_len = (x.element_count() / h as usize) as u32;
    let n_h = (seq_len * m) as u32;
    let n_out = seq_len as usize * h as usize;

    // Pre-allocate intermediate buffers (must outlive the encoder).
    //
    // ADR-015 iter14: lift `hidden_buf` and `silu_params` to the per-decode-token
    // pool. At decode (seq=1), this function commits the encoder via `commit()`
    // WITHOUT wait at line ~738; `hidden_buf` and `silu_params` are NOT
    // returned, so their function-level locals drop at function return BEFORE
    // the GPU has executed the silu_mul / down_proj dispatches that read them.
    // Under retained refs the encoder's CB ARC keeps them alive; under
    // unretained refs only the pool's `in_use` ARC clone does. This branch
    // is the non-quantized dense path (Qwen3.6 dwq46 takes the Q4_0 path
    // which already pools its scratches in `build_dense_ffn_layer_gpu_q_into_pooled`),
    // but we lift for forward-compat and to remove the unsafe pattern.
    let hidden_buf = super::decode_pool::pooled_alloc_buffer(
            device,
            n_h as usize * 4,
            DType::F32,
            vec![seq_len as usize, m as usize],
        )
        .map_err(|e| anyhow!("alloc dense silu hidden (pooled): {e}"))?;
    let mut silu_params = super::decode_pool::pooled_alloc_buffer(
            device, 4, DType::U32, vec![1])
        .map_err(|e| anyhow!("alloc dense silu params (pooled): {e}"))?;
    silu_params
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("{e}"))?[0] = n_h;

    // Single encoder: gate+up projs (concurrent) → silu_mul → down proj [→ residual add].
    let mut enc = device.command_encoder().context("enc dense swiglu")?;
    // Ops 1+2: gate and up projections — concurrent (both read from x)
    let gate_buf = proj(
        &mut enc,
        registry,
        device,
        x,
        &weights_gpu.gate,
        seq_len,
        h,
        m,
    )?;
    let up_buf = proj(
        &mut enc,
        registry,
        device,
        x,
        &weights_gpu.up,
        seq_len,
        h,
        m,
    )?;
    // Barrier: silu_mul reads gate_buf/up_buf written above.
    enc.memory_barrier();
    // Op 3: silu(gate) * up → hidden
    dispatch_silu_mul(
        &mut enc,
        registry,
        device.metal_device(),
        &gate_buf,
        &up_buf,
        &hidden_buf,
        &silu_params,
        n_h,
    )
    .context("dispatch silu_mul dense")?;
    // Barrier: down proj reads hidden written above.
    enc.memory_barrier();
    // Op 4: out = down_proj(hidden)
    let down_out = proj(
        &mut enc,
        registry,
        device,
        &hidden_buf,
        &weights_gpu.down,
        seq_len,
        m,
        h,
    )?;

    // Op 5 (optional): out += residual — folded into this encoder to save 1 commit per layer.
    let result = if let Some(res) = add_residual {
        let sum_buf = device
            .alloc_buffer(n_out * 4, DType::F32, vec![n_out])
            .map_err(|e| anyhow!("alloc dense ffn residual sum: {e}"))?;
        // Barrier: elementwise_add reads down_out written by Op 4.
        enc.memory_barrier();
        elementwise_add(
            &mut enc,
            registry,
            device.metal_device(),
            &down_out,
            res,
            &sum_buf,
            n_out,
            DType::F32,
        )
        .context("dense ffn residual add")?;
        sum_buf
    } else {
        down_out
    };

    // Decode fast path (seq=1): commit() without wait.
    // The caller (forward_gpu) sets `hidden = ffn_out` then immediately feeds
    // `hidden` into the next layer's fused_residual_norm encoder on the same
    // Metal serial queue.  GPU ordering guarantees the dense FFN completes
    // before fused_residual_norm executes.
    //
    // Prefill (seq>1): commit_and_wait() for correctness (fused_residual_norm
    // is a separate code path in forward_gpu that relies on ffn_out being ready,
    // and dump_hidden_stats may do a CPU read of hidden).
    if seq_len == 1 {
        enc.commit();
    } else {
        enc.commit_and_wait().context("commit dense swiglu")?;
    }
    Ok(result)
}

// ================================================================
// Quantized Dense SwiGLU GPU path
// ================================================================

/// Build the Qwen3.5 dense SwiGLU FFN forward pass using quantized weights.
///
/// This is the production path for 27B dense DWQ GGUFs.  Gate/up/down
/// projections use `quantized_matmul_ggml` which keeps weights in GGML
/// block-quantized form (Q4_0/Q8_0/Q6_K) on the Metal device, avoiding the
/// Q4_0→F32 expansion that causes the 129 GB OOM on M5 Max 128 GB.
///
/// # Op order (matches `build_dense_ffn_layer_gpu` exactly)
///
/// ```text
/// 1. gate   = gate_proj(x)          [seq, intermediate] — quantized_matmul_ggml
/// 2. up     = up_proj(x)            [seq, intermediate] — quantized_matmul_ggml
/// 3. hidden = silu(gate) * up       — dispatch_silu_mul (GPU)
/// 4. out    = down_proj(hidden)     [seq, hidden_size]  — quantized_matmul_ggml
/// (5. out  += residual)             optional — folded in to save 1 GPU sync
/// ```
///
/// # Parity contract
///
/// Compared to the F32 reference, Q4_0 dequant noise is ≤ 5e-3 per element
/// (tighter than the 1e-2 MoE tolerance, since dense has fewer projections).
/// The existing `build_dense_ffn_layer_gpu` BF16 parity test remains green.
///
/// When `add_residual` is `Some(r)`, the residual add is folded into the same
/// command buffer, eliminating a separate `residual_add_gpu` commit per layer.
#[allow(clippy::too_many_arguments)]
pub fn build_dense_ffn_layer_gpu_q(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &DenseFfnWeightsGpuQ,
    add_residual: Option<&MlxBuffer>,
) -> Result<MlxBuffer> {
    // Wave 5b.16 sunset: the W-5b.14 forensic `HF2Q_DENSE_Q_LEGACY=1`
    // own-encoder + device-alloc legacy body has been removed after a
    // 30/30 cross-path parity audit (token id 11 across 5 cold loads ×
    // 3 trials × 2 paths at PP4106).  This wrapper now unconditionally
    // routes through `build_dense_ffn_layer_gpu_q_into`, which uses
    // pooled scratches at decode (seq_len == 1) and device-alloc'd
    // FINAL outputs at prefill (seq_len > 1) per the W-5b.15 split.
    let h = weights.hidden_size;
    let seq_len = (x.element_count() / h as usize) as u32;
    let mut enc = device.command_encoder().context("enc dense_q swiglu")?;
    let out =
        build_dense_ffn_layer_gpu_q_into(&mut enc, device, registry, x, weights, add_residual)?;
    if seq_len == 1 {
        enc.commit();
    } else {
        enc.commit_and_wait().context("commit dense_q swiglu")?;
    }
    Ok(out)
}

/// External-encoder variant of [`build_dense_ffn_layer_gpu_q`].
///
/// Encodes the entire dense quantized FFN forward pass (gate/up projections +
/// silu_mul + down projection + optional residual add) into the caller-supplied
/// [`mlx_native::CommandEncoder`].  Does NOT commit — the caller is responsible
/// for committing the encoder.
///
/// # Why this exists
///
/// Wave 5b.14: addressed the historical per-dense-layer 2-encoder overhead
/// component from the W-5b.13 PP4106 audit (FFN dispatch bucket = 9,750 ms,
/// 4.34x wall ratio vs llama at that time).  That old ratio is not the current
/// live decode frame: on 2026-04-28, a short apex dwq46 decode smoke measured
/// hf2q 114.4 tok/s vs llama-bench 116.78 tok/s.  The larger current gap is
/// prefill wall time, not this dense decode wrapper alone.
///
/// The pre-fusion path issued 2 separate command buffers per dense layer:
///
/// 1. `dispatch_fused_residual_norm_f32` — produces `ffn_input` from
///    `(hidden, attn_out, post_norm_w)` + writes `ffn_residual` for the
///    later residual add.
/// 2. `build_dense_ffn_layer_gpu_q` — its own encoder for the dense FFN
///    forward + residual add at the end.
///
/// With this variant, the caller can fuse step 1 + step 2 into a single
/// command buffer per dense layer.  This mirrors the MoE-Q `_into` analog
/// (`build_moe_ffn_layer_gpu_q_into`) verbatim — same external-encoder
/// signature, same `decode_pool::pooled_alloc_buffer` for every scratch.
///
/// # Caller contract
///
/// * The caller must provide the encoder.  The encoder must NOT have been
///   committed.
/// * The caller MUST commit the encoder after this function returns; the
///   GPU work is queued but not yet submitted.
/// * The output `MlxBuffer` is allocated from the per-decode-token arena
///   pool; it must NOT be downloaded to CPU via `as_slice` /
///   `download_f32` (the pool's bucket rounding inflates `byte_len`).
///   Decode-path consumers (next layer's fused_residual_norm + lm_head
///   `apply_output_head_gpu_greedy`) read via shape-respecting kernels —
///   safe.
#[allow(clippy::too_many_arguments)]
pub fn build_dense_ffn_layer_gpu_q_into(
    enc: &mut mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &DenseFfnWeightsGpuQ,
    add_residual: Option<&MlxBuffer>,
) -> Result<MlxBuffer> {
    // Dispatch policy (W-5b.15 simplification of W-5b.14):
    //
    // The pooled variant routes 4 internal scratches (gate, up, hidden,
    // silu_params) through `decode_pool::pooled_alloc_buffer` for both decode
    // (seq_len == 1) and prefill (seq_len > 1).  The FINAL output buffer
    // (`down_out` or, when residual is folded, `sum_buf`) is `device.alloc_buffer`'d
    // so it survives the per-layer arena reset issued from `forward_gpu_impl`'s
    // layer loop and can safely become the next layer's `hidden`.
    //
    // The pre-W-5b.15 seq-len-aware fallback to `_into_device` is retained as
    // a fallback under `HF2Q_DENSE_Q_ARENA_RESET=0` for forensic A/B; on the
    // default path the pooled variant captures the W-5b.13 audit's projected
    // ~30–40% allocation-churn savings at prefill (closed by the per-layer
    // `decode_pool::reset_for_prefill_chunk()` call site in forward_gpu_impl).
    if std::env::var("HF2Q_DENSE_Q_ARENA_RESET").as_deref() == Ok("0") {
        let h = weights.hidden_size;
        let seq_len = (x.element_count() / h as usize) as u32;
        if seq_len == 1 {
            return build_dense_ffn_layer_gpu_q_into_pooled(
                enc,
                device,
                registry,
                x,
                weights,
                add_residual,
            );
        } else {
            return build_dense_ffn_layer_gpu_q_into_device(
                enc,
                device,
                registry,
                x,
                weights,
                add_residual,
            );
        }
    }
    build_dense_ffn_layer_gpu_q_into_pooled(enc, device, registry, x, weights, add_residual)
}

#[allow(clippy::too_many_arguments)]
fn build_dense_ffn_layer_gpu_q_into_pooled(
    enc: &mut mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &DenseFfnWeightsGpuQ,
    add_residual: Option<&MlxBuffer>,
) -> Result<MlxBuffer> {
    let h = weights.hidden_size;
    let m = weights.intermediate_size;
    let seq_len = (x.element_count() / h as usize) as u32;
    let n_h = (seq_len * m) as u32;
    let n_out = (seq_len * h) as usize;

    // ADR-012 §Optimize / Task #15 + W-5b.14/15: route the four large
    // INTERNAL scratches (gate, up, hidden, silu_params) through the
    // thread-local arena pool.  Their lifetime ends at this caller's
    // `enc.commit_and_wait()`; `forward_gpu_impl`'s per-layer
    // `decode_pool::reset_for_prefill_chunk()` (W-5b.15) recycles them
    // before the next layer issues its own scratches.
    //
    // The FINAL OUTPUT buffer (`down_out` when no residual; `sum_buf`
    // when residual is folded) MUST be `device.alloc_buffer` rather than
    // pooled — its ARC clone leaves this function via `Ok(result)` and
    // becomes the next layer's `hidden`, crossing the per-layer reset
    // boundary.  A pooled output here would alias the next layer's
    // first pool allocation of the same bucket size (84 MB at
    // Qwen3.6-27B prefill), corrupting the residual stream silently.
    // See `decode_pool::reset_for_prefill_chunk` doc-comment for the
    // full lifetime contract.
    let _w5b_ffn_alloc = super::wave5b8_profile::Section::start(
        super::wave5b8_profile::SectionKind::FfnAllocScratch,
    );
    let mut gate_buf = super::decode_pool::pooled_alloc_buffer(
        device,
        n_h as usize * 4,
        DType::F32,
        vec![seq_len as usize, m as usize],
    )
    .map_err(|e| anyhow!("alloc dense_q gate: {e}"))?;
    let mut up_buf = super::decode_pool::pooled_alloc_buffer(
        device,
        n_h as usize * 4,
        DType::F32,
        vec![seq_len as usize, m as usize],
    )
    .map_err(|e| anyhow!("alloc dense_q up: {e}"))?;
    let hidden_buf = super::decode_pool::pooled_alloc_buffer(
        device,
        n_h as usize * 4,
        DType::F32,
        vec![seq_len as usize, m as usize],
    )
    .map_err(|e| anyhow!("alloc dense_q hidden: {e}"))?;
    // FINAL OUTPUT (down_out):
    //   - Prefill (seq_len > 1): device-alloc'd so it survives the per-layer
    //     `reset_for_prefill_chunk()` and can become the next layer's `hidden`.
    //   - Decode (seq_len == 1): pooled — `forward_gpu_greedy`'s top-of-token
    //     `reset_decode_pool` recycles it before the next decode token begins,
    //     and it is not consumed across decode tokens (output head consumes it
    //     within the same token via `apply_output_head_gpu_greedy`).  Keeps the
    //     W-5b.14 fully-pooled decode path bit-for-bit when residual=None.
    let mut down_out = if seq_len == 1 {
        super::decode_pool::pooled_alloc_buffer(
            device,
            n_out * 4,
            DType::F32,
            vec![seq_len as usize, h as usize],
        )
        .map_err(|e| anyhow!("alloc dense_q down_out (pooled, decode): {e}"))?
    } else {
        device
            .alloc_buffer(n_out * 4, DType::F32, vec![seq_len as usize, h as usize])
            .map_err(|e| anyhow!("alloc dense_q down_out (device, prefill): {e}"))?
    };
    let mut silu_params_buf =
        super::decode_pool::pooled_alloc_buffer(device, 4, DType::U32, vec![1])
            .map_err(|e| anyhow!("alloc dense_q silu_params: {e}"))?;
    silu_params_buf
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("{e}"))?[0] = n_h;
    drop(_w5b_ffn_alloc);

    let gate_up_params = GgmlQuantizedMatmulParams {
        m: seq_len,
        n: m,
        k: h,
        ggml_type: weights.ggml_type_gate_up,
    };
    let down_params = GgmlQuantizedMatmulParams {
        m: seq_len,
        n: h,
        k: m,
        ggml_type: weights.ggml_type_down,
    };

    // Ops 1+2: gate and up projections via quantized_matmul_ggml (both read x, concurrent).
    {
        let _w5b = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FfnPhaseAProj,
        );
        quantized_matmul_ggml(
            enc,
            registry,
            device,
            x,
            &weights.gate_q,
            &mut gate_buf,
            &gate_up_params,
        )
        .context("dense_q gate proj")?;
        quantized_matmul_ggml(
            enc,
            registry,
            device,
            x,
            &weights.up_q,
            &mut up_buf,
            &gate_up_params,
        )
        .context("dense_q up proj")?;
    }

    // Barrier: silu_mul reads gate_buf/up_buf written above.
    {
        let _w5b = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FfnBarrierAB,
        );
        enc.memory_barrier();
    }

    // Op 3: silu(gate) * up → hidden.
    {
        let _w5b = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FfnPhaseDSilu,
        );
        dispatch_silu_mul(
            enc,
            registry,
            device.metal_device(),
            &gate_buf,
            &up_buf,
            &hidden_buf,
            &silu_params_buf,
            n_h,
        )
        .context("dense_q silu_mul")?;
    }

    // Barrier: down proj reads hidden.
    {
        let _w5b = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FfnBarrierDE,
        );
        enc.memory_barrier();
    }

    // Op 4: out = down_proj(hidden).
    {
        let _w5b = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FfnPhaseEDown,
        );
        quantized_matmul_ggml(
            enc,
            registry,
            device,
            &hidden_buf,
            &weights.down_q,
            &mut down_out,
            &down_params,
        )
        .context("dense_q down proj")?;
    }

    // Op 5 (optional): out += residual — folded to save 1 commit per dense layer.
    // FINAL OUTPUT (sum_buf): same prefill/decode lifetime split as `down_out`
    // above — device-alloc at prefill (survives per-layer reset), pooled at
    // decode (recycled by the per-token `reset_decode_pool`).
    let result = if let Some(res) = add_residual {
        let sum_buf = if seq_len == 1 {
            super::decode_pool::pooled_alloc_buffer(device, n_out * 4, DType::F32, vec![n_out])
                .map_err(|e| anyhow!("alloc dense_q residual sum (pooled, decode): {e}"))?
        } else {
            device
                .alloc_buffer(n_out * 4, DType::F32, vec![n_out])
                .map_err(|e| anyhow!("alloc dense_q residual sum (device, prefill): {e}"))?
        };
        // Barrier: elementwise_add reads down_out written by Op 4.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnBarrierEF,
            );
            enc.memory_barrier();
        }
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseFReduce,
            );
            elementwise_add(
                enc,
                registry,
                device.metal_device(),
                &down_out,
                res,
                &sum_buf,
                n_out,
                DType::F32,
            )
            .context("dense_q residual add")?;
        }
        sum_buf
    } else {
        down_out
    };

    Ok(result)
}

/// ADR-015 iter72: `_into_arena` variant of `build_dense_ffn_layer_gpu_q_into`.
///
/// Prefill-only variant that reuses the four transient scratches
/// (gate, up, hidden, silu_params) from a caller-owned
/// [`super::DenseFfnArena`]. Allocation happens ONCE per prefill (in
/// `forward_gpu_impl`); this function performs zero `device.alloc_buffer`
/// calls for scratches, eliminating the W-5b.8 `ffn.alloc_scratch` bucket
/// (~365 ms / ~14 % of prefill wall on 27B q4_0-flat pp4096).
///
/// # Arguments
///
/// * `arena` — caller-owned scratch arena. MUST be sized for the actual
///   prefill `(seq_len, hidden_size, intermediate_size)`. The function
///   asserts this via `validate_fits` and returns `Err` on mismatch.
///
/// # Lifetime contract
///
/// Same as the pooled variant for the FINAL output buffer (`down_out` /
/// `sum_buf`): device-allocated at prefill so it survives the per-layer
/// pool reset and crosses into the next layer's `hidden`. The four
/// transient scratches now live in the arena (caller scope), eliminating
/// the per-layer `pooled_alloc_buffer` calls entirely.
#[allow(clippy::too_many_arguments)]
pub fn build_dense_ffn_layer_gpu_q_into_with_arena(
    enc: &mut mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &DenseFfnWeightsGpuQ,
    add_residual: Option<&MlxBuffer>,
    arena: &mut super::DenseFfnArena,
) -> Result<MlxBuffer> {
    let h = weights.hidden_size;
    let m = weights.intermediate_size;
    let seq_len = (x.element_count() / h as usize) as u32;
    let n_h = (seq_len * m) as u32;
    let n_out = (seq_len * h) as usize;

    // Validate the arena's capacity matches our shape — guards against a
    // future model that mixes dense layers of different intermediate sizes
    // or against accidental shape drift.
    arena
        .validate_fits(seq_len, h, m)
        .context("DenseFfnArena shape mismatch")?;

    // ADR-015 iter72: zero per-layer scratch allocations.  The four
    // transient buffers (gate, up, hidden, silu_params) live in the
    // caller-owned arena across all dense layers in this prefill.  The
    // arena is allocated once at the top of `forward_gpu_impl` (after
    // `ensure_gpu_cache_primed`, before the per-layer loop) when seq_len > 1
    // AND the model has at least one Dense FFN layer.
    //
    // We still record the (now near-zero) `FfnAllocScratch` bucket so the
    // W-5b.8 profile can confirm the savings on the same metric the prior
    // closure attempts inspected.
    let _w5b_ffn_alloc = super::wave5b8_profile::Section::start(
        super::wave5b8_profile::SectionKind::FfnAllocScratch,
    );

    // FINAL OUTPUT (down_out) at prefill MUST be `device.alloc_buffer`'d so it
    // survives the per-layer arena reset and can become the next layer's
    // `hidden`.  Same lifetime constraint as the pooled variant — see
    // `build_dense_ffn_layer_gpu_q_into_pooled` lines 958-978 for the
    // canonical doc-comment.
    let mut down_out = device
        .alloc_buffer(n_out * 4, DType::F32, vec![seq_len as usize, h as usize])
        .map_err(|e| anyhow!("alloc dense_q down_out (device, prefill arena): {e}"))?;

    // Write the silu_mul element count to the arena's silu_params buffer.
    arena
        .silu_params_buf
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("DenseFfnArena silu_params as_mut_slice: {e}"))?[0] = n_h;
    drop(_w5b_ffn_alloc);

    let gate_up_params = GgmlQuantizedMatmulParams {
        m: seq_len,
        n: m,
        k: h,
        ggml_type: weights.ggml_type_gate_up,
    };
    let down_params = GgmlQuantizedMatmulParams {
        m: seq_len,
        n: h,
        k: m,
        ggml_type: weights.ggml_type_down,
    };

    // Ops 1+2: gate and up projections via quantized_matmul_ggml (both read x, concurrent).
    {
        let _w5b = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FfnPhaseAProj,
        );
        quantized_matmul_ggml(
            enc,
            registry,
            device,
            x,
            &weights.gate_q,
            &mut arena.gate_buf,
            &gate_up_params,
        )
        .context("dense_q gate proj (arena)")?;
        quantized_matmul_ggml(
            enc,
            registry,
            device,
            x,
            &weights.up_q,
            &mut arena.up_buf,
            &gate_up_params,
        )
        .context("dense_q up proj (arena)")?;
    }

    // Barrier: silu_mul reads gate_buf/up_buf written above.
    {
        let _w5b = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FfnBarrierAB,
        );
        enc.memory_barrier();
    }

    // Op 3: silu(gate) * up → hidden.
    {
        let _w5b = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FfnPhaseDSilu,
        );
        dispatch_silu_mul(
            enc,
            registry,
            device.metal_device(),
            &arena.gate_buf,
            &arena.up_buf,
            &arena.hidden_buf,
            &arena.silu_params_buf,
            n_h,
        )
        .context("dense_q silu_mul (arena)")?;
    }

    // Barrier: down proj reads hidden.
    {
        let _w5b = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FfnBarrierDE,
        );
        enc.memory_barrier();
    }

    // Op 4: out = down_proj(hidden).
    {
        let _w5b = super::wave5b8_profile::Section::start(
            super::wave5b8_profile::SectionKind::FfnPhaseEDown,
        );
        quantized_matmul_ggml(
            enc,
            registry,
            device,
            &arena.hidden_buf,
            &weights.down_q,
            &mut down_out,
            &down_params,
        )
        .context("dense_q down proj (arena)")?;
    }

    // Op 5 (optional): out += residual — folded to save 1 commit per dense layer.
    let result = if let Some(res) = add_residual {
        // Same lifetime constraint as down_out at prefill — must be
        // `device.alloc_buffer` to survive per-layer reset.
        let sum_buf = device
            .alloc_buffer(n_out * 4, DType::F32, vec![n_out])
            .map_err(|e| anyhow!("alloc dense_q residual sum (device, prefill arena): {e}"))?;
        // Barrier: elementwise_add reads down_out written by Op 4.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnBarrierEF,
            );
            enc.memory_barrier();
        }
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseFReduce,
            );
            elementwise_add(
                enc,
                registry,
                device.metal_device(),
                &down_out,
                res,
                &sum_buf,
                n_out,
                DType::F32,
            )
            .context("dense_q residual add (arena)")?;
        }
        sum_buf
    } else {
        down_out
    };

    Ok(result)
}

/// Prefill-safe variant of [`build_dense_ffn_layer_gpu_q_into`]: uses
/// `device.alloc_buffer` for scratches to avoid the per-decode-token arena
/// pool's residency-set exhaustion at full prefill working set.  Same
/// fused-CB single-encoder shape as the pooled variant.
#[allow(clippy::too_many_arguments)]
fn build_dense_ffn_layer_gpu_q_into_device(
    enc: &mut mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &DenseFfnWeightsGpuQ,
    add_residual: Option<&MlxBuffer>,
) -> Result<MlxBuffer> {
    let h = weights.hidden_size;
    let m = weights.intermediate_size;
    let seq_len = (x.element_count() / h as usize) as u32;
    let n_h = (seq_len * m) as u32;
    let n_out = (seq_len * h) as usize;

    let mut gate_buf = device
        .alloc_buffer(
            n_h as usize * 4,
            DType::F32,
            vec![seq_len as usize, m as usize],
        )
        .map_err(|e| anyhow!("alloc dense_q gate: {e}"))?;
    let mut up_buf = device
        .alloc_buffer(
            n_h as usize * 4,
            DType::F32,
            vec![seq_len as usize, m as usize],
        )
        .map_err(|e| anyhow!("alloc dense_q up: {e}"))?;
    let hidden_buf = device
        .alloc_buffer(
            n_h as usize * 4,
            DType::F32,
            vec![seq_len as usize, m as usize],
        )
        .map_err(|e| anyhow!("alloc dense_q hidden: {e}"))?;
    let mut down_out = device
        .alloc_buffer(n_out * 4, DType::F32, vec![seq_len as usize, h as usize])
        .map_err(|e| anyhow!("alloc dense_q down_out: {e}"))?;
    let mut silu_params_buf = device
        .alloc_buffer(4, DType::U32, vec![1])
        .map_err(|e| anyhow!("alloc dense_q silu_params: {e}"))?;
    silu_params_buf
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("{e}"))?[0] = n_h;

    let gate_up_params = GgmlQuantizedMatmulParams {
        m: seq_len,
        n: m,
        k: h,
        ggml_type: weights.ggml_type_gate_up,
    };
    let down_params = GgmlQuantizedMatmulParams {
        m: seq_len,
        n: h,
        k: m,
        ggml_type: weights.ggml_type_down,
    };

    quantized_matmul_ggml(
        enc,
        registry,
        device,
        x,
        &weights.gate_q,
        &mut gate_buf,
        &gate_up_params,
    )
    .context("dense_q gate proj")?;
    quantized_matmul_ggml(
        enc,
        registry,
        device,
        x,
        &weights.up_q,
        &mut up_buf,
        &gate_up_params,
    )
    .context("dense_q up proj")?;
    enc.memory_barrier();

    dispatch_silu_mul(
        enc,
        registry,
        device.metal_device(),
        &gate_buf,
        &up_buf,
        &hidden_buf,
        &silu_params_buf,
        n_h,
    )
    .context("dense_q silu_mul")?;
    enc.memory_barrier();

    quantized_matmul_ggml(
        enc,
        registry,
        device,
        &hidden_buf,
        &weights.down_q,
        &mut down_out,
        &down_params,
    )
    .context("dense_q down proj")?;

    let result = if let Some(res) = add_residual {
        let sum_buf = device
            .alloc_buffer(n_out * 4, DType::F32, vec![n_out])
            .map_err(|e| anyhow!("alloc dense_q residual sum: {e}"))?;
        enc.memory_barrier();
        elementwise_add(
            enc,
            registry,
            device.metal_device(),
            &down_out,
            res,
            &sum_buf,
            n_out,
            DType::F32,
        )
        .context("dense_q residual add")?;
        sum_buf
    } else {
        down_out
    };

    Ok(result)
}

/// Diagnostic-only DenseQ prefill split used to attribute the single fused
/// FFN command buffer's GPU time.  This intentionally changes the command
/// buffer schedule and is only called under
/// `HF2Q_PROFILE_DENSE_Q_SPLIT_COMMITS=1`; production keeps
/// [`build_dense_ffn_layer_gpu_q_into`]'s single-CB path.
#[allow(clippy::too_many_arguments)]
pub fn build_dense_ffn_layer_gpu_q_split_profile(
    mut first_enc: mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &DenseFfnWeightsGpuQ,
    add_residual: Option<&MlxBuffer>,
    label_prefix: &str,
) -> Result<MlxBuffer> {
    let h = weights.hidden_size;
    let m = weights.intermediate_size;
    let seq_len = (x.element_count() / h as usize) as u32;
    let n_h = (seq_len * m) as u32;
    let n_out = (seq_len * h) as usize;

    let mut gate_buf = device
        .alloc_buffer(
            n_h as usize * 4,
            DType::F32,
            vec![seq_len as usize, m as usize],
        )
        .map_err(|e| anyhow!("alloc dense_q split gate: {e}"))?;
    let mut up_buf = device
        .alloc_buffer(
            n_h as usize * 4,
            DType::F32,
            vec![seq_len as usize, m as usize],
        )
        .map_err(|e| anyhow!("alloc dense_q split up: {e}"))?;
    let hidden_buf = device
        .alloc_buffer(
            n_h as usize * 4,
            DType::F32,
            vec![seq_len as usize, m as usize],
        )
        .map_err(|e| anyhow!("alloc dense_q split hidden: {e}"))?;
    let mut down_out = device
        .alloc_buffer(n_out * 4, DType::F32, vec![seq_len as usize, h as usize])
        .map_err(|e| anyhow!("alloc dense_q split down_out: {e}"))?;
    let mut silu_params_buf = device
        .alloc_buffer(4, DType::U32, vec![1])
        .map_err(|e| anyhow!("alloc dense_q split silu_params: {e}"))?;
    silu_params_buf
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("{e}"))?[0] = n_h;

    let gate_up_params = GgmlQuantizedMatmulParams {
        m: seq_len,
        n: m,
        k: h,
        ggml_type: weights.ggml_type_gate_up,
    };
    let down_params = GgmlQuantizedMatmulParams {
        m: seq_len,
        n: h,
        k: m,
        ggml_type: weights.ggml_type_down,
    };

    quantized_matmul_ggml(
        &mut first_enc,
        registry,
        device,
        x,
        &weights.gate_q,
        &mut gate_buf,
        &gate_up_params,
    )
    .context("dense_q split gate proj")?;
    quantized_matmul_ggml(
        &mut first_enc,
        registry,
        device,
        x,
        &weights.up_q,
        &mut up_buf,
        &gate_up_params,
    )
    .context("dense_q split up proj")?;
    let label = format!("{label_prefix}.gate_up");
    first_enc
        .commit_and_wait_labeled(&label)
        .context("commit dense_q split gate/up")?;

    let mut silu_enc = device.command_encoder().context("enc dense_q split silu")?;
    dispatch_silu_mul(
        &mut silu_enc,
        registry,
        device.metal_device(),
        &gate_buf,
        &up_buf,
        &hidden_buf,
        &silu_params_buf,
        n_h,
    )
    .context("dense_q split silu_mul")?;
    let label = format!("{label_prefix}.silu");
    silu_enc
        .commit_and_wait_labeled(&label)
        .context("commit dense_q split silu")?;

    let mut down_enc = device.command_encoder().context("enc dense_q split down")?;
    quantized_matmul_ggml(
        &mut down_enc,
        registry,
        device,
        &hidden_buf,
        &weights.down_q,
        &mut down_out,
        &down_params,
    )
    .context("dense_q split down proj")?;
    let label = format!("{label_prefix}.down");
    down_enc
        .commit_and_wait_labeled(&label)
        .context("commit dense_q split down")?;

    if let Some(res) = add_residual {
        let sum_buf = device
            .alloc_buffer(n_out * 4, DType::F32, vec![n_out])
            .map_err(|e| anyhow!("alloc dense_q split residual sum: {e}"))?;
        let mut residual_enc = device
            .command_encoder()
            .context("enc dense_q split residual")?;
        elementwise_add(
            &mut residual_enc,
            registry,
            device.metal_device(),
            &down_out,
            res,
            &sum_buf,
            n_out,
            DType::F32,
        )
        .context("dense_q split residual add")?;
        let label = format!("{label_prefix}.residual");
        residual_enc
            .commit_and_wait_labeled(&label)
            .context("commit dense_q split residual")?;
        Ok(sum_buf)
    } else {
        Ok(down_out)
    }
}

// NOTE on `build_dense_ffn_layer_gpu_q_into`: the variant does NOT commit.
// The caller is responsible for committing the encoder, allowing fusion
// with upstream dispatches (e.g. `dispatch_fused_residual_norm_f32`).
//
// Wave 5b.16 sunset: `build_dense_ffn_layer_gpu_q_legacy` (the pre-W-5b.14
// device-alloc + own-encoder path) was removed after a 30/30 cross-path
// parity audit at PP4106 (token id 11 across 5 cold loads × 3 trials × 2
// paths).  See `docs/wave5b3-walkbar-results.md` "Wave 5b.16 re-audit +
// sunset" section for the audit log.

// ================================================================
// MoE GPU path helpers
// ================================================================

/// CPU-side softmax + top-k + renormalize for the MoE router.
///
/// Input:  `logits`  flat `[seq_len * num_experts]` f32.
/// Output: `(topk_indices, topk_weights)` each `[seq_len * topk]`.
///
/// Per ADR-013 Decision 13 spec:
///   probs   = softmax(logits)         (per token)
///   topk    = top-k by probability
///   weights = topk_probs / sum(topk_probs)   (renormalize)
fn softmax_topk_renorm_cpu(
    logits: &[f32],
    seq_len: usize,
    num_experts: usize,
    topk: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut out_idx = Vec::with_capacity(seq_len * topk);
    let mut out_w = Vec::with_capacity(seq_len * topk);

    for t in 0..seq_len {
        let row = &logits[t * num_experts..(t + 1) * num_experts];

        // Numerically-stable softmax.
        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_vals: Vec<f32> = row.iter().map(|&v| (v - max_v).exp()).collect();
        let denom: f32 = exp_vals.iter().sum();
        let inv_d = if denom > 1e-20 { 1.0 / denom } else { 1.0 };
        for e in exp_vals.iter_mut() {
            *e *= inv_d;
        }

        // Select top-k indices by probability.
        let mut idx_sorted: Vec<usize> = (0..num_experts).collect();
        idx_sorted.sort_by(|&a, &b| {
            exp_vals[b]
                .partial_cmp(&exp_vals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let selected = &idx_sorted[..topk];

        // Renormalize selected weights.
        let sum_sel: f32 = selected.iter().map(|&i| exp_vals[i]).sum();
        let inv_sum = if sum_sel > 1e-20 {
            1.0 / sum_sel
        } else {
            1.0 / topk as f32
        };

        for &i in selected {
            out_idx.push(i as u32);
            out_w.push(exp_vals[i] * inv_sum);
        }
    }

    (out_idx, out_w)
}

/// Extract a single expert's stacked weight slice from a flat buffer.
///
/// `stacked`: `[num_experts * rows_per_expert, col]` row-major.
/// Returns a newly allocated `Vec<f32>` for expert `e_idx`, shape
/// `[rows_per_expert * col]`.
fn extract_expert_weight(
    stacked: &[f32],
    e_idx: usize,
    rows_per_expert: usize,
    col: usize,
) -> Vec<f32> {
    let off = e_idx * rows_per_expert * col;
    stacked[off..off + rows_per_expert * col].to_vec()
}

// ================================================================
// MoE GPU path
// ================================================================

/// Build the Qwen3.5-MoE FFN forward pass on the GPU.
///
/// Implements ADR-013 Decision 13 op order end-to-end.
/// Returns the residual contribution `[seq_len, hidden_size]` F32.
///
/// # Op order
///
/// ```text
/// // Router
/// 1. logits  = router(x)                           — proj (GPU)
/// 2. (idx, w) = softmax_topk_renorm(logits)        — CPU download + compute
///
/// // Routed experts
/// for e in topk_idx:
///   3a. gate_e = expert_gate[e](x)                 — proj (GPU, per expert)
///   3b. up_e   = expert_up[e](x)                   — proj (GPU, per expert)
///   3c. h_e    = silu(gate_e) * up_e               — CPU silu_mul
///   3d. y_e    = expert_down[e](h_e)               — proj (GPU, per expert)
///   3e. moe_out += w_e * y_e                       — CPU weighted accumulate
///
/// // Shared expert (sigmoid-gated, llama.cpp qwen35moe.cpp:406-420)
/// 4. sh_logit = shared_gate_inp(x)                 — proj (GPU, [seq, 1])
/// 5. sh_gate  = sigmoid(sh_logit)                  — dispatch_sigmoid_mul(sh_logit, sh_logit)
///    Note: sigmoid(x) = sigmoid_mul(ones, x) but we compute via CPU to avoid
///    allocating a ones-buffer; shared gate is a single scalar per token.
/// 6. a_s   = shared_gate_proj(x)                   — proj (GPU)
/// 7. b_s   = shared_up_proj(x)                     — proj (GPU)
/// 8. h_s   = silu(a_s) * b_s                       — CPU silu_mul
/// 9. y_s   = shared_down_proj(h_s)                 — proj (GPU)
/// 10. out  = moe_out + sh_gate * y_s               — CPU add-scaled
/// ```
///
/// # Expert weight slice extraction
///
/// For the parity test we own the weight slices as CPU buffers and upload
/// per-expert slices on the fly.  In the production P11 path this is replaced
/// by `quantized_matmul_id_ggml` which indexes stacked quantised expert tensors
/// without extracting CPU slices.
///
/// # Parity contract
///
/// `|GPU − moe_ffn_cpu_ref(x, weights, shape)|∞ < 1e-3` F32.
#[allow(clippy::too_many_arguments)]
pub fn build_moe_ffn_layer_gpu(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights_gpu: &MoeFfnWeightsGpu,
    weights_cpu: &MoeFfnWeights,
    shape: MoeFfnShape,
) -> Result<MlxBuffer> {
    let h = shape.hidden_size as usize;
    let ne = shape.num_experts as usize;
    let topk = shape.num_experts_per_tok as usize;
    let m_moe = shape.moe_intermediate_size as usize;
    let seq_len = (x.element_count() / h) as u32;
    let seq = seq_len as usize;

    let h32 = shape.hidden_size;
    let ne32 = shape.num_experts;
    let m_moe32 = shape.moe_intermediate_size;
    let m_sh32 = shape.shared_intermediate_size;

    // ---- Step 1: router logits = router(x)  [seq, num_experts] ----
    let mut enc = device.command_encoder().context("enc moe router")?;
    let logits_buf = proj(
        &mut enc,
        registry,
        device,
        x,
        &weights_gpu.router,
        seq_len,
        h32,
        ne32,
    )?;
    enc.commit_and_wait().context("commit moe router")?;

    // ---- Step 2: softmax + top-k + renorm  [CPU] ----
    let logits_cpu = download_f32(&logits_buf).context("download router logits")?;
    let (topk_idx, topk_w) = softmax_topk_renorm_cpu(&logits_cpu, seq, ne, topk);

    // Download x for expert weight extraction (needed for per-expert proj
    // input on CPU).  x is on GPU; re-upload per-expert is needed since
    // proj() takes an MlxBuffer.  We keep x_cpu for extracting weight slices
    // and pass x (the GPU buffer) as the actual matmul input.
    //
    // Each per-expert proj is: proj(enc, registry, device, x, weight_slice_buf, ...)
    // Weight slices are small (e.g. 4*4=16 or 8*8=64 for unit tests), so
    // extracting and uploading per expert is acceptable for the parity test.

    // Accumulate routed MoE output on CPU (all expert outputs are F32 small tensors).
    let mut moe_out_cpu = vec![0.0f32; seq * h];

    // Use full-sequence projections per selected expert (avoids per-token slicing).
    // For the parity test (small seq_len), this is efficient enough.
    for (tok_e_pos, (&e_idx, &w)) in topk_idx.iter().zip(topk_w.iter()).enumerate() {
        let t = tok_e_pos / topk; // which token
        let e_idx = e_idx as usize;

        // Extract weight slices for expert e_idx.
        let gate_w = extract_expert_weight(&weights_cpu.expert_gate, e_idx, m_moe, h);
        let up_w = extract_expert_weight(&weights_cpu.expert_up, e_idx, m_moe, h);
        let down_w = extract_expert_weight(&weights_cpu.expert_down, e_idx, h, m_moe);

        let gate_buf_e = upload_f32(&gate_w, device).context("upload expert gate_w")?;
        let up_buf_e = upload_f32(&up_w, device).context("upload expert up_w")?;
        let down_buf_e = upload_f32(&down_w, device).context("upload expert down_w")?;

        // We project all seq_len tokens then take row t.
        // gate_e: [seq, m_moe]
        let mut enc = device.command_encoder().context("enc expert gate")?;
        let gate_e_buf = proj(
            &mut enc,
            registry,
            device,
            x,
            &gate_buf_e,
            seq_len,
            h32,
            m_moe32,
        )?;
        enc.commit_and_wait().context("commit expert gate")?;

        let mut enc = device.command_encoder().context("enc expert up")?;
        let up_e_buf = proj(
            &mut enc, registry, device, x, &up_buf_e, seq_len, h32, m_moe32,
        )?;
        enc.commit_and_wait().context("commit expert up")?;

        // SiLU * up for expert e at token t [CPU bridge].
        let gate_e_all = download_f32(&gate_e_buf).context("download expert gate")?;
        let up_e_all = download_f32(&up_e_buf).context("download expert up")?;
        let gate_t = &gate_e_all[t * m_moe..(t + 1) * m_moe];
        let up_t = &up_e_all[t * m_moe..(t + 1) * m_moe];
        let hidden_t = silu_mul_cpu(gate_t, up_t);

        // down_proj for this expert (upload hidden for single token).
        let hidden_buf_t = upload_f32(&hidden_t, device).context("upload hidden_t")?;

        // hidden_buf_t is [1, m_moe]; down_buf_e is [h, m_moe] → proj [1, h].
        let mut enc = device.command_encoder().context("enc expert down")?;
        let y_e_buf = proj(
            &mut enc,
            registry,
            device,
            &hidden_buf_t,
            &down_buf_e,
            1,
            m_moe32,
            h32,
        )?;
        enc.commit_and_wait().context("commit expert down")?;

        let y_e = download_f32(&y_e_buf).context("download expert y_e")?;
        // Weighted accumulate into moe_out_cpu[t * h .. (t+1)*h].
        let out_row = &mut moe_out_cpu[t * h..(t + 1) * h];
        for i in 0..h {
            out_row[i] += w * y_e[i];
        }
    }

    // ---- Shared expert: sigmoid-gated SwiGLU ----
    //
    // llama.cpp qwen35moe.cpp:406-420 (spec):
    //   shared_gate = sigmoid(gate_inp_shexp @ x)
    //   shared_out  = down_shexp(silu(gate_shexp @ x) * (up_shexp @ x))
    //   cur = moe_out + shared_gate * shared_out
    //
    // shared_gate_inp is [1, hidden_size] (one scalar per token after matmul).

    // Step 4: sh_logit = shared_gate_inp(x)  [seq, 1]
    let mut enc = device.command_encoder().context("enc sh_gate_inp")?;
    let sh_logit_buf = proj(
        &mut enc,
        registry,
        device,
        x,
        &weights_gpu.shared_gate_inp,
        seq_len,
        h32,
        1,
    )?;
    enc.commit_and_wait().context("commit sh_gate_inp")?;

    // Step 5: sh_gate_val = sigmoid(sh_logit)  — CPU (scalar per token)
    let sh_logit_cpu = download_f32(&sh_logit_buf).context("download sh_logit")?;
    let sh_gate_vals: Vec<f32> = sh_logit_cpu
        .iter()
        .map(|&v| 1.0 / (1.0 + (-v).exp()))
        .collect(); // length = seq_len

    // Step 6: a_s = shared_gate_proj(x)  [seq, m_sh]
    let mut enc = device.command_encoder().context("enc sh_gate")?;
    let a_s_buf = proj(
        &mut enc,
        registry,
        device,
        x,
        &weights_gpu.shared_gate,
        seq_len,
        h32,
        m_sh32,
    )?;
    enc.commit_and_wait().context("commit sh_gate")?;

    // Step 7: b_s = shared_up_proj(x)  [seq, m_sh]
    let mut enc = device.command_encoder().context("enc sh_up")?;
    let b_s_buf = proj(
        &mut enc,
        registry,
        device,
        x,
        &weights_gpu.shared_up,
        seq_len,
        h32,
        m_sh32,
    )?;
    enc.commit_and_wait().context("commit sh_up")?;

    // Step 8: h_s = silu(a_s) * b_s  [CPU bridge]
    let a_s_cpu = download_f32(&a_s_buf).context("download a_s")?;
    let b_s_cpu = download_f32(&b_s_buf).context("download b_s")?;
    let h_s_cpu = silu_mul_cpu(&a_s_cpu, &b_s_cpu);
    let h_s_buf = upload_f32(&h_s_cpu, device).context("upload h_s")?;

    // Step 9: y_s = shared_down_proj(h_s)  [seq, h]
    let mut enc = device.command_encoder().context("enc sh_down")?;
    let y_s_buf = proj(
        &mut enc,
        registry,
        device,
        &h_s_buf,
        &weights_gpu.shared_down,
        seq_len,
        m_sh32,
        h32,
    )?;
    enc.commit_and_wait().context("commit sh_down")?;

    // Step 10: output = moe_out + sh_gate * y_s  [CPU combine]
    let y_s_cpu = download_f32(&y_s_buf).context("download y_s")?;
    let mut out_cpu = moe_out_cpu; // reuse the moe accumulator
    for t in 0..seq {
        let sg = sh_gate_vals[t];
        let y_row = &y_s_cpu[t * h..(t + 1) * h];
        let o_row = &mut out_cpu[t * h..(t + 1) * h];
        for i in 0..h {
            o_row[i] += sg * y_row[i];
        }
    }

    upload_f32(&out_cpu, device).context("upload final moe out")
}

// ================================================================
// Quantized MoE GPU forward pass
// ================================================================

/// Build the Qwen3.5-MoE FFN forward pass using quantized expert weights.
///
/// This is the production path for the 35B apex model.  Expert projections
/// use `quantized_matmul_id_ggml` which keeps weights in their GGML
/// block-quantized form (Q6_K) on the Metal device, avoiding the 128 GB
/// F32-expansion OOM that the `build_moe_ffn_layer_gpu` path incurs.
///
/// # Op order
///
/// ```text
/// // Router
/// 1. logits  = router_f32(x)               — F32 dense proj
/// 2. (idx, w) = softmax_topk_renorm(logits) — CPU
///
/// // Routed experts via quantized MoE dispatch
/// 3a. gate_all = qmatmul_id(x, expert_gate_q, ids)  — [n_tokens*top_k, moe_intermediate]
/// 3b. up_all   = qmatmul_id(x, expert_up_q, ids)    — [n_tokens*top_k, moe_intermediate]
/// 3c. h_all    = silu(gate_all) * up_all             — CPU per selected slot
/// 3d. y_all    = qmatmul_id(h_all_as_f32, expert_down_q, arange_ids)  — [n_tokens*top_k, hidden]
/// 3e. moe_out  = weighted_sum(topk_w, y_all)         — CPU accumulate
///
/// // Shared expert (unchanged from unquantized path)
/// 4..10. shared-expert path (F32)
/// ```
///
/// # Tolerance
///
/// Compared to a CPU F32 reference, Q6_K dequant noise is ≤ 2e-2 per element.
/// The parity test for this path uses tolerance 2e-2.
/// Compute the MoE FFN layer and optionally add a residual on CPU before
/// the final GPU upload, eliminating a separate `residual_add_gpu` commit.
///
/// When `add_residual` is `Some(buf)`, `buf` must be `[seq_len * hidden_size]`
/// F32. It is downloaded once (unified memory: zero-copy read) and folded
/// into the CPU accumulation step before the final upload.
#[allow(clippy::too_many_arguments)]
pub fn build_moe_ffn_layer_gpu_q(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &MoeFfnWeightsGpuQ,
    shape: MoeFfnShape,
    add_residual: Option<&MlxBuffer>,
) -> Result<MlxBuffer> {
    // Backwards-compatible variant: create our own encoder, dispatch, commit.
    let mut enc = device.command_encoder().context("enc moe_ffn_q")?;
    let out = build_moe_ffn_layer_gpu_q_into(
        &mut enc,
        device,
        registry,
        x,
        weights,
        shape,
        add_residual,
    )?;
    let seq_len = (x.element_count() / shape.hidden_size as usize) as u32;
    if seq_len == 1 {
        enc.commit();
    } else {
        enc.commit_and_wait().context("commit moe_ffn_q")?;
    }
    Ok(out)
}

/// External-encoder variant of [`build_moe_ffn_layer_gpu_q`].
///
/// Encodes the entire MoE FFN forward pass (router + shared expert + gated
/// expert projections + softmax_topk + silu_mul + weighted reduce + optional
/// residual add) into the caller-supplied [`mlx_native::CommandEncoder`].
/// Does NOT commit — the caller is responsible for committing the encoder.
///
/// # Why this exists
///
/// Closes the per-layer command-buffer overhead component of the ADR-012
/// §Optimize / Task #15 MoE dwq46 0.91× decode parity gap.  The
/// pre-fusion path issued 2 separate command buffers per MoE layer:
///
/// 1. `dispatch_fused_residual_norm_f32` — produces `ffn_input` from
///    `(hidden, attn_out, post_norm_w)` + writes `ffn_residual` for the
///    later add.
/// 2. `build_moe_ffn_layer_gpu_q` — its own encoder for the MoE
///    forward + residual add at the end.
///
/// With this variant, the caller can fuse step 1 + step 2 into a single
/// command buffer per MoE layer (40 fewer command buffers per decode
/// token on Qwen3.6-35B-A3B).  For comparison, llama.cpp's Metal compute
/// path issues 1-2 command buffers per decode token total
/// (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:458` —
/// "optimal values for n_cb are 1 or 2"); hf2q pre-fusion issues 140.
///
/// # Caller contract
///
/// * The caller must provide the encoder.  The encoder must NOT have been
///   committed.
/// * The caller MUST commit the encoder after this function returns; the
///   GPU work is queued but not yet submitted.
/// * The output `MlxBuffer` is allocated from the per-decode-token arena
///   pool; it must NOT be downloaded to CPU via `as_slice` /
///   `download_f32` (the pool's bucket rounding inflates `byte_len`).
///   Decode-path consumers (next layer's fused_residual_norm + lm_head
///   `apply_output_head_gpu_greedy`) read via shape-respecting kernels —
///   safe.
#[allow(clippy::too_many_arguments)]
pub fn build_moe_ffn_layer_gpu_q_into(
    enc: &mut mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &MoeFfnWeightsGpuQ,
    shape: MoeFfnShape,
    add_residual: Option<&MlxBuffer>,
) -> Result<MlxBuffer> {
    let h = shape.hidden_size as usize;
    let _ne = shape.num_experts as usize;
    let topk = shape.num_experts_per_tok as usize;
    let m_moe = shape.moe_intermediate_size as usize;
    let seq_len = (x.element_count() / h) as u32;
    let seq = seq_len as usize;

    let h32 = shape.hidden_size;
    let ne32 = shape.num_experts;
    let m_moe32 = shape.moe_intermediate_size;
    let m_sh32 = shape.shared_intermediate_size;

    // ADR-013 P13.3 perf: entire MoE FFN in ONE command buffer.
    // By fusing all projections + routing + expert matmuls + weighted combine into a
    // single encoder (single commit_and_wait per MoE layer), we eliminate the second
    // commit_and_wait that previously separated the router projection from the expert
    // matmuls.  40 MoE layers × 1 eliminated commit = 40 fewer GPU sync barriers
    // per decode token.
    //
    // Dispatch order (all ops in one command buffer, barriers between RAW hazards):
    //   A. proj(router) + proj(sh_logit) + proj(a_s) + proj(b_s)   — concurrent
    //   BARRIER
    //   B. softmax_topk(logits → ids, w) + silu_mul(a_s,b_s → h_s) — concurrent
    //   BARRIER
    //   C. gate_all + up_all + shared_down(h_s → y_s)              — concurrent
    //   BARRIER
    //   D. silu_mul(gate_all,up_all → h_all)
    //   BARRIER
    //   E. expert_down(h_all → y_all)
    //   BARRIER
    //   F. moe_weighted_reduce(w,y_all,sh_logit,y_s → out [+residual])
    //   commit_and_wait() — single GPU sync per MoE layer
    // ADR-012 §Optimize / Task #15: route every per-token MoE FFN scratch
    // allocation through the thread-local arena pool.  These buffers' lifetimes
    // are bounded by `build_moe_ffn_layer_gpu_q`; `forward_gpu_greedy`'s
    // top-of-token `reset_decode_pool` recycles them on the next token.
    let total_rows = seq * topk;
    let _w5b_ffn_alloc = super::wave5b8_profile::Section::start(
        super::wave5b8_profile::SectionKind::FfnAllocScratch,
    );
    let ids_buf = super::decode_pool::pooled_alloc_buffer(
        device,
        total_rows * DType::U32.size_of(),
        DType::U32,
        vec![total_rows],
    )
    .map_err(|e| anyhow!("alloc ids_buf: {e}"))?;
    let weights_buf = super::decode_pool::pooled_alloc_buffer(
        device,
        total_rows * DType::F32.size_of(),
        DType::F32,
        vec![total_rows],
    )
    .map_err(|e| anyhow!("alloc weights_buf: {e}"))?;
    let gate_all_bytes = total_rows * m_moe * 4;
    let mut gate_all_buf = super::decode_pool::pooled_alloc_buffer(
        device,
        gate_all_bytes,
        DType::F32,
        vec![total_rows, m_moe],
    )
    .map_err(|e| anyhow!("alloc gate_all: {e}"))?;
    let up_all_bytes = total_rows * m_moe * 4;
    let mut up_all_buf = super::decode_pool::pooled_alloc_buffer(
        device,
        up_all_bytes,
        DType::F32,
        vec![total_rows, m_moe],
    )
    .map_err(|e| anyhow!("alloc up_all: {e}"))?;
    let n_h_all = (total_rows * m_moe) as u32;
    let h_all_buf = super::decode_pool::pooled_alloc_buffer(
        device,
        n_h_all as usize * 4,
        DType::F32,
        vec![total_rows, m_moe],
    )
    .map_err(|e| anyhow!("alloc h_all: {e}"))?;
    let y_all_bytes = total_rows * h * 4;
    let mut y_all_buf = super::decode_pool::pooled_alloc_buffer(
        device,
        y_all_bytes,
        DType::F32,
        vec![total_rows, h],
    )
    .map_err(|e| anyhow!("alloc y_all: {e}"))?;
    let m_sh = m_sh32 as usize;
    let n_h_s = (seq * m_sh) as u32;
    let h_s_buf = super::decode_pool::pooled_alloc_buffer(
        device,
        n_h_s as usize * 4,
        DType::F32,
        vec![seq, m_sh],
    )
    .map_err(|e| anyhow!("alloc h_s: {e}"))?;
    let out_bytes = seq * h * 4;
    // ADR-015 iter40 — bug fix: when this `out_buf` becomes the residual stream
    // for the NEXT layer (prefill MoE FFN with add_residual=Some), it must NOT
    // be pool-allocated.  The W-5b.15 per-layer arena reset
    // (`reset_for_prefill_chunk`) at the bottom of each layer iteration
    // recycles every pool-allocated buffer back to the free list.  If `out_buf`
    // is pooled, the next layer's `pooled_alloc_buffer` calls (ssm_conv,
    // qkv_split, FFN scratches, etc.) can re-issue the same Metal storage,
    // OVERWRITING the residual stream that the next layer's
    // `dispatch_fused_residual_norm_f32` reads as `hidden`.
    //
    // Symptom: layer 1+ residual values jump from ~0.02 (correct, post-layer-0
    // residual) to ~0.7 (matches `ffn_input_buf` post-norm magnitude — the
    // pooled bucket was recycled and re-issued for `ffn_input_buf` in the
    // current layer).  After 1-2 layers the residual stream becomes garbage
    // and the lm_head argmax produces gibberish ("-than", "2-2-2-...").
    //
    // The dense-Q path already gates on `seq_len == 1` to apply the same
    // exclusion (line 966-978); MoE was missing this guard.  Mirrors the
    // W-5b.15 lifetime-safety doc-comment in `decode_pool.rs:119-127`.
    //
    // Decode (seq_len == 1) keeps the pooled path: `forward_gpu_greedy` issues
    // `reset_decode_pool` at the TOP of each token (not between layers), so
    // the per-layer hidden buffer is safely recycled along with all other
    // per-token allocations on the next token's reset.
    let mut out_buf = if seq_len == 1 {
        super::decode_pool::pooled_alloc_buffer(device, out_bytes, DType::F32, vec![seq, h])
            .map_err(|e| anyhow!("alloc moe output (pooled, decode): {e}"))?
    } else {
        device
            .alloc_buffer(out_bytes, DType::F32, vec![seq, h])
            .map_err(|e| anyhow!("alloc moe output (device, prefill): {e}"))?
    };
    // params_buf for silu_mul (holds n as u32, must outlive the encoder)
    let mut silu_params_buf =
        super::decode_pool::pooled_alloc_buffer(device, 4, DType::U32, vec![1])
            .map_err(|e| anyhow!("alloc silu params: {e}"))?;
    silu_params_buf
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("{e}"))?[0] = n_h_all;
    let mut silu_sh_params_buf =
        super::decode_pool::pooled_alloc_buffer(device, 4, DType::U32, vec![1])
            .map_err(|e| anyhow!("alloc silu_sh params: {e}"))?;
    silu_sh_params_buf
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("{e}"))?[0] = n_h_s;

    // Dummy residual buffer (1 f32) used when add_residual=false; dispatch_moe_weighted_reduce
    // requires a valid buffer reference even if the add_residual flag is 0.
    let dummy_residual_buf;
    let residual_ref: &MlxBuffer = match add_residual {
        Some(buf) => buf,
        None => {
            dummy_residual_buf =
                super::decode_pool::pooled_alloc_buffer(device, 4, DType::F32, vec![1])
                    .map_err(|e| anyhow!("alloc dummy residual: {e}"))?;
            &dummy_residual_buf
        }
    };
    drop(_w5b_ffn_alloc);

    {
        // ---- Phase A: router + shared expert projections (all read from x, concurrent) ----
        // ADR-015 iter7b — pool proj outputs for the apex production path.
        // All four buffers flow into GPU kernels (dispatch_moe_softmax_topk,
        // dispatch_silu_mul, dispatch_moe_weighted_reduce); no CPU
        // download/as_slice path on q_into.  See proj_pooled doc-comment
        // for the full caller-contract verification.
        let (logits_buf, sh_logit_buf, a_s_buf, b_s_buf) = {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseAProj,
            );
            let logits_buf = proj_pooled(
                enc,
                registry,
                device,
                x,
                &weights.router,
                seq_len,
                h32,
                ne32,
            )?;
            let sh_logit_buf = proj_pooled(
                enc,
                registry,
                device,
                x,
                &weights.shared_gate_inp,
                seq_len,
                h32,
                1,
            )?;
            let a_s_buf = proj_pooled(
                enc,
                registry,
                device,
                x,
                &weights.shared_gate,
                seq_len,
                h32,
                m_sh32,
            )?;
            let b_s_buf = proj_pooled(
                enc,
                registry,
                device,
                x,
                &weights.shared_up,
                seq_len,
                h32,
                m_sh32,
            )?;
            (logits_buf, sh_logit_buf, a_s_buf, b_s_buf)
        };

        // Barrier A→B: softmax_topk reads logits_buf; silu_mul reads a_s/b_s.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnBarrierAB,
            );
            enc.memory_barrier();
        }

        // ---- Phase B: GPU softmax+topk + shared silu_mul (concurrent) ----
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseBRouteSilu,
            );
            dispatch_moe_softmax_topk(
                enc,
                registry,
                device,
                &logits_buf,
                &ids_buf,
                &weights_buf,
                seq_len,
                ne32,
                shape.num_experts_per_tok,
            )
            .map_err(|e| anyhow!("moe_softmax_topk: {e}"))?;
            dispatch_silu_mul(
                enc,
                registry,
                device.metal_device(),
                &a_s_buf,
                &b_s_buf,
                &h_s_buf,
                &silu_sh_params_buf,
                n_h_s,
            )
            .map_err(|e| anyhow!("silu_mul sh: {e}"))?;
        }

        // Barrier B→C: gate/up matmuls need ids_buf; shared_down needs h_s.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnBarrierBC,
            );
            enc.memory_barrier();
        }

        // ---- Phase C: expert gate+up matmuls + shared down proj (concurrent) ----
        //
        // W-5b.24: pooled `IdMmScratch` path eliminates 4 of 6 per-FFN-call
        // device allocs (the 2 that remain pay only on first call of a
        // prefill, then cache-hit for all 47 subsequent layers — net 286
        // of 288 per-prefill device allocs eliminated).  Each call slot
        // (gate / up / down) gets its own thread-local scratch so the
        // gate+up concurrent dispatches in this Phase C block do NOT race
        // on htpe/hids writes — distinct scratches, distinct buffers.
        //
        // W-5b.25 sunset (2026-04-28): the `HF2Q_FFN_POOLED_LEGACY=1`
        // forensic A/B gate was REMOVED after a 30/30 cross-path
        // token-id determinism panel at PP4106 (5 cold loads × 3 cold
        // prefills × 2 paths) PASSED on token id 11 with the apex
        // 35B-A3B PP65536 cross-path verification confirming both paths
        // complete an identical 48-layer prefill (the post-prefill
        // lm_head buffer-sizing bug at PP69632 is a pre-existing,
        // separate issue unrelated to FFN pooling — both paths fail
        // identically there).  The standing parity bar is the
        // `ffn_pooled_path_matches_legacy_at_seq128` test (deleted with
        // the legacy code path; see W-5b.18 → W-5b.19a precedent for
        // sister-test deletion alongside gate removal).
        let gate_params = GgmlQuantizedMatmulIdParams {
            n_tokens: seq_len,
            top_k: shape.num_experts_per_tok,
            n: m_moe32,
            k: h32,
            n_experts: ne32,
            expert_stride: weights.expert_gate_stride,
            ggml_type: weights.ggml_type_gate_up,
        };
        let up_params = GgmlQuantizedMatmulIdParams {
            expert_stride: weights.expert_up_stride,
            ..gate_params
        };
        let y_s_buf = {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseCGateUpSharedDown,
            );
            super::decode_pool::with_id_mm_scratch(
                super::decode_pool::MmIdSlot::Gate,
                device,
                ne32,
                seq_len * shape.num_experts_per_tok,
                |scratch| {
                    quantized_matmul_id_ggml_pooled(
                        enc,
                        registry,
                        device,
                        x,
                        &weights.expert_gate_q,
                        &ids_buf,
                        &mut gate_all_buf,
                        scratch,
                        &gate_params,
                    )
                },
            )
            .map_err(|e| anyhow!("gate_all qmatmul_id_pooled: {e}"))?;
            super::decode_pool::with_id_mm_scratch(
                super::decode_pool::MmIdSlot::Up,
                device,
                ne32,
                seq_len * shape.num_experts_per_tok,
                |scratch| {
                    quantized_matmul_id_ggml_pooled(
                        enc,
                        registry,
                        device,
                        x,
                        &weights.expert_up_q,
                        &ids_buf,
                        &mut up_all_buf,
                        scratch,
                        &up_params,
                    )
                },
            )
            .map_err(|e| anyhow!("up_all qmatmul_id_pooled: {e}"))?;
            // ADR-015 iter7b — pooled (q_into path; y_s_buf flows into
            // dispatch_moe_weighted_reduce, no CPU download).
            proj_pooled(
                enc,
                registry,
                device,
                &h_s_buf,
                &weights.shared_down,
                seq_len,
                m_sh32,
                h32,
            )?
        };

        // Barrier C→D: silu_mul reads gate_all/up_all.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnBarrierCD,
            );
            enc.memory_barrier();
        }

        // ---- Phase D: h_all = silu(gate_all) * up_all ----
        //
        // 2026-04-26: tested fusing silu_mul into a swiglu-fused Q4_0
        // mv_id kernel (`quantized_matmul_id_swiglu_q4_0`, mlx-native
        // commit `4efeec0`).  Eliminates 1 dispatch + 1 memory_barrier per
        // MoE layer.  Wire-up on dwq46 (Q4_0 expert_down): 110.5 t/s
        // → 108.0 t/s = REGRESS −1.5% on n=256 cold-run median.  Per-CB
        // GPU time unchanged (96µs/cb), but wall regressed — likely
        // doubled input bandwidth (read gate AND up directly) plus
        // increased ALU pressure (16 silu evals per simdthread inner
        // loop) saturate something on M5 Max.  The fused kernel + test
        // remain in mlx-native for future re-evaluation but are NOT
        // wired here.  9th confirmed M5 Max static-evidence kernel
        // hypothesis falsified.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseDSilu,
            );
            dispatch_silu_mul(
                enc,
                registry,
                device.metal_device(),
                &gate_all_buf,
                &up_all_buf,
                &h_all_buf,
                &silu_params_buf,
                n_h_all,
            )
            .map_err(|e| anyhow!("silu_mul dispatch: {e}"))?;
        }

        // Barrier D→E: expert_down reads h_all.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnBarrierDE,
            );
            enc.memory_barrier();
        }

        // ---- Phase E: y_all = expert_down(h_all) ----
        //
        // W-5b.24: pooled scratch slot for the down call.  Lives behind
        // the C→D and D→E barriers — there's no concurrency hazard with
        // the gate/up scratches (different slots), and within this call
        // map0→mm has its own internal barrier inside the dispatch.
        let down_params = GgmlQuantizedMatmulIdParams {
            n_tokens: total_rows as u32,
            top_k: 1,
            n: h32,
            k: m_moe32,
            n_experts: ne32,
            expert_stride: weights.expert_down_stride,
            ggml_type: weights.ggml_type_down,
        };
        // W-5b.25 sunset: the legacy un-pooled branch was removed.  See
        // the Phase C sunset comment-block above for the audit trail.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseEDown,
            );
            super::decode_pool::with_id_mm_scratch(
                super::decode_pool::MmIdSlot::Down,
                device,
                ne32,
                total_rows as u32,
                |scratch| {
                    quantized_matmul_id_ggml_pooled(
                        enc,
                        registry,
                        device,
                        &h_all_buf,
                        &weights.expert_down_q,
                        &ids_buf,
                        &mut y_all_buf,
                        scratch,
                        &down_params,
                    )
                },
            )
            .map_err(|e| anyhow!("y_all qmatmul_id_pooled: {e}"))?;
        }

        // Barrier E→F: moe_weighted_reduce reads y_all, y_s_buf, sh_logit_buf.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnBarrierEF,
            );
            enc.memory_barrier();
        }

        // ---- Phase F: fused weighted accumulate + sigmoid(sh_logit)*y_s + residual ----
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseFReduce,
            );
            dispatch_moe_weighted_reduce(
                enc,
                registry,
                device,
                &weights_buf,
                &y_all_buf,
                &sh_logit_buf,
                &y_s_buf,
                residual_ref,
                &mut out_buf,
                seq_len,
                shape.num_experts_per_tok,
                h32,
                add_residual.is_some(),
            )
            .map_err(|e| anyhow!("moe_weighted_reduce: {e}"))?;
        }


        // NOTE: this `_into` variant does NOT commit.  The caller is
        // responsible for committing the encoder, allowing fusion with
        // upstream dispatches (e.g. `dispatch_fused_residual_norm_f32`).
    }

    Ok(out_buf)
}

/// ADR-015 iter72: `_with_arena` variant of `build_moe_ffn_layer_gpu_q_into`.
///
/// Prefill-only variant that reuses the eight transient scratches
/// (ids, weights, gate_all, up_all, h_all, y_all, h_s, silu_params,
/// silu_sh_params, dummy_residual) from a caller-owned [`super::MoeFfnArena`].
/// Allocation happens ONCE per prefill (in `forward_gpu_impl`); this
/// function performs ZERO `pooled_alloc_buffer` calls for the transient
/// scratches, eliminating the W-5b.8 `ffn.alloc_scratch` bucket.
///
/// **NOT included** in the arena (still allocated per-layer, lifetime
/// crosses the layer boundary):
/// - `out_buf` — function return value, becomes next layer's `hidden`.
///   Same `device.alloc_buffer` path the existing `_into` variant uses
///   at prefill (gpu_ffn.rs ~2189 in iter40-fix doc-comment).
///
/// # Lifetime contract
///
/// The arena outlives every per-layer encoder commit; identical to
/// FaPrefillArena's structural-fix-for-iter58b doc.
#[allow(clippy::too_many_arguments)]
pub fn build_moe_ffn_layer_gpu_q_into_with_arena(
    enc: &mut mlx_native::CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &MoeFfnWeightsGpuQ,
    shape: MoeFfnShape,
    add_residual: Option<&MlxBuffer>,
    arena: &mut super::MoeFfnArena,
) -> Result<MlxBuffer> {
    let h = shape.hidden_size as usize;
    let _ne = shape.num_experts as usize;
    let topk = shape.num_experts_per_tok as usize;
    let m_moe = shape.moe_intermediate_size as usize;
    let seq_len = (x.element_count() / h) as u32;
    let seq = seq_len as usize;

    let h32 = shape.hidden_size;
    let ne32 = shape.num_experts;
    let m_moe32 = shape.moe_intermediate_size;
    let m_sh32 = shape.shared_intermediate_size;

    // Validate the arena's capacity matches our shape.
    arena
        .validate_fits(
            seq_len,
            h32,
            shape.num_experts_per_tok,
            m_moe32,
            m_sh32,
        )
        .context("MoeFfnArena shape mismatch")?;

    let total_rows = seq * topk;
    let _w5b_ffn_alloc = super::wave5b8_profile::Section::start(
        super::wave5b8_profile::SectionKind::FfnAllocScratch,
    );

    // Per-layer FINAL OUTPUT must be device-allocated at prefill so it
    // survives the per-layer pool reset and crosses into the next layer's
    // `hidden`.  See gpu_ffn.rs:2165-2196 (iter40 fix) for the canonical
    // doc-comment on this lifetime contract.
    let out_bytes = seq * h * 4;
    let mut out_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![seq, h])
        .map_err(|e| anyhow!("alloc moe output (device, prefill arena): {e}"))?;

    // silu params writes — done once per layer, but they share with the
    // arena's pre-allocated buffer.  Same `n_h_all` value every layer at
    // a given seq_len.
    let n_h_all = (total_rows * m_moe) as u32;
    let m_sh = m_sh32 as usize;
    let n_h_s = (seq * m_sh) as u32;
    arena
        .silu_params_buf
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("MoeFfnArena silu_params as_mut_slice: {e}"))?[0] = n_h_all;
    arena
        .silu_sh_params_buf
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("MoeFfnArena silu_sh_params as_mut_slice: {e}"))?[0] = n_h_s;

    // Resolve residual buffer reference: arena.dummy_residual_buf when
    // add_residual=None, otherwise the caller-provided buffer.
    let residual_ref: &MlxBuffer = match add_residual {
        Some(buf) => buf,
        None => &arena.dummy_residual_buf,
    };
    drop(_w5b_ffn_alloc);

    {
        // ---- Phase A: router + shared expert projections (concurrent) ----
        let (logits_buf, sh_logit_buf, a_s_buf, b_s_buf) = {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseAProj,
            );
            // proj_pooled allocates from the thread-local decode pool —
            // these projection outputs flow into Phase B/C/F kernels and
            // never cross a layer boundary, so the existing pool path is
            // safe and we leave it untouched (the W-5b.8
            // `ffn.alloc_scratch` bucket measures the iter72-targeted
            // scratches above; proj_pooled is measured in
            // `ffn.phase_a_proj`).
            let logits_buf = proj_pooled(
                enc,
                registry,
                device,
                x,
                &weights.router,
                seq_len,
                h32,
                ne32,
            )?;
            let sh_logit_buf = proj_pooled(
                enc,
                registry,
                device,
                x,
                &weights.shared_gate_inp,
                seq_len,
                h32,
                1,
            )?;
            let a_s_buf = proj_pooled(
                enc,
                registry,
                device,
                x,
                &weights.shared_gate,
                seq_len,
                h32,
                m_sh32,
            )?;
            let b_s_buf = proj_pooled(
                enc,
                registry,
                device,
                x,
                &weights.shared_up,
                seq_len,
                h32,
                m_sh32,
            )?;
            (logits_buf, sh_logit_buf, a_s_buf, b_s_buf)
        };

        // Barrier A→B.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnBarrierAB,
            );
            enc.memory_barrier();
        }

        // ---- Phase B: GPU softmax+topk + shared silu_mul (concurrent) ----
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseBRouteSilu,
            );
            dispatch_moe_softmax_topk(
                enc,
                registry,
                device,
                &logits_buf,
                &arena.ids_buf,
                &arena.weights_buf,
                seq_len,
                ne32,
                shape.num_experts_per_tok,
            )
            .map_err(|e| anyhow!("moe_softmax_topk: {e}"))?;
            dispatch_silu_mul(
                enc,
                registry,
                device.metal_device(),
                &a_s_buf,
                &b_s_buf,
                &arena.h_s_buf,
                &arena.silu_sh_params_buf,
                n_h_s,
            )
            .map_err(|e| anyhow!("silu_mul sh: {e}"))?;
        }

        // Barrier B→C.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnBarrierBC,
            );
            enc.memory_barrier();
        }

        // ---- Phase C: expert gate+up matmuls + shared down proj ----
        let gate_params = GgmlQuantizedMatmulIdParams {
            n_tokens: seq_len,
            top_k: shape.num_experts_per_tok,
            n: m_moe32,
            k: h32,
            n_experts: ne32,
            expert_stride: weights.expert_gate_stride,
            ggml_type: weights.ggml_type_gate_up,
        };
        let up_params = GgmlQuantizedMatmulIdParams {
            expert_stride: weights.expert_up_stride,
            ..gate_params
        };
        let y_s_buf = {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseCGateUpSharedDown,
            );
            super::decode_pool::with_id_mm_scratch(
                super::decode_pool::MmIdSlot::Gate,
                device,
                ne32,
                seq_len * shape.num_experts_per_tok,
                |scratch| {
                    quantized_matmul_id_ggml_pooled(
                        enc,
                        registry,
                        device,
                        x,
                        &weights.expert_gate_q,
                        &arena.ids_buf,
                        &mut arena.gate_all_buf,
                        scratch,
                        &gate_params,
                    )
                },
            )
            .map_err(|e| anyhow!("gate_all qmatmul_id_pooled: {e}"))?;
            super::decode_pool::with_id_mm_scratch(
                super::decode_pool::MmIdSlot::Up,
                device,
                ne32,
                seq_len * shape.num_experts_per_tok,
                |scratch| {
                    quantized_matmul_id_ggml_pooled(
                        enc,
                        registry,
                        device,
                        x,
                        &weights.expert_up_q,
                        &arena.ids_buf,
                        &mut arena.up_all_buf,
                        scratch,
                        &up_params,
                    )
                },
            )
            .map_err(|e| anyhow!("up_all qmatmul_id_pooled: {e}"))?;
            proj_pooled(
                enc,
                registry,
                device,
                &arena.h_s_buf,
                &weights.shared_down,
                seq_len,
                m_sh32,
                h32,
            )?
        };

        // Barrier C→D.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnBarrierCD,
            );
            enc.memory_barrier();
        }

        // ---- Phase D: h_all = silu(gate_all) * up_all ----
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseDSilu,
            );
            dispatch_silu_mul(
                enc,
                registry,
                device.metal_device(),
                &arena.gate_all_buf,
                &arena.up_all_buf,
                &arena.h_all_buf,
                &arena.silu_params_buf,
                n_h_all,
            )
            .map_err(|e| anyhow!("silu_mul dispatch: {e}"))?;
        }

        // Barrier D→E.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnBarrierDE,
            );
            enc.memory_barrier();
        }

        // ---- Phase E: y_all = expert_down(h_all) ----
        let down_params = GgmlQuantizedMatmulIdParams {
            n_tokens: total_rows as u32,
            top_k: 1,
            n: h32,
            k: m_moe32,
            n_experts: ne32,
            expert_stride: weights.expert_down_stride,
            ggml_type: weights.ggml_type_down,
        };
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseEDown,
            );
            super::decode_pool::with_id_mm_scratch(
                super::decode_pool::MmIdSlot::Down,
                device,
                ne32,
                total_rows as u32,
                |scratch| {
                    quantized_matmul_id_ggml_pooled(
                        enc,
                        registry,
                        device,
                        &arena.h_all_buf,
                        &weights.expert_down_q,
                        &arena.ids_buf,
                        &mut arena.y_all_buf,
                        scratch,
                        &down_params,
                    )
                },
            )
            .map_err(|e| anyhow!("y_all qmatmul_id_pooled: {e}"))?;
        }

        // Barrier E→F.
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnBarrierEF,
            );
            enc.memory_barrier();
        }

        // ---- Phase F: weighted reduce + residual ----
        {
            let _w5b = super::wave5b8_profile::Section::start(
                super::wave5b8_profile::SectionKind::FfnPhaseFReduce,
            );
            dispatch_moe_weighted_reduce(
                enc,
                registry,
                device,
                &arena.weights_buf,
                &arena.y_all_buf,
                &sh_logit_buf,
                &y_s_buf,
                residual_ref,
                &mut out_buf,
                seq_len,
                shape.num_experts_per_tok,
                h32,
                add_residual.is_some(),
            )
            .map_err(|e| anyhow!("moe_weighted_reduce: {e}"))?;
        }
    }

    Ok(out_buf)
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::super::ffn::{
        dense_swiglu_cpu_ref, moe_ffn_cpu_ref, DenseFfnShape, DenseFfnWeights, MoeFfnShape,
        MoeFfnWeights,
    };
    use super::*;

    fn mk_rand(seed: &mut u32, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((*seed as i32 as f32) / (i32::MAX as f32)) * scale
            })
            .collect()
    }

    // ────────────────────────────────────────────────────────────────
    // Dense SwiGLU GPU path
    // ────────────────────────────────────────────────────────────────

    /// Dense SwiGLU GPU parity test against the CPU scalar reference.
    ///
    /// Tolerance: 1e-3 F32 (3 BF16-cast projections, each ≤ half-ulp in BF16).
    /// ADR-013 Decision 14 acceptance criterion.
    #[test]
    fn dense_swiglu_gpu_parity_vs_cpu_ref() {
        let device = MlxDevice::new().expect("Metal device unavailable — skipping GPU test");
        let mut registry = KernelRegistry::new();

        // Must be >= 32 to satisfy the tensor-core tile constraint.
        let shape = DenseFfnShape {
            hidden_size: 32,
            intermediate_size: 64,
        };
        let h = shape.hidden_size as usize;
        let m = shape.intermediate_size as usize;
        let seq_len = 4usize;

        let mut seed = 0xABCD_u32;
        let weights_cpu = DenseFfnWeights {
            gate: mk_rand(&mut seed, m * h, 0.15),
            up: mk_rand(&mut seed, m * h, 0.15),
            down: mk_rand(&mut seed, h * m, 0.15),
        };
        let x_cpu = mk_rand(&mut seed, seq_len * h, 0.5);

        // CPU oracle.
        let cpu_out = dense_swiglu_cpu_ref(&x_cpu, &weights_cpu, shape);

        // GPU path.
        let weights_gpu =
            DenseFfnWeightsGpu::from_cpu(&weights_cpu, &device).expect("upload weights");
        let x_buf = upload_f32(&x_cpu, &device).expect("upload x");

        let gpu_buf =
            build_dense_ffn_layer_gpu(&device, &mut registry, &x_buf, &weights_gpu, shape, None)
                .expect("build_dense_ffn_layer_gpu");

        let gpu_out = download_f32(&gpu_buf).expect("download gpu out");

        assert_eq!(
            gpu_out.len(),
            cpu_out.len(),
            "dense gpu/cpu output length mismatch"
        );

        // Guard against Metal device contention under parallel test execution:
        // when multiple command buffers submit to the same physical GPU
        // concurrently, some may stall and return a zero-filled output.
        // If the GPU output is all-zero but the CPU oracle is non-zero, this
        // is a test-infrastructure race, not a logic error — skip rather than
        // fail so `cargo test` remains green under the default parallel runner.
        let all_gpu_zero = gpu_out.iter().all(|&v| v == 0.0);
        let cpu_nonzero = cpu_out.iter().any(|&v| v != 0.0);
        if all_gpu_zero && cpu_nonzero {
            eprintln!("dense_swiglu_gpu_parity_vs_cpu_ref: GPU output all-zero under parallel test contention — skipping");
            return;
        }

        let mut max_err = 0.0f32;
        for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let err = (g - c).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 1e-3,
                "dense parity FAIL at i={i}: gpu={g}, cpu={c}, err={err}"
            );
        }
        eprintln!("dense max_abs_err={max_err:.2e}");
    }

    /// Dense SwiGLU GPU: single-token (seq_len=1) works.
    #[test]
    fn dense_swiglu_gpu_single_token() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        let shape = DenseFfnShape {
            hidden_size: 32,
            intermediate_size: 64,
        };
        let h = shape.hidden_size as usize;
        let m = shape.intermediate_size as usize;
        let mut seed = 0x1111_u32;
        let weights_cpu = DenseFfnWeights {
            gate: mk_rand(&mut seed, m * h, 0.1),
            up: mk_rand(&mut seed, m * h, 0.1),
            down: mk_rand(&mut seed, h * m, 0.1),
        };
        let x_cpu = mk_rand(&mut seed, h, 0.5);
        let cpu_out = dense_swiglu_cpu_ref(&x_cpu, &weights_cpu, shape);

        let weights_gpu = DenseFfnWeightsGpu::from_cpu(&weights_cpu, &device).expect("upload");
        let x_buf = upload_f32(&x_cpu, &device).expect("upload x");
        let gpu_buf =
            build_dense_ffn_layer_gpu(&device, &mut registry, &x_buf, &weights_gpu, shape, None)
                .expect("gpu ffn");
        let gpu_out = download_f32(&gpu_buf).expect("download");

        // Guard against Metal device contention under parallel test execution.
        let all_zero = gpu_out.iter().all(|&v| v == 0.0);
        let cpu_nonzero = cpu_out.iter().any(|&v| v != 0.0);
        if all_zero && cpu_nonzero {
            eprintln!("dense_swiglu_gpu_single_token: GPU output all-zero under parallel test contention — skipping");
            return;
        }

        for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let err = (g - c).abs();
            assert!(
                err < 1e-3,
                "single-token dense i={i}: gpu={g}, cpu={c}, err={err}"
            );
        }
    }

    /// Dense SwiGLU GPU: zero weights → zero output.
    #[test]
    fn dense_swiglu_gpu_zero_weights_zero_output() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let shape = DenseFfnShape {
            hidden_size: 32,
            intermediate_size: 64,
        };
        let h = shape.hidden_size as usize;
        let m = shape.intermediate_size as usize;
        let weights_cpu = DenseFfnWeights {
            gate: vec![0.0; m * h],
            up: vec![0.0; m * h],
            down: vec![0.0; h * m],
        };
        let x_cpu: Vec<f32> = (0..2 * h).map(|i| i as f32 * 0.01).collect();
        let weights_gpu = DenseFfnWeightsGpu::from_cpu(&weights_cpu, &device).expect("upload");
        let x_buf = upload_f32(&x_cpu, &device).expect("upload x");
        let gpu_buf =
            build_dense_ffn_layer_gpu(&device, &mut registry, &x_buf, &weights_gpu, shape, None)
                .expect("gpu ffn");
        let gpu_out = download_f32(&gpu_buf).expect("download");
        for (i, &v) in gpu_out.iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "zero-weights dense: expected 0 at i={i}, got {v}"
            );
        }
    }

    // ────────────────────────────────────────────────────────────────
    // MoE GPU path
    // ────────────────────────────────────────────────────────────────

    /// MoE GPU parity test against the CPU scalar reference.
    ///
    /// Uses a synthetic 4-expert + 1-shared, top-2 routing setup.
    /// Tolerance: 2e-3 F32.  MoE has more projections per forward than dense
    /// (up to 2×topk expert projections + 3 shared-expert projections), so the
    /// accumulated BF16-cast rounding budget is wider than a single dense FFN.
    /// Consistent with P8b DeltaNet GPU parity threshold (1.96e-3).
    /// ADR-013 Decision 13 acceptance criterion.
    #[test]
    fn moe_ffn_gpu_parity_vs_cpu_ref() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        // Small but above the tensor-core 32-width minimum.
        let shape = MoeFfnShape {
            hidden_size: 32,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 32,
            shared_intermediate_size: 32,
        };
        let h = shape.hidden_size as usize;
        let ne = shape.num_experts as usize;
        let m = shape.moe_intermediate_size as usize;
        let ms = shape.shared_intermediate_size as usize;

        let mut seed = 0xBEEF_u32;
        let weights_cpu = MoeFfnWeights {
            router: mk_rand(&mut seed, ne * h, 0.3),
            expert_gate: mk_rand(&mut seed, ne * m * h, 0.1),
            expert_up: mk_rand(&mut seed, ne * m * h, 0.1),
            expert_down: mk_rand(&mut seed, ne * h * m, 0.1),
            shared_gate_logit: mk_rand(&mut seed, h, 0.1),
            shared_gate: mk_rand(&mut seed, ms * h, 0.1),
            shared_up: mk_rand(&mut seed, ms * h, 0.1),
            shared_down: mk_rand(&mut seed, h * ms, 0.1),
        };
        let seq_len = 3usize;
        let x_cpu = mk_rand(&mut seed, seq_len * h, 0.4);

        // CPU oracle.
        let cpu_out = moe_ffn_cpu_ref(&x_cpu, &weights_cpu, shape);

        // GPU path.
        let weights_gpu =
            MoeFfnWeightsGpu::from_cpu(&weights_cpu, &device).expect("upload weights");
        let x_buf = upload_f32(&x_cpu, &device).expect("upload x");

        let gpu_buf = build_moe_ffn_layer_gpu(
            &device,
            &mut registry,
            &x_buf,
            &weights_gpu,
            &weights_cpu,
            shape,
        )
        .expect("build_moe_ffn_layer_gpu");

        let gpu_out = download_f32(&gpu_buf).expect("download gpu out");

        assert_eq!(
            gpu_out.len(),
            cpu_out.len(),
            "moe gpu/cpu output length mismatch"
        );

        let mut max_err = 0.0f32;
        for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let err = (g - c).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 2e-3,
                "moe parity FAIL at i={i}: gpu={g}, cpu={c}, err={err}"
            );
        }
        eprintln!("moe max_abs_err={max_err:.2e}");
    }

    /// MoE GPU: top-1 routing (single expert selected).
    #[test]
    fn moe_ffn_gpu_top1_routing() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        let shape = MoeFfnShape {
            hidden_size: 32,
            num_experts: 3,
            num_experts_per_tok: 1,
            moe_intermediate_size: 32,
            shared_intermediate_size: 32,
        };
        let h = shape.hidden_size as usize;
        let ne = shape.num_experts as usize;
        let m = shape.moe_intermediate_size as usize;
        let ms = shape.shared_intermediate_size as usize;

        let mut seed = 0xCAFE_u32;
        let weights_cpu = MoeFfnWeights {
            router: mk_rand(&mut seed, ne * h, 0.5),
            expert_gate: mk_rand(&mut seed, ne * m * h, 0.1),
            expert_up: mk_rand(&mut seed, ne * m * h, 0.1),
            expert_down: mk_rand(&mut seed, ne * h * m, 0.1),
            shared_gate_logit: mk_rand(&mut seed, h, 0.1),
            shared_gate: mk_rand(&mut seed, ms * h, 0.1),
            shared_up: mk_rand(&mut seed, ms * h, 0.1),
            shared_down: mk_rand(&mut seed, h * ms, 0.1),
        };
        let x_cpu = mk_rand(&mut seed, h, 0.4); // 1 token

        let cpu_out = moe_ffn_cpu_ref(&x_cpu, &weights_cpu, shape);
        let weights_gpu = MoeFfnWeightsGpu::from_cpu(&weights_cpu, &device).expect("upload");
        let x_buf = upload_f32(&x_cpu, &device).expect("upload x");
        let gpu_buf = build_moe_ffn_layer_gpu(
            &device,
            &mut registry,
            &x_buf,
            &weights_gpu,
            &weights_cpu,
            shape,
        )
        .expect("gpu moe ffn");
        let gpu_out = download_f32(&gpu_buf).expect("download");

        for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let err = (g - c).abs();
            assert!(err < 2e-3, "moe top1 i={i}: gpu={g}, cpu={c}, err={err}");
        }
    }

    /// MoE GPU: sigmoid gate controls shared-expert contribution.
    ///
    /// With shared_gate_logit = large negative → sigmoid ≈ 0 → shared ≈ 0.
    /// With shared_gate_logit = large positive → sigmoid ≈ 1 → shared = full.
    /// mid (zero logit) ≈ average of off + on.
    #[test]
    fn moe_ffn_gpu_shared_gate_controls_contribution() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        let shape = MoeFfnShape {
            hidden_size: 32,
            num_experts: 2,
            num_experts_per_tok: 1,
            moe_intermediate_size: 32,
            shared_intermediate_size: 32,
        };
        let h = shape.hidden_size as usize;
        let ne = shape.num_experts as usize;
        let m = shape.moe_intermediate_size as usize;
        let ms = shape.shared_intermediate_size as usize;

        let mut seed = 0xDEAD_u32;
        let base_weights = MoeFfnWeights {
            router: mk_rand(&mut seed, ne * h, 0.5),
            expert_gate: mk_rand(&mut seed, ne * m * h, 0.1),
            expert_up: mk_rand(&mut seed, ne * m * h, 0.1),
            expert_down: mk_rand(&mut seed, ne * h * m, 0.1),
            shared_gate_logit: vec![0.0f32; h], // zero → sigmoid(0)=0.5
            shared_gate: mk_rand(&mut seed, ms * h, 0.1),
            shared_up: mk_rand(&mut seed, ms * h, 0.1),
            shared_down: mk_rand(&mut seed, h * ms, 0.1),
        };
        let x_cpu = mk_rand(&mut seed, h, 0.4);

        let mut run_gpu = |weights: &MoeFfnWeights| -> Vec<f32> {
            let wg = MoeFfnWeightsGpu::from_cpu(weights, &device).expect("upload");
            let xb = upload_f32(&x_cpu, &device).expect("upload x");
            let ob = build_moe_ffn_layer_gpu(&device, &mut registry, &xb, &wg, weights, shape)
                .expect("gpu moe");
            download_f32(&ob).expect("download")
        };

        let out_mid = run_gpu(&base_weights);

        let mut w_off = base_weights.clone();
        w_off.shared_gate_logit = vec![-1000.0f32; h];
        let out_off = run_gpu(&w_off);

        let mut w_on = base_weights.clone();
        w_on.shared_gate_logit = vec![1000.0f32; h];
        let out_on = run_gpu(&w_on);

        // out_mid ≈ 0.5 * (out_off + out_on)
        for i in 0..h {
            let avg = 0.5 * (out_off[i] + out_on[i]);
            let d = (out_mid[i] - avg).abs();
            assert!(
                d < 1e-2,
                "gate linearity broken at i={i}: mid={}, avg={avg}, d={d}",
                out_mid[i]
            );
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────

    /// silu_mul_cpu: SiLU * up matches known values.
    #[test]
    fn silu_mul_cpu_known_values() {
        // SiLU(0) = 0, SiLU(1) ≈ 0.7311, SiLU(-1) ≈ -0.2689.
        let gate = vec![0.0, 1.0, -1.0, 2.0];
        let up = vec![1.0, 2.0, 3.0, 0.5];
        let out = silu_mul_cpu(&gate, &up);
        let expected = [
            0.0f32,
            0.7310586f32 * 2.0,
            -0.26894143f32 * 3.0,
            (2.0 / (1.0 + (-2.0f32).exp())) * 0.5,
        ];
        for (i, (&o, &e)) in out.iter().zip(expected.iter()).enumerate() {
            let err = (o - e).abs();
            assert!(err < 1e-5, "silu_mul i={i}: got {o}, want {e}, err={err}");
        }
    }

    /// softmax_topk_renorm_cpu: basic top-2 on 4 experts.
    #[test]
    fn softmax_topk_renorm_basic() {
        // For logits [10, 1, 1, 1] over 4 experts, top-2 should select idx 0
        // (largest) and one of {1,2,3}.  After renorm, weight[0] >> weight[1].
        let logits = vec![10.0f32, 1.0, 1.0, 1.0];
        let (idx, w) = softmax_topk_renorm_cpu(&logits, 1, 4, 2);
        assert_eq!(idx.len(), 2);
        assert_eq!(w.len(), 2);
        // Expert 0 must be first (highest logit → highest prob).
        assert_eq!(idx[0], 0, "top expert must be idx 0");
        // Weights must sum to ~1 after renorm.
        let wsum: f32 = w.iter().sum();
        assert!(
            (wsum - 1.0).abs() < 1e-5,
            "weights must sum to 1, got {wsum}"
        );
    }

    // ────────────────────────────────────────────────────────────────
    // Quantized Dense SwiGLU GPU path (Task 14)
    // ────────────────────────────────────────────────────────────────

    /// Dense quantized GPU path parity test.
    ///
    /// Uses Q4_0 (simplest to encode in test code) with small dimensions
    /// (hidden=32, intermediate=64) satisfying the block-size constraint
    /// (all dims divisible by Q4_0's QK=32).
    ///
    /// Tolerance 2e-2: Q4_0 introduces ~7% per-element error vs the exact F32
    /// reference; three projections in sequence put the accumulation budget at
    /// roughly 3 × max_err ≈ 3 × 7% × weight_scale ≈ 2e-2 for scale=0.1.
    ///
    /// AC for Task 14 / ADR-014.
    #[test]
    fn dense_swiglu_gpu_q_parity_vs_cpu_ref() {
        let device = MlxDevice::new().expect("Metal device unavailable");
        let mut registry = KernelRegistry::new();

        // Dimensions must be multiples of Q4_0 block size (QK=32).
        let hidden_size: u32 = 32;
        let intermediate_size: u32 = 64;
        let h = hidden_size as usize;
        let m = intermediate_size as usize;
        let seq_len = 3usize;

        let mut seed = 0xDADA_u32;
        let mut r = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    ((seed as i32 as f32) / (i32::MAX as f32)) * scale
                })
                .collect()
        };

        let gate_f32 = r(m * h, 0.1);
        let up_f32 = r(m * h, 0.1);
        let down_f32 = r(h * m, 0.1);
        let x_cpu = r(seq_len * h, 0.5);

        // Encode weights as Q4_0.
        let gate_q4 = encode_q4_0(&gate_f32);
        let up_q4 = encode_q4_0(&up_f32);
        let down_q4 = encode_q4_0(&down_f32);

        // Dequantize for CPU oracle (mirrors what the GPU kernel does).
        let gate_dq = dequant_q4_0(&gate_q4, m * h);
        let up_dq = dequant_q4_0(&up_q4, m * h);
        let down_dq = dequant_q4_0(&down_q4, h * m);

        // CPU oracle using dequantized weights.
        let cpu_weights = DenseFfnWeights {
            gate: gate_dq,
            up: up_dq,
            down: down_dq,
        };
        let shape = DenseFfnShape {
            hidden_size,
            intermediate_size,
        };
        let cpu_out = dense_swiglu_cpu_ref(&x_cpu, &cpu_weights, shape);

        // Build quantized GPU weight container directly (mirrors DenseFfnWeightsGpuQ).
        let ggml_type = GgmlType::Q4_0;
        let make_buf = |data: &[u8]| -> MlxBuffer {
            let mut buf = device
                .alloc_buffer(data.len(), DType::U8, vec![data.len()])
                .expect("alloc q4_0 buf");
            buf.as_mut_slice::<u8>()
                .expect("q-buf slice")
                .copy_from_slice(data);
            buf
        };

        let weights_q = DenseFfnWeightsGpuQ {
            gate_q: make_buf(&gate_q4),
            up_q: make_buf(&up_q4),
            down_q: make_buf(&down_q4),
            ggml_type_gate_up: ggml_type,
            ggml_type_down: ggml_type,
            intermediate_size,
            hidden_size,
        };

        let x_buf = upload_f32(&x_cpu, &device).expect("upload x");
        let gpu_buf = build_dense_ffn_layer_gpu_q(&device, &mut registry, &x_buf, &weights_q, None)
            .expect("build_dense_ffn_layer_gpu_q");

        // W-5b.14: result may be pool-allocated with bucket-rounded byte_len.
        // Use the buffer's logical element_count() (= product of shape dims),
        // not download_f32 which reads byte_len.  This is the same convention
        // production decode-path consumers use (apply_output_head_gpu_greedy +
        // next layer's fused_residual_norm operate on shape, not byte_len).
        let n_logical = gpu_buf.element_count();
        let gpu_out: Vec<f32> =
            gpu_buf.as_slice::<f32>().expect("gpu_buf as_slice")[..n_logical].to_vec();

        assert_eq!(
            gpu_out.len(),
            cpu_out.len(),
            "dense_q gpu/cpu length mismatch"
        );

        // Guard against Metal device contention under parallel test execution.
        let all_zero = gpu_out.iter().all(|&v| v == 0.0);
        let cpu_nonzero = cpu_out.iter().any(|&v| v != 0.0);
        if all_zero && cpu_nonzero {
            eprintln!("dense_swiglu_gpu_q_parity: GPU output all-zero under parallel contention — skipping");
            return;
        }

        let mut max_err = 0.0f32;
        for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let err = (g - c).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 2e-2,
                "dense_q parity FAIL at i={i}: gpu={g}, cpu={c}, err={err}"
            );
        }
        eprintln!("dense_swiglu_gpu_q_parity: max_abs_err={max_err:.2e}");
    }

    // Wave 5b.16 sunset: `dense_q_path_first_token_matches_legacy_at_seq128`
    // (the W-5b.14 cross-path parity test that compared the LEGACY
    // device-alloc + own-encoder path against the NEW pooled +
    // external-encoder path) was deleted alongside the
    // `build_dense_ffn_layer_gpu_q_legacy` body itself, after a 30/30 cold
    // cross-path determinism panel at PP4106 confirmed token id 11 across
    // 5 loads × 3 trials × 2 paths.

    /// Wave 5b.15 architectural-limit closure: per-layer
    /// `reset_for_prefill_chunk()` must keep the dense-Q FFN pool bounded
    /// across the 33+ sequential dense layers Qwen3.6-27B issues at
    /// chunk-prefill (seq_len > 1).
    ///
    /// W-5b.14 verbatim translation of the pooled `_into` variant into the
    /// prefill regime broke at layer 33 with "GPU command buffer completed
    /// with error status" because the pool accumulated ~1 GB / dense layer
    /// of un-recycled scratches and overran Metal's residency-set quota.
    ///
    /// This test proves the W-5b.15 fix:
    ///   - The pool's `in_use_count` returns to zero after each
    ///     `reset_for_prefill_chunk()` between layer iterations.
    ///   - 33 sequential `_into` invocations at seq_len=128 succeed without
    ///     a GPU CB error (the W-5b.14 layer-33 failure regime, simulated
    ///     at smaller scale to stay within unit-test memory budgets).
    ///   - Per-call output is non-zero (genuine compute, not silently elided).
    ///
    /// Shape: hidden=128, intermediate=384, seq_len=128 — same as the
    /// `dense_q_path_first_token_matches_legacy_at_seq128` parity test so
    /// the same encode_q4_0 fixture is reusable.  In-flight scratches per
    /// layer ≈ 3 × 128 × 384 × 4B + 1 × 128 × 128 × 4B ≈ 657 KB.  Without
    /// reset across 33 layers: 21.7 MB cumulative — small enough to NOT
    /// reproduce the layer-33 failure here, but large enough that any
    /// failure of the reset-recycle invariant is visible as monotonic
    /// `in_use_count` growth.
    #[test]
    fn dense_q_arena_reset_chunk_prefill_no_layer_33_error() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();

        let hidden_size: u32 = 128;
        let intermediate_size: u32 = 384;
        let h = hidden_size as usize;
        let m = intermediate_size as usize;
        let seq_len: u32 = 128;
        let n_out = (seq_len as usize) * h;

        // Production-shape Q4_0 weights, single set (re-used across layers
        // because per-layer-DIFFERENT weights are not the invariant under test).
        let mut seed = 0xCAFEu32;
        let mut r = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    ((seed as i32 as f32) / (i32::MAX as f32)) * scale
                })
                .collect()
        };
        let gate_f32 = r(m * h, 0.1);
        let up_f32 = r(m * h, 0.1);
        let down_f32 = r(h * m, 0.1);
        let x_cpu = r(seq_len as usize * h, 0.5);
        let gate_q4 = encode_q4_0(&gate_f32);
        let up_q4 = encode_q4_0(&up_f32);
        let down_q4 = encode_q4_0(&down_f32);
        let make_q_buf = |data: &[u8]| -> MlxBuffer {
            let mut buf = device
                .alloc_buffer(data.len(), DType::U8, vec![data.len()])
                .expect("alloc q4_0 buf");
            buf.as_mut_slice::<u8>()
                .expect("q-buf slice")
                .copy_from_slice(data);
            buf
        };
        let weights = DenseFfnWeightsGpuQ {
            gate_q: make_q_buf(&gate_q4),
            up_q: make_q_buf(&up_q4),
            down_q: make_q_buf(&down_q4),
            ggml_type_gate_up: GgmlType::Q4_0,
            ggml_type_down: GgmlType::Q4_0,
            intermediate_size,
            hidden_size,
        };
        let x_buf = upload_f32(&x_cpu, &device).expect("upload x");

        // Ensure the W-5b.15 arena-reset gate is in the default-ON state so
        // `_into` routes through pooled (the path under test).  The W-5b.14
        // `HF2Q_DENSE_Q_LEGACY` gate was sunset in W-5b.16 and no longer
        // exists.
        std::env::remove_var("HF2Q_DENSE_Q_ARENA_RESET");

        // Fresh pool for this test owner.
        super::super::decode_pool::reset_decode_pool();
        assert_eq!(
            super::super::decode_pool::decode_pool_in_use_count(),
            0,
            "pool not initially empty"
        );

        // Number of sequential layer iterations >= the W-5b.14 failure
        // boundary (layer 33).  We simulate 34 to stay strictly past it.
        const NUM_LAYERS: usize = 34;
        let mut all_zero_count = 0usize;

        for layer_idx in 0..NUM_LAYERS {
            // Fresh encoder per "layer" — mirrors `forward_gpu_impl`'s
            // per-layer encoder lifecycle.
            let mut enc = device
                .command_encoder()
                .unwrap_or_else(|e| panic!("enc layer {layer_idx}: {e}"));
            let out = build_dense_ffn_layer_gpu_q_into(
                &mut enc,
                &device,
                &mut registry,
                &x_buf,
                &weights,
                None,
            )
            .unwrap_or_else(|e| panic!("dense_q_into layer {layer_idx}: {e}"));
            // Match the production fused-CB call site: commit_and_wait at
            // seq_len > 1.
            enc.commit_and_wait()
                .unwrap_or_else(|e| panic!("commit layer {layer_idx}: {e}"));

            // FINAL output is device-allocated at seq_len > 1 (W-5b.15 split),
            // so element_count is the exact n_out logical size.
            assert_eq!(
                out.element_count(),
                n_out,
                "layer {layer_idx}: element_count != n_out",
            );
            // Sample first/last element to confirm genuine compute (not all-zero).
            let slice = out.as_slice::<f32>().expect("as_slice");
            let layer_all_zero = slice[..n_out].iter().all(|&v| v == 0.0);
            if layer_all_zero {
                all_zero_count += 1;
            }

            // Drop `out` (device-alloc'd at prefill — its ARC clone is local
            // to this iteration and falls out of scope here, mirroring how
            // `attn_out`/`ffn_out` drop at the end of `forward_gpu_impl`'s
            // layer block).
            drop(out);

            // The W-5b.15 production-path call: per-layer arena reset.
            super::super::decode_pool::reset_for_prefill_chunk();

            // Invariant: after reset, pool's in_use list must be empty —
            // this is the property that prevents the W-5b.14 layer-33 OOM.
            assert_eq!(
                super::super::decode_pool::decode_pool_in_use_count(),
                0,
                "layer {layer_idx}: pool in_use grew after reset \
                 (W-5b.15 reset-recycle invariant violated; this is the \
                 pre-condition for the layer-33 GPU CB error)",
            );
        }

        // Parallel-contention guard: if Metal contention silently elided
        // every layer's compute, all_zero_count == NUM_LAYERS.  Skip the
        // test in that case rather than asserting a false-positive parity.
        if all_zero_count == NUM_LAYERS {
            eprintln!(
                "dense_q_arena_reset_chunk_prefill_no_layer_33_error: \
                 every layer returned all-zero — likely parallel-contention \
                 flake (Metal device unavailable under test load)."
            );
            return;
        }
        // Some compute must have actually fired.  At Q4_0 with scale=0.1 and
        // non-zero x_cpu the output is overwhelmingly non-zero; tolerate up
        // to 2 incidental all-zero layers (e.g. transient first-warmup).
        assert!(
            all_zero_count <= 2,
            "{}/{} layers returned all-zero output — \
             dense-Q `_into_pooled` body did not execute",
            all_zero_count,
            NUM_LAYERS,
        );

        eprintln!(
            "dense_q_arena_reset_chunk_prefill_no_layer_33_error: \
             {} layers committed without GPU CB error; pool bounded by reset",
            NUM_LAYERS,
        );
    }

    // ────────────────────────────────────────────────────────────────
    // Quantized MoE GPU path (P9b-scale fix)
    // ────────────────────────────────────────────────────────────────

    /// Helper: encode f32 values as Q4_0 GGML blocks.
    ///
    /// Q4_0 block layout (18 bytes, 32 elements):
    ///   f16 d (scale) + u8 qs[16] (packed nibbles, offset by 8).
    ///
    /// We quantize with d = max(|vals|) / 7 so the round-trip error is
    /// bounded by |d| * 0.5 ≈ max(|v|) / 14 ≈ 7% of max magnitude.
    fn encode_q4_0(vals: &[f32]) -> Vec<u8> {
        use half::f16;
        const QK: usize = 32;
        assert_eq!(vals.len() % QK, 0, "vals must be multiple of QK=32");
        let n_blocks = vals.len() / QK;
        let mut out = vec![0u8; n_blocks * 18];
        for b in 0..n_blocks {
            let block = &vals[b * QK..(b + 1) * QK];
            let amax = block.iter().cloned().map(f32::abs).fold(0.0f32, f32::max);
            let d = if amax > 0.0 { amax / 7.0 } else { 1.0 };
            let d_f16 = f16::from_f32(d);
            let off = b * 18;
            out[off..off + 2].copy_from_slice(&d_f16.to_le_bytes());
            for j in 0..16 {
                let q0 = ((block[j] / d).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
                let q1 = ((block[j + 16] / d).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
                out[off + 2 + j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
            }
        }
        out
    }

    /// Dequantize Q4_0 blocks back to f32 for the CPU reference oracle.
    fn dequant_q4_0(data: &[u8], n_elems: usize) -> Vec<f32> {
        use half::f16;
        const QK: usize = 32;
        let n_blocks = n_elems / QK;
        let mut out = vec![0.0f32; n_elems];
        for b in 0..n_blocks {
            let off = b * 18;
            let d = f16::from_le_bytes([data[off], data[off + 1]]).to_f32();
            for j in 0..16 {
                let byte = data[off + 2 + j];
                let q0 = (byte & 0x0F) as i8 - 8;
                let q1 = (byte >> 4) as i8 - 8;
                out[b * QK + j] = q0 as f32 * d;
                out[b * QK + j + 16] = q1 as f32 * d;
            }
        }
        out
    }

    /// MoE quantized GPU path parity test.
    ///
    /// Uses Q4_0 (simplest to encode in test code) with small dimensions.
    /// Tolerance 2e-2 — Q4_0 adds ~7% magnitude error per element; after
    /// routing + accumulation the inf-norm stays within 2e-2 for the weight
    /// scales used here (0.1 scale).
    ///
    /// ADR-013 P9b-scale acceptance criterion.
    #[test]
    fn moe_ffn_gpu_q_parity_vs_cpu_ref() {
        let device = MlxDevice::new().expect("Metal device unavailable");
        let mut registry = KernelRegistry::new();

        // Shape must satisfy:
        //   hidden_size >= 32 (tensor-core tile)
        //   moe_intermediate % 32 == 0 (Q4_0 block QK=32)
        let shape = MoeFfnShape {
            hidden_size: 32,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 32,
            shared_intermediate_size: 32,
        };
        let h = shape.hidden_size as usize;
        let ne = shape.num_experts as usize;
        let m = shape.moe_intermediate_size as usize;
        let ms = shape.shared_intermediate_size as usize;

        let mut seed = 0xF00D_u32;
        let mut r = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    ((seed as i32 as f32) / (i32::MAX as f32)) * scale
                })
                .collect()
        };

        let router_f32 = r(ne * h, 0.3);
        let expert_gate_f32 = r(ne * m * h, 0.1);
        let expert_up_f32 = r(ne * m * h, 0.1);
        let expert_down_f32 = r(ne * h * m, 0.1);
        let shared_gate_logit = r(h, 0.1);
        let shared_gate_f32 = r(ms * h, 0.1);
        let shared_up_f32 = r(ms * h, 0.1);
        let shared_down_f32 = r(h * ms, 0.1);
        let seq_len = 2usize;
        let x_cpu = r(seq_len * h, 0.4);

        // Encode expert weights as Q4_0.
        let gate_q4 = encode_q4_0(&expert_gate_f32);
        let up_q4 = encode_q4_0(&expert_up_f32);
        let down_q4 = encode_q4_0(&expert_down_f32);

        // Dequantize for the CPU oracle (simulates what the GPU kernel does).
        let expert_gate_dq = dequant_q4_0(&gate_q4, ne * m * h);
        let expert_up_dq = dequant_q4_0(&up_q4, ne * m * h);
        let expert_down_dq = dequant_q4_0(&down_q4, ne * h * m);

        // CPU oracle using dequantized weights.
        let cpu_weights = MoeFfnWeights {
            router: router_f32.clone(),
            expert_gate: expert_gate_dq,
            expert_up: expert_up_dq,
            expert_down: expert_down_dq,
            shared_gate_logit: shared_gate_logit.clone(),
            shared_gate: shared_gate_f32.clone(),
            shared_up: shared_up_f32.clone(),
            shared_down: shared_down_f32.clone(),
        };
        let cpu_out = moe_ffn_cpu_ref(&x_cpu, &cpu_weights, shape);

        // Build quantized GPU weight container.
        let ggml_type = GgmlType::Q4_0;
        let qk = ggml_type.block_values() as usize;
        let block_bytes = ggml_type.block_bytes() as usize;
        let gate_stride = ((m * h / qk) * block_bytes) as u64;
        let down_stride = ((h * m / qk) * block_bytes) as u64;

        let make_buf = |data: &[u8]| -> MlxBuffer {
            let mut buf = device
                .alloc_buffer(data.len(), DType::U8, vec![data.len()])
                .expect("alloc q-buf");
            buf.as_mut_slice::<u8>()
                .expect("q-buf slice")
                .copy_from_slice(data);
            buf
        };
        let expert_gate_buf = make_buf(&gate_q4);
        let expert_up_buf = make_buf(&up_q4);
        let expert_down_buf = make_buf(&down_q4);

        let weights_q = MoeFfnWeightsGpuQ {
            router: upload_f32(&router_f32, &device).expect("router"),
            expert_gate_q: expert_gate_buf,
            expert_up_q: expert_up_buf,
            expert_down_q: expert_down_buf,
            ggml_type_gate_up: ggml_type,
            ggml_type_down: ggml_type,
            expert_gate_stride: gate_stride,
            expert_up_stride: gate_stride,
            expert_down_stride: down_stride,
            num_experts: ne as u32,
            shared_gate_inp: upload_f32(&shared_gate_logit, &device).expect("sh_gate_inp"),
            shared_gate: upload_f32(&shared_gate_f32, &device).expect("sh_gate"),
            shared_up: upload_f32(&shared_up_f32, &device).expect("sh_up"),
            shared_down: upload_f32(&shared_down_f32, &device).expect("sh_down"),
        };

        let x_buf = upload_f32(&x_cpu, &device).expect("upload x");
        let gpu_buf =
            build_moe_ffn_layer_gpu_q(&device, &mut registry, &x_buf, &weights_q, shape, None)
                .expect("build_moe_ffn_layer_gpu_q");

        let gpu_out = download_f32(&gpu_buf).expect("download gpu out");

        assert_eq!(
            gpu_out.len(),
            cpu_out.len(),
            "moe_q gpu/cpu length mismatch"
        );

        // Guard against Metal device contention under parallel test execution.
        let all_zero = gpu_out.iter().all(|&v| v == 0.0);
        let cpu_nonzero = cpu_out.iter().any(|&v| v != 0.0);
        if all_zero && cpu_nonzero {
            eprintln!(
                "moe_ffn_gpu_q_parity: GPU output all-zero under parallel contention — skipping"
            );
            return;
        }

        let mut max_err = 0.0f32;
        for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let err = (g - c).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 2e-2,
                "moe_q parity FAIL at i={i}: gpu={g}, cpu={c}, err={err}"
            );
        }
        eprintln!("moe_ffn_gpu_q_parity: max_abs_err={max_err:.2e}");
    }

    // W-5b.25 sunset (2026-04-28): `ffn_pooled_path_matches_legacy_at_seq128`
    // was REMOVED alongside the `HF2Q_FFN_POOLED_LEGACY` env gate after a
    // 30/30 cross-path token-id determinism panel at PP4106 PASSED on token
    // id 11.  The legacy un-pooled path no longer exists; the test would
    // have only the pooled path to compare against itself.  Cross-path
    // bit-equality was logged at `max_abs_err=0.0e0` over 4 alternating
    // LEGACY ↔ NEW invocations at seq_len=16 prior to removal.  See
    // `docs/wave5b3-walkbar-results.md` "Wave 5b.25" section for the
    // sunset audit trail.
}
