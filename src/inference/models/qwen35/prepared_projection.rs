//! Prepared Q4_K / Q6_K planes for the attention and DeltaNet projections of
//! a width-32 Qwen joint target step (`HF2Q_REPACKED_GGUF_M32_PROJ=1`, on top
//! of `HF2Q_REPACKED_GGUF_M32=1`).
//!
//! The stock 32-row GGML matmul has a latency floor of about 0.1 ms per
//! serialized dispatch whatever the output width; the prepared BF16-input
//! kernel does not. Each projection's planes are prepared once, on first use,
//! and owned by the layer's weights. Projections that read the same
//! activation share one BF16 copy of it.

use std::sync::OnceLock;

use anyhow::{anyhow, Context, Result};
use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::quantized_matmul_ggml::{GgmlQuantizedMatmulParams, GgmlType};
use mlx_native::ops::repacked_gguf_mm::{
    dispatch_repacked_gguf_mm_bf16_input, prepare_repacked_gguf_weight, PreparedGgufWeight,
};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::decode_pool::pooled_alloc_buffer;

/// Load-once prepared planes of one projection weight.
pub(crate) type Planes = OnceLock<PreparedGgufWeight>;

/// Prepared planes of a DeltaNet layer's large projections.
#[derive(Default)]
pub(crate) struct DeltaNetPlanes {
    pub(crate) qkv: Planes,
    pub(crate) gate: Planes,
    pub(crate) out: Planes,
}

/// Prepared planes of a full-attention layer's projections.
#[derive(Default)]
pub(crate) struct FullAttnPlanes {
    pub(crate) q_gate: Planes,
    pub(crate) k: Planes,
    pub(crate) v: Planes,
    pub(crate) out: Planes,
}

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HF2Q_REPACKED_GGUF_M32_PROJ")
            .map(|value| value == "1")
            .unwrap_or(false)
    })
}

fn params(
    rows: u32,
    ggml_type: GgmlType,
    in_features: u32,
    out_features: u32,
) -> GgmlQuantizedMatmulParams {
    GgmlQuantizedMatmulParams {
        m: rows,
        n: out_features,
        k: in_features,
        ggml_type,
    }
}

/// Whether this product takes the prepared route: the switch is on, the
/// weight is a raw GGML block matrix, and the dense prepared route admits its
/// shape inside the open width-32 joint scope.
pub(crate) fn admits(
    rows: u32,
    weight: &MlxBuffer,
    ggml_type: GgmlType,
    in_features: u32,
    out_features: u32,
) -> bool {
    enabled()
        && weight.dtype() == DType::U8
        && super::gpu_ffn::repacked_dense_m32_admits(&params(
            rows,
            ggml_type,
            in_features,
            out_features,
        ))
}

/// Encode a BF16 copy of the F32 `[rows, in_features]` activation. The caller
/// places the barrier before the projections that read it.
pub(crate) fn bf16_input(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    rows: u32,
    in_features: u32,
) -> Result<MlxBuffer> {
    let elements = rows as usize * in_features as usize;
    let output = pooled_alloc_buffer(
        device,
        elements * std::mem::size_of::<half::bf16>(),
        DType::BF16,
        vec![rows as usize, in_features as usize],
    )
    .map_err(|error| anyhow!("allocate prepared projection input: {error}"))?;
    cast(
        enc,
        registry,
        device.metal_device(),
        input,
        &output,
        elements,
        CastDirection::F32ToBF16,
    )
    .context("cast prepared projection input F32→BF16")?;
    Ok(output)
}

/// Encode one prepared projection of `input_bf16` into a pooled F32
/// `[rows, out_features]` buffer, preparing the weight's planes on first use.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input_bf16: &MlxBuffer,
    weight: &MlxBuffer,
    ggml_type: GgmlType,
    planes: &Planes,
    rows: u32,
    in_features: u32,
    out_features: u32,
) -> Result<MlxBuffer> {
    let params = params(rows, ggml_type, in_features, out_features);
    if planes.get().is_none() {
        let prepared = prepare_repacked_gguf_weight(enc, registry, device, weight, &params)
            .context("prepare projection GGUF weight")?;
        let _ = planes.set(prepared);
        // The projection below reads the planes this repack writes.
        enc.memory_barrier();
    }
    let prepared = planes
        .get()
        .context("prepared projection GGUF weight was not retained")?;
    let output = pooled_alloc_buffer(
        device,
        rows as usize * out_features as usize * std::mem::size_of::<f32>(),
        DType::F32,
        vec![rows as usize, out_features as usize],
    )
    .map_err(|error| anyhow!("allocate prepared projection output: {error}"))?;
    dispatch_repacked_gguf_mm_bf16_input(
        enc, registry, device, input_bf16, prepared, &output, &params,
    )
    .context("dispatch prepared GGUF projection")?;
    Ok(output)
}
