//! Prepared Q4_K / Q6_K planes for the attention and DeltaNet projections of
//! a width-32 Qwen joint target step (`HF2Q_REPACKED_GGUF_M32_PROJ=1`, on top
//! of `HF2Q_REPACKED_GGUF_M32=1`).
//!
//! The stock 32-row GGML matmul has a latency floor of about 0.1 ms per
//! serialized dispatch whatever the output width; the prepared BF16-input
//! kernel does not. Each projection's planes are prepared once, on first use,
//! and owned by the layer's weights. Projections that read the same
//! activation share one BF16 copy of it.

use std::sync::{LazyLock, OnceLock};

use anyhow::{anyhow, Context, Result};
use mlx_native::ops::copy::{dispatch_strided_copy_f32, StridedCopyParams};
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
    /// The alpha and beta logit projections as one weight: alpha's rows,
    /// beta's rows, then zero rows up to a multiple of 64 outputs.
    pub(crate) alpha_beta_weight: OnceLock<MlxBuffer>,
    pub(crate) alpha_beta: Planes,
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

fn head_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HF2Q_REPACKED_GGUF_M32_HEAD")
            .map(|value| value == "1")
            .unwrap_or(false)
    })
}

fn shared_head_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("HF2Q_REPACKED_GGUF_SHARED_HEAD").as_deref() == Ok("1")
    });
    *ENABLED
}

/// The model's output head, prepared once for every prepared consumer
/// (`HF2Q_REPACKED_GGUF_SHARED_HEAD=1`): the joint target step's head and the
/// staged DFlash2 draft's vocabulary projection read the same planes instead
/// of preparing the head once each.
///
/// The planes are written by a command buffer of their own that completes
/// before the head is handed out, so no consumer can hold planes that an
/// abandoned command never wrote.
#[derive(Default)]
pub(crate) struct SharedHead {
    /// The source blocks' address and the planes prepared from them.
    prepared: OnceLock<(usize, PreparedGgufWeight)>,
}

impl SharedHead {
    /// Whether the switch is on and the prepared kernels take this head.
    pub(crate) fn serves(
        weight: &MlxBuffer,
        ggml_type: GgmlType,
        in_features: u32,
        out_features: u32,
    ) -> bool {
        shared_head_enabled()
            && weight.dtype() == DType::U8
            && out_features % 64 == 0
            && in_features % 256 == 0
            && matches!(ggml_type, GgmlType::Q4_K | GgmlType::Q6_K)
    }

    /// The planes of `weight`, preparing them on first use. Fails if the
    /// model's head was replaced after they were prepared.
    pub(crate) fn get_or_prepare(
        &self,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        weight: &MlxBuffer,
        ggml_type: GgmlType,
        in_features: u32,
        out_features: u32,
    ) -> Result<&PreparedGgufWeight> {
        let source = weight.contents_ptr() as usize + weight.byte_offset() as usize;
        if self.prepared.get().is_none() {
            let mut enc = device
                .command_encoder()
                .map_err(|error| anyhow!("shared output head encoder: {error}"))?;
            let prepared = prepare_repacked_gguf_weight(
                &mut enc,
                registry,
                device,
                weight,
                &params(1, ggml_type, in_features, out_features),
            )
            .context("prepare shared output head")?;
            enc.commit_and_wait_labeled("qwen35.shared_output_head.prepare")
                .map_err(|error| anyhow!("prepare shared output head: {error}"))?;
            let _ = self.prepared.set((source, prepared));
        }
        let (prepared_source, prepared) = self
            .prepared
            .get()
            .context("shared output head was not retained")?;
        anyhow::ensure!(
            *prepared_source == source,
            "the output head was replaced after its shared planes were prepared"
        );
        Ok(prepared)
    }
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

/// Whether the output head of a `rows`-wide joint step takes the prepared
/// route. The head runs after the layer chain's scope has closed, so the
/// caller vouches for the joint width.
pub(crate) fn admits_head(
    rows: u32,
    weight: &MlxBuffer,
    ggml_type: GgmlType,
    in_features: u32,
    out_features: u32,
) -> bool {
    head_enabled()
        && weight.dtype() == DType::U8
        && super::gpu_ffn::repacked_dense_m32_shape_admits(&params(
            rows,
            ggml_type,
            in_features,
            out_features,
        ))
}

/// Output rows of the fused alpha/beta weight for `heads` rows each.
pub(crate) fn alpha_beta_rows(heads: u32) -> u32 {
    (2 * heads).div_ceil(64) * 64
}

/// Whether the alpha and beta logit projections run as one prepared product.
pub(crate) fn admits_alpha_beta(
    rows: u32,
    alpha: &MlxBuffer,
    alpha_type: GgmlType,
    beta: &MlxBuffer,
    beta_type: GgmlType,
    in_features: u32,
    heads: u32,
) -> bool {
    alpha_type == beta_type
        && beta.dtype() == DType::U8
        && admits(rows, alpha, alpha_type, in_features, alpha_beta_rows(heads))
}

/// Encode the fused alpha/beta projection of `input_bf16` and return its
/// `[rows, alpha_beta_rows(heads)]` output. The caller splits it with
/// [`split_alpha_beta`] after a barrier.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_alpha_beta(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input_bf16: &MlxBuffer,
    alpha: &MlxBuffer,
    beta: &MlxBuffer,
    ggml_type: GgmlType,
    planes: &DeltaNetPlanes,
    rows: u32,
    in_features: u32,
    heads: u32,
) -> Result<MlxBuffer> {
    let fused_rows = alpha_beta_rows(heads);
    if planes.alpha_beta_weight.get().is_none() {
        let row_bytes =
            (in_features / ggml_type.block_values()) as usize * ggml_type.block_bytes() as usize;
        let each = heads as usize * row_bytes;
        let total = fused_rows as usize * row_bytes;
        let mut fused = device
            .alloc_buffer(total, DType::U8, vec![total])
            .map_err(|error| anyhow!("allocate fused alpha/beta weight: {error}"))?;
        {
            let alpha_bytes = alpha
                .as_slice::<u8>()
                .map_err(|error| anyhow!("read alpha projection blocks: {error}"))?;
            let beta_bytes = beta
                .as_slice::<u8>()
                .map_err(|error| anyhow!("read beta projection blocks: {error}"))?;
            anyhow::ensure!(
                alpha_bytes.len() >= each && beta_bytes.len() >= each,
                "alpha/beta projection holds {} and {} bytes, {each} needed",
                alpha_bytes.len(),
                beta_bytes.len()
            );
            let bytes = fused
                .as_mut_slice::<u8>()
                .map_err(|error| anyhow!("write fused alpha/beta weight: {error}"))?;
            // A zero GGML block dequantizes to zeros, so the padding rows
            // produce zero outputs that nothing reads.
            bytes.fill(0);
            bytes[..each].copy_from_slice(&alpha_bytes[..each]);
            bytes[each..2 * each].copy_from_slice(&beta_bytes[..each]);
        }
        let _ = planes.alpha_beta_weight.set(fused);
    }
    let weight = planes
        .alpha_beta_weight
        .get()
        .context("fused alpha/beta weight was not retained")?;
    project(
        enc,
        registry,
        device,
        input_bf16,
        weight,
        ggml_type,
        &planes.alpha_beta,
        rows,
        in_features,
        fused_rows,
    )
}

/// Encode the split of a fused alpha/beta output into contiguous
/// `[rows, heads]` alpha and beta logits.
pub(crate) fn split_alpha_beta(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    fused: &MlxBuffer,
    rows: u32,
    heads: u32,
) -> Result<(MlxBuffer, MlxBuffer)> {
    let fused_rows = alpha_beta_rows(heads);
    let params = StridedCopyParams {
        rows,
        cols: heads,
        stride_row: fused_rows,
        stride_col: 1,
    };
    let output = |label: &str| {
        pooled_alloc_buffer(
            device,
            rows as usize * heads as usize * std::mem::size_of::<f32>(),
            DType::F32,
            vec![rows as usize, heads as usize],
        )
        .map_err(|error| anyhow!("allocate {label} logits: {error}"))
    };
    let alpha = output("alpha")?;
    let beta = output("beta")?;
    dispatch_strided_copy_f32(enc, registry, device.metal_device(), fused, &alpha, &params)
        .context("split alpha logits")?;
    let beta_source = fused.slice_view(
        u64::from(heads) * std::mem::size_of::<f32>() as u64,
        (rows as usize * fused_rows as usize) - heads as usize,
    );
    dispatch_strided_copy_f32(
        enc,
        registry,
        device.metal_device(),
        &beta_source,
        &beta,
        &params,
    )
    .context("split beta logits")?;
    Ok((alpha, beta))
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
    let output = pooled_alloc_buffer(
        device,
        rows as usize * out_features as usize * std::mem::size_of::<f32>(),
        DType::F32,
        vec![rows as usize, out_features as usize],
    )
    .map_err(|error| anyhow!("allocate prepared projection output: {error}"))?;
    project_into(
        enc,
        registry,
        device,
        input_bf16,
        weight,
        ggml_type,
        planes,
        &output,
        rows,
        in_features,
        out_features,
    )?;
    Ok(output)
}

/// Encode the projection of `input_bf16` through already prepared planes into
/// a pooled F32 `[rows, out_features]` buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_prepared(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input_bf16: &MlxBuffer,
    prepared: &PreparedGgufWeight,
    ggml_type: GgmlType,
    rows: u32,
    in_features: u32,
    out_features: u32,
) -> Result<MlxBuffer> {
    let output = pooled_alloc_buffer(
        device,
        rows as usize * out_features as usize * std::mem::size_of::<f32>(),
        DType::F32,
        vec![rows as usize, out_features as usize],
    )
    .map_err(|error| anyhow!("allocate prepared projection output: {error}"))?;
    dispatch_repacked_gguf_mm_bf16_input(
        enc,
        registry,
        device,
        input_bf16,
        prepared,
        &output,
        &params(rows, ggml_type, in_features, out_features),
    )
    .context("dispatch prepared GGUF projection")?;
    Ok(output)
}

/// Encode one prepared projection of `input_bf16` into the caller's F32
/// `[rows, out_features]` buffer, preparing the weight's planes on first use.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_into(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input_bf16: &MlxBuffer,
    weight: &MlxBuffer,
    ggml_type: GgmlType,
    planes: &Planes,
    output: &MlxBuffer,
    rows: u32,
    in_features: u32,
    out_features: u32,
) -> Result<()> {
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
    dispatch_repacked_gguf_mm_bf16_input(
        enc, registry, device, input_bf16, prepared, output, &params,
    )
    .context("dispatch prepared GGUF projection")
}
