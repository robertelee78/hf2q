//! Prepared Q4_K / Q6_K planes for the attention and DeltaNet projections of
//! a Qwen joint target step (`HF2Q_REPACKED_GGUF_M32_PROJ=1`, on top of
//! `HF2Q_REPACKED_GGUF_M32=1`): a 32-row step, or a DFlash2 target verify
//! step of any width under `HF2Q_REPACKED_GGUF_ANY_WIDTH=1`.
//!
//! The stock 32-row GGML matmul has a latency floor of about 0.1 ms per
//! serialized dispatch whatever the output width; the prepared BF16-input
//! kernel does not. Each projection's planes are prepared once, on first use,
//! and owned by the layer's weights. Projections that read the same
//! activation share one BF16 copy of it.

use std::sync::OnceLock;

use anyhow::{anyhow, Context, Result};
use mlx_native::ops::copy::{dispatch_strided_copy_f32, StridedCopyParams};
use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::quantized_matmul_ggml::{GgmlQuantizedMatmulParams, GgmlType};
use mlx_native::ops::repacked_gguf_mm::{
    dispatch_repacked_gguf_mm_bf16_input, dispatch_repacked_gguf_mm_bf16_input_one_pass,
    prepare_repacked_gguf_weight, PreparedGgufWeight,
};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::decode_pool::pooled_alloc_buffer;
use crate::inference::spec_decode::dflash2::route_profile;

/// Load-once prepared planes of one projection weight.
pub(crate) type Planes = OnceLock<PreparedGgufWeight>;

/// How a prepared product is issued. Both give the same bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Pass {
    /// The weight's own schedule: with several K splits a split-grid pass, a
    /// memory barrier and a reduce pass.
    Scheduled,
    /// One kernel pass and no barrier
    /// (`dispatch_repacked_gguf_mm_bf16_input_one_pass`): for the products of
    /// one activation that are issued together and fill the GPU jointly.
    One,
}

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
    #[cfg(test)]
    if let Some(switches) = super::prepared_switch_override::current() {
        return switches.projections;
    }
    const SWITCH: route_profile::Switch = route_profile::switch("HF2Q_REPACKED_GGUF_M32_PROJ");
    route_profile::is_one(SWITCH)
}

fn head_enabled() -> bool {
    #[cfg(test)]
    if let Some(switches) = super::prepared_switch_override::current() {
        return switches.head;
    }
    const SWITCH: route_profile::Switch = route_profile::switch("HF2Q_REPACKED_GGUF_M32_HEAD");
    route_profile::is_one(SWITCH)
}

fn shared_head_enabled() -> bool {
    #[cfg(test)]
    if let Some(switches) = super::prepared_switch_override::current() {
        return switches.shared_head;
    }
    const SWITCH: route_profile::Switch =
        route_profile::switch("HF2Q_REPACKED_GGUF_SHARED_HEAD");
    route_profile::is_one(SWITCH)
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

    /// Whether the planes have been prepared.
    #[cfg(test)]
    pub(crate) fn is_prepared(&self) -> bool {
        self.prepared.get().is_some()
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

/// One weight's planes cell and what prepares it.
pub(crate) struct PlanesSource<'a> {
    pub(crate) planes: &'a Planes,
    /// Raw GGML blocks `[params.n, params.k]`; never the output of a
    /// dispatch that is still being encoded.
    pub(crate) weight: &'a MlxBuffer,
    pub(crate) params: GgmlQuantizedMatmulParams,
}

/// Prepare the planes of every source whose cell is empty and publish them.
///
/// The repacks run in a command buffer of their own, and a cell is set only
/// after that command buffer has completed. A cell therefore never holds
/// planes that were encoded but not written: if the caller's encoder is
/// dropped on a later error, or its command buffer fails, the planes it read
/// are still valid for the life of the model; if the repack itself fails,
/// the cells stay empty and the next use prepares again. (Published at
/// encode time, the planes of an abandoned encoder stayed cached as zeros.)
/// The wait is paid once per weight; sources prepared together share one.
pub(crate) fn ensure_planes(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    sources: &[PlanesSource<'_>],
) -> Result<()> {
    if sources.iter().all(|source| source.planes.get().is_some()) {
        return Ok(());
    }
    let mut enc = device
        .command_encoder()
        .map_err(|error| anyhow!("prepared GGUF planes encoder: {error}"))?;
    let mut prepared = Vec::with_capacity(sources.len());
    for source in sources {
        if source.planes.get().is_none() {
            let planes = prepare_repacked_gguf_weight(
                &mut enc,
                registry,
                device,
                source.weight,
                &source.params,
            )
            .context("prepare GGUF weight planes")?;
            prepared.push((source.planes, planes));
        }
    }
    enc.commit_and_wait_labeled("qwen35.prepared_planes.prepare")
        .map_err(|error| anyhow!("prepare GGUF weight planes: {error}"))?;
    for (cell, planes) in prepared {
        let _ = cell.set(planes);
    }
    Ok(())
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
/// shape inside the open joint scope (32 rows; any width up to 32 in a
/// DFlash2 target verify step under `HF2Q_REPACKED_GGUF_ANY_WIDTH=1`).
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
/// caller vouches for the joint width and says whether the step is a DFlash2
/// target verify: only such a step is admitted below 32 rows.
pub(crate) fn admits_head(
    rows: u32,
    weight: &MlxBuffer,
    ggml_type: GgmlType,
    in_features: u32,
    out_features: u32,
    dflash2_verify: bool,
) -> bool {
    head_enabled()
        && weight.dtype() == DType::U8
        && super::gpu_ffn::repacked_dense_m32_shape_admits(
            &params(rows, ggml_type, in_features, out_features),
            dflash2_verify,
        )
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
    project_alpha_beta_pass(
        enc,
        registry,
        device,
        input_bf16,
        alpha,
        beta,
        ggml_type,
        planes,
        rows,
        in_features,
        heads,
        Pass::Scheduled,
    )
}

/// [`project_alpha_beta`] issued as `pass`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_alpha_beta_pass(
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
    pass: Pass,
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
    project_pass(
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
        pass,
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

/// Whether [`normed_bf16_input`] serves activations of `in_features`
/// columns on this device: the stock RMS norm would run its float4 kernel on
/// them, whose bits the BF16-output kernel reproduces, and the kernel's
/// thread count and threadgroup memory are within the device's limits.
pub(crate) fn normed_bf16_available(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    in_features: u32,
) -> Result<bool> {
    mlx_native::ops::rms_norm::rms_norm_f32_to_bf16_supported(
        registry,
        device.metal_device(),
        in_features,
    )
    .map_err(|error| anyhow!("probe the BF16-output RMS norm: {error}"))
}

/// Whether [`normed_bf16_input`] publishes the trace witness of its norm.
/// It follows the stock norm it replaces: the DeltaNet pre-norm publishes
/// one, the attention pre-norm does not.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NormWitness {
    Publish,
    None,
}

/// Encode the RMS norm of the F32 `[rows, in_features]` activation straight
/// into the BF16 rows the prepared projections read: bitwise the stock norm
/// followed by [`bf16_input`], without the cast dispatch and its barrier.
/// The caller checked [`normed_bf16_available`], says whether the norm it
/// replaces publishes a trace witness, and places the barrier before the
/// projections.
#[allow(clippy::too_many_arguments)]
pub(crate) fn normed_bf16_input(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    norm_weight: &MlxBuffer,
    rows: u32,
    in_features: u32,
    eps: f32,
    witness: NormWitness,
) -> Result<MlxBuffer> {
    let elements = rows as usize * in_features as usize;
    let output = pooled_alloc_buffer(
        device,
        elements * std::mem::size_of::<half::bf16>(),
        DType::BF16,
        vec![rows as usize, in_features as usize],
    )
    .map_err(|error| anyhow!("allocate normalized prepared projection input: {error}"))?;
    let mut params = pooled_alloc_buffer(device, 8, DType::F32, vec![2])
        .map_err(|error| anyhow!("allocate norm params: {error}"))?;
    params
        .as_mut_slice::<f32>()
        .map_err(|error| anyhow!("write norm params: {error}"))?
        .copy_from_slice(&[eps, in_features as f32]);
    let mut dispatch = |enc: &mut CommandEncoder| {
        mlx_native::ops::rms_norm::dispatch_rms_norm_f32_to_bf16(
            enc,
            registry,
            device.metal_device(),
            input,
            norm_weight,
            &output,
            &params,
            rows,
            in_features,
        )
    };
    match witness {
        NormWitness::Publish => crate::inference::native_observation::witness_command_rms_norm(
            enc,
            input,
            rows as usize,
            in_features as usize,
            dispatch,
        ),
        NormWitness::None => dispatch(enc),
    }
    .context("RMS norm into the prepared projection input")?;
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
    project_pass(
        enc,
        registry,
        device,
        input_bf16,
        weight,
        ggml_type,
        planes,
        rows,
        in_features,
        out_features,
        Pass::Scheduled,
    )
}

/// [`project`] issued as `pass`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_pass(
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
    pass: Pass,
) -> Result<MlxBuffer> {
    let output = pooled_alloc_buffer(
        device,
        rows as usize * out_features as usize * std::mem::size_of::<f32>(),
        DType::F32,
        vec![rows as usize, out_features as usize],
    )
    .map_err(|error| anyhow!("allocate prepared projection output: {error}"))?;
    project_into_pass(
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
        pass,
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
    project_into_pass(
        enc,
        registry,
        device,
        input_bf16,
        weight,
        ggml_type,
        planes,
        output,
        rows,
        in_features,
        out_features,
        Pass::Scheduled,
    )
}

/// [`project_into`] issued as `pass`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_into_pass(
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
    pass: Pass,
) -> Result<()> {
    let params = params(rows, ggml_type, in_features, out_features);
    ensure_planes(
        registry,
        device,
        &[PlanesSource {
            planes,
            weight,
            params,
        }],
    )?;
    let prepared = planes
        .get()
        .context("prepared projection GGUF weight was not retained")?;
    match pass {
        Pass::Scheduled => dispatch_repacked_gguf_mm_bf16_input(
            enc, registry, device, input_bf16, prepared, output, &params,
        ),
        Pass::One => dispatch_repacked_gguf_mm_bf16_input_one_pass(
            enc, registry, device, input_bf16, prepared, output, &params,
        ),
    }
    .context("dispatch prepared GGUF projection")
}
