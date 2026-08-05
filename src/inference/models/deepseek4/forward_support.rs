//! Shared allocation and resident block-matmul helpers for DeepSeek-V4 graphs.

use anyhow::{bail, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::quantized_matmul_ggml::{GgmlQuantizedMatmulParams, GgmlType};
use mlx_native::ops::quantized_matmul_id_ggml::GgmlQuantizedMatmulIdParams;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::residency::RawMatrixRef;

pub(super) fn alloc(
    device: &MlxDevice,
    dtype: DType,
    shape: Vec<usize>,
    label: &str,
) -> Result<MlxBuffer> {
    let elements = shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .with_context(|| format!("DeepSeek-V4 {label} shape overflow"))?;
    let bytes = elements
        .checked_mul(dtype.size_of())
        .with_context(|| format!("DeepSeek-V4 {label} byte size overflow"))?;
    device
        .alloc_buffer(bytes, dtype, shape)
        .with_context(|| format!("allocate DeepSeek-V4 {label}"))
}

pub(super) fn rms_params(
    device: &MlxDevice,
    epsilon: f32,
    dim: usize,
    label: &str,
) -> Result<MlxBuffer> {
    let mut params = alloc(device, DType::F32, vec![2], label)?;
    params
        .as_mut_slice::<f32>()?
        .copy_from_slice(&[epsilon, dim as f32]);
    Ok(params)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn raw_matmul(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &RawMatrixRef<'_>,
    output: &MlxBuffer,
    m: usize,
    n: usize,
    k: usize,
    label: &str,
) -> Result<()> {
    if weight.shape != [n, k] {
        bail!(
            "DeepSeek-V4 {label} shape drift: got {:?}, expected [{n}, {k}]",
            weight.shape
        );
    }
    if matches!(
        weight.ggml_type,
        GgmlType::F32 | GgmlType::F16 | GgmlType::I16 | GgmlType::I32
    ) {
        bail!(
            "DeepSeek-V4 optimized {label} requires block-quantized storage, got {:?}",
            weight.ggml_type
        );
    }
    session.barrier_between(&[input, weight.buffer], &[output]);
    session
        .quantized_matmul_ggml(
            registry,
            device,
            input,
            weight.buffer,
            output,
            &GgmlQuantizedMatmulParams {
                m: u32::try_from(m).context("DeepSeek-V4 matmul rows exceed u32")?,
                n: u32::try_from(n).context("DeepSeek-V4 matmul outputs exceed u32")?,
                k: u32::try_from(k).context("DeepSeek-V4 matmul input exceeds u32")?,
                ggml_type: weight.ggml_type,
            },
        )
        .with_context(|| format!("encode DeepSeek-V4 {label}"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn expert_matmul(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &RawMatrixRef<'_>,
    safe_ids: &MlxBuffer,
    output: &MlxBuffer,
    n_tokens: usize,
    top_k: usize,
    experts: usize,
    n: usize,
    k: usize,
    label: &str,
) -> Result<()> {
    if weight.shape != [experts, n, k] {
        bail!(
            "DeepSeek-V4 {label} shape drift: got {:?}, expected [{experts}, {n}, {k}]",
            weight.shape
        );
    }
    if safe_ids.dtype() != DType::U32 {
        bail!("DeepSeek-V4 {label} expert IDs must be sanitized U32");
    }
    let block = weight.ggml_type.block_values() as usize;
    if k % block != 0 {
        bail!("DeepSeek-V4 {label} input dimension is not quant-block aligned");
    }
    let expert_stride = n
        .checked_mul(k / block)
        .and_then(|blocks| blocks.checked_mul(weight.ggml_type.block_bytes() as usize))
        .context("DeepSeek-V4 expert stride overflow")?;
    let n_tokens = u32::try_from(n_tokens).context("DeepSeek-V4 expert token count exceeds u32")?;
    let top_k = u32::try_from(top_k).context("DeepSeek-V4 expert top-k exceeds u32")?;
    let n = u32::try_from(n).context("DeepSeek-V4 expert output width exceeds u32")?;
    let k = u32::try_from(k).context("DeepSeek-V4 expert input width exceeds u32")?;
    let n_experts = u32::try_from(experts).context("DeepSeek-V4 expert count exceeds u32")?;
    let expert_stride =
        u64::try_from(expert_stride).context("DeepSeek-V4 expert stride exceeds u64")?;
    session.barrier_between(&[input, weight.buffer, safe_ids], &[output]);
    session
        .quantized_matmul_id_ggml(
            registry,
            device,
            input,
            weight.buffer,
            safe_ids,
            output,
            &GgmlQuantizedMatmulIdParams {
                n_tokens,
                top_k,
                n,
                k,
                n_experts,
                expert_stride,
                ggml_type: weight.ggml_type,
            },
        )
        .with_context(|| format!("encode DeepSeek-V4 {label}"))
}
