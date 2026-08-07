//! Shared allocation and resident block-matmul helpers for DeepSeek-V4 graphs.

use std::cell::{Cell, RefCell};

use anyhow::{bail, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::dense_gemm::{dispatch_dense_matvec_f32, DenseGemmF16Params};
use mlx_native::ops::dense_mm_f16::{dense_matmul_f16_f32_tensor, DenseMmF16F32Params};
use mlx_native::ops::dense_mm_f32_f32::{dense_matmul_f32_f32_tensor, DenseMmF32F32Params};
use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml_batched_mv, GgmlBatchedQuantizedMatmulInputStrides,
    GgmlBatchedQuantizedMatmulParams, GgmlQuantizedMatmulParams, GgmlType, MM_ROUTING_THRESHOLD,
};
use mlx_native::ops::quantized_matmul_id_ggml::{
    GgmlQuantizedMatmulIdParams, IdMmScratch,
};
use mlx_native::ops::transpose::permute_021_f32;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxBufferPool, MlxDevice};

use super::residency::RawMatrixRef;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransientPoolPhase {
    Inactive,
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct TransientScratchStats {
    pub free_buffers: usize,
    pub free_bytes: usize,
}

thread_local! {
    // Prefill scratch grows with prompt position (notably the ratio-four
    // indexer score/top-k workspaces). Keeping it in the decode arena retains
    // the cold prompt's multi-gigabyte high-water for the rest of an agentic
    // session. Separate arenas let serving release prefill scratch after TTFT
    // while preserving the small steady-state decode working set.
    static PREFILL_POOL: RefCell<MlxBufferPool> = RefCell::new(MlxBufferPool::new());
    static DECODE_POOL: RefCell<MlxBufferPool> = RefCell::new(MlxBufferPool::new());
    static TRANSIENT_POOL_PHASE: Cell<TransientPoolPhase> = const { Cell::new(TransientPoolPhase::Inactive) };
}

pub(super) fn begin_prefill_pool_layer() {
    begin_transient_pool_cycle(TransientPoolPhase::Prefill);
}

pub(super) fn end_prefill_pool_layer() {
    end_transient_pool_cycle(TransientPoolPhase::Prefill);
}

pub(super) fn begin_decode_pool_token() {
    begin_transient_pool_cycle(TransientPoolPhase::Decode);
}

pub(super) fn end_decode_pool_token() {
    end_transient_pool_cycle(TransientPoolPhase::Decode);
}

fn begin_transient_pool_cycle(phase: TransientPoolPhase) {
    TRANSIENT_POOL_PHASE.with(|active| {
        debug_assert_eq!(
            active.get(),
            TransientPoolPhase::Inactive,
            "nested DeepSeek-V4 transient pool cycle"
        );
        active.set(phase);
    });
}

fn end_transient_pool_cycle(phase: TransientPoolPhase) {
    TRANSIENT_POOL_PHASE.with(|active| {
        debug_assert_eq!(
            active.get(),
            phase,
            "DeepSeek-V4 transient pool phase drift"
        );
        active.set(TransientPoolPhase::Inactive);
    });
    match phase {
        TransientPoolPhase::Prefill => PREFILL_POOL.with(|pool| pool.borrow_mut().reset()),
        TransientPoolPhase::Decode => DECODE_POOL.with(|pool| pool.borrow_mut().reset()),
        TransientPoolPhase::Inactive => unreachable!("cannot end an inactive transient pool"),
    }
}

fn pool_stats(pool: &RefCell<MlxBufferPool>) -> TransientScratchStats {
    let pool = pool.borrow();
    TransientScratchStats {
        free_buffers: pool.free_count(),
        free_bytes: pool.free_bytes(),
    }
}

fn release_pool(pool: &RefCell<MlxBufferPool>) -> TransientScratchStats {
    let mut pool = pool.borrow_mut();
    debug_assert_eq!(
        pool.in_use_count(),
        0,
        "cannot release active DeepSeek-V4 transient scratch"
    );
    let stats = TransientScratchStats {
        free_buffers: pool.free_count(),
        free_bytes: pool.free_bytes(),
    };
    pool.clear();
    stats
}

pub(crate) fn prefill_scratch_stats() -> TransientScratchStats {
    PREFILL_POOL.with(pool_stats)
}

pub(crate) fn decode_scratch_stats() -> TransientScratchStats {
    DECODE_POOL.with(pool_stats)
}

pub(crate) fn release_prefill_scratch() -> TransientScratchStats {
    debug_assert_eq!(
        TRANSIENT_POOL_PHASE.with(Cell::get),
        TransientPoolPhase::Inactive,
        "cannot release DeepSeek-V4 prefill scratch during a graph"
    );
    PREFILL_POOL.with(release_pool)
}

pub(crate) fn release_decode_scratch() -> TransientScratchStats {
    debug_assert_eq!(
        TRANSIENT_POOL_PHASE.with(Cell::get),
        TransientPoolPhase::Inactive,
        "cannot release DeepSeek-V4 decode scratch during a graph"
    );
    DECODE_POOL.with(release_pool)
}

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
    match TRANSIENT_POOL_PHASE.with(Cell::get) {
        TransientPoolPhase::Prefill => PREFILL_POOL
            .with(|pool| pool.borrow_mut().alloc(device, bytes, dtype, shape))
            .with_context(|| format!("allocate pooled DeepSeek-V4 prefill {label}")),
        TransientPoolPhase::Decode => DECODE_POOL
            .with(|pool| pool.borrow_mut().alloc(device, bytes, dtype, shape))
            .with_context(|| format!("allocate pooled DeepSeek-V4 decode {label}")),
        TransientPoolPhase::Inactive => device
            .alloc_buffer(bytes, dtype, shape)
            .with_context(|| format!("allocate DeepSeek-V4 {label}")),
    }
}

pub(super) fn alloc_persistent(
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
        .with_context(|| format!("allocate persistent DeepSeek-V4 {label}"))
}

pub(super) fn rms_params(
    device: &MlxDevice,
    epsilon: f32,
    dim: usize,
    label: &str,
) -> Result<MlxBuffer> {
    let mut params = alloc(device, DType::F32, vec![2], label)?;
    params
        .as_logical_mut_slice::<f32>()?
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
    session.barrier_between(&[input, weight.buffer], &[output]);
    match weight.ggml_type {
        GgmlType::F32 if m == 1 => dispatch_dense_matvec_f32(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            input,
            weight.buffer,
            output,
            &DenseGemmF16Params {
                m: 1,
                n: u32::try_from(n).context("DeepSeek-V4 matvec outputs exceed u32")?,
                k: u32::try_from(k).context("DeepSeek-V4 matvec input exceeds u32")?,
            },
        ),
        GgmlType::F32 => dense_matmul_f32_f32_tensor(
            session.encoder_mut(),
            registry,
            device,
            weight.buffer,
            input,
            output,
            &DenseMmF32F32Params {
                m: u32::try_from(m).context("DeepSeek-V4 matmul rows exceed u32")?,
                n: u32::try_from(n).context("DeepSeek-V4 matmul outputs exceed u32")?,
                k: u32::try_from(k).context("DeepSeek-V4 matmul input exceeds u32")?,
                src0_batch: 1,
                src1_batch: 1,
            },
        ),
        GgmlType::F16 => dense_matmul_f16_f32_tensor(
            session.encoder_mut(),
            registry,
            device,
            weight.buffer,
            input,
            output,
            &DenseMmF16F32Params {
                m: u32::try_from(m).context("DeepSeek-V4 matmul rows exceed u32")?,
                n: u32::try_from(n).context("DeepSeek-V4 matmul outputs exceed u32")?,
                k: u32::try_from(k).context("DeepSeek-V4 matmul input exceeds u32")?,
                src0_batch: 1,
                src1_batch: 1,
            },
        ),
        GgmlType::I16 | GgmlType::I32 => Err(mlx_native::MlxError::InvalidArgument(format!(
            "DeepSeek-V4 {label} cannot use integer-only matrix storage {:?}",
            weight.ggml_type
        ))),
        _ => session.quantized_matmul_ggml(
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
        ),
    }
    .with_context(|| format!("encode DeepSeek-V4 {label}"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn grouped_output_a(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &RawMatrixRef<'_>,
    output: &MlxBuffer,
    groups: usize,
    rank: usize,
    heads: usize,
    head_dim: usize,
) -> Result<()> {
    let group_width = heads
        .checked_mul(head_dim)
        .and_then(|width| width.checked_div(groups))
        .context("DeepSeek-V4 output-A group width overflow")?;
    let output_width = groups
        .checked_mul(rank)
        .context("DeepSeek-V4 output-A width overflow")?;
    if weight.shape != [output_width, group_width] {
        bail!(
            "DeepSeek-V4 output-A weight shape drift: got {:?}, expected [{output_width}, {group_width}]",
            weight.shape
        );
    }
    if weight.buffer.dtype() != DType::U8 {
        bail!("DeepSeek-V4 output-A grouped projection requires block-quantized storage");
    }
    let block = weight.ggml_type.block_values() as usize;
    if group_width % block != 0 {
        bail!("DeepSeek-V4 output-A group width is not block aligned");
    }
    let row_bytes = group_width
        .checked_div(block)
        .and_then(|blocks| blocks.checked_mul(weight.ggml_type.block_bytes() as usize))
        .context("DeepSeek-V4 output-A row-byte overflow")?;
    if matches!(weight.ggml_type, GgmlType::Q2_K | GgmlType::Q8_0) {
        session.barrier_between(&[input, weight.buffer], &[output]);
        return quantized_matmul_ggml_batched_mv(
            session.encoder_mut(),
            registry,
            device,
            input,
            weight.buffer,
            output,
            &GgmlBatchedQuantizedMatmulParams {
                batch: u32::try_from(groups)
                    .context("DeepSeek-V4 output-A groups exceed u32")?,
                m: 1,
                n: u32::try_from(rank).context("DeepSeek-V4 output-A rank exceeds u32")?,
                k: u32::try_from(group_width)
                    .context("DeepSeek-V4 output-A group width exceeds u32")?,
                ggml_type: weight.ggml_type,
            },
        )
        .context("encode batched DeepSeek-V4 output-A matvec");
    }
    for group in 0..groups {
        let input_view = input
            .slice_view(
                u64::try_from(group * group_width * DType::F32.size_of())
                    .context("DeepSeek-V4 output-A input offset exceeds u64")?,
                group_width,
            )
            .with_shape(vec![1, group_width])?;
        let weight_view = weight.buffer.slice_view(
            u64::try_from(group * rank * row_bytes)
                .context("DeepSeek-V4 output-A weight offset exceeds u64")?,
            rank * row_bytes,
        );
        let output_view = output
            .slice_view(
                u64::try_from(group * rank * DType::F32.size_of())
                    .context("DeepSeek-V4 output-A output offset exceeds u64")?,
                rank,
            )
            .with_shape(vec![1, rank])?;
        session.barrier_between(&[&input_view, &weight_view], &[&output_view]);
        session.quantized_matmul_ggml(
            registry,
            device,
            &input_view,
            &weight_view,
            &output_view,
            &GgmlQuantizedMatmulParams {
                m: 1,
                n: u32::try_from(rank).context("DeepSeek-V4 output-A rank exceeds u32")?,
                k: u32::try_from(group_width)
                    .context("DeepSeek-V4 output-A group width exceeds u32")?,
                ggml_type: weight.ggml_type,
            },
        )?;
    }
    Ok(())
}

pub(super) struct BatchedGroupedOutputArena {
    input_group_major: MlxBuffer,
    output_group_major: MlxBuffer,
}

impl BatchedGroupedOutputArena {
    pub(super) fn new(
        device: &MlxDevice,
        rows: usize,
        groups: usize,
        group_width: usize,
        rank: usize,
    ) -> Result<Self> {
        Ok(Self {
            input_group_major: alloc(
                device,
                DType::F32,
                vec![groups, rows, group_width],
                "batched output-A group-major input",
            )?,
            output_group_major: alloc(
                device,
                DType::F32,
                vec![groups, rows, rank],
                "batched output-A group-major output",
            )?,
        })
    }
}

/// Apply independently-quantized output-A groups with a true `m=rows`
/// matrix dispatch. Q8_0 consumes token-major attention through explicit
/// input strides; other formats retain the group-major input permutation.
#[allow(clippy::too_many_arguments)]
pub(super) fn grouped_output_a_batched(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &RawMatrixRef<'_>,
    output: &MlxBuffer,
    arena: &BatchedGroupedOutputArena,
    rows: usize,
    groups: usize,
    rank: usize,
    heads: usize,
    head_dim: usize,
) -> Result<()> {
    let group_width = heads
        .checked_mul(head_dim)
        .and_then(|width| width.checked_div(groups))
        .context("DeepSeek-V4 batched output-A group width overflow")?;
    let output_width = groups
        .checked_mul(rank)
        .context("DeepSeek-V4 batched output-A width overflow")?;
    if weight.shape != [output_width, group_width] {
        bail!(
            "DeepSeek-V4 output-A weight shape drift: got {:?}, expected [{output_width}, {group_width}]",
            weight.shape
        );
    }
    if weight.buffer.dtype() != DType::U8 {
        bail!("DeepSeek-V4 output-A grouped projection requires block-quantized storage");
    }
    let block = weight.ggml_type.block_values() as usize;
    if group_width % block != 0 {
        bail!("DeepSeek-V4 output-A group width is not block aligned");
    }
    let row_bytes = group_width
        .checked_div(block)
        .and_then(|blocks| blocks.checked_mul(weight.ggml_type.block_bytes() as usize))
        .context("DeepSeek-V4 output-A row-byte overflow")?;

    if weight.ggml_type == GgmlType::Q8_0 && rows > MM_ROUTING_THRESHOLD as usize {
        let input_row_bytes = groups
            .checked_mul(group_width)
            .and_then(|elements| elements.checked_mul(DType::F32.size_of()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .context("DeepSeek-V4 output-A token-major row stride overflow")?;
        let input_group_bytes = group_width
            .checked_mul(DType::F32.size_of())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .context("DeepSeek-V4 output-A token-major group stride overflow")?;
        session.barrier_between(&[input, weight.buffer], &[&arena.output_group_major]);
        session
            .quantized_matmul_ggml_batched_mm_strided_input(
                registry,
                device,
                input,
                weight.buffer,
                &arena.output_group_major,
                &GgmlBatchedQuantizedMatmulParams {
                    batch: u32::try_from(groups)
                        .context("DeepSeek-V4 output-A groups exceed u32")?,
                    m: u32::try_from(rows).context("DeepSeek-V4 output-A rows exceed u32")?,
                    n: u32::try_from(rank).context("DeepSeek-V4 output-A rank exceeds u32")?,
                    k: u32::try_from(group_width)
                        .context("DeepSeek-V4 output-A group width exceeds u32")?,
                    ggml_type: weight.ggml_type,
                },
                &GgmlBatchedQuantizedMatmulInputStrides {
                    row_bytes: input_row_bytes,
                    batch_bytes: input_group_bytes,
                },
            )
            .context("encode strided native-batched DeepSeek-V4 Q8_0 output-A projection")?;
    } else {
        session.barrier_between(&[input], &[&arena.input_group_major]);
        permute_021_f32(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            input,
            &arena.input_group_major,
            rows,
            groups,
            group_width,
        )?;
        for group in 0..groups {
            let input_view = arena
                .input_group_major
                .slice_view(
                    u64::try_from(group * rows * group_width * DType::F32.size_of())
                        .context("DeepSeek-V4 batched output-A input offset exceeds u64")?,
                    rows * group_width,
                )
                .with_shape(vec![rows, group_width])?;
            let weight_view = weight.buffer.slice_view(
                u64::try_from(group * rank * row_bytes)
                    .context("DeepSeek-V4 batched output-A weight offset exceeds u64")?,
                rank * row_bytes,
            );
            let output_view = arena
                .output_group_major
                .slice_view(
                    u64::try_from(group * rows * rank * DType::F32.size_of())
                        .context("DeepSeek-V4 batched output-A output offset exceeds u64")?,
                    rows * rank,
                )
                .with_shape(vec![rows, rank])?;
            session.barrier_between(&[&input_view, &weight_view], &[&output_view]);
            session.quantized_matmul_ggml(
                registry,
                device,
                &input_view,
                &weight_view,
                &output_view,
                &GgmlQuantizedMatmulParams {
                    m: u32::try_from(rows).context("DeepSeek-V4 output-A rows exceed u32")?,
                    n: u32::try_from(rank).context("DeepSeek-V4 output-A rank exceeds u32")?,
                    k: u32::try_from(group_width)
                        .context("DeepSeek-V4 output-A group width exceeds u32")?,
                    ggml_type: weight.ggml_type,
                },
            )?;
        }
    }
    session.barrier_between(&[&arena.output_group_major], &[output]);
    permute_021_f32(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        &arena.output_group_major,
        output,
        groups,
        rows,
        rank,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ExpertMatmulRoute {
    Auto,
    ForceMv,
    SlottedMm,
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
    route: ExpertMatmulRoute,
    scratch: Option<&mut IdMmScratch>,
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
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens,
        top_k,
        n,
        k,
        n_experts,
        expert_stride,
        ggml_type: weight.ggml_type,
    };
    match (route, scratch) {
        (ExpertMatmulRoute::Auto, Some(scratch)) => session
            .quantized_matmul_id_ggml_pooled(
                registry,
                device,
                input,
                weight.buffer,
                safe_ids,
                output,
                scratch,
                &params,
            ),
        (ExpertMatmulRoute::Auto, None) => session.quantized_matmul_id_ggml(
            registry,
            device,
            input,
            weight.buffer,
            safe_ids,
            output,
            &params,
        ),
        (ExpertMatmulRoute::ForceMv, _) => session.quantized_matmul_id_ggml_mv(
            registry,
            device,
            input,
            weight.buffer,
            safe_ids,
            output,
            &params,
        ),
        (ExpertMatmulRoute::SlottedMm, Some(scratch)) => session
            .quantized_matmul_id_ggml_pooled_slotted(
                registry,
                device,
                input,
                weight.buffer,
                safe_ids,
                output,
                scratch,
                &params,
            ),
        (ExpertMatmulRoute::SlottedMm, None) => {
            bail!("DeepSeek-V4 {label} slotted mm_id requires caller-owned scratch")
        }
    }
    .with_context(|| format!("encode DeepSeek-V4 {label}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::gpu::GpuContext;

    #[test]
    fn raw_matmul_accepts_quality_sensitive_f32_weights() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let mut ctx = GpuContext::new().unwrap();
        let device = ctx.device().clone();
        let mut input = alloc(&device, DType::F32, vec![1, 32], "test input").unwrap();
        let mut weight = alloc(&device, DType::F32, vec![32, 32], "test weight").unwrap();
        let output = alloc(&device, DType::F32, vec![1, 32], "test output").unwrap();
        for (index, value) in input.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
            *value = index as f32 * 0.25 - 2.0;
        }
        for row in 0..32 {
            weight.as_mut_slice::<f32>().unwrap()[row * 32 + row] = 1.0;
        }
        let shape = [32, 32];
        let weight = RawMatrixRef {
            buffer: &weight,
            ggml_type: GgmlType::F32,
            shape: &shape,
        };
        let (executor, registry) = ctx.split();
        let mut session = executor.begin().unwrap();
        raw_matmul(
            &mut session,
            registry,
            &device,
            &input,
            &weight,
            &output,
            1,
            32,
            32,
            "F32 identity",
        )
        .unwrap();
        session.finish().unwrap();
        assert_eq!(
            output.as_slice::<f32>().unwrap(),
            input.as_slice::<f32>().unwrap()
        );
    }

    #[test]
    fn decode_pool_reuses_transient_metal_allocations_between_tokens() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let ctx = GpuContext::new().unwrap();
        let device = ctx.device().clone();

        begin_decode_pool_token();
        let first = alloc(&device, DType::F32, vec![257], "first decode scratch").unwrap();
        let first_ptr = first.contents_ptr();
        drop(first);
        end_decode_pool_token();

        begin_decode_pool_token();
        let second = alloc(&device, DType::F32, vec![257], "second decode scratch").unwrap();
        let second_ptr = second.contents_ptr();
        drop(second);
        end_decode_pool_token();

        assert_eq!(second_ptr, first_ptr);
    }

    #[test]
    fn prefill_and_decode_scratch_have_independent_release_lifecycles() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let ctx = GpuContext::new().unwrap();
        let device = ctx.device().clone();
        release_prefill_scratch();
        release_decode_scratch();

        begin_prefill_pool_layer();
        {
            let _buffer = alloc(&device, DType::F32, vec![16], "prefill lifecycle test")
                .expect("allocate prefill test buffer");
        }
        end_prefill_pool_layer();
        assert!(prefill_scratch_stats().free_bytes >= 64);
        assert_eq!(decode_scratch_stats().free_bytes, 0);

        let released_prefill = release_prefill_scratch();
        assert!(released_prefill.free_bytes >= 64);
        assert_eq!(prefill_scratch_stats().free_bytes, 0);

        begin_decode_pool_token();
        {
            let _buffer = alloc(&device, DType::F32, vec![32], "decode lifecycle test")
                .expect("allocate decode test buffer");
        }
        end_decode_pool_token();
        assert!(decode_scratch_stats().free_bytes >= 128);
        assert_eq!(prefill_scratch_stats().free_bytes, 0);

        let released_decode = release_decode_scratch();
        assert!(released_decode.free_bytes >= 128);
        assert_eq!(decode_scratch_stats().free_bytes, 0);
    }
}
