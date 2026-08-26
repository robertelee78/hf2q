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
use mlx_native::ops::quantized_matmul_id_ggml::{GgmlQuantizedMatmulIdParams, IdMmScratch};
use mlx_native::ops::transpose::permute_021_f32;
use mlx_native::{
    DType, DenseMatmulIdInputLayout, DenseMatmulIdMultiplicity, DenseMatmulIdParams,
    DenseMatmulIdRoute, DenseMatmulIdScratch, DenseMmBf16F32Params, KernelRegistry, MlxBuffer,
    MlxBufferPool, MlxDevice,
};

use crate::inference::dense_expert_activation::DenseExpertScratchCache;

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
    // Multi-layer recorded prefill may recycle GPU-only scratch between
    // layers, but CPU-written parameters must remain unique until the grouped
    // command buffer completes.
    static PREFILL_SUBMISSION_INPUT_POOL: RefCell<MlxBufferPool> = RefCell::new(MlxBufferPool::new());
    static PREFILL_SUBMISSION_INPUTS_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static DECODE_POOL: RefCell<MlxBufferPool> = RefCell::new(MlxBufferPool::new());
    static TRANSIENT_POOL_PHASE: Cell<TransientPoolPhase> = const { Cell::new(TransientPoolPhase::Inactive) };
    static DENSE_ID_SCRATCH_GATE: RefCell<DenseExpertScratchCache> = RefCell::new(DenseExpertScratchCache::default());
    static DENSE_ID_SCRATCH_UP: RefCell<DenseExpertScratchCache> = RefCell::new(DenseExpertScratchCache::default());
    static DENSE_ID_SCRATCH_DOWN: RefCell<DenseExpertScratchCache> = RefCell::new(DenseExpertScratchCache::default());
}

#[derive(Clone, Copy)]
pub(super) enum DenseExpertScratchSlot {
    Gate,
    Up,
    Down,
}

fn with_dense_expert_scratch<R>(
    slot: DenseExpertScratchSlot,
    activation_epoch: u64,
    device: &MlxDevice,
    n_experts: u32,
    max_tokens: u32,
    f: impl FnOnce(&DenseMatmulIdScratch) -> mlx_native::Result<R>,
) -> mlx_native::Result<R> {
    let cell = match slot {
        DenseExpertScratchSlot::Gate => &DENSE_ID_SCRATCH_GATE,
        DenseExpertScratchSlot::Up => &DENSE_ID_SCRATCH_UP,
        DenseExpertScratchSlot::Down => &DENSE_ID_SCRATCH_DOWN,
    };
    cell.with(|cell| {
        cell.borrow_mut()
            .with(activation_epoch, device, n_experts, max_tokens, f)
    })
}

pub(super) fn begin_prefill_pool_layer() {
    begin_transient_pool_cycle(TransientPoolPhase::Prefill);
}

pub(super) fn end_prefill_pool_layer() {
    end_transient_pool_cycle(TransientPoolPhase::Prefill);
}

pub(super) fn begin_prefill_submission_inputs() {
    debug_assert_eq!(
        TRANSIENT_POOL_PHASE.with(Cell::get),
        TransientPoolPhase::Inactive,
        "DeepSeek-V4 prefill submission inputs must begin between layer cycles"
    );
    PREFILL_SUBMISSION_INPUTS_ACTIVE.with(|active| {
        debug_assert!(!active.get(), "nested DeepSeek-V4 prefill input submission");
        active.set(true);
    });
}

pub(super) fn end_prefill_submission_inputs() {
    debug_assert_eq!(
        TRANSIENT_POOL_PHASE.with(Cell::get),
        TransientPoolPhase::Inactive,
        "DeepSeek-V4 prefill submission inputs must end between layer cycles"
    );
    PREFILL_SUBMISSION_INPUTS_ACTIVE.with(|active| {
        debug_assert!(
            active.get(),
            "inactive DeepSeek-V4 prefill input submission"
        );
        active.set(false);
    });
    PREFILL_SUBMISSION_INPUT_POOL.with(|pool| pool.borrow_mut().reset());
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

fn add_stats(left: TransientScratchStats, right: TransientScratchStats) -> TransientScratchStats {
    TransientScratchStats {
        free_buffers: left.free_buffers.saturating_add(right.free_buffers),
        free_bytes: left.free_bytes.saturating_add(right.free_bytes),
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
    add_stats(
        PREFILL_POOL.with(pool_stats),
        PREFILL_SUBMISSION_INPUT_POOL.with(pool_stats),
    )
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
    debug_assert!(
        !PREFILL_SUBMISSION_INPUTS_ACTIVE.with(Cell::get),
        "cannot release active DeepSeek-V4 prefill submission inputs"
    );
    add_stats(
        PREFILL_POOL.with(release_pool),
        PREFILL_SUBMISSION_INPUT_POOL.with(release_pool),
    )
}

pub(crate) fn release_decode_scratch() -> TransientScratchStats {
    debug_assert_eq!(
        TRANSIENT_POOL_PHASE.with(Cell::get),
        TransientPoolPhase::Inactive,
        "cannot release DeepSeek-V4 decode scratch during a graph"
    );
    DECODE_POOL.with(release_pool)
}

pub(crate) fn idle_runtime_scratch_bytes() -> Result<u64> {
    anyhow::ensure!(
        TRANSIENT_POOL_PHASE.with(Cell::get) == TransientPoolPhase::Inactive,
        "DeepSeek-V4 transient scratch is active"
    );
    anyhow::ensure!(
        !PREFILL_SUBMISSION_INPUTS_ACTIVE.with(Cell::get),
        "DeepSeek-V4 prefill submission inputs are active"
    );
    let mut bytes = 0u64;
    for pool in [&PREFILL_POOL, &PREFILL_SUBMISSION_INPUT_POOL, &DECODE_POOL] {
        pool.with(|pool| {
            let pool = pool.borrow();
            anyhow::ensure!(
                pool.in_use_count() == 0,
                "DeepSeek-V4 transient scratch still has in-use buffers"
            );
            bytes = bytes
                .checked_add(pool.free_bytes() as u64)
                .ok_or_else(|| anyhow::anyhow!("DeepSeek-V4 scratch byte total overflow"))?;
            Ok::<_, anyhow::Error>(())
        })?;
    }
    for scratch in [
        &DENSE_ID_SCRATCH_GATE,
        &DENSE_ID_SCRATCH_UP,
        &DENSE_ID_SCRATCH_DOWN,
    ] {
        scratch.with(|scratch| {
            bytes = bytes
                .checked_add(scratch.borrow().owned_bytes())
                .ok_or_else(|| anyhow::anyhow!("DeepSeek-V4 expert scratch byte overflow"))?;
            Ok::<_, anyhow::Error>(())
        })?;
    }
    Ok(bytes)
}

pub(crate) fn release_idle_runtime_scratch() -> Result<u64> {
    let expected = idle_runtime_scratch_bytes()?;
    for pool in [&PREFILL_POOL, &PREFILL_SUBMISSION_INPUT_POOL, &DECODE_POOL] {
        pool.with(|pool| pool.borrow_mut().clear());
    }
    for scratch in [
        &DENSE_ID_SCRATCH_GATE,
        &DENSE_ID_SCRATCH_UP,
        &DENSE_ID_SCRATCH_DOWN,
    ] {
        scratch.with(|scratch| {
            scratch.borrow_mut().release_owned_bytes();
        });
    }
    Ok(expected)
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
    // DeepSeek-V4 transient graph outputs fully cover these buffers before
    // any consumer reads them. ADR-042 records the hostile-fill proof: all
    // 11,954 prefill/decode state dumps were byte-identical for artifact
    // 936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d.
    // Keep this opt-in at the family boundary; other model families retain
    // mlx-native's zero-on-fresh default until they have equivalent proof.
    match TRANSIENT_POOL_PHASE.with(Cell::get) {
        TransientPoolPhase::Prefill => PREFILL_POOL
            .with(|pool| {
                pool.borrow_mut()
                    .alloc_uninitialized(device, bytes, dtype, shape)
            })
            .with_context(|| format!("allocate pooled DeepSeek-V4 prefill {label}")),
        TransientPoolPhase::Decode => DECODE_POOL
            .with(|pool| {
                pool.borrow_mut()
                    .alloc_uninitialized(device, bytes, dtype, shape)
            })
            .with_context(|| format!("allocate pooled DeepSeek-V4 decode {label}")),
        TransientPoolPhase::Inactive => device
            .alloc_buffer(bytes, dtype, shape)
            .with_context(|| format!("allocate DeepSeek-V4 {label}")),
    }
}

pub(super) fn alloc_host_input(
    device: &MlxDevice,
    dtype: DType,
    shape: Vec<usize>,
    label: &str,
) -> Result<MlxBuffer> {
    if !PREFILL_SUBMISSION_INPUTS_ACTIVE.with(Cell::get) {
        return alloc(device, dtype, shape, label);
    }
    debug_assert_eq!(
        TRANSIENT_POOL_PHASE.with(Cell::get),
        TransientPoolPhase::Prefill,
        "DeepSeek-V4 grouped prefill input allocated outside a layer cycle"
    );
    let elements = shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .with_context(|| format!("DeepSeek-V4 {label} shape overflow"))?;
    let bytes = elements
        .checked_mul(dtype.size_of())
        .with_context(|| format!("DeepSeek-V4 {label} byte size overflow"))?;
    PREFILL_SUBMISSION_INPUT_POOL
        .with(|pool| {
            pool.borrow_mut()
                .alloc_uninitialized(device, bytes, dtype, shape)
        })
        .with_context(|| format!("allocate pooled DeepSeek-V4 prefill input {label}"))
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
    let mut params = alloc_host_input(device, DType::F32, vec![2], label)?;
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
        GgmlType::BF16 => mlx_native::dense_matmul_bf16_f32_auto(
            session.encoder_mut(),
            registry,
            device,
            weight.buffer,
            input,
            output,
            &DenseMmBf16F32Params {
                m: u32::try_from(m).context("DeepSeek-V4 matmul rows exceed u32")?,
                n: u32::try_from(n).context("DeepSeek-V4 matmul outputs exceed u32")?,
                k: u32::try_from(k).context("DeepSeek-V4 matmul input exceeds u32")?,
                src0_batch: 1,
                src1_batch: 1,
            },
        )
        .map(|_| ()),
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
    if matches!(
        weight.ggml_type,
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
    ) {
        let params = DenseMmBf16F32Params {
            m: 1,
            n: u32::try_from(rank).context("DeepSeek-V4 output-A rank exceeds u32")?,
            k: u32::try_from(group_width)
                .context("DeepSeek-V4 output-A group width exceeds u32")?,
            src0_batch: u32::try_from(groups).context("DeepSeek-V4 output-A groups exceed u32")?,
            src1_batch: u32::try_from(groups).context("DeepSeek-V4 output-A groups exceed u32")?,
        };
        session.barrier_between(&[input, weight.buffer], &[output]);
        match weight.ggml_type {
            GgmlType::F32 => dense_matmul_f32_f32_tensor(
                session.encoder_mut(),
                registry,
                device,
                weight.buffer,
                input,
                output,
                &DenseMmF32F32Params {
                    m: params.m,
                    n: params.n,
                    k: params.k,
                    src0_batch: params.src0_batch,
                    src1_batch: params.src1_batch,
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
                    m: params.m,
                    n: params.n,
                    k: params.k,
                    src0_batch: params.src0_batch,
                    src1_batch: params.src1_batch,
                },
            ),
            GgmlType::BF16 => mlx_native::dense_matmul_bf16_f32_auto(
                session.encoder_mut(),
                registry,
                device,
                weight.buffer,
                input,
                output,
                &params,
            )
            .map(|_| ()),
            _ => unreachable!(),
        }
        .context("encode native-scalar batched DeepSeek-V4 output-A projection")?;
        return Ok(());
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
    if matches!(
        weight.ggml_type,
        GgmlType::Q2_K | GgmlType::Q5_0 | GgmlType::Q8_0
    ) {
        session.barrier_between(&[input, weight.buffer], &[output]);
        return quantized_matmul_ggml_batched_mv(
            session.encoder_mut(),
            registry,
            device,
            input,
            weight.buffer,
            output,
            &GgmlBatchedQuantizedMatmulParams {
                batch: u32::try_from(groups).context("DeepSeek-V4 output-A groups exceed u32")?,
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
/// matrix dispatch. Formats with a native strided-input kernel consume
/// token-major attention directly; other formats retain the group-major
/// input permutation.
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
    if matches!(
        weight.ggml_type,
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
    ) {
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
        session.barrier_between(
            &[&arena.input_group_major, weight.buffer],
            &[&arena.output_group_major],
        );
        let m = u32::try_from(rows).context("DeepSeek-V4 output-A rows exceed u32")?;
        let n = u32::try_from(rank).context("DeepSeek-V4 output-A rank exceeds u32")?;
        let k =
            u32::try_from(group_width).context("DeepSeek-V4 output-A group width exceeds u32")?;
        let batches = u32::try_from(groups).context("DeepSeek-V4 output-A groups exceed u32")?;
        match weight.ggml_type {
            GgmlType::F32 => dense_matmul_f32_f32_tensor(
                session.encoder_mut(),
                registry,
                device,
                weight.buffer,
                &arena.input_group_major,
                &arena.output_group_major,
                &DenseMmF32F32Params {
                    m,
                    n,
                    k,
                    src0_batch: batches,
                    src1_batch: batches,
                },
            ),
            GgmlType::F16 => dense_matmul_f16_f32_tensor(
                session.encoder_mut(),
                registry,
                device,
                weight.buffer,
                &arena.input_group_major,
                &arena.output_group_major,
                &DenseMmF16F32Params {
                    m,
                    n,
                    k,
                    src0_batch: batches,
                    src1_batch: batches,
                },
            ),
            GgmlType::BF16 => mlx_native::dense_matmul_bf16_f32_auto(
                session.encoder_mut(),
                registry,
                device,
                weight.buffer,
                &arena.input_group_major,
                &arena.output_group_major,
                &DenseMmBf16F32Params {
                    m,
                    n,
                    k,
                    src0_batch: batches,
                    src1_batch: batches,
                },
            )
            .map(|_| ()),
            _ => unreachable!(),
        }
        .context("encode native-scalar batched DeepSeek-V4 output-A prefill")?;
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
        return Ok(());
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

    if matches!(weight.ggml_type, GgmlType::Q5_0 | GgmlType::Q8_0)
        && rows > MM_ROUTING_THRESHOLD as usize
    {
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
            .with_context(|| {
                format!(
                    "encode strided native-batched DeepSeek-V4 {:?} output-A projection",
                    weight.ggml_type
                )
            })?;
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
pub(super) fn expert_matmul_pair(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    first_weight: &RawMatrixRef<'_>,
    second_weight: &RawMatrixRef<'_>,
    safe_ids: &MlxBuffer,
    first_output: &MlxBuffer,
    second_output: &MlxBuffer,
    n_tokens: usize,
    top_k: usize,
    experts: usize,
    n: usize,
    k: usize,
    scratch: &mut IdMmScratch,
    label: &str,
) -> Result<()> {
    let expected_shape = [experts, n, k];
    if first_weight.shape != expected_shape || second_weight.shape != expected_shape {
        bail!(
            "DeepSeek-V4 {label} shape drift: first {:?}, second {:?}, expected [{experts}, {n}, {k}]",
            first_weight.shape,
            second_weight.shape,
        );
    }
    if first_weight.ggml_type != second_weight.ggml_type {
        bail!(
            "DeepSeek-V4 {label} quantization drift: first {:?}, second {:?}",
            first_weight.ggml_type,
            second_weight.ggml_type,
        );
    }
    if safe_ids.dtype() != DType::U32 {
        bail!("DeepSeek-V4 {label} expert IDs must be sanitized U32");
    }
    let block = first_weight.ggml_type.block_values() as usize;
    if k % block != 0 {
        bail!("DeepSeek-V4 {label} input dimension is not quant-block aligned");
    }
    let expert_stride = n
        .checked_mul(k / block)
        .and_then(|blocks| blocks.checked_mul(first_weight.ggml_type.block_bytes() as usize))
        .context("DeepSeek-V4 paired expert stride overflow")?;
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: u32::try_from(n_tokens)
            .context("DeepSeek-V4 paired expert token count exceeds u32")?,
        top_k: u32::try_from(top_k).context("DeepSeek-V4 paired expert top-k exceeds u32")?,
        n: u32::try_from(n).context("DeepSeek-V4 paired expert output width exceeds u32")?,
        k: u32::try_from(k).context("DeepSeek-V4 paired expert input width exceeds u32")?,
        n_experts: u32::try_from(experts).context("DeepSeek-V4 paired expert count exceeds u32")?,
        expert_stride: u64::try_from(expert_stride)
            .context("DeepSeek-V4 paired expert stride exceeds u64")?,
        ggml_type: first_weight.ggml_type,
    };
    session.barrier_between(
        &[input, first_weight.buffer, second_weight.buffer, safe_ids],
        &[first_output, second_output],
    );
    session
        .quantized_matmul_id_ggml_pooled_pair(
            registry,
            device,
            input,
            first_weight.buffer,
            second_weight.buffer,
            safe_ids,
            first_output,
            second_output,
            scratch,
            &params,
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
    route: ExpertMatmulRoute,
    scratch: Option<&mut IdMmScratch>,
    activation_epoch: u64,
    id_multiplicity: DenseMatmulIdMultiplicity,
    dense_scratch_slot: DenseExpertScratchSlot,
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
    if matches!(
        weight.ggml_type,
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
    ) {
        let expected_dtype = match weight.ggml_type {
            GgmlType::F32 => DType::F32,
            GgmlType::F16 => DType::F16,
            GgmlType::BF16 => DType::BF16,
            _ => unreachable!(),
        };
        anyhow::ensure!(
            weight.buffer.dtype() == expected_dtype,
            "DeepSeek-V4 {label} declared {:?} but maps as {}",
            weight.ggml_type,
            weight.buffer.dtype()
        );
        let n_tokens =
            u32::try_from(n_tokens).context("DeepSeek-V4 scalar expert token count exceeds u32")?;
        let top_k = u32::try_from(top_k).context("DeepSeek-V4 scalar expert top-k exceeds u32")?;
        let n = u32::try_from(n).context("DeepSeek-V4 scalar expert output width exceeds u32")?;
        let k = u32::try_from(k).context("DeepSeek-V4 scalar expert input width exceeds u32")?;
        let n_experts =
            u32::try_from(experts).context("DeepSeek-V4 scalar expert count exceeds u32")?;
        let expert_stride_bytes = u64::from(n)
            .checked_mul(u64::from(k))
            .and_then(|elements| elements.checked_mul(expected_dtype.size_of() as u64))
            .context("DeepSeek-V4 scalar expert stride overflow")?;
        let params = DenseMatmulIdParams {
            m: n_tokens,
            n,
            k,
            top_k,
            n_experts,
            expert_stride_bytes,
            input_layout: if route == ExpertMatmulRoute::SlottedMm {
                DenseMatmulIdInputLayout::Slotted
            } else {
                DenseMatmulIdInputLayout::SharedPerToken
            },
            id_multiplicity,
            route: DenseMatmulIdRoute::Direct,
        };
        return with_dense_expert_scratch(
            dense_scratch_slot,
            activation_epoch,
            device,
            n_experts,
            n_tokens,
            |dense_scratch| {
                session
                    .dense_matmul_id_auto(
                        registry,
                        device,
                        activation_epoch,
                        weight.buffer,
                        input,
                        safe_ids,
                        output,
                        Some(dense_scratch),
                        &params,
                    )
                    .map(|_| ())
            },
        )
        .with_context(|| format!("encode DeepSeek-V4 {label} native scalar expert"));
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
        (ExpertMatmulRoute::Auto, Some(scratch)) => session.quantized_matmul_id_ggml_pooled(
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
    use crate::inference::dense_bf16_activation::NativeBf16Matrix;
    use crate::serve::gpu::GpuContext;

    fn encode_expert_test_matrix(values: &[f32], row_width: usize, ggml_type: GgmlType) -> Vec<u8> {
        match ggml_type {
            GgmlType::Q4_0 => crate::quantize::ggml_quants::q4_0::quantize(values, row_width, None),
            GgmlType::Q5_0 => crate::quantize::ggml_quants::q5_0::quantize(values, row_width, None),
            GgmlType::Q5_1 => crate::quantize::ggml_quants::q5_1::quantize(values, row_width, None),
            GgmlType::Q8_0 => crate::quantize::ggml_quants::q8_0::quantize(values, row_width, None),
            other => panic!("unsupported DeepSeek mixed expert test codec {other:?}"),
        }
    }

    fn upload_expert_test_matrix(
        device: &MlxDevice,
        bytes: &[u8],
        ggml_type: GgmlType,
        shape: &[usize; 3],
    ) -> MlxBuffer {
        let mut buffer = alloc(
            device,
            DType::U8,
            vec![bytes.len()],
            "mixed expert test matrix",
        )
        .unwrap();
        buffer.as_mut_slice::<u8>().unwrap().copy_from_slice(bytes);
        let expected = shape[0]
            * shape[1]
            * (shape[2] / ggml_type.block_values() as usize)
            * ggml_type.block_bytes() as usize;
        assert_eq!(bytes.len(), expected);
        buffer
    }

    fn expert_test_oracle(
        input: &[f32],
        ids: &[u32],
        packed: &[u8],
        ggml_type: GgmlType,
        experts: usize,
        rows: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut weights = vec![0.0_f32; experts * n * k];
        mlx_native::gguf::test_only_dequantize(packed, ggml_type, &mut weights)
            .expect("dequantize DeepSeek mixed expert oracle");
        let mut output = vec![0.0_f32; rows * n];
        for row in 0..rows {
            let expert = ids[row] as usize;
            for column in 0..n {
                output[row * n + column] = (0..k)
                    .map(|inner| {
                        input[row * k + inner] * weights[(expert * n + column) * k + inner]
                    })
                    .sum();
            }
        }
        output
    }

    fn assert_expert_test_output(label: &str, actual: &[f32], expected: &[f32]) {
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(actual.iter().any(|value| *value != 0.0));
        let (index, max_error) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .map(|(index, (&actual, &expected))| (index, (actual - expected).abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap();
        assert!(
            max_error < 2e-3,
            "{label} max error {max_error:.3e} at {index}: actual={} expected={}",
            actual[index],
            expected[index]
        );
    }

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
    fn mixed_expert_codecs_execute_independently_at_decode_and_mm_widths() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let mut ctx = GpuContext::new().unwrap();
        let device = ctx.device().clone();
        let (experts, n, k) = (3_usize, 32_usize, 32_usize);
        let shape = [experts, n, k];
        let codecs = [
            GgmlType::Q4_0,
            GgmlType::Q5_0,
            GgmlType::Q8_0,
            GgmlType::Q5_1,
        ];
        let mut seed = 0x4453_4D58_u32;
        let mut random = |count: usize, scale: f32| {
            (0..count)
                .map(|_| {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (seed as i32 as f32 / i32::MAX as f32) * scale
                })
                .collect::<Vec<_>>()
        };
        let source_weights = codecs
            .iter()
            .map(|_| random(experts * n * k, 0.08))
            .collect::<Vec<_>>();
        let packed = codecs
            .iter()
            .zip(&source_weights)
            .map(|(&codec, values)| encode_expert_test_matrix(values, k, codec))
            .collect::<Vec<_>>();
        let buffers = codecs
            .iter()
            .zip(&packed)
            .map(|(&codec, bytes)| upload_expert_test_matrix(&device, bytes, codec, &shape))
            .collect::<Vec<_>>();
        let strides = codecs
            .iter()
            .map(|codec| n * (k / codec.block_values() as usize) * codec.block_bytes() as usize)
            .collect::<Vec<_>>();
        assert_ne!(strides[0], strides[1]);
        assert_ne!(strides[0], strides[2]);
        assert_ne!(strides[1], strides[2]);

        for rows in [1_usize, 33] {
            let input_values = random(rows * k, 0.3);
            let ids_values = (0..rows)
                .map(|row| (row % experts) as u32)
                .collect::<Vec<_>>();
            let mut input =
                alloc(&device, DType::F32, vec![rows, k], "mixed expert input").unwrap();
            input
                .as_mut_slice::<f32>()
                .unwrap()
                .copy_from_slice(&input_values);
            let mut ids = alloc(&device, DType::U32, vec![rows], "mixed expert IDs").unwrap();
            ids.as_mut_slice::<u32>()
                .unwrap()
                .copy_from_slice(&ids_values);
            let outputs = codecs
                .iter()
                .map(|_| alloc(&device, DType::F32, vec![rows, n], "mixed expert output").unwrap())
                .collect::<Vec<_>>();

            let activation_epoch = ctx.activation_epoch();
            let (executor, registry) = ctx.split();
            let mut session = executor.begin().unwrap();
            for index in 0..codecs.len() {
                let weight = RawMatrixRef {
                    buffer: &buffers[index],
                    ggml_type: codecs[index],
                    shape: &shape,
                };
                expert_matmul(
                    &mut session,
                    registry,
                    &device,
                    &input,
                    &weight,
                    &ids,
                    &outputs[index],
                    rows,
                    1,
                    experts,
                    n,
                    k,
                    if rows == 1 {
                        ExpertMatmulRoute::ForceMv
                    } else {
                        ExpertMatmulRoute::Auto
                    },
                    None,
                    activation_epoch,
                    DenseMatmulIdMultiplicity::DistinctPerToken,
                    match index {
                        0 => DenseExpertScratchSlot::Gate,
                        1 => DenseExpertScratchSlot::Up,
                        _ => DenseExpertScratchSlot::Down,
                    },
                    "mixed-codec fallback projection",
                )
                .unwrap();
            }
            session.finish().unwrap();

            for index in 0..codecs.len() {
                let expected = expert_test_oracle(
                    &input_values,
                    &ids_values,
                    &packed[index],
                    codecs[index],
                    experts,
                    rows,
                    n,
                    k,
                );
                let actual = outputs[index].as_slice::<f32>().unwrap();
                assert_expert_test_output(
                    &format!("DeepSeek {:?} M={rows}", codecs[index]),
                    actual,
                    &expected,
                );
            }
        }
    }

    #[test]
    fn raw_matmul_consumes_native_bf16_without_a_shadow_weight() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let mut ctx = GpuContext::new().unwrap();
        let device = ctx.device().clone();
        let mut input = alloc(&device, DType::F32, vec![2, 4], "BF16 test input").unwrap();
        input
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&[0.5, -1.0, 2.0, 0.25, -2.0, 0.75, 1.5, -0.5]);
        let mut weight = alloc(&device, DType::BF16, vec![3, 4], "BF16 test weight").unwrap();
        let weights = [
            1.0, 0.5, -0.25, 2.0, -1.5, 0.25, 1.0, 0.5, 0.125, -2.0, 0.75, 1.0,
        ];
        for (dst, value) in weight
            .as_mut_slice::<half::bf16>()
            .unwrap()
            .iter_mut()
            .zip(weights)
        {
            *dst = half::bf16::from_f32(value);
        }
        let output = alloc(&device, DType::F32, vec![2, 3], "BF16 test output").unwrap();
        ctx.activate_native_bf16_dense(&[NativeBf16Matrix::unbatched_through(
            "DeepSeek BF16 test projection",
            &weight,
            3,
            4,
            2,
        )
        .unwrap()])
            .unwrap();
        let shape = [3, 4];
        let weight_ref = RawMatrixRef {
            buffer: &weight,
            ggml_type: GgmlType::BF16,
            shape: &shape,
        };
        let (executor, registry) = ctx.split();
        let mut session = executor.begin().unwrap();
        raw_matmul(
            &mut session,
            registry,
            &device,
            &input,
            &weight_ref,
            &output,
            2,
            3,
            4,
            "native BF16",
        )
        .unwrap();
        session.finish().unwrap();

        let expected: Vec<f32> = input
            .as_slice::<f32>()
            .unwrap()
            .chunks_exact(4)
            .flat_map(|row| {
                weights
                    .chunks_exact(4)
                    .map(move |w| row.iter().zip(w).map(|(x, y)| x * y).sum::<f32>())
            })
            .collect();
        let actual = output.as_slice::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "native BF16 output[{index}]={actual}, expected {expected}"
            );
        }
        assert_eq!(weight.dtype(), DType::BF16);
        assert_eq!(
            weight.data_byte_len(),
            weights.len() * DType::BF16.size_of()
        );
    }

    #[test]
    fn grouped_output_a_native_bf16_matches_independent_groups_for_decode_and_prefill() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let mut ctx = GpuContext::new().unwrap();
        let device = ctx.device().clone();
        let groups = 2usize;
        let rank = 3usize;
        let heads = 2usize;
        let head_dim = 4usize;
        let group_width = heads * head_dim / groups;
        let rows = 3usize;
        let weight_values = [
            1.0, 0.5, -0.25, 2.0, -1.5, 0.25, 1.0, 0.5, 0.125, -2.0, 0.75, 1.0, -0.5, 1.25, 0.5,
            -1.0, 2.0, -0.25, 0.75, 0.5, 1.0, 1.5, -0.5, 0.25,
        ];
        let mut weight = alloc(
            &device,
            DType::BF16,
            vec![groups * rank, group_width],
            "grouped BF16 test weight",
        )
        .unwrap();
        for (dst, value) in weight
            .as_mut_slice::<half::bf16>()
            .unwrap()
            .iter_mut()
            .zip(weight_values)
        {
            *dst = half::bf16::from_f32(value);
        }
        let matrix = NativeBf16Matrix {
            label: "DeepSeek grouped BF16 test projection",
            weight: &weight,
            n: rank as u32,
            k: group_width as u32,
            src0_batch: groups as u32,
            src1_batch: groups as u32,
            reachable_row_mask: 1 | (1 << (rows - 1)),
        };
        ctx.activate_native_bf16_dense(&[matrix]).unwrap();
        let shape = [groups * rank, group_width];
        let weight_ref = RawMatrixRef {
            buffer: &weight,
            ggml_type: GgmlType::BF16,
            shape: &shape,
        };

        let input_values = [
            0.5, -1.0, 2.0, 0.25, -2.0, 0.75, 1.5, -0.5, 1.0, 0.25, -0.75, 2.0, 0.5, 1.25, -1.5,
            0.75, -0.25, 2.0, 0.5, -1.0, 1.5, -0.5, 0.25, 1.0,
        ];
        let reference = |row: usize| -> Vec<f32> {
            (0..groups)
                .flat_map(|group| {
                    (0..rank).map(move |out| {
                        let input_base = (row * groups + group) * group_width;
                        let weight_base = (group * rank + out) * group_width;
                        (0..group_width)
                            .map(|column| {
                                input_values[input_base + column]
                                    * weight_values[weight_base + column]
                            })
                            .sum::<f32>()
                    })
                })
                .collect()
        };

        let mut decode_input = alloc(
            &device,
            DType::F32,
            vec![1, groups, group_width],
            "grouped BF16 decode input",
        )
        .unwrap();
        decode_input
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&input_values[..groups * group_width]);
        let decode_output = alloc(
            &device,
            DType::F32,
            vec![1, groups, rank],
            "grouped BF16 decode output",
        )
        .unwrap();
        {
            let (executor, registry) = ctx.split();
            let mut session = executor.begin().unwrap();
            grouped_output_a(
                &mut session,
                registry,
                &device,
                &decode_input,
                &weight_ref,
                &decode_output,
                groups,
                rank,
                heads,
                head_dim,
            )
            .unwrap();
            session.finish().unwrap();
        }
        assert_eq!(decode_output.as_slice::<f32>().unwrap(), reference(0));

        let mut prefill_input = alloc(
            &device,
            DType::F32,
            vec![rows, groups, group_width],
            "grouped BF16 prefill input",
        )
        .unwrap();
        prefill_input
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&input_values);
        let prefill_output = alloc(
            &device,
            DType::F32,
            vec![rows, groups, rank],
            "grouped BF16 prefill output",
        )
        .unwrap();
        let arena =
            BatchedGroupedOutputArena::new(&device, rows, groups, group_width, rank).unwrap();
        {
            let (executor, registry) = ctx.split();
            let mut session = executor.begin().unwrap();
            grouped_output_a_batched(
                &mut session,
                registry,
                &device,
                &prefill_input,
                &weight_ref,
                &prefill_output,
                &arena,
                rows,
                groups,
                rank,
                heads,
                head_dim,
            )
            .unwrap();
            session.finish().unwrap();
        }
        let expected: Vec<_> = (0..rows).flat_map(reference).collect();
        assert_eq!(prefill_output.as_slice::<f32>().unwrap(), expected);
        assert_eq!(weight.dtype(), DType::BF16);
    }

    #[test]
    fn grouped_output_a_native_q5_0_matches_dequantized_oracle_for_mv_and_strided_mm() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let mut ctx = GpuContext::new().unwrap();
        let device = ctx.device().clone();
        let groups = 2usize;
        let rank = 3usize;
        let group_width = 32usize;
        let heads = groups;
        let head_dim = group_width;
        let rows = MM_ROUTING_THRESHOLD as usize + 1;
        let source_weight = (0..groups * rank * group_width)
            .map(|index| ((index * 29 % 97) as f32 - 48.0) / 53.0)
            .collect::<Vec<_>>();
        let packed =
            crate::quantize::ggml_quants::q5_0::quantize(&source_weight, group_width, None);
        let mut dequantized = vec![0.0_f32; source_weight.len()];
        mlx_native::gguf::test_only_dequantize(&packed, GgmlType::Q5_0, &mut dequantized).unwrap();
        let mut weight = alloc(
            &device,
            DType::U8,
            vec![packed.len()],
            "grouped Q5_0 test weight",
        )
        .unwrap();
        weight
            .as_mut_slice::<u8>()
            .unwrap()
            .copy_from_slice(&packed);
        let shape = [groups * rank, group_width];
        let weight_ref = RawMatrixRef {
            buffer: &weight,
            ggml_type: GgmlType::Q5_0,
            shape: &shape,
        };
        let input_values = (0..rows * groups * group_width)
            .map(|index| ((index * 17 % 71) as f32 - 35.0) / 41.0)
            .collect::<Vec<_>>();
        let reference = |row: usize| -> Vec<f32> {
            (0..groups)
                .flat_map(|group| {
                    (0..rank).map({
                        let input_values = &input_values;
                        let dequantized = &dequantized;
                        move |out| {
                            let input_base = (row * groups + group) * group_width;
                            let weight_base = (group * rank + out) * group_width;
                            (0..group_width)
                                .map(|column| {
                                    input_values[input_base + column]
                                        * dequantized[weight_base + column]
                                })
                                .sum::<f32>()
                        }
                    })
                })
                .collect()
        };
        let assert_close = |actual: &[f32], expected: &[f32]| {
            assert_eq!(actual.len(), expected.len());
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 5e-3,
                    "native Q5_0 output[{index}]={actual}, expected {expected}"
                );
            }
        };

        let mut decode_input = alloc(
            &device,
            DType::F32,
            vec![1, groups, group_width],
            "grouped Q5_0 decode input",
        )
        .unwrap();
        decode_input
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&input_values[..groups * group_width]);
        let decode_output = alloc(
            &device,
            DType::F32,
            vec![1, groups, rank],
            "grouped Q5_0 decode output",
        )
        .unwrap();
        {
            let (executor, registry) = ctx.split();
            let mut session = executor.begin().unwrap();
            grouped_output_a(
                &mut session,
                registry,
                &device,
                &decode_input,
                &weight_ref,
                &decode_output,
                groups,
                rank,
                heads,
                head_dim,
            )
            .unwrap();
            session.finish().unwrap();
        }
        assert_close(decode_output.as_slice::<f32>().unwrap(), &reference(0));

        let mut prefill_input = alloc(
            &device,
            DType::F32,
            vec![rows, groups, group_width],
            "grouped Q5_0 prefill input",
        )
        .unwrap();
        prefill_input
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&input_values);
        let prefill_output = alloc(
            &device,
            DType::F32,
            vec![rows, groups, rank],
            "grouped Q5_0 prefill output",
        )
        .unwrap();
        let arena =
            BatchedGroupedOutputArena::new(&device, rows, groups, group_width, rank).unwrap();
        {
            let (executor, registry) = ctx.split();
            let mut session = executor.begin().unwrap();
            grouped_output_a_batched(
                &mut session,
                registry,
                &device,
                &prefill_input,
                &weight_ref,
                &prefill_output,
                &arena,
                rows,
                groups,
                rank,
                heads,
                head_dim,
            )
            .unwrap();
            session.finish().unwrap();
        }
        let expected = (0..rows).flat_map(reference).collect::<Vec<_>>();
        assert_close(prefill_output.as_slice::<f32>().unwrap(), &expected);
        assert_eq!(weight.dtype(), DType::U8);
        assert_eq!(weight.data_byte_len(), packed.len());
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
    fn grouped_prefill_keeps_host_inputs_unique_while_recycling_gpu_scratch() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let ctx = GpuContext::new().unwrap();
        let device = ctx.device().clone();
        release_prefill_scratch();

        begin_prefill_submission_inputs();
        begin_prefill_pool_layer();
        let first_input =
            alloc_host_input(&device, DType::U32, vec![257], "first grouped input").unwrap();
        let first_input_ptr = first_input.contents_ptr();
        let first_scratch = alloc(&device, DType::F32, vec![257], "first grouped scratch").unwrap();
        let first_scratch_ptr = first_scratch.contents_ptr();
        drop((first_input, first_scratch));
        end_prefill_pool_layer();

        begin_prefill_pool_layer();
        let second_input =
            alloc_host_input(&device, DType::U32, vec![257], "second grouped input").unwrap();
        let second_input_ptr = second_input.contents_ptr();
        let second_scratch =
            alloc(&device, DType::F32, vec![257], "second grouped scratch").unwrap();
        let second_scratch_ptr = second_scratch.contents_ptr();
        drop((second_input, second_scratch));
        end_prefill_pool_layer();

        assert_ne!(second_input_ptr, first_input_ptr);
        assert_eq!(second_scratch_ptr, first_scratch_ptr);
        end_prefill_submission_inputs();

        begin_prefill_submission_inputs();
        begin_prefill_pool_layer();
        let reused_input =
            alloc_host_input(&device, DType::U32, vec![257], "reused grouped input").unwrap();
        assert_eq!(reused_input.contents_ptr(), second_input_ptr);
        drop(reused_input);
        end_prefill_pool_layer();
        end_prefill_submission_inputs();
        release_prefill_scratch();
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
