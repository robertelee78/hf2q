//! Per-row, cache-dependent attention cores for joint decode and rows steps.
//!
//! Every function here encodes exactly the one-token decode branch of
//! `forward_uncompressed_attention_rows` / `forward_compressed_attention_rows`
//! for one row of a multi-row step: the row's window write, compressor and
//! indexer recurrences, and sparse attention read only that row's cache at
//! that row's position. Inputs are zero-copy row views of the batched
//! projection buffers, so the dispatched arithmetic is the one-row graph's.
//! A joint step's rows own independent caches; a rows step's rows are
//! consecutive positions of one cache ([`RowCaches`]).

use anyhow::{ensure, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_sparse_attention::{
    dispatch_deepseek_sparse_attention, dispatch_deepseek_sparse_attention_flash_decode,
    dispatch_deepseek_sparse_attention_with_scratch, DeepSeekSparseAttentionParams,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::attention::window_indices;
use super::cache::{CacheStep, Deepseek4Cache, LayerCache, LayerCacheStep};
use super::cache_window_write::encode_window_write;
use super::compressed_attention::{
    sparse_attention_scratch_shape, SparseFlashDecodeArena, SPARSE_FLASH_MIN_TOP_K,
};
use super::compressed_attention_indexer::layout::prepare_layout;
use super::compressed_attention_indexer::selection::SelectionRequest;
use super::compressed_attention_indexer::{encode_ratio_four_indexer, RatioFourIndexerArena};
use super::compressed_attention_main::{encode_main_compressor, MainCompressorArena};
use super::compressed_attention_weights::CompressedAttentionWeightsRef;
use super::forward_support::{alloc, alloc_host_input};
use super::Deepseek4Config;

/// Row views of one joint row's batched projections and its outputs.
pub(super) struct JointRowInputs {
    pub q_rope: MlxBuffer,
    pub kv_rope: MlxBuffer,
    pub attention: MlxBuffer,
    pub attn_norm: MlxBuffer,
    pub q_a_norm: MlxBuffer,
    pub position: MlxBuffer,
}

/// Host-prepared state for one row of an uncompressed (sliding-window) layer.
pub(super) struct WindowRowCore {
    window_capacity: usize,
    index_width: usize,
    indices: MlxBuffer,
}

enum SparseDecodeWorkspace {
    Flash(SparseFlashDecodeArena),
    Logits(MlxBuffer),
}

/// Host-prepared state for one row of a ratio-4 or ratio-128 layer.
pub(super) struct CompressedRowCore {
    ratio: u32,
    position: usize,
    graft: usize,
    compressed_count: usize,
    window_capacity: usize,
    physical_kv_len: usize,
    attention_width: usize,
    compressed_positions: MlxBuffer,
    indices: MlxBuffer,
    attention_indices: MlxBuffer,
    selection: Option<SelectionRequest>,
    main_compressor: MainCompressorArena,
    indexer: Option<RatioFourIndexerArena>,
    sparse: SparseDecodeWorkspace,
}

/// One row's planned cache-dependent core.
pub(super) enum JointRowCore {
    Window(WindowRowCore),
    Compressed(CompressedRowCore),
}

/// How the rows of one multi-row step share caches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RowCaches {
    /// Joint decode: one row per independent cache.
    Independent,
    /// Rows step: consecutive positions of one cache. Row `i + 1`'s window
    /// write overwrites the slot of position `p + i + 1 - window`, which row
    /// `i` still attends to, and its compressor and indexer read row `i`'s
    /// recurrent state; so the cores run in row order, each behind a full
    /// barrier, exactly as consecutive one-token steps order them.
    Sequential,
}

/// Plan every row's core for `layer` (host only).
pub(super) fn plan_row_cores(
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    caches: &[&Deepseek4Cache],
    steps: &[CacheStep],
    layer: usize,
    ratio: u32,
) -> Result<Vec<JointRowCore>> {
    ensure!(
        caches.len() == steps.len(),
        "DeepSeek-V4 row cores need one step per cache row, got {} caches and {} steps",
        caches.len(),
        steps.len()
    );
    let head_dim = cfg.head_dim as usize;
    caches
        .iter()
        .zip(steps)
        .enumerate()
        .map(|(row, (&cache, step))| {
            let planned = if ratio == 0 {
                plan_window_row(device, cache, layer, step.position, head_dim)
                    .map(JointRowCore::Window)
            } else {
                let layer_step = step
                    .layers
                    .get(layer)
                    .with_context(|| format!("missing layer-{layer} row cache step"))?;
                plan_compressed_row(device, cfg, cache, layer, ratio, step.position, layer_step)
                    .map(JointRowCore::Compressed)
            };
            planned.with_context(|| format!("plan DeepSeek-V4 layer-{layer} row {row} core"))
        })
        .collect()
}

/// Encode every row's core for `layer`, in row order.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_row_cores(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    caches: &[&Deepseek4Cache],
    steps: &[CacheStep],
    layer: usize,
    cores: &[JointRowCore],
    inputs: &[JointRowInputs],
    sinks: &MlxBuffer,
    compressed_weights: Option<&CompressedAttentionWeightsRef<'_>>,
    frequencies: &MlxBuffer,
    order: RowCaches,
) -> Result<()> {
    ensure!(
        caches.len() == steps.len() && cores.len() == steps.len() && inputs.len() == steps.len(),
        "DeepSeek-V4 layer-{layer} row cores are misaligned"
    );
    for (row, ((&cache, step), (core, inputs))) in caches
        .iter()
        .zip(steps)
        .zip(cores.iter().zip(inputs))
        .enumerate()
    {
        if order == RowCaches::Sequential && row > 0 {
            session.barrier();
        }
        let layer_cache = layer_cache(cache, layer)?;
        let layer_step = step
            .layers
            .get(layer)
            .with_context(|| format!("missing layer-{layer} row-{row} cache step"))?;
        let encoded = match core {
            JointRowCore::Window(window) => encode_window_row(
                session,
                registry,
                device,
                cfg,
                layer_cache,
                layer_step,
                window,
                sinks,
                inputs,
            ),
            JointRowCore::Compressed(compressed) => encode_compressed_row(
                session,
                registry,
                device,
                cfg,
                layer_cache,
                layer_step,
                compressed,
                compressed_weights.context("DeepSeek-V4 compressed row weights missing")?,
                frequencies,
                inputs,
            ),
        };
        encoded.with_context(|| format!("encode DeepSeek-V4 layer-{layer} row {row} core"))?;
    }
    Ok(())
}
#[cfg(test)]
impl CompressedRowCore {
    /// The row's attention index storage (window, graft and indexer slots).
    pub(super) fn index_storage(&self) -> &MlxBuffer {
        &self.indices
    }
}

fn layer_cache<'a>(cache: &'a Deepseek4Cache, layer: usize) -> Result<&'a LayerCache> {
    cache
        .layers()
        .get(layer)
        .with_context(|| format!("missing layer-{layer} joint decode cache"))
}

pub(super) fn plan_window_row(
    device: &MlxDevice,
    cache: &Deepseek4Cache,
    layer: usize,
    position: usize,
    head_dim: usize,
) -> Result<WindowRowCore> {
    let layer_cache = layer_cache(cache, layer)?;
    let window_capacity = layer_cache.window_kv.shape()[0];
    ensure!(
        layer_cache.window_kv.shape() == [window_capacity, head_dim],
        "DeepSeek-V4 layer-{layer} window cache shape drift"
    );
    let index_rows = window_indices(window_capacity, 1, position)?;
    let row = index_rows
        .first()
        .context("DeepSeek-V4 window index row missing")?;
    let index_width = row.len();
    let mut indices = alloc_host_input(
        device,
        DType::I32,
        vec![1, 1, index_width],
        "joint attention indices",
    )?;
    indices.as_logical_mut_slice::<i32>()?.copy_from_slice(row);
    Ok(WindowRowCore {
        window_capacity,
        index_width,
        indices,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_window_row(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    layer_cache: &LayerCache,
    step: &LayerCacheStep,
    core: &WindowRowCore,
    sinks: &MlxBuffer,
    inputs: &JointRowInputs,
) -> Result<()> {
    let heads = cfg.num_attention_heads as usize;
    let head_dim = cfg.head_dim as usize;
    encode_window_write(
        session,
        registry,
        device,
        layer_cache,
        &inputs.kv_rope,
        &[&inputs.kv_rope],
        None,
        Some(step),
        1,
        core.window_capacity,
        head_dim,
    )?;
    let cache_view = layer_cache
        .window_kv
        .with_shape(vec![1, core.window_capacity, head_dim])?;
    session.barrier_between(
        &[&inputs.q_rope, &cache_view, sinks, &core.indices],
        &[&inputs.attention],
    );
    dispatch_deepseek_sparse_attention(
        session.encoder_mut(),
        registry,
        device,
        &inputs.q_rope,
        &cache_view,
        sinks,
        &core.indices,
        &inputs.attention,
        &DeepSeekSparseAttentionParams {
            batch: 1,
            query_len: 1,
            kv_len: core.window_capacity as u32,
            top_k: core.index_width as u32,
            heads: heads as u32,
            head_dim: head_dim as u32,
            scale: 1.0 / (head_dim as f32).sqrt(),
        },
    )?;
    Ok(())
}

pub(super) fn plan_compressed_row(
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    cache: &Deepseek4Cache,
    layer: usize,
    ratio: u32,
    position: usize,
    step: &LayerCacheStep,
) -> Result<CompressedRowCore> {
    ensure!(
        matches!(ratio, 4 | 128),
        "DeepSeek-V4 layer {layer} is not a compressed attention layer"
    );
    let head_dim = cfg.head_dim as usize;
    let graft = cache.graft_slots();
    let layer_cache = layer_cache(cache, layer)?;
    let window_capacity = layer_cache.window_kv.shape()[0];
    let compressed_capacity = layer_cache
        .compressed_kv
        .as_ref()
        .context("compressed KV cache is missing")?
        .shape()[0];
    ensure!(
        window_capacity == cfg.sliding_window as usize
            && layer_cache.attention_kv.shape()
                == [window_capacity + compressed_capacity, head_dim],
        "DeepSeek-V4 layer {layer} compressed attention cache shape drift"
    );
    let compressed_count = usize::from(step.compressed_write_slot.is_some());
    let valid_compressed = step.compressed_valid_after;
    let layout = prepare_layout(
        ratio,
        1,
        position,
        graft,
        window_capacity,
        valid_compressed,
        cfg.index_top_k as usize,
        false,
    )?;
    let selection = (ratio == 4)
        .then(|| {
            SelectionRequest::new(
                cfg,
                1,
                position,
                valid_compressed,
                layout
                    .compressed_attention_offset
                    .checked_add(graft)
                    .context("DeepSeek-V4 indexer offset overflow")?,
                layout.storage_stride,
                layout
                    .indexer_output_offset
                    .context("ratio-four output offset missing")?,
            )
        })
        .transpose()?;
    let main_compressor = MainCompressorArena::new(device, cfg, ratio, 1, compressed_count)?;
    let indexer = selection
        .as_ref()
        .map(|selection| RatioFourIndexerArena::new(device, cfg, 1, compressed_count, selection))
        .transpose()?;
    let mut compressed_positions = alloc_host_input(
        device,
        DType::U32,
        vec![compressed_count.max(1)],
        "joint compressed block positions",
    )?;
    if compressed_count == 1 {
        compressed_positions.as_logical_mut_slice::<u32>()?[0] =
            u32::try_from(position + 1 - ratio as usize)
                .context("DeepSeek-V4 compressed block position exceeds u32")?;
    }
    let storage_stride = layout.storage_stride;
    let attention_width = layout.attention_width;
    let mut indices = alloc_host_input(
        device,
        DType::I32,
        vec![1, 1, storage_stride],
        "joint compressed attention indices",
    )?;
    indices
        .as_logical_mut_slice::<i32>()?
        .copy_from_slice(&layout.index_storage);
    let attention_indices = if storage_stride == attention_width {
        indices.with_shape(vec![1, 1, attention_width])?
    } else {
        indices
            .slice_view(0, attention_width)
            .with_shape(vec![1, 1, attention_width])?
    };
    let sparse = if ratio == 4 && attention_width >= SPARSE_FLASH_MIN_TOP_K {
        SparseDecodeWorkspace::Flash(SparseFlashDecodeArena::new(
            device,
            attention_width,
            head_dim,
        )?)
    } else {
        SparseDecodeWorkspace::Logits(alloc(
            device,
            DType::F32,
            sparse_attention_scratch_shape(1, attention_width),
            "joint compressed sparse attention logits",
        )?)
    };
    Ok(CompressedRowCore {
        ratio,
        position,
        graft,
        compressed_count,
        window_capacity,
        physical_kv_len: window_capacity + compressed_capacity,
        attention_width,
        compressed_positions,
        indices,
        attention_indices,
        selection,
        main_compressor,
        indexer,
        sparse,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_compressed_row(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    layer_cache: &LayerCache,
    step: &LayerCacheStep,
    core: &CompressedRowCore,
    weights: &CompressedAttentionWeightsRef<'_>,
    frequencies: &MlxBuffer,
    inputs: &JointRowInputs,
) -> Result<()> {
    let heads = cfg.num_attention_heads as usize;
    let head_dim = cfg.head_dim as usize;
    encode_window_write(
        session,
        registry,
        device,
        layer_cache,
        &inputs.kv_rope,
        &[&inputs.kv_rope],
        None,
        Some(step),
        1,
        core.window_capacity,
        head_dim,
    )?;
    // ADR-059 #308: real compressed rows land after the reserved graft rows.
    encode_main_compressor(
        session,
        registry,
        device,
        cfg,
        core.ratio,
        1,
        core.position,
        step.compressed_write_slot.unwrap_or(0) + core.graft,
        core.compressed_count,
        layer_cache,
        &inputs.attn_norm,
        &core.compressed_positions,
        frequencies,
        &weights.compressor,
        &core.main_compressor,
    )?;
    if let (Some(indexer_weights), Some(indexer_arena), Some(selection)) = (
        weights.indexer.as_ref(),
        core.indexer.as_ref(),
        core.selection.as_ref(),
    ) {
        encode_ratio_four_indexer(
            session,
            registry,
            device,
            cfg,
            1,
            core.position,
            step.indexer_write_slot.unwrap_or(0),
            core.compressed_count,
            selection,
            layer_cache,
            &inputs.attn_norm,
            &inputs.q_a_norm,
            &inputs.position,
            &core.compressed_positions,
            frequencies,
            indexer_weights,
            indexer_arena,
            &core.indices,
        )?;
    }
    let cache_view =
        layer_cache
            .attention_kv
            .with_shape(vec![1, core.physical_kv_len, head_dim])?;
    let params = DeepSeekSparseAttentionParams {
        batch: 1,
        query_len: 1,
        kv_len: core.physical_kv_len as u32,
        top_k: core.attention_width as u32,
        heads: heads as u32,
        head_dim: head_dim as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
    };
    let sinks = weights.attention.sinks;
    match &core.sparse {
        SparseDecodeWorkspace::Flash(flash) => {
            session.barrier_between(
                &[&inputs.q_rope, &cache_view, sinks, &core.attention_indices],
                &[
                    &flash.gathered_kv,
                    &flash.mask,
                    &flash.invalid_global,
                    &flash.invalid_heads,
                    &inputs.attention,
                ],
            );
            dispatch_deepseek_sparse_attention_flash_decode(
                session.encoder_mut(),
                registry,
                device,
                &inputs.q_rope,
                &cache_view,
                sinks,
                &core.attention_indices,
                &flash.gathered_kv,
                &flash.mask,
                &flash.invalid_global,
                &flash.invalid_heads,
                &inputs.attention,
                &params,
            )?;
        }
        SparseDecodeWorkspace::Logits(logits) => {
            session.barrier_between(
                &[&inputs.q_rope, &cache_view, sinks, &core.attention_indices],
                &[&inputs.attention],
            );
            dispatch_deepseek_sparse_attention_with_scratch(
                session.encoder_mut(),
                registry,
                device,
                &inputs.q_rope,
                &cache_view,
                sinks,
                &core.attention_indices,
                logits,
                &inputs.attention,
                &params,
            )?;
        }
    }
    Ok(())
}
