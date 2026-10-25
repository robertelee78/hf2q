//! Ratio-four learned compressed-position selection.

use anyhow::{Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_activation_quant::dispatch_deepseek_hadamard_mxfp4_bf16;
use mlx_native::ops::deepseek_tail_rope::DeepSeekTailRopeParams;
use mlx_native::ops::elementwise::scalar_mul_f32;
use mlx_native::{KernelRegistry, MlxBuffer, MlxDevice};

use super::cache::LayerCache;
use super::compressed_attention_weights::IndexerWeightsRef;
use super::compressor_tail::{encode_indexer_tail, CompressorTail};
use super::forward_support::raw_matmul;
use super::prefill_annotations;
use super::Deepseek4Config;

pub(super) mod layout;
pub(super) mod plan;
pub(super) mod selection;
#[cfg(test)]
mod selection_tests;
use plan::RatioFourIndexerPlan;
use selection::{encode_selection, SelectionRequest};

pub(super) struct RatioFourIndexerArena {
    query_f32: MlxBuffer,
    query_bf16: MlxBuffer,
    weights: MlxBuffer,
    scaled_weights: MlxBuffer,
    compressor_kv: MlxBuffer,
    compressor_score: MlxBuffer,
    compressor_output: MlxBuffer,
    compressor_rope: MlxBuffer,
    score_scratch: MlxBuffer,
    topk_scratch: MlxBuffer,
}

impl RatioFourIndexerArena {
    pub(super) fn new(
        device: &MlxDevice,
        cfg: &Deepseek4Config,
        rows: usize,
        compressed_output_count: usize,
        selection: &SelectionRequest,
    ) -> Result<Self> {
        let [query_f32, query_bf16, weights, scaled_weights, compressor_kv, compressor_score, compressor_output, compressor_rope, score_scratch, topk_scratch] =
            RatioFourIndexerPlan::new(cfg, rows, compressed_output_count, selection)?
                .allocate_with(|buffer| buffer.allocate(device))?;
        Ok(Self {
            query_f32,
            query_bf16,
            weights,
            scaled_weights,
            compressor_kv,
            compressor_score,
            compressor_output,
            compressor_rope,
            score_scratch,
            topk_scratch,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_ratio_four_indexer(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    rows: usize,
    position: usize,
    compressed_write_start: usize,
    compressed_output_count: usize,
    selection: &SelectionRequest,
    layer_cache: &LayerCache,
    attn_norm: &MlxBuffer,
    query_rank: &MlxBuffer,
    positions: &MlxBuffer,
    compressed_positions: &MlxBuffer,
    frequencies: &MlxBuffer,
    weights: &IndexerWeightsRef<'_>,
    arena: &RatioFourIndexerArena,
    output_indices: &MlxBuffer,
) -> Result<()> {
    let hidden = cfg.hidden_size as usize;
    let q_rank = cfg.q_lora_rank as usize;
    let heads = cfg.index_num_heads as usize;
    let dim = cfg.index_head_dim as usize;
    let projected = 2 * dim;
    let cache = layer_cache
        .indexer_kv
        .as_ref()
        .context("ratio-four indexer cache is missing")?;
    let cache_capacity = cache.shape()[0];
    let kv_state = layer_cache
        .indexer_kv_state
        .as_ref()
        .context("ratio-four indexer KV state is missing")?;
    let score_state = layer_cache
        .indexer_score_state
        .as_ref()
        .context("ratio-four indexer score state is missing")?;

    raw_matmul(
        session,
        registry,
        device,
        query_rank,
        &weights.q_b,
        &arena.query_f32,
        rows,
        heads * dim,
        q_rank,
        "index query",
    )?;
    raw_matmul(
        session,
        registry,
        device,
        attn_norm,
        &weights.projection,
        &arena.weights,
        rows,
        heads,
        hidden,
        "index head weights",
    )?;
    raw_matmul(
        session,
        registry,
        device,
        attn_norm,
        &weights.compressor.kv,
        &arena.compressor_kv,
        rows,
        projected,
        hidden,
        "index compressor KV",
    )?;
    raw_matmul(
        session,
        registry,
        device,
        attn_norm,
        &weights.compressor.gate,
        &arena.compressor_score,
        rows,
        projected,
        hidden,
        "index compressor gate",
    )?;
    prefill_annotations::indexer_query_rope(
        session,
        registry,
        device,
        &arena.query_f32,
        positions,
        frequencies,
        &arena.query_bf16,
        &DeepSeekTailRopeParams {
            batch: 1,
            seq_len: rows as u32,
            heads: heads as u32,
            head_dim: dim as u32,
            rope_dim: cfg.rope_head_dim,
            inverse: 0,
        },
    )?;
    session.barrier_between(&[&arena.query_bf16], &[&arena.query_bf16]);
    dispatch_deepseek_hadamard_mxfp4_bf16(
        session.encoder_mut(),
        registry,
        device,
        &arena.query_bf16,
        (rows * heads) as u32,
    )?;
    session.barrier_between(&[&arena.weights], &[&arena.scaled_weights]);
    scalar_mul_f32(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        &arena.weights,
        &arena.scaled_weights,
        rows * heads,
        1.0 / ((dim * heads) as f32).sqrt(),
    )?;
    encode_indexer_tail(
        session,
        registry,
        device,
        cfg,
        rows,
        position,
        compressed_write_start,
        compressed_output_count,
        cache,
        cache_capacity,
        kv_state,
        score_state,
        &CompressorTail {
            kv: &arena.compressor_kv,
            score: &arena.compressor_score,
            output: &arena.compressor_output,
            rope: &arena.compressor_rope,
            ape: weights.compressor.ape,
            norm: weights.compressor.norm,
        },
        compressed_positions,
        frequencies,
    )?;

    encode_selection(
        session,
        registry,
        device,
        selection,
        &arena.query_bf16,
        cache,
        &arena.scaled_weights,
        &arena.score_scratch,
        &arena.topk_scratch,
        output_indices,
    )?;
    Ok(())
}
