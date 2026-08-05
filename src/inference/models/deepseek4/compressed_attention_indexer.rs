//! Ratio-four learned compressed-position selection.

use anyhow::{Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_activation_quant::dispatch_deepseek_hadamard_mxfp4_bf16;
use mlx_native::ops::deepseek_compressor::{
    dispatch_deepseek_compressor, DeepSeekCompressorParams,
};
use mlx_native::ops::deepseek_indexer::{
    dispatch_deepseek_indexer, DeepSeekIndexerParams, DEEPSEEK_INDEXER_TOP_K,
};
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_bf16, dispatch_deepseek_tail_rope_f32_to_bf16,
    DeepSeekTailRopeParams,
};
use mlx_native::ops::elementwise::scalar_mul_f32;
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::cache::{LayerCache, LayerCacheStep};
use super::compressed_attention_weights::IndexerWeightsRef;
use super::forward_support::{alloc, raw_matmul};
use super::Deepseek4Config;

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
}

impl RatioFourIndexerArena {
    pub(super) fn new(
        device: &MlxDevice,
        cfg: &Deepseek4Config,
        valid_compressed: usize,
    ) -> Result<Self> {
        let heads = cfg.index_num_heads as usize;
        let dim = cfg.index_head_dim as usize;
        let projected = 2 * dim;
        Ok(Self {
            query_f32: alloc(
                device,
                DType::F32,
                vec![1, 1, heads, dim],
                "index query f32",
            )?,
            query_bf16: alloc(
                device,
                DType::BF16,
                vec![1, 1, heads, dim],
                "index query bf16",
            )?,
            weights: alloc(device, DType::F32, vec![1, 1, heads], "index weights")?,
            scaled_weights: alloc(
                device,
                DType::F32,
                vec![1, 1, heads],
                "scaled index weights",
            )?,
            compressor_kv: alloc(
                device,
                DType::F32,
                vec![1, 1, projected],
                "index compressor KV",
            )?,
            compressor_score: alloc(
                device,
                DType::F32,
                vec![1, 1, projected],
                "index compressor score",
            )?,
            compressor_output: alloc(
                device,
                DType::BF16,
                vec![1, 1, dim],
                "index compressor output",
            )?,
            compressor_rope: alloc(
                device,
                DType::BF16,
                vec![1, 1, 1, dim],
                "index compressor rotated output",
            )?,
            score_scratch: alloc(
                device,
                DType::F32,
                vec![1, 1, valid_compressed.max(1)],
                "index score scratch",
            )?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_ratio_four_indexer(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    position: usize,
    layer_step: &LayerCacheStep,
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
        1,
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
        1,
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
        1,
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
        1,
        projected,
        hidden,
        "index compressor gate",
    )?;
    session.barrier_between(&[&arena.query_f32], &[&arena.query_bf16]);
    dispatch_deepseek_tail_rope_f32_to_bf16(
        session.encoder_mut(),
        registry,
        device,
        &arena.query_f32,
        positions,
        frequencies,
        &arena.query_bf16,
        &DeepSeekTailRopeParams {
            batch: 1,
            seq_len: 1,
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
        heads as u32,
    )?;
    session.barrier_between(&[&arena.weights], &[&arena.scaled_weights]);
    scalar_mul_f32(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        &arena.weights,
        &arena.scaled_weights,
        heads,
        1.0 / ((dim * heads) as f32).sqrt(),
    )?;
    session.barrier_between(
        &[
            &arena.compressor_kv,
            &arena.compressor_score,
            weights.compressor.ape,
            weights.compressor.norm,
            kv_state,
            score_state,
        ],
        &[kv_state, score_state, &arena.compressor_output],
    );
    dispatch_deepseek_compressor(
        session.encoder_mut(),
        registry,
        device,
        &arena.compressor_kv,
        &arena.compressor_score,
        weights.compressor.ape,
        weights.compressor.norm,
        kv_state,
        score_state,
        &arena.compressor_output,
        &arena.compressor_output,
        &DeepSeekCompressorParams {
            batch: 1,
            seq_len: 1,
            start_pos: position as u32,
            ratio: 4,
            head_dim: dim as u32,
            cache_len: cache_capacity.max(1) as u32,
            epsilon: cfg.rms_norm_eps,
            write_cache: 0,
        },
    )?;
    if let Some(slot) = layer_step.indexer_write_slot {
        session.barrier_between(
            &[&arena.compressor_output, compressed_positions, frequencies],
            &[&arena.compressor_rope],
        );
        dispatch_deepseek_tail_rope_bf16(
            session.encoder_mut(),
            registry,
            device,
            &arena.compressor_output,
            compressed_positions,
            frequencies,
            &arena.compressor_rope,
            &DeepSeekTailRopeParams {
                batch: 1,
                seq_len: 1,
                heads: 1,
                head_dim: dim as u32,
                rope_dim: cfg.rope_head_dim,
                inverse: 0,
            },
        )?;
        session.barrier_between(&[&arena.compressor_rope], &[&arena.compressor_rope]);
        dispatch_deepseek_hadamard_mxfp4_bf16(
            session.encoder_mut(),
            registry,
            device,
            &arena.compressor_rope,
            1,
        )?;
        session.barrier_between(&[&arena.compressor_rope], &[cache]);
        dispatch_kv_cache_copy(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            &arena.compressor_rope,
            cache,
            slot as u32,
            dim as u32,
            1,
            cache_capacity as u32,
            false,
        )?;
    }

    let valid = layer_step.indexer_valid_after;
    if valid > 0 {
        let cache_view = cache
            .slice_view(0, valid * dim)
            .with_shape(vec![1, valid, dim])?;
        let scratch = arena
            .score_scratch
            .slice_view(0, valid)
            .with_shape(vec![1, 1, valid])?;
        session.barrier_between(
            &[&arena.query_bf16, &cache_view, &arena.scaled_weights],
            &[&scratch, output_indices],
        );
        dispatch_deepseek_indexer(
            session.encoder_mut(),
            registry,
            device,
            &arena.query_bf16,
            &cache_view,
            &arena.scaled_weights,
            &scratch,
            output_indices,
            &DeepSeekIndexerParams {
                batch: 1,
                query_len: 1,
                kv_len: valid as u32,
                start_pos: position as u32,
                ratio: 4,
                heads: heads as u32,
                head_dim: dim as u32,
                top_k: DEEPSEEK_INDEXER_TOP_K as u32,
                offset: cfg.sliding_window as i32,
            },
        )?;
    }
    Ok(())
}
