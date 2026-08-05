//! Main ratio-four/ratio-128 learned KV compression graph.

use anyhow::{Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_activation_quant::{
    dispatch_deepseek_mxfp8_fake_quant_bf16, DeepSeekMxfp8Params,
};
use mlx_native::ops::deepseek_compressor::{
    dispatch_deepseek_compressor, DeepSeekCompressorParams,
};
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_bf16, DeepSeekTailRopeParams,
};
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::cache::LayerCache;
use super::compressed_attention_weights::CompressorWeightsRef;
use super::forward_support::{alloc, raw_matmul};
use super::Deepseek4Config;

pub(super) struct MainCompressorArena {
    kv: MlxBuffer,
    score: MlxBuffer,
    output: MlxBuffer,
    rope: MlxBuffer,
}

impl MainCompressorArena {
    pub(super) fn new(
        device: &MlxDevice,
        cfg: &Deepseek4Config,
        ratio: u32,
        rows: usize,
        compressed_count: usize,
    ) -> Result<Self> {
        let dim = cfg.head_dim as usize;
        let coefficient = usize::from(ratio == 4) + 1;
        let projected = coefficient * dim;
        Ok(Self {
            kv: alloc(
                device,
                DType::F32,
                vec![1, rows, projected],
                "main compressor KV",
            )?,
            score: alloc(
                device,
                DType::F32,
                vec![1, rows, projected],
                "main compressor score",
            )?,
            output: alloc(
                device,
                DType::BF16,
                vec![1, compressed_count.max(1), dim],
                "main compressor output",
            )?,
            rope: alloc(
                device,
                DType::BF16,
                vec![1, compressed_count.max(1), 1, dim],
                "main compressor rotated output",
            )?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_main_compressor(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    ratio: u32,
    rows: usize,
    position: usize,
    compressed_write_start: usize,
    compressed_count: usize,
    layer_cache: &LayerCache,
    attn_norm: &MlxBuffer,
    compressed_positions: &MlxBuffer,
    frequencies: &MlxBuffer,
    weights: &CompressorWeightsRef<'_>,
    arena: &MainCompressorArena,
) -> Result<()> {
    let hidden = cfg.hidden_size as usize;
    let dim = cfg.head_dim as usize;
    let coefficient = usize::from(ratio == 4) + 1;
    let projected = coefficient * dim;
    let cache = layer_cache
        .compressed_kv
        .as_ref()
        .context("compressed attention cache is missing")?;
    let cache_capacity = cache.shape()[0];
    let kv_state = layer_cache
        .main_kv_state
        .as_ref()
        .context("main compressor KV state is missing")?;
    let score_state = layer_cache
        .main_score_state
        .as_ref()
        .context("main compressor score state is missing")?;
    raw_matmul(
        session,
        registry,
        device,
        attn_norm,
        &weights.kv,
        &arena.kv,
        rows,
        projected,
        hidden,
        "main compressor KV",
    )?;
    raw_matmul(
        session,
        registry,
        device,
        attn_norm,
        &weights.gate,
        &arena.score,
        rows,
        projected,
        hidden,
        "main compressor gate",
    )?;
    session.barrier_between(
        &[
            &arena.kv,
            &arena.score,
            weights.ape,
            weights.norm,
            kv_state,
            score_state,
        ],
        &[kv_state, score_state, &arena.output],
    );
    dispatch_deepseek_compressor(
        session.encoder_mut(),
        registry,
        device,
        &arena.kv,
        &arena.score,
        weights.ape,
        weights.norm,
        kv_state,
        score_state,
        &arena.output,
        &arena.output,
        &DeepSeekCompressorParams {
            batch: 1,
            seq_len: rows as u32,
            start_pos: position as u32,
            ratio,
            head_dim: dim as u32,
            cache_len: cache_capacity.max(1) as u32,
            epsilon: cfg.rms_norm_eps,
            write_cache: 0,
        },
    )?;
    if compressed_count == 0 {
        return Ok(());
    }
    let rope_input = arena.output.with_shape(vec![1, compressed_count, 1, dim])?;
    session.barrier_between(
        &[&rope_input, compressed_positions, frequencies],
        &[&arena.rope],
    );
    dispatch_deepseek_tail_rope_bf16(
        session.encoder_mut(),
        registry,
        device,
        &rope_input,
        compressed_positions,
        frequencies,
        &arena.rope,
        &DeepSeekTailRopeParams {
            batch: 1,
            seq_len: compressed_count as u32,
            heads: 1,
            head_dim: dim as u32,
            rope_dim: cfg.rope_head_dim,
            inverse: 0,
        },
    )?;
    session.barrier_between(&[&arena.rope], &[&arena.rope]);
    dispatch_deepseek_mxfp8_fake_quant_bf16(
        session.encoder_mut(),
        registry,
        device,
        &arena.rope,
        &DeepSeekMxfp8Params {
            rows: compressed_count as u32,
            row_width: dim as u32,
            quantized_width: (dim - cfg.rope_head_dim as usize) as u32,
            block_size: 64,
        },
    )?;
    session.barrier_between(&[&arena.rope], &[cache]);
    dispatch_kv_cache_copy(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        &arena.rope,
        cache,
        compressed_write_start as u32,
        dim as u32,
        compressed_count as u32,
        cache_capacity as u32,
        false,
    )?;
    Ok(())
}
