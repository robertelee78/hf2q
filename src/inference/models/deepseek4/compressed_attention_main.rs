//! Main ratio-four/ratio-128 learned KV compression graph.

use anyhow::{Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::cache::LayerCache;
use super::compressor_tail::{encode_main_tail, CompressorTail};
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
    encode_main_tail(session, registry, device, cfg, ratio, rows, position,
        compressed_write_start, compressed_count, cache, cache_capacity, kv_state, score_state,
        &CompressorTail { kv: &arena.kv, score: &arena.score, output: &arena.output,
            rope: &arena.rope, ape: weights.ape, norm: weights.norm },
        compressed_positions, frequencies)
}
