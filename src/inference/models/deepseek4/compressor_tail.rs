//! Actual postprojection compressor/cache-write tails shared with the synthetic control.
use super::Deepseek4Config;
use anyhow::Result;
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_activation_quant::{
    dispatch_deepseek_hadamard_mxfp4_bf16, dispatch_deepseek_mxfp8_fake_quant_bf16,
    DeepSeekMxfp8Params,
};
use mlx_native::ops::deepseek_compressor::{
    dispatch_deepseek_compressor, DeepSeekCompressorParams,
};
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_bf16, DeepSeekTailRopeParams,
};
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy;
use mlx_native::{KernelRegistry, MlxBuffer, MlxDevice};

pub(super) struct CompressorTail<'a> {
    pub kv: &'a MlxBuffer,
    pub score: &'a MlxBuffer,
    pub output: &'a MlxBuffer,
    pub rope: &'a MlxBuffer,
    pub ape: &'a MlxBuffer,
    pub norm: &'a MlxBuffer,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_main_tail(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    ratio: u32,
    rows: usize,
    position: usize,
    compressed_write_start: usize,
    compressed_count: usize,
    cache: &MlxBuffer,
    cache_capacity: usize,
    kv_state: &MlxBuffer,
    score_state: &MlxBuffer,
    tail: &CompressorTail<'_>,
    compressed_positions: &MlxBuffer,
    frequencies: &MlxBuffer,
) -> Result<()> {
    let dim = cfg.head_dim as usize;
    session.barrier_between(
        &[
            tail.kv,
            tail.score,
            tail.ape,
            tail.norm,
            kv_state,
            score_state,
        ],
        &[kv_state, score_state, tail.output],
    );
    dispatch_deepseek_compressor(
        session.encoder_mut(),
        registry,
        device,
        tail.kv,
        tail.score,
        tail.ape,
        tail.norm,
        kv_state,
        score_state,
        tail.output,
        tail.output,
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
    let rope_input = tail.output.with_shape(vec![1, compressed_count, 1, dim])?;
    session.barrier_between(
        &[&rope_input, compressed_positions, frequencies],
        &[tail.rope],
    );
    dispatch_deepseek_tail_rope_bf16(
        session.encoder_mut(),
        registry,
        device,
        &rope_input,
        compressed_positions,
        frequencies,
        tail.rope,
        &DeepSeekTailRopeParams {
            batch: 1,
            seq_len: compressed_count as u32,
            heads: 1,
            head_dim: dim as u32,
            rope_dim: cfg.rope_head_dim,
            inverse: 0,
        },
    )?;
    session.barrier_between(&[tail.rope], &[tail.rope]);
    dispatch_deepseek_mxfp8_fake_quant_bf16(
        session.encoder_mut(),
        registry,
        device,
        tail.rope,
        &DeepSeekMxfp8Params {
            rows: compressed_count as u32,
            row_width: dim as u32,
            quantized_width: (dim - cfg.rope_head_dim as usize) as u32,
            block_size: 64,
        },
    )?;
    session.barrier_between(&[tail.rope], &[cache]);
    dispatch_kv_cache_copy(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        tail.rope,
        cache,
        compressed_write_start as u32,
        dim as u32,
        compressed_count as u32,
        cache_capacity as u32,
        false,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_indexer_tail(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    rows: usize,
    position: usize,
    compressed_write_start: usize,
    compressed_output_count: usize,
    cache: &MlxBuffer,
    cache_capacity: usize,
    kv_state: &MlxBuffer,
    score_state: &MlxBuffer,
    tail: &CompressorTail<'_>,
    compressed_positions: &MlxBuffer,
    frequencies: &MlxBuffer,
) -> Result<()> {
    let dim = cfg.index_head_dim as usize;
    session.barrier_between(
        &[
            tail.kv,
            tail.score,
            tail.ape,
            tail.norm,
            kv_state,
            score_state,
        ],
        &[kv_state, score_state, tail.output],
    );
    dispatch_deepseek_compressor(
        session.encoder_mut(),
        registry,
        device,
        tail.kv,
        tail.score,
        tail.ape,
        tail.norm,
        kv_state,
        score_state,
        tail.output,
        tail.output,
        &DeepSeekCompressorParams {
            batch: 1,
            seq_len: rows as u32,
            start_pos: position as u32,
            ratio: 4,
            head_dim: dim as u32,
            cache_len: cache_capacity.max(1) as u32,
            epsilon: cfg.rms_norm_eps,
            write_cache: 0,
        },
    )?;
    if compressed_output_count > 0 {
        let rope_input = tail
            .output
            .with_shape(vec![1, compressed_output_count, 1, dim])?;
        session.barrier_between(
            &[&rope_input, compressed_positions, frequencies],
            &[tail.rope],
        );
        dispatch_deepseek_tail_rope_bf16(
            session.encoder_mut(),
            registry,
            device,
            &rope_input,
            compressed_positions,
            frequencies,
            tail.rope,
            &DeepSeekTailRopeParams {
                batch: 1,
                seq_len: compressed_output_count as u32,
                heads: 1,
                head_dim: dim as u32,
                rope_dim: cfg.rope_head_dim,
                inverse: 0,
            },
        )?;
        session.barrier_between(&[tail.rope], &[tail.rope]);
        dispatch_deepseek_hadamard_mxfp4_bf16(
            session.encoder_mut(),
            registry,
            device,
            tail.rope,
            compressed_output_count as u32,
        )?;
        session.barrier_between(&[tail.rope], &[cache]);
        dispatch_kv_cache_copy(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            tail.rope,
            cache,
            compressed_write_start as u32,
            dim as u32,
            compressed_output_count as u32,
            cache_capacity as u32,
            false,
        )?;
    }

    Ok(())
}
