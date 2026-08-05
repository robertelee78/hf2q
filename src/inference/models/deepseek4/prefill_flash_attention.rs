//! DeepSeek sparse/sink adapter for mlx-native's llama.cpp flash prefill port.

use anyhow::{Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_sparse_prefill_mask::{
    dispatch_deepseek_sparse_prefill_mask_f16, DeepSeekSparsePrefillMaskParams,
};
use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::flash_attn_prefill::FlashAttnPrefillParams;
use mlx_native::ops::flash_attn_prefill_d512::dispatch_flash_attn_prefill_f16_d512_with_sinks;
use mlx_native::ops::transpose::permute_021_f16;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::forward_support::alloc;

pub(super) struct DeepseekPrefillFlashArena {
    q_head_major: MlxBuffer,
    kv: MlxBuffer,
    mask: MlxBuffer,
    output_head_major: MlxBuffer,
}

impl DeepseekPrefillFlashArena {
    pub(super) fn new(
        device: &MlxDevice,
        rows: usize,
        heads: usize,
        head_dim: usize,
        kv_len: usize,
    ) -> Result<Self> {
        Ok(Self {
            q_head_major: alloc(
                device,
                DType::F16,
                vec![1, heads, rows, head_dim],
                "flash prefill head-major query",
            )?,
            kv: alloc(
                device,
                DType::F16,
                vec![1, 1, kv_len, head_dim],
                "flash prefill KV",
            )?,
            mask: alloc(
                device,
                DType::F16,
                vec![1, heads, rows, kv_len],
                "flash prefill sparse mask",
            )?,
            output_head_major: alloc(
                device,
                DType::F16,
                vec![1, heads, rows, head_dim],
                "flash prefill head-major output",
            )?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_deepseek_flash_prefill(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    query_token_major: &MlxBuffer,
    kv_source: &MlxBuffer,
    compressed_kv_source: Option<&MlxBuffer>,
    sinks: &MlxBuffer,
    indices: &MlxBuffer,
    output_token_major: &MlxBuffer,
    arena: &DeepseekPrefillFlashArena,
    rows: usize,
    raw_kv_len: usize,
    kv_len: usize,
    top_k: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<()> {
    session.barrier_between(&[query_token_major], &[&arena.q_head_major]);
    permute_021_f16(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        query_token_major,
        &arena.q_head_major,
        rows,
        heads,
        head_dim,
    )?;
    let raw_destination = arena.kv.slice_view(0, raw_kv_len * head_dim);
    session.barrier_between(&[kv_source], &[&raw_destination]);
    permute_021_f16(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        kv_source,
        &raw_destination,
        raw_kv_len,
        1,
        head_dim,
    )?;
    if let Some(compressed) = compressed_kv_source {
        let compressed_len = kv_len
            .checked_sub(raw_kv_len)
            .context("DeepSeek-V4 compact prefill KV length underflow")?;
        let destination = arena.kv.slice_view(
            (raw_kv_len * head_dim * DType::F16.size_of()) as u64,
            compressed_len * head_dim,
        );
        if compressed_len > 0 {
            session.barrier_between(&[compressed], &[&destination]);
            cast(
                session.encoder_mut(),
                registry,
                device.metal_device(),
                compressed,
                &destination,
                compressed_len * head_dim,
                CastDirection::BF16ToF16,
            )?;
        }
    }
    session.barrier_between(&[indices], &[&arena.mask]);
    dispatch_deepseek_sparse_prefill_mask_f16(
        session.encoder_mut(),
        registry,
        device,
        indices,
        &arena.mask,
        &DeepSeekSparsePrefillMaskParams {
            batch: 1,
            query_len: rows as u32,
            kv_len: kv_len as u32,
            top_k: top_k as u32,
            heads: heads as u32,
        },
    )?;
    session.barrier_between(
        &[&arena.q_head_major, &arena.kv, &arena.mask, sinks],
        &[&arena.output_head_major],
    );
    dispatch_flash_attn_prefill_f16_d512_with_sinks(
        session.encoder_mut(),
        device,
        registry,
        &arena.q_head_major,
        &arena.kv,
        &arena.kv,
        Some(&arena.mask),
        sinks,
        &arena.output_head_major,
        &FlashAttnPrefillParams {
            n_heads: heads as u32,
            n_kv_heads: 1,
            head_dim: head_dim as u32,
            seq_len_q: rows as u32,
            seq_len_k: kv_len as u32,
            batch: 1,
            scale,
            do_causal: false,
        },
    )?;
    session.barrier_between(&[&arena.output_head_major], &[output_token_major]);
    permute_021_f16(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        &arena.output_head_major,
        output_token_major,
        heads,
        rows,
        head_dim,
    )?;
    Ok(())
}
