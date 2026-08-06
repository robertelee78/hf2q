//! DeepSeek sparse/sink adapter for mlx-native's llama.cpp flash prefill port.

use anyhow::{Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_sparse_attention::{
    dispatch_deepseek_sparse_attention_flash_prefill, DeepSeekSparseAttentionParams,
};
use mlx_native::ops::deepseek_sparse_prefill_mask::{
    dispatch_deepseek_sparse_prefill_mask, DeepSeekSparsePrefillMaskParams,
};
use mlx_native::ops::flash_attn_prefill::FlashAttnPrefillParams;
use mlx_native::ops::flash_attn_prefill_blk::{dispatch_flash_attn_prefill_blk, BlkParams};
use mlx_native::ops::flash_attn_prefill_d512::dispatch_flash_attn_prefill_bf16_d512_with_blk_and_sinks;
use mlx_native::ops::transpose::permute_021_bf16;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::forward_support::alloc;

/// Keep the gathered sparse working set bounded while presenting independent
/// query rows to the flash-attention kernel as a batch. At the production
/// width (640 selected rows, D=512), 256 queries require 160 MiB of BF16 KV.
const SPARSE_PREFILL_QUERY_TILE: usize = 256;

pub(super) struct DeepseekPrefillFlashArena {
    q_head_major: MlxBuffer,
    kv: MlxBuffer,
    mask: MlxBuffer,
    blk: MlxBuffer,
    output_head_major: MlxBuffer,
}

pub(super) struct DeepseekSparsePrefillFlashArena {
    kv: MlxBuffer,
    gathered_kv: MlxBuffer,
    mask: MlxBuffer,
    invalid_global: MlxBuffer,
    invalid_heads: MlxBuffer,
}

impl DeepseekSparsePrefillFlashArena {
    pub(super) fn new(
        device: &MlxDevice,
        rows: usize,
        heads: usize,
        head_dim: usize,
        kv_len: usize,
        top_k: usize,
    ) -> Result<Self> {
        let tile = rows.min(SPARSE_PREFILL_QUERY_TILE);
        let mut invalid_global = alloc(
            device,
            DType::U32,
            vec![1, rows],
            "sparse flash prefill global validity",
        )?;
        invalid_global.as_logical_mut_slice::<u32>()?.fill(0);
        let mut invalid_heads = alloc(
            device,
            DType::U32,
            vec![1, rows, heads],
            "sparse flash prefill head validity",
        )?;
        invalid_heads.as_logical_mut_slice::<u32>()?.fill(0);
        Ok(Self {
            kv: alloc(
                device,
                DType::BF16,
                vec![1, 1, kv_len, head_dim],
                "sparse flash prefill compact KV",
            )?,
            gathered_kv: alloc(
                device,
                DType::BF16,
                vec![1, tile, top_k, head_dim],
                "sparse flash prefill gathered KV tile",
            )?,
            mask: alloc(
                device,
                DType::BF16,
                vec![tile, 1, top_k],
                "sparse flash prefill selected mask tile",
            )?,
            invalid_global,
            invalid_heads,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_deepseek_sparse_flash_prefill(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    query_token_major: &MlxBuffer,
    raw_prefix: Option<&MlxBuffer>,
    kv_source: &MlxBuffer,
    compressed_kv_source: &MlxBuffer,
    sinks: &MlxBuffer,
    indices: &MlxBuffer,
    output_token_major: &MlxBuffer,
    arena: &DeepseekSparsePrefillFlashArena,
    rows: usize,
    raw_prefix_len: usize,
    raw_kv_len: usize,
    kv_len: usize,
    top_k: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<()> {
    if let Some(prefix) = raw_prefix {
        let prefix_destination = arena.kv.slice_view(0, raw_prefix_len * head_dim);
        session.barrier_between(&[prefix], &[&prefix_destination]);
        permute_021_bf16(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            prefix,
            &prefix_destination,
            raw_prefix_len,
            1,
            head_dim,
        )?;
    }
    let current_offset = raw_prefix_len * head_dim * DType::BF16.size_of();
    let current_destination = arena.kv.slice_view(current_offset as u64, rows * head_dim);
    session.barrier_between(&[kv_source], &[&current_destination]);
    permute_021_bf16(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        kv_source,
        &current_destination,
        rows,
        1,
        head_dim,
    )?;
    let compressed_len = kv_len
        .checked_sub(raw_kv_len)
        .context("DeepSeek-V4 sparse compact prefill KV length underflow")?;
    if compressed_len > 0 {
        let destination = arena.kv.slice_view(
            (raw_kv_len * head_dim * DType::BF16.size_of()) as u64,
            compressed_len * head_dim,
        );
        session.barrier_between(&[compressed_kv_source], &[&destination]);
        permute_021_bf16(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            compressed_kv_source,
            &destination,
            compressed_len,
            1,
            head_dim,
        )?;
    }

    for query_start in (0..rows).step_by(SPARSE_PREFILL_QUERY_TILE) {
        let query_count = (rows - query_start).min(SPARSE_PREFILL_QUERY_TILE);
        let query_elements = query_count * heads * head_dim;
        let query = query_token_major
            .slice_view(
                query_token_major.byte_offset()
                    + (query_start * heads * head_dim * DType::BF16.size_of()) as u64,
                query_elements,
            )
            .with_shape(vec![1, query_count, heads, head_dim])?;
        let selected = indices
            .slice_view(
                indices.byte_offset() + (query_start * top_k * DType::I32.size_of()) as u64,
                query_count * top_k,
            )
            .with_shape(vec![1, query_count, top_k])?;
        let output = output_token_major
            .slice_view(
                output_token_major.byte_offset()
                    + (query_start * heads * head_dim * DType::BF16.size_of()) as u64,
                query_elements,
            )
            .with_shape(vec![1, query_count, heads, head_dim])?;
        let gathered = arena
            .gathered_kv
            .slice_view(0, query_count * top_k * head_dim)
            .with_shape(vec![1, query_count, top_k, head_dim])?;
        let mask = arena
            .mask
            .slice_view(0, query_count * top_k)
            .with_shape(vec![query_count, 1, top_k])?;
        let invalid_global = arena
            .invalid_global
            .slice_view(
                arena.invalid_global.byte_offset() + (query_start * DType::U32.size_of()) as u64,
                query_count,
            )
            .with_shape(vec![1, query_count])?;
        let invalid_heads = arena
            .invalid_heads
            .slice_view(
                arena.invalid_heads.byte_offset()
                    + (query_start * heads * DType::U32.size_of()) as u64,
                query_count * heads,
            )
            .with_shape(vec![1, query_count, heads])?;
        session.barrier_between(
            &[&query, &arena.kv, sinks, &selected],
            &[&gathered, &mask, &invalid_global, &invalid_heads, &output],
        );
        dispatch_deepseek_sparse_attention_flash_prefill(
            session.encoder_mut(),
            registry,
            device,
            &query,
            &arena.kv.with_shape(vec![1, kv_len, head_dim])?,
            sinks,
            &selected,
            &gathered,
            &mask,
            &invalid_global,
            &invalid_heads,
            &output,
            &DeepSeekSparseAttentionParams {
                batch: 1,
                query_len: query_count as u32,
                kv_len: kv_len as u32,
                top_k: top_k as u32,
                heads: heads as u32,
                head_dim: head_dim as u32,
                scale,
            },
        )?;
    }
    Ok(())
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
                DType::BF16,
                vec![1, heads, rows, head_dim],
                "flash prefill head-major query",
            )?,
            kv: alloc(
                device,
                DType::BF16,
                vec![1, 1, kv_len, head_dim],
                "flash prefill KV",
            )?,
            mask: alloc(
                device,
                DType::BF16,
                vec![rows, kv_len],
                "flash prefill broadcast sparse mask",
            )?,
            blk: alloc(
                device,
                DType::U8,
                vec![rows.div_ceil(8), kv_len.div_ceil(64)],
                "flash prefill sparse tile map",
            )?,
            output_head_major: alloc(
                device,
                DType::BF16,
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
    raw_prefix: Option<&MlxBuffer>,
    kv_source: &MlxBuffer,
    compressed_kv_source: Option<&MlxBuffer>,
    sinks: &MlxBuffer,
    indices: &MlxBuffer,
    output_token_major: &MlxBuffer,
    arena: &DeepseekPrefillFlashArena,
    rows: usize,
    raw_prefix_len: usize,
    raw_kv_len: usize,
    kv_len: usize,
    top_k: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<()> {
    session.barrier_between(&[query_token_major], &[&arena.q_head_major]);
    permute_021_bf16(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        query_token_major,
        &arena.q_head_major,
        rows,
        heads,
        head_dim,
    )?;
    if let Some(prefix) = raw_prefix {
        let prefix_destination = arena.kv.slice_view(0, raw_prefix_len * head_dim);
        session.barrier_between(&[prefix], &[&prefix_destination]);
        permute_021_bf16(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            prefix,
            &prefix_destination,
            raw_prefix_len,
            1,
            head_dim,
        )?;
    }
    let current_offset = raw_prefix_len * head_dim * DType::BF16.size_of();
    let current_destination = arena.kv.slice_view(current_offset as u64, rows * head_dim);
    session.barrier_between(&[kv_source], &[&current_destination]);
    permute_021_bf16(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        kv_source,
        &current_destination,
        rows,
        1,
        head_dim,
    )?;
    if let Some(compressed) = compressed_kv_source {
        let compressed_len = kv_len
            .checked_sub(raw_kv_len)
            .context("DeepSeek-V4 compact prefill KV length underflow")?;
        let destination = arena.kv.slice_view(
            (raw_kv_len * head_dim * DType::BF16.size_of()) as u64,
            compressed_len * head_dim,
        );
        if compressed_len > 0 {
            session.barrier_between(&[compressed], &[&destination]);
            permute_021_bf16(
                session.encoder_mut(),
                registry,
                device.metal_device(),
                compressed,
                &destination,
                compressed_len,
                1,
                head_dim,
            )?;
        }
    }
    session.barrier_between(&[indices], &[&arena.mask]);
    dispatch_deepseek_sparse_prefill_mask(
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
    session.barrier_between(&[&arena.mask], &[&arena.blk]);
    dispatch_flash_attn_prefill_blk(
        session.encoder_mut(),
        device,
        registry,
        &arena.mask,
        &arena.blk,
        &BlkParams {
            seq_len_q: rows as u32,
            seq_len_k: kv_len as u32,
            bq: 8,
            bk: 64,
        },
    )?;
    session.barrier_between(
        &[
            &arena.q_head_major,
            &arena.kv,
            &arena.mask,
            &arena.blk,
            sinks,
        ],
        &[&arena.output_head_major],
    );
    dispatch_flash_attn_prefill_bf16_d512_with_blk_and_sinks(
        session.encoder_mut(),
        device,
        registry,
        &arena.q_head_major,
        &arena.kv,
        &arena.kv,
        &arena.mask,
        &arena.blk,
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
    permute_021_bf16(
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
