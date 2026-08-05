//! Exact one-token ratio-four and ratio-128 DeepSeek-V4 attention.

use anyhow::{bail, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_activation_quant::{
    dispatch_deepseek_mxfp8_fake_quant_bf16, DeepSeekMxfp8Params,
};
use mlx_native::ops::deepseek_sparse_attention::{
    dispatch_deepseek_sparse_attention, DeepSeekSparseAttentionParams,
};
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_f32_to_bf16, dispatch_deepseek_tail_rope_f32_to_f16,
    DeepSeekTailRopeParams,
};
use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy;
use mlx_native::{DType, GraphExecutor, MlxBuffer};

use super::attention::{compressed_attention_index_plan, compressed_indices, window_indices};
use super::cache::{CacheSpan, Deepseek4Cache};
use super::compressed_attention_common::{
    encode_compressed_attention_epilogue, encode_compressed_attention_prelude,
    CompressedAttentionCoreArena,
};
use super::compressed_attention_indexer::{encode_ratio_four_indexer, RatioFourIndexerArena};
use super::compressed_attention_main::{encode_main_compressor, MainCompressorArena};
use super::compressed_attention_weights::CompressedAttentionWeightsRef;
use super::forward_support::alloc;
use super::prefill_flash_attention::{encode_deepseek_flash_prefill, DeepseekPrefillFlashArena};
use super::rope::yarn_frequencies;
use super::submission::{finish_or_commit, SubmissionChain};
use super::Deepseek4Model;

impl Deepseek4Model {
    pub(super) fn forward_compressed_attention_one(
        &mut self,
        state: &MlxBuffer,
        layer: usize,
        cache: &mut Deepseek4Cache,
        commit_cache: bool,
        in_flight: Option<&mut SubmissionChain>,
        shared_session: Option<&mut GraphSession<'_>>,
    ) -> Result<MlxBuffer> {
        self.forward_compressed_attention_rows(
            state,
            layer,
            cache,
            commit_cache,
            None,
            in_flight,
            shared_session,
        )
    }

    pub(super) fn forward_compressed_attention_prefill(
        &mut self,
        state: &MlxBuffer,
        layer: usize,
        cache: &mut Deepseek4Cache,
        span: &CacheSpan,
        in_flight: Option<&mut SubmissionChain>,
        shared_session: Option<&mut GraphSession<'_>>,
    ) -> Result<MlxBuffer> {
        self.forward_compressed_attention_rows(
            state,
            layer,
            cache,
            false,
            Some(span),
            in_flight,
            shared_session,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_compressed_attention_rows(
        &mut self,
        state: &MlxBuffer,
        layer: usize,
        cache: &mut Deepseek4Cache,
        commit_cache: bool,
        prefill_span: Option<&CacheSpan>,
        in_flight: Option<&mut SubmissionChain>,
        shared_session: Option<&mut GraphSession<'_>>,
    ) -> Result<MlxBuffer> {
        let ratio = self
            .cfg
            .compress_ratios
            .get(layer)
            .copied()
            .with_context(|| format!("missing DeepSeek-V4 layer-{layer} compression ratio"))?;
        if !matches!(ratio, 4 | 128) {
            bail!("DeepSeek-V4 layer {layer} is not a compressed attention layer");
        }
        let hidden = self.cfg.hidden_size as usize;
        let hc = self.cfg.hyper_connection_count as usize;
        let heads = self.cfg.num_attention_heads as usize;
        let head_dim = self.cfg.head_dim as usize;
        let rows = state.shape().first().copied().unwrap_or(0);
        if rows == 0 || state.dtype() != DType::F32 || state.shape() != [rows, hc, hidden] {
            bail!(
                "DeepSeek-V4 layer {layer} compressed state must be F32 [rows, {hc}, {hidden}], got {} {:?}",
                state.dtype(),
                state.shape()
            );
        }
        if head_dim <= self.cfg.rope_head_dim as usize {
            bail!("DeepSeek-V4 compressed KV needs non-RoPE dimensions");
        }
        let cache_step = prefill_span
            .is_none()
            .then(|| {
                cache
                    .plan_next_step()
                    .context("plan compressed DeepSeek-V4 cache transaction")
            })
            .transpose()?;
        let start_position = prefill_span
            .map(|span| span.start_position)
            .or_else(|| cache_step.as_ref().map(|step| step.position))
            .context("DeepSeek-V4 compressed attention cache position is missing")?;
        let layer_step = cache_step
            .as_ref()
            .map(|step| {
                step.layers
                    .get(layer)
                    .with_context(|| format!("missing layer-{layer} compressed cache step"))
            })
            .transpose()?;
        let layer_span = prefill_span
            .map(|span| {
                if span.token_count != rows {
                    bail!(
                        "DeepSeek-V4 prefill span has {} rows, compressed attention received {rows}",
                        span.token_count
                    );
                }
                span.layers
                    .get(layer)
                    .with_context(|| format!("missing layer-{layer} compressed cache span"))
            })
            .transpose()?;
        let layer_cache = cache
            .layers()
            .get(layer)
            .with_context(|| format!("missing layer-{layer} compressed cache"))?;
        let window_capacity = layer_cache.window_kv.shape()[0];
        let compressed_capacity = layer_cache
            .compressed_kv
            .as_ref()
            .context("compressed KV cache is missing")?
            .shape()[0];
        if window_capacity != self.cfg.sliding_window as usize
            || layer_cache.attention_kv.shape() != [window_capacity + compressed_capacity, head_dim]
        {
            bail!("DeepSeek-V4 layer {layer} compressed attention cache shape drift");
        }

        let device = self.ctx.device().clone();
        let use_f16_prefill = rows > 1;
        let compressed_count = layer_span
            .map(|span| span.compressed_count)
            .unwrap_or_else(|| {
                usize::from(
                    layer_step
                        .and_then(|step| step.compressed_write_slot)
                        .is_some(),
                )
            });
        let valid_compressed = layer_span
            .map(|span| span.compressed_count)
            .or_else(|| layer_step.map(|step| step.compressed_valid_after))
            .unwrap_or(0);
        let core = CompressedAttentionCoreArena::new(&device, &self.cfg, rows)?;
        let main_compressor = MainCompressorArena::new(&device, &self.cfg, ratio, rows)?;
        let indexer = (ratio == 4)
            .then(|| {
                RatioFourIndexerArena::new(
                    &device,
                    &self.cfg,
                    rows,
                    compressed_count,
                    valid_compressed,
                )
            })
            .transpose()?;
        let q_rope = alloc(
            &device,
            if use_f16_prefill {
                DType::F16
            } else {
                DType::BF16
            },
            vec![1, rows, heads, head_dim],
            "compressed rotated query",
        )?;
        let kv_rope = alloc(
            &device,
            if use_f16_prefill {
                DType::F16
            } else {
                DType::BF16
            },
            vec![1, rows, 1, head_dim],
            "compressed rotated raw KV",
        )?;
        let attention = alloc(
            &device,
            if use_f16_prefill {
                DType::F16
            } else {
                DType::BF16
            },
            vec![1, rows, heads, head_dim],
            "compressed sparse attention",
        )?;
        let kv_cache_source = use_f16_prefill
            .then(|| {
                alloc(
                    &device,
                    DType::BF16,
                    vec![1, rows, 1, head_dim],
                    "compressed raw KV cache source",
                )
            })
            .transpose()?;
        let physical_kv_len = window_capacity + compressed_capacity;
        let compact_prefill_kv_len = rows + valid_compressed;
        let prefill_flash = (rows > 1)
            .then(|| {
                DeepseekPrefillFlashArena::new(
                    &device,
                    rows,
                    heads,
                    head_dim,
                    compact_prefill_kv_len,
                )
            })
            .transpose()?;
        let mut positions = alloc(&device, DType::U32, vec![rows], "compressed positions")?;
        for (offset, position) in positions
            .as_logical_mut_slice::<u32>()?
            .iter_mut()
            .enumerate()
        {
            *position = u32::try_from(start_position + offset)
                .context("DeepSeek-V4 compressed position exceeds u32")?;
        }
        let mut compressed_positions = alloc(
            &device,
            DType::U32,
            vec![compressed_count.max(1)],
            "compressed block positions",
        )?;
        if prefill_span.is_some() {
            for (group, position) in compressed_positions
                .as_logical_mut_slice::<u32>()?
                .iter_mut()
                .take(compressed_count)
                .enumerate()
            {
                *position = u32::try_from(group * ratio as usize)
                    .context("DeepSeek-V4 compressed block position exceeds u32")?;
            }
        } else if compressed_count == 1 {
            compressed_positions.as_logical_mut_slice::<u32>()?[0] =
                u32::try_from(start_position + 1 - ratio as usize)
                    .context("DeepSeek-V4 compressed block position exceeds u32")?;
        }
        let frequencies = yarn_frequencies(
            self.cfg.rope_head_dim as usize,
            self.cfg.original_context_length as usize,
            self.cfg.compress_rope_theta,
            self.cfg.rope_factor,
            self.cfg.yarn_beta_fast,
            self.cfg.yarn_beta_slow,
        )?;
        let mut frequency_buffer = alloc(
            &device,
            DType::F32,
            vec![frequencies.len()],
            "compressed RoPE frequencies",
        )?;
        frequency_buffer
            .as_logical_mut_slice::<f32>()?
            .copy_from_slice(&frequencies);

        let (index_storage, storage_stride, attention_width, indexer_output_offset) =
            if prefill_span.is_some() {
                let window = window_indices(window_capacity, rows, 0)?;
                let tail_width = if ratio == 4 {
                    self.cfg.index_top_k as usize
                } else {
                    compressed_count
                };
                let width = rows + tail_width;
                let mut storage = vec![-1_i32; rows * width];
                let compressed = (ratio == 128)
                    .then(|| compressed_indices(ratio as usize, rows, 0, rows))
                    .transpose()?;
                for query in 0..rows {
                    storage[query * width..query * width + rows].copy_from_slice(&window[query]);
                    if let Some(compressed) = compressed.as_ref() {
                        storage[query * width + rows..(query + 1) * width]
                            .copy_from_slice(&compressed[query]);
                    }
                }
                (storage, width, width, (ratio == 4).then_some(rows))
            } else {
                let plan = compressed_attention_index_plan(
                    ratio as usize,
                    window_capacity,
                    self.cfg.index_top_k as usize,
                    start_position,
                )?;
                let storage_stride = plan.storage.len();
                (
                    plan.storage,
                    storage_stride,
                    plan.attention_width,
                    plan.indexer_output_offset,
                )
            };
        let mut indices = alloc(
            &device,
            DType::I32,
            vec![1, rows, storage_stride],
            "compressed attention indices",
        )?;
        indices
            .as_logical_mut_slice::<i32>()?
            .copy_from_slice(&index_storage);
        let attention_indices = if storage_stride == attention_width {
            indices.with_shape(vec![1, rows, attention_width])?
        } else {
            indices
                .slice_view(indices.byte_offset(), rows * attention_width)
                .with_shape(vec![1, rows, attention_width])?
        };
        let weights = CompressedAttentionWeightsRef::get(&self.weights, layer, ratio)?;

        let registry = &mut self.ctx.registry;
        let mut encode = |session: &mut GraphSession<'_>| -> Result<()> {
            encode_compressed_attention_prelude(
                session,
                registry,
                &device,
                &self.cfg,
                state,
                &weights.attention,
                &core,
                rows,
            )?;
            let rope = DeepSeekTailRopeParams {
                batch: 1,
                seq_len: rows as u32,
                heads: heads as u32,
                head_dim: head_dim as u32,
                rope_dim: self.cfg.rope_head_dim,
                inverse: 0,
            };
            session.barrier_between(&[&core.q_norm, &positions, &frequency_buffer], &[&q_rope]);
            if use_f16_prefill {
                dispatch_deepseek_tail_rope_f32_to_f16(
                    session.encoder_mut(),
                    registry,
                    &device,
                    &core.q_norm,
                    &positions,
                    &frequency_buffer,
                    &q_rope,
                    &rope,
                )?;
            } else {
                dispatch_deepseek_tail_rope_f32_to_bf16(
                    session.encoder_mut(),
                    registry,
                    &device,
                    &core.q_norm,
                    &positions,
                    &frequency_buffer,
                    &q_rope,
                    &rope,
                )?;
            }
            session.barrier_between(&[&core.kv_norm, &positions, &frequency_buffer], &[&kv_rope]);
            if use_f16_prefill {
                dispatch_deepseek_tail_rope_f32_to_f16(
                    session.encoder_mut(),
                    registry,
                    &device,
                    &core.kv_norm,
                    &positions,
                    &frequency_buffer,
                    &kv_rope,
                    &DeepSeekTailRopeParams { heads: 1, ..rope },
                )?;
            } else {
                dispatch_deepseek_tail_rope_f32_to_bf16(
                    session.encoder_mut(),
                    registry,
                    &device,
                    &core.kv_norm,
                    &positions,
                    &frequency_buffer,
                    &kv_rope,
                    &DeepSeekTailRopeParams { heads: 1, ..rope },
                )?;
            }
            if !use_f16_prefill {
                session.barrier_between(&[&kv_rope], &[&kv_rope]);
                dispatch_deepseek_mxfp8_fake_quant_bf16(
                    session.encoder_mut(),
                    registry,
                    &device,
                    &kv_rope,
                    &DeepSeekMxfp8Params {
                        rows: rows as u32,
                        row_width: head_dim as u32,
                        quantized_width: (head_dim - self.cfg.rope_head_dim as usize) as u32,
                        block_size: 64,
                    },
                )?;
            }
            let kv_cache_input = if let Some(cache_source) = kv_cache_source.as_ref() {
                session.barrier_between(&[&kv_rope], &[cache_source]);
                cast(
                    session.encoder_mut(),
                    registry,
                    device.metal_device(),
                    &kv_rope,
                    cache_source,
                    rows * head_dim,
                    CastDirection::F16ToBF16,
                )?;
                cache_source
            } else {
                &kv_rope
            };
            session.barrier_between(&[kv_cache_input], &[&layer_cache.window_kv]);
            dispatch_kv_cache_copy(
                session.encoder_mut(),
                registry,
                device.metal_device(),
                kv_cache_input,
                &layer_cache.window_kv,
                layer_span
                    .map(|span| span.window_write_start)
                    .or_else(|| layer_step.map(|step| step.window_write_slot))
                    .context("DeepSeek-V4 compressed window write slot is missing")?
                    as u32,
                head_dim as u32,
                rows as u32,
                window_capacity as u32,
                true,
            )?;
            encode_main_compressor(
                session,
                registry,
                &device,
                &self.cfg,
                ratio,
                rows,
                start_position,
                layer_span
                    .map(|span| span.compressed_write_start)
                    .or_else(|| layer_step.and_then(|step| step.compressed_write_slot))
                    .unwrap_or(0),
                compressed_count,
                layer_cache,
                &core.attn_norm,
                &compressed_positions,
                &frequency_buffer,
                &weights.compressor,
                &main_compressor,
            )?;
            if let (Some(indexer_weights), Some(indexer_arena), Some(indexer_output_offset)) = (
                weights.indexer.as_ref(),
                indexer.as_ref(),
                indexer_output_offset,
            ) {
                encode_ratio_four_indexer(
                    session,
                    registry,
                    &device,
                    &self.cfg,
                    rows,
                    start_position,
                    layer_span
                        .map(|span| span.indexer_write_start)
                        .or_else(|| layer_step.and_then(|step| step.indexer_write_slot))
                        .unwrap_or(0),
                    compressed_count,
                    valid_compressed,
                    if prefill_span.is_some() {
                        rows
                    } else {
                        window_capacity
                    },
                    layer_cache,
                    &core.attn_norm,
                    &core.q_a_norm,
                    &positions,
                    &compressed_positions,
                    &frequency_buffer,
                    indexer_weights,
                    indexer_arena,
                    &indices,
                    storage_stride,
                    indexer_output_offset,
                )?;
            }

            if let Some(prefill_flash) = prefill_flash.as_ref() {
                encode_deepseek_flash_prefill(
                    session,
                    registry,
                    &device,
                    &q_rope,
                    &kv_rope,
                    layer_cache.compressed_kv.as_ref(),
                    weights.attention.sinks,
                    &attention_indices,
                    &attention,
                    prefill_flash,
                    rows,
                    rows,
                    compact_prefill_kv_len,
                    attention_width,
                    heads,
                    head_dim,
                    1.0 / (head_dim as f32).sqrt(),
                )?;
            } else {
                let cache_view =
                    layer_cache
                        .attention_kv
                        .with_shape(vec![1, physical_kv_len, head_dim])?;
                session.barrier_between(
                    &[
                        &q_rope,
                        &cache_view,
                        weights.attention.sinks,
                        &attention_indices,
                    ],
                    &[&attention],
                );
                dispatch_deepseek_sparse_attention(
                    session.encoder_mut(),
                    registry,
                    &device,
                    &q_rope,
                    &cache_view,
                    weights.attention.sinks,
                    &attention_indices,
                    &attention,
                    &DeepSeekSparseAttentionParams {
                        batch: 1,
                        query_len: rows as u32,
                        kv_len: physical_kv_len as u32,
                        top_k: attention_width as u32,
                        heads: heads as u32,
                        head_dim: head_dim as u32,
                        scale: 1.0 / (head_dim as f32).sqrt(),
                    },
                )?;
            }
            encode_compressed_attention_epilogue(
                session,
                registry,
                &device,
                &self.cfg,
                state,
                &weights.attention,
                &core,
                &attention,
                &positions,
                &frequency_buffer,
                rows,
            )?;
            Ok(())
        };
        if let Some(session) = shared_session {
            encode(session)?;
        } else {
            let local_executor = GraphExecutor::new(device.clone());
            let mut session = local_executor
                .begin()
                .with_context(|| format!("begin DeepSeek-V4 layer {layer} compressed attention"))?;
            encode(&mut session)?;
            finish_or_commit(
                session,
                in_flight,
                format!("execute DeepSeek-V4 layer {layer} compressed attention"),
            )?;
        }
        if commit_cache {
            cache
                .commit_step(
                    cache_step
                        .as_ref()
                        .context("DeepSeek-V4 compressed decode cache step is missing")?
                        .position,
                )
                .context("publish compressed DeepSeek-V4 cache transaction")?;
        }
        Ok(core.output_state)
    }
}
