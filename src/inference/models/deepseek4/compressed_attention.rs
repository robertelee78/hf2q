//! Exact one-token ratio-four and ratio-128 DeepSeek-V4 attention.

use anyhow::{bail, Context, Result};
use mlx_native::ops::deepseek_activation_quant::{
    dispatch_deepseek_mxfp8_fake_quant_bf16, DeepSeekMxfp8Params,
};
use mlx_native::ops::deepseek_sparse_attention::{
    dispatch_deepseek_sparse_attention, DeepSeekSparseAttentionParams,
};
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_f32_to_bf16, DeepSeekTailRopeParams,
};
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy;
use mlx_native::{DType, MlxBuffer};

use super::attention::{compressed_indices, window_indices};
use super::cache::Deepseek4Cache;
use super::compressed_attention_common::{
    encode_compressed_attention_epilogue, encode_compressed_attention_prelude,
    CompressedAttentionCoreArena,
};
use super::compressed_attention_indexer::{encode_ratio_four_indexer, RatioFourIndexerArena};
use super::compressed_attention_main::{encode_main_compressor, MainCompressorArena};
use super::compressed_attention_weights::CompressedAttentionWeightsRef;
use super::forward_support::alloc;
use super::rope::yarn_frequencies;
use super::Deepseek4Model;

impl Deepseek4Model {
    pub(super) fn forward_compressed_attention_one(
        &mut self,
        state: &MlxBuffer,
        layer: usize,
        cache: &mut Deepseek4Cache,
        commit_cache: bool,
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
        if state.dtype() != DType::F32 || state.shape() != [1, hc, hidden] {
            bail!(
                "DeepSeek-V4 layer {layer} compressed state must be F32 [1, {hc}, {hidden}], got {} {:?}",
                state.dtype(),
                state.shape()
            );
        }
        if head_dim <= self.cfg.rope_head_dim as usize {
            bail!("DeepSeek-V4 compressed KV needs non-RoPE dimensions");
        }
        let cache_step = cache
            .plan_next_step()
            .context("plan compressed DeepSeek-V4 cache transaction")?;
        let layer_step = cache_step
            .layers
            .get(layer)
            .with_context(|| format!("missing layer-{layer} compressed cache step"))?;
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
        let core = CompressedAttentionCoreArena::new(&device, &self.cfg)?;
        let main_compressor = MainCompressorArena::new(&device, &self.cfg, ratio)?;
        let indexer = (ratio == 4)
            .then(|| RatioFourIndexerArena::new(&device, &self.cfg, layer_step.indexer_valid_after))
            .transpose()?;
        let q_rope = alloc(
            &device,
            DType::BF16,
            vec![1, 1, heads, head_dim],
            "compressed rotated query",
        )?;
        let kv_rope = alloc(
            &device,
            DType::BF16,
            vec![1, 1, 1, head_dim],
            "compressed rotated raw KV",
        )?;
        let attention = alloc(
            &device,
            DType::BF16,
            vec![1, 1, heads, head_dim],
            "compressed sparse attention",
        )?;
        let mut positions = alloc(&device, DType::U32, vec![1], "compressed position")?;
        positions.as_mut_slice::<u32>()?[0] = u32::try_from(cache_step.position)
            .context("DeepSeek-V4 compressed position exceeds u32")?;
        let mut compressed_positions =
            alloc(&device, DType::U32, vec![1], "compressed block position")?;
        compressed_positions.as_mut_slice::<u32>()?[0] = layer_step
            .compressed_write_slot
            .map(|_| cache_step.position + 1 - ratio as usize)
            .unwrap_or(0)
            .try_into()
            .context("DeepSeek-V4 compressed block position exceeds u32")?;
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
            .as_mut_slice::<f32>()?
            .copy_from_slice(&frequencies);

        let mut index_values = window_indices(window_capacity, 1, cache_step.position)?
            .into_iter()
            .next()
            .context("compressed window indices are missing")?;
        if ratio == 4 {
            index_values.resize(window_capacity + self.cfg.index_top_k as usize, -1);
        } else {
            index_values.extend(
                compressed_indices(ratio as usize, 1, cache_step.position, window_capacity)?
                    .into_iter()
                    .next()
                    .context("ratio-128 compressed indices are missing")?,
            );
        }
        let mut indices = alloc(
            &device,
            DType::I32,
            vec![1, 1, index_values.len()],
            "compressed attention indices",
        )?;
        indices
            .as_mut_slice::<i32>()?
            .copy_from_slice(&index_values);
        let indexer_output = (ratio == 4)
            .then(|| {
                indices
                    .slice_view(
                        (window_capacity * DType::I32.size_of()) as u64,
                        self.cfg.index_top_k as usize,
                    )
                    .with_shape(vec![1, 1, self.cfg.index_top_k as usize])
            })
            .transpose()?;
        let weights = CompressedAttentionWeightsRef::get(&self.weights, layer, ratio)?;

        let (executor, registry) = self.ctx.split();
        let mut session = executor
            .begin()
            .with_context(|| format!("begin DeepSeek-V4 layer {layer} compressed attention"))?;
        encode_compressed_attention_prelude(
            &mut session,
            registry,
            &device,
            &self.cfg,
            state,
            &weights.attention,
            &core,
        )?;
        let rope = DeepSeekTailRopeParams {
            batch: 1,
            seq_len: 1,
            heads: heads as u32,
            head_dim: head_dim as u32,
            rope_dim: self.cfg.rope_head_dim,
            inverse: 0,
        };
        session.barrier_between(&[&core.q_norm, &positions, &frequency_buffer], &[&q_rope]);
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
        session.barrier_between(&[&core.kv_norm, &positions, &frequency_buffer], &[&kv_rope]);
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
        session.barrier_between(&[&kv_rope], &[&kv_rope]);
        dispatch_deepseek_mxfp8_fake_quant_bf16(
            session.encoder_mut(),
            registry,
            &device,
            &kv_rope,
            &DeepSeekMxfp8Params {
                rows: 1,
                row_width: head_dim as u32,
                quantized_width: (head_dim - self.cfg.rope_head_dim as usize) as u32,
                block_size: 64,
            },
        )?;
        session.barrier_between(&[&kv_rope], &[&layer_cache.window_kv]);
        dispatch_kv_cache_copy(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            &kv_rope,
            &layer_cache.window_kv,
            layer_step.window_write_slot as u32,
            head_dim as u32,
            1,
            window_capacity as u32,
            true,
        )?;
        encode_main_compressor(
            &mut session,
            registry,
            &device,
            &self.cfg,
            ratio,
            cache_step.position,
            layer_step,
            layer_cache,
            &core.attn_norm,
            &compressed_positions,
            &frequency_buffer,
            &weights.compressor,
            &main_compressor,
        )?;
        if let (Some(indexer_weights), Some(indexer_arena), Some(indexer_output)) = (
            weights.indexer.as_ref(),
            indexer.as_ref(),
            indexer_output.as_ref(),
        ) {
            encode_ratio_four_indexer(
                &mut session,
                registry,
                &device,
                &self.cfg,
                cache_step.position,
                layer_step,
                layer_cache,
                &core.attn_norm,
                &core.q_a_norm,
                &positions,
                &compressed_positions,
                &frequency_buffer,
                indexer_weights,
                indexer_arena,
                indexer_output,
            )?;
        }

        let cache_view = layer_cache.attention_kv.with_shape(vec![
            1,
            window_capacity + compressed_capacity,
            head_dim,
        ])?;
        session.barrier_between(
            &[&q_rope, &cache_view, weights.attention.sinks, &indices],
            &[&attention],
        );
        dispatch_deepseek_sparse_attention(
            session.encoder_mut(),
            registry,
            &device,
            &q_rope,
            &cache_view,
            weights.attention.sinks,
            &indices,
            &attention,
            &DeepSeekSparseAttentionParams {
                batch: 1,
                query_len: 1,
                kv_len: (window_capacity + compressed_capacity) as u32,
                top_k: index_values.len() as u32,
                heads: heads as u32,
                head_dim: head_dim as u32,
                scale: 1.0 / (head_dim as f32).sqrt(),
            },
        )?;
        encode_compressed_attention_epilogue(
            &mut session,
            registry,
            &device,
            &self.cfg,
            state,
            &weights.attention,
            &core,
            &attention,
            &positions,
            &frequency_buffer,
        )?;
        session
            .finish()
            .with_context(|| format!("execute DeepSeek-V4 layer {layer} compressed attention"))?;
        if commit_cache {
            cache
                .commit_step(cache_step.position)
                .context("publish compressed DeepSeek-V4 cache transaction")?;
        }
        Ok(core.output_state)
    }
}
