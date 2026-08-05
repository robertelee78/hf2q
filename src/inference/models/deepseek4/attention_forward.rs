//! Exact one-token uncompressed DeepSeek-V4 attention layers.

use anyhow::{bail, Context, Result};
use mlx_native::ops::deepseek_hyper_connection::{
    dispatch_hc_post, dispatch_hc_pre, dispatch_hc_split_sinkhorn,
};
use mlx_native::ops::deepseek_sparse_attention::{
    dispatch_deepseek_sparse_attention, DeepSeekSparseAttentionParams,
};
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_bf16, dispatch_deepseek_tail_rope_f32_to_bf16,
    DeepSeekTailRopeParams,
};
use mlx_native::ops::elementwise::dispatch_cast_bf16_to_f32_with_encoder;
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy;
use mlx_native::{DType, MlxBuffer};

use super::attention::window_indices;
use super::attention_weights::AttentionWeightsRef;
use super::cache::Deepseek4Cache;
use super::forward_support::{alloc, grouped_output_a, raw_matmul, rms_params};
use super::rope::yarn_frequencies;
use super::Deepseek4Model;

impl Deepseek4Model {
    /// Execute one uncompressed verifier attention layer.
    /// `state=None` embeds layer 0; full-token callers defer cache publication.
    pub(super) fn forward_uncompressed_attention_one(
        &mut self,
        state: Option<&MlxBuffer>,
        token_id: u32,
        layer: usize,
        cache: &mut Deepseek4Cache,
        commit_cache: bool,
    ) -> Result<MlxBuffer> {
        if layer >= self.cfg.num_hidden_layers as usize {
            bail!(
                "DeepSeek-V4 attention layer index {layer} is outside 0..{}",
                self.cfg.num_hidden_layers
            );
        }
        if self.cfg.compress_ratios.get(layer).copied() != Some(0) {
            bail!("DeepSeek-V4 layer {layer} requires the compressed attention path");
        }
        if state.is_none() && layer != 0 {
            bail!("DeepSeek-V4 token embedding is valid only at layer 0");
        }
        let cache_step = cache
            .plan_next_step()
            .context("plan DeepSeek-V4 cache transaction")?;
        let layer_step = cache_step
            .layers
            .get(layer)
            .with_context(|| format!("missing layer-{layer} cache plan"))?;
        let layer_cache = cache
            .layers()
            .get(layer)
            .with_context(|| format!("missing layer-{layer} cache"))?;

        let hidden = self.cfg.hidden_size as usize;
        let hc = self.cfg.hyper_connection_count as usize;
        let hc_hidden = hc * hidden;
        let mix_width = (2 + hc) * hc;
        let heads = self.cfg.num_attention_heads as usize;
        let head_dim = self.cfg.head_dim as usize;
        let q_rank = self.cfg.q_lora_rank as usize;
        let o_rank = self.cfg.o_lora_rank as usize;
        let groups = self.cfg.output_groups as usize;
        let output_a_width = groups * o_rank;
        let window_capacity = layer_cache.window_kv.shape()[0];
        if layer_cache.window_kv.shape() != [window_capacity, head_dim] {
            bail!("DeepSeek-V4 layer-0 window cache shape drift");
        }

        let embedding = state
            .is_none()
            .then(|| self.prepare_embedding_arena(&[token_id]))
            .transpose()?;
        let embedding_weight = embedding
            .as_ref()
            .map(|_| self.weights.raw_matrix_ref("token_embd.weight"))
            .transpose()?;
        let state = match (state, embedding.as_ref()) {
            (Some(state), _) => state,
            (None, Some(embedding)) => &embedding.state,
            (None, None) => bail!("DeepSeek-V4 layer {layer} has no input state"),
        };
        if state.dtype() != DType::F32 || state.shape() != [1, hc, hidden] {
            bail!(
                "DeepSeek-V4 layer {layer} attention state must be F32 [1, {hc}, {hidden}], got {} {:?}",
                state.dtype(),
                state.shape()
            );
        }
        let AttentionWeightsRef {
            hc_fn,
            hc_base,
            hc_scale,
            attn_norm: attn_norm_weight,
            q_a: q_a_weight,
            q_a_norm: q_a_norm_weight,
            q_b: q_b_weight,
            kv: kv_weight,
            kv_norm: kv_norm_weight,
            sinks,
            output_a: output_a_weight,
            output_b: output_b_weight,
        } = AttentionWeightsRef::get(&self.weights, layer)?;

        let device = self.ctx.device().clone();
        let state_flat = state.with_shape(vec![1, hc_hidden])?;
        let state_norm = alloc(
            &device,
            DType::F32,
            vec![1, hc_hidden],
            "HC normalized state",
        )?;
        let mixes = alloc(&device, DType::F32, vec![1, mix_width], "HC mixes")?;
        let pre = alloc(&device, DType::F32, vec![1, hc], "HC pre weights")?;
        let post = alloc(&device, DType::F32, vec![1, hc], "HC post weights")?;
        let comb = alloc(&device, DType::F32, vec![1, hc, hc], "HC combination")?;
        let attn_input = alloc(&device, DType::F32, vec![1, hidden], "attention input")?;
        let attn_norm = alloc(&device, DType::F32, vec![1, hidden], "attention norm")?;
        let q_a = alloc(&device, DType::F32, vec![1, q_rank], "query rank")?;
        let q_a_norm = alloc(&device, DType::F32, vec![1, q_rank], "query rank norm")?;
        let q = alloc(&device, DType::F32, vec![1, heads, head_dim], "query heads")?;
        let q_norm = alloc(
            &device,
            DType::F32,
            vec![1, 1, heads, head_dim],
            "query head norm",
        )?;
        let kv = alloc(&device, DType::F32, vec![1, head_dim], "shared KV")?;
        let kv_norm = alloc(
            &device,
            DType::F32,
            vec![1, 1, 1, head_dim],
            "shared KV norm",
        )?;
        let q_rope = alloc(
            &device,
            DType::BF16,
            vec![1, 1, heads, head_dim],
            "rotated query",
        )?;
        let kv_rope = alloc(&device, DType::BF16, vec![1, 1, 1, head_dim], "rotated KV")?;
        let attention = alloc(
            &device,
            DType::BF16,
            vec![1, 1, heads, head_dim],
            "attention",
        )?;
        let attention_unrotated = alloc(
            &device,
            DType::BF16,
            vec![1, 1, heads, head_dim],
            "inverse-rotated attention",
        )?;
        let attention_f32 = alloc(
            &device,
            DType::F32,
            vec![1, heads, head_dim],
            "attention f32",
        )?;
        let output_a = alloc(&device, DType::F32, vec![1, output_a_width], "output A")?;
        let output_b = alloc(&device, DType::F32, vec![1, hidden], "output B")?;
        let output_state = alloc(&device, DType::F32, vec![1, hc, hidden], "HC output state")?;

        let hc_params = rms_params(&device, self.cfg.rms_norm_eps, hc_hidden, "HC RMS params")?;
        let hidden_params =
            rms_params(&device, self.cfg.rms_norm_eps, hidden, "hidden RMS params")?;
        let rank_params = rms_params(&device, self.cfg.rms_norm_eps, q_rank, "rank RMS params")?;
        let head_params = rms_params(&device, self.cfg.rms_norm_eps, head_dim, "head RMS params")?;
        let mut positions = alloc(&device, DType::U32, vec![1], "RoPE position")?;
        positions.as_mut_slice::<u32>()?[0] =
            u32::try_from(cache_step.position).context("DeepSeek-V4 position exceeds u32")?;
        let frequencies = yarn_frequencies(
            self.cfg.rope_head_dim as usize,
            0,
            self.cfg.rope_theta,
            1.0,
            self.cfg.yarn_beta_fast,
            self.cfg.yarn_beta_slow,
        )?;
        let mut frequency_buffer = alloc(
            &device,
            DType::F32,
            vec![frequencies.len()],
            "RoPE frequencies",
        )?;
        frequency_buffer
            .as_mut_slice::<f32>()?
            .copy_from_slice(&frequencies);
        let index_row = window_indices(window_capacity, 1, cache_step.position)?
            .into_iter()
            .next()
            .context("DeepSeek-V4 window index row missing")?;
        let mut indices = alloc(
            &device,
            DType::I32,
            vec![1, 1, index_row.len()],
            "attention indices",
        )?;
        indices.as_mut_slice::<i32>()?.copy_from_slice(&index_row);

        let (executor, registry) = self.ctx.split();
        let mut session = executor
            .begin()
            .with_context(|| format!("begin DeepSeek-V4 layer {layer} attention"))?;
        if let (Some(embedding_weight), Some(embedding)) =
            (embedding_weight.as_ref(), embedding.as_ref())
        {
            Self::encode_embedding_hyper_state(
                &mut session,
                registry,
                executor.device(),
                embedding_weight.buffer,
                embedding_weight.ggml_type,
                embedding_weight.shape,
                embedding,
                1,
                self.cfg.vocab_size as usize,
                hidden,
                self.cfg.hyper_connection_count,
            )?;
        }

        session.barrier_between(&[&state_flat], &[&state_norm]);
        session.rms_norm_no_scale_f32(
            registry,
            device.metal_device(),
            &state_flat,
            &state_norm,
            &hc_params,
            1,
            hc_hidden as u32,
        )?;
        raw_matmul(
            &mut session,
            registry,
            &device,
            &state_norm,
            &hc_fn,
            &mixes,
            1,
            mix_width,
            hc_hidden,
            "HC attention function",
        )?;
        session.barrier_between(&[&mixes, hc_scale, hc_base], &[&pre, &post, &comb]);
        dispatch_hc_split_sinkhorn(
            session.encoder_mut(),
            registry,
            &device,
            &mixes,
            hc_scale,
            hc_base,
            &pre,
            &post,
            &comb,
            1,
        )?;
        session.barrier_between(&[state, &pre], &[&attn_input]);
        dispatch_hc_pre(
            session.encoder_mut(),
            registry,
            &device,
            state,
            &pre,
            &attn_input,
            1,
            hidden as u32,
        )?;
        session.barrier_between(&[&attn_input, attn_norm_weight], &[&attn_norm]);
        session.rms_norm(
            registry,
            device.metal_device(),
            &attn_input,
            attn_norm_weight,
            &attn_norm,
            &hidden_params,
            1,
            hidden as u32,
        )?;
        raw_matmul(
            &mut session,
            registry,
            &device,
            &attn_norm,
            &q_a_weight,
            &q_a,
            1,
            q_rank,
            hidden,
            "query A",
        )?;
        raw_matmul(
            &mut session,
            registry,
            &device,
            &attn_norm,
            &kv_weight,
            &kv,
            1,
            head_dim,
            hidden,
            "shared KV",
        )?;
        session.barrier_between(&[&q_a, q_a_norm_weight], &[&q_a_norm]);
        session.rms_norm(
            registry,
            device.metal_device(),
            &q_a,
            q_a_norm_weight,
            &q_a_norm,
            &rank_params,
            1,
            q_rank as u32,
        )?;
        raw_matmul(
            &mut session,
            registry,
            &device,
            &q_a_norm,
            &q_b_weight,
            &q,
            1,
            heads * head_dim,
            q_rank,
            "query B",
        )?;
        session.barrier_between(&[&q], &[&q_norm]);
        session.rms_norm_no_scale_f32(
            registry,
            device.metal_device(),
            &q,
            &q_norm,
            &head_params,
            heads as u32,
            head_dim as u32,
        )?;
        session.barrier_between(&[&kv, kv_norm_weight], &[&kv_norm]);
        session.rms_norm(
            registry,
            device.metal_device(),
            &kv,
            kv_norm_weight,
            &kv_norm,
            &head_params,
            1,
            head_dim as u32,
        )?;
        let rope_params = DeepSeekTailRopeParams {
            batch: 1,
            seq_len: 1,
            heads: heads as u32,
            head_dim: head_dim as u32,
            rope_dim: self.cfg.rope_head_dim,
            inverse: 0,
        };
        session.barrier_between(&[&q_norm, &positions, &frequency_buffer], &[&q_rope]);
        dispatch_deepseek_tail_rope_f32_to_bf16(
            session.encoder_mut(),
            registry,
            &device,
            &q_norm,
            &positions,
            &frequency_buffer,
            &q_rope,
            &rope_params,
        )?;
        let kv_rope_params = DeepSeekTailRopeParams {
            heads: 1,
            ..rope_params
        };
        session.barrier_between(&[&kv_norm, &positions, &frequency_buffer], &[&kv_rope]);
        dispatch_deepseek_tail_rope_f32_to_bf16(
            session.encoder_mut(),
            registry,
            &device,
            &kv_norm,
            &positions,
            &frequency_buffer,
            &kv_rope,
            &kv_rope_params,
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
        let cache_view = layer_cache
            .window_kv
            .with_shape(vec![1, window_capacity, head_dim])?;
        session.barrier_between(&[&q_rope, &cache_view, sinks, &indices], &[&attention]);
        dispatch_deepseek_sparse_attention(
            session.encoder_mut(),
            registry,
            &device,
            &q_rope,
            &cache_view,
            sinks,
            &indices,
            &attention,
            &DeepSeekSparseAttentionParams {
                batch: 1,
                query_len: 1,
                kv_len: window_capacity as u32,
                top_k: index_row.len() as u32,
                heads: heads as u32,
                head_dim: head_dim as u32,
                scale: 1.0 / (head_dim as f32).sqrt(),
            },
        )?;
        let inverse_params = DeepSeekTailRopeParams {
            inverse: 1,
            ..rope_params
        };
        session.barrier_between(
            &[&attention, &positions, &frequency_buffer],
            &[&attention_unrotated],
        );
        dispatch_deepseek_tail_rope_bf16(
            session.encoder_mut(),
            registry,
            &device,
            &attention,
            &positions,
            &frequency_buffer,
            &attention_unrotated,
            &inverse_params,
        )?;
        session.barrier_between(&[&attention_unrotated], &[&attention_f32]);
        dispatch_cast_bf16_to_f32_with_encoder(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            &attention_unrotated,
            &attention_f32,
            (heads * head_dim) as u32,
        )?;

        grouped_output_a(
            &mut session,
            registry,
            &device,
            &attention_f32,
            &output_a_weight,
            &output_a,
            groups,
            o_rank,
            heads,
            head_dim,
        )?;
        raw_matmul(
            &mut session,
            registry,
            &device,
            &output_a,
            &output_b_weight,
            &output_b,
            1,
            hidden,
            output_a_width,
            "output B",
        )?;
        session.barrier_between(&[&output_b, state, &post, &comb], &[&output_state]);
        dispatch_hc_post(
            session.encoder_mut(),
            registry,
            &device,
            &output_b,
            state,
            &post,
            &comb,
            &output_state,
            1,
            hidden as u32,
        )?;
        session
            .finish()
            .with_context(|| format!("execute DeepSeek-V4 layer {layer} attention"))?;
        if commit_cache {
            cache
                .commit_step(cache_step.position)
                .context("publish DeepSeek-V4 cache transaction")?;
        }
        Ok(output_state)
    }
}
