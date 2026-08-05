//! Exact one-token, layer-0 DeepSeek-V4 attention slice.
//!
//! The entire slice is encoded into one Metal command buffer. Logical cache
//! visibility advances only after that command buffer completes successfully.

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
use mlx_native::ops::quantized_matmul_ggml::GgmlQuantizedMatmulParams;
use mlx_native::{DType, MlxBuffer};

use super::attention::window_indices;
use super::cache::Deepseek4Cache;
use super::forward_support::{alloc, raw_matmul, rms_params};
use super::rope::yarn_frequencies;
use super::Deepseek4Model;

impl Deepseek4Model {
    /// Execute the exact verifier attention graph for one token in layer 0.
    ///
    /// This is the first coherent vertical slice: Q2_K embedding, HC-pre,
    /// MLA projections and norms, interleaved tail-RoPE, BF16 cache write,
    /// sparse attention, inverse RoPE, grouped output projection, and HC-post.
    pub fn forward_layer0_attention_one(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        if self.cfg.compress_ratios.first().copied() != Some(0) {
            bail!("DeepSeek-V4 layer-0 attention slice requires official compression ratio 0");
        }
        let cache_step = cache
            .plan_next_step()
            .context("plan DeepSeek-V4 cache transaction")?;
        let layer_step = cache_step
            .layers
            .first()
            .context("missing layer-0 cache plan")?;
        let layer_cache = cache.layers().first().context("missing layer-0 cache")?;

        let hidden = self.cfg.hidden_size as usize;
        let hc = self.cfg.hyper_connection_count as usize;
        let hc_hidden = hc * hidden;
        let mix_width = (2 + hc) * hc;
        let heads = self.cfg.num_attention_heads as usize;
        let head_dim = self.cfg.head_dim as usize;
        let q_rank = self.cfg.q_lora_rank as usize;
        let o_rank = self.cfg.o_lora_rank as usize;
        let groups = self.cfg.output_groups as usize;
        let group_width = heads * head_dim / groups;
        let output_a_width = groups * o_rank;
        let window_capacity = layer_cache.window_kv.shape()[0];
        if layer_cache.window_kv.shape() != [window_capacity, head_dim] {
            bail!("DeepSeek-V4 layer-0 window cache shape drift");
        }

        let embedding_weight = self.weights.raw_matrix_ref("token_embd.weight")?;
        let hc_fn = self.weights.raw_matrix_ref("blk.0.hc_attn_fn.weight")?;
        let hc_base = self.weights.f32_state("blk.0.hc_attn_base.weight")?;
        let hc_scale = self.weights.f32_state("blk.0.hc_attn_scale.weight")?;
        let attn_norm_weight = self.weights.f32_state("blk.0.attn_norm.weight")?;
        let q_a_weight = self.weights.raw_matrix_ref("blk.0.attn_q_a.weight")?;
        let q_a_norm_weight = self.weights.f32_state("blk.0.attn_q_a_norm.weight")?;
        let q_b_weight = self.weights.raw_matrix_ref("blk.0.attn_q_b.weight")?;
        let kv_weight = self.weights.raw_matrix_ref("blk.0.attn_kv.weight")?;
        let kv_norm_weight = self.weights.f32_state("blk.0.attn_kv_a_norm.weight")?;
        let sinks = self.weights.f32_state("blk.0.attn_sinks.weight")?;
        let output_a_weight = self.weights.raw_matrix_ref("blk.0.attn_output_a.weight")?;
        let output_b_weight = self.weights.raw_matrix_ref("blk.0.attn_output_b.weight")?;

        let embedding = self.prepare_embedding_arena(&[token_id])?;
        let device = self.ctx.device().clone();
        let state_flat = embedding.state.with_shape(vec![1, hc_hidden])?;
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
            .context("begin DeepSeek-V4 layer-0 attention")?;
        Self::encode_embedding_hyper_state(
            &mut session,
            registry,
            executor.device(),
            embedding_weight.buffer,
            embedding_weight.ggml_type,
            embedding_weight.shape,
            &embedding,
            1,
            self.cfg.vocab_size as usize,
            hidden,
            self.cfg.hyper_connection_count,
        )?;

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
        session.barrier_between(&[&embedding.state, &pre], &[&attn_input]);
        dispatch_hc_pre(
            session.encoder_mut(),
            registry,
            &device,
            &embedding.state,
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

        if output_a_weight.shape != [output_a_width, group_width] {
            bail!(
                "DeepSeek-V4 output-A weight shape drift: got {:?}",
                output_a_weight.shape
            );
        }
        if output_a_weight.buffer.dtype() != DType::U8 {
            bail!("DeepSeek-V4 output-A grouped projection requires block-quantized storage");
        }
        let block = output_a_weight.ggml_type.block_values() as usize;
        if group_width % block != 0 {
            bail!("DeepSeek-V4 output-A group width is not block aligned");
        }
        let row_bytes = group_width / block * output_a_weight.ggml_type.block_bytes() as usize;
        for group in 0..groups {
            let input_view = attention_f32
                .slice_view(
                    (group * group_width * DType::F32.size_of()) as u64,
                    group_width,
                )
                .with_shape(vec![1, group_width])?;
            let weight_view = output_a_weight
                .buffer
                .slice_view((group * o_rank * row_bytes) as u64, o_rank * row_bytes);
            let output_view = output_a
                .slice_view((group * o_rank * DType::F32.size_of()) as u64, o_rank)
                .with_shape(vec![1, o_rank])?;
            session.barrier_between(&[&input_view, &weight_view], &[&output_view]);
            session.quantized_matmul_ggml(
                registry,
                &device,
                &input_view,
                &weight_view,
                &output_view,
                &GgmlQuantizedMatmulParams {
                    m: 1,
                    n: o_rank as u32,
                    k: group_width as u32,
                    ggml_type: output_a_weight.ggml_type,
                },
            )?;
        }
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
        session.barrier_between(
            &[&output_b, &embedding.state, &post, &comb],
            &[&output_state],
        );
        dispatch_hc_post(
            session.encoder_mut(),
            registry,
            &device,
            &output_b,
            &embedding.state,
            &post,
            &comb,
            &output_state,
            1,
            hidden as u32,
        )?;
        session
            .finish()
            .context("execute DeepSeek-V4 layer-0 attention")?;
        cache
            .commit_step(cache_step.position)
            .context("publish DeepSeek-V4 cache transaction")?;
        Ok(output_state)
    }
}
