//! Exact one-token uncompressed DeepSeek-V4 attention layers.
use super::attention::window_indices;
use super::attention_weights::AttentionWeightsRef;
use super::cache::{CacheSpan, Deepseek4Cache};
use super::forward_support::{
    alloc, grouped_output_a, grouped_output_a_batched, raw_matmul, rms_params,
    BatchedGroupedOutputArena,
};
use super::prefill_flash_attention::{encode_deepseek_flash_prefill, DeepseekPrefillFlashArena};
use super::rope::yarn_frequencies;
use super::submission::{finish_or_commit, SubmissionChain};
use super::Deepseek4Model;
use anyhow::{bail, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_activation_quant::{
    dispatch_deepseek_mxfp8_fake_quant_bf16, DeepSeekMxfp8Params,
};
use mlx_native::ops::deepseek_hyper_connection::{
    dispatch_hc_post, dispatch_hc_pre, dispatch_hc_split_sinkhorn,
};
use mlx_native::ops::deepseek_sparse_attention::{
    dispatch_deepseek_sparse_attention, DeepSeekSparseAttentionParams,
};
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_bf16, dispatch_deepseek_tail_rope_f16_to_bf16,
    dispatch_deepseek_tail_rope_f32_to_bf16, dispatch_deepseek_tail_rope_f32_to_f16,
    DeepSeekTailRopeParams,
};
use mlx_native::ops::elementwise::{cast, dispatch_cast_bf16_to_f32_with_encoder, CastDirection};
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy;
use mlx_native::{DType, GraphExecutor, MlxBuffer};

impl Deepseek4Model {
    /// Execute one uncompressed verifier attention layer; `state=None` embeds layer 0.
    pub(super) fn forward_uncompressed_attention_one(
        &mut self,
        state: Option<&MlxBuffer>,
        token_id: u32,
        layer: usize,
        cache: &mut Deepseek4Cache,
        commit_cache: bool,
        in_flight: Option<&mut SubmissionChain>,
        shared_session: Option<&mut GraphSession<'_>>,
    ) -> Result<MlxBuffer> {
        self.forward_uncompressed_attention_rows(
            state,
            &[token_id],
            layer,
            cache,
            commit_cache,
            None,
            in_flight,
            shared_session,
        )
    }

    pub(super) fn forward_uncompressed_attention_prefill(
        &mut self,
        state: Option<&MlxBuffer>,
        token_ids: &[u32],
        layer: usize,
        cache: &mut Deepseek4Cache,
        span: &CacheSpan,
        in_flight: Option<&mut SubmissionChain>,
        shared_session: Option<&mut GraphSession<'_>>,
    ) -> Result<MlxBuffer> {
        self.forward_uncompressed_attention_rows(
            state,
            token_ids,
            layer,
            cache,
            false,
            Some(span),
            in_flight,
            shared_session,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_uncompressed_attention_rows(
        &mut self,
        state: Option<&MlxBuffer>,
        token_ids: &[u32],
        layer: usize,
        cache: &mut Deepseek4Cache,
        commit_cache: bool,
        prefill_span: Option<&CacheSpan>,
        in_flight: Option<&mut SubmissionChain>,
        shared_session: Option<&mut GraphSession<'_>>,
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
        let rows = token_ids.len();
        if rows == 0 {
            bail!("DeepSeek-V4 attention requires at least one token row");
        }
        let cache_step = prefill_span
            .is_none()
            .then(|| {
                cache
                    .plan_next_step()
                    .context("plan DeepSeek-V4 cache transaction")
            })
            .transpose()?;
        let start_position = prefill_span
            .map(|span| span.start_position)
            .or_else(|| cache_step.as_ref().map(|step| step.position))
            .context("DeepSeek-V4 attention cache position is missing")?;
        let layer_step = cache_step
            .as_ref()
            .map(|step| {
                step.layers
                    .get(layer)
                    .with_context(|| format!("missing layer-{layer} cache plan"))
            })
            .transpose()?;
        let layer_span = prefill_span
            .map(|span| {
                if span.token_count != rows {
                    bail!(
                        "DeepSeek-V4 prefill span has {} rows, attention received {rows}",
                        span.token_count
                    );
                }
                span.layers
                    .get(layer)
                    .with_context(|| format!("missing layer-{layer} prefill cache span"))
            })
            .transpose()?;
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
        let use_f16_prefill = rows > 1;
        let o_rank = self.cfg.o_lora_rank as usize;
        let groups = self.cfg.output_groups as usize;
        let output_a_width = groups * o_rank;
        let window_capacity = layer_cache.window_kv.shape()[0];
        if layer_cache.window_kv.shape() != [window_capacity, head_dim] {
            bail!("DeepSeek-V4 layer-0 window cache shape drift");
        }

        let embedding = state
            .is_none()
            .then(|| self.prepare_embedding_arena(token_ids))
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
        if state.dtype() != DType::F32 || state.shape() != [rows, hc, hidden] {
            bail!(
                "DeepSeek-V4 layer {layer} attention state must be F32 [{rows}, {hc}, {hidden}], got {} {:?}",
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
        let state_flat = state.with_shape(vec![rows, hc_hidden])?;
        let state_norm = alloc(
            &device,
            DType::F32,
            vec![rows, hc_hidden],
            "HC normalized state",
        )?;
        let mixes = alloc(&device, DType::F32, vec![rows, mix_width], "HC mixes")?;
        let pre = alloc(&device, DType::F32, vec![rows, hc], "HC pre weights")?;
        let post = alloc(&device, DType::F32, vec![rows, hc], "HC post weights")?;
        let comb = alloc(&device, DType::F32, vec![rows, hc, hc], "HC combination")?;
        let attn_input = alloc(&device, DType::F32, vec![rows, hidden], "attention input")?;
        let attn_norm = alloc(&device, DType::F32, vec![rows, hidden], "attention norm")?;
        let q_a = alloc(&device, DType::F32, vec![rows, q_rank], "query rank")?;
        let q_a_norm = alloc(&device, DType::F32, vec![rows, q_rank], "query rank norm")?;
        let q = alloc(
            &device,
            DType::F32,
            vec![rows, heads, head_dim],
            "query heads",
        )?;
        let q_norm = alloc(
            &device,
            DType::F32,
            vec![1, rows, heads, head_dim],
            "query head norm",
        )?;
        let kv = alloc(&device, DType::F32, vec![rows, head_dim], "shared KV")?;
        let kv_norm = alloc(
            &device,
            DType::F32,
            vec![1, rows, 1, head_dim],
            "shared KV norm",
        )?;
        let q_rope = alloc(
            &device,
            if use_f16_prefill {
                DType::F16
            } else {
                DType::BF16
            },
            vec![1, rows, heads, head_dim],
            "rotated query",
        )?;
        let kv_rope = alloc(
            &device,
            if use_f16_prefill {
                DType::F16
            } else {
                DType::BF16
            },
            vec![1, rows, 1, head_dim],
            "rotated KV",
        )?;
        let kv_cache_source = use_f16_prefill
            .then(|| {
                alloc(
                    &device,
                    DType::BF16,
                    vec![1, rows, 1, head_dim],
                    "rotated KV cache source",
                )
            })
            .transpose()?;
        let attention = alloc(
            &device,
            if use_f16_prefill {
                DType::F16
            } else {
                DType::BF16
            },
            vec![1, rows, heads, head_dim],
            "attention",
        )?;
        let attention_unrotated = alloc(
            &device,
            DType::BF16,
            vec![1, rows, heads, head_dim],
            "inverse-rotated attention",
        )?;
        let attention_f32 = alloc(
            &device,
            DType::F32,
            vec![rows, heads, head_dim],
            "attention f32",
        )?;
        let output_a = alloc(&device, DType::F32, vec![rows, output_a_width], "output A")?;
        let output_b = alloc(&device, DType::F32, vec![rows, hidden], "output B")?;
        let output_state = alloc(
            &device,
            DType::F32,
            vec![rows, hc, hidden],
            "HC output state",
        )?;
        let grouped_output = (rows > 1)
            .then(|| {
                BatchedGroupedOutputArena::new(
                    &device,
                    rows,
                    groups,
                    heads * head_dim / groups,
                    o_rank,
                )
            })
            .transpose()?;
        let prefill_flash = (rows > 1)
            .then(|| DeepseekPrefillFlashArena::new(&device, rows, heads, head_dim, rows))
            .transpose()?;

        let hc_params = rms_params(&device, self.cfg.rms_norm_eps, hc_hidden, "HC RMS params")?;
        let hidden_params =
            rms_params(&device, self.cfg.rms_norm_eps, hidden, "hidden RMS params")?;
        let rank_params = rms_params(&device, self.cfg.rms_norm_eps, q_rank, "rank RMS params")?;
        let head_params = rms_params(&device, self.cfg.rms_norm_eps, head_dim, "head RMS params")?;
        let mut positions = alloc(&device, DType::U32, vec![rows], "RoPE positions")?;
        for (offset, position) in positions
            .as_logical_mut_slice::<u32>()?
            .iter_mut()
            .enumerate()
        {
            *position = u32::try_from(start_position + offset)
                .context("DeepSeek-V4 position exceeds u32")?;
        }
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
            .as_logical_mut_slice::<f32>()?
            .copy_from_slice(&frequencies);
        let index_rows = window_indices(window_capacity, rows, start_position)?;
        let index_width = index_rows
            .first()
            .context("DeepSeek-V4 window index row missing")?
            .len();
        let mut indices = alloc(
            &device,
            DType::I32,
            vec![1, rows, index_width],
            "attention indices",
        )?;
        for (destination, row) in indices
            .as_logical_mut_slice::<i32>()?
            .chunks_exact_mut(index_width)
            .zip(&index_rows)
        {
            destination.copy_from_slice(row);
        }
        let rows_u32 = u32::try_from(rows).context("DeepSeek-V4 attention rows exceed u32")?;

        let registry = &mut self.ctx.registry;
        let mut encode = |session: &mut GraphSession<'_>| -> Result<()> {
            if let (Some(embedding_weight), Some(embedding)) =
                (embedding_weight.as_ref(), embedding.as_ref())
            {
                Self::encode_embedding_hyper_state(
                    session,
                    registry,
                    &device,
                    embedding_weight.buffer,
                    embedding_weight.ggml_type,
                    embedding_weight.shape,
                    embedding,
                    rows,
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
                rows_u32,
                hc_hidden as u32,
            )?;
            raw_matmul(
                session,
                registry,
                &device,
                &state_norm,
                &hc_fn,
                &mixes,
                rows,
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
                rows_u32,
            )?;
            session.barrier_between(&[state, &pre], &[&attn_input]);
            dispatch_hc_pre(
                session.encoder_mut(),
                registry,
                &device,
                state,
                &pre,
                &attn_input,
                rows_u32,
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
                rows_u32,
                hidden as u32,
            )?;
            raw_matmul(
                session,
                registry,
                &device,
                &attn_norm,
                &q_a_weight,
                &q_a,
                rows,
                q_rank,
                hidden,
                "query A",
            )?;
            raw_matmul(
                session,
                registry,
                &device,
                &attn_norm,
                &kv_weight,
                &kv,
                rows,
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
                rows_u32,
                q_rank as u32,
            )?;
            raw_matmul(
                session,
                registry,
                &device,
                &q_a_norm,
                &q_b_weight,
                &q,
                rows,
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
                (rows * heads) as u32,
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
                rows_u32,
                head_dim as u32,
            )?;
            let rope_params = DeepSeekTailRopeParams {
                batch: 1,
                seq_len: rows_u32,
                heads: heads as u32,
                head_dim: head_dim as u32,
                rope_dim: self.cfg.rope_head_dim,
                inverse: 0,
            };
            session.barrier_between(&[&q_norm, &positions, &frequency_buffer], &[&q_rope]);
            if use_f16_prefill {
                dispatch_deepseek_tail_rope_f32_to_f16(
                    session.encoder_mut(),
                    registry,
                    &device,
                    &q_norm,
                    &positions,
                    &frequency_buffer,
                    &q_rope,
                    &rope_params,
                )?;
            } else {
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
            }
            let kv_rope_params = DeepSeekTailRopeParams {
                heads: 1,
                ..rope_params
            };
            session.barrier_between(&[&kv_norm, &positions, &frequency_buffer], &[&kv_rope]);
            if use_f16_prefill {
                dispatch_deepseek_tail_rope_f32_to_f16(
                    session.encoder_mut(),
                    registry,
                    &device,
                    &kv_norm,
                    &positions,
                    &frequency_buffer,
                    &kv_rope,
                    &kv_rope_params,
                )?;
            } else {
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
            }

            // Official DeepSeek-V4 QAT applies block-64 E4M3/E8M0 fake
            // quantization to the non-RoPE KV prefix in every attention
            // layer, including the ratio-zero prefix/suffix layers.
            if !use_f16_prefill {
                session.barrier_between(&[&kv_rope], &[&kv_rope]);
                dispatch_deepseek_mxfp8_fake_quant_bf16(
                    session.encoder_mut(),
                    registry,
                    &device,
                    &kv_rope,
                    &DeepSeekMxfp8Params {
                        rows: rows_u32,
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
                u32::try_from(
                    layer_span
                        .map(|span| span.window_write_start)
                        .or_else(|| layer_step.map(|step| step.window_write_slot))
                        .context("DeepSeek-V4 window cache write slot is missing")?,
                )
                .context("DeepSeek-V4 window write slot exceeds u32")?,
                head_dim as u32,
                rows_u32,
                window_capacity as u32,
                true,
            )?;
            if let Some(prefill_flash) = prefill_flash.as_ref() {
                encode_deepseek_flash_prefill(
                    session,
                    registry,
                    &device,
                    &q_rope,
                    &kv_rope,
                    None,
                    sinks,
                    &indices,
                    &attention,
                    prefill_flash,
                    rows,
                    rows,
                    rows,
                    index_width,
                    heads,
                    head_dim,
                    1.0 / (head_dim as f32).sqrt(),
                )?;
            } else {
                let cache_view =
                    layer_cache
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
                        query_len: rows_u32,
                        kv_len: window_capacity as u32,
                        top_k: index_width as u32,
                        heads: heads as u32,
                        head_dim: head_dim as u32,
                        scale: 1.0 / (head_dim as f32).sqrt(),
                    },
                )?;
            }
            let inverse_params = DeepSeekTailRopeParams {
                inverse: 1,
                ..rope_params
            };
            session.barrier_between(
                &[&attention, &positions, &frequency_buffer],
                &[&attention_unrotated],
            );
            if use_f16_prefill {
                dispatch_deepseek_tail_rope_f16_to_bf16(
                    session.encoder_mut(),
                    registry,
                    &device,
                    &attention,
                    &positions,
                    &frequency_buffer,
                    &attention_unrotated,
                    &inverse_params,
                )?;
            } else {
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
            }
            session.barrier_between(&[&attention_unrotated], &[&attention_f32]);
            dispatch_cast_bf16_to_f32_with_encoder(
                session.encoder_mut(),
                registry,
                device.metal_device(),
                &attention_unrotated,
                &attention_f32,
                (rows * heads * head_dim) as u32,
            )?;

            if let Some(grouped_output) = grouped_output.as_ref() {
                grouped_output_a_batched(
                    session,
                    registry,
                    &device,
                    &attention_f32,
                    &output_a_weight,
                    &output_a,
                    grouped_output,
                    rows,
                    groups,
                    o_rank,
                    heads,
                    head_dim,
                )?;
            } else {
                grouped_output_a(
                    session,
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
            }
            raw_matmul(
                session,
                registry,
                &device,
                &output_a,
                &output_b_weight,
                &output_b,
                rows,
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
                rows_u32,
                hidden as u32,
            )?;
            Ok(())
        };
        if let Some(session) = shared_session {
            encode(session)?;
        } else {
            let local_executor = GraphExecutor::new(device.clone());
            let mut session = local_executor
                .begin()
                .with_context(|| format!("begin DeepSeek-V4 layer {layer} attention"))?;
            encode(&mut session)?;
            finish_or_commit(
                session,
                in_flight,
                format!("execute DeepSeek-V4 layer {layer} uncompressed attention"),
            )?;
        }
        if commit_cache {
            cache
                .commit_step(
                    cache_step
                        .as_ref()
                        .context("DeepSeek-V4 decode cache step is missing")?
                        .position,
                )
                .context("publish DeepSeek-V4 cache transaction")?;
        }
        Ok(output_state)
    }
}
