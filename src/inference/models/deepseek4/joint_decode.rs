//! Layer-major joint decode over independent DeepSeek-V4 request caches.
//!
//! One transaction advances `B` requests, each with its own cache, position,
//! and token, by one token. Row-parallel work runs once with `m = B`: the
//! attention hyper-connection prelude and Q8_0 projections, inverse RoPE and
//! output projections, the complete FFN (router, routed experts, shared
//! expert), and — at the caller — the output head. Work that reads or writes
//! a request cache (window write, compressor and indexer recurrences, sparse
//! attention) runs per row against that row's cache and position. Every row is
//! bit-identical to `forward_verifier_one` on the same cache; see
//! `row_matmul` for the matmul half of that contract.

use anyhow::{ensure, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_activation_quant::{
    dispatch_deepseek_mxfp8_fake_quant_bf16, DeepSeekMxfp8Params,
};
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_f32_to_bf16, DeepSeekTailRopeParams,
};
use mlx_native::ops::quantized_matmul_ggml::MM_ROUTING_THRESHOLD;
use mlx_native::{DType, GraphExecutor, MlxBuffer, MlxDevice};

use super::attention_weights::AttentionWeightsRef;
use super::cache::{CacheStep, Deepseek4Cache};
use super::compressed_attention_common::{
    encode_compressed_attention_epilogue, encode_compressed_attention_prelude,
    CompressedAttentionCoreArena,
};
use super::compressed_attention_weights::CompressedAttentionWeightsRef;
use super::decode_cohort::publish_verifier_cohort_after_gate;
use super::forward_support::{
    alloc, alloc_host_input, begin_decode_pool_token, end_decode_pool_token,
};
use super::joint_decode_core::{
    encode_compressed_row, encode_window_row, plan_compressed_row, plan_window_row,
    CompressedRowCore, JointRowInputs, WindowRowCore,
};
use super::rope::yarn_frequencies;
use super::row_matmul::{row_view, RowMatmul};
use super::submission::{drain, retained_reference_pipeline_enabled, SubmissionChain};
use super::verifier_forward::decode_layers_per_command_buffer;
use super::{Deepseek4Config, Deepseek4Model};

/// Widest joint step whose dense products stay on the per-row matvec route.
/// Serving caps the default width lower; see the scheduler's row cap.
pub(crate) const MAX_JOINT_DECODE_ROWS: usize = MM_ROUTING_THRESHOLD as usize;

impl Deepseek4Model {
    /// `None` when joint decode is admissible for this artifact and process,
    /// otherwise the reason it must stay serial. Diagnostic layer dumps and
    /// stage profiling change the one-token graph's submission shape,
    /// unretained Metal references require the synchronous serial path,
    /// `HF2Q_DECODE_MV_EXT=1` moves Q8_0 at 2..=8 rows onto the non-exact
    /// `mul_mv_ext` kernel, and unproven matrix storage could make a row
    /// depend on its batch peers.
    pub(crate) fn joint_decode_unavailable_reason(&self) -> Option<String> {
        if !retained_reference_pipeline_enabled() {
            return Some("MLX_UNRETAINED_REFS requires the synchronous serial path".into());
        }
        for name in [
            "HF2Q_DEEPSEEK_DUMP_LAYER_DIR",
            "HF2Q_DEEPSEEK_DUMP_ATTENTION_DIR",
        ] {
            if std::env::var_os(name).is_some() {
                return Some(format!("{name} diagnostics require the serial graph"));
            }
        }
        if std::env::var("HF2Q_DEEPSEEK_STAGE_PROFILE").as_deref() == Ok("1") {
            return Some("HF2Q_DEEPSEEK_STAGE_PROFILE=1 requires the serial graph".into());
        }
        if std::env::var("HF2Q_DECODE_MV_EXT").as_deref() == Ok("1") {
            return Some(
                "HF2Q_DECODE_MV_EXT=1 routes Q8_0 at 2..=8 rows through non-exact mul_mv_ext"
                    .into(),
            );
        }
        self.joint_route_unproven.clone()
    }
}

/// RoPE tables for both attention families, built once per joint step.
struct JointRopeFrequencies {
    window: MlxBuffer,
    compressed: MlxBuffer,
}

impl JointRopeFrequencies {
    fn new(device: &MlxDevice, cfg: &Deepseek4Config) -> Result<Self> {
        let table = |values: Vec<f32>, label: &str| -> Result<MlxBuffer> {
            let mut buffer = alloc_host_input(device, DType::F32, vec![values.len()], label)?;
            buffer
                .as_logical_mut_slice::<f32>()?
                .copy_from_slice(&values);
            Ok(buffer)
        };
        Ok(Self {
            window: table(
                yarn_frequencies(
                    cfg.rope_head_dim as usize,
                    0,
                    cfg.rope_theta,
                    1.0,
                    cfg.yarn_beta_fast,
                    cfg.yarn_beta_slow,
                )?,
                "joint RoPE frequencies",
            )?,
            compressed: table(
                yarn_frequencies(
                    cfg.rope_head_dim as usize,
                    cfg.original_context_length as usize,
                    cfg.compress_rope_theta,
                    cfg.rope_factor,
                    cfg.yarn_beta_fast,
                    cfg.yarn_beta_slow,
                )?,
                "joint compressed RoPE frequencies",
            )?,
        })
    }
}

enum JointRowCore {
    Window(WindowRowCore),
    Compressed(CompressedRowCore),
}

impl Deepseek4Model {
    /// Advance `token_ids.len()` independent caches by one token each in one
    /// layer-major transaction and return their `[B, hc, hidden]` final
    /// states in input order. Caches publish only after every command buffer
    /// drains and `commit_gate` accepts; a submitted failure poisons them all.
    pub(crate) fn forward_verifier_decode_joint_with_commit_gate(
        &mut self,
        token_ids: &[u32],
        caches: &mut [&mut Deepseek4Cache],
        commit_gate: impl FnOnce() -> Result<()>,
    ) -> Result<MlxBuffer> {
        let rows = token_ids.len();
        ensure!(
            (2..=MAX_JOINT_DECODE_ROWS).contains(&rows) && caches.len() == rows,
            "DeepSeek-V4 joint decode requires 2..={MAX_JOINT_DECODE_ROWS} aligned rows, got {rows} tokens for {} caches",
            caches.len()
        );
        if let Some(reason) = self.joint_decode_unavailable_reason() {
            anyhow::bail!("DeepSeek-V4 joint decode unavailable: {reason}");
        }
        let steps = caches
            .iter()
            .map(|cache| {
                cache
                    .plan_next_step()
                    .context("plan DeepSeek-V4 joint decode cache transaction")
            })
            .collect::<Result<Vec<_>>>()?;
        let positions = steps.iter().map(|step| step.position).collect::<Vec<_>>();

        begin_decode_pool_token();
        let (result, submitted_any) =
            self.forward_verifier_decode_joint_uncommitted(token_ids, caches, &steps);
        end_decode_pool_token();
        match result {
            Ok(state) => {
                publish_verifier_cohort_after_gate(caches, &positions, commit_gate)?;
                Ok(state)
            }
            Err(error) => {
                if submitted_any {
                    for cache in caches.iter_mut() {
                        cache.poison();
                    }
                }
                Err(error).context("DeepSeek-V4 joint decode transaction failed")
            }
        }
    }

    fn forward_verifier_decode_joint_uncommitted(
        &mut self,
        token_ids: &[u32],
        caches: &[&mut Deepseek4Cache],
        steps: &[CacheStep],
    ) -> (Result<MlxBuffer>, bool) {
        let mut submitted_any = false;
        let result = (|| -> Result<MlxBuffer> {
            let layers = self.cfg.num_hidden_layers as usize;
            ensure!(layers > 0, "DeepSeek-V4 joint decode encoded zero layers");
            let layers_per_command_buffer = decode_layers_per_command_buffer(layers)?;
            let device = self.ctx.device().clone();
            let executor = GraphExecutor::new(device.clone());
            let frequencies = JointRopeFrequencies::new(&device, &self.cfg)?;
            let mut positions = alloc_host_input(
                &device,
                DType::U32,
                vec![token_ids.len()],
                "joint decode positions",
            )?;
            for (destination, step) in positions
                .as_logical_mut_slice::<u32>()?
                .iter_mut()
                .zip(steps)
            {
                *destination = u32::try_from(step.position)
                    .context("DeepSeek-V4 joint decode position exceeds u32")?;
            }
            let mut state: Option<MlxBuffer> = None;
            let mut in_flight =
                SubmissionChain::with_capacity(layers.div_ceil(layers_per_command_buffer));
            let encode_result = (|| -> Result<()> {
                for start in (0..layers).step_by(layers_per_command_buffer) {
                    let end = (start + layers_per_command_buffer).min(layers);
                    let mut session = executor
                        .begin()
                        .map(crate::inference::native_observation::session)
                        .with_context(|| {
                            format!("begin DeepSeek-V4 joint decode layers {start}..{end}")
                        })?;
                    for layer in start..end {
                        let attention = self
                            .encode_joint_attention_layer(
                                &mut session,
                                layer,
                                state.as_ref(),
                                token_ids,
                                caches,
                                steps,
                                &positions,
                                &frequencies,
                            )
                            .with_context(|| {
                                format!("encode DeepSeek-V4 joint layer-{layer} attention")
                            })?;
                        state = Some(
                            self.forward_ffn_rows(
                                &attention,
                                token_ids,
                                layer,
                                Some(&mut in_flight),
                                Some(&mut session),
                                None,
                                None,
                                None,
                                RowMatmul::IndependentRows,
                            )
                            .with_context(|| {
                                format!("encode DeepSeek-V4 joint layer-{layer} FFN")
                            })?,
                        );
                    }
                    submitted_any = true;
                    in_flight.push((
                        format!("execute DeepSeek-V4 joint decode layers {start}..{end}"),
                        session.commit(),
                    ));
                }
                Ok(())
            })();
            let drained = drain(&in_flight).context("drain DeepSeek-V4 joint decode pipeline");
            drop(in_flight);
            match (encode_result, drained) {
                (Ok(()), Ok(())) => state.context("DeepSeek-V4 joint decode produced no state"),
                (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
                (Err(error), Err(drain_error)) => Err(error).context(format!(
                    "joint decode pipeline drain also failed: {drain_error:#}"
                )),
            }
        })();
        (result, submitted_any)
    }

    /// Encode one attention layer for every joint row. Mirrors the one-token
    /// graph: the uncompressed path's inline HC/QKV prelude and output
    /// epilogue are the same dispatch sequence as the compressed helpers.
    #[allow(clippy::too_many_arguments)]
    fn encode_joint_attention_layer(
        &mut self,
        session: &mut GraphSession<'_>,
        layer: usize,
        state: Option<&MlxBuffer>,
        token_ids: &[u32],
        caches: &[&mut Deepseek4Cache],
        steps: &[CacheStep],
        positions: &MlxBuffer,
        frequencies: &JointRopeFrequencies,
    ) -> Result<MlxBuffer> {
        let rows = token_ids.len();
        let ratio = self
            .cfg
            .compress_ratios
            .get(layer)
            .copied()
            .with_context(|| format!("missing DeepSeek-V4 layer-{layer} compression ratio"))?;
        ensure!(
            matches!(ratio, 0 | 4 | 128),
            "DeepSeek-V4 layer {layer} has unsupported compression ratio {ratio}"
        );
        let hidden = self.cfg.hidden_size as usize;
        let hc = self.cfg.hyper_connection_count as usize;
        let heads = self.cfg.num_attention_heads as usize;
        let head_dim = self.cfg.head_dim as usize;
        let q_rank = self.cfg.q_lora_rank as usize;
        let device = self.ctx.device().clone();

        let embedding = match (layer, state) {
            (0, None) => {
                ensure!(ratio == 0, "DeepSeek-V4 layer 0 must be uncompressed");
                Some(self.prepare_embedding_arena(token_ids)?)
            }
            (0, Some(_)) => anyhow::bail!("DeepSeek-V4 joint layer 0 must embed each row"),
            (_, None) => anyhow::bail!("DeepSeek-V4 joint layer {layer} is missing its state"),
            (_, Some(_)) => None,
        };
        let embedding_weight = embedding
            .as_ref()
            .map(|_| self.weights.raw_matrix_ref("token_embd.weight"))
            .transpose()?;
        let state = match (state, embedding.as_ref()) {
            (Some(state), _) => state,
            (None, Some(embedding)) => &embedding.state,
            (None, None) => anyhow::bail!("DeepSeek-V4 joint layer {layer} has no input state"),
        };
        ensure!(
            state.dtype() == DType::F32 && state.shape() == [rows, hc, hidden],
            "DeepSeek-V4 joint layer {layer} state must be F32 [{rows}, {hc}, {hidden}], got {} {:?}",
            state.dtype(),
            state.shape()
        );

        let compressed_weights = (ratio != 0)
            .then(|| CompressedAttentionWeightsRef::get(&self.weights, layer, ratio))
            .transpose()?;
        let window_weights = (ratio == 0)
            .then(|| AttentionWeightsRef::get(&self.weights, layer))
            .transpose()?;
        let attention_weights = match (compressed_weights.as_ref(), window_weights.as_ref()) {
            (Some(weights), _) => &weights.attention,
            (None, Some(weights)) => weights,
            (None, None) => anyhow::bail!("DeepSeek-V4 joint layer {layer} has no weights"),
        };
        let frequency_buffer = if ratio == 0 {
            &frequencies.window
        } else {
            &frequencies.compressed
        };

        let core = CompressedAttentionCoreArena::new(&device, &self.cfg, rows)?;
        let q_rope = alloc(
            &device,
            DType::BF16,
            vec![1, rows, heads, head_dim],
            "joint rotated query",
        )?;
        let kv_rope = alloc(
            &device,
            DType::BF16,
            vec![1, rows, 1, head_dim],
            "joint rotated KV",
        )?;
        let attention = alloc(
            &device,
            DType::BF16,
            vec![1, rows, heads, head_dim],
            "joint sparse attention",
        )?;
        let row_cores = (0..rows)
            .map(|row| {
                let step = &steps[row];
                let planned = if ratio == 0 {
                    plan_window_row(&device, caches[row], layer, step.position, head_dim)
                        .map(JointRowCore::Window)
                } else {
                    let layer_step = step
                        .layers
                        .get(layer)
                        .with_context(|| format!("missing layer-{layer} joint cache step"))?;
                    plan_compressed_row(
                        &device,
                        &self.cfg,
                        caches[row],
                        layer,
                        ratio,
                        step.position,
                        layer_step,
                    )
                    .map(JointRowCore::Compressed)
                };
                planned.with_context(|| format!("plan DeepSeek-V4 joint layer-{layer} row {row}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let row_inputs = (0..rows)
            .map(|row| {
                Ok(JointRowInputs {
                    q_rope: row_view(&q_rope, row, vec![1, 1, heads, head_dim])?,
                    kv_rope: row_view(&kv_rope, row, vec![1, 1, 1, head_dim])?,
                    attention: row_view(&attention, row, vec![1, 1, heads, head_dim])?,
                    attn_norm: row_view(&core.attn_norm, row, vec![1, hidden])?,
                    q_a_norm: row_view(&core.q_a_norm, row, vec![1, q_rank])?,
                    position: row_view(positions, row, vec![1])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let cfg = &self.cfg;
        let registry = &mut self.ctx.registry;
        if let (Some(weight), Some(embedding)) = (embedding_weight.as_ref(), embedding.as_ref()) {
            Self::encode_embedding_hyper_state(
                session,
                registry,
                &device,
                weight.buffer,
                weight.ggml_type,
                weight.shape,
                embedding,
                rows,
                cfg.vocab_size as usize,
                hidden,
                cfg.hyper_connection_count,
            )?;
        }
        encode_compressed_attention_prelude(
            session,
            registry,
            &device,
            cfg,
            state,
            attention_weights,
            &core,
            rows,
            RowMatmul::IndependentRows,
        )?;
        let rope = DeepSeekTailRopeParams {
            batch: 1,
            seq_len: rows as u32,
            heads: heads as u32,
            head_dim: head_dim as u32,
            rope_dim: cfg.rope_head_dim,
            inverse: 0,
        };
        session.barrier_between(&[&core.q_norm, positions, frequency_buffer], &[&q_rope]);
        dispatch_deepseek_tail_rope_f32_to_bf16(
            session.encoder_mut(),
            registry,
            &device,
            &core.q_norm,
            positions,
            frequency_buffer,
            &q_rope,
            &rope,
        )?;
        session.barrier_between(&[&core.kv_norm, positions, frequency_buffer], &[&kv_rope]);
        dispatch_deepseek_tail_rope_f32_to_bf16(
            session.encoder_mut(),
            registry,
            &device,
            &core.kv_norm,
            positions,
            frequency_buffer,
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
                rows: rows as u32,
                row_width: head_dim as u32,
                quantized_width: (head_dim - cfg.rope_head_dim as usize) as u32,
                block_size: 64,
            },
        )?;

        for (row, (row_core, inputs)) in row_cores.iter().zip(&row_inputs).enumerate() {
            let layer_cache = caches[row]
                .layers()
                .get(layer)
                .with_context(|| format!("missing layer-{layer} joint row-{row} cache"))?;
            let layer_step = steps[row]
                .layers
                .get(layer)
                .with_context(|| format!("missing layer-{layer} joint row-{row} cache step"))?;
            let encoded = match row_core {
                JointRowCore::Window(window) => encode_window_row(
                    session,
                    registry,
                    &device,
                    cfg,
                    layer_cache,
                    layer_step,
                    window,
                    attention_weights.sinks,
                    inputs,
                ),
                JointRowCore::Compressed(compressed) => encode_compressed_row(
                    session,
                    registry,
                    &device,
                    cfg,
                    layer_cache,
                    layer_step,
                    compressed,
                    compressed_weights
                        .as_ref()
                        .context("DeepSeek-V4 compressed joint weights missing")?,
                    frequency_buffer,
                    inputs,
                ),
            };
            encoded.with_context(|| {
                format!("encode DeepSeek-V4 joint layer-{layer} row {row} core")
            })?;
        }

        encode_compressed_attention_epilogue(
            session,
            registry,
            &device,
            cfg,
            state,
            attention_weights,
            &core,
            &attention,
            positions,
            frequency_buffer,
            rows,
            RowMatmul::IndependentRows,
        )?;
        Ok(core.output_state)
    }
}
