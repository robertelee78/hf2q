//! Full one-token DeepSeek-V4 verifier execution and cache publication.

use anyhow::{Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::{GraphExecutor, MlxBuffer};

use super::cache::{CacheSpan, Deepseek4Cache};
use super::forward_support::{begin_prefill_pool_layer, end_prefill_pool_layer};
use super::submission::{drain, retained_reference_pipeline_enabled, SubmissionChain};
use super::Deepseek4Model;

pub(crate) const FRESH_MATRIX_PREFILL_WINDOW_MULTIPLIER: usize = 8;

impl Deepseek4Model {
    /// Execute one bounded prompt chunk layer-major with true matrix rows.
    pub fn forward_verifier_prefill(
        &mut self,
        token_ids: &[u32],
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        let span = cache
            .plan_prefill(token_ids.len())
            .context("plan DeepSeek-V4 batched prefill transaction")?;
        let result = self.forward_verifier_prefill_uncommitted(token_ids, cache, &span);
        match result {
            Ok(state) => {
                if let Err(error) = cache.commit_prefill(span.start_position, span.token_count) {
                    cache.poison();
                    return Err(error).context("publish complete DeepSeek-V4 prompt prefill");
                }
                Ok(state)
            }
            Err(error) => {
                cache.poison();
                Err(error).context("DeepSeek-V4 prefill partially executed; cache poisoned")
            }
        }
    }

    /// Prefill the leading portion of a fresh prompt as one native matrix
    /// transaction. Once a cache exists, extend it with the exact incremental
    /// verifier so follow-up turns preserve decode-equivalent recurrent and
    /// quantization semantics.
    pub fn forward_verifier_prompt(
        &mut self,
        token_ids: &[u32],
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        anyhow::ensure!(!token_ids.is_empty(), "DeepSeek-V4 prompt is empty");
        let mut state = None;
        let mut offset = 0;
        if cache.position() == 0 {
            let batch = token_ids.len().min(
                self.cfg.sliding_window as usize * FRESH_MATRIX_PREFILL_WINDOW_MULTIPLIER,
            );
            state = Some(self.forward_verifier_prefill(&token_ids[..batch], cache)?);
            offset = batch;
        }
        for &token in &token_ids[offset..] {
            state = Some(self.forward_verifier_one(token, cache)?);
        }
        state.context("DeepSeek-V4 prompt encoded zero chunks")
    }

    fn forward_verifier_prefill_uncommitted(
        &mut self,
        token_ids: &[u32],
        cache: &mut Deepseek4Cache,
        span: &CacheSpan,
    ) -> Result<MlxBuffer> {
        let layers = self.cfg.num_hidden_layers as usize;
        let executor = GraphExecutor::new(self.ctx.device().clone());
        let profile_timing = std::env::var_os("HF2Q_DEEPSEEK_PREFILL_TIMING").is_some();
        let mut state = None;
        for layer in 0..layers {
            begin_prefill_pool_layer();
            let layer_start = std::time::Instant::now();
            let layer_result: Result<MlxBuffer> = (|| {
                let mut session = executor
                    .begin()
                    .with_context(|| format!("begin DeepSeek-V4 prefill layer {layer}"))?;
                let next_state = self.encode_verifier_layer_prefill(
                    token_ids,
                    layer,
                    state.as_ref(),
                    cache,
                    span,
                    &mut session,
                )?;
                if profile_timing {
                    let (encode_ns, wait_ns) = session
                        .finish_with_timing(layer_start)
                        .with_context(|| format!("execute DeepSeek-V4 prefill layer {layer}"))?;
                    eprintln!(
                        "DeepSeek-V4 prefill layer {layer}: encode {:.3} ms; commit/wait {:.3} ms",
                        encode_ns as f64 / 1e6,
                        wait_ns as f64 / 1e6
                    );
                } else {
                    session
                        .finish()
                        .with_context(|| format!("execute DeepSeek-V4 prefill layer {layer}"))?;
                }
                Ok(next_state)
            })();
            let reset_start = std::time::Instant::now();
            end_prefill_pool_layer();
            if profile_timing {
                eprintln!(
                    "DeepSeek-V4 prefill layer {layer}: pool reset {:.3} ms; total {:.3} ms",
                    reset_start.elapsed().as_secs_f64() * 1e3,
                    layer_start.elapsed().as_secs_f64() * 1e3
                );
            }
            state = Some(layer_result?);
        }
        state.context("DeepSeek-V4 prefill encoded zero layers")
    }

    fn encode_verifier_layer_prefill(
        &mut self,
        token_ids: &[u32],
        layer: usize,
        state: Option<&MlxBuffer>,
        cache: &mut Deepseek4Cache,
        span: &CacheSpan,
        session: &mut GraphSession<'_>,
    ) -> Result<MlxBuffer> {
        let attention = if layer == 0 {
            anyhow::ensure!(state.is_none(), "DeepSeek-V4 layer 0 must embed the prompt");
            self.forward_uncompressed_attention_prefill(
                None,
                token_ids,
                layer,
                cache,
                span,
                None,
                Some(session),
            )
        } else {
            let state = state.context("DeepSeek-V4 nonzero prefill layer is missing state")?;
            if self.cfg.compress_ratios[layer] == 0 {
                self.forward_uncompressed_attention_prefill(
                    Some(state),
                    token_ids,
                    layer,
                    cache,
                    span,
                    None,
                    Some(session),
                )
            } else {
                self.forward_compressed_attention_prefill(
                    state,
                    layer,
                    cache,
                    span,
                    None,
                    Some(session),
                )
            }
        }
        .with_context(|| format!("encode DeepSeek-V4 prefill layer-{layer} attention"))?;
        self.forward_ffn_rows(&attention, token_ids, layer, None, Some(session))
            .with_context(|| format!("encode DeepSeek-V4 prefill layer-{layer} FFN"))
    }

    /// Execute all verifier blocks for one token.
    ///
    /// The normal retained-resource path encodes several complete layers per
    /// command buffer, preserving CPU/GPU overlap without the original
    /// attention/FFN submission fragmentation. Diagnostic stage profiling and
    /// unretained-resource mode keep the isolated stage path. Any submitted GPU
    /// failure poisons the cache; the caller must reset and replay rather than
    /// retry only the failed token.
    pub fn forward_verifier_one(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        let position = cache.position();
        let result = self.forward_verifier_one_uncommitted(token_id, cache);
        match result {
            Ok(state) => {
                if let Err(error) = cache.commit_step(position) {
                    cache.poison();
                    return Err(error).context("publish complete DeepSeek-V4 verifier token");
                }
                Ok(state)
            }
            Err(error) => {
                cache.poison();
                Err(error).context("DeepSeek-V4 verifier token partially executed; cache poisoned")
            }
        }
    }

    fn forward_verifier_one_uncommitted(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        let layers = self.cfg.num_hidden_layers as usize;
        let retained = retained_reference_pipeline_enabled();
        let profile_stages = std::env::var("HF2Q_DEEPSEEK_STAGE_PROFILE").as_deref() == Ok("1");
        if retained && !profile_stages {
            return self.forward_verifier_one_chunked(token_id, cache);
        }

        let pipelined = retained;
        let mut in_flight = SubmissionChain::with_capacity(layers.saturating_mul(2));
        let result =
            self.encode_verifier_layers(token_id, cache, None, pipelined.then_some(&mut in_flight));
        let drained = drain(&in_flight).context("drain DeepSeek-V4 verifier pipeline");
        match (result, drained) {
            (Ok(state), Ok(())) => Ok(state),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(drain_error)) => {
                Err(error).context(format!("pipeline drain also failed: {drain_error:#}"))
            }
        }
    }

    fn forward_verifier_one_chunked(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        let layers = self.cfg.num_hidden_layers as usize;
        let layers_per_command_buffer = std::env::var("HF2Q_DEEPSEEK_LAYERS_PER_CB")
            .ok()
            .map(|value| {
                value
                    .parse::<usize>()
                    .context("HF2Q_DEEPSEEK_LAYERS_PER_CB must be a positive integer")
            })
            .transpose()?
            .unwrap_or(1);
        anyhow::ensure!(
            (1..=layers).contains(&layers_per_command_buffer),
            "HF2Q_DEEPSEEK_LAYERS_PER_CB must be in 1..={layers}"
        );

        let executor = GraphExecutor::new(self.ctx.device().clone());
        let command_buffers = layers.div_ceil(layers_per_command_buffer);
        let mut in_flight = SubmissionChain::with_capacity(command_buffers);
        let result = (|| {
            let mut state = None;
            for start in (0..layers).step_by(layers_per_command_buffer) {
                let end = (start + layers_per_command_buffer).min(layers);
                let mut session = executor
                    .begin()
                    .with_context(|| format!("begin DeepSeek-V4 verifier layers {start}..{end}"))?;
                for layer in start..end {
                    state = Some(self.encode_verifier_layer(
                        token_id,
                        layer,
                        state.as_ref(),
                        cache,
                        Some(&mut session),
                        None,
                    )?);
                }
                in_flight.push((
                    format!("execute DeepSeek-V4 verifier layers {start}..{end}"),
                    session.commit(),
                ));
            }
            state.context("DeepSeek-V4 verifier encoded zero layers")
        })();
        let drained = drain(&in_flight).context("drain chunked DeepSeek-V4 verifier pipeline");
        match (result, drained) {
            (Ok(state), Ok(())) => Ok(state),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(drain_error)) => Err(error).context(format!(
                "chunked pipeline drain also failed: {drain_error:#}"
            )),
        }
    }

    fn encode_verifier_layers(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
        mut shared_session: Option<&mut GraphSession<'_>>,
        mut in_flight: Option<&mut SubmissionChain>,
    ) -> Result<MlxBuffer> {
        let layers = self.cfg.num_hidden_layers as usize;
        let mut state = None;
        for layer in 0..layers {
            state = Some(self.encode_verifier_layer(
                token_id,
                layer,
                state.as_ref(),
                cache,
                shared_session.as_deref_mut(),
                in_flight.as_deref_mut(),
            )?);
        }
        state.context("DeepSeek-V4 verifier encoded zero layers")
    }

    fn encode_verifier_layer(
        &mut self,
        token_id: u32,
        layer: usize,
        state: Option<&MlxBuffer>,
        cache: &mut Deepseek4Cache,
        shared_session: Option<&mut GraphSession<'_>>,
        in_flight: Option<&mut SubmissionChain>,
    ) -> Result<MlxBuffer> {
        let mut shared_session = shared_session;
        let mut in_flight = in_flight;
        let attention = if layer == 0 {
            anyhow::ensure!(
                state.is_none(),
                "DeepSeek-V4 layer 0 must embed the input token"
            );
            self.forward_uncompressed_attention_one(
                None,
                token_id,
                layer,
                cache,
                false,
                in_flight.as_deref_mut(),
                shared_session.as_deref_mut(),
            )
        } else {
            let state = state.context("DeepSeek-V4 nonzero layer is missing its input state")?;
            if self.cfg.compress_ratios[layer] == 0 {
                self.forward_uncompressed_attention_one(
                    Some(state),
                    token_id,
                    layer,
                    cache,
                    false,
                    in_flight.as_deref_mut(),
                    shared_session.as_deref_mut(),
                )
            } else {
                self.forward_compressed_attention_one(
                    state,
                    layer,
                    cache,
                    false,
                    in_flight.as_deref_mut(),
                    shared_session.as_deref_mut(),
                )
            }
        }
        .with_context(|| format!("execute DeepSeek-V4 layer-{layer} attention"))?;
        self.forward_ffn_one(
            &attention,
            token_id,
            layer,
            in_flight.as_deref_mut(),
            shared_session.as_deref_mut(),
        )
        .with_context(|| format!("execute DeepSeek-V4 layer-{layer} FFN"))
    }
}
