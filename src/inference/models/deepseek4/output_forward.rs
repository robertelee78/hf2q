//! Native DeepSeek-V4 Hyper-Connection output head and vocabulary logits.

use anyhow::{bail, Context, Result};
use mlx_native::ops::argmax::dispatch_argmax_f32;
use mlx_native::ops::deepseek_hyper_connection::{dispatch_hc_head_weights, dispatch_hc_pre};
use mlx_native::{DType, MlxBuffer};

use super::forward_support::{alloc, raw_matmul, rms_params};
use super::Deepseek4Model;

impl Deepseek4Model {
    /// Collapse completed verifier states and compute vocabulary logits.
    ///
    /// The input is the exact `[tokens, 4, hidden]` F32 state emitted by the
    /// final verifier FFN. This graph does not mutate request cache state, so
    /// callers may safely retry it after a command-buffer failure.
    pub fn forward_logits(&mut self, state: &MlxBuffer) -> Result<MlxBuffer> {
        let hidden = self.cfg.hidden_size as usize;
        let hc = self.cfg.hyper_connection_count as usize;
        let hc_hidden = hc
            .checked_mul(hidden)
            .context("DeepSeek-V4 output HC width overflow")?;
        let vocab = self.cfg.vocab_size as usize;
        if state.dtype() != DType::F32
            || state.shape().len() != 3
            || state.shape()[1..] != [hc, hidden]
        {
            bail!(
                "DeepSeek-V4 output state must be F32 [tokens, {hc}, {hidden}], got {} {:?}",
                state.dtype(),
                state.shape()
            );
        }
        let tokens = state.shape()[0];
        if tokens == 0 {
            bail!("DeepSeek-V4 output state must contain at least one token");
        }
        let tokens_u32 =
            u32::try_from(tokens).context("DeepSeek-V4 output token count exceeds u32")?;
        let hidden_u32 =
            u32::try_from(hidden).context("DeepSeek-V4 output hidden size exceeds u32")?;
        let hc_hidden_u32 =
            u32::try_from(hc_hidden).context("DeepSeek-V4 output HC width exceeds u32")?;

        let hc_fn = self.weights.raw_matrix_ref("output_hc_fn.weight")?;
        let hc_scale = self.weights.f32_state("output_hc_scale.weight")?;
        let hc_base = self.weights.f32_state("output_hc_base.weight")?;
        let output_norm_weight = self.weights.f32_state("output_norm.weight")?;
        let output_weight = self.weights.raw_matrix_ref("output.weight")?;

        let device = self.ctx.device().clone();
        let state_flat = state.with_shape(vec![tokens, hc_hidden])?;
        let state_norm = alloc(
            &device,
            DType::F32,
            vec![tokens, hc_hidden],
            "output HC normalized state",
        )?;
        let mixes = alloc(&device, DType::F32, vec![tokens, hc], "output HC mixes")?;
        let pre = alloc(&device, DType::F32, vec![tokens, hc], "output HC weights")?;
        let collapsed = alloc(
            &device,
            DType::F32,
            vec![tokens, hidden],
            "output HC collapsed state",
        )?;
        let normalized = alloc(
            &device,
            DType::F32,
            vec![tokens, hidden],
            "output normalized state",
        )?;
        let logits = alloc(
            &device,
            DType::F32,
            vec![tokens, vocab],
            "vocabulary logits",
        )?;
        let hc_params = rms_params(
            &device,
            self.cfg.rms_norm_eps,
            hc_hidden,
            "output HC RMS params",
        )?;
        let hidden_params =
            rms_params(&device, self.cfg.rms_norm_eps, hidden, "output RMS params")?;

        let (executor, registry) = self.ctx.split();
        let mut session = executor.begin().context("begin DeepSeek-V4 output head")?;
        session.barrier_between(&[&state_flat], &[&state_norm]);
        session.rms_norm_no_scale_f32(
            registry,
            device.metal_device(),
            &state_flat,
            &state_norm,
            &hc_params,
            tokens_u32,
            hc_hidden_u32,
        )?;
        raw_matmul(
            &mut session,
            registry,
            &device,
            &state_norm,
            &hc_fn,
            &mixes,
            tokens,
            hc,
            hc_hidden,
            "output HC function",
        )?;
        session.barrier_between(&[&mixes, hc_scale, hc_base], &[&pre]);
        dispatch_hc_head_weights(
            session.encoder_mut(),
            registry,
            &device,
            &mixes,
            hc_scale,
            hc_base,
            &pre,
            tokens_u32,
        )?;
        session.barrier_between(&[state, &pre], &[&collapsed]);
        dispatch_hc_pre(
            session.encoder_mut(),
            registry,
            &device,
            state,
            &pre,
            &collapsed,
            tokens_u32,
            hidden_u32,
        )?;
        session.barrier_between(&[&collapsed, output_norm_weight], &[&normalized]);
        session.rms_norm(
            registry,
            device.metal_device(),
            &collapsed,
            output_norm_weight,
            &normalized,
            &hidden_params,
            tokens_u32,
            hidden_u32,
        )?;
        raw_matmul(
            &mut session,
            registry,
            &device,
            &normalized,
            &output_weight,
            &logits,
            tokens,
            vocab,
            hidden,
            "vocabulary projection",
        )?;
        session
            .finish()
            .context("execute DeepSeek-V4 output head")?;
        Ok(logits)
    }

    /// Select the first maximum from one vocabulary-logit row on Metal.
    pub fn greedy_token(&mut self, logits: &MlxBuffer) -> Result<u32> {
        let vocab = self.cfg.vocab_size as usize;
        if logits.dtype() != DType::F32 || logits.shape() != [1, vocab] {
            bail!(
                "DeepSeek-V4 greedy logits must be F32 [1, {vocab}], got {} {:?}",
                logits.dtype(),
                logits.shape()
            );
        }
        let vocab_u32 = u32::try_from(vocab).context("DeepSeek-V4 vocabulary size exceeds u32")?;
        let device = self.ctx.device().clone();
        let out_index = alloc(&device, DType::U32, vec![1], "greedy token index")?;
        let out_value = alloc(&device, DType::F32, vec![1], "greedy token value")?;
        let mut params = alloc(&device, DType::U32, vec![1], "greedy token params")?;
        params.as_mut_slice::<u32>()?[0] = vocab_u32;

        let (executor, registry) = self.ctx.split();
        let mut session = executor
            .begin()
            .context("begin DeepSeek-V4 greedy argmax")?;
        session.barrier_between(&[logits, &params], &[&out_index, &out_value]);
        dispatch_argmax_f32(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            logits,
            &out_index,
            &out_value,
            &params,
            vocab_u32,
        )?;
        session
            .finish()
            .context("execute DeepSeek-V4 greedy argmax")?;
        Ok(out_index.as_slice::<u32>()?[0])
    }
}
