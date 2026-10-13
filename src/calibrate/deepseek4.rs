//! ADR-054 calibration adapter for DeepSeek-V4 (#300 per-family
//! extraction). Everything here is the existing ADR-054 pipeline's
//! DeepSeek specifics, moved out of `mod.rs` behind the
//! [`CalibrationFamily`] trait — the default deepseek4 path (residual
//! capture, writer-hook export) runs byte-identically to the
//! pre-extraction pipeline: same capture arm
//! (`forward_verifier_prefill_capture_layer`), same `[hc, hidden]`
//! committed-state shape, same probe/canary sequence, same export keys
//! and defaults.
//!
//! New with #300: the FFN-writer capture arm
//! (`forward_verifier_prefill_capture_ffn_writer`, the `[hidden]` writer
//! row immediately before the `dispatch_hc_post` fold) — the
//! capture-at-the-apply-site read for ladder arm 2 (ADR-054 Deferred
//! follow-ups item 4).

use anyhow::{Context, Result};
use mlx_native::gguf::GgufFile;
use tokenizers::Tokenizer;

use crate::core::deepseek_v4_encoding::{
    encode_json, EncodeOptions, ReasoningEffort, ThinkingMode,
};
use crate::inference::glp::bind::{BoundGlp, GlpResidualSite};
use crate::inference::glp::reader::{GlpHookPoint, GlpVector};
use crate::inference::models::deepseek4::cache::{Deepseek4Cache, Deepseek4CachePlan};
use crate::inference::models::deepseek4::model::Deepseek4Model;
use crate::inference::models::deepseek4::tokenizer as deepseek4_tokenizer;

use super::family::{
    bound_glp_debug_line, CalibrationFamily, CaptureSite, CapturedState, FamilyFacts,
    FAMILY_HOOK_SET,
};

/// Calibration cache context — the pre-extraction pipeline's constant.
const CALIBRATION_CONTEXT_LENGTH: usize = 4096;
/// Probe/canary cache context — the pre-extraction pipeline's constant.
const PROBE_CONTEXT_LENGTH: usize = 2048;

pub struct Deepseek4Family {
    model: Deepseek4Model,
    tokenizer: Tokenizer,
    facts: FamilyFacts,
    eos: u32,
}

impl Deepseek4Family {
    /// Config-only family facts — cheap, for fail-fast validation before
    /// the (large) weight load.
    pub(super) fn facts_from_gguf(gguf: &GgufFile) -> Result<FamilyFacts> {
        let config = Deepseek4Model::load_config_only(gguf).context("load DeepSeek4 config")?;
        Ok(FamilyFacts {
            family: "deepseek4",
            num_layers: config.num_hidden_layers,
            hidden: config.hidden_size,
            hc: Some(config.hyper_connection_count),
            capture_sites: &[
                CaptureSite::ResidualStreamPostLayer,
                CaptureSite::FfnOutPreResidual,
            ],
            // The GLP-29-declared pair: applied at the FFN writer,
            // derived at the post-layer residual — kept until the
            // ADR-054 same-site ladder lands a measured winner.
            default_hook: GlpHookPoint::FfnOutPreResidual,
            default_derived_at: CaptureSite::ResidualStreamPostLayer.as_spec_str(),
            // The measured export layer of the 2026-09-10 / #298 runs.
            default_layer: 29,
        })
    }

    pub fn load(gguf: &GgufFile) -> Result<Self> {
        let facts = Self::facts_from_gguf(gguf)?;
        let tokenizer = deepseek4_tokenizer::build_tokenizer_from_gguf(gguf)
            .context("build tokenizer from model GGUF")?;
        let model = Deepseek4Model::load_from_gguf(gguf).context("load DeepSeek4 model")?;
        let eos = gguf
            .metadata_u32("tokenizer.ggml.eos_token_id")
            .unwrap_or(1);
        Ok(Self {
            model,
            tokenizer,
            facts,
            eos,
        })
    }

    /// The existing ADR-054 committed-state capture: fresh cache per
    /// prompt, `[hc, hidden]` at the last sequence position.
    fn capture_residual(&mut self, tokens: &[u32], layer: usize) -> Result<Vec<f32>> {
        let plan = Deepseek4CachePlan::for_context(&self.model.cfg, CALIBRATION_CONTEXT_LENGTH)
            .context("plan calibration cache")?;
        let device = self.model.ctx.device().clone();
        let mut cache =
            Deepseek4Cache::allocate(&plan, device).context("allocate calibration cache")?;
        self.model
            .forward_verifier_prefill_capture_layer(tokens, &mut cache, layer)
    }
}

impl CalibrationFamily for Deepseek4Family {
    fn facts(&self) -> &FamilyFacts {
        &self.facts
    }

    fn encode(&mut self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!("tokenize: {error}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn capture(
        &mut self,
        tokens: &[u32],
        site: CaptureSite,
        layer: usize,
    ) -> Result<CapturedState> {
        self.facts.ensure_capture_site(site)?;
        match site {
            CaptureSite::ResidualStreamPostLayer => Ok(CapturedState::MhcStreams(
                self.capture_residual(tokens, layer)?,
            )),
            CaptureSite::FfnOutPreResidual => {
                let plan =
                    Deepseek4CachePlan::for_context(&self.model.cfg, CALIBRATION_CONTEXT_LENGTH)
                        .context("plan calibration cache")?;
                let device = self.model.ctx.device().clone();
                let mut cache = Deepseek4Cache::allocate(&plan, device)
                    .context("allocate calibration cache")?;
                let row = self
                    .model
                    .forward_verifier_prefill_capture_ffn_writer(tokens, &mut cache, layer)?;
                Ok(CapturedState::Dense(row))
            }
        }
    }

    fn probe_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        probe_logits_deepseek4(&mut self.model, tokens)
    }

    fn bind_glp(&mut self, vector: GlpVector, alpha: Option<f32>) -> Result<()> {
        // The serve-side family bind contract: both spec hook sites, the
        // mHC residual-site shape (project-only there, `hidden` or
        // `hc*hidden` widths — bind refuses the rest with named errors).
        let device = self.model.ctx.device().clone();
        let bound = BoundGlp::bind_for_family(
            vector,
            alpha,
            &device,
            &FAMILY_HOOK_SET,
            self.model.cfg.num_hidden_layers,
            self.model.cfg.hidden_size,
            GlpResidualSite::MhcStreams {
                hc: self.model.cfg.hyper_connection_count,
            },
        )
        .map_err(|error| anyhow::anyhow!("bind GLP: {error}"))?;
        eprintln!("{}", bound_glp_debug_line(&bound));
        self.model.glp = Some(bound);
        Ok(())
    }

    fn clear_glp(&mut self) {
        self.model.glp = None;
    }

    fn generate_openings(&mut self, probes: &[String], max_tokens: usize) -> Result<Vec<String>> {
        anyhow::ensure!(!probes.is_empty(), "register probe set is empty");
        let mut openings = Vec::with_capacity(probes.len());
        for (index, probe) in probes.iter().enumerate() {
            // The deployment register: the DeepSeek-V4 chat encoding with
            // thinking off (the panel protocol's setting), greedy.
            let messages =
                serde_json::to_string(&vec![serde_json::json!({"role": "user", "content": probe})])
                    .context("serialize DeepSeek register probe message")?;
            let rendered = encode_json(
                &messages,
                EncodeOptions {
                    thinking_mode: ThinkingMode::Chat,
                    drop_thinking: true,
                    add_bos: true,
                    reasoning_effort: ReasoningEffort::Low,
                },
            )
            .with_context(|| format!("encode DeepSeek register probe {index}"))?;
            let prompt = self
                .tokenizer
                .encode(rendered, false)
                .map_err(|error| anyhow::anyhow!("tokenize DeepSeek register probe: {error}"))?
                .get_ids()
                .to_vec();
            anyhow::ensure!(
                !prompt.is_empty(),
                "DeepSeek register probe encoded to zero tokens"
            );
            anyhow::ensure!(
                prompt
                    .iter()
                    .all(|token| *token < self.model.cfg.vocab_size),
                "DeepSeek register probe contains a token outside the model vocabulary"
            );
            let requested = prompt
                .len()
                .checked_add(max_tokens)
                .context("DeepSeek register-probe context overflow")?;
            anyhow::ensure!(
                requested <= self.model.cfg.max_position_embeddings as usize,
                "DeepSeek register probe + generation length {} exceeds context {}",
                requested,
                self.model.cfg.max_position_embeddings
            );
            let cache_context = requested.max(self.model.cfg.sliding_window as usize);
            let mut cache = self
                .model
                .allocate_cache(cache_context)
                .context("allocate DeepSeek register-probe cache")?;
            let final_state = self
                .model
                .forward_verifier_prompt(&prompt, &mut cache)
                .context("DeepSeek register-probe prefill")?;
            let last = self
                .model
                .last_token_state(&final_state)
                .context("view DeepSeek register-probe final state")?;
            let mut logits = self
                .model
                .forward_logits(&last)
                .context("DeepSeek register-probe output head")?;
            let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens);
            let mut next = self
                .model
                .greedy_token(&logits)
                .context("select DeepSeek register-probe greedy token")?;
            for _ in 0..max_tokens {
                if next == self.eos {
                    break;
                }
                tokens.push(next);
                if tokens.len() >= max_tokens {
                    break;
                }
                let state = self
                    .model
                    .forward_verifier_one(next, &mut cache)
                    .context("DeepSeek register-probe decode token")?;
                logits = self
                    .model
                    .forward_logits(&state)
                    .context("DeepSeek register-probe decode output head")?;
                next = self
                    .model
                    .greedy_token(&logits)
                    .context("select DeepSeek register-probe decode token")?;
            }
            let text = self.tokenizer.decode(&tokens, false).unwrap_or_default();
            openings.push(text);
        }
        Ok(openings)
    }
}

/// The pre-extraction `probe_logits`: one prefill + head read on a fresh
/// 2048-position cache, unchanged.
fn probe_logits_deepseek4(model: &mut Deepseek4Model, tokens: &[u32]) -> Result<Vec<f32>> {
    let plan = Deepseek4CachePlan::for_context(&model.cfg.clone(), PROBE_CONTEXT_LENGTH)
        .context("plan probe cache")?;
    let device = model.ctx.device().clone();
    let mut cache = Deepseek4Cache::allocate(&plan, device).context("allocate probe cache")?;
    let state = model
        .forward_verifier_prefill(tokens, &mut cache)
        .context("probe prefill")?;
    let last = model.last_token_state(&state).context("probe last token")?;
    let logits = model.forward_logits(&last).context("probe logits")?;
    let data: &[f32] = logits
        .as_logical_slice()
        .map_err(|error| anyhow::anyhow!("probe logits read: {error}"))?;
    Ok(data.to_vec())
}
