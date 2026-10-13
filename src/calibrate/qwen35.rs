//! ADR-054 per-family calibration adapter for Qwen3.5/3.6/3.8 (#300) —
//! one adapter for both variants (`qwen35` dense, `qwen35moe` MoE):
//! `Qwen35Model` loads both shapes and `cfg.variant` records which.
//!
//! Capture: the post-layer residual `hidden` at the layer-loop tail (the
//! exact tensor the `residual_stream_post_layer` steering mutates), via
//! `forward_gpu_capture_residual_layer` — prefill-only, one fresh
//! `HybridKvCache` per prompt, positions per the existing `forward_gpu`
//! contract. The Qwen family lists ONLY this capture site: no FFN-writer
//! read arm exists in the Qwen forward path, so a writer-site capture is
//! refused here by construction (a writer-site EXPORT remains available —
//! `ffn_out_pre_residual` is in the family's bind set since #296).

use anyhow::{Context, Result};
use mlx_native::gguf::GgufFile;
use mlx_native::MlxDevice;
use tokenizers::Tokenizer;

use crate::inference::glp::bind::BoundGlp;
use crate::inference::glp::reader::GlpVector;
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::inference::models::qwen35::tokenizer as qwen35_tokenizer;
use crate::inference::models::qwen35::Qwen35Config;
use crate::serve::multi_seq_kv::SlotId;

use super::family::{
    bound_glp_debug_line, CalibrationFamily, CaptureSite, CapturedState, FamilyFacts,
    FAMILY_HOOK_SET,
};

/// The published GLP-49 span on the Qwen3.8 dense (`-L10-58`): the
/// per-family default capture layer is the span's midpoint. A dial, not a
/// claim — the multi-layer ladder arm makes single-layer defaults less
/// load-bearing, and the chosen default is recorded in the export
/// metadata.
const QWEN_GLP49_SPAN: (usize, usize) = (10, 58);

/// Text-only axis-major positions `[4 * seq_len]` (each MROPE sector
/// carries the plain token index) — the same contract
/// `run_calibration_prompt_gpu` and the serve prefill use.
fn text_positions(seq_len: usize) -> Vec<i32> {
    let mut flat = Vec::with_capacity(4 * seq_len);
    for _ in 0..4 {
        for i in 0..seq_len {
            flat.push(i as i32);
        }
    }
    flat
}

pub struct Qwen35Family {
    model: Qwen35Model,
    tokenizer: Tokenizer,
    facts: FamilyFacts,
    /// The chat template actually in effect: GGUF
    /// `tokenizer.chat_template`, falling back to the Qwen3 ChatML
    /// constant — the same resolution `hf2q info`/serve perform.
    chat_template: String,
    /// Multi-source EOS set (metadata `eos_token_id`/`eot_token_id` + the
    /// canonical qwen3 chat-template stop token names scanned from
    /// `tokenizer.ggml.tokens`, legacy fallback `<|im_end|>`).
    eos_token_ids: Vec<u32>,
    /// BOS to prepend when the GGUF declares `add_bos_token` (the serve
    /// tokenize contract).
    add_bos_token: Option<u32>,
    device: MlxDevice,
}

impl Qwen35Family {
    /// Config-only family facts — cheap, for fail-fast validation before
    /// the (large) weight load.
    pub(super) fn facts_from_gguf(gguf: &GgufFile) -> Result<FamilyFacts> {
        let architecture = gguf
            .metadata_string("general.architecture")
            .filter(|value| !value.is_empty())
            .context("Qwen calibration: missing general.architecture")?;
        let config = Qwen35Model::load_config_only(gguf)
            .context("load Qwen3.5/3.6/3.8 config for calibration")?;
        Self::facts_from_config(architecture, &config)
    }

    fn facts_from_config(architecture: &str, config: &Qwen35Config) -> Result<FamilyFacts> {
        // Family string is the GGUF architecture verbatim ("qwen35" /
        // "qwen35moe") so the export metadata names the exact variant.
        let family: &'static str = match architecture {
            "qwen35moe" => "qwen35moe",
            _ => "qwen35",
        };
        // Default capture layer: the midpoint of the published GLP-49 span
        // (`-L10-58` on the 3.8 dense), clamped to this artifact's layers.
        // A dial, not a claim — the chosen default is recorded in the
        // export metadata and the sidecar report.
        let default_layer = QWEN_GLP49_SPAN.0 / 2 + QWEN_GLP49_SPAN.1 / 2;
        let default_layer = default_layer
            .max(1)
            .min(config.num_hidden_layers.saturating_sub(1) as usize);
        Ok(FamilyFacts {
            family,
            num_layers: config.num_hidden_layers,
            hidden: config.hidden_size,
            hc: None,
            capture_sites: &[CaptureSite::ResidualStreamPostLayer],
            default_hook: CaptureSite::ResidualStreamPostLayer.as_hook_point(),
            default_derived_at: CaptureSite::ResidualStreamPostLayer.as_spec_str(),
            default_layer,
        })
    }

    pub fn load(gguf: &GgufFile) -> Result<Self> {
        let architecture = gguf
            .metadata_string("general.architecture")
            .filter(|value| !value.is_empty())
            .context("Qwen calibration: missing general.architecture")?;
        let config = Qwen35Model::load_config_only(gguf)
            .context("load Qwen3.5/3.6/3.8 config for calibration")?;
        let facts = Self::facts_from_config(architecture, &config)?;
        let tokenizer = qwen35_tokenizer::build_tokenizer_from_gguf(gguf)
            .context("build Qwen tokenizer from model GGUF")?;
        // Silent progress: calibrate is a batch CLI tool; per-layer
        // progress lines would interleave with the pipeline's own stderr
        // report (same pattern as `RealActivationCapture::new`).
        let mut progress =
            crate::serve::header::LoadProgress::new(false, 1, config.num_hidden_layers as usize);
        let model = Qwen35Model::load_from_gguf(gguf, &mut progress)
            .context("load Qwen3.5/3.6/3.8 model for calibration")?;
        let device = MlxDevice::new().context("MlxDevice for Qwen calibration")?;

        let chat_template = gguf
            .metadata_string("tokenizer.chat_template")
            .unwrap_or(crate::core::chat_templates::QWEN3_CHATML)
            .to_string();

        // EOS resolution mirrors the serve engine (iter-267 multi-source):
        // metadata ids first, then a token-array scan for the canonical
        // qwen3 chat-template stop names, legacy fallback last.
        let mut eos_token_ids: Vec<u32> = Vec::new();
        for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"] {
            if let Some(id) = gguf.metadata_u32(key) {
                if !eos_token_ids.contains(&id) {
                    eos_token_ids.push(id);
                }
            }
        }
        if let Some(mlx_native::gguf::MetadataValue::Array(elems)) =
            gguf.metadata("tokenizer.ggml.tokens")
        {
            for (index, element) in elems.iter().enumerate() {
                if let mlx_native::gguf::MetadataValue::String(text) = element {
                    if text == "<|im_end|>" || text == "<|endoftext|>" {
                        let id = index as u32;
                        if !eos_token_ids.contains(&id) {
                            eos_token_ids.push(id);
                        }
                    }
                }
            }
        }
        if eos_token_ids.is_empty() {
            eos_token_ids.push(151_645);
        }

        let add_bos_token = match gguf.metadata("tokenizer.ggml.add_bos_token") {
            Some(mlx_native::gguf::MetadataValue::Bool(true)) => gguf
                .metadata_u32("tokenizer.ggml.bos_token_id")
                .filter(|id| tokenizer.id_to_token(*id).is_some()),
            _ => None,
        };

        Ok(Self {
            model,
            tokenizer,
            facts,
            chat_template,
            eos_token_ids,
            add_bos_token,
            device,
        })
    }

    /// Tokenize a rendered chat prompt with the serve-side contract
    /// (`add_special=false`, specials parsed by the rendered template
    /// text, BOS prepended only when the GGUF declares it).
    fn tokenize_rendered(&self, rendered: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(rendered, false)
            .map_err(|error| anyhow::anyhow!("Qwen calibration tokenize: {error}"))?;
        let mut tokens = encoding.get_ids().to_vec();
        if let Some(bos) = self.add_bos_token {
            if tokens.first() != Some(&bos) {
                tokens.insert(0, bos);
            }
        }
        Ok(tokens)
    }
}

impl CalibrationFamily for Qwen35Family {
    fn facts(&self) -> &FamilyFacts {
        &self.facts
    }

    fn encode(&mut self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!("Qwen calibration tokenize: {error}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn capture(
        &mut self,
        tokens: &[u32],
        site: CaptureSite,
        layer: usize,
    ) -> Result<CapturedState> {
        self.facts.ensure_capture_site(site)?;
        anyhow::ensure!(!tokens.is_empty(), "Qwen calibration prompt is empty");
        let seq_len = tokens.len() as u32;
        let positions = text_positions(tokens.len());
        let mut cache = HybridKvCache::new(&self.model.cfg, &self.device, seq_len.max(1), 1)
            .context("allocate Qwen calibration cache")?;
        let row = self
            .model
            .forward_gpu_capture_residual_layer(tokens, &positions, &mut cache, layer, SlotId(0))
            .context("Qwen calibration residual capture")?;
        Ok(CapturedState::Dense(row))
    }

    fn probe_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "Qwen calibration probe is empty");
        let positions = text_positions(tokens.len());
        let mut cache = HybridKvCache::new(&self.model.cfg, &self.device, tokens.len() as u32, 1)
            .context("allocate Qwen probe cache")?;
        self.model
            .forward_gpu_last_logits(tokens, &positions, &mut cache, SlotId(0))
            .context("Qwen calibration probe prefill")
    }

    fn bind_glp(&mut self, vector: GlpVector, alpha: Option<f32>) -> Result<()> {
        // The dense residual-site contract (qwen): every site accepts
        // `hidden`-width directions; the family hook set is both spec
        // sites (ADR-053 dual hook sites, #296).
        let bound = BoundGlp::bind(
            vector,
            alpha,
            &self.device,
            &FAMILY_HOOK_SET,
            self.facts.num_layers,
            self.facts.hidden,
        )
        .map_err(|error| anyhow::anyhow!("Qwen calibration GLP bind: {error}"))?;
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
            // The deployment register: the family chat template with
            // thinking off (the panel protocol's setting), greedy.
            let rendered =
                crate::serve::render_jinja_template(&self.chat_template, probe, Some(false))
                    .with_context(|| format!("render Qwen register probe {index}"))?;
            let prompt = self.tokenize_rendered(&rendered)?;
            anyhow::ensure!(
                !prompt.is_empty(),
                "Qwen register probe encoded to zero tokens"
            );
            let prompt_len = prompt.len();
            let max_seq = (prompt_len + max_tokens + 64)
                .max(128)
                .min(self.model.cfg.max_position_embeddings as usize);
            let positions = text_positions(prompt_len);
            let mut cache = HybridKvCache::new(&self.model.cfg, &self.device, max_seq as u32, 1)
                .context("allocate Qwen register-probe cache")?;
            let logits = self
                .model
                .forward_gpu_last_logits(&prompt, &positions, &mut cache, SlotId(0))
                .context("Qwen register-probe prefill")?;
            let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens);
            let mut next = crate::inference::models::qwen35::io_heads::greedy_argmax_last_token(
                &logits,
                self.model.cfg.vocab_size,
            );
            let mut pos = prompt_len as i32;
            for _ in 0..max_tokens {
                if self.eos_token_ids.contains(&next) {
                    break;
                }
                tokens.push(next);
                if tokens.len() >= max_tokens {
                    break;
                }
                next = self
                    .model
                    .forward_gpu_greedy(&[next], &[pos; 4], &mut cache, SlotId(0))
                    .with_context(|| format!("Qwen register-probe decode at pos {pos}"))?;
                pos += 1;
            }
            let text = self.tokenizer.decode(&tokens, false).unwrap_or_default();
            openings.push(text);
        }
        Ok(openings)
    }
}
