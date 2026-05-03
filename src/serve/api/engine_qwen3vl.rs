//! Qwen3-VL text-LM SERVE-side load path (ADR-005 Wedge-4 / iter-228a).
//!
//! Mirror of [`super::engine_qwen35`] for the Qwen3-VL text family.
//!
//! # Scope (iter-228a MVP)
//!
//! - [`Qwen3VlTextLoadedModel::load`] — opens the GGUF, parses
//!   [`Qwen3VlTextConfig`], loads every weight via
//!   [`Qwen3VlTextWeights::load_from_gguf`], resolves tokenizer + chat
//!   template + EOS + provenance, and populates the metadata surface
//!   [`Engine::model_id`](super::engine::Engine) / `/v1/models` /
//!   `/metrics` consume.
//! - **No forward pass.** The Engine worker thread arm for this variant
//!   returns the sentinel
//!   [`crate::inference::models::qwen3vl_text::forward::QWEN3VL_TEXT_FORWARD_PENDING_SENTINEL`]
//!   for every chat / streaming / embed / soft-tokens request, mapped
//!   to HTTP 501 by the chat handler. **Model is loaded; chat is 501.**
//!
//! This is the same load-then-forward split that landed Qwen3.5/3.6
//! (iter-215 → Wedge-3). iter-228b replaces the 501 sentinel arms in
//! [`super::engine::worker_run`] with the real forward chain.
//!
//! # Why a separate file
//!
//! [`super::engine`] is already 11K LOC (mostly Gemma-shaped chat /
//! streaming / grammar / soft-token / KV-spill machinery). Co-locating
//! the Qwen3-VL text surface in this dedicated file keeps the SERVE-path
//! arch dispatch visible in one place + leaves room for iter-228b
//! (forward wiring) to land without further `engine.rs` bloat. Same
//! rationale `engine_qwen35.rs` carries.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::inference::models::qwen3vl_text::Qwen3VlTextModel;
use crate::serve::load_info::{
    self, ArchFamily, ChatTemplateSource, LoadInfo, LoadInfoBuilder, TokenizerSource,
};
use crate::serve::provenance::{self, Provenance};

use super::engine::LoadOptions;

/// All artifacts the SERVE worker needs to handle requests against a
/// Qwen3-VL text-LM GGUF.
///
/// iter-228a MVP: every field except `model` is also surfaced through
/// [`super::engine::LoadedModel`] accessor methods (model_id,
/// hidden_size, vocab_size, …) so the `/v1/models` + `/metrics` +
/// Engine handle surface is identical to the Gemma + Qwen35 variants.
/// `model` is held by-value for iter-228b (forward_gpu wiring) — the
/// worker takes ownership when the LoadedModel moves into the worker
/// thread.
pub struct Qwen3VlTextLoadedModel {
    /// Loaded weights + config + GPU context, ready for `forward_*`
    /// calls. iter-228b consumes this through the worker thread.
    pub model: Qwen3VlTextModel,
    /// Tokenizer (truncation disabled; GGUF-driven, mirroring the
    /// pattern Qwen3.5/3.6 uses to avoid HF-tokenizer / GGUF-vocab
    /// drift).
    pub tokenizer: Tokenizer,
    /// GGUF-embedded chat template; empty string when absent.
    /// iter-228a MVP returns 501 before this is consumed; iter-228b
    /// will validate non-empty for the live chat path.
    pub chat_template: String,
    /// Surfaced via `/v1/models[*].id` and `Engine::model_id()`.
    /// Derived from `general.name` if present, else file stem.
    pub model_id: String,
    /// Filesystem path to the GGUF opened by this loaded model.
    pub model_path: PathBuf,
    /// EOS tokens — Qwen3-VL ships `tokenizer.ggml.eos_token_id =
    /// 151645` (`<|im_end|>`); fallback to that value when the metadata
    /// key is absent.
    pub eos_token_ids: Vec<u32>,
    /// Hidden-state dimensionality (mirrors `model.cfg.hidden_size`).
    pub hidden_size: usize,
    /// Vocabulary size (mirrors `model.cfg.vocab_size`).
    pub vocab_size: usize,
    /// Maximum context length declared by the GGUF.
    pub context_length: Option<usize>,
    /// Dominant non-fp tensor type for `/v1/models` ("Q4_0" for the
    /// Wedge-4f-converted Qwen3-VL-2B GGUF).
    pub quant_type: Option<String>,
    /// Wall-clock from start to finish of [`Self::load`].
    pub load_duration: Duration,
    /// ADR-017 §F4 — GGUF provenance captured at load time. Stored for
    /// the common [`super::engine::LoadedModel::provenance`] surface.
    pub provenance: Provenance,
}

impl Qwen3VlTextLoadedModel {
    /// Open a Qwen3-VL text-LM GGUF and populate every field.
    ///
    /// Mirrors [`super::engine_qwen35::Qwen35LoadedModel::load`] in
    /// shape. Errors propagate from:
    /// - GGUF open / parse
    /// - [`Qwen3VlTextModel::load_from_gguf`]
    /// - tokenizer file resolution + parse
    pub fn load(opts: &LoadOptions) -> Result<Self> {
        let load_start = Instant::now();
        let model_path = &opts.model_path;
        anyhow::ensure!(
            model_path.exists(),
            "Model not found: {}",
            model_path.display()
        );

        // Open GGUF (header + metadata only — re-opens after the
        // dispatcher-level open in `LoadedModel::load`; the cost is a
        // memory-mapped header parse, small relative to the full
        // weights load below).
        let gguf = mlx_native::gguf::GgufFile::open(model_path)
            .map_err(|e| anyhow::anyhow!("GGUF open: {e}"))?;
        let provenance = provenance::detect(&gguf);

        // ---- Pre-flight config parse ----
        // Run this BEFORE any tokenizer or weight-load work so that
        // operators who passed a malformed (or non-Qwen3-VL) GGUF see
        // the structurally-honest error from the config parser, not a
        // misleading "tokenizer.json not found" from `find_tokenizer`
        // or a downstream weight-load shape error. Iter-228a's
        // synthetic-GGUF dispatch tests assert on this parser-error
        // shape.
        let cfg_preview =
            Qwen3VlTextModel::load_config_only(&gguf).context("config preview")?;

        // ---- Tokenizer path ----
        // Qwen3-VL ships GGUF metadata `tokenizer.ggml.pre = 'default'`
        // (gpt2-style, NOT the qwen35-specific pre-type that
        // engine_qwen35::build_tokenizer_from_gguf accepts). We resolve
        // a sibling `tokenizer.json` on disk (same pattern Gemma uses)
        // and load it directly. Wedge-4f's converter copies the HF
        // `tokenizer.json` next to the GGUF (see scripts/wedge4_qwen3vl.sh).
        let tokenizer_path =
            crate::serve::find_tokenizer(model_path, opts.tokenizer_path.as_deref())?;

        // ---- Load weights (full mlx-native pipeline) ----
        let stderr_is_tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
        let verbosity = if tracing::enabled!(tracing::Level::INFO) {
            1
        } else {
            0
        };
        let mut progress = crate::serve::header::LoadProgress::new(
            stderr_is_tty,
            verbosity,
            cfg_preview.num_hidden_layers as usize,
        );
        let model = Qwen3VlTextModel::load_from_gguf(&gguf, &mut progress)
            .context("Qwen3VlTextModel::load_from_gguf")?;

        // ---- Resolve EOS ----
        let eos_token: u32 = gguf
            .metadata_u32("tokenizer.ggml.eos_token_id")
            .unwrap_or(151645);
        let eos_token_ids: Vec<u32> = vec![eos_token];

        // ---- Load tokenizer from disk ----
        // Qwen3-VL ships an HF-format `tokenizer.json` next to the
        // GGUF. The qwen35 GGUF-driven builder doesn't accept this
        // family's `tokenizer.ggml.pre = 'default'` (qwen35-specific
        // regex; see qwen35/tokenizer.rs:118). Loading the HF
        // tokenizer.json is the simpler + safer path.
        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer.json from {}: {e}", tokenizer_path.display()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {e}"))?;

        // ---- Chat template ----
        let chat_template = gguf
            .metadata_string("tokenizer.chat_template")
            .map(|s| s.to_string())
            .unwrap_or_default();

        // ---- model_id ----
        let model_id = gguf
            .metadata_string("general.name")
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                model_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "qwen3vl-text-model".to_string())
            });

        // ---- Surface fields from cfg ----
        let hidden_size = model.cfg.hidden_size as usize;
        let vocab_size = model.cfg.vocab_size as usize;
        let context_length = if model.cfg.max_position_embeddings > 0 {
            Some(model.cfg.max_position_embeddings as usize)
        } else {
            None
        };

        // ---- Quant label ----
        let quant_type = crate::serve::load_info::infer_quant_label(&gguf);

        let load_duration = load_start.elapsed();

        Ok(Self {
            model,
            tokenizer,
            chat_template,
            model_id,
            model_path: model_path.clone(),
            eos_token_ids,
            hidden_size,
            vocab_size,
            context_length,
            quant_type,
            load_duration,
            provenance,
        })
    }
}

impl LoadInfoBuilder for Qwen3VlTextLoadedModel {
    fn build_load_info(
        &self,
        gguf: &mlx_native::gguf::GgufFile,
        load_wall_clock: Duration,
        kv_cache_budget_bytes: Option<u64>,
        kv_spill_active: bool,
    ) -> LoadInfo {
        let cfg = &self.model.cfg;
        LoadInfo {
            model_id: self.model_id.clone(),
            arch_str: load_info::arch_str_from_gguf(gguf),
            arch_family: ArchFamily::Qwen3VlText,
            model_path: self.model_path.clone(),
            on_disk_bytes: load_info::on_disk_bytes(&self.model_path),
            backend_chip: mlx_native::MlxDevice::new()
                .map(|d| d.name())
                .unwrap_or_else(|_| "Apple GPU".to_string()),
            backend: "mlx-native",
            n_layers: cfg.num_hidden_layers,
            hidden_size: self.hidden_size as u32,
            vocab_size: self.vocab_size as u32,
            n_attention_heads: cfg.num_attention_heads,
            n_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            sliding_window: None,
            full_attention_interval: None,
            max_context_length: self.context_length.map(|v| v as u32),
            moe: None,
            quant_label: self.quant_type.clone(),
            quant_bpw: load_info::compute_bpw(gguf),
            tokenizer_source: TokenizerSource::GgufEmbedded,
            eos_token_ids: self.eos_token_ids.clone(),
            bos_token_id: gguf.metadata_u32("tokenizer.ggml.bos_token_id"),
            chat_template_source: if gguf.metadata_string("tokenizer.chat_template").is_some() {
                ChatTemplateSource::GgufEmbedded
            } else {
                ChatTemplateSource::None
            },
            provenance: self.provenance.clone(),
            vision_projector: None,
            load_wall_clock,
            resident_weight_bytes: None,
            kv_cache_budget_bytes,
            kv_spill_active,
        }
    }
}
