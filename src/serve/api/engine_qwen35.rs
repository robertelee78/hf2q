//! Qwen3.5 / Qwen3.6 SERVE-side load path (ADR-005 Phase 4 reopen iter-215
//! Wedge-2 MVP).
//!
//! # Scope
//!
//! Iter-215 Wedge-2 MVP:
//! - `Qwen35LoadedModel::load` — opens the GGUF, loads weights via
//!   `Qwen35Model::load_from_gguf`, resolves tokenizer + chat template +
//!   EOS, populates the metadata surface the engine handle (model_id,
//!   hidden_size, vocab_size, context_length, quant_type) and `/v1/models`
//!   need.
//! - **No forward pass.**  The Engine worker thread arm for this variant
//!   returns the iter-215 sentinel (`QWEN35_NOT_IMPLEMENTED_SENTINEL`)
//!   for every chat / embed / vision request, mapped to HTTP 501 by the
//!   chat handler (Phase D).  Model is loaded; chat is 501.
//!
//! # Why `engine_qwen35.rs` and not `engine.rs`
//!
//! `engine.rs` is already large (~7K LOC at iter-215 entry, mostly Gemma-
//! shaped chat / streaming / grammar / soft-token machinery).  Co-locating
//! the Qwen3.5/3.6 surface here keeps the SERVE-path arch dispatch
//! visible in one place + leaves room for Wedge-3 (forward_gpu wiring)
//! to land without further engine.rs bloat.
//!
//! # Wedge-3 (deferred follow-up)
//!
//! - Wire `Qwen35Model::forward_*` (prefill + decode) into the worker
//!   thread, mirroring the `cmd_generate_qwen35` inference loop at
//!   `serve/mod.rs:1037-1110+`.
//! - Replace the 501 sentinel arms in `engine.rs::worker_run` with the
//!   real generate/stream/embed paths.
//! - Add Qwen3.5/3.6 prompt-cache (currently `LoadedModel::prompt_cache()`
//!   returns `None` for the Qwen35 variant; that path needs review when
//!   live inference lands).

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use crate::inference::models::qwen35::model::Qwen35Model;

use super::engine::LoadOptions;

/// All artifacts the SERVE worker needs to handle requests against a
/// Qwen3.5 or Qwen3.6 GGUF.
///
/// Iter-215 Wedge-2 MVP: every field except `model` is also surfaced
/// through `LoadedModel` accessor methods (model_id, hidden_size,
/// vocab_size, …) so the `/v1/models` + `/metrics` + Engine handle
/// surface is identical to the Gemma variant.  `model` is held by-value
/// for Wedge-3 (forward_gpu wiring) — the worker takes ownership when
/// the LoadedModel moves into the worker thread.
pub struct Qwen35LoadedModel {
    /// Loaded weights + per-layer config, ready for `forward_*` calls.
    /// Wedge-3 consumes this through the worker thread.
    pub model: Qwen35Model,
    /// Tokenizer (truncation disabled, matching `cmd_generate_qwen35`).
    pub tokenizer: Tokenizer,
    /// GGUF-embedded chat template; empty string when absent.  Iter-215
    /// MVP returns 501 before this is consumed, so an empty template is
    /// acceptable.  Wedge-3 will validate non-empty for the chat path.
    pub chat_template: String,
    /// Surfaced via `/v1/models[*].id` and `Engine::model_id()`.
    /// Derived from `general.name` if present, else file stem.
    pub model_id: String,
    /// EOS tokens — Qwen3.5/3.6 typically uses 151645 (`<|im_end|>`).
    /// Resolved from `tokenizer.ggml.eos_token_id` metadata; default is
    /// the HF Qwen3.5 default (151645) per `cmd_generate_qwen35`.
    pub eos_token_ids: Vec<u32>,
    /// Hidden-state dimensionality.  From `model.cfg.hidden_size`.
    pub hidden_size: usize,
    /// Vocabulary size.  From `model.cfg.vocab_size` (post pad-row
    /// reconciliation in `Qwen35Model::load_from_gguf`).
    pub vocab_size: usize,
    /// Maximum context length.  From `model.cfg.max_position_embeddings`.
    pub context_length: Option<usize>,
    /// Dominant quant label ("Q4_0" / "Q6_K" / etc.) for `/v1/models`.
    pub quant_type: Option<String>,
    /// Wall-clock from start to finish of `load`.
    pub load_duration: Duration,
}

impl Qwen35LoadedModel {
    /// Open a Qwen3.5/3.6 GGUF and populate every field.
    ///
    /// Mirrors `cmd_generate_qwen35` (`serve/mod.rs:1037-1110`) for
    /// model + tokenizer + EOS + chat-template resolution — the SERVE
    /// path uses the same load logic to ensure parity with `hf2q
    /// generate` (the working chat path today).
    ///
    /// # Errors
    ///
    /// Propagates from:
    /// - GGUF open / parse
    /// - `Qwen35Model::load_from_gguf` (weights load via mlx-native)
    /// - tokenizer file resolution + parse
    pub fn load(opts: &LoadOptions) -> Result<Self> {
        let load_start = Instant::now();
        let model_path = &opts.model_path;
        anyhow::ensure!(
            model_path.exists(),
            "Model not found: {}",
            model_path.display()
        );

        // Open GGUF (header + metadata only).  Re-opens after the
        // dispatcher-level open in `LoadedModel::load`; the cost is a
        // memory-mapped header parse (~ms), small relative to the full
        // weights load below.
        let gguf = mlx_native::gguf::GgufFile::open(model_path)
            .map_err(|e| anyhow::anyhow!("GGUF open: {e}"))?;

        tracing::info!(
            "Qwen35 SERVE load: model = {}",
            model_path.display()
        );

        // ---- Resolve tokenizer path ----
        // Reuse the shared `find_tokenizer` helper from serve/mod.rs so
        // the SERVE path resolves the tokenizer the same way
        // `cmd_generate_qwen35` does.  Caller may override via
        // `--tokenizer` (threaded through `LoadOptions::tokenizer_path`).
        let tokenizer_path =
            crate::serve::find_tokenizer(model_path, opts.tokenizer_path.as_deref())?;
        tracing::info!(
            "Qwen35 SERVE load: tokenizer = {}",
            tokenizer_path.display()
        );

        // ---- Load weights (full mlx-native pipeline) ----
        let model = Qwen35Model::load_from_gguf(&gguf).context("Qwen35Model::load_from_gguf")?;
        let n_layers = model.layers.len();
        tracing::info!(
            "Qwen35 SERVE load: weights loaded ({} layers, variant={:?})",
            n_layers,
            model.cfg.variant
        );

        // ---- Resolve EOS ----
        // Qwen3.5/3.6: `tokenizer.ggml.eos_token_id` is typically 151645
        // (`<|im_end|>`) per `cmd_generate_qwen35:1066-1069`.  When the
        // GGUF metadata is absent we fall back to the HF Qwen3.5 default.
        let eos_token: u32 = gguf
            .metadata_u32("tokenizer.ggml.eos_token_id")
            .unwrap_or(151645);
        let eos_token_ids: Vec<u32> = vec![eos_token];

        // ---- Load tokenizer ----
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {e}"))?;

        // ---- Chat template ----
        // GGUFs lacking the embedded template (some Qwen3.6 dumps) yield
        // an empty string here; iter-215 MVP returns 501 before any
        // template render runs, so empty is acceptable.  Wedge-3 will
        // require non-empty for the live chat path.
        let chat_template = gguf
            .metadata_string("tokenizer.chat_template")
            .map(|s| s.to_string())
            .unwrap_or_default();

        // ---- model_id ----
        // Prefer `general.name` (matches Engine::model_id() Gemma path),
        // fall back to the file stem (matches `pool_key_for_path` used
        // when auto_pipeline returns `repo_id: None`).
        let model_id = gguf
            .metadata_string("general.name")
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                model_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "qwen35-model".to_string())
            });

        // ---- Surface fields from cfg ----
        let hidden_size = model.cfg.hidden_size as usize;
        let vocab_size = model.cfg.vocab_size as usize;
        let context_length = if model.cfg.max_position_embeddings > 0 {
            Some(model.cfg.max_position_embeddings as usize)
        } else {
            None
        };

        // ---- Quant label (matches Gemma path) ----
        let quant_type = infer_quant_type_from_gguf(&gguf);

        let load_duration = load_start.elapsed();
        tracing::info!(
            "Qwen35 SERVE load: complete in {:.1}s ({} layers, ctx_len={:?}, quant={:?})",
            load_duration.as_secs_f64(),
            n_layers,
            context_length,
            quant_type
        );

        Ok(Self {
            model,
            tokenizer,
            chat_template,
            model_id,
            eos_token_ids,
            hidden_size,
            vocab_size,
            context_length,
            quant_type,
            load_duration,
        })
    }
}

/// Dominant non-fp tensor type label.  Mirrors
/// `engine::infer_quant_type_from_gguf` (kept private to that module);
/// duplicated here rather than refactored into a shared helper because
/// the algorithm is 25 LOC and a refactor would touch a load-bearing
/// file beyond iter-215's scope.
fn infer_quant_type_from_gguf(gguf: &mlx_native::gguf::GgufFile) -> Option<String> {
    use mlx_native::GgmlType;
    use std::collections::HashMap;

    let mut histogram: HashMap<&'static str, usize> = HashMap::new();
    for name in gguf.tensor_names() {
        let Some(info) = gguf.tensor_info(name) else { continue };
        if matches!(info.ggml_type, GgmlType::F32 | GgmlType::F16) {
            continue;
        }
        let label = match info.ggml_type {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q4_K => "Q4_K",
            GgmlType::Q5_K => "Q5_K",
            GgmlType::Q6_K => "Q6_K",
            GgmlType::I16 => "I16",
        };
        *histogram.entry(label).or_insert(0) += 1;
    }
    histogram
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(k, _)| k.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Negative-path: `Qwen35LoadedModel::load` against a non-existent
    /// path returns an Err with the path in the message.  Smoke test
    /// that the constructor's exists-check fires before any GGUF parse.
    #[test]
    fn qwen35_loaded_model_load_errors_when_path_missing() {
        let opts = LoadOptions {
            model_path: std::path::PathBuf::from("/tmp/iter-215-does-not-exist.gguf"),
            tokenizer_path: None,
            config_path: None,
        };
        let res = Qwen35LoadedModel::load(&opts);
        assert!(res.is_err());
        let msg = format!("{:#}", res.err().unwrap());
        assert!(
            msg.contains("Model not found"),
            "expected 'Model not found' in error; got: {msg}"
        );
    }
}
